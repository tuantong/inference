import ast
import datetime as dt
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import dcor
import numpy as np
import pandas as pd
from dateutil.relativedelta import MO, relativedelta
from tqdm import tqdm

from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler
from forecasting.data.preprocessing.preprocess_data import preprocess_for_inference
from forecasting.data.preprocessing.utils import find_similar_items
from forecasting.data.util import (
    calc_adi,
    calc_created_time,
    calc_sale_per_day,
    find_leading_zeros_cumsum,
    get_series_type,
    remove_leading_zeros,
)
from forecasting.evaluation.aggregation_utils import (
    aggregate_bottom_up,
    aggregate_daily_ts_to_monthly,
    aggregate_monthly_ts_to_daily,
    aggregate_top_down_based_on_sale_distribution,
    aggregate_weekly_ts_to_daily,
    build_result_df_from_pd,
    clip_channel_pred_smaller_than_all_channel_pred,
)
from forecasting.models.trend_detector import trend_detector
from forecasting.monitor.forecast_monitor import ForecastMonitor
from forecasting.util import NpEncoder, get_formatted_duration
from forecasting.utils.common_utils import save_dict_to_json
from inference.find_similar_items_with_image import (
    find_similar_set_for_new_products_with_image,
    preprocess_image_urls,
)
from inference.trend_detection import get_trend

warnings.filterwarnings("ignore")

np.random.seed(42)
NEW_BRANDS = ["ipolita"]
BRANDS_TO_CREATE_ALL_CHANNEL = ["mizmooz", "as98"]
prediction_col = "predictions"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def check_value(pred_ts, sale_per_day):
    check_val = np.sum(pred_ts) / (sale_per_day * len(pred_ts))
    return check_val


def blend_with_ratio(pred_ts, sale_per_day, created_time):
    pred_arr = np.array(pred_ts)
    if (pred_arr.sum() < 1) | ((created_time <= 31) & (sale_per_day == 0)):
        ratio = 1
    else:
        ratio = (sale_per_day * len(pred_ts)) / (pred_arr.sum())
    blend_pred_arr = ratio * pred_arr
    return list(blend_pred_arr)


def blend_with_spd(pred_ts, sale_per_day, check_val):
    pred_arr = np.array(pred_ts)
    if check_val > 1.5 or check_val < 0.5:
        blend_fc = np.round(0.6 * sale_per_day + 0.4 * pred_arr, 7)
    else:
        blend_fc = pred_arr
    return list(blend_fc)


def calc_mean(pred_ts, sale_per_day):
    mean_pred_ts = [(val + sale_per_day) / 2 for val in pred_ts]
    return mean_pred_ts


def calc_clip(pred_ts, sale_per_day):
    # pred_clip = pred_ts[:90]
    # pred_unclip = pred_ts[90:]

    # pred_clip = np.clip(pred_clip, a_min= 0.5*sale_per_day, a_max=1.5*sale_per_day)
    # return [*pred_clip, *pred_unclip]

    pred_clip = np.clip(pred_ts, a_min=0.5 * sale_per_day, a_max=1.5 * sale_per_day)
    return pred_clip


def get_slope(history_ts, forecast_ts, forecast_date, historical_length, future_length):
    if pd.Timestamp(forecast_date).day != 1:
        forecast_ts[0] += history_ts[-1]
        history_ts.pop(-1)

    history_ts = history_ts[-historical_length:]
    forecast_ts = forecast_ts[:future_length]

    # Calculate slope for predictions
    array_of_data = history_ts + forecast_ts
    indices = np.arange(len(array_of_data))
    (slope, beta) = trend_detector(indices, array_of_data)
    return slope, beta


def detect_trend(
    result_df: pd.DataFrame,
    forecast_date,
    historical_length: int = 3,
    future_length: int = 0,
):
    platform_list = result_df.from_source.unique().tolist()

    final_df = pd.DataFrame()
    for platform in platform_list:
        platform_result_df = result_df[result_df.from_source == platform]

        channel_list = platform_result_df.channel_id.unique().tolist()
        for ch_id in channel_list:  # Calculate trend slope for each channel separately
            ch_result_df = platform_result_df[platform_result_df.channel_id == ch_id]
            ch_result_df["slope"] = ch_result_df.apply(
                lambda x: get_slope(
                    list(x["monthly_train_ts"]),
                    list(x["monthly_pred_ts"]),
                    forecast_date,
                    historical_length,
                    future_length,
                )[0],
                axis=1,
            )

            results_slope = ch_result_df.slope.tolist()
            # Calculate threshold to use for slope sensitivity on predictions
            mask_negative = (slope < 0 for slope in results_slope)
            mask_positive = (slope > 0 for slope in results_slope)

            thresh_up = 0
            thresh_down = 0
            if any(mask_negative):
                thresh_down = np.quantile(
                    [slope for slope in results_slope if slope < 0], 0.07
                )
            if any(mask_positive):
                thresh_up = np.quantile(
                    [slope for slope in results_slope if slope > 0], 0.7
                )
            logger.info(
                f"Platform: {platform}, Channel ID: {ch_id}, Thresh up: {thresh_up}, Thresh down: {thresh_down}"
            )

            ch_result_df["trend"] = ch_result_df.apply(
                lambda x: get_trend(x["slope"], thresh_up, thresh_down), axis=1
            )
            final_df = pd.concat([final_df, ch_result_df])
    return final_df.reset_index(drop=True)


def last_date_of_month(date):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = date.replace(day=28) + dt.timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - dt.timedelta(days=next_month.day)


def check_forecast_pattern(
    required_len,
    pred_df,
    corr_threshold=0.9,
    corr_method="distance",
    crop_history_length=True,
    plot=False,
):
    def calc_corr():
        if corr_method == "distance":
            corr = np.round(
                dcor.distance_correlation(
                    compare_df["prediction_total"],
                    compare_df["true_value_previous_year"],
                ),
                4,
            )
        else:
            corr = np.round(
                compare_df["prediction_total"].corr(
                    compare_df["true_value_previous_year"], method=corr_method
                ),
                4,
            )
        return corr

    assert corr_method in [
        "pearson",
        "spearman",
        "kendall",
        "distance",
    ], f"corr_method: `{corr_method}` is not recognized"

    forecast_date = pd.to_datetime(pred_df.first_pred_date.values[0])
    adjusted_pred_dict = dict()
    #     unadjusted_item_list = []
    for item_pred in tqdm(
        pred_df.itertuples(),
        total=pred_df.shape[0],
        desc="Check seasonality and adjust forecast",
    ):
        # print("-"*64)
        history_ts = np.array(item_pred.monthly_train_ts)
        pred_ts = np.array(item_pred.monthly_pred_ts)
        item_id = item_pred.id

        series_type, _, _ = get_series_type(history_ts[:-1])

        # Get the last 13 values of historical data and the corresponding dates
        # Include the last day of the month of forecast_date in the historical data
        if crop_history_length:
            # history_ts = history_ts[-13:] if len(history_ts) >= 13 else history_ts
            history_ts = history_ts[-14:] if len(history_ts) >= 14 else history_ts

        # Get next 12 months of forecast
        # pred_ts = pred_ts[:13]
        pred_ts = pred_ts[:14]

        # Create time range of history and future
        thresh_date = last_date_of_month(forecast_date)
        if forecast_date.date().day != 1:
            start_date = thresh_date + relativedelta(months=-len(history_ts) + 1)
        else:
            start_date = thresh_date + relativedelta(months=-len(history_ts))

        end_date = thresh_date + relativedelta(months=+len(pred_ts) - 1)

        # Create dataframe for plotting
        plot_period = pd.period_range(start_date, end_date, freq="M")

        hist_df = pd.DataFrame(
            {"date": plot_period[: len(history_ts)], "true_value": history_ts}
        )
        pred_df = pd.DataFrame(
            {"date": plot_period[-len(pred_ts) :], "prediction": pred_ts}
        )

        # tinh chỉnh max forecast <= 1.2 max hist value

        # Find the month and year of the highest gross sales quantity in history
        max_hist_month = hist_df["date"].iloc[hist_df["true_value"].idxmax()]
        max_hist_year = max_hist_month.year

        # Find the gross sales quantity of the highest forecast month in history
        max_hist_value = hist_df.loc[
            (hist_df["date"].dt.month == max_hist_month.month)
            & (hist_df["date"].dt.year == max_hist_year),
            "true_value",
        ].values[0]

        # Calculate the threshold value as 1.2 times the historical value
        threshold = max_hist_value * 1.2

        # Find the indices of the forecast values that exceed the threshold
        exceed_indices = pred_df[pred_df["prediction"] > threshold].index

        # Check if there are any forecast values that exceed the threshold
        if len(exceed_indices) > 0:
            # Adjust the forecast values to the threshold
            pred_df.loc[exceed_indices, "prediction"] = threshold

        plot_df = hist_df.merge(pred_df, how="outer", on="date")
        plot_df.date = plot_df.date.apply(
            lambda x: x.to_timestamp(freq="M").strftime("%b,%y")
        )

        # Get value of previous year
        plot_df.loc[plot_df["true_value"] == 0, "true_value"] += 1
        plot_df.loc[plot_df["prediction"] == 0, "prediction"] += 1
        plot_df["prediction_total"] = plot_df["true_value"].fillna(0.0) + plot_df[
            "prediction"
        ].fillna(0.0)
        plot_df["true_value_previous_year"] = (
            plot_df.set_index("date")["prediction_total"].shift(12).values
        )
        plot_df.loc[
            plot_df["true_value_previous_year"] == 0, "true_value_previous_year"
        ] += 1
        compare_df = plot_df.loc[~plot_df["prediction"].isna()].set_index("date")

        # compare_df['prediction_total'] = compare_df.apply(
        #     lambda x: x['true_value'] + x['prediction'] if not np.isnan(x['true_value']) else x['prediction'],
        #     axis=1
        # )
        # compare_df['smi_last_year'] = compare_df['true_value_previous_year'] / compare_df['true_value_previous_year'].sum()
        # compare_df['smi_forecast'] = compare_df['prediction_total'] / compare_df['prediction_total'].sum()
        compare_df = compare_df.round(4)
        compare_df["true_value"] = compare_df["true_value"].fillna(0.0)

        # print(f"Correlation threshold: {corr_threshold}")

        sales_corr = calc_corr()

        # print(f"{corr_method.title()} correlation for SMI between forecast and last year: {sales_corr}")

        if np.round(sales_corr, 2) >= np.round(corr_threshold, 2):
            logger.debug(
                f"Item ID: {item_id} forecast pattern follows last year's pattern"
            )
            if len(exceed_indices) > 0:
                adjusted_pred_dict[item_id] = pred_df.prediction.tolist()
        else:
            logger.debug(
                f"Item ID: {item_id} Forecast pattern does not follow last year's pattern, performing adjustments"
            )
            for idx, row in enumerate(compare_df.itertuples()):
                # date = datetime.datetime.strptime(row.Index, "%b,%y")
                # print(f"Sales for {date.month} - year {date.year - 1}: {row.true_value_previous_year}" \
                #     f" | Sales for {date.month} - year {date.year}: {row.prediction_total}")
                sales_ratio = row.prediction_total / row.true_value_previous_year
                # print(f"sales_ratio: {sales_ratio:.2f}" )
                if sales_ratio >= 1.2 or sales_ratio <= 0.8:
                    thresh = (
                        np.random.uniform(1.2, sales_ratio)
                        if sales_ratio >= 1.2
                        else np.random.uniform(0.8 * sales_ratio, 1.25 * sales_ratio)
                    )
                    new_value = np.round(row.prediction / thresh)
                    new_value = np.round(
                        np.clip(
                            new_value,
                            0.8 * row.true_value_previous_year,
                            1.2 * row.true_value_previous_year,
                        )
                    )
                    compare_df.at[row.Index, "prediction"] = new_value
                    # Replace the true_value_previous_year with adjusted_new_value
                    if idx < compare_df.shape[0] - 12:
                        compare_df.iat[idx + 12, 3] = new_value + row.true_value

            # Recalculate SMI with new forecast
            compare_df["prediction_total"] = compare_df.apply(
                lambda x: (
                    x["true_value"] + x["prediction"]
                    if not np.isnan(x["true_value"])
                    else x["prediction"]
                ),
                axis=1,
            )
            # TODO: Fix hardcode position here
            # Replace 'previous year value' for current month with new prediction value
            compare_df.iat[-1, 3] = compare_df.iat[0, 2]
            compare_df = compare_df.round(4)
            adjusted_pred_dict[item_id] = compare_df.prediction.tolist()
    return adjusted_pred_dict


def find_item_type(row):
    if row["brand_name"] in NEW_BRANDS:
        if row["adi"] <= 1.4:
            if row["item_length"] < 6:
                return "new"
            return "seasonal_adjusted"
        return "new"
    if row["created_time"] > 360:
        if row["adi"] >= 2.01:
            return "lumpy"
        return "seasonal_adjusted"
    else:
        return "new"


def run_seasonal_pattern_check(
    brand_list, pred_pivot_df, monthly_pivot, inference_week=1
):
    logger.info(f"List of brands: {brand_list}")
    pred_pivot_df = pred_pivot_df[pred_pivot_df.brand_name.isin(brand_list)]

    # Aggregate results
    logger.info("Preprocess predictions data...")
    pred_pivot_df = pred_pivot_df.assign(
        # Create daily_pred_ts
        daily_pred_ts=(
            pred_pivot_df[prediction_col].apply(
                lambda x: aggregate_weekly_ts_to_daily(x)
            )
        )
    ).assign(
        # Create daily_train_ts
        daily_train_ts=(
            pred_pivot_df["train_ts"].apply(lambda x: aggregate_weekly_ts_to_daily(x))
        )
    )
    # Remove leading zeros of every time series to calculate sale_per_day
    logger.info("Removing leading zeros values for daily_train_ts...")
    pred_pivot_df = pred_pivot_df.assign(
        # Create daily_train_ts_cut
        daily_train_ts_cut=(
            pred_pivot_df.daily_train_ts.apply(lambda x: remove_leading_zeros(x))
        )
    )
    # Join predictions df with monthly_history df
    monthly_pivot = monthly_pivot[monthly_pivot.id.isin(pred_pivot_df.id.unique())]
    pred_pivot_df = pd.merge(pred_pivot_df, monthly_pivot, on="id")
    logger.info(f"Number of all items: {len(pred_pivot_df.id.unique())}")

    pred_pivot_df["sale_per_day"] = pred_pivot_df.daily_train_ts_cut.apply(
        lambda x: calc_sale_per_day(x)
    )
    inference_date = datetime.today().date() + relativedelta(
        weekday=MO(-inference_week)
    )
    logger.info(f"Inference date: {inference_date}")
    pred_pivot_df = pred_pivot_df.assign(
        # Calculate created_time
        created_time=(
            pred_pivot_df.created_date.apply(
                lambda x: calc_created_time(inference_date, x)
            )
        )
    ).assign(
        # Calculate length of monthly historical sales after removing leading zeros
        item_length=(
            pred_pivot_df.monthly_train_ts.apply(remove_leading_zeros).apply(len)
        )
    )

    pred_pivot_df = pred_pivot_df.assign(
        **{"type": pred_pivot_df.apply(find_item_type, axis=1)}
    )
    logger.info(
        f'Count type of product: {pred_pivot_df[pred_pivot_df.is_product == True]["type"].value_counts()}'
    )

    # Create unique_id column
    # remove_id_list = []
    # for row in pred_pivot_df.itertuples():
    #     remove_id = row.brand_name + "_" + row.product_id + "_" + row.channel_id
    #     remove_id_list.append(remove_id)

    # pred_pivot_df = pred_pivot_df.assign(remove_id=remove_id_list)

    # logger.info("Checking for remove lumpy items -> Not forecast...")
    # Create zero historical product df
    # no_historical_product_df = pred_pivot_df[
    #     (pred_pivot_df.is_product == True)
    #     & (pred_pivot_df.created_time >= 12 * 7)
    #     & (pred_pivot_df.adi.isnull() == True)
    # ]
    # logger.info(f'Shape of no_historical_product_df: {no_historical_product_df.shape[0]}')
    # # Create lumpy product df
    # lumpy_product_df = pred_pivot_df[
    #     (pred_pivot_df.is_product == True)
    #     & (pred_pivot_df.created_time >= 12 * 7)
    #     & (pred_pivot_df.adi >= 2.01)
    #     & (pred_pivot_df.brand_name != "ipolita")
    # ]
    # remove_product_df = pd.concat([no_historical_product_df, lumpy_product_df])
    # remove_product_df = no_historical_product_df
    # remove_product_df = remove_product_df[
    #     ~remove_product_df.remove_id.isin(no_remove_item_list)
    # ]
    # remove_product_list = remove_product_df.remove_id.unique()
    # Check if product is removed -> remove all its variants
    # remove_variant_df = pred_pivot_df[
    #     (pred_pivot_df.is_product == False)
    #     & (pred_pivot_df.remove_id.isin(remove_product_list))
    # ]
    # removed_ids = list(
    #     set(remove_product_df.id.tolist() + remove_variant_df.id.tolist())
    # )

    # logger.info(f"Number of removed items: {len(removed_ids)}")
    # pred_df = pred_pivot_df[~pred_pivot_df.id.isin(removed_ids)]
    pred_df = pred_pivot_df

    # Create 'monthly_pred_ts' - Use 'assign' for adaptability
    pred_df = pred_df.assign(
        **{
            "monthly_pred_ts": pred_df.apply(
                lambda x: aggregate_daily_ts_to_monthly(
                    x["daily_pred_ts"], x["first_pred_date"]
                ),
                axis=1,
            )
        }
    )
    # pred_df["monthly_pred_ts"] = pred_df.monthly_pred_ts.apply(lambda x: x[:13])
    pred_df["monthly_pred_ts"] = pred_df.monthly_pred_ts.apply(lambda x: x[:14])

    variant_pred_df = pred_df[~pred_df.is_product]
    product_pred_df = pred_df[pred_df.is_product]

    # Check and adjust forecast on monthly_pred_ts
    logger.info("Check and adjust seasonal pattern...")
    adjusted_pred_dict = check_forecast_pattern(
        required_len=12,
        pred_df=product_pred_df,
        corr_threshold=0.99,
        corr_method="distance",
        crop_history_length=False,
        plot=False,
    )
    logger.info(
        f"Number of items adjusted: {len(adjusted_pred_dict)}/{product_pred_df.shape[0]}"
    )

    # Create 'adjusted_monthly_pred_ts' - Use 'assign' for adaptability
    product_pred_df = product_pred_df.assign(
        **{
            "adjusted_monthly_pred_ts": product_pred_df.apply(
                lambda row: (
                    adjusted_pred_dict[row.id]
                    if row.id in adjusted_pred_dict.keys()
                    else row.monthly_pred_ts
                ),
                axis=1,
            )
        }
    )

    # Create 'daily_pred_ts' - Use 'assign' for adaptability
    product_pred_df = product_pred_df.assign(
        **{
            "daily_pred_ts": product_pred_df.apply(
                lambda row: (
                    aggregate_monthly_ts_to_daily(
                        row.adjusted_monthly_pred_ts,
                        row.first_pred_date,
                    )
                    if row.id in adjusted_pred_dict.keys()
                    else row.daily_pred_ts
                ),
                axis=1,
            )
        }
    )
    min_len_daily_pred_ts = min(
        [
            len(product_pred_df.daily_pred_ts.values[i])
            for i in range(product_pred_df.shape[0])
        ]
    )

    logger.info(f"Shape of product_pred_df: {product_pred_df.shape[0]}")

    final_df = pd.concat([variant_pred_df, product_pred_df], ignore_index=True)
    final_df["daily_pred_ts"] = final_df.daily_pred_ts.apply(
        lambda x: x[:min_len_daily_pred_ts]
    )

    return final_df


def find_similar_set_for_new_products(full_pred_df, lookup_item_list):
    result = {}
    for (brand_name, channel_id), group_df in full_pred_df.groupby(
        ["brand_name", "channel_id"]
    ):
        # print(brand_name, channel_id)
        all_product_df = group_df[group_df["is_product"]]
        new_product_df = all_product_df[all_product_df["type"] == "new"]
        if brand_name in ["ipolita"]:
            old_product_df = all_product_df[all_product_df["id"].isin(lookup_item_list)]
        else:
            old_product_df = all_product_df[
                all_product_df["id"].isin(lookup_item_list)
                & (all_product_df["id"] != "new")
            ]
        if len(old_product_df) == 0:
            similar_product_dict = {
                item_id: None for item_id in new_product_df["id"].tolist()
            }
        else:
            item_text = (
                all_product_df["brand_name"]
                + "_"
                + all_product_df["channel_id"]
                + "_"
                + all_product_df["category"]
                + "_"
                + all_product_df["name"]
            ).tolist()

            similar_product_dict = find_similar_items(
                all_product_df, old_product_df, new_product_df, item_text
            )

        result.update(similar_product_dict)

    return result


def adjust_forecast_to_last_year_pattern(hist_df, pred_df):
    # Add temporary 'month' and 'year' columns for comparison, derived from the 'date' column
    hist_df_temp = hist_df.copy()
    pred_df_temp = pred_df.copy()
    hist_df_temp["month"] = hist_df_temp["date"].dt.month
    hist_df_temp["year"] = hist_df_temp["date"].dt.year
    pred_df_temp["month"] = pred_df_temp["date"].dt.month
    pred_df_temp["year"] = (
        pred_df_temp["date"].dt.year - 1
    )  # Adjust year for previous year comparison
    max_hist = (
        max(hist_df_temp["values"][:-1])
        if hist_df_temp.shape[0] > 1
        else hist_df_temp["values"].values[0]
    )

    # Loop through the pred_df
    for i, row in pred_df_temp.iterrows():
        pred_month = row["month"]
        pred_year = row["year"]

        corresponding_hist = hist_df_temp[
            (hist_df_temp["month"] == pred_month) & (hist_df_temp["year"] == pred_year)
        ]

        if i == 0:
            last_hist_value = hist_df_temp.iloc[-1]["values"]
            row["values"] += last_hist_value

            if not corresponding_hist.empty:
                last_year_value = corresponding_hist["values"].values[-1]
                adjusted_value = np.clip(
                    row["values"], 1.1 * last_year_value, 1.2 * last_year_value
                )
                pred_df_temp.at[i, "values"] = max(
                    int(round(adjusted_value - last_hist_value)), 0
                )
        else:
            if not corresponding_hist.empty:
                # Use the most recent historical value
                if i < 12:
                    last_year_value = corresponding_hist["values"].values[-1]
                    adjusted_value = np.clip(
                        row["values"], 1.1 * last_year_value, 1.2 * last_year_value
                    )
                else:
                    last_year_value = (
                        corresponding_hist["values"].values[-1]
                        + pred_df_temp["values"].values[i - 12]
                    )
                    adjusted_value = np.clip(
                        row["values"], last_year_value, 1.5 * max_hist
                    )

                # pred_df_temp.at[i, "values"] = int(round(adjusted_value))
                pred_df_temp.at[i, "values"] = max(int(round(adjusted_value)), 0)

    # Remove the temporary 'month' and 'year' columns before returning the result
    adjusted_pred_df = pred_df_temp.drop(columns=["month", "year"])

    return adjusted_pred_df


def adjust_product_forecast_from_similar_items_avg_w_scale_ratio_and_historical_pattern(
    full_pred_df,
):
    product_df = full_pred_df[
        (full_pred_df["is_product"] == True) & (full_pred_df["type"] == "new")
    ]
    adjust_pred_dict = {}
    for item_pred in tqdm(
        product_df.itertuples(),
        total=product_df.shape[0],
        desc="Adjusting products forecast based on similar products",
    ):
        item_id = item_pred.id
        similar_items_of_item = item_pred.similar_items
        item_first_pred_date = item_pred.first_pred_date
        item_created_length = item_pred.item_length  # months after remove_leading_zeros
        item_created_time = (
            item_pred.created_time
        )  # days from created_date to inference_date

        if similar_items_of_item is not None:
            similar_items_df = full_pred_df[
                full_pred_df["id"].isin(similar_items_of_item)
            ]
            similar_items_preds = np.stack(
                similar_items_df["adjusted_monthly_pred_ts"].apply(np.array), axis=0
            )

            item_monthly_hist = np.array(item_pred.monthly_train_ts)
            if (item_created_time > 14) or (item_monthly_hist.max() > 0):
                min_len_similar_items = (
                    similar_items_df["monthly_train_ts"].apply(len).min()
                )
                similar_items_hist = np.stack(
                    similar_items_df["monthly_train_ts"]
                    .apply(np.array)
                    .apply(lambda x: x[-min_len_similar_items:]),
                    axis=0,
                )

                item_created_time_hist_sum = item_monthly_hist[
                    -item_created_length:
                ].sum()
                similar_items_hist_mean = (
                    similar_items_hist[:, -item_created_length:].sum(axis=1).mean()
                )
                scale_ratio = (
                    item_created_time_hist_sum / similar_items_hist_mean
                    if similar_items_hist_mean != 0
                    else 1
                )

                monthly_raw_avg_sim_forecast = np.mean(similar_items_preds, axis=0)
                #             print(monthly_raw_avg_sim_forecast)
                new_ratio = (
                    1.5 * item_monthly_hist.max() / monthly_raw_avg_sim_forecast.max()
                )
                final_ratio = min(scale_ratio, new_ratio)
                #             print(final_ratio)
                monthly_raw_avg_sim_forecast_scaled = np.round(
                    monthly_raw_avg_sim_forecast * final_ratio
                )

                # pred_ts = monthly_raw_avg_sim_forecast_scaled[:13]
                pred_ts = monthly_raw_avg_sim_forecast_scaled[:14]

                # Create time range of history and future
                thresh_date = last_date_of_month(item_first_pred_date)
                if item_first_pred_date.date().day != 1:
                    start_date = thresh_date + relativedelta(
                        months=-len(item_monthly_hist) + 1
                    )
                else:
                    start_date = thresh_date + relativedelta(
                        months=-len(item_monthly_hist)
                    )

                end_date = thresh_date + relativedelta(months=+len(pred_ts) - 1)

                # Create dataframe for plotting
                plot_period = pd.period_range(start_date, end_date, freq="M")

                hist_df = pd.DataFrame(
                    {
                        "date": plot_period[: len(item_monthly_hist)],
                        "values": item_monthly_hist,
                    }
                )
                pred_df = pd.DataFrame(
                    {"date": plot_period[-len(pred_ts) :], "values": pred_ts}
                )

                monthly_pred_adjusted = adjust_forecast_to_last_year_pattern(
                    hist_df, pred_df
                )

                daily_pred_adjusted = aggregate_monthly_ts_to_daily(
                    monthly_pred_adjusted["values"].tolist(),
                    item_first_pred_date,
                )
            else:
                monthly_pred_adjusted = np.mean(similar_items_preds, axis=0)
                daily_pred_adjusted = aggregate_monthly_ts_to_daily(
                    monthly_pred_adjusted.tolist(), item_first_pred_date
                )

            adjust_pred_dict[item_id] = daily_pred_adjusted

    return adjust_pred_dict


def adjust_forecast_with_all_constraints(hist_df, pred_df, similar_items_type, avg_adi):
    # Add temporary 'month' and 'year' columns for comparison, derived from the 'date' column
    hist_df_temp = hist_df.copy()
    pred_df_temp = pred_df.copy()
    hist_df_temp["month"] = hist_df_temp["date"].dt.month
    hist_df_temp["year"] = hist_df_temp["date"].dt.year
    pred_df_temp["month"] = pred_df_temp["date"].dt.month
    pred_df_temp["year"] = (
        pred_df_temp["date"].dt.year - 1
    )  # Adjust year for previous year comparison
    max_hist = (
        max(hist_df_temp["values"][:-1])
        if hist_df_temp.shape[0] > 1
        else hist_df_temp["values"].values[0]
    )

    non_hist_months_indices = (
        []
    )  # Track indices of pred_df_temp with no corresponding historical data

    # Loop through the pred_df starting from the second row
    for i, row in pred_df_temp.iterrows():
        pred_month = row["month"]
        pred_year = row["year"]
        # Find corresponding month in the previous year in hist_df
        corresponding_hist = hist_df_temp[
            (hist_df_temp["month"] == pred_month) & (hist_df_temp["year"] == pred_year)
        ]

        if corresponding_hist.empty:
            non_hist_months_indices.append(i)
        else:
            if i == 0:
                last_hist_value = hist_df_temp.iloc[-1]["values"]
                row["values"] += last_hist_value
                last_year_value = corresponding_hist["values"].values[-1]
                adjusted_value = np.clip(
                    row["values"], 1.1 * last_year_value, 1.2 * last_year_value
                )
                pred_df_temp.at[i, "values"] = max(
                    int(round(adjusted_value - last_hist_value)), 0
                )

            else:

                if i < 12:
                    last_year_value = corresponding_hist["values"].values[-1]
                    adjusted_value = np.clip(
                        row["values"], 1.1 * last_year_value, 1.2 * last_year_value
                    )
                else:
                    last_year_value = (
                        corresponding_hist["values"].values[-1]
                        + pred_df_temp["values"].values[i - 12]
                    )
                    adjusted_value = np.clip(
                        row["values"], last_year_value, 1.5 * max_hist
                    )

    # Calculate S as specified
    average_of_historical_values = np.mean(hist_df_temp["values"].values)
    number_of_non_historical_months = len(non_hist_months_indices)
    S = average_of_historical_values * number_of_non_historical_months
    # print(f"Average of historical values: {average_of_historical_values}")
    # print(f"Number of non_historical months: {number_of_non_historical_months}")
    # print(f"Expected sum of non_historical months: S={S}")

    allowable_max_factor = 1.5
    max_allowable_forecast_value = int(
        round(allowable_max_factor * hist_df_temp["values"].max())
    )
    # print(f"Max allowable adjustment for forecast values: {max_allowable_forecast_value}")

    forecast_sum_factor = 1.15
    allowable_forecast_sum = int(round(forecast_sum_factor * S))
    # print(f"Allowable sum of forecast values for adjustments: {allowable_forecast_sum}")

    # Adjust forecasts for months without historical sales, ensuring their sum does not exceed S
    if non_hist_months_indices:
        non_hist_forecasts = pred_df_temp.loc[non_hist_months_indices, "values"]
        total_forecast_sum = non_hist_forecasts.sum()
        # print(f"Current total forecast sum: {total_forecast_sum}")
        if total_forecast_sum > allowable_forecast_sum:
            # print(
            #     f"Total forecast sum exceeded {forecast_sum_factor} * S = {allowable_forecast_sum},"
            #       " adjusting values down"
            # )
            adjustment_factor = allowable_forecast_sum / total_forecast_sum
            fluctuation_range = 0.05  # Allows for +/-5% fluctuation on the base factor
            adjusted_values = []
            for value in non_hist_forecasts:
                # Apply a fluctuating adjustment factor within the defined range
                fluctuating_factor = np.random.uniform(
                    adjustment_factor - fluctuation_range,
                    adjustment_factor + fluctuation_range,
                )
                adjusted_value = np.clip(
                    np.floor(value * fluctuating_factor).astype(int),
                    0,
                    max_allowable_forecast_value,
                )
                adjusted_values.append(adjusted_value)

            # Ensure the sum of adjusted values is as close to S as possible
            adjusted_sum = sum(adjusted_values)
            # print(f"Adjusted Sum before fine-tuning: {adjusted_sum}")
            # Fine-tuning phase: Adjust the individual values to ensure the sum exactly equals S
            while adjusted_sum != allowable_forecast_sum:
                for i in range(len(adjusted_values)):
                    if (
                        adjusted_sum < allowable_forecast_sum
                        and (adjusted_values[i] + 1) <= max_allowable_forecast_value
                    ):
                        adjusted_values[i] += 1
                        adjusted_sum += 1
                    elif (
                        adjusted_sum > allowable_forecast_sum and adjusted_values[i] > 0
                    ):
                        adjusted_values[i] -= 1
                        adjusted_sum -= 1

                    if adjusted_sum == allowable_forecast_sum:
                        break

            # Update the forecast values with the adjusted values
            for idx, new_val in zip(non_hist_months_indices, adjusted_values):
                pred_df_temp.at[idx, "values"] = new_val

        elif total_forecast_sum < allowable_forecast_sum:
            # print(
            #     f"Total forecast sum smaller than {forecast_sum_factor} * S = {allowable_forecast_sum},"
            #     " adjusting values up"
            # )
            difference = allowable_forecast_sum - total_forecast_sum
            max_value_reached = set()
            while difference > 0:
                # Randomly select an index to increment
                random_index = np.random.choice(non_hist_months_indices)
                if random_index in max_value_reached:
                    # To ensure we don't get stuck in an infinite loop, check if all indices have reached max value
                    if len(max_value_reached) == len(non_hist_months_indices):
                        break  # Exit the loop if there are no more indices to adjust
                    continue
                if (
                    pred_df_temp.at[random_index, "values"] + 1
                    <= max_allowable_forecast_value
                ):
                    pred_df_temp.at[random_index, "values"] += 1
                    difference -= 1
                else:
                    # If incrementing exceeds max value, add to max_value_reached and skip this index in future iterations
                    max_value_reached.add(random_index)

                # Additionally add to max_value_reached if the current increment brings the value to the max allowable
                if (
                    pred_df_temp.at[random_index, "values"]
                    >= max_allowable_forecast_value
                ):
                    max_value_reached.add(random_index)

    # print(f"Sum forecast of non_hist_month after adjustments: {pred_df_temp.loc[non_hist_months_indices, 'values'].sum()}")

    # print("*" * 20 + " Extra Steps " + "*" * 20)
    # Additional step before returning the adjusted_pred_df
    if (len(hist_df) >= 3) or (
        len(hist_df) < 3 and (similar_items_type == "intermittent")
    ):
        if len(hist_df) >= 3:
            # Calulate ratio of number of zero_sales months over total hist_df
            zero_sale_ratio = len(hist_df[hist_df["values"] == 0]) / len(hist_df)
            # print(f"Calculated zero sale ratio from historical data: {zero_sale_ratio}")
        else:
            # print(f"Average ADI of similar items: {avg_adi}")
            zero_sale_ratio = min(3 / 4, 1 - 1 / avg_adi)
            # print(f"Calculated zero sale ratio based on Average ADI: {zero_sale_ratio}")

        # Calculate number of months in the future that should be zero based on the ratio
        num_to_zero = int(round(len(non_hist_months_indices) * zero_sale_ratio))
        # print(f"Total non-historical months indices: {len(non_hist_months_indices)}")
        # print(f"Number of non-historical month forecasts to be set to 0, based on ratio: {num_to_zero}")

        # Split non_hist_months_indices based on whether their forecast value is zero
        zero_forecast_indices = [
            index
            for index in non_hist_months_indices
            if pred_df_temp.at[index, "values"] == 0
        ]
        non_zero_forecast_indices = [
            index
            for index in non_hist_months_indices
            if pred_df_temp.at[index, "values"] != 0
        ]
        # print(f"Number of indices with zero forecasts: {len(zero_forecast_indices)}")
        # print(f"Number of indices with non-zero forecasts: {len(non_zero_forecast_indices)}")

        # Determine how many indices to select from each group
        if len(zero_forecast_indices) >= num_to_zero:
            # If there are enough zero forecast indices, select randomly from them
            indices_to_zero = np.random.choice(
                zero_forecast_indices, num_to_zero, replace=False
            )
            # print(f"Enough zero forecast indices available, selected only from zero forecasts.")
        else:
            # If not enough, take all zero forecast indices and fill the rest from the non-zero forecast indices
            num_zeros_needed = num_to_zero - len(zero_forecast_indices)
            additional_indices_to_zero = (
                np.random.choice(
                    non_zero_forecast_indices, num_zeros_needed, replace=False
                )
                if num_zeros_needed > 0
                else []
            )
            indices_to_zero = (
                zero_forecast_indices + additional_indices_to_zero.tolist()
            )

        # print(f"Number of non_hist_month forecast to be set to 0: {num_to_zero}")
        # print(f"Indices that will be set to 0: {indices_to_zero}")
        # Set the selected forecast values to 0
        for idx in indices_to_zero:
            pred_df_temp.at[idx, "values"] = 0
    # else:
    # Otherwise len(ts) < 3 and similar_items_type == 'smooth'
    # Then we use avg forecast of similar items, meaning no random drop is applied
    # print(f"Similar Items type for this item is smooth, skipping random drop")

    # Remove the temporary 'month' and 'year' columns before returning the result
    adjusted_pred_df = pred_df_temp.drop(columns=["month", "year"])

    # print(f"Indices of non_hist_month forecast: {non_hist_months_indices}")

    return adjusted_pred_df


def adjust_product_forecast_for_lumpy_item(pred_df):
    new_df = pred_df[
        pred_df["brand_name"].isin(NEW_BRANDS)
        & pred_df["is_product"]
        & (pred_df["type"] == "new")
    ]

    lumpy_df = pred_df[pred_df["is_product"] & (pred_df["type"] == "lumpy")]
    product_df = pd.concat([new_df, lumpy_df])

    adjust_pred_dict = {}
    for item_pred in tqdm(
        product_df.itertuples(),
        total=product_df.shape[0],
        desc="Adjusting products forecast based on similar products",
    ):
        item_id = item_pred.id
        # print("*" * 64)
        # print(f"Item ID: {item_id}")
        similar_items_of_item = item_pred.similar_items
        item_first_pred_date = item_pred.first_pred_date

        if similar_items_of_item is not None:
            similar_items_df = pred_df[pred_df["id"].isin(similar_items_of_item)]
            similar_items_adi = similar_items_df["adi"].values.tolist()
            similar_items_type = (
                "smooth"
                if sum(adi <= 1.4 for adi in similar_items_adi)
                > len(similar_items_adi) / 2
                else "intermittent"
            )
            avg_adi = np.mean(similar_items_adi)
            if np.isnan(avg_adi):
                raise ValueError("Average ADI of similar items is NaN")
        else:
            # raise ValueError(f"Similar items not found for item ID: {item_id}")
            continue

        history_ts = item_pred.monthly_train_ts
        pred_ts = aggregate_daily_ts_to_monthly(
            item_pred.daily_pred_ts, item_first_pred_date
        )

        # pred_ts = pred_ts[:13]
        pred_ts = pred_ts[:14]

        # Create time range of history and future
        thresh_date = last_date_of_month(item_first_pred_date)
        if item_first_pred_date.date().day != 1:
            start_date = thresh_date + relativedelta(months=-len(history_ts) + 1)
        else:
            start_date = thresh_date + relativedelta(months=-len(history_ts))

        end_date = thresh_date + relativedelta(months=+len(pred_ts) - 1)

        # Create dataframe for plotting
        plot_period = pd.period_range(start_date, end_date, freq="M")

        hist_df = pd.DataFrame(
            {"date": plot_period[: len(history_ts)], "values": history_ts}
        )
        prediction_df = pd.DataFrame(
            {"date": plot_period[-len(pred_ts) :], "values": pred_ts}
        )

        pred_df_adjusted = adjust_forecast_with_all_constraints(
            hist_df, prediction_df, similar_items_type, avg_adi
        )
        #         print(pred_df)
        #         print(pred_df_adjusted)
        #         return

        daily_pred_adjusted = aggregate_monthly_ts_to_daily(
            pred_df_adjusted["values"].tolist(),
            item_first_pred_date,
        )

        adjust_pred_dict[item_id] = daily_pred_adjusted

    return adjust_pred_dict


def fill_stockout(stockout_ts, daily_train_ts, daily_pred_ts, item_id):
    pred_len = len(daily_pred_ts)
    len_avg = 90
    daily_train_ts = daily_train_ts[-len_avg:]
    # Fill Nan for daily_val if stockout and pred = 0
    daily_pred_ts_stockout = [
        (
            np.nan
            # if (stockout_ts[i] == 1) & (daily_pred_ts[i] == 0)
            if stockout_ts[i] == 1
            else daily_pred_ts[i]
        )
        for i in range(pred_len)
    ]
    if np.isnan(daily_pred_ts_stockout).sum() == 0:
        return daily_pred_ts
    full_ts = [*daily_train_ts, *daily_pred_ts_stockout]
    for i in range(len(daily_train_ts), len(full_ts)):
        if np.isnan(full_ts[i]):
            full_ts[i] = (
                np.mean(full_ts[i - len_avg : i])
                if i >= len_avg
                else np.mean(full_ts[:i])
            )
    final_daily_pred_ts = full_ts[-pred_len:]
    if np.sum(final_daily_pred_ts) != np.sum(daily_pred_ts):
        logger.debug(f"Post-process stockout for {item_id}")

    return final_daily_pred_ts


def tweak_variant(daily_ts, item_id):
    arr = np.array(daily_ts)
    if "B0CWR7M9J3__HP-5N1WASH" in item_id:
        logger.debug(f"Tweak variant_level: {item_id}")
        tweak_arr = 1 / 5 * arr[:]
    elif "B0CWQJ6TZQ__HP-N-SNC-CON-P" in item_id:
        logger.debug(f"Tweak variant_level: {item_id}")
        tweak_arr = np.concatenate((arr[:53], np.array([7] * 30), arr[83:]))
    elif "annmarie_cbd_shopify_7685560893654" in item_id:
        logger.debug(f"Tweak variant_level: {item_id}")
        tweak_arr = np.concatenate((arr[:323], 10 * arr[323:353], arr[353:]))
    else:
        tweak_arr = arr
    return list(tweak_arr)


def generate_results(
    inference_week,
    infer_data_handler,
    config_dict,  # For loading stock data of brand
    monthly_df,
    prediction_set,
    seasonal_set,
    similar_product_dict,
    model_dir,
    result_save_dir,
    trend_config_dict,
    csv_file=False,
    post_process=False,
):
    os.makedirs(result_save_dir, exist_ok=True)
    start = time.time()
    logger.info("Loading inference dataset")
    infer_freq_df = infer_data_handler.load_data()

    infer_freq_df, similar_item_dict = preprocess_for_inference(
        df=infer_freq_df, preprocess_method="fill_zero", min_length_for_new_item=12
    )

    seasonal_items_list = seasonal_set

    logger.info("Processing monthly_df...")
    monthly_df_mask = monthly_df.groupby("id")["quantity_order"].transform(
        find_leading_zeros_cumsum
    )
    monthly_df_cut = monthly_df[~monthly_df_mask].reset_index(drop=True)
    # Map monthly historical values
    monthly_pivot = (
        pd.pivot_table(
            monthly_df_cut,
            index="id",
            values=["quantity_order"],
            aggfunc=lambda x: list(x),
        )
        .rename(columns={"quantity_order": "monthly_train_ts"})
        .reset_index()
    )
    # Cut zeros of monthly_train_ts
    monthly_pivot["adi"] = monthly_pivot.monthly_train_ts.apply(
        lambda x: calc_adi(np.array(x))
    )

    # Generate submission
    logger.info("Generating submission")
    ids_list = []
    date_list = []
    pred_list = []
    id_list = infer_freq_df["id"].unique().tolist()
    if post_process:
        logger.info("Post-processing with fill_avg_sim")
    for u_id, pred_ts in tqdm(zip(id_list, prediction_set), total=len(prediction_set)):
        len_pred_ts = len(pred_ts)
        ts_date = pred_ts.time_index

        if post_process:
            similar_items_of_item = similar_item_dict.get(u_id, [])
            if len(similar_items_of_item):
                # Calculate avg forecast of similar items and use as prediction
                avg_sim_fc = pd.DataFrame()
                for item in similar_items_of_item:
                    item_index = id_list.index(item)
                    item_fc = prediction_set[item_index].pd_dataframe().round()
                    avg_sim_fc = pd.concat([avg_sim_fc, item_fc], axis=1)

                pred_values = avg_sim_fc.mean(axis=1).clip(0.0).round().values
            else:
                pred_values = pred_ts.univariate_values().clip(0.0).round()
        else:
            # Get the actual prediction of model
            pred_values = pred_ts.univariate_values().clip(0.0).round()

        ids_list.extend([u_id] * len_pred_ts)
        date_list.extend(ts_date)
        pred_list.extend(pred_values)

    pred_df = pd.DataFrame(
        {"id": ids_list, "date": date_list, "predictions": pred_list}
    )

    # Read metadata of inference set (up-to-date)
    infer_meta_df = infer_data_handler.load_metadata(drop_price=False)
    pred_df = pred_df.merge(infer_meta_df, on="id", how="left")

    # Read frequency_inference_df (history = last 25 weeks)
    pred_pivot_df = build_result_df_from_pd(
        infer_freq_df, pred_df, pred_cols=["predictions"]
    )

    # Merge metadata
    pred_pivot_df = pred_pivot_df.merge(
        infer_meta_df, on=infer_data_handler.metadata.id_col, how="left"
    )
    logger.info(
        f"Time for processing dataset: {get_formatted_duration(time.time()-start)}"
    )

    brand_list = pred_pivot_df.brand_name.unique().tolist()
    logger.info(f"Brand list: {brand_list}")

    start_adjust = time.time()
    #### STEP 1: ADJUST SEASONALITY
    final_df = run_seasonal_pattern_check(
        brand_list=brand_list,
        pred_pivot_df=pred_pivot_df,
        monthly_pivot=monthly_pivot,
        inference_week=inference_week,
    )

    ### STEP 2: FIND SIMILAR PRODUCT -> ADJUST NEW_ITEM = AVG_SIMILAR
    attribute_weights = {
        "brand_name": 1,
        "platform": 1,
        "channel_id": 1,
        "category": 1.5,
        "name": 1,
        "color": 1,
        "size": 1,
        "price": 1.5,
        "image": 2,
    }
    final_df = preprocess_image_urls(final_df)
    if similar_product_dict is None:
        logger.info("Find similar_product and adjust for new_item")
        similar_product_dict = find_similar_set_for_new_products_with_image(
            final_df,
            seasonal_items_list,
            new_items_list=None,
            attribute_weights=attribute_weights,
            image_model_name="facebook/dinov2-small",
            verbose=False,
        )
    else:
        logger.info(
            f"Number of products in similar_product_dict: {len(similar_product_dict)}"
        )
        new_and_lumpy_product_df = final_df[
            (final_df.type.isin(["new", "lumpy"])) & (final_df.is_product == True)
        ]
        new_and_lumpy_product_df_wo_similar_set_df = new_and_lumpy_product_df[
            ~(new_and_lumpy_product_df.id.isin(similar_product_dict.keys()))
        ]
        logger.info(
            f"Number of new_and_lumpy_product_df without similar_set: {new_and_lumpy_product_df_wo_similar_set_df.shape[0]}"
        )

        if new_and_lumpy_product_df_wo_similar_set_df.shape[0] > 0:
            logger.info(
                f"Finding similar item for more {new_and_lumpy_product_df_wo_similar_set_df.shape[0]} new items..."
            )
            more_similar_product_dict = find_similar_set_for_new_products_with_image(
                final_df,
                seasonal_items_list,
                new_items_list=new_and_lumpy_product_df_wo_similar_set_df.id.tolist(),
                attribute_weights=attribute_weights,
                image_model_name="facebook/dinov2-small",
                verbose=False,
            )
            similar_product_dict.update(more_similar_product_dict)

    logger.info(
        f"Number of products in final_similar_product_dict: {len(similar_product_dict)}"
    )

    save_dict_to_json(
        similar_product_dict, os.path.join(model_dir, "similar_product_dict.json")
    )
    save_dict_to_json(
        similar_product_dict, os.path.join(result_save_dir, "similar_product_dict.json")
    )

    similar_product_df = pd.DataFrame(
        similar_product_dict.items(), columns=["id", "similar_items"]
    )

    full_pred_df = final_df.merge(similar_product_df, on="id", how="left")
    adjusted_item_dict = adjust_product_forecast_from_similar_items_avg_w_scale_ratio_and_historical_pattern(
        full_pred_df
    )

    full_pred_df["daily_pred_ts"] = full_pred_df.apply(
        lambda row: (
            adjusted_item_dict[row.id]
            if row.id in adjusted_item_dict
            else row.daily_pred_ts
        ),
        axis=1,
    )

    ### STEP 3: FIX FOR LUMPY ITEMS (ESPECIALLY FOR IPOLITA)
    logger.info("Adjust for lumpy item")
    lumpy_adjust_dict = adjust_product_forecast_for_lumpy_item(full_pred_df)

    full_pred_df["daily_pred_ts"] = full_pred_df.apply(
        lambda row: (
            lumpy_adjust_dict[row.id]
            if row.id in lumpy_adjust_dict
            else row.daily_pred_ts
        ),
        axis=1,
    )
    logger.info(
        f"Time for adjust product level: {get_formatted_duration(time.time() - start_adjust)}"
    )
    logger.info(f"Shape of full_pred_df: {full_pred_df.shape[0]}")

    forecast_date = pd.to_datetime(full_pred_df.first_pred_date.values[0]).date()
    daily_forecast_length = len(full_pred_df.daily_pred_ts.values[0])
    forecast_range = pd.date_range(
        forecast_date, periods=daily_forecast_length, freq="D"
    ).tolist()
    logger.info(
        f"Inference_date: {str(forecast_date)}, daily_forecast_len={daily_forecast_length}"
    )

    # adi_nan_df = full_pred_df[full_pred_df.adi.isna()]
    # logger.info(f"Number of items with Nan history: {adi_nan_df.shape[0]}")
    # ### Assign forecast 0 for items created >= 3 months and no-sale
    # adi_nan_df["daily_pred_ts"] = adi_nan_df.apply(
    #     lambda row: (
    #         [0] * daily_forecast_length
    #         if row["created_time"] > 90
    #         else row.daily_pred_ts
    #     ),
    #     axis=1,
    # )
    # ### Assign forecast 0 for items with no recent-sale (in latest 6 months -> zero-sales)
    # no_recent_sale_df = full_pred_df[~full_pred_df.adi.isna()]
    # no_recent_sale_df["daily_pred_ts"] = no_recent_sale_df.apply(
    #     lambda row: (
    #         [0] * daily_forecast_length
    #         if sum(row["monthly_train_ts"][-6:]) == 0
    #         else row.daily_pred_ts
    #     ),
    #     axis=1,
    # )
    # full_pred_df = pd.concat([no_recent_sale_df, adi_nan_df])
    # full_pred_df = pd.concat([full_pred_df[~full_pred_df.adi.isna()], adi_nan_df])

    full_pred_df["daily_pred_ts"] = full_pred_df.apply(
        lambda row: (
            [0] * daily_forecast_length
            if (
                (pd.isna(row["adi"]) is True) and (row["created_time"] > 90)
            )  # all history no sale
            or (
                (pd.isna(row["adi"]) is False)
                and (sum(row["monthly_train_ts"][-6:]) == 0)
            )  # latest 6 months no sale -> zero-sales
            else row["daily_pred_ts"]
        ),
        axis=1,
    )

    logger.info(f"Shape of full_pred_df: {full_pred_df.shape[0]}")

    start_filter = time.time()
    # Filter channel_list and aggregate top-down
    channel_list = [
        "0",
        "580111",
        "c7d674e95445867e3488398e8b2cd2d8",
        "d724f9a653c53c6964282141d8fe9c84",
        "1",
        "910843e3c4805fe3e06524c14e939262",
    ]
    full_pred_df = full_pred_df[full_pred_df.channel_id.isin(channel_list)]
    logger.info(
        f"Full_pred_df after filtering channel_id: {full_pred_df.shape[0]}, {full_pred_df.id.unique().shape}"
    )

    logger.info("Clip pred_ts online channel <= all_channel...")
    final_full_pred_df = clip_channel_pred_smaller_than_all_channel_pred(full_pred_df)
    logger.info(
        f"Final_full_pred_df after clip online_channel pred_ts: {final_full_pred_df.shape[0]}, {final_full_pred_df.id.unique().shape}"
    )

    agg_result_df = aggregate_top_down_based_on_sale_distribution(
        final_full_pred_df, pred_column="daily_pred_ts"
    )
    logger.info(f"Full_pred_df after aggregation top-down: {agg_result_df.shape[0]}")
    logger.info(
        f"Time for filtering channel and aggregate top-down: {get_formatted_duration(time.time() - start_filter)}"
    )

    start_stockout = time.time()
    ### STEP 4: USE STOCKOUT FEATURE
    # Read and process stockout data
    logger.info("Process stockout dataset...")
    # data_folder = os.path.join(PROJECT_DATA_PATH, "downloaded_multiple_datasource")

    date_forecast_range = [date.strftime("%m-%d") for date in forecast_range]
    date_last_year = pd.to_datetime(forecast_date - relativedelta(years=1))
    stockout_df = pd.DataFrame()
    for brand in brand_list:
        config_dict["data"]["configs"]["name"] = brand
        brand_infer_data_handler = DataHandler(config_dict, subset="inference")
        brand_stock_df = brand_infer_data_handler.load_stock_data()
        brand_stock_df = brand_stock_df[
            (brand_stock_df.is_product == False)
            & (
                brand_stock_df.date.between(
                    date_last_year, pd.to_datetime(forecast_date)
                )
            )
        ]

        brand_stockout_df = brand_stock_df[brand_stock_df.is_stockout == 1]
        brand_stockout_df["stockout_id"] = brand_stockout_df.apply(
            lambda row: f"{row['platform']}_{row['variant_id']}", axis=1
        )
        brand_stockout_df["date"] = brand_stockout_df.date.apply(
            lambda x: x.strftime("%m-%d")
        )
        item_list = brand_stockout_df.stockout_id.unique().tolist()
        brand_stockout_df = brand_stockout_df.set_index(["date", "stockout_id"])
        brand_stockout_df = brand_stockout_df[~brand_stockout_df.index.duplicated()]

        multi_index = pd.MultiIndex.from_product([date_forecast_range, item_list])
        multi_index = multi_index.set_names(["date", "stockout_id"])
        brand_stockout_df = brand_stockout_df.reindex(multi_index).reset_index()
        brand_stockout_df["brand_name"] = brand

        stockout_df = pd.concat([stockout_df, brand_stockout_df])
        # stock_path = os.path.join(data_folder, brand, "product-ts.csv")
        # stock_df = (
        #     pd.read_csv(
        #         stock_path,
        #         parse_dates=["load_date"],
        #         dtype={"variant_id": "string", "stock": "float32"},
        #     )
        #     .drop(columns=["product_id"])
        #     .rename(columns={"from_source": "platform"})
        # )

        # # Concat stock data of ATuan
        # if brand in ["chanluu", "melinda_maria", "naked_cashmere", "honest_paws"]:
        #     logger.info(f"Concat more historical stock data for {brand}...")
        #     plus_stock_path = os.path.join(
        #         PROJECT_DATA_PATH, "stockout", f"{brand}.csv"
        #     )
        #     plus_stock_df = (
        #         pd.read_csv(
        #             plus_stock_path,
        #             parse_dates=["load_date"],
        #             dtype={"variant_id": "string", "stock": "float32"},
        #         )
        #         .drop(columns=["product_id"])
        #         .rename(columns={"from_source": "platform"})
        #     )
        #     plus_stock_df = plus_stock_df[
        #         plus_stock_df.load_date < stock_df.load_date.min()
        #     ]
        #     stock_df = pd.concat([plus_stock_df, stock_df])

        # stock_df = stock_df[
        #     (stock_df.load_date >= date_last_year)
        #     & (stock_df.load_date < pd.to_datetime(forecast_date))
        # ]
        if brand_stock_df.shape[0] > 0:
            logger.info(
                f"Brand {brand}: min_date={brand_stock_df.date.min()}, max_date={brand_stock_df.date.max()}"
            )
        else:
            logger.info(f"Brand {brand} no have stockout data")
            continue
        # stock_df = stock_df[~stock_df.variant_id.isna()]
        # stock_df = stock_df[stock_df.stock <= 0]
        # stock_df["stockout_id"] = stock_df.apply(
        #     lambda row: f"{row['platform']}_{row['variant_id']}", axis=1
        # )
        # item_list = stock_df.stockout_id.unique().tolist()
        # stock_df["stockout"] = 1
        # stock_df["date"] = stock_df.load_date.apply(lambda x: x.strftime("%m-%d"))
        # stock_df = stock_df.set_index(["date", "stockout_id"])

        # multi_index = pd.MultiIndex.from_product([date_forecast_range, item_list])
        # multi_index = multi_index.set_names(["date", "stockout_id"])
        # stock_df = stock_df.reindex(multi_index).reset_index()
        # stock_df["brand_name"] = brand

        # stock_df = stock_df.drop(columns=["stock", "load_date"])
        # stockout_df = pd.concat([stockout_df, stock_df])

    if stockout_df.shape[0] > 0:
        stockout_df = stockout_df.reset_index(drop=True)

        agg_product_df = agg_result_df[agg_result_df.is_product == True]
        agg_variant_df = agg_result_df[agg_result_df.is_product == False]

        # Process and merge stockout_ts to agg_variant_df
        stockout_df.is_stockout = stockout_df.is_stockout.fillna(value=0)
        stockout_df.is_stockout = stockout_df.is_stockout.astype(float)

        pivot_stockout_df = pd.pivot_table(
            stockout_df,
            index=["brand_name", "stockout_id"],
            values=["is_stockout"],
            aggfunc=lambda x: list(x),
        ).reset_index()
        agg_variant_df["stockout_id"] = agg_variant_df.apply(
            lambda row: f"{row['platform']}_{row['variant_id']}", axis=1
        )
        agg_variant_df = agg_variant_df.merge(
            pivot_stockout_df, on=["brand_name", "stockout_id"], how="left"
        )
        stockout_item_list = pivot_stockout_df.set_index(
            ["brand_name", "stockout_id"]
        ).index.tolist()
        logger.info(f"Number of variant with stockout: {len(stockout_item_list)}")

        # Run filling stockout daily_pred_ts for variant level
        logger.info(
            "Fill stockout daily_pred_ts with average quantity over previous days..."
        )
        agg_variant_df["daily_pred_ts"] = agg_variant_df.apply(
            lambda row: (
                fill_stockout(
                    stockout_ts=row.is_stockout,
                    daily_train_ts=row.daily_train_ts,
                    daily_pred_ts=row.daily_pred_ts,
                    item_id=row.id,
                )
                if (row.brand_name, row.stockout_id) in stockout_item_list
                else row.daily_pred_ts
            ),
            axis=1,
        )
        logger.info(
            f"Time for filling stockout predictions: {get_formatted_duration(time.time() - start_stockout)}"
        )
        stockout_result_df = pd.concat([agg_product_df, agg_variant_df])

        # Check again if online_pred > all_channel_pred
        final_stockout_result_df = clip_channel_pred_smaller_than_all_channel_pred(
            stockout_result_df
        )
        final_stockout_result_df["daily_pred_ts"] = final_stockout_result_df.apply(
            lambda row: tweak_variant(row["daily_pred_ts"], row["id"]), axis=1
        )

        logger.info("Aggregate bottom-up again...")
        final_agg_result_df = aggregate_bottom_up(
            final_stockout_result_df, pred_column="daily_pred_ts"
        )
    else:
        final_agg_result_df = agg_result_df

    final_agg_result_df["monthly_pred_ts"] = final_agg_result_df.apply(
        lambda x: aggregate_daily_ts_to_monthly(
            daily_ts=x.daily_pred_ts, first_date=x.first_pred_date
        ),
        axis=1,
    )
    logger.info(f"Final_agg_result_df: {final_agg_result_df.shape}")

    # Save results
    level_list = ["variant", "product"]
    if csv_file is False:
        # Generate and save results to json
        for brand in brand_list:
            logger.info(f"Generate results for {brand}...")
            brand_df = final_agg_result_df[final_agg_result_df.brand_name == brand]

            # Map variant_id with product_id
            variant_df = brand_df[brand_df.is_product == False].drop_duplicates(
                subset=["variant_id"]
            )
            dict_variant_product = dict(
                zip(variant_df.variant_id, variant_df.product_id)
            )

            # Calculate accuracy_score
            logger.info(f"Monitor accuracy for {brand}")
            forecast_monitor = ForecastMonitor(brand)
            # Accuracy last_3_months for confidence_score
            results = forecast_monitor.get_forecast_accuracy_all_items(
                period="3 months",
                method="mape",
                group_method="sale_category",
                forecast_date=forecast_date,
            )

            if results is not None:
                error_results = results["error_results"]
                logger.info(f"Len of last_3_months_acc_result: {len(error_results)}")
            else:
                logger.info("No have enough forecast for monitoring by month")
                error_results = None

            brand_save_path = Path(result_save_dir) / brand
            os.makedirs(brand_save_path, exist_ok=True)

            for level in level_list:
                result_list = []
                if level == "variant":
                    field_name = "variant_id"
                    full_level_df = brand_df[brand_df.is_product == False]
                    full_level_df["item_id"] = full_level_df.variant_id
                else:
                    field_name = "product_id"
                    full_level_df = brand_df[brand_df.is_product == True]
                    full_level_df["item_id"] = full_level_df.product_id
                logger.info(
                    f"Number of unique {level} IDs:{full_level_df.id.unique().shape}"
                )
                logger.info(
                    f"Shape of {level} level's dataframe: {full_level_df.shape}"
                )

                for row in tqdm(
                    full_level_df.itertuples(),
                    total=full_level_df.shape[0],
                    desc=f"Generating results of {level} level...",
                ):
                    result = {
                        field_name: row.item_id,
                        "h_key": row.h_key if pd.isna(row.h_key) == False else None,
                        "from_source": row.platform,
                        "channel_id": row.channel_id,
                        "forecast_date": str(forecast_date),
                        "weekly_historical_val": str(row.train_ts),
                        "monthly_historical_val": str(row.monthly_train_ts),
                        "monthly_prediction_val": str(row.monthly_pred_ts),
                        "predictions": {
                            "sale_per_day": row.sale_per_day,
                            "forecast_val": str(row.daily_pred_ts),
                            "trend": None,
                        },
                    }
                    if (error_results is not None) and (row.id in error_results.keys()):
                        result["sale_pattern"] = error_results[row.id]["sale_pattern"]
                        result["confidence_score"] = error_results[row.id][
                            "confidence_score"
                        ]
                    else:
                        result["sale_pattern"] = None
                        result["confidence_score"] = None

                    # Append result into the list of results
                    result_list.append(result)
                logger.info(f"Number of {level} results: {len(result_list)}")

                # Sum forecast all_channel
                # if "0" not in full_level_df.channel_id.unique():
                if brand in BRANDS_TO_CREATE_ALL_CHANNEL:
                    logger.info(f"Generate forecast result for {brand} all_channel...")
                    ids_list = list({result[field_name] for result in result_list})
                    for item in tqdm(ids_list, total=len(ids_list)):
                        item_results = [
                            res for res in result_list if res[field_name] == item
                        ]
                        weekly_history_list = [
                            ast.literal_eval(res["weekly_historical_val"])
                            for res in item_results
                        ]
                        len_weekly_history = min([len(x) for x in weekly_history_list])
                        weekly_history_list = [
                            x[-len_weekly_history:] for x in weekly_history_list
                        ]
                        monthly_history_list = [
                            ast.literal_eval(res["monthly_historical_val"])
                            for res in item_results
                        ]
                        len_history = min([len(x) for x in monthly_history_list])
                        monthly_history_list = [
                            x[-len_history:] for x in monthly_history_list
                        ]
                        monthly_prediction_list = [
                            ast.literal_eval(res["monthly_prediction_val"])
                            for res in item_results
                        ]
                        sale_per_day_list = [
                            res["predictions"]["sale_per_day"] for res in item_results
                        ]
                        forecast_list = [
                            ast.literal_eval(res["predictions"]["forecast_val"])
                            for res in item_results
                        ]
                        platform = item_results[0]["from_source"]
                        h_key = item_results[0]["h_key"]

                        if level == "product":
                            unique_id = brand + "_" + platform + "_" + item + "_NA_0"
                        else:
                            unique_id = (
                                brand
                                + "_"
                                + platform
                                + "_"
                                + dict_variant_product[item]
                                + "_"
                                + item
                                + "_0"
                            )

                        all_channel_result = {
                            field_name: item,
                            "h_key": h_key,
                            "from_source": platform,
                            "channel_id": "0",
                            "forecast_date": str(forecast_date),
                            "weekly_historical_val": str(
                                [round(sum(x), 2) for x in zip(*weekly_history_list)]
                            ),
                            "monthly_historical_val": str(
                                [round(sum(x), 2) for x in zip(*monthly_history_list)]
                            ),
                            "monthly_prediction_val": str(
                                [
                                    round(sum(x), 2)
                                    for x in zip(*monthly_prediction_list)
                                ]
                            ),
                            "predictions": {
                                "sale_per_day": round(np.sum(sale_per_day_list), 7),
                                "forecast_val": str(
                                    [round(sum(x), 7) for x in zip(*forecast_list)]
                                ),
                                "trend": None,
                            },
                        }
                        if (error_results is not None) and (
                            unique_id in error_results.keys()
                        ):
                            all_channel_result["sale_pattern"] = error_results[
                                unique_id
                            ]["sale_pattern"]
                            all_channel_result["confidence_score"] = error_results[
                                unique_id
                            ]["confidence_score"]
                        else:
                            all_channel_result["sale_pattern"] = None
                            all_channel_result["confidence_score"] = None

                        result_list.append(all_channel_result)
                    logger.info(
                        f"Number of {level} results after generating all_channel results: {len(result_list)}"
                    )

                # Saving results
                logger.info(f"Saving results for {level} level...")
                save_path = Path(brand_save_path) / f"{level}_result_forecast.json"
                try:
                    with open(save_path, "w") as file:
                        json.dump(result_list, file, cls=NpEncoder, indent=4)
                except Exception as err:
                    logger.exception("An error occured while saving: ", err)
                    raise
                else:
                    logger.info(
                        f"Successfully saved results to {Path(brand_save_path).absolute()}"
                    )

    else:
        # Save results to csv
        for brand in brand_list:
            logger.info(f"Generate prediction dataframe for {brand}...")
            brand_df = final_agg_result_df[final_agg_result_df.brand_name == brand]

            brand_save_path = Path(result_save_dir) / brand
            os.makedirs(brand_save_path, exist_ok=True)

            brand_config = [
                cf for cf in trend_config_dict if cf["brand_name"] == brand
            ][0]
            trend_history_length = brand_config["trend_history_length"]

            brand_df["item_id"] = brand_df.apply(
                lambda x: (
                    x["product_id"] if x["is_product"] == True else x["variant_id"]
                ),
                axis=1,
            )
            brand_df = brand_df[
                [
                    "item_id",
                    "is_product",
                    "channel_id",
                    "first_pred_date",
                    "sale_per_day",
                    "daily_pred_ts",
                    "monthly_train_ts",
                    "monthly_pred_ts",
                ]
            ]

            history_len_dict = dict()
            for item in brand_df.item_id.unique():
                history_list = brand_df[
                    brand_df.item_id == item
                ].monthly_train_ts.to_list()
                history_len = min([len(x) for x in history_list])
                history_len_dict[item] = history_len
            brand_df["monthly_train_ts"] = brand_df.apply(
                lambda x: x["monthly_train_ts"][-history_len_dict[x["item_id"]] :],
                axis=1,
            )

            brand_df["daily_pred_ts"] = brand_df["daily_pred_ts"].apply(
                lambda x: np.array(x)
            )
            brand_df["monthly_train_ts"] = brand_df["monthly_train_ts"].apply(
                lambda x: np.array(x)
            )
            brand_df["monthly_pred_ts"] = brand_df["monthly_pred_ts"].apply(
                lambda x: np.array(x)
            )

            # Sum foreast all_channel
            if "0" not in brand_df.channel_id.unique():
                logger.info(f"Generate forecast result for {brand} all_channel...")
                all_channel_brand_df = (
                    brand_df.groupby(by=["item_id", "is_product"])
                    .agg(
                        {
                            "sale_per_day": np.sum,
                            "daily_pred_ts": np.sum,
                            "monthly_train_ts": np.sum,
                            "monthly_pred_ts": np.sum,
                        }
                    )
                    .reset_index()
                )
                all_channel_brand_df["channel_id"] = "0"
                all_channel_brand_df["first_pred_date"] = forecast_date
                brand_df = pd.concat([brand_df, all_channel_brand_df]).reset_index(
                    drop=True
                )

            logger.info(f"Generate trend dataframe for {brand}...")
            trend_df = pd.DataFrame()
            for level in level_list:
                level_result_df = (
                    brand_df[brand_df.is_product == True]
                    if level == "product"
                    else brand_df[brand_df.is_product == False]
                )
                level_trend_df = detect_trend(
                    level_result_df, forecast_date, trend_history_length
                )
                trend_df = pd.concat([trend_df, level_trend_df]).reset_index(drop=True)

            trend_df = trend_df[["item_id", "is_product", "channel_id", "trend"]]

            final_pred_df = pd.concat(
                pd.DataFrame(
                    {
                        "item_id": row.item_id,
                        "is_product": row.is_product,
                        "channel_id": row.channel_id,
                        "date": forecast_range,
                        "sale_per_day": row.sale_per_day,
                        "forecast_value": list(row.daily_pred_ts),
                    }
                )
                for row in brand_df.itertuples()
            )

            logger.info("Saving results...")
            pred_save_path = Path(brand_save_path) / "predictions.csv"
            final_pred_df.to_csv(pred_save_path, index=False)
            trend_save_path = Path(brand_save_path) / "trend.csv"
            trend_df.to_csv(trend_save_path, index=False)

    ela = time.time() - start
    logger.info(f"Time for generation results: {get_formatted_duration(ela)}")
