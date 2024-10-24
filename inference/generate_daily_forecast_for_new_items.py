import ast
import glob
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import MO, relativedelta
from tqdm import tqdm

from forecasting import PROJECT_ROOT_PATH, S3_PROCESSED_DATA_URI
from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler
from forecasting.evaluation.aggregation_utils import (
    aggregate_bottom_up_update,
    aggregate_daily_ts_to_monthly,
    clip_channel_pred_smaller_than_all_channel_pred,
)
from forecasting.util import NpEncoder, get_formatted_duration
from forecasting.utils.common_utils import load_dict_from_yaml, save_dict_to_json
from inference.find_similar_items_with_image import (
    find_similar_set_for_new_variants_with_image,
    preprocess_image_urls,
)
from inference.trend_detection import run_single_level_detection

logger.info(S3_PROCESSED_DATA_URI)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.random.seed(42)

NEW_BRANDS = ["ipolita"]
BRANDS_TO_CREATE_ALL_CHANNEL = ["mizmooz", "as98"]
prediction_col = "predictions"

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT_PATH, "config", "sample_config.yaml")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT_PATH, "saved_models/TFT")
DEFAULT_JSON_FORECAST_DIR = os.path.join(
    PROJECT_ROOT_PATH, "forecasting", "visualization"
)


def create_all_channel_metadata(meta_df):
    all_channel_df = meta_df[meta_df.brand_name.isin(BRANDS_TO_CREATE_ALL_CHANNEL)]
    groupby_subset = ["brand_name", "platform", "product_id", "variant_id"]
    aggregate_fields = [
        field for field in meta_df.columns if field not in groupby_subset
    ]
    aggregate_fields = [
        field for field in aggregate_fields if field not in ["id", "channel_id"]
    ]
    aggregate_function_dict = dict.fromkeys(aggregate_fields, "first")

    all_channel_df = (
        all_channel_df.groupby(groupby_subset)
        .agg(aggregate_function_dict)
        .reset_index()
    )
    all_channel_df["channel_id"] = "0"
    all_channel_df["id"] = all_channel_df.apply(
        lambda row: row["brand_name"]
        + "_"
        + row["platform"]
        + "_"
        + row["product_id"]
        + "_"
        + row["variant_id"]
        + "_"
        + row["channel_id"],
        axis=1,
    )
    full_meta_df = pd.concat([meta_df, all_channel_df])
    return full_meta_df


def load_and_process_json_forecast_results(
    forecast_result_folder, brand, level, meta_df
):
    forecast_path = os.path.join(
        forecast_result_folder, brand, f"{level}_result_forecast.json"
    )
    with open(forecast_path) as file:
        result_list = json.load(file)

    for res in result_list:
        res["daily_pred_ts"] = res["predictions"]["forecast_val"]
        res["trend"] = res["predictions"]["trend"]
        res["platform"] = res["from_source"]

    result_df = pd.DataFrame(result_list)
    result_df = result_df.drop(columns=["predictions", "slope", "beta", "from_source"])
    result_df.daily_pred_ts = result_df.daily_pred_ts.apply(
        lambda x: ast.literal_eval(x)
    )
    result_df["brand_name"] = brand
    level_meta_df = (
        meta_df[meta_df.is_product == True]
        if level == "product"
        else meta_df[meta_df.is_product == False]
    )
    result_df = result_df.merge(
        level_meta_df[
            [
                "brand_name",
                "platform",
                "product_id",
                "variant_id",
                "channel_id",
                "id",
                "is_product",
            ]
        ],
        on=["brand_name", "platform", f"{level}_id", "channel_id"],
        how="left",
    )

    return result_df


def adjust_variant_forecast_from_similar_items_avg(result_df, new_variant_df):
    adjust_pred_dict = {}
    for item_pred in tqdm(
        new_variant_df.itertuples(),
        total=new_variant_df.shape[0],
        desc="Adjusting products forecast based on similar products",
    ):
        item_id = item_pred.id
        similar_items_of_item = item_pred.similar_items

        if similar_items_of_item is not None:
            similar_items_df = result_df[result_df["id"].isin(similar_items_of_item)]
            similar_items_preds = np.stack(
                similar_items_df["daily_pred_ts"].apply(np.array), axis=0
            )
            daily_avg_sim_forecast = np.mean(similar_items_preds, axis=0).tolist()
            adjust_pred_dict[item_id] = daily_avg_sim_forecast

    return adjust_pred_dict


def find_similar_item_for_new_variant(
    full_meta_df, seasonal_items_list, similar_items_dict, new_items_list
):
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
    full_meta_df = preprocess_image_urls(full_meta_df)

    new_variants_without_similar_items = [
        item_id
        for item_id in new_items_list
        if item_id not in similar_items_dict.keys()
    ]
    if len(new_variants_without_similar_items) > 0:
        logger.info("Find similar items for newly-launched items")
        new_similar_variant_dict = find_similar_set_for_new_variants_with_image(
            meta_df=full_meta_df,
            lookup_item_list=seasonal_items_list,
            new_items_list=new_variants_without_similar_items,
            attribute_weights=attribute_weights,
            image_model_name="facebook/dinov2-small",
            verbose=False,
        )
        similar_items_dict.update(new_similar_variant_dict)

    return similar_items_dict


def save_results_and_detect_trend_for_each_brand(
    df, result_save_dir, all_channel_old_product_result_df
):
    level_list = ["variant", "product"]
    brand = df.brand_name.unique()[0]
    if all_channel_old_product_result_df is not None:
        dict_trend = dict(
            zip(
                all_channel_old_product_result_df.product_id,
                all_channel_old_product_result_df.trend,
            )
        )
        dict_sale_pattern = dict(
            zip(
                all_channel_old_product_result_df.product_id,
                all_channel_old_product_result_df.sale_pattern,
            )
        )
        dict_confidence_score = dict(
            zip(
                all_channel_old_product_result_df.product_id,
                all_channel_old_product_result_df.confidence_score,
            )
        )

    for level in level_list:
        logger.info(f"Generate results for level {level} - {brand}...")
        result_list = []
        if level == "variant":
            field_name = "variant_id"
            full_level_df = df[df.is_product == False]
            full_level_df["item_id"] = full_level_df.variant_id
        else:
            field_name = "product_id"
            full_level_df = df[df.is_product == True]
            full_level_df["item_id"] = full_level_df.product_id

        for row in tqdm(
            full_level_df.itertuples(),
            total=full_level_df.shape[0],
            desc=f"Generating results of {level} level...",
        ):
            result = {
                field_name: row.item_id,
                "h_key": row.h_key if pd.isna(row.h_key) is False else None,
                "from_source": row.platform,
                "channel_id": row.channel_id,
                "forecast_date": str(row.forecast_date),
                "daily_historical_val": (
                    str(row.daily_historical_val)
                    if row.daily_historical_val is not None
                    else None
                ),
                "monthly_prediction_val": str(row.monthly_pred_ts),
                "predictions": {
                    "sale_per_day": row.sale_per_day,
                    "forecast_val": str(row.daily_pred_ts),
                    "trend": row.trend if pd.isna(row.trend) is False else None,
                },
                "sale_pattern": (
                    row.sale_pattern if pd.isna(row.sale_pattern) is False else None
                ),
                "confidence_score": (
                    row.confidence_score
                    if pd.isna(row.confidence_score) is False
                    else None
                ),
            }

            # Append result into the list of results
            result_list.append(result)

        # Sum forecast all_channel
        # if "0" not in full_level_df.channel_id.unique():
        if brand in BRANDS_TO_CREATE_ALL_CHANNEL:
            logger.info(f"Generate forecast result for {brand} all_channel...")
            ids_list = list({result[field_name] for result in result_list})
            for item in tqdm(ids_list, total=len(ids_list)):
                item_results = [res for res in result_list if res[field_name] == item]
                daily_history_list = [
                    ast.literal_eval(res["daily_historical_val"])
                    for res in item_results
                ]
                len_daily_history = min([len(x) for x in daily_history_list])
                daily_history_list = [
                    x[-len_daily_history:] for x in daily_history_list
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
                forecast_date = item_results[0]["forecast_date"]
                trend = (
                    dict_trend[item]
                    if item in dict_trend.keys()
                    else item_results[0]["predictions"]["trend"]
                )
                sale_pattern = (
                    dict_sale_pattern[item]
                    if item in dict_sale_pattern.keys()
                    else item_results[0]["sale_pattern"]
                )
                confidence_score = (
                    dict_confidence_score[item]
                    if item in dict_confidence_score.keys()
                    else item_results[0]["confidence_score"]
                )

                all_channel_result = {
                    field_name: item,
                    "h_key": h_key,
                    "from_source": platform,
                    "channel_id": "0",
                    "forecast_date": forecast_date,
                    "daily_historical_val": str(
                        [round(sum(x), 2) for x in zip(*daily_history_list)]
                    ),
                    "monthly_prediction_val": str(
                        [round(sum(x), 2) for x in zip(*monthly_prediction_list)]
                    ),
                    "predictions": {
                        "sale_per_day": round(np.sum(sale_per_day_list), 7),
                        "forecast_val": str(
                            [round(sum(x), 7) for x in zip(*forecast_list)]
                        ),
                        "trend": trend if pd.isna(trend) is False else None,
                    },
                    "sale_pattern": (
                        sale_pattern if pd.isna(sale_pattern) is False else None
                    ),
                    "confidence_score": (
                        confidence_score if pd.isna(confidence_score) is False else None
                    ),
                }
                result_list.append(all_channel_result)

        # if level == "product":
        # final_list = [res for res in result_list if res["item_id"] in list_no_need_to_detect_trend]
        # result_list = [res for res in result_list if res["item_id"] not in list_no_need_to_detect_trend]
        # else:
        #     final_list = []

        result_list_no_need_to_detect_trend = [
            res for res in result_list if res["predictions"]["trend"] is not None
        ]
        result_list = [
            res for res in result_list if res["predictions"]["trend"] is None
        ]

        logger.info(
            f"Result list no need to detect_trend: {len(result_list_no_need_to_detect_trend)}"
        )
        logger.info(f"Result list for detecting trend: {len(result_list)}")

        if len(result_list) > 0:
            # Detect trend for level result_list
            _, result_list = run_single_level_detection(
                result_list=result_list,
                historical_length=None,
                future_length=0,
                frequency_point="daily",
            )
        final_list = [*result_list_no_need_to_detect_trend, *result_list]
        logger.info(f"Final result_list: {len(final_list)}")

        # Save results
        logger.info(f"Saving results for {level} level...")
        save_path = Path(result_save_dir) / f"{level}_result_forecast.json"
        try:
            with open(save_path, "w") as file:
                json.dump(final_list, file, cls=NpEncoder, indent=4)
        except Exception as err:
            logger.exception("An error occured while saving: ", err)
            raise
        else:
            logger.info(
                f"Successfully saved results to {Path(result_save_dir).absolute()}"
            )


def main():

    logger.info("Loading config")
    # Load config
    config_dict = load_dict_from_yaml(DEFAULT_CONFIG_PATH)
    today = datetime.today().date()
    inference_data_date = ("").join(str(today).split("-"))

    config_dict["data"]["configs"]["version"] = inference_data_date
    config_dict["data"]["configs"]["freq"] = "D"
    logger.info(config_dict["data"]["configs"])

    infer_data_handler = DataHandler(config_dict, subset="full")
    meta_df = infer_data_handler.load_metadata(drop_price=False)
    meta_df.variant_id = meta_df.apply(
        lambda row: row["variant_id"] if row["is_product"] == False else "NA", axis=1
    )

    # Create forecast_result for partial folder
    result_save_dir = os.path.join(
        DEFAULT_JSON_FORECAST_DIR, f"{inference_data_date}_partial"
    )
    os.makedirs(result_save_dir, exist_ok=True)
    logger.info(f"Save directory: {result_save_dir}")

    # Check for the latest model folder -> Load seasonal_list and similar_items_list
    latest_model_folder = max(Path(DEFAULT_MODEL_DIR).glob("*/"), key=os.path.getmtime)
    model_dir = os.path.join(latest_model_folder, "best_model")
    logger.info(f"Model dir: {model_dir}")

    seasonal_file_path, seasonal_items_list = (
        infer_data_handler.create_or_load_seasonal_item_list(model_dir)
    )

    similar_file_paths = glob.glob(os.path.join(model_dir, "similar_product_dict.json"))
    if len(similar_file_paths) >= 1:
        similar_items_file = similar_file_paths[0]
        similar_items_dict = load_dict_from_yaml(similar_items_file)
    else:
        similar_items_dict = None
    logger.info(
        f"Seasonal list: {len(seasonal_items_list)}, Similar items dict: {len(similar_items_dict)}"
    )

    # Check for JSON forecast results: maximum 3 weeks ago
    for i in range(1, 4):
        last_monday = datetime.today().date() + relativedelta(weekday=MO(-i))
        latest_forecast_folder_name = ("").join(str(last_monday).split("-"))
        latest_forecast_dir = os.path.join(
            DEFAULT_JSON_FORECAST_DIR, latest_forecast_folder_name
        )
        if os.path.isdir(latest_forecast_dir):
            logger.info(
                f"Latest JSON forecast results: {latest_forecast_dir} version {str(last_monday)}"
            )
            break

    # Check if brand has new_items
    new_item_meta_df = meta_df[
        meta_df.created_date.between(str(last_monday), str(today))
    ]
    logger.info(
        f"New items created from {new_item_meta_df.created_date.min()} to {new_item_meta_df.created_date.max()}"
    )
    brand_list = new_item_meta_df.brand_name.unique().tolist()
    logger.info(f"Brand_list with new_items: {brand_list}")

    for brand in brand_list:
        logger.info(f"Generate daily forecast for {brand}")
        brand_meta_df = meta_df[meta_df.brand_name == brand]
        brand_new_item_meta_df = new_item_meta_df[new_item_meta_df.brand_name == brand]

        new_variant_df = brand_new_item_meta_df[
            brand_new_item_meta_df.is_product == False
        ]
        new_product_df = brand_new_item_meta_df[
            brand_new_item_meta_df.is_product == True
        ]
        new_variant_list = new_variant_df.id.unique().tolist()
        old_product_with_new_variant_list = [
            prod_id
            for prod_id in new_variant_df.product_id.unique()
            if prod_id not in new_product_df.product_id.unique()
        ]
        product_list = brand_new_item_meta_df.product_id.unique().tolist()

        # Find similar items for new variant
        similar_items_dict = find_similar_item_for_new_variant(
            full_meta_df=brand_meta_df,
            seasonal_items_list=seasonal_items_list,
            similar_items_dict=similar_items_dict,
            new_items_list=new_variant_list,
        )
        save_dict_to_json(similar_items_dict, similar_items_file)

        new_variant_df["similar_items"] = new_variant_df.id.apply(
            lambda x: similar_items_dict[x]
        )

        logger.info(
            "Load latest forecast results and generate forecast with similar_items_avg"
        )
        # Load JSON forecast results
        latest_variant_result_df = load_and_process_json_forecast_results(
            forecast_result_folder=latest_forecast_dir,
            brand=brand,
            level="variant",
            meta_df=brand_meta_df,
        )
        latest_product_result_df = load_and_process_json_forecast_results(
            forecast_result_folder=latest_forecast_dir,
            brand=brand,
            level="product",
            meta_df=brand_meta_df,
        )
        latest_forecast_date = pd.to_datetime(
            latest_product_result_df.forecast_date.unique()[0]
        )
        delta_date = (pd.to_datetime(today) - latest_forecast_date).days
        latest_variant_result_df.daily_pred_ts = (
            latest_variant_result_df.daily_pred_ts.apply(lambda x: x[delta_date:])
        )
        latest_product_result_df.daily_pred_ts = (
            latest_product_result_df.daily_pred_ts.apply(lambda x: x[delta_date:])
        )
        daily_pred_length = len(latest_product_result_df.daily_pred_ts.values[0])

        # Get average forecast of similar items
        new_variant_result_df = new_variant_df
        new_variant_result_df["forecast_date"] = str(today)
        adjust_similar_daily_pred_dict = adjust_variant_forecast_from_similar_items_avg(
            result_df=latest_variant_result_df, new_variant_df=new_variant_result_df
        )
        new_variant_result_df["daily_pred_ts"] = new_variant_result_df.id.apply(
            lambda x: (
                adjust_similar_daily_pred_dict[x]
                if x in adjust_similar_daily_pred_dict.keys()
                else [0] * daily_pred_length
            )
        )

        # Blend with sale_per_day
        logger.info("Load daily_dataset and scale forecast with sale_per_day")
        brand_config_dict = config_dict.copy()
        brand_config_dict["data"]["configs"]["name"] = brand

        daily_infer_data_handler = DataHandler(brand_config_dict, "full")
        daily_infer_df = daily_infer_data_handler.load_data()

        daily_infer_df = daily_infer_df[daily_infer_df.id.isin(new_variant_list)]
        daily_pivot_df = (
            pd.pivot_table(
                daily_infer_df,
                index="id",
                values=["quantity_order"],
                aggfunc=lambda x: list(x),
            )
            .rename(columns={"quantity_order": "daily_historical_val"})
            .reset_index()
        )
        daily_pivot_df["sale_per_day"] = daily_pivot_df.daily_historical_val.apply(
            lambda x: np.mean(np.array(x))
        )

        new_variant_result_df = new_variant_result_df.merge(
            daily_pivot_df, on="id", how="left"
        )
        new_variant_result_df["daily_pred_ts"] = new_variant_result_df.apply(
            lambda row: (
                row["sale_per_day"] * 0.5 + np.array(row["daily_pred_ts"]) * 0.5
                if row["sale_per_day"]
                != 0  # Just scale with recent sale if item has sale
                else row["daily_pred_ts"]
            ),
            axis=1,
        )
        new_variant_result_df = clip_channel_pred_smaller_than_all_channel_pred(
            new_variant_result_df
        )

        new_item_result_df = pd.concat([new_variant_result_df, new_product_df])
        new_item_result_df["trend"] = None
        new_item_result_df["sale_pattern"] = "newly-launched"
        new_item_result_df["confidence_score"] = None

        logger.info("Aggregate bottom-up")
        # Aggregate bottom-up for product forecast
        # Get latest_forecast_result of old_product and its old_variant for updating forecast with new variant
        # if len(old_product_with_new_variant_list) > 0:
        #     old_product_df = latest_product_result_df[
        #         latest_product_result_df.product_id.isin(
        #             old_product_with_new_variant_list
        #         )
        #     ]
        #     old_variant_df = latest_variant_result_df[
        #         latest_variant_result_df.product_id.isin(
        #             old_product_with_new_variant_list
        #         )
        #     ]
        #     old_product_with_new_variant_latest_result_df = pd.concat(
        #         [old_product_df, old_variant_df]
        #     )

        #     aggregate_df = pd.concat(
        #         [new_item_result_df, old_product_with_new_variant_latest_result_df]
        #     )
        # else:
        #     aggregate_df = new_item_result_df
        old_product_df = latest_product_result_df[
            latest_product_result_df.product_id.isin(product_list)
        ]
        old_variant_df = latest_variant_result_df[
            latest_variant_result_df.product_id.isin(product_list)
        ]
        old_item_with_latest_result_df = pd.concat([old_product_df, old_variant_df])
        aggregate_df = pd.concat([new_item_result_df, old_item_with_latest_result_df])

        aggregate_df.forecast_date = str(today)

        aggregate_df = aggregate_bottom_up_update(aggregate_df, "daily_pred_ts")
        aggregate_df = aggregate_bottom_up_update(aggregate_df, "daily_historical_val")
        aggregate_df = aggregate_bottom_up_update(aggregate_df, "sale_per_day")

        # Filter items for final results
        final_df = pd.concat(
            [
                aggregate_df[aggregate_df.id.isin(new_item_result_df.id.unique())],
                aggregate_df[
                    (aggregate_df.product_id.isin(old_product_with_new_variant_list))
                    & (aggregate_df.is_product == True)
                ],
            ]
        )
        logger.info(f"Shape of Brand_new_item_meta_df: {new_item_result_df.shape[0]}")
        logger.info(
            f"Number of old product with new_variant_list: {len(old_product_with_new_variant_list)}"
        )
        logger.info(f"Shape of final_df: {final_df.shape[0]}")

        final_df["monthly_pred_ts"] = final_df.apply(
            lambda row: aggregate_daily_ts_to_monthly(
                daily_ts=row["daily_pred_ts"], first_date=row["forecast_date"]
            ),
            axis=1,
        )

        if brand in BRANDS_TO_CREATE_ALL_CHANNEL:
            all_channel_old_product_latest_result_df = latest_product_result_df[
                (
                    latest_product_result_df.product_id.isin(
                        old_product_with_new_variant_list
                    )
                )
                & (latest_product_result_df.channel_id == "0")
            ]
        else:
            all_channel_old_product_latest_result_df = None

        logger.info("Save results and detect trend by day")
        brand_save_dir = Path(result_save_dir) / brand
        os.makedirs(brand_save_dir, exist_ok=True)

        final_df.to_parquet("test_final_df.parquet")
        save_results_and_detect_trend_for_each_brand(
            final_df, brand_save_dir, all_channel_old_product_latest_result_df
        )

    return result_save_dir, inference_data_date


if __name__ == "__main__":

    start = time.time()
    result_save_dir, inference_data_date = main()
    ela = time.time() - start
    logger.info(f"Time for inference: {get_formatted_duration(ela)}")
