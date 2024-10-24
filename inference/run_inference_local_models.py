import argparse
import ast
import copy
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from forecasting import MODEL_DIR, PROJECT_ROOT_PATH
from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler, merge_updated_data
from forecasting.data.util import calc_sale_per_day, remove_leading_zeros
from forecasting.evaluation.aggregation_utils import (
    aggregate_bottom_up,
    aggregate_daily_ts_to_monthly,
    aggregate_top_down_based_on_sale_distribution,
    aggregate_weekly_ts_to_daily,
    build_result_df_from_pd,
)
from forecasting.monitor.forecast_monitor import ForecastMonitor

# from forecasting.data.utils.darts_utils import convert_df_to_darts_dataset
from forecasting.util import (
    NpEncoder,
    get_formatted_duration,
    get_latest_date,
    import_class,
)
from forecasting.utils.common_utils import load_dict_from_yaml
from inference.utils import (
    BRANDS_TO_CREATE_ALL_CHANNEL,
    MULTIPLE_SOURCE_BRAND_LIST,
    clip_channel_pred_smaller_than_all_channel_pred,
    detect_trend,
    fill_stockout_update,
)

YAML_CONFIGS_DIR = (
    Path(__file__).resolve().parents[1] / "forecasting" / "data" / "yaml_configs"
)


def _setup_parser():
    """Setup Argument Parser"""
    parser = argparse.ArgumentParser(description="Script to run inference")

    parser.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        default=1,
        required=False,
        help="Number of times a prediction is sampled from a probabilistic model. Default=1",
    )
    parser.add_argument(
        "-id", "--inference_data_date", type=str, default=None, required=False
    )
    parser.add_argument("-wd", "--model_dir", type=str, default=None, required=False)
    parser.add_argument(
        "-s",
        "--result_save_dir",
        required=False,
        help="Directory of json forecast results for all brands",
    )
    parser.add_argument(
        "-cf",
        "--config_file_path",
        type=str,
        required=False,
        default="unified_model/config.yaml",
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        default=False,
        help="Only generate predictions, do not generate results",
    )

    return parser


def main():
    """Inference main function"""
    parser = _setup_parser()
    args = parser.parse_args()

    kwargs = vars(args)

    total_start = time.time()

    # num_samples = kwargs["num_samples"]
    model_dir = kwargs["model_dir"]
    inference_data_date = kwargs["inference_data_date"]
    result_save_dir = kwargs["result_save_dir"]
    config_file_path = kwargs["config_file_path"]
    predict_only = kwargs["predict_only"]

    # Read file config
    config_path = YAML_CONFIGS_DIR / config_file_path
    with open(config_path, encoding="utf-8") as f:
        all_config = yaml.safe_load(f)
    trend_data_config = all_config["data"]

    default_model_dir = MODEL_DIR
    if model_dir is None:
        logger.warning(
            f"Keyword argument `model_dir` not found, looking for latest trained model at {default_model_dir}"
        )
        # Get latest trained model
        latest_model_folder = get_latest_date(default_model_dir)
        model_dir = os.path.join(default_model_dir, latest_model_folder, "best_model")

    prediction_save_dir = os.path.join(
        model_dir, "inference_results", inference_data_date
    )
    os.makedirs(prediction_save_dir, exist_ok=True)

    if result_save_dir is None:
        logger.warning(
            f"Keyword argument `result_save_dir` not found, looking for latest trained model at {default_model_dir}"
        )
        result_save_dir = os.path.join(
            model_dir, "inference_results", inference_data_date
        )
    os.makedirs(result_save_dir, exist_ok=True)

    logger.add(
        os.path.join(model_dir, "inference.log"),
        format="{time} - {file.name} - {level} - {message}",
    )
    # Get config file from model_dir
    config_file = glob.glob(os.path.join(model_dir, "*.yaml"))

    assert (
        len(config_file) == 1
    ), f"There should be 1 config file, found {len(config_file)}."

    config_file = config_file[0]
    config_dict = load_dict_from_yaml(config_file)
    config_dict["data"]["configs"]["version"] = inference_data_date

    # Load config
    data_config = config_dict["data"]
    model_config = config_dict["model"]
    model_class_str = model_config["class"]

    # Data configs
    data_configs = data_config["configs"]
    prediction_length = data_configs["prediction_length"]
    data_configs["version"] = inference_data_date
    print(f"Data configs: {data_configs}")

    # Model configs
    model_type = model_config["model_type"]
    basic_configs = model_config["configs"]
    model_name = basic_configs.get("name")
    verbose = basic_configs.get("verbose", False)

    assert model_type == "local", "Only model_type `local` is supported"

    logger.info(f"Running inference for model {model_name} at {model_dir}")
    # Load data
    logger.info("Loading data")
    start = time.time()

    # Load infer_freq_dataset
    infer_data_handler = DataHandler(config_dict, subset="inference")
    df = infer_data_handler.load_data()

    # Load and process monthly_dataset
    monthly_config_dict = copy.deepcopy(config_dict)
    monthly_config_dict["data"]["configs"]["freq"] = "M"
    full_monthly_data_handler = DataHandler(monthly_config_dict, subset="full")
    infer_monthly_data_handler = DataHandler(monthly_config_dict, subset="inference")

    full_monthly_df = full_monthly_data_handler.load_data()
    infer_monthly_df = infer_monthly_data_handler.load_data()
    monthly_df = merge_updated_data(full_monthly_df, infer_monthly_df)

    ela = time.time() - start
    logger.info(
        f"Runtime for loading and preparing data: {get_formatted_duration(ela)}"
    )

    # Load trained model
    model_save_path = os.path.join(model_dir, "model", "saved_model.pkl")
    logger.info(f"Initiating model class {model_class_str}")
    model_class = import_class(f"forecasting.models.{model_class_str}")
    logger.info(f"Loading trained model from {model_save_path}")
    model = model_class.load(model_save_path)

    # Make predictions
    logger.info("Running inference")
    pred_df: pd.DataFrame = model.predict(
        prediction_length,
        freq=data_configs["freq"],
        verbose=verbose,
    )

    ela = time.time() - start
    logger.info(f"Time for loading data and inference: {get_formatted_duration(ela)}")

    if not predict_only:
        generate_results(
            pred_df=pred_df,
            infer_df=df,
            infer_data_handler=infer_data_handler,
            monthly_df=monthly_df,
            result_save_dir=result_save_dir,
            trend_config_dict=trend_data_config,
            csv_file=False,
        )

    # Record time
    total_ela = time.time() - total_start
    logger.info(f"Total run time: {get_formatted_duration(total_ela)}")


def generate_results(
    pred_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    infer_data_handler: DataHandler,
    monthly_df: pd.DataFrame,
    result_save_dir: str,
    trend_config_dict: dict,
    csv_file: bool = False,
):
    os.makedirs(result_save_dir, exist_ok=True)

    # Create monthly pivot df
    monthly_pivot = (
        pd.pivot_table(
            monthly_df,
            index="id",
            values=["quantity_order"],
            aggfunc=lambda x: list(x),
        )
        .rename(columns={"quantity_order": "monthly_train_ts"})
        .reset_index()
    )

    logger.info("Generating results...")

    start = time.time()
    # Read metadata of inference set (up-to-date)
    infer_meta_df = infer_data_handler.load_metadata(drop_price=False)

    # Build the result df in wide format
    pred_pivot_df = build_result_df_from_pd(
        infer_df, pred_df, pred_cols=["predictions"]
    )

    # Merge metadata
    pred_pivot_df = pred_pivot_df.merge(
        infer_meta_df, on=infer_data_handler.metadata.id_col, how="left"
    )
    logger.info(
        f"Time for processing dataset: {get_formatted_duration(time.time()-start)}"
    )

    # Create daily_pred_ts and daily_train_ts columns
    pred_pivot_df["daily_train_ts"] = pred_pivot_df["train_ts"].apply(
        lambda x: aggregate_weekly_ts_to_daily(x)
    )
    pred_pivot_df["daily_pred_ts"] = pred_pivot_df["predictions"].apply(
        lambda x: aggregate_weekly_ts_to_daily(x)
    )
    # Remove leading zeros of every time series to calculate sale_per_day
    logger.info("Removing leading zeros values for daily_train_ts...")
    pred_pivot_df = pred_pivot_df.assign(
        # Create daily_train_ts_cut
        daily_train_ts_cut=(
            pred_pivot_df.daily_train_ts.apply(lambda x: remove_leading_zeros(x))
        )
    )

    # Join predictions df with monthly_pivot to get monthly_train_ts
    monthly_pivot = monthly_pivot[monthly_pivot.id.isin(pred_pivot_df.id.unique())]
    pred_pivot_df = pd.merge(pred_pivot_df, monthly_pivot, on="id")
    logger.info(f"Number of all items: {len(pred_pivot_df.id.unique())}")

    pred_pivot_df["sale_per_day"] = pred_pivot_df.daily_train_ts_cut.apply(
        lambda x: calc_sale_per_day(x)
    )

    brand_list = pred_pivot_df.brand_name.unique().tolist()
    brand_list = [
        brand for brand in brand_list if brand not in MULTIPLE_SOURCE_BRAND_LIST
    ]
    # brand_list = ["chanluu"]
    logger.info(f"Brand list: {brand_list}")

    logger.info("Aggregating top-down...")
    start_filter = time.time()
    # Filter channel_list and aggregate top-down
    channel_list = [
        "0",
        "580111",
        "c7d674e95445867e3488398e8b2cd2d8",
        "d724f9a653c53c6964282141d8fe9c84",
        "1",
    ]
    full_pred_df = pred_pivot_df[
        pred_pivot_df.channel_id.isin(channel_list)
        & pred_pivot_df.brand_name.isin(brand_list)
    ]
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
    # Read and process stockout data
    logger.info("Process stockout dataset...")
    data_folder = os.path.join(PROJECT_ROOT_PATH, "data/downloaded")

    forecast_date = pd.to_datetime(full_pred_df.first_pred_date.values[0]).date()
    daily_forecast_length = len(full_pred_df.daily_pred_ts.values[0])
    forecast_range = pd.date_range(
        forecast_date, periods=daily_forecast_length, freq="D"
    ).tolist()
    logger.info(
        f"Inference_date: {str(forecast_date)}, daily_forecast_len={daily_forecast_length}"
    )

    date_forecast_range = [date.strftime("%m-%d") for date in forecast_range]
    stockout_df = pd.DataFrame()
    for brand in brand_list:
        stock_path = os.path.join(data_folder, brand, "product-ts.csv")
        stock_df = pd.read_csv(
            stock_path,
            parse_dates=["load_date"],
            dtype={"variant_id": "string", "stock": "float32"},
        ).drop(columns=["product_id"])
        if stock_df.shape[0] > 0:
            print(
                f"Brand {brand}: min_date={stock_df.load_date.min()}, max_date={stock_df.load_date.max()}"
            )
        else:
            print(f"Brand {brand} no have stockout data")
        stock_df = stock_df[stock_df.stock == 0]
        item_list = stock_df.variant_id.unique().tolist()
        stock_df["stockout"] = 1
        stock_df["date"] = stock_df.load_date.apply(lambda x: x.strftime("%m-%d"))
        stock_df = stock_df.set_index(["date", "variant_id"])

        multi_index = pd.MultiIndex.from_product([date_forecast_range, item_list])
        multi_index = multi_index.set_names(["date", "variant_id"])
        stock_df = stock_df.reindex(multi_index).reset_index()
        stock_df["brand_name"] = brand

        stock_df = stock_df.drop(columns=["stock", "load_date"])
        stockout_df = pd.concat([stockout_df, stock_df])

    stockout_df = stockout_df.reset_index(drop=True)

    agg_product_df = agg_result_df[agg_result_df.is_product == True]
    agg_variant_df = agg_result_df[agg_result_df.is_product == False]

    # Process and merge stockout_ts to agg_variant_df
    stockout_df.stockout = stockout_df.stockout.fillna(value=0)
    pivot_stockout_df = pd.pivot_table(
        stockout_df,
        index=["brand_name", "variant_id"],
        values=["stockout"],
        aggfunc=lambda x: list(x),
    ).reset_index()
    agg_variant_df = agg_variant_df.merge(
        pivot_stockout_df, on=["brand_name", "variant_id"], how="left"
    )
    stockout_item_list = pivot_stockout_df.set_index(
        ["brand_name", "variant_id"]
    ).index.tolist()

    # Run filling stockout daily_pred_ts for variant level
    logger.info(
        "Fill stockout daily_pred_ts with average quantity over previous days..."
    )
    agg_variant_df["daily_pred_ts"] = agg_variant_df.apply(
        lambda row: (
            fill_stockout_update(
                stockout_ts=row.stockout,
                daily_train_ts=row.daily_train_ts,
                daily_pred_ts=row.daily_pred_ts,
            )
            if (row.brand_name, row.variant_id) in stockout_item_list
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

    logger.info("Aggregate bottom-up again...")
    final_agg_result_df = aggregate_bottom_up(
        final_stockout_result_df, pred_column="daily_pred_ts"
    )

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
            results = forecast_monitor.get_forecast_accuracy_all_items(
                period="3 months",
                method="avg_abs_error",
                group_method="sale_category",
                forecast_date=forecast_date,
            )
            if results is not None:
                acc_results = results["error_results"]
                logger.info(f"Len of acc_result: {len(acc_results)}")
            else:
                acc_results = None

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
                        "channel_id": row.channel_id,
                        "forecast_date": str(forecast_date),
                        "monthly_historical_val": str(row.monthly_train_ts),
                        "monthly_prediction_val": str(row.monthly_pred_ts),
                        "predictions": {
                            "sale_per_day": row.sale_per_day,
                            "forecast_val": str(row.daily_pred_ts),
                            "trend": None,
                        },
                    }
                    if (acc_results is not None) and (row.id in acc_results.keys()):
                        result["sale_pattern"] = acc_results[row.id]["sale_pattern"]
                        result["confidence_score"] = acc_results[row.id][
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
                        if level == "product":
                            unique_id = brand + "_" + item + "_NA_0"
                        else:
                            unique_id = (
                                brand
                                + "_"
                                + dict_variant_product[item]
                                + "_"
                                + item
                                + "_0"
                            )

                        all_channel_result = {
                            field_name: item,
                            "channel_id": "0",
                            "forecast_date": str(forecast_date),
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
                        if (acc_results is not None) and (
                            unique_id in acc_results.keys()
                        ):
                            all_channel_result["sale_pattern"] = acc_results[unique_id][
                                "sale_pattern"
                            ]
                            all_channel_result["confidence_score"] = acc_results[
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


if __name__ == "__main__":
    main()
