import argparse
import ast
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from forecasting.configs.logging_config import logger
from forecasting.models.trend_detector import trend_detector
from forecasting.util import NpEncoder, get_formatted_duration, get_latest_date

YAML_CONFIGS_DIR = (
    Path(__file__).resolve().parents[1] / "forecasting" / "data" / "yaml_configs"
)


def get_trend(slope, threshold_up, threshold_down):
    """Get trend for each item id

    Args:
        slope (float): slope of the line
        threshold_up (float): threshold to use for defining "Up".
        threshold_down (float): threshold to use for defining "Down".

    Returns:
        [str]: Trend of the time series, "Up", "Down" or "Normal"
    """
    # logger.debug(f"Slope: {slope}")
    if slope <= threshold_down and slope < 0:
        trend = "Down"
    elif slope >= threshold_up and slope > 0:
        trend = "Up"
    else:
        trend = "Normal"

    return trend


def run_single_level_detection(
    result_list: list,
    frequency_point: str,
    historical_length: int = 3,
    future_length: int = 0,
):

    platform_list = list({result["from_source"] for result in result_list})
    forecast_date = list({result["forecast_date"] for result in result_list})[0]

    for platform in platform_list:
        platform_result_list = [
            result for result in result_list if result["from_source"] == platform
        ]
        channel_list = list({result["channel_id"] for result in platform_result_list})

        for ch_id in channel_list:  # Calculate trend slope for each channel separately
            results_slope = []

            ch_result_list = [
                result
                for result in platform_result_list
                if result["channel_id"] == ch_id
            ]
            for row in tqdm(
                ch_result_list,
                total=len(ch_result_list),
                desc=f"Calculating slope for channel_id {ch_id}...",
            ):

                # if frequency_point == "monthly":
                #     historical_val = ast.literal_eval(
                #         str(row["monthly_historical_val"])
                #     )
                # elif frequency_point == "weekly":
                #     historical_val = ast.literal_eval(str(row["weekly_historical_val"]))
                # else:
                #     historical_val = ast.literal_eval(str(row["daily_historical_val"]))

                historical_val = ast.literal_eval(
                    str(row[f"{frequency_point}_historical_val"])
                )
                forecast_val = ast.literal_eval(str(row["monthly_prediction_val"]))

                if frequency_point == "monthly":
                    # Check if the value of predictions contains the remaining value for last month
                    # If the forecast day is not 1st then we need to add to last historical value
                    if pd.Timestamp(forecast_date).day != 1:
                        forecast_val[0] += historical_val[-1]
                        historical_val.pop(-1)

                # Get list of history values and forecast values by month
                historical_ts = (
                    historical_val[-historical_length:]
                    if historical_length is not None
                    else historical_val
                )
                forecast_ts = forecast_val[:future_length]

                # Calculate slope for predictions
                array_of_data = historical_ts + forecast_ts
                indices = np.arange(len(array_of_data))
                (slope, beta) = trend_detector(indices, array_of_data)
                results_slope.append(slope)
                row["slope"] = slope
                row["beta"] = beta

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

            for row in tqdm(
                ch_result_list,
                total=len(ch_result_list),
                desc="Saving trend to result file...",
            ):
                slope = row["slope"]
                trend = get_trend(slope, thresh_up, thresh_down)
                row["predictions"]["trend"] = trend

    infer_list = copy.deepcopy(result_list)
    # for sub in infer_list:
    #     del sub["slope"]
    #     del sub["beta"]
    #     del sub["monthly_historical_val"]
    #     del sub["monthly_prediction_val"]

    logger.debug(f"List of keys in beta list: {result_list[0].keys()}")
    logger.debug(f"List of keys in inference list: {infer_list[0].keys()}")

    return result_list, infer_list


def run_trend_detection(
    brand_name: str,
    forecast_results_dir: str,
    inference_save_dir: str,
    frequency_point: str,
    historical_length: int = 3,
    future_length: int = 0,
):
    """Trend detection

    Args:
        forecast_results_dir (str): Directory containing the forecast results of variant, product and collection levels.
        inference_save_dir (str): Directory of final inference results (delete unnecessary keys)
        historical_length (int, optional): Historical length to use. Defaults to 13.
        future_length (int, optional): Forecast length to use. Defaults to 13

    Returns:
        [str]: Path of the trend detection result csv file
        i.e., <trend_results_save_dir>/trend_detection_result.csv
    """
    levels_map = ["variant", "product"]

    # assert 0.0 <= forecast_type <= 1.0, "Forecast type must between 0 and 1"
    variant_forecast_results_path = (
        Path(forecast_results_dir) / "variant_result_forecast.json"
    )
    product_forecast_results_path = (
        Path(forecast_results_dir) / "product_result_forecast.json"
    )

    # Inference result paths
    variant_infer_path = Path(inference_save_dir) / "variant_result_forecast.json"
    product_infer_path = Path(inference_save_dir) / "product_result_forecast.json"

    # Read forecast results
    logger.info("Reading forecast results...")
    with open(variant_forecast_results_path) as variant_file:
        variant_results = json.load(variant_file)

    with open(product_forecast_results_path) as product_file:
        product_results = json.load(product_file)

    all_levels_results = [variant_results, product_results]
    all_levels_infer = []
    # Run detection
    # all_results = []  # For debug
    for level, results_forecast in zip(levels_map, all_levels_results):
        logger.info(f"Running trend detection on {level} level")
        result_list, infer_list = run_single_level_detection(
            result_list=results_forecast,
            historical_length=historical_length,
            future_length=future_length,
            frequency_point=frequency_point,
        )
        all_levels_infer.append(infer_list)

    save_paths = [variant_forecast_results_path, product_forecast_results_path]
    results_dict = [variant_results, product_results]
    infer_paths = [variant_infer_path, product_infer_path]

    # Saving results
    for level, path, res in zip(levels_map, save_paths, results_dict):
        # Level is used for saving to separate result files
        try:
            # res.to_csv(path, index=False)
            with open(path, "w") as file:
                json.dump(res, file, cls=NpEncoder, indent=4)

        except Exception as err:
            logger.error(err)
            raise
        else:
            logger.debug(f"Sucessfully saved {level} trend detection results to {path}")

    for level, infer_path, res in zip(levels_map, infer_paths, all_levels_infer):
        # Level is used for saving to separate result files
        try:
            # res.to_csv(path, index=False)
            with open(infer_path, "w") as file:
                json.dump(res, file, cls=NpEncoder, indent=4)

        except Exception as err:
            logger.error(err)
            raise
        else:
            logger.debug(
                f"Sucessfully saved {level} trend detection results to {infer_path}"
            )


#    return save_paths


def main(kwargs):
    result_save_path = kwargs["results_save_path"]
    inference_path = kwargs["inference_path"]
    config_file_path = kwargs["config_path"]
    frequency_point = kwargs["frequency_point"]

    # Read file config
    config_path = YAML_CONFIGS_DIR / config_file_path
    with open(config_path, encoding="utf-8") as f:
        all_config = yaml.safe_load(f)
    data_config = all_config["data"]

    if result_save_path is None:
        logger.warning(
            "Keyword argument `result_save_path` not found. Looking for latest JSON results from latest model"
        )

        model_dir = os.environ["MODEL_DIR"]
        latest_model_date = get_latest_date(Path(model_dir))
        latest_model_path = Path(
            os.path.join(
                model_dir, latest_model_date, "best_model", "inference_results"
            )
        )
        latest_infer_date = get_latest_date(latest_model_path)

        result_save_path = os.path.join(latest_model_path, latest_infer_date)

    if inference_path is None:
        inference_path = result_save_path

    brand_list = next(os.walk(result_save_path))[1]
    logger.info(f"List of brands: {brand_list}")

    for brand in brand_list:
        logger.info(f"Detecting trend for {brand}...")
        brand_save_path = Path(result_save_path) / brand
        brand_inference_path = Path(inference_path) / brand
        os.makedirs(brand_inference_path, exist_ok=True)

        brand_config = [cf for cf in data_config if cf["brand_name"] == brand][0]
        if frequency_point == "monthly":
            trend_history_length = brand_config["trend_history_length"]
        elif frequency_point == "weekly":
            trend_history_length = 8
        else:
            trend_history_length = (
                None  # Get full daily_val for daily forecast of newly-launched items
            )

        run_trend_detection(
            brand_name=brand,
            forecast_results_dir=brand_save_path,
            inference_save_dir=brand_inference_path,
            historical_length=trend_history_length,
            frequency_point=frequency_point,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate submission files for test set from model predictions"
    )

    parser.add_argument(
        "-s",
        "--results_save_path",
        help="Path to save visualization files",
        required=False,
    )
    parser.add_argument(
        "-ip", "--inference_path", help="Path to save deployment files", required=False
    )
    parser.add_argument(
        "-f",
        "--frequency_point",
        help="Chosse frequency point for fitting in linear",
        choices=["monthly", "weekly", "daily"],
        default="monthly",
    )
    parser.add_argument(
        "-cf",
        "--config_path",
        help="Path to data_config file",
        default="unified_model/config_multiple_sources.yaml",
    )

    args = parser.parse_args()

    kwargs = vars(args)

    start = time.time()
    main(**kwargs)
    ela = time.time() - start
    logger.info(f"Time for inference: {get_formatted_duration(ela)}")
