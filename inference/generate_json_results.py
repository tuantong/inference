import argparse
import copy
import glob
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import yaml
from dateutil.relativedelta import MO, relativedelta

from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler, merge_updated_data
from forecasting.util import get_formatted_duration, get_latest_date
from forecasting.utils.common_utils import load_dict_from_yaml
from inference.utils import generate_results

YAML_CONFIGS_DIR = (
    Path(__file__).resolve().parents[1] / "forecasting" / "data" / "yaml_configs"
)


def _setup_parser():
    """Setup Argument Parser"""
    parser = argparse.ArgumentParser(
        description="Generate submission files for test set from model predictions",
        add_help=False,
    )

    parser.add_argument("-id", "--inference_data_date", type=str, required=False)
    parser.add_argument("-wd", "--model_dir", type=str, required=False)
    parser.add_argument("-p", "--model_predictions_path", type=str, required=False)
    parser.add_argument("-s", "--results_save_dir", type=str, required=False)
    parser.add_argument(
        "-cf",
        "--config_file_path",
        type=str,
        required=False,
        default="unified_model/config.yaml",
    )

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    inference_week = 1
    logger.info(
        f"Inference_date: {datetime.today().date() + relativedelta(weekday=MO(-inference_week))}"
    )

    parser = _setup_parser()
    args = parser.parse_args()

    kwargs = vars(args)

    start = time.time()
    inference_data_date = kwargs["inference_data_date"]
    model_dir = kwargs["model_dir"]
    model_predictions_path = kwargs["model_predictions_path"]
    result_save_dir = kwargs["results_save_dir"]
    config_file_path = kwargs["config_file_path"]

    # Read file config
    config_path = YAML_CONFIGS_DIR / config_file_path
    with open(config_path, encoding="utf-8") as f:
        all_config = yaml.safe_load(f)
    trend_data_config = all_config["data"]

    default_model_dir = Path(os.environ["MODEL_DIR"])
    if model_dir is None:
        logger.warning(
            f"Keyword argument `model_dir` not found, looking for latest trained model at {default_model_dir}"
        )
        # Get latest trained model
        latest_model_folder = get_latest_date(default_model_dir)
        model_dir = os.path.join(default_model_dir, latest_model_folder, "best_model")

    # Add logger file
    logger.add(
        os.path.join(model_dir, f"generate_json_{inference_data_date}.log"),
        format="{time} - {file.name} - {level} - {message}",
    )

    logger.info(f"Generating JSON results for model at {model_dir}")

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

    # Data configs
    data_configs = data_config["configs"]
    data_configs["version"] = inference_data_date
    logger.info(f"Data configs: {data_configs}")

    # Load data
    logger.info("Loading data")
    start = time.time()

    # Load infer_freq_dataset
    infer_data_handler = DataHandler(config_dict, subset="inference")

    # Load and process monthly_dataset
    monthly_config_dict = copy.deepcopy(config_dict)
    monthly_config_dict["data"]["configs"]["freq"] = "M"
    full_monthly_data_handler = DataHandler(monthly_config_dict, subset="full")
    infer_monthly_data_handler = DataHandler(monthly_config_dict, subset="inference")

    full_monthly_df = full_monthly_data_handler.load_data()
    infer_monthly_df = infer_monthly_data_handler.load_data()
    monthly_df = merge_updated_data(full_monthly_df, infer_monthly_df)

    if model_predictions_path is None:
        if inference_data_date is None:
            inference_data_date = datetime.now().strftime("%Y%m%d")

        model_predictions_path = os.path.join(
            model_dir,
            "inference_results",
            inference_data_date,
            "inference_predictions.pkl",
        )

        logger.warning(
            f"Keyword argument `model_predictions_path` not found, looking for model predictions at {default_model_dir}"
        )

    if result_save_dir is None:
        logger.warning(
            f"Keyword argument `result_save_dir` not found, looking for latest trained model at {default_model_dir}"
        )
        result_save_dir = os.path.join(
            model_dir, "inference_results", inference_data_date
        )
    os.makedirs(result_save_dir, exist_ok=True)

    # Read predictions
    logger.info("Reading predictions file")
    with open(model_predictions_path, "rb") as f:
        pred_target_set = pickle.load(f)

    # Load seasonal list
    seasonal_file = glob.glob(os.path.join(model_dir, "seasonal_set.txt"))
    if len(seasonal_file) != 1:
        raise ValueError(
            f"Expected to find seasonal_set.txt, found {len(seasonal_file)}. Run to find seasonal_set first."
        )

    seasonal_file = seasonal_file[0]
    seasonal_list = []
    with open(seasonal_file, encoding="utf-8") as f:
        for line in f:
            item = line[:-1]
            seasonal_list.append(item)
    logger.info(f"Seasonal_list: {len(seasonal_list)}")

    # Load similar_item_dict
    similar_product_file = glob.glob(
        os.path.join(model_dir, "similar_product_dict.json")
    )
    if len(similar_product_file) >= 1:
        similar_product_file = similar_product_file[0]
        similar_product_dict = load_dict_from_yaml(similar_product_file)
        logger.info(f"Similar_product_dict: {len(similar_product_dict)}")
    else:
        similar_product_dict = None

    generate_results(
        inference_week=inference_week,
        infer_data_handler=infer_data_handler,
        config_dict=config_dict,
        monthly_df=monthly_df,
        prediction_set=pred_target_set,
        seasonal_set=seasonal_list,
        similar_product_dict=similar_product_dict,
        model_dir=model_dir,
        result_save_dir=result_save_dir,
        trend_config_dict=trend_data_config,
        csv_file=False,
        post_process=False,
    )

    ela = time.time() - start
    logger.info(f"Time for generation JSON results: {get_formatted_duration(ela)}")


if __name__ == "__main__":
    main()
