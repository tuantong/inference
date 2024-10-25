import argparse
import copy
import glob
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import yaml
from dateutil.relativedelta import MO, relativedelta

from forecasting import MODEL_DIR
from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler, merge_updated_data
from forecasting.data.preprocessing.preprocess_data import preprocess_for_inference
from forecasting.data.preprocessing.utils import extend_df
from forecasting.data.utils.darts_utils import convert_df_to_darts_dataset
from forecasting.util import get_formatted_duration, get_latest_date, import_class
from forecasting.utils.common_utils import load_dict_from_yaml
from inference.utils import generate_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        "-m",
        "--preprocess_method",
        type=str,
        choices=["fill_avg_sim", "fill_zero"],
        default="fill_zero",
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

    num_samples = kwargs["num_samples"]
    model_dir = kwargs["model_dir"]
    inference_data_date = kwargs["inference_data_date"]
    result_save_dir = kwargs["result_save_dir"]
    preprocess_method = kwargs["preprocess_method"]
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

    logger.info(f"Running inference for model at {model_dir}")
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
    logger.info(f"Data configs: {data_configs}")

    # Model configs
    basic_configs = model_config["configs"]
    model_name = basic_configs.get("name")
    num_loader_workers = basic_configs.get("num_loader_workers")

    parameter_configs = model_config["parameters"]
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

    # Preprocessing
    logger.info("Running preprocessing step for inference")
    # input_chunk_length = parameter_configs['input_chunk_length']
    output_chunk_length = parameter_configs["output_chunk_length"]

    # Extend target and past covariates into the past and fill the extended periods
    df, _ = preprocess_for_inference(
        df, preprocess_method=preprocess_method, min_length_for_new_item=12
    )
    logger.info(
        f"preprocessed_df shape: {df.shape}, preprocessed_df unique ids: {df.id.unique().shape}"
    )
    # Extend past covariates into the future if prediction length is bigger than output_chunk_length
    if infer_data_handler.metadata.past_cov_cols:
        past_cov_df = df.drop(columns=infer_data_handler.metadata.target_col)
        if prediction_length > output_chunk_length:
            logger.info("Extending past covariates into the future")
            past_cov_df = extend_df(
                df=past_cov_df,
                value_col=infer_data_handler.metadata.past_cov_cols,
                avg_sim_df=None,
                required_len=prediction_length - output_chunk_length,
                direction="future",
                freq=infer_data_handler.metadata.freq,
            )
            logger.info(
                f"past_cov_df shape: {past_cov_df.shape}, past_cov_df unique ids: {past_cov_df.id.unique().shape}"
            )

    # Convert Pandas to Darts
    logger.info("Converting Pandas to Darts")
    infer_target = convert_df_to_darts_dataset(
        df=df,
        id_col=infer_data_handler.metadata.id_col,
        time_col=infer_data_handler.metadata.time_col,
        value_cols=infer_data_handler.metadata.target_col,
        static_cols=infer_data_handler.metadata.static_cov_cols,
        freq=infer_data_handler.metadata.freq,
        dtype=np.float32,
    )

    infer_past_cov = (
        convert_df_to_darts_dataset(
            df=past_cov_df,
            id_col=infer_data_handler.metadata.id_col,
            time_col=infer_data_handler.metadata.time_col,
            value_cols=infer_data_handler.metadata.past_cov_cols,
            static_cols=infer_data_handler.metadata.static_cov_cols,
            freq=infer_data_handler.metadata.freq,
            dtype=np.float32,
        )
        if infer_data_handler.metadata.past_cov_cols
        else None
    )

    # Load model class from configs
    logger.info(f"Initiating model class {model_class_str}")
    model_class = import_class(f"forecasting.models.{model_class_str}")

    # Transform (scaling) data
    if infer_data_handler.dataset.pipeline:
        logger.info("Transforming target (and past covariates)")

        infer_target = infer_data_handler.dataset.pipeline.fit_transform(infer_target)
        if infer_past_cov:
            infer_past_cov = infer_data_handler.dataset.past_cov_pipeline.fit_transform(
                infer_past_cov
            )

    if infer_data_handler.metadata.static_cov_cols:
        # Static covariates transform
        # Load the fitted static_cov_transformer
        logger.info("Loading static covariates transformer")
        static_cov_path = os.path.join(model_dir, "static_cov_transformer.pkl")
        with open(static_cov_path, "rb") as f:
            static_cov_transformer = pickle.load(f)
        logger.info("Transforming static covariates")
        infer_target = static_cov_transformer.transform(infer_target)

    ela = time.time() - start
    logger.info(
        f"Runtime for loading and preparing data: {get_formatted_duration(ela)}"
    )

    # Load trained model
    logger.info(f"Loading trained model from {model_dir}")
    model = model_class.load_from_checkpoint(
        model_name=model_name, work_dir=model_dir, best=False
    )

    trainer_params = model.trainer_params
    trainer_params.update({"logger": False, "enable_model_summary": False})

    # Make predictions in chunks
    preds = []
    chunk_size = 100000
    log_interval = 1
    total_chunks = (len(infer_target) + chunk_size - 1) // chunk_size

    for i in range(0, len(infer_target), chunk_size):
        preds.extend(
            model.predict(
                n=prediction_length,
                series=infer_target[i : i + chunk_size],
                past_covariates=(
                    infer_past_cov[i : i + chunk_size] if infer_past_cov else None
                ),
                num_loader_workers=num_loader_workers,
                trainer=pl.Trainer(**trainer_params),
                num_samples=num_samples,
            )
        )

        # Log progress after processing each chunk
        if (i // chunk_size + 1) % log_interval == 0 or (i + chunk_size) >= len(
            infer_target
        ):
            logger.info(f"Predicted {i // chunk_size + 1} of {total_chunks} chunks")

    # preds = model.predict(
    #     n=prediction_length,
    #     series=infer_target,
    #     past_covariates=infer_past_cov,
    #     num_loader_workers=num_loader_workers,
    #     trainer=pl.Trainer(**trainer_params),
    #     num_samples=num_samples
    # )

    if infer_data_handler.dataset.pipeline:
        # Inverse scale the prediction
        logger.info("Inverse transform predictions")
        preds = infer_data_handler.dataset.pipeline.inverse_transform(preds)

    ela = time.time() - start
    logger.info(f"Time for loading data and inference: {get_formatted_duration(ela)}")

    prediction_save_path = os.path.join(prediction_save_dir, f"{preprocess_method}.pkl")
    logger.info(f"Saving predictions at {prediction_save_path}")
    with open(prediction_save_path, "wb") as f:
        pickle.dump(preds, f)

    if not predict_only:
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

        inference_week = 1
        logger.info(
            f"Inference_date: {datetime.today().date() + relativedelta(weekday=MO(-inference_week))}"
        )

        generate_results(
            inference_week=inference_week,
            infer_data_handler=infer_data_handler,
            config_dict=config_dict,
            monthly_df=monthly_df,
            prediction_set=preds,
            seasonal_set=seasonal_list,
            similar_product_dict=similar_product_dict,
            model_dir=model_dir,
            result_save_dir=result_save_dir,
            trend_config_dict=trend_data_config,
            csv_file=False,
            post_process=False,
        )

    # Record time
    total_ela = time.time() - total_start
    logger.info(f"Total run time: {get_formatted_duration(total_ela)}")


if __name__ == "__main__":
    main()
