import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from forecasting import PROJECT_ROOT_PATH, PROJECT_SRC_PATH
from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler
from forecasting.models.model_handler import ModelHandler
from forecasting.utils.common_utils import (
    load_data_from_pickle,
    load_dict_from_yaml,
    save_data_to_pickle,
)
from forecasting.utils.mlflow_utils import MlflowManager

# from inference.utils import generate_results_with_new_rule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

YAML_CONFIGS_DIR = Path(PROJECT_SRC_PATH) / "data" / "yaml_configs"


def _setup_parser():
    """Setup Argument Parser"""
    parser = argparse.ArgumentParser(description="Script to run inference")

    parser.add_argument(
        "-n",
        "--exp_name",
        type=str,
        required=True,
        help="Name of the training experiment on MlFlow. (i.e. 'TFT_training')",
    )
    parser.add_argument(
        "-rid",
        "--run_id",
        type=str,
        required=False,
        default=None,
        help="Run ID of the training experiment on MlFlow. If not provided, the latest run will be used.",
    )
    parser.add_argument(
        "-id",
        "--inference_data_date",
        type=str,
        default=None,
        required=False,
        help="Date of the inference data to be used for forecasting. If not provided, the latest date will be used.",
    )
    parser.add_argument(
        "-m",
        "--preprocess_method",
        type=str,
        choices=["fill_avg_sim", "fill_zero"],
        default="fill_zero",
        help="Method to preprocess the inference data.",
    )
    parser.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        default=1,
        required=False,
        help="Number of times a prediction is sampled from a probabilistic model. Default=1",
    )
    parser.add_argument(
        "-cf",
        "--config_file_path",
        type=str,
        required=False,
        default=os.path.join(YAML_CONFIGS_DIR, "unified_model/config.yaml"),
        help="Path to the config file to be used for trend detection.",
    )

    return parser


class Inference:
    """
    Orchestrates the inference process for time-series forecasting models.
    """

    def __init__(
        self,
        exp_name,
        run_id=None,
        inference_data_date=None,
        preprocess_method="fill_zero",
        num_samples=1,
        config_file_path=os.path.join(YAML_CONFIGS_DIR, "unified_model/config.yaml"),
    ):
        self.exp_name = exp_name
        self.inference_data_date = self._get_inference_date(inference_data_date)
        self.preprocess_method = preprocess_method
        self.trend_data_config = load_dict_from_yaml(config_file_path)["data"]

        # Initialize MlflowManager
        self.mlflow_manager = MlflowManager()
        # Get run info
        self.run_info = self._get_run_info(run_id)

        self.local_model_dir = self._get_local_model_dir()
        os.makedirs(self.local_model_dir, exist_ok=True)
        self.result_save_dir = os.path.join(
            PROJECT_ROOT_PATH, "inference_results", self.inference_data_date
        )
        os.makedirs(self.result_save_dir, exist_ok=True)
        self.log_path = os.path.join(
            PROJECT_ROOT_PATH, "logs", f"inference_{self.inference_data_date}.log"
        )

        self.config_path = self._get_necessary_artifacts()
        self.config_dict = self._load_config_dict()
        self.config_dict["data"]["configs"]["version"] = self.inference_data_date
        # Initialize inference data handler
        self.inference_data_handler = DataHandler(self.config_dict, subset="inference")
        # Initialize inference model handler
        self.inference_model_handler = ModelHandler(
            self.config_dict, model_dir=self.local_model_dir
        )
        self.output_chunk_length = self.inference_model_handler.parameters[
            "output_chunk_length"
        ]

        # Initialize prediction parameters
        self.num_samples = num_samples
        self.prediction_chunk_size = 100_000
        self._setup_logger()

    def run_inference(self, log_interval=1):
        infer_target_path = os.path.join(self.result_save_dir, "infer_target.pkl")
        infer_past_cov_path = os.path.join(self.result_save_dir, "infer_past_cov.pkl")
        # Load from cache if exists otherwise run the whole inference process
        if os.path.exists(infer_target_path) and os.path.exists(infer_past_cov_path):
            logger.info("Loading preprocessed data from cache")
            infer_target = load_data_from_pickle(infer_target_path)
            infer_past_cov = load_data_from_pickle(infer_past_cov_path)
        else:
            # Load inference data
            infer_df = self.inference_data_handler.load_data()
            # Preprocess for inference
            infer_target, infer_past_cov = self._preprocess_data(infer_df)
            # Transform (scaling) data
            infer_target, infer_past_cov = self._transform_data(
                infer_target, infer_past_cov
            )
            # Save the processed data
            save_data_to_pickle(infer_target, infer_target_path)
            save_data_to_pickle(infer_past_cov, infer_past_cov_path)

        # Load model
        model = self.inference_model_handler.load_model()

        # Make predictions in chunks
        preds = []
        total_chunks = (len(infer_target) + self.chunk_size - 1) // self.chunk_size

        for i in range(0, len(infer_target), self.chunk_size):
            # Predict
            pred = model.predict(
                n=self.num_samples,
                series=infer_target[i : i + self.chunk_size],
                past_covariates=(
                    infer_past_cov[i : i + self.chunk_size]
                    if infer_past_cov is not None
                    else None
                ),
            )
            preds.extend(pred)
            # Log progress after processing each chunk
            if (i // self.chunk_size + 1) % log_interval == 0 or (
                i + self.chunk_size
            ) >= len(infer_target):
                logger.info(
                    f"Predicted {i // self.chunk_size + 1} of {total_chunks} chunks"
                )

        # Inverse transform predictions
        preds = self.inference_data_handler.dataset.pipeline.inverse_transform(preds)
        # Save predictions
        save_data_to_pickle(
            preds, os.path.join(self.result_save_dir, f"{self.preprocess_method}.pkl")
        )

    def _get_necessary_artifacts(self):
        # Download necessary artifacts for inference
        logger.info("Downloading necessary artifacts for inference")
        self.mlflow_manager.client.download_artifacts(
            self.run_info.run_id, "model", self.local_model_dir
        )
        self.mlflow_manager.client.download_artifacts(
            self.run_info.run_id, "static_cov_transformer.pkl", self.local_model_dir
        )
        config_path = self.mlflow_manager.client.download_artifacts(
            self.run_info.run_id, "config.yaml", self.local_model_dir
        )
        return config_path

    def _get_local_model_dir(self):
        return os.path.join(PROJECT_ROOT_PATH, "saved_models", self.run_info.run_name)

    def _setup_logger(self):
        logger.add(
            self.log_path,
            format="{time} - {file.name} - {level} - {message}",
            retention="7 days",
        )

    def _get_run_info(self, run_id=None):
        if run_id is None:
            logger.info("Run ID not provided. Getting the latest run from MlFlow.")
            return self.mlflow_manager.get_latest_run(self.exp_name).info
        else:
            return self.mlflow_manager.client.get_run(run_id).info

    def _get_inference_date(self, inference_date):
        if inference_date is None:
            logger.info(
                "Inference date not provided. Using today's date with format 'YYYYMMDD'."
            )
            inference_date = datetime.now().strftime("%Y%m%d")
        return inference_date

    def _load_config_dict(self):
        return load_dict_from_yaml(self.config_path)

    def _preprocess_data(self, df: pd.DataFrame):
        logger.info("Running preprocessing step for inference")
        target_df, past_cov_df, _ = self.inference_data_handler.preprocess_data(
            df,
            stage="inference",
            preprocess_method=self.preprocess_method,
            output_chunk_length=self.output_chunk_length,
        )
        infer_target, infer_past_cov = (
            self.inference_data_handler.convert_df_to_darts_format(
                target_df, past_cov_df
            )
        )
        return infer_target, infer_past_cov

    def _transform_data(self, target_series, past_cov_series):
        if self.inference_data_handler.dataset.pipeline:
            logger.info("Transforming target (and past covariates)")
            target_series = self.inference_data_handler.dataset.pipeline.fit_transform(
                target_series
            )
            past_cov_series = (
                (
                    self.inference_data_handler.dataset.pipeline.fit_transform(
                        past_cov_series
                    )
                )
                if past_cov_series
                else None
            )

        if self.inference_data_handler.dataset._metadata.static_cov_cols:
            logger.info("Transforming static covariates")
            target_series = (
                self.inference_model_handler.static_cov_transformer.transform(
                    target_series
                )
            )
        return target_series, past_cov_series


def main():
    """Inference main function"""
    parser = _setup_parser()
    args = parser.parse_args()

    kwargs = vars(args)
    inference = Inference(**kwargs)
    inference.run_inference()


if __name__ == "__main__":
    main()
