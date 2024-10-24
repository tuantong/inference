import argparse
import os

from forecasting import PROCESSED_DATA_BUCKET, PROJECT_ROOT_PATH
from forecasting.configs.logging_config import logger
from forecasting.data.s3_utils import aws_service

FORECAST_DATA_S3_DIR = "forecasting/forecast_data"


def upload_forecast_data_to_s3(local_dir, brand_folders, aws_storage):
    """
    Upload processed data to S3.

    Args:
        brand_folders (list): List of local directories for each brand.
        aws_storage (AWSStorage): AWSStorage instance for interacting with S3.

    Returns:
        None
    """
    # Iterate over each brand folder
    for brand_folder_name in sorted(brand_folders):
        # Set the local and S3 folder paths
        local_folder = os.path.join(local_dir, brand_folder_name)
        s3_folder = os.path.join(FORECAST_DATA_S3_DIR, brand_folder_name)

        # Upload all folders (predictions.csv and trend.csv)
        for f_name in os.listdir(local_folder):
            f_path = os.path.join(local_folder, f_name)
            # Check if the path is a file or a folder
            if os.path.isfile(f_path):
                aws_storage.upload(f_path, os.path.join(s3_folder, f_name))
            else:
                aws_storage.upload_folder(f_path, os.path.join(s3_folder, f_name))


def main(**kwargs):
    forecast_folder = kwargs["forecast_path"]
    forecast_local_path = os.path.join(PROJECT_ROOT_PATH, forecast_folder)

    brand_list = next(os.walk(forecast_local_path))[1]

    logger.info("Uploading data to S3 storage")
    client = aws_service.create_client()
    aws_storage = aws_service.AWSStorage(client, PROCESSED_DATA_BUCKET)

    upload_forecast_data_to_s3(
        local_dir=forecast_local_path, brand_folders=brand_list, aws_storage=aws_storage
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload csv forecast to S3")

    parser.add_argument(
        "-p",
        "--forecast_path",
        help="Path of forecast results",
        required=True,
    )
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
