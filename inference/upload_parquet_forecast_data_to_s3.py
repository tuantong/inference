import argparse
import os

from forecasting import AWS_FORECAST_DATA_BUCKET, PROJECT_ROOT_PATH
from forecasting.configs.logging_config import logger
from forecasting.data.s3_utils import aws_service

dict_brand_name = {
    "annmarie": "annmarieskincare",
    "naked_cashmere": "nakedcashmere",
}


def upload_forecast_data_to_s3(local_dir, subset, brand_folders, aws_storage):
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
        # s3_folder = os.path.join(AWS_FORECAST_DATA_BUCKET, brand_folder_name)
        s3_folder = (
            brand_folder_name
            if brand_folder_name not in dict_brand_name.keys()
            else dict_brand_name[brand_folder_name]
        )
        logger.info(f"Push local_folder {brand_folder_name} to S3 folder {s3_folder}")

        folder_list = ["trend", "predictions"]
        # Upload all subfiles in 2 folders
        for folder in folder_list:
            folder_path = os.path.join(local_folder, folder)
            for f_name in sorted(os.listdir(folder_path)):
                f_path = os.path.join(folder_path, f_name)
                aws_storage.upload(
                    f_path,
                    os.path.join(s3_folder, subset, folder, f_name),
                    # f_path,
                    # os.path.join(s3_folder, folder, f_name),
                )


def main(**kwargs):
    forecast_folder = kwargs["forecast_path"]
    subset = kwargs["subset"]
    forecast_local_path = os.path.join(PROJECT_ROOT_PATH, forecast_folder)

    brand_list = next(os.walk(forecast_local_path))[1]

    logger.info("Uploading data to S3 storage")
    client = aws_service.create_client()
    aws_storage = aws_service.AWSStorage(client, AWS_FORECAST_DATA_BUCKET)

    upload_forecast_data_to_s3(
        local_dir=forecast_local_path,
        subset=subset,
        brand_folders=brand_list,
        aws_storage=aws_storage,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload parquet forecast to S3")

    parser.add_argument(
        "-p",
        "--forecast_path",
        help="Path of forecast results",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--subset",
        help="Subset for pushing (all/partial)",
        choices=["all", "partial"],
        default="all",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
