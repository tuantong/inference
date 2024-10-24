import argparse
import ast
import json
import math
import os
from collections import Counter
from pathlib import Path

# import numpy as np
import pandas as pd
from tqdm import tqdm

from forecasting import PROJECT_DATA_PATH, PROJECT_ROOT_PATH
from forecasting.configs.logging_config import logger


def main(kwargs):
    result_folder = kwargs["forecast_json_path"]
    format = kwargs["convert_format"]
    save_folder = kwargs["results_save_path"]
    drop_old_items_melinda = kwargs["drop_old_items_melinda"]

    result_save_path = os.path.join(PROJECT_ROOT_PATH, result_folder)

    if save_folder is None:
        result_folder_name = result_folder.split("/")[-2]
        if format == "csv":
            save_folder = os.path.join(
                "inference_results", f"{result_folder_name}_new_system"
            )
        else:
            save_folder = os.path.join(
                "inference_results", f"{result_folder_name}_parquet"
            )

    save_path = os.path.join(PROJECT_ROOT_PATH, save_folder)
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Save_path: {save_path}")

    brand_list = next(os.walk(result_save_path))[1]
    logger.info(brand_list)

    for brand in brand_list:
        logger.info(brand)

        full_pred_df = pd.DataFrame()
        full_trend_df = pd.DataFrame()
        level_list = ["product", "variant"]
        for level in level_list:
            path = f"{result_save_path}/{brand}/{level}_result_forecast.json"
            with open(path) as file:
                fc_list = json.load(file)
            logger.info(f"{level} level: {len(fc_list)} items")

            fc_list = [res for res in fc_list if res["h_key"] is not None]
            if len(fc_list) == 0:
                continue

            forecast_date = pd.to_datetime(fc_list[0]["forecast_date"])
            pred_len = len(ast.literal_eval(fc_list[0]["predictions"]["forecast_val"]))
            date_list = pd.date_range(start=forecast_date, periods=pred_len, freq="D")

            # Reformat list of dictionary
            reformat_fc_list = [
                {
                    "from_source": item_fc["from_source"],
                    "h_key": item_fc["h_key"],
                    "item_id": item_fc[f"{level}_id"],
                    "is_product": True if level == "product" else False,
                    "channel_id": item_fc["channel_id"],
                    "date": date_list,
                    "sale_per_day": item_fc["predictions"]["sale_per_day"],
                    "forecast_value": ast.literal_eval(
                        item_fc["predictions"]["forecast_val"]
                    ),
                    "trend": item_fc["predictions"]["trend"],
                    "sale_pattern": item_fc["sale_pattern"],
                    "confidence_score": item_fc["confidence_score"],
                }
                for item_fc in fc_list
            ]

            # Create predictions dataframe
            level_pred_df = pd.concat(
                pd.DataFrame(
                    item_dict,
                    columns=[
                        "from_source",
                        "h_key",
                        "item_id",
                        "is_product",
                        "channel_id",
                        "date",
                        "sale_per_day",
                        "forecast_value",
                    ],
                )
                for item_dict in reformat_fc_list
            )

            # Create trend dataframe
            level_trend_df = pd.concat(
                pd.DataFrame(
                    item_dict,
                    columns=[
                        "from_source",
                        "h_key",
                        "item_id",
                        "is_product",
                        "channel_id",
                        "trend",
                        "sale_pattern",
                        "confidence_score",
                    ],
                    index=[0],
                )
                for item_dict in reformat_fc_list
            )

            full_pred_df = pd.concat([full_pred_df, level_pred_df])
            full_trend_df = pd.concat([full_trend_df, level_trend_df])

        logger.info(f"Number of items: {full_trend_df.item_id.unique().shape}")

        if drop_old_items_melinda:
            # Drop old_structure item for Melinda
            if brand == "melinda_maria":
                logger.info("Drop old_items Melinda")
                # Drop variant_item
                variant_path = os.path.join(
                    PROJECT_DATA_PATH,
                    "downloaded_multiple_datasource",
                    brand,
                    "variant.csv",
                )
                variant_df = pd.read_csv(variant_path).reset_index(drop=True)

                counter = Counter(variant_df.sku.dropna())
                duplicate_sku_list = {
                    item for item, count in counter.items() if count > 1
                }

                drop_variant_list = []
                for sku in duplicate_sku_list:
                    var_df = variant_df[variant_df.sku == sku]
                    if var_df[var_df.status == "active"].shape[0] <= 1:
                        drop_df = var_df[var_df.status != "active"]
                    else:  # have more than 1 active variant ID with the same SKU -> get the newest variant ID
                        max_created_date = var_df[
                            var_df.status == "active"
                        ].created_date.max()
                        drop_df = var_df[var_df.created_date != max_created_date]

                    for variant_id in drop_df.variant_id.values:
                        drop_variant_list.append(variant_id)
                logger.info(f"Number of drop_variant_id: {len(drop_variant_list)}")
                filtered_variant_df = variant_df[
                    ~variant_df.variant_id.isin(drop_variant_list)
                ]

                # Drop product_item
                product_path = os.path.join(
                    PROJECT_DATA_PATH,
                    "downloaded_multiple_datasource",
                    brand,
                    "product.csv",
                )
                product_df = pd.read_csv(product_path).reset_index(drop=True)

                drop_product_list = []
                for prod_id in product_df.product_id.values:
                    var_df = filtered_variant_df[
                        filtered_variant_df.product_id == prod_id
                    ]
                    if var_df.shape[0] < 1:  # all variant of prod_id is dropped
                        drop_product_list.append(prod_id)
                    else:
                        non_active_var_df = var_df[var_df.status != "active"]
                        if (
                            non_active_var_df.shape[0] == var_df.shape[0]
                        ):  # all variant of prod_is is not active
                            drop_product_list.append(prod_id)
                logger.info(f"Number of drop_product_id: {len(drop_product_list)}")
                more_drop_variant_list = (
                    filtered_variant_df[
                        filtered_variant_df.product_id.isin(drop_product_list)
                    ]
                    .variant_id.unique()
                    .tolist()
                )
                logger.info(
                    f"Number of drop_non-active_variant_id: {len(more_drop_variant_list)}"
                )
                drop_variant_list = list(
                    set(drop_variant_list + more_drop_variant_list)
                )

                var_pred_df = full_pred_df[
                    (full_pred_df.is_product == False)
                    & ~(full_pred_df.item_id.isin(drop_variant_list))
                ]
                prod_pred_df = full_pred_df[
                    (full_pred_df.is_product == True)
                    & ~(full_pred_df.item_id.isin(drop_product_list))
                ]
                full_pred_df = pd.concat([var_pred_df, prod_pred_df])

                var_trend_df = full_trend_df[
                    (full_trend_df.is_product == False)
                    & ~(full_trend_df.item_id.isin(drop_variant_list))
                ]
                prod_trend_df = full_trend_df[
                    (full_trend_df.is_product == True)
                    & ~(full_trend_df.item_id.isin(drop_product_list))
                ]
                full_trend_df = pd.concat([var_trend_df, prod_trend_df])
                logger.info(
                    f"Number of items of {brand} after filtering: {full_trend_df.item_id.unique().shape}"
                )

        # Save forecast as csv
        if format == "csv":
            logger.info("Save data as csv...")
            save_dir = f"{save_path}/{brand}"
            os.makedirs(save_dir, exist_ok=True)

            pred_save_path = Path(save_dir) / "predictions.csv"
            full_pred_df.to_csv(pred_save_path, index=False)
            trend_save_path = Path(save_dir) / "trend.csv"
            full_trend_df.to_csv(trend_save_path, index=False)

        # Save forecast as parquet
        if format == "parquet":
            logger.info("Save data as parquet...")
            save_dir = f"{save_path}/{brand}"
            os.makedirs(save_dir, exist_ok=True)

            pred_dir = f"{save_dir}/predictions"
            os.makedirs(pred_dir, exist_ok=True)
            trend_dir = f"{save_dir}/trend"
            os.makedirs(trend_dir, exist_ok=True)

            no_items = full_trend_df.shape[0]
            # n_items = 1000 # Split into files of 1000 items
            # n_chunks = math.ceil(no_items / n_items)
            # logger.info(f'No. all items: {no_items}, No. chunks: {n_chunks}')

            n_chunks = 10  # Split into 10 files
            n_items = math.ceil(no_items / n_chunks)
            logger.info(f"No. all items: {no_items}, No. item per chunk: {n_items}")
            pred_len_chunk = n_items * pred_len
            trend_len_chunk = n_items

            for chunk in tqdm(range(n_chunks)):
                pred_index = chunk * pred_len_chunk
                chunk_pred_df = full_pred_df[pred_index : (pred_index + pred_len_chunk)]
                trend_index = chunk * trend_len_chunk
                chunk_trend_df = full_trend_df[
                    trend_index : (trend_index + trend_len_chunk)
                ]

                chunk_pred_save_path = Path(pred_dir) / f"predictions-{chunk+1}.parquet"
                chunk_pred_df.to_parquet(chunk_pred_save_path, index=False)
                chunk_trend_save_path = Path(trend_dir) / f"trend-{chunk+1}.parquet"
                chunk_trend_df.to_parquet(chunk_trend_save_path, index=False)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert forecast result to CSV/Parquet"
    )

    parser.add_argument(
        "-p",
        "--forecast_json_path",
        help="Path of json forecast results",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--convert_format",
        help="Format of converted files",
        default="csv",
        choices=["csv", "parquet"],
    )
    parser.add_argument(
        "-s",
        "--results_save_path",
        help="Path to save converted files",
        default=None,
    )
    parser.add_argument(
        "--drop_old_items_melinda",
        action="store_true",
        help="Whether to drop old items of Melinda while convert for weekly forecast",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    save_dir = main(**kwargs)
