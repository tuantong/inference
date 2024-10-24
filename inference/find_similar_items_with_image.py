import os
import urllib.request
from io import BytesIO

import billiard as multiprocessing
import numpy as np
import pandas as pd
import torch
from filelock import FileLock, Timeout
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler, normalize
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from forecasting import PROJECT_DATA_PATH
from forecasting.configs.logging_config import logger


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


# def download_and_save_image(args):
#     url, folder_path, item_id = args
#     image_path = os.path.join(folder_path, f"{item_id}.jpg")
#     if url is None or pd.isna(url) or url.lower() in ["na", "<na>", "none", "null"]:
#         #         print(f"No valid URL for item ID {item_id}. Skipping...")
#         return item_id, None
#     if os.path.exists(image_path):
#         #         print(f"Image for item ID: {item_id} already downloaded. Skipping...")
#         return item_id, image_path

#     try:
#         signal.signal(signal.SIGALRM, timeout_handler)
#         signal.alarm(120)  # Set timeout to 120 seconds

#         with urllib.request.urlopen(url) as res:
#             image = Image.open(BytesIO(res.read()))
#             # Convert images to RGB, dropping alpha if present
#             if image.mode in ["RGBA", "LA"] or (
#                 image.mode == "P" and "transparency" in image.info
#             ):
#                 image = image.convert("RGBA")  # Ensure it's RGBA to preserve blending
#                 background = Image.new("RGBA", image.size, (255, 255, 255, 255))
#                 background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
#                 image = background.convert("RGB")
#             else:
#                 image = image.convert("RGB")

#             image.save(image_path)
#             #             print(f"Downloaded and saved image for item ID {item_id} to {image_path}")
#             signal.alarm(0)  # Disable the alarm
#             return item_id, image_path
#     except TimeoutException:
#         print(f"Download for item ID {item_id} timed out. Skipping...")
#         return item_id, None
#     except Exception as e:
#         print(f"Error downloading image from {url}: {e}")
#         return item_id, None
#     finally:
#         signal.alarm(0)  # Ensure the alarm is disabled


# Image download and save function
def download_and_save_image(args):
    url, folder_path, item_id = args
    image_path = os.path.join(folder_path, f"{item_id}.jpg")
    lock_path = image_path + ".lock"  # File lock to avoid race conditions

    # Handle invalid or empty URLs
    if url is None or pd.isna(url) or url.lower() in ["na", "<na>", "none", "null"]:
        return item_id, None

    try:
        # Acquire the file lock to ensure only one process writes the file,
        # with a timeout to avoid deadlocks
        with FileLock(lock_path, timeout=60):
            # Re-check existence after acquiring the lock (prevent race conditions)
            if os.path.exists(image_path):
                return item_id, image_path

            try:
                # Download the image
                with urllib.request.urlopen(url) as res:
                    image = Image.open(BytesIO(res.read()))

                    # Convert to RGB (remove alpha channel if necessary)
                    if image.mode in ["RGBA", "LA"] or (
                        image.mode == "P" and "transparency" in image.info
                    ):
                        image = image.convert("RGBA")
                        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                        background.paste(image, mask=image.split()[3])
                        image = background.convert("RGB")
                    else:
                        image = image.convert("RGB")

                    # Save the image directly to the final path
                    image.save(image_path)

                    return item_id, image_path
            except Exception as e:
                logger.error(f"Error downloading image from {url}: {e}")
                return item_id, None

    except Timeout:
        logger.error(f"Timeout acquiring lock for {item_id}. Skipping...")
        return item_id, None

    finally:
        # Ensure the lock is released even if an exception occurs
        if os.path.exists(lock_path):
            os.remove(lock_path)  # Manual cleanup in case the lock persists


def download_images_in_parallel_and_save(image_data, folder_path, max_workers=10):
    """
    Downloads images from a list of URLs in parallel using multiprocessing
    and saves them to a specified local folder.

    Parameters:
    - image_data (dict): Dictionary with item IDs as keys and image URLs as values.
    - folder_path (str): Path to the folder where images will be saved.
    - max_workers (int): Maximum number of worker processes.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    pool_data = [(url, folder_path, filename) for filename, url in image_data.items()]
    results = {}

    with multiprocessing.Pool(processes=max_workers) as pool, tqdm(
        total=len(pool_data), desc="Downloading images", disable=False
    ) as pbar:

        async_results = [
            pool.apply_async(download_and_save_image, (data,)) for data in pool_data
        ]

        for async_result in async_results:
            try:
                # Set a timeout of 120 seconds for each download task
                item_id, image_path = async_result.get(timeout=120)
                results[item_id] = image_path
            except TimeoutError:
                logger.error(f"Download for item ID {item_id} timed out. Skipping...")
            pbar.update()

    return results


# def download_images_in_parallel_and_save(
#     image_data, folder_path, max_workers=10, verbose=True
# ):
#     """
#     Downloads images from a list of URLs in parallel using multiprocessing and saves them to a specified local folder.

#     Parameters:
#     - image_data (dict): Dictionary with item IDs as keys and image URLs as values.
#     - folder_path (str): Path to the folder where images will be saved.
#     - max_workers (int): Maximum number of worker processes.

#     """
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     pool_data = [(url, folder_path, filename) for filename, url in image_data.items()]
#     results = {}
#     with multiprocessing.Pool(processes=max_workers) as pool, tqdm(
#         total=len(pool_data), desc="Downloading images", disable=False
#     ) as pbar:

#         for item_id, image_path in pool.imap_unordered(
#             download_and_save_image, pool_data
#         ):
#             results[item_id] = image_path
#             pbar.update()

#     return results


def load_image_model(image_model_name, device):
    """Load image model and processor."""
    processor = AutoImageProcessor.from_pretrained(image_model_name)
    model = AutoModel.from_pretrained(image_model_name).to(device)
    model.eval()
    return processor, model


def extract_image_features(
    item_image_path_dict, processor, model, device, verbose=False
):
    """
    Extracts features for a list of PIL Image objects using a pre-trained image model.
    Parameters:
    - item_image_path_dict: Dictionary of item_id and image_path mappings.
    - processor: The image processor for preprocessing images.
    - model: The pre-trained image model.
    - device: The device to run the model on.
    - verbose: Show progress bar
    Returns:
    - A tensor containing image features.
    """
    disable = verbose == False
    image_features = []

    for item_id, image_path in tqdm(
        item_image_path_dict.items(), desc="Extracting image features", disable=disable
    ):
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            image_features.append(embeddings)
        else:
            # print("Image is None, using zeros vector")
            # Handle failed image downloads or missing images with zero vectors
            zero_vec = torch.zeros((1, model.config.hidden_size), device=device)
            image_features.append(zero_vec)

    # Combine all image features into a single tensor
    return torch.cat(image_features, dim=0)


def preprocess_and_encode_attributes(
    df,
    text_model,
    processor,
    image_model,
    device,
    item_image_path_dict,
    attribute_weights={
        "brand_name": 1,
        "platform": 1,
        "channel_id": 1,
        "category": 1.5,
        "name": 1,
        "color": 1,
        "size": 1,
        "price": 1.5,
        "image": 2,
    },
    batch_size=100,
    verbose=False,
):
    """
    Preprocesses and encodes all specified attributes of items, combining them into a single vector per item.
    Utilizes batch processing for textual attributes and handles image attributes separately.

    Parameters:
    - df: DataFrame containing the items and their attributes.
    - text_model: The SentenceTransformer model for encoding textual attributes.
    - processor: The image processor from the transformers library for image attributes.
    - image_model: The model for extracting features from images.
    - device: The device to run the image model on ('cpu' or 'cuda').
    - item_image_path_dict: The dictionary of item ID and image path mappings.
    - attribute_weights: Dictionary specifying weights for each attribute.
    - batch_size: The number of items to process in each batch.
    - verbose: Show progress

    Returns:
    - A numpy array of the combined vectors for all items, with efficient memory usage.
    """
    if not (
        df.index.is_monotonic_increasing
        and df.index.is_unique
        and df.index[0] == 0
        and df.index[-1] == len(df) - 1
    ):
        raise ValueError("DataFrame's index must be sequential starting from 0.")

    # Normalize price attribute before processing batches
    if "price" in attribute_weights:
        scaler = MinMaxScaler()
        df["normalized_price"] = (
            scaler.fit_transform(df[["price"]]) * attribute_weights["price"]
        )
        price_feature_dim = 1  # Since price is a single value
    else:
        price_feature_dim = 0

    # print(f"Starting attribute encoding with batch size: {batch_size}...")

    # Determine the size of the feature vector for text attributes
    text_feature_dim = text_model.get_sentence_embedding_dimension()  # 384
    image_feature_dim = image_model.config.hidden_size  # 384
    total_feature_dim = text_feature_dim + image_feature_dim + price_feature_dim

    # Initialize an empty list to store combined vectors
    combined_vectors = np.zeros((0, total_feature_dim), dtype=np.float32)

    for start_idx in tqdm(
        range(0, len(df), batch_size), desc="Batch processing", disable=False
    ):
        end_idx = start_idx + batch_size
        batch_df = df.iloc[start_idx:end_idx]
        batch_item_image_path_dict = {
            item_id: item_image_path_dict[item_id]
            for item_id in batch_df["id"].tolist()
        }

        # Initialize batch combined vectors with zeros
        batch_combined_vectors = np.zeros(
            (len(batch_df), total_feature_dim), dtype=np.float32
        )

        # Textual and categorical attribute processing
        for attribute, weight in attribute_weights.items():
            if attribute not in ["image", "price"]:
                # if verbose:
                # print(f"Encoding and weighting {attribute}...")
                preprocessed_text = (
                    batch_df[attribute]
                    .fillna("unknown")
                    .str.strip()
                    .replace("", "unknown")
                )
                encoded_vectors = (
                    np.array(text_model.encode(preprocessed_text.tolist())) * weight
                )
                batch_combined_vectors[:, :text_feature_dim] += encoded_vectors

        # Process image attribute if included
        if "image" in attribute_weights:
            # if verbose:
            # print("Processing image attributes...")
            image_features = (
                extract_image_features(
                    batch_item_image_path_dict, processor, image_model, device, verbose
                )
                * attribute_weights["image"]
            )
            batch_combined_vectors[
                :, text_feature_dim : text_feature_dim + image_feature_dim
            ] = (image_features.cpu().detach().numpy())

        # Adding normalized and weighted price to the combined vectors
        if "price" in attribute_weights:
            # if verbose:
            #     print("Processing price attributes...")
            batch_combined_vectors[:, -price_feature_dim:] = batch_df[
                ["normalized_price"]
            ].to_numpy()

        # Concatenate the batch results
        combined_vectors = np.vstack((combined_vectors, batch_combined_vectors))

    print("Completed encoding all attributes.")
    return combined_vectors


def find_similar_items_with_image(
    all_item_df: pd.DataFrame,
    old_item_df: pd.DataFrame,
    new_item_df: pd.DataFrame,
    text_model_name: str = "all-MiniLM-L6-v2",
    image_model_name: str = "facebook/dinov2-small",
    top_k: int = 5,
    item_image_path_dict: dict = None,
    **kwargs,
):
    """
    Finds similar old items for each new item based on a combination of textual,
    categorical, and image attributes.

    Parameters:
    - all_item_df (pd.DataFrame): DataFrame containing all items with their attributes.
    - old_item_df (pd.DataFrame): DataFrame containing only old items.
    - new_item_df (pd.DataFrame): DataFrame containing only new items.

    - **kwargs: Additional keyword arguments passed directly to `preprocess_and_encode_attributes` function.
      Could include `attribute_weights`, `verbose`, etc.
        - attribute_weights (dict): Weights for each attribute to be considered in the similarity calculation.
        - batch_size: Batch size to process each subset.
        - verbose (bool): Show progress bar.
        - [TODO] n_components (int, optional): Number of dimensions for PCA reduction. If None, PCA is not applied.

    Returns:
    - dict: A dictionary where keys are IDs of new items and values are lists of IDs of the top_k most similar old items.

    The function encodes textual, categorical, and image attributes using the specified models,
    combines these attributes into a single feature vector per item (applying the specified weights),
    and then calculates the cosine similarity between new and old items to find the most similar old items
    for each new item. An optional PCA step can be applied for dimensionality reduction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model = SentenceTransformer(text_model_name)
    processor, image_model = load_image_model(image_model_name, device)

    if not (
        all_item_df.index.is_monotonic_increasing
        and all_item_df.index.is_unique
        and all_item_df.index[0] == 0
        and all_item_df.index[-1] == len(all_item_df) - 1
    ):
        # print("Dataframe's index is not sequential, resetting index")
        all_item_df = all_item_df.reset_index(drop=True)

    # Prepare and encode attributes, including images
    combined_vectors = preprocess_and_encode_attributes(
        all_item_df,
        text_model,
        processor,
        image_model,
        device,
        item_image_path_dict,
        **kwargs,
    )

    # Normalize combined feature vectors
    normalized_vectors = normalize(combined_vectors, axis=1)

    # Separate vectors for new and old items based on their IDs
    new_item_ids = new_item_df["id"].tolist()
    old_item_ids = old_item_df["id"].tolist()

    new_vectors = normalized_vectors[all_item_df["id"].isin(new_item_ids)]
    old_vectors = normalized_vectors[all_item_df["id"].isin(old_item_ids)]

    # Calculate cosine similarity between new and old items' vectors
    cos_sim = linear_kernel(new_vectors, old_vectors)

    # Identify top k old items with highest cosine similarity for each new item
    top_indices = np.argsort(-cos_sim, axis=1)[:, :top_k]

    # Map new items to their most similar old items
    similar_item_dict = {
        new_id: [old_item_ids[index] for index in indices]
        for new_id, indices in zip(new_item_ids, top_indices)
    }

    return similar_item_dict


def find_similar_set_for_new_products_with_image(
    full_pred_df, lookup_item_list, new_items_list, **kwargs
):
    """
    Finds similar products for new products within each brand and channel group,
    considering both textual and image attributes.

    Parameters:
    - full_pred_df (pd.DataFrame): DataFrame containing all product predictions.
    - lookup_item_list (list): List of item IDs to look up.
    - **kwargs: Additional keyword arguments passed directly to find_similar_items_with_image function.
      Could include:
          `text_model_name`, `image_model_name`, `top_k`,
          `attribute_weights`, `batch_size`, `verbose`

    Returns:
    - dict: A dictionary with new product IDs as keys and lists of similar old product IDs as values.
    """
    # verbose = kwargs.get("verbose", False)
    max_workers = kwargs.get("max_workers", 10)

    result = {}
    for (brand_name, platform, channel_id), group_df in full_pred_df.groupby(
        ["brand_name", "platform", "channel_id"]
    ):
        print(brand_name, platform, channel_id)
        folder_path = os.path.join(PROJECT_DATA_PATH, "images", brand_name)
        os.makedirs(folder_path, exist_ok=True)
        all_product_df = group_df[group_df["is_product"] == True]
        # all_product_df = preprocess_image_urls(all_product_df)
        # new_product_df = all_product_df[all_product_df["type"] == "new"]
        if new_items_list is None:
            new_product_df = all_product_df[
                all_product_df["type"].isin(["new", "lumpy"])
            ]
        else:
            new_product_df = all_product_df[all_product_df["id"].isin(new_items_list)]

        if new_product_df.shape[0] > 0:
            if brand_name in ["ipolita"]:
                old_product_df = all_product_df[
                    all_product_df["id"].isin(lookup_item_list)
                ]
            else:
                old_product_df = all_product_df[
                    all_product_df["id"].isin(lookup_item_list)
                    # & (all_product_df["id"] != "new")
                    & ~(all_product_df["id"].isin(["new", "lumpy"]))
                ]
            if len(old_product_df) == 0:
                similar_product_dict = {
                    item_id: None for item_id in new_product_df["id"].tolist()
                }
            else:
                # Unique identifier to prevent re-download of the same images across different channels
                all_product_df["unique_image_id"] = all_product_df.apply(
                    lambda x: f"{x['brand_name']}_{x['platform']}_{x['product_id']}_NA",
                    axis=1,
                )
                image_data = (
                    all_product_df.drop_duplicates("unique_image_id")
                    .set_index("unique_image_id")["first_image_url"]
                    .to_dict()
                )
                # Item ID is only: <brand_name>_<product_id>_NA
                results = download_images_in_parallel_and_save(
                    image_data,
                    folder_path,
                    max_workers=max_workers,
                )
                # Post process results to get actual item_id
                results = {
                    f"{filename}_{channel_id}": image_path
                    for filename, image_path in results.items()
                }

                count_none = sum(val is None for val in results.values())
                print(f"Total number of `None` images: {count_none} / {len(results)}")

                similar_product_dict = find_similar_items_with_image(
                    all_product_df,
                    old_product_df,
                    new_product_df,
                    item_image_path_dict=results,
                    **kwargs,
                )

            result.update(similar_product_dict)

    return result


def find_similar_set_for_new_variants_with_image(
    meta_df, lookup_item_list, new_items_list, **kwargs
):
    """
    Finds similar products for new products within each brand and channel group,
    considering both textual and image attributes.

    Parameters:
    - full_pred_df (pd.DataFrame): DataFrame containing all product predictions.
    - lookup_item_list (list): List of item IDs to look up.
    - **kwargs: Additional keyword arguments passed directly to find_similar_items_with_image function.
      Could include:
          `text_model_name`, `image_model_name`, `top_k`,
          `attribute_weights`, `batch_size`, `verbose`

    Returns:
    - dict: A dictionary with new product IDs as keys and lists of similar old product IDs as values.
    """
    verbose = kwargs.get("verbose", False)
    max_workers = kwargs.get("max_workers", 10)

    result = {}
    for (brand_name, platform, channel_id), group_df in meta_df.groupby(
        ["brand_name", "platform", "channel_id"]
    ):
        print(brand_name, platform, channel_id)
        folder_path = os.path.join(PROJECT_DATA_PATH, "images", brand_name)
        os.makedirs(folder_path, exist_ok=True)
        all_df = group_df[group_df["is_product"] == False]
        if new_items_list is None:
            new_df = all_df[all_df["type"].isin(["new", "lumpy"])]
        else:
            new_df = all_df[all_df["id"].isin(new_items_list)]

        if new_df.shape[0] > 0:
            if brand_name in ["ipolita"]:
                old_df = all_df[all_df["id"].isin(lookup_item_list)]
            else:
                old_df = all_df[
                    all_df["id"].isin(lookup_item_list)
                    & ~(all_df["id"].isin(["new", "lumpy"]))
                ]
            if len(old_df) == 0:
                similar_product_dict = {
                    item_id: None for item_id in new_df["id"].tolist()
                }
            else:
                # Unique identifier to prevent re-download of the same images across different channels
                all_df["unique_image_id"] = all_df.apply(
                    lambda x: f"{x['brand_name']}_{x['platform']}_{x['product_id']}_{x['variant_id']}",
                    axis=1,
                )
                image_data = (
                    all_df.drop_duplicates("unique_image_id")
                    .set_index("unique_image_id")["first_image_url"]
                    .to_dict()
                )
                # Item ID is only: <brand_name>_<product_id>_NA
                results = download_images_in_parallel_and_save(
                    image_data, folder_path, max_workers=max_workers, verbose=verbose
                )
                # Post process results to get actual item_id
                results = {
                    f"{filename}_{channel_id}": image_path
                    for filename, image_path in results.items()
                }

                count_none = sum(val is None for val in results.values())
                print(f"Total number of `None` images: {count_none} / {len(results)}")

                similar_product_dict = find_similar_items_with_image(
                    all_df,
                    old_df,
                    new_df,
                    item_image_path_dict=results,
                    **kwargs,
                )

            result.update(similar_product_dict)

    return result


def preprocess_image_urls(df):
    """
    Extracts the first valid image URL from a string of URLs separated by ', ' for each item.
    Invalid URLs, including 'NA', empty strings, or whitespace, are ignored.

    Parameters:
    - df: DataFrame containing the items and their image URLs as strings.

    Returns:
    - DataFrame with an added 'first_image_url' column containing the first valid image URL.
    """

    def first_valid_url(url_string):
        if pd.isna(url_string):
            return pd.NA
        urls = url_string.split(", ")
        # Filter out 'NA', empty strings, or any placeholder indicating an invalid URL
        valid_urls = [
            url for url in urls if url.strip().lower() not in ["na", "", "none"]
        ]
        return valid_urls[0] if valid_urls else None

    return df.assign(first_image_url=df["image"].apply(first_valid_url))
