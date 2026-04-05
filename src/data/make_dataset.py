import os
import urllib.request
import gzip
import json
import random
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

URLS = [
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Home_and_Kitchen.jsonl.gz",
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Beauty_and_Personal_Care.jsonl.gz",
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz"
]

def download_file(url, out_dir):
    """Download file from URL to the target directory with a progress bar."""
    os.makedirs(out_dir, exist_ok=True)
    filename = url.split("/")[-1]
    out_path = Path(out_dir) / filename
    
    if out_path.exists():
        logger.info(f"File {filename} already exists, skipping download.")
        return out_path
        
    logger.info(f"Downloading {filename}...")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
            
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)
        
    return out_path

def reservoir_sample_gz(file_path, n_samples):
    """Reservoir sampling for large gzipped jsonl files."""
    sampled_lines = []
    
    logger.info(f"Sampling {n_samples} from {file_path}")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Reading file")):
            if i < n_samples:
                sampled_lines.append(line)
            else:
                j = random.randint(0, i)
                if j < n_samples:
                    sampled_lines[j] = line
                    
    # Parse json and handle errors safely
    records = []
    for line in tqdm(sampled_lines, desc="Parsing JSON"):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
            
    return pd.DataFrame(records)

def clean_and_map(df, category_name):
    """Filter columns and map ratings to binary/tertiary sentiment."""
    # Keep only necessary columns
    keep_cols = ['rating', 'text', 'title', 'parent_asin', 'helpful_vote', 'verified_purchase', 'timestamp']
    # Not all datasets guarantee these columns, keep the ones we have
    available_cols = [c for c in keep_cols if c in df.columns]
    
    df = df[available_cols].copy()
    
    # Add category
    df['main_category'] = category_name
    
    # Map sentiment (1-2: negative, 3: neutral, 4-5: positive)
    def map_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
            
    if 'rating' in df.columns:
        df['sentiment'] = df['rating'].apply(map_sentiment)
        
    return df

def process_data(data_dir="data", n_samples=500000, random_seed=42):
    """Main execution function to download, sample and save the dataset."""
    random.seed(random_seed)
    
    raw_dir = Path(data_dir) / "raw" / "amazon_reviews_2023" / "reviews"
    interim_dir = Path(data_dir) / "interim"
    os.makedirs(interim_dir, exist_ok=True)
    
    master_colab_file = interim_dir / "amazon_reviews_4cat_2m.parquet"
    if master_colab_file.exists():
        logger.info(f"Found master dataset, loading directly: {master_colab_file}")
        df_master = pd.read_parquet(master_colab_file)
        # Colab compatibility: rename "category" to "main_category" if needed
        if "category" in df_master.columns and "main_category" not in df_master.columns:
            df_master = df_master.rename(columns={"category": "main_category"})
            
        # Ensure sentiment is present
        if "sentiment" not in df_master.columns and "rating" in df_master.columns:
            def map_sentiment(rating):
                if rating <= 2:
                    return 'negative'
                elif rating == 3:
                    return 'neutral'
                else:
                    return 'positive'
            df_master['sentiment'] = df_master['rating'].apply(map_sentiment)
            
        return df_master
    
    sampled_dfs = []
    for url in URLS:
        category_name = url.split('/')[-1].replace('.jsonl.gz', '')
        interim_path = interim_dir / f"{category_name}_sample_{n_samples}.parquet"
        
        if interim_path.exists():
            logger.info(f"Loading already sampled data for {category_name}")
            df = pd.read_parquet(interim_path)
        else:
            gz_path = download_file(url, raw_dir)
            df = reservoir_sample_gz(gz_path, n_samples)
            df = clean_and_map(df, category_name)
            df.to_parquet(interim_path)
            
        sampled_dfs.append(df)
        
    logger.info("Combining all categories...")
    combined_df = pd.concat(sampled_dfs, ignore_index=True)
    
    final_output = interim_dir / f"amazon_reviews_4_categories_{len(combined_df)}.parquet"
    combined_df.to_parquet(final_output)
    logger.info(f"Saved combined dataset to {final_output}")
    
    return combined_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_data()
