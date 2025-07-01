"""
Data loading utilities for the Stock Tweet User Credibility project.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from datasets import load_dataset

from .constants import (
    FLARE_DATASET_SPLITS,
    FLARE_DATASET_URL_PREFIX,
    ACL18_PRICE_DATA_DIR,
    ACL18_TWEET_DATA_DIR,
    STOCK_TABLE_PATH
)
from .utils import setup_logging

logger = setup_logging(__name__)


class DataLoader:
    """Centralized data loading for the project."""
    
    def __init__(self):
        self.price_data_cache = {}
        self.stock_mapping = self._load_stock_mapping()
    
    def _load_stock_mapping(self) -> Dict[str, str]:
        """Load ticker to company name mapping."""
        mapping = {}
        try:
            with open(STOCK_TABLE_PATH, 'r') as f:
                next(f)  # Skip header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        ticker = parts[1].strip().replace('$', '')
                        company = parts[2].strip()
                        mapping[ticker] = company
            logger.info(f"Loaded stock mapping for {len(mapping)} tickers")
        except Exception as e:
            logger.warning(f"Could not load stock mapping: {e}")
        return mapping
    
    def load_flare_dataset(self, split_name: str) -> pd.DataFrame:
        """Load FLARE dataset for specified split."""
        file_path = FLARE_DATASET_SPLITS.get(split_name)
        if not file_path:
            raise ValueError(f"Invalid split name: {split_name}. Available: {list(FLARE_DATASET_SPLITS.keys())}")
        
        try:
            # Check if local file exists
            local_path = Path(file_path)
            if local_path.exists():
                logger.info(f"Loading local Parquet file: {local_path}")
                return pd.read_parquet(local_path)
            else:
                # Load from Hugging Face
                logger.info(f"Loading from Hugging Face: {FLARE_DATASET_URL_PREFIX + file_path}")
                dataset = load_dataset(
                    "parquet",
                    data_files=FLARE_DATASET_URL_PREFIX + file_path,
                    split="train"
                )
                return dataset.to_pandas()
        except Exception as e:
            logger.error(f"Error loading dataset for split {split_name}: {e}")
            raise
    
    def load_stock_price_data(self, ticker: str) -> pd.DataFrame:
        """Load and cache stock price data for a ticker."""
        if ticker in self.price_data_cache:
            return self.price_data_cache[ticker]
        
        try:
            # Try with and without $ prefix
            file_path = ACL18_PRICE_DATA_DIR / f"{ticker.replace('$', '')}.csv"
            if not file_path.exists():
                file_path = ACL18_PRICE_DATA_DIR / f"{ticker}.csv"
            
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)
            
            self.price_data_cache[ticker] = df
            return df
        
        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def load_tweet_file(self, ticker: str, date_str: str) -> List[Dict[str, Any]]:
        """Load tweets from a specific file."""
        file_path = ACL18_TWEET_DATA_DIR / ticker.upper() / date_str
        tweets = []
        
        if not file_path.exists():
            logger.debug(f"Tweet file not found: {file_path}")
            return tweets
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        tweet_data = json.loads(line)
                        tweets.append(tweet_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in {file_path} at line {line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
        
        return tweets
    
    def load_cached_dataframe(self, file_path: Path, file_type: str = 'auto') -> pd.DataFrame:
        """
        Load a cached DataFrame from file.
        
        Args:
            file_path: Path to the file
            file_type: 'csv', 'parquet', 'pickle', or 'auto' to detect from extension
            
        Returns:
            Loaded DataFrame
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_type == 'auto':
            if file_path.suffix == '.csv':
                file_type = 'csv'
            elif file_path.suffix == '.parquet':
                file_type = 'parquet'
            elif file_path.suffix in ['.pkl', '.pickle']:
                file_type = 'pickle'
            else:
                raise ValueError(f"Cannot auto-detect file type for: {file_path}")
        
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path, encoding='utf-8-sig')
            elif file_type == 'parquet':
                return pd.read_parquet(file_path)
            elif file_type == 'pickle':
                return pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logger.error(f"Error loading {file_type} file {file_path}: {e}")
            raise
    
    def save_dataframe(self, df: pd.DataFrame, file_path: Path, file_type: str = 'auto') -> None:
        """
        Save a DataFrame to file.
        
        Args:
            df: DataFrame to save
            file_path: Path to save to
            file_type: 'csv', 'parquet', 'pickle', or 'auto' to detect from extension
        """
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_type == 'auto':
            if file_path.suffix == '.csv':
                file_type = 'csv'
            elif file_path.suffix == '.parquet':
                file_type = 'parquet'
            elif file_path.suffix in ['.pkl', '.pickle']:
                file_type = 'pickle'
            else:
                raise ValueError(f"Cannot auto-detect file type for: {file_path}")
        
        try:
            if file_type == 'csv':
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            elif file_type == 'parquet':
                df.to_parquet(file_path, index=False)
            elif file_type == 'pickle':
                df.to_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            logger.info(f"Saved DataFrame to {file_path}")
        except Exception as e:
            logger.error(f"Error saving {file_type} file {file_path}: {e}")
            raise
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise
    
    def parse_twitter_data(self, input_path: str, output_path: str, output_format: str = "csv"):
        """
        Parse Twitter JSON data into flattened structure.
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output file
            output_format: Output format ('csv' or 'parquet')
        """
        logger.info(f"Parsing Twitter data from {input_path}")
        
        try:
            import json
            
            # Read JSON data
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flatten the data structure (this would need specific implementation based on Twitter data format)
            # For now, just convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
            
            # Save based on format
            if output_format == "csv":
                df.to_csv(output_path, index=False)
            elif output_format == "parquet":
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
            logger.info(f"Parsed {len(df)} records to {output_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing Twitter data: {e}")
            raise