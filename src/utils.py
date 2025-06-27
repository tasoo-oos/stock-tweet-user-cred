"""
Common utility functions for the Stock Tweet User Credibility project.
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
import numpy as np
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging with consistent format across the project.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    console = Console()
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    return logging.getLogger(name)


def get_business_days_later(date_str: str, business_days: int = 10) -> str:
    """
    Calculate date that is N business days after given date.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        business_days: Number of business days to add
        
    Returns:
        Date string in YYYY-MM-DD format
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    days_added = 0
    current_date = date
    
    while days_added < business_days:
        current_date += timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            days_added += 1
    
    return current_date.strftime('%Y-%m-%d')


def load_stock_data(ticker: str, price_data_path: Path) -> pd.DataFrame:
    """
    Load stock price data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        price_data_path: Path to price data directory
        
    Returns:
        DataFrame with stock price data, empty if not found
    """
    csv_file = price_data_path / f"{ticker}.csv"
    
    if not csv_file.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        logging.error(f"Error loading stock data for {ticker}: {e}")
        return pd.DataFrame()


def load_stock_mapping(stock_table_path: Path) -> Dict[str, str]:
    """
    Load ticker to company name mapping from StockTable file.
    
    Args:
        stock_table_path: Path to StockTable file
        
    Returns:
        Dictionary mapping ticker symbols to company names
    """
    mapping = {}
    try:
        with open(stock_table_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    ticker = parts[1].strip().replace('$', '')
                    company = parts[2].strip()
                    mapping[ticker] = company
        logging.info(f"Loaded stock mapping for {len(mapping)} tickers")
    except Exception as e:
        logging.warning(f"Could not load stock mapping: {e}")
    return mapping


def extract_tweet_id_from_custom_id(custom_id: str) -> Optional[int]:
    """
    Extract tweet ID from custom ID string.
    
    Args:
        custom_id: Custom ID string (e.g., "request-20250624_114958-0-123456-rise")
        
    Returns:
        Tweet ID as integer, or None if extraction fails
    """
    try:
        return int(custom_id.split('-')[-1])
    except (ValueError, AttributeError, IndexError):
        logging.warning(f"Could not extract tweet ID from: {custom_id}")
        return None


def parse_custom_id(custom_id: str) -> Dict[str, Any]:
    """
    Parse components from a custom ID string.
    
    Args:
        custom_id: Custom ID string
        
    Returns:
        Dictionary with parsed components
    """
    try:
        parts = custom_id.split('-')
        result = {
            'ticker': custom_id.split('-20')[0].split('tweet-sentiment-')[-1] if 'tweet-sentiment-' in custom_id else None,
            'date': '201' + custom_id.split('-201')[1].split(' ')[0] if '-201' in custom_id else None,
            'tweet_id': parts[-1] if parts else None
        }
        return result
    except Exception as e:
        logging.error(f"Error parsing custom ID {custom_id}: {e}")
        return {'ticker': None, 'date': None, 'tweet_id': None}


def ensure_directory_exists(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)