import json
import os

import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any

# Set up logging to track progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def flatten_tweet_data(tweet_data: Dict[str, Any], stock_ticker: str, date_str: str, file_path: str) -> Dict[str, Any]:
    """
    Flatten nested Twitter JSON data into a flat dictionary suitable for DataFrame.

    Args:
        tweet_data: Raw tweet JSON data
        stock_ticker: Stock ticker symbol
        date_str: Date string from directory
        file_path: Path to source file

    Returns:
        Flattened dictionary with all nested data as separate columns
    """
    flattened = {}

    # Add our metadata first
    flattened['stock_ticker'] = stock_ticker
    flattened['file_date'] = date_str
    flattened['file_path'] = file_path

    # Main tweet fields (non-nested ones)
    main_fields = [
        'created_at', 'id', 'id_str', 'text', 'source', 'truncated',
        'in_reply_to_status_id', 'in_reply_to_status_id_str',
        'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name',
        'retweet_count', 'favorite_count', 'favorited', 'retweeted',
        'possibly_sensitive', 'filter_level', 'lang'
    ]

    for field in main_fields:
        flattened[field] = tweet_data.get(field)

    # Flatten user data
    user_data = tweet_data.get('user', {})
    if user_data:
        flattened.update(flatten_user_data(user_data))

    # Flatten entities data
    entities_data = tweet_data.get('entities', {})
    if entities_data:
        flattened.update(flatten_entities_data(entities_data))

    # Handle location data
    flattened.update(flatten_location_data(tweet_data))

    # Handle retweet data (if this is a retweet)
    retweeted_status = tweet_data.get('retweeted_status', {})
    if retweeted_status:
        flattened.update(flatten_retweet_data(retweeted_status))

    return flattened


def flatten_user_data(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten user object into user_* columns."""
    user_fields = {}

    # Direct user fields
    user_mapping = {
        'id': 'user_id',
        'id_str': 'user_id_str',
        'name': 'user_name',
        'screen_name': 'user_screen_name',
        'location': 'user_location',
        'url': 'user_url',
        'description': 'user_description',
        'protected': 'user_protected',
        'followers_count': 'user_followers_count',
        'friends_count': 'user_friends_count',
        'listed_count': 'user_listed_count',
        'created_at': 'user_created_at',
        'favourites_count': 'user_favourites_count',
        'utc_offset': 'user_utc_offset',
        'time_zone': 'user_time_zone',
        'geo_enabled': 'user_geo_enabled',
        'verified': 'user_verified',
        'statuses_count': 'user_statuses_count',
        'lang': 'user_lang',
        'contributors_enabled': 'user_contributors_enabled',
        'is_translator': 'user_is_translator',
        'profile_background_color': 'user_profile_background_color',
        'profile_background_image_url': 'user_profile_background_image_url',
        'profile_background_image_url_https': 'user_profile_background_image_url_https',
        'profile_background_tile': 'user_profile_background_tile',
        'profile_image_url': 'user_profile_image_url',
        'profile_image_url_https': 'user_profile_image_url_https',
        'profile_banner_url': 'user_profile_banner_url',
        'profile_link_color': 'user_profile_link_color',
        'profile_sidebar_border_color': 'user_profile_sidebar_border_color',
        'profile_sidebar_fill_color': 'user_profile_sidebar_fill_color',
        'profile_text_color': 'user_profile_text_color',
        'profile_use_background_image': 'user_profile_use_background_image',
        'default_profile': 'user_default_profile',
        'default_profile_image': 'user_default_profile_image',
        'following': 'user_following',
        'follow_request_sent': 'user_follow_request_sent',
        'notifications': 'user_notifications'
    }

    for original_field, new_field in user_mapping.items():
        user_fields[new_field] = user_data.get(original_field)

    return user_fields


def flatten_entities_data(entities_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten entities object into separate columns for hashtags, symbols, URLs, mentions."""
    entities_fields = {}

    # Extract hashtags
    hashtags = entities_data.get('hashtags', [])
    hashtag_texts = [tag.get('text', '') for tag in hashtags if isinstance(tag, dict)]
    entities_fields['hashtags'] = '|'.join(hashtag_texts) if hashtag_texts else None
    entities_fields['hashtags_count'] = len(hashtag_texts)

    # Extract stock symbols
    symbols = entities_data.get('symbols', [])
    symbol_texts = [symbol.get('text', '') for symbol in symbols if isinstance(symbol, dict)]
    entities_fields['symbols'] = '|'.join(symbol_texts) if symbol_texts else None
    entities_fields['symbols_count'] = len(symbol_texts)

    # Extract URLs
    urls = entities_data.get('urls', [])
    url_data = []
    expanded_urls = []
    display_urls = []

    for url in urls:
        if isinstance(url, dict):
            url_data.append(url.get('url', ''))
            expanded_urls.append(url.get('expanded_url', ''))
            display_urls.append(url.get('display_url', ''))

    entities_fields['urls'] = '|'.join(url_data) if url_data else None
    entities_fields['expanded_urls'] = '|'.join(expanded_urls) if expanded_urls else None
    entities_fields['display_urls'] = '|'.join(display_urls) if display_urls else None
    entities_fields['urls_count'] = len(url_data)

    # Extract user mentions
    mentions = entities_data.get('user_mentions', [])
    mentioned_users = []
    mentioned_user_ids = []

    for mention in mentions:
        if isinstance(mention, dict):
            mentioned_users.append(mention.get('screen_name', ''))
            mentioned_user_ids.append(str(mention.get('id', '')))

    entities_fields['mentioned_users'] = '|'.join(mentioned_users) if mentioned_users else None
    entities_fields['mentioned_user_ids'] = '|'.join(mentioned_user_ids) if mentioned_user_ids else None
    entities_fields['mentions_count'] = len(mentioned_users)

    return entities_fields


def flatten_location_data(tweet_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract location-related fields."""
    location_fields = {}

    # Geo coordinates
    geo = tweet_data.get('geo')
    if geo and isinstance(geo, dict):
        coordinates = geo.get('coordinates', [])
        if coordinates and len(coordinates) >= 2:
            location_fields['geo_latitude'] = coordinates[0]
            location_fields['geo_longitude'] = coordinates[1]

    # Coordinates (different format)
    coordinates = tweet_data.get('coordinates')
    if coordinates and isinstance(coordinates, dict):
        coords = coordinates.get('coordinates', [])
        if coords and len(coords) >= 2:
            location_fields['coordinates_longitude'] = coords[0]
            location_fields['coordinates_latitude'] = coords[1]

    # Place information
    place = tweet_data.get('place')
    if place and isinstance(place, dict):
        location_fields['place_id'] = place.get('id')
        location_fields['place_name'] = place.get('name')
        location_fields['place_full_name'] = place.get('full_name')
        location_fields['place_country'] = place.get('country')
        location_fields['place_country_code'] = place.get('country_code')
        location_fields['place_type'] = place.get('place_type')

    return location_fields


def flatten_retweet_data(retweeted_status: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten retweeted status data with rt_ prefix."""
    retweet_fields = {}

    # Main retweet fields
    rt_mapping = {
        'created_at': 'rt_created_at',
        'id': 'rt_id',
        'id_str': 'rt_id_str',
        'text': 'rt_text',
        'source': 'rt_source',
        'retweet_count': 'rt_retweet_count',
        'favorite_count': 'rt_favorite_count',
        'lang': 'rt_lang'
    }

    for original_field, new_field in rt_mapping.items():
        retweet_fields[new_field] = retweeted_status.get(original_field)

    # Retweeted user info
    rt_user = retweeted_status.get('user', {})
    if rt_user:
        retweet_fields['rt_user_id'] = rt_user.get('id')
        retweet_fields['rt_user_name'] = rt_user.get('name')
        retweet_fields['rt_user_screen_name'] = rt_user.get('screen_name')
        retweet_fields['rt_user_followers_count'] = rt_user.get('followers_count')
        retweet_fields['rt_user_verified'] = rt_user.get('verified')

    # Retweeted entities
    rt_entities = retweeted_status.get('entities', {})
    if rt_entities:
        # Hashtags in retweet
        rt_hashtags = rt_entities.get('hashtags', [])
        rt_hashtag_texts = [tag.get('text', '') for tag in rt_hashtags if isinstance(tag, dict)]
        retweet_fields['rt_hashtags'] = '|'.join(rt_hashtag_texts) if rt_hashtag_texts else None

        # Symbols in retweet
        rt_symbols = rt_entities.get('symbols', [])
        rt_symbol_texts = [symbol.get('text', '') for symbol in rt_symbols if isinstance(symbol, dict)]
        retweet_fields['rt_symbols'] = '|'.join(rt_symbol_texts) if rt_symbol_texts else None

        # URLs in retweet
        rt_urls = rt_entities.get('urls', [])
        rt_url_texts = [url.get('expanded_url', url.get('url', '')) for url in rt_urls if isinstance(url, dict)]
        retweet_fields['rt_urls'] = '|'.join(rt_url_texts) if rt_url_texts else None

    return retweet_fields


def load_twitter_data(base_path="dataset/tweet/raw"):
    """
    Load all Twitter data from the structured directory into a pandas DataFrame with flattened columns.

    Args:
        base_path (str): Path to the root directory containing stock ticker folders

    Returns:
        pd.DataFrame: Combined DataFrame with all tweet data flattened into columns
    """

    all_tweets = []  # List to collect all flattened tweet records
    processed_files = 0
    skipped_files = 0

    # Convert to Path object for easier manipulation
    base_path = Path(base_path)

    # Check if base directory exists
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory {base_path} does not exist")

    logger.info(f"Starting to process files from {base_path}")

    # Walk through each stock ticker directory
    for stock_dir in base_path.iterdir():
        if not stock_dir.is_dir():
            continue

        stock_ticker = stock_dir.name
        logger.info(f"Processing stock ticker: {stock_ticker}")

        # Walk through each date directory for this stock
        for file_path in stock_dir.iterdir():
            date_str = file_path.name

            # Validate date format (assuming YYYY-MM-DD)
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                logger.warning(f"Skipping invalid date directory: {date_str}")
                continue

            if file_path.is_file():
                try:
                    tweets_from_file = process_tweet_file(file_path, stock_ticker, date_str)
                    all_tweets.extend(tweets_from_file)
                    processed_files += 1

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    skipped_files += 1
                    continue

    logger.info(f"Completed processing. Files processed: {processed_files}, Files skipped: {skipped_files}")
    logger.info(f"Total tweets collected: {len(all_tweets)}")

    # Convert to DataFrame
    if all_tweets:
        df = pd.DataFrame(all_tweets)

        # Convert date columns to proper datetime format
        date_columns = ['created_at', 'user_created_at', 'rt_created_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')

        df['file_date'] = pd.to_datetime(df['file_date'])

        # Convert numeric columns
        numeric_columns = [
            'user_followers_count', 'user_friends_count', 'user_listed_count', 'user_favourites_count',
            'user_statuses_count',
            'retweet_count', 'favorite_count', 'hashtags_count', 'symbols_count', 'urls_count', 'mentions_count',
            'rt_retweet_count', 'rt_favorite_count', 'rt_user_followers_count'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info("DataFrame created successfully with flattened structure")
        return df
    else:
        logger.warning("No tweets found. Returning empty DataFrame")
        return pd.DataFrame()


def process_tweet_file(file_path, stock_ticker, date_str):
    """
    Process a single tweet file and extract all tweets with flattened structure.

    Args:
        file_path (Path): Path to the file containing tweets
        stock_ticker (str): Stock ticker symbol
        date_str (str): Date string in YYYY-MM-DD format

    Returns:
        list: List of flattened tweet dictionaries
    """
    tweets = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Parse JSON from each line
                    tweet_data = json.loads(line)

                    # Flatten the nested structure
                    flattened_tweet = flatten_tweet_data(tweet_data, stock_ticker, date_str, str(file_path))

                    tweets.append(flattened_tweet)

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in {file_path} at line {line_num}: {str(e)}")
                    continue

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

    return tweets


def analyze_dataframe(df):
    """
    Provide basic analysis of the loaded DataFrame.

    Args:
        df (pd.DataFrame): The loaded Twitter data DataFrame
    """
    if df.empty:
        print("DataFrame is empty - no data to analyze")
        return

    print("\n")
    print("=== TWITTER DATA ANALYSIS ===")
    print(f"Total tweets: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Date range: {df['file_date'].min()} to {df['file_date'].max()}")
    print(f"Number of stock tickers: {df['stock_ticker'].nunique()}")
    print(f"Stock tickers: {sorted(df['stock_ticker'].unique())}")

    print("\n=== TWEETS PER STOCK ===")
    stock_counts = df['stock_ticker'].value_counts()
    print(stock_counts.head(10))

    print(f"\n=== TWEETS PER USER ===")
    user_counts = df['user_id'].value_counts().reset_index().rename(columns={'count': 'tweet_count'}).sort_values(by='tweet_count', ascending=False)
    print(user_counts.head(10))
    print(f"Total unique users: {df['user_id'].nunique()}")
    print(f"User with more than 1000 tweets: {user_counts[user_counts['tweet_count'] > 1000].shape[0]}")
    print(f"User with more than 100 tweets: {user_counts[user_counts['tweet_count'] > 100].shape[0]}")
    print(f"User with more than 10 tweets: {user_counts[user_counts['tweet_count'] > 10].shape[0]}")

    print("\n=== RETWEET ANALYSIS ===")
    retweet_mask = df['rt_id'].notna()
    print(f"Total retweets: {retweet_mask.sum():,} ({retweet_mask.mean() * 100:.1f}%)")
    print(f"Original tweets: {(~retweet_mask).sum():,} ({(~retweet_mask).mean() * 100:.1f}%)")

    print("\n=== HASHTAG ANALYSIS ===")
    has_hashtags = df['hashtags'].notna()
    print(f"Tweets with hashtags: {has_hashtags.sum():,} ({has_hashtags.mean() * 100:.1f}%)")

    print("\n=== STOCK SYMBOL ANALYSIS ===")
    has_symbols = df['symbols'].notna()
    print(f"Tweets with stock symbols: {has_symbols.sum():,} ({has_symbols.mean() * 100:.1f}%)")

    print("\n=== COLUMN CATEGORIES ===")
    user_cols = [col for col in df.columns if col.startswith('user_')]
    entity_cols = [col for col in df.columns if col.startswith(('hashtags', 'symbols', 'urls', 'mentions'))]
    rt_cols = [col for col in df.columns if col.startswith('rt_')]
    location_cols = [col for col in df.columns if col.startswith(('geo_', 'coordinates_', 'place_'))]

    print(f"User columns: {len(user_cols)}")
    print(f"Entity columns: {len(entity_cols)}")
    print(f"Retweet columns: {len(rt_cols)}")
    print(f"Location columns: {len(location_cols)}")

    print("\n=== SAMPLE FLATTENED DATA ===")
    sample_cols = ['stock_ticker', 'file_date', 'text', 'user_screen_name', 'user_followers_count', 'hashtags',
                   'symbols']
    available_sample_cols = [col for col in sample_cols if col in df.columns]
    print(df[available_sample_cols].head(3).to_string())


def filter_overlapping_data(target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out overlapping data in the DataFrame.
    If "stock_ticker" and "text" are the same, keep only the first occurrence.

    Args:
        target_df (pd.DataFrame): The DataFrame to filter

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    # First, validate that our required columns exist in the DataFrame
    target_columns = ['text', 'file_date', 'stock_ticker']
    missing_columns = [col for col in target_columns if col not in target_df.columns]

    if missing_columns:
        logger.warning(f"Required columns {missing_columns} not found. Returning original DataFrame.")
        return target_df

    # Log the initial state before filtering
    initial_count = len(target_df)
    logger.info(f"Starting deduplication with {initial_count:,} total tweets")

    # Check for potential duplicates before removing them
    # This helps us understand the scope of the duplicate issue
    duplicate_mask = target_df.duplicated(subset=target_columns, keep=False)
    potential_duplicates = duplicate_mask.sum()

    if potential_duplicates > 0:
        logger.info(f"Found {potential_duplicates:,} tweets that are part of duplicate groups")

        # Show some examples of what we're about to remove (for debugging/validation)
        example_duplicates = target_df[duplicate_mask].groupby(target_columns).size()
        top_duplicates = example_duplicates.sort_values(ascending=False).head(3)
        logger.info(f"Top duplicate patterns: {dict(top_duplicates)}")

    # Remove duplicates based on the target_columns
    # keep='first' means we preserve the first occurrence of each duplicate group
    # This maintains chronological order since we typically process files in date order
    filtered_df = target_df.drop_duplicates(subset=target_columns, keep='first')

    # Calculate and log the results
    final_count = len(filtered_df)
    removed_count = initial_count - final_count

    if removed_count > 0:
        logger.info(f"Successfully removed {removed_count:,} duplicate tweets")
        logger.info(f"Deduplication rate: {(removed_count / initial_count) * 100:.2f}%")
        logger.info(f"Remaining unique tweets: {final_count:,}")
    else:
        logger.info("No duplicate tweets found - dataset is already clean")

    # Additional validation: ensure we haven't accidentally removed too much data
    if removed_count > initial_count * 0.5:  # More than 50% removed
        logger.warning(f"High deduplication rate detected ({(removed_count / initial_count) * 100:.1f}%). "
                       "Please verify this is expected for your dataset.")

    return filtered_df


# Main execution
if __name__ == "__main__":
    try:
        # Load all the data with flattened structure
        df = load_twitter_data("dataset/tweet/raw")

        if df.empty:
            print("No data found. Please check the dataset directory.")
            exit(1)

        # Filter out any overlapping data
        df = filter_overlapping_data(df)

        # Analyze the results
        analyze_dataframe(df)

        save_path = Path(os.getcwd()) / "cache"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save to file for future use
        pickle_file = "flattened_twitter_data.pkl"
        pickle_dir = save_path / pickle_file

        csv_file = "flattened_twitter_data.csv"
        csv_dir = save_path / csv_file

        df.to_pickle(pickle_dir)
        df.to_csv(csv_dir, index=False)

        print(f"\nFlattened DataFrame saved to {pickle_dir}")
        print(f"CSV version saved to {csv_dir}")
        print("You can load it later with: df = pd.read_pickle('flattened_twitter_data.pkl')")

        # Also save a CSV sample (first 1000 rows) for inspection
        # sample_file = "flattened_twitter_sample.csv"
        # df.head(1000).to_csv(sample_file, index=False)
        # print(f"Sample (first 1000 rows) saved to {sample_file}")

        # Save column information
        column_info_path = save_path / "column_info.txt"

        with open(column_info_path, 'wt', encoding='utf-8') as f:
            f.write("FLATTENED TWITTER DATA COLUMNS\n")
            f.write("=" * 50 + "\n\n")

            categories = {
                "Metadata": [col for col in df.columns if col in ['stock_ticker', 'file_date', 'file_path']],
                "Tweet Basic": [col for col in df.columns if not any(col.startswith(prefix) for prefix in
                                                                     ['user_', 'rt_', 'geo_', 'coordinates_',
                                                                      'place_', 'hashtags', 'symbols', 'urls',
                                                                      'mentions']) and col not in ['stock_ticker',
                                                                                                   'file_date',
                                                                                                   'file_path']],
                "User Info": [col for col in df.columns if col.startswith('user_')],
                "Entities": [col for col in df.columns if
                             col.startswith(('hashtags', 'symbols', 'urls', 'mentions'))],
                "Location": [col for col in df.columns if col.startswith(('geo_', 'coordinates_', 'place_'))],
                "Retweet": [col for col in df.columns if col.startswith('rt_')]
            }

            for category, cols in categories.items():
                if cols:
                    f.write(f"{category} ({len(cols)} columns):\n")
                    for col in sorted(cols):
                        f.write(f"  - {col}\n")
                    f.write("\n")

        print("Column information saved to column_info.txt")

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise