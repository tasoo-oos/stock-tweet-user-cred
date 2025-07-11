"""
Extract sentiment from tweets and merge with stock price data.
Refactored version of step1_extract_sentiment_and_price.py
"""
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from .constants import (
    ACL18_COMBINED_JSONL_PATH,
    OUTPUT_TABLE_STOCK_PATH,
    ACL18_PRICE_DATA_DIR,
    SENTIMENT_LEVELS,
    BUSINESS_DAYS_FORWARD
)
from .data_loader import DataLoader
from .utils import setup_logging, get_business_days_later, parse_custom_id

logger = setup_logging(__name__)


class SentimentPriceExtractor:
    """Extract sentiment and merge with stock price data."""
    
    def __init__(self):
        self.data_loader = DataLoader()
    
    def extract_sentiment_from_response(self, response_data: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """
        Extract sentiment and probabilities from API response.
        
        Returns:
            Tuple of (highest_sentiment, probability_dict)
        """
        highest_sentiment = ''
        outlook_results = {sentiment: 0.0 for sentiment in SENTIMENT_LEVELS}
        
        if not response_data:
            return highest_sentiment, outlook_results
        
        body_data = response_data.get("body", {})
        if not body_data:
            return highest_sentiment, outlook_results
        
        choices = body_data.get("choices", [])
        if not choices or not isinstance(choices, list):
            return highest_sentiment, outlook_results
        
        choice_item = choices[0]
        logprobs_info = choice_item.get("logprobs", {})
        
        if logprobs_info:
            content_logprobs = logprobs_info.get("content", [])
            if content_logprobs and isinstance(content_logprobs, list):
                token_data = content_logprobs[0]
                top_logprobs_list = token_data.get("top_logprobs", [])
                
                if top_logprobs_list:
                    # Extract probabilities for each sentiment
                    for prob in top_logprobs_list:
                        sentiment = prob.get("token")
                        if sentiment in SENTIMENT_LEVELS:
                            outlook_results[sentiment] = np.exp(prob.get("logprob", 0)).item()
                    
                    # Highest probability sentiment
                    highest_prob_token_info = top_logprobs_list[0]
                    highest_sentiment = highest_prob_token_info.get("token", '')
        
        return highest_sentiment, outlook_results
    
    def get_stock_performance(
        self, 
        ticker: str, 
        tweet_date: str
    ) -> Dict[str, Optional[float]]:
        """
        Calculate stock performance metrics for the period after tweet date.
        
        Returns:
            Dictionary with performance metrics
        """
        result = {
            'tweet_close_price': None,
            'two_weeks_close_price': None,
            'price_change_pct': None,
            'period_high': None,
            'period_low': None,
            'high_change_pct': None,
            'low_change_pct': None
        }
        
        # Load stock data
        stock_df = self.data_loader.load_stock_price_data(ticker)
        if stock_df.empty:
            logger.warning(f"No stock data found for {ticker}")
            return result
        
        try:
            tweet_datetime = pd.to_datetime(tweet_date)
            
            # Find tweet day close price
            tweet_day_data = stock_df[stock_df.index >= tweet_datetime]
            if tweet_day_data.empty:
                logger.warning(f"No stock data for {ticker} on or after {tweet_date}")
                return result
            
            tweet_close_price = tweet_day_data.iloc[0]['Close']
            actual_tweet_date = tweet_day_data.index[0]
            
            # Calculate 2-week later date
            two_weeks_later = get_business_days_later(tweet_date, BUSINESS_DAYS_FORWARD)
            two_weeks_datetime = pd.to_datetime(two_weeks_later)
            
            # Find 2-week later close price
            two_weeks_data = stock_df[stock_df.index >= two_weeks_datetime]
            if not two_weeks_data.empty:
                two_weeks_close_price = two_weeks_data.iloc[0]['Close']
                price_change_pct = ((two_weeks_close_price - tweet_close_price) / tweet_close_price) * 100
                
                result['two_weeks_close_price'] = two_weeks_close_price
                result['price_change_pct'] = price_change_pct
            
            # Get period high/low
            period_data = stock_df[
                (stock_df.index >= actual_tweet_date) & 
                (stock_df.index <= two_weeks_datetime)
            ]
            
            if not period_data.empty:
                period_high = period_data['High'].max()
                period_low = period_data['Low'].min()
                
                result['period_high'] = period_high
                result['period_low'] = period_low
                result['high_change_pct'] = ((period_high - tweet_close_price) / tweet_close_price) * 100
                result['low_change_pct'] = ((period_low - tweet_close_price) / tweet_close_price) * 100
            
            result['tweet_close_price'] = tweet_close_price
            
        except Exception as e:
            logger.error(f"Error calculating performance for {ticker}: {e}")
        
        return result
    
    def process_combined_file(self, input_path: str = None) -> pd.DataFrame:
        """
        Process the combined JSONL file and extract sentiment with stock data.
        
        Args:
            input_path: Path to combined.jsonl file (defaults to ACL18_COMBINED_JSONL_PATH)
            
        Returns:
            DataFrame with extracted data
        """
        if input_path is None:
            input_path = ACL18_COMBINED_JSONL_PATH
        
        extracted_data = []
        
        logger.info(f"Processing {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Parse custom ID
                custom_id = data.get("custom_id", "")
                parsed_id = parse_custom_id(custom_id)
                
                ticker = parsed_id['ticker']
                date = parsed_id['date']
                tweet_id = parsed_id['tweet_id']
                
                # Extract sentiment
                response_data = data.get("response", {})
                highest_sentiment, outlook_results = self.extract_sentiment_from_response(response_data)
                
                # Get stock performance
                performance = self.get_stock_performance(ticker, date)
                
                # Combine all data
                row = {
                    "Custom ID": custom_id,
                    "Ticker": ticker,
                    "Tweet Date": date,
                    "Sentiment": highest_sentiment,
                    "Prob-positive": outlook_results.get('positive', 0),
                    "Prob-neutral": outlook_results.get('neutral', 0),
                    "Prob-negative": outlook_results.get('negative', 0),
                    "Tweet Close Price": performance['tweet_close_price'],
                    "Two Weeks Close Price": performance['two_weeks_close_price'],
                    "Price Change (%)": round(performance['price_change_pct'], 2) if performance['price_change_pct'] is not None else None,
                    "Period High": performance['period_high'],
                    "Period Low": performance['period_low'],
                    "High Change from Tweet (%)": round(performance['high_change_pct'], 2) if performance['high_change_pct'] is not None else None,
                    "Low Change from Tweet (%)": round(performance['low_change_pct'], 2) if performance['low_change_pct'] is not None else None
                }
                
                extracted_data.append(row)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e} for line: {line[:100]}...")
            except Exception as e:
                logger.error(f"Unexpected error: {e} for line: {line[:100]}...")
        
        # Create DataFrame
        df = pd.DataFrame(extracted_data)
        
        # Log statistics
        logger.info(f"Total records processed: {len(df)}")
        logger.info(f"Stock price data found for: {df['Tweet Close Price'].notna().sum()} records")
        logger.info(f"Records with complete 2-week data: {df['Two Weeks Close Price'].notna().sum()}")
        
        if df['Price Change (%)'].notna().sum() > 0:
            logger.info(f"Mean 2-week price change: {df['Price Change (%)'].mean():.2f}%")
            logger.info(f"Median 2-week price change: {df['Price Change (%)'].median():.2f}%")
            logger.info(f"Std deviation: {df['Price Change (%)'].std():.2f}%")
        
        return df


def main():
    """Main execution function."""
    extractor = SentimentPriceExtractor()
    
    # Process the combined file
    df = extractor.process_combined_file()
    
    # Save results
    OUTPUT_TABLE_STOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_TABLE_STOCK_PATH, index=False, encoding='utf-8-sig')
    
    logger.info(f"Results saved to {OUTPUT_TABLE_STOCK_PATH}")
    
    # Show sample data
    logger.info("\nSample data:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    main()