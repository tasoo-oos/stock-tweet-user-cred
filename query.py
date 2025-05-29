import json
from ossaudiodev import error

import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Dict, Optional
import numpy as np
from openai import OpenAI
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
import time
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize Rich console for logging
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class TweetSentimentAnalyzer:
    """Analyzes tweet sentiment using OpenAI's model"""

    def __init__(self, openai_api_key: str = None):
        self.client = OpenAI(api_key=openai_api_key)
        self.sentiment_levels = [
            "negative",
            "neutral",
            "positive"
        ]

    def analyze_sentiment(self, tweet_text: str, stock_ticker: str, company_name: str = None) \
            -> Optional[Dict[str, float]]:
        """
        Analyze sentiment of a single tweet about a stock
        Returns sentiment level and reasoning
        """

        system_prompt = \
f"""You are a precise financial sentiment analyst. Your job is to classify the sentiment of tweets about stocks. Think of the timeframe as the next 2 weeks after the post.

Classify the sentiment as one of these exact phrases:
- negative: Bearish sentiment about the stock, implying a decline in price in the near future.
- neutral: No clear sentiment, or mixed signals about the stock's future.
- positive: Bullish sentiment about the stock, implying an increase in price in the near future.
    
Consider:
1. Direct statements about the stock's future
2. Emotional tone and language intensity
3. Specific predictions or targets mentioned
4. Overall market sentiment conveyed
5. Any relevant context about the company or sector

Respond in one word, nothing else.
"""

        # Build the prompt with context about stock analysis
        user_prompt = \
f"""Target stock: {stock_ticker} {f'({company_name})' if company_name else ''}

Tweet:
```txt
{tweet_text}
```"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=1,
                max_tokens=1,
                top_p=1.0,
                seed=42,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logprobs=True,
                top_logprobs=5
            )

            outlook_results = {}

            for sentiment in self.sentiment_levels:
                outlook_results[sentiment] = 0.0

            for prob in response.choices[0].logprobs.content[0].top_logprobs:
                if prob.token in self.sentiment_levels:
                    outlook_results[prob.token] = np.exp(prob.logprob).item()

            return outlook_results

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None

class StockPerformanceAnalyzer:
    """Analyzes stock performance after tweet dates"""

    def __init__(self, price_data_path: str = "./dataset/price/raw"):
        self.price_data_path = Path(price_data_path)
        self.price_data_cache = {}

    def load_stock_data(self, ticker: str) -> pd.DataFrame:
        """Load and cache stock price data for a given ticker"""

        if ticker in self.price_data_cache:
            return self.price_data_cache[ticker]

        try:
            # Try with and without $ prefix
            file_path = self.price_data_path / f"{ticker.replace('$', '')}.csv"
            if not file_path.exists():
                file_path = self.price_data_path / f"{ticker}.csv"

            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)

            self.price_data_cache[ticker] = df
            return df

        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_performance(self, ticker: str, start_date: datetime, days_forward: int = 14) -> Dict[str, Any]:
        """
        Calculate stock performance for specified period after start_date
        Returns various performance metrics
        """
        df = self.load_stock_data(ticker)
        if df.empty:
            return {"error": "No price data available"}

        try:
            # Normalize start_date to just the date part for comparison
            # This handles timezone-aware timestamps and time components
            start_date_normalized = pd.Timestamp(start_date.date())

            # Find the first available date on or after start_date
            available_dates = df.index[df.index >= start_date_normalized]

            if available_dates.empty:
                return {"error": f"No data available on or after {start_date.date()}"}

            closest_date = available_dates.min()

            # Find end date (approximately days_forward trading days later)
            end_date = closest_date + timedelta(days=days_forward)  # Add buffer for weekends
            future_data = df.loc[closest_date:end_date].iloc[:days_forward+1]

            if len(future_data) < 2:
                return {"error": "Insufficient future data"}

            # Calculate various metrics
            start_price = future_data['Close'].iloc[0]
            end_price = future_data['Close'].iloc[-1]

            # Basic return calculation
            simple_return = (end_price - start_price) / start_price

            # Calculate volatility during period
            daily_returns = future_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized

            # Find max and min during period
            period_high = future_data['High'].max()
            period_low = future_data['Low'].min()
            max_gain = (period_high - start_price) / start_price
            max_loss = (period_low - start_price) / start_price

            return {
                "start_date": closest_date.strftime("%Y-%m-%d"),
                "end_date": future_data.index[-1].strftime("%Y-%m-%d"),
                "trading_days": len(future_data) - 1,
                "start_price": start_price,
                "end_price": end_price,
                "simple_return": simple_return,
                "simple_return_pct": simple_return * 100,
                "volatility_annualized": volatility,
                "max_gain_pct": max_gain * 100,
                "max_loss_pct": max_loss * 100,
                "price_range": period_high - period_low
            }

        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {"error": str(e)}

class TweetStockAnalyzer:
    """Main analyzer combining sentiment and stock performance"""

    def __init__(self, openai_api_key: str = None):
        self.sentiment_analyzer = TweetSentimentAnalyzer(openai_api_key=openai_api_key)
        self.performance_analyzer = StockPerformanceAnalyzer()
        self.stock_mapping = self.load_stock_mapping()

    def load_stock_mapping(self) -> Dict[str, str]:
        """Load ticker to company name mapping"""
        mapping = {}
        try:
            with open('./dataset/StockTable', 'r') as f:
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

    def analyze_tweet_performance(self, tweet_data: pd.Series) -> Dict[str, Any]:
        """
        Analyze a single tweet's sentiment and corresponding stock performance
        """

        # Extract necessary data
        tweet_text = tweet_data.get('text', '')
        if not tweet_text or pd.isna(tweet_text):
            return {"error": "No tweet text available"}

        ticker = tweet_data.get('stock_ticker', '')
        tweet_date = pd.to_datetime(tweet_data.get('created_at'))

        if pd.isna(tweet_date):
            return {"error": "Invalid tweet date"}

        # Get company name if available
        company_name = self.stock_mapping.get(ticker, None)

        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(
            tweet_text, ticker, company_name
        )

        # Calculate stock performance
        performance_result = self.performance_analyzer.calculate_performance(
            ticker, tweet_date, days_forward=14
        )

        # Combine results
        return {
            "tweet_id": tweet_data.get('id_str', ''),
            "tweet_date": tweet_date.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "company": company_name,
            "tweet_text": tweet_text[:200] + "..." if len(tweet_text) > 200 else tweet_text,
            "sentiment": sentiment_result,
            "performance": performance_result
        }


def main():
    """Main execution function"""

    save_path = Path(os.getcwd()) / "cache"

    # Load tweet data
    logger.info("Loading tweet data...")
    df = pd.read_pickle(save_path / 'flattened_twitter_data.pkl')

    # Get Environment Variables
    openai_api_key = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

    # Filter for tweets with text
    df_with_text = df[df['text'].notna()].copy()
    logger.info(f"Found {len(df_with_text)} tweets with text out of {len(df)} total")

    # Initialize analyzer (you'll need to set your OpenAI API key)
    # You can set it as an environment variable: export OPENAI_API_KEY='your-key'
    analyzer = TweetStockAnalyzer(openai_api_key=openai_api_key)

    # Sample analysis (full dataset might be expensive with GPT-4.1)
    # You can adjust sample_size or implement batching
    sample_size = 10  # Start with small sample
    sample_df = df_with_text.sample(n=min(sample_size, len(df_with_text)), random_state=42)

    results = []

    logger.info(f"Analyzing {len(sample_df)} tweets...")

    with Progress(
            TextColumn("[bold blue]Analyzing tweets..."),
            BarColumn(),
            TaskProgressColumn(),
            console=console  # Important: use the same console instance
    ) as progress:

        task = progress.add_task("Analyzing tweets...", total=len(sample_df))

        for idx, row in sample_df.iterrows():
            result = analyzer.analyze_tweet_performance(row)
            results.append(result)
            progress.advance(task, advance=1)

    json_path = save_path / 'tweet_sentiment_analysis_results.json'
    with open(json_path, 'wt', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Convert results to DataFrame for analysis
    flatten_results = []
    error_results = []
    error_cnt = 0
    for res in results:
        if 'error' not in res:
            flatten_result = {}
            for key in res:
                if type(res[key]) is dict:
                    for sub_key, value in res[key].items():
                        flatten_result[f"{key}_{sub_key}"] = value
                else:
                    flatten_result[key] = res[key]
            flatten_results.append(flatten_result)
        else:
            error_cnt += 1
            error_results.append({
                "tweet_id": res.get('tweet_id', ''),
                "error": res['error']
            })

    results_df = pd.DataFrame(flatten_results)

    # Save results
    results_df.to_csv(save_path / 'tweet_sentiment_analysis_results.csv', index=False)

    return results_df

if __name__ == "__main__":
    # Run the analysis
    results = main()