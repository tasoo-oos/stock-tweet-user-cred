import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from openai import OpenAI
import time
from tqdm import tqdm
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TweetSentimentAnalyzer:
    """Analyzes tweet sentiment using GPT-4.1 API"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.sentiment_levels = [
            "strongly negative",
            "weakly negative",
            "neutral",
            "weakly positive",
            "strongly positive"
        ]

    def analyze_sentiment(self, tweet_text: str, stock_ticker: str, company_name: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment of a single tweet about a stock
        Returns sentiment level and reasoning
        """

        # Build the prompt with context about stock analysis
        prompt = \
f"""
You are a financial sentiment analyst. Your job is to classify the sentiment of tweets about stocks. Think of the timeframe as the next 14 trading days after the post.

Classify the sentiment as one of these exact phrases:
- strongly negative: Very bearish, predicting significant decline
- weakly negative: Slightly bearish, some concerns
- neutral: No clear positive or negative stance
- weakly positive: Slightly bullish, some optimism  
- strongly positive: Very bullish, predicting significant gains

Consider:
1. Direct statements about the stock's future
2. Emotional tone and language intensity
3. Specific predictions or targets mentioned
4. Overall market sentiment conveyed
5. Any relevant context about the company or sector

Respond in JSON format:
{{
    "sentiment": "one of the five sentiment levels",
    "confidence": 0.0-1.0
}}

Now, analyze the tweet about {stock_ticker} {f'({company_name})' if company_name else ''} and classify its outlook on the stock.

Tweet: 
```txt
{tweet_text}
```"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise financial sentiment analyst. Always respond only in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Lower temperature for more consistent classification
                max_tokens=200,
                seed=42,  # For reproducibility
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            # Parse the response
            result = json.loads(response.choices[0].message.content)

            # Validate sentiment is one of our categories
            if result.get('sentiment') not in self.sentiment_levels:
                logger.warning(f"Invalid sentiment returned: {result.get('sentiment')}")
                result['sentiment'] = 'neutral'

            return result

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reasoning": f"Error in analysis: {str(e)}"
            }

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
        self.sentiment_analyzer = TweetSentimentAnalyzer(api_key=openai_api_key)
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
            "performance": performance_result,
            "sentiment_correct": self.evaluate_prediction(
                sentiment_result.get('sentiment', 'neutral'),
                performance_result.get('simple_return_pct', 0)
            )
        }

    def evaluate_prediction(self, sentiment: str, return_pct: float) -> Dict[str, Any]:
        """
        Evaluate if sentiment prediction was correct based on actual returns
        """

        # Define thresholds for different sentiment levels
        thresholds = {
            "strongly negative": -5.0,  # Expecting > 5% decline
            "weakly negative": -2.0,    # Expecting 2-5% decline
            "neutral": (-2.0, 2.0),     # Expecting -2% to +2%
            "weakly positive": 2.0,      # Expecting 2-5% gain
            "strongly positive": 5.0     # Expecting > 5% gain
        }

        # Check if prediction was correct
        if sentiment == "neutral":
            correct = thresholds["neutral"][0] <= return_pct <= thresholds["neutral"][1]
        elif sentiment == "strongly negative":
            correct = return_pct <= thresholds["strongly negative"]
        elif sentiment == "weakly negative":
            correct = thresholds["strongly negative"] < return_pct <= thresholds["weakly negative"]
        elif sentiment == "weakly positive":
            correct = thresholds["weakly positive"] <= return_pct < thresholds["strongly positive"]
        elif sentiment == "strongly positive":
            correct = return_pct >= thresholds["strongly positive"]
        else:
            correct = False

        return {
            "correct": correct,
            "expected_direction": "negative" if "negative" in sentiment else ("positive" if "positive" in sentiment else "neutral"),
            "actual_direction": "negative" if return_pct < -2 else ("positive" if return_pct > 2 else "neutral"),
            "magnitude_match": self.check_magnitude_match(sentiment, return_pct)
        }

    def check_magnitude_match(self, sentiment: str, return_pct: float) -> str:
        """Check if the magnitude of the move matched the sentiment strength"""

        abs_return = abs(return_pct)

        if "strongly" in sentiment and abs_return >= 5:
            return "matched"
        elif "weakly" in sentiment and 2 <= abs_return < 5:
            return "matched"
        elif sentiment == "neutral" and abs_return < 2:
            return "matched"
        elif abs_return < 2:
            return "too_small"
        else:
            return "too_large"

def main():
    """Main execution function"""

    # Load tweet data
    logger.info("Loading tweet data...")
    df = pd.read_pickle('flattened_twitter_data.pkl')

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

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        result = analyzer.analyze_tweet_performance(row)
        results.append(result)

        # Rate limiting for API calls
        time.sleep(0.5)  # Adjust based on your API limits

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_json('tweet_sentiment_analysis_results.json', orient='records', indent=2)
    results_df.to_csv('tweet_sentiment_analysis_results.csv', index=False)

    # Calculate summary statistics
    valid_results = results_df[~results_df['performance'].apply(lambda x: 'error' in x)]

    if len(valid_results) > 0:
        accuracy = valid_results['sentiment_correct'].apply(lambda x: x['correct']).mean()

        logger.info(f"\n=== Analysis Summary ===")
        logger.info(f"Total tweets analyzed: {len(results_df)}")
        logger.info(f"Valid results: {len(valid_results)}")
        logger.info(f"Overall accuracy: {accuracy:.2%}")

        # Accuracy by sentiment type
        for sentiment in ["strongly negative", "weakly negative", "neutral", "weakly positive", "strongly positive"]:
            sentiment_results = valid_results[
                valid_results['sentiment'].apply(lambda x: x.get('sentiment') == sentiment)
            ]
            if len(sentiment_results) > 0:
                sent_accuracy = sentiment_results['sentiment_correct'].apply(lambda x: x['correct']).mean()
                logger.info(f"{sentiment}: {sent_accuracy:.2%} ({len(sentiment_results)} tweets)")

    return results_df

if __name__ == "__main__":
    # Run the analysis
    results = main()