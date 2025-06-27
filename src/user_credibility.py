"""
User credibility analysis module.
Refactored version of step3_user_credibility_analysis.py
"""
import math
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from scipy.stats import norm

from .constants import (
    OUTPUT_TABLE_USER_STOCK_PATH,
    CACHE_DIR,
    PRICE_RISE_THRESHOLD,
    PRICE_FALL_THRESHOLD,
    WILSON_CONFIDENCE
)
from .utils import setup_logging

logger = setup_logging(__name__)


class UserCredibilityAnalyzer:
    """Analyzes user credibility based on tweet sentiment and stock performance."""
    
    def __init__(self):
        self.rise_threshold = PRICE_RISE_THRESHOLD
        self.fall_threshold = PRICE_FALL_THRESHOLD
        self.wilson_confidence = WILSON_CONFIDENCE
    
    def calculate_price_label(self, row: pd.Series) -> str:
        """
        Determine price movement label based on thresholds.
        
        Args:
            row: DataFrame row with price change data
            
        Returns:
            Price label: 'rise', 'fall', 'both', or 'flat'
        """
        up = row["High Change from Tweet (%)"] >= self.rise_threshold
        down = row["Low Change from Tweet (%)"] <= self.fall_threshold
        
        if up and down:
            return "both"
        elif up:
            return "rise"
        elif down:
            return "fall"
        else:
            return "flat"
    
    def is_prediction_correct(self, sentiment: str, price_label: str) -> int:
        """
        Check if sentiment prediction matches actual price movement.
        
        Args:
            sentiment: Predicted sentiment ('positive', 'negative', 'neutral')
            price_label: Actual price movement ('rise', 'fall', 'both', 'flat')
            
        Returns:
            1 if correct, 0 if incorrect
        """
        if price_label == "both":  # Any direction is correct
            return 1
        
        return int(
            (sentiment == "positive" and price_label == "rise") or
            (sentiment == "negative" and price_label == "fall")
        )
    
    def wilson_lower_bound(self, positives: int, total: int, confidence: float = None) -> float:
        """
        Calculate Wilson score confidence interval lower bound.
        
        Args:
            positives: Number of positive outcomes
            total: Total number of trials
            confidence: Confidence level (defaults to self.wilson_confidence)
            
        Returns:
            Lower bound of Wilson score interval
        """
        if confidence is None:
            confidence = self.wilson_confidence
        
        if total == 0:
            return 0
        
        z = norm.ppf(1 - (1 - confidence) / 2)
        phat = positives / total
        
        denominator = 1 + z * z / total
        center = phat + z * z / (2 * total)
        margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
        
        return (center - margin) / denominator
    
    def calculate_user_statistics(self, user_tweets: pd.DataFrame) -> pd.Series:
        """
        Calculate credibility statistics for a single user.
        
        Args:
            user_tweets: DataFrame of tweets from one user
            
        Returns:
            Series with user statistics
        """
        # Add weight column if it exists
        if "Log Prob" in user_tweets.columns:
            user_tweets["weight"] = np.exp(user_tweets["Log Prob"])
        else:
            user_tweets["weight"] = 1.0
        
        # Calculate metrics
        n = len(user_tweets)
        hits = user_tweets["hit"].sum()
        weighted_hits = (user_tweets["hit"] * user_tweets["weight"]).sum()
        weight_sum = user_tweets["weight"].sum()
        
        # Wilson score
        wilson = self.wilson_lower_bound(hits, n)
        
        # Pearson correlation between sentiment and price change
        sentiment_score = user_tweets["Sentiment"].map({"positive": 1, "negative": -1})
        
        if sentiment_score.nunique() > 1 and "Price Change (%)" in user_tweets.columns:
            correlation = sentiment_score.corr(user_tweets["Price Change (%)"])
        else:
            correlation = np.nan
        
        return pd.Series({
            "tweets": n,
            "acc_plain": user_tweets["hit"].mean(),
            "acc_weighted": weighted_hits / weight_sum if weight_sum else 0,
            "wilson_lower": wilson,
            "pearson_corr": correlation
        })
    
    def analyze_credibility(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform complete user credibility analysis.
        
        Args:
            df: DataFrame with tweet sentiment and stock data
            
        Returns:
            Tuple of (user_credibility_df, overall_sentiment_df, user_sentiment_df)
        """
        # Add price labels
        df["price_label"] = df.apply(self.calculate_price_label, axis=1)
        
        # Filter out neutral sentiment
        df_filtered = df[df["Sentiment"] != "neutral"].copy()
        
        # Calculate hit/miss
        df_filtered["hit"] = df_filtered.apply(
            lambda row: self.is_prediction_correct(row["Sentiment"], row["price_label"]), 
            axis=1
        )
        
        # User-level statistics
        user_stats = df_filtered.groupby("user_id", group_keys=True).apply(
            self.calculate_user_statistics, 
            include_groups=False
        ).reset_index()
        
        # Overall sentiment distribution
        overall_sentiment = (
            df["Sentiment"]
            .value_counts(normalize=True)
            .rename_axis("Sentiment")
            .reset_index(name="Ratio")
        )
        
        # User-level sentiment distribution
        user_sentiment = pd.pivot_table(
            df,
            index='user_id',
            columns='Sentiment',
            aggfunc='size',
            fill_value=0
        )

        # Normalize to get ratios
        user_sentiment = user_sentiment.div(user_sentiment.sum(axis=1), axis=0)

        # Reset index to convert user_id from index to column
        user_sentiment = user_sentiment.reset_index()
        
        return user_stats, overall_sentiment, user_sentiment
    
    def process_and_save(self, input_path: str = None) -> Dict[str, pd.DataFrame]:
        """
        Process user credibility analysis and save results.
        
        Args:
            input_path: Path to input CSV (defaults to OUTPUT_TABLE_USER_STOCK_PATH)
            
        Returns:
            Dictionary with result DataFrames
        """
        if input_path is None:
            input_path = OUTPUT_TABLE_USER_STOCK_PATH
        
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path, parse_dates=["Tweet Date"])
        
        logger.info("Calculating user credibility...")
        user_cred, overall_sentiment, user_sentiment = self.analyze_credibility(df)
        
        # Log top performers
        logger.info("\n=== Top 10 Users by Wilson Score ===")
        top_users = user_cred.sort_values("wilson_lower", ascending=False).head(10)
        print(top_users)
        
        logger.info("\n=== Overall Sentiment Distribution ===")
        print(overall_sentiment)
        
        # Save results
        user_cred.to_csv(CACHE_DIR / "user_credibility.csv", index=False)
        overall_sentiment.to_csv(CACHE_DIR / "overall_sentiment_ratio.csv", index=False)
        user_sentiment.to_csv(CACHE_DIR / "user_sentiment_ratio.csv", index=False)
        
        # Create merged output
        merged = user_sentiment.merge(
            user_cred[["user_id", "acc_plain"]], 
            on="user_id"
        )
        merged.to_csv(CACHE_DIR / "user_sentiment_and_accuracy.csv", index=False)
        
        logger.info("Results saved successfully")
        
        return {
            "user_credibility": user_cred,
            "overall_sentiment": overall_sentiment,
            "user_sentiment": user_sentiment,
            "merged": merged
        }


def main():
    """Main execution function."""
    analyzer = UserCredibilityAnalyzer()
    results = analyzer.process_and_save()
    
    logger.info("\nAnalysis completed successfully.")
    
    return results


if __name__ == "__main__":
    main()