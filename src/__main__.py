#!/usr/bin/env python3
"""
Stock Tweet User Credibility Analysis - Main CLI Interface

This module provides a unified command-line interface for all analysis functions.
Usage: python -m src [command] [options]
"""

import argparse
import json
import sys
from pathlib import Path

from .utils import setup_logging
from .constants import (
    DEFAULT_GPT_MODEL,
    OUTPUT_TABLE_USER_STOCK_PATH,
    CACHE_DIR,
    FLATTENED_TWITTER_CSV_PATH,
)
from .extract_sentiment import main as extract_main
from .user_credibility import UserCredibilityAnalyzer
from .sentiment_analyze import SentimentAnalyze
from .benchmark import BenchmarkRunner
from .data_loader import DataLoader
from .openai_client import OpenAIClient
from .benchmark_cli import check_batch, list_batch_id, get_default_configs
from .generate_prompts import GeneratePrompts
from .p_value import PValueCalculator

logger = setup_logging(__name__)



def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Tweet User Credibility Analysis Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src query --input tweets.csv --model gpt-4.1-mini-2025-04-14
  python -m src parse --input raw_tweets.json --output parsed_tweets.csv
  python -m src extract-sentiment --input dataset/tweets.csv
  python -m src user-credibility --analyze
  python -m src generate-prompts --dataset flare-acl --split train --example
  python -m src benchmark --scenario gpt-batch-sample --num-samples 100 --query-type basic_query
  python -m src stats --inputs non_neutral batch_685ce8241c648190bf57f433f69ac8a4
  python -m src tools --find-batch-req batch_685ce8241c648190bf57f433f69ac8a4
        """
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Available Commands",
        help="Choose a command to run"
    )


    # ========== Query Command ==========
    query_parser = subparsers.add_parser(
        "query",
        help="Analyze individual tweets with OpenAI"
    )
    query_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with tweets"
    )
    query_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GPT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_GPT_MODEL})"
    )
    query_parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch API instead of real-time API"
    )
    query_parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens for response (default: 500)"
    )
    query_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)"
    )

    # ========== Parse Command ==========
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse raw Twitter data into flattened structure"
    )
    parse_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file path"
    )
    parse_parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path"
    )
    parse_parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)"
    )

    # ========== Sentiment Analysis Command ==========
    sentiment_parse = subparsers.add_parser(
        "sentiment-analyze",
        help="Analyze sentiment of tweets using OpenAI Batch API"
    )
    sentiment_parse.add_argument(
        "dataset-choice",
        type=str,
        choices=['acl18', 'kaggle1'],
        default='kaggle1',
        help="Dataset choice for sentiment analysis (choices: [acl18, kaggle1])"
    )
    sentiment_parse.add_argument(
        "dataset-path",
        type=str,
        default=FLATTENED_TWITTER_CSV_PATH,
        help="Path to the dataset containing tweets"
    )
    sentiment_parse.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GPT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_GPT_MODEL})"
    )

    # ========== Extract Sentiment Command ==========
    extract_parser = subparsers.add_parser(
        "extract-sentiment",
        help="Extract sentiment from tweets and merge with stock price data"
    )
    extract_parser.add_argument(
        "--input", 
        type=str,
        help="Input CSV file path (optional, uses default dataset if not provided)"
    )
    extract_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional, uses default naming if not provided)"
    )
    
    # ========== User Credibility Command ==========
    cred_parser = subparsers.add_parser(
        "user-credibility",
        help="Analyze user credibility based on tweet sentiment and stock performance"
    )
    cred_parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUT_TABLE_USER_STOCK_PATH),
        help=f"Input CSV file (default: {OUTPUT_TABLE_USER_STOCK_PATH})"
    )
    cred_parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run full credibility analysis"
    )
    cred_parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CACHE_DIR),
        help=f"Output file path for credibility results (default: {CACHE_DIR})"
    )


    # ========== Generate Prompts Command =========
    prompts_parse = subparsers.add_parser(
        "generate-prompts",
        help="Generate prompts for benchmark experiments"
    )
    prompts_parse.add_argument(
        "--dataset",
        type=str,
        choices=["flare-acl", "kaggle-1"],
        default="flare-acl",
        help="Configuration"
    )
    prompts_parse.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "test", "valid"],
        default="all",
        help="Configuration"
    )
    prompts_parse.add_argument(
        "--example",
        action="store_true",
        help="Generate example prompts for the dataset"
    )
    prompts_parse.add_argument(
        "--num-examples",
        type=int,
        default=1,
        help="Number of example prompts to generate (default: 1)"
    )
    prompts_parse.add_argument(
        "--example-type",
        type=str,
        choices=["from-code", "from-file"],
        default="from-code",
        help="Type of example to generate (choices: [from-code, from-file]) (default: from-code)"
    )
    
    # ========== Benchmark Command =========
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run stock movement prediction benchmarks"
    )
    bench_parser.add_argument(
        "--scenario",
        type=str,
        choices=["gpt-batch-full", "gpt-batch-sample", "gpt-batch-flare-original", "finma-batch"],
        help="Predefined scenario to run"
    )
    bench_parser.add_argument(
        "--config",
        type=str,
        help="JSON configuration file path"
    )
    bench_parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to process (0 for all)"
    )
    bench_parser.add_argument(
        "--query-type",
        type=str,
        default="basic_query",
        help="Query type for flare_edited dataset"
    )
    bench_parser.add_argument(
        "--check-batch",
        nargs="?",
        const="recent",
        type=str,
        help="Check status of existing batch ID"
    )
    bench_parser.add_argument(
        "--list-batches",
        type=int,
        nargs="?",
        const=10,
        help="List recent batch IDs (default: 10, use 0 for all)"
    )

    
    # ========== Statistics Command =========
    stats_parser = subparsers.add_parser(
        "stats",
        help="Run statistical analysis (p-value calculations)"
    )
    # 복수 개의 input 받기
    stats_parser.add_argument(
        "--inputs",
        type=str,
        nargs='+',
        help="Input batch_id or nickname of batch for statistical comparison. If not provided, 모두에 대해 실행. (e.g., --inputs non_neutral batch_685ce8241c648190bf57f433f69ac8a4)"
    )

    stats_parser.add_argument(
        "--test",
        type=str,
        choices=["mcnemar", "exact"],
        default="mcnemar",
        help="Statistical test to perform (default: mcnemar)"
    )


    # ========== Tools Command =========
    tools_parse = subparsers.add_parser(
        "tools",
        help="Utility tools for batch management and analysis"
    )
    tools_parse.add_argument(
        "--find-batch-req",
        type=str,
        help="Find batch ID for a specific requirement (e.g., batch_685ce8241c648190bf57f433f69ac8a4)"
    )
    tools_parse.add_argument(
        "--num",
        type=int,
        default=0,
        help= "Generic numeric argument for various tools (default: 0)"
    )
    tools_parse.add_argument(
        '--price',
        type=str,
        help="Calculate total price for a specific batch ID (e.g., batch_685ce8241c648190bf57f433f69ac8a4)"
    )


    # ========== Process command-line arguments ==========
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "query":
            client = OpenAIClient(model=args.model)
            loader = DataLoader()

            logger.info(f"Loading tweets from {args.input}")
            df = loader.load_csv(args.input)

            if args.batch:
                logger.info("Using batch API")
                # TODO: Implement batch processing
                logger.warning("Batch API functionality needs specific implementation")
            else:
                logger.info("Using real-time API")
                # TODO: Implement single tweet processing
                logger.warning("Single tweet processing needs specific implementation")

        elif args.command == "parse":
            loader = DataLoader()

            # Determine output path
            if args.output:
                output_path = args.output
            else:
                input_path = Path(args.input)
                output_path = str(input_path.with_suffix(f".{args.format}"))

            # Parse the data
            logger.info(f"Parsing {args.input} to {output_path}")
            result = loader.parse_twitter_data(args.input, output_path, args.format)
            logger.info("Parse completed successfully")

        elif args.command == "sentiment-analyze":
            logger.info(f"Analyzing sentiment for dataset at {args.dataset_path}")
            analyzer = SentimentAnalyze(args.datset_choice, args.dataset_path, args.model)

            logger.info("Sentiment analysis completed successfully")

        elif args.command == "extract-sentiment":
            result = extract_main()
            logger.info("Sentiment extraction completed successfully")
            
        elif args.command == "user-credibility":
            analyzer = UserCredibilityAnalyzer()
            results = analyzer.process_and_save(args.input)
            logger.info("User credibility analysis completed successfully")

        elif args.command == "generate-prompts":
            logger.info(f"dataset: {args.dataset} / split: {args.split}")
            generator = GeneratePrompts(args.dataset, args.split)
            if args.example:
                generator.show_example_queries(args.example_type, args.num_examples)
            else:
                generator.process_and_save()
            logger.info("Completed successfully")
            
        elif args.command == "benchmark":
            if args.check_batch:
                check_batch(args.check_batch)
                
            elif args.list_batches is not None:
                list_batch_id(args.list_batches)

            else:
                if args.scenario:
                    config = get_default_configs()[args.scenario]
                elif args.config:
                    with open(args.config, 'r') as f:
                        config = json.load(f)
                else:
                    if args.num_samples > 0:
                        config = get_default_configs()["gpt-batch-sample"]
                    else:
                        config = get_default_configs()["gpt-batch-full"]
                
                runner = BenchmarkRunner(config)
                results = runner.run_evaluation(args.num_samples, args.query_type)
                logger.info("Benchmark completed successfully")

                
        elif args.command == "stats":
            logger.info(f"Running {args.test} test on inputs: {args.inputs}")
            calculator = PValueCalculator()
            calculator.show_pvalue(
                batches=args.inputs,
                test_func=args.test
            )
            logger.info("Statistical analysis completed successfully")

        elif args.command == "tools":
            if args.find_batch_req:
                batch_id = args.find_batch_req

                openai_client = OpenAIClient()
                openai_client.find_batch_req_file(batch_id, args.num)

                logger.info(f"Batch request file for {batch_id} found successfully")
            if args.price:
                batch_id = args.price

                openai_client = OpenAIClient()
                total_price = openai_client.calculate_price(batch_id)

                logger.info(f"Total price for batch {batch_id}: ${total_price:.4f}")

        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error executing {args.command}: {e}")
        raise
        sys.exit(1)


if __name__ == "__main__":
    main()