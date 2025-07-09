"""
Run stock movement prediction benchmarks.
Refactored version of run_benchmark.py
"""
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

from .constants import (
    DEFAULT_GPT_MODEL,
    DEFAULT_GPT_SYSTEM_INSTRUCTION,
    FLARE_EDITED_TEST_PATH,
    BATCH_OUTPUT_DIR,
    STOCK_MOVEMENT_LABELS,
    STOCK_MOVEMENT_CHOICE_MAPPING,
    STOCK_MOVEMENT_DEFAULT
)
from .data_loader import DataLoader
from .openai_client import OpenAIClient
from .utils import setup_logging

logger = setup_logging(__name__)


class StockMovementEvaluator:
    """Evaluates stock movement predictions."""
    
    def __init__(self):
        self.choice_mapping = STOCK_MOVEMENT_CHOICE_MAPPING
        self.default = STOCK_MOVEMENT_DEFAULT
        self.gold_label_classes = STOCK_MOVEMENT_LABELS

    def create_prompt(self, doc: Dict[str, Any], query_column) -> str:
        """Create prompt from document."""
        if query_column not in doc:
            raise ValueError(f"Query column '{query_column}' not found in document. !!!!!!!!!")
        return doc.get(query_column)

    def process_single_result(self, generated_text: str) -> str:
        """Convert model output to normalized label."""

        try:
            json_data = json.loads(generated_text.strip())
        except json.JSONDecodeError:
            text = generated_text.lower().strip()
        else:
            for key in json_data:
                if "answer" in key.lower():
                    text = json_data[key].lower().strip()
                    break

        if "rise" in text or any(val in text for val in self.choice_mapping["rise"]):
            return "rise"
        if "fall" in text or any(val in text for val in self.choice_mapping["fall"]):
            return "fall"

        return self.default

    def get_gold_label(self, doc: Dict[str, Any]) -> str:
        """Extract gold label from document."""
        return doc["answer"].lower()

    def evaluate_predictions(
        self,
        gold_labels: List[str],
        predicted_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate evaluation metrics.

        Returns:
            Dictionary with various metrics
        """

        # Create DataFrame for analysis
        df = pd.DataFrame({
            'gold_label': gold_labels,
            'pred_label': predicted_labels
        })

        # Count errors
        error_count, error_count_when_rise, error_count_when_fall = 0, 0, 0
        for pred_label, gold_label in zip(predicted_labels, gold_labels):
            if pred_label not in self.gold_label_classes:
                error_count += 1
                if gold_label == 'fall':
                    error_count_when_fall += 1
                elif gold_label == 'rise':
                    error_count_when_rise += 1

        # Calculate metrics
        metrics = {
            'total_samples': len(gold_labels),
            'error_count': error_count,
            'error_count_when_rise': error_count_when_rise,
            'error_count_when_fall': error_count_when_fall,
            'error_rate': error_count / len(gold_labels) if gold_labels else 0
        }

        # Metrics excluding errors
        valid_mask = df['pred_label'].isin(self.gold_label_classes)
        if valid_mask.sum() > 0:
            valid_gold = df.loc[valid_mask, 'gold_label']
            valid_pred = df.loc[valid_mask, 'pred_label']

            metrics['accuracy'] = accuracy_score(valid_gold, valid_pred)
            metrics['f1_macro'] = f1_score(valid_gold, valid_pred, average='macro')
            metrics['mcc'] = matthews_corrcoef(valid_gold, valid_pred)

        # Confusion matrix
        cm_labels = self.gold_label_classes
        if error_count > 0:
            cm_labels = self.gold_label_classes + [self.default]
        
        metrics['confusion_matrix'] = confusion_matrix(
            df['gold_label'], 
            df['pred_label'], 
            labels=cm_labels
        )
        
        # Classification report
        metrics['classification_report'] = classification_report(
            df['gold_label'], 
            df['pred_label'],
            zero_division=0,
            output_dict=True
        )
        
        return metrics


class BenchmarkRunner:
    """Runs benchmarks for stock movement prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = DataLoader()
        self.evaluator = StockMovementEvaluator()
        self.batch_id = None
        
        # Initialize model based on type
        model_type = config.get('model_type', 'gpt')
        
        if model_type == 'gpt':
            self.model = OpenAIClient(
                model=config.get('model_path_or_name', DEFAULT_GPT_MODEL)
            )
            self.use_batch = config.get('use_batch_api', True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset for evaluation."""
        dataset_path = self.config.get('dataset_path', str(FLARE_EDITED_TEST_PATH))
        dataset_split = self.config.get('dataset_split', 'acl_test')
        
        # Check if local file
        if Path(dataset_path).exists():
            logger.info(f"Loading local file: {dataset_path}")
            if dataset_path.endswith('.parquet'):
                return pd.read_parquet(dataset_path)
            else:
                return pd.read_csv(dataset_path)
        else:
            # Load from Hugging Face
            return self.data_loader.load_flare_dataset(dataset_split)

    def sample_dataset(self, df: pd.DataFrame, num_samples: int) -> pd.DataFrame: # rise와 fall 비율을 똑같게 샘플링
        """
        Sample the dataset to balance rise and fall labels.

        Args:
            df: The full dataset
            num_samples: Number of samples to return

        Returns:
            Sampled DataFrame with balanced rise and fall labels
        """
        split_size = (num_samples+1)//2

        min_size = df.groupby('gold').size().min()
        if min_size < split_size:
            split_size = min_size

        sampled_df = df.groupby('gold').sample(n=split_size, random_state=42).reset_index(drop=True)
        if num_samples % 2 == 1:
            sampled_df.drop(len(sampled_df) - 1, inplace=True, axis=0)  # Drop last row if odd

        return sampled_df
    
    def run_evaluation(
        self, 
        num_samples: int = 0, 
        query_type: str = "basic_query"
    ) -> Dict[str, Any]:
        """
        Run the evaluation.
        
        Args:
            num_samples: Number of samples to evaluate (0 for all)
            query_type: Column name for query type
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation: {self.config.get('experiment_name', 'Untitled')}")
        logger.info(f"Model: {self.config.get('model_path_or_name')}")
        logger.info(f"Query type: {query_type}")
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Sample if requested
        if num_samples > 0:
            dataset = self.sample_dataset(dataset, num_samples)
        
        logger.info(f"Evaluating {len(dataset)} samples")
        
        # Prepare data
        prompts = [self.evaluator.create_prompt(row, query_type) for _, row in dataset.iterrows()]
        gold_labels = [self.evaluator.get_gold_label(row) for _, row in dataset.iterrows()]
        
        # Generate predictions
        logger.info("Generating predictions...")
        
        if hasattr(self, 'model') and isinstance(self.model, OpenAIClient):
            # Create custom IDs for batch tracking
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_ids = []
            for i, (_, row) in enumerate(dataset.iterrows()):
                custom_id = f"request-{timestamp}-{i}-{row.get('id', i)}-{gold_labels[i]}"
                custom_ids.append(custom_id)
            
            batch_id, generated_texts, original_results = self.model.generate_with_retry(
                prompts,
                system_instruction=self.config.get('system_instruction', DEFAULT_GPT_SYSTEM_INSTRUCTION),
                use_batch=self.use_batch,
                custom_ids=custom_ids,
                temperature=self.config.get('temperature', 0.0),
                max_tokens=self.config.get('max_tokens', 20)
            )
            self.batch_id = batch_id  # Save batch ID for later use
        else:
            raise NotImplementedError("Only GPT models are currently supported")
        
        # Process predictions
        predicted_labels = [self.evaluator.process_single_result(text) for text in generated_texts]
        resonings = [result.get('reasoning', '') for result in original_results] if original_results else None
        
        # Evaluate
        logger.info("Calculating metrics...")
        metrics = self.evaluator.evaluate_predictions(gold_labels, predicted_labels)
        
        # Log results
        self._log_results(metrics)
        
        # Save detailed results if batch ID provided
        if self.batch_id:
            self._save_batch_results_to_csv(
                self.batch_id,
                custom_ids,
                gold_labels,
                predicted_labels,
                generated_texts,
                resonings=resonings
            )
        else:
            logger.warning("No batch ID available, detailed results will not be saved.")
        
        return metrics
    
    def _log_results(self, metrics: Dict[str, Any]):
        """Log evaluation results."""
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        
        logger.info(f"Total samples: {metrics['total_samples']}")
        logger.info(f"Error count: {metrics['error_count']} ({metrics['error_rate']:.1%})")
        
        if 'accuracy' in metrics:
            logger.info(f"Accuracy (excluding errors): {metrics['accuracy']:.3f}")
            logger.info(f"F1 Macro: {metrics['f1_macro']:.3f}")
            logger.info(f"MCC: {metrics['mcc']:.3f}")
        
        # Confusion matrix
        logger.info("Confusion Matrix:")
        cm = metrics['confusion_matrix']
        cm_df = pd.DataFrame(cm)
        
        labels = self.evaluator.gold_label_classes
        if metrics['error_count'] > 0:
            labels = labels + [self.evaluator.default]
        
        cm_df.index = [f'true_{label}' for label in labels[:len(cm_df)]]
        cm_df.columns = [f'pred_{label}' for label in labels[:len(cm_df.columns)]]

        # Counfusion Matrix에 error column 추가
        if metrics['error_count'] > 0:
            pred_error = []
            for label in self.evaluator.gold_label_classes: # 순서 일치시키기 위함
                pred_error.append(metrics['error_count_when_'+label])
            cm_df['pred_error'] = pred_error

        # Sum rows and columns 추가
        cm_df['| sum'] = cm_df.sum(axis=1)
        cm_df.loc['-- sum --'] = cm_df.sum(axis=0)

        print('[Confusion Matrix]')
        print(cm_df)
        
        # Classification report
        logger.info("Classification Report:")
        report = metrics['classification_report']
        for label, scores in report.items():
            if isinstance(scores, dict):
                logger.info(f"{label}: precision={scores.get('precision', 0):.3f}, "
                          f"recall={scores.get('recall', 0):.3f}, "
                          f"f1={scores.get('f1-score', 0):.3f}")
    
    def _save_batch_results_to_csv(
        self,
        batch_id: str,
        custom_ids: List[str],
        gold_labels: List[str],
        predicted_labels: List[str],
        generated_texts: List[str],
        resonings: Optional[List[str]] = None
    ):
        """Save detailed batch results."""
        output_csv_dir = BATCH_OUTPUT_DIR / "csv"
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        
        output_csv_path = output_csv_dir / f"{batch_id}.csv"
        
        df = pd.DataFrame({
            'custom_id': custom_ids,
            'gold_label': gold_labels,
            'pred_label': predicted_labels,
            'pred_text': generated_texts
        })
        if resonings:
            df['reasoning'] = resonings
        
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved batch results to {output_csv_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run stock movement prediction benchmark")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to evaluate (0 for all)"
    )
    parser.add_argument(
        "--query-type",
        type=str,
        default="basic_query",
        help="Query column to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GPT_MODEL,
        help="Model to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(FLARE_EDITED_TEST_PATH),
        help="Dataset path"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "experiment_name": "Stock Movement Prediction Benchmark",
        "model_type": "gpt",
        "model_path_or_name": args.model,
        "dataset_path": args.dataset,
        "dataset_split": "train",
        "use_batch_api": True,
        "system_instruction": DEFAULT_GPT_SYSTEM_INSTRUCTION
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_evaluation(
        num_samples=args.num_samples,
        query_type=args.query_type
    )
    
    logger.info("Benchmark completed successfully")
    
    return results


if __name__ == "__main__":
    main()