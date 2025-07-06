"""
Benchmark CLI functionality moved from run_benchmark.py.
Contains functions for batch management and configuration.
"""
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from .openai_client import OpenAIClient
from .benchmark import StockMovementEvaluator
from .constants import (
    DEFAULT_GPT_MODEL,
    DEFAULT_GPT_SYSTEM_INSTRUCTION,
    FLARE_EDITED_TEST_PATH,
    BATCH_OUTPUT_DIR,
    BATCH_ID_MATCH
)
from .utils import setup_logging

logger = setup_logging(__name__)


def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for benchmark scenarios."""
    basic_gpt_model = "gpt-4.1-mini-2025-04-14"
    flare_edited_dataset_path = str(FLARE_EDITED_TEST_PATH)
    
    gpt_batch_base_config = {
        "experiment_name": "[FULL] GPT-4.1 mini on Stock Movement",
        "model_type": "gpt",
        "model_path_or_name": basic_gpt_model,
        "system_instruction": DEFAULT_GPT_SYSTEM_INSTRUCTION,
        "dataset_path": flare_edited_dataset_path,
        "dataset_split": "train",
        "use_batch_api": True,
        "batch_check_interval": 15,
        "max_batch_wait_time": 7200,
        "max_tokens": 20,
        "temperature": 0.0
    }
    
    gpt_batch_full_config = gpt_batch_base_config.copy()
    gpt_batch_sample_config = gpt_batch_base_config.copy()
    gpt_batch_sample_config["max_batch_wait_time"] = 600
    
    gpt_batch_flare_original_config = {
        "experiment_name": "GPT 3.5 Turbo on Stock Movement (ACL18)",
        "model_type": "gpt",
        "model_path_or_name": "gpt-3.5-turbo",
        "system_instruction": DEFAULT_GPT_SYSTEM_INSTRUCTION,
        "dataset_path": "TheFinAI/flare-sm-acl",
        "dataset_split": "acl_test",
        "use_batch_api": True,
        "batch_check_interval": 10,
        "max_batch_wait_time": 3600
    }
    
    finma_batch_config = {
        "experiment_name": "FinMA 7B Full on Stock Movement (ACL18)",
        "model_type": "vllm",
        "model_path_or_name": "TheFinAI/finma-7b-full",
        "system_instruction": None,
        "dataset_path": "TheFinAI/flare-sm-acl",
        "dataset_split": "acl_test",
        "tensor_parallel_size": 1,
        "trust_remote_code": True
    }
    
    return {
        "gpt-batch-full": gpt_batch_full_config,
        "gpt-batch-sample": gpt_batch_sample_config,
        "gpt-batch-flare-original": gpt_batch_flare_original_config,
        "finma-batch": finma_batch_config
    }


def check_batch(batch_id: str) -> None:
    """Check batch status and results."""
    if batch_id in BATCH_ID_MATCH.keys():
        batch_id = BATCH_ID_MATCH[batch_id]

    output_jsonl_path = BATCH_OUTPUT_DIR / "jsonl" / f"{batch_id}.jsonl"
    output_csv_path = BATCH_OUTPUT_DIR / "csv" / f"{batch_id}.csv"
    
    try:
        client = OpenAIClient()
        batch_job = client.client.batches.retrieve(batch_id)
    except Exception:
        logger.error('Invalid batch ID')
        return
    
    # Wait for completion
    while True:
        batch_job = client.client.batches.retrieve(batch_id)
        logger.info(f'Current status: {batch_job.status}')
        
        if batch_job.status == 'completed':
            logger.info('Batch API completed')
            break
        elif batch_job.status in ['failed', 'cancelled']:
            logger.error(f'Batch API {batch_job.status}')
            return
        
        time.sleep(15)
    
    # Process results
    if batch_job.output_file_id:
        output_file_content = client.client.files.content(batch_job.output_file_id).read()
        output_file_content = output_file_content.decode('utf-8')
        
        # Save JSONL
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl_path.open('w', encoding='utf-8') as f:
            for line in output_file_content.strip().split('\n'):
                f.write(json.dumps(json.loads(line), ensure_ascii=False) + '\n')
        
        logger.info(f'Output saved to: {output_jsonl_path}')
        
        # Process results for CSV
        evaluator = StockMovementEvaluator()
        results = []
        
        for line in output_file_content.strip().split('\n'):
            data = json.loads(line)
            custom_id = data.get('custom_id')
            response_body = data.get('response', {}).get('body', {})
            
            if response_body and 'choices' in response_body:
                content = response_body['choices'][0]['message']['content']
                gold_label = custom_id.split('-')[-1]
                pred_label = evaluator.process_single_result(content)
                
                results.append({
                    'custom_id': custom_id,
                    'gold_label': gold_label,
                    'pred_label': pred_label,
                    'pred_text': content
                })
        
        # Save CSV
        df = pd.DataFrame(results)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        # Show metrics
        gold_labels = df['gold_label'].tolist()
        pred_labels = df['pred_label'].tolist()
        metrics = evaluator.evaluate_predictions(gold_labels, pred_labels)
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.3f}")


def list_batch_id(list_len: int) -> None:
    """List batch IDs."""
    try:
        client = OpenAIClient()
        
        if list_len:
            batch_list = list(client.client.batches.list(limit=list_len))
        else:
            batch_list = list(client.client.batches.list())
        
        if not batch_list:
            print('-' * 30)
            print('No batch API history found.')
            print('-' * 30)
        else:
            print('-' * 30, '(1 is most recent)')
            for idx, batch in enumerate(batch_list, start=1):
                try:
                    request_counts = batch.request_counts.total
                except Exception:
                    request_counts = '?'
                print(f'{idx}. {batch.id} ({request_counts} requests)')
            print('-' * 30)
    except Exception as e:
        logger.error(f"Error listing batches: {e}")