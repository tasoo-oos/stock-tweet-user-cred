"""
OpenAI client wrapper for the Stock Tweet User Credibility project.
"""
import os
import time
import json
import logging
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime

import openai
import numpy as np
from tqdm import tqdm

from .constants import (
    DEFAULT_GPT_MODEL,
    BATCH_CHECK_INTERVAL,
    MAX_BATCH_WAIT_TIME,
    BATCH_API_MAX_RETRIES,
    BATCH_API_SLEEP_TIME
)
from .utils import setup_logging

logger = setup_logging(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API interactions with batch support."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        model: str = DEFAULT_GPT_MODEL
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            organization: OpenAI organization ID (defaults to environment variable)
            model: Model to use for completions
        """
        # Set up API key
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set."
            )
        
        # Set up organization if provided
        if organization:
            openai.organization = organization
        elif "OPENAI_ORG_ID" in os.environ:
            openai.organization = os.environ["OPENAI_ORG_ID"]
        
        self.client = openai.OpenAI()
        self.model = model
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def generate_single(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 20,
        **kwargs
    ) -> str:
        """
        Generate a single completion.
        
        Args:
            prompt: User prompt
            system_instruction: System instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated text
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in single generation: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        custom_ids: List[str],
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 20,
        batch_check_interval: int = BATCH_CHECK_INTERVAL,
        max_wait_time: int = MAX_BATCH_WAIT_TIME,
        **kwargs
    ) -> List[str]:
        """
        Generate completions using the Batch API.
        
        Args:
            prompts: List of user prompts
            custom_ids: List of custom IDs for tracking
            system_instruction: System instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            batch_check_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait
            **kwargs: Additional parameters for the API
            
        Returns:
            List of generated texts in the same order as prompts
        """
        logger.info(f"Starting batch generation for {len(prompts)} prompts")
        
        # Create batch requests
        batch_requests = []
        for i, (prompt, custom_id) in enumerate(zip(prompts, custom_ids)):
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
            }
            batch_requests.append(request)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', encoding='utf-8', delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
            batch_file_path = f.name
        
        try:
            # Upload file
            logger.info("Uploading batch file...")
            with open(batch_file_path, 'rb') as file:
                batch_input_file = self.client.files.create(
                    file=file,
                    purpose="batch"
                )
            
            # Create batch job
            logger.info("Creating batch job...")
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # Wait for completion
            batch_job = self._wait_for_batch_completion(
                batch_job.id, 
                batch_check_interval, 
                max_wait_time
            )
            
            # Download and parse results
            logger.info("Downloading batch results...")
            results = self._download_batch_results(batch_job.output_file_id, len(prompts))
            
            return results
            
        finally:
            # Clean up temporary file
            if os.path.exists(batch_file_path):
                os.unlink(batch_file_path)
    
    def _wait_for_batch_completion(
        self, 
        batch_id: str, 
        check_interval: int, 
        max_wait_time: int
    ):
        """Wait for batch job to complete."""
        start_time = time.time()
        
        while True:
            batch_job = self.client.batches.retrieve(batch_id)
            status = batch_job.status
            
            logger.info(f"Batch status: {status}")
            
            if status == "completed":
                logger.info("Batch job completed successfully")
                return batch_job
            elif status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch job {status}: {batch_job}")
            
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                raise Exception(f"Batch job timed out after {max_wait_time} seconds")
            
            time.sleep(check_interval)
    
    def _download_batch_results(self, output_file_id: str, expected_count: int) -> List[str]:
        """Download and parse batch results."""
        # Download file
        result_file_response = self.client.files.content(output_file_id)
        result_content = result_file_response.content.decode('utf-8')
        
        # Parse results
        results = [""] * expected_count
        
        for line in result_content.strip().split('\n'):
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id", "")
                
                # Extract index from custom_id
                try:
                    # Assuming custom_id format includes index
                    parts = custom_id.split('-')
                    for i, part in enumerate(parts):
                        if part.isdigit() and i < len(parts) - 1:
                            index = int(part)
                            break
                    
                    if result.get("response") and result["response"].get("body"):
                        response_body = result["response"]["body"]
                        if response_body.get("choices"):
                            content = response_body["choices"][0]["message"]["content"]
                            results[index] = content
                        else:
                            results[index] = "BATCH_ERROR"
                    else:
                        results[index] = "BATCH_ERROR"
                        
                except Exception as e:
                    logger.warning(f"Error parsing result: {e}")
                    continue
        
        # Replace empty results with error marker
        results = [r if r else "BATCH_ERROR" for r in results]
        
        success_count = sum(1 for r in results if r != "BATCH_ERROR")
        logger.info(f"Batch completed: {success_count}/{len(results)} successful")
        
        return results
    
    def generate_with_retry(
        self,
        prompts: List[str],
        system_instruction: Optional[str] = None,
        use_batch: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate completions with automatic retry and batch/sequential fallback.
        
        Args:
            prompts: List of prompts
            system_instruction: System instruction
            use_batch: Whether to use batch API for multiple prompts
            show_progress: Whether to show progress bar
            **kwargs: Additional parameters
            
        Returns:
            List of generated texts
        """
        if use_batch and len(prompts) > 1:
            # Create custom IDs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            custom_ids = [f"request-{timestamp}-{i}" for i in range(len(prompts))]
            
            try:
                return self.generate_batch(
                    prompts, 
                    custom_ids, 
                    system_instruction,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Batch API failed, falling back to sequential: {e}")
        
        # Sequential generation with retry
        results = []
        iterator = tqdm(prompts, desc="Generating") if show_progress else prompts
        
        for prompt in iterator:
            for attempt in range(BATCH_API_MAX_RETRIES):
                try:
                    result = self.generate_single(
                        prompt,
                        system_instruction,
                        **kwargs
                    )
                    results.append(result)
                    break
                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit error (attempt {attempt + 1}): {e}")
                    time.sleep(BATCH_API_SLEEP_TIME * 2)
                except Exception as e:
                    logger.error(f"Error (attempt {attempt + 1}): {e}")
                    if attempt < BATCH_API_MAX_RETRIES - 1:
                        time.sleep(BATCH_API_SLEEP_TIME)
                    else:
                        results.append("API_ERROR")
        
        return results