"""
Gemini client wrapper for the Stock Tweet User Credibility project
(adapted for the 2025-07 Google Gen AI SDK: google-genai).

Public API
----------
- generate_single(...)
- generate_batch(...)
- generate_with_retry(...)

Author: ChatGPT
"""
from __future__ import annotations

import os
import json
import time
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Google Gen AI SDK (pip install --upgrade google-genai)
# ──────────────────────────────────────────────────────────────────────────────
from google import genai
from google.api_core import exceptions as gexc
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# Local utilities / constants
# ──────────────────────────────────────────────────────────────────────────────
from .utils import setup_logging
from .constants import (
    BATCH_CHECK_INTERVAL,
    MAX_BATCH_WAIT_TIME,
    BATCH_API_MAX_RETRIES,
    BATCH_API_SLEEP_TIME,
)

logger = setup_logging(__name__)


class GeminiClient:
    """Wrapper around Google Generative AI with batch support."""

    # --------------------------------------------------------------------- #
    # Init
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ):
        """
        Parameters
        ----------
        api_key : str | None
            Google Gen AI API Key. Falls back to the ``GEMINI_API_KEY`` env var.
        model : str
            Model name, e.g. ``gemini-2.5-flash`` or ``gemini-1.5-pro-latest``.
        """
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not provided and GEMINI_API_KEY env var not set."
            )

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)
        self.client = genai.Client()
        self.model_name = model

        logger.info(f"Initialized GeminiClient with model: {self.model_name}")

    # --------------------------------------------------------------------- #
    # Low-level helpers
    # --------------------------------------------------------------------- #
    def _gen_config(
        self,
        temperature: float,
        max_tokens: int,
        extra: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        cfg = {"temperature": temperature, "max_output_tokens": max_tokens}
        if extra:
            cfg.update(extra)
        return cfg

    # --------------------------------------------------------------------- #
    # Single generation
    # --------------------------------------------------------------------- #
    def generate_single(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 20,
        **kwargs,
    ) -> str:
        """
        Synchronous single-prompt completion.

        Returns
        -------
        str
            Model’s text response.
        """
        full_prompt = ""
        if system_instruction:
            full_prompt += system_instruction + "\n"
        full_prompt += prompt

        try:
            resp = self.model.generate_content(
                contents=full_prompt,
                generation_config=self._gen_config(temperature, max_tokens, kwargs)
            )
            # 1.25.x: resp.candidates[0].content.parts[0].text
            try:
                return resp.text
            except AttributeError:
                return resp.candidates[0].content.parts[0].text
        except gexc.GoogleAPIError as e:
            logger.error(f"Gemini API error: {e}")
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Unexpected error: {e}")
            raise

    # --------------------------------------------------------------------- #
    # Batch helpers
    # --------------------------------------------------------------------- #
    def _wait_for_batch(
        self,
        job_name: str,
        check_interval: int = BATCH_CHECK_INTERVAL,
        max_wait: int = MAX_BATCH_WAIT_TIME,
    ) -> genai.BatchJob:
        """Poll batch job until completion or timeout."""
        start = time.time()
        while True:
            job = self.client.batches.get(name=job_name)
            state = job.state.name  # e.g. JOB_STATE_SUCCEEDED
            logger.info(f"[Gemini Batch] {job_name} → {state}")

            if state == "JOB_STATE_SUCCEEDED":
                return job
            if state in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                raise RuntimeError(f"Batch job ended with state: {state}")
            if time.time() - start > max_wait:
                raise TimeoutError(f"Batch job timed out after {max_wait} s")
            time.sleep(check_interval)

    def _download_batch_results(
        self,
        job: genai.BatchJob,
        expected: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Fetch results and map to list[str] aligned with request order.

        Notes
        -----
        - If `job.dest.inline_responses` is used, results are inline.
        - Otherwise a file is produced; download via client.files.download.
        """
        results: list[str] = [""] * expected
        originals: list[dict[str, Any]] = []

        # Prefer inline dest (smaller jobs)
        if job.dest.HasField("inline_responses"):
            for idx, proto in enumerate(job.dest.inline_responses.responses):
                resp = types.GenerateContentResponse.from_proto(proto)
                originals.append(resp.to_dict())
                results[idx] = resp.text or "BATCH_ERROR"
        else:  # file dest
            file_name = job.dest.file.name
            file_bytes = self.client.files.download(name=file_name)
            for idx, line in enumerate(file_bytes.decode("utf-8").splitlines()):
                if not line.strip():
                    continue
                js = json.loads(line)
                originals.append(js)
                # Google's batch JSON line has the structure:
                # { "response": { "candidates": [ { "content": {...} } ] } }
                try:
                    results[idx] = (
                        js["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    )
                except Exception:
                    results[idx] = "BATCH_ERROR"

        success = sum(1 for r in results if r != "BATCH_ERROR")
        logger.info(f"[Gemini Batch] success {success}/{expected}")
        return results, originals

    # --------------------------------------------------------------------- #
    # Batch generation
    # --------------------------------------------------------------------- #
    def generate_batch(
        self,
        prompts: List[str],
        custom_ids: List[str] | None = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 20,
        batch_check_interval: int = BATCH_CHECK_INTERVAL,
        max_wait_time: int = MAX_BATCH_WAIT_TIME,
        **kwargs,
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Submit prompts as a batch job and wait for completion.

        Returns
        -------
        (batch_job_name, responses, raw_responses)
        """
        if custom_ids and len(custom_ids) != len(prompts):
            raise ValueError("custom_ids length must match prompts length")

        # 1. Build JSONL file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fp:
            for idx, prompt in enumerate(prompts):
                full_prompt = ""
                if system_instruction:
                    full_prompt += system_instruction + "\n"
                full_prompt += prompt

                req = {
                    "contents": full_prompt,
                    "generation_config": self._gen_config(
                        temperature, max_tokens, kwargs
                    ),
                }
                fp.write(json.dumps(req, ensure_ascii=False) + "\n")
            jsonl_path = Path(fp.name)

        try:
            # 2. Upload file
            upload = self.client.files.upload(file=jsonl_path)
            logger.info(f"Uploaded JSONL → {upload.name}")

            # 3. Create batch job
            job = self.client.batches.create(
                source=upload.name,
                model=self.model_name,
                dest_inline=True  # inline responses for <=50 prompts
            )
            logger.info(f"Created batch job: {job.name}")

            # 4. Wait
            job = self._wait_for_batch(job.name, batch_check_interval, max_wait_time)

            # 5. Download / parse
            texts, originals = self._download_batch_results(job, len(prompts))
            return job.name, texts, originals

        finally:
            try:
                jsonl_path.unlink(missing_ok=True)
            except Exception:
                pass  # pragma: no cover

    # --------------------------------------------------------------------- #
    # generate_with_retry (sequential fallback)
    # --------------------------------------------------------------------- #
    def generate_with_retry(
        self,
        prompts: List[str],
        system_instruction: Optional[str] = None,
        use_batch: bool = True,
        custom_ids: Optional[List[str]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> Tuple[Optional[str], List[str], List[Dict[str, Any]]]:
        """
        Wrapper mimicking OpenAIClient.generate_with_retry.
        """
        if use_batch and len(prompts) > 1:
            try:
                return self.generate_batch(
                    prompts=prompts,
                    custom_ids=custom_ids,
                    system_instruction=system_instruction,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"Gemini batch failed → sequential fallback. ({e})")

        # Sequential loop with rudimentary backoff
        from tqdm import tqdm  # local import to avoid unnecessary dep
        results: list[str] = []
        originals: list[dict[str, Any]] = []
        iterator = tqdm(prompts, desc="Gemini (sequential)") if show_progress else prompts

        for prompt in iterator:
            for attempt in range(BATCH_API_MAX_RETRIES):
                try:
                    txt = self.generate_single(
                        prompt,
                        system_instruction=system_instruction,
                        **kwargs,
                    )
                    results.append(txt)
                    originals.append({"text": txt})
                    break
                except gexc.ResourceExhausted as e:
                    logger.warning(f"Rate-limited ({attempt+1}) – {e}")
                    time.sleep(BATCH_API_SLEEP_TIME * (attempt + 1))
                except Exception as e:
                    logger.error(f"Error ({attempt+1}) – {e}")
                    if attempt == BATCH_API_MAX_RETRIES - 1:
                        results.append("API_ERROR")
                        originals.append({"error": str(e)})

        return None, results, originals
