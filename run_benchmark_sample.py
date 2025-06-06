# 설명: FLARE 벤치마크의 일부 샘플(기본 5개)만 사용하여 Batch API를 테스트하는 스크립트.
#      전체 흐름을 빠르고 저렴하게 확인하는 용도입니다.
#
# 실행 전 필수 설치 항목:
# pip install torch datasets scikit-learn tqdm vllm openai pandas pyarrow
#
# 사용법:
# 1. 환경변수 설정 (필수):
#    export OPENAI_API_KEY='your-api-key-here'
#    export OPENAI_ORG_ID='your-org-id-here'  # 선택사항
# 2. 명령줄 실행 (기본 gpt-flare-edited 시나리오, 5개 샘플):
#    python run_benchmark_sample.py
# 3. 샘플 개수 조정하여 실행:
#    python run_benchmark_sample.py --num_samples 3
#
import os
import time
import json
from abc import ABC, abstractmethod
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import logging
from typing import List, Dict, Any, Optional
import sys

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 모델 인터페이스 추상화 ---
class BaseLLM(ABC):
    """모든 언어 모델이 따라야 할 기본 인터페이스 (ABC: Abstract Base Class)"""

    @abstractmethod
    def __init__(self, model_path_or_name: str, **kwargs):
        pass

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        pass


class VLLMModel(BaseLLM):
    """vLLM을 사용하여 로컬 Hugging Face 모델을 실행하는 클래스"""

    def __init__(self, model_path_or_name: str, tensor_parallel_size: int = 1, trust_remote_code: bool = True,
                 **kwargs):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM을 사용하려면 'pip install vllm'을 실행해주세요.")

        logger.info(f"[VLLM] 모델 로딩 중: {model_path_or_name}")
        self.llm = LLM(
            model=model_path_or_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code
        )
        self.sampling_params = self._create_sampling_params(**kwargs)

    def _create_sampling_params(self, **kwargs):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 20)
        )

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [out.outputs[0].text for out in outputs]


class GPTModel(BaseLLM):
    """OpenAI의 GPT API를 호출하는 클래스 (Batch API 지원)"""

    def __init__(self, model_path_or_name: str, **kwargs):
        try:
            import openai
        except ImportError:
            raise ImportError("GPT API를 사용하려면 'pip install openai'를 실행해주세요.")

        logger.info(f"[GPT] API 클라이언트 초기화 중: {model_path_or_name}")

        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("GPT를 사용하려면 환경 변수에 OPENAI_API_KEY를 설정해야 합니다.")

        if "OPENAI_ORG_ID" in os.environ:
            openai.organization = os.environ["OPENAI_ORG_ID"]

        self.client = openai.OpenAI()
        self.model_name = model_path_or_name
        self.request_timeout = kwargs.get("request_timeout", 30)
        self.max_retries = kwargs.get("max_retries", 3)
        self.sleep_time = kwargs.get("sleep_time", 5)
        self.use_batch_api = kwargs.get("use_batch_api", True)
        self.batch_check_interval = kwargs.get("batch_check_interval", 10)
        self.max_batch_wait_time = kwargs.get("max_batch_wait_time", 3600)
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 20)

    def generate(self, prompts: List[str]) -> List[str]:
        """프롬프트 리스트를 받아 생성된 텍스트 리스트를 반환합니다."""
        if self.use_batch_api and len(prompts) > 1:
            return self._generate_batch(prompts)
        else:
            return self._generate_sequential(prompts)

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Batch API를 사용하여 여러 프롬프트를 처리합니다."""
        logger.info(f"[GPT Batch] {len(prompts)}개 프롬프트를 Batch API로 처리 시작")

        batch_requests = []
        for i, prompt in enumerate(prompts):
            request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
            }
            batch_requests.append(request)

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
            batch_file_path = f.name

        try:
            logger.info("[GPT Batch] 배치 요청 파일 업로드 중...")
            with open(batch_file_path, 'rb') as file:
                batch_input_file = self.client.files.create(file=file, purpose="batch")

            logger.info("[GPT Batch] 배치 작업 생성 중...")
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            logger.info(f"[GPT Batch] 배치 작업 대기 중... (ID: {batch_job.id})")
            batch_job = self._wait_for_batch_completion(batch_job.id)

            logger.info("[GPT Batch] 결과 다운로드 및 파싱 중...")
            results = self._download_batch_results(batch_job.output_file_id, len(prompts))
            return results
        finally:
            if os.path.exists(batch_file_path):
                os.unlink(batch_file_path)

    def _wait_for_batch_completion(self, batch_id: str):
        """배치 작업이 완료될 때까지 대기합니다."""
        start_time = time.time()
        while True:
            batch_job = self.client.batches.retrieve(batch_id)
            status = batch_job.status
            logger.info(f"[GPT Batch] 현재 상태: {status} (경과 시간: {int(time.time() - start_time)}초)")

            if status in ["completed", "failed", "expired", "cancelled"]:
                if status == "completed":
                    logger.info("[GPT Batch] 배치 작업이 완료되었습니다.")
                else:
                    logger.error(f"[GPT Batch] 배치 작업이 {status} 상태로 종료되었습니다: {batch_job.errors}")
                return batch_job

            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_batch_wait_time:
                raise Exception(f"배치 작업 대기 시간 초과 ({self.max_batch_wait_time}초)")

            time.sleep(self.batch_check_interval)

    def _download_batch_results(self, output_file_id: str, expected_count: int) -> List[str]:
        """배치 결과를 다운로드하고 파싱합니다."""
        if not output_file_id:
            logger.error("[GPT Batch] 출력 파일 ID가 없습니다. 모든 요청이 실패했을 수 있습니다.")
            return ["BATCH_ERROR"] * expected_count

        result_file_response = self.client.files.content(output_file_id)
        result_content = result_file_response.content.decode('utf-8')
        results = [""] * expected_count

        for line in result_content.strip().split('\n'):
            if not line.strip(): continue
            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            try:
                index = int(custom_id.split('-')[1])
                if result.get("response") and result["response"].get("body"):
                    response_body = result["response"]["body"]
                    if response_body.get("choices") and len(response_body["choices"]) > 0:
                        content = response_body["choices"][0]["message"]["content"]
                        results[index] = content
                    else:
                        results[index] = "BATCH_ERROR: No choices in response"
                else:
                    error_message = result.get("error", {}).get("message", "Unknown error")
                    results[index] = f"BATCH_ERROR: {error_message}"
            except (ValueError, IndexError, KeyError) as e:
                logger.warning(f"결과 파싱 오류: {e} - 라인: {line}")
                continue

        results = [r if r else "BATCH_ERROR: Empty result" for r in results]
        success_count = sum(1 for r in results if not r.startswith('BATCH_ERROR'))
        logger.info(f"[GPT Batch] 총 {len(results)}개 결과 중 {success_count}개 성공")
        return results

    def _generate_sequential(self, prompts: List[str]) -> List[str]:
        """순차적으로 API를 호출합니다 (기존 방식)."""
        logger.info(f"[GPT Sequential] {len(prompts)}개 프롬프트를 순차 처리 시작")
        results = []
        for prompt in tqdm(prompts, desc="[GPT] API 요청 중"):
            # 샘플 테스트에서는 재시도 로직을 단순화합니다.
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.request_timeout
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"순차 API 호출 오류: {e}")
                results.append("API_ERROR")
        return results


# --- 프롬프트 및 태스크 로직 ---

def finma_prompt(ctx: str) -> str:
    return f'Human: \n{ctx}\n\nAssistant: \n'


MODEL_PROMPT_MAP = {"finma_prompt": finma_prompt, "no_prompt": lambda ctx: ctx}


class StockMovementEvaluator:
    """주가 변동 예측 태스크의 로직 담당"""
    CHOICE_MAPPING = {"rise": ["yes", "positive"], "fall": ["no", "negative", "neutral"]}
    DEFAULT_CHOICE = "fall"
    GOLD_LABELS = ["fall", "rise"]

    def create_prompt(self, doc: Dict[str, Any], template_func) -> str:
        return template_func(doc["query"])

    def process_single_result(self, generated_text: str) -> str:
        text = generated_text.lower().strip()
        if "rise" in text or any(val in text for val in self.CHOICE_MAPPING["rise"]):
            return "rise"
        if "fall" in text or any(val in text for val in self.CHOICE_MAPPING["fall"]):
            return "fall"
        return self.DEFAULT_CHOICE

    def get_gold_label(self, doc: Dict[str, Any]) -> str:
        """데이터 샘플에서 정답 레이블을 추출하고 소문자로 변환합니다."""
        # 중요: 데이터셋의 'Fall', 'Rise'를 소문자로 통일
        return doc["choices"][doc["gold"]].lower()


# --- 메인 평가 함수 ---

def run_evaluation(config: Dict[str, Any], num_samples: int) -> Dict[str, Any]:
    """설정값을 바탕으로 평가를 수행하는 메인 함수"""
    logger.info("=" * 50)
    logger.info(f"샘플 벤치마크 평가를 시작합니다: {config.get('experiment_name', 'Untitled')}")
    logger.info(f"  - 테스트할 샘플 수: {num_samples}")
    logger.info("=" * 50)

    config = config.copy()
    model_type = config.pop("model_type")
    model_path_or_name = config.pop("model_path_or_name")

    if model_type == 'vllm':
        model = VLLMModel(model_path_or_name, **config)
    elif model_type == 'gpt':
        model = GPTModel(model_path_or_name, **config)
    else:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    logger.info("[1/4] 데이터셋 로드 및 프롬프트 생성...")
    full_dataset = load_dataset("parquet",  # 데이터 타입을 'parquet'으로 지정
        data_files=config['dataset_path'],
        split=config['dataset_split']
    )

    # *** 핵심 변경점: 전체 데이터셋에서 일부 샘플만 선택 ***
    sample_dataset = full_dataset.select(range(min(num_samples, len(full_dataset))))
    logger.info(f"전체 {len(full_dataset)}개 데이터 중 {len(sample_dataset)}개를 샘플링했습니다.")

    evaluator = StockMovementEvaluator()
    prompt_template_func = MODEL_PROMPT_MAP[config['prompt_template']]

    prompts = [evaluator.create_prompt(doc, prompt_template_func) for doc in sample_dataset]
    gold_labels = [evaluator.get_gold_label(doc) for doc in sample_dataset]

    logger.info("[2/4] 모델 추론 시작...")
    generated_texts = model.generate(prompts)
    logger.info("모델 추론 완료.")

    # 생성된 텍스트와 정답을 함께 출력하여 직접 비교하기 용이하게 함
    logger.info("-" * 20 + " 추론 결과 비교 " + "-" * 20)
    for i, (gold, pred_text) in enumerate(zip(gold_labels, generated_texts)):
        processed_pred = evaluator.process_single_result(pred_text)
        status = "✅" if gold == processed_pred else "❌"
        logger.info(f"샘플 {i + 1}: {status} | 정답: {gold:<5} | 모델 출력: '{pred_text.strip()}' -> 처리 결과: {processed_pred}")
    logger.info("-" * 58)

    logger.info("[3/4] 결과 처리 및 성능 지표 계산...")
    predicted_labels = [evaluator.process_single_result(text) for text in generated_texts]
    accuracy = accuracy_score(gold_labels, predicted_labels)
    f1_macro = f1_score(gold_labels, predicted_labels, average='macro', labels=evaluator.GOLD_LABELS)
    mcc = matthews_corrcoef(gold_labels, predicted_labels)

    results = {
        "metrics": {"accuracy": accuracy, "f1_macro": f1_macro, "mcc": mcc},
        "num_samples": len(gold_labels)
    }

    logger.info("[4/4] 최종 결과")
    logger.info("=" * 50)
    logger.info(json.dumps(results, indent=2))
    logger.info("=" * 50)

    return results


def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """기본 설정들을 반환합니다."""
    return {
        "gpt-flare-edited": {
            "experiment_name": "[SAMPLE] GPT-4.1 mini on Stock Movement",
            "model_type": "gpt",
            "model_path_or_name": "gpt-4.1-mini-2025-04-14",
            "prompt_template": "no_prompt",
            # 데이터 파일의 실제 경로를 직접 지정합니다.
            "dataset_path": "./cache/flare_edited_test.parquet",
            # 단일 파일을 로드할 때, datasets 라이브러리는 기본적으로 'train' 스플릿으로 인식합니다.
            "dataset_split": "train",
            "use_batch_api": True,
            "batch_check_interval": 10,
            "max_batch_wait_time": 600,  # 샘플 테스트는 오래 기다릴 필요 없음
            "max_tokens": 20,
            "temperature": 0.0
        }
    }


# --- 실험 실행 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLARE 벤치마크 샘플 실행")
    # 샘플 개수를 조절할 수 있는 인자 추가
    parser.add_argument("--num_samples", type=int, default=5, help="테스트에 사용할 샘플 데이터의 수")
    parser.add_argument("--config", type=str, help="JSON 설정 파일 경로 (미지정 시 기본값 사용)")

    args = parser.parse_args()

    # 샘플 테스트는 gpt-flare-edited 시나리오만 사용하도록 고정
    config = get_default_configs()["gpt-flare-edited"]
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
            logger.info(f"사용자 설정 파일 '{args.config}'을(를) 로드했습니다.")
        except FileNotFoundError:
            logger.error(f"설정 파일 '{args.config}'을(를) 찾을 수 없습니다. 기본 설정을 사용합니다.")
        except json.JSONDecodeError:
            logger.error(f"설정 파일 '{args.config}' 파싱 오류. 기본 설정을 사용합니다.")

    try:
        results = run_evaluation(config, args.num_samples)
        logger.info("샘플 평가가 성공적으로 완료되었습니다.")
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}", exc_info=True)  # exc_info=True로 traceback 출력
        sys.exit(1)