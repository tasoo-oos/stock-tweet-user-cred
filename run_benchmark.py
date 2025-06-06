# 설명: FLARE의 주가 변동 예측 벤치마크를 독립적으로 실행하는 확장형 스크립트.
#      로컬 Hugging Face 모델(vLLM)과 GPT API를 모두 지원합니다.
#
# 실행 전 필수 설치 항목:
# pip install torch datasets scikit-learn tqdm vllm openai
#
# 사용법:
# 1. 환경변수 설정 (GPT 사용 시):
#    export OPENAI_API_KEY='your-api-key-here'
#    export OPENAI_ORG_ID='your-org-id-here'  # 선택사항
# 2. 명령줄 실행:
#    python run_benchmark.py --scenario gpt4
#    python run_benchmark.py --config config.json
# 3. 설정은 config.json 파일로 관리하거나 환경변수로 설정하세요.
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

        # API 키 확인
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "GPT를 사용하려면 환경 변수에 OPENAI_API_KEY를 설정해야 합니다.\n"
                "예: export OPENAI_API_KEY='your-api-key-here'"
            )

        # Organization ID 설정 (선택사항)
        if "OPENAI_ORG_ID" in os.environ:
            openai.organization = os.environ["OPENAI_ORG_ID"]

        self.client = openai.OpenAI()
        self.model_name = model_path_or_name
        self.request_timeout = kwargs.get("request_timeout", 30)
        self.max_retries = kwargs.get("max_retries", 3)
        self.sleep_time = kwargs.get("sleep_time", 5)
        self.use_batch_api = kwargs.get("use_batch_api", True)
        self.batch_check_interval = kwargs.get("batch_check_interval", 10)  # 10초마다 상태 확인
        self.max_batch_wait_time = kwargs.get("max_batch_wait_time", 3600)  # 최대 1시간 대기

    def generate(self, prompts: List[str]) -> List[str]:
        """프롬프트 리스트를 받아 생성된 텍스트 리스트를 반환합니다."""
        if self.use_batch_api and len(prompts) > 1:
            return self._generate_batch(prompts)
        else:
            return self._generate_sequential(prompts)

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Batch API를 사용하여 여러 프롬프트를 처리합니다."""
        logger.info(f"[GPT Batch] {len(prompts)}개 프롬프트를 Batch API로 처리 시작")

        # 1. Batch 요청 파일 생성
        batch_requests = []
        for i, prompt in enumerate(prompts):
            request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 20
                }
            }
            batch_requests.append(request)

        # 2. 임시 파일에 저장
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
            batch_file_path = f.name

        try:
            # 3. 파일 업로드
            logger.info("[GPT Batch] 배치 요청 파일 업로드 중...")
            with open(batch_file_path, 'rb') as file:
                batch_input_file = self.client.files.create(
                    file=file,
                    purpose="batch"
                )

            # 4. 배치 작업 생성
            logger.info("[GPT Batch] 배치 작업 생성 중...")
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            # 5. 배치 작업 완료 대기
            logger.info(f"[GPT Batch] 배치 작업 대기 중... (ID: {batch_job.id})")
            batch_job = self._wait_for_batch_completion(batch_job.id)

            # 6. 결과 다운로드 및 파싱
            logger.info("[GPT Batch] 결과 다운로드 및 파싱 중...")
            results = self._download_batch_results(batch_job.output_file_id, len(prompts))

            return results

        finally:
            # 임시 파일 정리
            import os
            if os.path.exists(batch_file_path):
                os.unlink(batch_file_path)

    def _wait_for_batch_completion(self, batch_id: str):
        """배치 작업이 완료될 때까지 대기합니다."""
        start_time = time.time()

        while True:
            batch_job = self.client.batches.retrieve(batch_id)
            status = batch_job.status

            logger.info(f"[GPT Batch] 현재 상태: {status}")

            if status == "completed":
                logger.info("[GPT Batch] 배치 작업이 완료되었습니다.")
                return batch_job
            elif status == "failed":
                raise Exception(f"배치 작업이 실패했습니다: {batch_job}")
            elif status == "expired":
                raise Exception("배치 작업이 만료되었습니다.")
            elif status == "cancelled":
                raise Exception("배치 작업이 취소되었습니다.")

            # 최대 대기 시간 확인
            elapsed_time = time.time() - start_time
            if elapsed_time > self.max_batch_wait_time:
                raise Exception(f"배치 작업 대기 시간 초과 ({self.max_batch_wait_time}초)")

            # 대기
            time.sleep(self.batch_check_interval)

    def _download_batch_results(self, output_file_id: str, expected_count: int) -> List[str]:
        """배치 결과를 다운로드하고 파싱합니다."""
        # 결과 파일 다운로드
        result_file_response = self.client.files.content(output_file_id)
        result_content = result_file_response.content.decode('utf-8')

        # 결과 파싱
        results = [""] * expected_count  # 순서 유지를 위해 초기화

        for line in result_content.strip().split('\n'):
            if line.strip():
                result = json.loads(line)
                custom_id = result.get("custom_id", "")

                # custom_id에서 인덱스 추출 (예: "request-0" -> 0)
                try:
                    index = int(custom_id.split('-')[1])

                    if result.get("response") and result["response"].get("body"):
                        response_body = result["response"]["body"]
                        if response_body.get("choices") and len(response_body["choices"]) > 0:
                            content = response_body["choices"][0]["message"]["content"]
                            results[index] = content
                        else:
                            results[index] = "BATCH_ERROR"
                    else:
                        results[index] = "BATCH_ERROR"

                except (ValueError, IndexError, KeyError) as e:
                    logger.warning(f"결과 파싱 오류: {e}")
                    continue

        # 빈 결과를 에러로 표시
        results = [r if r else "BATCH_ERROR" for r in results]

        logger.info(f"[GPT Batch] 총 {len(results)}개 결과 중 {sum(1 for r in results if r != 'BATCH_ERROR')}개 성공")

        return results

    def _generate_sequential(self, prompts: List[str]) -> List[str]:
        """순차적으로 API를 호출합니다 (기존 방식)."""
        logger.info(f"[GPT Sequential] {len(prompts)}개 프롬프트를 순차 처리 시작")
        results = []
        for prompt in tqdm(prompts, desc="[GPT] API 요청 중"):
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=20,
                        timeout=self.request_timeout
                    )
                    results.append(response.choices[0].message.content)
                    break  # 성공 시 루프 탈출
                except openai.RateLimitError as e:
                    logger.warning(f"Rate limit 오류 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.sleep_time * 2)  # Rate limit의 경우 더 오래 대기
                except openai.APIError as e:
                    logger.warning(f"API 오류 발생 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.sleep_time)
                except Exception as e:
                    logger.error(f"예상치 못한 오류 (시도 {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.sleep_time)
                    else:
                        logger.error("최대 재시도 횟수 초과. 'API_ERROR'로 처리합니다.")
                        results.append("API_ERROR")
        return results


# --- 프롬프트 및 태스크 로직 ---

def finma_prompt(ctx: str) -> str:
    """FinMA 모델을 위한 프롬프트 템플릿 (from src/model_prompt.py)"""
    return f'Human: \n{ctx}\n\nAssistant: \n'


MODEL_PROMPT_MAP = {
    "finma_prompt": finma_prompt,
    "no_prompt": lambda ctx: ctx,
}


class StockMovementEvaluator:
    """주가 변동 예측 태스크의 로직 담당 (from src/tasks/flare.py)"""
    # 이 규칙은 flare.py의 StockMovement 클래스에서 정확히 가져온 것입니다.
    CHOICE_MAPPING = {
        "rise": ["yes", "positive"],
        "fall": ["no", "negative", "neutral"],
    }
    DEFAULT_CHOICE = "fall"
    GOLD_LABELS = ["fall", "rise"]

    def create_prompt(self, doc: Dict[str, Any], template_func) -> str:
        """데이터 샘플로부터 모델 입력 프롬프트를 생성합니다."""
        return template_func(doc["query"])

    def process_single_result(self, generated_text: str) -> str:
        """모델 출력을 정규화된 레이블('rise' 또는 'fall')로 변환합니다."""
        text = generated_text.lower().strip()
        if "rise" in text or any(val in text for val in self.CHOICE_MAPPING["rise"]):
            return "rise"
        if "fall" in text or any(val in text for val in self.CHOICE_MAPPING["fall"]):
            return "fall"
        return self.DEFAULT_CHOICE

    def get_gold_label(self, doc: Dict[str, Any]) -> str:
        """데이터 샘플에서 정답 레이블을 추출하고 소문자로 변환합니다."""
        # .lower()를 추가하여 반환값을 항상 소문자로 만듭니다.
        return doc["choices"][doc["gold"]].lower()


# --- 메인 평가 함수 ---

def run_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """설정값을 바탕으로 평가를 수행하는 메인 함수"""
    logger.info("=" * 50)
    logger.info(f"벤치마크 평가를 시작합니다: {config.get('experiment_name', 'Untitled')}")
    logger.info(f"  - 모델 타입: {config['model_type']}")
    logger.info(f"  - 모델 이름/경로: {config['model_path_or_name']}")
    logger.info(f"  - 데이터셋 경로: {config['dataset_path']} ({config['dataset_split']} split)")
    logger.info("=" * 50)

    # 설정 복사본 생성 (원본 변경 방지)
    config = config.copy()

    # 1. 모델 인스턴스 생성
    model_type = config.pop("model_type")
    model_path_or_name = config.pop("model_path_or_name")

    if model_type == 'vllm':
        model = VLLMModel(model_path_or_name, **config)
    elif model_type == 'gpt':
        model = GPTModel(model_path_or_name, **config)
    else:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    # 2. 데이터셋 로드 및 프롬프트 생성
    logger.info("[1/4] 데이터셋 로드 및 프롬프트 생성...")
    try:
        # 데이터셋 경로가 로컬 파일(.parquet 등)인지 Hugging Face Hub 경로인지 확인
        is_local_file = os.path.isfile(config['dataset_path'])
        if is_local_file:
            # 로컬 파일인 경우, 데이터 타입을 명시적으로 지정
            logger.info(f"로컬 Parquet 파일 로드: {config['dataset_path']}")
            dataset = load_dataset(
                "parquet",
                data_files=config['dataset_path'],
                split=config['dataset_split']
            )
        else:
            # Hugging Face Hub 경로인 경우, 기존 방식대로 로드
            logger.info(f"Hugging Face Hub 데이터셋 로드: {config['dataset_path']}")
            dataset = load_dataset(config['dataset_path'], split=config['dataset_split'])

    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        raise

    evaluator = StockMovementEvaluator()
    prompt_template_func = MODEL_PROMPT_MAP[config['prompt_template']]

    prompts = [evaluator.create_prompt(doc, prompt_template_func) for doc in dataset]
    gold_labels = [evaluator.get_gold_label(doc) for doc in dataset]
    logger.info(f"총 {len(prompts)}개의 평가 데이터 준비 완료.")

    # 3. 모델 추론 실행
    logger.info("[2/4] 모델 추론 시작...")
    generated_texts = model.generate(prompts)
    logger.info("모델 추론 완료.")

    # 4. 결과 처리 및 지표 계산
    logger.info("[3/4] 결과 처리 및 성능 지표 계산...")
    predicted_labels = [evaluator.process_single_result(text) for text in generated_texts]

    accuracy = accuracy_score(gold_labels, predicted_labels)
    f1_macro = f1_score(gold_labels, predicted_labels, average='macro', labels=evaluator.GOLD_LABELS)
    mcc = matthews_corrcoef(gold_labels, predicted_labels)

    results = {
        "metrics": {"accuracy": accuracy, "f1_macro": f1_macro, "mcc": mcc},
        "num_samples": len(gold_labels)
    }

    # 최종 결과 출력
    logger.info("[4/4] 최종 결과")
    logger.info("=" * 50)
    logger.info(json.dumps(results, indent=2))
    logger.info("=" * 50)

    return results


def load_config_from_file(config_path: str) -> Optional[Dict[str, Any]]:
    """JSON 설정 파일에서 설정을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"설정 파일 파싱 오류: {e}")
        return None


def get_default_configs() -> Dict[str, Dict[str, Any]]:
    """기본 설정들을 반환합니다."""
    return {
        "finma": {
            "experiment_name": "FinMA 7B Full on Stock Movement (ACL18)",
            "model_type": "vllm",
            "model_path_or_name": "TheFinAI/finma-7b-full",
            "prompt_template": "finma_prompt",
            "dataset_path": "TheFinAI/flare-sm-acl",
            "dataset_split": "test",
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
        },
        "gpt-flare": {
            "experiment_name": "GPT 3.5 Turbo on Stock Movement (ACL18)",
            "model_type": "gpt",
            "model_path_or_name": "gpt-3.5-turbo",
            "prompt_template": "no_prompt",
            "dataset_path": "TheFinAI/flare-sm-acl",
            "dataset_split": "test",
            "max_retries": 3,
            "sleep_time": 5,
            "use_batch_api": True,
            "batch_check_interval": 10,
            "max_batch_wait_time": 3600
        },
        "gpt-flare-edited": {
            "experiment_name": "GPT-4.1 mini on Stock Movement (with flare-edited_test.csv)",
            "model_type": "gpt",
            "model_path_or_name": "gpt-4.1-mini-2025-04-14",
            "prompt_template": "no_prompt",
            # 데이터 파일의 실제 경로를 직접 지정합니다.
            "dataset_path": "./cache/flare_edited_test.parquet",
            # 단일 파일을 로드할 때, datasets 라이브러리는 기본적으로 'train' 스플릿으로 인식합니다.
            "dataset_split": "train",
            "use_batch_api": True,
            "batch_check_interval": 15,
            "max_batch_wait_time": 7200  # GPT-4는 더 오래 걸릴 수 있음
        }
    }


# --- 실험 실행 ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLARE 벤치마크 실행")
    parser.add_argument("--config", type=str, help="JSON 설정 파일 경로")
    parser.add_argument("--scenario", type=str, choices=["finma", "gpt-flare", "gpt-flare-edited"],
                        default="gpt-flare-edited", help="실행할 시나리오 선택")

    args = parser.parse_args()

    # 설정 로드
    if args.config:
        config = load_config_from_file(args.config)
        if config is None:
            logger.error("설정 파일 로드 실패. 기본 설정을 사용합니다.")
            config = get_default_configs()[args.scenario]
    else:
        config = get_default_configs()[args.scenario]

    try:
        results = run_evaluation(config)
        logger.info("평가가 성공적으로 완료되었습니다.")
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        sys.exit(1)