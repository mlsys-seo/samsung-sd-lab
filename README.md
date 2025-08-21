# Speculative Decoding Engine

이 실습은 LLM 추론 최적화를 위한 Speculative Decoding 기법을 단계적으로 구현하는 것을 목표로 합니다. 총 4단계의 문제(Problem A–D)를 해결하며, Draft Model과 Target Model을 활용한 효율적인 토큰 생성, Bonus Token 처리, KV Cache 관리 및 Cutoff 기능을 구현하게 됩니다.

## 1. 환경 세팅

1. 강의 자료 드라이브에서 run_sd_container.sh 파일을 다운로드 후 서버에 복사합니다.
2. 서버에서 스크립트를 실행하여 환경을 설정합니다.
3. 설정 완료 후, VSCode Dev Containers를 이용해 컨테이너 내부에 접속합니다.

실습에 제공된 주요 파일:
- 실행 스크립트: run_sd_container.sh
- Speculative Decoding 엔진: engine.py, spec_engine.py

단계별 구현용 파일:
- spec_engine_A.py, spec_engine_B.py, spec_engine_D.py
- model_runner_C.py

## 주요 구현 목표

1.  **Speculative Decoding 기본 구조 구현 (Problem A):** `spec_engine_A.py`에서 Draft model을 사용하여 후보 토큰들을 생성하고, Target model을 통해 검증하여 토큰 생성 속도를 향상시키는 기본적인 Speculative Decoding 파이프라인을 완성합니다.
2.  **Bonus Token 처리 기능 추가 (Problem B):** `spec_engine_B.py`에서 Draft model이 제안한 후보 토큰들이 모두 채택되었을 경우, 추가적인 "Bonus Token"을 활용하여 다음 단계의 추론을 더욱 효율적으로 만드는 기능을 구현합니다.
3.  **KV Cache 관리 기능 구현 (Problem C):** `model_runner_C.py`의 `ExpendedStaticCache` 클래스에 `cutoff` 기능을 구현합니다. 이 기능은 추론 과정에서 불필요해진 Key-Value Cache의 일부를 제거하여, 메모리를 효율적으로 관리하고 올바른 추론이 가능하게 하는 핵심적인 역할을 합니다.
4.  **Cache Cutoff 적용 (Problem D):** `spec_engine_D.py`에서 검증 단계 이후, 채택되지 않은 토큰들에 해당하는 KV Cache를 `cutoff` 기능을 이용해 실제로 제거하는 로직을 구현합니다.

---

## TODO 구현 목록

### `utils/spec_engine_A.py` (Problem A)

*   **목표:** 기본적인 Speculative Decoding 로직을 완성합니다.
*   **`step_problem` 함수 내 TODO 구현:**
    1.  **Draft Model 실행:** `input_ids`로 시작하여 `self.num_draft_tokens` 만큼 후보 토큰을 생성하고 `self.draft_tokens_buffer`에 저장합니다.
    2.  **Target Model 실행:** 원본 `input_ids`와 `draft_tokens_buffer`를 합쳐 Target model을 실행하고, `target_tokens`를 얻습니다.
    3.  **검증 (Verification):** `draft_tokens_buffer`와 `target_tokens`를 비교하여 채택된 토큰 수(`num_accepted_tokens`)를 계산하고, 최종 `emitted_tokens`를 결정합니다.

### `utils/spec_engine_B.py` (Problem B)

*   **목표:** Bonus Token 메커니즘을 추가하여 Speculative Decoding 효율을 높입니다.
*   **`__init__` 함수 내 TODO 구현:** `self.all_accepted`와 `self.last_bonus_token_id_buffer` 변수를 초기화합니다.
*   **`step_problem` 함수 내 TODO 구현:**
    1.  **Draft Input 준비:** 이전 스텝에서 모든 토큰이 채택되었다면(`self.all_accepted`), `last_bonus_token_id_buffer`의 토큰을 다음 Draft 생성의 시작으로 사용합니다.
    2.  **Bonus Token 처리:** 검증 후, 모든 후보 토큰이 채택되었는지 여부를 `self.all_accepted`에 기록하고, 채택되었다면 마지막 토큰을 `last_bonus_token_id_buffer`에 저장합니다.

### `utils/model_runner_C.py` (Problem C)

*   **목표:** Speculative Decoding 과정에서 채택되지 않은 토큰에 해당하는 불필요한 Key-Value(KV) Cache를 제거하는 기능을 구현합니다. `transformers`의 기본 `StaticCache`에는 이 기능이 없으므로, 이를 상속받는 `ExpendedStaticCache` 클래스를 만들고 `cutoff` 로직을 추가하는 것이 핵심입니다.
*   **`ExpendedStaticCache.cutoff_problem` 함수 내 TODO 구현:**
    *   이 파일에는 이미 완전하게 구현된 `cutoff` 메소드가 예시로 제공됩니다. **이 `cutoff` 메소드와 동일한 로직을 `cutoff_problem` 메소드 안에 구현하는 것이 목표입니다.**
    *   **구현 절차:**
        1.  `self.get_seq_length()`를 호출하여 현재 캐시에 저장된 토큰의 총 길이(`seq_length`)를 가져옵니다.
        2.  `num_tokens_to_cutoff` (잘라낼 토큰의 수)를 사용하여, 캐시에서 0으로 만들 영역의 시작 인덱스(`fill_zero_start_idx`)와 끝 인덱스(`fill_zero_end_idx`)를 계산합니다.
        3.  `for` 루프를 사용하여 `self.layers`의 모든 레이어를 순회합니다.
        4.  각 레이어의 `keys`와 `values` 텐서에 대해, 위에서 계산한 인덱스 범위(`fill_zero_start_idx`부터 `fill_zero_end_idx`까지)의 값을 `.zero_()` 메소드를 사용하여 0으로 만듭니다.

### `utils/spec_engine_D.py` (Problem D)

*   **목표:** 검증 단계 이후, 불필요해진 KV Cache를 `cutoff`하여 메모리를 관리하고 추론을 올바르게 수행합니다.
*   **`step_problem` 함수 내 "cutoff cache" TODO 구현:**
    *   검증 단계에서 얻은 `num_accepted_tokens`를 사용하여, Draft model과 Target model의 캐시에서 각각 몇 개의 토큰을 잘라낼지 계산해야 합니다.
    *   **`last_draft_num_cutoff` 계산:**
        *   Draft model이 추론한 `num_draft_tokens` 중, 채택되지 않고 버려지는 토큰의 수를 계산합니다. (수식: `max(0, self.num_draft_tokens - num_accepted_tokens - 1)`)
    *   **`last_target_num_cutoff` 계산:**
        *   Target model이 추론한 `num_draft_tokens` 중, 채택되지 않고 버려지는 토큰의 수를 계산합니다. (수식: `max(0, self.num_draft_tokens - num_accepted_tokens)`)
    *   계산된 값을 사용하여 `self.draft_model_runner.past_key_values.cutoff()`와 `self.model_runner.past_key_values.cutoff()`를 각각 호출하여 캐시를 잘라냅니다.

---

## 실행 방법

각 문제에 해당하는 솔루션 코드를 테스트하려면 `speculative` 디렉터리에 있는 쉘 스크립트를 실행합니다.

**주의:** `run_test.py`는 `--problem` 인자로 `A`, `B`, `D`와 같은 알파벳을 기대하도록 수정되었습니다. 하지만 쉘 스크립트들은 기본적으로 숫자 (`1`, `2`, `3`, `4`)를 사용하고 있어 **오류가 발생합니다.** 각 스크립트를 실행하기 전에 아래 설명에 따라 **반드시 수정**해야 합니다.

### Problem A 실행

`speculative/run_problemA.sh` 스크립트를 실행합니다.

'''bash
# ... (상략) ...
python run_test.py \
    # ... (중략) ...
    --problem A
'''

### Problem B 실행

`speculative/run_problemB.sh` 스크립트를 실행합니다.

'''bash
# ... (상략) ...
python run_test.py \
    # ... (중략) ...
    --problem B
'''

### Problem C 실행

**실행 불가:** Problem C는 `model_runner_C.py`에 `cutoff`라는 **기능을 구현**하는 단계입니다. 직접 실행하여 테스트하는 `spec_engine`이 없으므로 `run_problemC.sh`는 의도대로 동작하지 않습니다. Problem C에서 구현한 기능은 Problem D를 실행할 때 내부적으로 사용되고 테스트됩니다.

### Problem D 실행

`speculative/run_problemD.sh` 스크립트를 실행합니다.

**수정 필요:**
스크립트 마지막 줄의 `--problem 4`를 `--problem D`로 변경해야 합니다.
'''bash
# ... (상략) ...
python run_test.py \
    # ... (중략) ...
    --problem D
'''
