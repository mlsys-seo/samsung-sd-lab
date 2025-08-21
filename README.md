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
3.  **Cache manipulation 구현 (Problem C):** `model_runner_C.py`의 ...
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

*   **목표:** Acceptance에 따른 예외처리를 구현합니다. 
*   **`__init__` 함수 내 TODO 구현:** `self.all_accepted`와 `self.last_bonus_token_id_buffer` 변수를 초기화합니다.
*   **`step_problem` 함수 내 TODO 구현:**
    1.  **Draft Input 준비:** 이전 스텝에서 모든 토큰이 채택되었다면(`self.all_accepted`), `last_bonus_token_id_buffer`의 토큰을 다음 Draft 생성의 시작으로 사용합니다.
    2.  **Bonus Token 처리:** 검증 후, 모든 후보 토큰이 채택되었는지 여부를 `self.all_accepted`에 기록하고, 채택되었다면 마지막 토큰을 `last_bonus_token_id_buffer`에 저장합니다.

### `utils/model_runner_C.py` (Problem C)

...

### `utils/spec_engine_D.py` (Problem D)

*   **목표:** 검증 단계 이후, 불필요해진 KV Cache를 `cutoff`하여 메모리를 관리하고 추론을 올바르게 수행합니다.
*   **`step_problem` 함수 내 "cutoff cache" TODO 구현:**
    *   검증 단계에서 얻은 `num_accepted_tokens`를 사용하여, Draft model과 Target model의 캐시에서 각각 몇 개의 토큰을 잘라낼지 계산해야 합니다.
    *   **`last_draft_num_cutoff` 계산:**
        *   Draft model이 추론한 `num_draft_tokens` 중, 채택되지 않고 버려지는 토큰의 수를 계산합니다. (hint: `max(0, self.num_draft_tokens - num_accepted_tokens - 1)`)
    *   **`last_target_num_cutoff` 계산:**
        *   Target model이 추론한 `num_draft_tokens` 중, 채택되지 않고 버려지는 토큰의 수를 계산합니다. (hint: `max(0, self.num_draft_tokens - num_accepted_tokens)`)
    *   계산된 값을 사용하여 `self.draft_model_runner.past_key_values.cutoff()`와 `self.model_runner.past_key_values.cutoff()`를 각각 호출하여 캐시를 잘라냅니다.