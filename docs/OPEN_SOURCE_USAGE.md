# FiTuna의 오픈소스SW 활용

> 심사·검토 시에는 [부문별 국문 요약](#국문-요약--부문별-오픈소스sw-활용)에서
> 전체 구성과 라이선스 의무를 먼저 확인할 수 있습니다. 아래 상세 근거에서는
> 프로젝트명, API 응답, 라이선스 표기와 검증 문구를 원문 그대로 보존합니다.

## 의존성 목록을 읽기 전에

FiTuna의 핵심 특징은 **런타임 의존성 0개**입니다(`pyproject.toml`:
`dependencies = []`). 이 문장만 보면 오픈소스SW를 전혀 사용하지 않는다고
오해하기 쉽습니다.

실제로는 많은 오픈소스SW를 활용합니다. FiTuna는 tensor 연산, 양자화, 추론,
perplexity 계산을 직접 하지 않습니다. 보고하는 모든 수치는 별도 process로 실행한
**llama.cpp**가 **GGUF** 파일과 **open-weight 모델**을 처리하고 **공개
라이선스 corpus**로 평가해 만든 결과입니다. Agent와는 **공개 protocol**로
연결하며 **공개 CI**에서 검사합니다. FiTuna가 더하는 가치는 탐색, 출력 해석,
cache, 그리고 무엇을 측정했는지 숨기지 않는 데 있습니다.

정확한 표현은 이렇습니다. **FiTuna는 아무것도 vendor 방식으로 포함하거나 link하지
않지만 많은 도구에 의존합니다.** "의존성 0개"는 전체 software stack이 아니라
Python import graph와 설치 wheel에 관한 설명입니다. 이 문서는 그 stack을
정리합니다.

아래 라이선스는 모두 **2026-07-30**에 1차 출처에서 확인했습니다. 각 프로젝트의
`LICENSE`·`License.txt` 원문 또는 모델·데이터셋에 대한 HuggingFace API의
`cardData.license` 값입니다. 항목마다 정확한 출처와 반환값을 적었으며 기억에
의존해 쓰지 않았습니다. 확인하지 못한 내용은 **미확인**이라고 명시합니다.

## 결합 방식이 중요한 이유

라이선스 영향은 사용 여부보다 코드를 *어떻게* 결합했는지에 따라 달라집니다.
FiTuna는 다섯 가지 결합 방식을 사용하며, 그중 하나만 이 저장소 배포물에
라이선스 조건을 전달할 수 있습니다.

| 방식 | 의미 | FiTuna에 미치는 영향 |
|---|---|---|
| **import** | FiTuna Python process 안에서 실행 | FiTuna 자체 배포물에 조건을 부과할 수 있는 유일한 방식 |
| **subprocess** | 별도 OS process에 argv를 전달하고 stdout 해석 | Link 없음, 배포 없음, 사용자가 직접 설치 |
| **파일 형식** | 공개 specification에 따라 data 읽기·쓰기 | Link 없음, 형식 준수는 파생 저작물이 아님 |
| **network / IPC protocol** | HTTP 또는 stdio로 통신 | Link 없음, FiTuna가 client/server를 직접 구현 |
| **개발 / CI 전용** | 설치 package에 들어가지 않음 | downstream 사용자에게 영향 없음 |

이 문서에서 **import**하는 항목은 CPython 표준 라이브러리 하나뿐입니다. 이것이
"런타임 의존성 0개"의 정확한 뜻입니다.

## 전체 목록

| # | 부문 | 오픈소스SW | 라이선스(확인함) | 결합 방식 | 저장소 내 위치 |
|---|---|---|---|---|---|
| 1 | 추론·양자화 엔진 | llama.cpp (`llama-quantize`, `llama-bench`, `llama-perplexity`, `convert_hf_to_gguf.py`) | MIT | 서브프로세스(`llama-cli`/`llama-server`는 결과 명령에만 표시하고 실행하지 않음) | `quantize.py`, `bench.py`, `quality.py`, `model_info.py`, `binaries.py`, `report.py` |
| 2 | 모델 파일 형식 | GGUF (ggml) | MIT(사양 저장소) | 파일 형식 | `model_info.py:202` `read_model_info()` |
| 3 | 모델 가중치 | SmolLM2-135M-Instruct | Apache-2.0 | 파일 형식(사용자 제공) | `docs/RESULTS.md` 실행 1·4, `notebooks/` |
| 4 | 모델 가중치 | Qwen3-4B-Instruct-2507 | Apache-2.0 | 파일 형식(사용자 제공) | `docs/RESULTS.md` 실행 2·3 |
| 4b | 모델 가중치 | Midm-2.0-Mini-Instruct (KT Corporation) | MIT | 파일 형식(사용자 제공) | `docs/RESULTS.md` 실행 5 |
| 5 | 평가 말뭉치(영어) | `Salesforce/wikitext`, `wikitext-2-raw-v1` | CC BY-SA 3.0 / GFDL | 내려받은 파일 | `corpus.py` `PRESETS["en"]` |
| 6 | 평가 말뭉치(한국어) | `wikimedia/wikipedia`, `20231101.ko` | CC BY-SA 3.0 / GFDL | 내려받은 파일 | `corpus.py` `PRESETS["ko"]` |
| 7 | 데이터 접근 API | HuggingFace dataset-viewer | Apache-2.0(서버 구현체) | 네트워크 프로토콜(HTTP GET) | `corpus.py:52` `API_BASE` |
| 8 | 에이전트 프로토콜 | Model Context Protocol `2024-11-05`, JSON-RPC 2.0 기반 | Apache-2.0 / MIT(전환 중) | IPC 프로토콜(stdio) | `mcp_server.py` |
| 9 | 언어 runtime | CPython 3.11+ 표준 라이브러리 | PSF License Agreement v2 | **import** | package 전체, `pyproject.toml` |
| 10 | 빌드 backend | setuptools ≥ 77 | MIT | 빌드 시점만 | `pyproject.toml` `[build-system]` |
| 11 | CI | `actions/checkout@v5`, `actions/setup-python@v6` | MIT | 개발·CI 전용 | `.github/workflows/ci.yml` |
| 12 | test framework | pytest | MIT | 개발 전용(`[dev]` extra) | `tests/`, `pyproject.toml` |
| 13 | 검증 환경 | Jupyter notebook 형식, git, CMake, pip, `huggingface_hub` | BSD-3-Clause, GPL-2.0, BSD-3-Clause, MIT, Apache-2.0 | 개발·notebook 전용 | `notebooks/colab_nvidia_verification.ipynb` |
| 14 | GPU/RAM 감지 | `rocm-smi` (ROCm) | MIT | 서브프로세스(선택) | `hardware.py:112` |

FiTuna가 호출하지만 재배포하지 않는 비오픈소스 구성요소인 `nvidia-smi`,
`system_profiler`(§14), CUDA toolkit(§13), hosted GitHub Actions(§11), Google
Colab(§13)도 빠짐없는 stack 설명을 위해 이 문서에 기록합니다. `sysctl`은 여기에
묶지 않습니다. 다른 도구처럼 호출만 하고 재배포하지 않지만, §14에서
BSD-3-Clause 오픈소스임을 확인했기 때문입니다.

---

## 1. 추론·양자화 엔진 — llama.cpp

**무엇인가.** 이 프로젝트가 조정하려는 C/C++ LLM inference stack입니다.
FiTuna는 그 기능을 직접 구현하지 않습니다.

**사용 위치.** llama.cpp 산출물 네 개를 각각 하나의 wrapper로 실행합니다. 앞의
세 개는 컴파일된 바이너리이고 네 번째는 `sys.executable`로 실행하는 Python
스크립트입니다.

| 바이너리 | 호출 위치 | 용도 |
|---|---|---|
| `llama-quantize` | `fituna/quantize.py:72` (`quantize()`) | F16/F32 기반 모델을 양자화 GGUF로 변환 |
| `llama-bench` | `fituna/bench.py:115` (`run_bench()`) | `-o json`으로 prompt·생성 처리량 측정 |
| `llama-perplexity` | `fituna/quality.py:57` (`compute_perplexity()`) | 품질 손실 gate용 perplexity 측정 |
| `convert_hf_to_gguf.py` | `fituna/model_info.py:167` (`ensure_base_gguf()`) | HF 형식 디렉터리를 F16 기반 GGUF로 변환 |

세 파일은 *위치만 찾고 실행하지 않습니다.* `llama-cli`와 `llama-server`는 결과의
`3) terminal chat`, `1) local API server` 줄에 실제 경로를 넣기 위해
`fituna/report.py:33`의 `_find_beside_binaries()`가 `_find_llama_cli()`와
`_find_llama_server()`를 거쳐 찾습니다. `llama-cli`는 `fituna doctor`의 선택
점검 항목(`doctor.py:313`)이기도 합니다. `llama-imatrix`는
`fituna/binaries.py:100`에서 찾아 `fituna list-binaries`(`cli.py:268`)가
출력하지만 현재 실행하는 코드 경로는 없습니다. FiTuna가 `llama-imatrix`를
"사용한다"고 하면 관계를 과장하게 되므로 이 구분이 중요합니다.

탐색과 버전 처리는 `fituna/binaries.py`에 있습니다. `locate_binaries()`는
`shutil.which()`로 `PATH` 또는 `--llama-bin-dir`에서 필수 바이너리 세 개를
찾습니다. 없으면 모호하게 실패하지 않고 upstream 빌드 안내를 가리키는
`BinaryNotFoundError`를 발생시킵니다.

**라이선스와 확인 방법.** MIT입니다. 2026-07-30에
`https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE`를
확인했으며 첫 줄은 `MIT License`, `Copyright (c) 2023-2026 The ggml authors`
입니다. 저장소에 llama.cpp byte가 없지만 MIT 고지 보존 조건을 충족하도록
`THIRD_PARTY_NOTICES.md` §1에 전문을 실었습니다.

**결합 방식.** 모두 subprocess입니다. argv 목록으로 `subprocess.run()`을 호출하고
FiTuna는 stdout, stderr, 종료 코드만 읽습니다. llama.cpp header, source,
shared library, 바이너리를 대상으로 compile하거나 link하지 않으며 vendor 방식
포함, packaging, 재배포도 하지 않습니다. 사용자가 llama.cpp를 설치하면 FiTuna가
찾습니다. 프로젝트 자체 측정에 사용한 macOS Homebrew와 Colab notebook의 CUDA
source build도 같은 방식입니다. Link가 없으므로 FiTuna에 적용되는 MIT 조건은
고지 보존뿐이며 `THIRD_PARTY_NOTICES.md`가 이를 충족합니다.

**이 선택이 적절한 이유.** llama.cpp는 하나의 codebase로 Metal, CUDA, ROCm,
CPU를 지원하는 사실상의 portable local inference runtime입니다. 이를 직접
조정해야 FiTuna의 답을 사용자 컴퓨터에서 쓸 수 있습니다. Subprocess 경계에는
다음과 같은 의도한 결과가 있습니다.

- FiTuna는 **quant 목록을 hardcode하지 않습니다.**
  `list_supported_quant_types()`(`binaries.py:125`)가 실행 중
  `llama-quantize --help`를 해석하므로 quant 종류가 다른 빌드도 가정 없이
  처리합니다.
- FiTuna는 **upstream 변경을 견딥니다.** llama.cpp 릴리스마다 문구가 달라져
  `get_llama_cpp_version()`(`binaries.py:140`)은 banner 형식 세 가지를
  시도합니다. `"pp512"` 같은 label 형식도 바뀌었으므로 `bench.py:44`은
  `n_prompt`·`n_gen`으로 prompt와 생성 record를 구분합니다. `report.py:29`은
  `llama-cli`뿐 아니라 예전 upstream 바이너리 이름인 `main`도 허용합니다.

**검토한 대안.** Shell 호출 대신 ctypes 또는 compile extension으로 `libllama`에
binding하는 방식을 검토했습니다. 그러면 FiTuna 설치가 사용자의 llama.cpp ABI
버전에 의존하고 upstream API 변경마다 빌드가 깨지며 compile 산출물을 배포하거나
버전을 맞춰야 합니다. 우리가 보고할 수치를 만드는 도구가 바로 `llama-bench`라
subprocess 경계만으로 탐색에 필요한 모든 것을 얻을 수 있어 채택하지 않았습니다.

## 2. 모델 파일 형식 — GGUF

**무엇인가.** llama.cpp가 사용하는 단일 파일 모델 container 형식이며 ggml
저장소에 specification이 공개돼 있습니다.

**사용 위치.** `fituna/model_info.py:202`의 `read_model_info()`가 파일을 열고
표준 라이브러리 `struct`로 GGUF header를 직접 해석합니다. Magic, version,
tensor·KV 개수, metadata KV table(`general.architecture`,
`<arch>.block_count`, `general.file_type`), tensor 정보 구간을 읽고 tensor element
수를 합산해 실제 parameter 수를 구합니다. 여기서 얻은 `n_layers`는 `-ngl`
이진탐색의 상한입니다. `general.file_type`은 사용자가 이미 양자화된 파일을
"기준"으로 전달했을 때 경고하는 `is_already_quantized()`
(`model_info.py:270`)에 사용됩니다.

**라이선스와 확인 방법.** Specification 문서는
`https://github.com/ggml-org/ggml/blob/master/docs/gguf.md` (HTTP 200 on
2026-07-30)에 있습니다. 저장소 `LICENSE`를
`https://raw.githubusercontent.com/ggml-org/ggml/master/LICENSE`, reads
에서 확인했으며 `MIT License`와 `Copyright (c) 2023-2026 The ggml authors`가
적힌 llama.cpp와 같은 원문입니다.

**결합 방식.** 파일 형식만 따릅니다. ggml 코드를 import, link, 복사하지 않으며
`model_info.py`는 공개 specification에 맞춰 독립적으로 작성한 reader입니다.
문서화된 파일 형식을 읽는 것은 파생 저작물을 만들지 않습니다.

**이 선택이 적절한 이유.** `llama-quantize`가 만들고 `llama-bench`가 읽는 형식이
GGUF이므로 이 pipeline에는 다른 container 선택지가 없습니다.

**검토한 대안.** llama.cpp 바이너리에 metadata 출력을 맡기거나 PyPI `gguf`
package에 의존하는 방식을 검토했습니다. 첫 번째는 버전 사이에서 안정된 "metadata를
JSON으로 출력" 계약을 제공하는 llama.cpp 바이너리가 없다는
`model_info.py` docstring의 이유로 제외했습니다. 두 번째는 약 100줄의 `struct`
parser 때문에 의존성 0개 보장을 포기하기에는 규모가 작다고 판단했습니다. Parser는
사용자가 받은 GGUF를 신뢰하지 않는 입력으로 취급합니다. `_read_exact()`과
`_read_value()`가 메모리를 할당하기 전에 길이·개수 field를 실제 파일 크기와
대조해 제한합니다.

## 3~4. Open-weight 모델

FiTuna는 weight를 포함하지 않고 모델도 고정하지 않습니다. `--model`은 사용자가
이미 디스크에 보유한 파일입니다. 아래 세 모델은 프로젝트의 공개 측정에 사용했기
때문에 *활용한 오픈소스 산출물*로 기록하며 의존성은 아닙니다.

| 모델 | 사용 위치 | 라이선스 | 확인 방법 |
|---|---|---|---|
| SmolLM2-135M-Instruct | `docs/RESULTS.md` 실행 1·4, `notebooks/colab_nvidia_verification.ipynb` 5번 cell, `docs/DEMO_SCRIPT.md` 실시간 시연 | Apache-2.0 | `https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M-Instruct` → `cardData.license: apache-2.0`(2026-07-30). 실제로 받은 GGUF는 `bartowski/SmolLM2-135M-Instruct-GGUF`이며 같은 조회에서 자체 card도 `apache-2.0`으로 보고 |
| Qwen3-4B-Instruct-2507 | `docs/RESULTS.md` 실행 2·3 | Apache-2.0 | `https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507` → `cardData.license: apache-2.0`. 사용한 `unsloth/Qwen3-4B-Instruct-2507-GGUF`도 `apache-2.0`으로 보고 |
| Midm-2.0-Mini-Instruct (KT Corporation) | `docs/RESULTS.md` 실행 5 | MIT | `https://huggingface.co/api/models/K-intelligence/Midm-2.0-Mini-Instruct` → `cardData.license: mit`(2026-07-30 재확인). 실제로 받은 `mykor/Midm-2.0-Mini-Instruct-gguf`도 같은 조회에서 `cardData.license: mit`, `base_model: K-intelligence/Midm-2.0-Mini-Instruct`로 보고. 두 저장소 모두 동일한 MIT `LICENSE.txt` 원문("Copyright (c) 2025 KT Corporation")을 포함하고 gated 상태가 아님. `docs/RESULTS.md` 실행 5 참조 |

**결합 방식.** 사용자가 제공하는 파일 형식입니다. Weight는 llama.cpp subprocess가
읽고 FiTuna가 직접 읽지 않으며 commit, packaging, 재배포하지 않습니다. FiTuna는
학습, fine-tuning, distillation, merge를 하지 않고 `llama-quantize`가 llama.cpp
process 안에서 수치 정밀도만 바꿉니다. 모델별 공개 template은
`docs/AI_MODEL_USAGE.md`에 있습니다.

**세 모델을 고른 이유.** 앞의 두 모델은 같은 명령으로 의미 있는 크기 범위를
아우릅니다. 135M은 전체 cold 탐색이 약 76초라 `docs/DEMO_SCRIPT.md`처럼 실시간
녹화할 만큼 작습니다. 4B는 답이 뻔하지 않을 만큼 커서 Run 2에서 Q6_K의 실측
품질이 Q8_0보다 좋아 관례적인 순위가 뒤집혔습니다. 세 번째는 한국어 모델입니다.
Run 1~4가 주로 영어로 학습한 모델을 사용했으므로 Run 5는 Run 3의 영어·한국어
corpus 실험을 한국어 open-weight 모델에서 반복합니다. MIT, ungated이며 GGUF
재배포자도 MIT라 별도 약관에 동의하지 않고 재현할 수 있다는 라이선스 조건도
선정 이유입니다.

**제외한 대안과 라이선스 판단.** 중간 크기 실행의 초기 후보였던
Qwen2.5-3B-Instruct는 라이선스 때문에 제외했습니다.
`https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct`는
2026-07-30 기준 `license: other`, `license_name: qwen-research`를 반환합니다.
OSI 승인 라이선스가 아닌 연구용 custom license입니다. 공개 결과의 모든 모델을
permissive open license로 맞추고 심사자가 별도 약관 없이 재현할 수 있도록
Apache-2.0인 Qwen3-4B-Instruct-2507로 교체했습니다. 같은 이유로 재현 경로에는
gated 모델을 쓰지 않습니다. 비교하면 `meta-llama/Meta-Llama-3-8B-Instruct`는
별도 조건이 있는 community license인 `license: llama3`를 보고합니다.

## 5~6. 평가 corpus

**무엇인가.** `llama-perplexity`가 읽어 탐색의 품질 gate 수치를 만드는 plain-text
corpus입니다.

**사용 위치.** `fituna/corpus.py`의 `PRESETS`는 정확히 두 개를 정의합니다.

- `en` → `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test`,
  기본 1,000행
- `ko` → `wikimedia/wikipedia`, config `20231101.ko`, split `train`,
  기본 500행

`fituna fetch-corpus`가 사용자의 `--out` 경로에 텍스트를 쓰면 이후
`fituna/quality.py:57`이 그 경로를 `llama-perplexity -f`에 전달합니다.

**라이선스와 확인 방법.** 둘 다 **CC BY-SA 3.0이며 GFDL로도 이중
라이선스**됩니다. 저작자 표시와 동일조건변경허락 의무가 적용됩니다. 2026-07-30에
각 데이터셋 metadata를 조회했습니다.
`https://huggingface.co/api/datasets/Salesforce/wikitext` and
`https://huggingface.co/api/datasets/wikimedia/wikipedia` both return
두 API 모두 `cardData.license: ["cc-by-sa-3.0", "gfdl"]`를 반환합니다.
`corpus.py:56` 주석에도 같은 출처를 기록했고 각 preset의 `license_note`에 저작자
표시 문구가 있습니다.

**결합 방식.** 필요할 때 받는 파일 형식입니다. Corpus 텍스트는 저장소에 commit하지
않고 `.gitignore`가 `*.txt` 출력을 제외합니다. 사용자가 고른 경로로 다운로드하고
FiTuna가 재배포하지 않으므로 FiTuna 자체에 CC BY-SA 동일조건 의무가 발생하지는
않습니다. 다만 사용자는 재배포할 수 있으므로 `cli.py:321`의
`_cmd_fetch_corpus`는 **성공할 때마다 라이선스 고지와 출처 URL을 stdout에
출력합니다.** `--dataset/--config/--split`으로 preset을 바꾸면 확인하지 않은
다른 데이터셋에 CC BY-SA라고 잘못 주장하지 않도록 일반적인 "해당 데이터셋의
라이선스를 확인하라"는 문구를 출력합니다.

**이 프로젝트에서 특히 중요한 이유.** `docs/RESULTS.md` Run 3은 같은 양자화
파일을 두 corpus에서 측정했고 품질 손실이 두 배 넘게 달랐습니다(Q4_K_M 영어
1.73%, 한국어 0.77%). 1% 예산에서는 *corpus만으로* 가능 여부가 뒤집혔습니다.
품질 gate의 의미는 corpus에 달려 있으므로 숨은 상수가 아니라 문서화하고 공개
라이선스를 확인한 일급 입력으로 취급합니다.

**검토한 대안.** `corpus.py` 전 README에 있던 일반적인 `pip install datasets` +
snippet 방식을 검토했습니다. 텍스트 파일 하나를 받으려고 `datasets`가 수백 MB의
pyarrow·pandas를 끌어와 의존성 0개 보장과 충돌한다는 `corpus.py` docstring의
이유로 제외했습니다. 공개 REST API를 호출하는 표준 라이브러리 `urllib` 약
40줄로 대체했습니다.

## 7. 데이터 접근 API — HuggingFace dataset-viewer

**무엇인가.** 인증 없이 데이터셋 행을 JSON으로 제공하는 HuggingFace 공개 REST
service입니다.

**사용 위치.** `fituna/corpus.py:52`의 `API_BASE =
"https://datasets-server.huggingface.co/rows"`를 `_fetch_page()`가
`urllib.request.urlopen`으로 호출합니다. `fetch_corpus()`는 `offset`·`length`로
page를 넘기며 server는 `length`를 100으로 제한합니다. 쓰기 전 예상한 text
field가 실제 응답에 있는지 검사하고, 임시 파일 + `os.replace`로 atomic하게 써서
연결이 끊겨도 일부 corpus를 남기지 않습니다. 2026-07-30에 실제 API로
request·response 구조, 422 제한, "마지막 이후 offset은 빈 `rows`와 HTTP 200을
반환"하는 동작을 직접 확인해 모듈 docstring에 기록했습니다.

**라이선스와 확인 방법.** Service의 server 구현은 오픈소스이며
`https://api.github.com/repos/huggingface/dataset-viewer/license`가
2026-07-30에 SPDX `Apache-2.0`을 반환했습니다. **이 라이선스가 FiTuna의 service
사용을 규율하지는 않습니다.** 해당 코드를 import하지 않고 hosted service에 익명
HTTP GET만 보냅니다. 반환 byte에 실제 적용되는 의무는 §5~6의 *데이터셋* CC
BY-SA입니다. Service 자체 이용약관은 검토하지 않아 **미확인**이며, FiTuna는
문서화된 공개 endpoint를 기본 header·무인증으로만 사용합니다.

**결합 방식.** Network protocol이며 표준 라이브러리 HTTP client만 사용합니다.

**이 방식이 적절한 이유.** 의존성 0개를 유지하면서 빠른 시작의 마지막 수동
다운로드 단계를 없앱니다. 실패 시 retry로 멈추지 않고 `_MANUAL_FALLBACK`의 실행
가능한 안내를 냅니다. 제한된 network의 심사위원도 "직접 받거나
`--quality-corpus`에 UTF-8 text 파일을 지정하라"는 분명한 메시지를 받습니다.
프로젝트의 유일한 network 의존 지점이며 선택 사항입니다. `fituna run`은
network를 사용하지 않습니다.

## 8. 에이전트 프로토콜 — JSON-RPC 2.0 기반 Model Context Protocol

**무엇인가.** MCP는 AI agent에 도구를 노출하는 공개 protocol입니다. stdio
transport는 줄 단위 JSON-RPC 2.0을 사용합니다.

**사용 위치.** `fituna/mcp_server.py`는 표준 라이브러리만으로 작성한 305줄짜리
완전한 server입니다. `serve()`는 stdin/stdout에서 줄 단위 JSON-RPC loop를
실행합니다. `_handle()`은 `initialize`, `tools/list`, `tools/call`, `ping`을
구현하고, 알 수 없는 method에는 `-32601`, 잘못된 JSON에는 `-32700`을 반환합니다.
JSON-RPC 2.0에 따라 `id`가 없는 notification에는 아무것도 반환하지 않습니다.
`PROTOCOL_VERSION = "2024-11-05"`를 알리고 `fituna_detect_hardware`와
`fituna_recommend` 두 도구를 제공합니다. `pyproject.toml`의 진입점은
`fituna-mcp`입니다.

**라이선스와 확인 방법.** 사양 저장소의 `LICENSE`
(`https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/LICENSE`,
2026-07-30 확인)에 따르면 이 프로젝트는 **MIT에서 Apache-2.0으로 전환 중**입니다.
새 코드와 사양 기여는 Apache-2.0, 사양을 제외한 문서는 CC-BY-4.0이며, 저자가
재라이선스에 동의하지 않은 기여는 MIT로 남습니다. 따라서 MCP는 Apache-2.0/MIT
전환 중이라고 표현해야 하며, 단순히 MIT라고 하면 부정확합니다. FiTuna가 구현한
`2024-11-05` 판은 upstream에 존재합니다.
`https://modelcontextprotocol.io/specification/2024-11-05`는 2026-07-30에 HTTP
200을 반환했습니다. 다만 JSON Schema는 같은 형태의 `modelcontextprotocol.io`
URL에 있지 않습니다. 다음 두 주소는 2026-07-30 확인 당시 404였습니다.
`.../specification/2024-11-05/schema.json`과
`.../schema/2024-11-05/schema.json`. 실제 schema는 다음 주소에서 확인됩니다.
`https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/schema/2024-11-05/schema.json`
(같은 날 HTTP 200). 인용해야 할 주소는 앞의 두 주소가 아니라 이 주소입니다.
JSON-RPC 2.0 자체는 `https://www.jsonrpc.org/specification`에 공개된 사양이며,
원문의 저작권 표기는 "Copyright (C) 2007-2010 by the JSON-RPC Working Group"입니다.
이는 라이선스된 software가 아니라 사양 문서이므로 여기서 software 라이선스를
별도로 주장하지 않습니다.

**결합 방식.** IPC protocol입니다. FiTuna가 MCP로 통신할 뿐, MCP 구현체를
사용하지 않습니다. SDK·schema library·사양 저장소의 코드를 사용하지 않습니다.

**검토한 대안.** 공식 MCP Python SDK입니다. `mcp_server.py`의 module docstring과
`docs/ARCHITECTURE.md`에 기록했듯, runtime 의존성 0개 보장을 지키기 위해 채택하지
않았습니다. stdio transport인 줄 단위 JSON-RPC 2.0은 `json`과 `sys.stdin`만으로
완전히 처리할 수 있습니다. 새 protocol 판을 직접 구현해야 한다는 비용이 있지만,
`pip install fituna`가 여전히 package 하나만 설치한다는 이점이 있습니다. 이 구현은
추정에 의존하지 않고 protocol 수준에서 검사합니다. `python -m
fituna.mcp_server --selfcheck`가 `serve()`를 통해 실제 `initialize` → `tools/list` →
잘못된 `tools/call` → notification → 잘못된 JSON 순서를 실행해 응답을 검증하며,
CI가 push마다 실행합니다.

## 9. 언어 runtime — CPython 표준 라이브러리

**무엇인가.** FiTuna가 실제로 import하는 유일한 구성요소입니다.

**사용 위치.** package 전체입니다. `pyproject.toml`은
`requires-python = ">=3.11"`과 `dependencies = []`를 선언합니다. 영역별 핵심
module은 다음과 같습니다. 모든 llama.cpp 호출에는 `subprocess`, GGUF header
해석에는 `struct`, `cache.py`의 `--resume` 결과 cache에는 `sqlite3`,
`corpus.py`에는 `urllib`, llama-bench 출력·JSON-RPC·report에는 `json`, stdout
해석에는 `re`, `cli.py`에는 `argparse`, 모델 fingerprint에는 `hashlib`, binary
탐색에는 `shutil.which`를 사용합니다. `config.py`의 frozen dataclass interface
계약에는 `dataclasses`·`enum`·`typing`, Windows RAM 조회에는 `ctypes`, 그 밖에
`platform`·`os`·`pathlib`·`logging`을 사용합니다.

`corpus.py`, `quantize.py`, `binaries.py`, `cache.py`, `model_info.py`,
`report.py`에 걸친 원자적 write 후 rename에는 `tempfile`·`stat`·`io`를 사용합니다.
`sys`는 `convert_hf_to_gguf.py`를 실행할 `sys.executable`과 `mcp_server.py`의
stdin/stdout에 사용합니다. `search.py`의 경과 시간에는 `time`, `cache.py` 결과
시간에는 `datetime`, `doctor.py` 출력 형식에는 `textwrap`을 사용합니다.

`unittest.mock`은 정확히 한 self-check인 `bench.py:148`에서 세 가지 실패 상황을
검증하기 위해 `subprocess.run`을 바꿉니다. `subprocess`를 쓰는 다른 self-check는
대신 해당 module 자체 이름을 직접 monkeypatch합니다. `search.py:435-437`의
docstring도 그 이유를 "rather than pulling in unittest.mock/pytest"라고 적습니다.
17개 self-check module 중 `__init__`, `errors`, `config`, `corpus`, `doctor`,
`mcp_server` 여섯 개는 `subprocess`를 전혀 참조하지 않습니다.

이 목록에서 실제로 Python 3.11 전용인 module은 `tomllib` 하나입니다.
`fituna/__init__.py:12`의 version drift self-check가 `pyproject.toml`을 해석할 때
사용하며, 개발·CI 전용 명령 `python -m fituna.__init__`에서만 실행됩니다. 일반
`import fituna`나 CLI 사용에는 실행되지 않습니다. 설치본에는 항상
`pyproject.toml`이 없으므로 `_self_check()`도 비교를 마치기 전에 반환합니다
(`fituna/__init__.py:16-17`). 설치한 package의 runtime 경로는 `tomllib`을
import하지 않으므로 installer에 3.11 하한을 강제하는 직접 원인은 아닙니다.
`requires-python`은 선언된 제약이며, 두 단락 뒤 설명처럼 CI에서 검증합니다.

`fituna/*.py`의 최상위 import를 AST로 검사하면 정확히 29개 module입니다.
parse 시 처리되는 compiler directive인 `__future__`까지 세면 30개지만 runtime
import는 아닙니다. **이 문서의 이전 판과 달리 `csv`는 포함되지 않습니다.**
문자열 `"csv"`는 `nvidia-smi --format=csv,noheader,nounits` 인수
(`hardware.py:80`)와 함수명 `_parse_nvidia_csv`에만 등장하며, `fituna/` 아래 어떤
module도 `csv` package를 import하지 않습니다. `bench.py`는 llama-bench의
`-o json` 출력을, `quality.py`는 정규식 `_PPL_RE`(`quality.py:24`)로 일반 text를
해석합니다. 둘 다 CSV를 해석하지 않습니다. AST로 확인한 전체 목록은
`docs/SBOM.md` 2~27행이며, 1행은 interpreter 자체입니다.

**라이선스와 확인 방법.** PSF License Agreement Version 2(SPDX
`PSF-2.0`)입니다. 2026-07-30에
`https://raw.githubusercontent.com/python/cpython/main/LICENSE`를 확인했으며,
license section 제목은 `Python Software Foundation License Version 2`입니다.
허용적 라이선스로 FiTuna에 copyleft를 부과하지 않습니다.

**결합 방식.** 유일한 import 항목입니다. 선언된 하한은
`requires-python = ">=3.11"`이며 **상한은 선언하지 않았습니다.** CI는 push마다
3개 OS에서 하한 3.11과 현재 검사하는 최신 판 3.13을 검증합니다. 3.13은 현재까지
검사한 최신 판일 뿐, 프로젝트가 선언한 상한이 아닙니다.

**이 방식이 적절한 이유.** 의존성 0개는 그 자체를 위한 절제가 아닙니다. FiTuna
사용자는 이미 크고 취약한 local LLM toolchain을 다뤄야 하므로 tuner가 별도
dependency tree까지 더하면 설치 부담이 커집니다. 서브프로세스 orchestrator에
필요한 process 제어, SQLite, HTTP, binary struct 해석, JSON은 모두 표준
라이브러리에 있습니다.

**검토한 대안.** hardware 감지용 `psutil` 대신 Windows에서는 약 20줄짜리
`ctypes`의 `GlobalMemoryStatusEx` 호출, 다른 OS에서는 `/proc/meminfo` 또는
`sysctl`을 사용했습니다(`hardware.py:177`). NUMA별 정밀도가 필요해지면 `psutil`로
전환한다는 주석도 코드에 있습니다. corpus 내려받기에는 `requests` 대신
`urllib.request`를 사용했습니다.

## 10. 빌드 backend — setuptools

`pyproject.toml`의 `[build-system]`은 `requires = ["setuptools>=77"]`과
`build-backend = "setuptools.build_meta"`를 선언합니다. 라이선스는
`https://api.github.com/repos/pypa/setuptools/license` → SPDX `MIT`
(2026-07-30 확인)에 따른 MIT입니다. **결합 방식은 빌드 시점만**입니다. pip가
격리된 build environment에 준비하며, 설치된 package의 runtime 의존성이 아닙니다.
이 프로젝트는 console script 두 개를 가진 단순한 pure Python package이므로 현대
setuptools에 없는 기능이 필요하지 않습니다.

## 11. CI — GitHub Actions

**사용 위치.** `.github/workflows/ci.yml`은 `main` branch의 모든 push와 pull
request에서 실행됩니다. `fail-fast: false`로 3개 OS × 2개 Python 조합
(ubuntu/macos/windows-latest × 3.11/3.13)을 검사합니다. 먼저 `pytest -q`를
실행하고, 이어서 독립 self-check가 있는 모든 module을 실행합니다.

`fituna/` 아래 파일 18개 모두 `if __name__ == "__main__":` block이 있지만,
그중 17개만 assert 기반 검사를 실행합니다. CI는 정확히 그 17개를 호출합니다.
`python -m fituna.config`, `.cache`, `.search`, `.model_info`,
`.quantize`, `.quality`, `.bench`, `.hardware`, `.binaries`, `.report`,
`.corpus`, `.doctor`, `.quickstart`, `.cli`, `.mcp_server`, `.__init__`,
`.errors`. 따라서 각 module의 assert 기반 검사는 장식이 아니라 실제 CI
gate입니다. 18번째 파일 `fituna/__main__.py`는 `python -m fituna` 진입점으로
`cli.main()`만 호출하고 자체 self-check가 없습니다. 이 목록에서 빠진 것은 누락이
아니라 의도한 동작입니다.

**라이선스와 확인 방법.** 재사용하는 action 두 개는 모두 open source MIT입니다.
`https://api.github.com/repos/actions/checkout/license`와
`.../actions/setup-python/license`는 2026-07-30에 모두 SPDX `MIT`를 반환했습니다.
runner 자체(`actions/runner`)와 runner image(`actions/runner-images`)도 같은 조회에서
MIT였습니다. **호스팅 GitHub Actions service는 GitHub의 독점 제품**입니다.
FiTuna는 이를 이용할 뿐 재배포하지 않으며, 여기서 open source라고 주장하지
않습니다.

**결합 방식.** 개발·CI 전용이며 CI의 어떤 구성요소도 설치 package에 들어가지
않습니다.

**이 방식이 적절한 이유.** FiTuna에서 가장 오류가 나기 쉬운 부분은
cross-platform 동작입니다. `shutil.which` 의미, Windows의 `Path.replace`와
`rename` 차이, UTF-8이 아닌 locale에서의 subprocess text decoding을 각각 source의
방어 코드가 처리합니다. 3개 OS matrix가 이 주장을 실제로 검증합니다. 다만
`.github/workflows/ci.yml`에서 직접 확인되듯 GPU runner와 llama.cpp build step은
없습니다. CI는 FiTuna의 logic만 검사하며 실제 benchmark 실행은
`docs/RESULTS.md`와 Colab notebook에 기록합니다.

## 12. Test framework — pytest

**사용 위치.** `tests/test_cache.py`, `test_config.py`, `test_corpus.py`,
`test_doctor.py`, `test_hardware.py`, `test_search.py`에서 fixture,
`parametrize`, `raises`를 사용합니다. 다음과 같이 선언합니다.
`[project.optional-dependencies].dev = ["pytest"]`,
`[tool.pytest.ini_options] testpaths = ["tests"]`.

**라이선스와 확인 방법.** 다음 조회에서 확인한 MIT입니다.
`https://api.github.com/repos/pytest-dev/pytest/license` → SPDX `MIT`
(2026-07-30).

**결합 방식.** 개발 전용입니다. `pip install fituna`는 설치하지 않고
`pip install fituna[dev]`만 설치합니다. 따라서 배포 software의 의존성이 아니며
`docs/SBOM.md`에도 개발 전용으로 표시합니다.

**pytest와 표준 라이브러리 assert를 함께 쓰는 이유.** 두 계층은 중복이 아니라
의도한 구성입니다. 각 module에는 framework 없이 `python -m fituna.<module>`로
실행할 수 있는 assert 기반 `_self_check()` 또는 `demo()`가 있습니다.
`model_info.py`에도 그 이유를 "no test framework needed for a single module's
worth of parsing logic"이라고 적었습니다. 반면 pytest는 fixture와 parametrization이
실제로 유용한 cache key matrix, search algorithm 상황, hardware 해석 변형 등
여러 module에 걸친 suite를 담당합니다. CI는 둘 다 실행합니다. 표준 라이브러리
계층 덕분에 심사위원은 아무것도 더 설치하지 않고 개별 module을 검증할 수 있습니다.

## 13. 검증 환경 — Colab notebook

`notebooks/colab_nvidia_verification.ipynb`는 maintainer가 보유하지 않은 NVIDIA
T4(Linux/CUDA)에서 측정 탐색을 재현합니다. 기록된 출력은 `docs/RESULTS.md` 실행
4입니다. 이 경로에서 사용하는 open source 구성요소는 다음과 같습니다.

| 구성요소 | notebook에서의 역할 | 라이선스 | 확인 방법 |
|---|---|---|---|
| Jupyter notebook 형식(`nbformat`) | `.ipynb` 파일 자체 | BSD-3-Clause | `https://api.github.com/repos/jupyter/nbformat/license` → SPDX `BSD-3-Clause` (2026-07-30) |
| `git` | 2번 cell이 llama.cpp를 clone(`git clone --depth 1`)하고, 3번 cell의 `pip install git+https://...`도 이 저장소를 clone할 때 호출 | GPL-2.0 | `https://raw.githubusercontent.com/git/git/master/COPYING` 원문의 "GNU GENERAL PUBLIC LICENSE Version 2, June 1991"(2026-07-30). GitHub license 감지 API가 `git/git`에 `NOASSERTION`을 반환해 API를 신뢰하는 대신 원문을 직접 확인 |
| CMake | 2번 cell이 `-DGGML_CUDA=ON`으로 llama.cpp build | BSD-3-Clause | `https://api.github.com/repos/Kitware/CMake/license` → SPDX `BSD-3-Clause` (2026-07-30) |
| `pip` | 3·5번 cell이 Colab runtime에 FiTuna와 `huggingface_hub` 설치 | MIT | `https://api.github.com/repos/pypa/pip/license` → SPDX `MIT` (2026-07-30) |
| `huggingface_hub` | 5번 cell이 시연용 GGUF 내려받기 | Apache-2.0 | `https://api.github.com/repos/huggingface/huggingface_hub/license` → SPDX `Apache-2.0` (2026-07-30) |
| llama.cpp | 2번 cell이 source를 clone해 build | MIT | §1 |

**결합 방식.** notebook·개발 전용입니다. `huggingface_hub`는 파일을 받기 위해
*notebook 안에서* `pip install`하며 FiTuna 의존성이 아니고 `pyproject.toml`에도
없습니다. Google Colab은 실행 환경으로 사용하는 Google의 독점 호스팅 service입니다.
llama.cpp CUDA backend를 빌드하는 CUDA toolkit은 Colab image가 제공하는 NVIDIA
독점 software이며 재배포하지 않습니다. notebook은 tag를 고정하지 않고
`git clone --depth 1`로 llama.cpp를 clone하므로 실행 4의 llama.cpp 판은 주장하지
않습니다. 반면 `docs/RESULTS.md`의 macOS 실행은 기록이 있으므로 build(Homebrew
9960)를 명시합니다.

## 14. 하드웨어 감지 도구

`fituna/hardware.py:detect_hardware()`는 설치되어 있는 도구를 서브프로세스로
실행하고, 아무것도 없으면 `platform`을 통해 CPU 전용 profile로 fallback합니다.
이 중 `rocm-smi`와 `sysctl`은 open source이므로 다른 도구와 모호하게 묶지 않고
구분합니다.

| 도구 | 호출 위치 | 출처 | 라이선스 | 결합 방식 |
|---|---|---|---|---|
| `rocm-smi` | `hardware.py:112` | ROCm (`ROCm/rocm_smi_lib`, `python_smi_tools/rocm_smi.py`) | **MIT** | 서브프로세스, 선택 |
| `nvidia-smi` | `hardware.py:79` | NVIDIA driver package | NVIDIA 독점 | 서브프로세스, 선택 |
| `system_profiler` | `hardware.py:156` | macOS | Apple 독점 | 서브프로세스, 선택 |
| `sysctl` | `hardware.py:188` | macOS/BSD base system(`apple-oss-distributions/system_cmds`) | **BSD-3-Clause** — 아래 근거 참조 | 서브프로세스, 선택 |
| `/proc/meminfo` | `hardware.py:181` | Linux kernel interface | 해당 없음(kernel 제공 가상 파일) | 파일 읽기 |
| `GlobalMemoryStatusEx` | `hardware.py:215` | Windows kernel32, `ctypes` 경유 | Microsoft 독점 | OS API 호출 |

**`rocm-smi` 라이선스 확인.** 저장소의 현행 기본 branch에 있는
`https://raw.githubusercontent.com/ROCm/rocm_smi_lib/amd-staging_deprecated/License.txt`는
2026-07-30 확인 당시 `MIT License` / `Copyright (c) 2023-2025, Advanced Micro
Devices, Inc.`라고 적혀 있습니다. 정확한 기록이 필요한 이유는 이전 `master`
branch에는 여전히 **University of Illinois/NCSA Open Source License**가 있어 다른
branch를 확인하면 답이 달라지기 때문입니다. 현재 ROCm 설치에 적용되는 것은 현행
branch의 MIT text입니다.

**`sysctl` 라이선스 확인.** macOS에서 호출하는 `sysctl` binary는 open source
`apple-oss-distributions/system_cmds` package로 build한 Apple base system
구성요소입니다. 주 source인
`https://raw.githubusercontent.com/apple-oss-distributions/system_cmds/main/sysctl/sysctl.c`
(2026-07-30 HTTP 200)의 header에는 `SPDX-License-Identifier: BSD-3-Clause`와
`Copyright (c) 1993 The Regents of the University of California. All rights
reserved.`라는 전형적인 3-clause BSD text가 있습니다. `system_profiler`처럼 호출만
하고 재배포하지 않는 기본 OS utility입니다. 차이는 이 binary의 upstream 자체가
open source인 반면 `system_profiler`는 아니라는 점입니다.

이 도구들은 어느 것도 bundle하지 않으며 모두 선택 사항입니다. 모든 감지에
실패해도 `detect_hardware()`는 CPU 전용 `HardwareProfile`을 반환하고 탐색을
계속합니다. 이때 `docs/ARCHITECTURE.md` 설명대로 `-ngl` binary search는
건너뜁니다.

---

## FiTuna가 실제로 지는 라이선스 의무

vendor 방식으로 포함한 항목이 없고 PSF 라이선스의 표준 라이브러리만 import하므로,
라이선스 의무가 적으며 저장소 안에서 모두 이행합니다.

| 의무 | 출처 | 준수 방법 |
|---|---|---|
| MIT 고지 보존 | llama.cpp, ggml | `THIRD_PARTY_NOTICES.md` §1에 MIT 전문 수록 |
| 저작자 표시 + 동일조건 | CC BY-SA 3.0 말뭉치 | `fituna fetch-corpus`가 매번 고지와 출처 URL 출력(`cli.py:321`), 말뭉치 text는 commit하지 않음 |
| 모델 라이선스 준수 | 사용자가 선택한 weight | Weight를 재배포하지 않음. `docs/AI_MODEL_USAGE.md`에 공개 template 제공. 공개 결과의 모델은 Apache-2.0 또는 MIT |
| FiTuna 자체 조건 | MIT (`LICENSE`) | 허용적 라이선스로 사용자에게 추가 의무를 부과하지 않음 |

FiTuna에 link한 copyleft software는 없습니다. CC BY-SA 동일조건은 말뭉치
*text*에 적용되며, FiTuna는 이를 사용자가 선택한 경로로 전달할 뿐 재게시하지
않습니다.

## Upstream에 환원하는 것

2026-07-30 기준이며 과장 없이 기록합니다.

**현재 공개한 것.**

- **llama.cpp toolchain의 공개 cross-platform 측정값.** 한곳에서 보기 어려운 결과를
  `docs/RESULTS.md`의 실행 5개에 기록했습니다. 한국어 open-weight 모델(실행 5),
  *동일한* 양자화 파일과 *동일한* 말뭉치가 Metal에서 품질 손실 4.74%, CUDA에서
  5.22%를 보여 5% gate의 통과 여부가 바뀐 사례, 같은 장비에서 더 작은 quant가 더
  큰 quant보다 *느린* 사례를 포함합니다. 각 수치에는 명령, hardware, 기록이 있는
  경우 llama.cpp build를 명시했습니다.
- **누구나 무료로 빌릴 수 있는 hardware에서 한 번에 재현하는 경로.** 실제 cell
  출력이 담긴 `notebooks/colab_nvidia_verification.ipynb`가 CUDA로 llama.cpp를
  build하고 Colab T4에서 공개 실험을 다시 실행합니다.
- **품질 gate가 언어에 따라 달라진다는 실측 자료.** `docs/RESULTS.md` 실행 3에서
  동일 파일의 영어·한국어 perplexity 말뭉치 결과가 2배 넘게 다르고, 말뭉치만으로
  1% budget에서 가능 여부가 바뀝니다.
- **도구 자체.** MIT 라이선스로 공개해 누구나 llama.cpp 양자화를 benchmark할 수
  있습니다.
- **숨기지 않고 공개한 실행 간 분산.** thermal throttle outlier와 이를 outlier로
  확인한 직접 `llama-bench` 반복 실행까지 기록했습니다.

**아직 없으며 있다고 주장하지 않는 것.** 이 프로젝트의 patch, pull request,
issue 가운데 llama.cpp, ggml, Model Context Protocol, 데이터셋 또는 모델 저장소에
병합된 것은 없습니다. FiTuna는 이 프로젝트들을 소비하면서 측정값을 공개하지만,
현재 이들 코드의 contributor는 아닙니다. 정확한 요약은 **실측 data와 재현 가능한
notebook을 공개했으며, 아직 upstream에 병합된 것은 없다**입니다.

## 관련 문서

- `docs/LICENSE_COMPLIANCE.md` — 실제 sdist·wheel 구성, 라이선스 비충돌 근거,
  실행한 scan의 실제 출력과 재현 명령
- `docs/SBOM.md` — 번호를 붙인 SBOM(표준 라이브러리 모듈 + 외부 실행 파일)
- `THIRD_PARTY_NOTICES.md` — llama.cpp MIT 전문을 포함한 필수 라이선스 고지
- `docs/AI_MODEL_USAGE.md` — 모델별 AI 활용 공개
- `docs/ARCHITECTURE.md` — subprocess 경계와 설계 이유
- `LICENSE` — FiTuna 자체 MIT license

---

## 국문 요약 — 부문별 오픈소스SW 활용

FiTuna는 **런타임 의존성이 0개**이지만, 이는 "타 오픈소스SW를 쓰지 않는다"는
뜻이 아니다. **어떤 코드도 저장소에 포함(vendoring)하거나 프로세스에 링크하지
않을 뿐**, 양자화·추론·품질측정 연산 전부를 llama.cpp에 위임하고, GGUF 포맷과
공개 가중치·공개 데이터셋·공개 프로토콜 위에서 동작한다. 아래 표의 **결합
방식**이 라이선스 의무를 결정하는 핵심 항목이다.

| 부문 | 활용 오픈소스SW | 라이선스(확인함) | 결합 방식 | 저장소 내 사용 위치 |
|---|---|---|---|---|
| 추론·양자화 엔진 | llama.cpp (`llama-quantize`/`llama-bench`/`llama-perplexity`/`convert_hf_to_gguf.py`) | MIT | **서브프로세스** (링크·포함 없음) | `quantize.py`, `bench.py`, `quality.py`, `model_info.py`, `binaries.py`, `report.py` |
| 모델 파일 포맷 | GGUF (ggml 사양) | MIT | **파일 포맷 준수** (`struct`로 직접 파싱) | `model_info.py:202` |
| 모델 가중치 | SmolLM2-135M-Instruct, Qwen3-4B-Instruct-2507, Midm-2.0-Mini-Instruct(KT) | 앞의 둘 Apache-2.0, Midm은 MIT | **사용자 제공 파일** (재배포 없음) | `docs/RESULTS.md`, `notebooks/` |
| 평가 데이터셋 | `Salesforce/wikitext`(영), `wikimedia/wikipedia` `20231101.ko`(한) | CC BY-SA 3.0 / GFDL | **내려받아 파일로 사용** (저장소 미포함) | `corpus.py` `PRESETS` |
| 데이터 조회 API | HuggingFace dataset-viewer | 서버 구현체 Apache-2.0 (FiTuna는 HTTP 호출만) | **네트워크 프로토콜** | `corpus.py:52` |
| 에이전트 연동 프로토콜 | Model Context Protocol(`2024-11-05`) / JSON-RPC 2.0 | Apache-2.0·MIT 전환 중 | **프로토콜 자체 구현** (SDK 미사용) | `mcp_server.py` |
| 언어·런타임 | CPython 3.11+ 표준 라이브러리 | PSF License Agreement v2 | **임포트** (유일한 링크 항목) | 패키지 전체, `pyproject.toml` |
| 빌드 백엔드 | setuptools ≥ 77 | MIT | 빌드 시점 한정 | `pyproject.toml` |
| CI | `actions/checkout@v5`, `actions/setup-python@v6` | 둘 다 MIT | 개발·CI 한정 | `.github/workflows/ci.yml` |
| 테스트 | pytest | MIT | 개발 한정(`[dev]` 옵션) | `tests/`, `pyproject.toml` |
| 검증 환경 | Jupyter 노트북 포맷, git, CMake, pip, `huggingface_hub` | BSD-3-Clause, GPL-2.0, BSD-3-Clause, MIT, Apache-2.0 | 노트북 한정 | `notebooks/colab_nvidia_verification.ipynb` |
| GPU 감지 | `rocm-smi` (ROCm) | MIT (현행 기본 브랜치 `License.txt`) | 서브프로세스(선택) | `hardware.py:112` |

**라이선스 확인 방법.** 위 라이선스는 모두 2026-07-30에 1차 출처에서 직접
확인했다 — 각 프로젝트의 `LICENSE`/`License.txt` 원문, 또는 모델·데이터셋의
경우 HuggingFace API의 `cardData.license` 값이다. macOS `sysctl`도
`apple-oss-distributions/system_cmds`의 `sysctl/sysctl.c` 헤더
(`SPDX-License-Identifier: BSD-3-Clause`)로 직접 확인해 §14에 반영했다 —
더 이상 미확인이 아니다. 끝내 확인하지 못했거나 의도적으로 특정하지 않은
항목은 두 가지다: HuggingFace dataset-viewer **서비스 자체**의 이용약관
(§7 — 서버 코드의 Apache-2.0 라이선스와는 별개이며, 검토하지 않았다)과
Colab Run 4의 llama.cpp 정확한 빌드 버전(§13 — 태그를 고정하지 않고 클론해
특정 버전을 주장하지 않는다).

**오픈소스가 아닌 연동 대상.** `nvidia-smi`(NVIDIA 독점),
`system_profiler`(Apple 독점), CUDA 툴킷(NVIDIA 독점), GitHub Actions·Google
Colab(각 사 호스팅 서비스)도 호출·이용하지만, 모두 재배포하지 않으며 오픈소스로
표기하지 않는다.

**FiTuna가 지는 라이선스 의무.** 벤더링·링크가 없으므로 의무는 네 가지로
정리된다: (1) llama.cpp MIT 고지 보존 — `THIRD_PARTY_NOTICES.md`에 전문
수록, (2) 코퍼스 CC BY-SA 저작자표시·동일조건 — `fituna fetch-corpus` 실행
시마다 출처·라이선스 고지를 표준출력에 인쇄하고 코퍼스 원문은 저장소에 넣지
않음, (3) 사용자가 선택한 모델의 라이선스 준수 — 가중치를 재배포하지 않고
`docs/AI_MODEL_USAGE.md`에 고지 템플릿을 제공하며 시연에 사용한 모델은
Apache-2.0 또는 MIT, (4) FiTuna 자체 라이선스(MIT) 준수 — 사용자에게 아무 의무도
부과하지 않는 permissive 라이선스. 링크된 카피레프트 소프트웨어는 없다 —
§13의 `git`(GPL-2.0)은 검증 노트북 안에서만 호출되는 개발/노트북 한정
도구이며 FiTuna 배포물에는 링크되지 않는다.

**상류 기여 현황(과장 없이).** 현재까지 llama.cpp·ggml·MCP·데이터셋 저장소에
**병합된 기여는 없다**. 대신 공개한 것은 실측 데이터
(`docs/RESULTS.md` — 동일 파일·동일 코퍼스인데 Metal 4.74 % vs CUDA 5.22 %로
품질 게이트 판정이 뒤집힌 사례 포함)와 무료 Colab T4에서 그대로 재현되는
노트북, 그리고 MIT로 공개한 도구 자체다.
