# FiTuna 심사·검증용 재현 가이드

이 문서는 **대회 심사위원과 검증 전문기관**이 FiTuna를 직접 구동해 정상 동작을
확인하기 위한 안내서다. FiTuna를 처음 보는 사람이 위에서부터 순서대로 따라오면
추가 조사 없이 성공적인 실행에 도달하도록 작성했다. (이 문서와 docs/ 아래 기술 문서는 한국어가 기본이다. README는 해외 기여자를 위해
영어가 기본이며, 한국어판은 [README.ko.md](README.ko.md)로 제공한다.)

이 문서의 모든 명령·출력·소요시간은 저장소의
[`docs/RESULTS.md`](docs/RESULTS.md)에 이미 실측으로 기록된 값이거나, 이 문서를
작성하면서 **직접 실행해 관찰한 값**이다. 각 수치의 출처는 그 자리에 명시했다.
막히는 것이 있으면 `fituna help`가 언제나 한국어·영어 병기로 다음 할 일을
알려준다.

> **먼저 이것부터 — 실측 결과를 90초 안에 본다.** llama.cpp만 설치돼 있다면
> (4-1절), 아래 두 줄이 전부다. HuggingFace에서 모델을 자동으로 받아 곧바로
> 탐색한다 — README의 "1분 데모"와 동일한, v0.2.0 `--hf` 플래그다:
>
> ```bash
> fituna fetch-corpus --lang en --out wiki.txt
> fituna run --hf bartowski/SmolLM2-135M-Instruct-GGUF \
>   --target-tps 240 --max-quality-loss 5 --ctx 2048 \
>   --quality-corpus wiki.txt --out ./out --resume
> ```
>
> 이 문서를 작성하며 직접 실행한 결과(2026-08-25, Apple M3 Pro, 콜드): 다운로드
> 포함 **82.57초**만에 `FiTuna result: MEETS TARGET` — Q8_0 @ ngl=24, 247.23
> tok/s, 손실 0.29% — 와 함께 `llama-server`/Ollama/`llama-cli` 세 가지 사용법이
> 출력된다. 같은 명령을 그대로 한 번 더 실행하면 캐시에서 **1.10초**에 동일
> 결과가 나온다(7장). 전체 출력·단계별 설명·대안 경로는 4장에 이어진다.

---

## 1. 3분 요약 — 무엇을 넣으면 무엇이 나오는가

FiTuna는 로컬 LLM(llama.cpp)을 돌릴 때 **어떤 양자화 레벨(quant)과 GPU 오프로드
레이어 수(`-ngl`), 컨텍스트 길이(ctx) 조합을 써야 하는지**를 사양서 추정이 아니라
**그 기계에서 실제로 재서** 찾아 주는 CLI 도구다.

| | |
|---|---|
| **입력** | F16 GGUF 모델 파일(로컬 경로 또는 `--hf`로 HuggingFace 저장소 지정), 목표 생성속도(tok/s), 허용 품질저하(%), 컨텍스트 길이, 품질 측정용 텍스트 코퍼스 |
| **처리** | llama.cpp 바이너리(`llama-quantize` → `llama-perplexity` → `llama-bench`)를 직접 호출해 후보를 **실제로 양자화하고, 실제로 perplexity를 재고, 실제로 벤치마크**한다 |
| **출력** | 목표를 만족하는 구성(목표 속도를 내는 quant 중 품질이 가장 높은 것 × 최소 `-ngl` × ctx) + 실측 tok/s + 실측 품질손실 + **이미 만들어진 gguf 산출물(artifact)** 과 그걸 바로 쓰는 세 가지 방법 — 로컬 API 서버(`llama-server`) / Ollama(`--export-ollama`) / 대화형 확인용 `llama-cli` 커맨드 |

**무엇을 확인하면 "정상 동작"인가.** `fituna doctor`가 11개 점검 항목을 출력하고
실패 0건이면 환경 준비가 끝난 것이고, `fituna run`이 단계별 진행 로그(양자화 →
품질 평가 → 벤치)를 흘린 뒤 마지막에 `FiTuna result:` 블록과 산출물(artifact)
블록 — 생성된 gguf 경로·크기, 그리고 그걸 바로 쓰는 세 가지 방법(로컬 API 서버 /
Ollama / 대화형 `llama-cli`) — 을 출력하면 정상 동작이다. **종료 코드 0(목표
달성)과 종료 코드 3(목표 미달 + 최선 구성 보고)은 둘 다 정상 동작이며, 3은
버그가 아니다.** 이 구분은 5장에서 자세히 설명한다 — 검증 전에 5장을 먼저
읽어도 좋다.

**심사 관점별로 바로 갈 곳.** 무엇을 검증하고 싶은지에 따라 아래 표에서 바로
찾아가도 된다 — 순서대로 읽지 않아도 된다.

| 확인하고 싶은 것 | 보면 되는 곳 |
|---|---|
| 정말 설치·실행이 되는가 (설치 → 실행 → 결과) | 4장, 또는 위 "먼저 이것부터" |
| 목표 미달일 때 수치를 지어내지 않고 정직하게 보고하는가 | 5-2절 (종료 코드 3, best-effort) |
| 크래시·traceback을 사용자에게 그대로 노출하지 않는가 | 5-3절 |
| 다른 하드웨어(NVIDIA/Linux)에서도 동작하는가 | 3장 (경로 A, Tesla T4, 브라우저만) |
| 재실행 시 캐시가 실제로 재계산을 건너뛰는가 | 7장 |
| 테스트가 실제로 통과하는가 | 5-4절 (`pytest` 322개, 직접 실행 가능) |
| AI 에이전트 연동(MCP)이 실제로 응답하는가 | 4-8절 |
| 알려진 한계를 숨기지 않고 고지하는가 | 6장, README "범위와 한계" |

> **버전 참고.** 이 문서의 본문 명령은 2026-08-25에 PyPI 0.2.0을 새 가상환경에
> 설치해 `--hf`(HuggingFace 자동 다운로드, v0.2.0 신규 기능)로 재검증했다. 이미
> 받아 둔 GGUF 파일로 재현하는 예전 방식(`--model` + 수동 `curl`)과 그 경로로
> 측정된 2026-07-30 실행 기록(결과 블록이 `run command:` 한 줄로 끝나던 0.2
> 이전 형식 포함) — 은 각 절 아래 접이식 상자에 **손대지 않고 그대로** 남겨
> 뒀다. 어느 경로로 재현하든 결과는 동일하다.

---

## 2. 두 경로 중 선택

| | **경로 A · 브라우저만** | **경로 B · 로컬 실행** |
|---|---|---|
| 필요한 것 | 웹 브라우저 + 구글 계정 | macOS 또는 Linux, Python 3.11+, 그리고 llama.cpp 설치 수단 — Homebrew(`brew install llama.cpp`) 또는 소스 빌드용 `git` + `cmake`(4-1 참고) |
| 설치 | 없음 (Colab 무료 T4 GPU 사용) | llama.cpp + FiTuna 설치 필요 |
| 소요 시간 | 약 20~30분 (대부분 llama.cpp CUDA 빌드) | 약 2분 (llama.cpp 설치 시간 제외) |
| 검증 대상 하드웨어 | NVIDIA Tesla T4 / Linux / CUDA | 검증자의 실제 기기 |

경로 B의 "약 2분"은 4장의 실측 단계(코퍼스 내려받기 4-4 + 탐색 실행
4-6 — 모델 다운로드 포함 + 재현 확인 4-7)를 합산한 값이다. 설치(4-2)·점검(4-3)은
수 초 내로 끝나 반올림에 흡수된다.

설치를 전혀 하지 않고 확인하려면 **경로 A**, 로컬에 llama.cpp를 설치할 수 있는
환경이면 **경로 B**가 더 빠르다. 둘 중 하나만 수행해도 기능 확인에는 충분하다.

---

## 3. 경로 A · 브라우저만으로 (Google Colab, NVIDIA T4)

노트북: [`notebooks/colab_nvidia_verification.ipynb`](notebooks/colab_nvidia_verification.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leeyunseokarchive/fituna/blob/main/notebooks/colab_nvidia_verification.ipynb)

위 배지를 클릭하면 Colab에서 노트북이 열린다. **실행 전 반드시** 상단 메뉴
`런타임 → 런타임 유형 변경 → T4 GPU`(무료 등급으로 가능)를 설정한다. 그 다음
셀을 위에서부터 순서대로 실행한다. 노트북 안내문·주석은 전부 한국어이며,
"셀 6이 `BEST EFFORT`로 나오는 것이 정상"이라는 설명도 노트북 맨 위에
포함돼 있다 — 이 문서 없이 노트북만 열어도 판정 기준을 놓치지 않는다.

| 셀 | 하는 일 | 기대 출력 | 소요 |
|---|---|---|---|
| 1 | `nvidia-smi` | `Tesla T4` / `15360MiB` 표. 노트북 파일에 실제 실행 출력이 저장돼 있어 미리 확인 가능 | 수 초 |
| 2 | llama.cpp를 CUDA로 빌드 | 마지막 줄에 `llama.cpp ready` | **10~20분** (가장 긴 셀) |
| 3 | FiTuna 설치(PyPI) + `fituna --help` | 두 줄로 감기는 `usage: fituna [-h] [-v]` / `{run,detect-hw,list-binaries,doctor,fetch-corpus,quickstart,help} ...` | 수 초 |
| 4 | `fituna detect-hw` | `gpu_vendor: nvidia` / `gpu_name: Tesla T4` / `vram_mb: 15360` / `os_name: linux` | 수 초 |
| 5 | 품질 측정용 코퍼스 다운로드 | `Wrote 1000 rows ...` + 라이선스 고지 | 수 초~수십 초 |
| 6 | **측정 탐색 본체** (`fituna run --hf`) — 데모 모델(SmolLM2-135M F16, Apache 2.0, 258 MB)도 이 셀이 자동 다운로드 | 아래 설명 참고 — T4에서는 `BEST EFFORT` | 약 1~2분 |
| 7 | 동일 명령 재실행(`--resume`) | 6번과 **동일한** 결과 블록 | 1초대 |

셀 3은 매번 PyPI 게시본을 `pip install fituna`로 설치한다 — 언제 실행하든
심사 시점에 게시돼 있는 버전이 그대로 검증된다. 셀 6은 로컬 경로 B(4-6)와
같은 v0.2.0 `--hf` 플래그를 쓰므로, 모델을 따로 받는 단계가 없다.

`docs/RESULTS.md`의
[Run 4](docs/RESULTS.md#run-4--nvidia-tesla-t4-linux-google-colab)가 같은
조건의 T4 실측 기록이다(콜드 61초 / 캐시 1.45초 — 당시 노트북은 모델을
별도 셀에서 받는 이전 구성이라, 현재 셀 6은 다운로드가 합쳐진 만큼 조금
더 걸린다).

> ### ⚠️ 셀 6의 출력이 `BEST EFFORT`인 것은 정상이다
>
> 노트북의 목표치는 240 tok/s인데, 이 값은 개발 기기(Apple M3 Pro)에서는
> 달성되지만 **Tesla T4에서는 달성되지 않는다**. 그래서 셀 6은 종료 코드 3과
> 함께 다음을 출력한다:
>
> ```
> FiTuna result: BEST EFFORT (target not met)
>   quant : Q6_K   ngl : ...   ctx : 4096
>   gen tok/s (tg): 205.50      quality loss : 0.83%
>
>   artifact: .../SmolLM2-...-Q6_K.gguf  (... already produced during the search)
>
>   1) local API server (OpenAI-compatible):
>        ...llama-server -m ... -ngl ... -c 4096 --port 8080
>   2) import into Ollama: re-run with --export-ollama to write a Modelfile ...
>   3) terminal chat (interactive check):
>        ...llama-cli -m ... -ngl ... -c 4096
> ```
>
> 이것은 **오류가 아니라 이 도구의 핵심 기능**이다. "목표를 달성했다"고
> 지어내는 대신 "이 하드웨어에서는 불가능하며, 가장 근접한 구성은 이것"이라고
> 정직하게 보고하는 동작이다. 자세한 판정 기준은 **5장**을 참고한다.
>
> 참고로 같은 명령이 Apple M3 Pro에서는 목표를 달성한다(종료 코드 0). 동일 명령이
> 한 기기에서는 통과하고 다른 기기에서는 미달로 보고되는 것 자체가, 하드웨어를
> 실제로 측정해야만 얻을 수 있는 답이다.

---

## 4. 경로 B · 로컬 실행 (macOS / Linux)

아래 명령은 **그대로 복사해 붙여넣을 수 있다**. 각 단계의 기대 출력과 소요시간은
이 문서를 작성하면서 **Apple M3 Pro(18 GB) + llama.cpp Homebrew build 9960 +
Python 3.13.7** 환경에서 직접 실행해 관찰한 값이다.

### 4-1. llama.cpp 설치

FiTuna는 연산을 직접 하지 않고 llama.cpp 바이너리를 오케스트레이션한다. 따라서
`llama-quantize`, `llama-bench`, `llama-perplexity`(+ 결과 실행용 `llama-cli`)가
필요하다.

```bash
brew install llama.cpp
```

Homebrew가 없거나 소스 빌드가 필요하면(NVIDIA GPU는 `-DGGML_CUDA=ON` 추가):

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --target llama-quantize llama-bench llama-perplexity llama-cli -j
```

소스 빌드한 경우, 이후 모든 `fituna` 명령에
`--llama-bin-dir llama.cpp/build/bin`을 덧붙이면 된다.

### 4-2. FiTuna 설치 (Python 3.11 이상)

가상환경을 만들어 그 안에 설치한다(이유는 아래 참고):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install fituna
```

PyPI 게시본(v0.2.0+)이 저장소 최신 릴리스와 같다. 최신 main을 검증하려면
GitHub에서 바로 설치해도 된다:

```bash
pip install git+https://github.com/leeyunseokarchive/fituna
```

또는 소스에서 설치:

```bash
git clone https://github.com/leeyunseokarchive/fituna
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e fituna
```

- 런타임 의존성이 0개(표준 라이브러리만 사용)라 설치는 수 초면 끝난다.
- 시스템 `python3`가 3.11 미만이면(예: macOS 기본 python3는 3.9.6) 3.11 이상
  인터프리터로 가상환경을 만들어야 한다. 위처럼 `python3.13 -m venv .venv`.
  `python3.13: command not found`가 나오면 설치돼 있지 않은 것이다 —
  `brew install python@3.13` 한 줄로 받을 수 있고, 이미 있는 다른 3.11+
  버전(`python3.12` 등)을 써도 된다.
- **가상환경 없이 시스템 `python3`에 그대로 `pip install`하면 최신 Homebrew·
  Debian/Ubuntu 파이썬에서 `error: externally-managed-environment`로 실패한다**
  (PEP 668 — 직접 재현 확인, 6장 참고). 위 가상환경 절차가 이를 피하는 방법이다.
- 가상환경을 활성화한 채로는 `fituna` 명령이 곧바로 PATH에 잡힌다. 활성화를
  깜빡했거나 `--user`로 전역 설치해 `fituna: command not found`가 나오면,
  **`python3.13 -m fituna` 로 완전히 동일하게 사용할 수 있다**(6장 참고).
- 막히는 지점이 있으면 어느 단계에서든 `fituna help` (또는
  `fituna help <명령>`)가 한국어·영어를 함께 보여 준다.

### 4-3. 환경 점검 — `fituna doctor`

```bash
fituna doctor
```

실제 출력 (직접 실행, 종료 코드 0):

```
FiTuna doctor
  [PASS] python            3.13.7
  [PASS] llama-quantize    /opt/homebrew/bin/llama-quantize
  [PASS] llama-bench       /opt/homebrew/bin/llama-bench
  [PASS] llama-perplexity  /opt/homebrew/bin/llama-perplexity
  [PASS] llama-cli         /opt/homebrew/bin/llama-cli
  [PASS] llama-server      /opt/homebrew/bin/llama-server
  [PASS] gpu-backend       MTL0: Apple M3 Pro (13639 MiB, 13639 MiB free)
  [PASS] llama.cpp version 9960 (a935fbffe)
  [PASS] hardware          gpu=apple (Apple M3 Pro), vram=18432MB, cpu=11 cores, ram=18432MB, os=darwin
  [PASS] out-dir           out does not exist yet; can be created under .
  [PASS] disk-space        52.7 GB free at .

11 checks passed, 0 warnings, 0 failed.
Next: run `fituna quickstart` for a guided first run, or see `fituna run -h`.
```

경로/버전/하드웨어 값은 검증 기기에 따라 다르다. **`failed`가 0이면 다음
단계로 진행해도 된다.** `PASS`가 아닌 항목에는 항상 `->`로 시작하는 구체적인
조치 방법이 함께 출력된다. 바이너리는 경로만 찾는 게 아니라 실제로 실행해 보고
(dylib 누락·macOS 격리로 실행이 안 되면 `FAIL`), `gpu-backend`는 GPU가 있는데
llama.cpp 빌드가 CPU 전용이면 `WARN`을 낸다. 기계 수집용으로는 `fituna doctor --json`을 쓴다.

### 4-4. 품질 측정용 코퍼스 내려받기

품질 저하는 "텍스트 코퍼스에 대한 perplexity 증가율"로 측정한다. 아래 명령이
표준 라이브러리(`urllib`)만으로 코퍼스를 받아 온다. **별도 패키지 설치는 필요
없다.**

```bash
fituna fetch-corpus --lang en --out wiki.txt
```

실제 출력 (직접 실행, 종료 코드 0):

```
... (100행 단위 진행 로그, 이 문서에서는 앞부분 생략)
2026-08-25 03:15:20,263 INFO fituna: fetched 800/1000 rows
2026-08-25 03:15:20,648 INFO fituna: fetched 900/1000 rows
2026-08-25 03:15:21,023 INFO fituna: fetched 1000/1000 rows
Wrote 1000 rows to wiki.txt
Corpus: Salesforce/wikitext (wikitext-2-raw-v1, test split). License: CC BY-SA 3.0 (also dual-licensed GFDL) -- attribution and share-alike apply. Source: https://huggingface.co/datasets/Salesforce/wikitext
```

직접 실행: 약 4.6초. **네트워크·HuggingFace API 상태에 따라 다름** — 이전
측정(2026-07-30)에서는 최초 실행 38초, 재실행 4~5초였다(캐시 없이 매번
새로 받는 구조라 소요시간은 그때그때의 네트워크 상태를 반영한다).

한국어 코퍼스가 필요하면 `fituna fetch-corpus --lang ko --out kowiki-corpus.txt
--rows 500` (직접 실행: 약 5초, 5.9 MB).

### 4-5. 모델 준비 — `--hf`가 있으면 이 단계가 필요 없다

FiTuna는 로컬 GGUF 파일(`--model`)뿐 아니라, HuggingFace 저장소를 바로 받아
탐색하는 `--hf repo[:filename]`을 지원한다(v0.2.0). `--hf`를 쓰면 모델을 따로
받아 둘 필요 없이 다음 절의 `fituna run` 한 줄로 다운로드부터 탐색까지 끝난다 —
license가 자동으로 출력되고, 파일은 `--out`에 저장돼 이후 재실행에서 재사용된다.

<details>
<summary><b>다른 방법 — 이미 받아 둔 GGUF 파일로 재현 (<code>--model</code>, 2026-07-30 실측 기록)</b></summary>

오프라인이거나, 정확히 어떤 파일이 쓰이는지 다운로드 시점에 직접 확인하고
싶다면 이 방법도 동일하게 동작한다:

```bash
curl -L -o SmolLM2-135M-Instruct-f16.gguf \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf
```

직접 실행 결과(2026-07-30): 258 MB(270,885,952 바이트), 약 41초(네트워크
속도에 따라 다름). 135M 파라미터 모델이라 전체 탐색이 1~2분 안에 끝나므로
검증용으로 적합하다. 이 파일로 4-6을 재현하려면 아래 4-6의 접이식 상자
("이전 실측 기록")에 있는 `--model` 명령을 그대로 쓰면 된다.

</details>

### 4-6. 탐색 실행 — `fituna run`

```bash
fituna run --hf bartowski/SmolLM2-135M-Instruct-GGUF \
  --target-tps 240 --max-quality-loss 5 --ctx 2048 \
  --quality-corpus wiki.txt --out ./out --resume
```

직접 실행한 전체 출력 (2026-08-25, 콜드 캐시, **82.57초**, 종료 코드 **0**):

```
2026-08-25 03:22:26,473 INFO fituna: computing baseline perplexity on base GGUF
2026-08-25 03:22:28,563 INFO fituna: [Q8_0] quantizing
2026-08-25 03:22:28,951 INFO fituna: [Q8_0] evaluating quality
2026-08-25 03:22:30,858 INFO fituna: [Q6_K] quantizing
2026-08-25 03:22:31,328 INFO fituna: [Q6_K] evaluating quality
2026-08-25 03:22:33,278 INFO fituna: [Q5_K_M] quantizing
2026-08-25 03:22:33,762 INFO fituna: [Q5_K_M] evaluating quality
2026-08-25 03:22:35,768 INFO fituna: [Q4_K_M] quantizing
2026-08-25 03:22:36,352 INFO fituna: [Q4_K_M] evaluating quality
2026-08-25 03:22:38,394 INFO fituna: [Q3_K_M] quantizing
2026-08-25 03:22:38,915 INFO fituna: [Q3_K_M] evaluating quality
2026-08-25 03:22:40,855 INFO fituna: [Q3_K_M] quality loss 19.54% > 5.00% cap, skipping (early-exit A)
2026-08-25 03:22:40,855 INFO fituna: [Q2_K] quantizing
2026-08-25 03:22:41,219 INFO fituna: [Q2_K] evaluating quality
2026-08-25 03:22:43,169 INFO fituna: [Q2_K] quality loss 38.34% > 5.00% cap, skipping (early-exit A)
2026-08-25 03:22:43,169 INFO fituna: [Q8_0] bench full-offload (ngl=30)
2026-08-25 03:23:23,632 INFO fituna: [Q8_0] found ngl=24 meeting target -- done
license: apache-2.0 (weights published by bartowski/SmolLM2-135M-Instruct-GGUF; their terms, not FiTuna's)
downloading https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf
    0% (1.0 MB / 270.9 MB)
    ...(진행률 로그, 이 문서에서는 중간 생략)...
  100% (270.9 MB / 270.9 MB)
FiTuna result: MEETS TARGET

  quant           : Q8_0
  ngl             : 24
  ctx             : 2048

  prompt tok/s (pp): 1702.06
  gen tok/s    (tg): 247.23

  perplexity      : 18.2931 (baseline 18.2407)
  quality loss    : 0.29%

  artifact: out/SmolLM2-135M-Instruct-14784d04f46c-Q8_0.gguf  (144.8 MB -- already produced during the search)

  1) local API server (OpenAI-compatible):
       /opt/homebrew/bin/llama-server -m out/SmolLM2-135M-Instruct-14784d04f46c-Q8_0.gguf -ngl 24 -c 2048 --port 8080
  2) import into Ollama: re-run with --export-ollama to write a Modelfile beside the artifact
  3) terminal chat (interactive check):
       /opt/homebrew/bin/llama-cli -m out/SmolLM2-135M-Instruct-14784d04f46c-Q8_0.gguf -ngl 24 -c 2048
```

읽는 법:

- `computing baseline perplexity` → 후보별 `quantizing` / `evaluating quality`:
  1단계(품질 실측). 모든 후보의 품질을 먼저 잰다. `license:`와 `downloading`
  줄은 로그(stderr)와는 별도로 표준출력(stdout)에 찍히므로, 터미널을 파이프로
  넘기지 않는 한 다운로드 진행률이 위 로그와 실시간으로 섞여 보인다 — 위
  인용문은 파이프로 캡처하며 표준출력이 뒤늦게 flush된 결과라 순서가 바뀌어
  보일 뿐, 실제 터미널에서는 다운로드가 먼저 진행된다.
- `bench full-offload` → `found ngl=24 meeting target -- done`: 2단계(속도 탐색).
  품질이 좋은 순서로 벤치하고, 목표를 만족하면 **즉시 멈춘다**(조기종료). 이때
  `-ngl`은 전량 오프로드(30)가 아니라 **목표를 만족하는 최소값(24)**을 이진
  탐색으로 찾은 결과다.
- **`--resume`은 첫 실행에도 붙여야 한다.** `--resume`이 있어야 이번 결과가
  캐시에 저장되고, 다음 실행(4-7)이 그 캐시를 읽는다 — 빠뜨리면 재실행도
  처음부터 다시 계산한다(7장에서 캐시 키 구조를 설명한다).
- 작업 디렉토리 `./out`에 양자화된 GGUF 6개(전체 quant 후보)와 원본 F16
  (270.9 MB)이 함께 생성된다. 이번 실행 기준 `./out` 전체 합계는 **924 MB**
  (직접 측정, `du -sh`).

> **주의 — 승자 quant와 절대 수치는 기기·세션마다 다르다.** 위 실행에서는
> Q8_0이 247.23 tok/s로 목표를 통과했지만, 같은 기기에서 이전에 측정된 기록
> (`docs/RESULTS.md` Run 1)에서는 Q8_0이 205.91 tok/s로 미달이고 Q6_K(249.50
> tok/s)가 승자였다. 벤치마크 수치는 발열·부하 상태에 민감하기 때문이며, 이
> 편차는 프로젝트가 숨기지 않고
> [런간 편차 절](docs/RESULTS.md#run-to-run-variance-measured-not-hidden)에
> 실측과 함께 문서화해 두었다. **판정 기준은 절대값 일치가 아니라 로그의
> 형태**다(8장).

<details>
<summary><b>이전 실측 기록 (2026-07-30, <code>--model</code> 방식) — 손대지 않은 원본</b></summary>

4-5의 접이식 상자에서 `curl`로 받은 파일을 그대로 쓰는 명령과, 그 시점의 실제
출력이다. `--hf` 도입 이전 결과이며, 결과 블록이 `run command:` 한 줄로 끝나는
0.2 이전 형식을 그대로 보존한다.

```bash
fituna run --model SmolLM2-135M-Instruct-f16.gguf \
  --target-tps 240 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --quality-corpus wikitext-2-raw-test.txt --out ./out --resume
```

직접 실행한 전체 출력 (콜드 캐시, **1분 23초**, 종료 코드 **0**):

```
2026-07-30 03:09:49,872 INFO fituna: computing baseline perplexity on base GGUF
2026-07-30 03:09:51,822 INFO fituna: [Q8_0] quantizing
2026-07-30 03:09:52,177 INFO fituna: [Q8_0] evaluating quality
2026-07-30 03:09:54,048 INFO fituna: [Q6_K] quantizing
2026-07-30 03:09:54,477 INFO fituna: [Q6_K] evaluating quality
2026-07-30 03:09:56,361 INFO fituna: [Q5_K_M] quantizing
2026-07-30 03:09:56,809 INFO fituna: [Q5_K_M] evaluating quality
2026-07-30 03:09:58,747 INFO fituna: [Q4_K_M] quantizing
2026-07-30 03:09:59,213 INFO fituna: [Q4_K_M] evaluating quality
2026-07-30 03:10:01,140 INFO fituna: [Q8_0] bench full-offload (ngl=30)
2026-07-30 03:11:11,932 INFO fituna: [Q8_0] found ngl=28 meeting target -- done
FiTuna result: MEETS TARGET

  quant           : Q8_0
  ngl             : 28
  ctx             : 4096
  gguf            : out/SmolLM2-135M-Instruct-83beb8b331ac-Q8_0.gguf

  prompt tok/s (pp): 1669.37
  gen tok/s    (tg): 257.62

  perplexity      : 18.2931 (baseline 18.2407)
  quality loss    : 0.29%

  run command:
    /opt/homebrew/bin/llama-cli -m out/SmolLM2-135M-Instruct-83beb8b331ac-Q8_0.gguf -ngl 28 -c 4096
```

현재 버전(0.2+)은 마지막 `run command:` 한 줄 대신 산출물(artifact) 블록 + 세
가지 사용법을 출력한다 — 위 본문 4-6의 출력이 바로 그 현재 형태이며, 같은
측정값(quant/tok-s/perplexity)을 현재 코드의 `to_human()`에 그대로 통과시켜도
같은 모양이 나오는 것을 이 문서 작성 중 별도로 확인했다.

</details>

### 4-7. 재현성 확인 — 같은 명령 한 번 더

4-6과 **완전히 같은 명령**을 그대로 다시 실행한다.

직접 실행 결과(2026-08-25): **1.10초**, 결과 블록은 4-6과 완전히 동일, 종료
코드 0. `quantizing` / `bench` 로그가 같은 타임스탬프에 한꺼번에 찍히는데,
실제로 다시 계산한 것이 아니라 `./out/.fituna_cache.sqlite3` 캐시에서 즉시
읽어 온 것이다(7장). `--model` 방식의 과거 기록(0.64초, 2026-07-30)도 같은
자릿수이며 `docs/RESULTS.md` Run 1(0.75초)·Run 2(0.88초)와도 일관된다.

### 4-8. (선택) MCP 서버 확인 — AI 에이전트 연동

README가 강조하는 또 다른 표면인 `fituna-mcp`도 표준 라이브러리만으로 구현된
JSON-RPC 2.0/stdio 서버라 별도 설치 없이 바로 왕복 확인이 가능하다:

```bash
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fituna_detect_hardware","arguments":{}}}\n' \
  | fituna-mcp
```

직접 실행 결과(2026-08-25, 두 줄):

```
{"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "fituna", "version": "0.2.0"}}}
{"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "{\"gpu_vendor\": \"apple\", \"gpu_name\": \"Apple M3 Pro\", \"vram_mb\": 18432, \"cpu_cores\": 11, \"ram_mb\": 18432, \"os_name\": \"darwin\"}"}]}}
```

`serverInfo.version`이 설치한 FiTuna 버전과 일치하는지, `fituna_detect_hardware`
결과가 4-3의 `doctor` 출력과 같은 하드웨어를 가리키는지가 확인 포인트다.
`fituna_recommend` 도구는 같은 형태로 4-6과 동일한 탐색을 수행해 추천을
JSON으로 반환한다 — 실제 AI 에이전트가 이 서버를 붙이는 방법은
[README "MCP 서버"](README.ko.md#mcp-서버--ai-에이전트에게-추측이-아닌-실측-답변을)를
참고한다.

---

## 5. 정상 동작 판정 기준 ⭐

### 5-1. 종료 코드

`fituna run`의 종료 코드는 다음 네 가지뿐이다 (`echo $?`로 확인).

| 코드 | 의미 | 정상인가 |
|---|---|---|
| **0** | 목표를 만족하는 구성을 찾음 (`MEETS TARGET`) | ✅ **정상** |
| **1** | 일반 오류 — 모델 파일 없음, 네트워크 실패 등 | 메시지 확인 필요 (5-3절·6장) |
| **2** | 인자 오류(argparse) **또는** llama.cpp 바이너리를 찾지 못함 — 원인 두 가지, 아래에서 구분 | 메시지로 원인 구분 (6장) |
| **3** | 목표를 만족하는 구성이 없음 + **가장 근접한 구성 보고** | ✅ **정상** |

`fituna quickstart`는 여기에 한 줄이 더 붙는다.

| 코드 | 의미 | 정상인가 |
|---|---|---|
| **1** | stdin이 TTY가 아님 — 파이프·리다이렉트·CI에서 실행됨 | ✅ **정상**(설계된 거부. 아래 6장 참고) |

**종료 코드 2는 원인이 두 가지다.** 필수 플래그 누락이나 오타난 플래그는
argparse가 처리하며 stderr 첫 줄이 항상 `usage: ...`로 시작한다. 반면 llama.cpp
바이너리를 찾지 못한 경우(`BinaryNotFoundError`)는 FiTuna 자체 로거가 남기는
`... ERROR fituna: ...` 형태의 메시지이며 `usage:` 줄이 없다. **stderr 첫 줄이
`usage:`면 인자 오류, 아니면 환경(바이너리) 문제다.** 두 사례의 실제 출력은
6장 "종료 코드 2"에서 확인할 수 있다.

### 5-2. 🔴 종료 코드 3은 버그가 아니라 정직한 보고다

**이 항목을 오작동으로 판정하지 말 것.** FiTuna의 존재 이유가 바로 이 동작이다.

목표 속도가 해당 하드웨어에서 물리적으로 달성 불가능할 때, FiTuna는 그럴듯한
값을 지어내거나 조용히 실패하지 않는다. **"이 하드웨어에서는 목표 달성이
불가능하다"고 보고하고, 동시에 측정된 것 중 가장 근접했던 구성(best effort)을
바로 실행 가능한 커맨드와 함께 반환하도록 설계**되어 있다. 사용자는 이 답을
받아 목표치를 낮추거나 하드웨어를 바꾸는 판단을 할 수 있다.

이 상황은 목표치를 비현실적으로 높게 주면 언제든 재현할 수 있다. 아래는 이
문서를 작성하면서 **직접 실행한 실제 출력**이다(4-6과 같은 모델·같은 코퍼스,
목표만 5000 tok/s로 변경, `--export-ollama`도 함께 켜서 Ollama 연동까지 한 번에
확인한다):

```bash
fituna run --hf bartowski/SmolLM2-135M-Instruct-GGUF \
  --target-tps 5000 --max-quality-loss 5 --ctx 2048 \
  --quality-corpus wiki.txt --out ./out --resume --export-ollama
echo $?
```

```
2026-08-25 03:19:05,686 INFO fituna: [Q8_0] quantizing
2026-08-25 03:19:05,686 INFO fituna: [Q6_K] quantizing
2026-08-25 03:19:05,686 INFO fituna: [Q5_K_M] quantizing
2026-08-25 03:19:05,686 INFO fituna: [Q4_K_M] quantizing
2026-08-25 03:19:05,686 INFO fituna: [Q3_K_M] quantizing
2026-08-25 03:19:05,686 INFO fituna: [Q3_K_M] quality loss 19.54% > 5.00% cap, skipping (early-exit A)
2026-08-25 03:19:05,686 INFO fituna: [Q2_K] quantizing
2026-08-25 03:19:05,686 INFO fituna: [Q2_K] quality loss 38.34% > 5.00% cap, skipping (early-exit A)
2026-08-25 03:19:05,686 INFO fituna: [Q8_0] bench full-offload (ngl=30)
2026-08-25 03:19:05,686 INFO fituna: [Q8_0] full-offload 335.99 tok/s < target 5000.00, skipping (early-exit B)
2026-08-25 03:19:08,590 INFO fituna: [Q6_K] bench full-offload (ngl=30)
2026-08-25 03:19:11,458 INFO fituna: [Q6_K] full-offload 330.75 tok/s < target 5000.00, skipping (early-exit B)
2026-08-25 03:19:11,458 INFO fituna: [Q5_K_M] bench full-offload (ngl=30)
2026-08-25 03:19:14,293 INFO fituna: [Q5_K_M] full-offload 328.19 tok/s < target 5000.00, skipping (early-exit B)
2026-08-25 03:19:14,293 INFO fituna: [Q4_K_M] bench full-offload (ngl=30)
2026-08-25 03:19:14,293 INFO fituna: [Q4_K_M] full-offload 333.84 tok/s < target 5000.00, skipping (early-exit B)
2026-08-25 03:19:14,293 INFO fituna: no candidate met target after exhausting all quant candidates
2026-08-25 03:19:14,293 ERROR fituna: no quant/ngl/ctx combination met target_tokens_per_sec within max_quality_loss_pct
2026-08-25 03:19:14,293 INFO fituna: closest best-effort attempt:
FiTuna result: BEST EFFORT (target not met)

  quant           : Q8_0
  ngl             : 30
  ctx             : 2048

  prompt tok/s (pp): 5497.56
  gen tok/s    (tg): 335.99

  perplexity      : 18.2931 (baseline 18.2407)
  quality loss    : 0.29%

  artifact: out/SmolLM2-135M-Instruct-b76f0d9ed1fe-Q8_0.gguf  (144.8 MB -- already produced during the search)

  1) local API server (OpenAI-compatible):
       /opt/homebrew/bin/llama-server -m out/SmolLM2-135M-Instruct-b76f0d9ed1fe-Q8_0.gguf -ngl 30 -c 2048 --port 8080
  2) import into Ollama:
       ollama create <name> -f out/Modelfile
  3) terminal chat (interactive check):
       /opt/homebrew/bin/llama-cli -m out/SmolLM2-135M-Instruct-b76f0d9ed1fe-Q8_0.gguf -ngl 30 -c 2048
```

`echo $?`의 출력:

```
3
```

`out/Modelfile`도 실제로 쓰여 있다(직접 확인):

```
FROM ./SmolLM2-135M-Instruct-b76f0d9ed1fe-Q8_0.gguf
PARAMETER num_gpu 30
PARAMETER num_ctx 2048
```

주목할 점 — `2)`가 "재실행하라"는 안내가 아니라 실제 `ollama create` 명령이다.
`--export-ollama`를 준 사용자는 목표를 달성하지 못한 경우에도(종료 코드 3)
`out/Modelfile`을 실제로 받는다.

> **주의 — 여기서도 승자 quant와 절대 수치는 기기·세션마다 다르다.** 위
> 출력은 Q8_0을 최선으로 보고하지만, 이는 고정된 기대 출력이 아니라 이
> 세션에서 관측된 값이다. 목표 미달일 때 어느 후보가 "가장 근접"으로 뽑히는지는
> 4-6과 똑같이 발열·부하에 민감한 벤치 수치로 결정되므로, 다른 기기·다른
> 세션에서는 다른 quant와 다른 품질손실이 나올 수 있다. 4-6의 주의사항이 그대로 적용된다 —
> **판정 기준은 절대값 일치가 아니라, 종료 코드 3과 함께 완결된 best-effort
> 결과 블록이 나오는가**이다.

`ERROR` 로그가 한 줄 찍히지만, 그 뒤에 **`closest best-effort attempt:`와 완결된
결과 블록, 실행 가능한 커맨드가 정상적으로 출력**된다. 즉 사용자는 빈손으로
끝나지 않는다. 위 실행은 8.6초가 걸렸다(4-6에서 만든 캐시를 재사용한 상태).

`docs/RESULTS.md`의 [Run 4](docs/RESULTS.md#run-4--nvidia-tesla-t4-linux-google-colab)
(NVIDIA Tesla T4)도 실제 종료 코드 3 사례이며, 대회 제출 문서에 **정상 실측
결과**로 기재되어 있다. 경로 A(Colab)를 따라가면 셀 6에서 이 상황을 그대로 보게
된다.

<details>
<summary><b>이전 실측 기록 (2026-07-30, <code>--model</code> 방식, <code>--export-ollama</code> 미사용)</b></summary>

```bash
fituna run --model SmolLM2-135M-Instruct-f16.gguf \
  --target-tps 5000 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --quality-corpus wikitext-2-raw-test.txt --out ./out --resume
echo $?
```

```
2026-07-30 03:11:49,742 INFO fituna: [Q8_0] quantizing
2026-07-30 03:11:49,743 INFO fituna: [Q6_K] quantizing
2026-07-30 03:11:49,743 INFO fituna: [Q5_K_M] quantizing
2026-07-30 03:11:49,743 INFO fituna: [Q4_K_M] quantizing
2026-07-30 03:11:49,743 INFO fituna: [Q8_0] bench full-offload (ngl=30)
2026-07-30 03:11:49,743 INFO fituna: [Q8_0] full-offload 298.59 tok/s < target 5000.00, skipping (early-exit B)
2026-07-30 03:11:53,407 INFO fituna: [Q6_K] bench full-offload (ngl=30)
2026-07-30 03:11:57,161 INFO fituna: [Q6_K] full-offload 296.04 tok/s < target 5000.00, skipping (early-exit B)
2026-07-30 03:11:57,161 INFO fituna: [Q5_K_M] bench full-offload (ngl=30)
2026-07-30 03:12:00,852 INFO fituna: [Q5_K_M] full-offload 288.35 tok/s < target 5000.00, skipping (early-exit B)
2026-07-30 03:12:00,852 INFO fituna: [Q4_K_M] bench full-offload (ngl=30)
2026-07-30 03:12:00,852 INFO fituna: [Q4_K_M] full-offload 295.31 tok/s < target 5000.00, skipping (early-exit B)
2026-07-30 03:12:00,852 INFO fituna: no candidate met target after exhausting all quant candidates
2026-07-30 03:12:00,852 ERROR fituna: no quant/ngl/ctx combination met target_tokens_per_sec within max_quality_loss_pct
2026-07-30 03:12:00,852 INFO fituna: closest best-effort attempt:
FiTuna result: BEST EFFORT (target not met)

  quant           : Q8_0
  ngl             : 30
  ctx             : 4096
  gguf            : out/SmolLM2-135M-Instruct-83beb8b331ac-Q8_0.gguf

  prompt tok/s (pp): 3111.47
  gen tok/s    (tg): 298.59

  perplexity      : 18.2931 (baseline 18.2407)
  quality loss    : 0.29%

  run command:
    /opt/homebrew/bin/llama-cli -m out/SmolLM2-135M-Instruct-83beb8b331ac-Q8_0.gguf -ngl 30 -c 4096
```

종료 코드: `3`. 이때는 `--export-ollama`를 주지 않았고, 현재 버전과 달리
"`--export-ollama`로 재실행하라"는 문구만 출력했다 — 본문에서 보듯 현재
버전은 이 경로에서도 `out/Modelfile`을 실제로 쓴다(수정됨).

</details>

### 5-3. 오작동·크래시와 구분하는 방법

| | 정상 동작 (코드 0 또는 3) | 실제 오작동 |
|---|---|---|
| 결과 블록 | `FiTuna result:` 블록이 완결된 형태로 출력됨 | 출력되지 않음 |
| 산출물(artifact) 블록 | `artifact:` 줄 + 세 가지 사용법(서버/Ollama/`llama-cli`)이 항상 출력됨 | 없음 |
| Python traceback | **없음** | `Traceback (most recent call last):` 가 그대로 노출됨 |
| 종료 코드 | 0 또는 3 | 1 (예상치 못한 예외는 `unexpected error`로 로깅) |

즉 **Python traceback이 화면에 노출되었는가**가 핵심 구분선이다. 파일 없음·
네트워크 실패 같은 예상 가능한 실패는 traceback 없이 한 줄짜리 영문 설명
메시지와 종료 코드 1로 처리된다(인자 오류는 이와 별개로 종료 코드 2이며 5-1절
참고). 아래 명령은 그대로 붙여넣을 수 있고, 모델 파일이 없어도 즉시 끝난다 —
직접 실행:

```bash
fituna run --model nope.gguf \
  --target-tps 30 --max-quality-loss 5 --ctx 4096 \
  --quality-corpus wiki.txt --out ./out
echo $?
```

```
2026-08-25 03:19:25,380 ERROR fituna: nope.gguf is neither a .gguf file nor an HF-format model directory -- nothing to convert.
1
```

### 5-4. 모델·GPU 없이 할 수 있는 오프라인 점검

llama.cpp나 모델 파일 없이도, 소스 저장소만 있으면 다음을 확인할 수 있다.
**이미 4-2에서 소스 설치 경로(`git clone` + `pip install -e fituna`)를 따라왔다면
`fituna/`와 `.venv/`가 이미 있으니, 아래에서 `git clone`·`python3.13 -m venv`
줄은 건너뛴다.**

```bash
git clone https://github.com/leeyunseokarchive/fituna
python3.13 -m venv .venv          # 4-2와 같은 위치(저장소 바깥)에 만든다
source .venv/bin/activate
pip install pytest
cd fituna
python3.13 -m pytest -q
```

직접 실행 결과: `360 passed in 3.91s` (2026-10-05, `v0.3.3` 기준. 테스트가
추가되면 개수는 늘어난다 — 중요한 것은 **실패 0건**이다).

> **`python3`가 아니라 `python3.13`인 이유.** macOS 기본 `python3`는 3.9.6이고,
> FiTuna는 3.11 이상을 요구한다. 가상환경을 활성화했다면 `python3`도 3.13을
> 가리키지만, 새 셸을 열었거나 `source`를 빠뜨리면 시스템 3.9.6이 잡혀
> 테스트가 실패한다(직접 재현: 시스템 `python3` — 3.9.6 — 로 실행하면
> `test_doctor.py`의 python 버전 체크 하나가 FAIL로 갈리면서 `1 failed, 255
> passed`가 된다). 인터프리터를 명시하면 이 함정 자체가 사라진다.

모듈별 자체 점검도 개별 실행할 수 있다 (직접 실행, 모두 종료 코드 0, 각각
`OK` 한 줄만 출력):

```bash
python3.13 -m fituna.cli --selfcheck        # -> fituna.cli self-check OK
python3.13 -m fituna.doctor                 # -> fituna.doctor self-check OK
python3.13 -m fituna.report                 # -> fituna.report self-check OK
python3.13 -m fituna.quickstart --selfcheck # -> fituna.quickstart self-check OK
```

---

## 6. 자주 걸리는 문제

### `fituna: command not found`

파이썬 콘솔 스크립트 디렉토리가 `PATH`에 없는 경우다(특히 `pip install --user`
형태로 설치되면 자주 발생한다. 이 문서 작성 환경에서도 실제로 발생했다).

- 조치 1: `fituna` 대신 **`python3.13 -m fituna`** 를 쓴다. 완전히 동일한
  CLI다. 예: `python3.13 -m fituna doctor`. 여기서 `python3`가 아니라
  `python3.13`인 것이 중요하다 — 이 오류의 흔한 원인이 가상환경 미활성화인데,
  그 상태의 `python3`는 macOS 기본 3.9.6이라 FiTuna가 아예 실행되지 않는다.
- 조치 2: `python3.13 -m pip show -f fituna`가 알려주는 스크립트 경로를 `PATH`에
  추가한다.

### `fituna quickstart`가 아무것도 묻지 않고 끝난다

`quickstart`는 대화형 마법사이므로 stdin이 TTY일 때만 동작한다. 파이프·리다이렉트
(`fituna quickstart < /dev/null`)나 CI에서는 묻지 않고 안내 두 줄을 출력한 뒤
**종료 코드 1**로 끝난다(직접 재현: `fituna quickstart는 대화형 터미널(TTY)이
필요합니다.`). 오작동이 아니라 설계된 거부다 — 답을 받을 수 없는 질문을 던지느니
`fituna run`을 직접 쓰라고 안내한다.

마법사 안에서도 막다른 길은 종료가 아니라 **메뉴로 되돌아간다**: 라이선스 조건에
맞는 후보가 하나도 남지 않았을 때, HuggingFace 검색 결과가 비었을 때, 다운로드가
실패했을 때 — 세 경우 모두 다시 고를 수 있게 모델 선택 메뉴로 돌아온다.

`quickstart`는 `run`이 못 하는 일을 하지 않는다. 조립한 `fituna run` 명령을 그대로
출력한 뒤 같은 코드를 호출할 뿐이므로, 이 문서의 검증 경로는 계속 **경로 B
(`fituna run` 직접 실행)** 이다.

### `error: externally-managed-environment`

Homebrew·Debian/Ubuntu 등 최신 배포판 Python은 PEP 668에 따라 시스템 전역
`pip install`을 막는다. 직접 재현한 실제 메시지의 첫 줄:

```
error: externally-managed-environment
```

- 조치: 4-2절의 가상환경 절차(`python3.13 -m venv .venv` →
  `source .venv/bin/activate`)를 따른다. 이 문서의 모든 설치 명령은 이미
  가상환경을 전제로 고쳐져 있다.

### 종료 코드 2 — 원인 두 가지 (인자 오류 / llama.cpp 바이너리를 찾지 못함)

**인자 오류.** 필수 플래그를 빼고 `fituna run`만 실행해 직접 재현한 실제
출력이다(2026-08-25, v0.2.0 — `--model`과 `--hf`가 상호배타 그룹으로 묶여
`usage:` 줄이 `(--model MODEL | --hf HF)`로 바뀌었다). stderr 첫 줄이
`usage:`로 시작한다.

```
usage: fituna run [-h] (--model MODEL | --hf HF) --target-tps TARGET_TPS
                  --max-quality-loss MAX_QUALITY_LOSS [--ctx CTX]
                  [--quant QUANT] [--gpu {none,nvidia,amd,apple}]
                  [--vram-mb VRAM_MB] [--llama-bin-dir LLAMA_BIN_DIR]
                  --quality-corpus WIKITEXT [--ppl-chunks PPL_CHUNKS]
                  [--out OUT] [--json] [--resume] [--export-ollama]
fituna run: error: the following arguments are required: --target-tps, --max-quality-loss, --quality-corpus/--wikitext
```

오타난 플래그(`fituna run ... --bogus-flag foo`)도 같은 종료 코드 2, 같은
`usage:` 형태다(직접 재현). 다만 이때는 최상위 파서의 usage가 나오고, 이것이
두 줄로 감기기 때문에 `fituna: error:` 줄은 **세 번째** 줄이다:

```
usage: fituna [-h] [-v]
              {run,detect-hw,list-binaries,doctor,fetch-corpus,quickstart,help} ...
fituna: error: unrecognized arguments: --bogus-flag foo
```

- 조치: `usage:` 줄이 보여주는 필수 인자를 채우거나 오타난 플래그명을 고친다.

**llama.cpp 바이너리를 찾지 못함(`BinaryNotFoundError`).** 아래는 빈 디렉토리
(`emptybin`)를 `--llama-bin-dir`로 넘겨 직접 재현한 실제 출력이다. `usage:` 줄이
없고, 대신 FiTuna 자체 로거가 남기는 메시지다.

```
ERROR fituna: Required llama.cpp binaries not found under --llama-bin-dir emptybin: llama-quantize, llama-bench, llama-perplexity.
Build llama.cpp (https://github.com/ggml-org/llama.cpp#building-the-project) and either add its build output directory (e.g. build/bin) to your PATH, or pass --llama-bin-dir /path/to/llama.cpp/build/bin.
```

- 조치: 바이너리가 있는 디렉토리를 `--llama-bin-dir`로 넘긴다. 소스 빌드했다면
  보통 `llama.cpp/build/bin`이다.
- 먼저 `fituna doctor --llama-bin-dir /path/to/bin`으로 확인하면 어떤 바이너리가
  없는지 항목별로 `[FAIL]`/`[WARN]` 표에 정리해 알려 준다(이때 doctor 자체도
  종료 코드 2를 반환한다 — 직접 확인).

**구분법**: stderr 첫 줄이 `usage:`면 인자 오류, `ERROR fituna:`면 환경(바이너리)
문제다.

### GPU 오검출 / GPU가 없는 환경

- `fituna detect-hw`로 FiTuna가 무엇을 감지했는지 먼저 본다.
- 자동 감지 결과가 틀렸거나 특정 하드웨어를 가정하고 싶으면 수동 지정한다:
  `--gpu {none,nvidia,amd,apple} --vram-mb 8192`
- GPU가 없어도 CPU만으로 동작한다(느릴 뿐이다). doctor에서는 `WARN`으로 표시된다.
- **알려진 제약**: Windows + AMD 조합은 `rocm-smi`의 표준 배포판이 없어 자동
  감지가 불가능하다. `--gpu amd --vram-mb <N>`로 수동 지정해야 한다.

### 디스크 부족

- `fituna doctor`의 `disk-space` 항목이 20 GB 미만이면 `WARN`을 낸다.
- 필요 용량 실측: SmolLM2-135M 전체 6개 후보 + F16 원본 = **924 MB**(이 문서
  작성 중 `--hf`로 직접 측정, `--quant`로 후보를 좁히면 그만큼 줄어든다),
  Qwen3-4B 4개 후보 = **12.1 GB**(`docs/RESULTS.md` Run 2).
- 조치: `--quant`로 후보를 줄인다. 예: `--quant Q6_K,Q4_K_M`. 양자화된 파일은
  재실행 시 재사용되므로 반복 실행에서 추가로 늘지 않는다.

### 코퍼스 다운로드 실패 / 네트워크 차단

이 문서를 작성하면서 실제로 한 번 겪은 상황이다(HuggingFace API 응답 지연).

```
ERROR fituna: could not reach the HuggingFace dataset-viewer API: The read operation timed out. Download it manually from the dataset's HuggingFace page instead, or point --quality-corpus at any UTF-8 plain-text file you already have -- FiTuna's quality gate only needs text resembling your workload, not this specific corpus.
```

- 종료 코드는 1이며, **미완성 파일이 남지 않는다**(원자적 쓰기).
- 조치 1: 그대로 재실행한다(위 사례는 두 번째 시도에서 정상 완료됐다).
- 조치 2: 네트워크가 차단된 환경이라면 **아무 UTF-8 평문 텍스트 파일이나**
  `--quality-corpus`로 지정하면 된다. 품질 게이트는 특정 코퍼스가 아니라 "워크로드와
  닮은 텍스트"만 필요로 한다. 예: 임의의 문서 몇 편을 이어붙인 `.txt` 파일.

### Python 버전

- 3.11 이상이 필요하다. macOS 기본 `python3`는 3.9.6이라 동작하지 않는다.
- `fituna doctor`의 첫 줄 `python` 항목이 `FAIL`이면 이 경우다. 3.11+ 인터프리터로
  다시 설치·실행한다(이 문서의 실측 환경은 3.13.7).

### 벤치 타임아웃 로그가 보일 때

`[Q4_K_M] ngl=0 bench timed out -- treating as 0 tok/s` 같은 줄은 **정상 처리**다.
제한 시간 안에 끝나지 못한 벤치를 "너무 느림"으로 기록하고 탐색을 계속하는
동작이며, 중단이나 크래시가 아니다(`docs/RESULTS.md` Run 2에 실측 사례 기록).

### Windows에서의 제약 (정직한 고지)

- Windows는 **실기 통합 검증이 수행되지 않았다.** 단위 테스트와 3-OS(Windows
  포함) × 2-Python CI는 통과하지만, 실제 llama.cpp 바이너리를 붙인 E2E 검증은
  macOS(Apple Silicon/Metal)와 Linux(NVIDIA T4/CUDA)에서만 수행했다.
- Windows 환경에서 검증하셔야 한다면 **경로 A(Colab)** 를 권장한다. 브라우저만
  있으면 되고 실제 검증은 Linux/CUDA에서 이뤄진다.

---

## 7. 재현성 확인 방법

**같은 명령을 `--resume`으로 다시 돌리면 1초 안에 동일한 결과가 나온다 —
단, `--resume`은 처음 실행할 때부터 붙여야 한다.** 캐시 객체 자체가
`--resume`이 있을 때만 생성되므로(`fituna/cli.py`), 첫 실행에 `--resume`을
빠뜨리면 그 실행은 캐시를 쓰지도 채우지도 않는다 — 이후 `--resume`을 붙여
다시 돌려도 "처음 보는 요청"이라 전체를 다시 계산한다. 4장의 모든 명령이
첫 실행부터 `--resume`을 포함하는 이유가 이것이다.

| 환경 / 모델 | 재실행 시간 | 출처 |
|---|---|---|
| M3 Pro / SmolLM2-135M (`--hf`) | **1.10초** | 이 문서 작성 중 직접 실행 (2026-08-25) |
| M3 Pro / SmolLM2-135M (`--model`) | 0.64초 | 이 문서 작성 중 직접 실행 (2026-07-30) |
| M3 Pro / SmolLM2-135M | 0.75초 | `docs/RESULTS.md` Run 1 |
| M3 Pro / Qwen3-4B | 0.88초 | `docs/RESULTS.md` Run 2 |
| Tesla T4 / SmolLM2-135M | 1.45초 | `docs/RESULTS.md` Run 4 |

- 캐시 파일 위치: `<--out으로 준 디렉토리>/.fituna_cache.sqlite3` (sqlite3).
- **캐시 키 구성**: 속도 측정치는 `(모델 지문 × 하드웨어·llama.cpp 빌드 지문 ×
  quant × ngl × ctx)`, 품질 측정치는 `(모델 지문 × quant × ppl-chunks × 코퍼스
  지문)`으로 저장된다. 모델·코퍼스 지문은 `sha256(파일명:크기:수정시각)`, 하드웨어
  지문은 `GPU 벤더·이름·VRAM·CPU 코어·RAM·OS + llama.cpp 빌드 버전`이다. 따라서
  **다른 llama.cpp 빌드나 다른 코퍼스에서 잰 수치가 재사용되는 일이 없다.**
- 콜드 상태에서 다시 재현하려면 작업 디렉토리를 지운다: `rm -rf ./out`.
- 출력 GGUF 파일명에 들어가는 12자리 문자열
  (`SmolLM2-135M-Instruct-14784d04f46c-Q8_0.gguf`의 `14784d04f46c`)은 위 모델
  지문의 앞부분이며 파일 수정시각을 포함한다. **검증 기기에서 이 값이 이 문서와
  달라지는 것은 정상이다.**
- 기계 판독용 출력이 필요하면 `--json`을 붙인다: `fituna doctor --json`,
  `fituna run ... --json`.

---

## 8. 하드웨어별 실측 기준값

**절대 수치는 기기마다 다르다. 로그의 *형태*가 같으면 정상이다.**

정상 실행의 형태 체크리스트:

1. `computing baseline perplexity on base GGUF` — 기준 품질 측정
2. 후보별 `[QUANT] quantizing` → `[QUANT] evaluating quality` — 1단계(품질 실측)
3. (품질 예산 초과 후보가 있으면) 품질 게이트에서 탈락 — 조기종료 A
4. 품질이 좋은 순서로 `[QUANT] bench full-offload (ngl=N)` — 2단계(속도 측정)
5. `full-offload ... < target ..., skipping (early-exit B)` 또는
   `found ngl=N meeting target -- done`
6. `FiTuna result: MEETS TARGET` **또는** `FiTuna result: BEST EFFORT (target not met)`
   + 산출물(artifact) 블록(gguf 경로·크기 + 서버/Ollama/`llama-cli` 세 가지 사용법)

이 6단계가 순서대로 나타나면, 숫자가 무엇이든 정상 동작이다.

**단, `--resume`으로 캐시가 적중한 재실행에서는 1번과 2번의
`evaluating quality`가 아예 나타나지 않는다** — 품질은 이미 측정돼 캐시에
있으므로 다시 재지 않는다(4-7이 그 경우다). 캐시 적중 실행에서 이 두 항목이
없는 것은 누락이 아니라 `--resume`이 동작한 증거다.

### 실측 기준값

| 환경 | 명령 | 결과 | 콜드 탐색 | 출처 |
|---|---|---|---|---|
| Apple M3 Pro | SmolLM2-135M (`--hf`), 목표 240 tok/s | Q8_0, `ngl=24`, 247.23 tok/s, 손실 0.29% (코드 0) | 82.57초 | 이 문서 작성 중 직접 실행 (2026-08-25) |
| Apple M3 Pro | SmolLM2-135M (`--model`), 목표 240 tok/s | Q6_K, `ngl=30`, 249.50 tok/s, 손실 0.53% (코드 0) | 75.7초 | `docs/RESULTS.md` Run 1 |
| Apple M3 Pro | 위와 동일 명령 (2026-07-30 재실행) | Q8_0, `ngl=28`, 257.62 tok/s, 손실 0.29% (코드 0) | 82.8초 | 이 문서 작성 중 직접 실행 |
| Apple M3 Pro | Qwen3-4B, 목표 30 tok/s | Q4_K_M, `ngl=33`, 30.81 tok/s, 손실 1.73% (코드 0) | 품질 5분 01초 + 속도 12분 53초 | `docs/RESULTS.md` Run 2 |
| NVIDIA Tesla T4 (Colab) | SmolLM2-135M, 목표 240 tok/s | **BEST EFFORT** Q6_K 205.50 tok/s, 손실 0.83% (**코드 3**) | 61초 | `docs/RESULTS.md` Run 4 |

위 표의 1~3행은 **같은 기기에서 사실상 같은 조건을 다른 시점에 돌린 결과인데
승자 quant나 절대 수치가 조금씩 다르다.** 벤치마크 수치가 발열·부하 상태에
민감하기 때문이며, 이는 숨겨진 결함이 아니라 측정 도구의 본질적 성질로서
[런간 편차 절](docs/RESULTS.md#run-to-run-variance-measured-not-hidden)에 실측
데이터와 함께 문서화되어 있다. 목표치와 몇 tok/s 이내로 근접한 판정은 marginal로
보고 기기가 평상 온도일 때 재실행하는 것을 권장한다.

5행은 **같은 명령이 다른 하드웨어에서는 목표 미달로 정직하게 보고되는** 사례다
(5장 참고).

전체 로그·타이밍·편차 분석: **[docs/RESULTS.md](docs/RESULTS.md)**

---

## 9. 함께 보면 좋은 문서

| 문서 | 내용 |
|---|---|
| [README.md](README.md) | 프로젝트 개요, 기능, 설계 요약 (영어 · [한국어판](README.ko.md)) |
| [docs/RESULTS.md](docs/RESULTS.md) | 실측 결과 전문 — 5회 실기 측정, 타이밍, 편차 분석 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 모듈 구조, 탐색 알고리즘, 오류·종료 코드 처리 |
| [docs/USE_CASES.md](docs/USE_CASES.md) | 사용 시나리오 |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | 3분 시연 영상 컷 단위 시나리오 (한국어) |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) · [docs/SBOM.md](docs/SBOM.md) | 제3자 구성요소 고지 및 SBOM |
| [docs/AI_MODEL_USAGE.md](docs/AI_MODEL_USAGE.md) | AI 활용 개발 공개 |
