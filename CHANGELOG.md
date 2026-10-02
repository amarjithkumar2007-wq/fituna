# 변경 이력

프로젝트의 주요 변경 사항을 이 문서에 기록합니다.

형식은 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)를 따르며,
버전은 [유의적 버전](https://semver.org/spec/v2.0.0.html)에 맞춰 관리합니다.

## [미출시]

### 수정

- **full offload 측정에서 layer 하나가 CPU에 남던 문제** — llama.cpp는 출력층을
  layer 하나로 더 세므로 full offload는 `-ngl <block 수 + 1>`입니다. 이전에는
  `-ngl <block 수>`를 상한으로 써서 layer 하나가 CPU에 남았고, 목표를 full
  offload에서만 넘는 quant가 조기 종료 B로 탈락할 수 있었습니다. SmolLM2-135M
  (block 30, Metal)에서 `-ngl 30`은 233~240 tok/s, `-ngl 31`은 258~262 tok/s였습니다.
  이제 상한과 이진탐색 범위가 `n_layers + 1`입니다.
- **GPU가 없는 하드웨어에서 측정과 추천 명령이 어긋나던 문제** — GPU가 없으면
  `-ngl <block 수>`로 측정한 뒤 결과에 `ngl=0`만 붙였습니다. GPU 백엔드가 있는
  llama.cpp 빌드에서는 그 측정이 GPU에서 돌아, `--gpu none`이나 GPU 자동 감지
  실패(#44) 때 GPU 속도를 CPU 실행 명령에 붙여 보고했습니다. 이제 추천하는
  `-ngl 0` 그대로 측정합니다.

## [0.3.1] — 2026-10-02

### 수정

- **0.3.0이 기존 품질 측정 캐시를 지우던 문제** — 0.3.0은 `metric` 열이 없는
  0.2.x 캐시를 열면 `quality_cache`를 통째로 삭제했습니다. 0.2.x의 기록은 모두
  perplexity 측정이라 빠진 값(`metric='ppl'`, `kld` 없음)을 알 수 있으므로, 이제
  삭제하지 않고 새 형식으로 옮깁니다. 옮기는 작업은 한 transaction 안에서 하므로
  중간에 실패해도 원래 표가 그대로 남고 다음 실행에서 다시 시도합니다. 실패하면
  "파일을 지우라"는 손상 파일 안내 대신 원인을 알려 줍니다. `corpus_fp`가 없는 더
  오래된 캐시는 corpus를 알 수 없으므로 전처럼 다시 측정합니다
  ([#57](https://github.com/leeyunseokarchive/fituna/pull/57), #55).
- **llama-bench 속도 값 누락을 0 tok/s로 처리하던 문제** — 생성 측정 기록이나
  `avg_ts`가 없으면 후보를 "느림"으로 탈락시키지 않고 원본 출력과 함께 오류를
  냅니다. 프롬프트 속도는 판정에 쓰이지 않으므로 계속 선택 항목입니다
  ([#52](https://github.com/leeyunseokarchive/fituna/pull/52), @varshith84, #47).

## [0.3.0] — 2026-10-02

### 추가

- **`fituna run --quality-metric kld`** — 품질 측정에 KL divergence를 선택형으로
  추가했습니다. F16 모델의 기준 logits를 한 번 만든 뒤 각 양자화 후보와
  비교하며, 사람용·JSON 보고서에 KLD가 표시됩니다. 기본값(`ppl`)의 동작은
  바뀌지 않았고, 통과/탈락 판정은 여전히 `--max-quality-loss`(perplexity)로
  합니다. KLD 기반 판정은 #49에서 추적합니다.
  ([#45](https://github.com/leeyunseokarchive/fituna/pull/45), @Chirudeva-Reddy)
- **llama.cpp 출력 해석 테스트** — `tests/test_bench.py`가 llama-bench JSON
  해석과 실행 명령 생성을 검사합니다
  ([#46](https://github.com/leeyunseokarchive/fituna/pull/46), @PandaHUN777).
  `tests/test_quality.py`는 실제 llama.cpp(b11342) 출력을 캡처한 fixture로
  KLD·PPL 해석을 검사합니다(#45).

## [0.2.1] — 2026-08-27

### 변경

- **PyPI README 최신화** — 0.2.0 배포물에 박제된 옛 README를 최신 문서
  (한국어 우선, 로고·데모 영상 추가)로 갱신.
- **`fituna-mcp` 요청 검증 강화 및 기본값 개선** — `out_dir` 기본값을
  `./fituna-out`에서 `./out`으로 변경. `model_path`를 생략하면 `out_dir`에서
  base-precision GGUF를 자동 탐색하고, `wikitext_path`도 기본값(`./wiki.txt`)을
  쓰도록 선택 인자로 완화. 비정상 JSON-RPC 요청에 에러 응답을 반환하도록 검증
  강화(`tools/call`의 `params`/`arguments`가 dict가 아니면 -32602 반환).

## [0.2.0] — 2026-08-24

### 추가

- **PyPI 배포** — 이제 `pip install fituna`로 설치할 수 있으며 git checkout은
  개발할 때만 필요합니다. 최초 정식 배포 버전은 0.2.0입니다. 같은 날 올라간
  0.1.0은 0.2.0 이전 snapshot으로 `--hf` flag가 없으므로 0.2.0을 사용하세요.

- **`fituna run --hf repo[:filename]`** — script 경로에서 대화 없이
  HuggingFace 모델을 다운로드합니다. 저장소 이름만 주면 HF API를 조회해 유일한
  F16/BF16 `.gguf`를 고릅니다. 파일이 없거나 여러 개면 추측하지 않고 선택할 수
  있는 `.gguf` 목록을 오류에 표시합니다. API에 모델 라이선스가 있으면 출력하고,
  없으면 없다는 사실을 알립니다. `repo:filename` 형식은 목록을 조회하지 않고
  지정한 파일만 받습니다. 파일은 `--out`에 저장해 다음 실행에서 재사용합니다.
  마법사와 같은 atomic 임시 파일 다운로드를 공유하므로 진행 문구는 특정 언어에
  의존하지 않도록 바꿨습니다.

- **`fituna quickstart`** — 환경 점검 → 목표 → 라이선스 요구사항 → 모델 → 품질
  corpus → 확인 및 실행의 6단계 대화형 마법사입니다. 조립하는 모든 탐색 매개변수는
  공개 `fituna run` flag에 대응하며 argv 동등성 test로 이를 검증했습니다. 엄선
  모델 다운로드와 HuggingFace 검색은 `run`에 없는 마법사 전용 편의 기능입니다.
  완성한 `fituna run ...` 명령을 보여 준 뒤 같은 argv를 CLI parser로 다시
  해석해 process 안에서 실행하므로 화면에 표시한 명령과 실제 실행이 달라질 수
  없습니다. TTY가 없으면 `fituna run`을 안내하고 종료 코드 1로 끝납니다.
  처리량을 예측하지 않으며 이 점을 화면에 명시합니다. 메모리 적합 여부는 공개된
  파일 크기와 감지된 VRAM/RAM을 20% 여유분 가정으로 단순 계산합니다. F16 파일이
  예산을 넘는 엄선 모델도 숨기지 않고 경고와 함께 선택할 수 있게 남깁니다. 실제
  실행 파일은 양자화본이기 때문입니다. Qwen3·Mi:dm처럼 실제 `LICENSE` 파일을
  받아 비교한 모델에는 "라이선스 원문 검증", upstream 저장소에 원문이 없는
  SmolLM2에는 "metadata만 확인" 표시를 붙입니다. 모든 HuggingFace 검색 결과는
  uploader가 제공한 metadata임을 uploader 링크와 함께 밝히며, gated 저장소는
  사용할 수 없다고 표시합니다. `docs/RESULTS.md`의 과거 측정값은 명시된
  하드웨어에서 측정한 기록으로만 보여 줍니다. Parser를 만들기 전에 실제
  HuggingFace `/api/models` 응답 구조를 확인했습니다.

- **`fituna help [topic]`** — 지정한 명령 이름에 대해 argparse와 같은 도움말을
  출력합니다(`fituna help run`). topic이 없으면 최상위 도움말을 보여 줍니다.
  알 수 없는 topic은 유효한 이름 목록과 함께 종료 코드 2로 끝나며, 잘못된
  명령에 대한 argparse의 종료 코드와 같습니다.

- **산출물 중심 결과 출력** — 탐색이 이미 만든 양자화 `.gguf`의 경로와 크기를
  결과의 첫 항목으로 보여 줍니다. 이어서 `llama-server` 명령(OpenAI 호환 로컬
  API), Ollama, 대화형 점검 도구인 `llama-cli` 순서로 사용법을 안내합니다.
  `report.build_server_command()`는 `llama-cli`와 같은 방식으로
  `llama-server`의 **위치만 찾고 실행하지 않습니다.** 설치되어 있지 않으면
  명령 이름만 사용하며 그 사실을 알립니다.
- **`fituna run --export-ollama`** — `.gguf` 옆에 Ollama `Modelfile`
  (`FROM ./<gguf>` + 실측 `num_gpu`/`num_ctx`)을 atomic 방식으로 씁니다.
  `FROM`은 상대경로라 `--out` 디렉터리를 옮길 수 있습니다. PARAMETER 이름은
  기억에 의존하지 않고 2026-08-02에 Ollama 문서에서 확인했습니다.
- **`run --json`에 `llama_server_command`와 `modelfile_path` 추가** —
  `--export-ollama`를 주지 않으면 후자는 `null`입니다. 기존 필드의 이름과 구조는
  모두 유지한 하위 호환 추가입니다.

## [0.1.0] — 2026-07-30

첫 공개 릴리스입니다. 아래 내용은 이 시점까지 저장소 이력에 기록된 작업입니다.
아직 PyPI에 배포하지 않았으므로 소스에서 설치해야 합니다(`pip install -e .`).

**수정** 항목 중 "실제 하드웨어에서 발견"이라고 표시한 문제는 test suite가
잡아내지 못했습니다. Test suite는 llama.cpp와 network 없이 실행되도록
subprocess 계층을 의도적으로 모의 처리하므로, 실제 llama.cpp 바이너리와 모델로
실행했을 때만 이 버그들이 드러났습니다. 이 검증 단계를 CI와 분리한 이유는
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)에 설명합니다.

### 추가

- **2단계 실측 탐색**(`fituna run`) — 모든 후보를 양자화하고 F16 기준 대비
  perplexity 손실을 측정합니다. 품질 예산을 넘는 후보는 제외한 뒤, 통과한 후보를
  *실측* 품질 순으로 순회합니다. 각 후보를 full offload로 벤치마크하고 처리량
  목표를 만족한 첫 quant에서 최소 `-ngl`을 이진탐색합니다.
- **CLI** — `run`, `detect-hw`, `list-binaries` 하위 명령, 일반·`--json`
  출력, 실패 종류별 종료 코드(0 성공, 1 일반 오류, 2 바이너리 없음, 3 가능한
  설정 없음)를 제공합니다. 종료 코드 2는 argparse의 사용법 오류와 공유합니다.
  필수 flag가 없거나 알 수 없는 flag도 2이며, stderr 첫 줄이 `usage:`로
  시작하는지 보고 바이너리 누락과 구분할 수 있습니다
  (`docs/ARCHITECTURE.md`의 종료 코드 표 참조).
- **하드웨어 자동 감지** — NVIDIA(`nvidia-smi`), AMD(`rocm-smi`), Apple
  Silicon 통합 메모리(`system_profiler`)를 감지하며 `--gpu`·`--vram-mb`로
  직접 지정할 수도 있습니다.
- **llama.cpp 바이너리 탐색** — `--llama-bin-dir` 또는 `PATH`에서 찾아 기능과
  빌드 버전을 확인합니다.
- **GGUF header 해석** — 표준 라이브러리 `struct`로 layer 수와 파일 형식을
  읽고, 사용할 수 있으면 `convert_hf_to_gguf.py`로 HF 디렉터리를 변환합니다.
- **`llama-quantize` wrapper** — 멱등이며 임시 파일을 rename하는 atomic
  방식으로 동작하고 출력 파일명에 모델 fingerprint를 넣습니다.
- **`llama-perplexity` wrapper** — corpus 평가량을 제한하는 `--ppl-chunks`
  옵션을 제공합니다(기본값 32).
- **`llama-bench` wrapper** — timeout을 처리하고 `-d/--n-depth`로 context를
  모사합니다.
- **sqlite3 결과 cache** — 모델 fingerprint × 하드웨어 profile × llama.cpp
  빌드 버전을 key로 사용하며 `--resume`을 지원합니다.
- **보고 계층** — 사람이 읽는 결과와 JSON 결과, 바로 실행할 수 있는
  `llama-cli` 명령을 만듭니다.
- **`fituna doctor`** — Python 버전, 필수·권장 바이너리, llama.cpp 버전,
  하드웨어 감지, 출력 디렉터리 쓰기 권한, 남은 디스크 공간을 먼저 진단합니다.
  각 항목은 해결 방법 한 줄과 함께 PASS/WARN/FAIL로 표시하며 일반·`--json`
  출력을 지원합니다.
- **`fituna fetch-corpus`** — 표준 라이브러리 `urllib`만 사용해 HuggingFace
  공개 dataset-viewer REST API에서 품질 corpus를 받습니다(`datasets`,
  pyarrow, pandas 미사용). 영어(wikitext-2-raw)·한국어(한국어 Wikipedia)
  프리셋과 `--dataset/--config/--split` 재정의, atomic 쓰기를 지원하고 성공 시
  CC BY-SA 저작자 표시를 출력합니다.
- **`--quality-corpus`** — `--wikitext`를 alias로 유지하면서 실제 작업과 닮은
  텍스트로 품질 gate를 측정할 수 있게 했습니다.
- **MCP server**(`fituna-mcp`) — 의존성 0개 보장을 지키려고 protocol을 직접
  구현한 stdio 기반 줄 단위 JSON-RPC 2.0 server입니다.
  `fituna_detect_hardware`와 `fituna_recommend`를 제공합니다.
- **모듈별 자체 점검** — `python -m fituna.<module>`로 실행할 수 있으며 CI가
  test suite와 함께 실행합니다.
- **단위 테스트 152개와 CI matrix** — subprocess·network 계층을 모의 처리하고
  3개 OS × Python 2개 버전(Ubuntu/macOS/Windows × 3.11/3.13)에서 실행합니다.
- **이미 양자화된 GGUF를 `--model`로 지정할 때 경고** — 이중 양자화와 잘못된
  기준값 사용을 막습니다.
- **문서** — 아키텍처, 실행 간 변동성과 발열 저하 이상치를 포함한 Run 1~5 실측
  결과, 사용 사례, 심사위원용 한국어 재현 가이드(`REVIEWERS.md`), AI 활용 개발
  공개, 오픈소스 활용 명세, 라이선스 준수 기록, SBOM, 제3자 고지, 모든 Python
  파일의 SPDX header와 `REUSE.toml`을 추가했습니다.
- **Colab notebook** — 무료 T4 환경에서 NVIDIA/Linux 결과를 한 번에 재현합니다.
- **커뮤니티 파일** — `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `REVIEWERS.md`, 이슈와 pull request template을 추가했습니다.

### 수정

- **`llama-bench` timeout 하나가 전체 탐색을 중단하던 문제** — 실제
  하드웨어에서 발견했습니다. 최소 `ngl` 이진탐색은 `ngl=0`도 측정하는데, 4B
  모델의 CPU 전용 실행은 300초 안에 끝나지 않아 이미 20분 넘게 수행한 탐색까지
  오류로 종료됐습니다. 한 번의 bench도 마치지 못할 만큼 느리다는 사실 자체가
  해당 설정의 결과이므로, 이제 timeout을 잡아 목표 미달 0 tok/s로 기록하고
  cache에 넣어 `--resume`에서 다시 기다리지 않습니다.
- **최신 빌드에서 llama.cpp 버전 감지가 조용히 실패하던 문제** — 실제
  하드웨어에서 발견했습니다. `llama-bench`가 `--version`을 거부하고 빌드
  banner도 출력하지 않아 감지 결과가 항상 `None`이었고, 버전별 cache의 모든
  결과가 `"unknown"` 아래 저장됐습니다. 이 때문에 다른 backend 빌드의 측정값을
  `--resume`이 내주지 않는다는 보장이 무너졌습니다. 이제
  `llama-perplexity`도 확인하고 단순한 `llama.cpp <build>` banner 형식도
  허용합니다.
- **품질 cache key에 corpus가 없던 문제** — 영어·한국어 비교를 준비하다 실제
  하드웨어에서 발견했습니다. Perplexity는 (model, quant, corpus)의 속성인데
  key가 (model, quant, chunk count)여서, `--resume` 상태에서
  `--quality-corpus`를 바꾸면 이전 corpus의 값이 새 결과로 나왔습니다. Corpus
  fingerprint를 key에 추가했습니다. Upgrade 전 cache는 migration하지 않고 품질
  table을 지운 뒤 다시 측정합니다.
- **`llama-bench`에 `-c/--ctx-size` flag가 없던 문제** — 실제 하드웨어에서
  발견했습니다. 가정했던 인자는 최신 빌드에서 parse 오류를 냈습니다. 이제
  context를 `-d/--n-depth`로 매핑해 시간 측정 구간에서 KV cache가 요청한
  context와 비슷하게 차도록 합니다.
- **Perplexity가 test set 전체를 평가하던 문제** — chunk 제한 없이
  `llama-perplexity`를 실행해 실제 하드웨어에서 후보 4개 탐색에 3시간 44분이
  걸렸습니다. `--ppl-chunks`(기본값 32)를 추가해 같은 탐색을 약 12분으로
  줄였습니다.
- **`--resume`이 다른 chunk 수로 계산한 perplexity를 반환할 수 있던 문제** —
  `ppl_chunks`를 품질 cache key에 추가했습니다.
- **같은 `--out`을 쓰는 FiTuna process 둘이 동일한 양자화 임시 파일에 쓸 수
  있던 문제** — PID suffix를 추가하고 임시 파일 정리를 glob으로 넓혀 이전
  비정상 종료가 남긴 파일까지 치웁니다.
- **서로 다른 모델이 한 양자화 출력 경로에서 충돌할 수 있던 문제** — 모든 HF
  디렉터리 입력이 관례상 같은 기반 파일명으로 변환됐습니다. 이제 출력 파일명에
  모델 fingerprint를 넣습니다.
- **비정상 종료한 `llama-quantize`가 잘린 파일을 남겨 cache처럼 보이던 문제** —
  임시 경로에 쓴 뒤 atomic rename합니다.
- **파일은 있지만 실행할 수 없는 `llama-quantize`·`llama-perplexity`가 원시
  `PermissionError` traceback으로 종료되던 문제** — `quantize.py`와
  `quality.py`가 `FileNotFoundError`뿐 아니라 `OSError`를 처리하고, 메시지에서
  "설치되지 않음"과 "파일은 있지만 손상됨"을 구분합니다.
- **cache 경로의 손상됐거나 sqlite가 아닌 파일이 원시
  `sqlite3.DatabaseError` traceback을 내던 문제** — 구체적인 복구 안내를 담은
  `FiTunaError`로 바꿨습니다. 잘못된 `--ctx` 값도 같은 방식으로 고쳤습니다.
- **최선 결과 fallback이 CPU 전용 bench를 `ngl=n_layers`로 보고할 수 있던
  문제** — 기록하기 전에 `ngl=0`으로 바로잡습니다.
- **첫 실제 Windows 실행에서 CI 두 job이 모두 실패한 문제** —
  `pytest.raises` match pattern에 들어간 Windows 경로의 `\U`가 잘못된 정규식
  escape였습니다. 이제 pattern에 `re.escape`를 적용합니다.
- **`fituna doctor`의 진단 loop 자체가 예외로 끝날 수 있던 문제** — 각 점검을
  따로 보호해 한 항목의 실패가 도구 전체를 종료하지 않고 해당 FAIL 행만 만듭니다.

### 변경

- **Run 5의 corpus 순위 변경 결론 철회** — 한국어 corpus가 중간 quant의 순서를
  바꾼다는 공개 측정 결론은 재검토를 통과하지 못했습니다. 두 실행이 독립적이지
  않고 중첩됐으며, chunk별 추적에서 한국어 margin의 부호가 아홉 번 바뀌었습니다.
  해당 결론을 철회하고 두 margin 모두 오차 범위와 함께 제시합니다. Run 3과 4의
  판정 역전은 철회하지 않았지만 본문에 조건을 명시했습니다. 자세한 내용은
  [docs/RESULTS.md](docs/RESULTS.md#how-big-is-a-perplexity-gap-the-error-bar-we-had-been-discarding)를
  참고하세요.
- **AI 활용 문서의 잘못된 모델 라이선스 주장 수정** — 오픈소스 활용 문서를 실제
  저장소 상태와 맞추고 라이선스 준수 기록의 오래된 수치도 고쳤습니다.
- `--wikitext`를 `--quality-corpus`로 바꿨습니다. 이전 이름을 alias로
  유지하므로 호환성을 깨는 변경은 아닙니다.

[0.3.1]: https://github.com/leeyunseokarchive/fituna/releases/tag/v0.3.1
[0.3.0]: https://github.com/leeyunseokarchive/fituna/releases/tag/v0.3.0
[0.2.1]: https://github.com/leeyunseokarchive/fituna/releases/tag/v0.2.1
[0.2.0]: https://github.com/leeyunseokarchive/fituna/releases/tag/v0.2.0
[0.1.0]: https://github.com/leeyunseokarchive/fituna/releases/tag/v0.1.0
