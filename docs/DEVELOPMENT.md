# 개발 및 검증 방법

이 문서는 FiTuna를 실제로 개발하고 검증하는 방식을 설명합니다. 여기 적힌 내용은
저장소가 이미 따르는 관행이며 코드, CI 설정, pull request 이력에서 확인할 수
있습니다. 아직 실행하지 않은 계획은 (계획)으로 표시합니다.

코드가 *무엇을 하는지*는 [ARCHITECTURE.md](ARCHITECTURE.md), 개발 환경과 기여
방법은 [CONTRIBUTING.md](../CONTRIBUTING.md), 릴리스 이력은
[CHANGELOG.md](../CHANGELOG.md)를 참고하세요.

## 1. 계약 우선 설계

모듈 경계를 넘는 모든 데이터 구조는 [`fituna/config.py`](../fituna/config.py)의
불변 dataclass 또는 enum으로 정의합니다: `HardwareProfile`, `TargetSpec`,
`BinaryPaths`, `ModelInfo`, `CandidateConfig`, `BenchResult`, `QualityResult`,
`SearchResult`, `DoctorCheck`, `CorpusPreset`. 이 모듈이 interface의 단일 기준
정보원이며, 주변 모듈(`hardware`, `binaries`, `quantize`, `quality`, `bench`,
`search`, `cache`, `report`)은 이 type만 주고받습니다.

`frozen=True`는 의도한 설정입니다. 측정하고 cache에 저장해 보고까지 마친 결과가
pipeline 뒤쪽에서 바뀌어서는 안 됩니다. `config.py` 자체 점검은
`HardwareProfile`이나 `SearchResult`를 수정할 때 조용히 성공하지 않고 예외가
나는지 확인합니다. 불변성은 관례가 아니라 검사하는 속성입니다.

따라서 `CONTRIBUTING.md`에 명시했듯 모듈 간 데이터 구조를 바꾸려면
`config.py`의 dataclass와 모든 사용처를 같은 pull request에서 함께 고쳐야
합니다.

## 2. 모듈별 자체 점검

`fituna/`의 모듈 17개는 각각 단독 실행할 수 있고 핵심 불변 조건을 assertion으로
검사합니다. `.py` 파일 18개 중 세 줄짜리 `__main__.py` shim만 제외한 수입니다.

```bash
python -m fituna.config          # 불변 dataclass 계약
python -m fituna.search          # 탐색 순서와 조기 종료 논리
python -m fituna.cli --selfcheck # entry point는 명시적 flag 사용
```

`__main__`이 실제 entry point인 `cli`와 `mcp_server`는 명시적 `--selfcheck`
flag가 있을 때만 점검하므로 `python -m fituna.cli`는 여전히 CLI를 실행합니다.
`doctor`와 `corpus`를 포함한 나머지 모듈은 `if __name__ == "__main__":`
아래에서 인자와 관계없이 점검합니다. CI는 `doctor`와 `corpus`에도
`--selfcheck`를 전달합니다. 동작에는 영향이 없고 인자를 받는 네 모듈의 호출
형식을 통일합니다. 자체 점검은 별도 test framework가 아니라 검사 대상 코드
옆에 둔 assertion입니다. pytest가 없는 환경에서도 모듈 하나를 격리해 확인하려고
만들었습니다.

이는 선택 사항이나 참고 정보가 아닙니다. [CI](../.github/workflows/ci.yml)는
matrix의 모든 OS·Python 버전에서 test suite와 자체 점검 17개를 필수 단계로
실행합니다. 자체 점검 하나라도 실패하면 빌드도 실패합니다.

## 3. Subprocess를 모의 처리한 단위 테스트

`tests/`의 suite는 package 전반을 다루며 **llama.cpp가 설치되지 않고 network에
접속할 수 없는 컴퓨터**에서도 통과하도록 설계했습니다. 모든 외부 효과는 경계에서
monkeypatch합니다.

- 하드웨어 확인 명령(`nvidia-smi`, `rocm-smi`, `system_profiler`)은
  `subprocess.run` 자체를 대체합니다. 실행 파일을 찾지 못한 경우를 포함한 기록된
  출력으로 parser를 구동합니다.
- `fituna fetch-corpus`는 `urllib.request.urlopen`을, atomic 쓰기 경로는
  `tempfile.mkstemp`를 대체합니다.
- orchestration을 검사할 때는 wrapper 함수를 대체합니다. `search`에는 가짜
  `quantize`·`bench`·`quality`를, `doctor`에는 가짜 `binaries.find_exe`,
  `get_llama_cpp_version`, `detect_hardware`, `shutil.disk_usage`,
  `os.access`를 넣습니다.

실제 도구 출력은 fixture로 보존합니다. `tests/fixtures/llama_bench_sample.json`은
`fituna/bench.py`의 `_self_check()`와 `tests/test_bench.py`가 함께 읽고(#46),
`tests/fixtures/llama_perplexity_kld_sample.txt`(llama.cpp b11342에서 캡처)는
`tests/test_quality.py`가 읽습니다(#45). 그래서 이 두 parser는 임의로 만든
문자열이 아니라 llama.cpp가 실제로 출력한 텍스트로 `pytest -q` 안에서
검사됩니다. 새 fixture는 실제 실행에서 캡처하고, 어느 llama.cpp 빌드에서
얻었는지 test에 적어 둡니다.

이 선택에는 분명한 대가가 있습니다. Suite 안의 llama.cpp는 예상한 대로 움직이는
mock이므로 **FiTuna와 실제 llama.cpp 사이의 통신 오류를 잡을 수 없습니다.**
0.1.0 변경 이력의 flag·protocol 버그, 곧 `llama-bench`에 `-c`가 없고
`--version`을 거부하며 CPU 전용 bench가 끝나지 않던 문제는 모두 전체 suite를
통과했습니다. #45의 KLD 해석도 같은 경우입니다. mock 출력이 llama.cpp가 실제로
출력하지 않는 형식이었는데 suite와 CI는 모두 통과했고, 병합 전 실제 llama.cpp로
실행해서야 발견했습니다. 이후 실제 출력을 fixture로 캡처해 test가 그것을 읽게
했습니다. 이를 보완하는 절차가 6절이며, test를 더 추가하는 대신 별도 단계로
둔 이유입니다.

Suite가 잘 다루는 범위는 subprocess 경계 위쪽입니다. 탐색 순서와 조기 종료,
cache key와 schema migration, 기록된 `nvidia-smi`·`rocm-smi`·
`system_profiler` 텍스트를 이용한 하드웨어 출력 해석(`test_hardware.py`),
하드웨어 감지 fallback, 오류 매핑과 종료 코드, 운영체제별 경로 처리를 검사합니다.
`llama-bench`와 `llama-perplexity` 출력 해석은 `test_bench.py`와
`test_quality.py`가 subprocess를 mock한 채 위 fixture로 검사합니다. 반면
`quantize.py`와 `binaries.py`(`llama-quantize --help`로 지원 양자화 타입을 알아내는
부분 포함)는 아직 pytest에서 직접 실행하지 않습니다([#10](https://github.com/leeyunseokarchive/fituna/issues/10)). `test_report.py`는
`report.py`의 순수한 부분인 명령 생성, Modelfile export, rendering만 검사하며
원래 subprocess를 시작하지 않습니다. `test_cli.py`는
`search.search()`와 `binaries.locate_binaries()`를 가짜로 바꾼 뒤
`cli.main()`을 end-to-end로 구동합니다. 그래서 argparse만 검사할 때 놓치는
`--export-ollama`와 `report.export_ollama_modelfile()` 사이의 연결 오류 등을
잡습니다. Subprocess wrapper 중 `model_info.py`만 예외입니다.
`test_config.py`가 `is_already_quantized` guard를 검사하지만 GGUF header
해석은 다루지 않습니다. 나머지 parser(`quantize.py`, `binaries.py`)는 2절의
모듈별 자체 점검으로만 검사합니다.

## 4. CI matrix: OS 3개 × Python 2개

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)은 모든 pull request와
`main` push에서 실행됩니다.

| | Python 3.11 | Python 3.13 |
|---|---|---|
| ubuntu-latest | ✅ | ✅ |
| macos-latest | ✅ | ✅ |
| windows-latest | ✅ | ✅ |

Job은 6개이며 `fail-fast: false`라 한 platform의 실패가 다른 platform 결과를
가리지 않습니다. 각 job은 `pip install -e .`과 pytest로 package를 설치한 뒤
`pytest -q`와 모듈 자체 점검 17개를 실행합니다. FiTuna가 선언하지 않은 것을
import하면 설치 단계에서 실패하므로 이 설치 자체도 런타임 의존성 0개 검사입니다.

3.11은 `pyproject.toml`에 선언한 최저 버전이고 3.13은 현재 릴리스입니다.
FiTuna가 경로와 도구 출력을 처리하며 이 둘이 Windows에서 다르기 때문에 matrix에
Windows를 넣었습니다. 첫 실제 Windows CI 실행은 Windows 경로의 `\U`가 잘못된
정규식 escape라서 생긴 test 실패를 잡았습니다.

의도적으로 CI에 넣지 않은 항목은 linter, formatter, type checker, coverage
gate입니다. Package는 `py.typed`를 포함하고 type annotation을 사용하지만 현재
이를 검사하는 단계는 없습니다 **(계획)**.

## 5. 브랜치 → pull request → 검토 → 병합

작업은 `main`이 아니라 topic branch(`feat/…`, `fix/…`, `docs/…`,
`chore/…`)에서 진행합니다. 각 branch는 pull request가 되고 `main`은 merge
commit을 통해서만 바뀝니다. PR #1~#6이 이 방식으로 병합된 이력이 있습니다.

FiTuna는 1인 프로젝트라 여기서 review는 병합 전 pull request에 정식 review로
남기는 서면 자체 검토를 뜻합니다. 검토 결과 수정한 내용은 같은 branch에 별도
commit으로 push합니다. 무엇을 추정하지 않고 확인했는지 review 본문에 적고 후속
commit을 추적할 수 있게 기록을 남기는 것이 목적입니다. 실제 이력은 다음과
같습니다.

- PR #1 (`fituna doctor`) → `fix: address doctor review findings (Windows tests,
  dual binary path, naming)` and `fix: close doctor's never-raises gaps`
- PR #2 (`fetch-corpus`) → `fix: address fetch-corpus review findings
  (os.replace guard, rows<=0, docs)` and a final cleanup pass
- PR #6(Run 5) → review 세 차례에서 앞선 검토의 과장된 결론을 차례로 찾아내고,
  공개했던 측정 결론을 최종 **철회**

마지막 사례가 review 기준을 보여 줍니다. 변경을 확인만 하는 review는 의미가
없습니다. PR #6은 chunk별 추적이 근거를 뒷받침하지 못하자 스스로 핵심 결과를
철회했고, PR #4는 이미 공개한 문서의 잘못된 모델 라이선스 주장을 고쳤습니다.
해당 branch 안에서 고치기에는 너무 큰 문제도 조용히 버리지 않고 review와 issue
이력에 남깁니다. 현재 사례는 `quality.py`가 `llama-perplexity` 출력의 `±` 표준
오차를 버리는 문제입니다. 이를 저장하려면 모든 공개 수치가 참조하는 cache가
무효화되므로 우연히 끼워 넣을 수정이 아니라 명시적인 측정 데이터 migration이
필요합니다.

문서의 실측 수치는 log 한 줄이나 cache 행까지 추적할 수 있어야 합니다. 근거를
추적할 수 없는 주장은 표현을 약하게 바꾸지 않고 삭제합니다.

## 6. 실제 하드웨어 end-to-end 검증

Test suite는 의도적으로 llama.cpp를 실행할 수 없으므로(3절), subprocess 경계를
건드리는 변경은 실제 바이너리와 모델로 별도 검증합니다. GitHub runner에는 GPU와
llama.cpp 빌드가 없어 이 단계는 수동이며 CI에 포함하지 않습니다.

한 번의 검증 실행은 다음을 포함합니다.

1. 실제 llama.cpp 빌드에 `fituna doctor`를 실행하고 `fituna list-binaries`로
   감지한 빌드 버전 확인
2. 실제 quantize·bench·perplexity subprocess로 전체 `fituna run`을 성공
   결과(종료 코드 0)까지 실행
3. 두 번째 실행에서 `--resume` cache hit 경로 확인
4. 바이너리 누락(종료 코드 2)과 가능한 설정 없이 최선 결과를 보고하는 실패
   경로(종료 코드 3) 확인

실행 log, 시간, 하드웨어는 [RESULTS.md](RESULTS.md)에 기록합니다. 실행 간
변동성과 버리지 않고 남긴 발열 저하 이상치도 포함해 독자가 수치를 그대로 믿는
대신 확인할 수 있게 했습니다. 현재 실제 platform 검증 범위는 macOS(Apple
Silicon/Metal)와 Linux(NVIDIA Tesla T4/CUDA)입니다. Linux 결과는
[Colab notebook](../notebooks/colab_nvidia_verification.ipynb)으로 누구나 재현할
수 있습니다. Windows는 단위 테스트와 CI를 통과하지만 실제 바이너리 검증은 하지
않았습니다. 지원한다고 과장하지 않고 README의 알려진 한계에 명시합니다.

[REVIEWERS.md](../REVIEWERS.md)는 제3자가 공개 결과를 재현할 수 있도록 이
절차를 심사용으로 정리한 문서입니다.

## 7. 의존성과 라이선스

런타임 의존성 0개는 선호가 아니라 반드시 지키는 제약입니다. 개발 의존성은
`pytest` 하나뿐입니다. MCP server도 SDK 대신 stdio 기반 JSON-RPC를 직접
구현했고 corpus 다운로드도 `datasets`가 아닌 `urllib`를 사용합니다. CI의
`pip install -e .` 단계가 매번 의존성 선언을 검사합니다.

라이선스는 추정하지 않고 추적합니다. 모든 Python 파일의 SPDX header,
`REUSE.toml`, FiTuna가 호출하는 도구와 실측에 사용한 corpus·모델을 다루는 준수
기록을 마련했습니다. 다음 문서를 참고하세요.

[LICENSE_COMPLIANCE.md](LICENSE_COMPLIANCE.md),
[SBOM.md](SBOM.md),
[OPEN_SOURCE_USAGE.md](OPEN_SOURCE_USAGE.md),
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
AI를 활용한 개발 내역은 [AI_MODEL_USAGE.md](AI_MODEL_USAGE.md)에 공개합니다.
