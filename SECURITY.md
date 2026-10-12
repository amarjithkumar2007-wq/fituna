# 보안 정책

## 지원 버전

FiTuna는 아직 1.0 이전 단계입니다. 최신 릴리스(`0.2.0`)와 `main`만 지원하며,
수정 사항은 `main`에 반영한 뒤 다음 릴리스에 포함합니다.

## 취약점 신고

GitHub의 비공개 취약점 신고 기능을 이용해 주세요:
[**Security → Report a vulnerability**](https://github.com/leeyunseokarchive/fituna/security/advisories/new).
이 기능을 이용할 수 없다면 제목에 `[fituna security]`를 넣어 메인테이너
`dbstjr3576@gmail.com`에게 이메일을 보내 주세요.

실행한 FiTuna 명령, OS, llama.cpp 빌드(`fituna doctor --json`으로 둘 다 확인
가능), 공격자가 얻을 수 있는 권한이나 정보를 포함해 주세요. 개인이 자원봉사로
관리하는 프로젝트이므로 접수 확인까지 최대 일주일이 걸릴 수 있습니다. 수정본이
나오기 전에는 취약점을 공개 이슈로 등록하지 말아 주세요.

## 신뢰 경계

FiTuna는 추론이나 양자화를 직접 수행하지 않습니다. llama.cpp 바이너리를
subprocess로 실행하고 그 출력을 해석합니다. 바이너리 경로는 사용자가 지정한
`--llama-bin-dir` 또는 `PATH`에서 찾은 `llama-quantize`, `llama-bench`,
`llama-perplexity`이며, FiTuna는 이들을 사용자 권한으로 실행합니다. 사용자가
지정한 모델 파일도 읽습니다. `fituna fetch-corpus`는 고정된 호스트
`https://datasets-server.huggingface.co`에서 텍스트를 가져오며,
`--dataset`·`--config`·`--split`은 이 호스트 안에서 어떤 데이터셋을 받을지만
정합니다. 신뢰할 수 없는 바이너리나 디렉터리를 지정하면 해당 프로그램을 직접
실행하는 것과 같고, 검증하지 않은 데이터셋을 받으면 그 호스트의 신뢰할 수 없는
텍스트를 읽게 됩니다. FiTuna는 어느 쪽도 sandbox로 격리하거나 검증하지
않습니다. `--llama-bin-dir`, `PATH`, `--out` 디렉터리와
`--dataset`·프리셋으로 지정한 데이터셋은 사용자가 통제하는 입력으로 취급하세요.

가장 큰 노출 지점은 `--llama-bin-dir` 옵션 자체보다 이 옵션이 가리킬 수 있는
위치입니다. HF 형식 모델 디렉터리를 변환할 때
(`fituna/model_info.py:167-178`) `[sys.executable,
convert_hf_to_gguf.py, ...]`을 실행합니다. 즉 FiTuna와 같은 인터프리터가 임의의
Python 스크립트를 실행합니다. `fituna/binaries.py`의 `_find_script`는 이
스크립트를 `bin_dir`(사용자가 지정한 `--llama-bin-dir`)뿐 아니라 두 단계 위인
`bin_dir.parent.parent`에서도 찾습니다. 공격자가 쓸 수 있는
`convert_hf_to_gguf.py`보다 두 단계 아래를 `--llama-bin-dir`로 지정하면 해당
스크립트가 실행될 수 있습니다. 이 fallback도 신뢰 경계에 포함됩니다.
`--llama-bin-dir`부터 두 단계 위 디렉터리까지 모두 사용자가 통제해야 합니다.

FiTuna는 `llama-cli`나 `llama-server`를 실행하지 않습니다. `fituna doctor`와
`report.py`는 사용 가능 여부를 알려 주려고 `PATH` 또는 `--llama-bin-dir`에서
경로만 찾습니다. `fituna run`의 일반 출력과 `--json` 출력에는 사용자가 나중에
직접 실행할 `llama-cli ...`, `llama-server ...` 명령 문자열만 담깁니다
(`report.py`의 `build_run_command`, `build_server_command`).
`--export-ollama`도 생성된 `.gguf` 옆에 `Modelfile`을 쓸 뿐이며
(`report.py`의 `export_ollama_modelfile`), FiTuna가 `ollama`를 실행하지는
않습니다.

이 경계 안에서 발생하는 버그, 곧 FiTuna가 신뢰할 수 없는 경로·모델 파일·스크립트·
다운로드한 corpus를 잘못 처리해 위에서 설명한 범위를 넘는 권한을 내주는 문제는
보안 신고 대상입니다.
