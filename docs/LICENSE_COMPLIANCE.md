<!-- REUSE-IgnoreStart -->
<!--
  이 문서는 예시 출력으로 `SPDX-License-Identifier:` 줄을 인용한다.
  이 marker가 없으면 `reuse lint`(§3.5)가 주변 설명까지 이 파일의 license
  expression으로 해석해 invalid로 보고한다. 즉 문서가 자신이 공개한 scan 결과를
  오염시키게 된다. 이 marker는 인용한 tag에 대해 `reuse`가 문서화한 해결책이다.
-->

# 라이선스 준수 근거와 재현 방법

이 문서는 **라이선스 준수 근거**를 제시합니다. FiTuna가 사용하는 구성요소를
반복해서 나열하지 않습니다. 부문별 구성요소, 1차 출처로 확인한 라이선스,
결합 방식은 `docs/OPEN_SOURCE_USAGE.md`가 기준 문서입니다. 여기서는 라이선스
검증에서 실제로 묻는 네 가지 질문에 답합니다.

1. **배포물에는 무엇이 들어 있는가?** — 실제 빌드로 증명합니다.
2. **결합한 라이선스가 충돌하는가?** — 구성요소별 결합 방식으로 판단합니다.
3. **Scanner의 결과는 무엇인가?** — 실제 도구, 명령, 출력을 제시합니다.
4. **Scanner가 자체 파일을 분류할 수 있는가?** — SPDX 식별자를 추가했습니다.

아래 명령은 **2026-07-30**에 macOS 26.5.1(arm64, build 25F80), `Python
3.13.7 (main, Aug 14 2025, 11:12:11) [Clang 17.0.0]` 환경에서
`docs/license-compliance` branch의 당시 저장소 root를 기준으로 실행했습니다.
**문서의 출력을 손으로 작성하지 않았습니다.** `$` 뒤 block은 실제 실행에서
복사했습니다. 사실과 다른 인상을 주지 않도록 표현만 두 곳 통일했습니다. 실제로는
session 임시 디렉터리를 쓴 일회용 virtualenv 경로를 그대로 실행 가능한
`/tmp/buildenv`, `/tmp/runtimeenv`로 표시했고, 긴 빌드 log 한 곳은 `...`로
생략했다고 밝혔습니다. 도구를 사용할 수 없고 설치 비용이 지나치게 크면 그 사실과
대체 검사의 범위·한계를 §3.1과 §6에 명시합니다.

---

## 1. 실제 배포물의 구성

### 1.1 빌드

빌드 도구는 **저장소 밖의** 일회용 virtualenv에 설치하므로 어떤 배포물의
의존성도 되지 않습니다.

```
$ python3.13 -m venv /tmp/buildenv
$ /tmp/buildenv/bin/pip install build
$ /tmp/buildenv/bin/python -m build
...
Successfully built fituna-0.1.0.tar.gz and fituna-0.1.0-py3-none-any.whl
```

`build` 1.5.0은 `pyproject.toml`의 `[build-system]` 선언 그대로 자체 격리
환경에 `setuptools>=77`을 준비합니다. 설치한 package에는 `build`와 `setuptools`
어느 것도 들어가지 않습니다. §1.6에서 wheel을 빈 환경에 설치한 뒤 들어온 항목을
나열해 이를 증명합니다.

생략한 `...`는 setuptools 자체 build log입니다. 이제 license 선언에 관한
`SetuptoolsDeprecationWarning`은 없습니다. `pyproject.toml`이 PEP 639 표현식
`license = "MIT"`를 직접 선언하도록 바꾸고 이를 위해 `setuptools>=68`에서
`setuptools>=77`로 올렸으므로, 경고를 내던 deprecated 형식
`license = { text = "MIT" }`는 사라졌습니다. §6은 미해결 경고가 아니라 이 변경의
trade-off를 기록합니다.

### 1.2 sdist 전체 목록

```
$ tar -tzf dist/fituna-0.1.0.tar.gz | sort
fituna-0.1.0/
fituna-0.1.0/LICENSE
fituna-0.1.0/PKG-INFO
fituna-0.1.0/README.md
fituna-0.1.0/fituna.egg-info/
fituna-0.1.0/fituna.egg-info/PKG-INFO
fituna-0.1.0/fituna.egg-info/SOURCES.txt
fituna-0.1.0/fituna.egg-info/dependency_links.txt
fituna-0.1.0/fituna.egg-info/entry_points.txt
fituna-0.1.0/fituna.egg-info/requires.txt
fituna-0.1.0/fituna.egg-info/top_level.txt
fituna-0.1.0/fituna/
fituna-0.1.0/fituna/__init__.py
fituna-0.1.0/fituna/__main__.py
fituna-0.1.0/fituna/bench.py
fituna-0.1.0/fituna/binaries.py
fituna-0.1.0/fituna/cache.py
fituna-0.1.0/fituna/cli.py
fituna-0.1.0/fituna/config.py
fituna-0.1.0/fituna/corpus.py
fituna-0.1.0/fituna/doctor.py
fituna-0.1.0/fituna/errors.py
fituna-0.1.0/fituna/hardware.py
fituna-0.1.0/fituna/mcp_server.py
fituna-0.1.0/fituna/model_info.py
fituna-0.1.0/fituna/py.typed
fituna-0.1.0/fituna/quality.py
fituna-0.1.0/fituna/quantize.py
fituna-0.1.0/fituna/quickstart.py
fituna-0.1.0/fituna/report.py
fituna-0.1.0/fituna/search.py
fituna-0.1.0/pyproject.toml
fituna-0.1.0/setup.cfg
fituna-0.1.0/tests/
fituna-0.1.0/tests/test_cache.py
fituna-0.1.0/tests/test_cli.py
fituna-0.1.0/tests/test_config.py
fituna-0.1.0/tests/test_corpus.py
fituna-0.1.0/tests/test_doctor.py
fituna-0.1.0/tests/test_hardware.py
fituna-0.1.0/tests/test_quickstart.py
fituna-0.1.0/tests/test_report.py
fituna-0.1.0/tests/test_search.py
```

### 1.3 wheel 전체 목록

```
$ unzip -Z1 dist/fituna-0.1.0-py3-none-any.whl | sort
fituna-0.1.0.dist-info/METADATA
fituna-0.1.0.dist-info/RECORD
fituna-0.1.0.dist-info/WHEEL
fituna-0.1.0.dist-info/entry_points.txt
fituna-0.1.0.dist-info/licenses/LICENSE
fituna-0.1.0.dist-info/top_level.txt
fituna/__init__.py
fituna/__main__.py
fituna/bench.py
fituna/binaries.py
fituna/cache.py
fituna/cli.py
fituna/config.py
fituna/corpus.py
fituna/doctor.py
fituna/errors.py
fituna/hardware.py
fituna/mcp_server.py
fituna/model_info.py
fituna/py.typed
fituna/quality.py
fituna/quantize.py
fituna/quickstart.py
fituna/report.py
fituna/search.py
```

**총 25개 파일입니다. FiTuna `.py` source 18개, 빈 PEP 561 `py.typed` marker
1개, build가 생성한 metadata 6개입니다. 이 중 하나는 wheel 형식이
`.dist-info/licenses/` 아래에 둔 FiTuna 자체 MIT `LICENSE`입니다.** vendor
directory, bundle한 binary, 제3자 `.py`, 다른 주체의 복사된 license 파일은
없습니다.

### 1.4 두 산출물에 제3자 코드가 없다는 증명

목록은 이름만 보여 줍니다. 아래 검사는 *출처*를 확인합니다. 두 산출물에서 자동
생성되지 않은 모든 파일은 **이 저장소의 git index가 추적하는 파일과 byte 단위로
동일**하므로 build 중 삽입된 항목이 없습니다.

```
$ python3.13 - <<'PY'
import subprocess, tarfile, zipfile, hashlib, pathlib
tracked = {p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
           for p in subprocess.run(["git","ls-files"],capture_output=True,text=True).stdout.split()}
GEN = ("PKG-INFO","SOURCES.txt","dependency_links.txt","entry_points.txt","requires.txt",
       "top_level.txt","setup.cfg",".dist-info/")
def audit(label, items):
    same=gen=unknown=0
    for name, data in items:
        rel = name.split("/",1)[1] if label=="sdist" else name
        if any(g in name for g in GEN): gen+=1; continue
        if rel in tracked and hashlib.sha256(data).hexdigest()==tracked[rel]: same+=1
        else: unknown+=1; print(f"  !! {label}: {name} is NOT a byte-identical tracked repo file")
    print(f"{label}: {same} byte-identical to tracked repo files, "
          f"{gen} build-generated metadata, {unknown} unaccounted for")
with tarfile.open("dist/fituna-0.1.0.tar.gz") as t:
    audit("sdist", [(m.name, t.extractfile(m).read()) for m in t.getmembers() if m.isfile()])
with zipfile.ZipFile("dist/fituna-0.1.0-py3-none-any.whl") as z:
    audit("wheel", [(n, z.read(n)) for n in z.namelist()])
PY
sdist: 27 byte-identical to tracked repo files, 8 build-generated metadata, 0 unaccounted for
wheel: 18 byte-identical to tracked repo files, 6 build-generated metadata, 0 unaccounted for
```

**두 산출물 모두 출처 불명 파일이 0개입니다.**

### 1.5 빌드한 wheel에서 다시 읽은 의존성 선언

```
$ unzip -p dist/fituna-0.1.0-py3-none-any.whl fituna-0.1.0.dist-info/METADATA \
    | grep -E '^(Name|Version|Requires-Python|License-Expression|License-File|Provides-Extra|Requires-Dist):'
Name: fituna
Version: 0.1.0
License-Expression: MIT
Requires-Python: >=3.11
License-File: LICENSE
Provides-Extra: dev
Requires-Dist: pytest; extra == "dev"
```

원래 `grep` pattern은 `^License:`를 찾았습니다. `setuptools>=77`과 PEP 639
`license = "MIT"` 표현식을 사용하면 wheel이 단순한 `License:` 줄 대신
`License-Expression: MIT`(Metadata-Version 2.4)를 내보냅니다. 위 block을 재현할
수 있도록 pattern에 `License-Expression`을 추가했습니다. 이 변경의 비용은 §6에
기록했습니다.

유일한 `Requires-Dist` 줄은 `dev` extra에만 적용됩니다. 조건 없는 runtime
요구사항은 없습니다.

### 1.6 빈 환경에 설치

결정적인 검사입니다. `pip`가 실제 dependency graph를 해석하며 추가로 설치되는
항목이 없습니다.

```
$ python3.13 -m venv /tmp/runtimeenv
$ /tmp/runtimeenv/bin/python -m pip list --format=json
[{"name": "pip", "version": "25.2"}]

$ /tmp/runtimeenv/bin/python -m pip install dist/fituna-0.1.0-py3-none-any.whl
Processing ./dist/fituna-0.1.0-py3-none-any.whl
Installing collected packages: fituna
Successfully installed fituna-0.1.0

$ /tmp/runtimeenv/bin/python -m pip list --format=json
[{"name": "fituna", "version": "0.1.0"}, {"name": "pip", "version": "25.2"}]
```

`Installing collected packages: fituna`처럼 package 하나만 설치되고 transitive
dependency는 없습니다. 설치 후 환경은 설치 전 환경에 `fituna`만 더해진 상태입니다.
`pip` 자체는 venv의 bootstrap이며 FiTuna가 끌어온 항목이 아닙니다.

---

## 2. 라이선스가 충돌하지 않는 이유

`docs/OPEN_SOURCE_USAGE.md`의 결합 방식 표가 제시하는 원칙은 **라이선스 조건은
단순 사용이 아니라 link와 재배포를 통해 전달된다**는 것입니다. FiTuna는 Python
표준 라이브러리를 제외한 어떤 구성요소도 link하거나 재배포하지 않으며, 표준
라이브러리의 PSF-2.0도 permissive license입니다.

FiTuna 자체 라이선스는 MIT입니다(`LICENSE`, `pyproject.toml`의 PEP 639 SPDX
expression `license = "MIT"`). MIT는 고지 보존 하나만 요구하는 permissive
license이며 아래 모든 라이선스와 호환됩니다.

### 2.1 llama.cpp — MIT, subprocess만 사용

`fituna/quantize.py`, `bench.py`, `quality.py`, `model_info.py`는 argv 목록을
전달하는 `subprocess.run()`으로 `llama-quantize`, `llama-bench`,
`llama-perplexity`, `convert_hf_to_gguf.py`를 실행하고 stdout·stderr·종료 코드를
읽습니다. 이 경계를 넘는 것은 없습니다.

- **link하지 않음.** llama.cpp header를 `#include`하지 않고, `ctypes`나 compile
  extension으로 `libllama`를 load하지 않으며, static·dynamic link step도 없습니다.
  FiTuna에는 compile 구성요소 자체가 없습니다. §1.3의 wheel은 ABI tag가 없는 pure
  Python인 `py3-none-any`입니다.
- **import하지 않음.** §3.4의 AST scan에서 표준 라이브러리 외 import는 0개입니다.
- **vendor 방식 포함·재배포 없음.** §1.4에서 두 산출물의 모든 파일이 이 저장소 git
  index에서 왔음을 증명합니다. 사용자가 llama.cpp를 직접 설치하면
  `fituna/binaries.py:locate_binaries()`가 `PATH` 또는 `--llama-bin-dir`에서 찾습니다.
  없으면 upstream build 안내를 가리키는 `BinaryNotFoundError`를 발생시킵니다.

**llama.cpp 자체도 MIT이므로 어느 경우에도 충돌하지 않습니다.** 가장 밀접하게
결합해 하나의 binary에 static link하더라도 MIT끼리는 충돌하지 않고 두 고지만
보존하면 됩니다. subprocess 경계는 license 우회가 아니라 architecture 선택이며
그 이유는 `docs/OPEN_SOURCE_USAGE.md` §1에 기록했습니다. 실제로 적용되는 유일한
의무인 MIT 고지 보존은 `THIRD_PARTY_NOTICES.md` §1에서 이행합니다. llama.cpp
byte가 저장소에 없어도 MIT 전문을 수록했습니다.

같은 근거가 MIT인 **GGUF/ggml**에도 적용됩니다. `fituna/model_info.py`는 공개
사양에 따라 표준 라이브러리 `struct`로 작성한 독립 reader입니다. 문서화된 파일
형식을 준수하는 것은 파생이 아니며, 형식 저장소 자체도 MIT입니다.

### 2.2 평가 corpus — CC BY-SA 3.0 / GFDL, 저장소 미포함

`fituna/corpus.py`는 영어용 `Salesforce/wikitext`와 한국어용
`wikimedia/wikipedia` `20231101.ko` preset 두 개를 정의합니다. 둘 다 CC BY-SA
3.0이며 GFDL과 이중 라이선스된 동일조건 계열이라 가장 주의해서 검토해야 합니다.

다음 세 사실 때문에 충돌하지 않습니다.

1. **저장소에 말뭉치 text가 없습니다.** `.gitignore`는 `*.txt`와 말뭉치 임시 파일
   pattern `.*.tmp`를 제외합니다. 일반 저장소 text 파일은 명시적으로 다시 포함해
   정상 파일이 조용히 누락되지 않게 했습니다. §1.4 검사에서도 두 산출물에 말뭉치
   파일이 없음을 독립적으로 확인합니다.
2. **FiTuna가 배포하지 않습니다.** `fituna fetch-corpus`는 *사용자가 선택한*
   `--out` 경로에 내려받습니다. byte는 HuggingFace에서 사용자에게 이동하며 FiTuna는
   전송 수단이지 재배포자가 아닙니다.
3. **내려받을 때 고지합니다.** `fituna/cli.py:321`의 `_cmd_fetch_corpus`는 성공할
   때마다 source URL과 license 고지를 stdout에 출력합니다. text를 재배포하려는
   사용자도 조건을 확인할 수 있습니다. `--dataset/--config/--split`으로 preset을
   바꾸면 같은 코드 경로가 "check this dataset's own license"라는 일반 안내를
   출력합니다. 확인하지 않은 데이터셋에 CC BY-SA라고 단정하지 않기 위해서입니다.

동일조건은 말뭉치 text에 적용됩니다. FiTuna source는 말뭉치의 파생물이나 각색물이
아니라 다른 프로그램에 파일 경로를 전달하는 프로그램이므로 source code에는
영향을 주지 않습니다.

### 2.3 모델 weight — Apache-2.0 / MIT, 재배포하지 않음

`--model`은 사용자가 이미 가진 파일의 경로입니다. FiTuna는 weight를 bundle,
train, fine-tune, distil, merge하지 않습니다. `llama-quantize`가 llama.cpp 자체
process 안에서 수치 정밀도만 바꿉니다. §1.4에서 두 산출물에 `.gguf`가 없음을
확인했고 `.gitignore`도 `*.gguf`와 `*.gguf.tmp`를 제외합니다.

프로젝트 공개 측정(`docs/RESULTS.md`)에 사용한 세 모델은 Apache-2.0인
SmolLM2-135M-Instruct·Qwen3-4B-Instruct-2507과 MIT인
Midm-2.0-Mini-Instruct입니다. 확인 근거는 `docs/OPEN_SOURCE_USAGE.md` §3~4에
있습니다. Apache-2.0과 MIT 모두 허용적이며 MIT와 호환됩니다. 애초에 material을
결합하지도 않습니다. `docs/AI_MODEL_USAGE.md`에는 모델별 공개 template과 같은 세
모델을 기록했습니다.

모델 선택 자체에도 라이선스를 고려했습니다. Qwen2.5-3B-Instruct의
`qwen-research`는 OSI 승인 라이선스가 아닌 연구용 custom license라 결과에서
제외했습니다(`docs/OPEN_SOURCE_USAGE.md` §3~4에 확인 근거 기록). 재현 경로에는
접근 제한 모델이나 bespoke license 모델이 없습니다.

### 2.4 GPL / AGPL / LGPL — 직간접적으로 사용하지 않음

**주장.** FiTuna는 GPL 계열 코드를 import, link, vendor 방식으로 포함하거나
배포하지 않습니다.

**확인 방법.** 각각 재현 가능한 독립 검사 네 가지를 수행했습니다.

| 검사 | 확인 범위 | 결과 |
|---|---|---|
| 설치 환경 scan(§3.2) | 빈 환경 설치로 해석되는 모든 distribution | MIT인 `fituna`만 존재. `[dev]` 포함 시 MIT, BSD-2-Clause, Apache-2.0 OR BSD-2-Clause이며 GPL 계열 없음 |
| 배포물 출처 검사(§1.4) | sdist·wheel의 모든 파일 | 저장소 외부에서 온 파일 0개 |
| AST import scan(§3.4) | `fituna/*.py`의 모든 최상위 import | 표준 라이브러리 외 import 0개 |
| 저장소 text scan(§3.3) | git이 추적하는 모든 파일 | code·config·notebook·CI에서 copyleft 일치 0개. 의도적으로 주제를 설명하는 license 문서 세 개만 일치 |

**프로젝트 주변의 유일한 GPL 구성요소.**
`git`은 GPL-2.0이며 `notebooks/colab_nvidia_verification.ipynb`가 호출합니다.
2번 cell에서 `git clone --depth 1`로 llama.cpp를 받고, 3번 cell의
`pip install git+https://...`에서도 실행합니다. 이는 **Colab runtime 안에서 별도
process로 실행하는 개발·검증 시점 도구**입니다. import·link·배포하지 않으며 §1의
어느 산출물에도 없습니다.

GPL 프로그램을 별도 process로 호출한다고 호출자가 그 파생물이 되지는 않습니다.
FSF의 GPL FAQ는 다음 기준을 제시합니다
(`https://www.gnu.org/licenses/gpl-faq.html`, HTTP 200, fetched
2026-07-30):

> By contrast, pipes, sockets and command-line arguments are communication
> mechanisms normally used between two separate programs. So when they are
> used for communication, the modules normally are separate programs. But if
> the semantics of the communication are intimate enough, exchanging complex
> internal data structures, that too could be a basis to consider the two
> parts as combined into a larger program.

FAQ의 단서를 피하지 않고 실제 경계를 확인했습니다. `git`에는 argv 목록이 들어가고
stdout·stderr·종료 코드만 나오며, 복잡한 내부 data structure·shared memory·callback은
없습니다. command-line 인수와 stdout만 경계를 넘는다는 점에서 §2.1의 llama.cpp와
같고, package가 아닌 notebook이 실행하므로 FiTuna와 `git`의 관계는 더 멉니다.
`docs/OPEN_SOURCE_USAGE.md` §13에서 scanner가 "GPL-2.0"을 찾을 수 있는데 준수
문서가 이를 침묵하는 편이 더 부정확하므로 명시했습니다.

**이 방법의 한계**도 분명히 밝힙니다.

- 이 검사들은 제3자 *파일*이나 *package*가 FiTuna에 들어오지 않음을 증명하지만,
  header 없이 제3자 *snippet*을 source에 붙인 적이 전혀 없음을 증명하지는 못합니다.
  이를 찾으려면 ScanCode, FOSSology, Black Duck 같은 license text·code similarity
  matching이 필요합니다. 해당 도구를 실행하지 않은 이유와 대체 검사는 §3.1에
  설명합니다.
- text scan은 keyword 기반이므로 알아볼 수 있는 license 문자열이 없는 copyleft
  파일은 찾지 못합니다. §1.4의 출처 검사는 문자열이 아닌 hash 비교이므로 이 한계에
  의존하지 않습니다.
- `pip-licenses`는 각 package가 metadata에 선언한 내용을 보고합니다. package가
  license를 잘못 선언하면 결과도 잘못됩니다. 다만 이 프로젝트의 runtime 목록은
  FiTuna 자체 package 하나뿐이므로 잘못 선언할 제3자 항목이 없습니다.

### 2.5 충돌 검토표

FiTuna와 관계된 모든 라이선스와 검토 결과입니다.

| 구성요소 | 라이선스 | 결합 방식 | MIT와 충돌 여부 |
|---|---|---|---|
| CPython 표준 라이브러리 | PSF-2.0 | **import** | 없음 — 허용적이며 copyleft 없음 |
| llama.cpp, ggml/GGUF | MIT | subprocess / 파일 형식 | 없음 — 같은 라이선스이며 link하지 않음 |
| 모델 weight(SmolLM2, Qwen3) | Apache-2.0 | 사용자 제공 파일 | 없음 — 허용적이며 재배포하지 않음 |
| 모델 weight(Midm-2.0-Mini) | MIT | 사용자 제공 파일 | 없음 — 같은 라이선스이며 재배포하지 않음 |
| WikiText-2, 한국어 Wikipedia | CC BY-SA 3.0 / GFDL | 내려받은 파일 | 없음 — 동일조건은 저장소·배포물에 없는 text에 적용되며 내려받을 때 고지 |
| HF dataset-viewer(server) | Apache-2.0 | HTTP client | 없음 — 해당 코드를 사용하지 않음 |
| Model Context Protocol | Apache-2.0 / MIT, 전환 중 | 자체 protocol 구현 | 없음 — SDK·schema library·사양 저장소 코드 미사용 |
| setuptools | MIT | build 시점만 | 없음 — 격리된 build 환경에서만 사용하고 설치하지 않음 |
| pytest | MIT | 개발 extra만 | 없음 — `pip install fituna`로 설치되지 않음 |
| `rocm-smi`, `sysctl` | MIT, BSD-3-Clause | 선택적 subprocess | 없음 — 허용적이며 호출만 하고 재배포하지 않음 |
| `git`(notebook 전용) | GPL-2.0 | subprocess, 개발/notebook | 없음 — §2.4 참조 |
| CMake(notebook 전용) | BSD-3-Clause | subprocess, 개발/notebook | 없음 — 허용적이며 Colab 안에서 llama.cpp만 build |
| `pip`(notebook 전용) | MIT | subprocess, 개발/notebook | 없음 — 허용적이며 Colab 안에서 FiTuna와 `huggingface_hub` 설치 |
| `huggingface_hub`(notebook 전용) | Apache-2.0 | `pip install`, 개발/notebook | 없음 — 허용적이며 시연용 GGUF를 내려받을 뿐 FiTuna 의존성이 아님 |
| Jupyter notebook 형식(`nbformat`) | BSD-3-Clause | 파일 형식, 개발/notebook | 없음 — 허용적이며 `.ipynb` 파일이 형식을 따름 |
| `actions/checkout`, `actions/setup-python` | MIT | CI action, 개발 전용 | 없음 — 허용적이며 GitHub Actions에서만 실행, 배포하지 않음 |

FiTuna가 호출하지만 재배포하지 않는 비오픈소스 구성요소인 `nvidia-smi`,
`system_profiler`, CUDA toolkit, 호스팅 GitHub Actions·Google Colab service도
`docs/OPEN_SOURCE_USAGE.md`에 나열했으며 어느 곳에서도 open source라고 주장하지
않습니다.

**결론: 프로젝트의 어떤 라이선스 조합도 충돌하지 않으므로 별도 해결 조치가
필요하지 않습니다.** subprocess 경계, 의존성 0개 규칙, data를 포함하지 않고
내려받는 방식은 engineering 이유로 선택했으며 `docs/ARCHITECTURE.md`와
`docs/OPEN_SOURCE_USAGE.md`에 근거를 기록했습니다. 라이선스 충돌이 없는 것은 이
설계의 결과입니다.

---

## 3. 실제 실행한 scan

### 3.1 이 컴퓨터에서 사용할 수 있었던 도구

어떤 도구도 설치하기 전에 `command -v`로 확인했습니다.

| 도구 | 설치 여부 | 사용 여부 |
|---|---|---|
| `pip-licenses` | 없음 → `/tmp/buildenv`에 설치 | **사용**(§3.2) |
| `reuse` | 없음 → `/tmp/buildenv`에 설치 | **사용**(§3.5) |
| `python3.13 -m pip list` | 있음 | **사용**(§1.6) |
| `git grep` | 있음 | **사용**(§3.3) |
| Python `ast`(표준 라이브러리) | 있음 | **사용**(§3.4) |
| `pipdeptree` | 없음 | 불필요 — dependency graph가 node 하나(§1.6) |
| `scancode` | 없음 | **실행하지 않음** — 아래 참조 |
| `syft`, `trivy`, `osv-scanner`, `cyclonedx-py`, `licensecheck`, `askalono` | 없음 | 실행하지 않음 |

**ScanCode를 실행하지 않은 이유.** `scancode-toolkit`은 수백 MB의 license text
index를 포함하며 설치·index 시간이 이 저장소의 파일 수에 비해 큽니다. 추적 파일
57개, 실제 배포 source 18개인 codebase에는 과도하다고 판단했습니다. 대신 실행한
`reuse`(SPDX tag 추출), `pip-licenses`(package metadata), `git grep`(license 문자열
존재), AST import scan, §1.4의 hash 기반 출처 검사는 *선언된* 라이선스와 *배포물
출처*를 모두 확인합니다.

**검사하지 못한 범위도 명시합니다.** 여기서 실행한 도구는 license text similarity
matching이나 code clone 감지를 하지 않습니다. 미고지 snippet 복사 여부까지 보증해야
한다면 ScanCode 또는 동등한 도구가 적절합니다. §5의 재현 명령에는 이를 한 단계로
추가할 수 있습니다. 이 문서가 해당 위험에 대해 제공하는 근거는 통계가 아니라
구조입니다. FiTuna는 외부 code를 import·link하지 않으며 직접 작성하지 않은 파일을
배포하지 않습니다(§1.4).

### 3.2 설치 환경 라이선스 scan — `pip-licenses`

`pip-licenses`는 `--python`을 사용해 **build venv에서 실행하되 빈 runtime venv를
대상으로** 검사합니다. scanner 자체 의존성이 측정 환경에 들어가지 않습니다.

```
$ /tmp/buildenv/bin/pip-licenses --python /tmp/runtimeenv/bin/python \
      --from=mixed --with-authors --with-urls --format=markdown
```

| Name   | Version | License | Author              | URL                                         |
|--------|---------|---------|---------------------|---------------------------------------------|
| fituna | 0.1.0   | MIT     | FiTuna contributors | https://github.com/leeyunseokarchive/fituna |

**한 행이 설치된 FiTuna의 전체 라이선스 목록입니다.** 결과는 `MIT License`가
아니라 `MIT`입니다. `pip-licenses`는 `License :: OSI Approved :: ...` classifier가
있으면 여기서 이름을 얻지만 FiTuna wheel에는 없습니다(이유는 §6). 따라서
`License-Expression: MIT`만 보고 SPDX 식별자 그대로 출력합니다.

완전성을 위해 개발 extra(`fituna[dev]`)를 설치한 결과도 확인했습니다. 아래 항목은
배포물에 들어가지 않습니다.

```
$ /tmp/runtimeenv/bin/python -m pip install 'dist/fituna-0.1.0-py3-none-any.whl[dev]'
$ /tmp/buildenv/bin/pip-licenses --python /tmp/runtimeenv/bin/python --from=mixed --format=markdown
```

| Name      | Version | License                    |
|-----------|---------|----------------------------|
| Pygments  | 2.20.0  | BSD-2-Clause               |
| fituna    | 0.1.0   | MIT                        |
| iniconfig | 2.3.0   | MIT                        |
| packaging | 26.2    | Apache-2.0 OR BSD-2-Clause |
| pluggy    | 1.6.0   | MIT License                |
| pytest    | 9.1.1   | MIT                        |

모두 허용적이며 개발 환경에도 copyleft 라이선스가 없습니다. `Pygments`,
`iniconfig`, `packaging`, `pluggy`는 pip가 해석한 pytest 자체의 transitive
dependency이며 FiTuna는 `pytest`만 선언합니다.

### 3.3 저장소 전체 copyleft·독점 문구 scan — `git grep`

```
$ git grep -c -i -E 'GPL|AGPL|LGPL|copyleft|CDDL|EPL-|MPL-|SSPL|Commons Clause|proprietary|All Rights Reserved' -- .
THIRD_PARTY_NOTICES.md:4
docs/LICENSE_COMPLIANCE.md:25
docs/OPEN_SOURCE_USAGE.md:13
```

세 파일이 일치하며 모두 **제3자 라이선스를 의도적으로 설명하는 문서이기 때문에**
일치합니다. `THIRD_PARTY_NOTICES.md`의 독점 utility 표,
`docs/OPEN_SOURCE_USAGE.md` §13의 `git` GPL-2.0 항목과 §14 독점 도구 표, 그리고
copyleft 라이선스를 배제하기 위해 이름을 기록한 이 문서(§2.4)입니다. 같은 검사를
code·configuration·실행 가능 content로 제한하면 다음과 같습니다.

```
$ git grep -n -i -E 'GPL|AGPL|LGPL|copyleft|CDDL|EPL-|MPL-|SSPL|Commons Clause|proprietary|All Rights Reserved' \
      -- fituna tests pyproject.toml notebooks examples .github
(no matches)
```

**package, test suite, build configuration, notebook, example, CI workflow
전체에서 일치 항목은 0개입니다.**

방법과 한계: git이 추적하는 content에 대한 대소문자 무시 고정 keyword scan입니다.
*license 문자열이 없음*을 확인할 뿐 표시 없는 복사 code가 없음을 증명하지는
않습니다. §3.1을 참조하십시오.

### 3.4 Import graph scan — 표준 라이브러리 `ast`

`pyproject.toml` 선언을 그대로 신뢰하지 않고 syntax tree에서 의존성 0개 주장을
다시 확인합니다.

```
$ python3.13 - <<'PY'
import ast, pathlib, sys
mods = set()
for p in sorted(pathlib.Path("fituna").glob("*.py")):
    for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
std, first = sys.stdlib_module_names, {"fituna"}
print("total distinct top-level imports:", len(mods))
print("stdlib      :", sorted(mods & std))
print("first-party :", sorted(mods & first))
print("THIRD-PARTY :", sorted(mods - std - first) or "NONE")
PY
total distinct top-level imports: 28
stdlib      : ['__future__', 'argparse', 'ctypes', 'dataclasses', 'datetime', 'enum', 'hashlib', 'io', 'json', 'logging', 'os', 'pathlib', 'platform', 're', 'shutil', 'sqlite3', 'stat', 'struct', 'subprocess', 'sys', 'tempfile', 'textwrap', 'time', 'tomllib', 'typing', 'unittest', 'urllib']
first-party : ['fituna']
THIRD-PARTY : NONE
```

`THIRD-PARTY : NONE`입니다. 표준 라이브러리 module은 29개이며 runtime import가
아닌 compiler directive `__future__`까지 세면 30개입니다. 이름별로 확인한 결과
`docs/SBOM.md` 2~30행과 정확히 일치하므로 SBOM과 scan 결과가 같습니다.

### 3.5 SPDX 준수 scan — `reuse` 6.2.0

`reuse`는 FSFE의 SPDX/REUSE 도구입니다. git이 추적하는 모든 파일에서
`SPDX-License-Identifier` tag를 읽고 결과를 보고합니다.

```
$ /tmp/buildenv/bin/reuse lint
# SUMMARY

* Bad licenses: 0
* Deprecated licenses: 0
* Licenses without file extension: 0
* Missing licenses: MIT
* Unused licenses: 0
* Used licenses: MIT
* Read errors: 0
* Invalid SPDX License Expressions: 2
* Files with copyright information: 6 / 54
* Files with license information: 27 / 54

Unfortunately, your project is not compliant with version 3.3 of the REUSE Specification :-(
```

출력은 정확히 읽어야 합니다. 세 줄에는 설명이 필요하고, 그중 하나는 실제 발견
사항입니다.

- **`Bad licenses: 0`, `Deprecated licenses: 0`, `Used licenses: MIT`.** 핵심
  결과입니다. `reuse`가 저장소에서 찾은 모든 license 식별자는 `MIT`이며 deprecated
  또는 bad 목록에 든 항목이 없습니다.
- **`Missing licenses: MIT`.** "파일에 MIT가 없다"는 뜻이 아닙니다. REUSE 3.3
  사양은 tag와 함께 `LICENSES/MIT.txt` directory를 요구하지만 FiTuna는 관례적인
  root `LICENSE` 파일을 사용한다는 뜻입니다. REUSE 완전 준수에는 문서를 포함한
  54개 파일 모두에 `SPDX-FileCopyrightText` 줄도 필요합니다. 이번 검증보다 엄격한
  표준이라 프로젝트는 의도적으로 채택하지 않았습니다. `:-(` 줄은 license 유효성이
  아니라 REUSE 3.3 준수 여부에 관한 것입니다.
- **`Invalid SPDX License Expressions: 2`.** 두 개 모두 *문서에서 생긴 false
  positive*입니다. `docs/OPEN_SOURCE_USAGE.md`와 `docs/SBOM.md`가 macOS `sysctl`
  라이선스 확인 방법을 설명하며 `SPDX-License-Identifier: BSD-3-Clause` 문자열을
  본문에 인용했고, `reuse` line parser가 뒤의 한국어 설명까지 expression으로
  읽었습니다. source 파일에는 영향이 없습니다. 이 문서도 tag를 여러 번 인용하므로
  그대로 두면 false positive 네 개를 더 만듭니다. 그래서 `reuse`가 권장하는
  HTML comment ignore block marker 쌍으로 문서의 첫 줄과 마지막 줄을 감쌌습니다.
  marker가 없으면 scan 결과를 공개하는 행위 자체가 scan 결과를 바꿉니다. 닫는
  token을 본문에 쓰면 그 지점에서 block이 끝나 아래 내용이 다시 노출되므로, 두
  marker의 정확한 철자는 해당 위치에만 의도적으로 둡니다.
- **`Files with copyright information: 6 / 54`.** 준수 여부를 뜻하는 수치가
  아닙니다. `reuse`는 copyright 고지를 찾은 모든 파일을 세며, 여섯 파일은 root
  `LICENSE`, `THIRD_PARTY_NOTICES.md`, 그리고 license를 확인한 위치를 기록하면서
  다른 주체의 고지를 *인용한* 문서 네 개(`README.md`, `docs/ARCHITECTURE.md`,
  `docs/OPEN_SOURCE_USAGE.md`, `docs/RESULTS.md`)입니다. FiTuna `.py`에는
  `SPDX-License-Identifier`가 있지만 `SPDX-FileCopyrightText`는 의도적으로 없습니다.
  위 `Missing licenses` 설명과 §6을 참조하십시오. (2026-10-12 갱신: 이후 모든 `.py`
  파일의 둘째 줄에 root `LICENSE`와 같은 `# Copyright (c) 2026 FiTuna contributors`
  고지를 추가했습니다. 위 출력은 2026-07-30 실행 기록이므로 그대로 둡니다. 자세한
  내용은 §4 끝을 참조하십시오.)
- **`Files with license information: 27 / 54`.** tag가 있는 `.py` 파일 27개(§4)를
  모두 셉니다. 여기에는 `reuse`가 파일 자체 header를 읽지 못하는
  `fituna/config.py`도 포함되므로 설명이 필요합니다. `reuse`는 파일의 처음 2048
  byte(`HEURISTICS_CHUNK_SIZE`)만 보고 encoding을 추정하는데, `config.py`의 2048번째
  byte는 multi-byte UTF-8 한국어 주석 중간에 있습니다. `charset-normalizer`가 잘린
  chunk를 decode하지 못하면 `reuse`는 binary 파일로 판단하고 오류 없이 읽기를
  건너뜁니다. 다음과 같이 직접 확인했습니다.

  ```
  $ /tmp/buildenv/bin/python -c "
  from reuse import extract
  raw = open('fituna/config.py','rb').read()
  print(raw[2040:2056])
  print('2 KB chunk  ->', extract.detect_encoding(raw[:2048]))
  print('2050 bytes  ->', extract.detect_encoding(raw[:2050]))
  print('whole file  ->', extract.detect_encoding(raw))"
  b'\xec\xa0\x9c\xed\x95\x9c \xec\x97\x86\xec\x9d\x8c)\xeb\xa1'
  2 KB chunk  -> None
  2050 bytes  -> utf_8
  whole file  -> utf_8
  ```

  tag 자체는 파일에 있습니다. `head -1 fituna/config.py`는
  `# SPDX-License-Identifier: MIT`를 출력하고 §4의 독립 검사도 27개 중 27개를
  확인합니다. 다만 `reuse`가 직접 읽지 못하므로 `REUSE.toml`이 동일한 `MIT`
  식별자를 file-level annotation으로 제공합니다. 이는 이 상황에 대해 REUSE가
  문서화한 mechanism이며 파일 내 annotation 위치에 의존하지 않습니다. `reuse`가
  `config.py` encoding을 추정하기 전에 `REUSE.toml`을 읽으므로 2048 byte heuristic이
  잘못 작동할 기회가 없습니다.

  `reuse`는 54개 집계에서 `LICENSE` 자체와 빈 `fituna/py.typed` marker도 제외합니다.
  자체 configuration이라 annotation 대상이 아닌 `REUSE.toml`도 제외하므로
  54 + 3 = 57 = `git ls-files | wc -l`입니다. 근본 heuristic의 upstream 완화책은
  `libmagic`을 설치해 encoding 감지에 `charset-normalizer` 대신 `python-magic`을
  쓰게 하는 것이지만 system package라 설치하지 않았습니다. `REUSE.toml`로 같은
  문제를 해결했습니다. 심사위원의 SCA 도구는 같은 특성이 있을 수도, 없을 수도
  있습니다. 다른 보고서에 "unknown license" 파일 하나가 나타나더라도 원인을
  확인할 수 있도록 기록했습니다.

`reuse` 자체는 GPL-3.0입니다. 저장소 밖 일회용 venv에 설치해 별도 process로
실행했으며 이 프로젝트 어떤 항목의 의존성도 아닙니다. §2.4와 같은 독립된
관계입니다.

---

## 4. 소스의 SPDX 식별자

**변경 전에는 어떤 `.py` 파일에도 SPDX 식별자가 없었습니다.**
`grep -rn "SPDX" fituna/ tests/`가 아무것도 반환하지 않음을 확인했습니다.

**조치: Python 파일 27개 모두의 첫 줄에
`# SPDX-License-Identifier: MIT`를 추가했습니다.** `fituna/` package 18개와
`tests/` 9개입니다. 저장소에 shebang이 있는 파일이 없으므로 모두 첫 줄이 맞습니다.
tag는 module docstring 위에 있으며 comment는 문자열 literal이 아니므로 `__doc__`을
바꾸지 않습니다. 다음과 같이 확인했습니다.

```
$ ok=0; for f in fituna/*.py tests/*.py; do
    [ "$(head -1 "$f")" = "# SPDX-License-Identifier: MIT" ] && ok=$((ok+1)) || echo "MISSING: $f"
  done; echo "line-1 SPDX tag present: $ok/$(ls fituna/*.py tests/*.py | wc -l | tr -d ' ')"
line-1 SPDX tag present: 27/27
```

build한 wheel에서도 확인해 package 설치자에게 tag가 전달됨을 검증했습니다.

```
$ for f in $(unzip -Z1 dist/fituna-0.1.0-py3-none-any.whl | grep '\.py$'); do
    printf "%-24s %s\n" "$f" "$(unzip -p dist/fituna-0.1.0-py3-none-any.whl $f | head -1)"; done
fituna/__init__.py       # SPDX-License-Identifier: MIT
fituna/__main__.py       # SPDX-License-Identifier: MIT
fituna/bench.py          # SPDX-License-Identifier: MIT
fituna/binaries.py       # SPDX-License-Identifier: MIT
fituna/cache.py          # SPDX-License-Identifier: MIT
fituna/cli.py            # SPDX-License-Identifier: MIT
fituna/config.py         # SPDX-License-Identifier: MIT
fituna/corpus.py         # SPDX-License-Identifier: MIT
fituna/doctor.py         # SPDX-License-Identifier: MIT
fituna/errors.py         # SPDX-License-Identifier: MIT
fituna/hardware.py       # SPDX-License-Identifier: MIT
fituna/mcp_server.py     # SPDX-License-Identifier: MIT
fituna/model_info.py     # SPDX-License-Identifier: MIT
fituna/quality.py        # SPDX-License-Identifier: MIT
fituna/quantize.py       # SPDX-License-Identifier: MIT
fituna/quickstart.py     # SPDX-License-Identifier: MIT
fituna/report.py         # SPDX-License-Identifier: MIT
fituna/search.py         # SPDX-License-Identifier: MIT
```

중요한 이유: tag가 없으면 root `LICENSE`가 있어도 scanner는 각 파일을 *unknown
license*로 분류해 미해결 항목으로 보고합니다. 이제 source 파일마다 스스로 license를
표시합니다. 26개는 header를 직접 읽고, `reuse`가 자체 header를 읽지 못하는
`fituna/config.py`는 §3.5의 `REUSE.toml` annotation으로 식별합니다. 따라서
프로젝트 SCA scan(§3.5)에 "no copyright and licensing information"으로 남는
항목이 없습니다.

**저작권 고지 추가(2026-10-12).** 파일 단위로 저작권자도 드러나도록 모든 `.py`
파일의 둘째 줄에 root `LICENSE`와 같은 문구의 고지를 추가했습니다. 첫 줄의 SPDX tag는
그대로입니다. `tests/test_license_headers.py`가 `fituna/`와 `tests/`의 모든 `.py`
파일에서 두 줄을 검사하므로 새 파일이 header 없이 들어오면 CI가 실패합니다. 이때 SPDX
tag가 없던 `tests/test_mcp_server.py`(위 27개 확인 이후 추가된 파일)에도 두 줄을
넣었습니다. `REUSE.toml`의 `fituna/config.py` annotation도 같은 저작권자로 바꿨습니다.
줄이 하나씩 밀리므로 문서의 `file.py:NNN` 인용도 모두 1씩 옮겼습니다.

```
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
```

**Regression 검사.** 모든 파일에 한 줄을 추가하면 아래 line number가 전부
바뀝니다. 두 영향을 모두 확인하고 처리했습니다.

- `python3.13 -m pytest -q` → **246 passed**, `.github/workflows/ci.yml`에
  나열한 module self-check 17개도 모두 종료 코드 0을 유지했습니다.
- source 위치를 `file.py:NNN`으로 인용한 문서는 하나가 아니라 네 개입니다.
  `docs/OPEN_SOURCE_USAGE.md`에 38회(서로 다른 `file:line` 30개), `SECURITY.md`,
  `docs/DEVELOPMENT.md`, 이 문서(§2.2)에 각 1회가 있습니다. 처음에는
  `docs/OPEN_SOURCE_USAGE.md`만 검사하고 유일한 문서라고 잘못 적었습니다. 이후 네
  문서를 함께 검사하고 현재 source의 해당 줄을 읽어 모든 인용을 다시 확인했습니다.
  오래된 인용 다섯 개를 찾아 고쳤습니다. `_cmd_fetch_corpus`의 CC BY-SA 고지를
  가리키면서 `_cmd_doctor` 마지막 줄을 적은 세 곳(이 문서 한 곳,
  `docs/OPEN_SOURCE_USAGE.md` 두 곳), 그리고 하나의 `bench.py` self-check를 서로 다른
  위치로 적은 두 곳입니다. 이제 둘 다 `def _self_check()` 줄을 가리킵니다. 인용
  검사의 범위는 문서 하나가 아니라 저장소 전체입니다.

  ```
  $ grep -rn --include="*.md" -E "[a-z_]+[.]py:[0-9]+" .
  ```

---

## 5. 문서의 모든 주장을 재현하는 방법

Python 3.11+와 git이 있는 어느 platform에서든 빈 checkout부터 재현할 수 있습니다.
아래 명령은 프로젝트 환경에 아무것도 설치하지 않습니다.

```bash
# --- 0. 저장소 밖 일회용 venv에 도구 설치 ------------------------------
python3 -m venv /tmp/buildenv
/tmp/buildenv/bin/pip install build pip-licenses 'reuse[charset-normalizer]'

# --- 1. 배포물 구성 확인(§1) --------------------------------------------
rm -rf build dist fituna.egg-info
/tmp/buildenv/bin/python -m build
tar -tzf dist/fituna-0.1.0.tar.gz | sort
unzip -Z1 dist/fituna-0.1.0-py3-none-any.whl | sort
unzip -p dist/fituna-0.1.0-py3-none-any.whl fituna-0.1.0.dist-info/METADATA \
  | grep -E '^(Name|Version|Requires-Python|License-Expression|License-File|Provides-Extra|Requires-Dist):'

# --- 2. pip로 runtime 의존성 0개 확인(§1.6) -----------------------------
python3 -m venv /tmp/runtimeenv
/tmp/runtimeenv/bin/python -m pip list --format=json          # 기준 상태
/tmp/runtimeenv/bin/python -m pip install dist/fituna-0.1.0-py3-none-any.whl
/tmp/runtimeenv/bin/python -m pip list --format=json          # fituna 외 추가 없음

# --- 3. 설치 package의 license 목록(§3.2) ------------------------------
/tmp/buildenv/bin/pip-licenses --python /tmp/runtimeenv/bin/python \
  --from=mixed --with-authors --with-urls --format=markdown

# --- 4. code에 copyleft·독점 문구가 없는지 확인(§3.3) -------------------
git grep -c -i -E 'GPL|AGPL|LGPL|copyleft|CDDL|EPL-|MPL-|SSPL|Commons Clause|proprietary|All Rights Reserved' -- .
git grep -n -i -E 'GPL|AGPL|LGPL|copyleft|CDDL|EPL-|MPL-|SSPL|Commons Clause|proprietary|All Rights Reserved' \
  -- fituna tests pyproject.toml notebooks examples .github

# --- 5. SPDX tag 확인(§3.5, §4) ----------------------------------------
/tmp/buildenv/bin/reuse lint
for f in fituna/*.py tests/*.py; do head -1 "$f"; done | sort | uniq -c

# --- 6. 제3자 import가 없는지 확인(§3.4) -------------------------------
#      (§3.4의 AST snippet을 그대로 실행)

# --- 7. 산출물에 저장소 외부 파일이 없는지 확인(§1.4) -------------------
#      (§1.4의 출처 확인 snippet을 그대로 실행)

# --- 8. 회귀 검사(CI와 같은 두 단계) -----------------------------------
/tmp/runtimeenv/bin/python -m pip install pytest
/tmp/runtimeenv/bin/python -m pytest -q            # 기대값: 0 failed(작성 당시 246 passed)
for m in __init__ errors config cache search model_info quantize quality \
         bench hardware binaries report; do /tmp/runtimeenv/bin/python -m fituna.$m || break; done
for m in corpus doctor quickstart cli mcp_server; do /tmp/runtimeenv/bin/python -m fituna.$m --selfcheck || break; done
```

---

## 6. 알려진 한계와 남은 항목

이 문서가 실제 검증 범위보다 넓게 읽히지 않도록 한계를 기록합니다.

| 항목 | 상태 |
|---|---|
| Code similarity / 라이선스 원문 matching(ScanCode, FOSSology) | **실행하지 않음** — 추적 파일 57개에 비해 비용이 큼. 대체 검사와 남는 사각지대는 §3.1 참조 |
| `reuse`가 `fituna/config.py` 자체 header를 읽지 못함 | **도구 한계를 진단하고 우회함**(§3.5). `reuse`의 2 KB encoding heuristic이 multi-byte 문자를 중간에서 잘라 파일 안 tag를 읽지 못함. 파일 byte 배치를 바꾸는 방식은 주석 길이와 2048번째 byte 위치에 의존해 취약하므로 쓰지 않고, byte 위치와 무관하며 REUSE가 공식 지원하는 `REUSE.toml` 파일 단위 annotation으로 해결 |
| REUSE 3.3 완전 준수(`LICENSES/` 디렉터리, 파일별 copyright tag) | **일부 채택** — `.py` 파일에는 2026-10-12부터 SPDX tag와 copyright 고지가 모두 있음(§4). `LICENSES/` 디렉터리와 문서·이미지 파일의 tag는 채택하지 않고 root `LICENSE`를 유지 |
| `pyproject.toml` 라이선스 선언 | **변경함** — `license = { text = "MIT" }`를 PEP 639 SPDX expression `license = "MIT"`로 바꾸고 이를 지원하는 `setuptools>=77`로 build backend 최저 버전을 68에서 올림. Expression과 OSI classifier를 함께 두면 `setuptools>=77`이 `InvalidConfigError`를 내므로 `License :: OSI Approved :: MIT License` classifier는 제거. Wheel metadata는 기존 `License: MIT` 대신 Metadata-Version 2.4의 `License-Expression: MIT`를 포함하며 `pip-licenses`가 이를 읽음(§3.2). `importlib.metadata`의 `License` key나 OSI classifier 문자열만 보는 도구는 FiTuna를 찾지 못하므로 `License-Expression` 또는 SPDX 식별자를 읽어야 함 |
| HuggingFace dataset-viewer **service** 이용약관 | **미확인** — `docs/OPEN_SOURCE_USAGE.md` §7에 명시. Server 구현의 Apache-2.0은 확인했지만 hosted service 약관은 검토하지 않음 |
| `docs/RESULTS.md` Run 4의 llama.cpp 빌드 버전 | **주장하지 않음** — Colab notebook이 tag를 고정하지 않고 clone하므로 버전을 특정하지 않음(`docs/OPEN_SOURCE_USAGE.md` §13) |
| `docs/SBOM.md`의 pytest 버전이 "latest" | **유지** — version을 고정하지 않은 `dev` extra에 맞는 표현. Scan 당시 해석된 버전은 9.1.1(§3.2) |

---

## 관련 문서

- `docs/OPEN_SOURCE_USAGE.md` — FiTuna가 활용하는 구성요소, 결합 방식, 각
  라이선스 주장의 1차 출처
- `docs/SBOM.md` — 번호를 붙인 SBOM(표준 라이브러리 모듈 + 외부 실행 파일)
- `THIRD_PARTY_NOTICES.md` — llama.cpp MIT 전문을 포함한 필수 고지
- `docs/AI_MODEL_USAGE.md` — 모델별 AI 활용 공개
- `LICENSE` — FiTuna 자체 MIT license

<!-- REUSE-IgnoreEnd -->
