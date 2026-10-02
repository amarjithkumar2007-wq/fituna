# End-to-end 실측 결과

실제 하드웨어에서 실제 llama.cpp 바이너리와 open-weight 모델로 `fituna run`을
실행한 결과입니다. 아래 수치는 도구가 실제로 출력한 값이며 제시된 명령으로
재현할 수 있습니다. 절대값은 컴퓨터와 llama.cpp 빌드에 따라 달라지며, 여기서
중요한 것은 결과의 *양상*입니다.

**실행 환경(Run 1~3, 5)**

| | |
|---|---|
| 하드웨어 | Apple M3 Pro, 18 GB 통합 메모리(macOS, Apple Silicon) |
| llama.cpp | Homebrew build 9960 (`a935fbffe`) |
| 품질 corpus | wikitext-2-raw-v1 test split(`--ppl-chunks 32`), Run 3과 5는 한국어 Wikipedia도 측정 |
| FiTuna | 이 저장소 — Run 1~4는 `pip install -e .`, Run 5는 저장소 안에서 `python3.13 -m fituna`로 실행(같은 package라 동작 차이 없음) |

Run 4는 **NVIDIA Tesla T4(Linux, Google Colab)**에서 같은 실험을 반복한
결과입니다.

---

## Run 1 — SmolLM2-135M-Instruct(Apache 2.0)

기반 모델: `SmolLM2-135M-Instruct-f16.gguf`(F16, 258 MB).

```bash
fituna run --model SmolLM2-135M-Instruct-f16.gguf \
  --target-tps 240 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --wikitext wikitext-2-raw-test.txt --out ./out --resume
```

| 후보 | 파일 크기(MiB) | 실측 gen tok/s(full offload, ctx 4096) | F16 대비 실측 품질 손실 | 목표 240 판정 |
|---|---|---|---|---|
| Q8_0 | 138 MiB | 205.91 | 0.29 % | **미달** — 조기 종료 B |
| **Q6_K** | 132 MiB | **249.50** | **0.53 %** | **통과(ngl=30)** |
| Q5_K_M | 107 MiB | 233.26 | 3.32 % | 조기 종료로 측정하지 않음 |
| Q4_K_M | 101 MiB | 244.34 | 4.74 % | 조기 종료로 측정하지 않음 |

*(단위 정정: 이전에는 이 열을 "MB"로 표시했지만 수치는 처음부터 binary MiB였습니다.
다시 만든 파일을 `ls -l`로 재측정한 값은 Q8_0 144,811,072 B, Q6_K
138,382,912 B, Q5_K_M 112,103,488 B, Q4_K_M 105,454,144 B입니다. FiTuna의
`artifact:` 줄은 `report.human_size`가 decimal SI MB를 사용하므로 같은 Q6_K
파일을 `138.4 MB`로 출력합니다. 단위만 다를 뿐 불일치가 아닙니다.
`REVIEWERS.md` §5-2와 `docs/DEMO_SCRIPT.md` §4의 재구성한 보고서에 나오는
`110.0 MB`·`111.0 MB`는 그 재구성이 `artifact:`로 가리킨 대체 파일 크기이지
이 Q6_K의 크기가 아닙니다. 해당 block에서 Run 1 실측값은 tok/s와 perplexity뿐입니다.)*

*(Q5_K_M·Q4_K_M의 tok/s는 모든 후보를 측정한 아래 목표 300 실행에서 가져왔습니다.
목표 240에서는 Q6_K에서 탐색이 끝나 두 후보를 bench하지 않습니다. 조기 종료가
의도대로 동작한 결과입니다.)*

- 품질 손실은 F16 기준 대비 상대 perplexity 증가율입니다(기준 PPL 18.2407 →
  Q6_K 18.3377 = **+0.53%**).
- **당연해 보이는 순위가 두 번 틀립니다.** 최고 품질 quant인 Q8_0이 *가장
  느리고*, 이 하드웨어에서는 더 작은 Q4_K_M(244.34)이 더 큰 Q6_K(249.50)보다
  눈에 띄게 *느립니다*. 크기 기반 heuristic은 양쪽 모두 틀리지만 측정은 틀리지
  않습니다.
- 목표를 300 tok/s로 올리면 모든 후보가 미달합니다. FiTuna는 조용히 실패하지
  않고 종료 코드 3과 함께 가장 가까운 최선 설정(Q6_K, 249.50 tok/s)을
  보고합니다. Cold cache에서 품질 단계와 bench 4회를 포함해 **33.6초**가
  걸렸습니다.
- 목표 240 탐색 시간은 이진탐색 bench cache가 비어 있을 때 **75.7초**입니다.
  즉시 `--resume`으로 재실행하면 **0.75초**에 같은 답을 냅니다. 전체 결과는
  `out/.fituna_cache.sqlite3`에서 재현할 수 있습니다.
- 디스크 사용량은 양자화 파일 4개에 478 MB이며 실행 사이에 재사용합니다.

## Run 2 — Qwen3-4B-Instruct-2507 (Apache 2.0)

기반 모델: `Qwen3-4B-Instruct-2507-F16.gguf`(F16, 7.5 GB).

```bash
fituna run --model Qwen3-4B-Instruct-2507-F16.gguf \
  --target-tps 30 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --wikitext wikitext-2-raw-test.txt --out ./out --resume
```

| 후보 | 파일 크기 | F16 대비 실측 품질 손실 | 실측 gen tok/s(full offload, ctx 4096) | 목표 30 판정 |
|---|---|---|---|---|
| Q6_K | 3.1 GB | **−0.30 %**(F16보다 좋은 PPL) | 28.48 | 미달 — 조기 종료 B |
| Q8_0 | 4.0 GB | +0.07 % | 24.22 | 미달 — 조기 종료 B |
| Q5_K_M | 2.7 GB | +1.53 % | 29.59 | 0.41 tok/s 미달 |
| **Q4_K_M** | 2.3 GB | **+1.73 %** | 36.50 | **통과 — 최소 ngl=33 → 30.81 tok/s** |

이 실행은 조회표가 알 수 없는 세 가지를 보여 줍니다.

- **가정한 품질 순서가 틀렸습니다.** wikitext-2 32 chunk 측정에서 Q6_K의
  perplexity는 F16 기준 8.8688보다 *좋았지만* Q8_0은 0.07% 나빠졌습니다.
  실측 탐색 순서는 관례적인 Q8_0 우선이 아니라 Q6_K → Q8_0 → Q5_K_M →
  Q4_K_M이 됐습니다. FiTuna는 가정하지 않고 측정값으로 정렬합니다.
- **아슬아슬한 미달은 측정하지 않으면 보이지 않습니다.** Q5_K_M은 목표에 0.41
  tok/s 모자랍니다. "이 정도 하드웨어면 Q5로 30 tok/s가 충분하다"는 heuristic은
  목표를 놓치는 설정을 내놓습니다.
- **답은 quant 하나가 아니라 설정입니다.** 승자는 full offload 36이 아닌
  `-ngl 33`의 Q4_K_M입니다. 이진탐색이 목표를 만족하는 *최소* GPU offload
  30.81 tok/s를 찾았고, 실측 품질 손실은 PPL 8.8688 → 9.0225인 1.73%로 5%
  예산 안입니다.

같은 session에서 확인한 안정성 기록입니다.

- 최소 ngl 이진탐색에서 4B 모델의 `ngl=0` CPU 전용 probe는 300초 timeout 안에
  bench를 끝내지 못했습니다. FiTuna는 탐색을 중단하지 않고 목표 미달 값으로
  기록한 뒤 계속합니다(`[Q4_K_M] ngl=0 bench timed out -- treating as 0
  tok/s`). 실제 하드웨어에서 이 상황을 겪은 뒤 `BenchTimeoutError` 처리를
  만들었습니다.
- 즉시 `--resume`으로 재실행하면 **0.88초**에 같은 답을 얻습니다.
- 디스크는 양자화 파일 4개에 12.1 GB, F16 기반 파일에 7.5 GB를 사용합니다.
- 실측 단계별 시간은 cold 품질 단계(기준 PPL + quantize 4회 + PPL 4회)
  **5분 1초**, 속도 탐색 **12분 53초**입니다. 이 중 5분은 의도한 ngl=0
  timeout 한 번이며 full cache `--resume`은 **0.88초**입니다.

<a id="run-3--english-vs-korean-quality-corpus-same-model-same-quants"></a>

## Run 3 — 영어와 한국어 품질 corpus 비교(같은 모델·quant)

품질 손실은 corpus에 대한 perplexity 증가율로 측정합니다. 그렇다면 어떤 corpus를
써야 할까요? 같은 Qwen3-4B-Instruct-2507 양자화 파일을 영어 기본값인
wikitext-2 test split과 한국어 Wikipedia(`wikimedia/wikipedia`
`20231101.ko`, 처음 500개 문서, CC BY-SA)에서 `--ppl-chunks 32`로
측정했습니다.

| Quant | 품질 손실(영어 wikitext) | 품질 손실(한국어 kowiki) |
|---|---|---|
| Q6_K | −0.30 % | −0.06 % |
| Q8_0 | +0.07 % | −0.01 % |
| Q5_K_M | +1.53 % | +0.48 % |
| Q4_K_M | +1.73 % | +0.77 % |

측정에서 확인한 두 가지입니다.

- **순위는 우연히 같았습니다.** 두 corpus 모두 Q6_K가 가장 좋고 Q4_K_M이 가장
  나쁩니다. 순서가 항상 바뀐다고 주장하지 않습니다.
- **크기는 두 배 넘게 다르며 실제 판정도 바뀝니다.**
  `--max-quality-loss 1`에서 영어 corpus는 품질 gate 조기 종료 A로
  Q5_K_M·Q4_K_M을 탈락시킵니다. 남은 quant는 너무 느려 탐색 결과가 정직하게
  **BEST EFFORT(목표 미달)**가 됩니다. 한국어 corpus에서는 네 후보가 모두
  통과하고 Q4_K_M이 ngl=34에서 목표를 만족합니다(30.06 tok/s, 손실 0.77%).
  모델, 컴퓨터, 목표, 예산이 모두 같아도 **corpus 하나가 가능 여부를 뒤집습니다.**
  한국어 사용자를 위한 설정이라면 한국어 텍스트로 gate를 검사하세요
  (`fituna fetch-corpus --lang ko --out kowiki-corpus.txt`로 받은 뒤
  `--quality-corpus kowiki-corpus.txt`).

**Run 5 뒤에 추가한 단서가 있습니다.** 이 값은 `--ppl-chunks 32` 추정치입니다.
Run 5에서 FiTuna가 버리던 해당 chunk 수의 오차 범위를 다시 측정했더니 손실률로
약 ±3%p였습니다(아래 "Perplexity 차이는 얼마나 커야 하는가?" 참조). Run 5의
모델과 파일에서 측정한 값이고 Run 3의 Qwen3-4B 양자화 파일은 남아 있지 않아,
여기서는 ±3%p를 실측이 아닌 추정으로 봐야 합니다. 위 판정 역전은 도구의 실제
동작입니다. 사용자가 두 명령을 실행하면 서로 다른 판정을 얻으며 이것이 실험의
핵심입니다. 하지만 기저 품질 차이가 추정기의 해상도보다 크다는 사실은 **입증하지
못했습니다.** 영어 1.73%와 한국어 0.77%는 pairing되지 않은 corpus 간 0.96%p
차이로 위 오차 범위와 같은 규모이며, 두 값 모두 1% gate 양쪽에 걸쳐 있습니다.
"한국어 텍스트의 저하가 더 작다고 입증됐다"가 아니라 "이 예산에서는 gate에 쓰는
corpus가 판정을 정한다"로 읽어야 합니다. Run 5는 한국어 모델로 같은 비교를
측정했지만 전자의 주장을 입증하지 못했습니다.

이 실험을 설계하면서 실제 cache bug도 찾았습니다. 품질 결과 key가 (model,
quant, chunks)이고 corpus를 포함하지 않아 두 번째 corpus가 첫 번째 수치를 조용히
재사용했습니다. 이제 cache key에 corpus fingerprint를 포함하며 이 정직한 동작도
regression test로 확인합니다.

<a id="run-4--nvidia-tesla-t4-linux-google-colab"></a>

## Run 4 — NVIDIA Tesla T4, Linux(Google Colab)

Run 1과 같은 모델·명령·목표를 다른 하드웨어에서 실행했습니다.
[notebooks/colab_nvidia_verification.ipynb](../notebooks/colab_nvidia_verification.ipynb)
(무료 T4 tier, CUDA로 llama.cpp source build)로 재현했습니다. `fituna
detect-hw`는 `nvidia-smi` 해석 경로에서 `nvidia / Tesla T4 / 15360 MB VRAM /
linux`를 정확히 자동 감지했습니다.

| 후보 | 실측 품질 손실(T4/CUDA) | macOS/Metal | 실측 gen tok/s(T4) | macOS | 목표 240 판정 |
|---|---|---|---|---|---|
| Q8_0 | — | — | 202.70 | 205.91 | 미달 |
| Q6_K | 0.83 % | 0.53 % | **205.50** | 249.50 | 미달 → **최선 결과** |
| Q5_K_M | — | — | 172.03 | 233.26 | 미달 |
| Q4_K_M | **5.22 % → 5% gate 탈락(조기 종료 A)** | 4.74 % → 통과 | bench하지 않음 | 244.34 | — |

Cold 탐색은 **61초**, `--resume` 재실행은 같은 출력까지 **1.45초**입니다. 결과는
BEST EFFORT(Q6_K, 205.50 tok/s, 손실 0.83%)입니다. M3 Pro가 만족한 240 tok/s
목표를 T4에서는 달성할 수 없다고 정확히 보고합니다.

조회표가 알 수 없는 platform 간 사실 세 가지입니다.

- **Platform에 따라 품질 gate 판정이 뒤집혔습니다.** Q4_K_M은 Metal에서 손실
  4.74%로 5% 예산을 통과했지만 CUDA에서는 5.22%로 gate에서 탈락했습니다. 같은
  파일과 corpus라도 backend 수치가 달라 가능한 후보군이 달라집니다. Run 3과 같은
  단서가 있습니다. 32 chunk 추정치 사이 0.48%p는 Run 5가 자체 파일에서 측정한
  ±3%p 오차 범위와 같은 규모입니다. 두 backend가 여기서 다른 판정을 만들었다는
  사실만 입증했으며 0.48%p 차이가 재현된다는 뜻은 아닙니다. 그래도 Run 3의
  corpus 간 비교와 달리 같은 corpus·파일에서 backend만 달라 일부 pairing된
  비교입니다.
- **속도 순위는 platform마다 다릅니다.** T4에서는 Q6_K가 Q8_0과 세 후보 중
  가장 느린 Q5_K_M보다 빠릅니다. M3 Pro의 순서는 또 다릅니다.
- **가능 여부도 하드웨어에 상대적입니다.** 같은 명령이 한 컴퓨터에서는 통과하고
  다른 컴퓨터에서는 최선 결과에 그칩니다. 하드웨어를 고르거나 목표를 낮추기 전에
  사용자에게 필요한 답이 바로 이것입니다.

<a id="run-to-run-variance-measured-not-hidden"></a>

### 실행 간 변동성(숨기지 않고 실측)

노트북 benchmark 수치는 발열에 민감합니다. 새 `--out`으로 cache가 완전히 비어
있지만 한 시간 연속 bench로 컴퓨터는 뜨거운 두 번째 session에서
Q6_K·Q8_0·Q5_K_M은 ±0.5 tok/s 이내로 재현됐습니다. 반면 Q4_K_M full
offload는 첫 session 36.50 tok/s와 달리 22.73 tok/s였습니다. 같은 설정으로
`llama-bench`를 즉시 세 번 직접 반복한 결과입니다.

```
37.53 tok/s ± 0.20      31.97 tok/s ± 6.74      35.35 tok/s ± 3.26
```

지속 성능은 약 36 tok/s이고 22.73은 발열 저하 이상치입니다. 컴퓨터가 부하를
받을 때 내부 표준편차가 ±6.7까지 커진 점도 이를 보여 줍니다. 설계상 실무에
미치는 영향은 두 가지입니다.

- FiTuna는 *해당 session의 발열 조건*에서 측정한 값을 보고합니다. 결과 명령을
  바로 실행할 때 실제로 얻게 될 성능입니다.
- 목표가 후보의 지속 속도에서 몇 tok/s 안쪽이라면, 여기서 30과 Q5_K_M의
  29.6~29.7처럼 판정을 경계선으로 보고 컴퓨터가 평소 상태일 때 탐색을 다시
  실행하세요. llama-bench의 실행별 표준편차를 보고서에 표시해 경계 판정을 자동
  경고하는 기능은 roadmap에 있습니다.

## Run 5 — 한국어 open-weight 모델 Midm-2.0-Mini-Instruct(MIT)

Run 1~4는 주로 영어로 학습한 모델을 사용했습니다. Run 5는 모델 자체가 한국어일
때도 corpus 민감도가 유지되는지 보려고 **한국어** open-weight 모델에서 Run 3의
영어·한국어 품질 corpus 비교를 같은 모델·양자화 파일로 반복했습니다.

결과부터 말하면 손실 *크기*는 corpus마다 다르지만 최종 판정은 같았습니다. 처음
결론으로 보고했던 중간 순위 차이는 **자체 log를 이용한 안정성 검사를 통과하지
못했습니다.** 자세한 내용은 아래 "Perplexity 차이는 얼마나 커야 하는가?"에
있습니다.

기반 모델 `K-intelligence/Midm-2.0-Mini-Instruct`는 parameter
2,305,517,312개, 48 layer입니다. HuggingFace `license` field는 `mit`이고 저장소
`LICENSE.txt`는 "Copyright (c) 2025 KT Corporation"을 포함한 MIT 원문이며
gated 저장소가 아닙니다. 이미 만든 BF16 GGUF인
`mykor/Midm-2.0-Mini-Instruct-gguf`의
`Midm-2.0-Mini-Instruct-BF16.gguf`(4,617,053,184 bytes = 4.30 GB)를 써서
`convert_hf_to_gguf.py`, torch, transformers는 사용하지 않았습니다.

실제로 byte를 배포한 주체이므로 재배포자 라이선스도 따로 확인했습니다.
`mykor/Midm-2.0-Mini-Instruct-gguf`는 model card front matter에 `license:
mit`, `base_model: K-intelligence/Midm-2.0-Mini-Instruct`를 선언합니다. 자체
`LICENSE.txt`도 upstream과 같은 "Copyright (c) 2025 KT Corporation" 고지가
있는 MIT 원문입니다. HuggingFace API 결과는 `gated: false`, `private: false`입니다.
따라서 upstream weight와 실제 실행한 GGUF 변환본 모두 MIT입니다.

Corpus는 내장 표준 라이브러리 downloader로 받았으며 `pip install datasets`는
사용하지 않았습니다.

```bash
fituna fetch-corpus --lang ko --out kowiki-corpus.txt   # 500행, 5.9 MB
fituna fetch-corpus --lang en --out wikitext-2-raw-test.txt  # 1,000행, 316 KB
```

### 목표 선정 방법

목표는 실제로 실패할 가능성이 있어야 합니다. 정직하게 고르려고 먼저 일부러
달성할 수 없는 `--target-tps 9999`로 탐색했습니다. 첫 승자에서 조기 종료하지
않고 모든 후보가 품질 gate와 full offload bench를 거치도록 만든 것입니다. Cold
상태에서 **4분 59.81초**가 걸렸고 종료 코드 3(`BEST EFFORT`)과 다음 실측 full
offload 범위를 얻었습니다.

| Quant | 실측 gen tok/s @ ngl=48 |
|---|---|
| Q8_0 | 34.26 |
| Q5_K_M | 38.76 |
| Q6_K | 38.96 |
| Q4_K_M | 44.62 |

**40 tok/s**는 이 범위 안에서 후보 세 개보다 높고 정확히 하나보다 낮습니다.
Q8_0·Q6_K·Q5_K_M은 실제로 미달하고 Q4_K_M은 실제로 통과하므로 너무 쉽게
통과하거나 실패할 수 없는 값입니다. 실측 범위 밖이 아니라 범위를 가르는
수치입니다. 아래 값은 모두 `--target-tps 40`에서 측정했습니다.

### 두 차례 실행

```bash
# 한국어 품질 corpus
fituna run --model Midm-2.0-Mini-Instruct-BF16.gguf \
  --target-tps 40 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --quality-corpus kowiki-corpus.txt --out ./out --resume

# 영어 품질 corpus — corpus만 다름
fituna run --model Midm-2.0-Mini-Instruct-BF16.gguf \
  --target-tps 40 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --quality-corpus wikitext-2-raw-test.txt --out ./out --resume
```

| 후보 | 파일 크기 | 품질 손실(한국어 kowiki) | 품질 손실(영어 wikitext) | 실측 gen tok/s(ngl=48, ctx 4096) | 목표 40 판정 |
|---|---|---|---|---|---|
| Q8_0 | 2.29 GB | **−0.02 %** | +0.15 % | 34.26 | 미달 — 조기 종료 B |
| Q6_K | 1.77 GB | +0.59 % | +0.46 % | 38.96 | 미달 — 조기 종료 B |
| Q5_K_M | 1.54 GB | +0.53 % | +1.00 % | 38.76 | 미달 — 조기 종료 B |
| **Q4_K_M** | **1.33 GB** | **+2.58 %** | **+3.78 %** | **44.62** | **통과(ngl=48)** |

BF16 기준 perplexity는 한국어 **9.9511**, 영어 **8.7900**입니다. 표의 모든 품질
수치는 `--ppl-chunks 32` 추정치이며 `llama-perplexity`가 출력하지만 FiTuna가
저장하지 않는 오차 범위가 있습니다. 하나만 인용하고 나머지를 추정하지 않고 이
chunk 수에서 한국어 모든 행을 다시 측정했습니다: BF16 기준 **±0.29318**,
Q8_0 **±0.29307**, Q6_K **±0.29529**, Q5_K_M **±0.29372**, Q4_K_M
**±0.30055**. 손실률로 바꾸면 모든 행이 약 ±3%p입니다. 인접 행의 차이로 결론을
내리기 전에 아래 "Perplexity 차이는 얼마나 커야 하는가?"를 읽어 주세요. 두
실행은 같은 결과로 끝납니다.

```
FiTuna result: MEETS TARGET

  quant           : Q4_K_M
  ngl             : 48
  ctx             : 4096

  prompt tok/s (pp): 305.88
  gen tok/s    (tg): 44.62

  perplexity      : 10.2082 (baseline 9.9511)   # 한국어
  quality loss    : 2.58%
```

<a id="how-big-is-a-perplexity-gap-the-error-bar-we-had-been-discarding"></a>

### Perplexity 차이는 얼마나 커야 하는가?(버리고 있던 오차 범위)

위 품질 수치는 모두 32 chunk 추정치입니다. `llama-perplexity`는 각 값 옆에 오차
범위를 출력하지만 FiTuna는 이를 저장하지 않았습니다. 출력은 `Final estimate: PPL
= 10.0099 +/- 0.29529`인데 `quality.py` 정규식이 PPL만 잡고 `+/-`를 버립니다.
이 실행을 다시 읽었을 때 아래 순위 차이가 signal인지 말할 수 없어 직접
측정했습니다. 같은 파일 네 개와 두 corpus에 바이너리를 직접 실행하고 측정량을
**128 chunk**로 늘렸습니다. 기존 실행의 4배이자 영어 corpus가 지원하는 가장 큰
2의 거듭제곱입니다. wikitext-2 test는 `n_ctx=512`에서 143 chunk, kowiki는
2,368 chunk를 담습니다. 반복이 아니라 확장입니다. 아래 두 번째 항목처럼 128
chunk 실행이 32 chunk 실행을 *포함*하므로 두 열은 독립 반복 측정이 아닙니다.

```bash
llama-perplexity -m out/...-3e6680866409-Q6_K.gguf -f kowiki-corpus.txt --chunks 128
```

`--chunks 32`로 호출하면 cache의 모든 한국어 값(`9.9511`, `9.9488`,
`10.0099`, `10.0037`, `10.2082`)을 정확히 재현합니다. 도구와 같은 측정에서 오차
범위만 보존한 것입니다. 계산은 deterministic하며 아래 두 번째 항목처럼 같은
실행이므로, 값이 일치한다는 사실은 연결 경로 점검이지 반복 검증이 아닙니다.

| 한국어(kowiki) | PPL @128 chunks | 손실 @32 | 손실 @128 |
|---|---|---|---|
| BF16 기준 | 10.6238 ± 0.16350 | — | — |
| Q8_0 | 10.6187 ± 0.16335 | −0.02 % | −0.05 % |
| **Q5_K_M** | **10.7413 ± 0.16504** | **+0.53 %** | **+1.11 %** |
| **Q6_K** | **10.7470 ± 0.16569** | **+0.59 %** | **+1.16 %** |
| Q4_K_M | 11.0577 ± 0.17044 | +2.58 % | +4.08 % |

| 영어(wikitext-2) | PPL @128 chunks | 손실 @32 | 손실 @128 |
|---|---|---|---|
| BF16 기준 | 9.9155 ± 0.14863 | — | — |
| Q8_0 | 9.9217 ± 0.14873 | +0.15 % | +0.06 % |
| **Q6_K** | **9.9778 ± 0.14975** | **+0.46 %** | **+0.63 %** |
| **Q5_K_M** | **10.0197 ± 0.15035** | **+1.00 %** | **+1.05 %** |
| Q4_K_M | 10.2945 ± 0.15471 | +3.78 % | +3.82 % |

결론을 크게 바꾸는 순서대로 다섯 가지를 확인했습니다.

- **영어 순위는 전체 실행에서 안정적이지만 한국어 순위는 오르내립니다.**
  `llama-perplexity`는 chunk마다 누적 perplexity를 출력하므로 각 실행에는 중첩된
  추정치 128개가 있고, 이를 읽으면 별도 비용 없이 안정성을 검사할 수 있습니다.
  영어에서 두 quant의 margin은 n=4에 양수가 된 뒤 남은 125개 지점에서 모두
  양수입니다. 읽을 수 있는 **모든** chunk 수에서 Q6_K가 Q5_K_M보다 높고 부호가
  한 번도 바뀌지 않습니다. 한국어는 같은 n=4부터 반대이며 Q5_K_M이 0.0867%p
  앞섭니다. n=16 뒤에는 부호가 **아홉 번** 바뀌고, n=4부터 세면 열한 구간에서
  두 순서가 번갈아 나타나 총 열 번 바뀝니다.

  | Korean, n = | 4 | 5–16 | 17–22 | 23–24 | 25–66 | 67 | 68–76 | 77–99 | 100–106 | 107–108 | 109–128 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 앞선 quant | Q5_K_M | Q6_K | Q5_K_M | Q6_K | Q5_K_M | Q6_K | Q5_K_M | **Q6_K** | Q5_K_M | Q6_K | **Q5_K_M** |

  이 결과는 근소한 차이가 조용히 안정되는 모습이 아닙니다. 공개한 순서와 *반대*
  순서를 23개 지점 연속 유지한 n=77~99의 한국어 절대 margin 평균은
  **0.0545%p**로 n=128의 **0.0537%p**보다 조금 *큽니다*. 반대 순서도 같거나
  더 강하게 유지됩니다. n=96에서는 두 corpus 모두 Q6_K를 앞에 두어 순위 변경이
  아예 없습니다. 한국어 열의 순서는 실행이 어디서 멈췄는지에 따라 달라집니다.
- **32 chunk와 128 chunk 값은 두 측정이 아니라 하나입니다.**
  `llama-perplexity`는 항상 corpus 처음부터 순회하므로 128 chunk 실행이 자체 32
  chunk 실행을 *포함*합니다. 한국어 Q8_0 추적의 chunk `[32]` 값 `9.9488`은
  FiTuna가 `--ppl-chunks 32`에서 저장한 값과 정확히 같습니다. Q6_K `10.0099`,
  Q5_K_M `10.0037`, 한국어 기준 `9.9511`, 영어 기준 `8.7900`도 같습니다.
  "32 chunk와 128 chunk에서 다시 재현"했다고 하면 중첩된 한 측정을 두 번
  인용하는 셈입니다. 이전 문서는 이를 독립 확인 두 번으로 잘못 소개했습니다.
  안정성 근거는 위 chunk별 추적이지 이 일치가 아닙니다.
- **두 margin 모두 오차 범위를 넘지 못하며 강도도 전혀 같지 않습니다.** 128
  chunk에서 각 perplexity 오차는 ±0.15~0.17이며 손실로는 ±1.5%p입니다. 영어
  margin 0.423%p는 약 4분의 1, n=128 한국어 margin 0.054%p는 약 29분의
  1입니다. 영어 쪽 순위 차이는 한국어보다 약 8배 강하고, 한국어와 달리 모든
  지점에서 부호가 같습니다.
- **인용할 수 있는 `+/-`는 정확히 원하는 오차가 아닙니다.** 다른 한국어 표본에서
  절대 perplexity가 얼마나 움직이는지를 나타내는 corpus sampling 오차입니다.
  *같은* chunk로 평가한 두 quant 비교는 오차가 훨씬 작은 paired 비교입니다.
  그래서 영어 margin이 오차 범위의 4분의 1인데도 chunk 수 125개 연속으로 부호를
  유지할 수 있습니다. `llama-perplexity`는 paired 오차를 보고하지 않아 어느
  margin에도 신뢰 수준을 붙일 수 없습니다. 대신 chunk별 추적은 오차 model 없이
  직접 안정성을 검사하며 한국어 margin은 이 검사를 통과하지 못합니다.
- **절대 손실은 chunk 수에 크게 의존하므로 항상 chunk 수를 밝혀야 합니다.**
  Q4_K_M 한국어 손실은 32 chunk 2.58%에서 128 chunk 4.08%로, Q6_K는
  0.59%에서 1.16%로 바뀝니다. 위 32 chunk 값은 문서의 명령과 cache가 실제로
  만든 정확한 결과지만 quant 자체의 고정 속성이 아니라 32×512 token 추정치입니다.
  네 후보 중 Q4_K_M 저하만 자체 오차 범위를 넘을 만큼 큽니다(128 chunk에서 Δppl
  0.434, 오차 ±0.17). **따라서 판정도 결과 수치뿐 아니라 `--ppl-chunks`에
  따라 달라집니다.** 이 실행의 `--max-quality-loss 5`에서 Q4_K_M 한국어
  여유는 32 chunk 2.42%p, 128 chunk 0.92%p입니다. 같은 파일·corpus·gate에
  근거를 네 배로 늘리자 여유가 4분의 1이 됐습니다. 작은 chunk 수의 PASS는 그
  chunk 수에서만 PASS입니다. 후보가 예산 근처라면 chunk를 늘려 다시 측정한 뒤
  판정을 신뢰하세요.

### 조회표가 알 수 없는 것

- **Corpus가 순서를 바꿨다고 입증하지 못했습니다.** 실측 손실로 정렬한 두 실행은
  한국어 Q8_0 → **Q5_K_M → Q6_K** → Q4_K_M, 영어 Q8_0 → **Q6_K →
  Q5_K_M** → Q4_K_M으로 서로 다른 중간 순서를 실제 보고합니다. 하지만 순위
  변경을 주장하려면 양쪽 차이가 모두 실제여야 하며 영어 쪽만 측정을 견딥니다.
  영어는 n=4~128 모든 chunk 수에서 부호 변화 없이 Q6_K가 Q5_K_M보다 앞섭니다.
  한국어는 같은 두 quant의 부호가 아홉 번 바뀌며, n=77~99에서는 n=128의 반대인
  *영어* 순서를 같거나 더 강하게 유지합니다. n=96에서는 두 corpus 순위가 같아
  순위 변경이 없습니다. 따라서 corpus가 순위를 바꿨다고 말할 수 없고 한국어 열이
  순서를 정하기에 불안정하다고만 말할 수 있습니다. Run 3도 영어 학습 모델에서
  같은 비교를 하고 "순위가 우연히 같았으며 항상 바뀐다고 주장하지 않는다"고
  밝혔습니다. Run 5도 순위 변경을 주장할 수 없습니다. 이전 문서는 "corpus가
  실측 품질 순위를 뒤집었고 재현 가능하다"고 썼지만 위 chunk별 추적과 자체 log가
  두 주장을 반박합니다. 그럴듯한 가정이 자체 측정에서 실패한 것으로, 도구가
  의도대로 작동한 사례입니다.
- **Q8_0은 부호가 바뀌었고 이 관찰은 추적 전체에서 안정적입니다.** 한국어에서는
  32 chunk −0.02%, 128 chunk −0.05%로 양자화 기반 BF16보다 perplexity가 아주
  조금 *좋았습니다*. 영어에서는 각각 +0.15%, +0.06%입니다. 위 Q6_K·Q5_K_M
  순서와 달리 chunk별로도 유지됩니다. n=4~128의 125개 지점 중 한국어 Q8_0은
  117개에서 기준보다 낮고 초기에 부호가 한 번만 바뀝니다. 영어 Q8_0은 125개
  모두 기준보다 높습니다. "Q8_0 ≈ 무손실"이라고만 저장한 표는 0의 어느 쪽인지
  숨깁니다. 같은 32 chunk에서 Δppl은 한국어 0.0023, 영어 0.0128이고 실측 오차는
  ±0.26~0.29입니다. 다른 크기 실행의 128 chunk ±0.16을 재사용하지 않고 빠져
  있던 영어 32 chunk 오차도 직접 측정했습니다. 해결된 결론이라고 부르지는 않지만
  순위 차이보다는 일관된 관찰입니다.
- **Run 3의 "한국어 손실이 더 작다"는 양상은 일반화되지 않습니다.** Run 3은 네
  행 중 세 행에서 한국어 값이 영어보다 작지만 Q6_K가 이미 예외입니다(한국어
  −0.06%, 영어 −0.30%). 위 32 chunk 표에서도 Q8_0(−0.02 대 +0.15),
  Q5_K_M(0.53 대 1.00), Q4_K_M(2.58 대 3.78)은 같은 양상이지만 Q6_K만
  **반대**입니다(한국어 0.59, 영어 0.46). 0.14%p 차이는 ±1.5%p 오차 안입니다.
  Chunk별로 보면 n=4~128에서 한국어 값이 더 작은 지점은 Q8_0 125/125,
  Q4_K_M 73/125, Q5_K_M 50/125, Q6_K 4/125입니다. 애초에 깔끔한 규칙이
  아니며 안정적으로 따르는 quant는 Q8_0뿐입니다. Corpus 효과 방향은 quant마다
  측정해야 합니다.
- **이번에는 판정이 뒤집히지 않았으며 그대로 밝힙니다.** 1% 예산에서 corpus
  하나가 가능 여부를 정한 Run 3과 달리 `--max-quality-loss 5`에서는 네 후보가
  두 corpus의 gate를 모두 통과하고 같은 승리 설정을 고릅니다. Corpus는 수치를
  바꿨지만 최종 답은 바꾸지 않았고, 위 항목처럼 순위를 바꿨다고 입증하지도
  못했습니다. 측정한 그대로이며 판정 역전처럼 꾸미지 않습니다.
- **"최소 ngl"의 답은 48개 전부였고 여유는 layer 하나입니다.** `-ngl`
  이진탐색은 더 작은 값이 목표를 만족하지 못해 전체 48을 반환했습니다. Q4_K_M의
  실측 offload curve는 다음과 같습니다.

  > **정정(0.3.2).** 이 실행 당시 FiTuna는 `-ngl 48`(block 수)을 full offload로
  > 보았습니다. llama.cpp는 출력층을 layer 하나로 더 세므로 실제 full offload는
  > `-ngl 49`이고, `-ngl 48`에서는 layer 하나가 CPU에 남습니다. SmolLM2-135M(block 30)에서
  > `-ngl 30`은 "offloaded 30/31 layers"로 233~240 tok/s, `-ngl 31`은 258~262 tok/s를
  > 기록했습니다. 아래 curve는 48까지의 실측이며, 0.3.2부터 상한은 `n_layers + 1`입니다.

  | ngl | 0 | 24 | 36 | 42 | 45 | 47 | 48 |
  |---|---|---|---|---|---|---|---|
  | gen tok/s | timeout | 10.86 | 17.56 | 23.53 ⚠ | 32.43 | **39.32** | **44.62** |
  | llama-bench 표준편차 | — | ±0.05 | ±0.37 | **±4.94** ⚠ | ±0.26 | ±0.15 | ±1.25 |

  48개 중 layer **하나**를 GPU에서 내리면 5.30 ± 1.26 tok/s, 곧 4.0~6.6 또는
  처리량 11.9% ± 2.8%p를 잃습니다. 결과 39.32로 목표 40에 0.68 모자랍니다.
  이 차이는 ngl=47의 ±0.15에 비해 4.5σ라 분명합니다. Curve는 layer 수에
  선형도 아닙니다. 절반인 ngl=24는 10.86 tok/s로 full offload 속도의 50%가
  아닌 24%입니다. 고정된 경험칙으로 이 형태를 재현할 수 없습니다.

  ⚠ **ngl=42 지점은 발열 영향을 받았으므로 정상값으로 공개하지 않습니다.**
  llama-bench 표본 다섯 개는
  `[25.81, 25.69, 25.59, 25.84, 14.69]` — four tightly clustered at 25.73
  ±0.12에 모인 네 값과 붕괴한 한 값입니다. 이 한 번이 평균을 8.6% 낮춰 23.53으로
  만들고 표준편차를 curve의 다른 최악값보다 약 4배 큰 ±4.94로 키웁니다. Run 4의
  "실행 간 변동성"과 같은 징후입니다. 같은 명령(`llama-bench -m
  ...-Q4_K_M.gguf -ngl 42 -d 3456 -p 512 -n 128`)을 세 번 직접 다시 실행해도
  불안정성이 사라지지 않았습니다. 다만 한 시간 연속 perplexity 작업으로 컴퓨터가
  이미 뜨거운 상태라 이 설정이 발열 경계선이라는 사실만 입증하며 차가운 컴퓨터의
  값은 알 수 없습니다.

  ```
  19.84 ± 3.96      18.64 ± 3.57      19.32 ± 4.44
  ```

  세 실행 모두 자체 표본 다섯 개 안에서 같은 방식으로 무너졌습니다(예:
  `[21.97, 22.40, 23.13, 15.55, 13.53]`).

  같은 설정의 prompt 처리량도 첫 session 100.55 tok/s에서 세 반복 모두 약 80으로
  떨어져 컴퓨터 전체가 약 20% throttling됐습니다. 새 값으로 대체하지 않고 경고를
  붙인 이유입니다. 23.53은 오차 범위가 나쁜 하한으로 보세요. 정상 표본 네 개의
  평균도 25.73이라 ngl=42가 목표 40 미달이라는 이진탐색 판정에는 영향이 없습니다.
- **속도 역전 하나에도 단서를 명시합니다.** 1.77 GB인 Q6_K가 더 작은 1.54 GB
  Q5_K_M보다 *빠르게* 측정됐습니다(38.96 대 38.76). 크기가 작으면 빠르다는
  heuristic이 다시 반대로 작동했습니다. 하지만 llama-bench 실행별 표준편차가
  각각 ±0.53, ±0.11 tok/s라 Run 1의 분명한 역전과 달리 0.20 tok/s 차이는
  **실행 간 noise 안**입니다. Q6_K의 승리가 아니라 "구분할 수 없음"으로
  보고합니다. 그래도 Q6_K에 디스크를 15% 더 써서 측정 가능한 속도나 더 나은
  한국어 품질을 얻지 못했다는 실무 결론은 같습니다.

### 실측 시간과 디스크

| | |
|---|---|
| 목표 선정 probe(cold: 기준 PPL + quantize 4회 + PPL 4회 + bench 4회) | **4분 59.81초** |
| 목표 40 한국어 실행(cold `-ngl` 이진탐색, 새 bench 6회 — 7번째 curve 지점 ngl=48은 probe cache hit) | **12분 54.93초** |
| ↳ 의도한 `ngl=0` bench timeout | 5분 00초 |
| 목표 40 영어 실행(cold 품질 단계, 모든 bench cache hit) | **1분 53.19초** |
| 한국어 `--resume` 재실행 | **1.47초** |
| 영어 `--resume` 재실행 | **1.42초** |

두 `--resume` 재실행은 `out/.fituna_cache.sqlite3`에서 전체 보고서를 byte 단위로
재현합니다. 이번에는 산출물도 보존했습니다. 각 재실행 보고서 block과 원래 cold
실행 log(`run5-ko.log`, `run5-en.log`)의 보고서 block을 `diff -u`로 비교해
한국어 434 bytes, 영어 433 bytes가 완전히 같았고 둘 다 종료 코드 0이었습니다.
위 `--resume` 시간은 한 시간 연속 bench로 뜨거운 컴퓨터에서 수행한 이 검증
실행값입니다. 이전 session에는 0.955초·0.908초가 기록됐지만 디스크에 산출물을
남기지 않아 증명할 수 있는 값만 인용합니다. 디스크는 양자화 파일 네 개에
**6.9 GB**, BF16 기반 파일에 4.30 GB로 전체 작업 공간은 11 GB입니다.

2.3B 모델의 `ngl=0` CPU 전용 probe는 300초 timeout 안에 bench를 끝내지
못했습니다. FiTuna는 `[Q4_K_M] ngl=0 bench timed out -- treating as 0 tok/s
(below target)`를 log에 남기고 탐색을 계속했습니다. Run 2의 4B 모델에서 겪은
같은 `BenchTimeoutError` 경로를 두 번째 모델 종류에서도 재현했습니다.
