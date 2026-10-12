[한국어 원문](./RESULTS.md)

# End-to-End Measurement Results

These are results from running `fituna run` on real hardware with actual llama.cpp
binaries and open-weight models. The figures below are the values the tool
actually printed and can be reproduced with the commands shown. Absolute values
vary with the computer and llama.cpp build; what matters here is the *pattern*
of the results.

**Run environment (Runs 1-3, 5)**

| | |
|---|---|
| Hardware | Apple M3 Pro, 18 GB unified memory (macOS, Apple Silicon) |
| llama.cpp | Homebrew build 9960 (`a935fbffe`) |
| Quality corpus | wikitext-2-raw-v1 test split(`--ppl-chunks 32`), Runs 3 and 5 also measure Korean Wikipedia |
| FiTuna | This repository — Runs 1~4 use `pip install -e .`, Run 5 runs inside the repository with `python3.13 -m fituna` (the same package, with no behavior difference) |


Run 4 repeats the same experiment on **NVIDIA Tesla T4 (Linux, Google Colab)**.

---

## Run 1 - SmolLM2-135M-Instruct (Apache 2.0)

Base model: `SmolLM2-135M-Instruct-f16.gguf` (F16, 258 MB).


```bash
fituna run --model SmolLM2-135M-Instruct-f16.gguf \
  --target-tps 240 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --wikitext wikitext-2-raw-test.txt --out ./out --resume
```


| Candidate | File size (MiB) | Measured gen tok/s (full offload, ctx 4096) | Measured quality loss relative to F16 | Target 240 verdict |
|---|---|---|---|---|
| Q8_0 | 138 MiB | 205.91 | 0.29 % | **Below target** — Early exit B |
| **Q6_K** | 132 MiB | **249.50** | **0.53 %** | **Pass (ngl=30)** |
| Q5_K_M | 107 MiB | 233.26 | 3.32 % | Not measured due to early exit |
| Q4_K_M | 101 MiB | 244.34 | 4.74 % | Not measured due to early exit |


*(Unit correction: this column was previously labelled "MB", but the figures were
binary MiB from the start. Remeasuring regenerated files with `ls -l` gave Q8_0
144,811,072 B, Q6_K 138,382,912 B, Q5_K_M 112,103,488 B, and Q4_K_M
105,454,144 B. FiTuna's `artifact:` line uses decimal SI MB through
`report.human_size`, so the same Q6_K file is printed as `138.4 MB`. The units
are different, not the data. The `110.0 MB` and `111.0 MB` in the reconstructed
reports in `REVIEWERS.md` §5-2 and `docs/DEMO_SCRIPT.md` §4 are the sizes of
substitute files referenced by `artifact:` in those reconstructions, not the
size of this Q6_K file. Only tok/s and perplexity in those blocks are Run 1
measurements.)*

*(The Q5_K_M and Q4_K_M tok/s values come from the target-300 run below, which
measured every candidate. At target 240, search ends at Q6_K and does not benchmark
those two candidates. Early exit is working as intended.)*

- Quality loss is the relative increase in perplexity over the F16 baseline
  (baseline PPL 18.2407 → Q6_K 18.3377 = **+0.53%**).
- **The seemingly obvious ranking is wrong twice.** Q8_0, the highest-quality
  quantization, is the *slowest*. On this hardware, the smaller Q4_K_M (244.34)
  is noticeably *slower* than the larger Q6_K (249.50). A size-based heuristic
  gets both wrong; measurement does not.
- Raising the target to 300 tok/s leaves every candidate below target. FiTuna
  does not fail silently: it reports exit code 3 and the closest best-effort
  configuration (Q6_K, 249.50 tok/s). With a cold cache, including the quality
  stage and 4 benchmarks, this took **33.6 seconds**.
- Search for target 240 takes **75.7 seconds** when the binary-search benchmark
  cache is empty. An immediate rerun with `--resume` gives the same answer in
  **0.75 seconds**. Full results can be reproduced from
  `out/.fituna_cache.sqlite3`.
- The 4 quantized files use 478 MB of disk space and are reused between runs.

## Run 2 - Qwen3-4B-Instruct-2507 (Apache 2.0)

Base model: `Qwen3-4B-Instruct-2507-F16.gguf` (F16, 7.5 GB).


```bash
fituna run --model Qwen3-4B-Instruct-2507-F16.gguf \
  --target-tps 30 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --wikitext wikitext-2-raw-test.txt --out ./out --resume
```


| Candidate | File size | Measured quality loss relative to F16 | Measured gen tok/s (full offload, ctx 4096) | Target 30 verdict |
|---|---|---|---|---|
| Q6_K | 3.1 GB | **−0.30 %** (better PPL than F16) | 28.48 | Below target — Early exit B |
| Q8_0 | 4.0 GB | +0.07 % | 24.22 | Below target — Early exit B |
| Q5_K_M | 2.7 GB | +1.53 % | 29.59 | 0.41 tok/s below target |
| **Q4_K_M** | 2.3 GB | **+1.73 %** | 36.50 | **Pass — minimum ngl=33 → 30.81 tok/s** |


This run shows three things a lookup table cannot know.

- **The assumed quality order was wrong.** On wikitext-2 with 32 chunks, Q6_K's
  perplexity was *better* than the F16 baseline of 8.8688, while Q8_0 was 0.07%
  worse. The measured search order was Q6_K → Q8_0 → Q5_K_M → Q4_K_M, not the
  customary Q8_0-first order. FiTuna sorts by measurements, not assumptions.
- **A narrow miss is invisible without measurement.** Q5_K_M misses the target
  by 0.41 tok/s. The heuristic "this hardware should easily do 30 tok/s with Q5"
  produces a configuration below target.
- **The answer is a configuration, not just a quantization.** The winner is
  Q4_K_M at `-ngl 33`, not full offload at 36. Binary search found the *minimum*
  GPU offload that meets the target, at 30.81 tok/s. The measured quality loss,
  PPL 8.8688 → 9.0225, is 1.73%, within the 5% budget.

Stability observations from the same session:

- In the minimum-ngl binary search, the 4B model's `ngl=0` CPU-only probe did not
  finish its benchmark within the 300-second timeout. FiTuna records it as below
  target and continues rather than stopping search
  (`[Q4_K_M] ngl=0 bench timed out -- treating as 0
  tok/s`). `BenchTimeoutError` handling was added after encountering this on
  actual hardware.
- An immediate `--resume` rerun gives the same answer in **0.88 seconds**.
- The 4 quantized files use 12.1 GB of disk space, and the F16 base file uses
  7.5 GB.
- Measured stage times are **5 minutes 1 second** for the cold quality stage
  (baseline PPL + 4 quantizations + 4 PPL runs) and **12 minutes 53 seconds** for
  speed search. This includes one intentional 5-minute ngl=0 timeout. A fully
  cached `--resume` takes **0.88 seconds**.

<a id="run-3--english-vs-korean-quality-corpus-same-model-same-quants"></a>

## Run 3 - English vs. Korean Quality Corpus (Same Model and Quantizations)

Quality loss is measured as the increase in perplexity on a corpus. Which corpus
should be used? The same Qwen3-4B-Instruct-2507 quantized files were measured with
`--ppl-chunks 32` on the English default, the wikitext-2 test split, and on Korean
Wikipedia (`wikimedia/wikipedia` `20231101.ko`, first 500 articles, CC BY-SA).

| Quant | Quality loss (English wikitext) | Quality loss (Korean kowiki) |
|---|---|---|
| Q6_K | −0.30 % | −0.06 % |
| Q8_0 | +0.07 % | −0.01 % |
| Q5_K_M | +1.53 % | +0.48 % |
| Q4_K_M | +1.73 % | +0.77 % |


Two findings from the measurements:

- **The ranking happened to be the same.** Q6_K is best and Q4_K_M worst on both
  corpora. This does not claim that the order always changes.
- **The magnitude differs by more than a factor of two, and the verdict changes.**
  With `--max-quality-loss 1`, the English corpus rejects Q5_K_M and Q4_K_M at
  the quality gate (early exit A). The remaining quantizations are too slow, so
  search honestly returns **BEST EFFORT (below target)**. With the Korean corpus,
  all four candidates pass and Q4_K_M meets the target at ngl=34 (30.06 tok/s,
  0.77% loss). Even with the same model, computer, target, and budget, **the corpus
  alone reverses feasibility**. For Korean users, check the gate with Korean text:
  fetch it with `fituna fetch-corpus --lang ko --out kowiki-corpus.txt`, then use
  `--quality-corpus kowiki-corpus.txt`.

**A caveat was added after Run 5.** These are `--ppl-chunks 32` estimates.
Remeasuring the error bars FiTuna had discarded at that chunk count in Run 5 gave
about ±3 percentage points of loss (see "How Big Must a Perplexity Gap Be?" below).
Those measurements used Run 5's model and files. Run 3's Qwen3-4B quantized files
are no longer available, so ±3 percentage points here is an estimate, not a
measurement. The reversed verdict above is actual tool behavior: running the two
commands gives different verdicts, which is the experiment's main point. However,
it **did not establish** that the underlying quality difference exceeds the
estimator's resolution. English 1.73% versus Korean 0.77% is an unpaired
cross-corpus gap of 0.96 percentage points, of the same scale as that error bar;
the two values straddle the 1% gate. Read this as "at this budget, the corpus used
for the gate determines the verdict", not "Korean text was proved to degrade less".
Run 5 measured the same comparison with a Korean model but did not establish the
latter claim.

Designing this experiment also uncovered a real cache bug. The quality-result
key was (model, quant, chunks), without the corpus, so the second corpus silently
reused the first one's figures. The cache key now includes a corpus fingerprint,
and a regression test checks this behavior.

<a id="run-4--nvidia-tesla-t4-linux-google-colab"></a>

## Run 4 - NVIDIA Tesla T4, Linux (Google Colab)

The same model, command, and target as Run 1 were run on different hardware.
This was reproduced with
[notebooks/colab_nvidia_verification.ipynb](../notebooks/colab_nvidia_verification.ipynb)
(free T4 tier, llama.cpp built from source with CUDA). Through its `nvidia-smi`
parsing path, `fituna
detect-hw` correctly detected `nvidia / Tesla T4 / 15360 MB VRAM /
linux` automatically.

| Candidate | Measured quality loss (T4/CUDA) | macOS/Metal | Measured gen tok/s (T4) | macOS | Target 240 verdict |
|---|---|---|---|---|---|
| Q8_0 | — | — | 202.70 | 205.91 | Below target |
| Q6_K | 0.83 % | 0.53 % | **205.50** | 249.50 | Below target → **Best effort** |
| Q5_K_M | — | — | 172.03 | 233.26 | Below target |
| Q4_K_M | **5.22 % → 5% gate rejected (Early exit A)** | 4.74 % → Pass | Not benchmarked | 244.34 | — |


Cold search took **61 seconds**; a `--resume` rerun reached the same output in
**1.45 seconds**. The result is BEST EFFORT (Q6_K, 205.50 tok/s, 0.83% loss).
It correctly reports that the T4 cannot meet the 240 tok/s target that M3 Pro met.

Three cross-platform facts a lookup table cannot know:

- **The quality-gate verdict reversed between platforms.** Q4_K_M passed the 5%
  budget on Metal with 4.74% loss, but was rejected at the CUDA gate with 5.22%.
  Even with the same file and corpus, backend figures differ and change the set
  of feasible candidates. The Run 3 caveat also applies: the 0.48-percentage-point
  gap between 32-chunk estimates is of the same scale as the ±3-percentage-point
  error bar measured on Run 5's own files. This establishes only that the two
  backends produced different verdicts here, not that the 0.48-percentage-point
  difference is reproducible. Unlike Run 3's cross-corpus comparison, however,
  the same corpus and file are used and only the backend differs, making this a
  partially paired comparison.
- **Speed rankings differ by platform.** On T4, Q6_K is faster than Q8_0 and
  Q5_K_M, the slowest of the three. M3 Pro has a different order.
- **Feasibility is also hardware-relative.** The same command passes on one
  computer and gives only a best-effort result on another. This is the answer a
  user needs before choosing hardware or lowering the target.

<a id="run-to-run-variance-measured-not-hidden"></a>

### Run-to-Run Variability (Measured, Not Hidden)

Laptop benchmark figures are sensitive to heat. In a second session, the cache
was completely empty with a new `--out`, but the computer was hot after an hour
of continuous benchmarking. Q6_K, Q8_0, and Q5_K_M reproduced within ±0.5 tok/s.
Q4_K_M at full offload, however, gave 22.73 tok/s rather than the first session's
36.50 tok/s. Three immediate direct repeats of `llama-bench` with the same
configuration gave:


```
37.53 tok/s ± 0.20      31.97 tok/s ± 6.74      35.35 tok/s ± 3.26
```


Sustained performance is about 36 tok/s, and 22.73 is a thermal-throttling outlier.
The internal standard deviation growing to ±6.7 under load also shows this.
Two practical implications:

- FiTuna reports measurements under *that session's thermal conditions*. This
  is the performance obtained when running the resulting command immediately.
- If the target is within a few tok/s of a candidate's sustained speed, treat
  the verdict as borderline, as with 30 versus Q5_K_M's 29.6-29.7 here, and run
  search again with the computer in its usual state. Showing llama-bench's
  per-run standard deviation in the report and automatically warning about
  borderline verdicts is on the roadmap.

## Run 5 - Korean Open-Weight Model Midm-2.0-Mini-Instruct (MIT)

Runs 1-4 used models trained primarily in English. Run 5 repeated Run 3's English
versus Korean quality-corpus comparison on a **Korean** open-weight model, with
the same model and quantized files, to see whether corpus sensitivity persists
when the model itself is Korean.

The result: loss *magnitudes* differ by corpus, but the final verdict is the same.
The intermediate ranking difference initially reported as a conclusion **failed
the stability check using the project's own logs**. Details are in "How Big Must
a Perplexity Gap Be?" below.

The base model, `K-intelligence/Midm-2.0-Mini-Instruct`, has 2,305,517,312
parameters and 48 layers. Its HuggingFace `license` field is `mit`; the repository's
`LICENSE.txt` is the full MIT text with "Copyright (c) 2025 KT Corporation".
The repository is not gated. The already-converted BF16 GGUF,
`Midm-2.0-Mini-Instruct-BF16.gguf` from `mykor/Midm-2.0-Mini-Instruct-gguf`
(4,617,053,184 bytes = 4.30 GB), was used, without `convert_hf_to_gguf.py`, torch,
or transformers.

The redistributor's license was checked separately because it distributed the
actual bytes used. `mykor/Midm-2.0-Mini-Instruct-gguf` declares `license:
mit` and `base_model: K-intelligence/Midm-2.0-Mini-Instruct` in its model-card
front matter. Its own `LICENSE.txt` is also the full MIT text with the same
"Copyright (c) 2025 KT Corporation" notice as upstream. The HuggingFace API
returns `gated: false` and `private: false`. Thus, both the upstream weights
and the GGUF conversion actually used are MIT.

The corpora were fetched with the built-in standard-library downloader, without
`pip install datasets`.


```bash
fituna fetch-corpus --lang ko --out kowiki-corpus.txt   # 500행, 5.9 MB
fituna fetch-corpus --lang en --out wikitext-2-raw-test.txt  # 1,000행, 316 KB
```


### Choosing the Target

The target must have a real chance of failing. To choose honestly, search first
used a deliberately impossible `--target-tps 9999`. This prevented early exit at
the first winner and made every candidate go through the quality gate and a
full-offload benchmark. From cold, this took **4 minutes 59.81 seconds** and
returned exit code 3 (`BEST EFFORT`) with this measured full-offload range:

| Quant | Measured gen tok/s @ ngl=48 |
|---|---|
| Q8_0 | 34.26 |
| Q5_K_M | 38.76 |
| Q6_K | 38.96 |
| Q4_K_M | 44.62 |


**40 tok/s** lies within this range, above three candidates and below exactly one.
Q8_0, Q6_K, and Q5_K_M really fall short, while Q4_K_M really passes, so it is not
an easy target or one that cannot fail. It divides the measured range rather than
lying outside it. All figures below were measured with `--target-tps 40`.

### The Two Runs


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


| Candidate | File size | Quality loss (Korean kowiki) | Quality loss (English wikitext) | Measured gen tok/s (ngl=48, ctx 4096) | Target 40 verdict |
|---|---|---|---|---|---|
| Q8_0 | 2.29 GB | **−0.02 %** | +0.15 % | 34.26 | Below target — Early exit B |
| Q6_K | 1.77 GB | +0.59 % | +0.46 % | 38.96 | Below target — Early exit B |
| Q5_K_M | 1.54 GB | +0.53 % | +1.00 % | 38.76 | Below target — Early exit B |
| **Q4_K_M** | **1.33 GB** | **+2.58 %** | **+3.78 %** | **44.62** | **Pass (ngl=48)** |


BF16 baseline perplexity is **9.9511** for Korean and **8.7900** for English.
Every quality figure in the table is a `--ppl-chunks 32` estimate with an error
bar printed by `llama-perplexity` but not stored by FiTuna. Rather than quoting
one and guessing the rest, every Korean row was remeasured at this chunk count:
BF16 baseline **±0.29318**, Q8_0 **±0.29307**, Q6_K **±0.29529**, Q5_K_M
**±0.29372**, Q4_K_M **±0.30055**. Converted to loss, all rows are about ±3
percentage points. Before drawing conclusions from neighboring rows, read
"How Big Must a Perplexity Gap Be?" below. Both runs end with the same result:


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

### How Big Must a Perplexity Gap Be? (The Discarded Error Bars)

All quality figures above are 32-chunk estimates. `llama-perplexity` prints an
error bar beside each value, but FiTuna did not store it. The output is
`Final estimate: PPL
= 10.0099 +/- 0.29529`; the regular expression in `quality.py` captures only PPL
and discards `+/-`. Rereading this run did not establish whether the ranking gap
below was signal, so it was measured directly. The binaries were run directly on
the same four files and two corpora, increasing measurement to **128 chunks**:
4 times the original run, and the largest power of 2 the English corpus supports.
At `n_ctx=512`, wikitext-2 test contains 143 chunks and kowiki 2,368. This is an
extension, not a repeat. As the second point below explains, the 128-chunk run
*contains* the 32-chunk run; the two columns are not independent repeated
measurements.


```bash
llama-perplexity -m out/...-3e6680866409-Q6_K.gguf -f kowiki-corpus.txt --chunks 128
```


Calling with `--chunks 32` exactly reproduces every Korean value in the cache
(`9.9511`, `9.9488`, `10.0099`, `10.0037`, `10.2082`). Only the error bars have
been preserved from the same measurement as the tool. Calculation is deterministic;
as the second point below explains, it is the same run. Matching values checks
the wiring, not repeatability.

| Korean (kowiki) | PPL @128 chunks | Loss @32 | Loss @128 |
|---|---|---|---|
| BF16 baseline | 10.6238 ± 0.16350 | — | — |
| Q8_0 | 10.6187 ± 0.16335 | −0.02 % | −0.05 % |
| **Q5_K_M** | **10.7413 ± 0.16504** | **+0.53 %** | **+1.11 %** |
| **Q6_K** | **10.7470 ± 0.16569** | **+0.59 %** | **+1.16 %** |
| Q4_K_M | 11.0577 ± 0.17044 | +2.58 % | +4.08 % |


| English (wikitext-2) | PPL @128 chunks | Loss @32 | Loss @128 |
|---|---|---|---|
| BF16 baseline | 9.9155 ± 0.14863 | — | — |
| Q8_0 | 9.9217 ± 0.14873 | +0.15 % | +0.06 % |
| **Q6_K** | **9.9778 ± 0.14975** | **+0.46 %** | **+0.63 %** |
| **Q5_K_M** | **10.0197 ± 0.15035** | **+1.00 %** | **+1.05 %** |
| Q4_K_M | 10.2945 ± 0.15471 | +3.78 % | +3.82 % |


Five findings, ordered by how much they change the conclusion:

- **The English ranking is stable throughout the run; the Korean ranking fluctuates.**
  `llama-perplexity` prints cumulative perplexity at every chunk, giving 128 nested
  estimates per run. Reading these checks stability at no additional cost.
  For English, the margin between the two quantizations becomes positive at n=4
  and stays positive at all remaining 125 points. At **every** readable chunk
  count, Q6_K ranks above Q5_K_M; the sign never changes. Korean is opposite at
  the same n=4, with Q5_K_M ahead by 0.0867 percentage points. After n=16, the sign
  changes **nine times**. Counting from n=4, the two orders alternate over eleven
  intervals, with ten changes in total.

  | Korean, n = | 4 | 5–16 | 17–22 | 23–24 | 25–66 | 67 | 68–76 | 77–99 | 100–106 | 107–108 | 109–128 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Leading quant | Q5_K_M | Q6_K | Q5_K_M | Q6_K | Q5_K_M | Q6_K | Q5_K_M | **Q6_K** | Q5_K_M | Q6_K | **Q5_K_M** |


  This is not a tiny gap quietly settling down. At n=77-99, the *opposite* of the published order holds
  for 23 consecutive points. The Korean mean absolute margin there is
  **0.0545 percentage points**, slightly *larger* than **0.0537 percentage points**
  at n=128. The opposite order persists with equal or greater strength. At n=96,
  both corpora put Q6_K first, with no ranking reversal at all. The Korean column's
  order depends on where the run stops.
- **The 32-chunk and 128-chunk values are one measurement, not two.**
  `llama-perplexity` always starts at the beginning of the corpus, so a 128-chunk
  run *contains* its own 32-chunk run. Chunk `[32]` of the Korean Q8_0 trace,
  `9.9488`, exactly matches what FiTuna stored with `--ppl-chunks 32`. The same
  holds for Q6_K `10.0099`, Q5_K_M `10.0037`, the Korean baseline `9.9511`, and
  the English baseline `8.7900`. Saying "reproduced again at 32 and 128 chunks"
  cites one nested measurement twice. The previous document wrongly presented
  these as two independent confirmations. Stability evidence comes from the
  chunk-by-chunk traces above, not this agreement.
- **Neither margin exceeds the error bar, and their strengths are very different.**
  At 128 chunks, each perplexity error is ±0.15-0.17, or ±1.5 percentage points
  of loss. The English margin of 0.423 percentage points is about 1/4 of
  that; the Korean n=128 margin of 0.054 percentage points is about 1/29. The English
  ranking gap is about 8 times stronger than the Korean
  one and, unlike Korean, has the same sign at every point.
- **The available `+/-` is not exactly the error needed.** It is corpus-sampling
  error, describing how much absolute perplexity moves on another Korean sample.
  Comparing two quantizations on the *same* chunks is a paired comparison with
  much smaller error. This explains why the English margin keeps its sign for
  125 consecutive chunk counts despite being 1/4 of the error bar.
  `llama-perplexity` does not report paired error, so no confidence level can be
  assigned to either margin. Instead, chunk traces check stability directly
  without an error model, and the Korean margin fails this check.
- **Absolute loss depends strongly on chunk count; always report the count.**
  Korean Q4_K_M loss changes from 2.58% at 32 chunks to 4.08% at 128 chunks;
  Q6_K changes from 0.59% to 1.16%. The 32-chunk values above are exact results
  actually produced by the documented commands and cache, but are 32×512-token
  estimates, not fixed properties of the quantization. Only Q4_K_M's degradation
  among the four candidates exceeds its own error bar (128-chunk Δppl 0.434,
  error ±0.17). **The verdict therefore depends on `--ppl-chunks` as well as the
  result figures.** With this run's `--max-quality-loss 5`, Korean Q4_K_M has
  2.42 percentage points of headroom at 32 chunks and 0.92 at 128 chunks. With
  the same file, corpus, and gate, quadrupling the evidence reduces headroom to
  1/4. A PASS with a small chunk count is a PASS only at that count.
  When a candidate is close to budget, increase chunks and remeasure before
  trusting the verdict.

### What a Lookup Table Cannot Know

- **This did not establish that the corpus changed the ranking.** The two runs,
  sorted by measured loss, really report different intermediate orders:
  Korean Q8_0 → **Q5_K_M → Q6_K** → Q4_K_M, English Q8_0 → **Q6_K → Q5_K_M** →
  Q4_K_M. But claiming a reversal requires both gaps to be real, and only English
  survives measurement. English puts Q6_K ahead of Q5_K_M without a sign change
  at every chunk count from n=4-128. Korean changes the sign between the same
  quantizations nine times; at n=77-99 it maintains the *English* order, opposite
  to n=128, with equal or greater strength. At n=96, both corpora have the same
  ranking, with no reversal. The evidence therefore supports only that the
  Korean column is too unstable to determine an order, not that the corpus
  changes the ranking. Run 3 made the same comparison with an English-trained
  model and stated that the ranking happened to match, without claiming it
  always changes. Run 5 cannot claim a reversal either. The previous document
  said "the corpus reversed measured quality rankings, reproducibly", but the
  chunk traces and the project's own logs refute both claims. A plausible
  assumption failed against the project's own measurements; the tool worked
  as intended.
- **Q8_0's sign changed, and this observation is stable throughout the traces.**
  For Korean, its perplexity was very slightly *better* than the BF16 quantization
  baseline: −0.02% at 32 chunks and −0.05% at 128 chunks. For English it was
  +0.15% and +0.06%, respectively. Unlike the Q6_K/Q5_K_M order above, this holds
  chunk by chunk as well. Of the 125 points from n=4-128, Korean Q8_0 is below
  baseline at 117, with just one sign change early on. English Q8_0 is above
  baseline at all 125. A table recording only "Q8_0 ≈ lossless" hides which side
  of 0 it lies on. At the same 32 chunks, Δppl is 0.0023 for Korean and 0.0128
  for English; measured error is ±0.26-0.29. The missing English 32-chunk error
  was measured directly, rather than borrowing ±0.16 from a differently sized
  128-chunk run. This is not called a settled conclusion, but it is a more
  consistent observation than the ranking gap.
- **Run 3's pattern of "less Korean loss" does not generalize.** Three of Run 3's
  four rows have a smaller Korean value than English, but Q6_K is already an
  exception (Korean −0.06%, English −0.30%). In the 32-chunk table above, Q8_0
  (−0.02 versus +0.15), Q5_K_M (0.53 versus 1.00), and Q4_K_M (2.58 versus 3.78)
  follow that pattern, but Q6_K is **opposite** (Korean 0.59, English 0.46).
  The 0.14-percentage-point gap lies within ±1.5-percentage-point error.
  Chunk by chunk over n=4-128, the Korean value is lower at Q8_0 125/125,
  Q4_K_M 73/125, Q5_K_M 50/125, and Q6_K 4/125 points. It was never a clean rule;
  only Q8_0 follows it consistently. The direction of the corpus effect must be
  measured for each quantization.
- **The verdict did not reverse this time, and that is reported plainly.** Unlike
  Run 3, where the corpus determined feasibility at a 1% budget,
  `--max-quality-loss 5` lets all four candidates pass both corpus gates and
  selects the same winning configuration. The corpus changes the figures, not
  the final answer, and does not establish a ranking reversal as discussed above.
  These are the measured results, not dressed up as a reversed verdict.
- **The "minimum ngl" answer was all 48 layers, with just one layer of margin.**
  `-ngl` binary search returned the full 48 because smaller values failed to
  meet the target. The measured Q4_K_M offload curve follows.

  > **Correction (0.3.2).** During this run, FiTuna treated `-ngl 48` (the block
  > count) as full offload. llama.cpp counts the output layer as one extra layer,
  > so actual full offload is `-ngl 49`; at `-ngl 48`, one layer stays on the CPU.
  > For SmolLM2-135M (30 blocks), `-ngl 30` reported "offloaded 30/31 layers" at
  > 233-240 tok/s, while `-ngl 31` reached 258-262 tok/s. The curve below was
  > measured only up to 48. Since 0.3.2, the upper bound is `n_layers + 1`.

  | ngl | 0 | 24 | 36 | 42 | 45 | 47 | 48 |
  |---|---|---|---|---|---|---|---|
  | gen tok/s | timeout | 10.86 | 17.56 | 23.53 ⚠ | 32.43 | **39.32** | **44.62** |
  | llama-bench standard deviation | — | ±0.05 | ±0.37 | **±4.94** ⚠ | ±0.26 | ±0.15 | ±1.25 |


  Moving **one** of the 48 layers off the GPU costs 5.30 ± 1.26 tok/s, or 4.0-6.6, equivalent to throughput loss
  of 11.9% ± 2.8 percentage points. At 39.32, the result misses target 40 by 0.68.
  Relative to ngl=47's ±0.15, this gap is 4.5σ and clear. The curve is not linear
  in layer count either: half the layers, ngl=24, gives 10.86 tok/s, 24% of
  full-offload speed rather than 50%. A fixed rule of thumb cannot reproduce
  this shape.

  ⚠ **The ngl=42 point is thermally affected and is not published as a normal value.**
  The five llama-bench samples are `[25.81, 25.69, 25.59, 25.84, 14.69]`: four
  tightly clustered at 25.73 ±0.12 and one collapsed value. That single sample
  lowers the mean by 8.6% to 23.53 and increases standard deviation to ±4.94,
  about 4 times the curve's next-worst value. This is the same symptom as Run 4's
  "Run-to-Run Variability". Running the same command directly three more times
  (`llama-bench -m
  ...-Q4_K_M.gguf -ngl 42 -d 3456 -p 512 -n 128`) did not remove the instability.
  However, the computer was already hot after an hour of continuous perplexity
  work. This establishes only that the configuration is at a thermal boundary;
  its value on a cold computer is unknown.


  ```
  19.84 ± 3.96      18.64 ± 3.57      19.32 ± 4.44
  ```


  All three runs collapsed the same way within their five samples, for example:
  `[21.97, 22.40, 23.13, 15.55, 13.53]`.

  Prompt throughput at the same configuration also fell from the first session's
  100.55 tok/s to about 80 in all three repeats, indicating about 20% throttling
  across the computer. This is why the figure is flagged rather than replaced
  with the new values. Treat 23.53 as a lower bound with a poor error bar. Even
  the mean of the four normal samples is 25.73, so the binary-search verdict that
  ngl=42 misses target 40 is unchanged.
- **Even a speed reversal needs a caveat.** The 1.77 GB Q6_K measured *faster*
  than the smaller 1.54 GB Q5_K_M (38.96 versus 38.76), again contradicting the
  smaller-is-faster heuristic. But llama-bench per-run standard deviations are
  ±0.53 and ±0.11 tok/s, respectively. Unlike Run 1's clear reversal, the 0.20
  tok/s gap is **within run-to-run noise**. This is reported as "indistinguishable",
  not a Q6_K victory. The practical conclusion remains: using 15% more disk for
  Q6_K yielded neither measurable extra speed nor better Korean quality.

### Measured Time and Disk Usage

| | |
|---|---|
| Target-selection probe (cold: baseline PPL + quantize 4 times + PPL 4 times + bench 4 times) | **4 min 59.81 s** |
| Target 40 Korean run (cold `-ngl` binary search, 6 new benchmarks — curve point 7 at ngl=48 is a probe cache hit) | **12 min 54.93 s** |
| ↳ Intentional `ngl=0` benchmark timeout | 5 min 00 s |
| Target 40 English run (cold quality stage, all benchmarks are cache hits) | **1 min 53.19 s** |
| Korean `--resume` rerun | **1.47 s** |
| English `--resume` rerun | **1.42 s** |


Both `--resume` reruns reproduce the complete report byte for byte from
`out/.fituna_cache.sqlite3`. The artifacts were preserved this time. Comparing
each rerun's report block with the corresponding original cold-run log
(`run5-ko.log`, `run5-en.log`) using `diff -u` showed identical Korean 434-byte
and English 433-byte blocks; both exited with code 0. The `--resume` times above
are from this validation on a computer hot after an hour of continuous
benchmarking. The earlier session recorded 0.955 seconds and 0.908 seconds, but
its artifacts were not saved to disk, so only provable figures are quoted.
The four quantized files use **6.9 GB**, and the BF16 base file uses 4.30 GB,
for an 11 GB workspace in total.

The 2.3B model's `ngl=0` CPU-only probe failed to finish its benchmark within the
300-second timeout. FiTuna logged
`[Q4_K_M] ngl=0 bench timed out -- treating as 0 tok/s
(below target)` and continued search. This reproduces the same
`BenchTimeoutError` path seen with Run 2's 4B model on a second model type.
