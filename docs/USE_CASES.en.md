[Korean original](./USE_CASES.md)

# FiTuna Use Cases

Scenarios with measured figures use actual measurements (`docs/RESULTS.md`). The
others explain how the tool works. No figures are invented: avoiding guesses is
why this tool exists.

---

## Scenario 0 - First Day After Installation, Just `quickstart` (Measured Validation: 2026-08-02)

**Situation.** You have just installed FiTuna. You have neither a model file nor a
corpus, and do not know which flags to use.

**Run (the full process was measured on M3 Pro).** One command:
`fituna quickstart`. The wizard walks through these steps: select the "Coding
Assistant" preset (30 tok/s / 3% loss / ctx 8192) → choose "Personal/Research"
license requirements → select SmolLM2-135M from the verified model list →
**automatically download the 270.9 MB GGUF with progress reporting** → fetch
1,000 lines of English corpus text automatically (printing the CC BY-SA notice) →
show the complete assembled `fituna run ...` command, then run it after confirmation.

**What actually happened during the search.** The 3% quality gate rejected
Q5_K_M (3.32%), Q4_K_M (4.74%), Q3_K_M (19.54%), and Q2_K (38.34%) without
benchmarking them (early exit A). Binary search for the minimum offload of the
surviving Q8_0 returned **ngl=0**: **no GPU layers are needed at all** for this
model and target. CPU-only execution reached 35.45 tok/s (ctx 8192), meeting the
target of 30. It exited with code 0 and generated a Modelfile. "Minimum resources
that meet the target" can literally mean 0 GPU layers, an answer that intuition
based on a specification sheet cannot provide.

---

## Scenario 1 - Running a Local Coding Assistant on a Laptop (Based on Measurements)

**Situation.** An AI student wants to use a local LLM as a coding assistant on an
M3 Pro MacBook (18GB unified memory), without API costs. Responses should reach
30 tok/s to feel fluid, and code quality should not noticeably decline.

**The usual approach.** Q4? Q5? Q8? Community posts on HuggingFace disagree.
Download one and try it; if it is slow, download another, change `-ngl`, and
benchmark manually. Half a day is gone.

**FiTuna.**

```bash
fituna run --model Qwen3-4B-Instruct-2507-F16.gguf \
  --target-tps 30 --max-quality-loss 5 --ctx 4096 \
  --quant Q8_0,Q6_K,Q5_K_M,Q4_K_M --ppl-chunks 32 \
  --wikitext wikitext-2-raw-test.txt --out ./out --resume
```

**Measured result (M3 Pro).** Q8_0, supposedly the obvious best choice, was rejected
at 24.22 tok/s; Q6_K was rejected at 28.48. Q5_K_M reached 29.59 and missed by
**0.41**. Choosing by feel, you might have stopped there and used a configuration
below the target. The final answer was **Q4_K_M with only 33 layers offloaded to
the GPU, 30.81 tok/s, and 1.73% quality loss**. Copy the generated `llama-cli`
command and use it directly.

---

## Scenario 2 - "Can This Computer Actually Do It?" A Reality Check (exit code 3)

**Situation.** Someone with an older desktop or mini PC wants to run a local LLM
but does not even know whether the hardware can meet the target speed.

**FiTuna.** Run it with a target. If every candidate falls short, it does not fail
silently: it reports **exit code 3 plus the closest best-effort configuration**.

**Measured example (SmolLM2-135M, target 300 tok/s).** Every candidate fell short.
FiTuna reported "The best is Q6_K at 249.50 tok/s" in 33.6 seconds. The user can
choose between lowering the target, using a smaller model, or changing the
hardware: **the answer is known before trying configurations at random.**

**Revalidation (2026-08-02, target 5000 tok/s + `--export-ollama`).** Even with
exit 3, the full best-effort report was printed, including artifacts and the
llama-server command. **The Modelfile was also updated to match the best-effort
configuration** (num_gpu 29→30). This confirmed that falling short of the target
still produces useful output, consistently down to the artifacts.

---

## Scenario 3 - On-Premises Serving for a Lab or Small Team

**Situation.** A lab wants to run an internal LLM on its shared workstation. The
requirements are an acceptable response speed for concurrent users (for example,
20 tok/s) and quality loss within 3%. There is only one machine, so occupying it
with lengthy benchmarks is a burden.

**Measured validation from artifacts to serving (2026-08-02).** Run
`fituna run --json
--export-ollama` non-interactively → exit 0 → extract `llama_server_command`
from the JSON and run it unchanged → `/health` OK → send an actual request to
the OpenAI-compatible `/v1/chat/completions` endpoint and receive a response
(SmolLM2 Q8_0 ngl=29, 242.13 tok/s configuration). This verified the complete path
from a configuration measured by FiTuna to an internal API server. In the same
session, rerunning with `--resume` took **0.70 seconds**.

**FiTuna.**

- Pay the search cost once: results are stored in `.fituna_cache.sqlite3`.
  With the same model, hardware, and llama.cpp build, a `--resume` rerun reproduces
  the same answer in **less than 1 second** (measured at 0.88 seconds). Sharing the
  cache file lets team members on identical hardware skip the search itself.
- The cache key includes the llama.cpp build version, so upgrading the backend
  does not silently reuse old measurements: they are measured again automatically.
- When requirements change (20 → 25 tok/s), already-measured candidates are reused
  and only the missing measurements are run.

---

## Scenario 4 - Building a Hardware Profile with an Overnight Batch

**Situation.** The machine is needed during the day, so benchmarks cannot run then.

**FiTuna.** Start a broad search before leaving work (6 candidates in `--quant`,
`--ctx 8192,4096`). By the next day, the cache contains a table of measured
model × quantization × ngl × ctx combinations. After that, any target can be
answered from the cache in seconds. Asking for a target of 40 and then 30 does not
require another benchmark (measured: reruns with changed targets finished within
a few seconds). Even if power is lost, `--resume` continues from where the run
stopped. Quantized files are written atomically, so no corrupt file is left behind.

---

## Scenario 5 - A Measured Backend for AI Coding Agents (MCP Server, Implemented)

**Situation.** A user asks an AI agent, "Recommend a local model I can run on my
computer." Today's chatbots answer with **guesses** based on hardware specification
text. As Scenario 1 shows, even a 0.41 tok/s difference can make that guess wrong.

**FiTuna.** An agent that calls FiTuna's MCP server gets a configuration measured
on the spot instead of a guess. Its answer changes from "Q5 will probably work"
to "Just measured: Q4_K_M ngl=33, 30.81 tok/s."

The server is already working, not a roadmap item. It includes
`fituna/mcp_server.py` (a JSON-RPC 2.0 stdio server implemented using only the
standard library), the `fituna-mcp` console script, and CI self-checks
(`python -m fituna.mcp_server --selfcheck`). It is listed under Added in
`CHANGELOG.md`.

---

## Shared Assumptions and Limitations (Explicit Disclosure)

- FiTuna does not perform inference or quantization itself. llama.cpp binaries
  are required (`brew install llama.cpp` or a source build).
- The quality metric is a single axis: perplexity. It is a general-purpose proxy,
  not a guarantee of domain-specific quality. The measured text is not fixed,
  though: `--quality-corpus` accepts any UTF-8 text file, and Runs 3 and 5 in
  `docs/RESULTS.md` used a Korean corpus. The recommended approach is to measure
  text resembling the actual workload.
- Search creates all candidate quantized files (about 12GB of disk space for
  4 candidates of a 4B model). These files are reused on subsequent runs.
- Benchmark figures are sensitive to thermal conditions. If a result only narrowly
  meets or misses the target by a few tok/s, rerun with the machine in its normal
  state. See the measured variability in `docs/RESULTS.md`.
