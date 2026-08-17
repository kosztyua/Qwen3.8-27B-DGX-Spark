# Qwen3.8 27B for DGX Spark

[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Ready-to-run scripts to serve **Qwen3.8-27B** on an NVIDIA DGX Spark (GB10, aarch64) behind an OpenAI-compatible API. The default configuration is the fastest one measured on this hardware: **vLLM serving the [RadixArk NVFP4 checkpoint](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) with DSpark speculative decoding**, at 38-43 tok/s single-stream on real reasoning/code content and 122.8 tok/s aggregate at 8 concurrent streams. Alternative variants (the unsloth checkpoint with its MTP head, or the SGLang engine) are one environment variable away, and all performance and quality claims in this README were measured on one DGX Spark on 2026-08-15/16; see [Benchmarks](#benchmarks).

> Forked from [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000) (the original repo, which also covers the RTX 6000 PRO). This fork is DGX-Spark-only and adds the measured tuning, the DSpark configurations, and the SGLang alternative.

## Requirements

| Component | Detail |
|---|---|
| Hardware | NVIDIA DGX Spark / GB10 (aarch64, sm_121a, 128 GB unified memory) |
| Docker | With NVIDIA Container Toolkit / GPU passthrough working (`docker run --gpus all`) |
| Disk | ~24 GB for the default checkpoints (+23 GB with `--all`), plus the engine images |
| CLI tools | `docker`, `curl`, `python3`; the Hugging Face CLI (`hf`) for `download.sh` |
| Hugging Face token | `HF_TOKEN` in `~/.bashrc` (higher rate limits; not strictly required) |

Engine images are pinned by digest in the start scripts (the 2026-08-15 builds validated here): `vllm/vllm-openai@sha256:b5c860…` and `lmsysorg/sglang@sha256:febfb9…`. See [Known issues](#known-issues-and-safeguards) for why the vLLM pin matters.

## Quick start

```bash
# 1. Download the default checkpoints into ./.cache/huggingface (retries, resumes)
./download.sh              # add --mtp or --all for the unsloth checkpoint

# 2. Start the server (waits until the API is ready, then exits)
./start.sh

# 3. Use it
curl http://127.0.0.1:8888/v1/models

# 4. Benchmark it (optional, ~10 min)
./bench.sh mylabel

# 5. Stop it
./stop.sh
```

The start scripts are idempotent: if their container is already running they say so and exit; a stopped leftover container is removed first. Each start script refuses to run while the other engine's container is up (same port, same GPU), and refuses to boot until the previous server's memory is actually released; see [Known issues](#known-issues-and-safeguards).

## Scripts

| Script | What it does |
|---|---|
| `download.sh` | Pre-downloads the RadixArk target + DSpark drafter (`--mtp` for the unsloth checkpoint, `--all` for everything) into `./.cache/huggingface`, with up to 10 attempts per repo. |
| `start.sh` | Launches the pinned vLLM container for the selected `VARIANT` (drafter auto-provisioned for dspark), streams logs to `.vllm.log`, waits for readiness, runs a speculative-decode health probe, and starts the runtime memory guard. |
| `start-sglang.sh` | Launches the pinned SGLang container (RadixArk + DSpark; logs to `.sglang.log`), with the same memory pre-flight and a generation smoke test. |
| `stop.sh` | Stops whichever engine container is running plus the memory guard; leaves the stopped container for `docker logs` post-mortem. |
| `memguard.sh` | Runtime memory guard, started by the start scripts: force-removes the engine container if available memory stays under 4 GiB, because on the GB10 the alternative is the whole machine freezing. Logs to `.memguard.log`. |
| `bench.sh` | Benchmarks whatever is serving on the port (`vllm bench serve` in a throwaway client container, model name auto-detected). Fixed seeds for cross-config comparability; `--full` adds the 128k scenario. Results in `.cache/huggingface/bench-results/<label>/`. |

Runtime artifacts (`.vllm.log`, `.sglang.log`, `.memguard.log`, pid files, `.cache/`) are git-ignored.

## Configuration

### Choosing a variant

| | `./start.sh` (default) | `VARIANT=mtp ./start.sh` | `./start-sglang.sh` |
|---|---|---|---|
| Engine / checkpoint | vLLM, RadixArk NVFP4 | vLLM, unsloth NVFP4 | SGLang, RadixArk NVFP4 |
| Speculative decoding | DSpark drafter, k=7 | built-in MTP head, k=5 | DSpark drafter, k=7 |
| Real-content decode, 1 stream | **38-43 tok/s** | 28-31 tok/s | 38-45 tok/s |
| Aggregate decode, 8 streams | **122.8 tok/s** | 106.3 tok/s | capped at 7 streams |
| Random-content decode, 1 stream | 20.8 tok/s | **27.5 tok/s** | 35.4 tok/s* |
| Prefill (8×16k) | 1,827 tok/s | 1,339 tok/s | **1,844 tok/s** |
| Max context | 262k | 262k / **1M** (`CONTEXT_1M=1`) | 262k |
| `reasoning_effort` control | yes | yes | no (always `xhigh`) |
| Vision input | untested | validated | untested |
| Served model name | `qwen38-27b-radixark-nvfp4` | `qwen38-27b-unsloth-nvfp4` | `qwen38-27b-radixark-nvfp4` |

Pick the default for interactive and batched work; `VARIANT=mtp` when you need the 1M context, validated vision input, or your traffic is genuinely unpredictable text; SGLang mainly for its 128k-depth prefill (~1,125 vs ~590 tok/s single-stream at that depth). There is also `VARIANT=dspark DSPARK_TARGET=unsloth ./start.sh`, which runs the DSpark drafter over the unsloth checkpoint: it works and still beats MTP on real content (39.1/34.6/30.9 tok/s), but is 10-30% behind the RadixArk target because vLLM's compressed-tensors kernels are ~12ms/step slower than the modelopt path and draft acceptance is lower against this target. Quant quality between the two checkpoints measured indistinguishable; see [Quant quality](#quant-quality-radixark-vs-unsloth).

*SGLang's random-content lead comes partly from its looser draft verification: threshold-based acceptance rather than the lossless rejection sampling vLLM uses, trading exactness of the sampling distribution for speed.

### Serving settings (vLLM)

| Setting | Value | Notes |
|---|---|---|
| `IMAGE` | `vllm/vllm-openai@sha256:b5c860…` | Pinned digest = `nightly-aarch64` of 2026-08-15 (`v0.27.2rc1.dev110`) |
| `PORT` | `8888` | Listens on `0.0.0.0` via host networking |
| `--max-model-len` | `262,144` (native) | `CONTEXT_1M=1` serves 1,000,000 via YaRN (mtp variant only) |
| `--gpu-memory-utilization` | `0.84` | Profiling guard only; the pool size comes from `--kv-cache-memory` |
| `--kv-cache-memory` | `73014444032` (68 GiB) | `KV_CACHE_MEMORY=`; 1,274,196 tokens. Pinned because the utilization fraction is not reproducible — see below |
| `--max-num-seqs` | `12` | `MAX_SEQS=`; the measured knee, see [Concurrency and KV sizing](#concurrency-and-kv-sizing) |
| `--kv-cache-dtype` | `fp8` | ~2× KV memory savings vs bf16 |
| `--mamba-ssm-cache-dtype` | unset (`float32`) | `SSM_DTYPE=bfloat16` halves GDN state traffic; see below |
| `--attention-backend` | `triton_attn` | FlashInfer is faster but mis-drafts at high concurrency (see Known issues) |
| `--cudagraph-capture-sizes` | per variant, auto-extended | Must contain `c×(1+k)` for `c = 1..max-num-seqs`; `start.sh` extends the ladder automatically when `MAX_SEQS > 8` |

The `--max-num-seqs` and `--kv-cache-memory` defaults above apply to **`VARIANT=dspark` only**, since that is where they were measured. `VARIANT=mtp` keeps the previously shipped 8 sequences and a utilization-derived pool (it has no separate drafter and a ~2.1M-token pool, so dspark's 68 GiB would shrink it for no measured reason). `MAX_SEQS=8 KV_CACHE_MEMORY= ./start.sh` restores pre-tuning behaviour on either variant.

**Why the KV pool is pinned rather than derived**: `--gpu-memory-utilization` sets a budget as a fraction of total device memory, and vLLM then sizes the KV pool as that budget minus whatever weights, activations and CUDA graphs actually consumed at load time. Those measurements vary slightly run to run, so the resulting pool does too: two starts of the *identical* config measured 1,413,515 and 1,419,112 KV tokens. The drift is small, but it makes cross-restart benchmarking imprecise, so the pool is now pinned in bytes and reproduces exactly (1,274,196 tokens every start).

**Context length**: the model's native context is 262,144 tokens, served by default with no RoPE modification. `CONTEXT_1M=1 VARIANT=mtp ./start.sh` applies the model card's static-YaRN recipe (factor 4.0) for 1M tokens; per the model card, static YaRN can slightly degrade short-context quality, which is why it is opt-in. A single 1M-token sequence needs ~32 GB of KV.

## Using the API

OpenAI-compatible base URL: `http://127.0.0.1:8888/v1`. The model name depends on the variant (see the table above), or ask `GET /v1/models`.

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-27b-radixark-nvfp4",
    "messages": [{"role": "user", "content": "Explain YaRN in two sentences."}]
  }'
```

- **Thinking mode** is on by default; the `qwen3` reasoning parser separates `<think>` blocks into `reasoning_content`. Disable per request with `"chat_template_kwargs": {"enable_thinking": false}`.
- **Reasoning effort** defaults to `xhigh`, which produces very long thinking traces, often the dominant share of end-to-end latency. On vLLM, lower it per request with `"chat_template_kwargs": {"reasoning_effort": "medium"}` (or `"low"`), or server-wide with `--default-chat-template-kwargs '{"reasoning_effort": "medium"}'` in `start.sh`. SGLang currently ignores this kwarg.
- **Sampling defaults** come from the checkpoint's `generation_config.json`: `temperature=1.0, top_p=0.95, top_k=20` (thinking mode). For non-thinking requests the model card recommends `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` with `enable_thinking: false`.
- **Tool calling** is enabled everywhere (`qwen3_coder` parser); pass `tools` / `tool_choice` as in the OpenAI API.

## Benchmarks

All numbers: 1 × DGX Spark GB10, 2026-08-15/16. Random-content rows come from `./bench.sh` (random dataset, fixed seeds, thinking-mode sampling, `--ignore-eos`); speculative acceptance is much lower on random continuations than on real text, so those rows understate real-use decode. Real-workload rows are fixed code/reasoning/math prompts with wall-clock timing, all runs terminated naturally.

### Configuration comparison

Random-content decode (1k-token prompts, 1k-token generations) and prefill:

| Scenario | vLLM DSpark (default) | vLLM MTP | SGLang |
|---|---|---|---|
| Decode, 1 stream | 20.8 tok/s | **27.5 tok/s** | 35.4 tok/s* |
| Decode, 4 streams (aggregate) | **98.6 tok/s** | 65.9 tok/s | 79.3 tok/s |
| Decode, 6 streams (aggregate) | — | 98.6 tok/s | 90.1 tok/s |
| Decode, 8 streams (aggregate) | **122.8 tok/s** | 106.3 tok/s | not possible (scheduler cap 7) |
| Prefill 8×16k, 4 streams | 1,827 tok/s | 1,339 tok/s | **1,844 tok/s** |
| Prefill at 128k depth, 1 stream | — | ~590 tok/s | **~1,125 tok/s** |
| Decode at 128k context, 1 stream | — | 17.9 tok/s | 18.4 tok/s |

Real workloads, single stream (greedy):

| Prompt type | vLLM DSpark (default) | vLLM MTP | SGLang |
|---|---|---|---|
| Code task (thinking) | **42.6 tok/s** | 28.0 tok/s | 38.3 tok/s |
| Reasoning task (thinking) | **38.5 tok/s** | 31.5 tok/s | 37.5 tok/s |
| Math, temp 0.6 (thinking) | 42.6 tok/s | — | **45.2 tok/s** |

*See the verification-mode footnote under [Choosing a variant](#choosing-a-variant). The DSpark speedup is workload-dependent: acceptance ranges from ~2.2 tokens per step on prose to ~4.8 on step-by-step math, so reasoning-heavy traffic benefits most, and unpredictable content is where MTP keeps its single-stream edge.

### How the speed works

Both speculative methods pay the same ~90ms memory-bound target verify pass per step; they differ in drafting cost. MTP drafts sequentially: five recursive passes through one MTP layer, each re-reading the lm_head over the 250k vocab, ~51ms of every 142ms step. DSpark drafts a 7-token block in a single ~15ms forward of a dedicated 1.4B drafter, and its trained drafts also survive verification better on reasoning and code text.

vLLM's native DSpark support needs three things, the latter two found by a [community PSA](https://www.reddit.com/r/LocalLLM/comments/1vpo15s/psa_qwen3827b_dspark_works_in_vllm/): the drafter's `architectures` field patched from `DSparkDraftModel` (which vLLM routes to a DeepSeek-V4 class) to `Qwen3DSparkModel`, which `start.sh` maintains as a patched local copy automatically; `"draft_sample_method": "probabilistic"`, worth ~23% over the default greedy drafting; and `num_speculative_tokens: 7`, since smaller blocks bench worse and `enable_adaptive_verification` is rejected by the GDN attention backend. Verification in vLLM is lossless rejection sampling, so DSpark outputs follow the target model's distribution exactly.

### Quant quality: RadixArk vs unsloth

The two checkpoints are structurally similar mixed-precision quants, not different classes: both keep all attention and linear-attention projections at FP8 (e4m3) and quantize the MLPs to NVFP4 with calibrated scales, and both leave the vision tower unquantized. unsloth is more conservative in two places: its last 8 MLP layers stay at FP8, and its FP8 KV cache ships calibrated scales, while RadixArk's KV runs with 1.0 defaults (disclosed in its own qualification file). RadixArk ships a documented qualification: 97.27% on the full 1,319-example GSM8K, gated at a 96.5% minimum, converted directly from `Qwen/Qwen3.8-27B`.

Measured head-to-head on this box (same vLLM engine, greedy):

| | unsloth | RadixArk |
|---|---|---|
| Perplexity, 29,598 tokens of novel prose | 3.1952 | 3.1826 |
| GSM8K, 120-item subset (thinking, medium effort) | 96.7-97.5%* | 97.5% |

*Run-to-run spread from batching nondeterminism is ±1 item, so these are statistically indistinguishable: no measurable quality gap on these signals. Scope honestly: two signals (English prose likelihood, grade-school math); coding, long context (where the uncalibrated KV scales could in principle matter), multilingual, and vision quality were not compared.

### Capacity and KV cache

| | vLLM DSpark (default) | vLLM MTP | SGLang |
|---|---|---|---|
| KV cache dtype | FP8 e4m3 | FP8 (calibrated scales) | FP8 e4m3 |
| KV pool | 1,274,196 tokens (pinned 68 GiB) | ~2.1M tokens | 415k tokens |
| Concurrency cap | 12 (`MAX_SEQS`, measured knee) | 8 (same) | 7 (mamba state cache) |
| Max concurrent 128k requests | ~9 (memory), 12 (config) | ~16 (memory), 8 (config) | ~3 (KV-bound) |

Checkpoint sizes: RadixArk ~21.9 GB + 2.7 GB drafter; unsloth 22.6 GB including its MTP head. Live vLLM metrics: `curl -s localhost:8888/metrics | grep kv_cache_usage_perc` (KV usage) and `… | grep spec_decode` (acceptance). NVFP4 KV cache is not possible on this machine: vLLM accepts the flag, but the only implementing kernels (FlashInfer's TRT-LLM path) are gated to SM100-family GPUs, and GB10 is SM121, so FP8 is the floor here.

### Concurrency and KV sizing

Measured 2026-08-17 with the `sess*` scenarios (see Methodology), which model the
production shape: ~32k context, one 28k prefix per concurrent slot, high intra-session
prefix reuse. Server held at `MAX_SEQS=20` with the pool pinned at 68 GiB so that **only
client concurrency varies**; the ladder covered `c×8` up to 160 throughout. Labels
`tune-A/B/D/E`, reproducible with `./bench-table.sh tune-`.

**The measured result: +9.9% at 12 concurrent versus 8.** Six runs, three per arm,
identical prompt set on both arms (12 prefixes, seed 11908, 3 turns), full server restart
before every arm so neither inherits the other's prefix cache. Labels `ab-r{1,2,3}-c{8,12}`.

| Concurrency | out tok/s | acceptance length | seq-steps/s | wall clock |
|---|---|---|---|---|
| 8 | 40.68 ± 0.65 (3.2%) | 2.53 | 16.09 | 906 s |
| **12** | **44.73 ± 0.79 (3.5%)** | 2.49 | 17.98 | 824 s |

Within-arm spread is ~3% and the two arms do not overlap (2.63 tok/s between the worst
c=12 run and the best c=8 run), so the difference is real. Acceptance length came out
matched to within 1.7%, which is what should happen with identical prompts and is what
makes the throughput comparison trustworthy. Zero failed requests, zero preemptions.

These runs used `SESS_REQS=3`, so roughly 28% of wall clock is prefill. Production
sessions run ~28 turns, where prefill is a much smaller share and concurrency helps the
decode phase more — so **+9.9% is likely a floor for real traffic, not a ceiling**.

**Two earlier figures in this section were wrong and are withdrawn.** A claimed +33% came
from hand-picked steady-state windows; a claimed +27% came from whole-run data where each
concurrency level benchmarked a *different prompt set*, because `sess<c>` ties
`num_prefixes` to the concurrency. Use `SESS_PREFIXES` and `SESS_SEED` to hold content
fixed — and run one arm per invocation, since two arms in one invocation share the prefix
cache (measured: 90.5% of the second arm's input tokens were cache hits).

`c=16` was not measured under the controlled method. The uncontrolled runs suggested a
plateau rather than a regression, but that evidence has the same defect as the withdrawn
+27%, so treat 16 as unmeasured.

**Raising `MAX_SEQS` alone does nothing.** The engine has to be *offered* the extra work:
with 8 client connections, `num_requests_waiting` sat at 0 across 3,915 scheduler samples
of production traffic. This is a paired change — server cap and client worker count move
together.

### GDN state precision (`SSM_DTYPE`)

48 of the 64 layers are Gated DeltaNet, and their recurrent state is 48 v-heads × 128 ×
128 × 4 B = 3.15 MB per layer per sequence at `float32`. Under speculative decoding vLLM
keeps a state per draft position for rollback, so each step is 1 read + `k+1` writes — at
k=7 and c=12 roughly 16 GB of a ~55 GB step, the largest non-weight term.
`SSM_DTYPE=bfloat16` halves it.

Measured at c=12, `tune-B/sess12` (fp32) vs `tune-E-s20-kv68-bf16ssm/sess12` (bf16), same
seed and prompt set:

| | `float32` | `bfloat16` | |
|---|---|---|---|
| ms/step (steady state, batch matched) | 411.0 | 390.4 | −5.0% |
| Whole-run out tok/s | 50.96 | 52.20 | **+2.4%** |
| Whole-run acceptance length | 2.658 | 2.454 | −7.7% |
| KV pool @ 68 GiB | 1,274,196 tok | **1,344,234 tok** | +5.5% |

**Net +2.4%, and it is not shipped by default.** The engine step really does get faster —
that part is mechanical and matches the byte accounting. But the acceptance figure moved
−7.7%, which is inside the ±20% noise floor established above and therefore neither
confirms nor rules out the predicted cost: bf16 perturbs the sampled distribution, and
under probabilistic rejection sampling acceptance = 1 − TV(p, q). A real 7.7% acceptance
loss would more than cancel a 5% step-time gain. Resolving it needs repeats.

Given a single run, a gain inside the noise, and thousands of hours of long-context batch
work to protect, `SSM_DTYPE` stays unset. Enable it with `SSM_DTYPE=bfloat16 ./start.sh`
if you want to pursue it — and measure acceptance across repeats, not one run.

The KV gain is a second-order effect worth knowing about regardless: a smaller SSM state
shrinks the mamba page, so the forced attention block size drops 1648 → 880 tokens and
the same pinned pool holds ~70k more tokens (page padding rises 0.73% → 1.38%, the
smaller cost).

**Degeneration check.** The documented failure mode for reduced SSM precision is
verbosity and repetition at depth, not wrong answers, so a short-context accuracy
benchmark would not catch it. Ten requests at 58k context under bf16, generating up to
12,288 tokens each, showed **no repeated n-grams at all**. Verbosity was not measurable:
at the model's default `xhigh` reasoning effort nothing terminated naturally even at a
12,288-token cap, and no `float32` control arm was run.

### MTP variant tuning history

How the `VARIANT=mtp` configuration was chosen (random-content, measured under the 1M-YaRN config with `--max-num-seqs 4`; "accept" is tokens emitted per engine step, max k+1):

| Config | c=1 decode | c=4 decode (aggregate) | Prefill 8×16k | Notes |
|---|---|---|---|---|
| MTP k=2 (upstream default) | 23.8 tok/s | 70.7 tok/s | 1,339 tok/s | accept 2.69 |
| MTP k=2 + ladder fix | — | 71.8 tok/s | — | c=4 step time −3% |
| MTP k=3 + ladder | 14.5 tok/s ⚠ | 73.4 tok/s | — | mis-draft bug, see Known issues |
| **MTP k=5 + ladder (shipped)** | **27.5 tok/s** | 65.9 tok/s | ~1,339 tok/s | accept 3.96 |
| MTP k=5 + FlashInfer attention | 28.7 tok/s | 53.3 tok/s ⚠ | 1,448 tok/s | mis-draft bug, see Known issues |
| DSpark, greedy drafting | 15.9 tok/s | 41.2 tok/s | — | before the probabilistic fix |
| k=5 + `--max-num-batched-tokens 16384` | — | — | 1,309 tok/s | worse 16k TTFT; not shipped |

Long context under `VARIANT=mtp` (single stream, 128k-token prompt, 1M-YaRN config):

| | MTP k=2 | MTP k=5 (shipped) |
|---|---|---|
| Prefill throughput at 128k depth | ~555 tok/s | ~590 tok/s |
| Decode at 128k context | 8.4 tok/s | **17.9 tok/s** |

### Methodology

- Speculative acceptance (and therefore tok/s) is strongly content-dependent. Compare configs on step time (ITL) and acceptance length separately; `bench.sh` pins its seeds so prompts are identical across runs, and servers should be restarted between config comparisons so the prefix cache is cold.
- **Scenario choice matters more than it looks.** `dec1`/`dec4`/`pre16k` use 1k–16k random-token prompts. Production traffic here is ~32k median context with 89.6% prefix-cache hits arriving as multi-turn sessions, and concurrency measured on short random prompts does not predict concurrency on that shape. The `sess8`/`sess12`/`sess16`/`sess20` scenarios use the `prefix_repetition` dataset: one 28k prefix per concurrent slot, `SESS_REQS` turns of 4k suffix against it. `SESS_REQS=12` (the default) puts prefill at roughly 18% of wall clock and recorded acceptance 2.390 in `tune-D-s20-kv68/sess12.json`, close to the 2.42–2.46 seen in production; `SESS_REQS=4` puts prefill at roughly 28% and reads acceptance 2.5–3.5, i.e. optimistically. (Those percentages are solved from `tune-B/sess12` vs `tune-D/sess12` — identical config, 4 vs 12 turns — not from 1/n, which is the fraction of *turns* that are cold rather than the fraction of time spent prefilling.)
- **For concurrency comparisons use `seq-steps/s` = `steps/s × seqs/step`, not tok/s.** Engine steps come from `vllm:iteration_tokens_total_count`; `vllm:spec_decode_num_drafts_total` counts one draft per *sequence* per step, so `drafts/iterations` is the mean batch occupancy. Because `num_prefixes` tracks concurrency, each `sess<c>` benchmarks a different prompt set, and acceptance differences between them are content artifacts rather than concurrency effects — `seq-steps/s` is invariant to that and `tok/s` is not.
- `./bench-table.sh [label-prefix …]` renders `bench-results/` as a markdown comparison table.
- Don't measure vLLM throughput from streaming timestamps: it flushes stream deltas in multi-token bursts, which can inflate first-token-to-last-token rates about 2×. Use wall-clock timing or `bench.sh`.

## SGLang status

`./start-sglang.sh` serves the same RadixArk checkpoint + DSpark drafter under SGLang, with flags from the community GB10 setup at [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38) (use that repo directly for a hardened systemd deployment). Since vLLM's DSpark variant reached parity on real-content decode, SGLang's remaining edges are the 128k-depth prefill and random-content single-stream; it gives up `reasoning_effort` control, tops out at 7 concurrent requests, and its draft verification is looser than vLLM's (see the footnote above). Quality checks on this box were clean: 60+ battery runs with zero repetition loops and correct answers throughout.

**SGLang cannot serve the unsloth checkpoint**: its partial support for that compressed-tensors scheme (startup logs `Falling back to UnquantizedLinearMethod`, allocates bf16 KV) degrades output into hard repetition loops regardless of sampling or verification settings. `SGLANG_TARGET=unsloth SGLANG_EXPERIMENTAL=1` exists only for retesting after SGLang updates.

## Known issues and safeguards

- **Why the vLLM image is pinned**: stable vLLM releases cannot load this model (`'MergedColumnParallelLinear' object has no attribute 'data'`), and the pinned nightly contains the model-critical work: the fused CUDA GDN-MTP decode kernel (#51674), the SM12x FlashInfer XQA decode path (#49718), the packed-GDN decode launch fix (#52030), and the interleaved-mrope torch.compile fix (#52005). An open revert (#51987) may remove the XQA path from later nightlies, so a blind re-pull can regress decode. Worth re-evaluating the pin once these merge: **#52244** (prefix-cache hits under speculative decoding on hybrid models), #52000, #52013, #50862, #51954.
- **CUDA-graph capture ladder**: with spec decode, a uniform decode batch is `c×(1+k)` tokens at concurrency `c`; sizes missing from `--cudagraph-capture-sizes` run attention eagerly every step (PR #52000). Each variant in `start.sh` ships a ladder covering `c = 1..8` for its k.
- **MTP k=3 is broken in this build**: single-stream decode mis-drafts and collapses to ~15 tok/s (reproduced twice). k=5 is the measured MTP optimum.
- **FlashInfer attention mis-drafts at high concurrency**: +8% prefill and slightly faster single-stream in A/B, but spec-decode batches at width 24 mis-draft (c=4 falls to 53.3 tok/s), the same kernel family as upstream revert #51987. `triton_attn` stays the default until that settles.
- **Launch lottery + health probe**: some vLLM launches come up with a mis-drafting speculative state on a bit-identical config (measured under MTP: position-0 acceptance ~44% instead of ~77%, decode ~16 instead of ~27 tok/s); a container restart fixes it. After readiness, `start.sh` probes the acceptance counters and restarts once automatically if position-0 acceptance is below the variant's threshold (0.55 for mtp, 0.25 for dspark, whose position-0 rate is legitimately lower).
- **Prefix caching is currently defeated by speculative decoding** on this hybrid-GDN model (PR #52244 tracks the fix), so multi-turn conversations re-pay the full prefill each turn.
- **Unified-memory freeze protection**: a memory spiral on the GB10 freezes the whole machine, and container `--memory` caps cannot prevent it because GPU/unified allocations bypass the container cgroup (one hard freeze observed on this box during an engine start while the previous engine's memory was still held). Two safeguards cover it: the start scripts refuse to boot until 100 GiB is actually free, and `memguard.sh` kills the engine container if available memory stays under 4 GiB at runtime.
- **Day-1 tokenizer bug (unsloth checkpoint)**: copies downloaded before 2026-08-15 silently truncated every prompt to 2,048 tokens. Verify `truncation` is `None` in the cached `tokenizer.json`, or re-run `./download.sh --mtp`.

## Troubleshooting

- Tail the server log with `tail -f .vllm.log` or `tail -f .sglang.log` (or `docker logs -f <container>`).
- The start scripts print the last 200 log lines and exit if the container dies before becoming ready.
- Port already in use or won't start: `./stop.sh` (stops either engine), then start again.
- The server vanished while running: check `.memguard.log`; the memory guard removes the engine container to protect the host. Find what ate the memory, then restart.
- Slow decode right after a start: the launch lottery drew badly and the probe's one retry wasn't enough. `./stop.sh && ./start.sh`.
- Repeating or garbled responses: you are probably running SGLang with the experimental unsloth target; use the defaults instead.
- Download stalls: re-run `./download.sh`; it resumes.
- CUDA/arch errors at startup: confirm you're on the pinned images (the vLLM container sets `CUTE_DSL_ARCH=sm_121a` for GB10).

## Repository layout

```
.
├── download.sh       # fetch checkpoints into ./.cache/huggingface (retrying)
├── start.sh          # serve with vLLM (VARIANT=dspark default | mtp)
├── start-sglang.sh   # serve with SGLang (alternative engine)
├── stop.sh           # stop whichever engine is running
├── memguard.sh       # runtime memory guard (started by the start scripts)
├── bench.sh          # benchmark whatever is serving
├── bench-table.sh    # render bench-results/ as a markdown comparison table
├── .gitignore        # excludes .cache/, logs, pid files
├── LICENSE
└── README.md
```

## Credits

- [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) and [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark): the default checkpoint and the DSpark block drafter
- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4): the MTP-variant checkpoint and its docs (YaRN long-context recipe, sampling recommendations)
- [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang): the inference engines
- The r/LocalLLM [DSpark-in-vLLM PSA](https://www.reddit.com/r/LocalLLM/comments/1vpo15s/psa_qwen3827b_dspark_works_in_vllm/): the architecture patch and the probabilistic-drafting finding
- [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38): validated SGLang GB10 flags and a hardened systemd deployment
- [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000): the original repo this fork is based on
