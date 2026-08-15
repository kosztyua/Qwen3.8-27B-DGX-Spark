# Qwen3.8 27B for DGX Spark

[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Opinionated, ready-to-run scripts to serve **[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)** with **vLLM** in Docker on an NVIDIA DGX Spark (GB10, aarch64). One script downloads the weights, one starts an OpenAI-compatible server, one stops it, one benchmarks it.

> Forked from [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000) (the original repo, which also covers the RTX 6000 PRO). This fork is DGX-Spark-only and adds the measured performance tuning below (MTP `k=5`, CUDA-graph ladder, pinned image, spec-decode health probe).

- **NVFP4** quantized checkpoint (compressed-tensors) — 4-bit weights/activations with 8-bit groups for sensitive modules
- **262k-token native context** by default; the full **1M-token context** via static YaRN with `CONTEXT_1M=1 ./start.sh`
- **FP8 KV cache** (calibrated `kv_cache_scheme` from the checkpoint, passed explicitly via `--kv-cache-dtype fp8`, ~2× KV memory savings)
- **MTP speculative decoding** (`num_speculative_tokens: 5`) — measured fastest on this hardware: 27.5 tok/s single-stream (+15% vs `2`), 2.1× decode speed at 128k context
- **CUDA-graph capture ladder** sized for spec decode (`c×(1+k)` for every concurrency) so decode never falls back to eager attention
- **Pinned vLLM nightly image** (by digest) — the only vLLM that currently loads this model, with the SM12x XQA decode path an open upstream PR may remove
- Built-in **reasoning parsing** (Qwen3 thinking mode) and **tool calling** (`qwen3_coder` parser)

---

## Requirements

| Component | Detail |
|---|---|
| Hardware | NVIDIA DGX Spark / GB10 (aarch64, sm_121a) |
| Docker | With NVIDIA Container Toolkit / GPU passthrough working (`docker run --gpus all`) |
| vLLM image | `vllm/vllm-openai@sha256:b5c860…` (pinned 2026-08-15 nightly; stable releases cannot load this checkpoint) |
| CLI tools | `docker`, `curl`, and the Hugging Face CLI (`hf`) for the download step |
| Hugging Face token | `HF_TOKEN` defined in `~/.bashrc` (gated/repo access, higher rate limits) |

## Quick start

```bash
# 1. Download the checkpoint into this directory's HF cache (retries up to 10x)
./download.sh

# 2. Start the server (waits until the HTTP API is ready, then exits)
./start.sh

# 3. Use it
curl http://127.0.0.1:8888/v1/models

# 4. Stop it
./stop.sh
```

`start.sh` is idempotent: if the container is already running it says so and exits; if a stopped container exists it removes it first.

## Scripts

| Script | What it does |
|---|---|
| `download.sh` | Pre-downloads `unsloth/Qwen3.8-27B-NVFP4` into `./.cache/huggingface` with up to 10 retries (plain HTTP, xet disabled, so stalled downloads resume cleanly). Reads `HF_TOKEN` from `~/.bashrc`. |
| `start.sh` | Launches the vLLM container (`docker run -d`, host network), streams logs to `.vllm.log`, records the container ID in `.vllm.pid`, and polls `http://127.0.0.1:8888/v1/models` until the server is ready. |
| `stop.sh` | Stops the container, removes `.vllm.pid`, and leaves the stopped container in place for `docker logs` post-mortem (the next `start.sh` removes it). |
| `bench.sh` | Benchmarks the running server with `vllm bench serve` (decode c=1/c=4, 16k prefill, optional 128k long-context via `--full`). Fixed seeds for cross-config comparability; results in `.cache/huggingface/bench-results/<label>/`. |

Runtime artifacts: `.vllm.log` (server log), `.vllm.pid` (container ID), `.cache/` (HF + Triton caches). All are git-ignored.

## Configuration

Defaults live at the top of `start.sh`:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_ID` | `unsloth/Qwen3.8-27B-NVFP4` | Checkpoint to serve |
| `SERVED_MODEL_NAME` | `qwen38-27b-unsloth-nvfp4` | Name clients use in API requests |
| `IMAGE` | `vllm/vllm-openai@sha256:b5c860…` | **Pinned by digest** = `nightly-aarch64` as of 2026-08-15 (`v0.27.2rc1.dev110+gacb0f1dcd`); see "Why the image is pinned" |
| `CONTAINER_NAME` | `qwen3.8-27b-nvfp4` | Also used by `stop.sh` |
| `PORT` | `8888` | Listens on `0.0.0.0` via host networking |
| `--gpu-memory-utilization` | `0.84` | |
| `--max-model-len` | `262,144` (native) | `CONTEXT_1M=1 ./start.sh` serves 1,000,000 via YaRN; see notes below |
| `--max-num-seqs` | `4` | Concurrent sequences |
| `--speculative-config` | MTP, `num_speculative_tokens: 5` | Measured optimum on this box; see benchmark table |
| `--cudagraph-capture-sizes` | `1 2 4 6 8 12 16 18 24` | Must contain `c×(1+k)` for `c = 1..max-num-seqs`, `k` = spec tokens |

### Context length: native 262k by default, YaRN 1M opt-in

The model's native context is **262,144 tokens**, which `start.sh` serves by default — no RoPE modification, no quality caveat. For the full **1M-token** context, launch with `CONTEXT_1M=1 ./start.sh`: this applies the model card's static-YaRN recipe (factor 4.0 over the native 262,144 via `--hf-overrides`). Per the model card, static YaRN can slightly degrade short-context quality, which is why it is no longer the default. Decode/prefill speed is the same either way; the KV pool (~2.26M tokens) covers ~8× concurrent 262k-token requests or 2× 1M.

### Notable serving choices

- **Context:** 1M tokens via static YaRN (factor 4.0 over native 262,144, applied through `--hf-overrides`). Per the model card, static YaRN can slightly impact short-context quality. With the FP8 KV cache, a single 1M-token sequence needs ~32 GB of KV; concurrent long sequences are admitted as KV space allows. Override `--kv-cache-dtype bfloat16` in `start.sh` to force a bf16 KV cache instead of the checkpoint's FP8 scheme.
- **Quantization:** the checkpoint is dynamic NVFP4 in compressed-tensors format, hence `--quantization compressed-tensors`. Dense model (SwiGLU, no experts) — no `--moe-backend` needed. A same-hardware community A/B measured NVFP4 ~30% faster than the FP8 checkpoint on GB10 — stay on NVFP4.
- **Speculative decoding:** unsloth ships MTP weights (`model_mtp.safetensors`), used via `--speculative-config '{"method": "mtp", "num_speculative_tokens": 5}'`. `k=5` measured fastest single-stream on this box (see benchmark table); it costs ~7% aggregate throughput at 4 concurrent streams vs `k=2`, and wins big at long context. **Do not use `k=3`** — this vLLM build mis-drafts in the 4-token decode path (single stream at `k=3`) and single-stream decode collapses to ~15 tok/s.
- **CUDA-graph capture ladder:** with spec decode, a uniform decode batch is `c×(1+k)` tokens at concurrency `c`. Sizes missing from `--cudagraph-capture-sizes` run attention **eagerly** every step (vLLM PR #52000). The default ladder misses 12 (=4×3) at `k=2` and most sizes at other `k`, so `start.sh` passes an explicit ladder covering `c = 1..4` for `k=5`: 6/12/18/24.
- **Multimodal:** video support enabled via `--media-io-kwargs '{"video": {"num_frames": -1}}`.
- **Warm-restart caches:** `.cache/vllm` (torch.compile artifacts) and `.cache/flashinfer` (autotune) are mounted into the container; warm restarts skip ~35-45s of compilation.
- **Spec-decode health probe:** this vLLM build has a per-launch initialization lottery — some starts come up with a mis-drafting speculative state (MTP position-0 acceptance ~44% instead of ~77%, single-stream decode ~16 instead of ~27 tok/s; reproducible on identical configs, fixed by a plain container restart). After readiness, `start.sh` fires three short generations, reads the acceptance counters from `/metrics`, and restarts the container once automatically if the rate is below 0.55.

### Why the image is pinned

`start.sh` pins `vllm/vllm-openai@sha256:b5c860acda75d737a8e58cc99ba86ff13982695dceae194f906c2d7b54979358` (= `nightly-aarch64` of 2026-08-15, `v0.27.2rc1.dev110+gacb0f1dcd`):

- **Stable releases cannot load this model** — v0.27.1 and v0.25.1 fail with `'MergedColumnParallelLinear' object has no attribute 'data'`.
- This nightly contains the model-critical perf work: the fused CUDA GDN-MTP decode kernel (#51674), the SM12x FlashInfer XQA decode path (#49718) that enables FULL CUDA graphs for speculative uniform batches, the packed-GDN decode launch fix (#52030), and the interleaved-mrope torch.compile fix (#52005).
- An open revert (#51987) may **remove** the SM12x XQA path from later nightlies (gpt-oss produced gibberish on the Spark CI runner), so a blind re-pull can regress decode.

Worth re-pinning once these merge upstream: **#52244** (restores hybrid-GDN prefix-cache hits under MTP — today MTP largely defeats prefix caching on this model, so multi-turn TTFT re-pays the full prefill), **#52000** (uniform-decode graph dispatch), **#52013** (dedicated MTP draft `lm_head`), **#50862** (FlashInfer GDN prefill on SM12x), **#51954** (GDN decode gate copies).

## Attention backend note

`--attention-backend triton_attn` serves the 16 full-attention layers (the other 48 are Gated DeltaNet; the vision tower runs FlashAttention). FlashInfer also supports FP8 KV on SM121 in this build and was A/B-tested: it is **+8% prefill** (1448 vs 1339 tok/s @16k) and slightly faster single-stream, **but at `k=5` / 4 concurrent streams its 24-token decode batches mis-draft** (MTP position-0 acceptance drops from 69% to 42%), which also means its verify pass is numerically suspect — the same kernel family behind upstream revert #51987. Until that settles, `triton_attn` is the safe default. `flash_attn` would require `--kv-cache-dtype bfloat16` (FA2 cannot serve FP8 KV on SM121).

## Measured numbers (this box)

vLLM `v0.27.2rc1.dev110+gacb0f1dcd` (pinned nightly), 1 × DGX Spark GB10, `--gpu-memory-utilization 0.84`, YaRN 1M context, FP8 KV cache. `vllm bench serve`, random dataset with fixed seeds, thinking-mode sampling (temp 1.0 / top-p 0.95 / top-k 20), `--ignore-eos`.

Config sweep (2026-08-15), decode scenarios are 1k-token prompts with 1k-token generations:

| Config | c=1 decode | c=4 decode (aggregate) | Prefill 8×16k, c=4 | Notes |
|---|---|---|---|---|
| MTP k=2 (previous default) | 23.8 tok/s | 70.7 tok/s | 1,339 tok/s | accept 2.69/step |
| MTP k=2 + ladder 12 | — | 71.8 tok/s | — | c=4 step time −3% |
| MTP k=3 + ladder | **14.5 tok/s** ⚠ | 73.4 tok/s | — | width-4 mis-draft bug |
| **MTP k=5 + ladder (shipping)** | **27.5 tok/s** | 65.9 tok/s | ~1,339 tok/s | accept 3.96/step |
| MTP k=5 + FlashInfer attn | 28.7 tok/s | **53.3 tok/s** ⚠ | 1,448 tok/s | width-24 mis-draft bug |
| DSpark drafter (RadixArk 1.4B) | 15.9 tok/s | 41.2 tok/s | — | accept 1.89/step; see below |
| `--max-num-batched-tokens 16384` | — | — | 1,309 tok/s | worse 16k TTFT; kept 8192 |

Long-context (single stream, 128k-token prompt, 256-token generation):

| | MTP k=2 | MTP k=5 (shipping) |
|---|---|---|
| Prefill throughput at 128k depth | ~555 tok/s | ~590 tok/s |
| Decode at 128k context | 8.4 tok/s | **17.9 tok/s** (accept 4.48/step) |

KV / concurrency (from the engine startup log):

- GPU KV cache size: **~2.25M tokens** (FP8), max concurrency 2.25× @ 1M-context requests
- Hard cap from config: **4 concurrent sequences** (`--max-num-seqs`); extra requests queue
- Memory-wise with FP8 KV: 2 × 1M-context, ~4 × 512k, ~8 × 262k, 4 × ≤128k (config-capped)
- Checkpoint size: 22.6 GB (`model.safetensors` + `model_mtp.safetensors`)
- Live KV usage: `curl -s localhost:8888/metrics | grep kv_cache_usage_perc`; MTP acceptance: `grep spec_decode`

Real-workload single-stream decode (same three prompts on both engines, this box, wall-clock / per-token streaming):

| Engine | Code (greedy) | Reasoning (greedy) | Math (temp 0.6) |
|---|---|---|---|
| **vLLM MTP k=5 (this repo)** | 28.0 tok/s | 31.5 tok/s | ~31 tok/s |
| SGLang + DSpark (see below) | **38.3 tok/s** | **37.5 tok/s** | **45.2 tok/s** |

Benchmark methodology notes: MTP acceptance (and thus tok/s) is strongly content-dependent — real text accepts far better than random continuations (~89% position-0 vs ~66-77%), and run-to-run acceptance varied 2.4-2.7 at `k=2` on different random seeds. Compare configs on inter-token step time (ITL) and acceptance separately, with pinned seeds. Beware streaming-based tok/s measurements against vLLM: it flushes stream deltas in multi-token bursts, which can inflate first-token-to-last-token rates ~2× — use wall-clock or `vllm bench serve`.

### SGLang + DSpark: ~30% faster single-stream, fewer features (evaluated on this box)

[SGLang](https://github.com/sgl-project/sglang) with the DSpark block-drafter is the fastest known single-stream config for this model on DGX Spark, and it reproduced here: **38.3 / 37.5 / 45.2 tok/s** on code/reasoning/math prompts vs 28-31 for this repo's vLLM setup (`lmsysorg/sglang:qwen38-27b`, `RadixArk/Qwen3.8-27B-NVFP4` target + `RadixArk/Qwen3.8-27B-DSpark` 1.4B drafter, flags from [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38) — a hardened one-command systemd setup and the recommended way to run it).

Trade-offs measured/known vs this repo's vLLM setup:

- On random-content benchmarks the ranking flips (SGLang 20.4 tok/s c=1 / 51.5 c=4 aggregate vs vLLM 27.5 / 65.9): the 1.4B drafter only shines on predictable real text.
- 262k context (no validated YaRN-1M recipe), a different quantizer's NVFP4 checkpoint, and a patched chat template; tool-calling / vision parity should be spot-checked with your clients.
- vLLM-native DSpark (`{"method": "dspark", ...}` with the drafter's `architectures` patched to `Qwen3DSparkModel`) launches but mis-drafts against this target (1.89/step, position-0 53% → 15.9 tok/s) and shrinks the KV pool to ~1.47M tokens — not worth it today; a vLLM-tuned Qwen3.8 drafter would change that.

## Using the API

OpenAI-compatible base URL: `http://127.0.0.1:8888/v1` (model name: `qwen38-27b-unsloth-nvfp4`).

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-27b-unsloth-nvfp4",
    "messages": [{"role": "user", "content": "Explain YaRN in two sentences."}]
  }'
```

**Thinking mode** is on by default (`--reasoning-parser qwen3` separates `<think>` blocks into `reasoning_content`). Default sampling params (from the repo's `generation_config.json`): `temperature=1.0, top_p=0.95, top_k=20`.

For **non-thinking / instruct** requests, override per request (per the model card) — disable thinking via `chat_template_kwargs` and use:

```json
{
  "model": "qwen38-27b-unsloth-nvfp4",
  "messages": [{"role": "user", "content": "Write a haiku about GB10."}],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "presence_penalty": 1.5,
  "chat_template_kwargs": { "thinking": false }
}
```

Tool calling is enabled (`--tool-call-parser qwen3_coder --enable-auto-tool-choice`); pass `tools` / `tool_choice` as in the OpenAI API.

**Reasoning effort:** the chat template defaults to `reasoning_effort: "xhigh"`, which produces very long thinking traces — often the dominant share of end-to-end latency. Lower it per request via `chat_template_kwargs: {"reasoning_effort": "medium"}` (or `"low"`), or server-wide by adding `--default-chat-template-kwargs '{"reasoning_effort": "medium"}'` to `start.sh`.

## Logs & troubleshooting

- Tail the server log: `tail -f .vllm.log` (or `docker logs -f qwen3.8-27b-nvfp4`)
- `start.sh` prints the last 200 log lines and exits if the container dies before becoming ready
- If startup fails with CUDA/arch errors, confirm you're on the pinned aarch64 image (the container sets `CUTE_DSL_ARCH=sm_121a` for GB10's cutlass kernels)
- Model download stalls: just re-run `./download.sh`; it resumes
- **Day-1 tokenizer bug:** checkpoints downloaded before 2026-08-15 shipped a `tokenizer.json` that silently truncated every prompt to 2,048 tokens. Verify yours is fixed: `python3 -c "import json,glob; print(json.load(open(glob.glob('.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/*/tokenizer.json')[0]))['truncation'])"` must print `None`; if not, re-run `./download.sh`
- **Multi-turn TTFT:** with MTP on, prefix caching gets almost no hits on this hybrid-GDN model (vLLM PR #52244 tracks the fix), so each turn re-pays the full prefill for now

## Repository layout

```
.
├── download.sh   # fetch checkpoint into ./.cache/huggingface (retrying)
├── start.sh      # launch vLLM container, wait for readiness
├── stop.sh       # stop the container, clean up pid file
├── bench.sh      # benchmark the running server (vllm bench serve)
├── .gitignore    # excludes .cache/, logs, pid file
└── README.md
```

## Credits

- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — quantized checkpoint and docs (YaRN long-context recipe, MTP spec decode, sampling recommendations)
- [vLLM](https://github.com/vllm-project/vllm) — inference engine and OpenAI-compatible server
