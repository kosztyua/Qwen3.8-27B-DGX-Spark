# Qwen3.8 27B for DGX Spark

[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Ready-to-run scripts to serve **[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)** on an NVIDIA DGX Spark (GB10, aarch64), with two interchangeable engines:

- **vLLM** (`./start.sh`, the default) serves the unsloth checkpoint with the full feature set: 262k or 1M context, vision, tool calling, reasoning parsing, MTP speculative decoding.
- **SGLang + DSpark** (`./start-sglang.sh`) serves the [RadixArk NVFP4 checkpoint](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) of the same model, which decodes real content ~30-50% faster. SGLang cannot serve the unsloth checkpoint correctly today; see [SGLang status](#sglang-status).

Both serve an OpenAI-compatible API on the same port, under different model names (`qwen38-27b-unsloth-nvfp4` vs `qwen38-27b-radixark-nvfp4`) since they serve different quantizations. All performance claims in this README were measured on one DGX Spark on 2026-08-15/16; see [Benchmarks](#benchmarks).

> Forked from [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000) (the original repo, which also covers the RTX 6000 PRO). This fork is DGX-Spark-only and adds measured performance tuning and the SGLang alternative.

## Requirements

| Component | Detail |
|---|---|
| Hardware | NVIDIA DGX Spark / GB10 (aarch64, sm_121a, 128 GB unified memory) |
| Docker | With NVIDIA Container Toolkit / GPU passthrough working (`docker run --gpus all`) |
| Disk | ~25 GB for checkpoints (+3 GB with `--sglang`), plus the engine images |
| CLI tools | `docker`, `curl`, `python3` (health probe / smoke test); the Hugging Face CLI (`hf`) for `download.sh` |
| Hugging Face token | `HF_TOKEN` defined in `~/.bashrc` (higher rate limits; not strictly required) |

Engine images are **pinned by digest** in the start scripts (both are the 2026-08-15 builds validated here): `vllm/vllm-openai@sha256:b5c860…` and `lmsysorg/sglang@sha256:febfb9…`. See [Why the vLLM image is pinned](#why-the-vllm-image-is-pinned).

## Quick start

```bash
# 1. Download the checkpoint into ./.cache/huggingface (retries, resumes)
./download.sh            # add --sglang to also fetch the DSpark drafter

# 2. Start ONE engine (each waits until the API is ready, then exits)
./start.sh               # vLLM, unsloth checkpoint (full feature set)
# ...or
./start-sglang.sh        # SGLang, RadixArk checkpoint (faster decode)

# 3. Use it
curl http://127.0.0.1:8888/v1/models

# 4. Benchmark it (optional; ~10 min)
./bench.sh mylabel

# 5. Stop whichever engine is running
./stop.sh
```

The start scripts are idempotent: if their container is already running they say so and exit; a stopped leftover container is removed first. Each start script refuses to start while the other engine's container is running (same port, same GPU); run `./stop.sh` first.

## Scripts

| Script | What it does |
|---|---|
| `download.sh` | Pre-downloads `unsloth/Qwen3.8-27B-NVFP4` (and with `--sglang` the `RadixArk/Qwen3.8-27B-DSpark` drafter) into `./.cache/huggingface`, with up to 10 attempts per repo (plain HTTP, xet disabled, so stalled downloads resume cleanly). |
| `start.sh` | Launches the pinned vLLM container (`docker run -d`, host network), streams logs to `.vllm.log`, records the container ID in `.vllm.pid`, waits for readiness, then runs a speculative-decode health probe (see [Known issues](#tuning-notes--known-issues)). Refuses to boot until 100 GiB of memory is actually free, and starts the runtime memory guard. `CONTEXT_1M=1 ./start.sh` serves the 1M-token YaRN context. |
| `start-sglang.sh` | Launches the pinned SGLang container serving the RadixArk checkpoint with DSpark speculative decoding (logs to `.sglang.log`, ID in `.sglang.pid`), with a memory pre-flight that refuses to boot until the other engine's memory is released. Waits for readiness, runs a generation smoke test. `SGLANG_TARGET=unsloth` (broken, needs `SGLANG_EXPERIMENTAL=1`) exists for retesting; see [SGLang status](#sglang-status). |
| `stop.sh` | Stops whichever engine container is running (and the memory guard), removes the pid files, and leaves the stopped container in place for `docker logs` post-mortem (the next start removes it). |
| `memguard.sh` | Runtime memory guard, started automatically by both start scripts. Watches `MemAvailable` every 2 s and force-removes the engine container if it stays under 4 GiB — on the GB10 a memory spiral otherwise freezes the whole machine, and container `--memory` caps don't cover GPU/unified allocations. Actions are logged to `.memguard.log`. |
| `bench.sh` | Benchmarks whatever is serving on the port using `vllm bench serve` in a throwaway client container, so it works against both engines. Fixed seeds for cross-config comparability; results in `.cache/huggingface/bench-results/<label>/`. `--full` adds the 128k long-context scenario. |

Runtime artifacts (`.vllm.log`, `.sglang.log`, pid files, `.cache/`) are git-ignored.

## SGLang status

**Works with the RadixArk checkpoint (the default); broken with unsloth's.** `./start-sglang.sh` serves [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4), the NVIDIA-modelopt-format NVFP4 quantization the SGLang community validated. Quality check on this box (2026-08-16): 30+ battery runs across looping-prone prompts, code, and exact-answer questions, at both recommended sampling and greedy, produced zero repetition loops, zero non-terminating responses, and correct answers throughout.

Measured with the RadixArk checkpoint: **38.3 / 37.5 / 45.2 tok/s** single-stream on real code/reasoning/math prompts (vLLM does 28-31 on the same prompts) and 20.4 tok/s c=1 / 51.5 c=4 on the random-content benchmark (where vLLM wins with 27.5 / 65.9). Choose by workload: SGLang for interactive reasoning/code speed, vLLM for batched or random-content throughput and the bigger feature set.

Trade-offs vs the vLLM setup:

- It is a different quantization of the same base model, from NVIDIA's modelopt toolchain rather than unsloth's calibration pipeline. Quality differences between the two quants have not been evaluated head-to-head here.
- Context is the native 262k (no validated YaRN-1M recipe), and the `reasoning_effort` template kwarg is not passed through, so every thinking request runs at `xhigh` (`enable_thinking: false` works; a patched `--chat-template` with a different default is the workaround). Vision input is untested.
- Startup can pressure the GB10's unified memory (one hard freeze observed on this box during a cold boot; the container `--memory` caps don't fully account GPU allocations). Two safeguards now cover this, for both engines: the start scripts refuse to boot until at least 100 GiB is actually free (the freeze case was booting while the other engine still held its memory), and `memguard.sh` watches `MemAvailable` at runtime and kills the engine container before the host starves.

**The unsloth checkpoint remains broken under SGLang** (`SGLANG_TARGET=unsloth`, additionally gated behind `SGLANG_EXPERIMENTAL=1` for retesting after SGLang updates). Thinking-mode output degrades into hard repetition loops on ordinary prompts, with degraded fluency before the loop, at recommended sampling, and identically with strict rejection-sampling verification, so the speculative-decoding mode is not the cause. Leading suspect: SGLang's incomplete support for unsloth's compressed-tensors format, which mixes NVFP4 and FP8 quantization groups; at startup it logs `Falling back to UnquantizedLinearMethod` and allocates a bf16 KV cache instead of the checkpoint's calibrated FP8 scheme. The loop-contaminated speed numbers previously measured on that path (42-49 tok/s) are invalid: loops draft near-perfectly in speculative decoding.

## Configuration

### vLLM (`start.sh`)

| Setting | Default | Notes |
|---|---|---|
| `MODEL_ID` | `unsloth/Qwen3.8-27B-NVFP4` | Checkpoint to serve |
| `SERVED_MODEL_NAME` | `qwen38-27b-unsloth-nvfp4` | Name clients use in API requests (same for both engines) |
| `IMAGE` | `vllm/vllm-openai@sha256:b5c860…` | Pinned digest = `nightly-aarch64` of 2026-08-15 (`v0.27.2rc1.dev110`) |
| `PORT` | `8888` | Listens on `0.0.0.0` via host networking |
| `--max-model-len` | `262,144` (native) | `CONTEXT_1M=1 ./start.sh` serves 1,000,000 via YaRN |
| `--gpu-memory-utilization` | `0.84` | ~2.2M-token FP8 KV pool |
| `--max-num-seqs` | `4` | Concurrent sequences (extra requests queue) |
| `--speculative-config` | MTP, `num_speculative_tokens: 5` | Measured optimum; do **not** use 3 (see Known issues) |
| `--cudagraph-capture-sizes` | `1 2 4 6 8 12 16 18 24` | Must contain `c×(1+k)` for `c = 1..max-num-seqs` |
| `--attention-backend` | `triton_attn` | FlashInfer is faster but mis-drafts at 4-way concurrency (see Known issues) |
| `--kv-cache-dtype` | `fp8` | The checkpoint's calibrated FP8 KV scheme (~2× KV memory savings vs bf16) |

**Context length**: the model's native context is 262,144 tokens, served by default with no RoPE modification. `CONTEXT_1M=1 ./start.sh` applies the model card's static-YaRN recipe (factor 4.0) for 1M tokens; per the model card, static YaRN can slightly degrade short-context quality, which is why it is opt-in. Decode/prefill speed is the same either way. A single 1M-token sequence needs ~32 GB of KV; the pool covers ~8 concurrent 262k sequences or 2 × 1M.

### SGLang (`start-sglang.sh`)

Serves `RadixArk/Qwen3.8-27B-NVFP4` as `qwen38-27b-radixark-nvfp4` on the same port (see [SGLang status](#sglang-status) for why it is a different checkpoint). `SGLANG_TARGET=unsloth` selects the broken unsloth path for retesting and additionally requires `SGLANG_EXPERIMENTAL=1`. Key flags (from the community GB10 setup at [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38)): `--mem-fraction-static 0.50`, `--attention-backend flashinfer`, `--speculative-algorithm DSPARK` with `RadixArk/Qwen3.8-27B-DSpark` (block size 7, drafter unquantized), `--enable-torch-compile`, `--num-continuous-decode-steps 2`, plus hard `--memory 100g` caps and a pre-flight that waits for 100 GiB of free memory before booting (unified-memory freeze protection). First boot of a new configuration spends ~5-9 minutes in `torch.compile`; the cache under `.cache/sglang/` makes later boots faster. For a production SGLang deployment (systemd, API key, Anthropic-compatible endpoint), use the hasso5703 repo directly.

## Using the API

OpenAI-compatible base URL: `http://127.0.0.1:8888/v1`. Model name: `qwen38-27b-unsloth-nvfp4` on vLLM, `qwen38-27b-radixark-nvfp4` on SGLang (or ask `GET /v1/models`).

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38-27b-unsloth-nvfp4",
    "messages": [{"role": "user", "content": "Explain YaRN in two sentences."}]
  }'
```

- **Thinking mode** is on by default; the `qwen3` reasoning parser separates `<think>` blocks into `reasoning_content`. Disable it per request with `"chat_template_kwargs": {"enable_thinking": false}`.
- **Reasoning effort** defaults to `xhigh`, which produces very long thinking traces, often the dominant share of end-to-end latency. On vLLM, lower it per request with `"chat_template_kwargs": {"reasoning_effort": "medium"}` (or `"low"`), or server-wide by adding `--default-chat-template-kwargs '{"reasoning_effort": "medium"}'` to `start.sh`. On SGLang this kwarg is currently ignored (only the `enable_thinking` toggle works).
- **Sampling defaults** come from the checkpoint's `generation_config.json`: `temperature=1.0, top_p=0.95, top_k=20` (thinking mode). For non-thinking / instruct requests the model card recommends `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` together with `enable_thinking: false`.
- **Tool calling** is enabled on both engines (`qwen3_coder` parser + auto tool choice on vLLM); pass `tools` / `tool_choice` as in the OpenAI API.

## Benchmarks

All numbers: 1 × DGX Spark GB10, 2026-08-15, both engines serving `unsloth/Qwen3.8-27B-NVFP4`. Two kinds of measurements:

- **Random-content** (`./bench.sh`): `vllm bench serve`, random dataset, fixed seeds, thinking-mode sampling (temp 1.0 / top-p 0.95 / top-k 20), `--ignore-eos`. Comparable and repeatable, but speculative acceptance on random continuations is *lower* than on real text, so these understate real-use decode speed.
- **Real-workload**: fixed code/reasoning/prose prompts, wall-clock timing. These are the numbers you'll feel in interactive use.

### vLLM config sweep (random-content, measured under the 1M-YaRN config)

Decode scenarios are 1k-token prompts with 1k-token generations; "accept" is tokens emitted per engine step (max `k+1`).

| Config | c=1 decode | c=4 decode (aggregate) | Prefill 8×16k, c=4 | Notes |
|---|---|---|---|---|
| MTP k=2 (upstream default) | 23.8 tok/s | 70.7 tok/s | 1,339 tok/s | accept 2.69 |
| MTP k=2 + ladder fix | — | 71.8 tok/s | — | c=4 step time −3% |
| MTP k=3 + ladder | 14.5 tok/s ⚠ | 73.4 tok/s | — | mis-draft bug, see Known issues |
| **MTP k=5 + ladder (shipping)** | **27.5 tok/s** | 65.9 tok/s | ~1,339 tok/s | accept 3.96 |
| MTP k=5 + FlashInfer attention | 28.7 tok/s | 53.3 tok/s ⚠ | 1,448 tok/s | mis-draft bug, see Known issues |
| DSpark drafter inside vLLM | 15.9 tok/s | 41.2 tok/s | — | accept 1.89; not viable in vLLM today |
| k=5 + `--max-num-batched-tokens 16384` | — | — | 1,309 tok/s | worse 16k TTFT than 8192; not shipped |

Long context (single stream, 128k-token prompt, measured under the 1M-YaRN config; native-262k should be the same or better):

| | MTP k=2 | MTP k=5 (shipping) |
|---|---|---|
| Prefill throughput at 128k depth | ~555 tok/s | ~590 tok/s |
| Decode at 128k context | 8.4 tok/s | **17.9 tok/s** |

### Real workloads (vLLM, single stream)

| Prompt type | vLLM MTP k=5 |
|---|---|
| Code task (greedy, thinking, terminated naturally at 726 tokens) | 28.0 tok/s |
| Reasoning task (greedy, thinking, terminated naturally at 780 tokens) | 31.5 tok/s |
| Plain-prose continuation (template-free, greedy, fixed 700 tokens) | 26.7 tok/s |

SGLang with the RadixArk checkpoint measured 38.3 / 37.5 / 45.2 tok/s on the same code/reasoning/math prompts; see [SGLang status](#sglang-status) for that path's numbers and trade-offs.

### Capacity (vLLM, from the engine startup log)

- GPU KV cache pool: **~2.2M tokens** (FP8) at `--gpu-memory-utilization 0.84`
- Memory-wise: 2 × 1M-context, ~4 × 512k, ~8 × 262k concurrent sequences; the config caps at 4 concurrent (`--max-num-seqs`)
- Checkpoint size: 22.6 GB (`model.safetensors` + `model_mtp.safetensors`); DSpark drafter +2.7 GB
- Live KV usage: `curl -s localhost:8888/metrics | grep kv_cache_usage_perc`; MTP acceptance: `… | grep spec_decode`

### Methodology warnings

- Speculative acceptance (and therefore tok/s) is strongly content-dependent. Compare configs on step time (ITL) and acceptance length separately. `bench.sh` pins its seeds so prompts are identical across runs.
- Don't measure vLLM throughput from streaming timestamps: it flushes stream deltas in multi-token bursts, which can inflate first-token-to-last-token rates about 2×. Use wall-clock timing or `bench.sh`.
- Restart the server between config comparisons so the prefix cache is cold.

## Tuning notes & known issues

### Why the vLLM image is pinned

`start.sh` pins `vllm/vllm-openai@sha256:b5c860acda75d737a8e58cc99ba86ff13982695dceae194f906c2d7b54979358` (= `nightly-aarch64` of 2026-08-15, `v0.27.2rc1.dev110+gacb0f1dcd`):

- **Stable vLLM releases cannot load this model**: v0.27.1 and v0.25.1 fail with `'MergedColumnParallelLinear' object has no attribute 'data'`. Do not "upgrade" to stable.
- This nightly contains the work the model depends on: the fused CUDA GDN-MTP decode kernel (#51674), the SM12x FlashInfer XQA decode path (#49718), the packed-GDN decode launch fix (#52030), and the interleaved-mrope torch.compile fix (#52005).
- An open revert (#51987) may remove the SM12x XQA path from later nightlies, so a blind re-pull can regress decode.

Worth re-evaluating the pin once these merge upstream: **#52244** (prefix-cache hits under MTP), #52000 (uniform-decode graph dispatch), #52013 (dedicated MTP draft `lm_head`), #50862 (FlashInfer GDN prefill on SM12x), #51954 (GDN decode gate copies).

### Speculative decoding on this vLLM build

- **`num_speculative_tokens: 5` is the measured optimum** for interactive use: +15% single-stream over k=2 and 2.1× decode speed at 128k context, for about 7% less aggregate throughput at 4 concurrent streams. For pure 4-way batch work k=2 is slightly better; give it ladder `1 2 3 4 6 8 9 12 16 24` so 3/6/9/12 are covered (see the ladder rule below).
- **Do not use `k=3`**: this build mis-drafts in the single-stream decode path at k=3 and c=1 decode collapses to about 15 tok/s (reproduced twice).
- **CUDA-graph capture ladder**: with spec decode, a uniform decode batch is `c×(1+k)` tokens at concurrency `c`; sizes missing from `--cudagraph-capture-sizes` run attention eagerly every step (PR #52000). `start.sh` ships an explicit ladder covering `c = 1..4` at `k=5` (6/12/18/24).
- **Launch lottery + health probe**: some launches of this build come up with a mis-drafting speculative state (MTP position-0 acceptance around 44% instead of 77%, single-stream decode around 16 instead of 27 tok/s) on a bit-identical config, and a plain container restart fixes it. After readiness, `start.sh` fires three short generations, reads the acceptance counters from `/metrics`, and restarts the container once automatically if the position-0 rate is below 0.55.
- **Attention backend**: `triton_attn` serves the 16 full-attention layers (the other 48 are Gated DeltaNet; the vision tower runs FlashAttention). FlashInfer was A/B-tested: +8% prefill and slightly faster single-stream, but at 4-way concurrency its spec-decode batches mis-draft (acceptance drops from 3.51 to 2.41 per step and c=4 falls to 53.3 tok/s). That is the same kernel family as upstream revert #51987, so `triton_attn` stays the default until that settles.
- **Prefix caching is currently defeated by MTP** on this hybrid-GDN model (PR #52244 tracks the fix), so multi-turn conversations re-pay the full prefill each turn.

### Other notes

- **Quantization**: NVFP4 in compressed-tensors format (`--quantization compressed-tensors`); dense model, no MoE flags. A same-hardware community A/B measured NVFP4 about 30% faster than the FP8 checkpoint on GB10, so stay on NVFP4.
- **Warm restarts**: `.cache/vllm` (torch.compile) and `.cache/flashinfer` (autotune) are mounted into the vLLM container; engine init drops from about 3 minutes to about 30 seconds on a warm restart. SGLang keeps its inductor cache under `.cache/sglang/`.
- **Multimodal**: vLLM serves image/video input (`--media-io-kwargs '{"video": {"num_frames": -1}}'`).

## Troubleshooting

- Tail the server log with `tail -f .vllm.log` or `tail -f .sglang.log` (or `docker logs -f <container>`).
- The start scripts print the last 200 log lines and exit if the container dies before becoming ready.
- CUDA/arch errors at startup: confirm you're on the pinned images (the vLLM container sets `CUTE_DSL_ARCH=sm_121a` for GB10).
- Download stalls: re-run `./download.sh`; it resumes.
- Port already in use or won't start: `./stop.sh` (stops either engine), then start again.
- Slow decode right after a vLLM start (about 16 tok/s single-stream): the launch lottery drew badly and the probe's one retry wasn't enough. Run `./stop.sh && ./start.sh` again.
- Repeating, looping, or garbled responses: you are probably running SGLang with the broken unsloth target (`SGLANG_TARGET=unsloth`); use the default RadixArk target or `./stop.sh && ./start.sh` for vLLM (see [SGLang status](#sglang-status)).
- The server vanished while running: check `.memguard.log` — the memory guard force-removes the engine container when available memory stays under 4 GiB, because on the GB10 the alternative is the whole machine freezing. Find what ate the memory, then restart with the matching start script.
- Whole machine freezes or reboots around an engine start: the GB10 unified-memory failure mode. The start scripts' 100 GiB pre-flight and `memguard.sh` exist to prevent it; if it still happens, check whether something outside the repo's containers was holding memory.
- Day-1 tokenizer bug: checkpoints downloaded before 2026-08-15 shipped a `tokenizer.json` that silently truncated every prompt to 2,048 tokens. Verify with `python3 -c "import json,glob; print(json.load(open(glob.glob('.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/*/tokenizer.json')[0]))['truncation'])"`; it must print `None`. Otherwise re-run `./download.sh`.

## Repository layout

```
.
├── download.sh       # fetch checkpoint(s) into ./.cache/huggingface (retrying)
├── start.sh          # serve with vLLM (default engine)
├── start-sglang.sh   # serve with SGLang + DSpark (alternative engine)
├── stop.sh           # stop whichever engine is running
├── memguard.sh       # runtime memory guard (started by the start scripts)
├── bench.sh          # benchmark whatever is serving (works with both engines)
├── .gitignore        # excludes .cache/, logs, pid files
├── LICENSE
└── README.md
```

## Credits

- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4): quantized checkpoint and docs (YaRN long-context recipe, MTP spec decode, sampling recommendations)
- [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang): the inference engines
- [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark): the DSpark block-speculative drafter
- [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38): validated SGLang GB10 flags, and the place to go for a hardened systemd SGLang deployment
- [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000): the original repo this fork is based on
