# Qwen3.8 27B for DGX Spark / RTX 6000 PRO

Opinionated, ready-to-run scripts to serve **[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)** with **vLLM** in Docker on an NVIDIA DGX Spark (GB10, aarch64). One script downloads the weights, one starts an OpenAI-compatible server, one stops it.

- **NVFP4** quantized checkpoint (compressed-tensors) — 4-bit weights/activations with 8-bit groups for sensitive modules
- **1M-token context** via YaRN (factor 4.0 over the native 262,144)
- **FP8 KV cache** (applied automatically from the checkpoint's calibrated `kv_cache_scheme`, ~2× KV memory savings)
- **MTP speculative decoding** (`num_speculative_tokens: 2`) — faster decode
- Built-in **reasoning parsing** (Qwen3 thinking mode) and **tool calling** (`qwen3_coder` parser)

---

## Requirements

| Component | Detail |
|---|---|
| Hardware | NVIDIA DGX Spark / GB10 (aarch64, sm_121a) |
| Docker | With NVIDIA Container Toolkit / GPU passthrough working (`docker run --gpus all`) |
| vLLM image | `vllm/vllm-openai:nightly-aarch64` (needs vLLM ≥ 0.25.0 for this checkpoint) |
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

Runtime artifacts: `.vllm.log` (server log), `.vllm.pid` (container ID), `.cache/` (HF + Triton caches). All are git-ignored.

## Configuration

Defaults live at the top of `start.sh`:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_ID` | `unsloth/Qwen3.8-27B-NVFP4` | Checkpoint to serve |
| `SERVED_MODEL_NAME` | `qwen38-27b-unsloth-nvfp4` | Name clients use in API requests |
| `IMAGE` | `vllm/vllm-openai:nightly-aarch64` | Pin to a specific tag for reproducibility |
| `CONTAINER_NAME` | `qwen3.8-27b-nvfp4` | Also used by `stop.sh` |
| `PORT` | `8888` | Listens on `0.0.0.0` via host networking |
| `--gpu-memory-utilization` | `0.84` | |
| `--max-model-len` | `1,000,000` | YaRN-extended; see notes below |
| `--max-num-seqs` | `4` | Concurrent sequences |

### RTX 6000 PRO: lower `--gpu-memory-utilization`

The `0.84` default is tuned for **DGX Spark** (128 GB unified memory). The **RTX 6000 PRO** has 96 GB of dedicated VRAM that also feeds the display server and any other GPU processes — unlike the Spark, there is no unified-memory cushion. At `0.84` vLLM reserves ~81 GB and startup can OOM (usually during CUDA-graph capture) or starve the desktop.

On RTX 6000 PRO, reduce it in `start.sh`:

- headless: `--gpu-memory-utilization 0.80` is a good starting point
- with a display attached / other GPU processes: `0.75` or lower

Each 0.01 ≈ ~1 GB of KV cache; the engine logs the resulting `GPU KV cache size` at startup, so re-check the concurrency numbers above after changing it.

### Notable serving choices

- **Context:** 1M tokens via static YaRN (factor 4.0 over native 262,144, applied through `--hf-overrides`). Per the model card, static YaRN can slightly impact short-context quality. With the FP8 KV cache, a single 1M-token sequence needs ~32 GB of KV; concurrent long sequences are admitted as KV space allows. Override `--kv-cache-dtype bfloat16` in `start.sh` to force a bf16 KV cache instead of the checkpoint's FP8 scheme.
- **Quantization:** the checkpoint is dynamic NVFP4 in compressed-tensors format, hence `--quantization compressed-tensors`. Dense model (SwiGLU, no experts) — no `--moe-backend` needed.
- **Speculative decoding:** no DFlash drafter exists for Qwen3.8; unsloth ships MTP weights (`model_mtp.safetensors`), used via `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'`. Faster decode, somewhat lower peak throughput.
- **Multimodal:** video support enabled via `--media-io-kwargs '{"video": {"num_frames": -1}}`.

## Attention backend note

`triton_attn` is required for the FP8 KV cache: FlashAttention-2 cannot serve FP8 KV on GB10/SM121 (vLLM requires FA3 on SM90 or FA4 on SM100). Only the 16 full-attention layers use this backend — the other 48 layers are Gated DeltaNet (unaffected), and the vision tower still runs FA. Reverting to `--attention-backend flash_attn` requires `--kv-cache-dtype bfloat16`.

## Measured numbers (this box)

vLLM `v0.27.2rc1.dev77+gac7509e2b` (`vllm/vllm-openai:nightly-aarch64`), 1 × DGX Spark GB10, `--gpu-memory-utilization 0.84`, MTP speculative decoding on, YaRN 1M context. From the engine startup log:

| | bf16 KV cache | FP8 KV cache (default) |
|---|---|---|
| Available KV cache memory | 74.88 GiB | 75.89 GiB |
| GPU KV cache size | 1,143,423 tokens | **2,295,133 tokens** |
| Max concurrency @ 1M-context requests | 1.14× | **2.30×** |
| Max concurrency @ 262k-context (memory) | 4.4× | ~8.7× |

Practical concurrency:

- Hard cap from config: **4 concurrent sequences** (`--max-num-seqs`); extra requests queue.
- Memory-wise with FP8 KV: 2 × 1M-context, ~4 × 512k, ~8 × 262k, 4 × ≤128k (config-capped).
- Checkpoint size: 22.6 GB (`model.safetensors` + `model_mtp.safetensors`).
- Live KV usage: `curl -s localhost:8888/metrics | grep kv_cache_usage_perc`.

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

## Logs & troubleshooting

- Tail the server log: `tail -f .vllm.log` (or `docker logs -f qwen3.8-27b-nvfp4`)
- `start.sh` prints the last 200 log lines and exits if the container dies before becoming ready
- If startup fails with CUDA/arch errors, confirm you're on the `nightly-aarch64` image (the container sets `CUTE_DSL_ARCH=sm_121a` for GB10's cutlass kernels)
- Model download stalls: just re-run `./download.sh`; it resumes

## Repository layout

```
.
├── download.sh   # fetch checkpoint into ./.cache/huggingface (retrying)
├── start.sh      # launch vLLM container, wait for readiness
├── stop.sh       # stop the container, clean up pid file
├── .gitignore    # excludes .cache/, logs, pid file
└── README.md
```

## Credits

- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — quantized checkpoint and docs (YaRN long-context recipe, MTP spec decode, sampling recommendations)
- [vLLM](https://github.com/vllm-project/vllm) — inference engine and OpenAI-compatible server
