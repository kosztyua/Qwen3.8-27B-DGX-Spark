# DFlash 2 evaluation for Qwen3.8-27B on DGX Spark

Status: live evaluation completed on 2026-08-19. The unchanged DSpark default
was restored after testing and remains the running configuration.

## Recommendation

Do not switch the repository default to DFlash 2 yet.

DFlash 2 is a strong decode accelerator on this box: the local candidate was
63.8% faster than DSpark at concurrency 1 and 115.3% faster at concurrency 4
on matched 1k-prompt/1k-generation tests. It also completed clean c=1 and c=4
100-request soaks, and the previously reported SM120 concurrency crash did not
reproduce on GB10/SM121a.

The candidate nevertheless fails the existing promotion gates. A matched 16k
prefill test was 9.6% slower and mean TTFT regressed 44.0%. At 128k, total token
throughput was 36.5% lower and mean TTFT was 97.0% higher. It also needs an
unmerged vLLM PR, a local NVFP4 compatibility patch, and concurrency-specific
AOT cache isolation. Keep it as a fail-closed, decode-heavy experiment until
upstream support and prefill behavior improve.

Upstream references:

- [DFlash 2 announcement](https://inco.ai/blog/dflash2/)
- [Qwen3.8-27B DFlash 2 model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- [DFlash 2 model collection](https://huggingface.co/collections/z-lab/dflash-2)
- [DFlash reference repository](https://github.com/z-lab/dflash)
- [vLLM integration PR #52816](https://github.com/vllm-project/vllm/pull/52816)

## Local A/B results

The random datasets, seeds, sampling settings, target checkpoint, context
limit, attention backend, and seven-token speculation width were matched. The
candidate deployment necessarily used a different vLLM build: DSpark used the
repository's pinned `acb0f1dcdb66...` image, while DFlash 2 used PR #52816
rebased onto the 2026-08-19 aarch64 nightly commit `5a4c8d99242e...`. These are
deployment A/B numbers, not a drafter-only microbenchmark.

| Workload | Metric | DSpark | DFlash 2 | DFlash 2 delta |
|---|---|---:|---:|---:|
| 1k input / 1k output, c=1 | output tok/s | 21.92 | 35.90 | +63.8% |
| 1k input / 1k output, c=1 | acceptance length | 2.41 | 4.03 | +67.6% |
| 1k input / 1k output, c=1 | mean TPOT | 45.19 ms | 27.36 ms | -39.5% |
| 1k input / 1k output, c=4 | output tok/s | 66.07 | 142.27 | +115.3% |
| 1k input / 1k output, c=4 | acceptance length | 2.75 | 5.46 | +99.1% |
| 1k input / 1k output, c=4 | mean TPOT | 49.73 ms | 25.29 ms | -49.1% |
| 16k input / 16 output, c=4 | total tok/s | 1,818.58 | 1,643.45 | -9.6% |
| 16k input / 16 output, c=4 | mean TTFT | 18.50 s | 26.63 s | +44.0% |
| 128k input / 256 output, c=1 | total tok/s | 911.53 | 579.30 | -36.5% |
| 128k input / 256 output, c=1 | mean TTFT | 107.87 s | 212.47 s | +97.0% |
| 128k input / 256 output, c=1 | mean TPOT | 141.98 ms | 55.81 ms | -60.7% |

All listed requests completed without client failures. The raw JSON and logs
are under `.cache/huggingface/bench-results/live-ab-*20260819/` and are ignored
runtime artifacts rather than source-controlled fixtures.

The 128k harness sends an unreported probe before its two measured requests.
DSpark reported prefix-cache reuse on the repeated probe/sample while the
DFlash 2 candidate did not, despite prefix caching being enabled on both. The
raw result therefore describes the candidate stacks as operated, but it should
not be treated as a pure uncached-prefill kernel comparison. The 16k test was
also slower and independently fails the <=5% TTFT/prefill gate.

## Stability and correctness observations

- Two 100-request soaks completed with 0 failures: c=1 produced 6,400 tokens
  at 27.57 tok/s; c=4 produced 6,400 tokens at 82.80 tok/s.
- A sustained c=4 test with four 12k-token prompts and 512-token generations
  completed 4/4 with no CUDA assertion or engine reset.
- The matched c=4 decode and 16k-prefill tests completed 20/20 requests. The
  adjacent-SM120 out-of-bounds report was not reproduced on SM121a.
- Three deterministic greedy checks produced byte-identical DSpark and DFlash
  2 outputs: two arithmetic answers and a 91-token Python implementation.
  This is a smoke test of lossless verification, not a quality evaluation.
- No c=8/c=12 session soak, sampled-quality suite, or vision test was run.
  These remain required before any future promotion attempt.

## Memory and capacity

| Item | DSpark default | DFlash 2 c=1 | DFlash 2 c=4 |
|---|---:|---:|---:|
| Loaded target + drafter | 22.73 GiB | 23.75 GiB | 23.75 GiB |
| KV cache | 68.00 GiB pinned | 74.58 GiB profiled | 73.52 GiB profiled |
| KV token capacity | 1,274,196 | 1,674,122 | 1,650,131 |
| Max full-262k concurrency from KV | 4.86x | 6.39x | 6.29x |
| Peak activation during profile | not profiled | 1.60 GiB | 2.92 GiB |
| Captured graph memory | 0.96 GiB | 0.37 GiB | 0.57 GiB |

DFlash 2's checkpoint is 3.58 GiB and the DSpark checkpoint is 2.53 GiB. The
candidate's automatically profiled KV pool was intentionally not compared to
DSpark's 68 GiB pin as if it were a drafter efficiency result; it simply shows
that required 128k capacity remains available.

## Required vLLM and NVFP4 adaptation

PR #52816 remained open at tested head
`19c9351904df4c63042671bc67a866ca48dc7d6f`. It cherry-picked cleanly onto the
latest available aarch64 nightly source/image commit
`5a4c8d99242e9e069b604d0e9b969e77f7dd501d`.

The exact PR could not boot the RadixArk target. Its DFlash 2 selector requires
an `UnquantizedEmbeddingMethod`, while RadixArk quantizes `lm_head` with
`ModelOptNvFp4LinearMethod`. The DFlash checkpoint contains no independent
`lm_head` weights and the reference implementation applies the target head, so
the local compatibility patch permits any target head quantization method with
a callable `apply()` implementation. Verification and sampling logic are not
changed.

That adaptation is checked in as
`patches/vllm-dflash2-nvfp4-lm-head.patch`. It is locally exercised, not
upstream-reviewed, and is one reason the variant remains experimental.

Reproduce the tested source overlay as follows:

```bash
git clone https://github.com/vllm-project/vllm.git /path/to/vllm-dflash2
cd /path/to/vllm-dflash2
git checkout 5a4c8d99242e9e069b604d0e9b969e77f7dd501d
git fetch https://github.com/vllm-project/vllm.git refs/pull/52816/head
git cherry-pick 19c9351904df4c63042671bc67a866ca48dc7d6f
git apply /path/to/Qwen3.8-27B-DGX-Spark-RTX-6000/patches/vllm-dflash2-nvfp4-lm-head.patch

docker build \
  -f /path/to/Qwen3.8-27B-DGX-Spark-RTX-6000/Dockerfile.dflash2 \
  -t local/vllm-dflash2:pr-52816-nvfp4 .
```

`Dockerfile.dflash2` pins the exact official aarch64 base digest and overlays
only the PR's Python/Triton runtime files, preserving its compiled SM121a
dependencies. Supply the built image by immutable ID or registry digest; tags
are rejected:

```bash
DFLASH2_EXPERIMENTAL=1 \
DFLASH2_IMAGE=sha256:<local-image-id> \
VARIANT=dflash2 \
./start.sh
```

The tested DFlash checkpoint is also pinned in the scripts at Hugging Face
revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.

## Compile-cache isolation

The first c=4 boot reused a DFlash selector AOT artifact compiled during c=1
profiling and failed before readiness:

```text
AssertionError: expected size 4==1 ... at dim=0
```

A clean cache compiled and served c=4 successfully. `start.sh` therefore sets
a DFlash-only `VLLM_CACHE_ROOT` keyed by `MAX_SEQS`. DSpark and MTP retain the
existing cache path and behavior. This avoids cross-concurrency reuse without
deleting either cache.

## Prepared, fail-closed path

This branch provides:

- `VARIANT=dflash2` over the same RadixArk NVFP4 target;
- official `z-lab/Qwen3.8-27B-DFlash2`, pinned revision, `method=dflash`, k=7;
- gates requiring `DFLASH2_EXPERIMENTAL=1` and an immutable DFlash image;
- conservative `MAX_SEQS=1` and automatically profiled KV cache defaults;
- concurrency-keyed DFlash compile caches;
- `./download.sh --dflash2` for pinned artifact staging; and
- the exact source compatibility patch and overlay Dockerfile.

The normal `./start.sh` path is unchanged and remains DSpark.

## Promotion-gate result

| Gate | Result |
|---|---|
| Fixed greedy correctness smoke test | Pass, limited to three prompts |
| c=1 and c=4 stability | Pass |
| c=12 production soak | Not run |
| >=10% decode throughput gain | Pass at c=1 and c=4 |
| <=5% prefill/TTFT regression | **Fail** |
| 128k capacity | Pass |
| 128k end-to-end performance | **Fail** |
| Sampled quality and vision | Not run |
| Upstream, quantized-target support | **Fail; local patch required** |

Rollback remains the unchanged default and was performed after the test:

```bash
./stop.sh
./start.sh
```
