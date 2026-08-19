# DFlash 2 evaluation for Qwen3.8-27B on DGX Spark

Status: live evaluation completed on 2026-08-19. The unchanged DSpark default
was restored after testing and remains the running configuration.

## Recommendation

Do not switch the repository default to DFlash 2 yet.

DFlash 2 is a strong decode accelerator on this box: the local candidate was
63.8% faster than DSpark at concurrency 1 and 115.3% faster at concurrency 4
on matched 1k-prompt/1k-generation tests. It also completed c=12 tests without
a request failure, preemption, CUDA assertion, or engine reset.

It does not produce a net gain on the actual cache-heavy workload. At the
~35k p50 shape, DFlash 2's higher acceptance was offset by 52.1% more computed
input tokens and the run took 8.7% longer. In an intentionally severe all-90k
stress arm, it computed 40.6% more input tokens and took 38.8% longer. Cold
warmup time was better at 35k and essentially tied at 90k, so the evidence
points to lost effective prefix-cache reuse rather than a correspondingly slow
raw prefill kernel.

The candidate also needs an unmerged vLLM PR, a local NVFP4 compatibility
patch, and concurrency-specific AOT cache isolation. Keep it as a fail-closed,
decode-heavy experiment until upstream hybrid speculative-cache work lands and
the production-shaped cache A/B passes.

Upstream references:

- [DFlash 2 announcement](https://inco.ai/blog/dflash2/)
- [Qwen3.8-27B DFlash 2 model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- [DFlash 2 model collection](https://huggingface.co/collections/z-lab/dflash-2)
- [DFlash reference repository](https://github.com/z-lab/dflash)
- [vLLM integration PR #52816](https://github.com/vllm-project/vllm/pull/52816)
- [lookahead-aware prefix hashing PR #50897](https://github.com/vllm-project/vllm/pull/50897)
- [GDN/MTP prefix-cache PR #52244](https://github.com/vllm-project/vllm/pull/52244)
- [warmed-cache DFlash/DSpark issue #47930](https://github.com/vllm-project/vllm/issues/47930)

As checked on 2026-08-19, DFlash integration #52816 remained open at the tested
head. The two directly relevant cache PRs were also open and marked as needing
a rebase. This evaluation therefore tested the active upstream DFlash head; it
did not freeze vLLM at an older support state, but the pending fixes were not
available to include.

## Retest policy

Retest DFlash 2, but do not spend another multi-hour run on identical artifacts.
The first evaluation happened less than 24 hours after the checkpoint appeared,
so rapid follow-up changes are plausible. At the close of this evaluation the
reproducibility anchors were:

- DFlash model revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, last
  modified 2026-08-19 02:52 UTC;
- integration PR #52816 head `19c9351904df4c63042671bc67a866ca48dc7d6f`;
- lookahead-hashing PR #50897 head
  `2b7eaf105a364ee2a9873cde24049b8bb40dd635`; and
- hybrid GDN cache PR #52244 head
  `62cbf34259002207e237eec5b5af79f75cc1606c`.

Review those anchors on or after **2026-08-26**, and again by **2026-09-02**.
Run the full A/B as soon as any of the following occurs:

- the DFlash checkpoint revision changes;
- #52816 changes head, merges, or enters an official aarch64 vLLM image;
- #50897 or #52244 merges, is rebased into #52816, or equivalent cache work
  lands on the candidate build; or
- upstream supports the RadixArk NVFP4 target without the local LM-head patch.

If none of the artifacts changed, record the status check and defer the GPU
test; a bit-identical rerun will mostly measure run-to-run noise. Do not follow
an unpinned moving tag: record the candidate image digest, vLLM commit, PR head,
and model revision before every retest.

The minimum retest is the same c=12 matrix used here so results remain paired:

```bash
# Run each arm after a fresh server start; use the same seed for both variants.
SESS_REQS=4 SESS_OUTPUT_LEN=850 SESS_PREFIX_LEN=32768 \
SESS_SUFFIX_LEN=3072 SESS_PREFIXES=12 SESS_SEED=12358 \
./bench.sh <variant>-p50-c12-<date> --scenarios sess12

# Deliberately severe cache/prefill stress arm, not a production distribution.
SESS_REQS=2 SESS_OUTPUT_LEN=64 SESS_PREFIX_LEN=86016 \
SESS_SUFFIX_LEN=4096 SESS_PREFIXES=12 SESS_SEED=12890 \
./bench.sh <variant>-p95-c12-<date> --scenarios sess12
```

Capture metric boundaries, zero failures/preemptions, effective prefix-cache
hit rate, newly computed input tokens, TTFT, TPOT, E2EL, throughput, acceptance
length, and warmed-cache position-0 acceptance. Before changing the default,
require all of the following:

- DFlash cache hit rate is no more than 5 percentage points below DSpark and
  computed input tokens are no more than 5% higher in both cache arms;
- p50-shaped wall time improves by at least 5%, while the all-90k stress arm is
  no more than 5% slower;
- warmed prefix reuse does not collapse draft acceptance;
- the 128k capacity, deterministic correctness, sampled-quality, and vision
  checks pass; and
- the quantized-target path no longer depends on an unreviewed local patch.

Restore and health-check the pinned DSpark default after every experimental
run, regardless of outcome.

## Production-shaped c=12 cache A/B

The deployment workload is 12 continuously active security-benchmark streams,
with context p50 ~35k, p95 ~90k, max 128k, substantial context caching, and
roughly 850 output tokens. That profile is supplied by the operator. The local
`.vllm.log` is overwritten on every engine launch, so it could not reconstruct
the preceding 24-hour distribution after the fact; no OpenWebUI traffic was
used as a proxy.

Both variants used identical seeds, content, sampling, target checkpoint,
seven-token speculation width, and c=12 scheduling. Each arm started with a
fresh server. Counter snapshots before and after every arm preserve actual
prefix-cache and speculative-decoding deltas.

The p50-like arm used 12 distinct 32,768-token prefixes plus 3,072-token
suffixes (35,840 input), four requests per stream, and 850 requested output
tokens: 48 measured requests per variant.

| p50-like metric | DSpark | DFlash 2 | DFlash 2 delta |
|---|---:|---:|---:|
| Wall time | 964.2 s | 1,048.2 s | +8.7% |
| Total token throughput | 1,826.5 tok/s | 1,680.2 tok/s | -8.0% |
| Mean / median TTFT | 39.3 / 11.5 s | 60.9 / 28.4 s | +55.0% / +146.9% |
| Mean TPOT | 223.7 ms | 225.3 ms | +0.7% |
| Median E2EL | 173.3 s | 192.1 s | +10.8% |
| Acceptance length | 2.62 | 4.11 | +56.9% |
| Effective prefix-cache hit rate | 60.9% | 40.5% | -20.4 pp |
| Newly computed input tokens | 686,608 | 1,044,224 | +52.1% |
| Single-request cold warmup | 108.7 s | 89.5 s | -17.6% |

The p95 arm deliberately put **all** 12 streams at 90,112 input tokens (86,016
prefix + 4,096 suffix), two requests per stream, and 64 output tokens: 24
measured requests per variant. It is a cache/prefill stress test, not a model of
the real distribution—p95 means only about 5% of production requests reach
this length.

| all-90k stress metric | DSpark | DFlash 2 | DFlash 2 delta |
|---|---:|---:|---:|
| Wall time | 1,957.3 s | 2,715.8 s | +38.8% |
| Total token throughput | 1,105.7 tok/s | 796.9 tok/s | -27.9% |
| Mean / median TTFT | 645.0 / 657.4 s | 906.9 / 1,031.5 s | +40.6% / +56.9% |
| Mean TPOT | 4,164.5 ms | 3,971.5 ms | -4.6% |
| Median E2EL | 976.6 s | 1,258.3 s | +28.8% |
| Acceptance length | 1.70 | 2.00 | +17.6% |
| Effective prefix-cache hit rate | 28.9% | 0.0% | -28.9 pp |
| Newly computed input tokens | 1,601,840 | 2,252,800 | +40.6% |
| Single-request cold warmup | 116.5 s | 114.1 s | -2.0% |

Both variants completed every request with zero failures and zero preemptions.
DFlash 2's better acceptance and flat-to-better TPOT show why its generated
output is attractive. The lower hit rate causes enough extra prefill to erase
that advantage. The 35k result also proves DFlash caching is not globally
broken; its effectiveness is simply materially lower in this matched run and
collapses under the all-90k pressure arm.

Raw JSON, logs, and metric boundaries are under
`.cache/huggingface/bench-results/live-cache-*-20260819/`. They are ignored
runtime artifacts rather than source-controlled fixtures.

This is consistent with, but does not uniquely prove, the outstanding hybrid
speculative-cache issues upstream. PR #50897 adds successor/lookahead tokens to
cache hashes for EAGLE-style methods including DFlash and DSpark; PR #52244
addresses fine-grained GDN/MTP cache matching; issue #47930 covers a separate
warmed-cache draft-KV correctness failure. The catastrophic acceptance collapse
from #47930 did not reproduce here.

The candidate already resolves its hybrid cache groups to a 16-token matching
unit (the GCD of its 1,648-token attention block and 16-token Mamba alignment),
so explicitly setting `PREFIX_MATCH_UNIT=16` would be a no-op. `start.sh` now
exposes the knob for retesting future vLLM builds, but changing it cannot replace
the pending successor-aware hashing work.

## Local A/B results

The random datasets, seeds, sampling settings, target checkpoint, context
limit, attention backend, and seven-token speculation width were matched. The
candidate deployment necessarily used a different vLLM build: DSpark used the
repository's pinned `b5c860acda75...` image, while DFlash 2 used PR #52816
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
- The two c=12 DFlash 2 session arms completed 72 measured requests with zero
  failures and zero preemptions. A longer production-duration soak, sampled
  quality suite, and vision test remain required before any promotion attempt.

## Memory and capacity

| Item | DSpark default | DFlash 2 c=1 | DFlash 2 c=4 | DFlash 2 c=12 |
|---|---:|---:|---:|---:|
| Loaded target + drafter | 22.73 GiB | 23.75 GiB | 23.75 GiB | 23.75 GiB |
| KV cache | 68.00 GiB pinned | 74.58 GiB profiled | 73.52 GiB profiled | 73.47 GiB profiled |
| KV token capacity | 1,274,196 | 1,674,122 | 1,650,131 | 1,649,072 |
| Max full-262k concurrency from KV | 4.86x | 6.39x | 6.29x | 6.29x |
| Peak activation during profile | not profiled | 1.60 GiB | 2.92 GiB | 2.92 GiB |
| Captured graph memory | 0.96 GiB | 0.37 GiB | 0.57 GiB | 0.79 GiB |

DFlash 2's checkpoint is 3.58 GiB and the DSpark checkpoint is 2.53 GiB. The
candidate's automatically profiled KV pool was intentionally not compared to
DSpark's 68 GiB pin as if it were a drafter efficiency result; it simply shows
that required 128k capacity remains available.

## Required vLLM and NVFP4 adaptation

PR #52816 was still open at tested head
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
- optional, validated `PREFIX_MATCH_UNIT` plumbing for future cache-fix tests;
- `./download.sh --dflash2` for pinned artifact staging; and
- the exact source compatibility patch and overlay Dockerfile.

The normal `./start.sh` path is unchanged and remains DSpark.

## Promotion-gate result

| Gate | Result |
|---|---|
| Fixed greedy correctness smoke test | Pass, limited to three prompts |
| c=1 and c=4 stability | Pass |
| c=12 stability | Pass across 72 measured requests |
| c=12 production-shaped performance | **Fail** |
| >=10% decode throughput gain | Pass at c=1 and c=4 |
| <=5% prefill/TTFT regression | **Fail** |
| 128k capacity | Pass |
| 128k end-to-end performance | **Fail** |
| Prefix-cache effectiveness | **Fail versus DSpark at 35k and 90k** |
| Sampled quality and vision | Not run |
| Upstream, quantized-target support | **Fail; local patch required** |

Rollback remains the unchanged default and was performed after the test:

```bash
./stop.sh
./start.sh
```
