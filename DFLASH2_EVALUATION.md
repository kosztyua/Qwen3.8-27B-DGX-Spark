# DFlash 2 evaluation for Qwen3.8-27B on DGX Spark

Status: preparation only, 2026-08-19. Nothing in this document has been run
against the live server.

## Recommendation

DFlash 2 is worth a controlled DGX Spark trial, but it should not replace
DSpark by default yet.

The upside is material: the authors' matched H200/SGLang evaluation reports a
mean acceptance length of 4.80 for DFlash 2 versus 3.62 for the same community
DSpark drafter used here. Across their five tasks, DFlash 2 is 27-39% faster
than DSpark at concurrency 1 and 28-36% faster at concurrency 8. Both propose
seven draft tokens per verification step.

Those percentages are evidence for a trial, not a forecast for this machine.
The published run used a BF16 target, H200, SGLang, and FlashAttention 3. This
repository uses a RadixArk NVFP4 target, GB10/SM121a, vLLM, and Triton
attention. Its current DSpark setup is also already tuned locally.

Sources:

- [DFlash 2 announcement](https://inco.ai/blog/dflash2/)
- [Qwen3.8-27B DFlash 2 model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- [DFlash reference repository](https://github.com/z-lab/dflash)
- [vLLM integration PR #52816](https://github.com/vllm-project/vllm/pull/52816)

## Like-for-like facts

| Item | Current DSpark | DFlash 2 candidate |
|---|---:|---:|
| Draft parameters | 1,359,284,737 BF16 | 1,924,404,480 BF16 |
| Weight artifact | 2.7 GB | 3.8 GB |
| Weight delta | - | about +1.1 GB |
| Draft tokens per verify | 7 | 7 |
| Published mean acceptance length | 3.62 | 4.80 |
| Published runtime | H200, SGLang, FA3 | H200, SGLang, FA3 |
| Locally validated on GB10 | yes | no |
| Supported by the pinned vLLM image | yes | no |

DFlash 2 keeps the block-parallel draft but adds a top-16 candidate path
selector and two-tap dynamic convolutions. The model card describes lossless
verification: greedy output matches the target, while sampled output preserves
the target distribution.

## Blocking risks

1. **vLLM support is not merged.** The model card currently requires open
   vLLM PR #52816. Its reviewed head during this preparation was
   `19c9351904df4c63042671bc67a866ca48dc7d6f`. The repository's normal pinned
   image predates the PR.
2. **Adjacent-architecture concurrency crash.** A same-day PR comment reports
   an out-of-bounds gather at concurrency 4 on SM120. GB10 is SM121a, so this
   must be treated as relevant until a soak test disproves it.
3. **No transferable memory number.** The checkpoint itself adds about 1.1 GB,
   but draft workspaces, captured graphs, and GDN verify state can add more.
   The prepared variant therefore lets vLLM profile the KV pool instead of
   reusing DSpark's pinned 68 GiB pool.
4. **Published target mismatch.** Upstream numbers use `Qwen/Qwen3.8-27B`
   BF16. The prepared variant deliberately keeps the current RadixArk NVFP4
   target so the local A/B changes only the drafter, but acceptance must be
   remeasured.
5. **Vision, long-context behavior, and GB10 kernels are unvalidated.** The
   candidate is limited to native 262k context. Do not assume the MTP
   variant's vision or 1M-context validation carries over.
6. **SGLang is not the first test path.** Its published DFlash 2 results are
   encouraging, but this repository's pinned SGLang image is dated before the
   release and its local concurrency ceiling is seven. The preparation keeps
   the initial comparison on vLLM.

## What this branch prepares

- `VARIANT=dflash2` in `start.sh`, using the same RadixArk NVFP4 target.
- Official `z-lab/Qwen3.8-27B-DFlash2`, method `dflash`, and seven draft tokens.
- Two explicit gates:
  `DFLASH2_EXPERIMENTAL=1` and a caller-supplied, digest-pinned
  `DFLASH2_IMAGE`.
- Conservative first-boot defaults: `MAX_SEQS=1` and an automatically profiled
  KV pool.
- The existing memory preflight, runtime memory guard, readiness handling, and
  speculative-acceptance probe.
- `./download.sh --dflash2` for the target and draft artifacts.

The normal `./start.sh` path remains DSpark and uses its original pinned image.
No Docker image, model weights, or running service was changed while preparing
this branch.

## Image prerequisite

Build and validate an aarch64/SM121-compatible vLLM image containing PR #52816
(or a later merged replacement), then pin it by digest. It also needs the GB10
features relied on by the current image; merely installing the PR over an
arbitrary vLLM release is not equivalent.

The start script intentionally refuses an unqualified default image:

```bash
DFLASH2_EXPERIMENTAL=1 DFLASH2_IMAGE=registry.example/vllm-dflash2@sha256:<digest> VARIANT=dflash2 ./start.sh
```

That command is documentation for a future maintenance window. It was not run
during preparation.

## Controlled rollout

Perform each phase in a maintenance window, with an explicit rollback point.

1. **Stage artifacts only**

   ```bash
   ./download.sh --dflash2
   ```

2. **Capture a fresh DSpark baseline**

   Restart DSpark so the prefix cache is cold, then record:

   ```bash
   ./bench.sh ab-dspark-c1 --scenarios dec1,pre16k,long128k
   ```

   Also run the repository's fixed real code/reasoning/math prompts; synthetic
   random text understates speculative acceptance.

3. **Single-stream DFlash 2**

   Start with the prepared defaults (`MAX_SEQS=1`, automatic KV sizing). Check
   model load, generated text, `vllm:spec_decode_*` metrics, memory headroom,
   and a minimum 100-request soak before benchmarking the same scenarios.

4. **Concurrency staircase**

   Only after stage 3 passes, test `MAX_SEQS=4`, then 8, then 12. Restart
   between arms, keep `KV_CACHE_MEMORY=` unset/empty initially, and use one
   session scenario per invocation with identical controls:

   ```bash
   SESS_PREFIXES=12 SESS_REQS=3 SESS_SEED=11908      ./bench.sh ab-dflash2-c8 --scenarios sess8
   ```

   Repeat for DSpark and DFlash 2 with cold prefix caches. Do not combine arms
   in one invocation.

5. **Capacity and long-context soak**

   Record the profiled KV token capacity, peak available memory, preemptions,
   and acceptance at 32k, 128k, and near the required production limit.
   Long-context acceptance is a go/no-go metric, not an optional follow-up.

## Promotion gates

Promote DFlash 2 to the default only if all of these pass on GB10:

- Greedy outputs match the target on a fixed correctness suite; sampled quality
  is non-inferior on the same seeds and sampling parameters.
- No CUDA assertion, invalid draft token, server exit, preemption anomaly, or
  memory-guard trip during the concurrency-12 soak.
- At least 10% higher real-workload output throughput than DSpark at both
  concurrency 1 and the production concurrency, with acceptance and step time
  reported separately.
- No more than 5% prefill/TTFT regression.
- Required 128k+ context capacity remains available with safe host memory
  headroom.
- Vision is either validated or explicitly routed to the MTP variant.

If any gate fails, rollback is simply the unchanged default:

```bash
./start.sh
```
