#!/usr/bin/env bash
# Render the bench-results tree as a markdown comparison table.
#
#   ./bench-table.sh                 # every label
#   ./bench-table.sh tune-A tune-B   # only labels matching these prefixes
#
# Reads the JSONs written by `vllm bench serve` (bench.sh --save-result) and
# prints one row per (label, scenario). Speculative-decode fields are absent
# when the run was served by SGLang or with speculation disabled; those show
# as "-" rather than 0, because the two mean very different things.
set -uo pipefail
cd "$(dirname "$0")"
RESULTS=".cache/huggingface/bench-results"
[[ -d "${RESULTS}" ]] || { echo "no ${RESULTS}"; exit 1; }

python3 - "$@" <<'PY'
import glob, json, os, sys

results = ".cache/huggingface/bench-results"
prefixes = sys.argv[1:]
rows = []
for path in sorted(glob.glob(os.path.join(results, "*", "*.json"))):
    label = os.path.basename(os.path.dirname(path))
    scenario = os.path.basename(path)[:-5]
    if prefixes and not any(label.startswith(p) for p in prefixes):
        continue
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"<!-- skipped {label}/{scenario}: {exc} -->")
        continue
    failed = d.get("failed") or 0
    if failed:
        print(f"<!-- {label}/{scenario}: {failed} of {d.get('num_prompts')} requests FAILED -->")
    rows.append((label, scenario, d))

if not rows:
    print("no matching results")
    raise SystemExit(0)


def g(d, key, scale=1.0):
    v = d.get(key)
    return None if v is None else v * scale


def fmt(v, spec=".1f"):
    return "-" if v is None else format(v, spec)


# "reqs" matters: sess<c> rows produced with different SESS_REQS have different
# prefill fractions and are NOT directly comparable.
hdr = ["label", "scenario", "conc", "reqs", "req/s", "out tok/s", "tot tok/s",
       "TTFT p50 ms", "TTFT p99 ms", "ITL p50 ms", "ITL p99 ms", "E2EL p50 s",
       "acc len", "acc %"]
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for label, scenario, d in rows:
    print("| " + " | ".join([
        label,
        scenario,
        fmt(g(d, "max_concurrency"), ".0f"),
        fmt(g(d, "num_prompts"), ".0f"),
        fmt(g(d, "request_throughput"), ".4f"),
        fmt(g(d, "output_throughput"), ".2f"),
        fmt(g(d, "total_token_throughput"), ".1f"),
        fmt(g(d, "median_ttft_ms"), ".0f"),
        fmt(g(d, "p99_ttft_ms"), ".0f"),
        fmt(g(d, "median_itl_ms"), ".1f"),
        fmt(g(d, "p99_itl_ms"), ".1f"),
        fmt(g(d, "median_e2el_ms", 1e-3), ".1f"),
        fmt(g(d, "spec_decode_acceptance_length"), ".3f"),
        fmt(g(d, "spec_decode_acceptance_rate"), ".1f"),   # already a percentage
    ]) + (" |" if not (d.get("failed") or 0) else f" | **{d['failed']} FAILED** |"))
PY
