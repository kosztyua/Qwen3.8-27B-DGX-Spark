# Backlog

Repository-local follow-up work lives here. Items are triggered by relevant
dependency or artifact changes, not by calendar reminders.

## Open

- [ ] **Re-evaluate DFlash 2 for the default Qwen3.8-27B deployment.**

  The initial c=12 evaluation found excellent decode acceptance but a net loss
  on the cache-heavy production shape. Keep DSpark as the default until a
  changed DFlash/vLLM stack passes the promotion gates in
  [DFLASH2_EVALUATION.md](DFLASH2_EVALUATION.md#retest-policy).

  Re-open testing when at least one baseline anchor changes:

  - Hugging Face model revision differs from
    `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`;
  - vLLM DFlash integration PR #52816 differs from
    `19c9351904df4c63042671bc67a866ca48dc7d6f` or merges;
  - lookahead-hashing PR #50897 differs from
    `2b7eaf105a364ee2a9873cde24049b8bb40dd635` or merges;
  - hybrid GDN cache PR #52244 differs from
    `62cbf34259002207e237eec5b5af79f75cc1606c` or merges; or
  - RadixArk NVFP4 works without the local LM-head compatibility patch.

  Definition of done: repeat the pinned c=12 p50 and all-90k A/B matrix,
  capture cache and speculative counters, run the 128k/correctness/quality/
  vision gates, record the new immutable revisions and results, decide whether
  to promote, and restore the DSpark default after testing.

