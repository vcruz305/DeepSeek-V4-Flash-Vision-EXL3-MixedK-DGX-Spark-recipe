# Receipts

JSON outputs of the measurements quoted in [`docs/QUALITY.md`](../docs/QUALITY.md) and
[`docs/KALIBRATED.md`](../docs/KALIBRATED.md), as written by the scripts of the sibling recipe
[DeepSeek-V4-Flash-Vision-One-DGX-Spark](https://github.com/GaelicThunder/DeepSeek-V4-Flash-Vision-One-DGX-Spark)
(`scripts/nll_probe.py`, `scripts/bench/bench_all.py`, `scripts/ppl_probe.py`). Names here say what was
measured; the source repository keeps the same files under their build-time names (`dsvision` = MixedK on
the sparkinfer route, `aplus22` = Kalibrated's working name).

| file | engine | date | contents | name in the source repository |
|---|---|---|---|---|
| `nll-reference-fp8-20260906.json` | the abliterated Vision-Exp source at release precision (FP8 attention, MXFP4 experts), vLLM 0.28.1 nightly, tensor-parallel 2, 2×H200 | 2026-09-06 | 64,859 per-token log-probs in 23 chunks, per-source means | `receipts/nll/nll-ref-source-fp8-20260906.json` |
| `nll-mixedk-20260905.json` | MixedK, sparkinfer route | 2026-09-05 | same tokens | `receipts/nll/nll-dsvision-20260905.json` |
| `nll-kalibrated-20260907.json` | Kalibrated Vision Exp, sparkinfer route | 2026-09-07 | same tokens | `receipts/nll/nll-aplus22-20260907.json` |
| `bench-mixedk-20260903.json` | MixedK | 2026-09-03 | MMLU-Pro 251 items × 2 option orders with per-item answers and margins, MATH-500 level 5 (60 items), needle at 9 positions | `receipts/bench-dsvision-20260903.json` |
| `bench-mixedk-2nd-boot-20260903.json` | MixedK, second boot | 2026-09-03 | the same suite: the harness's run-to-run noise | `receipts/bench-dsvision-back-20260903.json` |
| `bench-kalibrated-20260907.json` | Kalibrated | 2026-09-07 | MMLU-Pro, two option orders | `receipts/kalibrated/bench-aplus22-20260907.json` |
| `ppl-mixedk-20260903.json` | MixedK | 2026-09-03 | 8 fixed passages, NLL and perplexity each | `receipts/ppl-dsvision-20260903.json` |
| `ppl-kalibrated-20260907.json` | Kalibrated | 2026-09-07 | same passages | `receipts/kalibrated/ppl-aplus22-20260907.json` |

Paired comparison of any two `nll-*.json` files (delta B − A with token-level and chunk-level standard
errors), with `nll_probe.py` from the sibling recipe:

```bash
python3 DeepSeek-V4-Flash-Vision-One-DGX-Spark/scripts/nll_probe.py --compare \
    receipts/nll-reference-fp8-20260906.json receipts/nll-mixedk-20260905.json
```

The `base` field in each file is the local endpoint the run used; `sec` is wall time; `chunks[*].logprobs`
are the per-token values that make the comparison paired.
