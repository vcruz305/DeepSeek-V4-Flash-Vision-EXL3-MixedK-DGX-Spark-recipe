# Route B (archived): the prebuilt fork runtime

This was the first route that served the pack, before stock vLLM 0.28.0 and the
nightly vision class. It is kept for people who already run the prebuilt GB10
fork wheels from
[GLM-5.3-Flash-EXL3-K2-spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm)
(vLLM fork + ExLlamaV3 + the vllm-exl3 plugin; install per that repo or the GLM
recipe's `install_prebuilt.sh`). It is **text only** and has **no speculative
decoding**; the recommended routes are in the [README](../README.md).

## Quick start

```bash
python scripts/fix_pack_config.py ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK --route b
python scripts/patch_dsv4_loader.py
MODEL_DIR=~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK bash scripts/serve_one_spark_dsv4.sh
```

Leave `SPEC_CONFIG` unset. The config block needs
`non_routed_dtype_policy: "official_source_native"` (what `--route b` writes).

## What is verified


Measured **2026-09-01** on one GB10, enforce-eager, no speculative decoding,
greedy smoke prompt.

| Item | Value |
|---|---|
| Architecture | `DeepseekV4ForCausalLM` (text path; 43 layers, 3 hash-routed + 40 noaux_tc, 256 routed experts top-6) |
| Pack | EXL3 MCG trellis, routed experts only; **layer_bits: six 3-bit layers (3, 13, 21, 22, 28, 41), rest 2-bit** |
| Load | **86.73 GiB** CUDA allocated (83.91 GiB parameters + buffers) |
| Resident dtypes | int16 74.09 GiB (packed trellis) · float8_e4m3fn 5.86 GiB (non-routed, quantized at load) · bf16 2.83 GiB |
| Serve | `max-model-len 65536`, util 0.92 → **GPU KV cache 1,069,556 tokens** (fp8 KV) |
| Output | coherent greedy completions (verified factual smoke) |
| Decode | **15.9 tok/s** on the smoke test, **enforce-eager, no spec decode**; CUDA graphs and MTP are untested upside, not included in this number |
| MTP | the pack carries 3 MTP layers (`mtp.*`, ~10 GB, skipped at load); speculative decoding not yet configured |

The same weights load at ~91 GB under ExLlamaV3 directly; route B is leaner
because the MTP and vision towers are skipped and non-routed weights are held as
block-FP8. Route B's fp8 dense path decodes faster than route A's BF16 dense
path without a draft (15.9 vs 11.5 tok/s); route A wins once the DSpark draft is
on. Load-time block-FP8 quantization plus the DSpark draft on stock 0.28 is the
obvious next experiment and is not measured yet.


## The loader patch: `scripts/patch_dsv4_loader.py`


Makes the fork's text-only `DeepseekV4ForCausalLM` accept this pack. Each edit
is exact-match anchored and idempotent:

| Defect at serve time | Fix |
|---|---|
| The strict loader raises on tensors the text class has no modules for (`vision.*`, `aligner.*`, `image_*` specials, and the `mtp.*` draft layers) | skip them at load; MTP loads separately if/when spec decode is configured |
| Layers 0–2 are **hash-routed** (`tid2eid` tables); their `gate.bias` is vestigial, and `gate.bias_vl` is unmapped in the text path | skip `gate.bias` below `num_hash_layers` and all `gate.bias_vl` |
| The pack's slimmed config omits `topk_method`, so the router's `e_score_correction_bias` parameter is never created and its checkpoint tensor has no home | default the gate branch to `noaux_tc` (the checkpoint carrying the bias tensors is the evidence) |
| Non-routed weights are **BF16 on disk**, but the model's forward path is fp8-specialized (deep-GEMM o-proj einsum, shared-expert path read `weight_scale_inv` unconditionally) | quantize them at load to real block-FP8: 128×128 blocks, `scale = amax/448`, emitting `weight` (`float8_e4m3fn`) + `weight_scale_inv` (fp32) the standard loaders already understand |

Two dead ends are documented in
[`LOADER_NOTES.md`](LOADER_NOTES.md) so nobody repeats them: the
~117 GiB OOM balloon caused by the leftover fp8 config declaration (wrong MoE
scaffolding), and the perfectly **uniform logits** (−ln vocab ≈ −11.77, endless
BOS) you get if BF16 weights are cast into fp8 parameters without real scales.

