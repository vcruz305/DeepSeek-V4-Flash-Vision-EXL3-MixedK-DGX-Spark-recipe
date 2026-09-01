# DeepSeek-V4-Flash-Vision EXL3 MixedK on one NVIDIA DGX Spark

[![Follow on X](https://img.shields.io/badge/Follow-%40ViC305-black?logo=x)](https://x.com/ViC305) [![Follow on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Follow-vcruz305-yellow)](https://huggingface.co/vcruz305)

Reproducible **vLLM** recipe for **[vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK)** on a **single NVIDIA DGX Spark / GB10 (SM121)**.

This pack keeps the **full 256 routed experts** (no expert pruning) at a **mixed
2/3-bit per-layer** EXL3 trellis (six layers at 3-bit, the rest at 2-bit), with
non-routed weights kept in their official source formats. It is served by the
same prebuilt GB10 runtime as the GLM-5.3-Flash recipes — one runtime, several
packs — plus a small set of loader patches shipped here.

> Independent community engineering. Not affiliated with or endorsed by
> DeepSeek, NVIDIA, or vLLM.

| What | Where |
|---|---|
| **Pack** | [vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK) — 48 shards, ~95 GB, full 256 experts |
| **Runtime** | [vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm) — prebuilt vLLM + ExLlamaV3 wheels for GB10, shared with the GLM recipes |
| **Plugin** | [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) — the EXL3 quantization plugin's canonical home (`--quantization exl3`) |
| **This repo** | loader patches, pack-config repair, serve script, memory census, and [`docs/LOADER_NOTES.md`](docs/LOADER_NOTES.md) |
| Engine | vLLM, `--quantization exl3`, TP=1. **Stock vLLM cannot load this pack** |

---

## Headline (what is verified)

Measured **2026-09-01** on one GB10 (~122 GiB visible unified memory), enforce-eager,
no speculative decoding, greedy smoke prompt.

| Item | Value |
|---|---|
| Architecture | `DeepseekV4ForCausalLM` (text path; 43 layers, 3 hash-routed + 40 noaux_tc, 256 routed experts top-6) |
| Pack | EXL3 MCG trellis, routed experts only; **layer_bits: six 3-bit layers (3, 13, 21, 22, 28, 41), rest 2-bit** |
| Load | **86.73 GiB** CUDA allocated (83.91 GiB parameters + buffers) |
| Resident dtypes | int16 74.09 GiB (packed trellis) · float8_e4m3fn 5.86 GiB (non-routed, quantized at load) · bf16 2.83 GiB |
| Serve | `max-model-len 65536`, util 0.92 → **GPU KV cache 1,069,556 tokens** (fp8 KV) |
| Output | coherent greedy completions (verified factual smoke) |
| Decode | **15.9 tok/s** on the smoke test, **enforce-eager, no spec decode** — CUDA graphs and MTP are untested upside, not included in this number |
| MTP | the pack carries 3 MTP layers (`mtp.*`, ~10 GB, skipped at load); speculative decoding not yet configured |

The same weights load at ~91 GB under ExLlamaV3 directly; the vLLM path above
is leaner because the MTP and vision towers are skipped and non-routed weights
are held as block-FP8.

## Requirements

- One DGX Spark (GB10 / SM121), NVMe with ~100 GB free for the pack.
- The prebuilt runtime wheels from
  [spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm)
  (vLLM fork + ExLlamaV3 + the [vllm-exl3](https://github.com/vcruz305/vllm-exl3)
  plugin). Install per that repo / the GLM recipe's `install_prebuilt.sh`.
- **`nvcc` and `ninja` on PATH** when serving. FlashInfer JIT-compiles its
  kernels on this box; without `nvcc` the only usable attention backends are
  rejected at engine init and vLLM dies with "No valid attention backend found".
  `export PATH=/usr/local/cuda-13.0/bin:$PATH` and use the venv's `ninja`.

## Quick start

```bash
# 0) Download the pack (resumable, multi-stream)
hf download vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK \
  --local-dir ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK

# 1) If your download predates 2026-09-01, repair its config in place
python scripts/fix_pack_config.py ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK

# 2) Patch the runtime's DeepSeek-V4 loader (idempotent; see the table below)
python scripts/patch_dsv4_loader.py

# 3) Serve
MODEL_DIR=~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK \
  bash scripts/serve_one_spark_dsv4.sh

# 4) Smoke
curl -s http://127.0.0.1:8899/v1/completions -H 'content-type: application/json' \
  -d '{"model":"DSV4-Flash","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

## The pack-config contract

vLLM resolves the quantization method from the pack's `config.json`
**before** it looks at your CLI flag, and a leftover base-model declaration
silently overrides `--quantization exl3`. The pack's `quantization_config`
must declare:

```json
{
  "quant_method": "exl3",
  "bits": 2,
  "codebook": "mcg",
  "layer_bits": {"3": 3, "13": 3, "21": 3, "22": 3, "28": 3, "41": 3},
  "non_routed_quantization": {
    "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
    "scale_fmt": "ue8m0", "weight_block_size": [128, 128]
  }
}
```

- `layer_bits` sizes the packed dest tensors per layer (MixedK packs fail with
  a trellis shape mismatch without it).
- `non_routed_quantization` tells the [vllm-exl3](https://github.com/vcruz305/vllm-exl3)
  plugin to delegate non-routed layers to the runtime's block-FP8 method, whose
  forward path this model class is specialized for.

The pack on the Hub ships the fixed config as of **2026-09-01**.
`scripts/fix_pack_config.py` repairs older local downloads in place (it also
scans the shards and regenerates `layer_bits` from the actual trellis shapes).

## Why the loader patches exist

`scripts/patch_dsv4_loader.py` makes the runtime's text-only
`DeepseekV4ForCausalLM` accept this pack. Each edit is exact-match anchored and
idempotent:

| Defect at serve time | Fix |
|---|---|
| The strict loader raises on tensors the text class has no modules for (`vision.*`, `aligner.*`, `image_*` specials, and the `mtp.*` draft layers) | skip them at load; MTP loads separately if/when spec decode is configured |
| Layers 0–2 are **hash-routed** (`tid2eid` tables); their `gate.bias` is vestigial, and `gate.bias_vl` is unmapped in the text path | skip `gate.bias` below `num_hash_layers` and all `gate.bias_vl` |
| The pack's slimmed config omits `topk_method`, so the router's `e_score_correction_bias` parameter is never created and its checkpoint tensor has no home | default the gate branch to `noaux_tc` (the checkpoint carrying the bias tensors is the evidence) |
| Non-routed weights are **BF16 on disk**, but the model's forward path is fp8-specialized (deep-GEMM o-proj einsum, shared-expert path read `weight_scale_inv` unconditionally) | quantize them at load to real block-FP8: 128×128 blocks, `scale = amax/448`, emitting `weight` (`float8_e4m3fn`) + `weight_scale_inv` (fp32) the standard loaders already understand |

Two dead ends are documented in
[`docs/LOADER_NOTES.md`](docs/LOADER_NOTES.md) so nobody repeats them: the
~117 GiB OOM balloon caused by the leftover fp8 config declaration (wrong MoE
scaffolding), and the perfectly **uniform logits** (−ln vocab ≈ −11.77, endless
BOS) you get if BF16 weights are cast into fp8 parameters without real scales.

## Unified-memory gotchas (GB10)

- **Page cache eats CUDA-free.** After the 95 GB download (or a previous boot),
  the model files sit in page cache and `torch.cuda.mem_get_info()` reports
  too little free memory for vLLM's startup check — the load then fails while
  `free -g` looks fine. The serve script drops the pack's page cache with
  `posix_fadvise(DONTNEED)` before booting.
- **`nvidia-smi` memory reads `[N/A]` on GB10.** Use
  `torch.cuda.mem_get_info()`, `free -g`, and
  `nvidia-smi --query-compute-apps=pid` instead.
- `scripts/memory_census.py` prints a disk-vs-resident per-component table —
  the tool that localized every defect above. Reach for it first when a load
  OOMs or a boot dies mid-warmup.

## Limitations

- **Text only.** The pack preserves the vision tower, but no current engine
  serves it; the vision/aligner tensors are skipped at load.
- **No speculative decoding yet.** The MTP layers are present in the pack and
  untested under vLLM; the 15.9 tok/s headline is plain eager decode.
- **Extreme context untested.** Verified at 65,536; the KV pool would allow
  far more, but >64k behavior on this model has not been measured.
- **Not stock vLLM.** Upstream declined EXL3 support
  ([vllm#19896](https://github.com/vllm-project/vllm/issues/19896)); this
  recipe requires the fork runtime above.

## Related repositories

| Repo | What |
|---|---|
| [DSV4-Flash-Vision-ablit-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK) | the pack this recipe serves |
| [vllm-exl3](https://github.com/vcruz305/vllm-exl3) | the EXL3 plugin: source, releases, issues |
| [GLM-5.3-Flash-EXL3-K2-spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm) | prebuilt GB10 runtime wheels (shared) |
| [GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) | the sibling GLM recipe |
| [GLM-5.3-Flash-EXL3-K2K3-mix-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix-DGX-Spark-recipe) | the GLM mixed-K sibling |

## License

Apache-2.0. If this recipe or its measurements are useful to you, please
credit **vcruz305**.
