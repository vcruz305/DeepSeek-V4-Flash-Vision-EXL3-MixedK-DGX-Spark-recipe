# DeepSeek-V4-Flash-Vision EXL3 MixedK on one NVIDIA DGX Spark

[![Follow on X](https://img.shields.io/badge/Follow-%40ViC305-black?logo=x)](https://x.com/ViC305) [![Follow on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Follow-vcruz305-yellow)](https://huggingface.co/vcruz305)

Reproducible **vLLM** recipe for **[vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK)** on a **single NVIDIA DGX Spark / GB10 (SM121)**.

This pack keeps the **full 256 routed experts** (no expert pruning) at a **mixed
2/3-bit per-layer** EXL3 trellis (six layers at 3-bit, the rest at 2-bit), with
non-routed weights kept in their official source formats. Two verified routes
serve it: **stock vLLM 0.28.0 from PyPI** plus the vllm-exl3 plugin (with the
pack's DSpark MTP layers as speculative draft, the faster route), or the same
prebuilt GB10 fork runtime the GLM-5.3-Flash recipes use. Each route needs a
small, idempotent loader patch shipped here.

> Independent community engineering. Not affiliated with or endorsed by
> DeepSeek, NVIDIA, or vLLM.

| What | Where |
|---|---|
| **Pack** | [vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK): 48 shards, ~95 GB, full 256 experts |
| **Runtime** | [vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm): prebuilt vLLM + ExLlamaV3 wheels for GB10, shared with the GLM recipes |
| **Plugin** | [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3): the EXL3 quantization plugin's canonical home (`--quantization exl3`) |
| **This repo** | loader patches, pack-config repair, serve script, memory census, and [`docs/LOADER_NOTES.md`](docs/LOADER_NOTES.md) |
| Engine | vLLM `--quantization exl3`, TP=1: stock vLLM 0.28.0 + plugin (route A, MTP spec decode) or the prebuilt fork runtime (route B) |

---

## Headline (what is verified)

### Route A: stock vLLM 0.28.0 + DSpark speculative decoding

Measured **2026-09-02** on one GB10 (~122 GiB visible unified memory),
enforce-eager, greedy 256/512-token completions, `--kv-cache-dtype fp8`.

| Item | Value |
|---|---|
| Runtime | vLLM **0.28.0** (PyPI) · exllamav3 1.4.5 · flashinfer-python 0.6.18 · torch 2.13 · vllm-exl3 >= 0.2.3 |
| Architecture | `DeepseekV4ForCausalLM` on vLLM's `deepseek_v4/nvidia` path; vision tower skipped |
| Non-routed weights | **BF16 as stored** (`non_routed_dtype_policy: "bf16_as_stored"`), no load-time requantization |
| Draft | the pack's three DSpark MTP layers (`mtp.0..2`, routed experts in their source format), `{"method":"dspark","num_speculative_tokens":3}` |
| Serve | `max-model-len 65536`, util 0.92 → GPU KV cache **462,355 tokens** (fp8 KV) |
| Decode, no spec | **11.5 tok/s** (BF16 dense path, coherent greedy output) |
| Decode, dspark3 | **22.3 tok/s** at 256 tokens, **23.6 tok/s** at 512; mean acceptance length **3.47** (188 accepted / 228 drafted) |

The same pack, the same shards, no fork: `pip install vllm==0.28.0`, the plugin,
`scripts/patch_dsv4_stock028.py`, and the config policy below.

### Route B: prebuilt fork runtime, non-routed quantized to block-FP8 at load

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

## Requirements

- One DGX Spark (GB10 / SM121), NVMe with ~100 GB free for the pack.
- Route A: `vllm==0.28.0` from PyPI, `exllamav3>=1.4.5`,
  `flashinfer-python==0.6.18` (older FlashInfer rejects this model's
  `index_topk=192`), and the [vllm-exl3](https://github.com/vcruz305/vllm-exl3)
  plugin **>= 0.2.3** (the `bf16_as_stored` policy).
- Route B: the prebuilt runtime wheels from
  [spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm)
  (vLLM fork + ExLlamaV3 + the plugin). Install per that repo / the GLM
  recipe's `install_prebuilt.sh`.
- **`nvcc` and `ninja` on PATH** when serving. FlashInfer JIT-compiles its
  kernels on this box; without `nvcc` the only usable attention backends are
  rejected at engine init and vLLM dies with "No valid attention backend found".
  `export PATH=/usr/local/cuda-13.0/bin:$PATH` and use the venv's `ninja`.

## Quick start

```bash
# 0) Download the pack (resumable, multi-stream)
hf download vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK \
  --local-dir ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK

# 1) Write the serving config for your route (idempotent; scans the shards for layer_bits)
python scripts/fix_pack_config.py ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK            # route A (default)
python scripts/fix_pack_config.py ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK --route b  # route B

# 2) Patch the runtime's DeepSeek-V4 files (idempotent; see the tables below)
python scripts/patch_dsv4_stock028.py   # route A: stock vLLM 0.28.0
python scripts/patch_dsv4_loader.py     # route B: fork runtime

# 3) Serve (route A: add the DSpark draft; leave SPEC_CONFIG unset on route B)
MODEL_DIR=~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK \
  SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":3}' \
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
  "non_routed_dtype_policy": "bf16_as_stored",
  "mtp_experts": "source",
  "mtp_experts_start_layer": 43,
  "non_routed_quantization": {
    "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
    "scale_fmt": "ue8m0", "weight_block_size": [128, 128]
  }
}
```

- `layer_bits` sizes the packed dest tensors per layer (MixedK packs fail with
  a trellis shape mismatch without it).
- `non_routed_quantization` names the runtime method the
  [vllm-exl3](https://github.com/vcruz305/vllm-exl3) plugin delegates
  non-EXL3 modules to. On route A it serves the DSpark draft's routed experts
  (`mtp.*`, layer index >= `mtp_experts_start_layer`, kept in their source
  format); on route B it also serves the dense layers, which the loader patch
  quantizes to real block-FP8 at load.
- `non_routed_dtype_policy: "bf16_as_stored"` (route A) makes the plugin hand
  dense linears to vLLM's unquantized method instead of the fp8 delegate. Set
  it to `"official_source_native"` for route B. Getting this wrong is silent:
  BF16 tensors loaded into fp8 parameters produce **empty or uniform output**
  with no error (see `docs/LOADER_NOTES.md`).

`scripts/fix_pack_config.py` writes this block (default route A;
`--route b` for the fork) after scanning the shards for the actual trellis
shapes. The pack on the Hub ships the route B config as of **2026-09-01**; run
the script once for route A.

## Why the loader patches exist

### Route A: `scripts/patch_dsv4_stock028.py` (stock vLLM 0.28.0)

Three exact-match, idempotent edits under `vllm/models/deepseek_v4/nvidia/`
(backups `*.orig`; the script verifies its own anchors and reports
`already patched` on a second run):

| Defect at serve time | Fix |
|---|---|
| `_o_proj` in `flashinfer_sparse.py` (both attention classes) calls the deep-GEMM fp8 einsum, which reads block scales a BF16 `wo_a` does not have | when `wo_a` is not `float8_e4m3fn`, run the Triton inverse-RoPE + bf16 einsum reference (`rocm_inv_rope_einsum`, device-agnostic despite its name) and the regular `wo_b` linear |
| Hash-routed layers (< `num_hash_layers`) have no `e_score_correction_bias` parameter, but the checkpoint ships one: `KeyError` in `model.py` `load_weights` | skip that tensor when no parameter exists for it |
| The DSpark draft loader (`dspark.py`) has no home for `gate.bias_vl` | skip it, as the target loader does |

Everything else in route A is stock: the vision tower is skipped by the
runtime, and the draft's routed experts load through the plugin's delegate.

### Route B: `scripts/patch_dsv4_loader.py` (fork runtime)

Makes the fork's text-only `DeepseekV4ForCausalLM` accept this pack. Each edit
is exact-match anchored and idempotent:

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
  too little free memory for vLLM's startup check, so the load fails while
  `free -g` looks fine. The serve script drops the pack's page cache with
  `posix_fadvise(DONTNEED)` before booting.
- **`nvidia-smi` memory reads `[N/A]` on GB10.** Use
  `torch.cuda.mem_get_info()`, `free -g`, and
  `nvidia-smi --query-compute-apps=pid` instead.
- `scripts/memory_census.py` prints a disk-vs-resident per-component table,
  the tool that localized every defect above. Reach for it first when a load
  OOMs or a boot dies mid-warmup.

## Limitations

- **Text only.** The pack preserves the vision tower, but no current engine
  serves it; the vision/aligner tensors are skipped at load.
- **Speculative decoding is route A only.** The fork loader skips the `mtp.*`
  layers; route B's 15.9 tok/s is plain eager decode.
- **Extreme context untested.** Verified at 65,536; the KV pool would allow
  far more, but >64k behavior on this model has not been measured.
- **Plugin required.** Upstream declined EXL3 support
  ([vllm#19896](https://github.com/vllm-project/vllm/issues/19896)); both
  routes need the vllm-exl3 plugin, and route A additionally needs the three
  `patch_dsv4_stock028.py` edits until they are upstreamed.

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
