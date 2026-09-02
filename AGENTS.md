# Agent notes for this recipe

## What this is
Serving recipe for DeepSeek-V4-Flash-Vision EXL3 MixedK (full 256 experts) on
one NVIDIA DGX Spark (GB10) with the vllm-exl3 plugin on either stock vLLM
0.28.0 (route A, DSpark MTP spec decode, `scripts/patch_dsv4_stock028.py`) or
the prebuilt vLLM fork runtime (route B, `scripts/patch_dsv4_loader.py`).
Serving-side only: never document how the pack was produced.

## Ground rules
- Commits authored by the repo owner only; no AI/tool attribution anywhere.
- No tokens/credentials in the tree; grep diffs before committing.
- Measured claims only. Every number in the README has a receipt; do not add
  aspirational numbers.

## The load-bearing facts
- The pack's non-routed weights are BF16 on disk with NO scale tensors; the
  nvidia forward path is fp8-specialized. Route B (`patch_dsv4_loader.py`)
  quantizes them to real 128x128 block-FP8 at load. Route A keeps them BF16:
  the config's `non_routed_dtype_policy: "bf16_as_stored"` makes the plugin
  (>= 0.2.3) use vLLM's unquantized linear method for dense layers, and
  `patch_dsv4_stock028.py` adds the bf16 `_o_proj` branch. The two policies
  are not interchangeable: BF16 tensors under the fp8 delegate load without
  error and produce empty text (route A) or uniform logits (route B). Do not "simplify" this to
  a dtype cast — casting without computed scales produces uniform logits
  (every token at -ln(vocab), endless BOS).
- Layers 0-2 are hash-routed (`tid2eid` is real); their `gate.bias` and every
  `gate.bias_vl` are vestigial and must be skipped, along with
  `vision./aligner./mtp./image_` tensors (text-only class).
- The pack config MUST declare exl3 (`scripts/fix_pack_config.py` repairs old
  downloads). A leftover fp8 declaration silently overrides `--quantization
  exl3` and balloons memory with wrong-format MoE scaffolding.
- GB10 unified memory: `nvidia-smi` memory reads are N/A — use
  `torch.cuda.mem_get_info`. Page cache from prior model reads lowers
  CUDA-free below the model size; the serve script fadvises the model dir
  first. Never remove that step.
- nvcc + ninja must be on PATH or vLLM rejects the attention backend at init.

## Diagnostics
`scripts/memory_census.py` = disk-vs-resident per-component table via a
tiny-context boot; the cheapest way to localize a loader defect. Full
methodology and the 12-defect history: `docs/LOADER_NOTES.md`.
