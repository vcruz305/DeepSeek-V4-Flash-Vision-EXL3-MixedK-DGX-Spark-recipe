# Getting the DSV4 EXL3 MixedK pack to serve on vLLM: every defect, with receipts

The first attempt to serve this pack on the fork runtime materialized ~117 GiB
of a ~95 GB checkpoint and OOM'd allocating the KV cache — while ExLlamaV3
loaded the identical weights at ~91 GB and generated correctly. One evening of
instrumented boots later, the same runtime loads it at **86.73 GiB** and serves
coherent text. Nothing here changed the weights; every defect was in how they
were declared, mapped, or fed forward. This file is the full list — symptom →
localization → fix — plus the two dead ends worth not repeating.

Method in one line: a **tiny-context census boot** (`scripts/memory_census.py`,
2048 ctx, explicit 2 GiB KV, enforce-eager) that prints disk-vs-resident
per-component tables, run in cycles where **every failure is cheap** — most
defects surfaced in under 90 seconds, and anything checkable offline (config
parsing, plugin wiring) was probed without touching the GPU.

## The defects

### 1. The pack declared fp8, and vLLM believed it (the 117 GiB balloon)

- **Symptom:** `--quantization exl3` on the CLI, yet the engine line read
  `quantization=deepseek_v4_fp8` and the log showed
  `Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend`. Load reached ~117 GiB, then OOM.
- **Cause:** the pack's `config.json` still carried the base model's
  `quantization_config: {quant_method: "fp8", ...}`. vLLM resolves the method
  from the model config via `override_quantization_method` **before** honoring
  the CLI, so the fork's fp8 method claimed the model and built MoE scaffolding
  in the wrong format on top of the EXL3 tensors.
- **Fix:** declare EXL3 in the pack config (`scripts/fix_pack_config.py`; the
  Hub pack ships it fixed as of 2026-09-01). Rule for every EXL3 pack: **ship
  the exl3 `quantization_config` or your CLI flag is a suggestion.**

### 2. Page cache vs CUDA-free on unified memory

- **Symptom:** `ValueError: Free memory on device cuda:0 (116.42/121.69 GiB)
  on startup is less than desired GPU memory utilization` — with `free -g`
  showing over 100 GiB available.
- **Cause:** GB10's GPU and host share one memory pool. The 95 GB download
  (and every prior boot) leaves the shards in page cache;
  `torch.cuda.mem_get_info()` reports the reclaimable pages as not-free.
  `nvidia-smi` is no help — its memory fields read `[N/A]` on GB10.
- **Fix:** `posix_fadvise(POSIX_FADV_DONTNEED)` over the pack before boot
  (built into the serve script and the census). Measured effect: CUDA-free
  86.8 → 123.4 GiB with the model untouched on disk.

### 3-5. The strict loader vs everything the text class doesn't model

- **Symptom (three rounds):** `There is no module or parameter named 'aligner'`
  → then `'image_end'` → then `KeyError` on gate tensors.
- **Cause:** the pack preserves the **vision tower** (`vision.*`, `aligner.*`,
  `image_*` token embeddings) and **3 MTP draft layers** (`mtp.*`, 10.27 GB);
  the text-only `DeepseekV4ForCausalLM` has no homes for them, and vLLM's
  loader raises on unknown tensors (ExLlamaV3 skips them silently — part of
  why its 91 GB "just worked").
- **Fix:** skip them at load (`scripts/patch_dsv4_loader.py`). The MTP tensors
  load separately if speculative decoding is ever configured.

### 6. Hash-routed layers and the vestigial gate bias

- **Symptom:** `KeyError: 'layers.0.ffn.gate.e_score_correction_bias'`.
- **Localization:** the checkpoint holds, for layers 0-2, **both**
  `gate.tid2eid` (hash routing tables — real, loaded) **and** `gate.bias`
  (vestigial: hash layers don't use a correction bias). It also holds
  `gate.bias_vl` on every layer — a vision-language variant with no text-path
  mapping at all.
- **Fix:** skip `gate.bias` for layer indices below `num_hash_layers` (read
  from the config at load; it is 3 and it is correct) and all `gate.bias_vl`.

### 7. The slimmed config omits `topk_method`

- **Symptom:** same `e_score_correction_bias` KeyError on the noaux layers.
- **Cause:** the model creates the bias parameter only under
  `topk_method == "noaux_tc"`, and the pack's config omits the field, so the
  parameter was never created — while the checkpoint carries the tensors.
- **Fix:** default the branch to `noaux_tc` in the model code. The checkpoint
  containing 40 layers of correction biases *is* the evidence for the routing
  mode. (Do **not** fix this by adding fields to the config — see dead end B.)

### 8. MixedK per-layer bits

- **Symptom:** `EXL3 load shape mismatch ... dest (256, 128, 32) != loaded
  (256, 128, 48)` at layer 3.
- **Cause:** the pack is mixed-bitrate — trellis last-dim 48 (3-bit) on layers
  3, 13, 21, 22, 28, 41; 32 (2-bit) elsewhere — and the plugin sizes dest
  tensors from a single `bits` value unless told otherwise.
- **Fix:** `quantization_config.layer_bits` (the
  [vllm-exl3](https://github.com/vcruz305/vllm-exl3) plugin's existing
  mechanism; `fix_pack_config.py` regenerates the map by scanning the shards).

### 9. The plugin swallowed its own config key

- **Symptom:** the delegation below never engaged; an offline probe showed the
  parsed config object had no `non_routed_quantization` attribute even though
  the JSON did.
- **Cause:** the plugin config's `__init__` accepts `**kwargs` without storing
  them.
- **Fix:** stash the key explicitly in `from_config` (shipped in the plugin
  ≥ 0.1.1 / vllm-exl3 0.2.0). Found offline in 60 seconds — no GPU cycle.

### 10-12. BF16 non-routed weights vs an fp8-specialized forward

- **Symptom (first form):** `AttributeError: 'ColumnParallelLinear' object has
  no attribute 'weight_scale_inv'` — raised **in forward**, from the deep-GEMM
  o-proj einsum, after a full successful load.
- **Localization:** the fork's nvidia path is fp8-hardwired — the o-proj
  (`deep_gemm_fp8_o_proj`), the shared-expert path, and the ue8m0 conversions
  all read fp8 scale attributes unconditionally; there is no bf16 branch. The
  pack, meanwhile, keeps non-routed weights **BF16 on disk with zero scale
  tensors** (its "official source native" policy as stored).
- **Fix (three parts):** declare `non_routed_quantization` in the pack config;
  have the plugin delegate non-routed layers to the runtime's block-FP8 method
  (creating fp8 params + `weight_scale_inv` the forward expects); and — the
  part that makes it *correct* — quantize the BF16 tensors at load with real
  128×128 block scales (`scale = amax/448`, emitting `weight` +
  `weight_scale_inv`), including through the fused destinations
  (`fused_wqa_wkv`, `compressor.fused_wkv_wgate`, shared-expert
  `gate_up_proj`/`down_proj`). Resident cost: 5.86 GiB of fp8 versus 11.7 GiB
  if the same tensors were held BF16.

## The two instructive dead ends

**A. Dequantize fp8 → bf16 at load.** Right diagnosis (dtype mismatch), wrong
direction: the forward has no bf16 branch to receive it. Reverted within one
cycle. The grain of the fork is fp8 — feed it fp8 done right, don't fight it.

**B. "Fix" the config by adding fields.** Adding `topk_method` into a nested
`text_config` looked harmless and broke config parsing entirely
(`ValidationError: text_config ... does not have num_attention_heads`): this
pack's config is **flat**, and injecting even an *empty* `text_config` makes
vLLM adopt it as the whole text config. The stripped config only works because
the *absence* of keys routes every reader to correct defaults. Repair
checkpoints' declarations (`quantization_config`), default behaviors in code —
and never invent nested structure a config never had.

**The signature worth memorizing:** after the delegation landed *without* real
scales (BF16 bytes cast into fp8 params), the model ran mechanically and
emitted endless `<begin_of_sentence>`. Logprobs showed every token at
**-11.77 = -ln(129,280)** — a perfectly uniform distribution. Flat-uniform
logits mean the hidden state died upstream while the machinery kept running;
check your weight values, not your kernels.

## The numbers after all fixes

| | |
|---|---|
| Disk | 101.44 GB total (74.49 experts, 10.27 mtp, ~9 attention, ~2 vision/specials) |
| Resident | **83.91 GiB** params+buffers; **86.73 GiB** CUDA allocated |
| By dtype | int16 74.09 (packed trellis) · fp8 5.86 · bf16 2.83 · fp32 0.68 |
| Serve | 65,536 ctx, util 0.92 → KV cache **1,069,556 tokens** |
| Smoke | coherent greedy factual completions, **15.9 tok/s** (enforce-eager, no spec decode) |

## The playbook, portable to the next pack

1. **Read the disk first.** The census's disk table (shard headers only) is
   free and tells you what the checkpoint actually holds — dtypes, scale
   tensors present or absent, per-layer trellis widths — before any theory.
2. **Make failures cheap.** Tiny context, explicit KV bytes, enforce-eager:
   most config/mapping defects then fail in under 90 seconds instead of after
   a 12-minute load.
3. **Probe offline before burning a GPU cycle.** Config parsing, plugin
   wiring, and method resolution all reproduce in a CPU-only interpreter.
4. **Trust dtype tables over narratives.** Both dead ends above were killed by
   a number (a resident tensor at half its disk size; a logprob equal to
   -ln vocab), not by reasoning about code.

## Addendum 2026-09-02: stock vLLM 0.28.0 (route A)

The same pack boots on PyPI vLLM 0.28.0 with the plugin. The first boot loaded
cleanly, served `200 OK`, and returned **empty text** at 15-17 tok/s: the pack
config still declared `non_routed_quantization` (fp8), so the plugin delegated
the BF16 dense linears to the fp8 method, whose `weight_scale_inv` parameters
were never written. Same family as dead end 2 above, a different symptom.

Fix, in three parts, each necessary:

1. Config: `non_routed_dtype_policy: "bf16_as_stored"` (plugin >= 0.2.3 hands
   dense linears to the unquantized method). `non_routed_quantization` stays,
   because the DSpark draft's routed experts (`mtp.*`, source format) still
   need the delegate; removing it breaks spec decode with
   `KeyError: 'w13_weight_scale'`.
2. `flashinfer_sparse.py`: `_o_proj` gets a BF16 branch (Triton inverse-RoPE
   einsum + `wo_b`), since the deep-GEMM fp8 einsum reads scales that do not
   exist for a BF16 `wo_a`.
3. Loader skips: hash-layer `e_score_correction_bias` (target) and
   `gate.bias_vl` (draft).

Receipts: no spec 11.5 tok/s, coherent; `{"method":"dspark",
"num_speculative_tokens":3}` 22.3 / 23.6 tok/s at 256 / 512 tokens, mean
acceptance length 3.47. FlashInfer must be 0.6.18 or newer (`index_topk=192`)
and `nvcc` on PATH.
