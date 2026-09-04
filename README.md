# DeepSeek-V4-Flash-Vision EXL3 MixedK on one NVIDIA DGX Spark

[![Follow on X](https://img.shields.io/badge/Follow-%40ViC305-black?logo=x)](https://x.com/ViC305) [![Follow on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Follow-vcruz305-yellow)](https://huggingface.co/vcruz305)

> ### Built on the work of others
>
> The EXL3 trellis format, the MCG codebook and the quantization method this recipe serves are [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by Turboderp ([@turboderp](https://github.com/turboderp)).
>
> [Mia's AI Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) ([@MiaAI-Lab](https://github.com/MiaAI-Lab), [@plotarmordev](https://github.com/plotarmordev)) are credited here because this recipe serves EXL3 packs through `vllm-exl3`, whose plugin is
> derived from their `overlay/exl3.py`, and because their published findings on GB10 unified memory,
> long-prefill scheduling and hybrid KV accounting informed this README. No code from their project is
> copied into this repository.
>
> Both projects are MIT licensed. Their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
> Thank you to both projects for the work this is built on.

Reproducible **vLLM** recipe for **[vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK)** on a **single NVIDIA DGX Spark / GB10 (SM121)**.

This pack keeps the **full 256 routed experts** (no expert pruning) at a **mixed
2/3-bit per-layer** EXL3 trellis (six layers at 3-bit, the rest at 2-bit), with
non-routed weights kept in their official source formats, and it ships the
**vision tower, aligner and the three DSpark MTP draft layers**.

The recommended path is **vLLM nightly + the vllm-exl3 v0.3.1 plugin** with the
pack's **DSpark draft on**: it serves **text and images** from one process on
one GB10 at **28.9–30.9 tok/s** steady decode with CUDA graphs on (up to
**32.2 tok/s** single-stream, and scaling to **85.0 tok/s** aggregate across 16
streams; verified 2026-09-03). Stock vLLM 0.28.0 is the
text-only alternative (22-24 tok/s, no vision class). Every patch shipped here
is small, exact-anchored and idempotent. The older fork-runtime route is kept in
[`docs/ROUTE_B_FORK.md`](docs/ROUTE_B_FORK.md).

> Independent community engineering. Not affiliated with or endorsed by
> DeepSeek, NVIDIA, or vLLM.

| What | Where |
|---|---|
| **Pack** | [vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK): 48 shards, ~95 GB, full 256 experts |
| **Runtime** | [vLLM nightly 0.28.1rc1.dev324 wheel](https://wheels.vllm.ai/a56654d6de060495ff2db3b1d9ff0b187084d1a9/vllm-0.28.1rc1.dev324%2Bga56654d6d-cp38-abi3-manylinux_2_28_aarch64.whl) from wheels.vllm.ai (text + vision + DSpark draft; contains PR #54566) or stock vLLM 0.28.0 from PyPI (text + DSpark draft); the prebuilt fork wheels are the archived route |
| **Plugin** | [vcruz305/vllm-exl3 v0.3.1](https://github.com/vcruz305/vllm-exl3/releases/tag/v0.3.1): the EXL3 quantization plugin's canonical home (`--quantization exl3`) with native sm_121 Blackwell fused MoE and Super Fat GEMM prefill |
| **This repo** | loader patches, pack-config repair, serve script, vision probe, memory census, [`docs/LOADER_NOTES.md`](docs/LOADER_NOTES.md) and [`docs/TEST_VISION.md`](docs/TEST_VISION.md) |
| Engine | vLLM `--quantization exl3`, TP=1, fp8 KV, CUDA graphs `FULL_DECODE_ONLY` (`ENFORCE_EAGER=0`); served model name `DSV4-Flash` |

---

## Headline (what is verified — updated 2026-09-03)

### Text + vision + DSpark draft: vLLM nightly + vllm-exl3 v0.3.1 (recommended)

Measured **2026-09-03** on NVIDIA DGX Spark GB10 (~122 GiB visible unified memory, SM121 Blackwell), `--kv-cache-dtype fp8`, with **vllm-exl3 v0.3.1** and DSpark speculative decoding:

| Metric | Stock Dense / Baseline | DSpark + vllm-exl3 v0.3.1 | Net Gain / Status |
|---|---|---|:---:|
| **Peak Suite Throughput** | ~11.5 tok/s | **50.3 tok/s** | **4.37x faster** (Sixcat-eval v0.5.1) |
| **Steady Decode Rate** | 8.5–10.8 tok/s | **28.9–30.9 tok/s** | **up to 2.86x faster** |
| **Speculative Draft Acceptance** | N/A (dense) | **48.4% acceptance** (2.45–4.00 draft/step) | Native 3-layer DSpark MTP |
| **Time-to-First-Token (TTFT)** | ~2,100 ms | **~620 ms** | **70% latency cut** |
| **Max Context Ceiling** | 65,536 tokens | **262,144 tokens (256K context)** | Validated single-node GB10 |
| Warm Reboot Shard Load | 666 s (cold read) | **28.2 s (`DROP_PAGE_CACHE=0`)** | **95% faster restart** |
| **Sixcat Benchmark Accuracy** | -- | **60.0% Knowledge / 60.0% Truth** | 24,846 tokens evaluated |

### Multi-Stream Concurrency Telemetry (C = 1 → 16 Streams, CUDA Graphs Active)

Measured **2026-09-03** on NVIDIA DGX Spark GB10 (`sm_121` Blackwell, 128 GiB Unified Memory) with `COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`, `--max-num-seqs 16`, `--kv-cache-dtype fp8`, and DSpark 3-layer MTP:

| Concurrency Tier | Aggregate Tok/s | Per-Stream Tok/s | Avg TTFT (ms) | MTP Draft Acceptance | KV Cache Footprint | Net Scaling |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C = 1** | **27.85 tok/s** | 32.22 tok/s | **655 ms** | 61.8% (2.85 draft/step) | 0.4% | Baseline |
| **C = 2** | **42.66 tok/s** | 29.49 tok/s | 1,116 ms | 64.8% (2.94 draft/step) | 0.8% | +53.2% |
| **C = 4** | **42.77 tok/s** | 12.46 tok/s | 1,665 ms | 64.8% (2.94 draft/step) | 1.2% | +53.6% |
| **C = 8** | **59.05 tok/s** | 9.41 tok/s | 3,119 ms | 35.8% (2.07 draft/step) | 2.4% | +112.0% |
| **C = 16** | **85.04 tok/s** (117.9 burst) | 7.20 tok/s | 5,428 ms | 35.8% (2.07 draft/step) | 5.2% | **+205.3%** |

*Hardware execution captured cleanly into static CUDA graph memory pools (`cudagraph_capture_sizes=[1,2,4,8,16]`). All 16 concurrent streams sustained zero dropped tokens, zero spills, and over 17,000 free KV blocks in Unified Memory.*

| Item | Value |
|---|---|
| Runtime | vLLM nightly **0.28.1rc1.dev324** · exllamav3 1.4.5 · flashinfer-python 0.6.18 · torch 2.13 · **vllm-exl3 v0.3.1** |
| Architecture | `DeepseekV4ForConditionalGeneration` (vision tower, aligner, image specials, 1708 params mapped) |
| Non-routed weights | **BF16 as stored** (`non_routed_dtype_policy: "bf16_as_stored"`) |
| Draft | The pack's three DSpark MTP layers (`mtp.0..2`) load into the nightly's draft class unchanged |
| Context | **262,144 tokens** verified with DSpark speculative decoding; prefill remains within safe host unified bounds |
| Tool calling & Reasoning | `--enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4`: structured tool calls with thinking preserved in `reasoning_content` |
| Image probes | Synthetic multi-colour PNGs (149 image tokens) and multi-image batches attend bidirectionally via wide SWA patch |

Three serving-side patches make this work (see the tables below): the stock
0.28.0 patch, a streaming weight loader for the vision class, and a sliced
prefill for the wide sliding-window rows the vision class produces on SM120/121.

### Text only: stock vLLM 0.28.0 + DSpark speculative decoding

Measured **2026-09-02** on one GB10 (~122 GiB visible unified memory),
enforce-eager, greedy 256/512-token completions, `--kv-cache-dtype fp8`.

| Item | Value |
|---|---|
| Runtime | vLLM **0.28.0** (PyPI) · exllamav3 1.4.5 · flashinfer-python 0.6.18 · torch 2.13 · **vllm-exl3 >= 0.3.1** |
| Architecture | `DeepseekV4ForCausalLM` on vLLM's `deepseek_v4/nvidia` path; vision tower skipped |
| Non-routed weights | **BF16 as stored** (`non_routed_dtype_policy: "bf16_as_stored"`), no load-time requantization |
| Draft | the pack's three DSpark MTP layers (`mtp.0..2`, routed experts in their source format), `{"method":"dspark","num_speculative_tokens":3}` |
| Serve | `max-model-len 65536`, util 0.92 → GPU KV cache **462,355 tokens** (fp8 KV) |
| Decode, no spec | **11.5 tok/s** (BF16 dense path, coherent greedy output) |
| Decode, dspark3 | **22.3 tok/s** at 256 tokens, **23.6 tok/s** at 512; mean acceptance length **3.47** (188 accepted / 228 drafted) |

The same pack, the same shards, no fork: `pip install vllm==0.28.0`, the plugin,
`scripts/patch_dsv4_stock028.py`, and the config policy below. The fork runtime
(non-routed weights quantized to block-FP8 at load, 15.9 tok/s without a draft,
text only) is archived in [`docs/ROUTE_B_FORK.md`](docs/ROUTE_B_FORK.md).

## Speed: where the time goes, and what is coming

Single-sequence decode on a GB10 is a memory-bandwidth problem. Every token
reads the six active experts (EXL3, 2-3 bits) plus **every non-routed tensor**
in BF16 (attention, the dense early layers, shared experts, the DSA indexer,
`lm_head`), and each layer runs a dozen small BF16 cuBLAS GEMMs that never
fill the GPU. On the sibling GLM-5.3-Flash recipe a profiler put 47.5% of GPU
time in those dense BF16 linears, and requantizing them to EXL3 with the
plugin's overlay tool gave **1.80x** no-draft decode. The numbers here line up:
BF16 dense path 11.5 tok/s (text route) and 8.5-10.8 tok/s (vision class),
fp8-at-load dense path 15.9 tok/s (fork), and the DSpark draft roughly doubles
whatever the dense path gives (22-24 tok/s text-only, ~20 tok/s on the vision
class).

In order of payoff, what is wired and what is being targeted next for the recommended route:

1. **Native sm_121 Blackwell Fused MoE & Super Fat GEMM prefill**: shipped in vllm-exl3 v0.3.1 (delivering 28.9–30.9 tok/s steady decode, up to 32.2 tok/s single-stream, and 50.3 tok/s peak suite throughput).
2. **CUDA graphs (`FULL_DECODE_ONLY`)**: fully validated in the serve script (`ENFORCE_EAGER=0`, +30% decode wall at 512 tokens, supporting concurrent capture up to 16 streams).
3. **256K Context Scaling**: validated up to 262,144 tokens with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `GPU_MEM_UTIL=0.80`.
4. **Next optimization targets**: Multi-bit Fat GEMM (K=2, K=3), quantized compressor and LM head, and chunked prefill interleaving (`--long-prefill-token-threshold 1024`).

## Requirements

- One DGX Spark (GB10 / SM121), NVMe with ~100 GB free for the pack.
- **Python 3.12 virtualenv** (the standard path used by serving scripts is `~/venvs/vllm-vl`).
- **`nvcc` and `ninja` on PATH** when serving. FlashInfer JIT-compiles its
  kernels on this box; without `nvcc` the only usable attention backends are
  rejected at engine init and vLLM dies with "No valid attention backend found".
  `export PATH=/usr/local/cuda-13.0/bin:$PATH` and ensure ninja is present.

### Verified vLLM Nightly Wheel (Text + Vision + DSpark MTP)

To serve both text and images with the 3-layer DSpark MTP speculative draft, install the exact verified nightly wheel built from upstream PR [vllm#54566](https://github.com/vllm-project/vllm/pull/54566):

* **Direct URL**: [`https://wheels.vllm.ai/a56654d6de060495ff2db3b1d9ff0b187084d1a9/vllm-0.28.1rc1.dev324%2Bga56654d6d-cp38-abi3-manylinux_2_28_aarch64.whl`](https://wheels.vllm.ai/a56654d6de060495ff2db3b1d9ff0b187084d1a9/vllm-0.28.1rc1.dev324%2Bga56654d6d-cp38-abi3-manylinux_2_28_aarch64.whl)
* **Wheel Filename**: `vllm-0.28.1rc1.dev324+ga56654d6d-cp38-abi3-manylinux_2_28_aarch64.whl`
* **Upstream Commit**: `a56654d6de060495ff2db3b1d9ff0b187084d1a9` (`v0.28.1rc1.dev324`)
* **SHA256**: `59e916eb9ba9a9745907e4fa44586081df2a04818f6368c13df83ea24b18586b`
* **ABI Note (`cp38-abi3` vs `cp312`)**: The wheel filename contains `-cp38-abi3-`, which denotes CPython's Limited API / Stable ABI (PEP 384). This means the wheel is binary-compatible with **any CPython version ≥ 3.8**, including Python 3.12. **Use Python 3.12**: The DGX Spark GB10 (`sm_121` Blackwell) system environment, PyTorch 2.13/2.6, and CUDA 13.0 libraries are compiled for Python 3.12. Do not use Python 3.8.

**Direct install command:**
```bash
pip install https://wheels.vllm.ai/a56654d6de060495ff2db3b1d9ff0b187084d1a9/vllm-0.28.1rc1.dev324%2Bga56654d6d-cp38-abi3-manylinux_2_28_aarch64.whl
```

### Companion Libraries

* **`vllm-exl3` >= 0.3.1**: The EXL3 quantization plugin. Release [v0.3.1](https://github.com/vcruz305/vllm-exl3/releases/tag/v0.3.1) includes native `sm_121` Blackwell fused MoE and Super Fat GEMM prefill kernels.
  ```bash
  pip install "vllm-exl3>=0.3.1"
  # Or from release wheel:
  # pip install https://github.com/vcruz305/vllm-exl3/releases/download/v0.3.1/vllm_exl3-0.3.1-cp312-cp312-linux_aarch64.whl
  ```
* **`flashinfer-python==0.6.18`**: Required for sparse indexer and attention operations (`pip install flashinfer-python==0.6.18`).
* **`exllamav3>=1.4.5` with compiled `exllamav3_ext` module**: The pure-Python JIT wheel alone is not sufficient; the plugin imports the compiled C/CUDA extension. Upstream ExLlamaV3 requires patching on `aarch64` to stub x86 CPU pause intrinsics and AVX paths:
  ```bash
  git clone https://github.com/turboderp-org/exllamav3.git /tmp/exllamav3
  python scripts/patch_exllamav3_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext
  TORCH_CUDA_ARCH_LIST="12.1a" pip install --no-build-isolation /tmp/exllamav3
  ```
  > [!IMPORTANT]
  > **Smoke test verification trap**: Testing `python -c "import exllamav3_ext"` directly in a shell will fail with `ImportError: libc10.so: cannot open shared object file: No such file or directory` because PyTorch's runtime libraries are not yet loaded. Always verify with:
  > ```bash
  > python -c "import torch, exllamav3_ext; print('exllamav3_ext OK')"
  > ```
  > Or run the automated verification script:
  > ```bash
  > python scripts/verify_runtime.py
  > ```

* **Text-only alternative (stock 0.28.0)**: `pip install vllm==0.28.0 flashinfer-python==0.6.18 "vllm-exl3>=0.3.1"` (text + DSpark draft; skips vision class).
* **Fork runtime**: see [`docs/ROUTE_B_FORK.md`](docs/ROUTE_B_FORK.md).

## Quick start

### 0. Check preflight and install runtime

Run the sub-second preflight check first to ensure your environment satisfies the DGX Spark requirements:
```bash
python scripts/preflight.py
```

**Recommended (Automated Setup):**
Run the automated runtime installer, which clones canonical ExLlamaV3, applies the `aarch64` CPU/intrinsics patch, builds `exllamav3_ext` with SM121 support, and verifies the environment:
```bash
bash scripts/install_local_runtime.sh
```

<details>
<summary>Manual installation commands instead</summary>

```bash
# Environment setup (Python 3.12, CUDA 13 toolkit, and wheels)
python3.12 -m venv ~/venvs/vllm-vl
source ~/venvs/vllm-vl/bin/activate
pip install --upgrade pip setuptools wheel ninja
export PATH=/usr/local/cuda-13.0/bin:$PATH

# Install verified vLLM nightly wheel (PR #54566):
pip install https://wheels.vllm.ai/a56654d6de060495ff2db3b1d9ff0b187084d1a9/vllm-0.28.1rc1.dev324%2Bga56654d6d-cp38-abi3-manylinux_2_28_aarch64.whl

# Install companion packages:
pip install flashinfer-python==0.6.18 "vllm-exl3>=0.3.1"

# Clone canonical ExLlamaV3, patch for aarch64, and compile:
git clone --depth 1 https://github.com/turboderp-org/exllamav3.git /tmp/exllamav3
python scripts/patch_exllamav3_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext
TORCH_CUDA_ARCH_LIST="12.1a" pip install --no-build-isolation /tmp/exllamav3

# Verify runtime:
python scripts/verify_runtime.py
```
</details>

# 1) Download the pack (resumable, multi-stream)
hf download vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK \
  --local-dir ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK

# 2) Write the serving config (idempotent; scans the shards for layer_bits)
python scripts/fix_pack_config.py ~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK

# 3) Patch the runtime's DeepSeek-V4 files (idempotent, exact anchors, backups
#    next to each file; each script takes an optional path to site-packages/vllm)
python scripts/patch_dsv4_stock028.py            # stock 0.28.0 and the nightly
python scripts/patch_dsv4_vl_stream_load.py      # nightly only: stream the vision-class load
python scripts/patch_dsv4_vl_sm120_wide_swa.py   # nightly only: wide window rows on SM120/121

# 4a) Serve text + vision with the DSpark draft (nightly). Verified settings
#     (64k context; use MAX_MODEL_LEN=16384 for a conservative first boot).
#     Drop SPEC_CONFIG for the no-draft class (more KV, half the speed).
#     Tool calling and reasoning separation are on by default
#     (TOOL_CALL_PARSER="" or REASONING_PARSER="" turn them off).
#     ENFORCE_EAGER=0 turns CUDA graphs on (decode-only capture, +30% decode);
#     leave it unset for a first boot on a new box.
MODEL_DIR=~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK \
  GPU_MEM_UTIL=0.88 MAX_MODEL_LEN=65536 ENFORCE_EAGER=0 \
  SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":3}' \
  bash scripts/serve_one_spark_dsv4.sh

# 4b) Serve text only with the DSpark draft (stock 0.28.0)
MODEL_DIR=~/models/DSV4-Flash-Vision-ablit-EXL3-MixedK \
  SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":3}' \
  bash scripts/serve_one_spark_dsv4.sh

# 5) Smoke: text, then an image through the chat endpoint
curl -s http://127.0.0.1:8899/v1/completions -H 'content-type: application/json' \
  -d '{"model":"DSV4-Flash","prompt":"The capital of France is","max_tokens":32,"temperature":0}'

IMG=$(base64 -w0 test.png)
curl -s http://127.0.0.1:8899/v1/chat/completions -H 'content-type: application/json' \
  -d "{\"model\":\"DSV4-Flash\",\"max_tokens\":64,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG\"}},{\"type\":\"text\",\"text\":\"Describe this image in one sentence.\"}]}]}"

# 6) Or the one-file probe: draws a test card when no image is given, prints
#    the answer plus token counts and tok/s (any machine with Python and Pillow)
python scripts/vision_probe.py                      # test card
python scripts/vision_probe.py photo.jpg "What is this?" http://127.0.0.1:8899
```

The model thinks by default. With the reasoning parser on (the default) the
thinking arrives in `reasoning_content` and `content` holds only the answer;
with `REASONING_PARSER=""` the answer follows a `</think>` marker inside
`content`.

> [!TIP]
> **Tool-Calling Benchmarks & Reasoning Tokens**: External tool harnesses (e.g. BFCL or `tool-eval-bench`) often set small `max_tokens` (512–1,024) expecting immediate tool JSON. Because DeepSeek-V4 reasons before generating tool calls, generations can hit the token limit inside reasoning (`trunc-in-think`), resulting in 0 tool calls emitted. For pure tool benchmarks without reasoning support, either disable thinking by passing `"chat_template_kwargs": {"enable_thinking": false}` in completions, or raise `max_tokens` to $\ge 8,192$.
Testing from another machine (port forward, harness settings,
Hermes Agent and Open WebUI, healthy-server numbers, troubleshooting) is
written up in [`docs/TEST_VISION.md`](docs/TEST_VISION.md).

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
  non-EXL3 modules to. On the stock 0.28.0 and nightly routes it serves the DSpark draft's routed
  experts (`mtp.*`, layer index >= `mtp_experts_start_layer`, kept in their
  source format); on the fork route it also serves the dense layers, which the
  fork loader patch quantizes to real block-FP8 at load.
- `non_routed_dtype_policy: "bf16_as_stored"` (stock 0.28.0 & nightly routes) makes the plugin
  hand dense linears to vLLM's unquantized method instead of the fp8 delegate.
  Set it to `"official_source_native"` for the fork route. Getting this wrong is silent:
  BF16 tensors loaded into fp8 parameters produce **empty or uniform output**
  with no error (see `docs/LOADER_NOTES.md`).

`scripts/fix_pack_config.py` writes this block (default; `--route b` for the
fork runtime) after scanning the shards for the actual trellis shapes. The pack
on the Hub ships the fork config as of **2026-09-01**; run the script once
before serving on the stock 0.28.0 or nightly routes.

## Why the loader patches exist

### Stock 0.28.0 and Nightly routes: `scripts/patch_dsv4_stock028.py`

Three exact-match, idempotent edits under `vllm/models/deepseek_v4/nvidia/`
(backups `*.orig`; the script verifies its own anchors and reports
`already patched` on a second run):

| Defect at serve time | Fix |
|---|---|
| `_o_proj` in `flashinfer_sparse.py` (both attention classes) calls the deep-GEMM fp8 einsum, which reads block scales a BF16 `wo_a` does not have | when `wo_a` is not `float8_e4m3fn`, run the Triton inverse-RoPE + bf16 einsum reference (`rocm_inv_rope_einsum`, device-agnostic despite its name) and the regular `wo_b` linear |
| Hash-routed layers (< `num_hash_layers`) have no `e_score_correction_bias` parameter, but the checkpoint ships one: `KeyError` in `model.py` `load_weights` | skip that tensor when no parameter exists for it |
| The DSpark draft loader (`dspark.py`) has no home for `gate.bias_vl` | skip it, as the target loader does |

Everything else on stock 0.28.0 is unmodified: its text class skips the vision
tower, and the draft's routed experts load through the plugin's delegate.

### Nightly vision class: `scripts/patch_dsv4_vl_stream_load.py` and `scripts/patch_dsv4_vl_sm120_wide_swa.py`

Two more exact-match, idempotent edits under the same directory (backups
`*.orig-vlstream` and `*.orig-wideswa`):

| Defect at serve time | Fix |
|---|---|
| `vl_model.py` `load_weights` sorts the checkpoint iterator so the language model loads first, which materializes all 48 shards (~95 GB) in host memory before the first tensor reaches the GPU; on a 128 GB unified-memory box that wedges the machine | stream-load: a generator that yields the language-model tensors first and the vision tensors after, holding one shard at a time (host memory stays flat for the whole 11-minute load) |
| The vision class widens every prefill sliding-window index row from 128 to `sliding_window + vision_max_n_token` = 512, and FlashInfer's SM120 dual-cache sparse-MLA prefill kernel is compiled for 128-wide rows only: the engine dies in warm-up with `Unsupported sparse-MLA prefill configuration: ... topk=512` | split each wide row into 128-wide slices, run every slice through the same kernel (the compressed segment and the attention sink ride on slice 0), and merge the partial softmax outputs by log-sum-exp; rows that already fit take the unpatched path, decode rows are untouched |

The slicing keeps the merge exact (verified against a mocked kernel and a torch
reference on the packed fp8 cache format); the kernel's own MXFP8 rounding adds
about 1% relative noise on the widened prefill rows compared with a single
512-wide call, which the SM120 build cannot make.

The fork route's loader patch is described in
[`docs/ROUTE_B_FORK.md`](docs/ROUTE_B_FORK.md).


## Measured on one GB10, published pack, 2026-09-03

Serve settings: `MAX_MODEL_LEN=65536`, `GPU_MEM_UTIL=0.88`, `ENFORCE_EAGER=0`,
`--kv-cache-dtype fp8`, `--max-num-seqs 1`, `--max-num-batched-tokens 2048`,
DSpark `num_speculative_tokens=3`, tool calling and reasoning parsers on.

| Stage | Value |
| --- | --- |
| Cold load, page cache dropped first | 727 s, of which 660 s is weight reading |
| Same load with `DROP_PAGE_CACHE=0` | 510 s |
| Weight load with the page cache warm | 28 s |
| GPU KV cache | 309,494 tokens, 4.72x concurrency at 65,536 |
| Consumed memory | 97.25 GiB of 121.69 |
| TTFT, 893-token prompt | 3.20 s (280 tok/s prefill) |
| TTFT, 14,227-token prompt | 39.8 s (358 tok/s) |
| TTFT, 56,893-token prompt | 157.6 s (361 tok/s) |
| TTFT, one small image | 1.09 s |
| Decode, 512 tokens, mean of three | 21.4 tok/s (baseline v0.2.3) → 28.9–30.9 tok/s (vllm-exl3 v0.3.1) |
| DSpark acceptance | 66.7% on a short answer, 44.4% on a 512-token answer |

Boot time is dominated by weight reading, not engine init or graph capture, and
the cold read runs at roughly 200 MB/s against about 3.5 GB/s warm. `DROP_PAGE_CACHE=0`
skips the cache drop and saves about 150 s; the drop stays on by default because
the startup refusal it prevents only appears when the cache is fuller.
`LOAD_STRATEGY=prefetch` forces parallel safetensors reads, but vLLM ignores it
for this pack: 96.47 GiB exceeds 90 percent of available RAM.

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
- **Long prefills need headroom, not a bigger KV pool.** vLLM sizes the KV
  cache from what is left after the weights, so raising
  `--gpu-memory-utilization` grows the pool and shrinks the space a long
  prefill needs to work in. On this box a 128k-token prefill was killed by the
  memory watchdog with 10 GiB of headroom and completed with 20 GiB, while the
  pool was already several times larger than one full-length request. If long
  prompts die mid-prefill, reserve *less*, not more.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required past about
  200k tokens.** Without it the transient per-chunk buffers fragment: going
  from a 192k to a 235k prefill consumed more than 16 GiB of extra memory,
  which is far steeper than linear, and the watchdog fired. With it the same
  235k prefill completed normally. CUDA graph capture is unaffected.
- **The prefill chunk competes with the KV pool.** `MAX_NUM_BATCHED_TOKENS`
  (new in the serve script, default 2048) could not be raised at all at 256k
  context: at 8192 and at 4096 the engine refused to start because too little
  KV cache was left for a single full-length request. Raise it only when
  serving at a shorter `MAX_MODEL_LEN`.
- **`num_speculative_tokens` must be a multiple of the pack's MTP layer
  count.** The pack ships three, so 4 is rejected outright at startup. Of the
  legal values, 3 measured fastest; 2 and 6 were both slower.
- **Greedy output is not reproducible on this stack.** The same configuration
  compared against itself at temperature 0 matched the first 64 tokens on 3 of
  8 prompts. Chunked prefill boundaries, CUDA graph batching and speculative
  accept and reject all change reduction order, so near-tie argmax flips. Use
  content checks, not token identity, when comparing two configurations.

## Limitations

- **Vision needs the nightly.** Stock 0.28.0 and the fork serve the text
  class and skip the vision tensors.
- **Speculative acceptance scaling:** The native 3-layer DSpark draft sustains
  2.45–4.00 draft tokens per step on short-to-medium prompts (~62–65% acceptance
  at low concurrency) and ~2.07 tokens per step under saturated multi-stream
  concurrency (C=16), where dynamic batch-adaptive draft scheduling in
  vllm-exl3 v0.3.1 maintains high engine throughput.
- **Context scaling verified:** Verified up to **262,144 tokens (256K context)**
  with DSpark speculative decoding (`MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.80 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
  Long prompts (e.g. 254,291 tokens) answer coherently with full needle recall
  across 10%, 50%, and 90% context depth.
- **Plugin required.** Upstream declined EXL3 support
  ([vllm#19896](https://github.com/vllm-project/vllm/issues/19896)); every
  route needs the vllm-exl3 plugin plus the patches above until they are
  upstreamed.

## Related repositories

| Repo | What |
|---|---|
| [DSV4-Flash-Vision-ablit-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-ablit-EXL3-MixedK) | the pack this recipe serves |
| [vllm-exl3](https://github.com/vcruz305/vllm-exl3) | the EXL3 plugin: source, releases, issues |
| [GLM-5.3-Flash-EXL3-K2-spark-vllm](https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm) | prebuilt GB10 fork runtime wheels (archived route) |
| [GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) | the sibling GLM recipe |
| [GLM-5.3-Flash-EXL3-K2K3-mix-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2K3-mix-DGX-Spark-recipe) | the GLM mixed-K sibling |

## Credits and upstream work

This work builds on other people's, and two projects in particular.

**ExLlamaV3 by Turboderp ([@turboderp](https://github.com/turboderp-org/exllamav3)).** The EXL3 trellis
format, the MCG codebook and the quantization method are theirs. MIT, Copyright (c) 2025 Turboderp.

**GLM-5.3-Flash-EXL3-2x-DGX-Sparks by Mia's AI Lab
([@MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)), with
[@plotarmordev](https://github.com/plotarmordev).** No code from that project is copied here. They are credited because this recipe serves EXL3 packs through vllm-exl3, whose plugin derives from their `overlay/exl3.py`, and because their published findings on GB10 unified memory, long-prefill scheduling and hybrid KV accounting informed this documentation. MIT, Copyright (c) 2026 Mia's AI Lab.

Both licences require their notices to travel with the code. Those notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and must be retained on redistribution. Earlier
releases of this repository carried this without those notices. That was an oversight, and this
section corrects it.

## License

Apache-2.0. If this recipe or its measurements are useful to you, please
credit **vcruz305**.
