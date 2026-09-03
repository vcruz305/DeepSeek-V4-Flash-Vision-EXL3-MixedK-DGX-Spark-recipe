#!/usr/bin/env bash
# Serve DSV4-Flash-Vision EXL3 MixedK from a local vLLM + ExLlamaV3 install on
# one DGX Spark (GB10). Run scripts/patch_dsv4_loader.py once first.
set -euo pipefail

# vLLM's has_flashinfer() returns False when nvcc is not on PATH (FlashInfer
# JIT-compiles its kernels on this box; there are no pre-downloaded cubins),
# and the usable attention backends are then rejected at engine init:
#   ValueError: No valid attention backend found for cuda ...
# Put the CUDA 13 toolkit on PATH before vLLM looks for it.
if ! command -v nvcc >/dev/null 2>&1; then
  for d in /usr/local/cuda-13.0/bin /usr/local/cuda/bin; do
    if [[ -x "$d/nvcc" ]]; then export PATH="$d:$PATH"; echo "nvcc was not on PATH; added $d"; break; fi
  done
fi
# FlashInfer's JIT runs ninja, which pip installs into the venv's bin/.
if ! command -v ninja >/dev/null 2>&1; then
  for d in "${VENV:-$HOME/venvs/glm53-exl3-local}/bin" "$(dirname "$(command -v vllm 2>/dev/null || echo /nonexistent/x)")"; do
    if [[ -x "$d/ninja" ]]; then export PATH="$d:$PATH"; echo "ninja was not on PATH; added $d"; break; fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install the CUDA 13 toolkit or: export PATH=/usr/local/cuda-13.0/bin:\$PATH" >&2
  exit 1
fi

MODEL_DIR="${MODEL_DIR:-${HOME}/models/DSV4-Flash-Vision-ablit-EXL3-MixedK}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8899}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# Prefill chunk. The engine warns at boot that speculative decoding pins
# max_num_scheduled_tokens to 2048 and suggests raising this.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
# Set to "prefetch" to force parallel safetensors reads. vLLM disables auto-prefetch
# on local filesystems, and the cold read here runs at about 190 MB/s without it.
LOAD_STRATEGY="${LOAD_STRATEGY:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
SERVED_NAME="${SERVED_NAME:-DSV4-Flash}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "missing $MODEL_DIR/config.json — download the pack first" >&2
  exit 1
fi
if ! python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if (c.get("quantization_config") or {}).get("quant_method")=="exl3" else 1)' "$MODEL_DIR/config.json"; then
  echo "config.json does not declare quant_method=exl3 — run scripts/fix_pack_config.py first" >&2
  echo "(a leftover fp8 declaration silently overrides --quantization exl3 and the load OOMs)" >&2
  exit 1
fi

export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

# Unified-memory pre-step: the pack's page cache (from the download or a prior
# boot) shrinks CUDA-free below vLLM's startup check even though the memory is
# reclaimable. Drop it. (nvidia-smi memory reads N/A on GB10; use
# torch.cuda.mem_get_info to inspect.)
# DROP_PAGE_CACHE=0 skips this. It costs about 630 s per boot: the same pack
# loads in 28 s warm and 660 s cold, so only drop it when the startup check
# actually fails.
if [[ "${DROP_PAGE_CACHE:-1}" != "0" ]]; then
python3 - "$MODEL_DIR" <<'PY'
import glob, os, sys
for fn in glob.glob(os.path.join(sys.argv[1], "*")):
    if os.path.isfile(fn):
        try:
            fd = os.open(fn, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass
print("page cache dropped for", sys.argv[1])
PY
else
  echo "page cache drop skipped (DROP_PAGE_CACHE=0)"
fi

# ENFORCE_EAGER=1 (default) is the verified configuration. ENFORCE_EAGER=0
# drops --enforce-eager and captures full CUDA graphs for the decode batches
# only (no Inductor compile, prefill stays eager); override the exact
# compilation config with COMPILATION_CONFIG. Speculative decoding is opt-in
# below.
ARGS=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size 1
  --quantization exl3
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 1
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)
if [[ -n "$LOAD_STRATEGY" ]]; then ARGS+=(--safetensors-load-strategy "$LOAD_STRATEGY"); fi
ARGS+=(
  --kv-cache-dtype fp8
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --no-enable-prefix-caching
  --trust-remote-code
)

ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  ARGS+=(--enforce-eager)
else
  COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"
  ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi
# Opt-in torch profiler: PROFILER_DIR=/path enables /start_profile and /stop_profile
# (Chrome traces land in that directory).
if [[ -n "${PROFILER_DIR:-}" ]]; then
  mkdir -p "$PROFILER_DIR"
  ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILER_DIR\"}")
fi

# Opt-in speculative decoding, e.g. SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":3}'
# (the pack ships the three DSpark MTP layers as mtp.*). Off by default.
if [[ -n "${SPEC_CONFIG:-}" ]]; then
  ARGS+=(--speculative-config "$SPEC_CONFIG")
fi

# Tool calling and reasoning separation, on by default so agent harnesses
# (Hermes, OpenClaw, Cline, ...) can send tools with tool_choice "auto" and
# receive the model's thinking in reasoning_content instead of the answer.
# The nightly ships DeepSeek-V4 parsers under the name "deepseek_v4".
# Set TOOL_CALL_PARSER="" or REASONING_PARSER="" to turn either off.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER-deepseek_v4}"
REASONING_PARSER="${REASONING_PARSER-deepseek_v4}"
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$REASONING_PARSER" ]]; then
  ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

echo "vllm ${ARGS[*]}"
vllm "${ARGS[@]}" &
SERVE_PID=$!

up=0
for _ in $(seq 1 130); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then up=1; break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then break; fi
  sleep 10
done
if [[ "$up" != 1 ]]; then
  echo "server did not come up (the ~95 GB load takes ~10-12 minutes on NVMe)" >&2
  exit 1
fi

echo "server up on :${PORT}; smoke:"
curl -s --max-time 120 "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"${SERVED_NAME}\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":0}" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(repr(d["choices"][0]["text"]))'

wait "$SERVE_PID"
