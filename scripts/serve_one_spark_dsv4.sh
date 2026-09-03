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

# Enforce-eager is the verified configuration (CUDA graphs untested on this
# model). No speculative decoding: the pack's MTP layers are skipped at load.
ARGS=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size 1
  --quantization exl3
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 1
  --max-num-batched-tokens 2048
  --kv-cache-dtype fp8
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enforce-eager
  --no-enable-prefix-caching
  --trust-remote-code
)

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
