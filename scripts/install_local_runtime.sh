#!/usr/bin/env bash
# Build and install the DeepSeek-V4 Flash Vision runtime on NVIDIA DGX Spark GB10 (sm_121).
# Clones canonical ExLlamaV3, applies the aarch64 CPU/intrinsics patch, builds exllamav3_ext,
# and installs the verified vLLM nightly wheel and vllm-exl3 plugin.
set -euo pipefail

RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/venvs/vllm-vl}"
PYTHON="${VENV}/bin/python"

# 1. Environment & Path configuration
if [[ -z "${CUDA_HOME:-}" ]]; then
  for d in /usr/local/cuda-13.0 /usr/local/cuda; do
    if [[ -d "$d" ]]; then export CUDA_HOME="$d"; echo "Set CUDA_HOME=$CUDA_HOME"; break; fi
  done
fi

if ! command -v nvcc >/dev/null 2>&1; then
  for d in "${CUDA_HOME:-}/bin" /usr/local/cuda-13.0/bin /usr/local/cuda/bin; do
    if [[ -x "$d/nvcc" ]]; then export PATH="$d:$PATH"; echo "Added $d to PATH"; break; fi
  done
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual environment not found at $VENV. Creating Python 3.12 venv..."
  python3.12 -m venv "$VENV" || python3 -m venv "$VENV"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${VENV}/bin:$PATH"

echo "=== DeepSeek-V4 Flash Vision Local Runtime Setup ==="
echo "Recipe root: $RECIPE_ROOT"
echo "Python: $("$PYTHON" --version) at $PYTHON"

# 2. Upgrade build prerequisites
"$PYTHON" -m pip install --upgrade pip setuptools wheel ninja

# 3. Install verified vLLM nightly wheel (PR #54566)
VLLM_WHEEL="https://wheels.vllm.ai/a56654d6de060495ff2db3b1d9ff0b187084d1a9/vllm-0.28.1rc1.dev324%2Bga56654d6d-cp38-abi3-manylinux_2_28_aarch64.whl"
echo "Installing vLLM nightly wheel..."
"$PYTHON" -m pip install "$VLLM_WHEEL"

# 4. Install companion packages
echo "Installing FlashInfer and vllm-exl3 plugin..."
"$PYTHON" -m pip install flashinfer-python==0.6.18 "vllm-exl3>=0.3.1"

# 5. Clone and patch ExLlamaV3 from canonical turboderp-org repository
EXLLAMAV3_SRC="/tmp/exllamav3_build"
EXLLAMAV3_REV="${EXLLAMAV3_REV:-master}"

echo "Fetching canonical ExLlamaV3 from turboderp-org..."
if [[ ! -d "$EXLLAMAV3_SRC/.git" ]]; then
  rm -rf "$EXLLAMAV3_SRC"
  git clone --depth 1 "https://github.com/turboderp-org/exllamav3.git" "$EXLLAMAV3_SRC"
else
  git -C "$EXLLAMAV3_SRC" fetch --depth 1 origin "$EXLLAMAV3_REV"
  git -C "$EXLLAMAV3_SRC" checkout -f "$EXLLAMAV3_REV"
fi

echo "Applying aarch64 CPU/intrinsics patch to ExLlamaV3..."
"$PYTHON" "$RECIPE_ROOT/scripts/patch_exllamav3_aarch64.py" "$EXLLAMAV3_SRC/exllamav3/exllamav3_ext"

# 6. Build and install ExLlamaV3 extension
echo "Building ExLlamaV3 extension with TORCH_CUDA_ARCH_LIST=12.1a..."
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
"$PYTHON" -m pip install --no-build-isolation "$EXLLAMAV3_SRC"

# 7. Apply runtime patches to vLLM
echo "Applying DeepSeek-V4 runtime patches..."
"$PYTHON" "$RECIPE_ROOT/scripts/patch_dsv4_stock028.py"
"$PYTHON" "$RECIPE_ROOT/scripts/patch_dsv4_vl_stream_load.py"
"$PYTHON" "$RECIPE_ROOT/scripts/patch_dsv4_vl_sm120_wide_swa.py"

# 8. Verify runtime installation
echo "Verifying runtime components..."
"$PYTHON" "$RECIPE_ROOT/scripts/verify_runtime.py"

echo ""
echo "=== Setup Complete! ==="
echo "Activate environment: source $VENV/bin/activate"
echo "Serve model: bash scripts/serve_one_spark_dsv4.sh"
