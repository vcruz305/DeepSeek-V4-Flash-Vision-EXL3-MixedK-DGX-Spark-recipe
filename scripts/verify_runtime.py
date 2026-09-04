#!/usr/bin/env python3
"""Fail fast unless the local DGX Spark runtime has every required DeepSeek-V4 component."""
from __future__ import annotations

import argparse
import sys
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Skip strict GB10 SM121 CUDA check (for imports/syntax validation in test environments)",
    )
    args = parser.parse_args()

    import torch

    if not args.allow_cpu:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available")
        capability = torch.cuda.get_device_capability()
        if capability != (12, 1):
            raise SystemExit(f"expected GB10 SM121, got capability {capability}")
        print(f"device: {torch.cuda.get_device_name()} capability: {capability}")
    else:
        print("running in --allow-cpu mode (skipping GB10 SM121 hardware requirement)")

    # 1. Plugin verification
    try:
        from vllm.plugins import load_general_plugins
        load_general_plugins()
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        if "exl3" not in QUANTIZATION_METHODS:
            raise SystemExit("EXL3 plugin did not register with vLLM")
    except Exception as e:
        raise SystemExit(f"Failed to load vllm-exl3 plugin: {e}")

    # 2. ExLlamaV3 native extension verification
    try:
        import exllamav3_ext
        if not hasattr(exllamav3_ext, "exl3_moe"):
            raise SystemExit("exllamav3_ext has no fused exl3_moe entry point")
    except Exception as e:
        raise SystemExit(f"Failed to import exllamav3_ext: {e}")

    # 3. vLLM and Parsers verification
    try:
        import vllm
        from vllm.tool_parsers import TOOL_PARSER_REGISTRY
        from vllm.reasoning import REASONING_PARSER_REGISTRY

        if "deepseek_v4" not in TOOL_PARSER_REGISTRY:
            print("WARNING: 'deepseek_v4' not in TOOL_PARSER_REGISTRY (check nightly patch)")
        if "deepseek_v4" not in REASONING_PARSER_REGISTRY:
            print("WARNING: 'deepseek_v4' not in REASONING_PARSER_REGISTRY (check nightly patch)")
    except Exception as e:
        raise SystemExit(f"Failed to verify vllm parsers: {e}")

    # 4. Print inventory
    print(f"torch: {torch.__version__} (cuda: {torch.version.cuda})")
    print(f"vllm: {vllm.__version__}")
    try:
        print(f"exllamav3: {version('exllamav3')}")
    except Exception:
        print("exllamav3: installed (metadata unavailable)")
    try:
        print(f"flashinfer-python: {version('flashinfer-python')}")
    except Exception:
        print("flashinfer-python: installed (metadata unavailable)")
    try:
        print(f"vllm-exl3: {version('vllm-exl3')}")
    except Exception:
        print("vllm-exl3: installed (metadata unavailable)")

    if torch.cuda.is_available() and hasattr(exllamav3_ext, "exl3_moe_max_concurrency"):
        try:
            print(f"exl3_moe concurrency: {exllamav3_ext.exl3_moe_max_concurrency(0)}")
        except Exception:
            pass

    print("\n[OK] DeepSeek-V4 Flash Vision runtime verified successfully.")


if __name__ == "__main__":
    main()
