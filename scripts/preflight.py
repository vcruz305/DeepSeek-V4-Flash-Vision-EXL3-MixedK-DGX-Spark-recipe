#!/usr/bin/env python3
"""Answer 'can this environment serve the DeepSeek-V4 pack?' in under a second.

Run this BEFORE downloading 95 GiB or starting the server. Validates that the
hardware, Python ABI, CUDA toolkit, and runtime libraries satisfy the requirements
for serving DeepSeek-V4 Flash Vision on NVIDIA DGX Spark GB10 (sm_121).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path


class Result:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []
        self.fatal = False

    def check(self, name: str, ok: bool, detail: str, fatal: bool = True) -> bool:
        self.rows.append((name, ok, detail))
        if not ok and fatal:
            self.fatal = True
        return ok

    def report(self) -> int:
        width = max(len(n) for n, _, _ in self.rows) if self.rows else 20
        for name, ok, detail in self.rows:
            print(f"  [{'OK ' if ok else 'X  '}] {name.ljust(width)}  {detail}")
        return 1 if self.fatal else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / "models" / "DSV4-Flash-Vision-ablit-EXL3-MixedK",
        help="Checked only if it exists; preflight never requires the weights on disk",
    )
    parser.add_argument(
        "--allow-non-gb10",
        action="store_true",
        help="Treat missing GB10 SM121 hardware as a warning rather than a fatal error",
    )
    args = parser.parse_args()
    r = Result()

    print("DeepSeek-V4 Flash Vision EXL3 preflight check\n")

    # 1. Architecture
    machine = platform.machine().lower()
    r.check(
        "arch aarch64",
        machine in {"aarch64", "arm64"},
        machine if machine in {"aarch64", "arm64"} else f"{machine} (DGX Spark requires ARM64 aarch64)",
        fatal=not args.allow_non_gb10,
    )

    # 2. Python version
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    r.check(
        "python 3.12",
        sys.version_info[:2] == (3, 12),
        f"{py} (DGX Spark wheels and CUDA 13 toolchain require Python 3.12)",
    )

    # 3. Build tools
    ninja = shutil.which("ninja") or (
        os.path.join(sys.prefix, "bin", "ninja")
        if os.path.exists(os.path.join(sys.prefix, "bin", "ninja"))
        else None
    )
    r.check(
        "ninja available",
        ninja is not None,
        ninja or "not found: pip install ninja into the venv or put on PATH",
    )

    nvcc = shutil.which("nvcc")
    if not nvcc:
        for d in ("/usr/local/cuda-13.0/bin/nvcc", "/usr/local/cuda/bin/nvcc"):
            if os.path.isfile(d) and os.access(d, os.X_OK):
                nvcc = d
                break
    r.check(
        "nvcc on PATH",
        nvcc is not None,
        nvcc or "not found: export PATH=/usr/local/cuda-13.0/bin:$PATH",
    )

    # 4. PyTorch & CUDA
    try:
        import torch
    except ImportError:
        r.check("torch importable", False, "not installed (pip install torch)", fatal=True)
        torch = None
    else:
        cuda = torch.version.cuda or "none"
        r.check(
            "torch CUDA 13",
            cuda.startswith("13."),
            f"{torch.__version__} cuda={cuda}",
            fatal=not args.allow_non_gb10,
        )
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            r.check(
                "GB10 SM121",
                cap == (12, 1),
                f"{torch.cuda.get_device_name()} capability {cap[0]}.{cap[1]}",
                fatal=not args.allow_non_gb10,
            )
        else:
            r.check(
                "CUDA available",
                False,
                "torch.cuda.is_available() is False",
                fatal=not args.allow_non_gb10,
            )

    # 5. Runtime dependencies
    vllm_spec = importlib.util.find_spec("vllm")
    r.check(
        "vllm installed",
        vllm_spec is not None,
        "installed" if vllm_spec else "not installed (install verified nightly wheel)",
    )

    fi_spec = importlib.util.find_spec("flashinfer")
    r.check(
        "flashinfer installed",
        fi_spec is not None,
        "installed" if fi_spec else "not installed (pip install flashinfer-python==0.6.18)",
    )

    exl3_spec = importlib.util.find_spec("vllm_exl3")
    r.check(
        "vllm-exl3 plugin",
        exl3_spec is not None,
        "installed" if exl3_spec else "not installed (pip install 'vllm-exl3>=0.3.1')",
    )

    # 6. Model dir check if present
    if args.model_dir.is_dir():
        cfg_path = args.model_dir / "config.json"
        if cfg_path.is_file():
            import json
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                qm = (cfg.get("quantization_config") or {}).get("quant_method")
                r.check("model pack quant_method", qm == "exl3", f"quant_method={qm}")
            except Exception as e:
                r.check("model pack config valid", False, str(e))
        else:
            r.check("model pack config.json", False, f"missing {cfg_path}")

    print()
    code = r.report()
    if code != 0:
        print("\n[!] Preflight identified issues. Please resolve the above items before serving.")
    else:
        print("\n[OK] Preflight checks passed. Environment is ready to serve.")
    return code


if __name__ == "__main__":
    sys.exit(main())
