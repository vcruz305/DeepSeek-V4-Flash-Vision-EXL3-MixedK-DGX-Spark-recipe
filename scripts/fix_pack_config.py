#!/usr/bin/env python3
"""Repair a local DSV4 EXL3 MixedK pack's ``config.json`` for vLLM serving.

The pack must declare EXL3 in ``quantization_config`` — vLLM resolves the
quantization method from the model config before the CLI flag, and a leftover
base-model fp8 declaration silently overrides ``--quantization exl3`` (the
engine then builds the wrong MoE scaffolding and OOMs). This script:

- backs up ``config.json`` to ``config.json.bak`` (first run only),
- writes the EXL3 ``quantization_config`` (method, codebook, per-layer
  ``layer_bits`` **scanned from the actual shard trellis shapes**, and the
  ``non_routed_quantization`` delegation block),
- removes an empty ``text_config`` key if present (the config is flat; an
  empty nested dict breaks vLLM's text-config extraction),
- touches nothing else.

Idempotent. Packs downloaded from the Hub after 2026-09-01 already ship this.

Usage: python scripts/fix_pack_config.py /path/to/DSV4-Flash-Vision-ablit-EXL3-MixedK
"""

import glob
import json
import re
import shutil
import struct
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_BITS = 2
TRELLIS_DIM_TO_BITS = {16: 1, 32: 2, 48: 3, 64: 4, 80: 5}


def scan_layer_bits(pack: Path) -> dict[str, int]:
    per_layer: dict[int, set[int]] = defaultdict(set)
    shards = sorted(glob.glob(str(pack / "*.safetensors")))
    if not shards:
        raise SystemExit(f"no safetensors shards under {pack}")
    pat = re.compile(r"layers\.(\d+)\.ffn\.experts\.\d+\.w[123]\.trellis$")
    for fn in shards:
        with open(fn, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            m = pat.match(name)
            if m:
                per_layer[int(m.group(1))].add(meta["shape"][-1])
    out: dict[str, int] = {}
    for layer, dims in sorted(per_layer.items()):
        assert len(dims) == 1, (
            f"layer {layer} mixes trellis widths {sorted(dims)}; "
            "per-layer bits are not expressible for this pack")
        bits = TRELLIS_DIM_TO_BITS.get(next(iter(dims)))
        assert bits is not None, (
            f"layer {layer}: unknown trellis width {next(iter(dims))}")
        if bits != DEFAULT_BITS:
            out[str(layer)] = bits
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    pack = Path(sys.argv[1]).expanduser()
    cfg_path = pack / "config.json"
    if not cfg_path.is_file():
        print(f"missing {cfg_path}")
        return 1

    cfg = json.loads(cfg_path.read_text())
    layer_bits = scan_layer_bits(pack)
    want = {
        "quant_method": "exl3",
        "bits": DEFAULT_BITS,
        "codebook": "mcg",
        "head_bits": 16,
        "non_routed_dtype_policy": "official_source_native",
        "scope": "dsv4_routed_experts_only",
        "serving_reader_qualified": False,
        "version": "0.0.43",
        "layer_bits": layer_bits,
        "non_routed_quantization": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }

    changed = False
    if cfg.get("quantization_config") != want:
        cfg["quantization_config"] = want
        changed = True
    if cfg.get("text_config") == {}:
        cfg.pop("text_config")
        changed = True

    if not changed:
        print("config already correct; nothing to do")
        return 0

    bak = cfg_path.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
        print(f"backed up original to {bak.name}")
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {cfg_path}")
    print(f"  layer_bits (non-default layers): {layer_bits or '{} (uniform)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
