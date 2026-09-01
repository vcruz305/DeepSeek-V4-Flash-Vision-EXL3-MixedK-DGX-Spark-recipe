#!/usr/bin/env python3
"""Disk-vs-resident memory census for an EXL3 pack under vLLM.

Loads the model at a tiny context (explicit 2 GiB KV, enforce-eager) and prints
three tables: per-component bytes **on disk**, per-component bytes **resident**
after load, and resident bytes **by dtype** — plus the CUDA allocator totals.

When to use: a load OOMs, a boot dies mid-warmup, or resident size is far from
disk size. The disk table prints before the load starts, so even a failing boot
tells you what the checkpoint holds; a completed run tells you which component
ballooned (this tool localized every defect in docs/LOADER_NOTES.md). Expect
~10-12 minutes for the load itself on NVMe.

Usage: python scripts/memory_census.py /path/to/pack
"""

import glob
import json
import os
import struct
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(2)
MD = os.path.expanduser(sys.argv[1])


def bucket(name: str) -> str:
    parts = ["*" if p.isdigit() else p for p in name.split(".")]
    return ".".join(parts[:5])


# ---- disk census (always prints, even if the load later fails) ----
disk = defaultdict(int)
for fn in sorted(glob.glob(os.path.join(MD, "*.safetensors"))):
    with open(fn, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        disk[bucket(k)] += b - a
print("== DISK top ==")
for k, v in sorted(disk.items(), key=lambda x: -x[1])[:15]:
    print(f"D {v/1e9:8.2f}GB {k}")
print(f"D TOTAL {sum(disk.values())/1e9:.2f}GB")
sys.stdout.flush()

# ---- resident census ----
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch  # noqa: E402

# Drop the pack's page cache: on unified-memory boxes (GB10) cached shards
# shrink CUDA-free below vLLM's startup check even though they are reclaimable.
for fn in glob.glob(os.path.join(MD, "*")):
    if os.path.isfile(fn):
        try:
            fd = os.open(fn, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass
free, total = torch.cuda.mem_get_info()
print(f"CUDA free after fadvise {free/1e9:.1f}GB / {total/1e9:.1f}GB")
sys.stdout.flush()

from vllm import LLM  # noqa: E402

kw = dict(model=MD, quantization="exl3", load_format="auto", max_model_len=2048,
          enforce_eager=True, kv_cache_dtype="fp8",
          trust_remote_code=True, max_num_seqs=1)
try:
    import dataclasses

    from vllm.engine.arg_utils import EngineArgs

    fields = {f.name for f in dataclasses.fields(EngineArgs)}
except Exception:
    fields = set()
if "kv_cache_memory_bytes" in fields:
    kw["kv_cache_memory_bytes"] = 2 * 1024**3
elif "kv_cache_memory" in fields:
    kw["kv_cache_memory"] = 2 * 1024**3
else:
    kw["gpu_memory_utilization"] = 0.97
print("LLM kwargs extra:",
      {k: v for k, v in kw.items() if "kv_cache" in k or "util" in k})
sys.stdout.flush()
llm = LLM(**kw)


def census(m):
    agg = defaultdict(int)
    dt = defaultdict(int)
    for coll in (m.named_parameters(), m.named_buffers()):
        for n, p in coll:
            b = p.numel() * p.element_size()
            agg[bucket(n)] += b
            dt[str(p.dtype)] += b
    print("== RESIDENT top ==")
    for k, v in sorted(agg.items(), key=lambda x: -x[1])[:15]:
        print(f"R {v/1e9:8.2f}GB {k}")
    print("== BY DTYPE ==")
    for k, v in sorted(dt.items(), key=lambda x: -x[1]):
        print(f"T {v/1e9:8.2f}GB {k}")
    print(f"R PARAM+BUF TOTAL {sum(agg.values())/1e9:.2f}GB")


llm.apply_model(census)
print(f"CUDA allocated {torch.cuda.memory_allocated()/1e9:.2f}GB "
      f"reserved {torch.cuda.memory_reserved()/1e9:.2f}GB")
print("CENSUS_DONE")
