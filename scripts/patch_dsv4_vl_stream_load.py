#!/usr/bin/env python3
"""Stream the DeepSeek-V4 vision loader instead of materializing the checkpoint.

vLLM's DeepseekV4ForConditionalGeneration.load_weights (vl_model.py) sorts the
whole mapped weight iterator before handing it to AutoWeightsLoader so that the
``language_model.`` group reaches the child loader as one contiguous block.
sorted() pulls every tensor of the checkpoint into memory first; on a GB10 with
unified memory that is the 95 GB checkpoint plus the 95 GB model, and the box
wedges before the load finishes. This patch keeps the contiguity guarantee by
streaming ``language_model.`` tensors straight through and deferring only the
small non-language tensors (vision tower, aligner, image specials) to the end.

Idempotent, exact-anchor, backup in ``vl_model.py.orig-vlstream``.

Usage: python patch_dsv4_vl_stream_load.py [path/to/site-packages/vllm]
"""
import ast
import shutil
import sys
from pathlib import Path

OLD = "        mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=lambda x: x[0])\n"
NEW = """        # vcruz305 recipe patch: stream instead of sorted() (which materializes
        # the whole checkpoint). language_model.* tensors pass straight through
        # and stay contiguous because every other tensor is deferred to the end.
        def _stream_language_model_first(mapped_iter):
            deferred = []
            for name, weight in mapped_iter:
                if name.startswith("language_model."):
                    yield name, weight
                else:
                    deferred.append((name, weight))
            yield from deferred

        mapped = _stream_language_model_first(self.hf_to_vllm_mapper.apply(weights))
"""
MARK = "vcruz305 recipe patch: stream instead of sorted()"


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        import vllm
        root = Path(vllm.__file__).parent
    target = root / "models" / "deepseek_v4" / "nvidia" / "vl_model.py"
    if not target.exists():
        print(f"missing {target}: this vLLM has no DeepSeek-V4 vision model", file=sys.stderr)
        return 2
    src = target.read_text(encoding="utf-8")
    if MARK in src:
        print(f"{target}: already patched")
        print("VL_STREAM_PATCH_OK")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"{target}: expected 1 anchor, found {n}", file=sys.stderr)
        return 1
    backup = target.with_name(target.name + ".orig-vlstream")
    if not backup.exists():
        shutil.copy2(target, backup)
    new_src = src.replace(OLD, NEW)
    ast.parse(new_src)
    target.write_text(new_src, encoding="utf-8")
    print(f"{target}: patched (backup {backup.name})")
    print("VL_STREAM_PATCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
