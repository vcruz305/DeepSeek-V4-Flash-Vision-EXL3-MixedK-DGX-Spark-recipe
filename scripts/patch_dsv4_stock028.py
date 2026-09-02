#!/usr/bin/env python3
"""Patch stock vLLM 0.28 (nvidia DeepSeek-V4 path) to serve the EXL3 MixedK pack
with its non-routed weights kept BF16, as stored in the pack.

Three exact-match, idempotent edits under ``vllm/models/deepseek_v4/nvidia/``:

1. ``flashinfer_sparse.py`` ``_o_proj`` (both attention classes): when
   ``wo_a`` is not ``float8_e4m3fn`` the deep-GEMM fp8 einsum has no block
   scales to read, so route through the Triton inverse-RoPE + bf16 einsum
   reference (``rocm_inv_rope_einsum``, device-agnostic despite the name)
   followed by the regular ``wo_b`` linear.

2. ``model.py`` ``load_weights``: hash-routed layers (< ``num_hash_layers``)
   have no ``e_score_correction_bias`` parameter, but the checkpoint ships one;
   skip it instead of raising ``KeyError``.

3. ``dspark.py`` (DSpark MTP draft) ``load_weights``: skip ``gate.bias_vl``
   (vision-token routing bias, unused, unmapped in the draft loader).

Pair this with the pack config's ``non_routed_dtype_policy: "bf16_as_stored"``
(``scripts/fix_pack_config.py``) and vllm-exl3 >= 0.2.3, which returns the
unquantized linear method for dense layers under that policy while still
delegating the draft's source-format routed experts.

Usage: python scripts/patch_dsv4_stock028.py [path/to/site-packages/vllm]
Defaults to the installed vllm package. Safe to re-run. Backups: ``*.orig``.
"""

import shutil
import sys
from pathlib import Path

OPROJ_ANCHOR = (
    "    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor)"
    " -> torch.Tensor:\n"
    "        return deep_gemm_fp8_o_proj(\n"
)
OPROJ_PATCHED = (
    "    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor)"
    " -> torch.Tensor:\n"
    "        if self.wo_a.weight.dtype != torch.float8_e4m3fn:\n"
    "            # bf16 wo_a (packs that keep non-routed weights unquantized):\n"
    "            # the fp8 einsum path needs block scales that do not exist, so\n"
    "            # use the Triton inverse-RoPE + bf16 einsum reference instead.\n"
    "            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (\n"
    "                rocm_inv_rope_einsum,\n"
    "            )\n"
    "\n"
    "            z = rocm_inv_rope_einsum(\n"
    "                self.rotary_emb,\n"
    "                o,\n"
    "                positions,\n"
    "                self.rope_head_dim,\n"
    "                self.n_local_groups,\n"
    "                self.o_lora_rank,\n"
    "                self.wo_a,\n"
    "            )\n"
    "            return self.wo_b(z.flatten(1))\n"
    "        return deep_gemm_fp8_o_proj(\n"
)

MODEL_ANCHOR = (
    "                    if is_pp_missing_parameter(name, self):\n"
    "                        continue\n"
    "                    param = params_dict[name]\n"
)
MODEL_PATCHED = (
    "                    if is_pp_missing_parameter(name, self):\n"
    "                        continue\n"
    "                    if (\n"
    "                        name.endswith(\".ffn.gate.e_score_correction_bias\")\n"
    "                        and name not in params_dict\n"
    "                    ):\n"
    "                        # Hash-routed layers (< num_hash_layers) have no\n"
    "                        # correction bias; some checkpoints still ship one.\n"
    "                        continue\n"
    "                    param = params_dict[name]\n"
)

DSPARK_ANCHOR = (
    "            name = mapped\n"
    "            if \"confidence_head.\" in name:\n"
)
DSPARK_PATCHED = (
    "            name = mapped\n"
    "            if name.endswith(\".ffn.gate.bias_vl\"):\n"
    "                # Vision-token routing bias; unused here, as in the target loader.\n"
    "                continue\n"
    "            if \"confidence_head.\" in name:\n"
)

EDITS = [
    ("flashinfer_sparse.py", OPROJ_ANCHOR, OPROJ_PATCHED, 2),
    ("model.py", MODEL_ANCHOR, MODEL_PATCHED, 1),
    ("dspark.py", DSPARK_ANCHOR, DSPARK_PATCHED, 1),
]


def patch_file(path: Path, anchor: str, patched: str, expect: int) -> str:
    text = path.read_text()
    have = text.count(patched)
    if have == expect:
        return "already patched"
    if have:
        raise SystemExit(f"{path}: partially patched ({have}/{expect}); restore .orig first")
    n = text.count(anchor)
    if n != expect:
        raise SystemExit(f"{path}: expected {expect} anchor(s), found {n}")
    orig = path.with_suffix(path.suffix + ".orig")
    if not orig.exists():
        shutil.copy2(path, orig)
    path.write_text(text.replace(anchor, patched))
    return f"patched x{expect}"


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        import vllm  # noqa: WPS433

        root = Path(vllm.__file__).resolve().parent
    d = root / "models" / "deepseek_v4" / "nvidia"
    if not d.is_dir():
        print(f"missing {d}")
        return 1
    for fname, anchor, patched, expect in EDITS:
        print(f"{fname}: {patch_file(d / fname, anchor, patched, expect)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
