#!/usr/bin/env python3
"""Patch the fork's DeepSeek-V4 text model to serve the EXL3 MixedK pack.

Three exact-match, idempotent edits to
``vllm/models/deepseek_v4/nvidia/model.py``:

1. **Checkpoint skip filter** in ``DeepseekV4ForCausalLM.load_weights``:
   the text-only class has no modules for ``vision.*`` / ``aligner.*`` /
   ``image_*`` specials or the ``mtp.*`` draft layers, and the strict loader
   raises on them. Also skipped: ``gate.bias_vl`` (unmapped in the text path)
   and ``gate.bias`` on hash-routed layers (< ``num_hash_layers``), which use
   ``tid2eid`` tables instead of a correction bias.

2. **Load-time block-FP8 quantization** of non-routed weights: the pack keeps
   them BF16 on disk, but the model's forward path is fp8-specialized (the
   deep-GEMM o-proj einsum and shared-expert path read ``weight_scale_inv``
   unconditionally). Any 2-D BF16/FP16 ``.weight`` destined for a
   ``float8_e4m3fn`` parameter is quantized with real 128x128 block scales
   (``scale = amax / 448``) and emitted as ``weight`` + ``weight_scale_inv``,
   which the standard loaders already understand. Fused destinations
   (``fused_wqa_wkv``, ``compressor.fused_wkv_wgate``, shared-expert
   ``gate_up_proj`` / ``down_proj``) are resolved by name remap.

3. **Gate-bias default**: the pack's slimmed config omits ``topk_method``, so
   the router's ``e_score_correction_bias`` parameter is never created and its
   checkpoint tensor has no home. The branch defaults to ``noaux_tc`` — the
   checkpoint carrying the bias tensors is the evidence for the routing mode.

Usage: python scripts/patch_dsv4_loader.py [path/to/model.py]
Defaults to the file inside the installed vllm package. Safe to re-run.
"""

import ast
import sys
from pathlib import Path

SIG = (
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]])"
    " -> set[str]:\n"
)

LOADER_BLOCK = '''        _nhash = getattr(self.config, "num_hash_layers", 0) or 0

        def _skip_ckpt(n, _nh=_nhash):
            if n.startswith(("vision.", "aligner.", "mtp.", "image_")):
                return True
            if n.endswith(".ffn.gate.bias_vl"):
                return True  # vision-language gate bias; text model has no param
            if n.endswith(".ffn.gate.bias"):
                seg = n.split(".")
                if len(seg) > 1 and seg[1].isdigit() and int(seg[1]) < _nh:
                    return True  # hash-routed layers have no e_score_correction_bias
            return False

        _pd0 = dict(self.named_parameters())
        _fp8w = {
            n for n, p in _pd0.items()
            if p.dtype == torch.float8_e4m3fn and n.endswith(".weight")
        }
        _remap = (("attn.wq_a", "attn.fused_wqa_wkv"),
                  ("attn.wkv", "attn.fused_wqa_wkv"),
                  ("compressor.wkv", "compressor.fused_wkv_wgate"),
                  ("compressor.wgate", "compressor.fused_wkv_wgate"),
                  ("shared_experts.w1", "shared_experts.gate_up_proj"),
                  ("shared_experts.w3", "shared_experts.gate_up_proj"),
                  ("shared_experts.w2", "shared_experts.down_proj"))

        def _fp8_dest(n):
            cands = ["model." + n]
            for a, b in _remap:
                if a in n:
                    cands.append("model." + n.replace(a, b))
            return any(c in _fp8w for c in cands)

        def _q128(w):
            O, I = w.shape[-2], w.shape[-1]
            ob, ib = -(-O // 128), -(-I // 128)
            wf = w.to(torch.float32)
            pad = torch.zeros(ob * 128, ib * 128, dtype=torch.float32)
            pad[:O, :I] = wf
            blocks = pad.view(ob, 128, ib, 128)
            amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
            scale = amax / 448.0
            q = (blocks / scale[:, None, :, None]).clamp(-448, 448)
            qw = q.view(ob * 128, ib * 128)[:O, :I].to(torch.float8_e4m3fn)
            return qw, scale.to(torch.float32)

        def _bf16_to_fp8_stream(ws):
            for n, w in ws:
                if (n.endswith(".weight") and w.dim() == 2
                        and w.dtype in (torch.bfloat16, torch.float16)
                        and _fp8_dest(n)):
                    qw, sc = _q128(w)
                    yield n, qw
                    yield n + "_scale_inv", sc
                    continue
                yield n, w

        weights = _bf16_to_fp8_stream(
            (n, w) for (n, w) in weights if not _skip_ckpt(n))
'''

GATE_OLD = 'elif getattr(config, "topk_method", None) == "noaux_tc":'
GATE_NEW = 'elif getattr(config, "topk_method", "noaux_tc") == "noaux_tc":'


def default_target() -> Path:
    import vllm

    return (Path(vllm.__file__).parent
            / "models" / "deepseek_v4" / "nvidia" / "model.py")


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_target()
    if not target.is_file():
        print(f"target not found: {target}")
        return 1
    s = target.read_text()
    changed = []

    # --- edit 1+2: skip filter + fp8 quant stream in ForCausalLM.load_weights
    if "_bf16_to_fp8_stream" in s and "_skip_ckpt" in s:
        print("loader block: already applied")
    else:
        assert "_skip_ckpt" not in s and "_bf16_to_fp8_stream" not in s, (
            "partial prior application; refusing to guess — restore the file "
            "from the runtime wheel and re-run")
        import re

        cls = re.search(r"^class DeepseekV4ForCausalLM\b", s, re.M)
        assert cls, "class DeepseekV4ForCausalLM not found"
        nxt = re.search(r"^class \w+", s[cls.end():], re.M)
        end = cls.end() + (nxt.start() if nxt else len(s) - cls.end())
        seg = s[cls.start():end]
        assert seg.count(SIG) == 1, (
            f"expected exactly one load_weights in ForCausalLM, got "
            f"{seg.count(SIG)}")
        seg = seg.replace(SIG, SIG + LOADER_BLOCK, 1)
        s = s[:cls.start()] + seg + s[end:]
        changed.append("loader block (skip filter + block-FP8 quant stream)")

    # --- edit 3: gate-bias default
    if GATE_NEW in s:
        print("gate-bias default: already applied")
    else:
        n = s.count(GATE_OLD)
        assert n >= 1, "gate condition anchor not found"
        s = s.replace(GATE_OLD, GATE_NEW)
        changed.append(f"gate-bias noaux_tc default ({n} site{'s'[:n > 1]})")

    ast.parse(s)
    target.write_text(s)
    print(f"patched {target}")
    for c in changed:
        print("  +", c)
    if not changed:
        print("  (no changes needed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
