#!/usr/bin/env python3
"""Let the DeepSeek-V4 vision variant prefill on SM120 (DGX Spark GB10).

The vision model widens every prefill sliding-window index row from
``sliding_window`` (128) to ``sliding_window + vision_max_n_token`` (512) so
image spans attend bidirectionally. FlashInfer's SM120 sparse-MLA prefill
kernel is only compiled for a 128-wide window row when a compressed segment is
attached (``sparse_mla_sm120_prefill.cu``: ``if (topk != 128) return false``),
so the engine dies in the warm-up dummy run with
``Unsupported sparse-MLA prefill configuration: ... topk=512``.

This patch keeps the math exact: each wide row is split into 128-wide slices,
every slice runs through the same kernel (the compressed segment and the
attention sink ride on slice 0 so they are counted once) and the partial
softmax outputs are merged with their log-sum-exp. Rows that already fit in
128 entries take the unpatched path.

Idempotent, exact-anchor, backup in ``flashinfer_sparse.py.orig-wideswa``.

Usage: python patch_dsv4_vl_sm120_wide_swa.py [path/to/site-packages/vllm]
"""
import ast
import shutil
import sys
from pathlib import Path

MARK = "vcruz305 recipe patch: sliced SM120 prefill for wide window rows"

OLD_CALL = """            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
                workspace_buffer=self._get_workspace(q.device),
                sparse_indices=swa_indices_chunk,
                compressed_kv_cache=extra_kv_paged,
                out=output[query_start:query_end],
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens_chunk,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_topk_lens=extra_sparse_lengths_chunk,
            )
"""

NEW_CALL = """            if swa_indices_chunk.shape[-1] > _SM120_SWA_SLICE_WIDTH:
                # vcruz305 recipe patch: sliced SM120 prefill for wide window rows
                _sm120_sparse_mla_sliced_prefill(
                    q_chunk,
                    swa_kv_paged,
                    swa_indices_chunk,
                    swa_lens_chunk,
                    extra_kv_paged,
                    extra_sparse_indices_chunk,
                    extra_sparse_lengths_chunk,
                    output[query_start:query_end],
                    self.scale,
                    self.attn_sink,
                    self._get_workspace(q.device),
                )
                continue
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=q_chunk,
                swa_kv_cache=swa_kv_paged,
                workspace_buffer=self._get_workspace(q.device),
                sparse_indices=swa_indices_chunk,
                compressed_kv_cache=extra_kv_paged,
                out=output[query_start:query_end],
                bmm1_scale=self.scale,
                sinks=self.attn_sink,
                kv_layout="NHD",
                swa_topk_lens=swa_lens_chunk,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_topk_lens=extra_sparse_lengths_chunk,
            )
"""

HELPER = '''

# vcruz305 recipe patch: sliced SM120 prefill for wide window rows (helper)
_SM120_SWA_SLICE_WIDTH = 128


def _sm120_sparse_mla_sliced_prefill(
    q: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    compressed_kv_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    out: torch.Tensor,
    sm_scale: float,
    sinks: torch.Tensor | None,
    workspace_buffer: torch.Tensor,
) -> None:
    """Run the SM120 DSV4 sparse MLA kernel over window rows wider than 128.

    The kernel's dual-cache prefill path is compiled for a 128-wide window row
    only. The vision variant widens prefill rows to
    ``sliding_window + vision_max_n_token``. Each row is split into 128-wide
    slices (indices ``[T, W]`` or ``[T, 1, W]``; ``swa_lens`` is a valid-prefix
    length, so slice s keeps
    ``clamp(len - 128 s, 0, 128)`` entries); the compressed segment and the
    attention sink ride on slice 0 so they are counted once, and the partial
    softmax outputs are merged with their log-sum-exp. The result equals one
    softmax over the whole row plus the compressed entries.
    """
    from flashinfer.mla import _core as fi_core

    width = swa_indices.shape[-1]
    slice_w = _SM120_SWA_SLICE_WIDTH
    if width % slice_w != 0:
        raise ValueError(f"window index width {width} is not a multiple of {slice_w}")
    swa_kv = fi_core._check_sm120_dsv4_kv_cache_layout(swa_kv_cache, "NHD", "swa_kv_cache")
    comp_kv = None
    if compressed_kv_cache is not None:
        if extra_indices is None or extra_lens is None:
            raise ValueError("compressed cache given without compressed indices")
        comp_kv = fi_core._check_sm120_dsv4_kv_cache_layout(
            compressed_kv_cache, "NHD", "compressed_kv_cache"
        )
    num_tokens, num_heads = q.shape[0], q.shape[1]
    q4 = q.unsqueeze(1)
    neg_inf = float("-inf")
    m = torch.full((num_tokens, num_heads), neg_inf, dtype=torch.float32, device=q.device)
    acc = torch.zeros((num_tokens, num_heads, 512), dtype=torch.float32, device=q.device)
    denom = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    for s in range(width // slice_w):
        # indices may arrive as [T, W] or [T, 1, W] (lens as [T] or [T, 1]);
        # slice the window axis only and keep the caller's rank.
        idx = swa_indices[..., s * slice_w : (s + 1) * slice_w].contiguous()
        lens = (swa_lens - s * slice_w).clamp(0, slice_w).contiguous()
        segments = [fi_core._SparseMLASegment(indices=idx, lengths=lens)]
        valid = lens.reshape(num_tokens) > 0
        if s == 0 and comp_kv is not None:
            segments.append(
                fi_core._SparseMLASegment(
                    indices=extra_indices, lengths=extra_lens, kv_cache=comp_kv
                )
            )
            valid = valid | (extra_lens.reshape(num_tokens) > 0)
        o, lse = fi_core._trtllm_batch_decode_sparse_mla_sm120(
            query=q4,
            kv_cache=swa_kv,
            workspace_buffer=workspace_buffer,
            sparse_mla_segments=segments,
            out=None,
            sm_scale=float(sm_scale),
            sinks=sinks if s == 0 else None,
            lse=None,
            return_lse=True,
            kv_scale_format="auto",
        )
        lse = lse.reshape(num_tokens, num_heads).float()
        valid = valid.view(num_tokens, 1)
        lse = torch.where(valid, lse, torch.full_like(lse, neg_inf))
        m_new = torch.maximum(m, lse)
        finite = m_new > neg_inf
        a = torch.where(finite, torch.exp(m - m_new), torch.zeros_like(m))
        b = torch.where(finite, torch.exp(lse - m_new), torch.zeros_like(m))
        o = torch.nan_to_num(o.reshape(num_tokens, num_heads, 512).float())
        acc.mul_(a.unsqueeze(-1)).add_(o * b.unsqueeze(-1))
        denom.mul_(a).add_(b)
        m = m_new
    out.copy_((acc / denom.clamp_min(1e-30).unsqueeze(-1)).to(out.dtype))
'''


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        import vllm
        root = Path(vllm.__file__).parent
    target = root / "models" / "deepseek_v4" / "nvidia" / "flashinfer_sparse.py"
    if not target.exists():
        print(f"missing {target}", file=sys.stderr)
        return 2
    src = target.read_text(encoding="utf-8")
    if MARK in src:
        print(f"{target}: already patched")
        print("VL_WIDE_SWA_PATCH_OK")
        return 0
    n = src.count(OLD_CALL)
    if n != 1:
        print(f"{target}: expected 1 anchor, found {n}", file=sys.stderr)
        return 1
    if "import torch\n" not in src:
        print(f"{target}: no torch import", file=sys.stderr)
        return 1
    backup = target.with_name(target.name + ".orig-wideswa")
    if not backup.exists():
        shutil.copy2(target, backup)
    new_src = src.replace(OLD_CALL, NEW_CALL) + HELPER
    ast.parse(new_src)
    target.write_text(new_src, encoding="utf-8")
    print(f"{target}: patched (backup {backup.name})")
    print("VL_WIDE_SWA_PATCH_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
