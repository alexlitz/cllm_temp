"""BLOCK-0 FUSED FFN (C4_BLOCK0_FUSED_FFN) — RUNG 3: fold block-0's DENSE SwiGLU
FFN into the fused-delta COO sparse path.

CONTEXT (this task, RUNG 3).  After the composed stack + RUNG 2 whole-step graph,
the NEW dominant *device* GEMM is block-0's ingest attention (W_o) + its dense
``SparseFFN`` (``_agent_composed_floor`` TASK 4 measured ~18.8 us/step of
``ampere_sgemm``, of which the FFN half is the dense ``[K,D]@[D,Dff]`` +
``[K,Dff]@[Dff,D]`` SwiGLU).  Block-0's FFN is EXTREMELY sparse (probed: Dff=33,
W_up 33 nnz == 1/row, W_down 33 nnz over just 6 ACTIVE output rows, b_down all-zero),
so the dense GEMM wastes ~99.98% of its FLOPs multiplying zeros — the same waste the
``[cut,N)`` dead-FFN blocks pay before the fused megablock ate them.

THE LEVER.  Swap block-0's ``.ffn`` for the SAME byte-exact ``FusedUpGateSiluDeltaFFN``
COO kernel the megablock uses on the dead-FFN chain: kernel-1 fuses up+gate+silu over
the 33 nonzeros; kernel-2 adds ``W_down @ hidden`` to ONLY the 6 residual rows W_down
writes, in place.  No dense ``[K, D] @ [D, Dff]`` GEMM, no ``[K, Dff]`` HBM hidden over
the full D.  Because block-0's FFN is captured INSIDE ``Block0ChunkGraph`` (RUNG 1) /
the whole-step graph (RUNG 2), the swap must happen BEFORE the graph captures — the
graph then bakes the sparse Triton kernels instead of the dense cuBLAS GEMM.

BYTE-EXACT.  ``FusedUpGateSiluDeltaFFN.forward`` computes the SAME nonzeros in the SAME
per-output accumulation order as ``SparseFFN.forward`` (zeros contribute 0); only the
fp-accumulation-order residue vs cuBLAS (~6e-2 on ~5-magnitude values) differs, far
below the integer nibble-snap decode margin — exactly as the megablock / fused-delta
paths already proven byte-exact.  b_down==0 is asserted (probed).  DEFAULT OFF -> the
dense ``SparseFFN.forward`` (golden 069cc32f unchanged; the SWAP is a runtime
``.forward`` rebind that leaves the stored ``W_up/W_gate/W_down`` byte-identical —
verified: the FFN-weight hash is unchanged across install + uninstall).

HONEST MEASURED OUTCOME (this task).  BYTE-EXACT (30+-program battery, all K), but the
block-0 device sgemm barely moves: 15.68 -> 15.53 us/step (-0.15 us).  Block-0's FFN is
so tiny (Dff=33) that its dense GEMM was NEVER the cost — the ~15.5 us block-0 sgemm is
its W_o ATTENTION projection ([K, D=1416] @ [D, D]), not the FFN.  So RUNG 3 correctly
folds the FFN into the sparse path (the right mechanism, byte-exact, composes with the
block-0 / whole-step graph) but the honest device win is negligible: the next real
block-0 lever is the W_o attention GEMM, not the FFN.
"""
from __future__ import annotations

import os
from typing import Optional

import torch


def block0_fused_ffn_enabled() -> bool:
    """``C4_BLOCK0_FUSED_FFN`` (DEFAULT OFF): fold block-0's dense SwiGLU FFN into the
    fused-delta COO sparse kernel (byte-exact at the nibble margin).  OFF -> the dense
    ``SparseFFN.forward`` (golden path)."""
    return os.environ.get("C4_BLOCK0_FUSED_FFN", "0") not in ("0", "", "false", "False")


class _Block0SparseFFNWrap:
    """Wraps block-0's dense ``SparseFFN`` with the fused-delta COO kernel while
    keeping the ``.forward(x)->x_out`` residual-add signature block0_graph / the eager
    block-0 path call.  Falls back to the 2-launch upgate form if b_down != 0."""

    def __init__(self, ffn, device, block_k: int = 256):
        from .fused_sparse_ffn import (FusedUpGateSiluDeltaFFN, FusedUpGateSiluFFN,
                                       _dense_of)
        bd = ffn.b_down.to(device).float() if hasattr(ffn, "b_down") else None
        b_down_zero = bd is None or bool((bd != 0).sum() == 0)
        # delta-inplace requires b_down==0 (asserted inside); else the 2-launch form.
        if b_down_zero:
            self._impl = FusedUpGateSiluDeltaFFN(ffn, device, block_k=block_k)
        else:
            self._impl = FusedUpGateSiluFFN(ffn, device, block_k=block_k)
        self.dim = self._impl.dim
        self.Dff = self._impl.Dff
        self._orig_forward = ffn.forward
        # keep the original weight tensors reachable (uninstall / other consumers).
        for a in ("W_up", "W_gate", "W_down", "b_up", "b_gate", "b_down"):
            if hasattr(ffn, a):
                setattr(self, a, getattr(ffn, a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl.forward(x)


def install_block0_fused_ffn(model, device, block_idx: int = 0, verbose: bool = False):
    """Swap block ``block_idx``'s FFN for the fused-delta COO sparse kernel.

    Must be called BEFORE block0_graph / whole-step-graph captures (so the graph bakes
    the sparse kernels).  Byte-exact at the nibble margin.  Returns the wrap (kept alive
    by the caller); stores the original ``.forward`` on the block so uninstall restores
    the dense path."""
    if not block0_fused_ffn_enabled():
        return None
    blk = model.blocks[block_idx]
    ffn = blk.ffn
    if getattr(ffn, "_block0_fused_installed", False):
        return getattr(ffn, "_block0_fused_wrap", None)
    # ensure the dense weights are materialised for the CSR pack.
    if hasattr(model, "materialize_dense"):
        model.materialize_dense(device=str(device))
    wrap = _Block0SparseFFNWrap(ffn, device)
    ffn._block0_fused_orig_forward = ffn.forward
    ffn.forward = wrap.forward          # bound method on the wrap instance
    ffn._block0_fused_wrap = wrap
    ffn._block0_fused_installed = True
    if verbose:
        print(f"[block0-fused-ffn] block {block_idx}: dense SwiGLU -> fused-delta COO "
              f"(Dff={wrap.Dff}, {type(wrap._impl).__name__})", flush=True)
    return wrap


def uninstall_block0_fused_ffn(model, block_idx: int = 0):
    for bi in ([block_idx] if isinstance(block_idx, int) else block_idx):
        ffn = model.blocks[bi].ffn
        if getattr(ffn, "_block0_fused_installed", False):
            ffn.forward = ffn._block0_fused_orig_forward
            del ffn._block0_fused_orig_forward
            del ffn._block0_fused_wrap
            ffn._block0_fused_installed = False


__all__ = ["block0_fused_ffn_enabled", "install_block0_fused_ffn",
           "uninstall_block0_fused_ffn"]
