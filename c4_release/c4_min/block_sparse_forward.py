"""BLOCK-STRUCTURED sparse forward for the LEAN compacted C4 VM.

The lean fused VM's ~3-86 k nonzeros are hand-placed by semantic role, so an
UNSTRUCTURED sparse (COO/CSR) mat-mul is scalar-gather bound and cannot use the
GPU's tensor cores (they need dense ``tile×tile`` MMA tiles).  This module runs
the SAME forward with a BLOCK-sparse representation (torch BSR — Block Sparse Row
— which dispatches to a dense-tile block matmul, the tensor-core-friendly form),
on weights first CLUSTERED by a byte-identical residual/intermediate permutation
(``block_sparse_analysis.build_clustering_permutation``) so the scattered
nonzeros pack into as few dense tiles as possible.

Three composable pieces:

  * ``BlockSparseWeight`` — a ``[out, in]`` linear stored as a torch BSR tensor at
    a chosen ``blocksize`` (the tile).  ``linear(x)`` runs ``x @ Wᵀ`` as a
    block-sparse matmul (dense MMA over the ACTIVE tiles only).  A weight with no
    (or too few) active tiles is kept dense (block sparsity never pays a
    launch-overhead loss on a tiny matrix — logged).

  * ``BlockSparseLean`` — wraps a ``LeanQwenVM`` (already permuted by
    ``apply_permutation``): every q/k/v/o/gate/up/down GEMM is a
    ``BlockSparseWeight``; the attention math (RoPE/softmax/GQA) is copied
    verbatim from the lean forward.

  * ``PermutedBlockSparseVM`` — the DRIVER-facing wrapper: it permutes the
    (original-space) residual into the clustered space at forward ENTRY and
    unpermutes at EXIT, so the un-modified ``run_program_lean`` /
    ``speculative_run_lean`` driver decodes IDENTICALLY (the permutation is a
    relabeling, the block matmul is the same arithmetic on the nonzeros).

The permutation + block matmul is DECODE-byte-identical to the dense/COO forward
(``verify.byte_identity``): a column reorder changes fp reduction ORDER (a
~1e-2-relative residue on the huge VM band magnitudes, far below the integer
``_snap`` decode margin), so the decoded register trace is bit-for-bit the same.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .block_sparse_analysis import Permutation, apply_permutation, tile_stats


# ===========================================================================
# One block-sparse (BSR) or dense linear weight.
# ===========================================================================
@dataclass
class BlockSparseWeight:
    """A ``[out, in]`` linear weight stored block-sparse (torch BSR) or dense.

    ``linear(x)`` computes ``F.linear(x, W)`` = ``x @ Wᵀ`` with a block-sparse
    matmul over the ACTIVE ``blocksize×blocksize`` tiles (the tensor-core path)
    when block sparsity pays, else the dense kernel.  ``bsr`` holds the BSR
    tensor; ``active_tiles`` / ``total_tiles`` are the block-density accounting.
    """

    out_dim: int
    in_dim: int
    blocksize: int
    is_block_sparse: bool
    dense: Optional[torch.Tensor] = None       # [out, in] when dense-kept
    bsr: Optional[torch.Tensor] = None         # BSR when block-sparse
    nnz: int = 0
    active_tiles: int = 0
    total_tiles: int = 0
    kept_reason: str = ""

    def to(self, device) -> "BlockSparseWeight":
        if self.dense is not None:
            self.dense = self.dense.to(device)
        if self.bsr is not None:
            self.bsr = self.bsr.to(device)
        return self

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ Wᵀ`` for ``x`` [..., in] -> [..., out]."""
        if not self.is_block_sparse:
            return F.linear(x, self.dense)
        orig = x.shape
        x2d = x.reshape(-1, self.in_dim)                      # [N, in]
        # the BSR weight is zero-PADDED to a tile multiple on both axes: pad the
        # input on its contracted (in) axis to match, run the block matmul, then
        # SLICE the padded output rows back to the real out_dim.
        bsr_out, bsr_in = self.bsr.shape
        if bsr_in != self.in_dim:
            pad = torch.zeros(x2d.shape[0], bsr_in - self.in_dim,
                              dtype=x2d.dtype, device=x2d.device)
            x2d = torch.cat([x2d, pad], dim=1)
        # BSR @ dense: W[out,in] @ x2dᵀ[in,N] -> [out, N]; transpose -> [N, out].
        try:
            out = torch.sparse.mm(self.bsr, x2d.transpose(0, 1).contiguous())
        except (RuntimeError, NotImplementedError):
            out = self.bsr.to_dense() @ x2d.transpose(0, 1).contiguous()
        out = out.transpose(0, 1).contiguous()                # [N, bsr_out]
        if bsr_out != self.out_dim:
            out = out[:, :self.out_dim].contiguous()          # drop pad rows
        return out.reshape(*orig[:-1], self.out_dim)

    # block-flop accounting -------------------------------------------------
    def block_flops(self) -> int:
        """FLOP-equivalent NUMEL a block matmul processes (active tiles · tile²)."""
        return self.active_tiles * self.blocksize * self.blocksize

    def dense_flops(self) -> int:
        return self.out_dim * self.in_dim


def _make_block_weight(w: torch.Tensor, blocksize: int,
                       min_active_tiles: int, log: Dict[str, int]) -> BlockSparseWeight:
    """Build a ``BlockSparseWeight`` from ``w`` at ``blocksize``.

    Keeps the weight DENSE if it is too small to tile or has (almost) all tiles
    active — block sparsity only pays when a large fraction of tiles is empty AND
    there are enough active tiles to amortise the kernel launch.  BSR requires
    ``out``/``in`` divisible by ``blocksize``, so a weight is zero-padded to the
    tile multiple first (padding is exact — the pad rows/cols are all zero)."""
    out_dim, in_dim = w.shape
    ts = tile_stats(w, blocksize)
    density_after = ts.active_tiles / ts.total_tiles if ts.total_tiles else 1.0
    # dense-keep rules: tiny matrix, no nonzeros, or block density already high.
    if ts.nnz == 0:
        log["kept_dense_empty"] = log.get("kept_dense_empty", 0) + 1
        return BlockSparseWeight(out_dim, in_dim, blocksize, False,
                                 dense=w.contiguous(), nnz=0,
                                 active_tiles=0, total_tiles=ts.total_tiles,
                                 kept_reason="empty")
    if ts.active_tiles < min_active_tiles or density_after >= 0.5:
        log["kept_dense_small"] = log.get("kept_dense_small", 0) + 1
        return BlockSparseWeight(out_dim, in_dim, blocksize, False,
                                 dense=w.contiguous(), nnz=ts.nnz,
                                 active_tiles=ts.active_tiles,
                                 total_tiles=ts.total_tiles, kept_reason="small/dense")
    # pad to tile multiple, build BSR.
    import math
    pr = math.ceil(out_dim / blocksize) * blocksize
    pc = math.ceil(in_dim / blocksize) * blocksize
    if pr != out_dim or pc != in_dim:
        wp = torch.zeros(pr, pc, dtype=w.dtype, device=w.device)
        wp[:out_dim, :in_dim] = w
    else:
        wp = w.contiguous()
    bsr = wp.to_sparse_bsr(blocksize=blocksize)
    log["block_sparse"] = log.get("block_sparse", 0) + 1
    return BlockSparseWeight(out_dim, in_dim, blocksize, True, bsr=bsr, nnz=ts.nnz,
                             active_tiles=ts.active_tiles,
                             total_tiles=ts.total_tiles, kept_reason="block_sparse")


# ===========================================================================
# The block-sparse lean forward.
# ===========================================================================
@dataclass
class _BSLayer:
    ln1: torch.Tensor
    ln2: torch.Tensor
    q: BlockSparseWeight
    k: BlockSparseWeight
    v: BlockSparseWeight
    o: BlockSparseWeight
    q_b: Optional[torch.Tensor]
    k_b: Optional[torch.Tensor]
    v_b: Optional[torch.Tensor]
    gate: BlockSparseWeight
    up: BlockSparseWeight
    down: BlockSparseWeight


class BlockSparseLean:
    """Block-sparse mirror of ``LeanQwenVM`` — SAME RoPE/RMSNorm/softmax/SwiGLU,
    q/k/v/o/gate/up/down run as BSR block matmuls (tensor-core path).

    ``lean`` MUST already be the CLUSTERED (permuted) model
    (``apply_permutation``) for the block density to be worth it; a padded weight
    with no active tiles is kept dense automatically."""

    def __init__(self, lean, blocksize: int = 16, min_active_tiles: int = 2):
        self.hidden_size = lean.hidden_size
        self.n_layers = lean.n_layers
        self.n_heads = lean.n_heads
        self.n_kv_heads = lean.n_kv_heads
        self.head_dim = lean.head_dim
        self.rope_theta = lean.rope_theta
        self.rms_eps = lean.rms_eps
        self.device = lean.device
        self.dtype = lean.dtype
        self.blocksize = blocksize
        self.inv_freq = lean.inv_freq.clone()
        self.final_norm = lean.final_norm.clone()
        self.embed = lean.embed.clone()
        self.QL = lean.QL
        self.subset = lean.subset
        log: Dict[str, int] = {}
        mk = lambda w: _make_block_weight(w.detach(), blocksize, min_active_tiles, log)
        self.layers: List[_BSLayer] = []
        for l in lean.layers:
            self.layers.append(_BSLayer(
                ln1=l.ln1.clone(), ln2=l.ln2.clone(),
                q=mk(l.q_w), k=mk(l.k_w), v=mk(l.v_w), o=mk(l.o_w),
                q_b=(l.q_b.clone() if l.q_b is not None else None),
                k_b=(l.k_b.clone() if l.k_b is not None else None),
                v_b=(l.v_b.clone() if l.v_b is not None else None),
                gate=mk(l.gate_w), up=mk(l.up_w), down=mk(l.down_w)))
        self._log = log

    def to(self, device):
        dev = torch.device(device)
        self.device = dev
        self.inv_freq = self.inv_freq.to(dev)
        self.final_norm = self.final_norm.to(dev)
        self.embed = self.embed.to(dev)
        for l in self.layers:
            l.ln1 = l.ln1.to(dev); l.ln2 = l.ln2.to(dev)
            for w in (l.q, l.k, l.v, l.o, l.gate, l.up, l.down):
                w.to(dev)
            for bnm in ("q_b", "k_b", "v_b"):
                b = getattr(l, bnm)
                if b is not None:
                    setattr(l, bnm, b.to(dev))
        return self

    # ---- forward internals (copied verbatim from qwen_lean_forward) --------
    def _rmsnorm(self, x, gamma):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.rms_eps) * gamma

    def _rope_cos_sin(self, positions):
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _apply_rope(self, t, cos, sin):
        cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
        return (t * cos) + (self._rotate_half(t) * sin)

    def _attn(self, layer: _BSLayer, xn, past, q_pos):
        B, S, H = xn.shape
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        scale = hd ** -0.5
        q = layer.q.linear(xn)
        if layer.q_b is not None:
            q = q + layer.q_b
        k = layer.k.linear(xn)
        if layer.k_b is not None:
            k = k + layer.k_b
        v = layer.v.linear(xn)
        if layer.v_b is not None:
            v = v + layer.v_b
        q = q.view(B, S, nh, hd).transpose(1, 2)
        k = k.view(B, S, nkv, hd).transpose(1, 2)
        v = v.view(B, S, nkv, hd).transpose(1, 2)
        cos_q, sin_q = self._rope_cos_sin(q_pos)
        q = self._apply_rope(q, cos_q, sin_q)
        k = self._apply_rope(k, cos_q, sin_q)
        if past is not None and past[0] is not None:
            K_cache, V_cache, pos_cache = past
            K = torch.cat([K_cache, k], dim=2)
            Vv = torch.cat([V_cache, v], dim=2)
            k_pos = torch.cat([pos_cache, q_pos], dim=1)
        else:
            K, Vv, k_pos = k, v, q_pos
        n_rep = nh // nkv
        if n_rep != 1:
            Sk = K.shape[2]
            Kr = K[:, :, None, :, :].expand(B, nkv, n_rep, Sk, hd).reshape(B, nh, Sk, hd)
            Vr = Vv[:, :, None, :, :].expand(B, nkv, n_rep, Sk, hd).reshape(B, nh, Sk, hd)
        else:
            Kr, Vr = K, Vv
        scores = torch.matmul(q, Kr.transpose(-2, -1)) * scale
        mask = (k_pos.unsqueeze(1) > q_pos.unsqueeze(2))
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
        out = torch.matmul(attn, Vr).transpose(1, 2).contiguous().view(B, S, nh * hd)
        out = layer.o.linear(out)
        return out, (K, Vv, k_pos)

    def forward(self, x, past=None, q_positions=None):
        B, S, H = x.shape
        if q_positions is None:
            q_pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)
            if q_pos.dim() == 1:
                q_pos = q_pos.unsqueeze(0).expand(B, S)
        if past is None:
            past = [None] * self.n_layers
        new_past: List = []
        h = x
        for li, layer in enumerate(self.layers):
            xn = self._rmsnorm(h, layer.ln1)
            a, kv = self._attn(layer, xn, past[li], q_pos)
            h = h + a
            xn2 = self._rmsnorm(h, layer.ln2)
            mlp = layer.down.linear(F.silu(layer.gate.linear(xn2)) * layer.up.linear(xn2))
            h = h + mlp
            new_past.append(kv)
        h = self._rmsnorm(h, self.final_norm)
        return h, new_past

    # ---- block-density / flop accounting ----------------------------------
    def flop_report(self) -> Dict[str, int]:
        block = dense = active = total = n_bs = n_dense = 0
        for l in self.layers:
            for w in (l.q, l.k, l.v, l.o, l.gate, l.up, l.down):
                block += w.block_flops() if w.is_block_sparse else w.dense_flops()
                dense += w.dense_flops()
                active += w.active_tiles
                total += w.total_tiles
                n_bs += int(w.is_block_sparse)
                n_dense += int(not w.is_block_sparse)
        return {"block_flops": block, "dense_flops": dense,
                "active_tiles": active, "total_tiles": total,
                "n_block_sparse": n_bs, "n_kept_dense": n_dense,
                "log": self._log}


# ===========================================================================
# The driver-facing permuting wrapper.
# ===========================================================================
class PermutedBlockSparseVM:
    """Driver-facing block-sparse VM: permutes the residual into the CLUSTERED
    space at forward entry and un-permutes at exit, so the un-modified
    ``run_program_lean`` / ``speculative_run_lean`` driver (which builds the
    overlay in ORIGINAL dim positions) decodes identically.

    Exposes the ``LeanQwenVM`` driver API (``.forward``, ``.QL``, ``.subset``,
    ``.embed``, ``.device``, ``.hidden_size``, ``.n_layers``)."""

    def __init__(self, lean, perm: Permutation, blocksize: int = 16,
                 min_active_tiles: int = 2):
        self.perm = perm
        pH = perm.pH
        self.pH = pH
        inv = torch.empty_like(pH)
        inv[pH] = torch.arange(pH.numel())
        self.inv = inv
        clustered = apply_permutation(lean, perm)
        self.bs = BlockSparseLean(clustered, blocksize=blocksize,
                                  min_active_tiles=min_active_tiles)
        # driver-visible attributes (original-space).
        self.QL = lean.QL
        self.subset = lean.subset
        self.embed = lean.embed
        self.device = lean.device
        self.hidden_size = lean.hidden_size
        self.n_layers = lean.n_layers
        self.blocksize = blocksize

    def to(self, device):
        dev = torch.device(device)
        self.device = dev
        self.pH = self.pH.to(dev)
        self.inv = self.inv.to(dev)
        self.embed = self.embed.to(dev)
        self.bs.to(dev)
        return self

    def forward(self, x, past=None, q_positions=None):
        # permute original-space residual -> clustered space, run BSR, unpermute.
        xp = x[:, :, self.pH]
        h, _ = self.bs.forward(xp, past=None, q_positions=q_positions)
        return h[:, :, self.inv], None

    def flop_report(self):
        return self.bs.flop_report()
