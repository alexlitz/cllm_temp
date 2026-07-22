"""Block-sparsity analysis + BYTE-IDENTICAL clustering permutation for the LEAN
compacted C4 VM forward.

The lean fused VM (``qwen_lean_forward.LeanQwenVM``) is a genuine Qwen2 decoder
stack whose ~3 k (mem+cmp) / ~86 k (full) nonzero weights are HAND-PLACED by
semantic role, NOT naturally block-structured.  Running its GEMMs dense wastes
essentially all FLOPs on zeros; running them unstructured-sparse (COO/CSR)
underutilizes the GPU's tensor cores (a CSR mat-mul is scalar-gather bound, no
dense-tile MMA).

This module answers, HONESTLY, whether a *consistent* relabeling of residual
dims / FFN units / heads — a permutation ``P`` applied as ``P·W·Pᵀ`` (a pure
relabeling that is BYTE-IDENTICAL: it reorders the residual axis the SAME way
everywhere, so every attention/FFN op reads/writes the SAME values, just at
renamed slots) — can CLUSTER the scattered nonzeros into dense tiles that run as
dense block-matmuls on tensor cores, and MEASURES the block density before/after.

Two axes are permutable while preserving the math EXACTLY:

  * the RESIDUAL (hidden) axis ``H`` — one global permutation ``pH`` applied to
    every weight's residual side: q/k/v/gate/up read columns ``[·, H]`` (permute
    columns by ``pH``); o/down write rows ``[H, ·]`` (permute rows by ``pH``);
    the token embedding writes ``[vocab, H]`` (permute columns).  Because the
    RMSNorm gamma is a per-dim scale it permutes with ``pH`` too, and the
    residual add is elementwise so a consistent ``pH`` end-to-end is invisible to
    the decoded output.  This is the ``P·W·Pᵀ`` relabeling.

  * the FFN HIDDEN (intermediate) axis ``I`` per layer — gate/up write rows
    ``[I, ·]`` and down reads columns ``[·, I]``; ``silu(up)*gate`` is elementwise
    so any per-layer permutation ``pI`` of the intermediate units is invisible.
    The bias vectors ``b_up/b_gate`` permute with ``pI``.

  * the ATTENTION HEAD axis is already block-structured (each head is a
    ``head_dim``-wide contiguous slab of q/k/v and o); GQA repeat is per-kv-head.
    We keep heads intact (permuting within/across heads would break RoPE lane
    pairing), so head-level structure is a GIVEN block, like MoE experts.

We DO NOT change any weight VALUE — only WHERE (which slot) it lives.  The
byte-identity gate (``verify_permuted_forward_byte_identical``) proves the
permuted lean forward decodes L∞=0 vs the un-permuted one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch


# ===========================================================================
# Block-density measurement.
# ===========================================================================
@dataclass
class TileStats:
    tile: int
    active_tiles: int
    total_tiles: int
    nnz: int
    numel: int

    @property
    def skip_frac(self) -> float:
        return 1.0 - (self.active_tiles / self.total_tiles) if self.total_tiles else 1.0

    @property
    def avg_fill(self) -> float:
        """Mean nonzeros per ACTIVE tile (the tensor-core-relevant density: how
        dense is the padded tile we would hand the MMA unit)."""
        if self.active_tiles == 0:
            return 0.0
        return self.nnz / self.active_tiles

    @property
    def tile_fill_frac(self) -> float:
        """avg_fill / tile² — the fraction of a scheduled tile that is real work
        (the rest is padding waste the tensor core still multiplies)."""
        return self.avg_fill / (self.tile * self.tile)

    @property
    def block_flops_ratio(self) -> float:
        """FLOPs a block-sparse GEMM does (active_tiles·tile²) relative to the
        DENSE GEMM (numel).  <1 means block-sparse skips work; the useful-work
        fraction inside that is ``tile_fill_frac``."""
        return (self.active_tiles * self.tile * self.tile) / self.numel if self.numel else 0.0


def tile_stats(mask: torch.Tensor, tile: int) -> TileStats:
    """Block-density of a boolean/weight matrix ``mask`` [out, in] at ``tile``.

    Pads ``mask`` up to a multiple of ``tile`` on both axes, counts how many
    ``tile×tile`` blocks contain at least one nonzero (the tiles a block-sparse
    kernel must schedule) and the mean fill of those active tiles."""
    m = (mask != 0)
    out, inn = m.shape
    pr = math.ceil(out / tile)
    pc = math.ceil(inn / tile)
    mp = torch.zeros(pr * tile, pc * tile, dtype=torch.bool, device=m.device)
    mp[:out, :inn] = m
    blocks = mp.view(pr, tile, pc, tile).permute(0, 2, 1, 3).reshape(pr * pc, tile * tile)
    per_block = blocks.sum(dim=1)
    active = int((per_block > 0).sum())
    nnz = int(m.sum())
    return TileStats(tile=tile, active_tiles=active, total_tiles=pr * pc,
                     nnz=nnz, numel=out * inn)


# ===========================================================================
# The clustering permutation.
#
# We want ONE global residual permutation ``pH`` (length H) and ONE per-layer
# intermediate permutation ``pI`` (length I) that, applied consistently, cluster
# the nonzeros of EVERY weight into as few dense tiles as possible.
#
# The residual axis is shared by ALL layers (it is the token stream), so pH must
# be chosen JOINTLY across the whole stack.  We build a co-occurrence graph over
# residual dims: two residual dims are "affine" if they co-appear in the support
# of the same weight matrix (same active row-set for a write, same active col-set
# for a read).  A reverse-Cuthill-McKee (bandwidth-reduction) ordering of that
# graph groups affine dims adjacent, so their nonzeros fall in the same tile.
# ===========================================================================
def _residual_cooccurrence(mats_read: List[torch.Tensor],
                           mats_write: List[torch.Tensor], H: int) -> torch.Tensor:
    """Symmetric [H,H] co-occurrence count over residual dims.

    ``mats_read`` are weights that READ the residual on their COLUMN axis
    (q/k/v/gate/up: [out, H]); ``mats_write`` WRITE it on their ROW axis
    (o/down: [H, out']).  Two residual dims co-occur when they are both active
    on the OTHER axis of the same matrix (both columns hit by some output row /
    both rows hit by some input col) — i.e. they participate together in a
    dense sub-block, so clustering them adjacent fills a tile."""
    C = torch.zeros(H, H, dtype=torch.float64)
    for w in mats_read:                      # [out, H]: residual is the column axis
        m = (w != 0)
        for r in range(m.shape[0]):
            cols = torch.nonzero(m[r], as_tuple=False).flatten()
            if cols.numel() > 1:
                C[cols.unsqueeze(1), cols.unsqueeze(0)] += 1.0
    for w in mats_write:                     # [H, out']: residual is the row axis
        m = (w != 0)
        for c in range(m.shape[1]):
            rows = torch.nonzero(m[:, c], as_tuple=False).flatten()
            if rows.numel() > 1:
                C[rows.unsqueeze(1), rows.unsqueeze(0)] += 1.0
    C.fill_diagonal_(0.0)
    return C


def _rcm_order(adj: torch.Tensor) -> List[int]:
    """Reverse Cuthill-McKee ordering of a symmetric adjacency (bandwidth
    reduction): a BFS from a low-degree seed, neighbours visited in increasing
    degree, reversed.  Isolated nodes appended at the end.  Returns a permutation
    (new position -> old index)."""
    n = adj.shape[0]
    deg = (adj != 0).sum(dim=1)
    adj_bool = (adj != 0)
    visited = [False] * n
    order: List[int] = []
    remaining = sorted(range(n), key=lambda i: int(deg[i]))
    for seed in remaining:
        if visited[seed]:
            continue
        queue = [seed]
        visited[seed] = True
        while queue:
            node = queue.pop(0)
            order.append(node)
            nbrs = torch.nonzero(adj_bool[node], as_tuple=False).flatten().tolist()
            nbrs = [x for x in nbrs if not visited[x]]
            nbrs.sort(key=lambda i: int(deg[i]))
            for x in nbrs:
                visited[x] = True
                queue.append(x)
    order.reverse()                          # REVERSE Cuthill-McKee
    return order


def _support_first_order(active: torch.Tensor) -> List[int]:
    """'active-first' order: all active indices (original order) then all
    inactive.  A baseline that just PACKs the used dims contiguous so the
    all-inactive tail forms skippable tiles, without RCM's within-active
    grouping."""
    act = torch.nonzero(active, as_tuple=False).flatten().tolist()
    inact = torch.nonzero(~active, as_tuple=False).flatten().tolist()
    return act + inact


@dataclass
class Permutation:
    """A byte-identical relabeling of the lean model's axes.

    ``pH`` [H]: new-position -> old residual-dim index (global, all layers).
    ``pI`` list of [I]: per-layer intermediate-unit permutation.
    All are ``old = perm[new]`` gather indices (apply with ``W[:, pH]`` for a
    residual-read column permute, ``W[pH]`` for a residual-write row permute)."""
    pH: torch.Tensor
    pI: List[torch.Tensor]
    method: str = ""


def build_clustering_permutation(lean, method: str = "rcm") -> Permutation:
    """Compute a byte-identical clustering permutation for ``lean``.

    ``method``:
      * ``"rcm"``       — reverse Cuthill-McKee on the residual co-occurrence
                          graph (+ RCM per-layer on each intermediate axis).
      * ``"pack"``      — active-first packing (cluster the used dims contiguous,
                          push the all-zero tail into skippable tiles).
      * ``"identity"``  — no reorder (the CURRENT semantic layout; baseline).
    """
    H = lean.hidden_size
    layers = lean.layers
    if method == "identity":
        pH = torch.arange(H)
        pI = [torch.arange(l.gate_w.shape[0]) for l in layers]
        return Permutation(pH=pH, pI=pI, method="identity")

    reads: List[torch.Tensor] = []
    writes: List[torch.Tensor] = []
    for l in layers:
        reads += [l.q_w, l.k_w, l.v_w, l.gate_w, l.up_w]     # [out, H]
        writes += [l.o_w, l.down_w]                           # [H, out']
    active_res = torch.zeros(H, dtype=torch.bool)
    for w in reads:
        active_res |= (w != 0).any(dim=0)
    for w in writes:
        active_res |= (w != 0).any(dim=1)

    if method == "pack":
        pH = torch.tensor(_support_first_order(active_res), dtype=torch.long)
    elif method == "rcm":
        C = _residual_cooccurrence(reads, writes, H)
        order = _rcm_order(C)
        # keep active dims first (packed + RCM-grouped), inactive tail after.
        act = [i for i in order if bool(active_res[i])]
        inact = [i for i in order if not bool(active_res[i])]
        pH = torch.tensor(act + inact, dtype=torch.long)
    else:
        raise ValueError(f"unknown method {method!r}")

    pI: List[torch.Tensor] = []
    for l in layers:
        I = l.gate_w.shape[0]
        active_i = torch.zeros(I, dtype=torch.bool)
        active_i |= (l.gate_w != 0).any(dim=1)
        active_i |= (l.up_w != 0).any(dim=1)
        active_i |= (l.down_w != 0).any(dim=0)
        if method == "pack":
            pI.append(torch.tensor(_support_first_order(active_i), dtype=torch.long))
        else:  # rcm: co-occurrence over intermediate units via shared residual dims
            Cg = _residual_cooccurrence([l.gate_w.t().contiguous(),
                                         l.up_w.t().contiguous()],
                                        [l.down_w.t().contiguous()], I)
            order = _rcm_order(Cg)
            act = [i for i in order if bool(active_i[i])]
            inact = [i for i in order if not bool(active_i[i])]
            pI.append(torch.tensor(act + inact, dtype=torch.long))
    return Permutation(pH=pH, pI=pI, method=method)


# ===========================================================================
# Apply a permutation to the lean model (byte-identical relabeling).
# ===========================================================================
def apply_permutation(lean, perm: Permutation):
    """Return a NEW ``LeanQwenVM`` whose weights are ``perm`` relabeled.

    Residual axis ``pH`` permutes: embed columns, every ln gamma, q/k/v/gate/up
    columns, o/down rows.  Per-layer ``pI`` permutes gate/up rows + b_up/b_gate,
    and down columns.  This is a pure relabeling — the forward decodes identically
    (proven by ``verify_permuted_forward_byte_identical``)."""
    from .qwen_lean_forward import _LeanLayer, LeanQwenVM

    pH = perm.pH
    new_layers: List[_LeanLayer] = []
    for l, pI in zip(lean.layers, perm.pI):
        new_layers.append(_LeanLayer(
            ln1=l.ln1[pH].clone(),
            ln2=l.ln2[pH].clone(),
            q_w=l.q_w[:, pH].clone(),
            k_w=l.k_w[:, pH].clone(),
            v_w=l.v_w[:, pH].clone(),
            o_w=l.o_w[pH, :].clone(),
            q_b=(l.q_b.clone() if l.q_b is not None else None),
            k_b=(l.k_b.clone() if l.k_b is not None else None),
            v_b=(l.v_b.clone() if l.v_b is not None else None),
            gate_w=l.gate_w[pI][:, pH].clone(),
            up_w=l.up_w[pI][:, pH].clone(),
            down_w=l.down_w[pH][:, pI].clone(),
        ))
    new = LeanQwenVM(
        layers=new_layers,
        final_norm=lean.final_norm[pH].clone(),
        embed=lean.embed[:, pH].clone(),
        hidden_size=lean.hidden_size, n_layers=lean.n_layers,
        n_heads=lean.n_heads, n_kv_heads=lean.n_kv_heads, head_dim=lean.head_dim,
        rope_theta=lean.rope_theta, rms_eps=lean.rms_eps,
        device=lean.device, dtype=lean.dtype, QL=lean.QL, subset=lean.subset,
        inv_freq=lean.inv_freq.clone(),
    )
    return new


# ===========================================================================
# Whole-model block-density report (per matrix family, aggregated).
# ===========================================================================
_FFN_MATS = ["gate_w", "up_w", "down_w"]
_ATTN_MATS = ["q_w", "k_w", "v_w", "o_w"]


def model_tile_report(lean, tiles=(8, 16, 32)):
    """Aggregate block density across ALL layers for each tile size.

    Returns ``{tile: {"attn": TileStats-agg, "ffn": ..., "all": ...}}`` where the
    aggregate sums active/total tiles and nnz/numel across every matrix in the
    family, so ``skip_frac`` / ``block_flops_ratio`` are the model-wide numbers."""
    out = {}
    for tile in tiles:
        agg = {"attn": [0, 0, 0, 0], "ffn": [0, 0, 0, 0], "all": [0, 0, 0, 0]}
        for l in lean.layers:
            for nm in _ATTN_MATS + _FFN_MATS:
                ts = tile_stats(getattr(l, nm), tile)
                fam = "attn" if nm in _ATTN_MATS else "ffn"
                for key in (fam, "all"):
                    agg[key][0] += ts.active_tiles
                    agg[key][1] += ts.total_tiles
                    agg[key][2] += ts.nnz
                    agg[key][3] += ts.numel
        res = {}
        for key, (a, t, nz, ne) in agg.items():
            res[key] = TileStats(tile=tile, active_tiles=a, total_tiles=t,
                                 nnz=nz, numel=ne)
        out[tile] = res
    return out
