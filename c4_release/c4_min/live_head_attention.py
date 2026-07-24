"""LIVE-HEAD-ONLY attention scoring — skip the Q@Kᵀ matmul for zero-value heads.

Motivation (attention-FLOP audit).  On the built model (237 physical blocks, 23
heads = 5451 head-slots) attention scoring is **99.7 % of per-step FLOPs**
(≈3.39 GFLOP/step) and 92× wasted: only **23 of the 5451 head-slots (0.42 %)**
have a NON-TRIVIAL value output; the other 5428 are ``_zero_attn`` heads
(``W_v`` slice == 0 AND ``W_o`` slice == 0).  A ``_zero_attn`` head's
contribution to the residual is provably ZERO — it produces ``matmul(attn, V)``
with ``V==0`` (so 0), and ``W_o`` reads that slice through all-zero columns
(so 0 again) — REGARDLESS of which keys it attends to.  Yet the current
``SparseAttn.forward`` still computes the full ``Q@Kᵀ`` score matrix
``[H, Sq, Sk]`` for ALL 23 heads of EVERY block.

So the win is: compute ``Q@Kᵀ`` + ``attn@V`` ONLY for the LIVE-value heads and
leave every DEAD head's per-head output slice at ZERO (which is exactly what the
full forward computes for it, since its ``V`` is zero).  This is
**byte-identical** to the full forward — a dead head's output slice is 0 in BOTH
paths, and ``W_o`` applied to the assembled ``[live … | 0 …]`` head-output is the
SAME matmul in the SAME accumulation order (the zero slices contribute nothing).

Head-slots scored collapse 5451 -> 23 (0.42 %); attention scoring FLOPs
3.39 GFLOP -> ~37 MFLOP/step.  On the 234 blocks with NO live head the whole
score+softmax+attend step is skipped and the block's attention output is exactly
``x``.

RELATIONSHIP TO ``local_attention.py``.  That module WINDOWS the score READ of
the (proven-local) heads and keeps the memory/stack/LEV heads GLOBAL — it still
runs a (short) score matmul for every head-slot.  This module is ORTHOGONAL and
strictly stronger on the DEAD slots: it runs NO score matmul at all for a dead
head (its output is 0 by construction, window-invariant).  The two compose: a
future build could window the 20 live INGEST heads AND skip the 5428 dead ones.
This module reuses ``local_attention.live_value_heads`` for the classification.

DESIGN.  ``classify_live_head_slots(model)`` -> ``{block_idx: bool_mask[H]}`` (a
head is live iff its ``W_v`` rows AND ``W_o`` cols in the head slice are
non-zero).  ``install_live_head_attention(model)`` sets ``attn._live_head_mask``
on every block and swaps in ``live_head_forward``.  ``uninstall_...`` reverts.
"""
from __future__ import annotations

from typing import Dict

import torch

from . import sparse_forward as _SF
from .blogspec_model import softmax1
from .local_attention import live_value_heads


# The ORIGINAL global attention forward (restored on uninstall / no-mask fallback).
_ORIG_FORWARD = _SF.SparseAttn.forward


def classify_live_head_slots(model) -> Dict[int, torch.Tensor]:
    """Return ``{block_idx: bool_mask[H]}`` — ``mask[h]`` True iff head ``h`` has a
    NON-TRIVIAL value output (``W_v`` rows AND ``W_o`` cols in head slice
    ``h·HD..(h+1)·HD`` non-zero).  A head not flagged outputs EXACTLY ``x + 0`` no
    matter which keys it reads, so its score matmul can be skipped byte-identically.

    CPU-only (structural nnz via ``live_value_heads``), so no GPU dense transient.
    """
    out: Dict[int, torch.Tensor] = {}
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        H = at.n_heads
        mask = torch.zeros(H, dtype=torch.bool)
        for h in live_value_heads(at):
            mask[h] = True
        out[bi] = mask
    return out


def live_head_attention_stats(model) -> Dict[str, object]:
    """Analytic work-reduction summary (no forward run).  Head-slots scored
    before/after and the per-block live-head map."""
    cls = classify_live_head_slots(model)
    n_blocks = len(model.blocks)
    H = model.blocks[0].attn.n_heads if n_blocks else 0
    total_slots = n_blocks * H
    live_slots = int(sum(int(m.sum()) for m in cls.values()))
    live_blocks = sum(1 for m in cls.values() if bool(m.any()))
    per_block_live = {bi: torch.nonzero(m, as_tuple=False).flatten().tolist()
                      for bi, m in cls.items() if bool(m.any())}
    return {
        "n_blocks": n_blocks,
        "n_heads": H,
        "total_head_slots": total_slots,
        "live_head_slots": live_slots,
        "dead_head_slots": total_slots - live_slots,
        "live_attention_blocks": live_blocks,
        "dead_attention_blocks": n_blocks - live_blocks,
        "per_block_live_heads": per_block_live,
        "frac_scored_after": live_slots / max(1, total_slots),
    }


# ===========================================================================
# The live-head-only forward — a drop-in for ``SparseAttn.forward`` that computes
# Q@Kᵀ + attn@V ONLY over the live-value heads and leaves every dead head's
# per-head output slice at 0 (byte-identical: a dead head's V is 0 anyway).
# ===========================================================================
def live_head_forward(self, x, past_kv=None, q_positions=None, use_cache=False):
    """Byte-identical to ``SparseAttn.forward``, but the ``Q@Kᵀ`` / ``attn@V``
    matmuls run ONLY for the heads in ``self._live_head_mask``.  Every OTHER head's
    per-head output slice is left ZERO — which is EXACTLY what the full forward
    computes for it (its ``W_v`` slice is 0, so ``matmul(attn, V)`` over that head
    is 0), so ``x + W_o·(assembled head output)`` is bit-for-bit the same.

    On a block with NO live head the whole score+softmax+attend is skipped and the
    output is exactly ``x + W_o·0 == x`` (``W_o`` slice is 0 for every head too).

    Cache contract MATCHES ``SparseAttn.forward``: when ``use_cache`` it still
    returns ``(out, (K, V, k_pos))`` with the FULL K/V of ALL heads (the dead
    heads' K/V are 0-valued but structurally present) so the driver's per-block
    cache concat is unchanged and byte-identical.  (K/V of every head is still
    materialised — cheap ``F.linear`` — only the O(S²) SCORE matmul is pruned.)
    """
    lmask = getattr(self, "_live_head_mask", None)
    if lmask is None:
        return _ORIG_FORWARD(self, x, past_kv=past_kv, q_positions=q_positions,
                             use_cache=use_cache)

    B, S, D = x.shape
    H, HD = self.n_heads, self.head_dim
    lmask = lmask.to(device=x.device)
    live_idx = torch.nonzero(lmask, as_tuple=False).flatten()

    # K/V are needed for the returned cache of EVERY head (a dead head's K/V are
    # 0 but the driver concats a full-H cache), so materialise them (cheap linear).
    Q = self.W_q.linear(x).view(B, S, H, HD).transpose(1, 2)        # [B,H,S,HD]
    Knew = self.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)
    Vnew = self.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)

    if q_positions is None:
        q_pos = torch.arange(S, device=x.device)
    else:
        q_pos = q_positions.to(device=x.device, dtype=torch.long)

    if past_kv is not None:
        K_cache, V_cache, pos_cache = past_kv
        K = torch.cat([K_cache, Knew], dim=2)
        V = torch.cat([V_cache, Vnew], dim=2)
        k_pos = torch.cat([pos_cache.to(x.device), q_pos], dim=0)
    else:
        K, V, k_pos = Knew, Vnew, q_pos

    # Assemble the per-head attention output; dead heads stay ZERO (their true value).
    attn_out = x.new_zeros(B, H, S, HD)

    if live_idx.numel() > 0:
        # Score ONLY the live heads: [B, Hl, S, Sk].
        Ql = Q[:, live_idx]
        Kl = K[:, live_idx]
        Vl = V[:, live_idx]
        slopes = self.alibi_slopes[live_idx].view(1, -1, 1, 1)
        scores = torch.matmul(Ql, Kl.transpose(-2, -1)) * self.scale
        if past_kv is None and q_positions is None:
            pos = torch.arange(S, device=x.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
            scores = scores - slopes * dist
            causal = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal
        else:
            dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs().float()
            scores = scores - slopes * dist.unsqueeze(0)
            mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))       # [Sq, Sk]
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        a = softmax1(scores, dim=-1)
        attn_out[:, live_idx] = torch.matmul(a, Vl)

    out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    out = x + self.W_o.linear(out)
    if use_cache:
        return out, (K, V, k_pos)
    return out


# ===========================================================================
# DEAD-BLOCK ATTENTION FUSION — bypass the ENTIRE attention sublayer of a block
# whose attention has NO live-value head (all heads ``_zero_attn``).
#
# ``live_head_forward`` already SKIPS the O(S²) score matmul on a dead block, but
# it STILL materialises Q/K/V (three ``F.linear``s), the ``W_o`` linear, and the
# full-H KV tuple for the returned cache.  Measured (this task's audit): those
# per-block linears are the wall on the 234 dead blocks (~98% of the attention-
# sublayer time even after score-skip).  A dead block's attention output is
# PROVABLY ``x`` (all heads ``_zero_attn`` -> ``W_v`` slice 0 -> ``attn@V == 0``;
# ``W_o`` slice 0 -> ``x + W_o·0 == x``; audited no output bias, L-inf=0), so we
# can bypass the WHOLE sublayer: output is ``x`` directly, no linears, no KV.
#
# KV-CACHE SAFETY (option (a), skip the write entirely).  A dead block's KV is
# NEVER read: (1) its OWN future attention ignores it — every head is dead, so a
# future step's attention over this block also outputs ``x`` regardless of K/V;
# (2) NO OTHER block reads block-k's cache — the driver's cache is strictly
# positional (``caches[b]`` <-> ``blocks[b]``, per-block ``forward_hidden_cached``
# append).  So returning ``None`` for a dead block's KV is byte-safe.  This is the
# EXACT contract the speculative driver's block-MoE skip already relies on
# (``pf_speculative._forward_hidden_cached_skip`` sets ``new_caches[b]=None`` and
# ``_commit_span`` skips ``None`` with the comment "its cache is never read, so a
# gap is harmless").  The single-program cached driver
# (``nibble_pure_forward_cached``) is taught the same one-line ``None``-guard.
# ===========================================================================
def dead_block_forward(self, x, past_kv=None, q_positions=None, use_cache=False):
    """The fused forward for a block whose attention has NO live-value head.

    Output is EXACTLY ``x`` (a dead block's attention sublayer is the identity on
    the residual — proven L-inf=0).  Skips ALL of Q/K/V linears + ``W_o`` + the
    KV materialisation.  On the cache path returns ``(x, None)``: the block writes
    NO KV entry (byte-safe — a dead block's KV is provably never read; see module
    docstring).  The driver tolerates the ``None`` (positional per-block cache).
    """
    if use_cache:
        return x, None
    return x


def install_dead_block_fusion(model, verbose: bool = False) -> Dict[str, object]:
    """Bypass the ENTIRE attention sublayer of every DEAD-attention block.

    Composes with (and requires the same classification as) the live-head install:
    a block with 0 live-value heads gets ``dead_block_forward`` (output ``x``, no
    linears, no KV write); a block with >=1 live head keeps its attention (the
    caller pairs this with ``install_live_head_attention`` so live blocks run the
    live-head-only forward, and the ~234 dead blocks are fully fused away).

    Sets ``attn._dead_block_fused = True`` on the fused blocks so
    ``uninstall_dead_block_fusion`` can revert.  Returns the same summary dict as
    ``live_head_attention_stats`` plus ``fused_blocks`` (the count bypassed).
    """
    cls = classify_live_head_slots(model)
    fused = 0
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        if not bool(cls[bi].any()):                # dead block: 0 live heads
            at._dead_block_fused = True
            at.forward = dead_block_forward.__get__(at, type(at))
            fused += 1
    stats = live_head_attention_stats(model)
    stats["fused_blocks"] = fused
    if verbose:
        print(f"[dead-block-fusion] bypassed {fused}/{stats['n_blocks']} "
              f"dead-attention blocks (output=x, no K/Q/V/W_o linears, no KV "
              f"write); {stats['live_attention_blocks']} live blocks keep "
              f"attention")
    return stats


def uninstall_dead_block_fusion(model) -> None:
    """Revert every fused dead block to its prior ``attn.forward``."""
    for blk in model.blocks:
        at = blk.attn
        if getattr(at, "_dead_block_fused", False):
            delattr(at, "_dead_block_fused")
        if "forward" in at.__dict__:
            del at.__dict__["forward"]


def install_live_head_attention(model, verbose: bool = False) -> Dict[str, object]:
    """Install live-head-only attention scoring on every block.

    Sets ``attn._live_head_mask`` (a ``[H]`` bool of the live-value heads) on every
    block and binds ``live_head_forward``.  Byte-identical to the global forward:
    every non-live head's per-head output is 0 in BOTH paths.  Returns a summary
    (classification + scored-slot fraction).  ``uninstall_live_head_attention``
    reverts to the original global forward.
    """
    cls = classify_live_head_slots(model)
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        at._live_head_mask = cls[bi].to(at.alibi_slopes.device)
        at.forward = live_head_forward.__get__(at, type(at))
    stats = live_head_attention_stats(model)
    if verbose:
        print(f"[live-head-attn] scoring {stats['live_head_slots']}/"
              f"{stats['total_head_slots']} head-slots "
              f"({stats['frac_scored_after']*100:.2f}%); "
              f"{stats['live_attention_blocks']}/{stats['n_blocks']} blocks have "
              f"live attention, {stats['dead_attention_blocks']} are pure x-passthrough")
        print(f"[live-head-attn] live heads (block->heads): "
              f"{stats['per_block_live_heads']}")
    return stats


def uninstall_live_head_attention(model) -> None:
    """Revert every block to the ORIGINAL global ``SparseAttn.forward``."""
    for blk in model.blocks:
        at = blk.attn
        if hasattr(at, "_live_head_mask"):
            delattr(at, "_live_head_mask")
        if "forward" in at.__dict__:
            del at.__dict__["forward"]
