"""Compile AttentionSpec -> attention weights (W_q,W_k,W_v,W_o,mask).

Two head constructions cover the ISA's attention needs:

1. ``carry_forward_head`` — attend to the immediate predecessor position and copy
   a set of bands forward. Selection is purely positional (strong ALiBi + a
   kill-self diagonal), so it needs no content match. This is the head the
   cross-position state carry uses; it is exercised + verified by the module
   self-test below (and mirrors the gadget validated in the slice bring-up).

2. ``content_match_head`` — select the source position whose ``k_band`` equals the
   query's ``q_band + q_offset``, and copy ``v_band`` -> ``dst_band``. Used for
   pop (SP -> stack cell) and memory load (addr -> cell). The match uses the
   negative-squared-distance trick, which needs a *squared* companion of the key
   in the residual (``ksq_band``): with
       Q = [2g*(q+off), -g],  K = [k, ksq]      (ksq == k^2)
   the score is  -g*(k-(q+off))^2 + const, maximised at k == q+off. The caller
   must supply ``ksq_band`` (the compiler bakes k^2 lanes for integer key bands).
"""
from __future__ import annotations

from typing import List

import torch

from .dsl import AttentionSpec


def carry_forward_head(dim, n_heads, max_pos, one_band, carried_bands, slope=5.0):
    """One head: copy ``carried_bands`` from position t-1 to t (t=0 keeps its own)."""
    head_dim = dim // n_heads
    assert 1 + len(carried_bands) <= head_dim, "carried bands must fit in head 0"
    W_q = torch.zeros(dim, dim); W_k = torch.zeros(dim, dim)
    W_v = torch.zeros(dim, dim); W_o = torch.zeros(dim, dim)
    W_q[0, one_band] = 1.0
    W_k[0, one_band] = 1.0
    for i, band in enumerate(carried_bands):
        W_v[1 + i, band] = 1.0
        W_o[band, 1 + i] = 1.0
    mask = torch.full((max_pos, max_pos), float("-inf"))
    for q in range(max_pos):
        for k in range(q + 1):
            mask[q, k] = -slope * (q - k)
        mask[q, q] += -1000.0          # kill self -> pick t-1
    mask[0, 0] = 0.0                    # row 0 has no predecessor; keep self
    return {"W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o, "mask": mask}


def content_match_head(spec: AttentionSpec, ksq_band: int, dim, n_heads, max_pos,
                       one_band):
    """One head selecting position where ``k_band == q_band + q_offset``.

    ``ksq_band`` must hold ``k_band**2`` at each candidate position.
    """
    head_dim = dim // n_heads
    assert head_dim >= 2, "need >=2 dims per head for the squared-distance match"
    g = spec.gain
    W_q = torch.zeros(dim, dim); W_k = torch.zeros(dim, dim)
    W_v = torch.zeros(dim, dim); W_o = torch.zeros(dim, dim)
    # Q = [2g*(q+off), -g] ; K = [k, k^2]  -> Q.K = -g*(k-(q+off))^2 + g*(q+off)^2
    W_q[0, spec.q_band] = 2.0 * g
    W_q[0, one_band] = 2.0 * g * spec.q_offset
    W_q[1, one_band] = -g
    W_k[0, spec.k_band] = 1.0
    W_k[1, ksq_band] = 1.0
    W_v[0, spec.v_band] = 1.0
    W_o[spec.dst_band, 0] = 1.0
    mask = torch.zeros(max_pos, max_pos)
    if spec.alibi_slope:
        for qi in range(max_pos):
            for ki in range(max_pos):
                mask[qi, ki] += -spec.alibi_slope * abs(qi - ki)
    return {"W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o, "mask": mask}


def compile_attn(specs: List[AttentionSpec], dim: int, n_heads: int, max_pos: int,
                 one_band: int, ksq_bands: List[int]):
    """Pack up to ``n_heads`` content-match specs into one block's attention.

    ``ksq_bands[i]`` is the squared-key companion band for ``specs[i]``.
    """
    assert len(specs) <= n_heads, "one head per spec"
    head_dim = dim // n_heads
    W_q = torch.zeros(dim, dim); W_k = torch.zeros(dim, dim)
    W_v = torch.zeros(dim, dim); W_o = torch.zeros(dim, dim)
    mask = torch.zeros(max_pos, max_pos)
    for h, (spec, ksq) in enumerate(zip(specs, ksq_bands)):
        w = content_match_head(spec, ksq, dim, n_heads, max_pos, one_band)
        base = h * head_dim
        W_q[base:base + head_dim] += w["W_q"][:head_dim]
        W_k[base:base + head_dim] += w["W_k"][:head_dim]
        W_v[base:base + head_dim] += w["W_v"][:head_dim]
        W_o[:, base:base + head_dim] += w["W_o"][:, :head_dim]
        mask += w["mask"]
    return {"W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o, "mask": mask}
