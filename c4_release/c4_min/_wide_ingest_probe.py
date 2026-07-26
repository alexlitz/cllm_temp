"""PROBE: single QUERY head + single KV head "wide-value" frame ingest.

Refute-or-confirm of agent a8c09504's "20 query heads are the minimum".  Tests the
user's hypothesis:

  If the value projection routes each register's value to its OWN distinct band (a
  wide concatenation), a SINGLE head attending to all register tokens produces
  ``out = Σ_i w_i·V_i`` — a SCALED CONCATENATION, not a destructive blend.  A fixed
  downstream FFN rescales lane i by 1/w_i, and the discrete nibble argmax re-quant
  snaps any fp residue to the exact nibble.

STAGE 1 (this file): the IDEAL wide gather physics, isolated.  Even granting a
per-role value routing (the wide concat's whole point), does the softmax1+ALiBi
weight spread across the 20 frame-byte positions leave every 1/w_i rescale inside
the 0.5 argmax-snap margin?  This is the make-or-break the user asked about (min w_i,
residue, ALiBi spread).

Run:  OMP_NUM_THREADS=4 PYTHONPATH=<c4_release> python -m c4_min._wide_ingest_probe
"""
from __future__ import annotations

import math

import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min.blogspec_model import softmax1
from c4_min.nibble_pure_forward import (
    PureForwardLayout, N_ROLES, INGEST_REGS, BYTES_PER_REG, INGEST_EFF,
    INGEST_RECENCY, build_frame_tokens, make_overlay, _snap_nib,
    _FRAME_ROLE_SLOTS,
)


# The a8c09504 reconstruct battery + a wide-32-bit case.
TEST_VECTORS = [
    (2, 13, 0x10000, 0x10000, 6),
    (0, 0, 0x10000, 0x10000, 0),
    (5, 255, 0xFFFC, 0x10000, 128),
    (0xABCD, 0x12345678, 0xFFFC, 0x10000, 0xDEADBEEF),
]

_ROLE_TO_LOCAL = {rr: lc for lc, rr in _FRAME_ROLE_SLOTS.items()}
_REGKEY = {"PC": "pc", "AX": "ax", "SP": "sp", "BP": "bp", "STACK0": "stk"}


def _mk_layout():
    L = PureForwardLayout(code_size=16, n_heads=20, include_memory=False,
                          include_cmp=False)
    from c4_min import nibble_bitwise as _bw
    _bw.extend_layout_for_bitwise(L)
    if _bw.tight_shift_enabled():
        for _op in (isa.SHL, isa.SHR):
            _bw.extend_layout_for_tight_shift(L, _op)
    L.D = L._off
    return L


def _embed(L):
    E = torch.zeros(V.VOCAB, L.D)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    return E


def probe_isolated(recency: float, n_frames: int = 1, verbose: bool = True):
    """Drive the IDEAL wide-value single-head gather (per-role value routing) + the
    fixed 1/w_i rescale on the battery, through the REAL softmax1 weights.

    ``n_frames`` > 1 stacks identical-role frames (a loop) so recency must select the
    LATEST — the case where recency is load-bearing."""
    L = _mk_layout()
    E = _embed(L)
    reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP, "STACK0": L.STACK0}
    all_ok = True
    min_w_global = 1.0
    max_resid_global = 0.0
    for regs in TEST_VECTORS:
        pc, ax, sp, bp, stk = regs
        # Build n_frames identical frames (loop); the LATEST is the correct one.
        frames = []
        for _ in range(n_frames):
            frames += build_frame_tokens(pc, ax, sp, bp, stk)
        toks = torch.tensor([[V.BOS] + frames])
        overlay = make_overlay(isa.assemble([("IMM", 0), ("HALT", 0)]), L)
        x = E[toks].clone()
        overlay(x)
        S = x.shape[1]
        qpos = S - 1
        # Softmax1 over ALL frame-byte rows across ALL frames (role-agnostic query).
        # Each frame-byte token scores M - recency*dist; non-frame rows excluded.
        M = INGEST_EFF
        scores = torch.full((S,), -1e30, dtype=torch.float64)
        for f in range(n_frames):
            fbase = 1 + f * V.FRAME_LEN
            for local in _FRAME_ROLE_SLOTS:
                p = fbase + local
                scores[p] = M - recency * abs(qpos - p)
        wsm = softmax1(scores.unsqueeze(0)).squeeze(0)
        got = {v: 0 for v in _REGKEY.values()}
        for r in range(N_ROLES):
            r_idx, bi = divmod(r, BYTES_PER_REG)
            reg = INGEST_REGS[r_idx]
            local_r = _ROLE_TO_LOCAL[r]
            # the LATEST frame's role-r token position
            pos_r = 1 + (n_frames - 1) * V.FRAME_LEN + local_r
            wr = float(wsm[pos_r])
            min_w_global = min(min_w_global, wr)
            # IDEAL wide gather: lane r accumulates ONLY role-r tokens across frames.
            # out_lane = Σ_{frames f} w_{f,r} * nib_{f,r}.  All frames have the SAME
            # nibble value here (identical frames), so out = (Σ_f w_{f,r}) * nib.
            # Rescale must divide by the TOTAL role-r weight, not just the latest.
            wtot_r = 0.0
            nib_lo = float(x[0, pos_r, L.CUR_NIB + 0])
            nib_hi = float(x[0, pos_r, L.CUR_NIB + 1])
            out_lo = out_hi = 0.0
            for f in range(n_frames):
                p = 1 + f * V.FRAME_LEN + local_r
                wf = float(wsm[p])
                wtot_r += wf
                out_lo += wf * float(x[0, p, L.CUR_NIB + 0])
                out_hi += wf * float(x[0, p, L.CUR_NIB + 1])
            # The fixed rescale can only use POSITIONAL constants.  BUT for a LOOP the
            # gather SUMS role-r tokens across ALL frames (all identical value), so
            # out = (Σ_f w_{f,r})·nib.  The correct rescale is by the TOTAL role-r
            # weight wtot_r — also a fixed positional constant.  PROBLEM: wtot_r depends
            # on n_frames, which GROWS every step (unknown at bake time).  This is the
            # fatal flaw: no single fixed constant recovers nib for all sequence lengths.
            rs_lo = out_lo / wtot_r if wtot_r > 0 else 0.0
            rs_hi = out_hi / wtot_r if wtot_r > 0 else 0.0
            resid_lo = abs(rs_lo - nib_lo)
            resid_hi = abs(rs_hi - nib_hi)
            max_resid_global = max(max_resid_global, resid_lo, resid_hi)
            snap_lo = _snap_nib(rs_lo)
            snap_hi = _snap_nib(rs_hi)
            got[_REGKEY[reg]] |= (snap_lo << (4 * (2 * bi + 0)))
            got[_REGKEY[reg]] |= (snap_hi << (4 * (2 * bi + 1)))
        exp = {"pc": pc, "ax": ax, "sp": sp, "bp": bp, "stk": stk}
        ok = all(got[k] == exp[k] for k in exp)
        all_ok = all_ok and ok
        if verbose:
            tag = "OK" if ok else "MISMATCH"
            print(f"  regs={regs} -> got={got} {tag}")
    if verbose:
        print(f"  [recency={recency}, n_frames={n_frames}] min w_i={min_w_global:.3e}  "
              f"max rescale residue={max_resid_global:.3e}  1/min_w={1.0/min_w_global:.3e}")
    return all_ok, min_w_global, max_resid_global


if __name__ == "__main__":
    print("=== IDEAL wide-value single-head ingest (per-role gate, 1/w rescale) ===\n")
    print("--- single frame (no loop): recency only sets the intra-frame w spread ---")
    for rec in (INGEST_RECENCY, 1.0, 0.1, 0.01, 0.0):
        ok, minw, resid = probe_isolated(rec, n_frames=1)
        print(f"  => recency={rec}: {'PASS' if ok else 'FAIL'}\n")
    print("--- 3 identical frames (loop): recency MUST pick the latest ---")
    for rec in (INGEST_RECENCY, 1.0, 0.1, 0.01):
        ok, minw, resid = probe_isolated(rec, n_frames=3)
        print(f"  => recency={rec}: {'PASS' if ok else 'FAIL'}\n")
