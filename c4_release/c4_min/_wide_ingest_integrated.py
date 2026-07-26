"""INTEGRATED wide-value single-head ingest — through the REAL FFN + Attention.

Refute-or-confirm of a8c09504's "20 query heads minimum".  Builds the FULL wide-value
single-head ingest and drives it through the codebase's REAL ``FFN.forward`` and
``Attention.forward`` (softmax1 + ALiBi), reconstructing the register frame BYTE-EXACT
on the a8c09504 battery (incl. multi-frame loops where recency must pick the latest).

The wide-value ingest has THREE stages (all in-weights, no python compute):

  (A) PRE-ROUTE FFN — a SwiGLU block at each frame-byte position that computes the
      per-role product  ``PREROUTE[2r+b] = CUR_NIB[b] · [ROLE==r]``  into 40 fresh
      residual dims.  This is the gating a LINEAR W_v cannot do (role & nibble are
      ADDED on the residual, never multiplied); the SwiGLU ``silu(up)·gate`` product
      supplies it.  Needed because otherwise all 20 tokens dump CUR_NIB into the SAME
      value lane => the destructive blend a8c09504 observed.
  (B) WIDE-VALUE SINGLE HEAD — ONE query head + ONE KV head.  Role-AGNOSTIC query
      (every frame-byte token gets the same base match M); V = identity copy of the 40
      PREROUTE dims.  out lane 2r+b = Σ_i w_i · PREROUTE_i[2r+b] = (Σ_frames w_{f,r}) ·
      nib_{r,b}  — a SCALED CONCATENATION, each register in its own lane.
  (C) RESCALE FFN — lane 2r+b *= 1/wtot_r, where wtot_r = Σ_frames w_{f,r} is the
      FIXED positional weight fraction (invariant to sequence length because the huge
      match logit M makes softmax1 scale-free in n_frames).  Then the nibble argmax
      re-quant snaps.
"""
from __future__ import annotations

from typing import Dict

import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min.blogspec_model import FFN, Attn, softmax1
from c4_min.nibble_vm import S, SILU_S, SILU_HALF
from c4_min.nibble_pure_forward import (
    PureForwardLayout, N_ROLES, INGEST_REGS, BYTES_PER_REG, INGEST_EFF,
    build_frame_tokens, make_overlay, _snap_nib, _FRAME_ROLE_SLOTS,
    WIDE_INGEST_RECENCY,
)


TEST_VECTORS = [
    (2, 13, 0x10000, 0x10000, 6),
    (0, 0, 0x10000, 0x10000, 0),
    (5, 255, 0xFFFC, 0x10000, 128),
    (0xABCD, 0x12345678, 0xFFFC, 0x10000, 0xDEADBEEF),
]
_ROLE_TO_LOCAL = {rr: lc for lc, rr in _FRAME_ROLE_SLOTS.items()}
_REGKEY = {"PC": "pc", "AX": "ax", "SP": "sp", "BP": "bp", "STACK0": "stk"}
WIDE_RECENCY = WIDE_INGEST_RECENCY   # 0.5: inside the fp32 byte-exact window


def _mk_layout():
    """The base pure-forward layout + a private 40-dim PREROUTE band."""
    L = PureForwardLayout(code_size=16, n_heads=20, include_memory=False,
                          include_cmp=False)
    from c4_min import nibble_bitwise as _bw
    _bw.extend_layout_for_bitwise(L)
    if _bw.tight_shift_enabled():
        for _op in (isa.SHL, isa.SHR):
            _bw.extend_layout_for_tight_shift(L, _op)
    # PREROUTE: 2*N_ROLES = 40 fresh dims (per (role, byte-half) value lane).
    # GATHER:   2*N_ROLES = 40 fresh dims where the attention W_o writes the SCALED
    #           concat (separate from PREROUTE so the query-row's own polluted PREROUTE
    #           is not summed into the gather).
    L.PREROUTE = L._band("PREROUTE", 2 * N_ROLES)
    L.GATHER = L._band("GATHER", 2 * N_ROLES)
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


def build_preroute_ffn(L, dim) -> FFN:
    """(A) SwiGLU: PREROUTE[2r+b] = CUR_NIB[b] · [ROLE==r]  (the per-role gate a
    linear W_v cannot do).  One hidden unit per (role r, byte-half b):
      gate = CUR_NIB[b]                       (the nibble value 0..15)
      up   = S·(ROLE+r)  - 0.5·S              (>0 iff ROLE==r ; silu -> S else 0)
      down = 1/silu(0.5S) · into PREROUTE[2r+b]
    => hidden = silu(up)·gate = (ROLE==r ? S : 0)·nib ; down scales silu(0.5S)->1."""
    n_units = 2 * N_ROLES
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    u = 0
    for r in range(N_ROLES):
        for b in range(2):
            W_gate[u, L.CUR_NIB + b] = 1.0          # gate = the nibble value
            W_up[u, L.ROLE + r] = S                  # up = S iff ROLE==r ...
            b_up[u] = -0.5 * S                       # ... (silu(0.5S) when on, ~0 off)
            W_down[L.PREROUTE + 2 * r + b, u] = 1.0 / SILU_HALF
            u += 1
    f = FFN(dim, n_units)
    with torch.no_grad():
        f.W_up.copy_(W_up); f.b_up.copy_(b_up)
        f.W_gate.copy_(W_gate); f.b_gate.copy_(b_gate)
        f.W_down.copy_(W_down); f.b_down.copy_(b_down)
    return f


def build_wide_ingest_attn(L, dim, recency) -> Attention:
    """(B) ONE query head + ONE KV head.  head_dim = dim (n_heads=1).
      K: match ch keys +smag on ONE (role-AGNOSTIC); penalty ch keys -p·ONE + p·IS_FRAME
      Q: match ch +smag on ONE ; penalty ch +p·ONE
      V: identity copy of the 40 PREROUTE dims into value channels 2..41
      O: copy value channels 2..41 -> PREROUTE (in place); rescale done in stage C."""
    attn = Attn(dim, n_heads=1, positional="alibi", sink="softmax1")
    hs = attn.scale
    smag = (INGEST_EFF / hs) ** 0.5
    PEN = 100.0 * INGEST_EFF
    p = (PEN / hs) ** 0.5
    for w in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        w.data.zero_()
    attn.alibi_slopes[0] = recency
    cMATCH, cPEN = 0, 1
    attn.W_k.data[cMATCH, L.ONE] = smag
    attn.W_q.data[cMATCH, L.ONE] = smag
    attn.W_k.data[cPEN, L.ONE] = -p
    attn.W_k.data[cPEN, L.IS_FRAME_BYTE] = p
    attn.W_q.data[cPEN, L.ONE] = p
    for k in range(2 * N_ROLES):
        vch = 2 + k
        attn.W_v.data[vch, L.PREROUTE + k] = 1.0
        # O writes the gathered (scaled) value into the FRESH GATHER band (initially 0
        # on every row), so out = x + W_o@gather leaves GATHER = the pure scaled concat
        # (no self-pollution from the query-row's own PREROUTE).
        attn.W_o.data[L.GATHER + k, vch] = 1.0
    return attn


def build_rescale_ffn(L, dim, wtot: Dict[int, float]) -> FFN:
    """(C) SwiGLU: register nibble band[reg,2*bi+b] = PREROUTE[2r+b] / wtot_r, and
    self-clear the register band first.  ``wtot`` maps role r -> Σ_frames w_{f,r} (the
    fixed positional weight fraction).  Also cancels the query-row PREROUTE self-term."""
    reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP, "STACK0": L.STACK0}
    # per role: read GATHER[2r+b]/wtot_r -> reg nibble band, plus clear the band.
    units = []
    for r in range(N_ROLES):
        r_idx, bi = divmod(r, BYTES_PER_REG)
        reg_base = reg_bases[INGEST_REGS[r_idx]]
        inv = 1.0 / wtot[r]
        for b in range(2):
            dst = reg_base + 2 * bi + b
            src = L.GATHER + 2 * r + b            # read the SCALED concat from GATHER
            units.append(("read", dst, src, inv))
    # self-clear each destination reg nibble dim first (SET semantics).
    clears = set()
    for _, dst, _, _ in units:
        clears.add(dst)
    n_units = len(clears) + len(units)
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    u = 0
    for dst in sorted(clears):
        # clear: hidden = silu(S)/S · (current dst) with down -1 => subtract dst.
        W_up[u, L.ONE] = S
        W_gate[u, dst] = 1.0
        W_down[dst, u] = -1.0 / SILU_S
        u += 1
    for _, dst, src, inv in units:
        # read: hidden = silu(S)/S · PREROUTE_src ; down (+inv) => add src*inv.
        W_up[u, L.ONE] = S
        W_gate[u, src] = 1.0
        W_down[dst, u] = inv / SILU_S
        u += 1
    f = FFN(dim, n_units)
    with torch.no_grad():
        f.W_up.copy_(W_up); f.b_up.copy_(b_up)
        f.W_gate.copy_(W_gate); f.b_gate.copy_(b_gate)
        f.W_down.copy_(W_down); f.b_down.copy_(b_down)
    return f


def _wtot_fractions(recency, qpos, S_len, n_frames):
    """Σ_frames w_{f,r} for each role r (the fixed rescale constants)."""
    M = INGEST_EFF
    scores = torch.full((S_len,), -1e30, dtype=torch.float64)
    for f in range(n_frames):
        fbase = 1 + f * V.FRAME_LEN
        for local in _FRAME_ROLE_SLOTS:
            scores[fbase + local] = M - recency * abs(qpos - (fbase + local))
    wsm = softmax1(scores.unsqueeze(0)).squeeze(0)
    wtot = {}
    for r in range(N_ROLES):
        local_r = _ROLE_TO_LOCAL[r]
        wtot[r] = sum(float(wsm[1 + f * V.FRAME_LEN + local_r]) for f in range(n_frames))
    return wtot


def _gather_reg(state, L, reg_bases):
    state = state.detach()
    got = {v: 0 for v in _REGKEY.values()}
    worst = 0.0
    for r in range(N_ROLES):
        r_idx, bi = divmod(r, BYTES_PER_REG)
        reg = INGEST_REGS[r_idx]
        reg_base = reg_bases[reg]
        vlo = float(state[reg_base + 2 * bi + 0])
        vhi = float(state[reg_base + 2 * bi + 1])
        worst = max(worst, abs(vlo - round(vlo)), abs(vhi - round(vhi)))
        got[_REGKEY[reg]] |= (_snap_nib(vlo) << (4 * (2 * bi + 0)))
        got[_REGKEY[reg]] |= (_snap_nib(vhi) << (4 * (2 * bi + 1)))
    return got, worst


def _run_stream(L, E, frame_specs, recency, dtype, wtot):
    """Drive [BOS] + frames through pre-route FFN -> wide head -> rescale FFN (REAL
    FFN.forward + Attn.forward) at ``dtype``.  Returns the decoded last-frame regs."""
    dim = L.D
    _cast = (lambda m: m.double()) if dtype == torch.float64 else (lambda m: m)
    pre = _cast(build_preroute_ffn(L, dim))
    attn = _cast(build_wide_ingest_attn(L, dim, recency))
    resc = _cast(build_rescale_ffn(L, dim, wtot))
    frames = []
    for fr in frame_specs:
        frames += build_frame_tokens(*fr)
    toks = torch.tensor([[V.BOS] + frames])
    overlay = make_overlay(isa.assemble([("IMM", 0), ("HALT", 0)]), L)
    x = E[toks].clone()
    if dtype == torch.float64:
        x = x.double()
    overlay(x)
    x = pre(x)                         # STAGE A: per-role gate (SwiGLU product)
    x = attn(x)                        # STAGE B: wide-value single head (scaled concat)
    x = resc(x)                        # STAGE C: 1/wtot rescale (fixed positional consts)
    return _gather_reg(x[0, -1], L, {"PC": L.PC, "AX": L.AX, "SP": L.SP,
                                     "BP": L.BP, "STACK0": L.STACK0})


def run(recency=WIDE_RECENCY, n_frames=1, dtype=torch.float64, verbose=True):
    """Wide-ingest byte-exactness on the a8c09504 battery (IDENTICAL loop frames).
    ``wtot`` is baked at n_frames=1 (length-invariant) and applied at ``n_frames``."""
    L = _mk_layout()
    E = _embed(L)
    S_bake = 1 + V.FRAME_LEN
    if 1.0 / min(_wtot_fractions(recency, S_bake - 1, S_bake, 1).values()) > 3.0e38:
        if verbose:
            print(f"  recency={recency}: 1/wtot > fp32 max — SKIP")
        return None
    wtot = _wtot_fractions(recency, S_bake - 1, S_bake, 1)
    all_ok = True
    for regs in TEST_VECTORS:
        got, _ = _run_stream(L, E, [regs] * n_frames, recency, dtype, wtot)
        exp = dict(zip(("pc", "ax", "sp", "bp", "stk"), regs))
        ok = all(got[k] == exp[k] for k in exp)
        all_ok = all_ok and ok
        if verbose:
            print(f"  regs={regs} -> got={got} {'OK' if ok else 'MISMATCH'}")
    if verbose:
        print(f"  [recency={recency} n_frames={n_frames} {dtype}] "
              f"min wtot={min(wtot.values()):.4e} 1/min={1.0/min(wtot.values()):.2f}")
    return all_ok


def run_differing(recency, dtype=torch.float32, n_prev=8, verbose=True):
    """The REAL loop-correctness test: n_prev DIFFERING prior frames + a distinct
    latest frame.  The ingest must recover the LATEST frame's regs, NOT blend earlier
    ones — the constraint that forces recency HIGH ENOUGH (vs the fp32-rescale
    constraint that forces it LOW).  Returns byte-exact bool + worst nibble residue."""
    L = _mk_layout()
    E = _embed(L)
    S_bake = 1 + V.FRAME_LEN
    wtot = _wtot_fractions(recency, S_bake - 1, S_bake, 1)
    if 1.0 / min(wtot.values()) > 3.0e38:
        return None, None
    prev = [(i, (10 * i) & 0xFF, 0x10000, 0x10000, (7 * i) & 0xFF)
            for i in range(1, n_prev + 1)]
    latest = (9, 42, 0xFFF0, 0x10000, 7)
    exp = {"pc": 9, "ax": 42, "sp": 0xFFF0, "bp": 0x10000, "stk": 7}
    got, worst = _run_stream(L, E, prev + [latest], recency, dtype, wtot)
    ok = all(got[k] == exp[k] for k in exp)
    return ok, worst


if __name__ == "__main__":
    print("=== INTEGRATED wide-value single-head ingest (real FFN + Attn) ===\n")
    print("--- IDENTICAL loop frames, a8c09504 battery (fp64, wtot baked @ n=1) ---")
    for nf in (1, 3, 8):
        print(f"n_frames={nf}:", "PASS" if run(n_frames=nf, verbose=False) else "FAIL")
    print("\n--- REAL loop: 8 DIFFERING prior frames, latest must win (fp32) ---")
    print("recency | byte-exact | worst residue | verdict")
    for rec in (1.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.15, 0.12, 0.10, 0.05):
        ok, worst = run_differing(rec)
        if ok is None:
            print(f"  {rec:5.2f} | SKIP (1/wtot > fp32 max)")
            continue
        print(f"  {rec:5.2f} | {'PASS' if ok else 'FAIL':4s} | {worst:.4f} | "
              f"{'byte-exact' if ok else ('rescale-noise' if worst>0.3 and rec>1 else 'frame-blend')}")
    print("\nfp32 byte-exact window (differing frames): recency ~[0.12, 1.0], "
          "sweet spot ~0.5")
