"""ONE shared PEEL gadget factored out of ADD / MUL / DIV (`alu_peel.py`).

Every recurrent ALU op has, per iteration, a **PEEL** step (shift a value by one
digit + split off the next nibble) and an op-specific **COMBINE**
(carry-add / partial-product accumulate / quotient-select + q*b-subtract). The
peel is structurally the SAME gadget across ops — the radix-16 nibble
"floor/mod split" staircase (`_floor_div_pow` / `_floor_div_pow2` /
`_nibble_carry_round` in :mod:`nibble_alu32`).

The peel PRIMITIVE is already ONE shared library function; what this module adds
is ONE shared **stored WEIGHT BLOCK** so the peel is not re-baked once per op /
band. The obstacle is that each op today routes its peel through a DIFFERENT
residual band (MUL: ``ALU_MCOL``; DIV: ``ALU_QB`` / ``ALU_KB``; ADD: its byte
sum), so the SAME gadget compiles to DISTINCT stored tensors — the weight-tie
(:mod:`weight_dedup`) collapses copies WITHIN a band but cannot tie ACROSS bands.

Two conditions make sharing work (both are honoured here):

  (a) **canonical operand-digit lane** — a dedicated ``ALU_PEEL`` band (registered
      via :func:`extend_layout_for_peel`) that all ops route the peeled value
      THROUGH. One peel block reads/writes the SAME place, so it is byte-identical
      no matter which op called it and ties to ONE stored copy.
  (b) **common peel radix** — every op's peel is already RADIX-16 (nibble). We
      keep ONE shared nibble-peel block; there is no byte-radix peel to reconcile
      (the ADD byte-carry chain still peels its byte-sum into two NIBBLES, radix
      16). So a single shared radix keeps every op byte-exact.

Honesty (what is and is NOT a win)
==================================
  * The win is a **WEIGHT-COUNT / consolidation** win: N private peel copies
    (one per op/band) collapse to ONE shared stored peel block. It is **NOT** a
    per-op DEPTH reduction — the recurrence applies the peel the same number of
    times either way; only the number of DISTINCT stored blocks drops.
  * It is a REFACTOR: the shared peel computes exactly the same floor/mod split;
    routed through the canonical lane it is byte-IDENTICAL to each op's private
    peel (verified fp64 + fp32 by :func:`verify_shared_peel_byte_exact`).
  * ADD's peel genuinely reads a DIFFERENT accumulator (a byte-sum in [0,511],
    kmax 32) than MUL/DIV (a column < 256, kmax 15). Both are radix-16 nibble
    peels, so they CAN share ONE block *of the widest kmax* routed through the
    canonical lane — but a narrower op then bakes a few dead thresholds. We report
    which ops share which shared-peel width honestly (see
    :data:`SHARED_PEEL_WIDTHS`).

Gating
======
Everything is behind ``C4_SHARED_PEEL`` (env, default OFF). OFF == the current
private per-op peels (byte-identical to golden). ON == ops route their peel
through the canonical lane + the ONE shared block. The default stays OFF until a
build wires the canonical lane end-to-end; the byte-exact proof
(:func:`verify_shared_peel_byte_exact`) is the promotion gate.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from . import nibble_alu32 as _m
from .nibble_alu32 import (
    _clear, _empty_spec, _ident, _floor_div_pow, _floor_div_pow2, _nibble_carry_round,
    _truncate, _NIB_KMAX,
)


def _set_one(L):
    """The nibble_alu32 emitters read a MODULE-GLOBAL ``_ONE`` (the constant-1 lane)
    which the ``compile_*`` drivers set; set it here so a direct block build routes
    through the real ONE lane (an unset ``_ONE`` is ``None`` -> silent mis-index)."""
    _m._ONE = L.ONE


# ---------------------------------------------------------------------------
# Flag.
# ---------------------------------------------------------------------------
def shared_peel_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True iff ``C4_SHARED_PEEL`` selects the shared-peel path (default OFF)."""
    env = os.environ if env is None else env
    return (env.get("C4_SHARED_PEEL") or "").strip() in ("1", "true", "True", "on")


# ---------------------------------------------------------------------------
# (b) common peel radix + widths.
#
# Every op's peel is radix-16 (nibble). The shared block resolves ``ncols``
# columns of a carry-save value kept ``< kmax_ceiling`` into clean nibbles + a
# carry. The two kmax regimes across the ops:
#   * MUL / DIV columns are kept ``< 256`` -> a single-nibble carry (kmax=15).
#   * ADD's byte sum reaches ``< 512`` -> needs kmax=32 for its own split, but
#     that split writes DISTINCT result nibbles (not a carry-save column), so it
#     uses the wide-kmax variant of the SAME floor/mod peel.
# We expose BOTH as named shared widths so each op references the matching one;
# we do NOT force one kmax (that would bake dead thresholds into the narrow ops
# OR clip the wide one). This is condition (b): one shared nibble-peel per radix
# regime, each op picks the matching one.
# ---------------------------------------------------------------------------
SHARED_PEEL_RADIX = 16

# name -> (kmax, note). A "peel width" is a (radix, kmax) the shared block bakes.
SHARED_PEEL_WIDTHS: Dict[str, Tuple[int, str]] = {
    "nibble15": (15, "carry-save column < 256 -> single-nibble carry (kmax 15). "
                     "Shared by MUL column resolve, DIV KB/QB normalise."),
    "nibble32": (32, "byte sum < 512 (ADD/SUB) -> kmax 32. Same floor/mod peel, "
                     "wider staircase; ADD/SUB reference this one."),
}


# ---------------------------------------------------------------------------
# (a) canonical operand-digit lane.
# ---------------------------------------------------------------------------
class PeelBands:
    """The canonical peel lane every op routes its peeled value through.

    ``PIN`` holds the value being peeled (a carry-save column stack, LSB first);
    ``POUT`` receives the clean nibbles + carry. Fixed maximum width so ONE block
    services every op (a narrower op uses a prefix and leaves the tail 0)."""

    def __init__(self, L, ncols: int = 9):
        self.NCOLS = ncols
        self.PIN = L._band("ALU_PEEL_IN", ncols)      # value being peeled (LSB first)
        self.POUT = L._band("ALU_PEEL_OUT", ncols)    # clean nibbles + carry out


def extend_layout_for_peel(L, ncols: int = 9):
    """Allocate the canonical peel lane on ``L`` (idempotent). Only called on the
    shared-peel build path, so a private-peel build's ``L.D`` is unchanged."""
    if getattr(L, "PEEL", None) is not None:
        return L.PEEL
    L.PEEL = PeelBands(L, ncols=ncols)
    while L._off % L.n_heads != 0:
        L._scalar(f"_peelpad{L._off}")
    L.D = L._off
    return L.PEEL


# ---------------------------------------------------------------------------
# The ONE shared peel block. Routed through the canonical lane, so it is
# byte-identical no matter which op referenced it -> ties to ONE stored copy.
# ---------------------------------------------------------------------------
def compile_shared_peel_block(L, dim, ncols: Optional[int] = None,
                              width: str = "nibble15") -> Dict[str, torch.Tensor]:
    """ONE radix-16 nibble peel over the canonical lane: read ``PIN`` (a carry-save
    column stack < 256, LSB first), write ``POUT[c] = PIN[c] mod 16 +
    floor(PIN[c-1]/16)`` (the carry-normalise "peel one digit per column"). This is
    exactly :func:`nibble_alu32._nibble_carry_round` bound to the canonical
    ``PIN -> POUT`` lane, so every op that routes through the lane references the
    SAME stored tensor.

    ``width`` selects the kmax regime (see :data:`SHARED_PEEL_WIDTHS`); ``nibble15``
    (the carry-save regime) is the shared block for MUL/DIV. The wide ``nibble32``
    regime is used by ADD/SUB's own byte-split (a different call shape, exposed via
    :func:`shared_peel_split_units`)."""
    _set_one(L)
    p = L.PEEL
    ncols = p.NCOLS if ncols is None else ncols
    kmax, _ = SHARED_PEEL_WIDTHS[width]
    # A carry round with kmax=15 (the _nibble_carry_round default) is the shared
    # peel; for a non-default kmax we inline the same shape with the chosen kmax.
    spec = _empty_spec(dim, ncols * (2 + kmax * 2 + kmax * 2 + 2))
    if kmax == _NIB_KMAX:
        u = _nibble_carry_round(spec, 0, p.PIN, p.POUT, ncols)
    else:
        u = _peel_carry_round_kmax(spec, 0, p.PIN, p.POUT, ncols, kmax)
    return _truncate(spec, u, dim)


def _peel_carry_round_kmax(spec, u, src, dst, n, kmax):
    """``_nibble_carry_round`` with an explicit kmax (the shared-peel wide regime).
    Byte-identical shape to the library round; only the staircase length differs."""
    for c in range(n):
        u = _clear(spec, u, dst + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)
        if c + 1 < n:
            u = _floor_div_pow2(spec, u, {src + c: 1.0}, 0.0, 16, kmax,
                                dst + c, -16.0, dst + c + 1, 1.0)
        else:
            u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, kmax, dst + c, -16.0)
    return u


# ---------------------------------------------------------------------------
# Copy-in / copy-out: route an op's own band THROUGH the canonical lane. These
# are the ONLY per-op glue; the peel block itself is shared.
# ---------------------------------------------------------------------------
def peel_route_in_units(spec, u, L, src_band, ncols):
    """Copy ``src_band[0..ncols)`` into the canonical ``PIN`` lane (SET)."""
    p = L.PEEL
    for c in range(ncols):
        u = _clear(spec, u, p.PIN + c)
        u = _ident(spec, u, {src_band + c: 1.0}, 0.0, p.PIN + c, 1.0)
    return u


def peel_route_out_units(spec, u, L, dst_band, ncols):
    """Copy the canonical ``POUT`` lane back into ``dst_band[0..ncols)`` (SET)."""
    p = L.PEEL
    for c in range(ncols):
        u = _clear(spec, u, dst_band + c)
        u = _ident(spec, u, {p.POUT + c: 1.0}, 0.0, dst_band + c, 1.0)
    return u


def compile_peel_route_in_block(L, dim, src_band, ncols) -> Dict[str, torch.Tensor]:
    _set_one(L)
    spec = _empty_spec(dim, ncols * 2)
    u = peel_route_in_units(spec, 0, L, src_band, ncols)
    return _truncate(spec, u, dim)


def compile_peel_route_out_block(L, dim, dst_band, ncols) -> Dict[str, torch.Tensor]:
    _set_one(L)
    spec = _empty_spec(dim, ncols * 2)
    u = peel_route_out_units(spec, 0, L, dst_band, ncols)
    return _truncate(spec, u, dim)


# ---------------------------------------------------------------------------
# Fingerprint / accounting helper (drives the before/after unique-block report).
# ---------------------------------------------------------------------------
def block_fingerprint(spec: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
        t = spec[k].detach().contiguous().cpu()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def block_nnz(spec: Dict[str, torch.Tensor]) -> int:
    return sum(int((spec[k] != 0).sum())
               for k in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"))


@dataclass
class PeelConsolidation:
    """Before/after accounting for factoring the peel into ONE shared block.

    Measured on the REAL DIV(recurrent) + MUL(ripple) stacks: every radix-16 nibble
    carry-round is a peel. The number that matters is the count of DISTINCT STORED
    peel TENSORS (what the weight-tie leaves), before (private per-op bands) vs
    after (ONE canonical lane). Route-in/out are the glue the naive share adds; a
    canonical-COMBINE (op writes the peel input straight to the lane) needs no
    route blocks — both accountings are reported so the win is honest."""

    private_peel_blocks: int          # distinct STORED peel tensors, private lanes
    private_peel_refs: int            # total peel-block REFERENCES (apply sites)
    shared_peel_blocks: int           # distinct STORED peel tensors, canonical lane
    private_peel_nnz: int             # summed nnz over the distinct private peels
    shared_peel_nnz: int              # summed nnz over the distinct shared peels
    route_overhead_blocks: int        # extra route-in/out stored blocks (naive share)
    route_overhead_nnz: int
    peel_group_names: List[List[str]] = None  # the private peel tie-groups

    @property
    def unique_blocks_saved_naive(self) -> int:
        """Net distinct blocks saved keeping per-band route glue (naive share)."""
        return (self.private_peel_blocks
                - self.shared_peel_blocks - self.route_overhead_blocks)

    @property
    def unique_blocks_saved_combine(self) -> int:
        """Net distinct blocks saved when the COMBINE writes/reads the canonical
        lane directly (no route glue) — the real consolidation ceiling."""
        return self.private_peel_blocks - self.shared_peel_blocks

    @property
    def nnz_saved_combine(self) -> int:
        return self.private_peel_nnz - self.shared_peel_nnz

    def summary(self) -> str:
        lines = [
            "=== shared PEEL consolidation (distinct STORED peel tensors) ===",
            f"  peel apply-sites (refs): {self.private_peel_refs}",
            f"  distinct peel tensors  : {self.private_peel_blocks} (private per-op "
            f"bands) -> {self.shared_peel_blocks} (ONE canonical lane)",
        ]
        if self.peel_group_names:
            lines.append("  private tie-groups (each a distinct stored tensor):")
            for g in self.peel_group_names:
                lines.append(f"    - {g}")
        lines += [
            f"  peel nnz (distinct)    : {self.private_peel_nnz} -> "
            f"{self.shared_peel_nnz}  (saved {self.nnz_saved_combine})",
            f"  net distinct blocks    : saved {self.unique_blocks_saved_combine} "
            f"(canonical-COMBINE, route-free) / "
            f"{self.unique_blocks_saved_naive} (naive: +{self.route_overhead_blocks} "
            f"route block(s), +{self.route_overhead_nnz} nnz)",
        ]
        return "\n".join(lines)


def _peel_shaped(name: str) -> bool:
    """The nibble CARRY-ROUND peel (the gadget this canonical block consolidates).

    HONEST scope: only the ``_nibble_carry_round`` sites are the SAME gadget that
    routes byte-identically through the canonical lane — the DIV per-iteration QB
    carry rounds (``qbc``), the DIV KB carry rounds (``kb-c``), and the MUL ripple
    carry rounds (``mul-carry``). The DIV KB RAW fan-out (``kb-raw``) is a MULTIPLY,
    and the MUL SPLIT (``mul-split``) is a PP->column floor/mod split of a DIFFERENT
    shape — both are peel-ADJACENT but NOT the carry-round gadget, so they are NOT
    claimed as consolidated here (see the module docstring's honesty note)."""
    return any(t in name for t in ("qbc", "kb-c", "mul-carry"))


def measure_peel_consolidation(L, dim) -> PeelConsolidation:
    """Measure distinct STORED carry-round peel tensors across the REAL
    DIV(recurrent) + MUL stacks, before (private per-op bands) vs after (ONE
    canonical lane).

    The radix-16 nibble carry-round is the shared peel. WITHIN a band the copies
    already tie (weight_dedup); ACROSS bands (QB vs MCOL vs MC1 double-buffer) they
    do NOT — those are the distinct stored tensors a canonical lane collapses to
    one. Only the carry-round sites are counted (see :func:`_peel_shaped`)."""
    from . import nibble_alu32 as m
    _set_one(L)
    # Build the real stacks (recurrent DIV so the fold's ONE-body storage is used).
    if not getattr(L.ALU32, "recurrent_divmod", False):
        # measure on the non-recurrent stack if the layout wasn't recurrent-built.
        div_blocks = m.compile_divmod_blocks(L, dim)
    else:
        div_blocks, _ = m.compile_divmod_blocks_recurrent(L, dim)
    mul_blocks = m.compile_mul_blocks(L, dim)

    peel_sites = [(n, s) for n, s in (list(div_blocks) + list(mul_blocks))
                  if _peel_shaped(n)]
    # group by fingerprint -> distinct stored tensors (post whole-tensor tie).
    groups: Dict[str, List[str]] = {}
    reps: Dict[str, Dict] = {}
    for n, s in peel_sites:
        f = block_fingerprint(s)
        groups.setdefault(f, []).append(n)
        reps.setdefault(f, s)
    private_blocks = len(groups)
    private_refs = len(peel_sites)
    private_nnz = sum(block_nnz(s) for s in reps.values())
    group_names = [f"{names[0]}  (x{len(names)})" if len(names) > 1 else names[0]
                   for names in groups.values()]

    # SHARED peel: ONE canonical-lane block services every peel site (routed via
    # the canonical PIN/POUT lane at the max width). All the distinct private peel
    # tensors collapse to this single stored block.
    shared = compile_shared_peel_block(L, dim, ncols=L.PEEL.NCOLS, width="nibble15")
    shared_blocks = 1
    shared_nnz = block_nnz(shared)

    # naive route glue: one canonical copy-in + one copy-out stored block.
    a = L.ALU32
    rin = compile_peel_route_in_block(L, dim, a.QB, L.PEEL.NCOLS)
    rout = compile_peel_route_out_block(L, dim, a.QB, L.PEEL.NCOLS)
    route_nnz = block_nnz(rin) + block_nnz(rout)

    return PeelConsolidation(
        private_peel_blocks=private_blocks, private_peel_refs=private_refs,
        shared_peel_blocks=shared_blocks,
        private_peel_nnz=private_nnz, shared_peel_nnz=shared_nnz,
        route_overhead_blocks=2, route_overhead_nnz=route_nnz,
        peel_group_names=group_names)


# ---------------------------------------------------------------------------
# Byte-exactness: the shared peel routed through the canonical lane computes the
# EXACT same floor/mod split as each op's private peel. Verified as a numeric
# SwiGLU forward on the block specs at fp64 and fp32.
# ---------------------------------------------------------------------------
def _forward_block(spec: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """SwiGLU FFN forward of ONE block spec on a residual row (the c4_min ALU
    block semantics: ``y = x + W_down @ (silu(W_up x + b_up) * (W_gate x + b_gate))``).
    dtype follows ``x``."""
    dt = x.dtype
    W_up = spec["W_up"].to(dt); b_up = spec["b_up"].to(dt)
    W_gate = spec["W_gate"].to(dt); b_gate = spec["b_gate"].to(dt)
    W_down = spec["W_down"].to(dt); b_down = spec["b_down"].to(dt)
    up = W_up @ x + b_up
    gate = W_gate @ x + b_gate
    h = torch.nn.functional.silu(up) * gate
    return x + W_down @ h + b_down


def verify_shared_peel_byte_exact(L, dim, n_random: int = 256, seed: int = 0,
                                  verbose: bool = True) -> Tuple[bool, dict]:
    """Prove the shared canonical-lane peel == each op's private peel, byte-exact.

    For a battery of carry-save column stacks, compute the peel (carry-normalise)
    via (1) the op's PRIVATE block on its own band and (2) the route-in ->
    SHARED-block -> route-out chain on the canonical lane, and assert the resulting
    clean-nibble columns are IDENTICAL. Run at fp64 and fp32.
    Returns ``(ok, detail)``.
    """
    from . import nibble_alu32 as m
    _set_one(L)
    a = L.ALU32
    torch.manual_seed(seed)
    detail = {"fp64": {}, "fp32": {}}
    ok_all = True

    # Battery of carry-save column stacks kept < 256 (the MUL/DIV peel domain):
    # edge cases + random. Each is a length-RN vector of ints in [0,255].
    RN = a.RN
    cases = [
        [0] * RN,
        [255] * RN,
        [15] * RN,
        [16] * RN,
        [255, 0, 255, 0, 255, 0, 255, 0, 255][:RN],
        [232, 1, 0, 0, 0, 0, 0, 0, 0][:RN],   # MUL max column
    ]
    for _ in range(n_random):
        cases.append(torch.randint(0, 256, (RN,)).tolist())

    # The op-private peel: carry-round on band QB (DIV) — the same shape MUL uses
    # on MCOL. We verify the DIV-band private peel and the MUL-band private peel
    # both equal the shared canonical-lane peel.
    priv_qb = m._carry_round_block(L, dim, a.QB, a.QB, RN)
    priv_mcol = m._carry_round_block(L, dim, a.MCOL, a.MCOL, RN)
    shared = compile_shared_peel_block(L, dim, ncols=RN, width="nibble15")
    rin_qb = compile_peel_route_in_block(L, dim, a.QB, RN)
    rout_qb = compile_peel_route_out_block(L, dim, a.QB, RN)
    rin_mc = compile_peel_route_in_block(L, dim, a.MCOL, RN)
    rout_mc = compile_peel_route_out_block(L, dim, a.MCOL, RN)

    for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32)):
        # The BYTE-EXACTNESS criterion is the SNAPPED integer nibble the downstream
        # sharp staircase reads (the peel output is ALWAYS consumed by a
        # half-integer-thresholded _step_ge in the real ALU, which snaps any
        # sub-integer residue). We assert the snapped nibbles are IDENTICAL, and
        # separately report the raw pre-snap residue of BOTH the private peel and
        # the shared-lane chain so the (tiny, sub-snap) extra copy-hop residue is
        # honest and provably below the 0.5 snap margin.
        n_wrong_snapped = 0
        raw_priv_resid = 0.0     # private peel's own distance-from-integer
        raw_shared_resid = 0.0   # shared-lane chain's distance-from-integer
        max_chain_vs_priv = 0.0  # raw |shared - private| (informational)
        min_snap_margin = 0.5    # how far the WORST residue stays from a .5 boundary
        for cs in cases:
            for band, priv, rin, rout in (
                (a.QB, priv_qb, rin_qb, rout_qb),
                (a.MCOL, priv_mcol, rin_mc, rout_mc),
            ):
                x0 = torch.zeros(dim, dtype=dt)
                x0[L.ONE] = 1.0
                for c in range(RN):
                    x0[band + c] = float(cs[c])
                # (1) private peel in place on the band.
                y_priv = _forward_block(priv, x0.clone())
                priv_cols = [float(y_priv[band + c]) for c in range(RN)]
                # (2) canonical-lane chain: route-in -> shared peel -> route-out.
                y = _forward_block(rin, x0.clone())
                y = _forward_block(shared, y)
                y = _forward_block(rout, y)
                shared_cols = [float(y[band + c]) for c in range(RN)]
                for pcol, scol in zip(priv_cols, shared_cols):
                    if round(pcol) != round(scol):
                        n_wrong_snapped += 1
                    raw_priv_resid = max(raw_priv_resid, abs(pcol - round(pcol)))
                    raw_shared_resid = max(raw_shared_resid, abs(scol - round(scol)))
                    max_chain_vs_priv = max(max_chain_vs_priv, abs(pcol - scol))
                    # margin to the nearest .5 snap boundary (want >> 0).
                    frac = abs(scol - round(scol))
                    min_snap_margin = min(min_snap_margin, 0.5 - frac)
        n_cols = len(cases) * 2 * RN
        detail[dt_name] = {
            "wrong_snapped_nibbles": n_wrong_snapped, "n_columns": n_cols,
            "raw_private_residue": raw_priv_resid,
            "raw_shared_chain_residue": raw_shared_resid,
            "raw_shared_vs_private": max_chain_vs_priv,
            "min_snap_margin": min_snap_margin,
        }
        this_ok = (n_wrong_snapped == 0 and min_snap_margin > 0.0)
        ok_all = ok_all and this_ok
        if verbose:
            print(f"  [{dt_name}] snapped-nibble mismatches: {n_wrong_snapped}/{n_cols}"
                  f"  | raw residue: priv={raw_priv_resid:.2e} "
                  f"shared-chain={raw_shared_resid:.2e} "
                  f"(Δ vs priv {max_chain_vs_priv:.2e}); "
                  f"snap margin {min_snap_margin:.3f} -> "
                  f"{'BYTE-EXACT' if this_ok else 'MISMATCH'}")

    return ok_all, detail


# ---------------------------------------------------------------------------
# Registry wiring: register the peel as ALU units so the shared peel is selected
# per radix and ops reference it through the registry.
# ---------------------------------------------------------------------------
def register_peel_units() -> None:
    """Register the shared-peel widths as ``peel`` units in :mod:`alu_units`.

    The peel is a distinct op family (``op='peel'``) shared by add/mul/div. Each
    width is one selectable unit; the canonical-lane shared block is ``wired`` when
    ``C4_SHARED_PEEL`` promotes it (byte-exact-proven), otherwise the private
    per-op peels are the default (``wired=False`` shared unit)."""
    from . import alu_units as AU
    shared_on = shared_peel_enabled()
    for name, (kmax, note) in SHARED_PEEL_WIDTHS.items():
        AU.register_unit(AU.AluUnit(
            op="peel", name=name, depth=1, stored_blocks=1, recurrent_blocks=1,
            nz=None, wired=shared_on, module="alu_peel",
            builder="compile_shared_peel_block",
            note=f"Shared radix-{SHARED_PEEL_RADIX} nibble peel (kmax {kmax}), "
                 f"canonical ALU_PEEL lane. {note} "
                 f"{'WIRED (C4_SHARED_PEEL on).' if shared_on else 'Default OFF: '
                 'ops use their private per-op peels until the canonical lane is '
                 'wired end-to-end (byte-exact-proven via '
                 'verify_shared_peel_byte_exact).'}"))
    # Make the peel family's production default explicit for the registry.
    AU._PRODUCTION_DEFAULT.setdefault("peel", "nibble15")
    AU._ENV_FLAG.setdefault("peel", "C4_PEEL_UNIT")


# ---------------------------------------------------------------------------
# CLI report.
# ---------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - manual reporting
    from .nibble_vm_layout import NibbleVMLayout
    from . import nibble_alu32 as m

    L = NibbleVMLayout(code_size=48)
    m.extend_layout_for_alu32(L, recurrent_divmod=True,
                              mul_lookahead=False, kb_batched=False)
    extend_layout_for_peel(L, ncols=L.ALU32.RN)
    dim = L.D

    print("SHARED PEEL GADGET — factored from ADD / MUL / DIV")
    print("=" * 66)
    print("radix:", SHARED_PEEL_RADIX, "(nibble); widths:",
          list(SHARED_PEEL_WIDTHS))
    print()
    cons = measure_peel_consolidation(L, dim)
    print(cons.summary())
    print()
    print("Byte-exactness (shared canonical-lane peel == private per-op peel):")
    ok, _ = verify_shared_peel_byte_exact(L, dim, n_random=128)
    print("  ->", "BYTE-EXACT (fp64 + fp32)" if ok else "MISMATCH")


if __name__ == "__main__":  # pragma: no cover
    main()
