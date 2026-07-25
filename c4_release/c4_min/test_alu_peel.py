"""Gate for the shared PEEL gadget (``c4_min.alu_peel``).

Covers:
  * BYTE-EXACTNESS: the canonical-lane shared peel computes the SAME radix-16
    nibble carry-normalise (peel) as each op's private per-op peel, at fp64 AND
    fp32 (snapped-nibble identical, residue below the 0.5 snap margin) — the
    promotion gate for the shared peel.
  * WIDTH-PREFIX SHARE: the width-8 MUL carry-round is byte-identical to the 8-col
    prefix of the width-9 canonical peel (so ONE shared block services both widths).
  * CONSOLIDATION: the distinct STORED carry-round peel tensors collapse from 3
    (QB / MCOL / MC1 private bands) to 1 (canonical lane); the accounting adds up.
  * GATING: ``C4_SHARED_PEEL`` default OFF -> the peel layout band is NOT allocated
    (a private-peel build's L.D is unchanged); registry unit wired follows the flag.
  * REGISTRY: the peel widths register as ``op='peel'`` units.

Run: ``OMP_NUM_THREADS=4 python -m pytest c4_min/test_alu_peel.py -v``
"""
from __future__ import annotations

import os

import pytest
import torch

from c4_min import alu_peel as P
from c4_min import nibble_alu32 as m
from c4_min.nibble_vm_layout import NibbleVMLayout


def _layout(recurrent=True):
    L = NibbleVMLayout(code_size=48)
    m.extend_layout_for_alu32(L, recurrent_divmod=recurrent,
                              mul_lookahead=False, kb_batched=False)
    P.extend_layout_for_peel(L, ncols=L.ALU32.RN)
    return L, L.D


# ---------------------------------------------------------------------------
# Byte-exactness: the shared peel == the private per-op peel.
# ---------------------------------------------------------------------------
def test_shared_peel_byte_exact_fp64_fp32():
    L, dim = _layout()
    ok, detail = P.verify_shared_peel_byte_exact(L, dim, n_random=256, verbose=False)
    assert ok, detail
    # snapped-nibble identity + snap margin strictly > 0 (residue below .5).
    for dt in ("fp64", "fp32"):
        assert detail[dt]["wrong_snapped_nibbles"] == 0, (dt, detail[dt])
        assert detail[dt]["min_snap_margin"] > 0.0, (dt, detail[dt])
    # the shared-chain residue stays tiny (extra copy hops) — well under the snap.
    assert detail["fp64"]["raw_shared_chain_residue"] < 1e-3
    assert detail["fp32"]["raw_shared_chain_residue"] < 1e-2


def test_width8_mul_carry_equals_width9_canonical_prefix():
    """The width-8 MUL carry-round IS the 8-col prefix of the width-9 canonical
    peel — so ONE shared block services both the DIV (9) and MUL (8) widths."""
    L, dim = _layout()
    a = L.ALU32
    P._set_one(L)
    priv = m._carry_round_block(L, dim, a.MCOL, a.MCOL, 8)
    shared = P.compile_shared_peel_block(L, dim, ncols=a.RN, width="nibble15")
    rin = P.compile_peel_route_in_block(L, dim, a.MCOL, 8)
    rout = P.compile_peel_route_out_block(L, dim, a.MCOL, 8)
    torch.manual_seed(3)
    for _ in range(200):
        cs = torch.randint(0, 256, (8,)).tolist()
        x = torch.zeros(dim, dtype=torch.float64)
        x[L.ONE] = 1.0
        for c in range(8):
            x[a.MCOL + c] = float(cs[c])
        yp = P._forward_block(priv, x.clone())
        pcols = [round(float(yp[a.MCOL + c])) for c in range(8)]
        y = P._forward_block(rin, x.clone())
        y = P._forward_block(shared, y)
        y = P._forward_block(rout, y)
        scols = [round(float(y[a.MCOL + c])) for c in range(8)]
        assert pcols == scols, (cs, pcols, scols)


# ---------------------------------------------------------------------------
# Consolidation accounting.
# ---------------------------------------------------------------------------
def test_consolidation_collapses_carry_round_peels_to_one():
    L, dim = _layout()
    cons = P.measure_peel_consolidation(L, dim)
    # 3 distinct private carry-round peels (QB / MCOL / MC1) -> 1 canonical.
    assert cons.private_peel_blocks == 3, cons.peel_group_names
    assert cons.shared_peel_blocks == 1
    # the private carry-rounds are applied more times than they are stored.
    assert cons.private_peel_refs > cons.private_peel_blocks
    # canonical-COMBINE (route-free) saves real distinct blocks + nnz.
    assert cons.unique_blocks_saved_combine == 2
    assert cons.nnz_saved_combine > 0
    # naive (with route glue) breaks even here (honest): net <= combine.
    assert cons.unique_blocks_saved_naive <= cons.unique_blocks_saved_combine


# ---------------------------------------------------------------------------
# Gating: default OFF leaves the ALU dim unchanged (no canonical band).
# ---------------------------------------------------------------------------
def test_default_off_does_not_allocate_peel_band():
    saved = os.environ.pop("C4_SHARED_PEEL", None)
    try:
        assert not P.shared_peel_enabled()
        L = NibbleVMLayout(code_size=48)
        m.extend_layout_for_alu32(L, recurrent_divmod=True,
                                  mul_lookahead=False, kb_batched=False)
        d_before = L.D
        # a private-peel build never calls extend_layout_for_peel, so L.D is the
        # ALU dim with NO canonical band appended.
        assert getattr(L, "PEEL", None) is None
        assert L.D == d_before
    finally:
        if saved is not None:
            os.environ["C4_SHARED_PEEL"] = saved


def test_shared_peel_flag_reader():
    for on in ("1", "true", "on"):
        assert P.shared_peel_enabled({"C4_SHARED_PEEL": on})
    for off in ("", "0", None):
        env = {} if off is None else {"C4_SHARED_PEEL": off}
        assert not P.shared_peel_enabled(env)


# ---------------------------------------------------------------------------
# Registry wiring.
# ---------------------------------------------------------------------------
def test_register_peel_units_adds_peel_family():
    from c4_min import alu_units as AU
    P.register_peel_units()
    units = AU.units_for("peel")
    assert set(units) == {"nibble15", "nibble32"}
    for u in units.values():
        assert u.op == "peel"
        assert u.module == "alu_peel"
        assert u.depth == 1 and u.stored_blocks == 1


def test_register_peel_units_wired_follows_flag():
    from c4_min import alu_units as AU
    # OFF -> shared peel not wired (ops use private peels).
    saved = os.environ.pop("C4_SHARED_PEEL", None)
    try:
        P.register_peel_units()
        assert AU.units_for("peel")["nibble15"].wired is False
        os.environ["C4_SHARED_PEEL"] = "1"
        P.register_peel_units()
        assert AU.units_for("peel")["nibble15"].wired is True
    finally:
        os.environ.pop("C4_SHARED_PEEL", None)
        if saved is not None:
            os.environ["C4_SHARED_PEEL"] = saved
