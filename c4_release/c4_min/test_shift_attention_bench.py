"""Gate for the SHIFT-AS-POSITIONAL-ATTENTION bench (``shift_attention_bench``).

Confirms the bench's headline claims WITHOUT a DIM-8192 forward — the gadgets run on
their small residual planes / token positions (seconds):

* all three variants (nibble mux-tree baseline, shift-as-attention, leaner-mux) are
  byte-exact vs the 32-bit reference for SHL and SHR, and fp32-exact;
* the coarse-CAM head does a genuine positional gather under a REAL softmax forward
  (RoPE slow lanes + BOS sink + in-range gate), including out-of-range = shifted-in
  zero;
* shift-as-attention beats the mux-tree, but only MODESTLY (~12%): the head is cheap
  (~19 nz) yet the coarse mux it replaces was only ~7% of the shifter, so the win is
  bounded — the honest verdict;
* the leaner-mux is byte-IDENTICAL to the baseline and barely smaller (coarse mux was
  never the expensive part).

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_shift_attention_bench.py -v
"""
from __future__ import annotations

import math
import random

import pytest
import torch
import torch.nn.functional as F

from c4_min import isa
from c4_min import shift_attention_bench as B
from c4_min import shifter_bakeoff as sb
from c4_min.qwen_full_vm import QWEN2_5_ARCH


def _byte_exact(run, left) -> None:
    rng = random.Random(2024)
    cases = [(x, n) for x in B._EDGE_XS for n in B._EDGE_NS]
    cases += [(rng.randint(0, B.MASK32), rng.randint(0, 0x3F)) for _ in range(300)]
    for x, n in cases:
        assert run(x, n) == B.ref_shift32(x, n, left), (hex(x), n)


@pytest.mark.parametrize("left", [True, False])
def test_baseline_byte_exact(left):
    blocks, L = B.build_baseline(left)
    _byte_exact(lambda x, n: sb.run_blocks(blocks, L, x, n), left)
    assert B._max_relu_arg(blocks) < B.FP32_INT_LIMIT


@pytest.mark.parametrize("left", [True, False])
def test_shift_attention_byte_exact(left):
    """The COARSE-CAM-gather + FINE-FFN hybrid is byte-exact vs the 32-bit ref."""
    fine, fL, _nz = B.build_fine_ffn(left)
    _byte_exact(lambda x, n: B.run_shift_attention(fine, fL, x, n, left), left)
    assert B._max_relu_arg(fine) < B.FP32_INT_LIMIT


@pytest.mark.parametrize("left", [True, False])
def test_leaner_mux_identical_to_baseline(left):
    """Leaner-mux is byte-IDENTICAL to the baseline (same result), byte-exact, and
    strictly (if only slightly) smaller — the coarse mux is not where the weight is."""
    base, bL = B.build_baseline(left)
    lean, lL = B.build_leaner_mux(left)
    rng = random.Random(5)
    for _ in range(300):
        x, n = rng.randint(0, B.MASK32), rng.randint(0, 0x3F)
        assert sb.run_blocks(base, bL, x, n) == sb.run_blocks(lean, lL, x, n)
    assert sb._blocks_nz(lean) < sb._blocks_nz(base)


@pytest.mark.parametrize("left", [True, False])
def test_cam_head_softmax_gather(left):
    """The coarse-CAM head genuinely gathers positionally under a REAL softmax
    forward (per-bit position agreement on slow lanes + BOS sink + in-range gate),
    for EVERY (pop, shift) case incl. out-of-range = shifted-in zero."""
    (q_w, k_w, v_w, o_w), L, _nz = B.bake_coarse_cam_head(left)
    hd = QWEN2_5_ARCH.head_dim

    def gather(pop, c):
        src = [(pop >> (4 * k)) & 0xF for k in range(B.N_NIB)]

        def krow(p=None, nib=0):
            x = torch.zeros(L.D); x[L.ONE] = 1.0
            if p is not None:
                for b in range(3):
                    x[L.POS_BIN + b] = float((p >> b) & 1)
                x[L.TOK_NIB] = float(nib); x[L.IS_TOK] = 1.0
            return x
        rows = [krow()] + [krow(p, src[p]) for p in range(B.N_NIB)]
        X = torch.stack(rows); K = F.linear(X, k_w); Vv = F.linear(X, v_w)
        out = []
        for jout in range(B.N_NIB):
            tgt = (jout - c) if left else (jout + c)
            valid = 0 <= tgt < B.N_NIB
            qx = torch.zeros(L.D); qx[L.ONE] = 1.0; qx[L.IS_QRY] = 1.0
            if valid:
                qx[L.QVALID] = 1.0
                for b in range(3):
                    qx[L.QPOS_BIN + b] = float((tgt >> b) & 1)
            Q = F.linear(qx.unsqueeze(0), q_w)
            w = F.softmax((Q @ K.T).squeeze(0) / math.sqrt(hd), dim=0)
            out.append(int(round(float((w @ Vv)[0]))))
        return out

    for pop in (0xDEADBEEF, 0xFFFFFFFF, 0x12345678, 0x80000000, 0x0, 0xF0F0F0F0):
        for c in range(8):
            assert gather(pop, c) == B._coarse_via_cam(pop, c, left), (hex(pop), c)


def test_attention_beats_muxtree_modestly():
    """Shift-as-attention (head + address FFN + fine FFN, all counted) beats the
    mux-tree — but by a MODEST margin (the head is cheap yet the coarse path it
    optimises is small); the fine peel is a shared hard floor in both."""
    rows = B.measure()
    for op in ("SHL", "SHR"):
        att = next(r for r in rows if r["name"] == "shift-as-attention"
                   and r["op"] == op)
        base = next(r for r in rows if r["name"] == "nibble mux-tree"
                    and r["op"] == op)
        # a real win...
        assert att["total_nz"] < base["total_nz"]
        # ...but modest (between ~5% and ~25%), NOT an order of magnitude.
        ratio = att["total_nz"] / base["total_nz"]
        assert 0.75 < ratio < 0.96, (op, ratio)
        # the head itself is cheap (< the coarse mux it replaces).
        assert att["attn_nz"] < 40
        # all-exact.
        assert att["exact"] == att["total"]


def test_table_builds():
    rows = B.measure()
    table = B.format_table(rows)
    note = B.format_phase_note(rows)
    assert "shift-as-attention" in table and "nibble mux-tree" in table
    assert "coarse mux" in note or "coarse_mux" in note
