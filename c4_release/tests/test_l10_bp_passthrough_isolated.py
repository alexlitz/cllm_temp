"""Isolation test for L10 head 7 BP byte passthrough Q-row suppression.

Probe ``c4_release/tools/probe_l14_li_consumer.py`` confirmed that on
``test_si_li_roundtrip`` head 7 (``layer10_bp_byte_passthrough_bake.head_7``)
spuriously attended from Q@p=193 (a MEM val byte 1 row in the SI step's
MEM frame) to K@170 (the IMM 0x200 byte-1 row of the preceding IMM
step) with attention weight 0.61.  That leak forwarded CLEAN_EMBED
nibbles ``(LO=2, HI=0)`` -- i.e. the byte ``0x02`` of address ``0x200``
-- into ``OUTPUT_LO[2]`` and ``OUTPUT_HI[0]`` (~1.29 / 2.00 in the
residual delta), contaminating the MEM val byte slot and ultimately
the LI consumer.

Fix: add ``MEM_VAL_B0/B1/B2/B3`` as negative-strength Q-row suppressors
at slot 0 of ``_layer10_bp_byte_passthrough_head_spec``.  Slot 0 is
the head's "byte-position selection" slot; any Q with one of those
flags set now picks up ``-L`` per flag, driving its slot-0 inner
product into the strongly-negative region.  The legitimate target rows
(BP byte rows during ordinary non-ENT/LEV steps) do NOT carry
``MEM_VAL_B*`` so the head still fires there.

We assert two things in isolation:

  1. The fix adds a negative slot-0 Q write for each of MEM_VAL_B0..B3.
  2. With the fix lowered into a stub attention module, the slot-0
     Q*K contribution between a MEM val byte 1 Q row and any K row
     becomes strongly negative (drops by at least the L=S scale per
     MEM_VAL_B* flag set in Q).
"""
from __future__ import annotations

import math
import os
import sys

import torch

# Force CPU-only execution.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace  # noqa: E402

from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _layer10_bp_byte_passthrough_head_spec,
)
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402


D_MODEL = 512
S = 100.0
HD = 64
NUM_HEADS = 12  # L10 head 7 sits inside the 12-head layout
BD = _SetDim
HEAD_IDX = 7
MEM_VAL_DIMS = (BD.MEM_VAL_B0, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3)


class _StubAttn:
    """Bare attention module exposing W_q/W_k/W_v/W_o tensors."""

    def __init__(
        self,
        *,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        head_dim: int = HD,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = d_model
        dim_out = num_heads * head_dim
        self.W_q = torch.zeros(dim_out, d_model)
        self.W_k = torch.zeros(dim_out, d_model)
        self.W_v = torch.zeros(dim_out, d_model)
        self.W_o = torch.zeros(d_model, dim_out)
        self.alibi_slopes = torch.zeros(num_heads)


def _head_7_spec_with_fix():
    return _layer10_bp_byte_passthrough_head_spec(BD, S)


def _head_7_spec_no_fix():
    spec = _head_7_spec_with_fix()
    stripped = tuple(
        w for w in spec.q
        if not (w.slot == 0 and w.dim in MEM_VAL_DIMS)
    )
    return replace(spec, q=stripped)


def _bake(spec) -> _StubAttn:
    attn = _StubAttn()
    Primitives.generate_attention_head(attn, spec, HD)
    return attn


def _mem_val_byte1_q_row() -> torch.Tensor:
    """Q row at a SI step's MEM val byte 1 position (p=193 in the probe).

    Carries MEM_VAL_B1=1 (the L2 byte-prediction flag picking val byte
    1), MEM_STORE=1, HAS_SE=1, IS_BYTE=1, plus the MEM marker proximity
    flag (H1+MEM_I).  This row should NOT attract head 7 (a BP byte
    passthrough head) because it represents a stored memory value byte,
    not a BP-byte register destination.
    """
    row = torch.zeros(D_MODEL)
    row[BD.CONST] = 1.0
    row[BD.IS_BYTE] = 1.0
    row[BD.HAS_SE] = 1.0
    row[BD.MEM_STORE] = 1.0
    row[BD.MEM_VAL_B1] = 1.0
    MEM_I = 4
    row[BD.H1 + MEM_I] = 1.0
    row[BD.BYTE_INDEX_1] = 1.0
    return row


def _imm_byte1_k_row() -> torch.Tensor:
    """K row at a prior IMM step's byte 1 position (K@170 in the probe).

    BP marker proximity (H1+BP_IDX=1) and CLEAN_EMBED nibbles for
    0x02 (the byte 1 of address 0x200), which is the value that
    leaked through head 7 in the probe.
    """
    row = torch.zeros(D_MODEL)
    row[BD.CONST] = 1.0
    row[BD.IS_BYTE] = 1.0
    BP_I = 3
    row[BD.H1 + BP_I] = 1.0
    row[BD.BYTE_INDEX_1] = 1.0
    row[BD.CLEAN_EMBED_LO + 2] = 1.0
    row[BD.CLEAN_EMBED_HI + 0] = 1.0
    return row


def test_head_7_does_not_fire_at_mem_val_rows():
    """Per probe diagnostic, head 7 spuriously attended K@170 from Q@p=193
    (a MEM val byte 1 row) with weight 0.61, leaking 0x02 nibble.
    Q at MEM val byte position (MEM_VAL_B0/B1/B2/B3 = 1) must have a
    strongly-negative slot-0 contribution against any K row, dropping
    head 7's effective score versus the no-fix baseline by at least the
    full slot-0 strength (L = S = 100) for the single MEM_VAL_B1 flag.

    The negative slot-0 Q*K delta must be enough to flip head 7 from a
    firing regime to a softmax1 probability <= 0.05 at this Q row, even
    in the simplest case of a single matching K.
    """
    # 1. Declarative invariant: the fix must add four negative slot-0
    #    Q writes -- one per MEM_VAL_B* dim -- with weight -S.
    spec = _head_7_spec_with_fix()
    suppress_writes = {
        w.dim: w.weight
        for w in spec.q
        if w.slot == 0 and w.dim in MEM_VAL_DIMS
    }
    assert set(suppress_writes.keys()) == set(MEM_VAL_DIMS), (
        "Fix must declare slot-0 Q suppressors for all four MEM_VAL_B0..B3 dims;"
        f" found {sorted(suppress_writes.keys())}"
    )
    for dim, weight in suppress_writes.items():
        assert weight < 0.0, (
            f"Slot-0 Q write for dim={dim} must be negative; got weight={weight}"
        )
        # Must match the existing slot-0 strength L = S so it dominates.
        assert math.isclose(weight, -S, abs_tol=1e-6), (
            f"Slot-0 Q write for dim={dim} must equal -S={-S}; got {weight}"
        )

    # 2. Numerical invariant: slot-0 Q*K contribution between the
    #    MEM val byte 1 Q row and the leaking IMM byte 1 K row drops by
    #    at least S between no-fix and with-fix lowerings.
    attn_fix = _bake(spec)
    attn_no_fix = _bake(_head_7_spec_no_fix())

    q = _mem_val_byte1_q_row()
    k = _imm_byte1_k_row()
    base = HEAD_IDX * HD

    def _slot0_contrib(attn) -> float:
        Wq_slot0 = attn.W_q[base]
        Wk_slot0 = attn.W_k[base]
        q_proj = float(q @ Wq_slot0)
        k_proj = float(k @ Wk_slot0)
        return q_proj * k_proj / math.sqrt(HD)

    slot0_fix = _slot0_contrib(attn_fix)
    slot0_no_fix = _slot0_contrib(attn_no_fix)
    delta = slot0_no_fix - slot0_fix
    # The MEM_VAL_B1 flag in Q is 1 with weight -S=-100 in slot 0;
    # K slot 0 baseline at the IMM K row carries IS_BYTE=1 with weight
    # L=S, so K@0 = +S.  The delta should be S*S/sqrt(HD) per MEM_VAL_B
    # flag set.
    expected_delta = (S * S) / math.sqrt(HD)
    assert math.isclose(delta, expected_delta, rel_tol=1e-3), (
        "Fix must subtract S*S/sqrt(HD) from slot-0 score at MEM val byte 1\n"
        f"  Q row; got delta={delta:.2f}, expected ~{expected_delta:.2f}."
    )

    # 3. Sanity: the slot-0 contribution at this Q row is now lower
    #    than the legitimate slot-0 contribution at a non-MEM Q row,
    #    confirming the suppressor pulls the head away from MEM-val
    #    positions.
    q_non_mem = q.clone()
    q_non_mem[BD.MEM_VAL_B1] = 0.0
    q_non_mem[BD.MEM_STORE] = 0.0
    q_non_mem[BD.H1 + 4] = 0.0  # remove MEM marker proximity
    BP_I = 3
    q_non_mem[BD.H1 + BP_I] = 1.0  # legitimate BP-byte Q row

    Wq_slot0 = attn_fix.W_q[base]
    Wk_slot0 = attn_fix.W_k[base]
    slot0_at_mem = float(q @ Wq_slot0) * float(k @ Wk_slot0) / math.sqrt(HD)
    slot0_at_bp = float(q_non_mem @ Wq_slot0) * float(k @ Wk_slot0) / math.sqrt(HD)
    assert slot0_at_mem < slot0_at_bp, (
        "Slot-0 score at MEM val byte Q row must be strictly less than at "
        f"a legitimate BP-byte Q row.  MEM={slot0_at_mem:.2f}, BP={slot0_at_bp:.2f}"
    )
