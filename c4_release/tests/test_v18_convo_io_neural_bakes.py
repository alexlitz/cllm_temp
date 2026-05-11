"""V18 Phase 1 bake tests: step resumption (3a) + PC/SP latch (3b).

Locks in the surface contract for the two new bakes that close the gap in
``docs/V18_CONVO_IO_NEURAL_PLAN.md §3``:

- 3a (``convo_io_step_resume``): L3 FFN unit 1035 that fires on
  ``LAST_WAS_THINKING_START`` and emits ``NEXT_PC = 1`` so the head decodes
  ``Token.REG_PC`` for the next token, autonomously starting a new VM step
  (replacing ``_inject_synthetic_step`` after THINKING_START).
- 3b (``convo_io_pc_sp_latch``): L6 FFN units 1402-1465 that, also on
  ``LAST_WAS_THINKING_START``, drive ``OUTPUT_LO/HI`` from the staged PC
  and SP nibble cache so the resumed step's REG_PC and REG_SP value bytes
  carry the advanced PC and popped SP.

Both ops are double-gated (``enable_conversational_io`` AND ``enable``).
For Phase 1 landing both default to ``enable=False`` so production builds
are byte-identical to the prior compile. These tests cover:

1. Both ops are registered in ``all_core_ops()`` (dep-graph stability).
2. Both default to no-op when ``enable=False``.
3. With ``enable=True`` + ``enable_conversational_io=True`` the helpers
   write the documented FFN unit ranges and the unit slots that other
   convo-IO bakes own (1034 in L3; 1400-1401 in L6; 1500-1532 V9 PUTCHAR
   in L6) remain untouched.
4. ``enable=True`` paired with ``enable_conversational_io=False`` is
   still a no-op (the outer gate dominates), so accidentally flipping
   just the inner switch doesn't change behavior.

These are static / direct-call checks against the bake helpers — no
model is built and no bytecode is run. The end-to-end run-the-model
check belongs in a future Phase 1b PR after the latch *capture* side
(at the PRTF AX marker) lands.
"""

import torch

from neural_vm.setup_helpers import (
    _set_convo_io_pc_sp_latch,
    _set_convo_io_step_resume,
)
from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
from neural_vm.unified_compiler.ops.flag_gated_ops import (
    make_convo_io_pc_sp_latch_op,
    make_convo_io_step_resume_op,
)
from neural_vm.vm_step import _SetDim as BD


class _StubFFN:
    """Minimal FFN stub exposing the W_up/W_gate/W_down/b_up/b_gate
    tensors the convo-IO bake helpers write into.

    Shape mirrors the production L3/L6 FFN: hidden_dim=4096,
    d_model=512 (the bake helpers only write a handful of rows / cols
    so the exact d_model doesn't matter here — any value >= the max
    BD dim the helpers touch will do).
    """

    def __init__(self, hidden_dim: int = 4096, d_model: int = 512):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.W_down = torch.zeros(d_model, hidden_dim)
        self.b_up = torch.zeros(hidden_dim)
        self.b_gate = torch.zeros(hidden_dim)


# Scaling factor matching the convention used by other convo-IO helpers
# (S = 10.0 in production; the absolute value doesn't matter for these
# tests since we only check which slots are non-zero).
S = 10.0


class TestV18Phase1BakesRegistered:
    """The two Phase 1 ops are registered in ``all_core_ops()``."""

    def test_step_resume_op_registered(self):
        names = {op.name for op in all_core_ops()}
        assert "convo_io_step_resume" in names, (
            "convo_io_step_resume missing from all_core_ops(). See "
            "V18_CONVO_IO_NEURAL_PLAN.md §3a."
        )

    def test_pc_sp_latch_op_registered(self):
        names = {op.name for op in all_core_ops()}
        assert "convo_io_pc_sp_latch" in names, (
            "convo_io_pc_sp_latch missing from all_core_ops(). See "
            "V18_CONVO_IO_NEURAL_PLAN.md §3b."
        )

    def test_phase1_ops_registered_with_convo_io_enabled_too(self):
        """Same registration check with ``enable_conversational_io=True``
        — the ops are always present, the gate only controls the bake_fn
        body (not the registration)."""
        names = {
            op.name for op in all_core_ops(enable_conversational_io=True)
        }
        assert "convo_io_step_resume" in names
        assert "convo_io_pc_sp_latch" in names


class TestV18Phase1BakesGatedOff:
    """With ``enable=False`` (the Phase 1 default) the bake bodies are
    no-ops regardless of ``enable_conversational_io``.

    Verifies the safety contract: the Phase 1 ops are registered but do
    not change any FFN weights, so production builds remain byte-identical
    to the prior compile.
    """

    def test_step_resume_noop_when_enable_false(self):
        # enable_conversational_io=True, enable=False → still no-op.
        op = make_convo_io_step_resume_op(
            enable_conversational_io=True, enable=False
        )
        ffn = _StubFFN()
        op.bake_fn(_Block(ffn), _dim_positions(), S)
        assert torch.all(ffn.W_up == 0)
        assert torch.all(ffn.W_down == 0)

    def test_pc_sp_latch_noop_when_enable_false(self):
        op = make_convo_io_pc_sp_latch_op(
            enable_conversational_io=True, enable=False
        )
        ffn = _StubFFN()
        op.bake_fn(_Block(ffn), _dim_positions(), S)
        assert torch.all(ffn.W_up == 0)
        assert torch.all(ffn.W_down == 0)

    def test_step_resume_noop_when_convo_io_false(self):
        # Outer gate dominates: enable=True is meaningless without convo_io.
        op = make_convo_io_step_resume_op(
            enable_conversational_io=False, enable=True
        )
        ffn = _StubFFN()
        op.bake_fn(_Block(ffn), _dim_positions(), S)
        assert torch.all(ffn.W_up == 0)
        assert torch.all(ffn.W_down == 0)

    def test_pc_sp_latch_noop_when_convo_io_false(self):
        op = make_convo_io_pc_sp_latch_op(
            enable_conversational_io=False, enable=True
        )
        ffn = _StubFFN()
        op.bake_fn(_Block(ffn), _dim_positions(), S)
        assert torch.all(ffn.W_up == 0)
        assert torch.all(ffn.W_down == 0)


class TestV18Phase1StepResumeBake:
    """Direct-call checks on ``_set_convo_io_step_resume``.

    Verifies the documented contract (V18 plan §3a):
    - L3 FFN unit 1035 is the only unit written.
    - Unit 1034 (the existing state-init unit) is untouched.
    - Up-projection reads LAST_WAS_THINKING_START.
    - Down-projection writes NEXT_PC (+), IO_STATE (-), IO_IN_OUTPUT_MODE (-).
    """

    def test_writes_only_unit_1035(self):
        ffn = _StubFFN()
        _set_convo_io_step_resume(ffn, S, BD)

        # Unit 1035 has non-zero up-projection on LAST_WAS_THINKING_START.
        assert ffn.W_up[1035, BD.LAST_WAS_THINKING_START] != 0
        # Unit 1034 is untouched (state-init owns it).
        assert torch.all(ffn.W_up[1034] == 0)
        # No spillover above unit 1035.
        assert torch.all(ffn.W_up[1036:1040] == 0)

    def test_emits_next_pc_and_clears_io_state(self):
        ffn = _StubFFN()
        _set_convo_io_step_resume(ffn, S, BD)

        # NEXT_PC positive → head decodes Token.REG_PC after THINKING_START.
        assert ffn.W_down[BD.NEXT_PC, 1035] > 0
        # IO_STATE and IO_IN_OUTPUT_MODE both negative (cleared).
        assert ffn.W_down[BD.IO_STATE, 1035] < 0
        assert ffn.W_down[BD.IO_IN_OUTPUT_MODE, 1035] < 0


class TestV18Phase1PCSPLatchBake:
    """Direct-call checks on ``_set_convo_io_pc_sp_latch``.

    Verifies the documented contract (V18 plan §3b):
    - Writes L6 FFN units 1402-1465 (64 units).
    - Units 1400-1401 (state-machine) untouched.
    - Units 1500-1532 (V9 PUTCHAR routing) untouched.
    - All units gate on LAST_WAS_THINKING_START.
    - Down-projection drives OUTPUT_LO/HI nibbles.
    """

    def test_writes_64_units_in_correct_range(self):
        ffn = _StubFFN()
        _set_convo_io_pc_sp_latch(ffn, S, BD)

        # Each of units 1402..1465 has a non-zero up-projection on
        # LAST_WAS_THINKING_START.
        for u in range(1402, 1466):
            assert (
                ffn.W_up[u, BD.LAST_WAS_THINKING_START] != 0
            ), f"unit {u} should gate on LAST_WAS_THINKING_START"

    def test_does_not_overlap_state_machine_units(self):
        ffn = _StubFFN()
        _set_convo_io_pc_sp_latch(ffn, S, BD)

        # 1400-1401 belong to _set_conversational_io_state_machine — must
        # remain zero after this latch bake runs in isolation.
        for u in (1400, 1401):
            assert torch.all(ffn.W_up[u] == 0)
            assert torch.all(ffn.W_down[:, u] == 0)

    def test_does_not_overlap_v9_putchar_units(self):
        """V9 PUTCHAR (parallel branch) uses L6 FFN units 1500-1532.
        V18 latch occupies 1402-1465 — must leave V9's range clean.
        """
        ffn = _StubFFN()
        _set_convo_io_pc_sp_latch(ffn, S, BD)

        for u in range(1500, 1533):
            assert torch.all(ffn.W_up[u] == 0)
            assert torch.all(ffn.W_down[:, u] == 0)

    def test_drives_output_lo_hi_nibbles(self):
        """Down-projection of the latch units writes into OUTPUT_LO/HI
        (the head decodes value bytes from these nibbles)."""
        ffn = _StubFFN()
        _set_convo_io_pc_sp_latch(ffn, S, BD)

        # At least one OUTPUT_LO column and one OUTPUT_HI column should
        # have non-zero coefficients across the latch unit band.
        latch_band = ffn.W_down[:, 1402:1466]
        out_lo_block = latch_band[BD.OUTPUT_LO:BD.OUTPUT_LO + 16]
        out_hi_block = latch_band[BD.OUTPUT_HI:BD.OUTPUT_HI + 16]
        assert (out_lo_block != 0).any()
        assert (out_hi_block != 0).any()


class TestV18Phase1BakesEnabled:
    """End-to-end gate flip: with both ``enable_conversational_io=True``
    and ``enable=True``, the factory-built ops write the documented units
    when invoked against a stub FFN.

    Does NOT build the full model — that's the job of a future Phase 1b
    PR after the latch *capture* side is in place. This just confirms the
    factory→bake_fn→helper plumbing is wired correctly.
    """

    def test_step_resume_factory_bake_writes_unit_1035(self):
        op = make_convo_io_step_resume_op(
            enable_conversational_io=True, enable=True
        )
        ffn = _StubFFN()
        op.bake_fn(_Block(ffn), _dim_positions(), S)
        # Unit 1035 should fire.
        assert ffn.W_up[1035, BD.LAST_WAS_THINKING_START] != 0
        assert ffn.W_down[BD.NEXT_PC, 1035] > 0

    def test_pc_sp_latch_factory_bake_writes_band(self):
        op = make_convo_io_pc_sp_latch_op(
            enable_conversational_io=True, enable=True
        )
        ffn = _StubFFN()
        op.bake_fn(_Block(ffn), _dim_positions(), S)
        # Span check: each unit in the band is wired.
        for u in (1402, 1417, 1418, 1433, 1434, 1465):
            assert ffn.W_up[u, BD.LAST_WAS_THINKING_START] != 0


# ----- helpers ---------------------------------------------------------------


class _Block:
    """Minimal block stub that exposes ``ffn`` as the helpers expect."""

    def __init__(self, ffn):
        self.ffn = ffn


def _dim_positions():
    """Build a dim_positions dict mapping every BD attribute name to its
    int value, so the proxy returned by ``_as_setdim_proxy`` resolves to
    the same positions as direct BD attribute access.

    This lets the factory-built bake_fn (which calls
    ``_as_setdim_proxy(dim_positions)``) end up writing to the same FFN
    slots as the direct-call tests above.
    """
    out = {}
    for name in dir(BD):
        if name.startswith("_"):
            continue
        val = getattr(BD, name)
        if isinstance(val, int):
            out[name] = val
    return out
