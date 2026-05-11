"""V18 conversational-I/O THINK-protocol parity test (Phase 0).

Locks in the surface contract that Phase 1's neural replacement of the
V18 handler at ``run_vm.py:534-583`` must reproduce. Specifically:

- The handler is present at the documented location.
- All 10 ``enable_conversational_io`` compiler ops are registered in
  ``all_core_ops()`` (no-ops when the flag is False).
- ``Token.THINKING_START/END`` and the BD dim names used by the bakes
  are stable.
- ``_inject_synthetic_step`` keeps its signature, and the handler
  invokes it after emitting ``THINKING_START``.

These are static / inventory checks — no model is built, no bytecode
is run. Phase 1 adds the end-to-end run once the missing bakes (3a +
3b in ``docs/V18_CONVO_IO_NEURAL_PLAN.md §3``) land.

Coordinates with V9 PUTCHAR (parallel branch). V9 PUTCHAR uses the
AX-marker → OUTPUT routing path at L6 FFN units 1500-1532; V18 uses
the THINKING_END/THINKING_START tag protocol triggered at NEXT_SE
via L6 attn heads 4-5 + L6 FFN units 1400-1401. The two paths share
no FFN units and no attention heads.
"""

import inspect


class TestV18HandlerParity:
    """Lock in the V18 handler-mode invariants that the eventual neural-
    side replacement must reproduce.

    The handler at ``run_vm.py:534-583`` is the production path today;
    these tests assert the surface contract (handler location, registered
    bake op names, reserved token IDs, BD dim names) so a future Phase 1
    bake has a stable target to reproduce.
    """

    def test_v18_handler_block_present_at_known_lineno(self):
        """The V18 handler block lives at run_vm.py:534-583. If it
        moves, update PURITY_AUDIT_2026_05_11.md §V18 and
        V18_CONVO_IO_NEURAL_PLAN.md §1 — both reference these
        line numbers."""
        from neural_vm import run_vm

        src = inspect.getsource(run_vm)
        assert (
            "if self.conversational_io and next_token == Token.THINKING_END:"
            in src
        ), (
            "V18 handler guard string not found in run_vm.py. The "
            "convo-IO THINKING_END handler may have moved or been "
            "deleted. Update PURITY_AUDIT_2026_05_11.md and "
            "V18_CONVO_IO_NEURAL_PLAN.md."
        )

    def test_convo_io_neural_bakes_registered(self):
        """The convo-IO neural pipeline (L2/L3/L5/L6/L7/L8/L9/L10/L15)
        has all 10 compiler ops registered. They're no-ops by default
        (``enable_conversational_io=False``) but should be present in
        ``all_core_ops()`` so the dep-graph stays stable.

        This is the inventory check that paves the way for Phase 1:
        when the gate flips to ``True``, these ops bake the neural
        pipeline described in V18_CONVO_IO_NEURAL_PLAN.md §2.
        """
        from neural_vm.unified_compiler.ops.all_core_ops import (
            all_core_ops,
        )
        ops = all_core_ops()
        names = {op.name for op in ops}

        expected = {
            # L2: lookback (sets LAST_WAS_THINKING_END / _START / _BYTE)
            "layer2_lookback_detection_head",
            # L3: state init (LAST_WAS_THINKING_END → IO_IN_OUTPUT_MODE)
            "layer3_convo_io_state_init",
            # L5: opcode decode (PRTF/READ → IO_IS_PRTF/IO_IS_READ)
            "convo_io_opcode_decode",
            # L6: relay heads + state machine
            "convo_io_relay_heads",
            "convo_io_state_machine",
            # L7: format pointer extraction (STACK0 → FORMAT_PTR)
            "format_pointer_extraction",
            # L8: FORMAT_POS counter increment
            "format_position_counter",
            # L9: format string fetch head (PTR+POS → OUTPUT_BYTE)
            "format_string_fetch_head",
            # L10: null terminator (OUTPUT_BYTE=0 → NEXT_THINKING_START)
            "null_terminator_detection",
            # L15: OUTPUT_BYTE → OUTPUT routing for emit
            "conversational_io_output_routing",
        }
        missing = expected - names
        assert not missing, (
            f"Missing convo-IO neural compiler ops: {sorted(missing)}. "
            "See V18_CONVO_IO_NEURAL_PLAN.md §2 for the canonical list."
        )

    def test_convo_io_thinking_tokens_reserved(self):
        """The token vocabulary reserves THINKING_START (272) and
        THINKING_END (273) plus IO_STATE_* markers. Locking these
        token IDs is a prerequisite for the neural emit path: the
        lm_head's ``NEXT_THINKING_START → 272`` routing depends on
        these IDs."""
        from neural_vm.vm_step import Token

        assert Token.THINKING_START == 272
        assert Token.THINKING_END == 273
        assert Token.IO_STATE_EMIT_BYTE == 274
        assert Token.IO_STATE_EMIT_THINKING == 275

    def test_convo_io_dim_layout_stable(self):
        """The residual-stream dimensions used by the convo-IO bakes
        (IO_IN_OUTPUT_MODE, FORMAT_PTR_LO/HI, MARK_THINKING_*,
        NEXT_THINKING_*, etc.) must be present in the BD layout.
        These are read by the L2/L3/L7/L9 attention/FFN bakes and
        the lm_head routing.

        Locking the dim names (not their numerical values, which can
        drift if the BD registry is renumbered) lets Phase 1's bakes
        rely on the names being stable.
        """
        from neural_vm.vm_step import _SetDim as BD

        for dim_name in (
            "IO_IN_OUTPUT_MODE",
            "IO_OUTPUT_COMPLETE",
            "FORMAT_PTR_LO",
            "FORMAT_PTR_HI",
            "IO_FORMAT_POS",
            "OUTPUT_BYTE_LO",
            "OUTPUT_BYTE_HI",
            "MARK_THINKING_START",
            "MARK_THINKING_END",
            "LAST_WAS_THINKING_START",
            "LAST_WAS_THINKING_END",
            "LAST_WAS_BYTE",
            "NEXT_THINKING_START",
            "NEXT_THINKING_END",
            "IO_IS_PRTF",
            "IO_IS_READ",
            "IO_STATE",
            "ACTIVE_OPCODE_PRTF",
            "ACTIVE_OPCODE_READ",
        ):
            assert hasattr(BD, dim_name), (
                f"BD.{dim_name} missing — convo-IO bakes will fail. "
                "Update V18_CONVO_IO_NEURAL_PLAN.md if the dim "
                "registry is being intentionally renamed."
            )


class TestV18ParityHarnessSelfCheck:
    """Sanity check that the parity test infrastructure itself works,
    independent of the neural path. These tests run regardless of
    whether the neural pipeline emits THINKING_END.

    Verifies:
    - The V18 handler module is importable.
    - The handler's helper methods (``_inject_synthetic_step``,
      ``_extract_register``) are present at known signatures.
    """

    def test_synthetic_step_injector_signature(self):
        """``_inject_synthetic_step(context, pc, ax, sp, bp, stack0=0)``
        is the post-THINKING_END resume mechanism today. Its signature
        is the contract that Phase 1's neural bake must reproduce:
        the new step must contain REG_PC + 4 bytes + REG_AX + 4 bytes
        + REG_SP + 4 bytes + REG_BP + 4 bytes + STACK0 + 4 bytes +
        MEM + 9 bytes + STEP_END = 35 tokens.
        """
        from neural_vm.run_vm import AutoregressiveVMRunner

        sig = inspect.signature(
            AutoregressiveVMRunner._inject_synthetic_step
        )
        params = list(sig.parameters.keys())
        assert params == ["self", "context", "pc", "ax", "sp", "bp", "stack0"]
        assert sig.parameters["stack0"].default == 0

    def test_handler_invokes_synthetic_step(self):
        """Static check: the V18 handler block in run_vm.py invokes
        ``self._inject_synthetic_step(...)`` after appending
        ``Token.THINKING_START`` to the context. If this pattern
        changes, ``V18_CONVO_IO_NEURAL_PLAN.md §1`` needs updating.
        """
        from neural_vm.run_vm import AutoregressiveVMRunner

        src = inspect.getsource(AutoregressiveVMRunner.run)
        assert "context.append(Token.THINKING_START)" in src
        assert "self._inject_synthetic_step(" in src
