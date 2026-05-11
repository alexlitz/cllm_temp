"""Verify OP_LEV decodes neurally at the PC marker without Python injection.

The L5 FFN bake (``_set_opcode_decode_ffn`` in ``vm_step.py``) includes an
all-step PC-marker OP_LEV decode (1 FFN unit gated on
``MARK_PC + OPCODE_BYTE_LO[8] + OPCODE_BYTE_HI[0]``). The L6 head 6 V[0]
OP_LEV relay propagates OP_LEV from AX → other markers. Together they
replace the Python-side ``_inject_active_opcode`` OP_LEV write for
pure_neural runs.

The test runs ``[LEV, EXIT]`` so LEV lands at instruction index 0. The
first-step PC marker has OPCODE_BYTE_LO/HI populated by L5 attention
head 2 (first-step fetch). It captures L5/L6 residual stream activations
via forward hooks and asserts OP_LEV >= 1 at the first PC marker.
"""

import pytest

from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _scan_pc_markers(token_ids):
    return [
        pos for pos in range(token_ids.shape[1])
        if token_ids[0, pos].item() == Token.REG_PC
    ]


class TestNeuralOpLevDecode:
    """OP_LEV must fire at the LEV step's PC marker without Python injection."""

    def test_op_lev_fires_at_pc_marker_without_injection(self, pure_neural_runner):
        """LEV at instruction index 0: assert OP_LEV >= 1 at the LEV step's
        PC marker in both L5 and L6 residual outputs.

        Baseline before the L5 all-step PC-marker OP_LEV decode landed: OP_LEV
        was only written at the AX marker by the main opcode decode (MARK_AX
        gated), staying zero at all PC marker positions, so pure_neural runs
        had to rely on ``_inject_active_opcode`` to write OP_LEV globally.
        With the new decode, OP_LEV >= 1 at the PC marker autonomously.
        """
        runner = pure_neural_runner
        if not getattr(runner, "pure_neural", False):
            pytest.skip("Requires pure_neural=True runner")

        # Precondition: no Python peek at the bytecode.
        assert runner.model._active_opcode is None, (
            "Test precondition: pure_neural runner must have "
            "_active_opcode=None (no Python peek)"
        )

        # LEV at idx 0, EXIT at idx 1. The first-step PC marker has
        # OPCODE_BYTE_LO/HI populated by L5 attention head 2 (first-step fetch).
        prog = [
            Opcode.LEV,
            Opcode.EXIT,
        ]
        bc = _make_bc(prog)

        # Collect captures from EVERY forward pass so we don't depend on which
        # generate_next iteration's tensors were retained.
        captures = []

        def _embed_hook(module, inputs, output):
            captures.append({"token_ids": inputs[0].detach().clone()})

        def _block_hook(name):
            def fn(module, inputs, output):
                if captures:
                    captures[-1][name] = output.detach().clone()
            return fn

        h_embed = runner.model.embed.register_forward_hook(_embed_hook)
        h5 = runner.model.blocks[5].register_forward_hook(_block_hook("after_L5"))
        h6 = runner.model.blocks[6].register_forward_hook(_block_hook("after_L6"))

        try:
            runner._memory = {}
            runner._mem_history = {}
            runner._mem_access_order = []
            try:
                runner.run(bc, b"", max_steps=3)
            except Exception:
                # Runner may fail mid-LEV; hooks captured residuals already.
                pass
        finally:
            h_embed.remove()
            h5.remove()
            h6.remove()

        assert len(captures) >= 1, "Expected at least one forward pass"

        # Use the compiler-allocated OP_LEV position (not legacy _SetDim.OP_LEV).
        # With pin_io_only=True, OP_LEV is repinned to a different position from
        # the legacy _SetDim value.
        op_lev_dim = runner.model.embed._dim_positions["OP_LEV"]

        # Scan all forward passes for one where the FIRST PC marker carries
        # OP_LEV (the LEV step). We expect this on every forward — the first
        # PC marker stays in the token stream throughout generation.
        best_l5 = 0.0
        best_l6 = 0.0
        for c in captures:
            if "after_L5" not in c or "after_L6" not in c:
                continue
            pc_positions = _scan_pc_markers(c["token_ids"])
            if not pc_positions:
                continue
            first_pc = pc_positions[0]
            l5 = c["after_L5"][0, first_pc, op_lev_dim].item()
            l6 = c["after_L6"][0, first_pc, op_lev_dim].item()
            if l5 > best_l5:
                best_l5 = l5
            if l6 > best_l6:
                best_l6 = l6

        # The L5 all-step PC-marker OP_LEV decode writes OP_LEV ≈ 5 (W_down
        # = 10/S, silu(S*0.5) ≈ S/2 → 10/S * S/2 = 5). Require >= 1.0
        # (well above the zero baseline before the decode existed).
        assert best_l5 >= 1.0, (
            f"After L5: OP_LEV not present at any forward pass's first PC "
            f"marker. best_l5={best_l5:.3f} across {len(captures)} forwards. "
            f"Expected ≥ 1.0 from the all-step PC-marker decode in "
            f"_set_opcode_decode_ffn (vm_step.py:3235-3252)."
        )

        # L6 head 6 V[0] relay propagates OP_LEV across markers (V=0.1 × W_o=10
        # = ×1.0 multiplier, summed with the AX-marker value). After L6 the PC
        # marker residual carries OP_LEV ≈ 5.
        assert best_l6 >= 1.0, (
            f"After L6: OP_LEV not present at any forward pass's first PC "
            f"marker. best_l6={best_l6:.3f} across {len(captures)} forwards. "
            f"Expected ≥ 1.0 from L6 head 6 V[0] relay (vm_step.py:7060)."
        )
