"""Per-op audit harness for Layer 5 (instruction fetch + opcode decode).

Layer 5 owns the fetch-and-decode pipeline that every downstream layer
depends on:

* ``layer5_fetch`` -- 6-of-8 attention heads (heads 0..5) implementing
  the instruction-fetch pattern at the AX / PC markers. Reads CODE-byte
  ADDR_KEY + CLEAN_EMBED_LO/HI; writes OPCODE_BYTE_LO/HI and the
  FETCH_LO/HI staging slots used by the immediate-decode chain. Ships
  with a populated 192-cell claim map (6 heads x 16 dims x 2 bands).
* ``opcode_decode_ffn`` -- FFN that converts OPCODE_BYTE_LO/HI into the
  one-hot OP_* flag set (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_JMP,
  OP_JSR, OP_BZ, OP_BNZ, OP_ENT, OP_ADJ, OP_LEV, OP_LI, OP_LC, OP_SI,
  OP_SC, OP_PSH, OP_OR, OP_XOR, OP_AND, OP_EQ, OP_NE, OP_LT, OP_GT,
  OP_LE, OP_GE, OP_SHL, OP_SHR, OP_EXIT, OP_NOP, OP_PUTCHAR,
  OP_GETCHAR, OP_LEA, OP_IMM = 34 opcodes per the layer-map). Ships
  with empty ``claims`` today (per-FFN-unit claim map not authored;
  ``writes`` set covers the broad strokes only) so the verifier skips it.
  Tracked below as a known-absent op so a future per-cell claim map is
  loud.
* ``_layer5_fetch_dep_anchor`` -- no-op companion of ``layer5_fetch``
  that reserves a dep-graph slot. Empty claims -> absent.
* ``_opcode_decode_ffn_dep_anchor`` -- no-op companion of
  ``opcode_decode_ffn``. Empty claims -> absent.
* ``layer5_user_input_gather`` -- V9 GETCHAR staging op gated by
  ``enable=False`` in the default build (see
  ``docs/V9_GETCHAR_READ_NEURAL_PLAN.md``). Empty claims -> absent.

A representative symbolic forward check is included for the
``opcode_decode_ffn`` opcode-table contract: feeding OPCODE_BYTE_LO/HI
encoded for OP_ADD must produce a positive OP_ADD residual write at
the AX marker. This guards against a structural drift in the opcode
table that the per-op claim audit cannot catch (since ``opcode_decode_ffn``
ships with empty per-cell claims).

See ``test_l0_marker_transitions.py`` for the shared fixture rationale.
"""

import pytest

from ._per_op_audit import assert_no_drift, assert_op_absent, assert_op_fires


# Ops with a populated ``claims`` map -- drift-checked + fires-checked.
L5_OPS_WITH_CLAIMS = (
    "layer5_fetch",
)

# Ops that intentionally ship with empty ``claims`` in the default
# build. ``opcode_decode_ffn`` declares its semantic ``writes`` set but
# has no per-cell claim map yet (the 34-opcode decode lowers via
# ``FFNRule.gated_write`` rules at runtime), so the verifier's claim
# scan skips it. The two ``_dep_anchor`` ops are no-op companions that
# reserve a dep-graph slot. ``layer5_user_input_gather`` is gated off
# by default per ``docs/V9_GETCHAR_READ_NEURAL_PLAN.md``. All four
# should be absent from the default-build verifier report; appearing
# means somebody added claims or flipped an ``enable=`` gate and this
# list needs to migrate the op into the drift-checked list.
L5_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "opcode_decode_ffn",
    "_layer5_fetch_dep_anchor",
    "_opcode_decode_ffn_dep_anchor",
    "layer5_user_input_gather",
)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L5_OPS_WITH_CLAIMS)
def test_l5_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L5", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L5_OPS_WITH_CLAIMS)
def test_l5_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    assert_op_fires(static_claims_report, "L5", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L5_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD)
def test_l5_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L5", op_name)


# ---------------------------------------------------------------------------
# Symbolic forward checks for opcode_decode_ffn
# ---------------------------------------------------------------------------
#
# ``opcode_decode_ffn`` ships with empty per-cell ``claims`` so the
# verifier cannot catch a structural drift in the opcode table (e.g., a
# transposed OPCODE_BYTE_LO/HI mapping for OP_ADD). The symbolic checks
# below complement the absent-from-report gate: they construct the L5
# FFN directly via the production factory, then run one-token forwards
# with the (OPCODE_BYTE_LO, OPCODE_BYTE_HI) encodings from
# ``_opcode_decode_main_rules`` under MARK_AX gating, and assert the
# named OP_* residual lights up. Two distinct (lo, hi) pairs are
# exercised so a transposed-axis regression is caught regardless of
# which axis flipped.

@pytest.fixture(scope="module")
def l5_opcode_decode_ffn():
    """Build the L5 FFN with ``opcode_decode_ffn.bake_fn`` applied.

    Module-scoped so the two symbolic forward tests share one build
    (~100ms vs ~25s for the session-scoped ``static_claims_report``
    fixture, which the drift/fires tests above use).
    """
    import torch

    from neural_vm.unified_compiler.decl_verifier import _build_layout_only
    from neural_vm.unified_compiler.ops.l5_ops import make_opcode_decode_ffn_op
    from neural_vm.vm_step import AutoregressiveVM

    layout = _build_layout_only(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        n_heads=8,
    )
    model = AutoregressiveVM(
        d_model=layout.d_model,
        n_layers=layout.n_layers,
        n_heads=8,
        dim_positions=layout.dim_positions,
    )
    op = make_opcode_decode_ffn_op()
    block = model.blocks[5]
    with torch.no_grad():
        op.bake_fn(block, layout.dim_positions, 100.0)
    return block.ffn, layout.dim_positions, layout.d_model


@pytest.mark.lowering
@pytest.mark.parametrize(
    "op_name, lo, hi",
    [
        # See ``_opcode_decode_main_rules`` in ``l5_ops.py`` for the table.
        ("OP_ADD", 9, 1),
        ("OP_EXIT", 6, 2),
    ],
)
def test_l5_opcode_decode_symbolic_forward(l5_opcode_decode_ffn, op_name, lo, hi):
    """Feed ``op_name`` encoding through the L5 FFN under MARK_AX gating;
    the named OP_* residual must light up positive.
    """
    import torch

    ffn, dim_positions, d_model = l5_opcode_decode_ffn
    x = torch.zeros(1, 1, d_model)
    x[0, 0, dim_positions["OPCODE_BYTE_LO"] + lo] = 1.0
    x[0, 0, dim_positions["OPCODE_BYTE_HI"] + hi] = 1.0
    x[0, 0, dim_positions["MARK_AX"]] = 1.0
    # CONST = 1.0 is the model-wide convention (see ``compile_full_vm_dynamic``);
    # the gated FFN rules read it implicitly via threshold subtraction.
    if "CONST" in dim_positions:
        x[0, 0, dim_positions["CONST"]] = 1.0
    with torch.no_grad():
        y = ffn(x)
    delta = y[0, 0, dim_positions[op_name]].item()
    assert delta > 0.0, (
        f"L5 opcode_decode_ffn did not light up {op_name} for "
        f"(OPCODE_BYTE_LO={lo}, OPCODE_BYTE_HI={hi}, MARK_AX=1); "
        f"{op_name} delta = {delta:.4f}. Either the opcode table in "
        f"_opcode_decode_main_rules drifted or MARK_AX gating broke."
    )
