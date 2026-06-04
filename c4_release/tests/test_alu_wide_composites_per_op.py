"""Per-op audit harness for the 4 wide-ALU composites at L10/L11/L12/L13.

Inventory of "wide-ALU composites" runtime-installed via post_op attach
ops in lookup mode (the default ``compile_full_vm_dynamic`` build):

  - L10.post_ops[0] = ``ALUAndOrXor``       (bitwise AND/OR/XOR)
  - L10.post_ops[-1] = ``FlattenedDivMod``  (DIV/MOD long-division pipeline)
  - L11.post_ops[0] = ``FlattenedALUMul``   (MUL 9-stage pipeline, instance A)
  - L12.post_ops[0] = ``FlattenedALUMul``   (MUL 9-stage pipeline, instance B)
  - L13.post_ops[0] = ``ALUShiftComposite`` (SHL/SHR 4-stage pipeline)

Per B6-L Section 3a, none of these have dedicated per-op test harnesses
today — regressions only surface as smoke failures. This file delivers
the per-op contract testing for all 4 composites:

1. **Drift checks via ``static_claims_report``**: the attach ops
   (``l10_alu_postop_attach``, ``l11_alu_postop_attach``,
   ``l12_alu_postop_attach``, ``l13_alu_postop_attach``,
   ``l10_alu_divmod_{bdtoge,longdiv,getobd,install}``) all ship with
   empty ``claims`` today; ``assert_op_absent`` watchdogs them. If any
   author adds claims (e.g. per-stage byte-row pins) this gate forces
   migration into the drift-checked list.

2. **Fires-during-bake** (no test needed for absent ops — covered by the
   drift check upgrade path).

3. **Symbolic forward**: each composite instantiated standalone via its
   constructor (``ALUAndOrXor(S, BD)``, ``FlattenedALUMul.build_fully_baked(
   S, BD)``, ``ALUShiftComposite(S, BD)``, ``FlattenedDivMod`` via
   sequential install_* calls). One representative operation is exercised
   per composite and the OUTPUT byte assertion pins the byte-identical
   semantics with the legacy ``PureNeuralALU`` wrappers.

Mirrors the ``test_l11_mul_partials.py`` / ``test_l12_mul_combine.py``
pattern (sibling pre-MUL test harnesses): instantiate a fresh module +
forward a synthetic AX-marker residual + assert the OUTPUT slice.
The MUL tests here are deliberately COMPLEMENTARY to those files —
they exercise the END-TO-END composite forward
(``FlattenedALUMul.build_fully_baked``) rather than the L11/L12
lookup-mode partial+combine helpers.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.vm_step import _SetDim  # noqa: E402

from ._per_op_audit import assert_op_absent  # noqa: E402


# ---------------------------------------------------------------------------
# Inventory.
# ---------------------------------------------------------------------------

# The 7 wide-ALU composite "owner" ops (attach + DIV/MOD install). None
# carry per-cell claims today, so they're absent from static_claims_report.
WIDE_ALU_COMPOSITE_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "l10_alu_postop_attach",      # ALUAndOrXor installer
    "l10_alu_divmod_bdtoge",      # FlattenedDivMod stage 0
    "l10_alu_divmod_longdiv",     # FlattenedDivMod stage 1
    "l10_alu_divmod_getobd",      # FlattenedDivMod stage 2
    "l10_alu_divmod_install",     # FlattenedDivMod post_op install
    "l11_alu_postop_attach",      # FlattenedALUMul installer (L11)
    "l12_alu_postop_attach",      # FlattenedALUMul installer (L12)
    "l13_alu_postop_attach",      # ALUShiftComposite installer
)


# ---------------------------------------------------------------------------
# Section 1: Drift watchdogs (absent → migrate when claims are added).
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize(
    "op_name", WIDE_ALU_COMPOSITE_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD
)
def test_wide_alu_composite_owner_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    """The composite owner ops have no per-cell claims today.

    If any author adds claims (e.g. byte-row pins for the schoolbook
    output slot, or for the DIV pipeline's RESULT writeback), this test
    fires loudly so the migration into a drift-checked list happens at
    test time rather than during the next 1096 backtrace.
    """
    assert_op_absent(static_claims_report, "L10/L11/L12/L13", op_name)


# ---------------------------------------------------------------------------
# Section 2: L10 bitwise (AND/OR/XOR) symbolic forward.
# ---------------------------------------------------------------------------
#
# Post-V8 (2026-06-04): the legacy ``ALUAndOrXor`` composite (a
# ``PureNeuralALU(operations='bitwise')`` subclass) was deleted; the
# production lookup-mode install is now a rule-derived ``PureFFN`` baked
# from ``wide_alu_dsl.bitwise_rules`` via the factory
# ``ops/alu_ops.py:make_lookup_mode_l10_bitwise_rules_op``. The tests
# below exercise that factory's bake_fn directly (mock block, real
# PureFFN install) and assert the decoded OUTPUT byte matches Python.


@pytest.fixture(scope="module")
def andorxor_composite():
    """Rule-derived ``PureFFN`` install (replacing the legacy ALUAndOrXor).

    Replays the production install path's bake function on a mock block
    and returns the ``PureFFN`` inserted into ``block.post_ops[0]``.
    """
    import torch.nn as nn

    from neural_vm.base_layers import PureFFN
    from neural_vm.unified_compiler.ops.alu_ops import (
        make_lookup_mode_l10_bitwise_rules_op,
    )
    from neural_vm.unified_compiler.ops.shared import _setdim_to_positions

    class _MockBlock:
        def __init__(self):
            self.ffn = PureFFN(dim=512, hidden_dim=64)
            self.post_ops = nn.ModuleList()

    block = _MockBlock()
    op = make_lookup_mode_l10_bitwise_rules_op()
    op.bake_fn(block, _setdim_to_positions(_SetDim), 100.0)
    return block.post_ops[0]


def _make_alu_byte_input(*, a: int, b: int, op_dim: int) -> torch.Tensor:
    """Build a one-position residual at MARK_AX exposing both bytes.

    Sets ALU_LO/HI to the operand-A nibbles, AX_CARRY_LO/HI to the
    operand-B nibbles, MARK_AX + the selected opcode hot, plus a CONST
    baseline that matches the rest of the production-residual norm.
    """
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, op_dim] = 1.0
    x[0, 0, _SetDim.ALU_LO + (a & 0xF)] = 1.0
    x[0, 0, _SetDim.ALU_HI + ((a >> 4) & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (b & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_HI + ((b >> 4) & 0xF)] = 1.0
    return x


def _decode_output_byte(y: torch.Tensor) -> int:
    lo = int(y[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16].argmax().item())
    hi = int(y[0, 0, _SetDim.OUTPUT_HI:_SetDim.OUTPUT_HI + 16].argmax().item())
    return lo | (hi << 4)


@pytest.mark.parametrize(
    ("op_dim", "op_name", "py_op"),
    [
        (_SetDim.OP_AND, "AND", lambda a, b: a & b),
        (_SetDim.OP_OR, "OR", lambda a, b: a | b),
        (_SetDim.OP_XOR, "XOR", lambda a, b: a ^ b),
    ],
)
@pytest.mark.parametrize(
    ("a", "b"),
    [
        (0x00, 0xFF),  # one operand all-zero, one all-ones
        (0xA5, 0x5A),  # alternating-bit pattern
        (0xCC, 0x33),  # nibble-aligned pattern
    ],
)
def test_alu_andorxor_byte_identical_to_python(
    andorxor_composite, op_dim, op_name, py_op, a, b
):
    """Rule-derived L10 bitwise install: OUTPUT byte matches Python.

    Each (a, b) pair exercises a different nibble pattern so a regression
    in either the lo or hi nibble path surfaces. Forward through the
    rule-lowered ``PureFFN`` and decode
    ``(OUTPUT_LO argmax) | (OUTPUT_HI argmax << 4)``; compare to the
    spec result.
    """
    x = _make_alu_byte_input(a=a, b=b, op_dim=op_dim)
    with torch.no_grad():
        y = andorxor_composite(x)

    got = _decode_output_byte(y)
    expected = py_op(a, b) & 0xFF
    assert got == expected, (
        f"L10 bitwise rule install {op_name}: a=0x{a:02X} {op_name} "
        f"b=0x{b:02X} expected 0x{expected:02X}, got 0x{got:02X}"
    )


def test_alu_andorxor_no_fire_without_mark_ax(andorxor_composite):
    """Without MARK_AX, the rule-derived install must not overwrite OUTPUT.

    The ``bitwise_rules`` 3-way AND gates OUTPUT writes on ``MARK_AX``.
    Sentinel for the most common silent-corruption mode where the gate
    drops and the bitwise op leaks at every position.
    """
    x = _make_alu_byte_input(a=0xAA, b=0x55, op_dim=_SetDim.OP_XOR)
    x[0, 0, _SetDim.MARK_AX] = 0.0
    # Initialize OUTPUT to a known sentinel so we can see a write.
    x[0, 0, _SetDim.OUTPUT_LO + 7] = 9.0
    x[0, 0, _SetDim.OUTPUT_HI + 11] = 11.0
    with torch.no_grad():
        y = andorxor_composite(x)
    # The MARK_AX gate should suppress the rule fires; OUTPUT should
    # therefore be passed through unchanged.
    assert torch.allclose(
        y[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16],
        x[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16],
        atol=1e-3,
    ), "L10 bitwise rule install wrote OUTPUT_LO without MARK_AX"


# ---------------------------------------------------------------------------
# Section 3: FlattenedALUMul symbolic forward (end-to-end MUL composite).
# ---------------------------------------------------------------------------
#
# FlattenedALUMul packages the 9-stage MUL pipeline used by L11.post_ops[0]
# and L12.post_ops[0]. ``build_fully_baked`` is the canonical drop-in
# replacement constructor used by the post-op attach ops. This test pins
# its forward against Python's % 256 reference at a representative product.
#
# COMPLEMENTARY to test_l11_mul_partials.py (which tests _set_layer11_mul_partial,
# the lookup-mode partial staging at TEMP[partial]) and test_l12_mul_combine.py
# (which tests _set_layer12_mul_combine at OUTPUT_HI). Those test files cover
# the LOOKUP-mode L11/L12 baking path; this file covers the WIDE-ALU
# composite — the actual module that ends up in post_ops in default mode.


@pytest.fixture(scope="module")
def flattened_mul_composite():
    from neural_vm.efficient_alu_neural import FlattenedALUMul

    return FlattenedALUMul.build_fully_baked(S=100.0, BD=_SetDim)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (3, 7),       # tiny, no overflow
        (15, 17),     # crosses nibble boundary
        (200, 200),   # overflow byte (40000 % 256 = 64)
        (255, 1),     # identity edge
    ],
)
def test_flattened_mul_byte_identical_to_python(
    flattened_mul_composite, a, b
):
    """``FlattenedALUMul`` writes OUTPUT byte == (a * b) % 256.

    This is the END-TO-END composite forward — through BDToGE, 7 mul
    sub-FFNs (schoolbook + 3 carry passes + genprop + binary lookahead +
    final correction), combine, GEToBD. Output is the low byte of the
    product (256-bit modulo); high bytes flow via the L12 tail-MUL
    propagation chain not exercised here.
    """
    x = _make_alu_byte_input(a=a, b=b, op_dim=_SetDim.OP_MUL)
    with torch.no_grad():
        y = flattened_mul_composite(x)

    got = _decode_output_byte(y)
    expected = (a * b) & 0xFF
    assert got == expected, (
        f"FlattenedALUMul: 0x{a:02X} * 0x{b:02X} expected low byte "
        f"0x{expected:02X}, got 0x{got:02X}"
    )


def test_flattened_mul_early_outs_when_no_op_mul(flattened_mul_composite):
    """Perf optimization sentinel (efficient_alu_neural.py:1281).

    Without OP_MUL > 0.1 anywhere in the batch, ``forward`` must short-
    circuit and return the input tensor unchanged. The early-out
    contributes a ~370 ms perf win per inactive call (the pipeline runs
    a 9-stage chain otherwise). Sentinel: if anyone removes the
    short-circuit, this test still passes but the perf regresses
    silently; if anyone breaks the masking such that OUTPUT writes
    without OP_MUL, this test fires.
    """
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, _SetDim.ALU_LO + 5] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + 7] = 1.0
    # Stage a sentinel value in OUTPUT to detect any write.
    x[0, 0, _SetDim.OUTPUT_LO + 3] = 9.0
    with torch.no_grad():
        y = flattened_mul_composite(x)
    assert torch.equal(y, x), (
        "FlattenedALUMul fired without OP_MUL — early-out at "
        "efficient_alu_neural.py:1281 has regressed"
    )


# ---------------------------------------------------------------------------
# Section 4: ALUShiftComposite symbolic forward.
# ---------------------------------------------------------------------------
#
# ALUShiftComposite is the 4-stage SHL/SHR composite at L13.post_ops[0]:
# bdtoge → precompute → select → getobd. Forward applies the shift
# selected by OP_SHL/OP_SHR with shift amount taken from AX_CARRY_LO.


@pytest.fixture(scope="module")
def shift_composite():
    from neural_vm.efficient_alu_neural import ALUShiftComposite

    return ALUShiftComposite(S=100.0, BD=_SetDim)


@pytest.mark.parametrize(
    ("a", "shift", "op_dim", "op_name", "py_op"),
    [
        (0x01, 4, _SetDim.OP_SHL, "SHL", lambda a, s: (a << s) & 0xFF),
        (0x0F, 4, _SetDim.OP_SHL, "SHL", lambda a, s: (a << s) & 0xFF),
        (0xF0, 4, _SetDim.OP_SHR, "SHR", lambda a, s: (a >> s) & 0xFF),
        (0x80, 3, _SetDim.OP_SHR, "SHR", lambda a, s: (a >> s) & 0xFF),
    ],
)
def test_alu_shift_composite_byte_identical_to_python(
    shift_composite, a, shift, op_dim, op_name, py_op
):
    """ALUShiftComposite OUTPUT byte == (a << shift) & 0xFF or (a >> shift).

    Shift amount is sourced from AX_CARRY_LO (lo nibble of the shift
    operand). Tests pin the byte-identical forward against PureNeuralALU
    via Python's reference shift.
    """
    x = _make_alu_byte_input(a=a, b=shift, op_dim=op_dim)
    with torch.no_grad():
        y = shift_composite(x)

    got = _decode_output_byte(y)
    expected = py_op(a, shift)
    assert got == expected, (
        f"ALUShiftComposite {op_name}: 0x{a:02X} {op_name} {shift} "
        f"expected 0x{expected:02X}, got 0x{got:02X}"
    )


# ---------------------------------------------------------------------------
# Section 5: FlattenedDivMod symbolic forward.
# ---------------------------------------------------------------------------
#
# FlattenedDivMod packages the 4-stage long-division composite at
# L10.post_ops[-1]: bdtoge → div_pipeline → mod_pipeline → getobd.
# Forward applies DIV (op_div=1) or MOD (op_mod=1) with operands from
# ALU_LO/HI (dividend) and AX_CARRY_LO/HI (divisor).


@pytest.fixture(scope="module")
def divmod_composite():
    from neural_vm.efficient_alu_divmod_split import FlattenedDivMod

    composite = FlattenedDivMod(S=100.0, BD=_SetDim)
    # Install all 4 stages so forward() is callable. The compiler does
    # this via 3 stage ops + 1 install op; here we call the installers
    # directly for the test (mirrors ``FlattenedALUMul.build_fully_baked``).
    composite.install_bdtoge()
    composite.install_longdiv()
    composite.install_getobd()
    return composite


@pytest.mark.parametrize(
    ("a", "b", "op_dim", "op_name", "py_op"),
    [
        (15, 4, _SetDim.OP_DIV, "DIV", lambda a, b: a // b),
        (15, 4, _SetDim.OP_MOD, "MOD", lambda a, b: a % b),
        (100, 7, _SetDim.OP_DIV, "DIV", lambda a, b: a // b),
        (100, 7, _SetDim.OP_MOD, "MOD", lambda a, b: a % b),
    ],
)
def test_flattened_divmod_byte_identical_to_python(
    divmod_composite, a, b, op_dim, op_name, py_op
):
    """FlattenedDivMod OUTPUT byte == a // b or a % b.

    Pin the byte-identical forward against PureNeuralALU(operations=
    'div_mod') / EfficientDivMod_Neural via Python's reference op.
    """
    x = _make_alu_byte_input(a=a, b=b, op_dim=op_dim)
    with torch.no_grad():
        y = divmod_composite(x)

    got = _decode_output_byte(y)
    expected = py_op(a, b)
    assert got == expected, (
        f"FlattenedDivMod {op_name}: 0x{a:02X} {op_name} 0x{b:02X} "
        f"expected 0x{expected:02X}, got 0x{got:02X}"
    )


def test_flattened_divmod_early_outs_when_no_op_div_or_mod(divmod_composite):
    """Perf sentinel (efficient_alu_divmod_split.py:464).

    Without OP_DIV or OP_MOD > 0.1, ``forward`` must return ``x_bd``
    unchanged (the long-division pipeline is ~370 ms otherwise).
    """
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, _SetDim.ALU_LO + 5] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + 7] = 1.0
    x[0, 0, _SetDim.OUTPUT_LO + 3] = 9.0
    with torch.no_grad():
        y = divmod_composite(x)
    assert torch.equal(y, x), (
        "FlattenedDivMod fired without OP_DIV/OP_MOD — early-out at "
        "efficient_alu_divmod_split.py:464 has regressed"
    )


# ---------------------------------------------------------------------------
# Section 6: Stage inventory pins.
# ---------------------------------------------------------------------------
#
# These tests pin the structural shape of each composite. Drift in the
# stage count or stage ordering would silently swap a stage out and we
# would see only a smoke-level failure.


def test_flattened_mul_has_10_stages_total(flattened_mul_composite):
    """FlattenedALUMul.build_fully_baked yields 10 pipeline stages.

    Stage breakdown (per docstring at efficient_alu_neural.py:1022):
      1 BDToGE + 1 schoolbook + 3 carry passes + 1 genprop +
      1 binary lookahead + 1 final correction + 1 combine + 1 GEToBD
      = 10 stages total. The 9 compiler installer ops register 8 of
      those stages explicitly; the 9th (`_MulCombineStage`) is appended
      implicitly when ``install_getobd`` runs (efficient_alu_neural.py:
      1199-1214), and the GEToBD stage follows. ``_stages`` and
      ``pipeline`` end up with the same 10 entries.
    """
    assert len(flattened_mul_composite._stages) == 10
    assert flattened_mul_composite.pipeline is not None
    assert len(list(flattened_mul_composite.pipeline)) == 10


def test_alu_shift_composite_has_4_stages(shift_composite):
    """ALUShiftComposite has 4 sub-stage attributes: bdtoge, precompute,
    select, getobd.
    """
    for stage_name in (
        "bdtoge_stage",
        "precompute_stage",
        "select_stage",
        "getobd_stage",
    ):
        assert hasattr(shift_composite, stage_name), (
            f"ALUShiftComposite missing stage {stage_name!r}; pipeline "
            f"layout drift?"
        )


def test_flattened_divmod_has_4_stages(divmod_composite):
    """FlattenedDivMod has 4 named stages after install: bdtoge, div, mod,
    getobd (the longdiv installer registers both div + mod).
    """
    expected = {"bdtoge", "div", "mod", "getobd"}
    assert set(divmod_composite.stages.keys()) == expected, (
        f"FlattenedDivMod stages={set(divmod_composite.stages.keys())}, "
        f"expected {expected}"
    )
    assert divmod_composite.__dict__.get("pipeline") is not None, (
        "FlattenedDivMod pipeline was not assembled after install_*"
    )
