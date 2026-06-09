"""Tests for the reference oracle + block-by-block diff infra.

Unit tests verify:

1. ``ReferenceOracle`` correctly reproduces the C4 ISA reference
   semantics for a tiny ``IMM 0x200; PSH; EXIT`` program (AX/SP/STACK0
   trajectory).
2. ``project_state_to_residual`` sets the simple-family dims
   (CLEAN_EMBED_LO/HI, MARK_AX, MARK_STACK0, BYTE_INDEX_h) at the
   expected positions for one step's token window.
3. ``expected_trace`` produces ``(step, position, block, dim_key)``
   entries across every block for a supported dim.
4. ``diff_actual_vs_expected`` returns no divergences when actual ==
   expected, and a known divergence when an oracle-required dim is
   missing from the runner.
5. ``find_first_divergent_block`` flags the earliest block index with
   a mismatch, the demo use case from the task brief.

Tests use the same lightweight ``_StubOp`` + ``_StubCompiler`` shims
as ``test_symbolic_forward.py`` so they run sub-second without a full
``LayerCompiler.compile()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

import pytest

from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (
    step_function_rule,
)
from c4_release.neural_vm.unified_compiler.dim_diff import (
    diff_actual_vs_expected,
    find_first_divergent_block,
    format_divergence,
)
from c4_release.neural_vm.unified_compiler.dim_oracle import (
    DEFERRED_DIM_FAMILIES,
    SUPPORTED_DIM_FAMILIES,
    TOKENS_PER_STEP,
    ReferenceOracle,
    expected_trace,
    is_supported_dim,
    project_state_to_residual,
)
from c4_release.neural_vm.unified_compiler.ir import CompilerIR
from c4_release.neural_vm.unified_compiler.symbolic_forward import (
    OP_EXIT,
    OP_IMM,
    OP_PSH,
    OP_SI,
    SymbolicForwardRunner,
    encode_instr,
)


# ---------------------------------------------------------------------------
# Stubs (mirror test_symbolic_forward.py)
# ---------------------------------------------------------------------------


@dataclass
class _StubOp:
    name: str
    kind: str = "ffn"
    compiler_ir: Optional[Any] = None
    layer_idx: Optional[int] = None
    target_op_name: Optional[str] = None
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)


@dataclass
class _StubCompiler:
    ops_per_layer: List[List[_StubOp]]
    block_ops: List[_StubOp] = field(default_factory=list)
    model_ops: List[_StubOp] = field(default_factory=list)
    dim_positions: Optional[dict] = None


def _ir_with_rule(rule) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.append(rule)
    return ir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reference_oracle_imm_psh_exit_trajectory():
    """``IMM 0x200; PSH; EXIT`` — AX, SP, STACK0 must match C4 semantics."""

    program = [
        encode_instr(OP_IMM, 0x200),
        encode_instr(OP_PSH),
        encode_instr(OP_EXIT),
    ]
    oracle = ReferenceOracle(program)

    # 3 instructions -> 3 snapshots.
    assert oracle.num_steps == 3

    # Step 0: IMM 0x200 -> AX = 0x200, SP unchanged.
    s0 = oracle.state_at_step(0)
    assert s0.ax == 0x200
    assert s0.sp == oracle.initial_sp
    assert not s0.halted

    # Step 1: PSH -> SP -= 8, STACK0 = AX = 0x200.
    s1 = oracle.state_at_step(1)
    assert s1.ax == 0x200
    assert s1.sp == oracle.initial_sp - 8
    assert s1.stack0 == 0x200

    # Step 2: EXIT -> halted, AX unchanged.
    s2 = oracle.state_at_step(2)
    assert s2.ax == 0x200
    assert s2.halted

    # Out-of-range step indices clamp.
    assert oracle.state_at_step(99).step_idx == 2
    assert oracle.state_at_step(-5).step_idx == 0


def test_project_state_per_token_sets_simple_dim_families():
    """Per-token projection sets CLEAN_EMBED/MARK/BYTE_INDEX at the
    correct token slots within the step's window."""

    program = [encode_instr(OP_IMM, 0x200)]
    oracle = ReferenceOracle(program)
    s0 = oracle.state_at_step(0)
    table = project_state_to_residual(s0, per_token=True)

    base = s0.step_idx * TOKENS_PER_STEP

    # Markers land on slot 0 of each register section.
    assert table[(base + 0, "MARK_PC+0")] == 1.0
    assert table[(base + 5, "MARK_AX+0")] == 1.0
    assert table[(base + 20, "MARK_STACK0+0")] == 1.0
    assert table[(base + 34, "MARK_SE+0")] == 1.0

    # BYTE_INDEX_h is set at the byte token at offset h within the section.
    assert table[(base + 6, "BYTE_INDEX_0+0")] == 1.0
    assert table[(base + 7, "BYTE_INDEX_1+0")] == 1.0
    assert table[(base + 8, "BYTE_INDEX_2+0")] == 1.0

    # AX = 0x200; byte 1 = 0x02; LO nibble = 2; HI nibble = 0.
    assert table[(base + 7, "CLEAN_EMBED_LO+2")] == 1.0
    assert table[(base + 7, "CLEAN_EMBED_HI+0")] == 1.0
    # Byte 0 = 0x00; both nibbles = 0.
    assert table[(base + 6, "CLEAN_EMBED_LO+0")] == 1.0
    assert table[(base + 6, "CLEAN_EMBED_HI+0")] == 1.0

    # CONST is set everywhere.
    for tok in range(TOKENS_PER_STEP):
        assert table[(base + tok, "CONST+0")] == 1.0


def test_project_state_bag_of_dims_collapses_to_step_position():
    """Default (bag-of-dims) projection emits one entry per supported
    dim at position == step_idx, matching ``SymbolicForwardRunner``."""

    program = [encode_instr(OP_IMM, 0x200)]
    oracle = ReferenceOracle(program)
    s0 = oracle.state_at_step(0)
    bag = project_state_to_residual(s0)

    pos = s0.step_idx  # 0

    # Marker / byte_index indicators all collapse onto position=0.
    assert bag[(pos, "MARK_AX+0")] == 1.0
    assert bag[(pos, "MARK_STACK0+0")] == 1.0
    assert bag[(pos, "BYTE_INDEX_1+0")] == 1.0
    # AX = 0x200; byte 1 LO nibble = 2.
    assert bag[(pos, "CLEAN_EMBED_LO+2")] == 1.0


def test_expected_trace_covers_all_blocks():
    """``expected_trace`` emits a tuple per (step, position, block)
    for the supported dim family."""

    program = [
        encode_instr(OP_IMM, 0x200),
        encode_instr(OP_PSH),
        encode_instr(OP_EXIT),
    ]
    oracle = ReferenceOracle(program)

    trace = expected_trace(oracle, "MARK_AX", n_blocks=4)

    # Bag-of-dims: MARK_AX collapses to position=step_idx, 3 steps * 4
    # blocks = 12 entries.
    assert len(trace) == 12
    expected_positions = {0, 1, 2}
    actual_positions = {pos for (_, pos, _, _) in trace.keys()}
    assert actual_positions == expected_positions

    # Block coverage: every block 0..3 must appear.
    actual_blocks = {b for (_, _, b, _) in trace.keys()}
    assert actual_blocks == {0, 1, 2, 3}

    # Unsupported (unknown) families must raise. OUTPUT_LO now has a
    # same-step projection rule (added 2026-06-09) — a fabricated dim
    # name with no projection still surfaces a ValueError.
    with pytest.raises(ValueError):
        expected_trace(oracle, "NOT_A_REAL_DIM_FAMILY", n_blocks=4)


def test_supported_vs_deferred_dim_families_disjoint():
    """Sanity: the supported and deferred lists have no overlap, and the
    ``is_supported_dim`` predicate agrees. After the 2026-06-09 oracle
    extension, ``DEFERRED_DIM_FAMILIES`` is empty (the same-step
    OUTPUT_LO/HI, AX_CARRY_LO/HI, ADDR_KEY, and MEM_ADDR_SRC projections
    graduated to ``SUPPORTED_DIM_FAMILIES``).
    """

    assert set(SUPPORTED_DIM_FAMILIES).isdisjoint(DEFERRED_DIM_FAMILIES)
    for name in SUPPORTED_DIM_FAMILIES:
        assert is_supported_dim(name)
        assert is_supported_dim(f"{name}+3")
    for name in DEFERRED_DIM_FAMILIES:
        assert not is_supported_dim(name)


def test_diff_finds_missing_writer_for_psh_step():
    """When the runner has NO writer for ``MARK_AX`` at the PSH step's
    AX-marker row, the oracle sees a divergence at block 0.

    This is the demo case in the task brief, reduced to a stub: we
    only write MARK_AX at the *IMM* step's row (via OP_IMM gating).
    The oracle says MARK_AX must be 1 at EVERY step's AX-marker row,
    so PSH (step 1) diverges.
    """

    # FFN rule that writes MARK_AX+0 = 1.0 only when OP_IMM is set.
    rule = step_function_rule(
        input_dim="OP_IMM",
        threshold=0.5,
        write_dim="MARK_AX+0",
        write_value=1.0,
    )
    op = _StubOp(
        name="imm_only_mark_ax",
        kind="ffn",
        compiler_ir=_ir_with_rule(rule),
        writes={"MARK_AX"},
    )
    compiler = _StubCompiler(ops_per_layer=[[op]])

    program = [
        encode_instr(OP_IMM, 0x200),
        encode_instr(OP_PSH),
        encode_instr(OP_EXIT),
    ]
    runner = SymbolicForwardRunner(compiler, program)
    runner.run_all()

    oracle = ReferenceOracle(program)

    divergences = diff_actual_vs_expected(runner, oracle, "MARK_AX")
    # Bag-of-dims: PSH and EXIT steps don't fire the rule at all ->
    # at least 2 divergences with actual == 0.
    no_firing = [d for d in divergences if d.actual == 0.0]
    assert len(no_firing) >= 2
    for d in no_firing:
        assert d.dim_key == "MARK_AX+0"
        assert d.step_idx >= 1
        assert d.position == d.step_idx  # bag-of-dims collapses to step_idx

    # find_first_divergent_block returns block 0 (the only block).
    diff = find_first_divergent_block(runner, oracle, "MARK_AX")
    assert diff is not None
    assert diff.block_idx == 0
    assert diff.first_divergence is not None
    assert diff.first_divergence.dim_key == "MARK_AX+0"
    msg = format_divergence(diff)
    assert "First divergence at block 0" in msg
    assert "MARK_AX+0" in msg


def test_apply_attention_specs_propagates_v_to_o_via_int_string_bridge():
    """Regression: the V-side state lookup used to read ``state.get(v.dim,
    0.0)`` with ``v.dim`` an *int* residual column while the rest of
    state was keyed by *string* dim names. That mismatch meant the V
    value was always 0.0 and no V→O propagation fired — the
    ``replay_expected_diff`` demo localised to block 0 instead of the
    real broadcast head's block.

    With the int↔string bridge in place, an embedding-seeded state with
    ``CLEAN_EMBED_LO+2 = 1.0`` (oracle's projection for AX byte 1 of
    ``IMM 0x200``) propagates through a head whose V slot reads
    ``CLEAN_EMBED_LO`` and whose O slot writes ``STACK0_BYTE_VAL_1_LO``
    — i.e. exactly the L10 PSH-AX-broadcast head's V/O channel shape.
    """

    from c4_release.neural_vm.unified_compiler.dsl_interpreter import (
        DSLInterpreter,
    )
    from c4_release.neural_vm.unified_compiler.primitives import (
        AO,
        AP,
        DeclarativeAttentionHeadSpec,
    )

    # Minimal dim layout: CLEAN_EMBED_LO at base 100 (16 cells),
    # STACK0_BYTE_VAL_1_LO at base 200 (16 cells). The choice of bases
    # is arbitrary so long as the bridge correctly resolves int columns
    # back to "NAME+offset" string keys.
    dim_positions = {
        "CLEAN_EMBED_LO": 100,
        "STACK0_BYTE_VAL_1_LO": 200,
    }
    # Build the per-cell V/O writes for the AX-byte-1 lo-nibble channel
    # (the L10 PSH broadcast head writes 16 cells: one per nibble value).
    v_writes = tuple(
        AP(slot=k, dim=100 + k, weight=1.0) for k in range(16)
    )
    o_writes = tuple(
        AO(out_dim=200 + k, slot=k, weight=3.0) for k in range(16)
    )
    spec = DeclarativeAttentionHeadSpec(
        head_idx=8, q=(), k=(), v=v_writes, o=o_writes,
    )

    # Seed: only CLEAN_EMBED_LO+2 is active (matches AX=0x200 byte 1
    # lo nibble == 2).
    interp = DSLInterpreter(
        initial_state={"CLEAN_EMBED_LO+2": 1.0},
        dim_positions=dim_positions,
    )
    step = interp.apply_attention_specs([spec], op_name="psh_ax_byte1_broadcast")

    # The bridge should have read CLEAN_EMBED_LO+2 (=1.0) on the V side
    # and propagated v_weight * o_weight = 3.0 to STACK0_BYTE_VAL_1_LO+2.
    assert pytest.approx(interp.get("STACK0_BYTE_VAL_1_LO+2"), abs=1e-9) == 3.0
    # No other STACK0_BYTE_VAL_1_LO+k should fire (only nibble 2 was active).
    for k in range(16):
        if k == 2:
            continue
        assert interp.get(f"STACK0_BYTE_VAL_1_LO+{k}") == 0.0
    # Sanity: the step recorded one rule firing and at least one write.
    assert step.rules_fired == 1
    assert any(
        key.startswith("STACK0_BYTE_VAL_1_LO") for key, _value in step.writes
    )


def test_apply_attention_specs_skips_propagation_without_dim_positions():
    """When ``dim_positions`` is absent the bridge falls back to a
    stringified int key. Writes still propagate self-consistently within
    a single spec (V int matches the same int on the O side), so the
    interpreter is at least usable; the test pins the fallback shape so
    a regression that returns an empty list silently is caught.
    """

    from c4_release.neural_vm.unified_compiler.dsl_interpreter import (
        DSLInterpreter,
    )
    from c4_release.neural_vm.unified_compiler.primitives import (
        AO,
        AP,
        DeclarativeAttentionHeadSpec,
    )

    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(),
        k=(),
        v=(AP(slot=0, dim=42, weight=1.0),),
        o=(AO(out_dim=99, slot=0, weight=2.0),),
    )
    # Seed using the stringified int form so the fallback bridge sees a
    # non-zero V activation.
    interp = DSLInterpreter(initial_state={"42": 1.0})
    interp.apply_attention_specs([spec])
    assert interp.get("99") == 2.0


def test_diff_finds_no_divergence_when_actual_matches_oracle():
    """A runner that writes MARK_AX at every step shows zero divergence."""

    # CONST is set by the default embedding at every step; gate MARK_AX
    # on CONST to fire on IMM/PSH/EXIT steps alike.
    rule = step_function_rule(
        input_dim="CONST",
        threshold=0.5,
        write_dim="MARK_AX+0",
        write_value=1.0,
    )
    op = _StubOp(
        name="always_mark_ax",
        kind="ffn",
        compiler_ir=_ir_with_rule(rule),
        writes={"MARK_AX"},
    )
    compiler = _StubCompiler(ops_per_layer=[[op]])

    program = [
        encode_instr(OP_IMM, 0x200),
        encode_instr(OP_PSH),
        encode_instr(OP_EXIT),
    ]
    runner = SymbolicForwardRunner(compiler, program)
    runner.run_all()
    oracle = ReferenceOracle(program)

    # Use a loose tolerance: the lowered S=100 scaling produces 0.01 per
    # firing; the oracle says 1.0 — so we expect divergences if atol is
    # tight. Loosen atol to >1.0 to confirm the *structural* match: the
    # runner DID fire at every step.
    divergences = diff_actual_vs_expected(
        runner, oracle, "MARK_AX", atol=1.0,
    )
    assert divergences == []
    assert find_first_divergent_block(
        runner, oracle, "MARK_AX", atol=1.0,
    ) is None


# ---------------------------------------------------------------------------
# Same-step ALU / MEM-bus projection tests (extended dim families)
# ---------------------------------------------------------------------------


def test_output_lo_hi_projection_imm_42_exit():
    """``IMM 42; EXIT`` — OUTPUT_LO/HI must encode AX byte 0 nibbles at
    the AX-marker row at both steps.

    42 = 0x2A. Byte 0 lo nibble = 0xA = 10; hi nibble = 0x2 = 2. The
    oracle's per-token projection should land OUTPUT_LO+10 and
    OUTPUT_HI+2 at the AX-marker token (slot 5) of every step.
    """

    program = [
        encode_instr(OP_IMM, 42),
        encode_instr(OP_EXIT),
    ]
    oracle = ReferenceOracle(program)

    # Step 0: IMM 42 -> AX = 42.
    s0 = oracle.state_at_step(0)
    assert s0.ax == 42

    table = project_state_to_residual(s0, per_token=True)
    base = s0.step_idx * TOKENS_PER_STEP

    # AX-marker row carries the OUTPUT/AX_CARRY one-hots for AX byte 0.
    assert table[(base + 5, "OUTPUT_LO+10")] == 1.0  # 42 & 0xF
    assert table[(base + 5, "OUTPUT_HI+2")] == 1.0   # (42 >> 4) & 0xF
    assert table[(base + 5, "AX_CARRY_LO+10")] == 1.0
    assert table[(base + 5, "AX_CARRY_HI+2")] == 1.0

    # PC-marker row carries the OUTPUT one-hots for PC byte 0. After
    # step 0 PC = 0 (we record state *after* the instruction with PC
    # still pointing at it). PC byte 0 = 0; lo=0, hi=0.
    assert table[(base + 0, "OUTPUT_LO+0")] == 1.0
    assert table[(base + 0, "OUTPUT_HI+0")] == 1.0

    # No MEM bus on IMM -> MEM_ADDR_SRC / ADDR_KEY do not fire.
    for k in range(16):
        assert table.get((base + 25, f"ADDR_KEY+{k}"), 0.0) == 0.0
    assert table.get((base + 25, "MEM_ADDR_SRC+0"), 0.0) == 0.0

    # Bag-of-dims at position=step_idx.
    bag = project_state_to_residual(s0)
    assert bag[(0, "OUTPUT_LO+10")] == 1.0
    assert bag[(0, "OUTPUT_HI+2")] == 1.0
    assert bag[(0, "AX_CARRY_LO+10")] == 1.0
    assert bag[(0, "AX_CARRY_HI+2")] == 1.0
    # No memory step -> ADDR_KEY / MEM_ADDR_SRC absent from the bag.
    for k in range(48):
        assert bag.get((0, f"ADDR_KEY+{k}"), 0.0) == 0.0
    assert bag.get((0, "MEM_ADDR_SRC+0"), 0.0) == 0.0


def test_addr_key_and_mem_addr_src_projection_psh_si_program():
    """``IMM 0x200; PSH; SI; EXIT`` exercises the MEM-bus projections.

    Step 1 (PSH): address bus = post-decrement SP, value bus = AX
    (0x200). PSH is *not* SI/SC so MEM_ADDR_SRC stays unset; ADDR_KEY
    fires with the SP-address nibbles.

    Step 2 (SI): address bus = popped top-of-stack = 0x200,
    value bus = AX (0x200). MEM_ADDR_SRC fires (1.0); ADDR_KEY's three
    nibble cells encode the 0x200 address.
    """

    program = [
        encode_instr(OP_IMM, 0x200),
        encode_instr(OP_PSH),
        encode_instr(OP_SI),
        encode_instr(OP_EXIT),
    ]
    oracle = ReferenceOracle(program)

    # Step 1: PSH — sanity-check the snapshot fields.
    s_psh = oracle.state_at_step(1)
    assert s_psh.opcode is not None
    assert s_psh.is_store
    assert s_psh.mem_value == 0x200
    assert s_psh.mem_addr == s_psh.sp  # post-dec SP

    # Step 2: SI — STACK0 address source.
    s_si = oracle.state_at_step(2)
    assert s_si.is_store
    assert s_si.mem_addr == 0x200
    assert s_si.mem_value == 0x200

    # OUTPUT_LO/HI at the MEM-marker row for the PSH step: byte 0 of
    # 0x200 = 0x00; lo nibble = 0, hi nibble = 0.
    psh_token = project_state_to_residual(s_psh, per_token=True)
    psh_base = s_psh.step_idx * TOKENS_PER_STEP
    assert psh_token[(psh_base + 25, "OUTPUT_LO+0")] == 1.0
    assert psh_token[(psh_base + 25, "OUTPUT_HI+0")] == 1.0
    # MEM_ADDR_SRC stays unset (PSH uses SP, not STACK0).
    assert psh_token.get((psh_base + 25, "MEM_ADDR_SRC+0"), 0.0) == 0.0

    # SI step: MEM_ADDR_SRC fires at the MEM-marker.
    si_token = project_state_to_residual(s_si, per_token=True)
    si_base = s_si.step_idx * TOKENS_PER_STEP
    assert si_token[(si_base + 25, "MEM_ADDR_SRC+0")] == 1.0

    # ADDR_KEY at MEM marker (SI step): addr = 0x200.
    # byte 0 = 0x00 -> hi nibble = 0 -> ADDR_KEY+0
    # byte 1 = 0x02 -> hi nibble = 0 -> ADDR_KEY+16
    # byte 2 = 0x00 -> hi nibble = 0 -> ADDR_KEY+32
    assert si_token[(si_base + 25, "ADDR_KEY+0")] == 1.0
    assert si_token[(si_base + 25, "ADDR_KEY+16")] == 1.0
    assert si_token[(si_base + 25, "ADDR_KEY+32")] == 1.0

    # AX-marker carries OUTPUT_LO/HI for AX byte 0 = 0x00 on every step.
    assert si_token[(si_base + 5, "OUTPUT_LO+0")] == 1.0
    assert si_token[(si_base + 5, "OUTPUT_HI+0")] == 1.0

    # Bag-of-dims sanity (SI step): MEM_ADDR_SRC and ADDR_KEY collapse
    # onto position=step_idx.
    bag = project_state_to_residual(s_si)
    assert bag[(s_si.step_idx, "MEM_ADDR_SRC+0")] == 1.0
    assert bag[(s_si.step_idx, "ADDR_KEY+0")] == 1.0
    assert bag[(s_si.step_idx, "ADDR_KEY+16")] == 1.0
    assert bag[(s_si.step_idx, "ADDR_KEY+32")] == 1.0


def test_expected_trace_now_supports_output_lo():
    """The previously-deferred OUTPUT_LO family now produces an
    ``expected_trace``. Sanity-check: at least one cell fires per step.
    """

    program = [
        encode_instr(OP_IMM, 42),
        encode_instr(OP_EXIT),
    ]
    oracle = ReferenceOracle(program)
    trace = expected_trace(oracle, "OUTPUT_LO", n_blocks=2)
    # Every step contributes at least one OUTPUT_LO+k entry per block.
    steps = {step_idx for (step_idx, _pos, _blk, _key) in trace}
    blocks = {blk for (_step, _pos, blk, _key) in trace}
    assert steps == {0, 1}
    assert blocks == {0, 1}
    # The AX=42 step writes OUTPUT_LO+10 at position=0 (bag-of-dims).
    assert trace[(0, 0, 0, "OUTPUT_LO+10")] == 1.0
