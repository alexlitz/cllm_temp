"""Symbolic byte-signature bounds checks against declarative lowering targets.

These tests pin down the *declarative* expected byte values for two clusters
that have historically diverged in the autoregressive neural pipeline:

* **id575 ENT saved-BP byte 2** (``int add(int a, int b)`` / ``add(57, 11)``):
  the outer ``main`` ENT step makes BP = 0xfff0, so the BP token row must
  emit bytes ``[0xf0, 0xff, 0x00, 0x00]``. ``BP_byte2`` is one of the slots
  that wobbles when the local-frame routing collides with the saved-BP word
  written at ``mem[0xfff0] = 0x10000`` (whose ``MEM_value_byte2`` is 0x01).
* **id800 STACK0_byte0 PSH cluster** (``22 + 24 * 22``): the first PSH stores
  ``22 = 0x16`` on the stack, so STACK0 must read back as
  ``[0x16, 0x00, 0x00, 0x00]``. ``STACK0_byte0`` collapses to lo nibble 6 /
  hi nibble 1 — a useful symbolic target for the L10 stack0 persistence /
  loaded-byte tail rules.

The tests do NOT depend on the neural model. They check the lowering target
itself, then drive a small declarative bake-rule probe with that target so a
future weight-bake regression that overwrites the loaded STACK0 byte trips
this file before the full 1096 diagnostic.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.compiler import compile_c  # noqa: E402
from neural_vm.unified_compiler.ir import CompilerIR  # noqa: E402
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _tail_bit32_result_correction_rules,
)
from tests.symbolic_byte_signature import (  # noqa: E402
    SymbolicStepByteSignature,
    find_steps_by_opcode,
    step_token_stream,
    symbolic_byte_signatures,
)
from tests.test_1096_neural_declarative_diagnostic import (  # noqa: E402
    _build_symbolic_expected_execution,
)


# Cached compile_c outputs for the test programs of interest.
ID575_SRC = (
    "int add(int a, int b) { return a + b; }\n"
    "int main() { return add(57, 11); }\n"
)
ID800_SRC = "int main() { return 22 + 24 * 22; }\n"
REC_FIB_SRC_TEMPLATE = (
    "int fib(int n) {\n"
    "    if (n < 2) return n;\n"
    "    return fib(n-1) + fib(n-2);\n"
    "}\n"
    "int main() { return fib(%d); }\n"
)


@pytest.fixture(scope="module")
def id575_signatures() -> list[SymbolicStepByteSignature]:
    bytecode, data = compile_c(ID575_SRC)
    return symbolic_byte_signatures(bytecode, data, max_steps=200)


@pytest.fixture(scope="module")
def id800_signatures() -> list[SymbolicStepByteSignature]:
    bytecode, data = compile_c(ID800_SRC)
    return symbolic_byte_signatures(bytecode, data, max_steps=200)


@pytest.fixture(scope="module")
def id740_signatures() -> list[SymbolicStepByteSignature]:
    bytecode, data = compile_c(REC_FIB_SRC_TEMPLATE % 1)
    return symbolic_byte_signatures(bytecode, data, max_steps=200)


# --------------------------------------------------------------------------- #
# Sanity: helper token stream agrees with the 1096-diagnostic token recorder. #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("src", [ID575_SRC, ID800_SRC])
def test_signature_token_stream_matches_diagnostic_recorder(src: str) -> None:
    bytecode, data = compile_c(src)
    sigs = symbolic_byte_signatures(bytecode, data, max_steps=200)
    expected = _build_symbolic_expected_execution(bytecode, data)
    assert step_token_stream(sigs) == expected.context[expected.prefix_len:]


def test_signature_byte_slot_count_matches_step_layout(
    id800_signatures: list[SymbolicStepByteSignature],
) -> None:
    assert id800_signatures, "id800 must execute at least one step"
    first = id800_signatures[0]
    # PC/AX/SP/BP/STACK0 = 5*4 = 20 byte slots, MEM_addr+MEM_value = 8.
    assert len(first.slot_bytes) == 28
    assert len(first.tokens) == 35


# --------------------------------------------------------------------------- #
# Cluster 1: id575 ENT saved BP / outer-frame BP byte 2 expectations.         #
# --------------------------------------------------------------------------- #


def test_id575_outer_main_ent_saved_bp_byte_signature(
    id575_signatures: list[SymbolicStepByteSignature],
) -> None:
    ent_steps = find_steps_by_opcode(id575_signatures, "ENT")
    assert len(ent_steps) >= 1, "id575 should have at least one ENT"

    # The first ENT is the outer ``main`` frame — saves the initial BP
    # (0x10000) onto the stack and reloads BP = 0xfff0. The neural BP byte
    # row must emit [0xf0, 0xff, 0x00, 0x00]; BP_byte2 = 0x00 is the
    # historically-divergent slot when the saved-BP MEM_value (byte 2 = 0x01)
    # leaks into the BP routing.
    outer_ent = ent_steps[0]
    assert outer_ent.byte("BP_byte0") == 0xF0
    assert outer_ent.byte("BP_byte1") == 0xFF
    assert outer_ent.byte("BP_byte2") == 0x00
    assert outer_ent.byte("BP_byte3") == 0x00

    # The same step writes saved_bp = 0x10000 to mem[0xfff0]; the MEM_value
    # byte 2 = 0x01 is the actual interfering signal that *can* leak into
    # BP_byte2 through declarative-vs-neural lowering drift. Pinning it down
    # here documents the collision target the bounds check must guard.
    assert outer_ent.byte("MEM_addr0") == 0xF0
    assert outer_ent.byte("MEM_addr1") == 0xFF
    assert outer_ent.byte("MEM_value0") == 0x00
    assert outer_ent.byte("MEM_value1") == 0x00
    assert outer_ent.byte("MEM_value2") == 0x01
    assert outer_ent.byte("MEM_value3") == 0x00


def test_id575_inner_add_ent_saved_bp_byte_signature(
    id575_signatures: list[SymbolicStepByteSignature],
) -> None:
    ent_steps = find_steps_by_opcode(id575_signatures, "ENT")
    assert len(ent_steps) >= 2, "id575 should have two ENT steps (main + add)"

    # The inner ``add`` frame: saved_bp = 0xfff0 (outer main BP) is written
    # to mem[0xffd0]; the new BP becomes 0xffd0. Now MEM_value byte 1 = 0xff
    # actively interferes with BP_byte1.
    inner_ent = ent_steps[1]
    assert inner_ent.byte("BP_byte0") == 0xD0
    assert inner_ent.byte("BP_byte1") == 0xFF
    assert inner_ent.byte("BP_byte2") == 0x00
    assert inner_ent.byte("BP_byte3") == 0x00

    assert inner_ent.byte("MEM_value0") == 0xF0
    assert inner_ent.byte("MEM_value1") == 0xFF
    assert inner_ent.byte("MEM_value2") == 0x00
    assert inner_ent.byte("MEM_value3") == 0x00


# --------------------------------------------------------------------------- #
# Cluster 2: id800 STACK0 byte 0 PSH expectations.                            #
# --------------------------------------------------------------------------- #


def test_id800_first_psh_stack0_byte_signature(
    id800_signatures: list[SymbolicStepByteSignature],
) -> None:
    psh_steps = find_steps_by_opcode(id800_signatures, "PSH")
    assert len(psh_steps) >= 1

    first_psh = psh_steps[0]
    # PSH 22 → STACK0 = [0x16, 0x00, 0x00, 0x00]; STACK0_byte0 lo=6/hi=1.
    assert first_psh.byte("STACK0_byte0") == 0x16
    assert first_psh.byte("STACK0_byte1") == 0x00
    assert first_psh.byte("STACK0_byte2") == 0x00
    assert first_psh.byte("STACK0_byte3") == 0x00
    assert first_psh.nibbles("STACK0_byte0") == (0x6, 0x1)

    # MEM row writes the same value (mem[0xfff8] = 0x16) — the L10 stack0
    # persistence / loaded-byte tail rules must keep this byte routed to the
    # STACK0_byte0 OUTPUT band even though MEM_value0 is identical.
    assert first_psh.byte("MEM_addr0") == 0xF8
    assert first_psh.byte("MEM_value0") == 0x16


def test_id800_second_psh_stack0_byte_signature(
    id800_signatures: list[SymbolicStepByteSignature],
) -> None:
    psh_steps = find_steps_by_opcode(id800_signatures, "PSH")
    assert len(psh_steps) >= 2

    # Second PSH (after IMM 24) → STACK0 = [0x18, 0, 0, 0]; lo=8/hi=1.
    second_psh = psh_steps[1]
    assert second_psh.byte("STACK0_byte0") == 0x18
    assert second_psh.nibbles("STACK0_byte0") == (0x8, 0x1)


# --------------------------------------------------------------------------- #
# Cluster 3: id740 recursive JSR return-address frame invariants.             #
# --------------------------------------------------------------------------- #


def test_id740_recursive_jsr_stack0_is_return_address_0x0122(
    id740_signatures: list[SymbolicStepByteSignature],
) -> None:
    jsr_steps = find_steps_by_opcode(id740_signatures, "JSR")
    assert len(jsr_steps) >= 2, "fib(1) should call main, then recursively call fib"

    recursive_call = jsr_steps[1]
    assert recursive_call.byte("STACK0_byte0") == 0x22
    assert recursive_call.byte("STACK0_byte1") == 0x01
    assert recursive_call.byte("STACK0_byte2") == 0x00
    assert recursive_call.byte("STACK0_byte3") == 0x00

    # JSR stores the same return address at the decremented stack top.
    assert recursive_call.byte("MEM_addr0") == 0xE0
    assert recursive_call.byte("MEM_addr1") == 0xFF
    assert recursive_call.byte("MEM_value0") == 0x22
    assert recursive_call.byte("MEM_value1") == 0x01
    assert recursive_call.byte("MEM_value2") == 0x00
    assert recursive_call.byte("MEM_value3") == 0x00


def test_id740_inner_ent_saves_caller_bp_not_return_address(
    id740_signatures: list[SymbolicStepByteSignature],
) -> None:
    ent_steps = find_steps_by_opcode(id740_signatures, "ENT")
    assert len(ent_steps) >= 2, "fib(1) should enter main and then fib"

    inner_fib_ent = ent_steps[1]
    assert inner_fib_ent.byte("BP_byte0") == 0xD8
    assert inner_fib_ent.byte("BP_byte1") == 0xFF
    assert inner_fib_ent.byte("STACK0_byte0") == 0xF0
    assert inner_fib_ent.byte("STACK0_byte1") == 0xFF

    # The saved frame pointer lives in MEM_value; the previous JSR return
    # address remains at BP+8 and must not be confused with saved BP.
    assert inner_fib_ent.byte("MEM_addr0") == 0xD8
    assert inner_fib_ent.byte("MEM_addr1") == 0xFF
    assert inner_fib_ent.byte("MEM_value0") == 0xF0
    assert inner_fib_ent.byte("MEM_value1") == 0xFF
    assert inner_fib_ent.byte("MEM_value2") == 0x00
    assert inner_fib_ent.byte("MEM_value3") == 0x00


# --------------------------------------------------------------------------- #
# Tail bake-rule bounds: feed the symbolic byte signature into the L10        #
# "loaded byte" tail rule and assert it doesn't overwrite the expected band.  #
# --------------------------------------------------------------------------- #


def _tail_rule(name: str):
    for rule in _tail_bit32_result_correction_rules():
        if rule.name == name:
            return rule
    raise AssertionError(f"missing tail rule {name}")


def _single_rule_ir(rule) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.append(rule)
    return ir


def test_l10_tail_loaded_byte01_preserves_stack0_byte0_from_signature(
    id800_signatures: list[SymbolicStepByteSignature],
) -> None:
    """The id800 PSH STACK0_byte0 = 0x16 must survive the loaded-byte tail.

    The ``tail_stack0_store_loaded_byte_01`` rule is the L10 tail that
    promotes a previously-loaded STACK0 byte through ``IS_BYTE / HAS_SE /
    CMP+3`` gating. We drive the rule with the OUTPUT_LO/HI bands set to the
    *symbolic* expected nibbles for STACK0_byte0 (0x16 → lo 6 / hi 1) and
    assert the residual stays at its strong loaded value with no spurious
    OUTPUT_LO+0 fall-through.
    """

    first_psh = find_steps_by_opcode(id800_signatures, "PSH")[0]
    lo, hi = first_psh.nibbles("STACK0_byte0")
    assert (lo, hi) == (0x6, 0x1)

    ir = _single_rule_ir(_tail_rule("tail_stack0_store_loaded_byte_01"))
    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "HAS_SE": 1.0,
        "CMP+3": 4.0,
        f"OUTPUT_LO+{lo}": 2_000_000_000.0,
        f"OUTPUT_HI+{hi}": 2_000_000_000.0,
    })

    # The rule must not displace the loaded band.
    assert out[f"OUTPUT_LO+{lo}"] == pytest.approx(2_000_000_000.0)
    assert out[f"OUTPUT_HI+{hi}"] == pytest.approx(2_000_000_000.0)
    # And it must not promote the (0,0) fallback that would corrupt byte 0.
    assert out.get("OUTPUT_LO+0", 0.0) <= 0.0


def test_l10_tail_loaded_byte02_preserves_bp_byte2_from_signature(
    id575_signatures: list[SymbolicStepByteSignature],
) -> None:
    """id575 ENT BP_byte2 = 0x00 must survive the loaded-byte tail.

    The lo/hi for 0x00 are both 0, so the residual probe puts a strong
    positive residual on OUTPUT_LO+0 / OUTPUT_HI+0 and asserts the
    ``tail_stack0_store_loaded_byte_02`` rule keeps them in place. The rule
    is the one that protects byte-2 spans (``H1+3``, ``BYTE_INDEX_2``), which
    is the gating used at the outer-frame BP_byte2 emission position.
    """

    outer_ent = find_steps_by_opcode(id575_signatures, "ENT")[0]
    lo, hi = outer_ent.nibbles("BP_byte2")
    assert (lo, hi) == (0x0, 0x0)

    ir = _single_rule_ir(_tail_rule("tail_stack0_store_loaded_byte_02"))
    out = ir.symbolic_ffn({
        "IS_BYTE": 1.0,
        "H1+3": 1.0,
        "BYTE_INDEX_2": 0.97,
        f"OUTPUT_LO+{lo}": 8_000_000.0,
        f"OUTPUT_HI+{hi}": 8_000_000.0,
    })

    assert out[f"OUTPUT_LO+{lo}"] == pytest.approx(8_000_000.0)
    assert out[f"OUTPUT_HI+{hi}"] == pytest.approx(8_000_000.0)


# --------------------------------------------------------------------------- #
# Cross-check the symbolic stream is internally consistent with declarative   #
# exit codes (suite-expected vs declarative-oracle).                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "src, expected_ax",
    [(ID575_SRC, 68), (ID800_SRC, 550)],
)
def test_signature_terminal_step_carries_declarative_exit(
    src: str, expected_ax: int
) -> None:
    bytecode, data = compile_c(src)
    sigs = symbolic_byte_signatures(bytecode, data, max_steps=200)

    halted = [s for s in sigs if s.halted]
    assert len(halted) == 1, "exactly one halted step expected"

    terminal = halted[0]
    assert terminal.opcode_name == "EXIT"
    # AX bytes on the EXIT step must spell the expected return value.
    ax_value = (
        terminal.byte("AX_byte0")
        | (terminal.byte("AX_byte1") << 8)
        | (terminal.byte("AX_byte2") << 16)
        | (terminal.byte("AX_byte3") << 24)
    )
    assert ax_value == expected_ax
