#!/usr/bin/env python3
"""
Smoke Tests - Pure Neural Mode

Pure-neural mirror of tests/test_smoke.py. Each test here mirrors a test
in test_smoke.py but uses the `pure_neural_runner` fixture
(AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)) so the
neural network drives execution with NO Python overrides.

Per F's Phase 8 scoping doc (docs/PHASE_8_RUNNER_SWITCH_SCOPE.md):
    Option B (Parallel) — keep handler-mode `test_smoke.py` unchanged and
    grow this parallel pure-neural suite as Phases 1-7 land. As each phase
    completes, xfail markers below get flipped to expect PASS.

Status convention (as of 2026-05-11):
    PASS    — Known-good in pure_neural per existing test_pure_neural_*.py
              files (no @xfail markers on the equivalent assertion AND the
              identical operands are in the existing Phase suite).
    @xfail  — Either (a) blocked on a later phase, OR (b) uses operands /
              bytecode patterns that are NOT in any existing test_pure_
              neural_*.py file, so the pure_neural pass-status is
              unverified. Reason cites which case and the suspected phase
              per F's matrix in docs/PHASE_8_RUNNER_SWITCH_SCOPE.md
              section 7.

Empirical sanity-check (2026-05-11): an initial pass attempted to mark
ADD/SUB/OR/AND/DIV/MOD/JMP-forward as PASS-expected (since their *opcodes*
work in pure_neural per the Phase suites). However the *specific operand
combinations* used in the smoke suite (e.g. ADD 10+32=42, JMP through
IMM-not-NOP) do not match the operands in the Phase suites. Empirically
they FAIL in pure_neural. The conservative choice taken here: every test
beyond `test_imm_exit` is xfail until the matching parametrization is
landed in the corresponding test_pure_neural_*.py file and confirmed.

This keeps the smoke pure_neural suite honest: PASS count today = 1
(IMM+EXIT only), and rises as Phase suites add the smoke operands or
fix the operand-specific bugs.

Run with:
    pytest tests/test_smoke_pure_neural.py -v --tb=short

Cross-references:
    - test_pure_neural_pc.py     (Phase 1, 13/13 PASS — source of PASSes)
    - test_pure_neural_psh_add.py (Phase 2 small ALU PASS, bitwise xfail)
    - test_pure_neural_multibyte.py (Phase 3 — all xfail)
    - test_pure_neural_jmp_bz.py  (Phase 4 — JMP forward PASS, rest xfail)
    - test_pure_neural_jsr_ent_lev.py (Phase 5 — JSR/LEV simple PASS)
    - test_pure_neural_io.py      (Phase 6 — PUTCHAR/GETCHAR PASS)
    - test_pure_neural_heap_div.py (Phase 7 — DIV/MOD small PASS)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from neural_vm.embedding import Opcode


# =============================================================================
# Basic Functionality Smoke Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralBasic:
    """Quick sanity checks in pure-neural mode.

    Phase 1 (IMM/EXIT) confirmed PASS via test_pure_neural_pc.py 13/13.
    Phase 2 small ADD/SUB confirmed PASS via test_pure_neural_psh_add.py.
    Phase 7 DIV/MOD small (no high-byte result) confirmed PASS via
    test_pure_neural_heap_div.py.
    """

    def test_imm_exit(self, pure_neural_runner, make_bytecode):
        """IMM + EXIT works in pure-neural (Phase 1)."""
        bytecode = make_bytecode([(Opcode.IMM, 42), Opcode.EXIT])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=10)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 2: ADD pure-neural is BROKEN for ALL operand pairs, "
               "not just (10,32). Empirical results 2026-05-11 (agent a2): "
               "  3+4   → 23 (0x17)   exp 7   (hi nibble off by +1) "
               "  10+20 → 46 (0x2E)   exp 30  (hi nibble off by +1) "
               "  5+0   → 21 (0x15)   exp 5   (hi nibble off by +1) "
               "  0+5   → 0  (0x00)   exp 5   (both nibbles wrong) "
               "  100+50→ 70 (0x46)   exp 150 (hi nibble wrong: got 4, exp 9) "
               "  10+32 → 58 (0x3A)   exp 42  (hi nibble off by +1) "
               "All operand pairs in test_pure_neural_psh_add.py::"
               "TestPureNeuralBinaryOps::test_add_small are xfailed for the "
               "same root cause (per commit 7b9d317 'Phase 2 triage'). "
               "Default mode (with handlers) passes all of these. "
               "Root cause: pure_neural pipeline does not wire AX-through-"
               "PSH preservation nor binary ALU readback of prev STACK0 "
               "from MEM into the L8 AddSub5StageBlock / L9 lookup ADD hi "
               "nibble units. Multiple OUTPUT_HI writers (L8 post_op, L9 "
               "lookup) appear to interact such that hi nibble result is "
               "consistently +1 for small operands. Fixing this is a "
               "Phase 2 effort, not an operand-specific patch. Promote "
               "when the entire Phase 2 ADD path is wired.",
        strict=False,
    )
    def test_add_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 32), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 2: SUB pure-neural passes for the specific operand "
               "set tested in test_pure_neural_psh_add.py "
               "(50-20, 100-50, 200-100, 42-42) but (50,8,42) is not in "
               "that set. Promote when added to Phase 2 suite and "
               "confirmed PASS.",
        strict=False,
    )
    def test_sub_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 50), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 7: MUL pure-neural blocked — _set_layer11_mul_partial "
               "/ _set_layer12_mul_combine not wired in pure_neural mode "
               "(see test_pure_neural_heap_div.py::test_mul_small).",
        strict=False,
    )
    def test_mul_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 6), Opcode.PSH,
            (Opcode.IMM, 7), Opcode.MUL,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 7: DIV pure-neural passes for the specific operand "
               "set tested in test_pure_neural_heap_div.py "
               "((20,5,4), (100,10,10), (255,1,255), (256,16,16)) but "
               "(84,2,42) is not in that set. Promote when added to Phase "
               "7 suite and confirmed PASS.",
        strict=False,
    )
    def test_div_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 84), Opcode.PSH,
            (Opcode.IMM, 2), Opcode.DIV,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 7: MOD pure-neural passes for the specific operand "
               "set tested in test_pure_neural_heap_div.py "
               "((7,3,1), (10,3,1), (15,4,3), (256,5,1)) but (43,10,3) is "
               "not in that set. Promote when added to Phase 7 suite and "
               "confirmed PASS.",
        strict=False,
    )
    def test_mod_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 43), Opcode.PSH,
            (Opcode.IMM, 10), Opcode.MOD,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 3


# =============================================================================
# Control Flow Smoke Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralControlFlow:
    """Control flow in pure-neural mode.

    Phase 4 confirmed JMP forward PASS (test_pure_neural_jmp_bz.py and
    git commit 6b10a10). BZ/BNZ remain blocked on _set_layer4_ffn PC
    carry-forward bug.
    """

    @pytest.mark.xfail(
        reason="Phase 4: JMP forward passes for the test_pure_neural_jmp_bz.py "
               "form `(JMP,2), NOP, (IMM,9), EXIT` returning 9 but FAILS "
               "for the smoke form `(JMP,2), (IMM,99), (IMM,42), EXIT` — "
               "empirically returns 0 instead of 42. Suggests the JMP "
               "target-resolution is sensitive to the instruction at the "
               "target index (NOP works, IMM doesn't relay correctly). "
               "Promote when this bytecode pattern is added to "
               "test_pure_neural_jmp_bz.py and confirmed.",
        strict=False,
    )
    def test_jmp_forward(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.JMP, 2),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=15)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 4: pure_neural BZ taken-path not yet supported (PC "
               "never redirects at step 1). See test_pure_neural_jmp_bz.py"
               "::TestPureNeuralBZ::test_bz_taken.",
        strict=False,
    )
    def test_bz_branch(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0),
            (Opcode.BZ, 3),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=15)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 4: pure_neural BNZ taken-path not yet supported (PC "
               "never redirects at step 1). See test_pure_neural_jmp_bz.py"
               "::TestPureNeuralBNZ::test_bnz_taken.",
        strict=False,
    )
    def test_bnz_branch(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 1),
            (Opcode.BNZ, 3),
            (Opcode.IMM, 99),
            (Opcode.IMM, 42),
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=15)
        assert result == 42


# =============================================================================
# Function Call Smoke Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralFunctionCall:
    """Function call in pure-neural mode.

    Phase 5: JSR-then-LEV simple roundtrip with AX-preservation works
    (see test_pure_neural_jsr_ent_lev.py::test_jsr_then_lev_simple).
    Callee-writes-AX is blocked on LEV neural path not restoring PC from
    mem[BP+8].
    """

    @pytest.mark.xfail(
        reason="Phase 5: callee-writes-AX blocked — _set_layer9_lev_bp_to_pc"
               "_relay does not restore PC from mem[BP+8]; callee body's IMM "
               "never reaches caller EXIT. "
               "See test_pure_neural_jsr_ent_lev.py::test_jsr_callee_writes_ax.",
        strict=False,
    )
    def test_simple_function(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.JSR, 3),
            Opcode.EXIT,
            Opcode.NOP,
            (Opcode.ENT, 0),
            (Opcode.IMM, 42),
            Opcode.LEV,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=30)
        assert result == 42


# =============================================================================
# Bitwise Smoke Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralBitwise:
    """Bitwise op smoke tests in pure-neural mode.

    Phase 2 bitwise (single-byte AND/OR) now PASS per recent fix
    fd45c7b "Phase 2 bitwise: fix BITWISE_OP relay/propagation at byte
    positions" and observed PASSes in test_pure_neural_psh_add.py. XOR
    is not exercised in test_pure_neural_psh_add.py but uses the same
    relay path; xfail until confirmed.
    """

    @pytest.mark.xfail(
        reason="Phase 2: OR pure-neural passes for the specific operand "
               "set tested in test_pure_neural_psh_add.py "
               "((0xF0,0x0F,0xFF), (0xAA,0x55,0xFF), (0,0,0)) but "
               "(0x0F,0x30,0x3F) is not in that set. Promote when added "
               "to Phase 2 suite and confirmed PASS.",
        strict=False,
    )
    def test_or_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x0F), Opcode.PSH,
            (Opcode.IMM, 0x30), Opcode.OR,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0x3F

    @pytest.mark.xfail(
        reason="Phase 2: AND pure-neural passes for the specific operand "
               "set tested in test_pure_neural_psh_add.py "
               "((0xFF,0x0F,0x0F), (0xF0,0x0F,0x00), (0xFF,0xFF,0xFF)) "
               "but (0xFF,0x2A,0x2A) is not in that set. Promote when "
               "added to Phase 2 suite and confirmed PASS.",
        strict=False,
    )
    def test_and_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0x2A), Opcode.AND,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0x2A

    @pytest.mark.xfail(
        reason="Phase 2: XOR pure-neural path not directly exercised by "
               "test_pure_neural_psh_add.py; uses BITWISE_OP relay same as "
               "AND/OR but not verified end-to-end. Promote to PASS when "
               "added to test_pure_neural_psh_add.py.",
        strict=False,
    )
    def test_xor_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0xD5), Opcode.XOR,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0x2A


# =============================================================================
# Comparison Smoke Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralComparison:
    """Comparison ops in pure-neural mode.

    Per F's matrix (PHASE_8_RUNNER_SWITCH_SCOPE.md Appendix B):
        EQ/NE/LT/GT/LE/GE in L4/L5 FFN — "likely works (under-tested)"

    Not exercised in any test_pure_neural_*.py file today. Marked xfail
    until added there and confirmed.
    """

    @pytest.mark.xfail(
        reason="Phase 2: EQ pure-neural path not directly exercised by any "
               "test_pure_neural_*.py file; F's matrix marks 'likely works "
               "(under-tested)'. Promote when added to Phase 2 suite.",
        strict=False,
    )
    def test_eq_true(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 42), Opcode.PSH,
            (Opcode.IMM, 42), Opcode.EQ,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 1

    @pytest.mark.xfail(
        reason="Phase 2: EQ pure-neural path not directly exercised; see "
               "test_eq_true note.",
        strict=False,
    )
    def test_eq_false(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.EQ,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0

    @pytest.mark.xfail(
        reason="Phase 2: LT pure-neural path not directly exercised; F's "
               "matrix marks 'likely works (under-tested)'.",
        strict=False,
    )
    def test_lt_true(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.LT,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 1

    @pytest.mark.xfail(
        reason="Phase 2: NE pure-neural path not directly exercised; F's "
               "matrix marks 'likely works (under-tested)'.",
        strict=False,
    )
    def test_ne_true(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.NE,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 1

    @pytest.mark.xfail(
        reason="Phase 2: GT pure-neural path not directly exercised; F's "
               "matrix marks 'likely works (under-tested)'.",
        strict=False,
    )
    def test_gt_true(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 20), Opcode.PSH,
            (Opcode.IMM, 10), Opcode.GT,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 1

    @pytest.mark.xfail(
        reason="Phase 2: LE pure-neural path not directly exercised; F's "
               "matrix marks 'likely works (under-tested)'.",
        strict=False,
    )
    def test_le_true(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 10), Opcode.PSH,
            (Opcode.IMM, 20), Opcode.LE,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 1

    @pytest.mark.xfail(
        reason="Phase 2: GE pure-neural path not directly exercised; F's "
               "matrix marks 'likely works (under-tested)'.",
        strict=False,
    )
    def test_ge_true(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 20), Opcode.PSH,
            (Opcode.IMM, 10), Opcode.GE,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 1


# =============================================================================
# Address/Stack Smoke Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralAddress:
    """LEA and ADJ in pure-neural mode.

    Neither LEA nor ADJ has dedicated pure_neural test coverage today.
    Per F's matrix (Appendix B), ADJ is "now fully neural (migrated)" but
    LEA is not analyzed. Both are marked xfail pending Phase 5 (LEA depends
    on ENT setting BP, which is xfail) and direct ADJ testing.
    """

    @pytest.mark.xfail(
        reason="Phase 5: LEA depends on ENT establishing BP; ENT with imm>0 "
               "blocked (see test_pure_neural_jsr_ent_lev.py::"
               "test_ent_decrements_sp_by_imm).",
        strict=False,
    )
    def test_lea_basic(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            Opcode.ENT,
            (Opcode.IMM, 0),
            (Opcode.LEA, 2),
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result != 0

    @pytest.mark.xfail(
        reason="Phase 2: ADJ migrated to fully-neural per F's matrix but "
               "not exercised by any test_pure_neural_*.py file today. "
               "Promote when added to Phase 2 suite.",
        strict=False,
    )
    def test_adj_sp(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 42),
            Opcode.PSH,
            Opcode.ADJ,
            Opcode.EXIT,
        ])
        bytecode[2] = (Opcode.ADJ | (8 << 8))
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 42


# =============================================================================
# Memory Operation Smoke Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralMemory:
    """Memory load/store in pure-neural mode.

    Per Phase 7 test_pure_neural_heap_div.py, SI/LI and SC/LC are blocked
    on _set_layer14_mem_generation / _set_layer15_memory_lookup gaps.
    """

    @pytest.mark.xfail(
        reason="Phase 7: SI/LI blocked — _set_layer14_mem_generation / "
               "_set_layer15_memory_lookup / _set_layer7_memory_heads gaps. "
               "See test_pure_neural_heap_div.py::test_si_then_li.",
        strict=False,
    )
    def test_si_li_roundtrip(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x200),
            Opcode.PSH,
            (Opcode.IMM, 42),
            Opcode.SI,
            (Opcode.IMM, 0x200),
            Opcode.LI,
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=30)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 7: SC/LC blocked — _set_layer14_mem_generation / "
               "_set_layer15_memory_lookup gaps for char-width. "
               "See test_pure_neural_heap_div.py::test_sc_then_lc.",
        strict=False,
    )
    def test_sc_lc_roundtrip(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x200),
            Opcode.PSH,
            (Opcode.IMM, 42),
            Opcode.SC,
            (Opcode.IMM, 0x200),
            Opcode.LC,
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=30)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 7: SI/LI blocked (see test_si_li_roundtrip).",
        strict=False,
    )
    def test_si_li_zero(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x300),
            Opcode.PSH,
            (Opcode.IMM, 0),
            Opcode.SI,
            (Opcode.IMM, 0x300),
            Opcode.LI,
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=30)
        assert result == 0

    @pytest.mark.xfail(
        reason="Phase 7: SI/LI multi-address blocked — needs per-step "
               "memory persistence in pure-neural. See "
               "test_pure_neural_heap_div.py::test_two_writes_two_reads.",
        strict=False,
    )
    def test_si_li_multiple_stores(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x200),
            Opcode.PSH,
            (Opcode.IMM, 10),
            Opcode.SI,
            (Opcode.IMM, 0x300),
            Opcode.PSH,
            (Opcode.IMM, 99),
            Opcode.SI,
            (Opcode.IMM, 0x300),
            Opcode.LI,
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=40)
        assert result == 99

    @pytest.mark.xfail(
        reason="Phase 7: SI/LI overwrite blocked (see test_si_li_roundtrip).",
        strict=False,
    )
    def test_si_li_overwrite(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x200),
            Opcode.PSH,
            (Opcode.IMM, 10),
            Opcode.SI,
            (Opcode.IMM, 0x200),
            Opcode.PSH,
            (Opcode.IMM, 55),
            Opcode.SI,
            (Opcode.IMM, 0x200),
            Opcode.LI,
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=40)
        assert result == 55

    @pytest.mark.xfail(
        reason="Phase 7: SI/LI blocked + Phase 3 multi-byte (0x1234 > 0xFF) "
               "depends on _set_layer11_mul_partial / "
               "_set_layer12_mul_combine multibyte routing. See "
               "test_pure_neural_multibyte.py.",
        strict=False,
    )
    def test_si_li_16bit_value(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x200),
            Opcode.PSH,
            (Opcode.IMM, 0x1234),
            Opcode.SI,
            (Opcode.IMM, 0x200),
            Opcode.LI,
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=30)
        assert result == 0x1234


# =============================================================================
# Shift Smoke Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralShift:
    """Shift ops in pure-neural mode.

    Per Phase 7 test_pure_neural_heap_div.py:
        - SHL: all small cases xfail (_set_layer13_shifts / ALUShift not wired)
        - SHR: 1>>1=0 passes; 8>>1, 256>>2, 255>>1 xfail
    """

    @pytest.mark.xfail(
        reason="Phase 7: SHL blocked — _set_layer13_shifts / ALUShift not "
               "wired in pure-neural. See test_pure_neural_heap_div.py::"
               "test_shl_small.",
        strict=False,
    )
    def test_shl(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 21), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.SHL,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 42

    @pytest.mark.xfail(
        reason="Phase 7: SHR blocked for shifts >0 — _set_layer13_shifts / "
               "ALUShift not wired in pure-neural. See "
               "test_pure_neural_heap_div.py::test_shr_small[84-1-42] "
               "(equivalent — 84>>1=42 is blocked, only 1>>1=0 passes).",
        strict=False,
    )
    def test_shr(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 84), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.SHR,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 42


# =============================================================================
# Quick Full Pipeline (Pure Neural)
# =============================================================================

class TestSmokePureNeuralPipeline:
    """End-to-end pipeline smoke test in pure-neural mode.

    Compiling C code (`return 6 * 7;`) exercises MUL, JSR/ENT/LEV, and
    function-call ABI in pure_neural. Blocked by Phases 5 + 7 (MUL).
    """

    @pytest.mark.slow
    @pytest.mark.timeout(900)
    @pytest.mark.xfail(
        reason="Phase 5 + Phase 7: compiled `return 6*7` uses MUL "
               "(Phase 7 blocked) inside a function frame (Phase 5 blocked). "
               "See PHASE_8_RUNNER_SWITCH_SCOPE.md sections 3+7.",
        strict=False,
    )
    def test_compile_and_run(self, pure_neural_runner):
        from src.compiler import compile_c

        source = """
        int main() {
            return 6 * 7;
        }
        """
        bytecode, data = compile_c(source)
        _, result = pure_neural_runner.run(bytecode, data, max_steps=50)
        assert result == 42


# =============================================================================
# 32-bit Value Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeural32Bit:
    """Multi-byte AX coherence in pure-neural mode.

    Per Phase 3 test_pure_neural_multibyte.py, ALL multi-byte arith
    (ADD overflow, SUB borrow, MUL high-byte) is xfail today.
    """

    @pytest.mark.xfail(
        reason="Phase 3: ADD overflow >255 blocked — _set_layer9_alu carry "
               "handling + post_op CarryPropagationPostOp not wired for "
               "multi-byte ADD in pure_neural. See test_pure_neural_"
               "multibyte.py::test_add_overflow_300.",
        strict=False,
    )
    def test_add_16bit(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 200), Opcode.PSH,
            (Opcode.IMM, 100), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 300

    @pytest.mark.xfail(
        reason="Phase 3: ADD carry cascade blocked — same as test_add_16bit. "
               "See test_pure_neural_multibyte.py::test_add_overflow_510.",
        strict=False,
    )
    def test_add_carry_cascade(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.ADD,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0x100

    @pytest.mark.xfail(
        reason="Phase 3: SUB borrow depends on multi-byte minuend which "
               "requires multi-byte ADD first (Phase 3 blocker). See "
               "test_pure_neural_multibyte.py::test_sub_borrow_into_byte1. "
               "Also IMM 0x100 (>255) is itself blocked.",
        strict=False,
    )
    def test_sub_16bit(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x100), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0xFF

    @pytest.mark.xfail(
        reason="Phase 3: SUB borrow cascade blocked — same as test_sub_16bit. "
               "32-bit underflow result (0xFFFFFFFF) needs full multi-byte "
               "borrow propagation.",
        strict=False,
    )
    def test_sub_borrow_cascade(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.SUB,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0xFFFFFFFF

    @pytest.mark.xfail(
        reason="Phase 3: 16-bit IMM (0x0F00) > 255 requires multi-byte "
               "IMM bake, not currently exercised. See test_pure_neural_"
               "multibyte.py.",
        strict=False,
    )
    def test_or_16bit(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x0F00), Opcode.PSH,
            (Opcode.IMM, 0x00FF), Opcode.OR,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0x0FFF

    @pytest.mark.xfail(
        reason="Phase 3: 16-bit AND blocked (same as test_or_16bit).",
        strict=False,
    )
    def test_and_16bit(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x0FFF), Opcode.PSH,
            (Opcode.IMM, 0x00FF), Opcode.AND,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0x00FF

    @pytest.mark.xfail(
        reason="Phase 3: 16-bit XOR blocked (same as test_or_16bit).",
        strict=False,
    )
    def test_xor_16bit(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x0F0F), Opcode.PSH,
            (Opcode.IMM, 0x00FF), Opcode.XOR,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 0x0FF0

    @pytest.mark.xfail(
        reason="Phase 7: MUL >255 blocked — _set_layer11_mul_partial / "
               "_set_layer12_mul_combine not wired. See "
               "test_pure_neural_multibyte.py::test_mul_to_high_byte_300.",
        strict=False,
    )
    def test_mul_overflow(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 100), Opcode.PSH,
            (Opcode.IMM, 5), Opcode.MUL,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 500

    @pytest.mark.xfail(
        reason="Phase 3+7: SHL by 8 blocked — cross-byte SHL not implemented "
               "(_set_layer13_shifts gap). See test_pure_neural_heap_div.py"
               "::test_shl_small + PHASE_8_RUNNER_SWITCH_SCOPE.md.",
        strict=False,
    )
    def test_shl_8bit(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 1), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SHL,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 256

    @pytest.mark.xfail(
        reason="Phase 3+7: SHR by 8 blocked — cross-byte SHR not implemented. "
               "Also IMM 0x100 requires multi-byte IMM bake.",
        strict=False,
    )
    def test_shr_8bit(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 0x100), Opcode.PSH,
            (Opcode.IMM, 8), Opcode.SHR,
            Opcode.EXIT
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=20)
        assert result == 1


# =============================================================================
# Integration Tests (Pure Neural)
# =============================================================================

class TestSmokePureNeuralIntegration:
    """Multi-step integration in pure-neural mode.

    Combining EQ + BZ requires Phase 2 (EQ under-tested) + Phase 4 (BZ
    fall-through xfail). Blocked.
    """

    @pytest.mark.xfail(
        reason="Phase 2 (EQ under-tested) + Phase 4 (BZ fall-through xfail "
               "per test_pure_neural_jmp_bz.py::test_bz_not_taken).",
        strict=False,
    )
    def test_cmp_and_branch(self, pure_neural_runner, make_bytecode):
        bytecode = make_bytecode([
            (Opcode.IMM, 5), Opcode.PSH,
            (Opcode.IMM, 5), Opcode.EQ,
            (Opcode.BZ, 6),
            (Opcode.IMM, 42),
            Opcode.EXIT,
            (Opcode.IMM, 0),
            Opcode.EXIT,
        ])
        _, result = pure_neural_runner.run(bytecode, b'', max_steps=30)
        assert result == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
