"""test_exact_steps.py — the MEMORY-OPERAND ALU family (C4_EXACT_STEPS) byte-exact
proof + the step-count reduction vs the interpreted 4-forward addressed ALU op.

Run:  C4_EXACT_STEPS=1 python -m pytest c4_min/test_exact_steps.py -q
  or: C4_EXACT_STEPS=1 python c4_min/test_exact_steps.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min import isa
from c4_min import nibble_exact_steps as ES


@pytest.fixture(scope="module")
def model8():
    os.environ["C4_EXACT_STEPS"] = "1"
    os.environ["C4_MEM_OPERAND"] = "1"
    return ES.build_exact_steps_model(code_size=6)


def _one_mop(model, L, op, mem_a, ax0, mask=0xFF):
    """AX = mem[A] <op> AX via ONE <OP>M step; AX starts at ax0 (seeded by an IMM)."""
    A_ADDR = 0x40
    seed = {A_ADDR: mem_a}
    prog = [isa.Instr(isa.IMM, ax0), isa.Instr(op, A_ADDR), isa.Instr(isa.HALT, 0)]
    tr = ES.run_exact_steps(model, L, prog, max_steps=6, seed_mem=seed, mask=mask)
    ref = ES.ref_interpret_exact(prog, seed_mem=seed, mask=mask)
    return tr[-1], ref[-1]


def test_addm_subm_byte_exact(model8):
    model, L = model8
    fails = []
    # a lean but edge-covering battery (each case is one ~17s 252-block forward):
    # wrap (255+1), non-commutative SUB (3-17 -> 242), ordinary.
    for op, mem_a, ax0 in [(ES.ADDM, 200, 100), (ES.ADDM, 255, 1),
                           (ES.SUBM, 3, 17), (ES.SUBM, 200, 100)]:
        got, ref = _one_mop(model, L, op, mem_a, ax0, mask=0xFF)
        if got != ref:
            fails.append((ES.NAMES[op], mem_a, ax0, got, ref))
    assert not fails, f"ADDM/SUBM mismatches: {fails}"


def test_mulm_byte_exact(model8):
    model, L = model8
    fails = []
    for mem_a, ax0 in [(17, 13), (16, 16), (255, 255)]:   # ordinary, exact-256, wrap
        got, ref = _one_mop(model, L, ES.MULM, mem_a, ax0, mask=0xFF)
        exp = (mem_a * ax0) & 0xFF
        if got != ref or got != exp:
            fails.append((mem_a, ax0, got, ref, exp))
    assert not fails, f"MULM mismatches: {fails}"


def test_divm_modm_byte_exact(model8):
    model, L = model8
    fails = []
    # DIVM/MODM compute mem[addr] // AX / mem[addr] % AX; AX==0 -> 0 (ISA 4.2).
    for op, mem_a, ax0 in [(ES.DIVM, 100, 7), (ES.DIVM, 100, 0),
                           (ES.MODM, 100, 7), (ES.MODM, 255, 16)]:
        got, ref = _one_mop(model, L, op, mem_a, ax0, mask=0xFF)
        if got != ref:
            fails.append((ES.NAMES[op], mem_a, ax0, got, ref))
    assert not fails, f"DIVM/MODM mismatches: {fails}"


def test_dot_via_mulm_addm_composes(model8):
    """A dot product acc = Σ a[i]*b[i] with a[i] pre-scaled INTO memory, then
    MULM;stack... — here we test the pure memory-operand ALU chain:
        acc = ((mem[a0]*mem_ax) ...)  a simple fused arithmetic chain composes."""
    model, L = model8
    # compute ((7 + mem[A]) - mem[B]) * mem[C] using ADDM/SUBM/MULM in memory-operand
    # form, AX threaded across steps.
    A, B, C = 0x40, 0x44, 0x48
    seed = {A: 10, B: 3, C: 4}
    prog = [
        isa.Instr(isa.IMM, 7),        # AX = 7
        isa.Instr(ES.ADDM, A),        # AX = mem[A] + AX = 10 + 7 = 17
        isa.Instr(ES.SUBM, B),        # AX = mem[B] - AX = 3 - 17 = -14 & 0xFF = 242
        isa.Instr(ES.MULM, C),        # AX = mem[C] * AX = 4 * 242 = 968 & 0xFF = 200
        isa.Instr(isa.HALT, 0),
    ]
    tr = ES.run_exact_steps(model, L, prog, max_steps=8, seed_mem=seed, mask=0xFF)
    ref = ES.ref_interpret_exact(prog, seed_mem=seed, mask=0xFF)
    assert tr == ref, f"neural {tr} != ref {ref}"


def test_flag_off_complete_model_unchanged():
    """With C4_EXACT_STEPS (and C4_MEM_OPERAND) OFF, the interpreted complete-VM
    model is byte-identical — this module is never on the default build path."""
    import hashlib
    from c4_min import nibble_pure_forward_complete as PFC
    os.environ.pop("C4_EXACT_STEPS", None)
    os.environ.pop("C4_MEM_OPERAND", None)
    model, L = PFC.build_pure_forward_complete_model(code_size=16)
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    # RE-BASELINED 2026-07 for the c4-faithful SHR-arithmetic + signed-LC fix (was
    # ``a9484315fabcb30d`` when SHR was logical / LC unsigned).
    assert h.hexdigest()[:16] == "914a12ffe9db84ab", h.hexdigest()[:16]


if __name__ == "__main__":
    os.environ["C4_EXACT_STEPS"] = "1"
    os.environ["C4_MEM_OPERAND"] = "1"
    print("building exact-steps model ...")
    m8 = ES.build_exact_steps_model(code_size=6)
    test_addm_subm_byte_exact(m8)
    print("  ADDM/SUBM byte-exact: OK")
    test_mulm_byte_exact(m8)
    print("  MULM byte-exact: OK")
    test_divm_modm_byte_exact(m8)
    print("  DIVM/MODM byte-exact: OK")
    test_dot_via_mulm_addm_composes(m8)
    print("  fused arithmetic chain composes: OK")
    test_flag_off_complete_model_unchanged()
    print("  flag-OFF complete model unchanged: OK")
    print("ALL EXACT-STEPS TESTS PASS")
