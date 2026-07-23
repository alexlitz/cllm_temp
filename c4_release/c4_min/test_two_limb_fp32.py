"""TWO-LIMB fp32 32-bit VM verification.

The width-32 value substrate historically ran the whole step in **fp64** because
``compile_nibble_to_scalar`` recomposed each register's 8 nibbles into a single
``16^7`` scalar that overflows fp32's 2^24 integer precision.  The TWO-LIMB path
(``nibble_vm.vm_two_limb``, the default when ``C4_VM_WIDTH32=1``) carries the DATA
registers AX and STACK0 as two fp32-exact limbs (low 4 nibbles / high 4 nibbles)
with a carry/borrow across the 2^16 limb boundary, so the whole model is **fp32**
(no fp64) yet byte-exact to the full 2^32.

These tests assert:
  1. the two-limb build is fp32 (NO fp64 parameter anywhere);
  2. 32-bit ADD/SUB (incl the cross-limb carry/borrow cases) are byte-exact vs
     Python ``(a±b) & 0xFFFFFFFF``;
  3. PSH/POP round-trips a full 32-bit value;
  4. the full base ISA (IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ + a deep countdown loop) is
     byte-exact vs a 32-bit reference interpreter;
  5. the fp64 single-scalar fallback (``C4_VM_TWO_LIMB=0``) still builds fp64.

Run: OMP_NUM_THREADS=4 PYTHONPATH=<repo> C4_VM_WIDTH32=1 \
        python -m pytest c4_min/test_two_limb_fp32.py -q
"""
from __future__ import annotations

import os

import pytest
import torch

from . import isa
from . import nibble_vm as N

WIDTH32 = os.environ.get("C4_VM_WIDTH32", "0") == "1"
_requires_w32 = pytest.mark.skipif(
    not WIDTH32, reason="two-limb fp32 32-bit tests require C4_VM_WIDTH32=1")


# --- a 32-bit reference interpreter (mask 2^32; LEA uses full BP, matching W32) ---
def _interp32(code, mem_size=0x10000):
    MASK = 0xFFFFFFFF
    ax, bp, sp, pc = 0, 0x10000, 0x10000, 0
    stack = [0] * (2 * mem_size + 8)
    emitted = []
    steps = 0

    def push(v):
        nonlocal sp
        sp -= 1
        stack[sp] = v & MASK

    def pop():
        nonlocal sp
        v = stack[sp]
        sp += 1
        return v & MASK

    while pc < len(code) and steps < 200000:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        pc += 1
        if op == isa.IMM:
            ax = imm & MASK
        elif op == isa.LEA:
            ax = (bp + imm) & MASK
        elif op == isa.PSH:
            push(ax)
        elif op == isa.ADD:
            ax = (pop() + ax) & MASK
        elif op == isa.SUB:
            ax = (pop() - ax) & MASK
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.HALT:
            emitted.append(ax)
            break
        else:
            raise NotImplementedError(op)
        emitted.append(ax)
    return emitted


@pytest.fixture(scope="module")
def vm():
    return N.NibbleVM(code_size=12)


def _alu(vm, a, b, op):
    prog = [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)]
    _, frames = vm.run(prog, max_steps=50)
    return N.decode_trace(frames)[-1]


# --- 1. the two-limb build is fp32 (no fp64) --------------------------------
@_requires_w32
def test_two_limb_build_is_fp32(vm):
    assert N.vm_two_limb(), "two-limb should be the width-32 default"
    dtypes = set(p.dtype for p in vm.model.parameters())
    assert torch.float64 not in dtypes, f"fp64 present: {dtypes}"
    assert dtypes == {torch.float32}, dtypes


# --- 2. 32-bit ADD / SUB incl the cross-limb carry / borrow -----------------
ADD_CASES = [
    ((1 << 32) - 1, 1),          # 0xFFFFFFFF + 1 -> 0 (full carry chain)
    (1 << 20, 1 << 20),          # 2^20 + 2^20
    (0xFFFFFFFF, 1),
    (0xDEADBEEF, 0x12345678),    # generic wide + wide
    (0x0000FFFF, 1),             # carry exactly at the 2^16 low-limb boundary
    (0x7FFFFFFF, 0x7FFFFFFF),
    (6, 7), (200, 100), (0, 0),  # small (hi limb = 0)
]
SUB_CASES = [
    (0xFFFFFFFF, 1),
    (0x00100000, 0x00000001),    # borrow across the limb boundary
    (0x00010000, 1),             # borrow exactly at the 2^16 boundary
    (0, 1),                      # 0 - 1 -> 0xFFFFFFFF (full borrow chain)
    (0x12345678, 0xDEADBEEF),    # negative result wraps mod 2^32
    (0x80000000, 1),
    (12, 7), (5, 5), (1000, 2000),
]


@_requires_w32
@pytest.mark.parametrize("a,b", ADD_CASES)
def test_add_32bit_byte_exact(vm, a, b):
    got = _alu(vm, a, b, "ADD")
    assert got == ((a + b) & 0xFFFFFFFF), f"ADD {a:#x}+{b:#x} = {got:#x}"


@_requires_w32
@pytest.mark.parametrize("a,b", SUB_CASES)
def test_sub_32bit_byte_exact(vm, a, b):
    got = _alu(vm, a, b, "SUB")
    assert got == ((a - b) & 0xFFFFFFFF), f"SUB {a:#x}-{b:#x} = {got:#x}"


# --- 3. PSH / POP round-trips a full 32-bit value ---------------------------
@_requires_w32
@pytest.mark.parametrize("v", [0, 255, 0x10000, 0xDEADBEEF, 0xFFFFFFFF,
                               0x7FFFFFFF, 0x12345678, 0x0000FFFF, 0x00010000])
def test_psh_pop_32bit_round_trip(vm, v):
    # IMM v ; PSH ; IMM 0 ; ADD  ==  pop()(=v) + 0  recovers the pushed value.
    assert _alu(vm, v, 0, "ADD") == v


# --- 4. the full base ISA is byte-exact vs the 32-bit reference -------------
BASE_PROGS = {
    "add_13":      [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)],
    "sub_wrap":    [("IMM", 3), ("PSH", 0), ("IMM", 10), ("SUB", 0), ("HALT", 0)],
    "lea":         [("LEA", 3), ("HALT", 0)],
    "lea_200":     [("LEA", 200), ("HALT", 0)],
    "jmp_skip":    [("IMM", 1), ("JMP", 3), ("IMM", 99), ("IMM", 42), ("HALT", 0)],
    "bz_taken":    [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)],
    "bz_nottaken": [("IMM", 5), ("BZ", 4), ("IMM", 9), ("HALT", 0)],
    "bnz_taken":   [("IMM", 5), ("BNZ", 3), ("IMM", 99), ("IMM", 8), ("HALT", 0)],
    "bnz_nottkn":  [("IMM", 0), ("BNZ", 4), ("IMM", 7), ("HALT", 0)],
    "add_big":     [("IMM", 0xDEADBEEF), ("PSH", 0), ("IMM", 0x12345678),
                    ("ADD", 0), ("HALT", 0)],
    "sub_borrow":  [("IMM", 0x00100000), ("PSH", 0), ("IMM", 0x00000001),
                    ("SUB", 0), ("HALT", 0)],
}


@_requires_w32
@pytest.mark.parametrize("name", list(BASE_PROGS))
def test_base_isa_vs_ref32(vm, name):
    prog = BASE_PROGS[name]
    code = isa.assemble(prog)
    _, frames = vm.run(prog, max_steps=200)
    assert N.decode_trace(frames) == _interp32(code), name


# --- 5. a deep 32-bit countdown loop (past 2^8) reaches 0 in the exact count --
@_requires_w32
@pytest.mark.parametrize("n", [3, 255, 256, 300])
def test_countdown_loop_32bit(vm, n):
    prog = [("IMM", n), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]
    _, frames = vm.run(prog, max_steps=n * 6 + 50)
    trace = N.decode_trace(frames)
    assert trace[-1] == 0, f"countdown({n}) ended at {trace[-1]}"
    assert len(frames) == 4 * n + 2, f"countdown({n}) took {len(frames)} steps"


# --- 6. the fp64 single-scalar fallback is intact (kill-switch) -------------
@_requires_w32
def test_fp64_fallback_intact():
    """``C4_VM_TWO_LIMB=0`` (with width-32 on) rebuilds the historical fp64
    single-scalar model.  Verified in-process by toggling the build context."""
    with N.two_limb_mode(False):
        assert not N.vm_two_limb()
        model, L = N.build_step_model(code_size=6)
        assert set(p.dtype for p in model.parameters()) == {torch.float64}
        # a small ADD still works on the fp64 fallback.
        prog = [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]
        _, frames = N.run_program(model, L, isa.assemble(prog), max_steps=50)
        assert N.decode_trace(frames)[-1] == 13


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
