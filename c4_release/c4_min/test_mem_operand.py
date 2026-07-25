"""test_mem_operand.py — the MEMORY-OPERAND MAC (C4_MEM_OPERAND) byte-exact proof.

Proves the fused ``MAC [addr_a],[addr_b]`` opcode (``acc = acc + mem[a]*mem[b]``,
one VM step / one model.forward) is byte-exact vs numpy fixed-point and vs the
extended SP-addressed reference, and that it composes into a dot product with the
accumulator round-tripping through AX.  Also asserts the interpreted complete-VM
model + golden are unchanged with the flag OFF (this module is a separate file
never imported on the default build path).

Run:  C4_MEM_OPERAND=1 python -m pytest c4_min/test_mem_operand.py -q
  or: C4_MEM_OPERAND=1 python c4_min/test_mem_operand.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min import isa
from c4_min import nibble_mem_operand as MO

MAC = MO.MAC


def _single_mac(model, L, a, b, acc=0, mask=0xFF):
    """acc = acc + mem[A]*mem[B] via ONE MAC step; return the model-decoded AX."""
    A_ADDR, B_ADDR = 0x40, 0x44
    seed = {A_ADDR: a, B_ADDR: b}
    prog = [isa.Instr(MAC, A_ADDR), isa.Instr(isa.HALT, 0)]
    # seed the accumulator into AX by prepending an IMM if acc != 0; for these tests
    # acc starts at 0 (AX init), so a lone MAC produces a*b.
    tr = MO.run_mem_operand(model, L, prog, {0: B_ADDR}, max_steps=6,
                            seed_mem=seed, mask=mask)
    return tr[-1]


@pytest.fixture(scope="module")
def model8():
    os.environ["C4_MEM_OPERAND"] = "1"
    return MO.build_mem_operand_model(code_size=4)


def test_single_mac_byte_exact(model8):
    model, L = model8
    cases = [(0, 0), (7, 6), (255, 255), (0, 200), (200, 0), (1, 255),
             (16, 16), (15, 15), (128, 2), (100, 100), (17, 13), (250, 3)]
    fails = []
    for a, b in cases:
        got = _single_mac(model, L, a, b, mask=0xFF)
        exp = (a * b) & 0xFF
        if got != exp:
            fails.append((a, b, got, exp))
    assert not fails, f"single-MAC byte mismatches: {fails}"


def test_dot_product_composes(model8):
    """A dot product = a chain of MACs; the accumulator round-trips through AX."""
    model, L = model8
    avec = [3, 5, 2, 7]
    bvec = [4, 1, 6, 2]
    seed = {}
    for i, v in enumerate(avec):
        seed[0x40 + 4 * i] = v
    for i, v in enumerate(bvec):
        seed[0x60 + 4 * i] = v
    prog, mac_b = [], {}
    for i in range(4):
        mac_b[len(prog)] = 0x60 + 4 * i
        prog.append(isa.Instr(MAC, 0x40 + 4 * i))
    prog.append(isa.Instr(isa.HALT, 0))
    tr = MO.run_mem_operand(model, L, prog, mac_b, max_steps=16, seed_mem=seed, mask=0xFF)
    ref = MO.ref_interpret_mac(prog, mac_b, seed_mem=seed, mask=0xFF)
    np_dot = int(np.dot(avec, bvec)) & 0xFF
    assert tr == ref, f"neural {tr} != ref {ref}"
    assert tr[-1] == np_dot, f"final {tr[-1]} != numpy dot {np_dot}"


def test_dot_product_32bit():
    """Full 32-bit fixed-point dot product (products/partial sums fit 32 bits)."""
    os.environ["C4_MEM_OPERAND"] = "1"
    MASK = 0xFFFFFFFF
    avec = [123, 4567, 89, 1000]
    bvec = [7, 12, 300, 42]
    seed = {}
    for i, v in enumerate(avec):
        seed[0x40 + 4 * i] = v
    for i, v in enumerate(bvec):
        seed[0x60 + 4 * i] = v
    prog, mac_b = [], {}
    for i in range(4):
        mac_b[len(prog)] = 0x60 + 4 * i
        prog.append(isa.Instr(MAC, 0x40 + 4 * i))
    prog.append(isa.Instr(isa.HALT, 0))
    model, L = MO.build_mem_operand_model(code_size=len(prog))
    tr = MO.run_mem_operand(model, L, prog, mac_b, max_steps=16, seed_mem=seed, mask=MASK)
    ref = MO.ref_interpret_mac(prog, mac_b, seed_mem=seed, mask=MASK)
    np_dot = int(np.dot(np.array(avec, dtype=np.int64),
                        np.array(bvec, dtype=np.int64))) & MASK
    assert tr == ref
    assert tr[-1] == np_dot, f"final {tr[-1]} != numpy 32-bit dot {np_dot}"


def test_flag_off_complete_model_unchanged():
    """With C4_MEM_OPERAND OFF, the interpreted complete-VM model is byte-identical
    (this module is a separate file, never on the default build path)."""
    import hashlib
    from c4_min import nibble_pure_forward_complete as PFC
    os.environ.pop("C4_MEM_OPERAND", None)
    model, L = PFC.build_pure_forward_complete_model(code_size=16)
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    # the flag-OFF complete-model hash (captured at build); MAC is additive & gated.
    assert h.hexdigest()[:16] == "a9484315fabcb30d"


if __name__ == "__main__":
    os.environ["C4_MEM_OPERAND"] = "1"
    print("building 8-bit mem-operand model ...")
    m8 = MO.build_mem_operand_model(code_size=4)
    test_single_mac_byte_exact(m8)
    print("  single-MAC byte-exact: OK")
    test_dot_product_composes(m8)
    print("  dot-product composition: OK")
    print("building 32-bit dot model ...")
    test_dot_product_32bit()
    print("  32-bit dot-product byte-exact: OK")
    test_flag_off_complete_model_unchanged()
    print("  flag-OFF complete model unchanged: OK")
    print("ALL MEM-OPERAND TESTS PASS")
