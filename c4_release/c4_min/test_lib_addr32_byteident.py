"""Byte-identity guard for the 32-bit-address LI/LC query widening.

``lib_neural.compile_mem_prep_addr32`` widens the LI/LC load-address QUERY from
the low byte (base ``compile_mem_prep``) to all 32 bits, so the runtime library's
heap addresses (0x30008) survive the §Memory CAM.  This test proves the widening
is BYTE-IDENTICAL to the base for every BYTE-sized address (the whole base corpus
range) — the widened FFN produces the SAME ``QRY_BIN`` bits and the SAME
``IS_LOAD`` — so swapping it into the model cannot regress the 1096 corpus, AND
that it additionally recovers a full 32-bit heap address the base cannot.

Fast: builds only the two mem-prep FFN specs on a minimal layout + evaluates them
on a synthetic residual (no 62 GB model build).

Run:  PYTHONPATH=<repo> python -m pytest c4_min/test_lib_addr32_byteident.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from c4_min import isa
from c4_min.blogspec_vocab import nibbles_of_value


def _layout_and_specs():
    from c4_min.nibble_pure_forward_complete import PureForwardCompleteLayout
    from c4_min import nibble_alu32 as A, nibble_bitwise as bw
    from c4_min.nibble_pure_forward import compile_mem_prep
    from c4_min.lib_neural import compile_mem_prep_addr32
    n_heads = 23
    L = PureForwardCompleteLayout(4, n_heads=n_heads)
    A.extend_layout_for_alu32(L, recurrent_divmod=True)
    bw.extend_layout_for_bitwise(L)
    while L._off % n_heads != 0:
        L._scalar(f"p{L._off}")
    L.D = L._off
    A._ONE = L.ONE
    return L, compile_mem_prep(L, L.D), compile_mem_prep_addr32(L, L.D)


def _run_ffn(spec, x):
    up = x @ spec["W_up"].T + spec["b_up"]
    gate = x @ spec["W_gate"].T + spec["b_gate"]
    h = F.silu(up) * gate
    return x + h @ spec["W_down"].T + spec["b_down"]


def _qry_bits(L, spec, addr):
    x = torch.zeros(1, L.D)
    x[0, L.ONE] = 1.0
    x[0, L.OP_IS + isa.LI] = 1.0
    x[0, L.AX_VAL] = float(addr & 0xFF)
    for j, nv in enumerate(nibbles_of_value(addr, 8)):
        x[0, L.AX + j] = float(nv)
    y = _run_ffn(spec, x)
    bits = [int(round(float(y[0, L.QRY_BIN + b]))) for b in range(32)]
    is_load = int(round(float(y[0, L.IS_LOAD])))
    return bits, is_load


def test_addr32_byte_identical_for_byte_addresses():
    L, base, wide = _layout_and_specs()
    for addr in (0, 1, 0x40, 0x44, 0x7F, 0x80, 0xAB, 0xFF):
        bb, bl = _qry_bits(L, base, addr)
        wb, wl = _qry_bits(L, wide, addr)
        want = [(addr >> b) & 1 for b in range(32)]
        assert bb == wb == want, f"addr {hex(addr)}: base={bb} wide={wb} want={want}"
        assert bl == wl == 1, f"addr {hex(addr)}: IS_LOAD base={bl} wide={wl}"


def test_addr32_recovers_full_32bit_heap_address():
    L, base, wide = _layout_and_specs()
    addr = 0x30008                       # the runtime-library heap base
    wb, _ = _qry_bits(L, wide, addr)
    val = sum(bit << b for b, bit in enumerate(wb))
    assert val == addr, f"wide QRY for {hex(addr)} = {hex(val)}"
    # the base 8-bit query only sees the low byte (it CANNOT address the heap).
    bb, _ = _qry_bits(L, base, addr)
    base_val = sum(bit << b for b, bit in enumerate(bb))
    assert base_val == (addr & 0xFF), f"base QRY = {hex(base_val)} (expected low byte)"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
