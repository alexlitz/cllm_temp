#!/usr/bin/env python3
"""Verify the EFFICIENT-ALU MUL/DIV/MOD bake through the genuine Qwen2Model.forward.

Builds the fused Qwen VM with ``efficient_alu=True`` (nibble_alu32 fp32 gadgets, NOT
the 256x256 lookup table) and runs a MUL/DIV/MOD battery + the mandelbrot z=z^2+c
inner loop, asserting argmax-exactness two ways:

  * 8-bit vs ``isa.interpret`` (the 8-bit reference used across c4_min): mask=0xFF.
  * 32-bit vs ``nibble_muldivmod`` (mul32/divmod32/mod32): mask=0xFFFFFFFF — the
    efficient ALU's genuine 32-bit-exact result (600 for 200*3, 256 for 16*16), which
    the 8-bit interpreter cannot represent (it masks them to 88 / 0).

Memory-safe, CPU-only.

Run:  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$(pwd) \
        python -m c4_min._mem_guard 12 c4_min._verify_qwen_efficient_alu [--recurrent]
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min.nibble_muldivmod import mul32, divmod32, mod32


def _bin(op, a, b):
    return [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)]


# MUL/DIV/MOD battery over VARIED operands (values in the corpus 8-bit operand-load
# range; the ALU gadget itself is 32-bit-exact — see the gadget test for 1e9/7 etc).
BATTERY = [
    ("MUL", 6, 7), ("MUL", 12, 12), ("MUL", 15, 17), ("MUL", 200, 3),
    ("MUL", 0, 5), ("MUL", 1, 255), ("MUL", 16, 16), ("MUL", 100, 2),
    ("DIV", 84, 7), ("DIV", 100, 7), ("DIV", 9, 4), ("DIV", 255, 16),
    ("DIV", 41, 42), ("DIV", 42, 42), ("DIV", 5, 0), ("DIV", 200, 3),
    ("MOD", 84, 5), ("MOD", 100, 7), ("MOD", 9, 4), ("MOD", 255, 16),
    ("MOD", 41, 42), ("MOD", 42, 42), ("MOD", 5, 0), ("MOD", 200, 3),
]

_REF32 = {"MUL": lambda a, b: mul32(a, b), "DIV": lambda a, b: divmod32(a, b)[0],
          "MOD": lambda a, b: mod32(a, b)}


def mandelbrot_inner_loop_progs():
    """The mandelbrot z=z^2+c inner-loop arithmetic (zx' = zx*zx - zy*zy + cx;
    zy' = 2*zx*zy + cy), exercised as MUL sub-terms + the combining ADD/SUB through
    the fused Qwen VM.  z=z^2+c chains the efficient MUL then ADD/SUB."""
    cases = []
    for (zx, zy, cx) in [(2, 1, 1), (3, 2, 0), (1, 3, 2), (4, 1, 3)]:
        zx2 = zx * zx
        cases.append((f"mand_zx2_{zx}", _bin("MUL", zx, zx), zx2))
        zy2 = zy * zy
        cases.append((f"mand_zy2_{zy}", _bin("MUL", zy, zy), zy2))
        cases.append((f"mand_zxzy_{zx}_{zy}", _bin("MUL", zx, zy), zx * zy))
        if zx2 >= zy2:
            prog = [("IMM", zx2), ("PSH", 0), ("IMM", zy2), ("SUB", 0),
                    ("PSH", 0), ("IMM", cx), ("ADD", 0), ("HALT", 0)]
            cases.append((f"mand_zxp_{zx}_{zy}_{cx}", prog, zx2 - zy2 + cx))
    return cases


def main():
    recurrent = "--recurrent" in sys.argv
    print(f"building efficient-ALU Qwen VM (recurrent_divmod={recurrent}) ...", flush=True)
    vm = Q.build(code_size=24, subset=Q.SUBSET_MULDIV, efficient_alu=True,
                 recurrent_divmod=recurrent)
    from transformers.models.qwen2 import Qwen2Model
    assert isinstance(vm.qmodel, Qwen2Model), "not a genuine Qwen2Model!"
    print(f"  Qwen2Model: hidden={vm.hidden_size} intermediate={vm.intermediate_size} "
          f"stored_layers={vm.n_layers} applied_layers={vm.n_applied} "
          f"heads={vm.qmodel.config.num_attention_heads}", flush=True)
    print(f"  efficient_alu={vm.efficient_alu} (NO lookup table — "
          f"intermediate {vm.intermediate_size} vs lookup ~160465)", flush=True)

    # -- 8-bit vs isa.interpret AND 32-bit vs nibble_muldivmod (one run @ 32-bit;
    #    the 8-bit trace is the 32-bit trace masked, so a single forward pass).
    #    The FULL 32-bit ALU result lives in the AX nibble band ONLY at the ALU-op
    #    step (the scalar AX_VAL read by a trailing HALT is folded mod 256), so the
    #    32-bit result is read at the op STEP (index 3 in the IMM;PSH;IMM;OP shape). --
    _OP_STEP = 3
    pass8 = pass32 = 0
    for op, a, b in BATTERY:
        prog = isa.assemble(_bin(op, a, b))
        r32 = Q.run_program(vm, prog, max_steps=48, mask=0xFFFFFFFF)
        got32 = r32["ax_trace"][_OP_STEP]       # the ALU-op step's 32-bit result
        trace8 = [v & 0xFF for v in r32["ax_trace"]]
        ref8 = r32["ref_trace"]
        ref32 = _REF32[op](a, b)
        ok8 = (trace8 == ref8)                  # vs isa.interpret (8-bit)
        ok32 = (got32 == ref32)                 # vs nibble_muldivmod (32-bit)
        pass8 += ok8
        pass32 += ok32
        wide = "  <32-bit>" if ref32 > 0xFF else ""
        print(f"[{'OK ' if (ok8 and ok32) else 'FAIL'}] {op} {a},{b}: "
              f"8bit got={got32 & 0xFF} ref={ref8[_OP_STEP]} | "
              f"32bit got={got32} ref={ref32}{wide}", flush=True)
    print(f"\nBATTERY 8-bit vs isa.interpret : {pass8}/{len(BATTERY)}", flush=True)
    print(f"BATTERY 32-bit vs nibble_muldivmod: {pass32}/{len(BATTERY)}", flush=True)

    print("\nmandelbrot z=z^2+c inner-loop arithmetic:", flush=True)
    mcases = mandelbrot_inner_loop_progs()
    mpass = 0
    for name, prog, expected in mcases:
        r = Q.run_program(vm, isa.assemble(prog), max_steps=48, mask=0xFFFFFFFF)
        got = r["ax_trace"][-1] if r["ax_trace"] else None
        ok = (got == expected)
        mpass += ok
        print(f"[{'OK ' if ok else 'FAIL'}] {name:22s} got={got} want={expected}", flush=True)
    print(f"\nMANDELBROT-INNER: {mpass}/{len(mcases)} exact", flush=True)

    ok_all = (pass8 == len(BATTERY) and pass32 == len(BATTERY) and mpass == len(mcases))
    print(f"\nRESULT: {'ALL PASS' if ok_all else 'SOME FAIL'} "
          f"(8bit {pass8}/{len(BATTERY)}, 32bit {pass32}/{len(BATTERY)}, "
          f"mandelbrot {mpass}/{len(mcases)}) through genuine Qwen2Model.forward")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
