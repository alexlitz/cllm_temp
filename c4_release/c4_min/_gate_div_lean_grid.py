"""GPU gate: DIV/MOD byte-exact grid for the radix-16 divide variant selected by
``C4_DIV_LEAN`` (lean 80-block vs hardened 88-block), run through the REAL fused
Qwen VM forward.

Reuses #735's grid approach: 8-bit IMM operands (dividends over byte edges +
spread, divisors incl b=0), asserts BOTH quotient AND remainder decoded from the
AX nibble band match ``nibble_muldivmod`` (== ISA semantics: b==0 -> q=0, r=0)
for every case.  Prints the pass count (target 288/288).

Run:  CUDA_VISIBLE_DEVICES=<n> C4_DIV_LEAN=<0|1> python -m c4_min._gate_div_lean_grid
"""
from __future__ import annotations

import os

import torch

import c4_min.qwen_full_vm as Q
from c4_min import isa
from c4_min.nibble_muldivmod import divmod32, mod32


# 24 dividends (byte edges {0,1,15,16,17,255} + spread) x 12 divisors = 288 cases.
DIVIDENDS = [0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 100,
             127, 128, 129, 170, 200, 240, 250, 253, 254, 255, 85, 51]
DIVISORS = [0, 2, 3, 7, 11, 13, 17, 4, 8, 16, 32, 128]


def _ref(a: int, b: int):
    """VM DIV/MOD semantics: q = a//b, r = a%b, with b==0 -> (0, 0)."""
    return divmod32(a, b)[0], mod32(a, b)


def main():
    # Resolve the variant via the PRODUCTION selectors (not a raw env read), so the
    # printed label always matches the model that actually builds — the default is
    # lean (``_div_lean()`` defaults ON), so an UNSET env means lean, not hardened.
    longdiv = Q._div_longdiv()
    lean = Q._div_lean()
    variant = ("LONGDIV(~262)" if longdiv else ("LEAN(80)" if lean else "HARDENED(88)"))
    print(f"C4_DIV_LEAN={os.environ.get('C4_DIV_LEAN','<unset->lean>')} "
          f"-> radix-16 variant: {variant}")

    vm = Q.build(code_size=16, subset=Q.SUBSET_MULDIV, efficient_alu=True,
                 recurrent_divmod=True)
    # Move the built model + the token->residual embed table onto the GPU so the
    # per-(a,b) forwards run on-device (x lands on cuda because it is gathered from
    # vm.embed).  The scalar read-back paths use float(...) so they cross devices.
    if torch.cuda.is_available():
        vm.qmodel = vm.qmodel.cuda().eval()
        vm.embed = vm.embed.cuda()
        print(f"device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("device: cpu (no CUDA visible)")
    # confirm which divmod builder actually assembled
    names = [n for n, _ in Q._block_specs(vm.QL.L, 24, Q.SUBSET_MULDIV,
                                          efficient_alu=True, recurrent_divmod=True)]
    n_lean_blocks = len([n for n in names if n.startswith("lean-")])
    print(f"assembled unique divmod blocks (lean-*): {n_lean_blocks}  "
          f"(applied depth: hardened 88 / lean 80)")

    cases = [(a, b) for a in DIVIDENDS for b in DIVISORS]
    total = len(cases)
    passed = 0
    fails = []
    for a, b in cases:
        rq, rr = _ref(a, b)
        # DIV: read AX (32-bit-decoded) at the op step (index 3: IMM;PSH;IMM;OP)
        prog_div = isa.assemble([("IMM", a), ("PSH", 0), ("IMM", b),
                                 ("DIV", 0), ("HALT", 0)])
        rd = Q.run_program(vm, prog_div, max_steps=32, mask=0xFFFFFFFF)
        gq = rd["ax_trace"][3]
        prog_mod = isa.assemble([("IMM", a), ("PSH", 0), ("IMM", b),
                                 ("MOD", 0), ("HALT", 0)])
        rm = Q.run_program(vm, prog_mod, max_steps=32, mask=0xFFFFFFFF)
        gr = rm["ax_trace"][3]
        if (gq, gr) == (rq, rr):
            passed += 1
        else:
            if len(fails) < 40:
                fails.append((a, b, (gq, gr), (rq, rr)))
    print(f"DIV/MOD byte-exact grid: {passed}/{total}  "
          f"({'ALL PASS' if passed == total else 'MISMATCH'})")
    if fails:
        print("  first fails (a, b, got(q,r), exp(q,r)):")
        for f in fails:
            print("   ", f)
    return passed, total


if __name__ == "__main__":
    main()
