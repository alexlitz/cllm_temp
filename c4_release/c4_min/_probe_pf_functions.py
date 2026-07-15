"""M4 — FUNCTIONS run 100%-in-forward, guard-clean.

Prove the full C4 calling convention executes entirely in ``model.forward`` +
argmax-generate-append, under the no-python-compute guard:

    func_identity(70) -> 70     (JSR return-PC, ENT frame, LEA/LI arg read, LEV)
    func_add(3, 4)    -> 7      (two args read from the KV-memory stack, add, LEV)

The stack IS the softmax1-KV memory: JSR pushes the return PC as a KV store, ENT
pushes the caller BP, the args pushed by the caller are earlier KV stores, LEA
computes a frame-relative byte address (BP + 4*slot) and LI content-addresses the
newest write at that address, LEV loads saved-BP and return-PC back out of the KV
log.  No _apply_op / DictMemStack / per-call gadget is ever entered.

C4 frame after ENT n (SP/BP are byte addresses, 4 bytes/slot, stack descends):

    caller pushes arg_k ... arg_1 (in that order) then  JSR fn
      => MEM[BP+8]   = arg_1        (first pushed = highest addr above ret-PC)
         MEM[BP+12]  = arg_2
         ...
      MEM[BP+4] = return-PC   (pushed by JSR)
      MEM[BP]   = saved caller BP  (pushed by ENT)
      MEM[BP-4] = local_1  ...      (reserved by ENT n)

  so inside the callee LEA 2 -> &MEM[BP+8] = &arg_1, LEA 3 -> &arg_2.
"""
from __future__ import annotations
import sys, time

import c4_min.nibble_pure_forward as PF
import c4_min.nibble_pure_forward_complete as C
# small stack base so frame-relative LEA (AX is 8-bit) reaches the frame + args.
PF.SP_INIT = 0xF0
C.SP_INIT = 0xF0

from c4_min import isa
from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete, ref_interpret)
from c4_min.nibble_pure_forward import assert_no_python_compute


def prog_identity(x: int):
    """int identity(int a){ return a; }  main(){ return identity(x); }

    [0] IMM x        ; push the argument
    [1] PSH
    [2] JSR 5        ; call identity  (pushes return-PC = 3)
    [3] ADJ 1        ; caller pops the 1 arg
    [4] HALT         ; AX = identity(x)
    -- identity:
    [5] ENT 0        ; frame, no locals
    [6] LEA 2        ; &arg_1 = BP+8
    [7] LI 0         ; AX = arg_1
    [8] LEV          ; return AX
    """
    return [("IMM", x), ("PSH", 0), ("JSR", 5), ("ADJ", 1), ("HALT", 0),
            ("ENT", 0), ("LEA", 2), ("LI", 0), ("LEV", 0)]


def prog_add(a: int, b: int):
    """int add(int a,int b){ return a+b; }  main(){ return add(a,b); }

    Caller pushes b then a  (so arg_1=a is at BP+8, arg_2=b at BP+12).
    [0] IMM b
    [1] PSH
    [2] IMM a
    [3] PSH
    [4] JSR 8        ; pushes return-PC = 5
    [5] ADJ 2        ; pop 2 args
    [6] HALT
    -- add:
    [7] ENT 0
    [8] LEA 3        ; &arg_2 = BP+12  -> AX
    [9] LI 0         ; AX = arg_2 = b
    [10] PSH          ; push b
    [11] LEA 2        ; &arg_1 = BP+8  -> AX
    [12] LI 0         ; AX = arg_1 = a
    [13] ADD 0        ; AX = b + a
    [14] LEV
    """
    return [("IMM", b), ("PSH", 0), ("IMM", a), ("PSH", 0), ("JSR", 7),
            ("ADJ", 2), ("HALT", 0),
            ("ENT", 0), ("LEA", 3), ("LI", 0), ("PSH", 0),
            ("LEA", 2), ("LI", 0), ("ADD", 0), ("LEV", 0)]


def main():
    t = time.time()
    m, L = build_pure_forward_complete_model(code_size=24, include_bitwise=True,
                                             include_divmod=False)
    print(f"LEAN complete model: dim={L.D} blocks={len(m.blocks)} "
          f"heads={m.blocks[0].attn.n_heads} ({time.time()-t:.1f}s)\n", flush=True)

    cases = [
        ("func_identity(70)", prog_identity(70), 70),
        ("func_add(3,4)",     prog_add(3, 4),    7),
        # a couple more to show it is not a fluke of the constants:
        ("func_identity(42)", prog_identity(42), 42),
        ("func_add(9,5)",     prog_add(9, 5),    14),
    ]
    npass = 0
    for name, prog, want in cases:
        code = isa.assemble(prog)
        ref = ref_interpret(code)
        try:
            got = assert_no_python_compute(run_pure_forward_complete, m, L, code,
                                           max_steps=64, verbose=False)
            guard = "CLEAN"
        except AssertionError as e:
            got = None
            guard = f"LEAK: {e}"
        final = got[-1] if got else None
        ok = (final == want) and (got == ref) and guard == "CLEAN"
        npass += int(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s} got={final} want={want} "
              f"ref_final={ref[-1] if ref else None} guard={guard}", flush=True)
        if got != ref:
            print(f"        got_trace={got}\n        ref_trace={ref}", flush=True)
    print(f"\n=== FUNCTIONS pure-forward: {npass}/{len(cases)} guard-clean byte-exact ===",
          flush=True)
    return 0 if npass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
