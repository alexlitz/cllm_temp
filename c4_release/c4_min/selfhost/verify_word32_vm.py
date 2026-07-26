#!/usr/bin/env python3
"""verify_word32_vm.py — Task 1 gate for the 32-bit-memory draft VM.

Three claims, all end-to-end through the REAL c4 toolchain
(``compile_c`` -> ``bytecode_to_isa`` -> the VM):

  (A) BYTE-IDENTICAL to the byte VM (``ref_interpret``) on fits-a-byte cases
      (<=255) — value AND step count.

  (B) The word32 VM keeps the FULL 32-bit value where the byte VM TRUNCATES.
      A program sums 10 copies of 100 (=1000, > 255), stores it (SI), loads it
      back (LI), and extracts the HIGH byte ``(v >> 8) & 0xFF``.  The byte VM
      stored ``1000 & 0xFF = 232`` so its high byte is 0; the word32 VM stored
      the full 1000 so its high byte is 3.  This DISCRIMINATES the two VMs
      (the PRTF low byte alone does not).

  (C) The word32 VM matches numpy 32-bit on a real fixed-point dot whose
      accumulator overflows a byte (the STORED accumulator high bytes are
      checked, not just the emitted low byte).

The step count is DATA-INDEPENDENT and IDENTICAL between the two VMs (masking is
value-only) — so the grounding reuses the byte VM's measured steps/MAC with the
word32 VM's CORRECT values.

Run:  python -m c4_min.selfhost.verify_word32_vm
"""
from __future__ import annotations

import sys

from c4_min import isa
from c4_min.selfhost.word32_draft_vm import ref_interpret_word32, WORD
from c4_min.selfhost._matmul_paged_src import paged_dot_c, _compile


def _byte_trace(code, max_steps=20_000_000):
    """The existing BYTE draft VM (SI/LI & 0xFF).  Returns (out, trace)."""
    from c4_min.nibble_pure_forward_complete import ref_interpret
    out = []
    tr = ref_interpret(code, max_steps=max_steps, mask=0xFFFFFFFF, out=out)
    return out, tr


def _numpy_paged_dot(w, x, scale=16):
    """32-bit reference for the paged fixed-point dot (acc wraps mod 2^32)."""
    acc = 0
    for i in range(len(w)):
        acc = (acc + (w[i] * scale * x[i] * scale) // scale) & WORD
    return acc & WORD


def main() -> int:
    ok = True

    # ------------------------------------------------------------------ #
    # (A) byte-identical on FITS-A-BYTE cases                             #
    # ------------------------------------------------------------------ #
    print("=" * 74)
    print("(A) word32 VM == byte VM on fits-a-byte cases (values <= 255)")
    print("=" * 74)
    from c4_min.nibble_pure_forward_complete import ref_interpret
    for w, x in [([1, 0, 1], [2, 0, 1]), ([1, 1], [1, 1]), ([0, 1, 0, 1], [3, 1, 2, 1])]:
        code = _compile(paged_dot_c(w, x))
        w32_trace, w32_n = ref_interpret_word32(code, out=(o32 := []))
        b_out = []
        b_trace = ref_interpret(code, max_steps=20_000_000, mask=0xFFFFFFFF, out=b_out)
        b_n = len(b_trace)
        eq_out, eq_steps = (o32 == b_out), (w32_n == b_n)
        eq_trace = (w32_trace == b_trace)          # full per-step AX trace
        print(f"  w={w} x={x}: word32_emit={o32} byte_emit={b_out}  "
              f"trace_eq={eq_trace}  steps word32={w32_n} byte={b_n}  "
              f"{'OK' if (eq_out and eq_steps and eq_trace) else 'MISMATCH'}")
        ok = ok and eq_out and eq_steps and eq_trace

    # ------------------------------------------------------------------ #
    # (B) DISCRIMINATING test: high byte of a stored value > 255          #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 74)
    print("(B) word32 keeps the HIGH byte the byte VM truncates (store 1000, load, >>8)")
    print("=" * 74)
    # C: sum 100 ten times -> 1000; store to a local (SI); load it back (LI);
    # emit high byte (acc / 256) and low byte (acc - hi*256).  1000 = 3*256 + 232.
    src = """
int main() {
  int acc, k, hi, lo, v;
  acc = 0; k = 0;
  while (k < 10) { acc = acc + 100; k = k + 1; }
  v = acc;                 /* SI: store 1000 (byte VM truncates to 232) */
  lo = v - (v / 256) * 256;
  hi = v / 256;
  printf(hi);              /* word32: 3   byte VM: 0 */
  printf(lo);              /* word32: 232 byte VM: 232 */
  return 0;
}
"""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, _ = compile_c(src)
    code = bytecode_to_isa(bc)
    w32_trace, w32_n = ref_interpret_word32(code, out=(o32 := []))
    b_out = []
    ref_interpret(code, max_steps=20_000_000, mask=0xFFFFFFFF, out=b_out)
    # word32 must emit [3, 232]; byte VM emits [0, 232] (1000 truncated to 232 on store)
    disc = (o32 == [3, 232]) and (b_out == [0, 232])
    print(f"  word32 emit (hi, lo) = {o32}  (expect [3, 232])")
    print(f"  byte VM emit (hi, lo) = {b_out}  (expect [0, 232] — 1000 truncated to 232)")
    print(f"  DISCRIMINATES the two VMs: {disc}  steps={w32_n} (identical both VMs)")
    ok = ok and disc

    # ------------------------------------------------------------------ #
    # (C) CORRECT vs numpy 32-bit on a real dot (full stored acc checked) #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 74)
    print("(C) word32 == numpy 32-bit on a fixed-point dot whose acc overflows a byte")
    print("=" * 74)
    import numpy as np
    rng = np.random.RandomState(0)
    for K in [8, 16, 64, 104]:
        w = [int(v) for v in rng.randint(1, 6, size=K)]
        x = [int(v) for v in rng.randint(1, 6, size=K)]
        # emit the acc's high and low byte so the FULL value is validated.
        code = _emit_hilo_dot(w, x)
        w32_trace, w32_n = ref_interpret_word32(code, out=(o32 := []))
        ref = _numpy_paged_dot(w, x)
        ref_hi, ref_lo = (ref >> 8) & 0xFF, ref & 0xFF
        correct = (o32 == [ref_hi, ref_lo])
        b_out = []
        ref_interpret(code, max_steps=20_000_000, mask=0xFFFFFFFF, out=b_out)
        byte_wrong = (b_out != [ref_hi, ref_lo])
        print(f"  K={K:>3}: numpy32 acc={ref} (hi={ref_hi} lo={ref_lo})  "
              f"word32 emit={o32}  byte VM emit={b_out}  steps={w32_n}  "
              f"{'OK' if correct else 'MISMATCH'}"
              f"{'  [byteVM wrong]' if byte_wrong else ''}")
        ok = ok and correct

    print()
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _emit_hilo_dot(w, x, scale=16):
    """Compile a fixed-point dot that stores its 32-bit acc and emits (hi, lo)
    bytes — so the FULL value is validated, not just the truncatable low byte.
    Uses the paged kernel body then appends the hi/lo extraction."""
    # build the paged-dot C but replace the trailing printf(acc) with hi/lo emit.
    body = paged_dot_c(w, x, scale=scale)
    # paged_dot_c ends with:  printf(acc);\n  return 0;\n}
    assert "printf(acc);" in body
    emit = ("hi = acc / 256; lo = acc - hi * 256; printf(hi); printf(lo);")
    body = body.replace("printf(acc);", emit)
    # declare hi, lo in the header (append to the first int decl line)
    body = body.replace("int s; int acc, k;", "int s; int acc, k; int hi; int lo;")
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, _ = compile_c(body)
    return bytecode_to_isa(bc)


if __name__ == "__main__":
    sys.exit(main())
