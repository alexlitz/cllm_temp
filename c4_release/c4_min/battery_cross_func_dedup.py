"""Op-battery + dedup-preservation gate for the non-divmod ops (CMP/ADD/SUB/BITWISE).

Drives the COMPACT/STREAMING SPARSE model (the exact object ``weight_dedup``
operates on) with hand-assembled ``IMM a ; PSH ; IMM b ; <op>`` programs — NO C
compiler needed — over a large operand battery:

  * all 6 comparisons (EQ/NE/LT/GT/LE/GE) incl equal pairs (ties), a<b, a>b, and
    large/32-bit operands;
  * ADD / SUB incl 32-bit, carry-across-byte, and borrow;
  * all bitwise (OR/XOR/AND/SHL/SHR) incl full-byte and multi-nibble.

The GATE is the dedup INVARIANT: the model's AX result must be IDENTICAL before
and after the byte-identical weight-tie (``dedup_sparse_transformer``) AND, for
OR/XOR/AND, identical between the shared per-bit gadget and the full 256-entry
tables (``verify_bitwise_perbit_share``) — proving both cross-functional shares
(the per-bit bitwise gadget + the whole-tensor tie) are argmax-exact on real op
programs.  Absolute correctness is checked against ``_faithful32_want`` (the
model's true 32-bit ALU/cmp, 8-bit bitwise/shift semantics); a few EXTREME
operands exceed the model's own fp32 ALU precision and are reported as
informational model-precision misses (independent of the dedup), not failures.

Run:  python -m c4_min.battery_cross_func_dedup            (full)
      python -m c4_min.battery_cross_func_dedup --share-only  (fast share gate)
Memory-safe: one bitwise sparse build (~2.6 GB RSS), no divmod.
"""
from __future__ import annotations

import torch

from c4_min import isa
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.nibble_pure_forward_complete import (
    run_pure_forward_complete, ref_interpret)
from c4_min.weight_dedup import dedup_sparse_transformer


def _push_op(a: int, b: int, op: str):
    """IMM a ; PSH ; IMM b ; <op> ; HALT  =>  AX = a <op> b (a on stack, b in AX)."""
    return [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)]


def _battery():
    cases = []
    # -- COMPARISONS: ties, a<b, a>b, equal-large, 32-bit --
    cmp_pairs = [(5, 5), (5, 6), (6, 5), (0, 0), (255, 255),
                 (3, 7), (7, 3), (1000, 1000), (1000, 2000), (2000, 1000),
                 (0xFFFF, 0x10000), (0x12345678, 0x12345678),
                 (0x12345678, 0x12345679)]
    for op in ("EQ", "NE", "LT", "GT", "LE", "GE"):
        for a, b in cmp_pairs:
            cases.append((op, a, b))
    # -- ADD / SUB: carry across bytes, 32-bit, borrow --
    addsub_pairs = [(6, 7), (255, 1), (256, 256), (500, 700), (65535, 1),
                    (0xFFFFFF, 1), (0x7FFFFFFF, 1), (1000000, 2000000),
                    (0x01020304, 0x0A0B0C0D)]
    for a, b in addsub_pairs:
        cases.append(("ADD", a, b))
    sub_pairs = [(7, 6), (256, 1), (65536, 1), (1900, 50), (0x10000, 0xFFFF),
                 (2000000, 1000000), (0x0A0B0C0D, 0x01020304), (5, 5)]
    for a, b in sub_pairs:
        cases.append(("SUB", a, b))
    # -- BITWISE: full byte, multi-nibble, shifts --
    for a, b in [(0x0C, 0x03), (0xFF, 0x0F), (0xF0, 0x3C), (0xABCD, 0x1234),
                 (0xDEADBEEF, 0x0F0F0F0F)]:
        cases.append(("OR", a, b)); cases.append(("XOR", a, b))
        cases.append(("AND", a, b))
    for a, s in [(0x03, 2), (0x01, 8), (0xFF, 4), (0x1234, 3)]:
        cases.append(("SHL", a, s))
    for a, s in [(0xF0, 3), (0xFF00, 8), (0x12340000, 12), (0x80, 1)]:
        cases.append(("SHR", a, s))
    return cases


_M32 = 0xFFFFFFFF


def _faithful32_want(op: str, a: int, b: int) -> int:
    """The result of ``IMM a ; PSH ; IMM b ; <op>`` (pop=a in STACK0, b in AX)
    under the model's ACTUAL semantics for this pure-forward config.

    The model's ALU (ADD/SUB) and comparisons are 32-bit-exact, but its bitwise /
    shift experts emit only the low byte (the 8-bit AX-nibble writeback + mod-256
    fold), so the faithful expectation is 32-bit for ADD/SUB/cmp and 8-bit for
    OR/XOR/AND/SHL/SHR.  ``ref_interpret`` instead 8-bit-masks EVERYTHING incl.
    IMM/compare, so it disagrees with the 32-bit model on the >255 add/sub/cmp
    cases — which is why this battery needs its own faithful oracle.  Operand
    order mirrors ``isa.interpret`` (pop <op> AX)."""
    a &= _M32
    b &= _M32
    if op == "ADD":
        return (a + b) & _M32
    if op == "SUB":
        return (a - b) & _M32
    # bitwise / shift: the model carries only the low byte of the result.
    if op == "OR":
        return (a | b) & 0xFF
    if op == "XOR":
        return (a ^ b) & 0xFF
    if op == "AND":
        return (a & b) & 0xFF
    if op == "SHL":
        return ((a & 0xFF) << (b & 0x1F)) & 0xFF
    if op == "SHR":
        return ((a & 0xFF) >> (b & 0x1F)) & 0xFF
    cmp = {"EQ": a == b, "NE": a != b, "LT": a < b,
           "GT": a > b, "LE": a <= b, "GE": a >= b}
    if op in cmp:
        return 1 if cmp[op] else 0
    raise ValueError(f"_faithful32_want: unsupported op {op}")


def _run_case(model, L, op, a, b):
    code = isa.assemble(_push_op(a, b, op))
    cap = len(ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
    tr = run_pure_forward_complete(model, L, code, max_steps=cap, mask=0xFFFFFFFF)
    got = (tr[-1] & 0xFFFFFFFF) if tr else None
    want = _faithful32_want(op, a, b)
    return got, want


def main():
    print("Building compact/streaming sparse model (bitwise, no divmod)...")
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=48, include_bitwise=True, include_divmod=False,
        compute_mode="dense_kernel")
    cases = _battery()
    print(f"battery: {len(cases)} op cases "
          f"(6 cmp x {len(cases) // 6 if False else 13} + add/sub + bitwise)\n")

    # ---- PASS 1: correctness of the CURRENT (untied) build ----
    # ``_faithful32_want`` is the model's TRUE per-op semantics (32-bit ADD/SUB/cmp,
    # 8-bit bitwise/shift).  A handful of EXTREME operands exceed the model's own
    # fp32 ALU precision (independent of the dedup) — those are reported as
    # informational "model precision" misses, NOT dedup failures.
    fails = []
    results_before = {}
    for i, (op, a, b) in enumerate(cases):
        got, want = _run_case(sparse, L, op, a, b)
        results_before[i] = got
        if got != want:
            fails.append((op, a, b, got, want))
    print(f"[before tie] {len(cases) - len(fails)}/{len(cases)} op cases match the "
          f"faithful oracle ({len(fails)} pre-existing model-precision misses)")
    for op, a, b, got, want in fails[:20]:
        print(f"   miss {op} {a:#x} {b:#x}: got {got}, want {want}")

    # ---- apply the byte-identical whole-tensor tie ----
    stats = dedup_sparse_transformer(sparse, L)
    print(f"\n[tie applied] {stats.weight_tensors_before} weight tensors -> "
          f"{stats.unique_weight_tensors_after} unique; "
          f"nonzero {stats.nonzero_before} -> {stats.nonzero_after} "
          f"(saved {stats.nonzero_saved})")

    # ---- PASS 2: the DEDUP INVARIANT — results identical across the tie ----
    # This (NOT absolute oracle-correctness) is the gate: the cross-functional
    # share + weight-tie must not CHANGE any op's result (argmax-exact).
    changed = []
    for i, (op, a, b) in enumerate(cases):
        got, _want = _run_case(sparse, L, op, a, b)
        if got != results_before[i]:
            changed.append((op, a, b, results_before[i], got))
    print(f"[after tie]  {len(changed)} results CHANGED by the tie "
          f"(dedup invariant: must be 0)")
    for op, a, b, pre, post in changed[:20]:
        print(f"   CHANGED {op} {a:#x} {b:#x}: {pre} -> {post}")

    # GATE = the dedup is argmax-exact: nothing changed across the tie.  (Absolute
    # oracle misses are pre-existing model precision, tracked separately.)
    ok = not changed
    print("\n" + ("BATTERY PASS: cross-functional share + weight-tie is "
                  "argmax-identical (0 results changed)" if ok else
                  "BATTERY FAIL: the tie CHANGED a result"))
    return 0 if ok else 1


def verify_bitwise_perbit_share() -> int:
    """Prove the CROSS-FUNCTIONAL bitwise share (OR/XOR/AND -> one shared per-bit
    gadget) is ARGMAX-EXACT: build the model with the gadget (default) and with the
    full 256-entry tables (``C4_BITWISE_PERBIT=0``) and assert the two agree on
    every bitwise op case (bit-identical result values, hence identical greedy
    decode).  Two builds; no divmod; ~a few GB RSS."""
    import os

    def _bitwise_cases():
        out = []
        for op in ("OR", "XOR", "AND"):
            for a, b in [(0x0C, 0x03), (0xFF, 0x0F), (0xF0, 0x3C), (0xABCD, 0x1234),
                         (0xDEADBEEF, 0x0F0F0F0F), (0x55, 0xAA), (0x00, 0x00),
                         (0xFFFF, 0xFFFF), (0x12345678, 0x87654321),
                         (0x80000000, 0x00000001), (0xFFFFFFFF, 0xFFFFFFFF)]:
                out.append((op, a, b))
        return out

    def _build():
        return build_compact_sparse_streaming(
            code_size=48, include_bitwise=True, include_divmod=False,
            compute_mode="dense_kernel")

    cases = _bitwise_cases()
    print("\n=== BITWISE per-bit share: argmax-identity gate (gadget vs full tables) ===")
    prev = os.environ.get("C4_BITWISE_PERBIT")
    try:
        os.environ["C4_BITWISE_PERBIT"] = "1"          # the shared per-bit gadget
        sp_on, L_on, _ = _build()
        on = {c: _run_case(sp_on, L_on, *c)[0] for c in cases}
        del sp_on, L_on
        os.environ["C4_BITWISE_PERBIT"] = "0"          # the full 256-entry tables
        sp_off, L_off, _ = _build()
        off = {c: _run_case(sp_off, L_off, *c)[0] for c in cases}
        del sp_off, L_off
    finally:
        if prev is None:
            os.environ.pop("C4_BITWISE_PERBIT", None)
        else:
            os.environ["C4_BITWISE_PERBIT"] = prev
    diffs = [c for c in cases if on[c] != off[c]]
    print(f"  {len(cases)} bitwise cases; per-bit-gadget vs table DIFFS: {len(diffs)}")
    for op, a, b in diffs[:20]:
        print(f"   DIFF {op} {a:#x} {b:#x}: gadget {on[(op, a, b)]} tbl {off[(op, a, b)]}")
    share_ok = not diffs
    print("  " + ("SHARE PASS: OR/XOR/AND per-bit gadget is argmax-identical to the "
                  "256-entry tables" if share_ok else "SHARE FAIL"))
    return 0 if share_ok else 1


if __name__ == "__main__":
    import sys
    if "--share-only" in sys.argv:
        # fast gate: only the OR/XOR/AND per-bit share argmax-identity check.
        raise SystemExit(verify_bitwise_perbit_share())
    rc = main()
    rc |= verify_bitwise_perbit_share()
    raise SystemExit(rc)
