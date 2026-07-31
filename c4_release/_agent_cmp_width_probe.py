"""5th-WALL probe: does the model's LT/GT/EQ compare 32-bit or 8-bit operands?

doom's init loops run counters up to CIRC=256 (`while (i < CIRC)`, `i <= CIRC/4`).
If the cmp path folds operands to 8 bits, `255 < 256` and `256 < 256` both mis-decode
(256&0xFF==0), so the loop never terminates -> the draft (and model) infinite-loops.

We push two operands that CROSS the byte boundary (e.g. 255 vs 256, 256 vs 256,
300 vs 256, 100 vs 256) and run LT/GT/EQ through the byte-exact cached driver, decoding
the 0/1 verdict.  Compared to the true signed 32-bit answer.
"""
from __future__ import annotations
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from c4_min import isa


def cmp_prog(a: int, b: int, op: str):
    # push a, set AX=b, then CMP: verdict = (a OP b).  Stack pops a as STACK0.
    return isa.assemble([("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)])


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=64, recurrent_divmod=True,
                                             addr32=True, compute_mode="dense_kernel")
    if device != "cpu":
        sparse = sparse.to(device); sparse.materialize_dense(device=device)
    print(f"[cmp] built in {time.time()-t0:.1f}s dev={device}", flush=True)

    cases = [
        # (a, b) crossing the byte boundary — the doom loop-counter regime
        (255, 256), (256, 256), (257, 256), (300, 256), (100, 256),
        (64, 64), (65, 64), (63, 64),           # i <= CIRC/4 == 64 regime (8-bit ok)
        (128, 200), (200, 128),
    ]
    print(f"\n  {'a':>5} {'b':>5} {'op':>3} {'model':>6} {'true':>5} {'8bit':>5} {'ok':>4}",
          flush=True)
    print("  " + "-" * 40, flush=True)
    all_ok = True
    for op in ("LT", "GT", "EQ"):
        for a, b in cases:
            code = cmp_prog(a, b, op)
            tr = run_pure_forward_cached(sparse, L, code, max_steps=8, mask=0xFFFFFFFF)
            model_v = tr[-1] if tr else None      # verdict is AX after CMP (before HALT it's last)
            # verdict step is the CMP op (step index 3, 0-based); HALT doesn't change AX
            model_v = tr[3] if len(tr) > 3 else model_v
            true_v = int({"LT": a < b, "GT": a > b, "EQ": a == b}[op])
            b8 = int({"LT": (a & 0xFF) < (b & 0xFF), "GT": (a & 0xFF) > (b & 0xFF),
                      "EQ": (a & 0xFF) == (b & 0xFF)}[op])
            ok = (model_v == true_v)
            all_ok = all_ok and ok
            flag = "" if ok else "  <-- MISMATCH (5th wall?)"
            print(f"  {a:>5} {b:>5} {op:>3} {str(model_v):>6} {true_v:>5} {b8:>5} "
                  f"{'OK' if ok else 'X':>4}{flag}", flush=True)

    print(f"\n[cmp] model comparisons are 32-bit-correct across the byte boundary: {all_ok}",
          flush=True)
    if not all_ok:
        print("[cmp] --> 5th WALL: cmp operands are 8-bit-folded; doom's i<256 loops "
              "won't terminate. Needs 32-bit cmp (widen the ingest recompose / STK_VAL "
              "AX_VAL to carry all nibbles, i.e. C4_VM_WIDTH32-style).", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
