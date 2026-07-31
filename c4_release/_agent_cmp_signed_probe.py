"""Probe the model's SIGNED cmp with NEGATIVE operands (doom uses `x >= 0`, `x < y`
with signed ints).  A negative in this VM is a large 32-bit word (2^32 - |x|).  We
form a negative by SUB (0 - k) then compare.  Check LT/GE against the true signed
answer, so the draft fix (32-bit signed cmp) matches the model exactly."""
from __future__ import annotations
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch
from c4_min import isa

M = 0xFFFFFFFF
def s32(v):
    v &= M; return v - (1 << 32) if v & 0x80000000 else v


def neg_cmp_prog(a: int, b: int, op: str):
    # Build operand A = a (may be negative via 0 - |a|), push it, set AX=b (or 0-|b|),
    # then CMP.  Uses SUB to synthesize negatives: 0 - k.
    prog = []
    def load(v):
        if v >= 0:
            prog.append(("IMM", v))
        else:
            prog.append(("IMM", 0)); prog.append(("PSH", 0))
            prog.append(("IMM", -v)); prog.append(("SUB", 0))   # 0 - |v|
    load(a); prog.append(("PSH", 0)); load(b); prog.append((op, 0)); prog.append(("HALT", 0))
    return isa.assemble(prog)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=64, recurrent_divmod=True,
                                             addr32=True, compute_mode="dense_kernel")
    if device != "cpu":
        sparse = sparse.to(device); sparse.materialize_dense(device=device)
    print(f"[signed-cmp] built {time.time()-t0:.1f}s dev={device}", flush=True)

    cases = [(-5, 0), (5, 0), (0, 0), (-1, -2), (-2, -1), (-100, 100),
             (100, -100), (-256, 0), (300, -1)]
    print(f"\n  {'a':>6} {'b':>6} {'op':>3} {'model':>6} {'true_signed':>12} {'ok':>4}",
          flush=True)
    print("  " + "-" * 44, flush=True)
    all_ok = True
    for op in ("LT", "GE", "GT", "LE"):
        for a, b in cases:
            code = neg_cmp_prog(a, b, op)
            tr = run_pure_forward_cached(sparse, L, code, max_steps=16, mask=0xFFFFFFFF)
            model_v = tr[-2] if len(tr) >= 2 else (tr[-1] if tr else None)  # CMP is second-to-last (before HALT)
            true_v = int({"LT": s32(a) < s32(b), "GE": s32(a) >= s32(b),
                          "GT": s32(a) > s32(b), "LE": s32(a) <= s32(b)}[op])
            ok = (model_v == true_v)
            all_ok = all_ok and ok
            print(f"  {a:>6} {b:>6} {op:>3} {str(model_v):>6} {true_v:>12} "
                  f"{'OK' if ok else 'X <--':>4}", flush=True)
    print(f"\n[signed-cmp] model does 32-bit SIGNED cmp: {all_ok}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
