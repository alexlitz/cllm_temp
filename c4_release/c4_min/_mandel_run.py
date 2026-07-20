"""Run the tiny fixed-point Mandelbrot through the ACTUAL c4_min model.forward,
byte-exact vs the reference interpreter.  Diagnostic runner (leading underscore);
the pytest-facing assertion lives in ``test_mandelbrot_neural.py``.

    OMP_NUM_THREADS=4 python -m c4_min._mandel_run [W H MAXITER]
"""
from __future__ import annotations
import os, sys, time, resource
os.environ.setdefault("OMP_NUM_THREADS", "4")

from c4_min import isa
from c4_min._mandel_src import mandel_c


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run(width, height, maxiter, max_steps=200000):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import ref_interpret
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    src = mandel_c(width, height, maxiter)
    bytecode, data = compile_c(src)                 # the REAL c4_min C compiler
    code = bytecode_to_isa(bytecode)
    big = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not big, f"IMM>255 present (would diverge from ref): {big}"

    # Reference (byte-masking, unsigned 32-bit) — the blessed oracle the neural
    # driver is byte-identical to.  Captures the PRTF stdout stream.
    ref_out = []
    ref_tr = ref_interpret(code, max_steps=max_steps, mask=0xFFFFFFFF, out=ref_out)
    n_ref_steps = len(ref_tr)

    print(f"grid {width}x{height} maxiter {maxiter}: "
          f"n_instrs={len(code)} ref_steps={n_ref_steps}")
    print(f"RSS before build: {_rss_gb():.2f} GB", flush=True)

    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True)
    t_build = time.time() - t0
    print(f"build wall: {t_build:.1f}s  RSS after build: {_rss_gb():.2f} GB", flush=True)

    out, stats = [], {}
    t0 = time.time()
    run_pure_forward_cached(sparse, L, code, max_steps=n_ref_steps + 6,
                            mask=0xFFFFFFFF, evict=True, prune_interval=60,
                            out=out, stats=stats, data_seg=data)
    t_run = time.time() - t0
    n = stats.get("steps", 0)

    def render(bytes_):
        return "".join(chr(b) if b != 10 else "\n" for b in bytes_)

    print(f"run wall: {t_run:.1f}s  neural_steps={n}  "
          f"per-step={t_run / max(1, n):.2f}s  peak RSS={_rss_gb():.2f} GB")
    print(f"max_seq={stats.get('max_seq_len')} max_cache={stats.get('max_cache_size')} "
          f"evicted={stats.get('total_evicted')}")
    print("--- reference render ---")
    print(render(ref_out).rstrip("\n"))
    print("--- neural model.forward render ---")
    print(render(out).rstrip("\n"))
    match = out == ref_out
    print(f"BYTE-EXACT MATCH: {match}  (neural {len(out)} bytes, ref {len(ref_out)} bytes)")
    return match, out, ref_out, dict(n_instrs=len(code), ref_steps=n_ref_steps,
                                     neural_steps=n, build_s=t_build, run_s=t_run,
                                     rss_gb=_rss_gb(), stats=stats)


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    MI = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ok, *_ = run(W, H, MI)
    sys.exit(0 if ok else 1)
