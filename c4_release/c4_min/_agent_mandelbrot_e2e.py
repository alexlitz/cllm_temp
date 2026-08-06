"""End-to-end Mandelbrot on the c4 stack: compile -> draft -> transformer, plus
HF-config feasibility + float caveat.  A read-only measurement harness (authors NO
weights; golden 069cc32f untouched).  Consolidates the numbers in
docs/MANDELBROT_E2E_C4_STACK_2026_08_05.md.

Subcommands (all memory-safe, single process):

  draft        the CPU word-oracle (ref_interpret, mask=0xFFFFFFFF) — steps + wall
               for BOTH mandelbrot paths, at 16x8 .. 80x48.
  gcc          compile each path's C/algorithm with the NATIVE gcc and diff the
               stdout byte-for-byte vs the c4 draft (byte-exactness gate).
  hf           build the VANILLA stock-Qwen2ForCausalLM VM per subset and report
               n_layers / hidden / fits-stock-0.5B (the HF-config feasibility).
  transformer  run the tiny 1x1 cell through the ACTUAL model.forward (streaming
               sparse + KV-cached driver) and assert byte-exact (heavy; GPU).

    PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._agent_mandelbrot_e2e draft
    PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._agent_mandelbrot_e2e hf
    PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._agent_mandelbrot_e2e transformer --device=cuda:0

Two Mandelbrot programs live in the tree, with different tradeoffs:

  * ``_mandel_src.mandel_c``  — a REAL C source the c4 compiler compiles (260
    instrs, a static grid loop).  Restructured to keep every AX value NON-NEGATIVE
    (unsigned first-quadrant window) so the transformer's AX-0xFF-leak weakness
    never fires => transformer-SAFE, but the render is the VM's own UNSIGNED
    escape-time slice (a degenerate ``*   *`` lattice, NOT the textbook set, and
    NOT gcc-byte-exact — gcc's signed ints escape where the unsigned VM does not).
  * ``mandelbrot_native``     — the REAL Mandelbrot (window cx in [-2.1,0.9], cy in
    [-1.2,1.2], toward-zero fixed-point).  A per-pixel ~182-instr LOOP (constant in
    max_iter), byte-exact vs gcc (verified here).  Uses SIGNED biased byte-storage,
    so it exercises the AX-leak weakness => NOT yet transformer-byte-exact past the
    first cells (a correctness wall, documented).
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")


def _render(bs):
    return "".join(chr(b) if b != 10 else "\n" for b in bs)


def draft():
    """CPU word-oracle draft: steps + wall for both paths."""
    from src.compiler import compile_c
    from c4_min import isa
    from c4_min._mandel_src import mandel_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import ref_interpret
    from c4_min import mandelbrot_native as MN

    print("=== _mandel_src (transformer-safe unsigned slice) — c4 draft ===")
    for (W, H, MI) in [(16, 8, 20), (32, 16, 30), (64, 32, 40), (80, 48, 50)]:
        code = bytecode_to_isa(compile_c(mandel_c(W, H, MI))[0])
        over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
        assert not over, f"IMM>255 {over}"
        out = []
        t0 = time.time()
        tr = ref_interpret(code, max_steps=2_000_000_000, mask=0xFFFFFFFF, out=out)
        dt = time.time() - t0
        print(f"  {W}x{H} mi{MI}: instrs={len(code)} steps={len(tr):,} "
              f"wall={dt:.3f}s ({len(tr) / dt / 1e6:.2f}M/s) out={len(out)}B")

    print("=== mandelbrot_native (REAL set, per-pixel loop) — c4 draft ===")
    for (W, H, MI) in [(16, 8, 20), (32, 16, 30), (64, 32, 40), (80, 48, 50)]:
        cxs, cys = MN.grid(W, H)
        t0 = time.time()
        total = 0
        for cy in cys:
            for cx in cxs:
                total += MN.steps_per_pixel(cx, cy, MI)
        dt = time.time() - t0
        print(f"  {W}x{H} mi{MI}: pixel_instrs={len(MN.pixel_program_native(0, 0, MI))} "
              f"total_steps={total:,} wall={dt:.2f}s ({total / dt / 1e6:.2f}M/s)")


def gcc():
    """Native-gcc byte-exactness gate for the native (real) Mandelbrot path."""
    import subprocess
    import tempfile
    from c4_min import mandelbrot_native as MN

    W, H, MI = 80, 48, 50
    c_src = r'''#include <stdio.h>
#include <math.h>
#define Q_FRAC 4
#define SCALE 16
#define FOUR 64
static long sfp_mul(long a,long b){long m=((a<0?-a:a)*(b<0?-b:b))>>Q_FRAC;return ((a<0)!=(b<0))?-m:m;}
static int pesc(long cx,long cy,int mi){long zx=0,zy=0;int i;for(i=0;i<mi;i++){long zx2=sfp_mul(zx,zx),zy2=sfp_mul(zy,zy);if(zx2+zy2>FOUR)return i;long two=2*sfp_mul(zx,zy);long nzx=zx2-zy2+cx,nzy=two+cy;zx=nzx;zy=nzy;}return mi;}
int main(){int W=%d,H=%d,MI=%d;const char*CS=" .:-=+*#%%@";int c,r;
for(r=0;r<H;r++){long cy=(long)nearbyint((r/(double)(H-1)*2.4-1.2)*SCALE);
for(c=0;c<W;c++){long cx=(long)nearbyint((c/(double)(W-1)*3.0-2.1)*SCALE);int e=pesc(cx,cy,MI);putchar(CS[e<9?e:9]);}putchar('\n');}return 0;}
''' % (W, H, MI)
    d = tempfile.mkdtemp()
    csrc = os.path.join(d, "m.c")
    exe = os.path.join(d, "m")
    open(csrc, "w").write(c_src)
    r = subprocess.run(["gcc", "-O2", "-static", "-o", exe, csrc, "-lm"],
                       capture_output=True)
    if r.returncode != 0:
        print("gcc failed:", r.stderr.decode()[:200])
        return
    gcc_out = subprocess.run([exe], capture_output=True).stdout
    c4_out = (MN.render_ascii_native(W, H, MI) + "\n").encode()
    match = gcc_out == c4_out
    print(f"native gcc vs c4-draft {W}x{H} mi{MI}: gcc={len(gcc_out)}B c4={len(c4_out)}B "
          f"BYTE-EXACT={match}")
    if not match:
        for i, (a, b) in enumerate(zip(gcc_out, c4_out)):
            if a != b:
                print(f"  first diff at byte {i}: gcc={a} c4={b}")
                break


def hf():
    """Stock-Qwen2ForCausalLM feasibility per subset."""
    from transformers.models.qwen2 import Qwen2ForCausalLM
    from c4_min import qwen_vanilla_vm as VV

    print("=== HF-config: does the VM fit a stock Qwen2ForCausalLM? ===")
    print("stock released Qwen2.5-0.5B: n_layers=24, hidden=896, inter=4864, qheads=14")
    subs = [("base", VV.SUBSET_BASE), ("base+mem", VV.SUBSET_MEM),
            ("mem+cmp", VV.SUBSET_MEM_CMP), ("+bitwise", VV.SUBSET_BITWISE),
            ("+muldiv", VV.SUBSET_MULDIV), ("full", VV.SUBSET_FULL)]
    for name, sub in subs:
        vm = VV.build(code_size=64, subset=sub, device="cpu")
        cfg = vm.model.config
        stock_class = isinstance(vm.model, Qwen2ForCausalLM)
        fits_05b = (vm.n_layers <= 24 and cfg.hidden_size <= 896
                    and cfg.intermediate_size <= 4864)
        print(f"  {name:10s} n_layers={vm.n_layers:3d} hidden={cfg.hidden_size:5d} "
              f"inter={cfg.intermediate_size:5d} stock_class={stock_class} "
              f"fits_released_0.5B={fits_05b}")
        del vm


def transformer(device="cpu"):
    """Run the tiny 1x1 cell through model.forward; assert byte-exact."""
    import resource
    from src.compiler import compile_c
    from c4_min._mandel_src import mandel_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import ref_interpret
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    W, H, MI = 1, 1, 1
    bc, data = compile_c(mandel_c(W, H, MI))
    code = bytecode_to_isa(bc)
    ref_out = []
    n_ref = len(ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF, out=ref_out))
    print(f"[compile] instrs={len(code)} ref_steps={n_ref} ref_out={bytes(ref_out)!r}")

    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True)
    print(f"[build] wall={time.time() - t0:.1f}s RSS={rss():.2f}GB")
    if device != "cpu":
        sparse = sparse.to(device)
        print(f"[build] moved to {device}")

    out, stats = [], {}
    t0 = time.time()
    run_pure_forward_cached(sparse, L, code, max_steps=n_ref + 6, mask=0xFFFFFFFF,
                            evict=True, prune_interval=60, out=out, stats=stats,
                            data_seg=data)
    dt = time.time() - t0
    n = stats.get("steps", 0)
    match = out == ref_out
    print(f"[run] wall={dt:.1f}s steps={n} per-step={dt / max(1, n):.3f}s "
          f"peakRSS={rss():.2f}GB max_cache={stats.get('max_cache_size')}")
    print(f"[RESULT] BYTE_EXACT={match} neural={bytes(out)!r} ref={bytes(ref_out)!r}")
    return match


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "draft"
    dev = "cpu"
    for a in sys.argv:
        if a.startswith("--device="):
            dev = a.split("=", 1)[1]
    if cmd == "draft":
        draft()
    elif cmd == "gcc":
        gcc()
    elif cmd == "hf":
        hf()
    elif cmd == "transformer":
        ok = transformer(device=dev)
        sys.exit(0 if ok else 2)
    else:
        print(__doc__)
