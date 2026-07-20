#!/usr/bin/env python3
"""FAST-PATH benchmark for the HEAVY neural-VM programs.

The heavy demos (mandelbrot, malloc corpus, deep loops, self-emulation matmul)
ran the NAIVE token-by-token driver (``run_pure_forward_cached``): ONE
``model.forward`` per VM step (~1-3 s each), no cross-step speculation, no
block-MoE — which is why a tiny grid is minutes and a full render is hours.

This script stacks the proven speedup levers on a SINGLE heavy program and
MEASURES the result vs the naive path, byte-exact:

  1. PERFECT-DRAFT SPECULATION.  ``draft_pf_program`` runs the logical VM (0 model
     forwards, deterministic -> 100% accept) to materialise the whole per-step
     token stream, then ``verify_blocks`` confirms the model's argmax register
     state == the draft at every step-query row in the FEWEST, LARGEST batched
     forwards (K = ``--block-steps`` VM steps per forward).  N one-step forwards
     collapse to ceil(N/K) batched forwards.

  2. BLOCK-MoE DIVMOD-SKIP (``--block-moe``).  A forward whose span has NO DIV/MOD
     step skips the ~262-block divmod range (identity attention -> byte-identical)
     -> ~7x fewer blocks/forward on the mul/add/compare-heavy spans.

  3. GPU (``--device cuda:0``) + eviction (bounded KV cache -> big K viable).

The NAIVE baseline per-step wall is measured directly on a bounded prefix
(``--naive-steps``) of the SAME model, then projected to the full step count; the
FAST path measures the whole program's actual wall + forwards.  Byte-exactness is
the fast path's decoded output/final-AX vs the naive prefix (both run the model)
AND vs the perfect draft (which the naive path provably matches).

    OMP_NUM_THREADS=4 python -m c4_min.bench_fast_path mandel 8 8 3 \
        --device cuda:0 --block-steps 64 --block-moe --naive-steps 60
"""
from __future__ import annotations

import argparse
import gc
import os
import resource
import sys
import time
from typing import List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Program builders: (code, expected, data_seg, label).  ``expected`` is the
# reference final AX (or None for I/O programs where the deliverable is stdout).
# ---------------------------------------------------------------------------
def build_mandel(width: int, height: int, maxiter: int):
    from c4_min._mandel_src import mandel_c
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    src = mandel_c(width, height, maxiter)
    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    return code, None, data, f"mandel {width}x{height} maxiter {maxiter}"


def build_loop_countdown(n: int):
    """A deep pure-arithmetic loop: countdown from n to 0 (no divmod)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    src = f"""
    int main() {{
        int i;
        i = {n};
        while (i > 0) {{ i = i - 1; }}
        return i;
    }}
    """
    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    return code, 0, data, f"loop_countdown n={n}"


def build_malloc(n: int):
    """A malloc + memset + memcmp heap program built from the c4_min runtime library
    (``nibble_runtime``: pure base-ISA, uses the addr32 heap at 0x30008).

    ``malloc(n)`` two buffers, ``memset`` both to the same byte, ``memcmp`` them ->
    the memcmp all-equal loop returns 0.  memcmp is the deepest looping subroutine
    (the deep-loop heavy case per the memory notes).  Reference AX via
    ``ref_interpret_words``."""
    import c4_min.nibble_runtime as R
    n = max(1, n)
    pa, pb = R.HEAP_BASE, R.HEAP_BASE + max(16, n + 8)
    a = R.Asm()
    a.splice(R.emit_malloc(pb - pa))                 # bump the heap (exercise malloc)
    a.splice(R.emit_memset(pa, 0x41, n))             # fill A with 'A'
    a.splice(R.emit_memset(pb, 0x41, n))             # fill B with 'A' (equal)
    a.splice(R.emit_memcmp(pa, pb, n))               # all-equal -> 0
    a.exit_()
    code = a.instrs()
    ref_ax, _ = R.ref_interpret_words(code, max_steps=200000)
    return code, ref_ax & 0xFFFFFFFF, None, f"malloc+memset+memcmp n={n} (equal->0)"


def build_matmul(dim: int):
    """The self-emulation matmul: an NxN integer matmul on malloc'd pointer arrays
    (mul/add-heavy — the block-MoE has nothing to skip, so this stresses the
    speculation lever).  Expected AX via the model's own 8-bit reference."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import ref_interpret
    N = dim
    # malloc'd int arrays keep the local frame small (byte-sized LEA offsets) — the
    # matmul products stay < 256 for small N so the model's 8-bit ALU is byte-exact.
    src = f"""
    int main() {{
        int *a; int *b; int *c;
        int i; int j; int k; int s;
        a = malloc({N*N*4}); b = malloc({N*N*4}); c = malloc({N*N*4});
        i = 0;
        while (i < {N*N}) {{ a[i] = i % 3; b[i] = i % 2; c[i] = 0; i = i + 1; }}
        i = 0;
        while (i < {N}) {{
            j = 0;
            while (j < {N}) {{
                s = 0; k = 0;
                while (k < {N}) {{
                    s = s + a[i*{N}+k] * b[k*{N}+j];
                    k = k + 1;
                }}
                c[i*{N}+j] = s;
                j = j + 1;
            }}
            i = i + 1;
        }}
        return c[{N*N-1}];
    }}
    """
    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    ref_ax = ref_interpret(code, max_steps=500000, mask=0xFFFFFFFF)
    exp = (ref_ax[-1] & 0xFFFFFFFF) if ref_ax else None
    return code, exp, data, f"matmul {N}x{N} (malloc arrays)"


# ---------------------------------------------------------------------------
def _reference(code, data, max_steps):
    from c4_min.nibble_pure_forward_complete import ref_interpret
    ref_out: List[int] = []
    ref_tr = ref_interpret(code, max_steps=max_steps, mask=0xFFFFFFFF, out=ref_out)
    return ref_tr, ref_out


def run_bench(kind: str, args) -> int:
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    # SP_INIT must be PINNED CONSISTENTLY across the draft, the naive cached driver,
    # AND the model's init-frame — else the draft's frame bookkeeping (bp/sp) drifts
    # from the model by a constant (e.g. LEA off by 4).  run_corpus_resumable pins
    # 0xFC (the deep-recursion value); the default 0x10000 overflows the 8-bit frame
    # window.  CRITICAL: nibble_pure_forward_cached does ``from ... import SP_INIT``
    # (an import-time SNAPSHOT), so we must also patch ITS namespace binding, not
    # just the source modules.
    sp_init = args.sp_init if args.sp_init is not None else 0xFC
    _PF.SP_INIT = sp_init
    _PFC.SP_INIT = sp_init

    import torch
    from c4_min.lib_neural import build_lib_model_streaming
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PFCa.SP_INIT = sp_init                     # patch the cached driver's snapshot
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    from c4_min.pf_speculative import draft_pf_program, verify_blocks

    # -- build the program ----------------------------------------------------
    if kind == "mandel":
        code, expected, data, label = build_mandel(*args.grid)
    elif kind == "loop":
        code, expected, data, label = build_loop_countdown(args.n)
    elif kind == "malloc":
        code, expected, data, label = build_malloc(args.n)
    elif kind == "matmul":
        code, expected, data, label = build_matmul(args.n)
    else:
        raise ValueError(kind)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[bench] CUDA unavailable; cpu", file=sys.stderr)
        device = "cpu"

    # -- the PERFECT DRAFT (free) --------------------------------------------
    t = time.time()
    draft = draft_pf_program(code, max_steps=args.max_steps, mask=0xFFFFFFFF)
    t_draft = time.time() - t
    ndivmod = sum(1 for d in (draft.prtf_steps and []) or []) or 0
    ndivmod = sum(1 for f in draft.frames if f["op"] in ("DIV", "MOD"))
    print(f"=== {label} ===")
    print(f"  n_instrs={len(code)}  draft_steps={draft.step_count}  "
          f"halted={draft.halted}  divmod_steps={ndivmod} "
          f"({100*ndivmod/max(draft.step_count,1):.1f}%)  "
          f"prtf_bytes={len(draft.out or [])}  draft_wall={t_draft:.3f}s", flush=True)
    if not draft.halted:
        print("  DRAFT DID NOT HALT within max_steps -> genuine wall", flush=True)
        return 2

    # reference (the 8-bit ref oracle — may differ from the 32-bit model for
    # values >255; we report both).
    ref_tr, ref_out = _reference(code, data, draft.step_count + 8)

    # -- build the model (streaming sparse, memory-safe) ----------------------
    print(f"  RSS before build: {_rss_gb():.2f} GB", flush=True)
    t = time.time()
    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True,
        compute_mode=args.compute_mode)
    t_build = time.time() - t
    n_blocks = len(sparse.blocks)
    print(f"  build wall: {t_build:.1f}s  blocks={n_blocks}  "
          f"RSS after build: {_rss_gb():.2f} GB", flush=True)
    if device != "cpu":
        sparse = sparse.to(device)
        print(f"  moved model to {device}", flush=True)

    # -- NAIVE baseline: measure per-step wall on a bounded prefix ------------
    naive_steps = min(args.naive_steps, draft.step_count)
    print(f"  NAIVE: timing {naive_steps} token-by-token steps "
          f"(one model.forward each) ...", flush=True)
    naive_out: List[int] = []
    naive_stats: dict = {}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t = time.time()
    naive_tr = run_pure_forward_cached(
        sparse, L, code, max_steps=naive_steps, mask=0xFFFFFFFF, evict=(not args.no_evict),
        prune_interval=args.prune_interval, out=naive_out, stats=naive_stats,
        data_seg=data)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_naive = time.time() - t
    naive_per_step = t_naive / max(len(naive_tr), 1)
    proj_naive_full = naive_per_step * draft.step_count
    print(f"  NAIVE: {len(naive_tr)} steps in {t_naive:.1f}s "
          f"-> {naive_per_step*1000:.0f} ms/step  "
          f"(projected full {draft.step_count} steps = "
          f"{proj_naive_full:.0f}s = {proj_naive_full/60:.1f} min "
          f"= {proj_naive_full/3600:.2f} hr)", flush=True)

    # byte-check the naive prefix vs the draft (the fast path targets the draft).
    naive_prefix_ax = [f["ax"] & 0xFFFFFFFF for f in draft.frames[:len(naive_tr)]]
    naive_ax_ok = (naive_tr == naive_prefix_ax)
    print(f"  naive-prefix AX == draft: {naive_ax_ok}", flush=True)

    # -- FAST path: whole-program draft + batched verify ---------------------
    print(f"  FAST: verify {draft.step_count} steps in blocks of "
          f"K={args.block_steps} (block_moe={args.block_moe}) ...", flush=True)
    fast_out: List[int] = []
    fast_stats: dict = {}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=args.block_steps,
                       device=device, evict=(not args.no_evict), prune_interval=args.prune_interval,
                       mask=0xFFFFFFFF, stats=fast_stats, fast=True,
                       collect_out=fast_out, block_moe=args.block_moe)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_fast = time.time() - t
    fast_per_step = t_fast / max(draft.step_count, 1)
    print(f"  FAST: {vr.forwards} forwards in {t_fast:.1f}s "
          f"({t_fast/60:.2f} min) -> {fast_per_step*1000:.1f} ms/step-equiv  "
          f"| all_matched={vr.all_matched} accepted={vr.accepted_steps}/"
          f"{draft.step_count}", flush=True)
    print(f"  FAST: max_cache={fast_stats.get('max_cache_size')} "
          f"evicted={fast_stats.get('total_evicted')}  "
          f"blocks_run={fast_stats.get('blocks_run')} "
          f"blocks_full={fast_stats.get('blocks_full')} "
          f"block_moe_speedup={fast_stats.get('block_moe_speedup', 1.0):.2f}x", flush=True)
    if not vr.all_matched:
        print(f"  FAST FAIL: {vr.first_mismatch}", flush=True)

    # -- BYTE-EXACTNESS + speedup --------------------------------------------
    # fast output == naive output on the shared prefix (both run the model).
    fast_prefix_out = fast_out[:len(naive_out)]
    out_match_naive = (fast_prefix_out == naive_out)
    out_match_draft = (fast_out == (draft.out or []))
    fast_final_ax = vr.decoded_final_ax
    draft_final_ax = draft.final_ax_masked
    final_ax_match = (fast_final_ax == draft_final_ax)
    # ref comparison (8-bit oracle — divergence expected for values >255).
    out_match_ref = (fast_out == ref_out)

    # measured speedup = projected naive full wall / actual fast full wall.
    speedup_wall = proj_naive_full / max(t_fast, 1e-9)
    speedup_fwd = draft.step_count / max(vr.forwards, 1)
    print(f"  --- byte-exactness ---")
    print(f"    fast output == naive prefix (model==model): {out_match_naive} "
          f"({len(fast_prefix_out)} bytes)")
    print(f"    fast output == perfect draft:               {out_match_draft} "
          f"({len(fast_out)} bytes)")
    print(f"    fast final AX == draft final AX:            {final_ax_match} "
          f"(fast={fast_final_ax} draft={draft_final_ax})")
    if expected is not None:
        print(f"    fast final AX == expected (32-bit ref):     "
              f"{fast_final_ax == (expected & 0xFFFFFFFF)} (expected={expected & 0xFFFFFFFF})")
    print(f"    fast output == 8-bit ref oracle:            {out_match_ref} "
          f"(divergence for values>255 is expected & honest)")
    print(f"  --- SPEEDUP ---")
    print(f"    forward-count: {draft.step_count} naive -> {vr.forwards} fast "
          f"= {speedup_fwd:.1f}x fewer forwards")
    print(f"    wall-clock:    naive {naive_per_step*1000:.0f} ms/step (proj "
          f"{proj_naive_full/60:.1f} min full) vs fast {fast_per_step*1000:.1f} "
          f"ms/step-equiv ({t_fast/60:.2f} min full)")
    print(f"    EFFECTIVE WALL SPEEDUP: {speedup_wall:.1f}x", flush=True)
    print(f"  RSS peak: {_rss_gb():.2f} GB", flush=True)

    # cleanup
    del sparse
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    ok = vr.all_matched and out_match_naive and final_ax_match
    print(f"  RESULT: {'OK' if ok else 'MISMATCH'} "
          f"(model byte-exact fast==naive, verify accepted)", flush=True)
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["mandel", "loop", "malloc", "matmul"])
    ap.add_argument("grid", nargs="*", type=int, default=[],
                    help="mandel: W H MAXITER")
    ap.add_argument("--n", type=int, default=64, help="loop/malloc/matmul size")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--compute-mode", type=str, default="dense_kernel",
                    choices=["dense_kernel", "sparse_mm"])
    ap.add_argument("--block-steps", type=int, default=64,
                    help="K = VM steps verified per batched forward.")
    ap.add_argument("--block-moe", action="store_true",
                    help="skip the divmod block span on non-divmod spans.")
    ap.add_argument("--naive-steps", type=int, default=50,
                    help="how many naive steps to time for the per-step baseline.")
    ap.add_argument("--prune-interval", type=int, default=60)
    ap.add_argument("--no-evict", action="store_true",
                    help="disable KV eviction (bounded VRAM off; larger cache).")
    ap.add_argument("--max-steps", type=int, default=5_000_000)
    ap.add_argument("--sp-init", type=lambda s: int(s, 0), default=None,
                    help="override SP_INIT (e.g. 0xFC for deep recursion).")
    args = ap.parse_args(argv)
    if args.kind == "mandel":
        if len(args.grid) < 3:
            args.grid = (args.grid + [1, 1, 3])[:3]
        args.grid = tuple(args.grid[:3])
    return run_bench(args.kind, args)


if __name__ == "__main__":
    raise SystemExit(main())
