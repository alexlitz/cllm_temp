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


class _GpuUtilSampler:
    """Poll ``nvidia-smi`` GPU utilisation in a background thread while a verify
    runs, so the report has an honest GPU-busy number (before vs after big-K)."""

    def __init__(self, device: str, interval: float = 0.25):
        import threading
        self._idx = 0
        if ":" in device:
            try:
                self._idx = int(device.split(":")[1])
            except ValueError:
                self._idx = 0
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.samples: List[float] = []

    def _poll(self) -> Optional[float]:
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits",
                 f"--id={self._idx}"], stderr=subprocess.DEVNULL, timeout=2)
            return float(out.decode().strip().splitlines()[0])
        except Exception:
            return None

    def start(self):
        import threading

        def _run():
            while not self._stop.wait(self._interval):
                u = self._poll()
                if u is not None:
                    self.samples.append(u)
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def n(self) -> int:
        return len(self.samples)


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


def build_nested(outer: int, inner: int):
    """A byte-SAFE deep nested loop (all values <= 255 so the model's 8-bit ALU is
    byte-exact to the 32-bit draft): ``outer`` times, count ``inner`` down to 0;
    return the outer count.  A genuinely DEEP program (outer*inner*~5 steps) that
    stays fully verifiable — the deep-loop headline."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    src = (f"int main(){{ int a; int b; int r; b=0; r={outer}; "
           f"while(r>0){{ a={inner}; while(a>0){{ a=a-1; }} b=b+1; r=r-1; }} "
           f"return b; }}")
    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    return code, outer, data, f"nested_loop {outer}x{inner} (byte-safe deep)"


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


def build_malloc_free(n: int):
    """A malloc + memset + FREE (zero-tombstone) + RE-USE heap program — exercises the
    heap free() -> zero-tombstone -> reclaim path (§689-691) the exact-evict schedule
    must reclaim.  ``malloc(n)``, fill with 'A', ``free`` (writes 0 == the tombstone),
    then re-fill the SAME buffer with 'B' (the reuse SUPERSEDES the tombstone), memcmp
    the reused buffer against a second buffer also filled with 'B' (equal -> 0).  So a
    live-heap slot is freed, its cell zeroed, then re-allocated and re-written — the
    tombstone row must go dead (superseded by the reuse) and its cache slot reclaimed."""
    import c4_min.nibble_runtime as R
    n = max(1, n)
    pa, pb = R.HEAP_BASE, R.HEAP_BASE + max(16, n + 8)
    a = R.Asm()
    a.splice(R.emit_malloc(pb - pa))                 # bump the heap
    a.splice(R.emit_memset(pa, 0x41, n))             # fill A with 'A'
    a.splice(R.emit_free(pa))                        # free(A): *(int*)pa = 0 (tombstone)
    a.splice(R.emit_memset(pa, 0x42, n))             # REUSE A: refill with 'B' (supersede)
    a.splice(R.emit_memset(pb, 0x42, n))             # fill B with 'B' (equal to reused A)
    a.splice(R.emit_memcmp(pa, pb, n))               # all-equal -> 0
    a.exit_()
    code = a.instrs()
    ref_ax, _ = R.ref_interpret_words(code, max_steps=200000)
    return code, ref_ax & 0xFFFFFFFF, None, f"malloc+free+reuse+memcmp n={n} (equal->0)"


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


def _free_vram_gb(device: str) -> float:
    """Free VRAM on ``device`` (GB) via nvidia-smi (contention-aware; torch's own
    mem_get_info can under-report other-process usage on a shared box)."""
    import subprocess
    idx = 0
    if ":" in device:
        try:
            idx = int(device.split(":")[1])
        except ValueError:
            idx = 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", f"--id={idx}"],
            stderr=subprocess.DEVNULL, timeout=3)
        return float(out.decode().strip().splitlines()[0]) / 1024.0
    except Exception:
        return 0.0


def _wait_for_vram(device: str, min_free_gb: float, timeout_s: float) -> None:
    """Block until ``device`` has ``min_free_gb`` free (or ``timeout_s`` elapses).
    The box is multi-agent GPU-contended; this avoids a load-time OOM when another
    process momentarily owns the card, without failing the run."""
    if min_free_gb <= 0:
        return
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        free = _free_vram_gb(device)
        if free >= min_free_gb:
            return
        print(f"  [vram-wait] {device} free={free:.1f}GB < {min_free_gb:.1f}GB; "
              f"waiting ({int(time.time()-t0)}s/{int(timeout_s)}s) ...", flush=True)
        time.sleep(10)
    print(f"  [vram-wait] proceeding after {int(timeout_s)}s "
          f"(free={_free_vram_gb(device):.1f}GB); OOM-backoff will adapt K",
          flush=True)


def _run_k_sweep(sparse, L, code, draft, device, args) -> int:
    """Sweep K over ``args.k_sweep`` on ONE built model + ONE draft; print the
    K vs amortized-ms/step vs peak-VRAM vs forwards vs GPU-util table.  The draft
    is byte-identical across K (K only changes how the SAME stream is verified),
    so ``all_matched`` / final AX MUST be K-invariant — the byte-exact gate."""
    import torch
    from c4_min.pf_speculative import verify_blocks
    ks = [int(x) for x in args.k_sweep.split(",") if x.strip()]
    n_steps = draft.step_count
    print(f"  === K-SWEEP over {ks} on {n_steps} steps "
          f"(evict_interval_steps={args.evict_interval_steps or 'per-block'}, "
          f"block_moe={args.block_moe}) ===", flush=True)
    header = (f"    {'K':>6} {'forwards':>9} {'eff_K':>6} {'wall_s':>9} "
              f"{'ms/step':>9} {'vram_GB':>8} {'gcache':>7} {'lcache':>7} {'evictR':>7} "
              f"{'GPU%':>5} {'match':>6} {'AX':>6}")
    print(header, flush=True)
    print("    " + "-" * (len(header) - 4), flush=True)
    ref_ax = None
    ref_match = None
    all_ok = True
    last_split_stats: dict = {}
    for K in ks:
        stats: dict = {}
        out: List[int] = []
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        sampler = _GpuUtilSampler(device) if args.gpu_util else None
        if sampler:
            sampler.start()
        t = time.time()
        try:
            vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=device,
                               evict=(not args.no_evict), prune_interval=args.prune_interval,
                               mask=0xFFFFFFFF, stats=stats, fast=True, collect_out=out,
                               block_moe=args.block_moe,
                               evict_interval_steps=args.evict_interval_steps,
                               oom_backoff=True, min_block_steps=args.min_block_steps)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # even the OOM-backoff floor (min_block_steps) did not fit — this K's
            # span exceeds the available VRAM.  Record it (the real ceiling under
            # this contention) and continue the sweep instead of aborting.
            if sampler:
                sampler.stop()
            if device.startswith("cuda"):
                gc.collect()
                torch.cuda.empty_cache()
            msg = "OOM" if "out of memory" in str(e).lower() else "ERR"
            print(f"    {K:>6} {'-':>9} {'-':>6} {'-':>9} {'-':>9} "
                  f"{'-':>8} {'-':>7} {'-':>7} {'-':>7} {'-':>5} {msg:>6} {'-':>6}"
                  f"   (K span did not fit even at min_block_steps)", flush=True)
            continue
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        wall = time.time() - t
        if sampler:
            sampler.stop()
        ms = wall / max(n_steps, 1) * 1000.0
        gpu = sampler.mean if sampler else 0.0
        ax = vr.decoded_final_ax
        # K-invariance byte-exact gate: match/AX must be identical across all K.
        if ref_ax is None:
            ref_ax, ref_match = ax, vr.all_matched
        elif ax != ref_ax or vr.all_matched != ref_match:
            all_ok = False
        if stats.get("split_active"):
            last_split_stats = stats
        print(f"    {K:>6} {vr.forwards:>9} "
              f"{stats.get('effective_block_steps', K):>6} {wall:>9.2f} "
              f"{ms:>9.2f} {stats.get('peak_vram_gb', 0.0):>8.2f} "
              f"{stats.get('max_cache_size', 0):>7} "
              f"{stats.get('max_local_cache', 0):>7} "
              f"{stats.get('evict_rounds', 0):>7} {gpu:>5.0f} "
              f"{str(vr.all_matched):>6} {str(ax):>6}", flush=True)
    print(f"    K-invariance (all completed K same match+AX): {all_ok}", flush=True)
    if last_split_stats:
        print(f"    DROP-KV split: eviction manages "
              f"{last_split_stats.get('n_global_caches')} global caches of "
              f"{len(sparse.blocks)} blocks (vs all {len(sparse.blocks)} classic); "
              f"peak global cache {last_split_stats.get('peak_global_cache')} rows, "
              f"local window rows/block <= {last_split_stats.get('max_local_cache')}; "
              f"KV-row footprint {last_split_stats.get('kv_rows_split')} vs classic full-H "
              f"{last_split_stats.get('kv_rows_classic')} "
              f"({last_split_stats.get('kv_row_reduction', 1.0):.1f}x fewer rows)",
              flush=True)
    print(f"  RSS peak: {_rss_gb():.2f} GB", flush=True)
    return 0 if all_ok else 1


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
    elif kind == "nested":
        code, expected, data, label = build_nested(args.outer, args.inner)
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
        _wait_for_vram(device, args.min_free_gb, args.wait_vram_s)
        sparse = sparse.to(device)
        print(f"  moved model to {device}", flush=True)

    # -- LOCAL (sliding-window) attention on the non-memory heads (opt-in) -----
    # The ~20 ingest heads only read the last W tokens (O(S*W)); the memory /
    # stack-pop / LEV KV heads stay GLOBAL (full causal).  Byte-identical: each
    # windowed head's true attention weight past W is exactly 0.
    if args.local_window is not None:
        from .local_attention import install_local_attention
        drop_kv = not args.local_mask_only
        la = install_local_attention(sparse, window=args.local_window,
                                     drop_local_kv=drop_kv,
                                     content_bound_global=args.content_bound_global,
                                     verbose=True)
        mode = "DROP-KV" if drop_kv else "MASK-ONLY"
        cb = ("  +CONTENT-BOUND global (store-only cache, working-set bounded)"
              if args.content_bound_global else "")
        print(f"  LOCAL-ATTN [{mode}]: window={la['window']}  "
              f"windowed {la['frac_windowed']*100:.1f}% of head-slots  "
              f"({la['n_global_head_slots']} global, {la['n_local_head_slots']} local)"
              + ("  -> local heads keep only last-W KV (cache ~Hg·S + Hl·W)"
                 if drop_kv else "  -> full KV kept, windowed READ only (compute win)")
              + cb, flush=True)

    # -- LIVE-HEAD-ONLY attention SCORING (opt-in) ---------------------------
    # Compute Q@Kᵀ + attn@V ONLY for the ~23 live-value head-slots (of 5451);
    # every _zero_attn head's per-head output is 0 by construction, so its score
    # matmul is skipped byte-identically.  Orthogonal to --local-window (that
    # windows the live heads' READ; this prunes the DEAD heads' matmul entirely).
    if getattr(args, "live_head", False) or getattr(args, "fuse_dead_blocks", False):
        from .live_head_attention import install_live_head_attention
        lh = install_live_head_attention(sparse, verbose=True)
        print(f"  LIVE-HEAD-ATTN: scoring {lh['live_head_slots']}/"
              f"{lh['total_head_slots']} head-slots "
              f"({lh['frac_scored_after']*100:.2f}%)  "
              f"{lh['dead_attention_blocks']}/{lh['n_blocks']} blocks are pure "
              f"x-passthrough (attention sublayer skipped)", flush=True)

    # -- DEAD-BLOCK ATTENTION FUSION (opt-in) --------------------------------
    # Bypass the ENTIRE attention sublayer of the ~234 dead-attention blocks
    # (output = x directly; NO K/Q/V/W_o linears, NO KV write).  --live-head only
    # skips the score matmul; this also skips the per-block linears (the measured
    # wall) and the KV materialisation.  Byte-identical: a dead block's attention
    # output is x (all heads _zero_attn), and its KV is provably never read.
    if getattr(args, "fuse_dead_blocks", False):
        from .live_head_attention import install_dead_block_fusion
        df = install_dead_block_fusion(sparse, verbose=True)
        print(f"  DEAD-BLOCK-FUSION: bypassed {df['fused_blocks']}/{df['n_blocks']} "
              f"dead-attention blocks (output=x, no K/Q/V/W_o linears, no KV write); "
              f"{df['live_attention_blocks']} live blocks keep attention", flush=True)

    # -- MATERIALIZE-DENSE (opt-in, the byte-exact forward win) --------------
    # Densify the CSR FFN/attn weights ONCE onto the device so ``.linear`` stops
    # re-running ``csr.to_dense()`` on EVERY forward (705 CSR densifies/forward was
    # the launch-bound wall at small S).  Byte-identical (same F.linear GEMM, same
    # accum order, L-inf=0).  Costs the full dense model VRAM (~5.8 GB lean).
    if getattr(args, "materialize_dense", False) or getattr(args, "graph_fused", False):
        if not device.startswith("cuda"):
            print("  MATERIALIZE-DENSE requires CUDA; skipping", flush=True)
        else:
            t = time.time()
            sparse.materialize_dense(device=device)
            print(f"  MATERIALIZE-DENSE: densified all CSR weights onto {device} "
                  f"in {time.time()-t:.1f}s (per-forward csr.to_dense eliminated, "
                  f"byte-identical)", flush=True)

    # -- STATIC BLOCK-SPARSE FFN (opt-in) ------------------------------------
    # Replace the WASTEFUL dense FFN GEMMs (99.93% sparse, ~60% of forward compute)
    # with a STATIC gather-scale-scatter (COO): the fixed nonzero pattern -> a fixed
    # index program (few-FLOP, no zero-multiply). Runs AFTER materialize-dense (it
    # reads the resident dense/CSR weights). Byte-exact up to fp-reduction order
    # (rel residue ~1e-7 << the integer decode margin; the 1-nnz-per-row majority
    # is bit-exact). At production span S~=961 this is ~2x faster than the dense FFN
    # GEMM (STEP 3 microbench); the scattered nonzeros keep it well above the FLOP
    # floor (see report). DEFAULT OFF -> golden byte-identical.
    if getattr(args, "block_sparse_ffn", False):
        from .block_sparse_ffn import install_block_sparse_ffn
        bs = install_block_sparse_ffn(sparse, mode=args.block_sparse_mode,
                                      verbose=True)

    # -- CUDA-GRAPHED dead-segment forward (opt-in) --------------------------
    # Collapse the ~235 dead-block FFN kernel LAUNCHES per forward: capture each
    # contiguous dead-attention segment into a CUDA graph (replayed per verify-
    # block).  The ~3 live-attention blocks stay eager.  Byte-identical (dead
    # segment graphed output == eager fused chain, L-inf=0).  NOTE: only wins at
    # SMALL span S (launch-bound); at the large verify spans the forward is
    # GEMM-compute-bound so this is ~neutral (see report).
    if getattr(args, "graph_fused", False):
        if device.startswith("cuda"):
            from .graphed_fused_forward import install_graphed_fused_forward
            install_graphed_fused_forward(sparse, device, verbose=True)

    # -- K-SWEEP mode: reuse this ONE build to verify at each K, print the table
    #    (K vs amortized ms/step vs peak-VRAM vs forwards vs GPU-util). ---------
    if args.k_sweep:
        return _run_k_sweep(sparse, L, code, draft, device, args)

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

    # free the naive driver's (large, growing) KV cache before the FAST phase so its
    # VRAM is returned to the allocator — else the fast path starts with a full card
    # and OOMs on the first (pre-flatten) prune's stack.  (The naive baseline is only
    # measured for the per-step wall; its cache is not needed afterward.)
    del naive_tr, naive_stats
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # -- FAST path: whole-program draft + batched verify ---------------------
    evict_iv = args.evict_interval_steps
    print(f"  FAST: verify {draft.step_count} steps in blocks of "
          f"K={args.block_steps} (block_moe={args.block_moe}, "
          f"evict_interval_steps={evict_iv if evict_iv is not None else 'per-block'}) "
          f"...", flush=True)
    fast_out: List[int] = []
    fast_stats: dict = {}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    # sample GPU util during the verify (a lightweight nvidia-smi poller thread).
    util_sampler = _GpuUtilSampler(device) if args.gpu_util else None
    if util_sampler:
        util_sampler.start()
    t = time.time()
    try:
        vr = verify_blocks(sparse, L, code, draft, block_steps=args.block_steps,
                           device=device, evict=(not args.no_evict), prune_interval=args.prune_interval,
                           mask=0xFFFFFFFF, stats=fast_stats, fast=True,
                           collect_out=fast_out, block_moe=args.block_moe,
                           evict_interval_steps=evict_iv, oom_backoff=True,
                           min_block_steps=args.min_block_steps)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        if util_sampler:
            util_sampler.stop()
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"  FAST: OOM even at min_block_steps={args.min_block_steps} "
              f"(GPU too contended / span too large); retry with a smaller "
              f"--block-steps or on a freer GPU. detail: {str(e).splitlines()[0]}",
              flush=True)
        del sparse
        gc.collect()
        return 3
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_fast = time.time() - t
    if util_sampler:
        util_sampler.stop()
    fast_per_step = t_fast / max(draft.step_count, 1)
    print(f"  FAST: {vr.forwards} forwards in {t_fast:.1f}s "
          f"({t_fast/60:.2f} min) -> {fast_per_step*1000:.2f} ms/step-equiv  "
          f"| all_matched={vr.all_matched} accepted={vr.accepted_steps}/"
          f"{draft.step_count}", flush=True)
    print(f"  FAST: max_cache={fast_stats.get('max_cache_size')} "
          f"evicted={fast_stats.get('total_evicted')} "
          f"evict_rounds={fast_stats.get('evict_rounds')} "
          f"eff_K={fast_stats.get('effective_block_steps')} "
          f"peak_vram={fast_stats.get('peak_vram_gb', 0.0):.2f}GB", flush=True)
    print(f"  FAST: blocks_run={fast_stats.get('blocks_run')} "
          f"blocks_full={fast_stats.get('blocks_full')} "
          f"block_moe_speedup={fast_stats.get('block_moe_speedup', 1.0):.2f}x", flush=True)
    if fast_stats.get("split_active"):
        print(f"  FAST: DROP-KV split — eviction manages "
              f"{fast_stats.get('n_global_caches')} global caches of "
              f"{len(sparse.blocks)} blocks (vs all {len(sparse.blocks)} classic); "
              f"peak global cache {fast_stats.get('peak_global_cache')} rows, "
              f"local window rows/block <= {fast_stats.get('max_local_cache')}; "
              f"KV-row footprint {fast_stats.get('kv_rows_split')} vs classic full-H "
              f"{fast_stats.get('kv_rows_classic')} "
              f"({fast_stats.get('kv_row_reduction', 1.0):.1f}x fewer rows)",
              flush=True)
    if util_sampler:
        print(f"  FAST: GPU util mean={util_sampler.mean:.0f}% "
              f"max={util_sampler.max:.0f}% (n={util_sampler.n})", flush=True)
    # eviction wall FRACTION (the #667/#670 bottleneck instrumentation): the fused
    # on-GPU eviction should now be a small fraction of the fast wall (it was the
    # dominant cost with the per-block host-synced loop that stalled the GPU).
    t_ev = fast_stats.get("t_evict", 0.0)
    n_pr = fast_stats.get("n_prunes", 0)
    t_rest = max(t_fast - t_ev, 0.0)
    print(f"  FAST wall split: evict={t_ev:.1f}s "
          f"({100*t_ev/max(t_fast,1e-9):.1f}%, {n_pr} prunes, "
          f"{1000*t_ev/max(n_pr,1):.1f} ms/prune)  forward+overlay={t_rest:.1f}s "
          f"({100*t_rest/max(t_fast,1e-9):.1f}%)", flush=True)
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
    ap.add_argument("kind", choices=["mandel", "loop", "nested", "malloc", "matmul"])
    ap.add_argument("grid", nargs="*", type=int, default=[],
                    help="mandel: W H MAXITER")
    ap.add_argument("--n", type=int, default=64, help="loop/malloc/matmul size")
    ap.add_argument("--outer", type=int, default=40, help="nested: outer loop count")
    ap.add_argument("--inner", type=int, default=200, help="nested: inner loop count")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--compute-mode", type=str, default="dense_kernel",
                    choices=["dense_kernel", "sparse_mm"])
    ap.add_argument("--block-steps", type=int, default=256,
                    help="K = VM steps verified per batched forward (cranked; the "
                         "OOM-backoff halves it if a span overflows VRAM).")
    ap.add_argument("--block-moe", action=argparse.BooleanOptionalAction, default=True,
                    help="block-sparse conditional dispatch: skip inert (non-active-op) "
                         "block spans per step (DEFAULT ON; --no-block-moe for dense).")
    ap.add_argument("--naive-steps", type=int, default=50,
                    help="how many naive steps to time for the per-step baseline.")
    ap.add_argument("--prune-interval", type=int, default=60)
    ap.add_argument("--evict-interval-steps", type=int, default=None,
                    help="evict once per this many VM steps (default: once per "
                         "verify block).  The user's lever: with a big K this is "
                         "one eviction sweep per ~30k-token block.")
    ap.add_argument("--min-block-steps", type=int, default=4,
                    help="OOM-backoff floor for K.")
    ap.add_argument("--gpu-util", action="store_true",
                    help="sample nvidia-smi GPU utilisation during the verify.")
    ap.add_argument("--min-free-gb", type=float, default=6.0,
                    help="wait until the GPU has this much free VRAM before loading "
                         "the model (multi-agent-contention guard); 0 disables.")
    ap.add_argument("--wait-vram-s", type=float, default=600.0,
                    help="max seconds to wait for --min-free-gb before proceeding.")
    ap.add_argument("--k-sweep", type=str, default=None,
                    help="comma list of K values to sweep (e.g. 32,64,128,256,512,"
                         "1000); prints a K vs ms/step vs VRAM vs forwards table.")
    ap.add_argument("--no-evict", action="store_true",
                    help="disable KV eviction (bounded VRAM off; larger cache).")
    ap.add_argument("--max-steps", type=int, default=5_000_000)
    ap.add_argument("--sp-init", type=lambda s: int(s, 0), default=None,
                    help="override SP_INIT (e.g. 0xFC for deep recursion).")
    ap.add_argument("--local-window", type=int, default=None,
                    help="sliding-window (LOCAL) attention on the non-memory heads: "
                         "the ~20 ingest heads only read the last W tokens (O(S*W)), "
                         "the memory/stack/LEV KV heads stay GLOBAL (full causal). "
                         "Byte-identical (ingest weight past W is 0). Try 64 (~2 VM "
                         "steps). Off by default (full global attention).  DEFAULT "
                         "DROPS the local heads' old KV (cache ~Hg·S + Hl·W; the VRAM "
                         "lever) — pass --local-mask-only for the old mask-only mode.")
    ap.add_argument("--local-mask-only", action="store_true",
                    help="with --local-window: keep the FULL KV cache and only WINDOW "
                         "the local heads' softmax read (compute win, no VRAM win). "
                         "The fallback for the default DROP-KV behavior.")
    ap.add_argument("--live-head", action="store_true",
                    help="LIVE-HEAD-ONLY attention scoring: compute Q@Kᵀ+attn@V ONLY "
                         "for the ~23 live-value head-slots (of 5451); every other "
                         "_zero_attn head's per-head output is 0 by construction, so "
                         "its score matmul is skipped.  Byte-identical (a zero-value "
                         "head outputs x regardless of scores).  Cuts attn-scoring "
                         "FLOPs ~99.6%%; 234/237 blocks become pure x-passthrough. "
                         "Composes with --local-window.")
    ap.add_argument("--fuse-dead-blocks", action="store_true",
                    help="DEAD-BLOCK ATTENTION FUSION: bypass the ENTIRE attention "
                         "sublayer of the ~234 blocks with NO live-value head "
                         "(output = x directly; NO K/Q/V/W_o linears, NO KV write). "
                         "Implies --live-head classification (installs live-head "
                         "scoring on the ~3 live blocks).  --live-head alone only "
                         "skips the score matmul; this ALSO removes the per-block "
                         "linears (the measured wall) + KV materialisation on the "
                         "dead blocks.  Byte-identical: a dead block's attention "
                         "output is x and its KV is provably never read.")
    ap.add_argument("--materialize-dense", action="store_true",
                    help="Densify the CSR FFN/attn weights ONCE onto the device so "
                         ".linear stops re-running csr.to_dense() every forward (the "
                         "launch-bound wall at small span S).  Byte-identical (same "
                         "GEMM, L-inf=0).  Costs the full dense-model VRAM (~5.8 GB).")
    ap.add_argument("--graph-fused", action="store_true",
                    help="CUDA-GRAPH the dead-attention segments (requires "
                         "--fuse-dead-blocks).  Densifies the CSR FFN weights ONCE "
                         "(materialize_dense -> .linear stops re-running "
                         "csr.to_dense per call, the launch-bound wall) and captures "
                         "each contiguous dead-block FFN segment into a CUDA graph "
                         "replayed per verify-block; the ~3 live-attention blocks "
                         "stay eager.  Collapses the ~705 tiny-GEMM kernel LAUNCHES "
                         "per forward to a few graph launches.  Byte-identical "
                         "(graphed dead segment == eager fused chain, L-inf=0).")
    ap.add_argument("--block-sparse-ffn", action="store_true",
                    help="STATIC BLOCK-SPARSE FFN: replace the wasteful dense FFN "
                         "GEMMs (99.93%% sparse) with a static gather-scale-scatter "
                         "(COO) over the fixed nonzero pattern — few-FLOP, no zero-"
                         "multiply. ~2x faster than the dense FFN GEMM at production "
                         "span S~=961 (the scattered nonzeros keep it above the FLOP "
                         "floor). Byte-exact up to fp-reduction order (rel ~1e-7 << "
                         "decode margin). Compose with --materialize-dense.")
    ap.add_argument("--block-sparse-mode", default="coo",
                    choices=["coo", "dense_active"],
                    help="block-sparse FFN form: coo (gather-scale-scatter, few-FLOP)"
                         " or dense_active (compact dense sub-block, ~1%% useful).")
    ap.add_argument("--content-bound-global", action="store_true",
                    help="with --local-window (DROP-KV): bound the GLOBAL "
                         "(memory/stack/LEV) cache BY CONTENT — each global head is "
                         "an address-CAM that keys ONLY store frames (a non-store "
                         "frame's role-gate key -> softmax1 weight 0), so its cache "
                         "keeps only store rows, bounded by the WORKING SET (distinct "
                         "live addresses, after latest-write-wins eviction), NOT step "
                         "count.  Byte-identical.  Makes the TOTAL KV runtime-"
                         "independent (the 2.4M-step self-emulation at constant mem).")
    args = ap.parse_args(argv)
    if args.kind == "mandel":
        if len(args.grid) < 3:
            args.grid = (args.grid + [1, 1, 3])[:3]
        args.grid = tuple(args.grid[:3])
    return run_bench(args.kind, args)


if __name__ == "__main__":
    raise SystemExit(main())
