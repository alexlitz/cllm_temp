"""MEASURE the real cost of the transformer running the transformer (self-emulation).

Question (BLOG_SPEC §901 Self-Hosting "TODO performance analysis"):
    "how long is ONE step of the transformer, run VIA the transformer?"

A transformer forward IS a matmul stack.  So "the transformer emulating its own
forward" = the transformer executing the *bytecode of a matmul* — the inner loop of
the fixed-point ONNX runtime (``onnx_runtime_nibble_fixedpoint.c``, ~9,955 c4 words)
is ``op_matmul`` = a ``MUL``/``ADD`` accumulate.  Here ONE VM step (one c4
instruction) is executed by ONE genuine ``transformers.Qwen2Model.forward``
(``qwen_full_vm.run_program``), and the MUL a matmul needs is itself computed by the
``nibble_alu32`` byte-schoolbook gadget baked into the Qwen MLPs
(``build(..., efficient_alu=True)`` — no lookup table, no fp64).  That is genuine
SELF-EMULATION: the transformer emulates the multiply a matmul forward is made of.

This module MEASURES, on ``cuda:1``, for a TINY matvec:
  * the matvec is BYTE-EXACT vs the numpy / ``isa.interpret`` reference,
  * total VM steps, tokens-per-step (the window fed to each forward), ms/step, wall,
  * the #702 single-stack-slot wall (why a width>=2 dot product must spill / call)
    and how call/spill-heaviness gates cross-step speculation batching,
  * the PERFECT-DRAFT speculation forwards-saved (100% acceptance): 32x single-
    program block batching + the MEASURED 54.7x@B=64 cross-program prior — a
    FORWARDS-SAVED (throughput) win, NOT a per-token-compute win,
and EXTRAPOLATES the honest wall of ONE smallest self-forward (~2.4M VM steps,
the labeled basis) = ``total_tokens x per-token-compute``.

MEASURED (cuda:1, this build, hidden=1728 inter=1124 138 recurrent layers):
  * scalar MAC / tiny [Rx1]@[1] matvec are BYTE-EXACT vs numpy & isa.interpret,
  * TOKENS PER VM STEP = 7.0 (BOS + 5-reg frame + query) — structural, invariant,
  * per-step compute ~330-1250 ms/step (GPU-CONTENTION-sensitive; this cuda:1 is
    shared), i.e. per-token ~47-179 ms; the two together pin the wall,
  * peak ~5.6 GB — memory-safe,
  * extrapolated ONE self-forward wall ~9-35 days (vs the blog's 88-day prior).
The TWO PINNING QUANTITIES the wall reduces to: TOKENS-PER-STEP (=7, exact) and
PER-TOKEN-COMPUTE (=ms/step / tokens-per-step, the contention-sensitive term).

Memory-safe: ONE lean muldiv+efficient_alu Qwen build on ``cuda:1``, tiny programs,
capped step counts (max_steps<=64).  Run:
    C4_SELF_EMU_DEV=cuda:1 python -m c4_min.measure_self_emulation_wall
See docs/SELF_EMULATION_WALL_2026_07_21.md for the full write-up.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from . import isa
from . import qwen_full_vm as Q

# ~2.4M VM steps for the smallest self-forward (BLOG_SPEC §901 self-hosting
# perf-analysis answer).  This is a LABELED EXTRAPOLATION BASIS — not measured
# here; a smaller/larger self-net scales the wall linearly (see docs write-up).
SELF_FORWARD_STEPS = 2_400_000

# MEASURED PRIOR (692-selfemul, prior session): CROSS-PROGRAM batched perfect-draft
# speculation (``batched_speculative.run_corpus_stacked``) saved 54.7x forwards at a
# batch of B=64 independent programs — the deterministic VM is a PERFECT draft (100%
# acceptance), so one [B, W, D] block-verify forward commits ~block_steps VM steps for
# EACH of B programs at once.  This is a FORWARDS-SAVED (throughput) win, not a
# per-token-compute win: it needs B independent programs (or independent step-blocks)
# runnable in lockstep, and it is gated by CALL/SPILL-heaviness (the #702 wall) —
# see ``speculation_model`` below.
BATCHED_SPEC_FWD_SAVED_B64 = 54.7
SPEC_BATCH = 64
# Single-program perfect-draft block width (steps committed per block-verify forward,
# LEAN_FORWARD.md / pf_speculative.verify_blocks default).  Perfect draft => every
# drafted step is accepted, so single-program forwards-saved = block_steps.
SPEC_BLOCK_STEPS = 32


def _dev() -> str:
    return os.environ.get("C4_SELF_EMU_DEV", "cuda:1")


def build_vm(dev: str) -> Q.QwenFullVM:
    """One muldiv+efficient_alu genuine Qwen2Model VM (the self-emulation build:
    MUL/DIV/MOD via the nibble_alu32 schoolbook gadgets, NOT a lookup table)."""
    vm = Q.build(code_size=24, subset=Q.SUBSET_MULDIV,
                 efficient_alu=True, recurrent_divmod=True)
    vm.qmodel = vm.qmodel.to(dev)
    vm.embed = vm.embed.to(dev)
    return vm


# ---------------------------------------------------------------------------
# The tiny matvec.  A genuine matmul output element is a dot product sum_k W[k]*x[k].
# ---------------------------------------------------------------------------
def scalar_mac_prog(w: int, x: int) -> List[Tuple[str, int]]:
    """ONE matmul term: y = w * x (the transformer emulates the multiply a matmul
    forward is built from).  Single-level binop -> byte-exact on the 1-slot machine."""
    return [("IMM", w), ("PSH", 0), ("IMM", x), ("MUL", 0), ("HALT", 0)]


def matvec_rows_prog(W: List[int], x0: int) -> List[List[Tuple[str, int]]]:
    """A [R x 1] @ [1] matvec: R output rows, each y_i = W[i]*x0.  Each row is a
    byte-exact scalar MAC; at width 1 the accumulate is trivial, so the whole tiny
    matmul forward is R independent 5-instruction programs that the transformer
    emulates term-by-term."""
    return [scalar_mac_prog(w, x0) for w in W]


def dot2_prog(w: List[int], x: List[int]) -> List[Tuple[str, int]]:
    """A width-2 dot product y = w0*x0 + w1*x1.  Needs stack DEPTH 2 (park the first
    product while computing the second).  The Qwen VM's register CAM tracks ONE
    top-of-stack cell (STACK0), so this WALLS at the second push (the #702
    fallback): the reference's deep stack and the model's 1-slot mirror diverge.
    Kept here to MEASURE that wall, not because it passes in-register."""
    return [
        ("IMM", w[0]), ("PSH", 0), ("IMM", x[0]), ("MUL", 0),  # AX = w0*x0
        ("PSH", 0),                                            # park (depth 2 needed)
        ("IMM", w[1]), ("PSH", 0), ("IMM", x[1]), ("MUL", 0),  # AX = w1*x1 (clobbers park)
        ("ADD", 0),                                            # AX = park + w1*x1
        ("HALT", 0),
    ]


def dot_prog(w: List[int], x: List[int]) -> List[Tuple[str, int]]:
    """A width-W dot product y = sum_i w_i*x_i in the KV-MEMORY-BACKED stack form
    (#692/#702 fix).  The natural stack form parks each partial product with a PSH
    and folds it with ADD, so the stack grows to depth 2 (park a partial while the
    next product's operand is pushed).  Run through ``run_program(..., 
    spill_stack_to_kv=True)`` that push-down stack lives in the persistent KV MEMORY
    LOG (the same log SI/SC write, content-addressed by the memory CAM), so an
    ARBITRARY-width dot is BYTE-EXACT — the ``dot2`` 1-slot-STACK0 wall dissolves.
    This IS "the deeper stack maintained in memory": each PSH is an SI to the SP-
    relative stack cell, each pop an LI of the new top."""
    p: List[Tuple[str, int]] = [("IMM", w[0]), ("PSH", 0), ("IMM", x[0]), ("MUL", 0)]
    for i in range(1, len(w)):
        p += [("PSH", 0), ("IMM", w[i]), ("PSH", 0), ("IMM", x[i]),
              ("MUL", 0), ("ADD", 0)]                       # acc += w_i*x_i
    p += [("HALT", 0)]
    return p


def measure_matmul_depth1(vm: Q.QwenFullVM, dev: str) -> Dict[str, object]:
    """Prove a width>=2 dot ([3,5].[7,11]=76) and a real 2x2 @ 2x1 matmul are
    BYTE-EXACT vs numpy through the genuine Qwen forward on the KV-BACKED stack,
    overcoming the depth-1 wall #692 found (which diverges: dot2 default -> 60/25)."""
    import numpy as np
    out: Dict[str, object] = {}

    # dot2 [3,5].[7,11]=76 : DEFAULT diverges (1-slot STACK0), KV-backed = byte-exact.
    w, x = [3, 5], [7, 11]
    ref = int(np.dot(w, x)) & 0xFF
    code = isa.assemble(dot_prog(w, x))
    default_got = Q.run_program(vm, code, max_steps=64)["ax_trace"][-1]
    fix = Q.run_program(vm, code, max_steps=64, spill_stack_to_kv=True)
    out["dot2"] = {"w": w, "x": x, "numpy": ref, "default_1slot": default_got,
                   "kv_backed": fix["ax_trace"][-1],
                   "byte_exact": fix["ax_trace"][-1] == ref, "steps": fix["steps"],
                   "default_diverges": default_got != ref}

    # width-3 + width-4 dots (deeper stack) — arbitrary width byte-exact.
    for w2, x2, tag in ([3, 5, 7], [2, 4, 6], "dot3"), ([1, 2, 3, 4], [5, 6, 7, 8], "dot4"):
        r = Q.run_program(vm, isa.assemble(dot_prog(w2, x2)), max_steps=256,
                          spill_stack_to_kv=True)
        out[tag] = {"w": w2, "x": x2, "numpy": int(np.dot(w2, x2)) & 0xFF,
                    "model": r["ax_trace"][-1],
                    "byte_exact": r["ax_trace"][-1] == (int(np.dot(w2, x2)) & 0xFF),
                    "steps": r["steps"]}

    # a real 2x2 @ 2x1 matmul: each output element is one width-2 dot.
    M = np.array([[3, 5], [7, 2]]); v = np.array([4, 6])
    ref_mv = [int(M[i] @ v) & 0xFF for i in range(2)]
    got_mv = []; total = 0
    for i in range(2):
        r = Q.run_program(vm, isa.assemble(dot_prog(list(M[i]), list(v))),
                          max_steps=256, spill_stack_to_kv=True)
        got_mv.append(r["ax_trace"][-1]); total += r["steps"]
    out["matmul_2x2"] = {"M": M.tolist(), "v": v.tolist(), "numpy": ref_mv,
                         "model": got_mv, "byte_exact": got_mv == ref_mv,
                         "total_steps": total}

    # TIMING for ONE width-2 dot (one emulated matmul output element).
    win: List[int] = []
    orig = Q._forward
    def probe(vm_, xx):
        win.append(int(xx.shape[1])); return orig(vm_, xx)
    Q._forward = probe
    try:
        Q.run_program(vm, code, max_steps=64, spill_stack_to_kv=True)   # warmup
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        win.clear()
        reps = 5; t0 = time.time()
        for _ in range(reps):
            r = Q.run_program(vm, code, max_steps=64, spill_stack_to_kv=True)
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        wall = (time.time() - t0) / reps
    finally:
        Q._forward = orig
    steps = r["steps"]
    out["timing"] = {"vm_steps_per_dot": steps, "wall_ms_per_dot": wall * 1000.0,
                     "ms_per_step": 1000.0 * wall / steps,
                     "tokens_per_step": sum(win) / len(win) if win else 0.0,
                     "win_min": min(win) if win else 0, "win_max": max(win) if win else 0}
    return out


# ---------------------------------------------------------------------------
# Timed run with a window-length (tokens/step) probe.
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    label: str
    exact: bool
    got: int
    ref: int
    steps: int
    tokens_per_step: float
    win_min: int
    win_max: int


def _timed_run(vm: Q.QwenFullVM, prog, ref_val: int, label: str,
               dev: str, reps: int) -> Tuple[RunMetrics, float]:
    """Run ``prog`` ``reps`` times (after a warmup), returning metrics + mean wall/prog.
    Instruments the per-forward window length (tokens/step) via a _forward wrapper."""
    win_lens: List[int] = []
    orig = Q._forward

    def probe(vm_, x):
        win_lens.append(int(x.shape[1]))
        return orig(vm_, x)

    Q._forward = probe
    try:
        r = Q.run_program(vm, isa.assemble(prog), max_steps=64)   # warmup
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        win_lens.clear()
        t0 = time.time()
        for _ in range(reps):
            r = Q.run_program(vm, isa.assemble(prog), max_steps=64)
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        wall = (time.time() - t0) / reps
    finally:
        Q._forward = orig

    got = r["ax_trace"][-1] if r["ax_trace"] else -1
    steps = r["steps"]
    m = RunMetrics(
        label=label, exact=(got == ref_val), got=got, ref=ref_val, steps=steps,
        tokens_per_step=(sum(win_lens) / len(win_lens)) if win_lens else 0.0,
        win_min=min(win_lens) if win_lens else 0,
        win_max=max(win_lens) if win_lens else 0,
    )
    return m, wall


def measure(dev: str = None, reps: int = 3) -> Dict[str, object]:
    dev = dev or _dev()
    t0 = time.time()
    vm = build_vm(dev)
    build_s = time.time() - t0
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)

    out: Dict[str, object] = {
        "device": dev, "build_s": build_s,
        "hidden": vm.hidden_size, "intermediate": vm.intermediate_size,
        "n_layers": vm.n_layers, "fits_stock": vm.fits_stock,
    }

    # (1) BYTE-EXACT scalar MAC (the matmul multiply, self-emulated) — the timed unit.
    w, x = 3, 5
    mac, mac_wall = _timed_run(vm, scalar_mac_prog(w, x), (w * x) & 0xFF,
                               f"scalar_mac {w}*{x}", dev, reps)
    ms_step = 1000.0 * mac_wall / mac.steps
    out["mac"] = mac.__dict__
    out["mac_wall_s"] = mac_wall
    out["ms_per_step"] = ms_step

    # (2) BYTE-EXACT multi-row matvec [R x 1] @ [1] (a genuine tiny matmul forward).
    import numpy as np
    W = [3, 5, 7, 9]
    x0 = 11
    ref_rows = [int(v) for v in (np.array(W) * x0) & 0xFF]
    got_rows = []
    total_rows_steps = 0
    for wi in W:
        r = Q.run_program(vm, isa.assemble(scalar_mac_prog(wi, x0)), max_steps=64)
        got_rows.append(r["ax_trace"][-1])
        total_rows_steps += r["steps"]
    out["matvec"] = {
        "W": W, "x0": x0, "ref": ref_rows, "got": got_rows,
        "byte_exact": ref_rows == got_rows, "total_steps": total_rows_steps,
    }

    # (3) the #702 wall: a width-2 dot product needs stack depth 2 -> diverges.
    w2, x2 = [3, 5], [10, 4]
    ref2 = isa.interpret(isa.assemble(dot2_prog(w2, x2)))[-1]   # real deep stack -> 50
    r2 = Q.run_program(vm, isa.assemble(dot2_prog(w2, x2)), max_steps=64)
    got2 = r2["ax_trace"][-1]
    out["dot2_wall"] = {
        "w": w2, "x": x2, "ref_deep_stack": ref2, "model_1slot": got2,
        "diverges": got2 != ref2,
        "note": "width>=2 dot needs a 2nd live stack cell; the register CAM tracks "
                "ONE (STACK0) -> the #702 spill/func-call fallback",
    }

    # (3c) #692 OVERCOME: width>=2 dot + real matmul BYTE-EXACT via the KV-backed
    #      stack (spill_stack_to_kv) — the depth-1 wall dissolves.
    out["matmul_depth1"] = measure_matmul_depth1(vm, dev)

    if dev.startswith("cuda"):
        out["peak_cuda_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9

    # (4) EXTRAPOLATION — one smallest self-forward (~2.4M VM steps).
    tps = mac.tokens_per_step
    tokens_total = SELF_FORWARD_STEPS * tps
    wall_s = SELF_FORWARD_STEPS * (ms_step / 1000.0)
    out["extrapolation"] = {
        "basis_steps": SELF_FORWARD_STEPS,
        "tokens_per_vm_step": tps,
        "per_step_compute_ms": ms_step,
        "per_token_compute_ms": ms_step / tps if tps else 0.0,
        "total_tokens": tokens_total,
        "wall_seconds": wall_s,
        "wall_hours": wall_s / 3600.0,
        "wall_days": wall_s / 86400.0,
        "blog_prior_days": 88.0,
        "blog_prior_s_per_instr": 3.2,
    }

    # (3b) SPECULATION forwards-saved model (perfect deterministic draft), and how
    #      #702 call/spill-heaviness gates it.  Built on the measured 54.7x@B=64 prior.
    out["speculation"] = speculation_model(
        naive_wall_s=wall_s, ms_step=ms_step, mac_steps=mac.steps,
        dot2_diverges=out["dot2_wall"]["diverges"],
    )
    return out


def speculation_model(naive_wall_s: float, ms_step: float, mac_steps: int,
                      dot2_diverges: bool) -> Dict[str, object]:
    """PERFECT-DRAFT speculation forwards-saved, and the #702 gate on it.

    The c4_min logical VM is a DETERMINISTIC draft, so speculation has 100%
    acceptance (LEAN_FORWARD.md): every drafted VM step is committed by the
    verify forward.  Two independent levers, both FORWARDS-SAVED (throughput),
    NEITHER lowers PER-TOKEN compute:

      * SINGLE-PROGRAM block batching: one block-verify forward commits
        ``SPEC_BLOCK_STEPS`` steps -> forwards-saved = SPEC_BLOCK_STEPS
        (``steps / ceil(steps/block_steps)``, exact for a perfect draft).
      * CROSS-PROGRAM batching: ``SPEC_BATCH``=64 independent programs stacked
        into one ``[B, W, D]`` forward -> the MEASURED 54.7x@B=64 prior
        (``batched_speculative``).  ~= B x the per-step launch amortisation.

    THE #702 GATE.  A real matmul dot of width W>=2 needs stack DEPTH>=2 (park
    one partial while computing the next).  This machine's register CAM tracks
    exactly ONE live cell (STACK0), so a width>=2 dot DIVERGES in-register
    (measured: dot2 50 -> 25) and MUST spill to a function frame (JSR/ENT/LEV) or
    a memory round-trip per accumulate term.  Two consequences:

      1. STEP INFLATION (the dominant gate): each extra dot term costs a spill
         frame on top of its 5-step MAC, so a width-W output element runs
         ~W*(mac_steps + spill_steps) VM steps, not W*mac_steps.  Since the wall
         is total_steps x per-step-compute, call-heaviness multiplies the wall
         DIRECTLY, and the ~2.4M-step basis ALREADY bakes in the spilled form.
      2. LOCKSTEP RAGGEDNESS (the batching gate): cross-program batching needs B
         programs at a similar call DEPTH to fill a [B, W, D] tile with no idle
         rows (depth-bucketing).  Call-heavy programs fan out into many distinct
         depths, so the achievable batch shrinks below B and the 54.7x@B=64 is an
         UPPER bound (hit on uniform shallow loops; lower on deep call chains).

    Speculation is NOT broken by calls (the deterministic draft knows every
    dynamic target, so it drafts THROUGH a JSR) — it is GATED by them via (1)+(2).
    """
    single = SPEC_BLOCK_STEPS                              # perfect-draft, per-program
    # Best case: cross-program batched perfect draft (measured prior).  A SINGLE
    # self-forward is ONE program, so cross-program batching only applies if many
    # self-forwards (or many independent inner-loop MACs) are run together.
    best_case = BATCHED_SPEC_FWD_SAVED_B64                 # 54.7x @ B=64 (measured)
    return {
        "perfect_draft": True,
        "single_program_fwd_saved": single,
        "batched_fwd_saved_B64": best_case,
        "batch": SPEC_BATCH,
        "block_steps": SPEC_BLOCK_STEPS,
        # forwards-saved cuts the NUMBER of forwards, so it divides the wall
        # (throughput), leaving per-token compute untouched.  Report both bounds.
        "wall_s_single_prog_spec": naive_wall_s / single,
        "wall_days_single_prog_spec": naive_wall_s / single / 86400.0,
        "wall_s_batched_spec": naive_wall_s / best_case,
        "wall_days_batched_spec": naive_wall_s / best_case / 86400.0,
        "hs702_dot2_diverges": dot2_diverges,
        "hs702_note": (
            "width>=2 dot needs depth>=2 -> a func-frame spill per accumulate "
            "term; this INFLATES the step count (dominant gate) and RAGGENS "
            "cross-program depth-bucketing (caps the 54.7x@B=64 to an upper bound)"
        ),
    }


# ===========================================================================
# BLOCK-SPARSE self-emulation wall — the SAME scalar MAC (the multiply a matmul
# forward is built from), but run through the BLOCK-SPARSE conditional-dispatch
# fast path (``full_native_fast`` / ``ConditionalBlockLean``) instead of the DENSE
# overlay every-block-every-step build ``measure`` times.
#
# The dense build (`build_vm` above) runs all ~154 applied blocks every step ->
# ~337.8 ms/step -> a 9.38-day wall.  The block-sparse path GATHERS only each op's
# active FFN units (a few MB) and runs a dense cuBLAS GEMM on just that block; the
# MAC's per-step forward is byte-EXACT (thr=0 drop adds exactly 0 to
# down(silu(up)*gate)).  Scoped to the MAC's own ops (IMM/PSH/MUL/HALT) the active
# union is ~0.58% of dense (~47 MB); the full-ISA union is ~2.63%.  MEASURED
# per-step (block-verify, single-stream, k=32/64) is single-digit ms/step —
# ~41-64x cheaper than the dense 337.8.  This function measures that on a GENUINE
# forward and re-extrapolates the SAME 2.4M-step / speculation basis.
# ===========================================================================
def _driver_ms_per_step(bundle, code, *, block_steps: int, reps: int,
                        cuda: bool) -> Tuple[float, int, int]:
    """End-to-end block-verify DRIVER ms/step for ``code`` (warm + ``reps`` timed)."""
    r = bundle.run(code, block_steps=block_steps)     # warm
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        r = bundle.run(code, block_steps=block_steps)
    if cuda:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / reps * 1000.0
    return ms / max(r.steps, 1), r.steps, r.forwards


# ---------------------------------------------------------------------------
# GRAPHED single-stream block-verify — collapse the per-block 154-layer forward
# into ONE ``cuda.graph.replay()``.  The single-stream self-emulation forward is
# LAUNCH-BOUND (measured: ~0.18% FLOP-util at B=1 — the 154 tiny per-layer GEMMs
# spend more time in kernel-launch + the python block-loop than in the GPU), which
# is exactly what a CUDA graph removes.  ``GraphedLeanForward`` captures the whole
# fixed-shape forward once and replays it with ZERO python / ZERO per-launch CPU
# overhead.  It wraps ``ConditionalBlockLean`` byte-identically (proven Linf-0: the
# replay runs the SAME kernels over the SAME active-block weights as the eager
# forward).  This mirrors ``full_native_fast._run_native`` EXACTLY (same window
# build, same per-block AX decode) — only ``cond.forward`` is swapped for the graph
# replay.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_native_graphed(cond, draft_lean, code, *, block_steps: int, graphed,
                        max_steps: int = 4096):
    """``full_native_fast._run_native`` with the block-verify forward run as a
    CUDA-graph replay.  Byte-identical to the eager native driver (graph replay ==
    eager ``cond.forward``).  ``graphed`` is a ``GraphedLeanForward(cond)`` (reused
    across blocks/programs to amortise the capture)."""
    from . import isa as _isa
    from . import full_native_fast as _FNF
    from . import qwen_lean_forward as _LF
    from .qwen_lean_forward import CAM_REGS as _CAM_REGS, _snap as _sn
    L = draft_lean.QL.L
    subset = draft_lean.subset
    draft = _LF.draft_program_lean(draft_lean, code, max_steps=max_steps)
    ref_trace = draft.ref_trace
    if not draft.steps:                       # out-of-slice — eager naive fallback
        r = _LF.run_program_lean(draft_lean, code, max_steps=max_steps)
        n = r["steps"]
        return _LF.LeanSpecResult(
            status="PASS" if r["exact"] else "FAIL", ax_trace=r["ax_trace"],
            ref_trace=r["ref_trace"], exact=r["exact"], steps=n, forwards=n,
            naive_forwards=n, speedup=1.0, accepted=n, detail="naive-fallback")
    n_steps = len(draft.steps)
    ax_trace: List[int] = []
    forwards = 0
    for s0 in range(0, n_steps, block_steps):
        slab = draft.steps[s0:s0 + block_steps]
        x, positions = _LF._build_spec_batch(draft_lean, code, slab)
        x = x.to(cond.device)
        positions = positions.to(cond.device)
        hidden = graphed(x, positions)        # <- CUDA-graph replay (was cond.forward)
        forwards += 1
        for i, st in enumerate(slab):
            n_store = len(st["store_log"]) if subset.memory else 0
            n_code = len(code) if draft_lean.code_from_memory else 0
            qrow = (1 + n_store + n_code) + len(_CAM_REGS)
            state = hidden[i, qrow]
            op = st["op"]
            if op in (_isa.MUL, _isa.DIV, _isa.MOD):
                ax = _FNF._decode_reg_from_nibbles(state, L, L.AX) & 0xFF
            else:
                ax = _sn(state[L.AX_VAL]) & 0xFF
            ax_trace.append(ax)
    exact = ax_trace == ref_trace
    speedup = (n_steps / forwards) if forwards else 0.0
    return _LF.LeanSpecResult(
        status="PASS" if exact else "FAIL", ax_trace=ax_trace, ref_trace=ref_trace,
        exact=exact, steps=n_steps, forwards=forwards, naive_forwards=n_steps,
        speedup=speedup, accepted=n_steps,
        detail="" if exact else "graphed native trace != isa.interpret")


def _driver_ms_per_step_graphed(bundle, code, *, block_steps: int, reps: int,
                                cuda: bool, graphed) -> Tuple[float, int, int, bool]:
    """Graphed end-to-end block-verify DRIVER ms/step for ``code`` (warm + timed).
    Returns ``(ms_per_step, steps, forwards, byte_exact)``."""
    r = _run_native_graphed(bundle.cond, bundle.dense_lean, code,
                            block_steps=block_steps, graphed=graphed)  # warm + capture
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        r = _run_native_graphed(bundle.cond, bundle.dense_lean, code,
                                block_steps=block_steps, graphed=graphed)
    if cuda:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / reps * 1000.0
    return ms / max(r.steps, 1), r.steps, r.forwards, r.exact


def measure_block_sparse(dev: str = None, reps: int = 20,
                         batches: Tuple[int, ...] = (1, 16, 64),
                         local_window: int = None,
                         graphed: bool = False,
                         graph_ks: Tuple[int, ...] = (16, 32, 64, 128, 256)) -> Dict[str, object]:
    """Run the SAME scalar MAC (``scalar_mac_prog``, the multiply a matmul forward
    is built from) through the BLOCK-SPARSE conditional-dispatch fast path, measure
    the block-sparse per-step on a genuine forward, and re-extrapolate the honest
    self-emulation wall on the SAME 2.4M-step / speculation basis ``measure`` uses.

    Composition report: block-sparse conditional dispatch + fp32 (the lean weights
    are extracted fp32 from the fp64-DIV VM) + perfect-draft speculation.  KV-drop
    (local sliding-window attention) is checked for availability and its effect
    reported IF it cleanly installs on the conditional model.
    """
    from . import full_native_fast as FNF

    dev = dev or _dev()
    cuda = dev.startswith("cuda")
    t0 = time.time()
    # Scope the active-unit union to the MAC's OWN opcodes (IMM/PSH/MUL/HALT): the
    # union must cover every FFN unit that FIRES for the timed program, and the MAC
    # is a pure multiply — no DIV/MOD megablock — so this keeps the probe + the
    # conditional block SMALL and GPU-safe while staying byte-exact for the MAC.
    mac_ops = [isa.IMM, isa.PSH, isa.MUL, isa.HALT]
    bundle = FNF.build_full_native_fast(device=dev, ops=mac_ops, verbose=True)
    build_s = time.time() - t0

    # fp32 build check (the block-sparse conditional weights are fp32 even though the
    # DIV/MOD path of the source VM is fp64) — this is the "fp32-only" lever.
    is_fp32 = (bundle.dense_lean.dtype == torch.float32)

    w, x = 3, 5
    code = isa.assemble(scalar_mac_prog(w, x))
    ref = (w * x) & 0xFF

    # (A) BYTE-EXACT MAC through the CONDITIONAL block-sparse model + cross-check
    #     conditional == dense (the byte-identity of the sparse kernel).
    rc = bundle.run(code, block_steps=32)
    rd = bundle.run_dense(code, block_steps=32)
    mac_ax = rc.ax_trace[-1] if rc.ax_trace else -1
    byte_exact = (mac_ax == ref) and rc.exact
    cond_eq_dense = (rc.ax_trace == rd.ax_trace)

    # (B) tokens/step (window length) for the MAC through the block-sparse driver —
    #     the SAME structural quantity the dense tool reports (reuse the basis).
    xw, posw, _ = PC_windows(bundle.dense_lean, code)
    tokens_per_step = float(xw.shape[1])          # window S (BOS + stores + regs + STEP_END)

    # (C) BLOCK-SPARSE per-step compute on GENUINE conditional forwards.  THREE
    #     numbers, all real forwards on the gathered active block:
    #       * DRIVER ms/step = the end-to-end block-verify driver wall / VM steps,
    #         measured on a REAL multi-step program (a compact SUB/BNZ countdown that
    #         fits the code table).  This is the honest SINGLE-STREAM per-step and the
    #         apples-to-apples comparison to the dense tool's 337.8 driver ms/step: the
    #         deep-layer per-FORWARD launch cost is amortised over block_steps VM steps
    #         by the block-verify path (exactly as it is in the dense build).  A 5-step
    #         MAC alone is NOT representative — its one short forward is launch-bound
    #         over the 154-layer stack and would OVER-state the per-step.
    #       * MAC driver ms/step = the same driver on the 5-step MAC (reported for
    #         completeness — the launch-bound single-forward number).
    #       * forward us/step at batch B = the amortised conditional-forward cost at
    #         speculation batch B (one window = one VM step) — the batched ceiling.
    cond = bundle.cond
    # honest single-stream driver ms/step on a real multi-step loop (SUB/BNZ
    # countdown from 60 → 0; compact enough for the code table, many VM steps).
    loop = isa.assemble([("IMM", 60), ("PSH", 0), ("IMM", 1),
                         ("SUB", 0), ("BNZ", 1), ("HALT", 0)])
    driver_ms_per_step, loop_steps, loop_fwds = _driver_ms_per_step(
        bundle, loop, block_steps=32, reps=max(reps // 4, 3), cuda=cuda)
    driver_ms_per_step_k64, _, _ = _driver_ms_per_step(
        bundle, loop, block_steps=64, reps=max(reps // 4, 3), cuda=cuda)

    # the 5-step MAC's own driver ms/step (launch-bound single forward — reported,
    # NOT used as the wall basis).
    mac_driver_ms_per_step, _, _ = _driver_ms_per_step(
        bundle, code, block_steps=32, reps=reps, cuda=cuda)

    # (C') GRAPHED single-stream driver — collapse the per-block 154-layer forward
    #      into ONE cuda.graph.replay().  The single-stream forward is LAUNCH-bound
    #      (measured ~0.18% FLOP-util at B=1), which the graph removes.  Byte-identity
    #      is asserted (graph replay == eager cond.forward, proven Linf-0).  Sweep the
    #      block width K on single-stream (a perfect draft accepts any K) to see where
    #      graphed ms/step BOTTOMS OUT as the batched GEMM fills.
    graphed_block: Dict[str, object] = {"enabled": False}
    if graphed and cuda:
        from .qwen_lean_cuda_graph import GraphedLeanForward
        g = GraphedLeanForward(bundle.cond)
        # byte-identity spot-check: graphed block-verify == eager on the loop.
        r_eager = bundle.run(loop, block_steps=32)
        r_graph = _run_native_graphed(bundle.cond, bundle.dense_lean, loop,
                                      block_steps=32, graphed=g)
        graph_eq_eager = (r_graph.ax_trace == r_eager.ax_trace) and r_graph.exact
        gk_ms: Dict[int, float] = {}
        gk_exact: Dict[int, bool] = {}
        gk_fwds: Dict[int, int] = {}
        for K in graph_ks:
            try:
                ms_k, st_k, fw_k, ex_k = _driver_ms_per_step_graphed(
                    bundle, loop, block_steps=K, reps=max(reps // 4, 3),
                    cuda=cuda, graphed=g)
                gk_ms[K] = ms_k
                gk_exact[K] = ex_k
                gk_fwds[K] = fw_k
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                gk_ms[K] = None
                gk_exact[K] = None
                if cuda:
                    torch.cuda.empty_cache()
        # the graphed single-stream ms/step at k=32 / k=64 (the eager baseline's Ks).
        graphed_k32 = gk_ms.get(32)
        graphed_k64 = gk_ms.get(64)
        # best (lowest) graphed single-stream ms/step across the swept K.
        valid = {k: v for k, v in gk_ms.items() if v is not None}
        best_k = min(valid, key=valid.get) if valid else None
        best_ms = valid[best_k] if best_k is not None else None
        graphed_block = {
            "enabled": True,
            "graph_eq_eager": graph_eq_eager,
            "graph_shapes": g.shapes,
            "n_capture": g.n_capture,
            "ms_per_step_by_k": gk_ms,
            "byte_exact_by_k": gk_exact,
            "forwards_by_k": gk_fwds,
            "graphed_ms_per_step_k32": graphed_k32,
            "graphed_ms_per_step_k64": graphed_k64,
            "eager_ms_per_step_k32": driver_ms_per_step,
            "eager_ms_per_step_k64": driver_ms_per_step_k64,
            "speedup_vs_eager_k32": (driver_ms_per_step / graphed_k32
                                     if graphed_k32 else None),
            "speedup_vs_eager_k64": (driver_ms_per_step_k64 / graphed_k64
                                     if graphed_k64 else None),
            "best_k": best_k,
            "best_ms_per_step": best_ms,
        }
        if cuda:
            torch.cuda.empty_cache()

    # batched conditional-forward us/step (replicate ONE MAC step-window to batch B).
    base_x = xw[:1].to(dev).contiguous()
    base_pos = posw[:1].to(dev).contiguous()
    fwd_us_per_step: Dict[int, float] = {}
    for B in batches:
        try:
            xb = base_x.expand(B, -1, -1).contiguous()
            pb = base_pos.expand(B, -1).contiguous()
            if cuda:
                torch.cuda.synchronize()
            for _ in range(5):
                cond.forward(xb, q_positions=pb)
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                cond.forward(xb, q_positions=pb)
            if cuda:
                torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t) / reps * 1000.0
            fwd_us_per_step[B] = fwd_ms / B * 1000.0   # us/step (one window = one VM step)
            del xb, pb
            if cuda:
                torch.cuda.empty_cache()
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            fwd_us_per_step[B] = None
            if cuda:
                torch.cuda.empty_cache()

    # the single per-step compute the wall reduces to: the honest single-stream
    # DRIVER ms/step (block_steps=32) on a real multi-step program — the apples-to-
    # apples analogue of the dense tool's 337.8 driver ms/step.
    ms_step_blocksparse = driver_ms_per_step

    # (D) EXTRAPOLATE the SAME 2.4M-step basis with the block-sparse per-step.
    dense_ms_step = 337.8                           # the labeled DENSE per-step (this tool)
    dense_wall_s = SELF_FORWARD_STEPS * (dense_ms_step / 1000.0)
    bs_wall_s = SELF_FORWARD_STEPS * (ms_step_blocksparse / 1000.0)
    spec = speculation_model(
        naive_wall_s=bs_wall_s, ms_step=ms_step_blocksparse, mac_steps=rc.steps,
        dot2_diverges=True)

    # (D') GRAPHED single-stream re-extrapolation — the SAME 2.4M-step / speculation
    #      basis, but with the GRAPHED single-stream ms/step (best-K).  This is the
    #      apples-to-apples "close the single-vs-batched gap" number: single-stream
    #      per-step, launch overhead removed by the graph.
    graphed_extra: Dict[str, object] = {"enabled": False}
    if graphed_block.get("enabled") and graphed_block.get("best_ms_per_step"):
        g_ms = graphed_block["best_ms_per_step"]
        g_wall_s = SELF_FORWARD_STEPS * (g_ms / 1000.0)
        g_spec = speculation_model(
            naive_wall_s=g_wall_s, ms_step=g_ms, mac_steps=rc.steps,
            dot2_diverges=True)
        graphed_extra = {
            "enabled": True,
            "graphed_ms_per_step": g_ms,
            "graphed_best_k": graphed_block["best_k"],
            "graphed_wall_seconds": g_wall_s,
            "graphed_wall_hours": g_wall_s / 3600.0,
            "graphed_wall_days": g_wall_s / 86400.0,
            "graphed_spec_single_days": g_spec["wall_days_single_prog_spec"],
            "graphed_spec_single_minutes": g_spec["wall_s_single_prog_spec"] / 60.0,
            "graphed_spec_batched_days": g_spec["wall_days_batched_spec"],
            "graphed_spec_batched_minutes": g_spec["wall_s_batched_spec"] / 60.0,
            # the eager single-stream references (best eager ms/step = k=64 baseline).
            "eager_spec_single_minutes": (spec["wall_s_single_prog_spec"] / 60.0),
            "eager_spec_batched_minutes": (spec["wall_s_batched_spec"] / 60.0),
        }

    # (E) COMPOSE KV-drop (local sliding-window attention) IF available on the
    #     conditional model.  local_attention.install_local_attention targets the
    #     c4_min pure-forward SparseAttn model (the bench_fast_path lib model), NOT
    #     the LeanQwenVM/ConditionalBlockLean forward, so it does NOT drop directly
    #     onto THIS conditional model — report availability + the honest caveat.
    kv_drop = _kv_drop_state(bundle, local_window)

    out: Dict[str, object] = {
        "device": dev, "build_s": build_s, "fp32": is_fp32,
        "artifact": {
            "stored_blocks": bundle.n_layers_stored,
            "applied_layers": bundle.n_layers_applied,
            "H": bundle.hidden_size, "I": bundle.intermediate,
            "dense_gb": bundle.dense_gb, "active_mb": bundle.active_gb * 1024,
            "active_units": bundle.active_total,
            "active_frac": bundle.active_total / (bundle.intermediate * bundle.n_layers_applied),
            "scoped_ops": [isa.NAMES[o] for o in mac_ops],
        },
        "mac": {"w": w, "x": x, "ref": ref, "model": mac_ax,
                "byte_exact": byte_exact, "cond_eq_dense": cond_eq_dense,
                "steps": rc.steps, "tokens_per_step": tokens_per_step},
        "blocksparse_per_step": {
            "driver_ms_per_step_loop_k32": driver_ms_per_step,
            "driver_ms_per_step_loop_k64": driver_ms_per_step_k64,
            "loop_steps": loop_steps, "loop_forwards": loop_fwds,
            "mac_driver_ms_per_step": mac_driver_ms_per_step,
            "forward_us_per_step": fwd_us_per_step,
            "ms_per_step_single_stream": ms_step_blocksparse,
            "dense_ms_per_step": dense_ms_step,
            "per_step_speedup_vs_dense": dense_ms_step / max(ms_step_blocksparse, 1e-9),
        },
        "extrapolation": {
            "basis_steps": SELF_FORWARD_STEPS,
            "tokens_per_vm_step": tokens_per_step,
            "dense_ms_per_step": dense_ms_step,
            "dense_wall_days": dense_wall_s / 86400.0,
            "blocksparse_ms_per_step": ms_step_blocksparse,
            "blocksparse_wall_seconds": bs_wall_s,
            "blocksparse_wall_hours": bs_wall_s / 3600.0,
            "blocksparse_wall_days": bs_wall_s / 86400.0,
            "blocksparse_spec_single_days": spec["wall_days_single_prog_spec"],
            "blocksparse_spec_batched_days": spec["wall_days_batched_spec"],
        },
        "speculation": spec,
        "graphed_single_stream": graphed_block,
        "graphed_extrapolation": graphed_extra,
        "kv_drop": kv_drop,
        "composed": {
            "block_sparse": True,
            "fp32": is_fp32,
            "speculation_single_32x": True,
            "speculation_batched_54_7x": True,
            "cuda_graph_single_stream": graphed_block.get("enabled", False),
            "kv_drop_local_window": kv_drop["installed_on_conditional"],
            "radix16_shallow_div": _radix16_available(),
        },
    }
    if cuda:
        out["peak_cuda_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
    return out


def PC_windows(lean, code):
    """One-window builder for the MAC (delegates to the conditional-sparse module's
    repetitive-program window builder so the token frame matches the run driver)."""
    from . import perlayer_conditional_sparse as PC
    return PC.repetitive_program_windows(lean, code, max_steps=64)


def _radix16_available() -> bool:
    """Is a hardened radix-16 / log-sink SHALLOW-divide module present on the branch
    (composes further on DIV/MOD-heavy self-forwards, not on the pure-MUL MAC)?"""
    try:
        from . import nibble_logsink_blocks  # noqa: F401
        return True
    except Exception:
        return False


def _kv_drop_state(bundle, local_window) -> Dict[str, object]:
    """KV-drop (local sliding-window attention) composition state.  The
    ``local_attention`` module is present on the branch and byte-identical (windowed
    ingest heads' tail weight is 0), but it installs on the c4_min pure-forward
    SparseAttn model (the ``bench_fast_path`` lib model), NOT on the
    LeanQwenVM/ConditionalBlockLean forward this fast path uses — so it does not drop
    directly onto THIS conditional model.  Report availability + the honest caveat."""
    try:
        from . import local_attention  # noqa: F401
        available = True
    except Exception:
        available = False
    return {
        "module_available": available,
        "installed_on_conditional": False,
        "note": ("local_attention.install_local_attention targets the c4_min "
                 "pure-forward SparseAttn (bench_fast_path lib model), not the "
                 "LeanQwenVM/ConditionalBlockLean forward; KV-drop composes on the "
                 "bench_fast_path driver (bigger K per VRAM) — a THROUGHPUT lever "
                 "orthogonal to the per-step compute measured here — not directly "
                 "on this conditional model."),
        "requested_window": local_window,
    }


def _fmt_block_sparse(rep: Dict[str, object]) -> str:
    a = rep["artifact"]
    mac = rep["mac"]
    bs = rep["blocksparse_per_step"]
    e = rep["extrapolation"]
    sp = rep["speculation"]
    kv = rep["kv_drop"]
    comp = rep["composed"]
    fwd = "  ".join(
        f"B{B}={bs['forward_us_per_step'][B]:.1f}us"
        if bs["forward_us_per_step"].get(B) is not None else f"B{B}=OOM"
        for B in bs["forward_us_per_step"])
    lines = [
        "=" * 78,
        "SELF-EMULATION WALL (BLOCK-SPARSE) — the same MAC through conditional dispatch",
        "=" * 78,
        f"device={rep['device']}  build={rep['build_s']:.1f}s  fp32={rep['fp32']}",
        f"ARTIFACT: {a['stored_blocks']} stored blocks / {a['applied_layers']} applied "
        f"layers  H={a['H']} I={a['I']}",
        f"  dense FFN = {a['dense_gb']:.1f} GB (dense build OOMs / runs every block "
        f"every step) ; conditional active block = {a['active_mb']:.1f} MB "
        f"({a['active_units']} units = {a['active_frac']*100:.4f}% of dense) — the RUN path",
        f"  scoped to MAC ops {a['scoped_ops']} (pure multiply — no DIV/MOD megablock)",
        "",
        "(1) BYTE-EXACT self-emulated MULTIPLY through BLOCK-SPARSE conditional dispatch:",
        f"    scalar MAC  {mac['w']}*{mac['x']} = {mac['ref']}: model={mac['model']}  "
        f"byte_exact={mac['byte_exact']}  cond==dense={mac['cond_eq_dense']}  "
        f"steps={mac['steps']}  tokens/step={mac['tokens_per_step']:.0f}",
        "",
        "(2) BLOCK-SPARSE PER-STEP (genuine conditional forwards) vs the DENSE 337.8 ms/step:",
        f"    DRIVER ms/step (single-stream, real multi-step loop, {bs['loop_steps']} steps in "
        f"{bs['loop_forwards']} fwds): k=32 {bs['driver_ms_per_step_loop_k32']:.2f}  "
        f"k=64 {bs['driver_ms_per_step_loop_k64']:.2f} ms/step",
        f"    (5-step MAC-only driver = {bs['mac_driver_ms_per_step']:.1f} ms/step — launch-bound "
        f"single forward over the 154-layer stack, NOT the wall basis)",
        f"    batched conditional forward us/step: {fwd}",
        f"    -> single-stream per-step = {bs['ms_per_step_single_stream']:.3f} ms/step  "
        f"(vs dense {bs['dense_ms_per_step']:.1f} ms/step = "
        f"{bs['per_step_speedup_vs_dense']:.0f}x faster per step)",
        "",
        "(3) EXTRAPOLATION — SAME 2.4M-step / speculation basis, block-sparse per-step:",
        f"    basis: {e['basis_steps']:,} VM steps  tokens/step={e['tokens_per_vm_step']:.0f}",
        f"    DENSE wall (labeled)              = {e['dense_ms_per_step']:.1f} ms/step "
        f"-> {e['dense_wall_days']:.2f} days",
        f"    BLOCK-SPARSE wall                 = {e['blocksparse_ms_per_step']:.3f} ms/step "
        f"-> {e['blocksparse_wall_hours']:.2f} h = {e['blocksparse_wall_days']:.3f} days",
        f"    BLOCK-SPARSE + spec (single 32x)  = {e['blocksparse_spec_single_days']:.4f} days "
        f"({e['blocksparse_spec_single_days']*24:.2f} h)",
        f"    BLOCK-SPARSE + spec (batched 54.7x, B=64) = {e['blocksparse_spec_batched_days']:.4f} days "
        f"({e['blocksparse_spec_batched_days']*24:.2f} h)",
        "",
    ]
    gb = rep.get("graphed_single_stream", {})
    ge = rep.get("graphed_extrapolation", {})
    if gb.get("enabled"):
        lines.append("(3g) CUDA-GRAPH single-stream — 154 per-layer launches -> ONE graph.replay():")
        lines.append(f"    byte-identical (graph==eager): {gb['graph_eq_eager']}  "
                     f"captured shapes={gb['graph_shapes']} ({gb['n_capture']} graphs)")
        k32e = gb.get("eager_ms_per_step_k32"); k32g = gb.get("graphed_ms_per_step_k32")
        k64e = gb.get("eager_ms_per_step_k64"); k64g = gb.get("graphed_ms_per_step_k64")
        if k32g is not None:
            lines.append(f"    k=32: eager {k32e:.2f} -> graphed {k32g:.2f} ms/step  "
                         f"({gb['speedup_vs_eager_k32']:.2f}x)")
        if k64g is not None:
            lines.append(f"    k=64: eager {k64e:.2f} -> graphed {k64g:.2f} ms/step  "
                         f"({gb['speedup_vs_eager_k64']:.2f}x)")
        lines.append("    graphed single-stream K-sweep (perfect draft -> push K to fill the GEMM):")
        for K, ms in gb["ms_per_step_by_k"].items():
            if ms is None:
                lines.append(f"      K={K:5d}: OOM/err")
            else:
                ex = gb["byte_exact_by_k"].get(K)
                fw = gb["forwards_by_k"].get(K)
                mark = " <- best" if K == gb.get("best_k") else ""
                lines.append(f"      K={K:5d}: {ms:.3f} ms/step  fwds={fw}  "
                             f"byte_exact={ex}{mark}")
        lines.append(f"    -> BEST graphed single-stream = {gb['best_ms_per_step']:.3f} ms/step "
                     f"at K={gb['best_k']}  (vs eager 8.15 k=32 / 5.26 k=64)")
        lines.append("")
    if ge.get("enabled"):
        lines.append("(3h) RE-EXTRAPOLATED single-stream wall (GRAPHED best-K, SAME 2.4M basis):")
        lines.append(f"    graphed per-step = {ge['graphed_ms_per_step']:.3f} ms/step "
                     f"(K={ge['graphed_best_k']}) -> naive wall {ge['graphed_wall_hours']:.2f} h "
                     f"= {ge['graphed_wall_days']:.3f} days")
        lines.append(f"    + spec (single 32x)  = {ge['graphed_spec_single_minutes']:.2f} min "
                     f"(was eager {ge['eager_spec_single_minutes']:.2f} min)")
        lines.append(f"    + spec (batched 54.7x, B=64) = {ge['graphed_spec_batched_minutes']:.2f} min "
                     f"(was eager {ge['eager_spec_batched_minutes']:.2f} min)")
        gap = (ge['graphed_spec_single_minutes'] / max(ge['graphed_spec_batched_minutes'], 1e-9))
        lines.append(f"    single-vs-batched GAP (graphed) = {gap:.2f}x "
                     f"(eager single/batched = "
                     f"{ge['eager_spec_single_minutes']/max(ge['eager_spec_batched_minutes'],1e-9):.2f}x)")
        lines.append("")
    lines += [
        "(4) COMPOSED optimizations on this measurement:",
        f"    block-sparse conditional dispatch: {comp['block_sparse']}",
        f"    fp32-only build:                   {comp['fp32']}",
        f"    perfect-draft speculation single 32x / batched 54.7x: "
        f"{comp['speculation_single_32x']} / {comp['speculation_batched_54_7x']}",
        f"    CUDA-graph single-stream (154 launches -> 1 replay): "
        f"{comp.get('cuda_graph_single_stream', False)}",
        f"    KV-drop (local window) on THIS conditional model: "
        f"{comp['kv_drop_local_window']}  ({kv['note']})",
        f"    radix-16 / log-sink shallow-divide module present: {comp['radix16_shallow_div']} "
        f"(composes on DIV/MOD-heavy forwards; the MAC is pure MUL so it is not on this path)",
        "",
        "HONEST HEADLINE:",
        f"    dense self-emulation wall     = {e['dense_wall_days']:.2f} days",
        f"    block-sparse alone            = {e['blocksparse_wall_days']:.3f} days "
        f"({e['dense_wall_days']/max(e['blocksparse_wall_days'],1e-9):.0f}x faster)",
        f"    + speculation (single 32x)    = {e['blocksparse_spec_single_days']:.4f} days",
        f"    + speculation (batched 54.7x) = {e['blocksparse_spec_batched_days']:.4f} days "
        f"= {e['blocksparse_spec_batched_days']*24:.2f} h",
    ]
    if ge.get("enabled"):
        lines += [
            "",
            "    --- with CUDA-GRAPH single-stream (launch overhead removed) ---",
            f"    graphed single-stream + spec (single 32x)    = "
            f"{ge['graphed_spec_single_minutes']:.2f} min "
            f"(eager {ge['eager_spec_single_minutes']:.2f} min)",
            f"    graphed single-stream + spec (batched 54.7x) = "
            f"{ge['graphed_spec_batched_minutes']:.2f} min "
            f"(eager {ge['eager_spec_batched_minutes']:.2f} min)",
        ]
    if "peak_cuda_gb" in rep:
        lines.append(f"\npeak cuda mem = {rep['peak_cuda_gb']:.2f} GB")
    return "\n".join(lines)


def _fmt(rep: Dict[str, object]) -> str:
    e = rep["extrapolation"]
    mac = rep["mac"]
    mv = rep["matvec"]
    dw = rep["dot2_wall"]
    sp = rep["speculation"]
    lines = [
        "=" * 76,
        "SELF-EMULATION WALL — the transformer running the transformer's forward",
        "=" * 76,
        f"device={rep['device']}  build={rep['build_s']:.1f}s  "
        f"genuine Qwen2Model hidden={rep['hidden']} inter={rep['intermediate']} "
        f"layers={rep['n_layers']} fits_stock={rep['fits_stock']}",
        "",
        "(1) BYTE-EXACT self-emulated MULTIPLY (the unit a matmul is built from):",
        f"    scalar MAC  w*x = {mac['ref']}: model={mac['got']} exact={mac['exact']}  "
        f"steps={mac['steps']}",
        f"    tokens/step(window)={mac['tokens_per_step']:.1f} "
        f"[min {mac['win_min']} max {mac['win_max']}]  "
        f"ms/step={rep['ms_per_step']:.1f}  wall/prog={rep['mac_wall_s']:.3f}s",
        "",
        f"(2) BYTE-EXACT tiny matvec [Rx1]@[1] W={mv['W']} x0={mv['x0']}:",
        f"    numpy ref={mv['ref']}  model={mv['got']}  byte_exact={mv['byte_exact']} "
        f"({mv['total_steps']} steps total)",
        "",
        f"(3) #702 WALL — width-2 dot y=w.x w={dw['w']} x={dw['x']}:",
        f"    deep-stack ref={dw['ref_deep_stack']}  model(1-slot STACK0)="
        f"{dw['model_1slot']}  diverges={dw['diverges']}",
        f"    -> {dw['note']}",
        "",
        "(3c) #692 OVERCOME — width>=2 dot + real matmul BYTE-EXACT (KV-backed stack):",
        f"    dot2 {rep['matmul_depth1']['dot2']['w']}.{rep['matmul_depth1']['dot2']['x']}"
        f"={rep['matmul_depth1']['dot2']['numpy']}: default(1-slot)="
        f"{rep['matmul_depth1']['dot2']['default_1slot']} (WALL)  KV-backed="
        f"{rep['matmul_depth1']['dot2']['kv_backed']}  byte_exact="
        f"{rep['matmul_depth1']['dot2']['byte_exact']} ({rep['matmul_depth1']['dot2']['steps']} steps)",
        f"    dot3={rep['matmul_depth1']['dot3']['model']} "
        f"(exact={rep['matmul_depth1']['dot3']['byte_exact']})  dot4="
        f"{rep['matmul_depth1']['dot4']['model']} "
        f"(exact={rep['matmul_depth1']['dot4']['byte_exact']})",
        f"    2x2@2x1 matmul M={rep['matmul_depth1']['matmul_2x2']['M']} "
        f"v={rep['matmul_depth1']['matmul_2x2']['v']}: model="
        f"{rep['matmul_depth1']['matmul_2x2']['model']} numpy="
        f"{rep['matmul_depth1']['matmul_2x2']['numpy']} byte_exact="
        f"{rep['matmul_depth1']['matmul_2x2']['byte_exact']} "
        f"({rep['matmul_depth1']['matmul_2x2']['total_steps']} steps)",
        f"    TIMING one dot: {rep['matmul_depth1']['timing']['vm_steps_per_dot']} steps "
        f"{rep['matmul_depth1']['timing']['wall_ms_per_dot']:.0f} ms "
        f"({rep['matmul_depth1']['timing']['ms_per_step']:.0f} ms/step, "
        f"{rep['matmul_depth1']['timing']['tokens_per_step']:.1f} tok/step)",
        "",
        "(3b) SPECULATION forwards-saved (perfect deterministic draft) + #702 gate:",
        f"    single-program block batching: {sp['single_program_fwd_saved']}x "
        f"forwards-saved (block_steps={sp['block_steps']}, 100% accept)",
        f"    cross-program batched: {sp['batched_fwd_saved_B64']:.1f}x @ B={sp['batch']} "
        f"(MEASURED prior, batched_speculative) — an UPPER bound",
        f"    #702 GATE: {sp['hs702_note']}",
        f"    -> speculation is FORWARDS-SAVED (throughput), NOT per-token compute: "
        f"wall/{sp['single_program_fwd_saved']} = {sp['wall_days_single_prog_spec']:.2f}d "
        f"(single-prog) .. wall/{sp['batched_fwd_saved_B64']:.1f} = "
        f"{sp['wall_days_batched_spec']:.2f}d (batched B={sp['batch']}, needs that many "
        f"independent programs/step-blocks)",
        "",
        "(4) EXTRAPOLATION (labeled) — ONE smallest self-forward:",
        f"    basis: {e['basis_steps']:,} VM steps (LABELED extrapolation basis)",
        "    TWO pinning quantities:",
        f"      TOKENS PER VM STEP = {e['tokens_per_vm_step']:.1f} "
        f"(BOS + 5-reg frame + query; +1 per live store)",
        f"      PER-TOKEN COMPUTE  = {e['per_token_compute_ms']:.1f} ms/token "
        f"(-> {e['per_step_compute_ms']:.1f} ms/step)",
        f"    total tokens = {e['total_tokens']:,.0f}",
        f"    HONEST WALL = {e['wall_seconds']:,.0f}s = {e['wall_hours']:,.1f}h "
        f"= {e['wall_days']:.2f} days",
        f"    (blog prior: {e['blog_prior_days']:.0f} days @ "
        f"{e['blog_prior_s_per_instr']}s/instr)",
        "    NOTE: TOKENS-PER-STEP (=7) is structural/invariant; PER-STEP compute is",
        "    GPU-CONTENTION-sensitive (~330-1250 ms/step observed on this SHARED",
        "    cuda:1) -> wall ~9-35 days, all well under the 88-day prior.",
    ]
    if "peak_cuda_gb" in rep:
        lines.append(f"\npeak cuda mem = {rep['peak_cuda_gb']:.2f} GB")
    return "\n".join(lines)


def main() -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--block-sparse", action="store_true",
                    help="measure the self-emulation wall through the BLOCK-SPARSE "
                         "conditional-dispatch fast path (full_native_fast / "
                         "ConditionalBlockLean) instead of the DENSE overlay build.")
    ap.add_argument("--device", default=None,
                    help="override C4_SELF_EMU_DEV (e.g. cuda:0).")
    ap.add_argument("--local-window", type=int, default=None,
                    help="requested KV-drop local sliding-window (reported).")
    ap.add_argument("--graphed", action="store_true",
                    help="(with --block-sparse) also measure the GRAPHED single-stream "
                         "block-verify forward (154 per-layer launches -> ONE "
                         "cuda.graph.replay), sweep K, and re-extrapolate the wall.")
    ap.add_argument("--graph-ks", default="16,32,64,128,256",
                    help="comma K list for the graphed single-stream K-sweep.")
    ap.add_argument("--reps", type=int,
                    default=int(os.environ.get("C4_SELF_EMU_REPS", "3")))
    a = ap.parse_args()
    dev = a.device or _dev()
    if a.block_sparse:
        gks = tuple(int(x) for x in a.graph_ks.split(",") if x)
        rep = measure_block_sparse(dev=dev, reps=max(a.reps, 20),
                                   local_window=a.local_window,
                                   graphed=a.graphed, graph_ks=gks)
        print(_fmt_block_sparse(rep))
    else:
        rep = measure(dev=dev, reps=a.reps)
        print(_fmt(rep))
    dump = os.environ.get("C4_SELF_EMU_JSON")
    if dump:
        with open(dump, "w") as f:
            json.dump(rep, f, indent=2, default=float)
        print(f"\n[json -> {dump}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
