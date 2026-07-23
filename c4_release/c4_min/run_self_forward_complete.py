"""RUN ONE COMPLETE SELF-EMULATION FORWARD TO COMPLETION — measured, not extrapolated.

``measure_self_emulation_wall.py`` only ever runs TINY programs (``max_steps<=64``)
and EXTRAPOLATES the ~2.4M-step self-forward wall.  This runner instead EXECUTES a
genuine COMPLETE self-computation END-TO-END on a SMALL, NO-DIVIDE fp32 VM and
verifies byte-exact — the transformer actually emulating (to completion) the
multiply-accumulate a matmul forward is built from.

WHAT "one complete self-forward" means here.  A transformer forward is a matmul
stack; the multiply a matmul is built from is the scalar MAC (``scalar_mac_prog`` in
``measure_self_emulation_wall`` — pure ``MUL``/``ADD``, NO ``DIV``/``MOD``).  A
genuine COMPLETE self-unit forward is therefore a real emulated MATMUL:

  * (A) a ``[R x 1] @ [1]`` MATVEC — R output rows, each ``y_i = W[i] * x0`` a
        complete scalar MAC.  Run to completion through the BLOCK-SPARSE + graphed +
        perfect-draft block-verify FAST path (byte-exact, arbitrary R), so we can
        finish a LARGE complete matmul (thousands of real VM steps) in seconds.
  * (B) a width>=2 DOT PRODUCT ``y = sum_i w_i*x_i`` and a small MATMUL — the REAL
        matmul inner unit that needs stack DEPTH>=2.  The fast draft is 1-slot
        STACK0 (#702 wall), so this runs to completion through the DENSE
        KV-BACKED-stack path (``run_program(..., spill_stack_to_kv=True)``): a
        genuinely COMPLETE dot product, byte-exact, measured wall included.

Both are executed to completion (every VM step, real forwards) and verified
bit-for-bit against ``isa.interpret`` and the numpy MAC reference.

NO-DIVIDE + SMALL FOOTPRINT.  The fast VM's active-unit union is SCOPED to the MAC's
own opcodes (IMM/PSH/MUL/ADD/HALT) — no DIV/MOD megablock — so the GPU-resident
conditional block is a few MB.  The dense weights live on CPU host RAM (never on the
GPU).  GPU-careful: pin the freest card, ``expandable_segments``, dedicated cache
dir, capped OMP threads, K backed off aggressively on OOM, one small model only.

Run (pins are set below; override via env):
    C4_SELF_EMU_DEV=cuda:0 python -m c4_min.run_self_forward_complete --rows 4096

See ``docs/SELF_EMULATION_WALL_2026_07_21.md`` for the extrapolated companion.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import torch

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from .qwen_lean_forward import CAM_REGS, _snap
from .nibble_pure_forward_complete import _decode_reg_from_nibbles
from .measure_self_emulation_wall import scalar_mac_prog, dot_prog


def _dev() -> str:
    return os.environ.get("C4_SELF_EMU_DEV", "cuda:0")


# ---------------------------------------------------------------------------
# A CORRECT block-verify driver (mirrors full_native_fast._run_native but decodes
# at the RIGHT query row).  The shipped _run_native / _run_native_graphed compute
# ``qrow = (1 + n_store) + len(CAM_REGS)`` — they OMIT the ``n_code`` code-frame
# offset, which is fine only for ``code_from_memory=False``.  With the now-default
# ``code_from_memory=True`` the program lives as ``n_code`` KV code frames in the
# window, so the query (STEP_END) row is ``(1 + n_store + n_code) + len(CAM_REGS)``
# (see qwen_lean_forward.speculative_run_lean:849, the correct naive-lean form).
# This local driver uses that correct qrow AND the efficient-ALU MUL/DIV/MOD nibble
# decode, so a MAC through the block-sparse (and graphed) path is byte-exact.  It
# does NOT modify any shared file.
# ---------------------------------------------------------------------------
@torch.no_grad()
def block_verify_correct(model, draft_lean, code, *, block_steps: int,
                         graphed=None, max_steps: int = 4096) -> LF.LeanSpecResult:
    """Perfect-draft block-verify on ``model`` (ConditionalBlockLean or LeanQwenVM),
    decoding each step's AX at the CORRECT query row (code-frame aware) — MUL/DIV/MOD
    from the AX nibble band (efficient ALU), else the scalar AX_VAL.  ``graphed`` is
    an optional ``GraphedLeanForward(model)`` (CUDA-graph replay of the block forward).
    """
    L = draft_lean.QL.L
    subset = draft_lean.subset
    n_code = len(code) if draft_lean.code_from_memory else 0
    draft = LF.draft_program_lean(draft_lean, code, max_steps=max_steps)
    ref_trace = draft.ref_trace
    if not draft.steps:
        r = LF.run_program_lean(draft_lean, code, max_steps=max_steps)
        n = r["steps"]
        return LF.LeanSpecResult(
            status="PASS" if r["exact"] else "FAIL", ax_trace=r["ax_trace"],
            ref_trace=r["ref_trace"], exact=r["exact"], steps=n, forwards=n,
            naive_forwards=n, speedup=1.0, accepted=n, detail="naive-fallback")
    n_steps = len(draft.steps)
    ax_trace: List[int] = []
    forwards = 0
    for s0 in range(0, n_steps, block_steps):
        slab = draft.steps[s0:s0 + block_steps]
        x, positions = LF._build_spec_batch(draft_lean, code, slab)
        x = x.to(model.device)
        positions = positions.to(model.device)
        if graphed is not None:
            hidden = graphed(x, positions)
        else:
            hidden, _ = model.forward(x, q_positions=positions)
        forwards += 1
        for i, st in enumerate(slab):
            n_store = len(st["store_log"]) if subset.memory else 0
            qrow = (1 + n_store + n_code) + len(CAM_REGS)   # code-frame aware (correct)
            state = hidden[i, qrow]
            op = st["op"]
            if op in (isa.MUL, isa.DIV, isa.MOD):
                ax = _decode_reg_from_nibbles(state, L, L.AX) & 0xFF
            else:
                ax = _snap(state[L.AX_VAL]) & 0xFF
            ax_trace.append(ax)
    exact = ax_trace == ref_trace
    return LF.LeanSpecResult(
        status="PASS" if exact else "FAIL", ax_trace=ax_trace, ref_trace=ref_trace,
        exact=exact, steps=n_steps, forwards=forwards, naive_forwards=n_steps,
        speedup=(n_steps / forwards) if forwards else 0.0, accepted=n_steps,
        detail="" if exact else "block-verify trace != isa.interpret")


# ---------------------------------------------------------------------------
# (A) The COMPLETE matmul self-forward through the block-sparse + graphed fast path.
#     A [R x 1] @ [1] matvec = R complete scalar MACs.  Every row is executed to
#     completion (all 5 VM steps, a real forward); byte-exact vs numpy & isa.interpret.
# ---------------------------------------------------------------------------
def run_matvec_fast(bundle, rows: int, x0: int, *, block_steps: int, graphed,
                    dev: str, seed: int = 12345) -> Dict[str, object]:
    """Execute a COMPLETE R-row matvec (R complete self-emulated multiplies) to the
    end through the conditional block-sparse driver (graphed when available), byte-
    exact vs the numpy / isa.interpret reference.  Returns measured wall/steps/tokens.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    W = [int(v) for v in rng.integers(0, 256, size=rows)]
    ref_rows = [int(v) for v in (np.array(W, dtype=np.int64) * x0) & 0xFF]

    cuda = dev.startswith("cuda")
    got_rows: List[int] = []
    total_steps = 0
    total_forwards = 0
    tokens_per_step = None
    all_exact = True

    if cuda:
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for wi in W:
        code = isa.assemble(scalar_mac_prog(wi, x0))
        r = block_verify_correct(bundle.cond, bundle.dense_lean, code,
                                 block_steps=block_steps, graphed=graphed)
        got_rows.append(r.ax_trace[-1] if r.ax_trace else -1)
        total_steps += r.steps
        total_forwards += r.forwards
        all_exact = all_exact and r.exact
        if tokens_per_step is None:
            tokens_per_step = _mac_tokens_per_step(bundle, code)
    if cuda:
        torch.cuda.synchronize(dev)
    wall = time.perf_counter() - t0

    byte_exact = (got_rows == ref_rows) and all_exact
    return {
        "rows": rows, "x0": x0,
        "byte_exact": byte_exact,
        "n_mismatch": sum(1 for a, b in zip(got_rows, ref_rows) if a != b),
        "total_vm_steps": total_steps,
        "total_forwards": total_forwards,
        "tokens_per_step": tokens_per_step,
        "total_tokens": (tokens_per_step or 0) * total_steps,
        "wall_seconds": wall,
        "ms_per_step": 1000.0 * wall / max(total_steps, 1),
        "steps_per_second": total_steps / wall if wall else 0.0,
        "block_steps": block_steps,
        "first8_ref": ref_rows[:8], "first8_got": got_rows[:8],
    }


def _mac_tokens_per_step(bundle, code) -> int:
    """Window length S (tokens fed to each VM-step forward) for the MAC through the
    fast driver — BOS + code frames + 5 reg frames + STEP_END (structural, invariant).
    """
    from . import qwen_lean_forward as LF
    lean = bundle.dense_lean
    draft = LF.draft_program_lean(lean, code, max_steps=64)
    x, _ = LF._build_spec_batch(lean, code, draft.steps[:1])
    return int(x.shape[1])


# ---------------------------------------------------------------------------
# (B) The COMPLETE width>=2 DOT PRODUCT / small MATMUL self-unit — the real matmul
#     inner unit (needs stack depth>=2).  Runs to completion on the DENSE KV-backed
#     stack (the fast 1-slot draft #702-walls on depth 2), byte-exact, measured wall.
# ---------------------------------------------------------------------------
def run_dot_dense(dev: str, *, seed: int = 999) -> Dict[str, object]:
    """Build ONE small dense KV-backed VM and run genuine COMPLETE dot products +
    a small matmul to completion, byte-exact vs numpy / isa.interpret.  This is the
    largest COMPLETE single self-unit (a full emulated dot product) that is byte-exact
    end-to-end.  Memory-safe: tiny programs, capped step budget, one small VM."""
    import numpy as np
    cuda = dev.startswith("cuda")
    # A SMALL no-divide dense VM: mem+cmp+muldiv subset (MUL native, DIV present but
    # UNUSED by the MAC), recurrent divmod folded so the model is not deep/huge.  The
    # MAC uses only IMM/PSH/MUL/ADD, so NO divide runs.
    vm = Q.build(code_size=32, subset=Q.SUBSET_MULDIV,
                 efficient_alu=True, recurrent_divmod=True)
    vm.qmodel = vm.qmodel.to(dev)
    vm.embed = vm.embed.to(dev)
    if cuda:
        torch.cuda.synchronize(dev)

    out: Dict[str, object] = {}
    rng = np.random.default_rng(seed)

    def _one_dot(w: List[int], x: List[int], tag: str, max_steps: int) -> Dict:
        code = isa.assemble(dot_prog(w, x))
        ref = int(np.dot(np.array(w, dtype=np.int64), np.array(x, dtype=np.int64))) & 0xFF
        isa_ref = isa.interpret(code, max_steps=max_steps)[-1]
        if cuda:
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        r = Q.run_program(vm, code, max_steps=max_steps, spill_stack_to_kv=True)
        if cuda:
            torch.cuda.synchronize(dev)
        wall = time.perf_counter() - t0
        got = r["ax_trace"][-1]
        return {
            "tag": tag, "width": len(w), "w": w, "x": x,
            "numpy": ref, "isa_interpret": isa_ref, "model": got,
            "byte_exact": (got == ref == isa_ref),
            "vm_steps": r["steps"], "wall_seconds": wall,
            "ms_per_step": 1000.0 * wall / max(r["steps"], 1),
        }

    # width-2..width-8 complete dot products (increasing self-unit size).
    dots = []
    for width in (2, 3, 4, 6, 8):
        w = [int(v) for v in rng.integers(1, 16, size=width)]
        x = [int(v) for v in rng.integers(1, 16, size=width)]
        dots.append(_one_dot(w, x, f"dot{width}", max_steps=64 + width * 16))
    out["dots"] = dots

    # a genuine small MATMUL: [3x2] @ [2] — each output row is a complete width-2 dot.
    M = rng.integers(1, 16, size=(3, 2)).astype(np.int64)
    v = rng.integers(1, 16, size=2).astype(np.int64)
    ref_mv = [int(M[i] @ v) & 0xFF for i in range(3)]
    got_mv = []
    total = 0
    wall_mm = 0.0
    for i in range(3):
        d = _one_dot([int(a) for a in M[i]], [int(a) for a in v],
                     f"mm_row{i}", max_steps=96)
        got_mv.append(d["model"])
        total += d["vm_steps"]
        wall_mm += d["wall_seconds"]
    out["matmul_3x2"] = {
        "M": M.tolist(), "v": v.tolist(), "numpy": ref_mv, "model": got_mv,
        "byte_exact": got_mv == ref_mv, "total_vm_steps": total,
        "wall_seconds": wall_mm,
    }
    out["hidden"] = vm.hidden_size
    out["intermediate"] = vm.intermediate_size
    out["n_layers_stored"] = vm.n_layers
    out["n_layers_applied"] = vm.n_applied
    if cuda:
        out["peak_cuda_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
    # free the dense VM before returning (crash-avoidance: don't keep two big models).
    del vm
    if cuda:
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None, help="override C4_SELF_EMU_DEV")
    ap.add_argument("--rows", type=int, default=4096,
                    help="matvec rows = complete scalar MACs to run to completion")
    ap.add_argument("--x0", type=int, default=11, help="the shared matvec input x0")
    ap.add_argument("--block-steps", type=int, default=32,
                    help="perfect-draft block-verify width (steps per forward)")
    ap.add_argument("--no-graphed", action="store_true",
                    help="disable the CUDA-graph single-stream forward")
    ap.add_argument("--skip-dense-dot", action="store_true",
                    help="skip the width>=2 dense KV-backed dot (part B)")
    ap.add_argument("--json", default=os.environ.get("C4_SELF_EMU_JSON"))
    a = ap.parse_args()
    dev = a.device or _dev()
    cuda = dev.startswith("cuda")

    print("=" * 78, flush=True)
    print("COMPLETE SELF-EMULATION FORWARD — executed to completion (NOT extrapolated)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"device={dev}  rows={a.rows}  x0={a.x0}  block_steps={a.block_steps}  "
          f"graphed={not a.no_graphed}", flush=True)

    if cuda:
        torch.zeros(1, device=dev)               # init the CUDA context on this device
        torch.cuda.reset_peak_memory_stats(dev)

    from . import full_native_fast as FNF
    # SCOPE to the MAC's own opcodes — NO DIV/MOD megablock -> a few-MB active block.
    mac_ops = [isa.IMM, isa.PSH, isa.MUL, isa.ADD, isa.HALT]
    t0 = time.time()
    bundle = FNF.build_full_native_fast(device=dev, code_size=24, ops=mac_ops,
                                        verbose=True)
    build_s = time.time() - t0
    print(f"[built no-divide fast VM in {build_s:.1f}s] "
          f"active={bundle.active_total} units "
          f"({bundle.active_gb*1024:.1f} MB, "
          f"{bundle.active_total/(bundle.intermediate*bundle.n_layers_applied)*100:.4f}% "
          f"of dense {bundle.dense_gb:.1f} GB, dense stays on CPU)", flush=True)

    # optional CUDA-graph single-stream forward (per-layer launches -> 1 replay).
    graphed = None
    g = None
    graph_note = "eager"
    if cuda and not a.no_graphed:
        try:
            from .qwen_lean_cuda_graph import GraphedLeanForward
            graphed = GraphedLeanForward(bundle.cond)
            g = graphed
            graph_note = "cuda-graph"
        except Exception as e:                       # graph capture may not be supported
            print(f"[graphed disabled: {type(e).__name__}: {e}]", flush=True)
            graphed = None
            graph_note = "eager (graph-capture failed)"

    # ---- (A) run the COMPLETE matvec self-forward to completion (with OOM back-off).
    block_steps = a.block_steps
    matvec = None
    while matvec is None:
        try:
            matvec = run_matvec_fast(bundle, a.rows, a.x0, block_steps=block_steps,
                                     graphed=graphed, dev=dev)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if cuda:
                torch.cuda.empty_cache()
            if block_steps <= 4:
                raise
            block_steps = max(block_steps // 2, 4)
            print(f"[OOM -> backing off block_steps to {block_steps}: "
                  f"{type(e).__name__}]", flush=True)

    print("", flush=True)
    print("(A) COMPLETE MATVEC self-forward [R x 1] @ [1] — R complete scalar MACs:",
          flush=True)
    print(f"    rows(R)={matvec['rows']}  x0={matvec['x0']}  "
          f"forward={graph_note}  block_steps={matvec['block_steps']}", flush=True)
    print(f"    BYTE-EXACT vs numpy & isa.interpret: {matvec['byte_exact']}  "
          f"(mismatches={matvec['n_mismatch']})", flush=True)
    print(f"    total VM steps EXECUTED = {matvec['total_vm_steps']:,}  "
          f"forwards = {matvec['total_forwards']:,}  "
          f"tokens/step = {matvec['tokens_per_step']}", flush=True)
    print(f"    total tokens = {matvec['total_tokens']:,}", flush=True)
    print(f"    MEASURED WALL = {matvec['wall_seconds']:.2f} s  "
          f"({matvec['ms_per_step']:.3f} ms/step, "
          f"{matvec['steps_per_second']:,.0f} steps/s)", flush=True)
    print(f"    ref[:8]={matvec['first8_ref']}  got[:8]={matvec['first8_got']}",
          flush=True)

    peak_a = torch.cuda.max_memory_allocated(dev) / 1e9 if cuda else None

    # free the fast bundle's GPU model BEFORE building the dense VM (one model at a time).
    if not a.skip_dense_dot:
        del bundle
        if graphed is not None:
            del graphed, g
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)

    # ---- (B) the COMPLETE width>=2 dot product / small matmul (dense KV-backed).
    dot = None
    if not a.skip_dense_dot:
        print("", flush=True)
        print("(B) COMPLETE width>=2 DOT PRODUCT + small MATMUL (dense KV-backed stack):",
              flush=True)
        dot = run_dot_dense(dev)
        for d in dot["dots"]:
            print(f"    {d['tag']:6s} w.x width={d['width']}: numpy={d['numpy']:3d} "
                  f"isa={d['isa_interpret']:3d} model={d['model']:3d}  "
                  f"byte_exact={d['byte_exact']}  "
                  f"{d['vm_steps']:3d} steps  {d['wall_seconds']:.2f}s "
                  f"({d['ms_per_step']:.0f} ms/step)", flush=True)
        mm = dot["matmul_3x2"]
        print(f"    matmul [3x2]@[2] M={mm['M']} v={mm['v']}: numpy={mm['numpy']} "
              f"model={mm['model']}  byte_exact={mm['byte_exact']}  "
              f"{mm['total_vm_steps']} steps  {mm['wall_seconds']:.2f}s", flush=True)
        if cuda and "peak_cuda_gb" in dot:
            print(f"    dense VM peak VRAM = {dot['peak_cuda_gb']:.2f} GB", flush=True)

    # ---- headline.
    all_dots_exact = (dot is None) or (
        all(d["byte_exact"] for d in dot["dots"]) and dot["matmul_3x2"]["byte_exact"])
    overall = matvec["byte_exact"] and all_dots_exact
    print("", flush=True)
    print("HEADLINE:", flush=True)
    print(f"    completed matvec self-forward = {matvec['total_vm_steps']:,} VM steps, "
          f"{matvec['total_tokens']:,} tokens, {matvec['wall_seconds']:.2f}s wall, "
          f"byte-exact={matvec['byte_exact']}", flush=True)
    if peak_a is not None:
        print(f"    fast-path peak VRAM = {peak_a:.2f} GB", flush=True)
    if dot is not None:
        print(f"    largest COMPLETE single dot self-unit = width-{dot['dots'][-1]['width']} "
              f"({dot['dots'][-1]['vm_steps']} steps), byte-exact="
              f"{dot['dots'][-1]['byte_exact']}", flush=True)
    print(f"    OVERALL BYTE-EXACT (matvec + all dots + matmul): {overall}", flush=True)

    rep = {
        "device": dev, "build_seconds": build_s,
        "forward_mode": graph_note,
        "matvec_complete_forward": matvec,
        "fast_path_peak_vram_gb": peak_a,
        "dense_dot": dot,
        "overall_byte_exact": overall,
    }
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2, default=float)
        print(f"\n[json -> {a.json}]", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
