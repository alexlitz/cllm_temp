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
and EXTRAPOLATES the honest wall of ONE smallest self-forward (~2.4M VM steps,
BLOG_SPEC_REVISIONS §7) = ``total_tokens x per-token-compute``.

Memory-safe: ONE lean muldiv+efficient_alu Qwen build on ``cuda:1``, tiny programs,
capped step counts.  Run:
    C4_SELF_EMU_DEV=cuda:1 python -m c4_min.measure_self_emulation_wall
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from . import isa
from . import qwen_full_vm as Q

# ~2.4M VM steps for the smallest self-forward (BLOG_SPEC §901 answer, recorded in
# docs/BLOG_SPEC_REVISIONS.md §7).  This is the EXTRAPOLATION basis (labeled).
SELF_FORWARD_STEPS = 2_400_000


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
    return out


def _fmt(rep: Dict[str, object]) -> str:
    e = rep["extrapolation"]
    mac = rep["mac"]
    mv = rep["matvec"]
    dw = rep["dot2_wall"]
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
        "(4) EXTRAPOLATION (labeled) — ONE smallest self-forward:",
        f"    basis: {e['basis_steps']:,} VM steps (BLOG_SPEC_REVISIONS §7)",
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
    ]
    if "peak_cuda_gb" in rep:
        lines.append(f"\npeak cuda mem = {rep['peak_cuda_gb']:.2f} GB")
    return "\n".join(lines)


def main() -> int:
    import json
    reps = int(os.environ.get("C4_SELF_EMU_REPS", "3"))
    rep = measure(reps=reps)
    print(_fmt(rep))
    dump = os.environ.get("C4_SELF_EMU_JSON")
    if dump:
        with open(dump, "w") as f:
            json.dump(rep, f, indent=2, default=float)
        print(f"\n[json -> {dump}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
