"""THREE-WAY head-to-head benchmark for the fused C4 VM forward.

Measures, on the SAME programs and the SAME device, the ms/VM-step of:

  1. **lean-native** — ``qwen_lean_forward.LeanQwenVM`` (this branch's foundation):
     a hand-written RoPE + RMSNorm + softmax + SwiGLU forward of the COMPACTED
     7-14-layer, 6-head baked weights.  No ``transformers`` machinery.
  2. **HF-Qwen2Model** — ``qwen_full_vm`` (the SAME compacted weights) driven through
     the genuine ``transformers.Qwen2Model.forward``.  The delta vs (1) is the pure
     HF-wrapper overhead.
  3. **deep pure-forward** — ``nibble_pure_forward_complete`` (the ~308-block, dim
     ~2392, 23-head reference).  The honest "can we beat 65ms/step" comparison.

Also: window-size scaling of the lean forward (ms vs stream length), and the
speculation speedup (forwards-saved) of ``speculative_run_lean``.

All measurements are wall-clock over ``run_program`` (naive one-forward-per-step)
so the ms/step is the amortised per-VM-step cost the CUDA-graph / fusion / async-KV
follow-on agents will drive down.  Run:

    python -m c4_min.bench_lean_forward --device cuda:1
    python -m c4_min.bench_lean_forward --device cuda:1 --deep   # include the 308-block ref
"""
from __future__ import annotations

import argparse
import time
import warnings
from typing import Dict, List, Tuple

import torch

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF


# ---------------------------------------------------------------------------
# The benchmark corpus — one representative per op family (base + mem/cmp), plus a
# few loop depths for the amortised ms/step.
# ---------------------------------------------------------------------------
def _bin(op, a, b):
    return [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)]


BASE_PROGS: List[Tuple[str, list]] = [
    ("arith_add", _bin("ADD", 100, 27)),
    ("arith_sub", _bin("SUB", 200, 55)),
    ("cmp_eq", _bin("EQ", 5, 5)),
    ("if_bz", [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)]),
    ("loop_cd5", [("IMM", 5), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]),
    ("loop_cd20", [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]),
]

MEM_PROGS: List[Tuple[str, list]] = [
    ("cmp_eq", _bin("EQ", 5, 5)),
    ("cmp_gt", _bin("GT", 9, 7)),
    ("mem_si_li", [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
                   ("IMM", 5), ("LI", 0), ("HALT", 0)]),
    ("mem_zfod", [("IMM", 50), ("LI", 0), ("HALT", 0)]),
    ("var_add", [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
                 ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)]),
    ("loop_cd20", [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]),
]


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _time_run(fn, *args, warmup: int = 1, iters: int = 3, device: str = "cpu",
              **kwargs) -> Tuple[float, object]:
    """Return (mean_ms_per_call, last_result).  Warms up then times ``iters`` runs."""
    res = None
    for _ in range(warmup):
        res = fn(*args, **kwargs)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        res = fn(*args, **kwargs)
    _sync(device)
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0, res


# ---------------------------------------------------------------------------
# Head-to-head: lean vs HF (same weights) on the SAME programs.
# ---------------------------------------------------------------------------
def head_to_head(device: str = "cpu", subset=Q.SUBSET_BASE,
                 progs=None, deep: bool = False, iters: int = 3) -> Dict:
    warnings.filterwarnings("ignore")
    progs = progs or BASE_PROGS
    print(f"\n=== building compacted VM (subset={subset.name}) ===")
    t0 = time.time()
    vm = Q.build(code_size=24, subset=subset)
    vm.qmodel = vm.qmodel.to(device)
    # the HF driver gathers the embedding on vm.embed's device, then feeds
    # inputs_embeds to the model — put both on the bench device for a fair GPU time.
    vm.embed = vm.embed.to(device)
    print(f"  HF Qwen2Model: {vm.n_layers} layers, hidden {vm.hidden_size}, "
          f"inter {vm.intermediate_size}, {vm.arch.num_attention_heads} q-heads "
          f"(built {time.time()-t0:.1f}s)")
    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    print(f"  lean-native: {lean.n_layers} layers, {lean.n_heads} q-heads "
          f"(6 CAM), head_dim {lean.head_dim}")

    deep_model = deep_L = None
    if deep:
        from .nibble_pure_forward_complete import (
            build_pure_forward_complete_model, run_pure_forward_complete)
        print("  building deep 308-block reference ...")
        t0 = time.time()
        deep_model, deep_L = build_pure_forward_complete_model(code_size=24)
        gb = sum(p.numel() * 4 for p in deep_model.parameters()) / 1e9
        print(f"    deep: {len(deep_model.blocks)} blocks, dim {deep_model.dim}, "
              f"{deep_model.blocks[0].attn.n_heads} heads, ~{gb:.1f} GB fp32 "
              f"(built {time.time()-t0:.1f}s)")
        try:
            deep_model = deep_model.to(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # noqa: F821
            # the deep model is ~128 GB fp32 — it does NOT fit a 24 GB GPU densely
            # (the honest finding).  Fall back to CPU timing (slow) or skip.
            print(f"    deep .to({device}) OOM ({type(e).__name__}); "
                  f"deep column left blank (128 GB fp32 needs the sparse/streamed "
                  f"runner — the source of the ~65 ms/step reference).")
            deep = False
            deep_model = None

    rows = []
    tot_lean = tot_hf = tot_deep = 0.0
    tot_steps = 0
    for name, prog in progs:
        code = isa.assemble(prog)
        # correctness: lean must match HF byte-for-byte.
        rh = Q.run_program(vm, code, max_steps=64)
        rl = LF.run_program_lean(lean, code, max_steps=64)
        match = rl["ax_trace"] == rh["ax_trace"]
        steps = rl["steps"]

        ms_lean, _ = _time_run(LF.run_program_lean, lean, code, device=device,
                               iters=iters, max_steps=64)
        ms_hf, _ = _time_run(Q.run_program, vm, code, device=device,
                             iters=iters, max_steps=64)
        row = {"name": name, "steps": steps, "lean_ms": ms_lean, "hf_ms": ms_hf,
               "lean_ms_step": ms_lean / max(steps, 1),
               "hf_ms_step": ms_hf / max(steps, 1),
               "lean_eq_hf": match, "lean_exact": rl["exact"]}
        if deep:
            ms_deep, _ = _time_run(
                run_pure_forward_complete, deep_model, deep_L, code,
                device=device, iters=max(1, iters // 2), max_steps=64)
            row["deep_ms"] = ms_deep
            row["deep_ms_step"] = ms_deep / max(steps, 1)
            tot_deep += ms_deep
        rows.append(row)
        tot_lean += ms_lean; tot_hf += ms_hf; tot_steps += steps

    print(f"\n  {'program':12s} {'steps':>5s} {'lean ms/step':>13s} "
          f"{'HF ms/step':>11s} {'deep ms/step':>13s} {'HF/lean':>8s} {'lean==HF':>9s}")
    for r in rows:
        deep_s = f"{r['deep_ms_step']:11.2f}" if deep else f"{'—':>11s}"
        ratio = r["hf_ms_step"] / r["lean_ms_step"] if r["lean_ms_step"] else 0.0
        print(f"  {r['name']:12s} {r['steps']:5d} {r['lean_ms_step']:11.2f}  "
              f"{r['hf_ms_step']:9.2f}  {deep_s}  {ratio:6.2f}x  "
              f"{str(r['lean_eq_hf']):>9s}")

    lean_step = tot_lean / max(tot_steps, 1)
    hf_step = tot_hf / max(tot_steps, 1)
    print(f"\n  AGGREGATE ms/VM-step (subset {subset.name}):")
    print(f"    lean-native : {lean_step:8.2f} ms/step")
    print(f"    HF-Qwen2    : {hf_step:8.2f} ms/step   "
          f"(HF wrapper overhead {hf_step - lean_step:+.2f} ms/step, "
          f"{hf_step / lean_step:.2f}x)")
    if deep:
        deep_step = tot_deep / max(tot_steps, 1)
        print(f"    deep 308blk : {deep_step:8.2f} ms/step   "
              f"(lean is {deep_step / lean_step:.1f}x faster than the deep ref)")
    all_match = all(r["lean_eq_hf"] for r in rows)
    print(f"\n  lean == HF byte-for-byte on ALL programs: {all_match}")
    return {"rows": rows, "lean_ms_step": lean_step, "hf_ms_step": hf_step,
            "all_match": all_match,
            "deep_ms_step": (tot_deep / max(tot_steps, 1)) if deep else None}


# ---------------------------------------------------------------------------
# Window-size scaling of the lean forward: ms vs stream length (steps in flight).
# ---------------------------------------------------------------------------
def window_scaling(device: str = "cpu", subset=Q.SUBSET_MEM_CMP,
                   iters: int = 5) -> Dict:
    warnings.filterwarnings("ignore")
    vm = Q.build(code_size=24, subset=subset)
    vm.qmodel = vm.qmodel.to(device)
    vm.embed = vm.embed.to(device)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    # a store-heavy program so the window (store log) grows with distinct addresses.
    print(f"\n=== lean window-size scaling (subset {subset.name}) ===")
    print(f"  {'n_store':>7s} {'stream_len':>10s} {'fwd ms':>8s} {'ms/token':>9s}")
    rows = []
    for n_store in (0, 4, 8, 16, 32, 64):
        # build a synthetic window: BOS + n_store store rows + 5 reg frame + query.
        reg_state = {"PC": 1, "AX": 3, "SP": LF.SP_INIT, "BP": LF.SP_INIT, "STACK0": 0}
        store_log = [{"addr": a, "val": a & 0xFF} for a in range(n_store)]
        code = isa.assemble(_bin("ADD", 3, 4))
        x, positions = LF._build_stream_and_overlay(lean, code, reg_state, store_log, None)
        S = x.shape[1]

        def _fwd():
            with torch.no_grad():
                return lean.forward(x, past=None, q_positions=positions)

        ms, _ = _time_run(_fwd, device=device, iters=iters)
        rows.append({"n_store": n_store, "stream_len": S, "fwd_ms": ms,
                     "ms_per_token": ms / S})
        print(f"  {n_store:7d} {S:10d} {ms:8.3f} {ms / S:9.4f}")
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Speculation speedup on the lean forward.
# ---------------------------------------------------------------------------
def speculation_bench(device: str = "cpu", subset=Q.SUBSET_MEM_CMP,
                      iters: int = 3) -> Dict:
    warnings.filterwarnings("ignore")
    vm = Q.build(code_size=24, subset=subset)
    vm.qmodel = vm.qmodel.to(device)
    vm.embed = vm.embed.to(device)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    progs = [
        ("loop_cd5", [("IMM", 5), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]),
        ("loop_cd20", [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]),
        ("loop_cd60", [("IMM", 60), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]),
        ("var_add", [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
                     ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)]),
    ]
    print(f"\n=== lean speculation speedup (subset {subset.name}, block_steps=32) ===")
    print(f"  {'program':10s} {'steps':>5s} {'naive fwd':>9s} {'spec fwd':>8s} "
          f"{'fwd saved':>9s} {'naive ms':>9s} {'spec ms':>8s} {'wall x':>7s} {'exact':>6s}")
    rows = []
    for name, p in progs:
        code = isa.assemble(p)
        rs = LF.speculative_run_lean(lean, code, block_steps=32)
        # wall-clock naive vs spec.
        ms_naive, _ = _time_run(LF.run_program_lean, lean, code, device=device,
                                iters=iters, max_steps=4096)
        ms_spec, _ = _time_run(LF.speculative_run_lean, lean, code, device=device,
                               iters=iters, block_steps=32, max_steps=4096)
        fwd_saved = rs.naive_forwards / rs.forwards if rs.forwards else 0.0
        wall_x = ms_naive / ms_spec if ms_spec else 0.0
        rows.append({"name": name, "steps": rs.steps, "naive_fwd": rs.naive_forwards,
                     "spec_fwd": rs.forwards, "fwd_saved": fwd_saved,
                     "naive_ms": ms_naive, "spec_ms": ms_spec, "wall_x": wall_x,
                     "exact": rs.exact})
        print(f"  {name:10s} {rs.steps:5d} {rs.naive_forwards:9d} {rs.forwards:8d} "
              f"{fwd_saved:8.1f}x {ms_naive:9.2f} {ms_spec:8.2f} {wall_x:6.1f}x "
              f"{str(rs.exact):>6s}")
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deep", action="store_true", help="include the 308-block reference")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--only", default="all",
                    help="all|head|window|spec")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(torch.device(dev))
    print(f"device: {dev}  torch {torch.__version__}")

    if args.only in ("all", "head"):
        print("\n########## BASE subset (7 layers, fits/near-stock) ##########")
        head_to_head(dev, subset=Q.SUBSET_BASE, progs=BASE_PROGS,
                     deep=args.deep, iters=args.iters)
        print("\n########## MEM+CMP subset (10 layers) ##########")
        head_to_head(dev, subset=Q.SUBSET_MEM_CMP, progs=MEM_PROGS,
                     deep=args.deep, iters=args.iters)
    if args.only in ("all", "window"):
        window_scaling(dev, iters=max(3, args.iters))
    if args.only in ("all", "spec"):
        speculation_bench(dev, iters=args.iters)


if __name__ == "__main__":
    main()
