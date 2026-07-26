"""#745 — COMPOSED byte-exact POSITION-SPARSE fast path, real GPU ms/step.

This is the culminating measurement for the position-sparsity lever (#745): the
highest-value per-step redundancy this session.  Every op-level-LIVE block computes
at all ~S stream positions but ONLY the query row (the last token, -1) affects the
decode; masking a block at any other position is byte-identical (STEP-1 verified per-
position ablation in ``_pos_sparse_measure``).  So the heavy long-division span (179
blocks) is recomputed ~S times more than needed.

We COMPOSE the position-sparse forward with the rest of the merged fast path onto ONE
byte-exact forward and measure the REAL ms/step:

  * ``PositionSparseRunner`` (``C4_POS_SPARSE``): per decoded op, run ONLY that op's
    live blocks (block-skip), and within each live block run the ATTENTION K/V over
    all S positions (the query attends every key) but Q / softmax / context / W_o and
    the WHOLE FFN at the SINGLE query row; every other row carries its exact input
    residual (byte-exact passthrough for the decode).
  * ``direct_cam_read`` (``C4_DIRECT_CAM_READ``): the perfect draft already knows the
    exact KV store row each CAM read resolves to, so the read direct-gathers the value
    (O(1) overwrite of the query-row dest band) instead of the O(K) softmax score.

Baselines measured against, all byte-exact to each other for the decode:
  * FULL-238 dense over ALL positions   (the vanilla golden reference).
  * block-skip only, dense over ALL positions (op's live blocks, every row).
  * POSITION-SPARSE  (op's live blocks, query-row-only heavy compute).  <-- #745

HONESTY: pos-sparse still runs K/V over ALL S positions (the query attends every
key), so the attention K/V projection is NOT reduced.  Only Q/O/FFN drop to 1 row.
Whether the 111-536x nnz-MAC reduction translates to a real ms/step win, or is capped
by that K/V-at-all-positions residual + the per-block kernel-launch overhead, is
exactly what this benchmark reports — measured, not assumed.

Gate: ``C4_POS_SPARSE`` default OFF (this bench sets it internally); the golden
build is byte-identical flag-OFF.

Run (needs a free >=18 GB CUDA card):
    OMP_NUM_THREADS=4 python -m c4_min.bench_pos_sparse_composed --S 300,900
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .step_block_skip import StepBlockSkipRunner, build_live_index
from .pos_sparse_forward import PositionSparseRunner, _block_query_only, block_macs
from .pf_speculative import draft_pf_program
from .nibble_evict_schedule import resolve_load_rows
from .bench_composed_fast_path import (
    wait_for_gpu, _battery, _nested_prog, _resolve_with_seed, _dest_band,
    _OP_MIX, _OP_LABEL,
)


# ---------------------------------------------------------------------------
# The COMPOSED position-sparse per-step forward: op's live blocks, query-row-only
# heavy compute, then the direct-CAM dest-band overwrite.
# ---------------------------------------------------------------------------
class PosSparseComposed:
    """PositionSparseRunner (block-skip + query-row-only) + direct-CAM.

    ``forward(x, op, resolved)`` runs the op's live blocks with the ATTENTION K/V
    over all S rows and Q/O/FFN at the single query row (-1), then overwrites the
    CAM-read dest band of the query row with the resolved draft gather.  Returns the
    query-row state ([D]).  Byte-identical to the full-238 dense-over-positions +
    softmax CAM iff every non-query position is a nil passthrough for the decode
    (STEP-1 verified) AND the direct-gather value == the softmax winner (proven by
    ``resolve_load_rows``)."""

    def __init__(self, model, L, direct_cam: bool = True):
        self.model = model
        self.L = L
        self.direct_cam = direct_cam
        self.runner = PositionSparseRunner(model, L)

    def live_count(self, op) -> int:
        return self.runner.live_count(op)

    def forward(self, x: torch.Tensor, op,
                resolved: Optional[List] = None) -> torch.Tensor:
        with torch.no_grad():
            x = self.runner.forward(x, op)
        state = x[0, -1].clone()
        if self.direct_cam and resolved:
            for r in resolved:
                band = _dest_band(self.L, r.head)
                for j, nv in enumerate(
                        V.nibbles_of_value(r.value & 0xFFFFFFFF, NIB_PER_REG)):
                    state[band + j] = float(nv)
        return state


# ---------------------------------------------------------------------------
# Byte-exact composition VERIFY: full-238 dense-over-positions vs the composed
# POSITION-SPARSE path, per-step decoded AX trace.
# ---------------------------------------------------------------------------
def _drive(model, L, code, *, composed: Optional[PosSparseComposed] = None,
           max_steps=200, seed_mem=None):
    from . import nibble_pure_forward_complete as pfc
    from .nibble_pure_forward_complete import (
        make_overlay_complete, _build_frame, SP_INIT, _seed_frames, _mem_top)

    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)
    resolved_by_frame = {}
    if composed is not None and composed.direct_cam:
        draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF)
        resolved_by_frame = _resolve_with_seed(draft, seed_mem or {})
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    frame_idx = n_seed
    trace: List[int] = []
    total_applies = 0
    dev = model.embed.device
    for _ in range(max_steps):
        overlay = make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream], device=dev)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            if composed is None:
                for blk in model.blocks:
                    x = blk(x)
                total_applies += len(model.blocks)
                state = x[0, -1]
            else:
                draft_frame = frame_idx + 1 - n_seed
                resolved = resolved_by_frame.get(draft_frame, [])
                state = composed.forward(x, op, resolved)
                total_applies += composed.live_count(op)
        pc = pfc._snap_lane(state[L.PC_VAL].cpu())
        sp = pfc._snap_lane(state[L.SP_VAL].cpu())
        bp = pfc._snap_lane(state[L.BP_VAL].cpu())
        stk = pfc._snap_lane(state[L.STK_VAL].cpu())
        halted = float(state[L.HALTED]) > 0.5
        ax = pfc._decode_reg_from_nibbles(state.cpu(), L, L.AX)
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & 0xFF
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & 0xFF
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        trace.append(ax & 0xFF)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        cur_pc, cur_sp, cur_bp = pc, sp, bp
        if halted or pc < 0 or pc >= len(code):
            break
    return trace, total_applies


def verify_composition(model, L, *, direct_cam=True, verbose=True) -> bool:
    """FULL-238 dense-over-positions vs the COMPOSED position-sparse path, byte-exact
    per-step AX, over the battery + a genuinely deep nested loop (vs the free draft)."""
    composed = PosSparseComposed(model, L, direct_cam=direct_cam)
    ok = True
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        base, base_ap = _drive(model, L, code, composed=None, seed_mem=seed)
        comp, comp_ap = _drive(model, L, code, composed=composed, seed_mem=seed)
        match = base == comp
        ok = ok and match
        if verbose:
            ratio = f"{base_ap/comp_ap:.1f}x" if comp_ap else "n/a"
            print(f"  {name:10s} {'OK ' if match else 'FAIL'} "
                  f"steps={len(base)} applies full={base_ap} pos-sparse={comp_ap} "
                  f"({ratio} fewer block-applies)", flush=True)
            if not match:
                print(f"    BASE={base}\n    COMP={comp}", flush=True)

    # DEEP-LOOP gate vs the FREE perfect draft (the byte-exact ground truth).
    deep_steps = 60
    deep = _nested_prog(3, 4)
    draft = draft_pf_program(deep, max_steps=deep_steps, mask=0xFF)
    draft_ax = [f["ax"] & 0xFF for f in draft.frames]
    comp, comp_ap = _drive(model, L, deep, composed=composed, seed_mem={},
                           max_steps=deep_steps)
    n = min(len(comp), len(draft_ax))
    dmatch = comp[:n] == draft_ax[:n] and n > 0
    ok = ok and dmatch
    if verbose:
        print(f"  {'nested_deep':10s} {'OK ' if dmatch else 'FAIL'} "
              f"steps={n} (pos-sparse==free-draft over a deep nested loop, "
              f"{comp_ap} block-applies)", flush=True)
        if not dmatch:
            print(f"    DRAFT={draft_ax[:20]}...\n    COMP ={comp[:20]}...", flush=True)
    return ok


# ---------------------------------------------------------------------------
# CUDA-graph capture of the query-row-only path.  The query-row-only shapes are
# STATIC (K/V over fixed S, Q/O/FFN over 1 row), so each op-class live schedule
# captures as one graph.  Keyed by the (live-schedule) tuple.
# ---------------------------------------------------------------------------
class PosSparseOpGraphs:
    """Capture + replay one CUDA graph per distinct pos-sparse live schedule.

    Runs ``_block_query_only`` for each live block on a static [1,S,D] buffer at
    fixed S and query row q=S-1.  Ops sharing a live-block set share a graph."""

    def __init__(self, model, L, S: int, device):
        self.model = model
        self.L = L
        self.S = S
        self.device = device
        self.q = S - 1
        self.runner = PositionSparseRunner(model, L)
        D = model.embed.shape[1]
        self.static_in = torch.zeros(1, S, D, device=device,
                                     dtype=model.embed.dtype)
        self.op_to_key: Dict[int, Tuple[int, ...]] = {}
        seen: Dict[Tuple[int, ...], List[int]] = {}
        for op, live in self.runner.live_index.items():
            if op is None:
                continue
            key = tuple(live)
            self.op_to_key[op] = key
            seen.setdefault(key, []).append(op)
        self.distinct_keys = list(seen.keys())
        self._seen = seen
        self.graphs: Dict[Tuple[int, ...], Tuple] = {}

    def _apply_live(self, x, live: List[int]):
        for bi in live:
            x = _block_query_only(self.model.blocks[bi], x, self.q)
        return x

    def try_capture(self, key: Tuple[int, ...]) -> Tuple[str, Optional[float]]:
        live = list(key)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    _ = self._apply_live(self.static_in, live)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        try:
            t0 = time.time()
            with torch.no_grad():
                with torch.cuda.graph(g):
                    static_out = self._apply_live(self.static_in, live)
            torch.cuda.synchronize()
            cap_ms = (time.time() - t0) * 1e3
        except Exception as e:
            return (f"skip:{type(e).__name__}:{str(e).splitlines()[0][:70]}", None)
        self.graphs[key] = (g, static_out)
        return ("ok", cap_ms)

    def replay_ms(self, key, iters=50, warmup=10) -> Optional[float]:
        ent = self.graphs.get(key)
        if ent is None:
            return None
        g, _ = ent
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1e3


def _time_fn(fn, n, warmup, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


# ---------------------------------------------------------------------------
# BENCHMARK: dense-over-positions vs position-sparse ms/step, per-op + weighted.
# ---------------------------------------------------------------------------
def bench_at_S(model, L, S: int, *, n=30, warmup=8, use_graphs=True,
               report_macs=True):
    dev = model.embed.device
    cuda = (dev.type == "cuda")
    skip_runner = StepBlockSkipRunner(model, L)     # dense-over-positions block-skip
    pos_runner = PositionSparseRunner(model, L)     # query-row-only
    q = S - 1
    D = model.embed.shape[1]
    x0 = torch.zeros(1, S, D, device=dev, dtype=model.embed.dtype)

    print(f"\n{'='*80}\n[bench] production S={S}  (n={n}, warmup={warmup})\n{'='*80}",
          flush=True)

    # -- FULL-238 dense over ALL positions (vanilla golden reference) --------
    def full():
        x = x0
        with torch.no_grad():
            for blk in model.blocks:
                x = blk(x)
        return x
    t_full = _time_fn(full, n, warmup, cuda)
    nb = len(model.blocks)
    print(f"  FULL {nb}-block dense-over-positions       = {t_full:9.3f} ms/step",
          flush=True)

    graphs = None
    if use_graphs and cuda:
        graphs = PosSparseOpGraphs(model, L, S, dev)

    per_op_dense: Dict[int, float] = {}     # block-skip, dense over all S rows
    per_op_pos: Dict[int, float] = {}       # position-sparse, query-row-only (eager)
    per_op_graph: Dict[int, float] = {}     # position-sparse, CUDA-graphed
    graph_status: Dict[Tuple[int, ...], str] = {}

    print(f"\n  {'op':>5} {'live':>5} {'dense-over-pos':>15} "
          f"{'pos-sparse(eager)':>18} {'pos-sparse(graph)':>18} {'dense/pos':>10}",
          flush=True)
    print("  " + "-" * 78, flush=True)
    for op in _OP_MIX:
        live = pos_runner.live_index[op]

        def dense(op=op):
            with torch.no_grad():
                return skip_runner.forward(x0, op)

        def pos(op=op):
            with torch.no_grad():
                return pos_runner.forward(x0, op)

        t_dense = _time_fn(dense, n, warmup, cuda)
        t_pos = _time_fn(pos, n, warmup, cuda)
        per_op_dense[op] = t_dense
        per_op_pos[op] = t_pos
        t_graph = None
        if graphs is not None:
            key = graphs.op_to_key[op]
            if key not in graphs.graphs and key not in graph_status:
                st, _cap = graphs.try_capture(key)
                graph_status[key] = st
            if graphs.graphs.get(key) is not None:
                t_graph = graphs.replay_ms(key, iters=n, warmup=warmup)
                per_op_graph[op] = t_graph
        lbl = _OP_LABEL.get(op, str(op))
        gtxt = f"{t_graph:9.3f}" if t_graph is not None else "   (no-graph)"
        print(f"  {lbl:>5} {len(live):>5} {t_dense:12.3f}    {t_pos:15.3f}    "
              f"{gtxt:>18} {t_dense/max(t_pos,1e-9):9.1f}x", flush=True)

    # -- weighted-mix ms/step (the headline numbers) ------------------------
    def _mix(times: Dict[int, float]) -> float:
        return sum(times[op] * w for op, w in _OP_MIX.items() if op in times)

    mix_dense = _mix(per_op_dense)
    mix_pos = _mix(per_op_pos)
    pos_best = {op: per_op_graph.get(op, per_op_pos[op]) for op in _OP_MIX}
    mix_pos_best = _mix(pos_best)
    print(f"\n  --- WEIGHTED OP-MIX ms/step at S={S} ---", flush=True)
    print(f"    full-{nb} dense-over-pos          : {t_full:9.3f} ms/step  (1.0x)",
          flush=True)
    print(f"    block-skip, dense-over-pos       : {mix_dense:9.3f} ms/step  "
          f"({t_full/max(mix_dense,1e-9):.1f}x vs full)", flush=True)
    print(f"    POSITION-SPARSE (eager)          : {mix_pos:9.3f} ms/step  "
          f"({mix_dense/max(mix_pos,1e-9):.1f}x vs dense-skip,  "
          f"{t_full/max(mix_pos,1e-9):.1f}x vs full)", flush=True)
    if per_op_graph:
        print(f"    POSITION-SPARSE + CUDA-graph     : {mix_pos_best:9.3f} ms/step  "
              f"({mix_dense/max(mix_pos_best,1e-9):.1f}x vs dense-skip,  "
              f"{t_full/max(mix_pos_best,1e-9):.1f}x vs full)   <-- COMPOSED",
              flush=True)

    # -- DIV focus: the 179-block divmod span, dense-over-pos vs query-row --
    div = isa.DIV
    div_live = pos_runner.live_index[div]
    print(f"\n  --- DIV focus (the {len(div_live)}-block divmod span) at S={S} ---",
          flush=True)
    print(f"    DIV dense-over-pos (block-skip)  : {per_op_dense[div]:9.3f} ms/step",
          flush=True)
    print(f"    DIV position-sparse (eager)      : {per_op_pos[div]:9.3f} ms/step  "
          f"({per_op_dense[div]/max(per_op_pos[div],1e-9):.1f}x)", flush=True)
    if div in per_op_graph:
        print(f"    DIV position-sparse (graph)      : {per_op_graph[div]:9.3f} "
              f"ms/step  ({per_op_dense[div]/max(per_op_graph[div],1e-9):.1f}x)",
              flush=True)

    # -- nnz-MAC accounting side-by-side (the claimed 111-536x) -------------
    if report_macs:
        dm_dense = sum(block_macs(model.blocks[bi], S, False) for bi in div_live)
        dm_pos = sum(block_macs(model.blocks[bi], S, True) for bi in div_live)
        print(f"    DIV nnz-MACs dense={dm_dense:,} pos-sparse={dm_pos:,}  "
              f"({dm_dense/max(1,dm_pos):.0f}x MAC reduction)", flush=True)
        print(f"    -> latency speedup {per_op_dense[div]/max(per_op_pos[div],1e-9):.1f}x"
              f" vs MAC reduction {dm_dense/max(1,dm_pos):.0f}x  "
              f"(gap = K/V-at-all-pos + launch overhead)", flush=True)

    if graphs is not None:
        n_ok = sum(1 for k in graphs.distinct_keys if graphs.graphs.get(k) is not None)
        print(f"\n  CUDA graphs (pos-sparse query-row-only): captured "
              f"{n_ok}/{len(graphs.distinct_keys)} distinct op-class shapes", flush=True)
        for k in sorted(graphs.distinct_keys, key=len):
            ops = ",".join(_OP_LABEL.get(o, str(o)) for o in graphs._seen[k])
            st = "ok" if graphs.graphs.get(k) is not None else graph_status.get(k, "?")
            print(f"    shape live={len(k):3d}  [{ops}]  -> {st}", flush=True)
        graphs.graphs.clear()
        del graphs

    return {"S": S, "full": t_full, "dense_skip": mix_dense,
            "pos_sparse": mix_pos, "pos_sparse_graph": mix_pos_best,
            "div_dense": per_op_dense[div], "div_pos": per_op_pos[div],
            "div_graph": per_op_graph.get(div)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--S", type=str, default="300,900",
                    help="comma list of production seq lengths to bench.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[pos-sparse] CUDA unavailable; falling back to cpu", file=sys.stderr)
        device = "cpu"
    idx = 0
    if device.startswith("cuda") and ":" in device:
        idx = int(device.split(":")[1])
    if device.startswith("cuda") and not args.no_wait:
        wait_for_gpu(idx, min_free_gb=args.min_free_gb, stable_s=args.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"      # this bench measures the pos-sparse lever
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"dev={device} build={time.time()-t0:.1f}s", flush=True)

    if args.no_verify:
        print("\n[verify] SKIPPED (--no-verify); bench only", flush=True)
        ok = True
    else:
        print("\n[verify] full-238 dense-over-positions == COMPOSED "
              "position-sparse (+direct-CAM):", flush=True)
        ok = verify_composition(model, L, direct_cam=True)
        print(f"[verify] {'ALL BYTE-EXACT' if ok else 'DIVERGENCE FOUND'}", flush=True)
    if not ok:
        print("[verify] position-sparse composition NOT byte-exact -> abort bench",
              flush=True)
        return 1

    results = []
    for S in (int(s) for s in args.S.split(",") if s.strip()):
        results.append(bench_at_S(model, L, S, n=args.n,
                                  use_graphs=(not args.no_graphs)))
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # -- the honest self-emulation-wall composition (pos-sparse ms/step x steps) --
    print(f"\n{'='*80}\n[self-emulation wall] pos-sparse ms/step x 23M steps "
          f"(a04549a2 position-sparse step count)\n{'='*80}", flush=True)
    for r in results:
        best = r["pos_sparse_graph"] if r.get("pos_sparse_graph") else r["pos_sparse"]
        hrs = best * 23e6 / 1e3 / 3600.0
        hrs_full = r["full"] * 23e6 / 1e3 / 3600.0
        print(f"  S={r['S']:5d}: pos-sparse {best:.3f} ms/step x 23M = {hrs:8.1f} hr "
              f"(vs full-238 {r['full']:.3f} ms/step = {hrs_full:8.1f} hr)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
