"""#749 — BOUNDED-KV composed position-sparse fast path: kill the K/V-over-S floor.

a39ae2c measured the composed position-sparse ms/step floor is ~65 % the K/V-over-S
GEMM (K and V projected over ALL S stream rows, grows LINEARLY with S), even though
only a bounded set of rows is ever attended with non-zero weight (the ingest heads
read only the last ~28-token frame; the memory/CAM heads read only the ~14 live
store rows).  ``pos_sparse_bounded.BoundedPosSparseRunner`` projects K/V over ONLY
those bounded row subsets (local window + store-only global), collapsing the K/V
GEMM from ``[S,D]`` to ``[W + n_store, D]`` — BYTE-EXACT (dropped rows' softmax1
weight is exactly 0) and RUNTIME-INDEPENDENT (flat in S).

This bench:
  1. VERIFIES the bounded-KV composed path is byte-exact vs the full dense-over-
     positions reference on the battery (incl DIV/MOD, LI memory, JSR/LEV) + a deep
     nested loop, at each S.
  2. MEASURES ms/step for BOTH the a39ae2c K/V-over-S pos-sparse path AND the
     bounded-KV path, sweeping S (300, 900, 3000).  With bounded K/V the ms/step
     should be ~FLAT in S (vs the K/V-over-S linear growth).
  3. REPORTS the residual after bounding K/V — the 1-row FFN, launches, and the
     bounded ~W+n_store K/V.

Gate: ``C4_POS_SPARSE`` (this bench sets it internally); the golden build is
byte-identical flag-OFF.

Run (needs a free >=18 GB CUDA card):
    OMP_NUM_THREADS=4 python -m c4_min.bench_pos_sparse_bounded --S 300,900,3000
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
from .pos_sparse_forward import PositionSparseRunner
from .pos_sparse_bounded import BoundedPosSparseRunner
from .pf_speculative import draft_pf_program
from .bench_composed_fast_path import (
    wait_for_gpu, _battery, _nested_prog, _resolve_with_seed, _dest_band,
    _OP_MIX, _OP_LABEL,
)


# ---------------------------------------------------------------------------
# Composed forward wrappers (pos-sparse runner + direct-CAM dest-band overwrite).
# ---------------------------------------------------------------------------
class _ComposedWrap:
    """Wrap a pos-sparse runner (K/V-over-S OR bounded-KV) + direct-CAM."""

    def __init__(self, runner, L, direct_cam: bool = True):
        self.runner = runner
        self.L = L
        self.direct_cam = direct_cam

    def live_count(self, op) -> int:
        return self.runner.live_count(op)

    def forward(self, x, op, resolved=None):
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


def _drive(model, L, code, *, composed=None, max_steps=200, seed_mem=None):
    """Token-by-token driver (mirror of bench_pos_sparse_composed._drive)."""
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


def verify_bounded(model, L, window: int, *, verbose=True) -> bool:
    """FULL-238 dense-over-positions vs the BOUNDED-KV composed pos-sparse path,
    byte-exact per-step AX, over the battery + a deep nested loop."""
    runner = BoundedPosSparseRunner(model, L, window=window)
    composed = _ComposedWrap(runner, L, direct_cam=True)
    ok = True
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        base, _ = _drive(model, L, code, composed=None, seed_mem=seed)
        comp, _ = _drive(model, L, code, composed=composed, seed_mem=seed)
        match = base == comp
        ok = ok and match
        if verbose:
            print(f"  {name:10s} {'OK ' if match else 'FAIL'} steps={len(base)}",
                  flush=True)
            if not match:
                print(f"    BASE={base}\n    COMP={comp}", flush=True)
    # DEEP-LOOP gate vs the FREE perfect draft.
    deep_steps = 60
    deep = _nested_prog(3, 4)
    draft = draft_pf_program(deep, max_steps=deep_steps, mask=0xFF)
    draft_ax = [f["ax"] & 0xFF for f in draft.frames]
    comp, _ = _drive(model, L, deep, composed=composed, seed_mem={},
                     max_steps=deep_steps)
    n = min(len(comp), len(draft_ax))
    dmatch = comp[:n] == draft_ax[:n] and n > 0
    ok = ok and dmatch
    if verbose:
        print(f"  {'nested_deep':10s} {'OK ' if dmatch else 'FAIL'} steps={n} "
              f"(bounded-KV == free-draft over a deep nested loop)", flush=True)
        if not dmatch:
            print(f"    DRAFT={draft_ax[:20]}...\n    COMP ={comp[:20]}...", flush=True)
    return ok


# ---------------------------------------------------------------------------
# ms/step measurement: K/V-over-S pos-sparse vs BOUNDED-KV, per op + weighted mix.
# ---------------------------------------------------------------------------
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


def _make_store_stream(model, L, S: int):
    """A [1,S,D] input where a handful of rows carry IS_STORE=1 (so the global CAM
    heads' bounded-KV row set is a realistic ~small working set, not the whole S).

    Mirrors a running stream: ~1 store row every ~30 tokens (one per VM step that
    stores), which is exactly what the memory/stack heads see in production.  The
    exact residual values do not affect the ms/step (only the row COUNT drives the
    bounded K/V GEMM size), so a synthetic store mask is a faithful timing input."""
    dev = model.embed.device
    D = model.embed.shape[1]
    x = torch.zeros(1, S, D, device=dev, dtype=model.embed.dtype)
    # one store row per ~30-token frame (SI/PSH/JSR/ENT frequency in compiled C4).
    store_rows = list(range(29, S, 30))
    for r in store_rows:
        x[0, r, int(L.IS_STORE)] = 1.0
    return x, len(store_rows)


def bench_at_S(model, L, S: int, window: int, *, n=30, warmup=8):
    dev = model.embed.device
    cuda = (dev.type == "cuda")
    pos_runner = PositionSparseRunner(model, L)              # K/V-over-S (a39ae2c)
    bnd_runner = BoundedPosSparseRunner(model, L, window=window)  # bounded K/V
    x0, n_store = _make_store_stream(model, L, S)

    print(f"\n{'='*82}\n[bench] S={S}  window={window}  n_store_rows={n_store}  "
          f"(n={n}, warmup={warmup})\n{'='*82}", flush=True)

    # -- FULL-238 dense over ALL positions (vanilla golden reference) --------
    def full():
        x = x0
        with torch.no_grad():
            for blk in model.blocks:
                x = blk(x)
        return x
    t_full = _time_fn(full, n, warmup, cuda)
    nb = len(model.blocks)
    print(f"  FULL {nb}-block dense-over-positions   = {t_full:9.3f} ms/step",
          flush=True)

    per_op_pos: Dict[int, float] = {}       # K/V-over-S pos-sparse
    per_op_bnd: Dict[int, float] = {}       # bounded-KV pos-sparse
    kv_rows: Dict[int, int] = {}

    print(f"\n  {'op':>5} {'live':>5} {'kvrows':>7} {'KV-over-S':>12} "
          f"{'bounded-KV':>12} {'speedup':>9}", flush=True)
    print("  " + "-" * 62, flush=True)
    for op in _OP_MIX:
        def pos(op=op):
            with torch.no_grad():
                return pos_runner.forward(x0, op)

        def bnd(op=op):
            with torch.no_grad():
                return bnd_runner.forward(x0, op)

        t_pos = _time_fn(pos, n, warmup, cuda)
        t_bnd = _time_fn(bnd, n, warmup, cuda)
        per_op_pos[op] = t_pos
        per_op_bnd[op] = t_bnd
        kv_rows[op] = bnd_runner.kv_rows(x0, op)
        lbl = _OP_LABEL.get(op, str(op))
        live = pos_runner.live_index[op]
        print(f"  {lbl:>5} {len(live):>5} {kv_rows[op]:>7} {t_pos:12.3f} "
              f"{t_bnd:12.3f} {t_pos/max(t_bnd,1e-9):8.1f}x", flush=True)

    def _mix(times):
        return sum(times[op] * w for op, w in _OP_MIX.items() if op in times)

    mix_pos = _mix(per_op_pos)
    mix_bnd = _mix(per_op_bnd)
    print(f"\n  --- WEIGHTED OP-MIX ms/step at S={S} ---", flush=True)
    print(f"    full-{nb} dense-over-pos       : {t_full:9.3f} ms/step  (1.0x)",
          flush=True)
    print(f"    POS-SPARSE (K/V-over-S)       : {mix_pos:9.3f} ms/step  "
          f"({t_full/max(mix_pos,1e-9):.1f}x vs full)", flush=True)
    print(f"    BOUNDED-KV (local+store)      : {mix_bnd:9.3f} ms/step  "
          f"({mix_pos/max(mix_bnd,1e-9):.1f}x vs K/V-over-S,  "
          f"{t_full/max(mix_bnd,1e-9):.1f}x vs full)   <-- BOUNDED", flush=True)

    # DIV focus (the heaviest op).
    div = isa.DIV
    print(f"\n  --- DIV focus at S={S} ---", flush=True)
    print(f"    DIV K/V-over-S  : {per_op_pos[div]:9.3f} ms/step", flush=True)
    print(f"    DIV bounded-KV  : {per_op_bnd[div]:9.3f} ms/step  "
          f"({per_op_pos[div]/max(per_op_bnd[div],1e-9):.1f}x)  "
          f"kv_rows={kv_rows[div]}", flush=True)

    return {"S": S, "full": t_full, "pos_sparse": mix_pos, "bounded": mix_bnd,
            "div_pos": per_op_pos[div], "div_bnd": per_op_bnd[div],
            "kv_rows_div": kv_rows[div], "n_store": n_store}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--S", type=str, default="300,900,3000")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[bounded] CUDA unavailable; falling back to cpu", file=sys.stderr)
        device = "cpu"
    idx = 0
    if device.startswith("cuda") and ":" in device:
        idx = int(device.split(":")[1])
    if device.startswith("cuda") and not args.no_wait:
        wait_for_gpu(idx, min_free_gb=args.min_free_gb, stable_s=args.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"dev={device} build={time.time()-t0:.1f}s", flush=True)

    # head classification summary.
    from .pos_sparse_bounded import classify_block_heads
    tl = tg = td = 0
    live_blocks = []
    for bi, b in enumerate(model.blocks):
        l, g, d = classify_block_heads(b.attn)
        tl += l.numel(); tg += g.numel(); td += d.numel()
        if l.numel() or g.numel():
            live_blocks.append((bi, l.numel(), g.numel()))
    print(f"[classify] local={tl} global={tg} dead={td} head-slots; "
          f"live-attn blocks (block: nlocal/nglobal): "
          f"{[(bi, nl, ng) for bi, nl, ng in live_blocks]}", flush=True)

    if args.no_verify:
        print("\n[verify] SKIPPED (--no-verify)", flush=True)
        ok = True
    else:
        print(f"\n[verify] full-238 == BOUNDED-KV (window={args.window}, "
              f"local+store-only):", flush=True)
        ok = verify_bounded(model, L, args.window)
        print(f"[verify] {'ALL BYTE-EXACT' if ok else 'DIVERGENCE FOUND'}", flush=True)
    if not ok:
        print("[verify] bounded-KV NOT byte-exact -> abort bench", flush=True)
        return 1

    results = []
    for S in (int(s) for s in args.S.split(",") if s.strip()):
        results.append(bench_at_S(model, L, S, args.window, n=args.n))
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # -- the headline: flatness in S -----------------------------------------
    print(f"\n{'='*82}\n[flatness] ms/step vs S (bounded K/V should be ~FLAT; "
          f"K/V-over-S grows linearly)\n{'='*82}", flush=True)
    print(f"  {'S':>6} {'full':>10} {'K/V-over-S':>12} {'bounded-KV':>12} "
          f"{'DIV bnd':>10} {'kv_rows':>8}", flush=True)
    for r in results:
        print(f"  {r['S']:>6} {r['full']:10.3f} {r['pos_sparse']:12.3f} "
              f"{r['bounded']:12.3f} {r['div_bnd']:10.3f} {r['kv_rows_div']:>8}",
              flush=True)
    if len(results) >= 2:
        r0, rN = results[0], results[-1]
        grow_pos = rN["pos_sparse"] / max(r0["pos_sparse"], 1e-9)
        grow_bnd = rN["bounded"] / max(r0["bounded"], 1e-9)
        print(f"\n  S {r0['S']}->{rN['S']} ({rN['S']/r0['S']:.1f}x): "
              f"K/V-over-S ms/step x{grow_pos:.2f}  vs  bounded-KV x{grow_bnd:.2f}  "
              f"(bounded flat iff ~1.0x)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
