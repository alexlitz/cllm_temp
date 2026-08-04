"""#747 — WITHIN-PROGRAM BIG-K batching on the BOUNDED-KV pos-sparse path.

Verify K VM steps per forward on the ``pf_kbatch.KBatchBoundedRunner`` (bounded-KV,
flat-in-S #746 path), to raise the ~0.004% FLOP util and cut the forward count to
steps/K.  The perfect draft (``pf_speculative.draft_pf_program``) materialises the K
steps deterministically (CPU, ~us/step); the model verifies K at a time -> forwards =
steps/K.

This bench:
  1. BYTE-EXACT: the K-batched verify decodes the identical K register frames as K
     sequential single-step bounded decodes, on the battery incl DIV/MOD, mem,
     JSR/LEV, and a nested loop, for every K in {1,2,4,8,16,32,64}.
  2. MEASURES per K: the ms/FORWARD, the effective ms/step (ms-forward / K), the GPU
     FLOP util, and kv_rows (should stay bounded).
  3. SELF-EMU WALL: (23.2M / K) forwards x ms-forward, vs the 18hr (K=1) baseline;
     reports the K that minimizes it and where it saturates.

Gate: ``C4_POS_SPARSE`` (set internally); golden flag-OFF untouched.

Run (needs a free >=18 GB CUDA card):
    OMP_NUM_THREADS=4 python -m c4_min.bench_pf_kbatch --K 1,2,4,8,16,32,64
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
from .pos_sparse_bounded import BoundedPosSparseRunner
from .pf_kbatch import KBatchBoundedRunner
from .pf_speculative import draft_pf_program
from .bench_composed_fast_path import (
    wait_for_gpu, _battery, _nested_prog, _resolve_with_seed, _dest_band,
)

SELF_EMU_STEPS = 23_200_000        # the self-emulation step count (memory note)


# ---------------------------------------------------------------------------
# Build the teacher-forced K-frame stream from a program + seed, decode all K
# query rows.  Mirrors bench_pos_sparse_bounded._drive but batches all steps.
# ---------------------------------------------------------------------------
def _prep_stream(model, L, code, *, max_steps, seed_mem):
    """Draft the program (CPU) and build the FULL teacher-forced token stream +
    the per-frame resolved CAM reads (for direct-CAM) + the per-step query row
    positions.  Returns (stream_tokens, q_positions, ops, resolved_by_qpos,
    draft_ax, n_seed)."""
    from . import nibble_pure_forward_complete as pfc
    from .nibble_pure_forward_complete import _build_frame, SP_INIT, _seed_frames

    seed_frames, _ = _seed_frames(seed_mem or {})
    n_seed = len(_seed_frames(seed_mem or {})[1])
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF)
    resolved_by_frame = _resolve_with_seed(draft, seed_mem or {})

    # stream = BOS + seed frames + init frame + one frame per drafted step.
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    # query row for step s = the LAST token of frame s (the row before frame s+1).
    # frame 0 = the init frame; step s appends frame s+1.  The driver decodes step
    # s from the last row of the stream BEFORE appending step s's frame, i.e. the
    # last token of the previous frame.  With seed frames leading, that is:
    #   base = 1 + n_seed*30 + 30  (end of init frame) - 1  == last tok of init.
    # We reconstruct the exact per-step stream + query positions below.
    q_positions: List[int] = []
    ops: List[int] = []
    draft_ax: List[int] = []
    resolved_by_qpos: Dict[int, list] = {}
    store_log: Dict[int, Tuple[int, int]] = {}
    # seed the store_log with the seed frames (they precede all program frames).
    _, store_log = _seed_frames(seed_mem or {})
    cur_pc = 0
    frame_idx = n_seed
    # STACK0-MIRROR recurrence.  The draft's ``f["stk"]`` is only updated on POP ops,
    # but the DRIVER's stk lane = the model's STACK0 = the MEM_VAL of the PREVIOUS
    # emitted frame (a store frame's s_val, else the persisting mirror).  Reproduce
    # it exactly: ``stk_N = MEM_VAL(frame N-1)`` (init frame's MEM_VAL = 0), else the
    # ADD operand-B the model ingests from STACK0 is WRONG (proven: without this, the
    # teacher-forced ADD reads 0 not the pushed value).
    prev_mem_val = 0                               # MEM_VAL of the init frame
    for s, f in enumerate(draft.frames):
        # the query row for THIS step is the last row of the current stream.
        q_positions.append(len(stream) - 1)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        ops.append(op)
        draft_ax.append(f["ax"] & 0xFF)
        draft_frame = frame_idx + 1 - n_seed
        resolved = resolved_by_frame.get(draft_frame, [])
        resolved_by_qpos[len(stream) - 1] = resolved
        # append this step's frame (teacher-forced from the draft), with the stk
        # mirror = the previous frame's MEM_VAL.
        is_store = f["is_store"]
        s_addr, s_val = f["s_addr"], f["s_val"]
        stk = prev_mem_val
        # the DRIVER emits the frame from the DIRECT-CAM-corrected register state: a
        # "mem" load (LI/LC) writes its resolved value into AX.  The draft's f["ax"]
        # is the EMPTY-memory result (0 for a SEED-only address), so override AX with
        # the resolved load value — else the NEXT step ingests the wrong AX (proven:
        # seeded ``li`` HALT reads 0 without this).
        frame_ax = f["ax"]
        for r in resolved:
            if r.head == "mem":
                frame_ax = r.value
        frame = _build_frame(f["pc"], frame_ax, f["sp"], f["bp"], stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        # this frame's MEM_VAL (store s_val, else the mirror stk) feeds the next stk.
        prev_mem_val = (s_val if is_store else stk) & 0xFFFFFFFF
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        cur_pc = f["pc"]
    return stream, q_positions, ops, resolved_by_qpos, draft_ax, store_log


def _decode_qrow(state, L):
    from . import nibble_pure_forward_complete as pfc
    ax = pfc._decode_reg_from_nibbles(state, L, L.AX)
    return ax


def _apply_direct_cam(state, L, resolved):
    for r in resolved:
        band = _dest_band(L, r.head)
        for j, nv in enumerate(V.nibbles_of_value(r.value & 0xFFFFFFFF, NIB_PER_REG)):
            state[band + j] = float(nv)
    return state


def drive_kbatch(model, L, runner: KBatchBoundedRunner, code, *,
                 K: int, max_steps=200, seed_mem=None, graph=False, perrow=False,
                 perrow_graphed=False):
    """Verify the whole program K VM steps per forward on the bounded-KV path.

    ``graph=True`` routes through the MEGAKERNEL (CUDA-graphed passthrough-FFN
    tail); ``perrow=True`` routes through the #874 PER-ROW block-skip forward_span
    (each block runs only the query rows whose op uses it, not the union of all K);
    ``perrow_graphed=True`` routes through the GROUPED graphed per-row path (same
    per-row skip, consecutive same-subset FFN runs compacted + CUDA-graphed).
    Returns (ax_trace, n_forwards)."""
    from .nibble_pure_forward_complete import make_overlay_complete
    stream, q_positions, ops, resolved_by_qpos, draft_ax, store_log = _prep_stream(
        model, L, code, max_steps=max_steps, seed_mem=seed_mem)
    n_steps = len(q_positions)
    dev = model.embed.device
    toks = torch.tensor([stream], device=dev)
    # ONE overlay of the whole teacher-forced stream; each query row gets the
    # all-ROLE ingest tag re-applied below (make_overlay_complete tags only the
    # last row, so we re-tag all K query rows explicitly).
    overlay = make_overlay_complete(code, L, store_log=store_log)
    with torch.no_grad():
        x_full = model.embed[toks].clone()
        overlay(x_full)
        from .nibble_pure_forward import N_ROLES
        for q in q_positions:
            for role in range(N_ROLES):
                x_full[0, q, L.ROLE + role] = 1.0
    # SELF-EMU DIRECT-CAM (C4_SELFEMU_DIRECT_CAM): arm the runner's global CAM heads
    # to DIRECT-GATHER the resolved values (O(1)) instead of softmax1 over the growing
    # store set (O(n_store)).  The per-qpos resolved reads are already computed above.
    from .selfemu_direct_cam import (selfemu_direct_cam_enabled,
                                     build_direct_cam_table)
    dcam_on = selfemu_direct_cam_enabled()
    if dcam_on:
        dcam_tbl = build_direct_cam_table(resolved_by_qpos)
        runner.arm_direct_cam(dcam_tbl)
    trace: List[int] = []
    n_forwards = 0
    s = 0
    while s < n_steps:
        e = min(s + K, n_steps)
        q_idxs = q_positions[s:e]
        span_ops = ops[s:e]
        with torch.no_grad():
            if perrow_graphed:
                x = runner.forward_span_perrow_graphed(x_full, span_ops, q_idxs)
            elif perrow:
                x = runner.forward_span_perrow(x_full, span_ops, q_idxs)
            elif graph:
                x = runner.forward_span_graphed(x_full, span_ops, q_idxs)
            else:
                x = runner.forward_span(x_full, span_ops, q_idxs)
        n_forwards += 1
        for q in q_idxs:
            state = x[0, q].clone()
            state = _apply_direct_cam(state, L, resolved_by_qpos.get(q, []))
            trace.append(_decode_qrow(state.cpu(), L) & 0xFF)
        s = e
    if dcam_on:
        runner.arm_direct_cam(None)          # disarm (clean state between programs)
    return trace, n_forwards


def drive_seq_bounded(model, L, code, *, max_steps=200, seed_mem=None):
    """The K=1 sequential bounded-KV reference (BoundedPosSparseRunner, one forward
    per step over the GROWING stream) — the byte-exact ground truth to match."""
    from . import nibble_pure_forward_complete as pfc
    from .nibble_pure_forward_complete import (
        make_overlay_complete, _build_frame, SP_INIT, _seed_frames, _mem_top)
    seq_runner = BoundedPosSparseRunner(model, L, window=runner_window)
    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF)
    resolved_by_frame = _resolve_with_seed(draft, seed_mem or {})
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    frame_idx = n_seed
    trace: List[int] = []
    dev = model.embed.device
    for _ in range(max_steps):
        overlay = make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream], device=dev)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            x = seq_runner.forward(x, op)
            state = x[0, -1].clone()
            draft_frame = frame_idx + 1 - n_seed
            resolved = resolved_by_frame.get(draft_frame, [])
            state = _apply_direct_cam(state, L, resolved)
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
    return trace


runner_window = 64   # set from args in main()


# ---------------------------------------------------------------------------
# BYTE-EXACT verify: K-batch trace == sequential bounded trace, per K.
# ---------------------------------------------------------------------------
def verify_kbatch(model, L, runner, K: int, *, verbose=True, graph=False) -> bool:
    ok = True
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        base = drive_seq_bounded(model, L, code, seed_mem=seed)
        kb, _ = drive_kbatch(model, L, runner, code, K=K, seed_mem=seed, graph=graph)
        n = min(len(base), len(kb))
        match = base[:n] == kb[:n] and len(base) == len(kb)
        ok = ok and match
        if verbose:
            print(f"    {name:10s} {'OK ' if match else 'FAIL'} steps={len(base)}",
                  flush=True)
            if not match:
                print(f"      SEQ ={base}\n      KB{K}={kb}", flush=True)
    # deep nested loop vs the free draft.
    deep_steps = 60
    deep = _nested_prog(3, 4)
    draft = draft_pf_program(deep, max_steps=deep_steps, mask=0xFF)
    draft_ax = [f["ax"] & 0xFF for f in draft.frames]
    kb, _ = drive_kbatch(model, L, runner, deep, K=K, seed_mem={},
                         max_steps=deep_steps, graph=graph)
    n = min(len(kb), len(draft_ax))
    dmatch = kb[:n] == draft_ax[:n] and n > 0
    ok = ok and dmatch
    if verbose:
        print(f"    {'nested_deep':10s} {'OK ' if dmatch else 'FAIL'} steps={n} "
              f"(K={K} batch == free-draft over a deep nested loop)", flush=True)
        if not dmatch:
            print(f"      DRAFT={draft_ax[:20]}...\n      KB{K} ={kb[:20]}...", flush=True)
    return ok


# ---------------------------------------------------------------------------
# ms/forward measurement.  Build a synthetic teacher-forced stream that exercises
# the bounded path at a fixed S, and time ONE K-batched forward_span over K query
# rows (the K last-tokens of the last K frames).
# ---------------------------------------------------------------------------
def _make_timing_stream(model, L, K: int, base_s: int, op: int, window: int):
    """A [1,S,D] stream with ~S rows, one store row per 30-token frame, and K query
    rows spaced 30 apart at the tail.  The exact residual values do not affect the
    ms/forward (only the row COUNT drives the bounded K/V GEMM), so a synthetic
    store mask + role tags is a faithful timing input."""
    dev = model.embed.device
    D = model.embed.shape[1]
    # S must hold at least the K frames plus a warm prefix so the store working set
    # and the local window are realistic.
    S = max(base_s, K * 30 + window + 60)
    x = torch.zeros(1, S, D, device=dev, dtype=model.embed.dtype)
    x[0, :, int(L.ONE)] = 1.0
    # one store row per ~30-token frame (realistic working set for the global CAM).
    for r in range(29, S, 30):
        x[0, r, int(L.IS_STORE)] = 1.0
    # K query rows spaced 30 apart at the tail (the last K frames' last tokens).
    q_idxs = [S - 1 - 30 * (K - 1 - j) for j in range(K)]
    q_idxs = [q for q in q_idxs if q >= 0]
    from .nibble_pure_forward import N_ROLES
    for q in q_idxs:
        for role in range(N_ROLES):
            x[0, q, L.ROLE + role] = 1.0
    return x, q_idxs, S


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


def _peak_flops(dev_idx: int) -> float:
    """Peak fp32 FLOP/s for the card (from its name), for the util estimate.  A
    coarse table; overrideable via C4_PEAK_TFLOPS."""
    env = os.environ.get("C4_PEAK_TFLOPS")
    if env:
        return float(env) * 1e12
    # the current CUDA device index (respects CUDA_VISIBLE_DEVICES remapping).
    try:
        name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
    except Exception:
        return 27.8e12   # A5000 fp32 peak (the bench card); override via C4_PEAK_TFLOPS
    table = {
        "a100": 19.5e12, "a5000": 27.8e12, "a6000": 38.7e12, "4090": 82.6e12,
        "3090": 35.6e12, "v100": 15.7e12, "a40": 37.4e12, "l40": 90.5e12,
        "h100": 67e12, "2080": 10.6e12, "titan": 16.3e12,
    }
    for k, v in table.items():
        if k in name:
            return v
    return 20e12   # conservative default


def _count_flops_forward(model, L, runner, K, q_idxs, ops, D, window, n_store):
    """Estimate the useful FLOPs in ONE K-batched forward_span (the live blocks'
    Q/K/V/O GEMMs over bounded rows + K-row FFN)."""
    live = runner.live_union(ops)
    flops = 0.0
    for bi in live:
        b = runner.kblocks[bi].b
        # FFN: K rows x (3 dense DxDff GEMMs).  Use the dense shapes (byte-exact path
        # runs dense fp64 on the query rows).
        F_ = b.ffn
        if not b.routed and hasattr(F_, "W_up"):
            def _sz(w):
                d = getattr(w, "dense_resident", None)
                if d is None:
                    d = getattr(w, "dense", None)
                return tuple(d.shape) if d is not None else (D, D)
            for w in (F_.W_up, F_.W_gate, F_.W_down):
                o, i = _sz(w)
                flops += 2.0 * K * o * i
        if b.is_passthrough:
            continue
        # attention: Q(K rows), K/V over bounded union rows, scores + ctx + O.
        nh_l = int(b.local_idx.numel())
        nh_g = int(b.global_idx.numel())
        HD = b.HD
        # Q: K rows x D x D ; O: K rows x D x D.
        flops += 2.0 * 2.0 * K * D * D
        # K/V over union rows; scores + ctx O(K * rows * HD * heads).
        rows_l = min(window, max(q_idxs) + 1)
        rows_g = n_store + 1
        flops += 2.0 * 2.0 * (rows_l + rows_g) * D * D   # K + V projections
        flops += 2.0 * 2.0 * K * (nh_l * rows_l + nh_g * rows_g) * HD  # scores+ctx
    return flops


def bench_K(model, L, runner, K: int, *, base_s: int, window: int, dev_idx: int,
            n=30, warmup=8, graph=False):
    dev = model.embed.device
    cuda = (dev.type == "cuda")
    op = isa.ADD    # a representative arith op (mid-size live-block set)
    x0, q_idxs, S = _make_timing_stream(model, L, K, base_s, op, window)
    n_store = int((x0[0, :, int(L.IS_STORE)] != 0).sum())
    ops = [op] * len(q_idxs)

    def fwd():
        with torch.no_grad():
            if graph:
                return runner.forward_span_graphed(x0, ops, q_idxs)
            return runner.forward_span(x0, ops, q_idxs)

    t_fwd = _time_fn(fwd, n, warmup, cuda)
    kv = runner.kv_rows_max(x0, q_idxs, ops)
    eff = t_fwd / max(len(q_idxs), 1)

    D = model.embed.shape[1]
    flops = _count_flops_forward(model, L, runner, K, q_idxs, ops, D, window, n_store)
    util = 0.0
    if cuda:
        peak = _peak_flops(dev_idx)
        util = (flops / (t_fwd / 1e3)) / peak * 100.0

    return {"K": K, "S": S, "ms_forward": t_fwd, "ms_eff_step": eff,
            "kv_rows": kv, "n_store": n_store, "flops": flops, "util_pct": util,
            "n_q": len(q_idxs)}


def main(argv: Optional[List[str]] = None) -> int:
    global runner_window
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--K", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--base-s", type=int, default=900)
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--fp64-all", action="store_true",
                    help="a39ae2c all-fp64 baseline (disable selective fp64)")
    ap.add_argument("--graph", action="store_true",
                    help="MEGAKERNEL: CUDA-graph the passthrough-FFN tail")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args(argv)
    runner_window = args.window

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[kbatch] CUDA unavailable; falling back to cpu", file=sys.stderr)
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

    runner = KBatchBoundedRunner(model, L, window=args.window,
                                 selective_fp64=not args.fp64_all)
    if args.fp64_all:
        runner.set_fp64_blocks(None)
    fp64_now = [bi for bi, kb in enumerate(runner.kblocks) if kb.b.fp64_ffn]
    names = list(getattr(L, "_block_names", []))
    fp64_names = [names[b] if b < len(names) else str(b) for b in fp64_now]
    print(f"[fp64] {'ALL blocks (a39ae2c baseline)' if args.fp64_all else 'SELECTIVE'}: "
          f"{len(fp64_now)}/{len(runner.kblocks)} blocks fp64 "
          f"{fp64_now if not args.fp64_all else '(all)'} "
          f"{fp64_names if not args.fp64_all else ''}", flush=True)

    Ks = [int(k) for k in args.K.split(",") if k.strip()]

    # -- BYTE-EXACT verify at each K -----------------------------------------
    if not args.no_verify:
        print(f"\n{'='*82}\n[verify] K-batch bounded == sequential bounded "
              f"(window={args.window}), per K\n{'='*82}", flush=True)
        for K in Ks:
            print(f"  --- K={K} ---", flush=True)
            ok = verify_kbatch(model, L, runner, K, graph=args.graph)
            print(f"  K={K}: {'ALL BYTE-EXACT' if ok else 'DIVERGENCE FOUND'}",
                  flush=True)
            if not ok:
                print(f"[verify] K={K} NOT byte-exact -> abort bench", flush=True)
                return 1
    else:
        print("\n[verify] SKIPPED (--no-verify)", flush=True)

    # -- ms/forward vs K ------------------------------------------------------
    print(f"\n{'='*82}\n[bench] ms/forward vs K (bounded-KV path, base_s="
          f"{args.base_s}, window={args.window})\n{'='*82}", flush=True)
    print(f"  {'K':>4} {'S':>6} {'kvrows':>7} {'ms/fwd':>10} {'ms/step':>10} "
          f"{'util%':>8} {'GFLOP/fwd':>10}", flush=True)
    print("  " + "-" * 62, flush=True)
    results = []
    for K in Ks:
        r = bench_K(model, L, runner, K, base_s=args.base_s, window=args.window,
                    dev_idx=idx, n=args.n, graph=args.graph)
        results.append(r)
        print(f"  {r['K']:>4} {r['S']:>6} {r['kv_rows']:>7} {r['ms_forward']:10.3f} "
              f"{r['ms_eff_step']:10.4f} {r['util_pct']:8.3f} "
              f"{r['flops']/1e9:10.2f}", flush=True)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # -- the headline: self-emulation wall vs K ------------------------------
    print(f"\n{'='*82}\n[self-emu wall] (23.2M / K) forwards x ms/forward, vs the "
          f"18hr K=1 baseline\n{'='*82}", flush=True)
    print(f"  {'K':>4} {'ms/fwd':>10} {'forwards':>12} {'wall(hr)':>10} "
          f"{'ms/step_eff':>12} {'vs K=1':>8}", flush=True)
    print("  " + "-" * 62, flush=True)
    wall1 = None
    best = None
    for r in results:
        forwards = SELF_EMU_STEPS / r["K"]
        wall_hr = forwards * r["ms_forward"] / 1e3 / 3600.0
        if r["K"] == 1:
            wall1 = wall_hr
        eff = r["ms_eff_step"]
        speed = (wall1 / wall_hr) if wall1 else 1.0
        print(f"  {r['K']:>4} {r['ms_forward']:10.3f} {forwards:12.0f} "
              f"{wall_hr:10.2f} {eff:12.4f} {speed:7.1f}x", flush=True)
        if best is None or wall_hr < best[1]:
            best = (r["K"], wall_hr, r)
    if best is not None:
        bk, bw, br = best
        print(f"\n  BEST K = {bk}: self-emu wall = {bw:.2f}hr "
              f"(ms/fwd={br['ms_forward']:.3f}, ms/step_eff={br['ms_eff_step']:.4f}, "
              f"FLOP util={br['util_pct']:.3f}%, kv_rows={br['kv_rows']})", flush=True)
        if wall1:
            print(f"  vs K=1 (18hr-class) baseline: {wall1/bw:.1f}x faster wall",
                  flush=True)

    # saturation note.
    if len(results) >= 2:
        print(f"\n[saturation] ms/forward growth per K doubling:", flush=True)
        for i in range(1, len(results)):
            r0, r1 = results[i - 1], results[i]
            kr = r1["K"] / max(r0["K"], 1)
            fr = r1["ms_forward"] / max(r0["ms_forward"], 1e-9)
            print(f"  K {r0['K']:>3} -> {r1['K']:>3} ({kr:.0f}x): "
                  f"ms/forward x{fr:.2f}  (linear-in-K iff ~{kr:.0f}x, "
                  f"sub-linear = amortization win)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
