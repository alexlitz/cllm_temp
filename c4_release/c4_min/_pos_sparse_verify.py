"""STEP 2 verify + STEP 3 payoff for the POSITION-SPARSE forward.

VERIFY: run a corpus through BOTH the full 238-block full-position driver and the
position-sparse driver (``PositionSparseRunner``: op-level block-skip + query-row-
only heavy compute) and assert the per-step AX trace is BYTE-IDENTICAL.

PAYOFF (STEP 3): the effective NONZERO-weight MAC count with position-sparsity vs
the dense-over-positions count, and (on GPU) ms/step, especially DIV/MOD.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch

from . import isa
from . import nibble_pure_forward_complete as pfc
from .nibble_pure_forward_complete import (
    make_overlay_complete, _build_frame, SP_INIT,
)
from .pos_sparse_forward import (
    PositionSparseRunner, block_macs, _block_query_only,
)
from ._step_block_skip_verify import _corpus


def _run_driver(model, L, code, runner: Optional[PositionSparseRunner],
                max_steps=64, seed_mem=None):
    """Pure-forward driver; block loop optionally goes through the position-sparse
    ``runner``.  ``runner=None`` -> full 238-block full-position forward (baseline).
    Returns (trace, n_block_applies, max_S)."""
    from .nibble_pure_forward_complete import _seed_frames, _mem_top
    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    frame_idx = n_seed
    trace: List[int] = []
    total_applies = 0
    max_S = 0
    dev = model.embed.device
    for _ in range(max_steps):
        overlay = make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream], device=dev)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        max_S = max(max_S, len(stream))
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            if runner is None:
                for blk in model.blocks:
                    x = blk(x)
                total_applies += len(model.blocks)
            else:
                x = runner.forward(x, op)
                total_applies += runner.live_count(op)
        state = x[0, -1]
        pc = pfc._snap_lane(state[L.PC_VAL].cpu())
        sp = pfc._snap_lane(state[L.SP_VAL].cpu())
        bp = pfc._snap_lane(state[L.BP_VAL].cpu())
        stk = pfc._snap_lane(state[L.STK_VAL].cpu())
        halted = float(state[L.HALTED]) > 0.5
        ax = pfc._decode_reg_from_nibbles(state.cpu(), L, L.AX)
        s_addr = s_val = 0; is_store = False
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
    return trace, total_applies, max_S


def verify(model, L, verbose=True) -> bool:
    runner = PositionSparseRunner(model, L)
    ok = True
    for name, prog, seed in _corpus():
        code = isa.assemble(prog)
        base_trace, _, _ = _run_driver(model, L, code, None, seed_mem=seed)
        ps_trace, _, _ = _run_driver(model, L, code, runner, seed_mem=seed)
        match = base_trace == ps_trace
        ok = ok and match
        if verbose:
            print(f"  {name:12s} {'OK ' if match else 'FAIL'} "
                  f"trace={base_trace[:6]}"
                  f"{'...' if len(base_trace) > 6 else ''}", flush=True)
        if not match and verbose:
            print(f"    BASE={base_trace}\n    POSSPARSE={ps_trace}", flush=True)
    return ok


def macs(model, L, S_list=(121, 300, 900)):
    """MAC payoff: for representative ops, dense-over-positions vs query-row-only."""
    runner = PositionSparseRunner(model, L)
    tot_nz = 0
    for b in {id(bb): bb for bb in model.blocks}.values():
        for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o):
            d = (w.dense if getattr(w, "dense", None) is not None else
                 (w.dense_resident if getattr(w, "dense_resident", None) is not None
                  else w.csr.to_dense()))
            tot_nz += int((d != 0).sum().item())
        for w in (b.ffn.W_up, b.ffn.W_gate, b.ffn.W_down):
            if hasattr(w, "dense") or hasattr(w, "csr") or hasattr(w, "dense_resident"):
                d = (w.dense if getattr(w, "dense", None) is not None else
                     (w.dense_resident if getattr(w, "dense_resident", None) is not None
                      else w.csr.to_dense()))
            else:
                d = w
            tot_nz += int((d != 0).sum().item())
    print(f"[macs] total distinct nonzero weights in model = {tot_nz:,}", flush=True)
    rep = {"IMM": isa.IMM, "ADD": isa.ADD, "MUL": isa.MUL, "DIV": isa.DIV,
           "MOD": isa.MOD, "SHL": isa.SHL, "LI": isa.LI, "EQ": isa.EQ}
    for S in S_list:
        print(f"\n[macs] S={S}  (op: dense-over-positions -> query-row-only)",
              flush=True)
        for nm, op in rep.items():
            live = runner.live_index[op]
            dense = sum(block_macs(model.blocks[bi], S, False) for bi in live)
            ps = sum(block_macs(model.blocks[bi], S, True) for bi in live)
            print(f"  {nm:4s} live={len(live):3d}  "
                  f"dense={dense:>13,}  pos-sparse={ps:>12,}  "
                  f"({dense/max(1,ps):.1f}x fewer MACs)", flush=True)


def bench(model, L, prod_S: int = 900, n: int = 30, warmup: int = 5):
    runner = PositionSparseRunner(model, L)
    dev = model.embed.device
    cuda = (dev.type == "cuda")
    toks = torch.zeros(1, prod_S, dtype=torch.long, device=dev)
    x0 = model.embed[toks].clone()

    def _time(fn):
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

    def full():
        x = x0
        with torch.no_grad():
            for blk in model.blocks:
                x = blk(x)
        return x

    t_full = _time(full)
    print(f"\n[bench] prod_S={prod_S} n={n}  FULL 238-block full-pos = "
          f"{t_full:.3f} ms/step", flush=True)
    rep = {"IMM": isa.IMM, "ADD": isa.ADD, "MUL": isa.MUL, "DIV": isa.DIV,
           "SHL": isa.SHL, "LI": isa.LI}
    for nm, op in rep.items():
        def ps(op=op):
            with torch.no_grad():
                return runner.forward(x0, op)
        t = _time(ps)
        print(f"  pos-sparse {nm:4s} live={runner.live_count(op):3d}/238  "
              f"{t:.3f} ms/step  ({t_full/t:.1f}x faster)", flush=True)


def main(code_size: int = 32, device: str = "cpu", do_bench: bool = False,
         S_list=(121, 300, 900)):
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device); model.materialize_dense(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.dim} dev={device} "
          f"build={time.time()-t0:.1f}s", flush=True)
    print("[verify] byte-exact full-pos vs position-sparse:", flush=True)
    ok = verify(model, L)
    print(f"[verify] {'ALL BYTE-EXACT' if ok else 'DIVERGENCE FOUND'}", flush=True)
    macs(model, L, S_list=S_list)
    if do_bench:
        bench(model, L, prod_S=max(S_list))
    return ok


if __name__ == "__main__":
    import sys
    dev = "cuda" if "--cuda" in sys.argv else "cpu"
    do_bench = "--bench" in sys.argv
    main(device=dev, do_bench=do_bench)
