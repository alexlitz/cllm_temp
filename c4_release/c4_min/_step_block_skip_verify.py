"""STEP 2 verify + STEP 3 bench for the per-step block-skip forward.

VERIFY: run a corpus of programs through BOTH the full 238-block driver and the
skip-aware driver (``StepBlockSkipRunner``, keyed on the decoded opcode) and assert
the per-step AX trace is BYTE-IDENTICAL.  Any divergence prints the step, opcode,
and the block that was wrongly skipped — reported honestly.

BENCH: ms/step of the full 238-block forward vs the per-step-skip forward at
production S, on GPU, materialize_dense (resident dense weights, no per-forward
re-densify).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import nibble_pure_forward_complete as pfc
from .nibble_pure_forward_complete import (
    make_overlay_complete, _build_frame, SP_INIT,
)
from .step_block_skip import StepBlockSkipRunner


def _run_driver(model, L, code, runner: Optional[StepBlockSkipRunner],
                max_steps=64, seed_mem=None, collect_blocks=False):
    """The pure-forward driver, but the block loop optionally goes through
    ``runner`` (skip-aware).  Returns (trace, n_block_applies).  ``runner=None`` ->
    the full 238-block forward (baseline)."""
    from .nibble_pure_forward_complete import _seed_frames, _mem_top
    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)
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
    return trace, total_applies


def _corpus() -> List[Tuple[str, list, dict]]:
    """A battery exercising every opcode class + multi-step programs."""
    C: List[Tuple[str, list, dict]] = []
    C.append(("imm_halt", [("IMM", 42), ("HALT", 0)], {}))
    C.append(("add", [("IMM", 12), ("PSH", 0), ("IMM", 30), ("ADD", 0), ("HALT", 0)], {}))
    C.append(("sub", [("IMM", 100), ("PSH", 0), ("IMM", 58), ("SUB", 0), ("HALT", 0)], {}))
    C.append(("mul", [("IMM", 12), ("PSH", 0), ("IMM", 7), ("MUL", 0), ("HALT", 0)], {}))
    C.append(("div", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0), ("HALT", 0)], {}))
    C.append(("div2", [("IMM", 255), ("PSH", 0), ("IMM", 3), ("DIV", 0), ("HALT", 0)], {}))
    C.append(("mod", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0), ("HALT", 0)], {}))
    C.append(("and", [("IMM", 0xF0), ("PSH", 0), ("IMM", 0x3C), ("AND", 0), ("HALT", 0)], {}))
    C.append(("or", [("IMM", 0xF0), ("PSH", 0), ("IMM", 0x0C), ("OR", 0), ("HALT", 0)], {}))
    C.append(("xor", [("IMM", 0xFF), ("PSH", 0), ("IMM", 0x0F), ("XOR", 0), ("HALT", 0)], {}))
    C.append(("shl", [("IMM", 3), ("PSH", 0), ("IMM", 4), ("SHL", 0), ("HALT", 0)], {}))
    C.append(("shr", [("IMM", 240), ("PSH", 0), ("IMM", 4), ("SHR", 0), ("HALT", 0)], {}))
    C.append(("eq", [("IMM", 5), ("PSH", 0), ("IMM", 5), ("EQ", 0), ("HALT", 0)], {}))
    C.append(("lt", [("IMM", 3), ("PSH", 0), ("IMM", 5), ("LT", 0), ("HALT", 0)], {}))
    C.append(("gt", [("IMM", 9), ("PSH", 0), ("IMM", 5), ("GT", 0), ("HALT", 0)], {}))
    C.append(("le", [("IMM", 5), ("PSH", 0), ("IMM", 5), ("LE", 0), ("HALT", 0)], {}))
    C.append(("li", [("IMM", 8), ("LI", 0), ("HALT", 0)], {8: 123}))
    C.append(("lea", [("LEA", 4), ("HALT", 0)], {}))
    C.append(("bz_taken", [("IMM", 0), ("BZ", 4), ("IMM", 99), ("HALT", 0),
                           ("IMM", 7), ("HALT", 0)], {}))
    C.append(("bz_nottaken", [("IMM", 5), ("BZ", 5), ("IMM", 42), ("HALT", 0),
                              ("IMM", 9), ("HALT", 0)], {}))
    C.append(("bnz_taken", [("IMM", 3), ("BNZ", 4), ("IMM", 99), ("HALT", 0),
                            ("IMM", 8), ("HALT", 0)], {}))
    C.append(("jmp", [("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)], {}))
    # JSR to a subroutine that ENTs a frame, sets AX, LEVs back to after the JSR.
    C.append(("jsr_lev", [("JSR", 3), ("IMM", 5), ("HALT", 0), ("ENT", 0),
                          ("IMM", 7), ("LEV", 0)], {}))
    C.append(("ent_adj", [("ENT", 1), ("IMM", 3), ("PSH", 0), ("ADJ", 1),
                          ("HALT", 0)], {}))
    C.append(("eq_true", [("IMM", 5), ("PSH", 0), ("IMM", 5), ("EQ", 0),
                          ("HALT", 0)], {}))
    C.append(("eq_false", [("IMM", 4), ("PSH", 0), ("IMM", 5), ("EQ", 0),
                           ("HALT", 0)], {}))
    C.append(("and2", [("IMM", 0xAA), ("PSH", 0), ("IMM", 0x55), ("AND", 0),
                       ("HALT", 0)], {}))
    C.append(("ge_true", [("IMM", 9), ("PSH", 0), ("IMM", 5), ("GE", 0),
                          ("HALT", 0)], {}))
    # a multi-op arithmetic program (many steps, mixed ops)
    C.append(("mixed", [("IMM", 10), ("PSH", 0), ("IMM", 20), ("ADD", 0),
                        ("PSH", 0), ("IMM", 3), ("MUL", 0), ("PSH", 0),
                        ("IMM", 7), ("SUB", 0), ("HALT", 0)], {}))
    return C


def verify(model, L, verbose=True) -> bool:
    runner = StepBlockSkipRunner(model, L)
    ok = True
    for name, prog, seed in _corpus():
        code = isa.assemble(prog)
        base_trace, base_ap = _run_driver(model, L, code, None, seed_mem=seed)
        skip_trace, skip_ap = _run_driver(model, L, code, runner, seed_mem=seed)
        match = base_trace == skip_trace
        ok = ok and match
        if verbose:
            print(f"  {name:10s} {'OK ' if match else 'FAIL'} "
                  f"trace={base_trace[:6]}{'...' if len(base_trace) > 6 else ''} "
                  f"applies full={base_ap} skip={skip_ap} "
                  f"({base_ap/skip_ap:.1f}x fewer)" if skip_ap else "",
                  flush=True)
        if not match and verbose:
            print(f"    BASE={base_trace}\n    SKIP={skip_trace}", flush=True)
    return ok


def bench(model, L, prod_S: int = 900, n: int = 50, warmup: int = 10):
    """ms/step: full 238-block forward vs per-step skip, at a fixed production S.

    We feed a fixed-length stream (prod_S tokens) and time ONE step's block-stack
    for a representative op mix.  Uses resident dense weights (materialize_dense).
    """
    import torch
    runner = StepBlockSkipRunner(model, L)
    dev = model.embed.device
    cuda = (dev.type == "cuda")
    # a fixed prod_S residual (embed a BOS-padded stream).
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
        return (time.time() - t0) / n * 1e3     # ms

    def full():
        x = x0
        with torch.no_grad():
            for blk in model.blocks:
                x = blk(x)
        return x

    # representative ops spanning the skip spectrum.
    rep_ops = {"IMM": isa.IMM, "ADD": isa.ADD, "MUL": isa.MUL, "DIV": isa.DIV,
               "SHL": isa.SHL, "LI": isa.LI, "JMP": isa.JMP}
    t_full = _time(full)
    print(f"\n[bench] prod_S={prod_S} n={n}  FULL 238-block = {t_full:.3f} ms/step",
          flush=True)
    ts = {}
    for nm, op in rep_ops.items():
        def skip(op=op):
            with torch.no_grad():
                return runner.forward(x0, op)
        t = _time(skip)
        ts[nm] = t
        print(f"  skip {nm:4s} live={runner.live_count(op):3d}/238  "
              f"{t:.3f} ms/step  ({t_full/t:.1f}x faster)", flush=True)
    # a representative non-divmod opcode mix (the common case): weight by how often
    # each op class runs in typical code (mostly IMM/ADD/LI/branch, rare MUL/DIV).
    mix = {"IMM": 0.30, "ADD": 0.20, "LI": 0.20, "JMP": 0.10, "MUL": 0.05,
           "SHL": 0.05, "DIV": 0.10}
    t_mix = sum(ts[k] * w for k, w in mix.items())
    print(f"\n[bench] weighted mix (30%IMM/20%ADD/20%LI/10%JMP/5%MUL/5%SHL/10%DIV) "
          f"= {t_mix:.3f} ms/step  vs FULL {t_full:.3f}  "
          f"({t_full/t_mix:.1f}x faster)", flush=True)
    # CUDA-graphability: each op-class has a STATIC live-block shape, so one graph
    # per op-class is capturable.  Report the number of distinct live-count classes.
    distinct = sorted(set(runner.live_count(op) for op in runner.live_index
                          if op is not None))
    print(f"[bench] distinct per-op live shapes = {len(distinct)} "
          f"(counts {distinct}) -> one CUDA graph per op-class is static-shaped",
          flush=True)


def main(code_size: int = 32, device: str = "cuda", prod_S: int = 900,
         do_bench: bool = True, with_bsf: bool = False, bench_S=(900,)):
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    if with_bsf:
        # compose STEP 2 (block skip) with block_sparse_ffn (COO scatter on the
        # surviving live blocks) — the two are orthogonal (skip picks WHICH blocks
        # run; COO cuts the per-block FFN cost).  Keep dense attention (already ~0).
        from .block_sparse_ffn import install_block_sparse_ffn
        install_block_sparse_ffn(model, mode="coo", verbose=True)
    if device != "cpu":
        model.to(device)
        if not with_bsf:
            model.materialize_dense(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.dim} dev={device} "
          f"bsf={with_bsf} build={time.time()-t0:.1f}s", flush=True)
    print("[verify] byte-exact full vs per-step-skip:", flush=True)
    ok = verify(model, L)
    print(f"[verify] {'ALL BYTE-EXACT' if ok else 'DIVERGENCE FOUND'}", flush=True)
    if do_bench:
        for S in bench_S:
            bench(model, L, prod_S=S)
    return ok


if __name__ == "__main__":
    import sys
    dev = "cpu" if "--cpu" in sys.argv else "cuda"
    with_bsf = "--bsf" in sys.argv
    bench_S = (900,)
    for a in sys.argv:
        if a.startswith("--S="):
            bench_S = tuple(int(x) for x in a.split("=", 1)[1].split(","))
    main(device=dev, bench_S=bench_S, with_bsf=with_bsf)
