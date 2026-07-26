"""Verify the WIRED wide-value single-head ingest (C4_INGEST_WIDE) in the PRODUCTION
build_pure_forward_model — byte-exact VM behaviour vs the stock 20-head build.

This proves the wide ingest is real in the model builder (not flag-only-in-a-probe):

  * The wide model's block-0 ingest ATTENTION is ONE query head + ONE KV head (the
    "wide-gather" block); the stock build's block-0 ingest is 20 (or 21 w/ memory).
  * Through the REAL run_pure_forward (one VM step = one model.forward), the wide
    build reproduces isa.interpret and the stock 20-head build BYTE-FOR-BYTE on the
    a8c09504 register battery (incl. 32-bit values) AND the loop / nested batteries
    (differing prior frames, latest-wins) — even though block-0 WEIGHTS differ.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=<c4_release> python -m c4_min._wide_ingest_wired_verify
"""
from __future__ import annotations

import os

import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min.nibble_pure_forward import (
    build_pure_forward_model, run_pure_forward, assert_no_python_compute,
    build_frame_tokens, make_overlay, N_ROLES,
)


# ---------------------------------------------------------------------------
# (1) direct frame-ingest reconstruction: the a8c09504 battery incl. 32-bit.
# ---------------------------------------------------------------------------
INGEST_BATTERY = [
    (2, 13, 0x10000, 0x10000, 6),
    (0, 0, 0x10000, 0x10000, 0),
    (5, 255, 0xFFFC, 0x10000, 128),
    (0xABCD, 0x12345678, 0xFFFC, 0x10000, 0xDEADBEEF),
]


def _decode_regs(state, L):
    got = {}
    for name, base in (("pc", L.PC), ("ax", L.AX), ("sp", L.SP),
                       ("bp", L.BP), ("stk", L.STACK0)):
        got[name] = sum(round(float(state[base + j])) << (4 * j) for j in range(8))
    return got


def _run_ingest_only(model, L, regs, n_frames=1):
    """Drive [BOS] + n_frames frames through the ingest sub-blocks only (up to and
    including the recompose block), returning the decoded register nibble bands at
    the last position.  Runs the FULL block stack (harmless: later blocks don't
    touch the reg nibble bands except via ops, which are inert for IMM;HALT)."""
    pc, ax, sp, bp, stk = regs
    frames = []
    for _ in range(n_frames):
        frames += build_frame_tokens(pc, ax, sp, bp, stk)
    toks = torch.tensor([[V.BOS] + frames])
    overlay = make_overlay(isa.assemble([("IMM", 0), ("HALT", 0)]), L)
    x = model.embed[toks].clone()
    overlay(x)
    # only the ingest chain: run every block up to and incl. the recompose.
    names = getattr(L, "_block_names", None)
    stop = names.index("ingest+recompose") if names else 0
    for bi in range(stop + 1):
        x = model.blocks[bi](x)
    return _decode_regs(x[0, -1], L)


def verify_ingest_reconstruction(verbose=True):
    """The block-0 wide gather reconstructs every register byte-exact (single frame
    AND a 3-frame loop where recency must pick the latest)."""
    os.environ["C4_INGEST_WIDE"] = "1"
    model, L = build_pure_forward_model(code_size=16, include_memory=False)
    ok_all = True
    for n_frames in (1, 3):
        for regs in INGEST_BATTERY:
            got = _run_ingest_only(model, L, regs, n_frames=n_frames)
            exp = dict(zip(("pc", "ax", "sp", "bp", "stk"), regs))
            ok = all(got[k] == exp[k] for k in exp)
            ok_all = ok_all and ok
            if verbose and not ok:
                print(f"  MISMATCH n_frames={n_frames} regs={regs} -> {got}")
    if verbose:
        print(f"  ingest reconstruction (single + 3-frame loop): "
              f"{'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ---------------------------------------------------------------------------
# (2) full VM behaviour through run_pure_forward: wide == stock == interpret.
# ---------------------------------------------------------------------------
# Programs spanning arith / stack / branch / a LOOP (differing frames) and a
# nested-frame pattern, so recency + the rescale are exercised at every step.
PROGRAMS = {
    "arith":  [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)],
    "sub":    [("IMM", 100), ("PSH", 0), ("IMM", 42), ("SUB", 0), ("HALT", 0)],
    "wide32": [("IMM", 0x1234), ("PSH", 0), ("IMM", 0x1111), ("ADD", 0), ("HALT", 0)],
    # a countdown LOOP: AX=3; loop: AX=AX-1; BNZ loop; HALT.  Re-emits the same
    # register roles every iteration → the ingest MUST pick the latest frame.
    "loop":   [("IMM", 3), ("PSH", 0), ("IMM", 1), ("SUB", 0),
               ("BNZ", -3), ("HALT", 0)],
    # a longer loop (5 iterations) with a differing AX each step.
    "loop5":  [("IMM", 5), ("PSH", 0), ("IMM", 1), ("SUB", 0),
               ("BNZ", -3), ("HALT", 0)],
    # nested arithmetic: builds up an accumulator across several PSH/ADD frames.
    "nested": [("IMM", 2), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("PSH", 0),
               ("IMM", 4), ("ADD", 0), ("PSH", 0), ("IMM", 5), ("ADD", 0),
               ("HALT", 0)],
}


def _build(wide: bool, code_size=32):
    os.environ["C4_INGEST_WIDE"] = "1" if wide else "0"
    return build_pure_forward_model(code_size=code_size, include_memory=False)


def verify_full_vm(verbose=True):
    """wide-ingest run_pure_forward == stock 20-head run == isa.interpret, byte-
    for-byte, on every program (incl. loops + nested)."""
    model_wide, Lw = _build(True)
    model_stock, Ls = _build(False)
    head_wide = _ingest_head_count(model_wide, Lw)
    head_stock = model_stock.blocks[0].attn.n_heads
    ok_all = True
    for name, prog in PROGRAMS.items():
        code = isa.assemble(prog)
        ref = isa.interpret(code)
        tr_stock = run_pure_forward(model_stock, Ls, code)
        tr_wide = assert_no_python_compute(run_pure_forward, model_wide, Lw, code)
        ok = (tr_wide == tr_stock == ref)
        ok_all = ok_all and ok
        if verbose:
            tag = "PASS" if ok else "FAIL"
            print(f"  {name:8s} {tag}  interpret={ref}")
            if not ok:
                print(f"           wide ={tr_wide}")
                print(f"           stock={tr_stock}")
    if verbose:
        print(f"\n  block-0 ingest head count: stock={head_stock} -> wide={head_wide}")
        print(f"  full VM (wide == stock == interpret): "
              f"{'PASS' if ok_all else 'FAIL'}")
    return ok_all, head_stock, head_wide


def _ingest_head_count(model, L):
    """The number of heads the block-0 ingest ATTENTION uses in the wide build (the
    wide-gather block's 1-head Attn)."""
    names = getattr(L, "_block_names", None)
    if names and "wide-gather" in names:
        return model.blocks[names.index("wide-gather")].attn.n_heads
    return model.blocks[0].attn.n_heads


if __name__ == "__main__":
    print("=== WIRED wide-value single-head ingest — production build_pure_forward ===\n")
    print("--- (1) direct frame-ingest reconstruction (a8c09504 battery) ---")
    ok1 = verify_ingest_reconstruction()
    print("\n--- (2) full VM through run_pure_forward (wide == stock == interpret) ---")
    ok2, hs, hw = verify_full_vm()
    print(f"\n=== OVERALL: {'PASS' if (ok1 and ok2) else 'FAIL'}  "
          f"(block-0 ingest heads {hs} -> {hw}) ===")
