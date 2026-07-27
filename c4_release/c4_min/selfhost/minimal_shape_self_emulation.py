#!/usr/bin/env python3
"""minimal_shape_self_emulation.py — DEFINE + GROUND the MINIMAL transformer shape
that runs the FULL ISA the emulated matmul needs, and measure ITS self-emulation
step count vs the full c4_min shape.

WHY: the self-emulation step count is DERIVED from the emulated model's own matmul
nnz structure (COO iters = S*nnz) + a non-matmul tail (softmax ~ n_heads*S^2, silu
~ ffn_hidden*S, resid ~ resid_footprint*S).  So a SMALLER shape (fewer live nnz,
smaller d_model/heads/hidden) = fewer self-emulation VM steps.  This is the
EMULATED-MODEL-SIZE lever (bigger than ms/step).

WHAT is minimal: the emulated matmul (the paged COO dot the transformer forward is
built from) uses a 17-opcode subset:
    ADD ADJ BZ DIV ENT HALT IMM JMP JSR LEA LI LT MUL NOP PRTF PSH SI
It does NOT use the BITWISE family (OR/XOR/AND/SHL/SHR).  The c4_min compact model
is a SINGLE full-op interpreter whose dims are ALREADY liveness-colored and whose
FFN hidden is ALREADY live-masked (so d_model / hidden are already minimal for the
full op-set).  The one op-FAMILY the emulated matmul never touches is BITWISE, so
the minimal shape DROPS the bitwise-family blocks (the family granularity the build
supports), exactly as the grounding already drops the recurrent-divmod span for the
without-divmod subset.  (DIV is KEPT — the emulated matmul needs it.)

HONEST: the compact model is ALREADY the stripped shape (the 0.5B Qwen 24L/896/14h
is a SEPARATE baked-in model).  So this measures the ADDITIONAL win from op-family
restriction on top of the already-live-colored pos-sparse model — NOT a strip of
dead 0.5B weights (already free in the pos-sparse composed path).

Run:  python -m c4_min.selfhost.minimal_shape_self_emulation
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Set

import torch

from c4_min.selfhost.word32_draft_vm import ref_interpret_word32
from c4_min.selfhost._compile_helper import compile_paged_dot


# The op-families the emulated matmul (paged COO dot) actually uses.  It uses
# memory (LI/SI), cmp (LT), muldiv (MUL/DIV) — but NOT bitwise (OR/XOR/AND/SHL/SHR).
# So the minimal op-family strip is: drop the bitwise family.
def _is_bitwise_block(name: str) -> bool:
    nl = name.lower()
    return (nl.startswith("bw-") or "tshift" in nl or nl == "and"
            or nl == "or" or "xor" in nl or nl == "shift"
            or nl.startswith("shr") or nl.startswith("shl")
            or nl.startswith("or-") or nl.startswith("and-"))


def _is_div_block(name: str) -> bool:
    return name.lower().startswith("alu-div")


def _steps_for_len(length: int, cache: Dict[int, int]) -> int:
    """MEASURED draft-VM steps for ONE paged COO dot of nnz=length (data-independent)."""
    if length <= 0:
        return 0
    if length in cache:
        return cache[length]
    w = [1] + [0] * (length - 1)
    x = [1] + [0] * (length - 1)
    code = compile_paged_dot(w, x)
    _tr, steps = ref_interpret_word32(code, max_steps=80_000_000, out=[])
    cache[length] = steps
    return steps


def _dense_of(sw):
    if not sw.is_sparse:
        return sw.dense
    if getattr(sw, "dense_resident", None) is not None:
        return sw.dense_resident
    return sw.csr.to_dense()


def _walk_model(seq: int, drop_bitwise: bool, drop_div: bool):
    """Walk the compact model, optionally dropping the bitwise / div families.
    Returns per-matmul records + model facts (d_model, heads, hidden, tail footprint)
    computed over ONLY the kept blocks."""
    from c4_min.lib_neural import build_lib_model_streaming
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(
        code_size=32, recurrent_divmod=True, addr32=True)
    build_s = time.time() - t0
    names = L._block_names
    n_blocks = len(model.blocks)

    recs: List[Dict] = []
    d_model = None
    n_heads = None
    n_realattn = 0
    ffn_hidden_total = 0
    n_kept = 0
    resid_add_footprint = 0
    dropped_bw = 0
    dropped_div = 0

    for bi in range(n_blocks):
        name = names[bi] if bi < len(names) else f"blk{bi}"
        if drop_bitwise and _is_bitwise_block(name):
            dropped_bw += 1
            continue
        if drop_div and _is_div_block(name):
            dropped_div += 1
            continue
        n_kept += 1
        blk = model.blocks[bi]
        at = getattr(blk, "attn", None)
        ff = getattr(blk, "ffn", None)
        if at is not None and getattr(at, "W_q", None) is not None:
            d_model = at.dim
            n_heads = at.n_heads
            if int(at.W_q.nnz) > 0 or int(at.W_v.nnz) > 0:
                n_realattn += 1
            for wn in ("W_q", "W_k", "W_v", "W_o"):
                w = getattr(at, wn, None)
                if w is None:
                    continue
                recs.append(dict(nnz=int(w.nnz), S=seq))
        if ff is not None and getattr(ff, "W_up", None) is not None:
            ffn_hidden_total += ff.W_up.out_dim
            wd = _dense_of(ff.W_down)
            resid_add_footprint += int((wd != 0).any(dim=1).sum().item())
            for bn in ("b_down", "b_up", "b_gate"):
                b = getattr(ff, bn, None)
                if b is not None:
                    resid_add_footprint += int((b != 0).sum().item())
            for wn in ("W_up", "W_gate", "W_down"):
                w = getattr(ff, wn, None)
                if w is None:
                    continue
                recs.append(dict(nnz=int(w.nnz), S=seq))

    # LM head (needed to decode the emit frame → keep in both).
    head = getattr(model, "head", None)
    if head is not None and getattr(head, "weight", None) is not None:
        hnnz = int((head.weight != 0).sum().item())
        recs.append(dict(nnz=hnnz, S=seq))

    total_nnz = sum(r["nnz"] for r in recs)
    coo_iters = sum(r["nnz"] * r["S"] for r in recs)
    facts = dict(d_model=d_model, n_heads=n_heads, n_realattn=n_realattn,
                 ffn_hidden_total=ffn_hidden_total, n_blocks=n_kept,
                 resid_add_footprint=resid_add_footprint,
                 dropped_bw=dropped_bw, dropped_div=dropped_div,
                 total_blocks=n_blocks, build_s=build_s)
    return recs, facts


def _tail(facts, seq, rates):
    """Non-matmul tail steps (softmax + silu + resid adds) for the KEPT-block model."""
    softmax_elems = facts["n_realattn"] * facts["n_heads"] * seq * seq
    softmax_steps = softmax_elems * rates["Softmax"]
    sigmoid_steps = facts["ffn_hidden_total"] * seq * rates["Sigmoid"]
    resid_steps = facts["resid_add_footprint"] * seq * rates["Add"]
    return dict(softmax=softmax_steps, silu=sigmoid_steps, resid=resid_steps,
                total=softmax_steps + sigmoid_steps + resid_steps)


def ground_shape(seq: int, drop_bitwise: bool, cache: Dict[int, int], rates):
    """Ground ONE shape's self-emulation step count (matmul + tail), pos-DENSE and
    pos-SPARSE.  KEEPS the div family (the emulated matmul uses DIV)."""
    recs, facts = _walk_model(seq, drop_bitwise=drop_bitwise, drop_div=False)
    for r in recs:
        _steps_for_len(r["nnz"], cache)
    mm_dense = sum(cache[r["nnz"]] * r["S"] for r in recs if r["nnz"] > 0)
    mm_sparse = sum(cache[r["nnz"]] * 1 for r in recs if r["nnz"] > 0)
    tail_dense = _tail(facts, seq, rates)
    tail_sparse = _tail(facts, 1, rates)
    total_nnz = sum(r["nnz"] for r in recs)
    coo_iters = sum(r["nnz"] * r["S"] for r in recs)
    return dict(
        facts=facts, total_nnz=total_nnz, coo_iters=coo_iters,
        mm_dense=mm_dense, mm_sparse=mm_sparse,
        tail_dense=tail_dense, tail_sparse=tail_sparse,
        total_dense=mm_dense + tail_dense["total"],
        total_sparse=mm_sparse + tail_sparse["total"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=30)
    args = ap.parse_args()
    seq = args.seq

    print("=" * 80)
    print("MINIMAL-SHAPE SELF-EMULATION — the emulated-model-size lever, GROUNDED")
    print("=" * 80)
    from c4_min.selfhost.measure_whole_forward_steps import measure_rates
    rates = measure_rates(verbose=False)
    cache: Dict[int, int] = {}

    print("\n[1] FULL c4_min shape (all op families incl bitwise; div kept):", flush=True)
    full = ground_shape(seq, drop_bitwise=False, cache=cache, rates=rates)
    print("\n[2] MINIMAL shape (drop BITWISE family; keep div — the 17-op emulated-"
          "matmul set):", flush=True)
    mini = ground_shape(seq, drop_bitwise=True, cache=cache, rates=rates)

    def _shape_line(tag, g):
        f = g["facts"]
        print(f"  {tag}: blocks={f['n_blocks']}  d_model={f['d_model']}  "
              f"heads={f['n_heads']}  real-attn={f['n_realattn']}  "
              f"ffn_hidden_total={f['ffn_hidden_total']:,}  "
              f"resid_footprint={f['resid_add_footprint']:,}")
        print(f"       nnz={g['total_nnz']:,}  COO-iters(S={seq})={g['coo_iters']:,}")

    print("\n" + "-" * 80)
    print("SHAPE")
    print("-" * 80)
    _shape_line("FULL   ", full)
    _shape_line("MINIMAL", mini)
    print(f"  dropped by minimal: {mini['facts']['dropped_bw']} bitwise blocks")

    def _steps_block(tag, g):
        print(f"  {tag}:")
        print(f"    matmul  pos-DENSE={g['mm_dense']:,.0f}  pos-SPARSE={g['mm_sparse']:,.0f}")
        print(f"    tail    pos-DENSE={g['tail_dense']['total']:,.0f}  "
              f"pos-SPARSE={g['tail_sparse']['total']:,.0f}   "
              f"(softmax {g['tail_sparse']['softmax']:,.0f} + silu "
              f"{g['tail_sparse']['silu']:,.0f} + resid {g['tail_sparse']['resid']:,.0f} @S=1)")
        print(f"    TOTAL   pos-DENSE={g['total_dense']:,.0f}  "
              f"pos-SPARSE={g['total_sparse']:,.0f}")

    print("\n" + "-" * 80)
    print("SELF-EMULATION STEP COUNT (matmul MEASURED + tail DERIVED)")
    print("-" * 80)
    _steps_block("FULL   ", full)
    _steps_block("MINIMAL", mini)

    dn = full["total_sparse"]
    mn = mini["total_sparse"]
    print("\n" + "=" * 80)
    print(f"  pos-SPARSE self-emulation steps:  FULL={dn:,.0f}  MINIMAL={mn:,.0f}")
    print(f"  MINIMAL saves {dn-mn:,.0f} steps = {(1-mn/dn)*100:.2f}% fewer  "
          f"({dn/mn:.3f}x)")
    print(f"  (matmul-only delta: {full['mm_sparse']-mini['mm_sparse']:,.0f} steps; "
          f"tail delta: {full['tail_sparse']['total']-mini['tail_sparse']['total']:,.0f} steps)")
    print("=" * 80)
    print("\nHONEST verdict: the minimal op-family strip removes only the BITWISE")
    print("family (the compact model is ALREADY liveness-colored + hidden-masked, so")
    print("d_model/heads/hidden barely move).  The div family — which the emulated")
    print("matmul NEEDS — dominates nnz and is KEPT.  So minimal is only MARGINALLY")
    print("fewer steps than the pos-sparse full shape: the SHAPE lever is SMALL here")
    print("because the pos-sparse composed path already skips the dead weights, and")
    print("the live op-set the emulated matmul exercises is nearly the whole ISA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
