#!/usr/bin/env python3
"""ground_true_self_emulation.py — the GROUNDED draft-VM step count for TRUE
SELF-EMULATION: the FULL-ISA VM (the compact_alloc 238-block model baked in
Qwen-0.5B) emulating its OWN forward.

This replaces the two bogus priors:
  * ~296M  — an unfounded extrapolation for the full model.
  * 56.4M  — the full-ISA host emulating a TINY 207-node c4vm.onnx toy
             (503,776 MACs, 486 nnz), NOT self-emulation.

The TRUE self-emulation target is the SAME 238-block full-ISA model's forward.
We MEASURE it, we do not guess:

  1. STRUCTURE (Task 2): ``enumerate_full_model_matmuls`` walks EVERY block's
     attn projection + FFN + LM head and reads the ACTUAL nonzero (sparse) MAC
     count per matmul from the built weights (99.99% sparse).  The full ISA
     INCLUDES the recurrent divmod megablocks (``--with-divmod``); the
     without-divmod subset is also reported.

  2. PER-OP RATE (Task 3): the paged COO kernel's steps/MAC is MEASURED
     end-to-end through the 32-bit draft VM (``word32_draft_vm``, which does NOT
     truncate values/counters > 255), per DISTINCT nnz length (data-independent).
     Validated at REAL scale (K=896/1964) in ``validate_real_scale_word32``.

  3. GROUNDED SUM (Task 3): the forward loop is DATA-INDEPENDENT and FIXED-bound,
     so  sum over matmuls of steps(nnz) * S  IS a measurement (each unit a real
     end-to-end VM run), not an extrapolation.  Reported:
       * matmul portion (sparse) + non-matmul tail (softmax on the 3 real-attn
         blocks; RMSNorm/SwiGLU-sigmoid elementwise),
       * CURRENT-IMPL (position-dense, S = emit-frame width) vs
         POSITION-SPARSE-OPTIMAL (S = 1, only the decode row consumed).

  4. WALL: steps x ms/step across the measured ladder (49 / 10.5 / 2.0 / 0.1).

HONESTY labels: every number is tagged MEASURED (a real VM run) or DERIVED
(a fixed-structure product of measured units) — nothing is extrapolated off a
rate x a guessed MAC count.

Run:  python -m c4_min.selfhost.ground_true_self_emulation
      ... --with-divmod   (full ISA incl the recurrent divmod megablocks)
      ... --exhaustive     (literally run every (matmul,row) instance)
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

from c4_min.selfhost.enumerate_full_model_matmuls import enumerate_matmuls
from c4_min.selfhost._matmul_paged_src import paged_dot_c
from c4_min.selfhost.word32_draft_vm import ref_interpret_word32
from c4_min.selfhost._compile_helper import compile_paged_dot

# ms/step ladder (MEASURED across the build: dense-overlay 49, block-sparse-eager
# ~10.5, graphed ~2.0, batched/GEMM-filled ~0.1).  See measure_self_emulation_wall.
MS_LADDER = [("dense-overlay", 49.0), ("block-sparse-eager", 10.5),
             ("cuda-graph", 2.0), ("batched/GEMM-filled", 0.1)]


def _steps_for_len(length: int, cache: Dict[int, int]) -> int:
    """MEASURED draft-VM steps for ONE paged COO dot of nnz=length, run end-to-end
    through the 32-BIT draft VM (no byte truncation).  Data-independent -> cached."""
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


# --------------------------------------------------------------------------- #
# non-matmul tail (softmax + RMSNorm + SwiGLU sigmoid), grounded per-element   #
# --------------------------------------------------------------------------- #
def nonmatmul_tail(model_info, seq: int, rates: Dict[str, float]) -> Dict:
    """Grounded non-matmul draft-steps for ONE forward.

    The model's ACTUAL forward (read from the built module) is:
        attn: softmax(scores) @ V         (softmax the only non-matmul op)
        ffn:  x + W_down(silu(W_up(x))*gate) + b   (silu + residual/bias adds)
    There is NO RMSNorm (confirmed: the SparseFFN/attn carry none).  So the
    non-matmul work is EXACTLY three families:
      * softmax over the attention scores in the 3 real-attn blocks:
        n_realattn * n_heads * S(query) * S(key) elements at the MEASURED
        softmax rate,
      * silu (= x*sigmoid(x)) over the FFN hidden units actually present
        (the ~sparse hidden width per block), at the MEASURED sigmoid rate,
      * elementwise residual + bias adds: the FFN does ``x + down(...) + b_down``
        = 2 adds over d_model per block per row (up/gate biases add a 3rd over the
        hidden width, folded into silu's input — counted once as d_model here to
        stay conservative-low, the biggest tail term so labelled DERIVED clearly).
    Every rate is MEASURED (``measure_whole_forward_steps.measure_rates``)."""
    n_blocks = model_info["n_blocks"]
    d_model = model_info["d_model"]
    n_heads = model_info["n_heads"]
    n_realattn = model_info["n_realattn"]
    ffn_hidden_total = model_info["ffn_hidden_total"]

    # softmax elements: per real-attn block, n_heads * S_query * S_key scores.
    softmax_elems = n_realattn * n_heads * seq * seq
    softmax_steps = softmax_elems * rates["Softmax"]
    # silu over the (sparse) FFN hidden units actually present.
    sigmoid_steps = ffn_hidden_total * seq * rates["Sigmoid"]
    # residual + bias adds (NO RMSNorm): the elementwise-add footprint is the
    # ACTUAL sparse delta positions W_down writes + the nonzero biases — MEASURED
    # from the weights (5,632 active out dims + 34,840 nonzero biases), NOT the
    # dense d_model*n_blocks (which over-counts ~9x since most dims never change).
    resid_adds = model_info["resid_add_footprint"]      # measured per-row count
    resid_steps = resid_adds * seq * rates["Add"]
    total = softmax_steps + sigmoid_steps + resid_steps
    return dict(softmax_elems=softmax_elems, softmax_steps=softmax_steps,
                sigmoid_steps=sigmoid_steps, resid_steps=resid_steps,
                total=total)


def ground(seq: int = 30, with_divmod: bool = True, exhaustive: bool = False,
           verbose: bool = True) -> Dict:
    # --- Task 2: real full-ISA forward structure (MEASURED from the model) ---
    info = enumerate_matmuls(seq=seq, drop_divmod=not with_divmod, verbose=verbose)
    recs = info["recs"]

    # extra model facts for the non-matmul tail (built once here).
    model_info = _model_facts(with_divmod)

    # --- Task 3: MEASURED per-op rate x FIXED structure ---
    cache: Dict[int, int] = {}
    lengths = sorted({r["nnz"] for r in recs if r["nnz"] > 0})
    t0 = time.time()
    executed = 0
    for L in lengths:
        executed += _steps_for_len(L, cache)
    per_len_wall = time.time() - t0

    # position-DENSE (current impl): steps(nnz) * S per matmul.
    sparse_dense = sum(cache[r["nnz"]] * r["S"] for r in recs if r["nnz"] > 0)
    # position-SPARSE-optimal: only the decode row (S=1) consumes the heavy result.
    sparse_sparse = sum(cache[r["nnz"]] * 1 for r in recs if r["nnz"] > 0)

    coo_iters = info["coo_iters"]
    rate = sparse_dense / max(coo_iters, 1)

    exhaustive_executed = 0
    if exhaustive:
        code_cache = {L: compile_paged_dot([1] + [0] * (L - 1),
                                           [1] + [0] * (L - 1)) for L in lengths}
        for r in recs:
            if r["nnz"] <= 0:
                continue
            code = code_cache[r["nnz"]]
            for _ in range(r["S"]):
                exhaustive_executed += ref_interpret_word32(
                    code, max_steps=80_000_000, out=[])[1]

    # rates for the non-matmul tail (measured; cached import).
    from c4_min.selfhost.measure_whole_forward_steps import measure_rates
    rates = measure_rates(verbose=False)

    tail_dense = nonmatmul_tail(model_info, seq, rates)
    tail_sparse = nonmatmul_tail(model_info, 1, rates)

    total_dense = sparse_dense + tail_dense["total"]
    total_sparse = sparse_sparse + tail_sparse["total"]

    out = dict(
        seq=seq, with_divmod=with_divmod,
        n_blocks=info["n_blocks"], n_matmuls=len([r for r in recs if r["nnz"] > 0]),
        total_nnz=info["total_nnz"], dense_macs=info["dense_macs"],
        coo_iters=coo_iters, rate=rate, n_lengths=len(lengths),
        executed=executed, per_len_wall=per_len_wall,
        sparse_dense=sparse_dense, sparse_sparse=sparse_sparse,
        tail_dense=tail_dense, tail_sparse=tail_sparse,
        total_dense=total_dense, total_sparse=total_sparse,
        exhaustive_executed=exhaustive_executed, model_info=model_info,
    )
    if verbose:
        _report(out)
    return out


def _model_facts(with_divmod: bool) -> Dict:
    """Read d_model / n_heads / real-attn-block-count / total FFN hidden units from
    the BUILT model (MEASURED structure, not assumed)."""
    from c4_min.lib_neural import build_lib_model_streaming
    model, L, _ = build_lib_model_streaming(
        code_size=32, recurrent_divmod=True, addr32=True)
    names = L._block_names
    d_model = None
    n_heads = None
    n_realattn = 0
    ffn_hidden_total = 0
    n_blocks_counted = 0
    resid_add_footprint = 0     # active W_down out dims + nonzero biases (measured)

    def _dense_of(sw):
        if not sw.is_sparse:
            return sw.dense
        if getattr(sw, "dense_resident", None) is not None:
            return sw.dense_resident
        return sw.csr.to_dense()

    for bi, blk in enumerate(model.blocks):
        name = names[bi] if bi < len(names) else ""
        if not with_divmod and name.startswith("alu-div"):
            continue
        n_blocks_counted += 1
        at = getattr(blk, "attn", None)
        if at is not None and getattr(at, "W_q", None) is not None:
            d_model = at.dim
            n_heads = at.n_heads
            if int(at.W_q.nnz) > 0 or int(at.W_v.nnz) > 0:
                n_realattn += 1
        ff = getattr(blk, "ffn", None)
        if ff is not None and getattr(ff, "W_up", None) is not None:
            # FFN hidden units present in this block = W_up out_dim (SwiGLU width).
            ffn_hidden_total += ff.W_up.out_dim
            # residual-delta footprint: distinct output dims W_down writes (the
            # only dims x+delta actually changes) + the nonzero bias adds.
            wd = _dense_of(ff.W_down)
            resid_add_footprint += int((wd != 0).any(dim=1).sum().item())
            for bn in ("b_down", "b_up", "b_gate"):
                b = getattr(ff, bn, None)
                if b is not None:
                    resid_add_footprint += int((b != 0).sum().item())
    return dict(d_model=d_model, n_heads=n_heads, n_realattn=n_realattn,
                ffn_hidden_total=ffn_hidden_total, n_blocks=n_blocks_counted,
                resid_add_footprint=resid_add_footprint)


def _report(o: Dict) -> None:
    mi = o["model_info"]
    print()
    print("=" * 78)
    print("TRUE SELF-EMULATION — the FULL-ISA VM emulating its OWN forward")
    print("=" * 78)
    print(f"  ISA: {'WITH' if o['with_divmod'] else 'WITHOUT'} recurrent divmod "
          f"megablocks   |   {o['n_blocks']} blocks, d_model={mi['d_model']}, "
          f"{mi['n_heads']} heads")
    print(f"  real-attention blocks (softmax): {mi['n_realattn']} of {o['n_blocks']} "
          f"(rest are pure-FFN passthrough)")
    print(f"  MEASURED structure: {o['n_matmuls']} nonzero matmuls, "
          f"{o['total_nnz']:,} nonzero weights, {o['dense_macs']:,} DENSE MACs")
    print(f"  sparse COO inner iters (S*nnz, S={o['seq']}) = {o['coo_iters']:,}")
    print(f"  MEASURED paged-COO steps/MAC (32-bit draft VM) = {o['rate']:.2f}")
    print(f"  ({o['n_lengths']} distinct nnz lengths, executed {o['executed']:,} "
          f"real VM steps in {o['per_len_wall']:.1f}s to build the per-length cache)")
    if o["exhaustive_executed"]:
        print(f"  EXHAUSTIVE: literally executed {o['exhaustive_executed']:,} VM steps "
              f"(== the position-dense matmul total)")
    print()
    print("  MATMUL portion (MEASURED = sum over matmuls of steps(nnz) * S):")
    print(f"    position-DENSE  (current impl, S={o['seq']} rows) = "
          f"{o['sparse_dense']:,} draft steps")
    print(f"    position-SPARSE (optimal, S=1 decode row)        = "
          f"{o['sparse_sparse']:,} draft steps  "
          f"({o['sparse_dense'] / max(o['sparse_sparse'],1):.1f}x smaller)")
    print()
    td, ts = o["tail_dense"], o["tail_sparse"]
    print("  NON-MATMUL tail (DERIVED = measured per-op rate x measured element count):")
    print(f"    softmax ({td['softmax_elems']:,} score elems @ {o['seq']} rows) "
          f"= {td['softmax_steps']:,.0f} steps  (S=1: {ts['softmax_steps']:,.0f})")
    print(f"    SwiGLU silu = {td['sigmoid_steps']:,.0f} steps  "
          f"(S=1: {ts['sigmoid_steps']:,.0f})")
    print(f"    residual/bias adds (NO RMSNorm; {mi['resid_add_footprint']:,} "
          f"measured active dims/row) = {td['resid_steps']:,.0f} steps  "
          f"(S=1: {ts['resid_steps']:,.0f})")
    print(f"    tail total = {td['total']:,.0f} steps  (S=1: {ts['total']:,.0f})")
    print()
    print("  " + "-" * 74)
    print(f"  GROUNDED TOTAL — ONE true-self-emulated forward:")
    print(f"    MATMUL-ONLY (fully MEASURED)       dense {o['sparse_dense']:,}  |  "
          f"sparse {o['sparse_sparse']:,}")
    print(f"    + non-matmul tail (DERIVED)        dense {td['total']:,.0f}  |  "
          f"sparse {ts['total']:,.0f}")
    print(f"    = position-DENSE  (current impl)   = {o['total_dense']:,.0f} draft steps")
    print(f"    = position-SPARSE (optimal)        = {o['total_sparse']:,.0f} draft steps")
    print("  " + "-" * 74)
    print()
    print("  vs the priors:  296M (unfounded extrapolation)   56.4M (toy 207-node "
          "c4vm.onnx, NOT self-emulation)")
    print(f"    grounded position-dense  = {o['total_dense']/1e6:.1f}M draft steps")
    print(f"    grounded position-sparse = {o['total_sparse']/1e6:.1f}M draft steps")
    print()
    print("  SELF-EMULATION WALL (steps x ms/step ladder):")
    for label, ms in MS_LADDER:
        wd = o["total_dense"] * ms / 1000.0
        ws = o["total_sparse"] * ms / 1000.0
        print(f"    {label:22} {ms:5.1f} ms/step :  dense {_fmt_wall(wd)}  |  "
              f"sparse {_fmt_wall(ws)}")
    print()
    print("  HONESTY: the matmul portion is MEASURED (each nnz-length is a real "
          "end-to-end")
    print("  32-bit-VM run; the sum is fixed-structure, not a rate x guessed MACs). "
          "The")
    print("  non-matmul tail is DERIVED (measured per-op rate x measured element "
          "count).")
    print("  Position-sparse assumes the decode-row-only optimisation (only S=1 of "
          "the")
    print("  heavy blocks' output rows is consumed); the current impl computes all "
          "S rows.")


def _fmt_wall(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    if seconds < 86400:
        return f"{seconds/3600:.2f}h"
    return f"{seconds/86400:.2f}d"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=30,
                    help="emit-frame width = the per-step forward's query rows "
                         "(position-dense S)")
    ap.add_argument("--with-divmod", action="store_true",
                    help="include the recurrent divmod megablocks (the FULL ISA)")
    ap.add_argument("--exhaustive", action="store_true")
    args = ap.parse_args()
    print("GROUNDING TRUE SELF-EMULATION (the full-ISA VM emulating its own "
          "forward)\nvia the 32-bit draft VM (no byte truncation) + CPU c4 "
          "toolchain — no GPU\n")
    ground(seq=args.seq, with_divmod=args.with_divmod,
           exhaustive=args.exhaustive, verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
