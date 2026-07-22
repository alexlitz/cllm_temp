"""Measure the cross-op structural dedup win on top of the byte-identical tie.

Pipeline per config:
  1. build_compact_sparse_streaming (compact / sparse / per-block hidden)
  2. dedup_sparse_transformer        (EXISTING byte-identical tie -> baseline)
  3. crossop_dedup                   (NEW permutation / cross-op tie)
  4. verify L-inf=0 vs a fresh UN-tied twin (built once) + argmax identity

Reports the unique-nonzero-scalar count and stored bytes at each stage, the
cross-op group detail (which opcode families share a core), and the ALMOST-
shareable candidates that are NOT tied (no exact permutation).

Run:  python -m c4_min.measure_crossop_dedup [--lean] [--bitwise] [--divmod]
                                             [--no-econ]
Memory-safe: two builds (tied + ref) per config; divmod peaks ~5 GB RSS each,
built serially.
"""
from __future__ import annotations

import argparse
import resource
import time

import torch

from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.weight_dedup import (
    dedup_sparse_transformer, count_unique_stored_tensors,
)
from c4_min.crossop_dedup import crossop_dedup, verify_crossop_identical


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _distinct_scalars_bytes(sparse):
    """(unique nonzero scalars, weight stored bytes) over DISTINCT storages."""
    seen = set()
    scal = 0
    byt = 0
    for b in sparse.blocks:
        for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o,
                  b.ffn.W_up, b.ffn.W_gate, b.ffn.W_down):
            if getattr(w, "_crossop_base", None) is not None:
                continue
            store = w.csr if w.is_sparse else w.dense
            if store is None or id(store) in seen or int(w.nnz) == 0:
                continue
            seen.add(id(store))
            scal += int(w.nnz)
            byt += int(w.storage_bytes())
    return scal, byt


def _streams():
    from c4_min.nibble_pure_forward_complete import _build_frame, SP_INIT
    from c4_min import blogspec_vocab as V
    streams = []
    for nf in (0, 2, 4):
        s = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        for k in range(nf):
            s += _build_frame(k + 1, (7 * (k + 1)) & 0xFF,
                              SP_INIT - 4 * (k + 1), SP_INIT, 3 * (k + 1))
        streams.append(s)
    return streams


def run_config(include_bitwise, include_divmod, econ):
    tag = ("divmod" if include_divmod else
           ("bitwise" if include_bitwise else "lean"))
    print(f"\n{'=' * 78}\nCONFIG: {tag}  (econ={econ})\n{'=' * 78}")

    t0 = time.time()
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=48, include_bitwise=include_bitwise,
        include_divmod=include_divmod, compute_mode="dense_kernel")
    print(f"built in {time.time()-t0:.1f}s  peak RSS {_rss_gb():.2f} GB  "
          f"n_blocks={len(sparse.blocks)} dim={sparse.dim}")

    # stage 1: existing byte-identical tie (baseline for THIS work)
    dedup_sparse_transformer(sparse, L)
    refs_b, uniq_b, _ = count_unique_stored_tensors(sparse)
    scal0, byt0 = _distinct_scalars_bytes(sparse)
    print(f"\n  [baseline: after existing byte-identical tie]")
    print(f"    distinct weight storages : {uniq_b}")
    print(f"    unique nonzero scalars   : {scal0}")
    print(f"    weight stored bytes      : {byt0/1e6:.3f} MB")

    # stage 2: NEW cross-op permutation tie
    st = crossop_dedup(sparse, L, econ=econ)
    print(f"\n{st.summary()}")

    scal1, byt1 = st.scalars_after, st.bytes_after
    print(f"\n  [after cross-op tie]")
    print(f"    unique nonzero scalars   : {scal0} -> {scal1} "
          f"(saved {scal0-scal1}, {100*(scal0-scal1)/max(1,scal0):.1f}%)")
    print(f"    weight stored bytes      : {byt0/1e6:.3f} MB -> {byt1/1e6:.3f} MB "
          f"(saved {(byt0-byt1)/1e6:.3f} MB, "
          f"{100*(byt0-byt1)/max(1,byt0):.1f}%)")

    # stage 3: verify byte-identity vs a FRESH un-tied twin
    ref, _, _ = build_compact_sparse_streaming(
        code_size=48, include_bitwise=include_bitwise,
        include_divmod=include_divmod, compute_mode="dense_kernel")
    worst, argmax_ok = verify_crossop_identical(sparse, ref, _streams())
    ok = (worst == 0.0 and argmax_ok)
    print(f"\n  VERIFY vs un-tied twin: forward L-inf = {worst}  "
          f"argmax-identical = {argmax_ok}  {'PASS' if ok else 'FAIL'}")
    del ref
    return st, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--bitwise", action="store_true")
    ap.add_argument("--divmod", action="store_true")
    ap.add_argument("--no-econ", action="store_true",
                    help="tie whenever it saves scalars (bytes may be neutral)")
    args = ap.parse_args()
    econ = not args.no_econ
    any_sel = args.lean or args.bitwise or args.divmod
    if not any_sel or args.lean:
        run_config(False, False, econ)
    if not any_sel or args.bitwise:
        run_config(True, False, econ)
    if args.divmod:
        run_config(True, True, econ)


if __name__ == "__main__":
    main()
