"""Measure the weight-dedup (weight-tying) win + the recurrent-block potential.

Builds the c4_min compact/streaming sparse model (LEAN / bitwise / divmod),
reports the UNIQUE-tensor / nonzero-weight / storage(MB) numbers BEFORE and AFTER
tying byte-identical weight tensors, verifies the tie is byte-identical
(forward L-inf = 0), and separately MEASURES (does not implement) how many of the
divmod loop's blocks are byte-identical — the size of the win a recurrent-block
refactor (share one block, loop it) would add on top of the tensor tie.

Run:  python -m c4_min.measure_weight_dedup [--divmod] [--lean] [--bitwise]
Memory-safe: streaming build (peak ~1 dense block); divmod peaks ~5 GB RSS.
"""
from __future__ import annotations

import argparse
import resource
import time
from collections import defaultdict

import torch

from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.weight_dedup import (
    dedup_sparse_transformer, count_unique_stored_tensors, _fingerprint,
    _dense_of,
)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _sparse_model_storage_bytes(sparse) -> int:
    """Total DISTINCT stored bytes of the sparse model, counting shared storage ONCE.

    ``sparse.stats()`` prices every block's weight independently, so it cannot see
    the ``id()``-level tie (it double-counts shared tensors).  This walks the model
    and charges each DISTINCT storage object (by ``id()``) exactly once — the true
    on-device / on-disk footprint after tying.  Non-weight tensors (embed, lm_head,
    lm_bias, biases, alibi) are charged at their dense size.
    """
    seen: set = set()
    total = 0
    # embed + lm head + biases (dense, per-object)
    for t in (sparse.embed, sparse.lm_head, sparse.lm_bias):
        if id(t) not in seen:
            seen.add(id(t)); total += t.numel() * 4
    for b in sparse.blocks:
        for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o,
                  b.ffn.W_up, b.ffn.W_gate, b.ffn.W_down):
            store = w.csr if w.is_sparse else w.dense
            if id(store) in seen:
                continue
            seen.add(id(store))
            total += int(w.storage_bytes())
        for v in (b.ffn.b_up, b.ffn.b_gate, b.ffn.b_down, b.attn.alibi_slopes):
            if id(v) not in seen:
                seen.add(id(v)); total += v.numel() * 4
    return total


def _block_signature(sparse, bi) -> tuple:
    """A byte-exact signature of block ``bi``'s full weight set (attn + ffn).

    Two blocks with the same signature are byte-identical VMs — a recurrent-block
    refactor could store ONE and loop it.  The signature ignores block INDEX, so
    it captures the loop-body replication the divmod stack has.
    """
    b = sparse.blocks[bi]
    parts = []
    for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o,
              b.ffn.W_up, b.ffn.W_gate, b.ffn.W_down):
        parts.append(_fingerprint(_dense_of(w)))
    for v in (b.ffn.b_up, b.ffn.b_gate, b.ffn.b_down, b.attn.alibi_slopes):
        parts.append(_fingerprint(v))
    parts.append(("scale", float(b.attn.scale)))
    return tuple(parts)


def measure_recurrent_block_potential(sparse, L):
    """Report how many WHOLE BLOCKS are byte-identical (recurrent-block potential).

    This is the STRUCTURAL win ABOVE the tensor tie: if K blocks are byte-identical
    the model could store 1 and loop K times.  Returns
    ``(n_blocks, n_unique_blocks, groups)`` where ``groups`` lists the big
    identical-block sets.
    """
    names = getattr(L, "_block_names", None)
    groups = defaultdict(list)
    for bi in range(len(sparse.blocks)):
        groups[_block_signature(sparse, bi)].append(bi)
    detail = []
    for sig, members in groups.items():
        if len(members) > 1:
            ex = sorted({names[m] if names and m < len(names) else f"blk{m}"
                         for m in members})
            detail.append((len(members), ex))
    detail.sort(key=lambda g: -g[0])
    return len(sparse.blocks), len(groups), detail


def run_config():
    print(f"\n{'=' * 78}\nCONFIG: full (single op set: incl. bitwise + divmod)"
          f"\n{'=' * 78}")
    t0 = time.time()
    sparse, L, cstats = build_compact_sparse_streaming(
        code_size=48, compute_mode="dense_kernel")
    print(f"built in {time.time() - t0:.1f}s  peak RSS {_rss_gb():.2f} GB")
    print(f"  n_blocks={len(sparse.blocks)}  dim={sparse.dim}  "
          f"vocab={sparse.vocab}")

    st = sparse.stats()
    model_bytes_before = _sparse_model_storage_bytes(sparse)
    print(f"  compact/sparse model: total_nnz={st.total_nnz}  "
          f"stored={model_bytes_before / 1e6:.2f} MB  "
          f"(dense-equiv {st.dense_gb:.2f} GB)")

    # ---- the recurrent-block potential (measure only) ----
    nb, nub, blk_detail = measure_recurrent_block_potential(sparse, L)
    ident_blocks_saved = nb - nub
    print(f"\n  [recurrent-block POTENTIAL — measured, NOT implemented]")
    print(f"    whole blocks: {nb} total -> {nub} unique "
          f"({ident_blocks_saved} byte-identical duplicate blocks a "
          f"recurrent-block refactor could drop, "
          f"{100 * ident_blocks_saved / max(1, nb):.1f}%)")
    for cnt, ex in blk_detail[:8]:
        print(f"      x{cnt:4d} identical blocks  e.g. {ex[:3]}")

    # ---- the tensor tie (implement + verify) ----
    refs_b, uniq_b, _ = count_unique_stored_tensors(sparse)
    stats = dedup_sparse_transformer(sparse, L)
    model_bytes_after = _sparse_model_storage_bytes(sparse)
    refs_a, uniq_a, uniq_nnz = count_unique_stored_tensors(sparse)

    print(f"\n{stats.summary()}")
    print(f"\n  id()-truth: {refs_a} weight refs -> {uniq_a} distinct storages "
          f"(unique nnz {uniq_nnz})")
    print(f"  FULL MODEL stored bytes: {model_bytes_before / 1e6:.2f} MB "
          f"-> {model_bytes_after / 1e6:.2f} MB "
          f"(saved {(model_bytes_before - model_bytes_after) / 1e6:.2f} MB, "
          f"{100 * (model_bytes_before - model_bytes_after) / max(1, model_bytes_before):.1f}%)")
    print(f"  [note: FULL-model bytes include embed/lm_head/lm_bias "
          f"({(st.sparse_bytes - stats.weight_storage_bytes_before) / 1e6:.2f} MB, "
          f"un-tied)]")

    # ---- verify byte-identity vs a fresh un-tied twin ----
    _verify(sparse, L)
    return stats


def _verify(tied, L):
    from c4_min.nibble_pure_forward_complete import _build_frame, SP_INIT
    from c4_min import blogspec_vocab as V
    ref, _, _ = build_compact_sparse_streaming(
        code_size=48, compute_mode="dense_kernel")
    streams = []
    for nf in (0, 2, 4):
        s = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        for k in range(nf):
            s += _build_frame(k + 1, (7 * (k + 1)) & 0xFF,
                              SP_INIT - 4 * (k + 1), SP_INIT, 3 * (k + 1))
        streams.append(s)
    worst = 0.0
    argmax_ok = True
    with torch.no_grad():
        for stream in streams:
            toks = torch.tensor([stream], dtype=torch.long)
            xt = tied.embed[toks].clone()
            xr = ref.embed[toks].clone()
            for bt, br in zip(tied.blocks, ref.blocks):
                xt = bt(xt); xr = br(xr)
            worst = max(worst, float((xt - xr).abs().max()))
            lt = torch.nn.functional.linear(xt, tied.lm_head, tied.lm_bias)
            lr = torch.nn.functional.linear(xr, ref.lm_head, ref.lm_bias)
            if not torch.equal(lt.argmax(-1), lr.argmax(-1)):
                argmax_ok = False
    print(f"  VERIFY vs un-tied twin: forward L-inf = {worst}  "
          f"argmax-identical = {argmax_ok}  "
          f"{'PASS' if worst == 0.0 and argmax_ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    # ONE full-op-set model — the former lean/bitwise/divmod configs are unified.
    run_config()


if __name__ == "__main__":
    main()
