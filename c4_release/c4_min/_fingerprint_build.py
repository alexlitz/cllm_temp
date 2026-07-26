"""Deterministic build fingerprint for the c4_min pure-forward VM.

Builds the SINGLE full-op interpreter memory-safely (streaming sparse build,
peak = one dense block) and hashes the DENSE-reconstructed per-block weights
in fp64 in a deterministic order.  Used by the consolidation byte-identity gate:
the fingerprint must stay identical across merges except where a change is
byte-identical-by-construction and known.

Run:  PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._fingerprint_build
"""
from __future__ import annotations

import hashlib

import torch


def _sw_dense(sw):
    """Reconstruct a SparseWeight's dense matrix (storage-format independent)."""
    if getattr(sw, "dense", None) is not None:
        return sw.dense
    if getattr(sw, "dense_resident", None) is not None:
        return sw.dense_resident
    if getattr(sw, "csr", None) is not None:
        return sw.csr.to_dense()
    raise RuntimeError("SparseWeight has no dense/csr backing")


def _hash_tensor(h, name, t):
    if t is None:
        h.update(f"{name}:None\n".encode())
        return
    t = t.detach().to(torch.float64).contiguous().cpu()
    h.update(f"{name}:{tuple(t.shape)}\n".encode())
    h.update(t.numpy().tobytes())


def _hash_sw(h, name, sw):
    _hash_tensor(h, name, _sw_dense(sw))


def _hash_model(model) -> str:
    h = hashlib.sha256()
    for nm in ("embed", "lm_head", "lm_bias"):
        _hash_tensor(h, nm, getattr(model, nm, None))
    h.update(f"n_blocks:{len(model.blocks)}\n".encode())
    for bi, blk in enumerate(model.blocks):
        h.update(f"block:{bi}\n".encode())
        attn = getattr(blk, "attn", None)
        if attn is not None:
            _hash_tensor(h, f"b{bi}.attn.n_heads", torch.tensor([attn.n_heads]))
            _hash_tensor(h, f"b{bi}.alibi", getattr(attn, "alibi_slopes", None))
            for wn in ("W_q", "W_k", "W_v", "W_o"):
                w = getattr(attn, wn)
                if hasattr(w, "dense") or hasattr(w, "csr") or \
                   hasattr(w, "dense_resident"):
                    _hash_sw(h, f"b{bi}.attn.{wn}", w)
                else:
                    _hash_tensor(h, f"b{bi}.attn.{wn}", w)
        ffn = getattr(blk, "ffn", None)
        if ffn is not None:
            for wn in ("W_up", "W_gate", "W_down"):
                w = getattr(ffn, wn)
                if hasattr(w, "dense") or hasattr(w, "csr") or \
                   hasattr(w, "dense_resident"):
                    _hash_sw(h, f"b{bi}.ffn.{wn}", w)
                else:
                    _hash_tensor(h, f"b{bi}.ffn.{wn}", w)
            for bn in ("b_up", "b_gate", "b_down"):
                _hash_tensor(h, f"b{bi}.ffn.{bn}", getattr(ffn, bn, None))
    return h.hexdigest()


def fingerprint(code_size: int = 32, recurrent_divmod: bool = False) -> str:
    # C4_INGEST_WIDE (default OFF): the 1-query/1-KV wide ingest restructures block 0
    # to a 1-head Attn.  The STREAMING builder now carries a PER-BLOCK n_heads (the
    # wide-gather block is 1-head; every other block keeps the build-wide n_heads), so
    # the same streaming path is fingerprinted in BOTH states — flag-ON is the intended
    # MOVED hash (packed-dim, differs from the dense build_pure_forward_complete_model
    # hash by construction), flag-OFF is the golden.
    #
    # CURRENT GOLDEN (flag-OFF): 069cc32fa7cecfbceae448a7dbf6e2140b3db6cf6857c8accec5639b9c55c0ca
    #   (short 069cc32f).  RE-BASELINED 2026-07 from 8f4dd780 by the c4-faithful
    #   SHR-ARITHMETIC (tight-shifter ``build_sign_fill``) + SIGNED-LC
    #   (``lc-sign-detect`` / ``lc-sign-extend``) fix — an INTENTIONAL move: the two
    #   sign-fill gadgets fire only on OP_IS[SHR]∧sign / OP_IS[LC]∧high-bit, so every
    #   non-shift / non-signed-char step (and thus the whole 1096 corpus) is unchanged.
    from c4_min.compact_alloc import build_compact_sparse_streaming

    model, L, stats = build_compact_sparse_streaming(
        code_size=code_size, recurrent_divmod=recurrent_divmod,
        compute_mode="dense_kernel")

    h = hashlib.sha256()
    for nm in ("embed", "lm_head", "lm_bias"):
        _hash_tensor(h, nm, getattr(model, nm, None))
    h.update(f"n_blocks:{len(model.blocks)}\n".encode())
    for bi, blk in enumerate(model.blocks):
        h.update(f"block:{bi}\n".encode())
        attn = getattr(blk, "attn", None)
        if attn is not None:
            _hash_tensor(h, f"b{bi}.alibi", getattr(attn, "alibi_slopes", None))
            for wn in ("W_q", "W_k", "W_v", "W_o"):
                _hash_sw(h, f"b{bi}.attn.{wn}", getattr(attn, wn))
        ffn = getattr(blk, "ffn", None)
        if ffn is not None:
            for wn in ("W_up", "W_gate", "W_down"):
                _hash_sw(h, f"b{bi}.ffn.{wn}", getattr(ffn, wn))
            for bn in ("b_up", "b_gate", "b_down"):
                _hash_tensor(h, f"b{bi}.ffn.{bn}", getattr(ffn, bn, None))
    return h.hexdigest()


if __name__ == "__main__":
    torch.manual_seed(0)
    fp = fingerprint()
    print(f"FINGERPRINT {fp}")
