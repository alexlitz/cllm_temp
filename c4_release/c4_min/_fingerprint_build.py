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
    # CURRENT GOLDEN (default): 3cabef6479d183296384b99d25caec9ecd6bc9b2a0f1511d6c89a3689a5ae7b2
    #   (short 3cabef64).  RE-BASELINED 2026-08-06 from 7d4afe61 by the DEFAULT-ON flip of
    #   ``C4_DIVMOD_SIGNED`` (#790 — the last C90-general fix: signed trunc-toward-zero DIV/MOD via
    #   the ``compile_divmod_sign_prep``/``compile_divmod_sign_apply`` sign-magnitude wrapper in
    #   ``nibble_alu32``; +4 C90 correctness, 0 regressions, doom + Mandelbrot byte-exact).
    #   ROLLBACK LADDER: ``C4_DIVMOD_SIGNED=0`` -> 7d4afe61 -> ``C4_BP_RESTORE_HIBYTE=0`` -> 069cc32f.
    #
    #   7d4afe61 (the ``C4_DIVMOD_SIGNED=0`` rung) was itself RE-BASELINED 2026-08-06 from 069cc32f
    #   by the DEFAULT-ON flip of ``C4_BP_RESTORE_HIBYTE`` (a general-program-correctness fix, see
    #   ``nibble_pure_forward_complete._bp_restore_hibyte_nibs``): the three LEV/pop scalar
    #   recomposes (``compile_stk_recompose`` / ``compile_unify_stk_recompose`` /
    #   ``compile_lev_ret_recompose``) now read the fp32-safe ``_recompose_hi_nibbles()`` = 5
    #   nibbles instead of the hardcoded fp64-only ``hi_nibbles=8``.  The dropped hi-nibble
    #   units (j∈[5,7], 3 per recompose op × 3 ops) carried only fp32 CAM residue on a saved
    #   BP / return-PC, so this is correct-by-construction (0x10000 → 0xFEEF residue drop
    #   eliminated).  MANDELBROT is now FULLY byte-exact (interior/escape/boundary) and DOOM
    #   stays byte-exact.  An INTENTIONAL move — the recompose FFN units shrink by construction.
    #
    #   ESCAPE HATCH (rollback + old-byte-identity): ``C4_BP_RESTORE_HIBYTE=0`` reverts to the
    #   old ``hi_nibbles=8`` recompose and rebuilds the PRE-FIX golden
    #   069cc32fa7cecfbceae448a7dbf6e2140b3db6cf6857c8accec5639b9c55c0ca (short 069cc32f).
    #   (069cc32f was itself RE-BASELINED 2026-07 from 8f4dd780 by the c4-faithful
    #   SHR-ARITHMETIC ``build_sign_fill`` + SIGNED-LC ``lc-sign-detect``/``lc-sign-extend``
    #   fix — an INTENTIONAL move where the sign-fill gadgets fire only on OP_IS[SHR]∧sign /
    #   OP_IS[LC]∧high-bit, so the whole 1096 corpus is unchanged.)
    #
    # C4_DOOM_DRAWSPAN (default OFF): the native DRAWSPAN render-macro opcode (47,
    #   ``c4_min.doom_drawspan``) that fuses Doom's V_DrawPatch column-copy loop.
    #   OFF -> ``isa.num_ops_effective`` stays NUM_OPS=40, so the OP_IS one-hot band
    #   and every downstream layout dim are byte-identical -> the DRAWSPAN-OFF
    #   fingerprint is the golden (3cabef64 by default; 7d4afe61 with ``C4_DIVMOD_SIGNED=0``;
    #   069cc32f with ``C4_DIVMOD_SIGNED=0 C4_BP_RESTORE_HIBYTE=0``).  ON -> ``num_ops_effective`` widens to
    #   NUM_OPS_DRAWSPAN=48 so the opcode-47 one-hot has a slot (the SAME lever
    #   NUM_OPS_FLOAT uses), which is an INTENTIONAL move of the fingerprint (wider
    #   OP_IS band by construction):
    #     C4_DOOM_DRAWSPAN=1 fingerprint (with the PRE-FIX ``C4_BP_RESTORE_HIBYTE=0``
    #       base): 2f69350f81d05c43df28d4c63bb78c0a2e32dd1aae07f05c7af3611fb67e164c
    #       (short 2f69350f).  DELIBERATE — do NOT treat as a regression.  NOTE: under
    #       the new BP-restore default this ON hash shifts (the recompose units also
    #       shrank); the DRAWSPAN-OFF fingerprint remains the golden gate.
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
