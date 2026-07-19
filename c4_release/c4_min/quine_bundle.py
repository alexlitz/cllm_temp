"""Assemble + run the QUINE BUNDLE — a single artifact that, when executed, runs
the quine bytecode through the SMALL c4_min pure-forward transformer and
reproduces the program's own source, byte-for-byte, via PRTF.

BLOG_SPEC §"Making a Quine" (line 928) frames the quine as a program *bundled with
its runtime + weights + bytecode*.  This module realises that bundle on the small
model (feasible now that the model is a few MB sparse, not 30 GB):

  a) RUNTIME / DRIVER — ``nibble_pure_forward_cached.run_pure_forward_cached`` (the
     KV-cached pure-forward VM) + ``sparse_forward.SparseTransformer`` + this file.
     Referenced by import from the bundle's ``run`` entrypoint (it is the c4_min
     package the bundle ships with).
  b) SMALL MODEL WEIGHTS — the compact + dim-shared pure-forward VM, stored SPARSE
     (COO) so the whole model is a few MB on disk (vs 2.5 GB dense-equiv / 30 GB
     naive).  This is what makes the bundle practical.
  c) QUINE BYTECODE — the ``(op, imm)`` code table of the PRTF string quine
     (``quine_prtf``) plus its data segment ``seed_mem`` (the program's own source
     bytes as the table ``Q``).

``build_bundle(path)`` bakes the small model once and writes (a-ref, b, c) to a
single ``.pt`` file.  ``run_bundle(path)`` loads it, runs the bytecode through the
model, and returns the visible output — which equals the bundled source bytes.

Memory-horizon note: the quine's data segment ``Q`` is loaded at arbitrary future
steps over a ~37 k-token run, beyond the default §Memory ALiBi reachability horizon
(``EFF/MEM_ALIBI_SLOPE ≈ 666`` tokens).  The bundle bakes the model with a widened
horizon (``MEM_ALIBI_SLOPE = 0.05`` → ``EFF/slope ≈ 80 k`` tokens) so a far-back
data store still reads weight ~1 while same-address counter writes (spaced ~1 frame
apart) still get decisive latest-write-wins recency.  This slope is a bundle-local
model parameter; the CHK-1 corpus model uses the default slope.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

from . import isa
from . import blogspec_vocab as V


# The widened §Memory reachability horizon the quine's persistent data segment
# needs (default is 6.0 → ~666-token horizon; 0.05 → ~80 k-token horizon).
QUINE_MEM_ALIBI_SLOPE = 0.05
SP_INIT_QUINE = 0xF0                 # matches the CHK-1 runner's pinned SP_INIT


def _pin_slope_and_sp(slope: float = QUINE_MEM_ALIBI_SLOPE):
    """Pin ``SP_INIT`` and widen the memory horizon for the quine build/run.  Must
    run BEFORE the model is baked (the memory head reads the slope at bake time)."""
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.blogspec_memory as _MEM
    _PF.SP_INIT = SP_INIT_QUINE
    _PFC.SP_INIT = SP_INIT_QUINE
    _MEM.MEM_ALIBI_SLOPE = slope


def _build_small_model(code_size: int = 64):
    """Bake the compact + dim-shared pure-forward VM (small model).  Returns
    ``(model, L, stats)``."""
    from .compact_alloc import build_compact_pure_forward_model
    return build_compact_pure_forward_model(
        code_size=code_size)


def _sparse_state_dict(model) -> Dict[str, object]:
    """COO-sparsify the model's state dict for compact storage (the ~180 k-nnz
    weights dwarfed by dense zeros → a few MB COO vs 2.5 GB dense)."""
    from .compact_alloc import sparse_state_dict
    return sparse_state_dict(model, min_dense_to_sparsify=1)


def build_bundle(path: str, code_size: int = 64) -> Dict[str, object]:
    """Bake the small model + quine and write the BUNDLE to ``path`` (a ``.pt``).

    The bundle dict carries:
      * ``config``       — everything to reconstruct the empty model shell + layout
                           (dim, n_heads, hidden, n_blocks, vocab, max_seq_len,
                           code_size, mem_alibi_slope, sp_init).
      * ``state_dict``   — the COO-sparse small-model weights (b).
      * ``code``         — the quine bytecode as ``[(op, imm), ...]`` (c).
      * ``seed_mem``     — the data segment ``{addr: byte}`` (c).
      * ``source_bytes`` — the expected self-output ``S`` (for the self-check).
    Returns a summary dict (sizes / counts).
    """
    from .quine_prtf import build_quine

    _pin_slope_and_sp()
    code, seed_mem, S = build_quine()
    model, L, stats = _build_small_model(code_size=code_size)

    sd = _sparse_state_dict(model)
    config = {
        "dim": model.dim,
        "n_heads": model.blocks[0].attn.n_heads,
        "hidden": model.blocks[0].ffn.W_up.shape[0],
        "n_blocks": len(model.blocks),
        "vocab": model.vocab,
        "max_seq_len": model.max_seq_len,
        "code_size": code_size,
        "mem_alibi_slope": QUINE_MEM_ALIBI_SLOPE,
        "sp_init": SP_INIT_QUINE,
    }
    bundle = {
        "config": config,
        "state_dict": sd,                       # COO-sparse small weights
        "layout": L,                            # the COMPACTED (dim-shared) layout the
                                                # compacted weights use — must be saved,
                                                # NOT rebuilt (compaction re-indexes dims)
        "code": [(int(ins.op), int(ins.imm)) for ins in code],
        "seed_mem": {int(a): int(b) for a, b in seed_mem.items()},
        "source_bytes": list(S),
        "format": "c4_min.quine_bundle.v1",
    }
    torch.save(bundle, path)
    size = os.path.getsize(path)
    return {
        "path": path, "bundle_bytes": size, "bundle_mb": size / 1e6,
        "n_instr": len(code), "n_source_bytes": len(S),
        "dim": model.dim, "n_blocks": len(model.blocks),
        "nnz": int(stats.nonzero_params) if hasattr(stats, "nonzero_params") else None,
    }


def _load_model_from_bundle(bundle) -> Tuple[object, object]:
    """Reconstruct the SparseTransformer + layout ``L`` from a loaded bundle."""
    from .blogspec_model import Transformer
    from .sparse_forward import SparseTransformer

    cfg = bundle["config"]
    _pin_slope_and_sp(cfg.get("mem_alibi_slope", QUINE_MEM_ALIBI_SLOPE))

    # Materialise the dense state dict from the COO-sparse storage.
    dense_sd = {}
    for name, entry in bundle["state_dict"].items():
        kind, t, shape = entry
        dense_sd[name] = (t.to_dense() if kind == "coo" else t).reshape(shape)
    model = Transformer(dim=cfg["dim"], n_heads=cfg["n_heads"], hidden=cfg["hidden"],
                        n_blocks=cfg["n_blocks"], vocab=cfg["vocab"],
                        max_seq_len=cfg["max_seq_len"])
    # The compact model's FFNs are RAGGED (each block sized to its own real hidden,
    # smaller than the uniform ``hidden`` a fresh Transformer allocates), so resize
    # each block's FFN Parameters to the stored shapes before loading.
    import torch.nn as _nn
    for bi in range(cfg["n_blocks"]):
        ffn = model.blocks[bi].ffn
        for pn in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
            key = f"blocks.{bi}.ffn.{pn}"
            if key in dense_sd:
                setattr(ffn, pn, _nn.Parameter(torch.zeros_like(dense_sd[key])))
    model.load_state_dict(dense_sd)
    # Use the SAVED compacted layout (the compaction re-indexes dims, so a rebuilt
    # PRE-compaction layout would read registers at the WRONG positions).
    L = bundle["layout"]
    sparse = SparseTransformer(model, compute_mode="dense_kernel")
    return sparse, L


def run_bundle(path: str, device: str = "cuda:0", max_steps: int = 4000,
               verbose: bool = False) -> Tuple[List[int], List[int], bool]:
    """Load the bundle at ``path`` and RUN the quine through the small model.

    Returns ``(visible, source, ok)`` where ``visible`` is the PRTF byte stream the
    model produced and ``source`` is the bundled source bytes; ``ok`` is
    ``visible == source`` (the byte-exact self-output verdict)."""
    from .nibble_pure_forward_cached import run_pure_forward_cached

    bundle = torch.load(path, weights_only=False)
    sparse, L = _load_model_from_bundle(bundle)
    if device.startswith("cuda") and torch.cuda.is_available():
        sparse = sparse.to(device)
    code = isa.assemble([(isa.NAMES[op], imm) for op, imm in bundle["code"]])
    seed_mem = {int(a): int(b) for a, b in bundle["seed_mem"].items()}
    source = list(bundle["source_bytes"])

    out: List[int] = []
    run_pure_forward_cached(sparse, L, code, max_steps=max_steps, mask=0xFFFFFFFF,
                            evict=True, prune_interval=90, out=out, seed_mem=seed_mem,
                            verbose=verbose)
    ok = out == source
    return out, source, ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build/run the c4_min quine bundle.")
    ap.add_argument("--path", default="/tmp/c4_quine_bundle.pt")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.build:
        info = build_bundle(args.path)
        print("BUNDLE WRITTEN:", info)
    if args.run:
        vis, src, ok = run_bundle(args.path, device=args.device)
        print("printed %d bytes" % len(vis))
        print("QUINE OK (bundle self-output == bundled source):", ok)
        if not ok:
            print("source :", src)
            print("printed:", vis)
