#!/usr/bin/env python3
"""Measure the divmod compact-sparse model: blocks, nonzero weights, artifact
size (MB), peak build RSS.  Run BEFORE and AFTER the recurrent refactor."""
import os, sys, time, resource, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from c4_min.compact_alloc import (build_compact_sparse_streaming,
                                  save_sparse_transformer)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def measure(tag, code_size=48, recurrent_divmod=False):
    t0 = time.monotonic()
    sparse, L, stats = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel",
        recurrent_divmod=recurrent_divmod)
    build_s = time.monotonic() - t0
    n_phys = len(getattr(sparse, "_phys_blocks", sparse.blocks))  # STORED blocks
    n_apply = len(sparse.blocks)                                  # APPLICATIONS
    st = sparse.stats()
    nnz = st.total_nnz
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    save_sparse_transformer(sparse, L, stats, path)
    size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    os.unlink(path)
    print(f"[{tag}] stored_blocks={n_phys} applied_blocks={n_apply} nnz={nnz} "
          f"artifact={size_mb:.2f}MB dim={L.D} build={build_s:.1f}s "
          f"peak_rss={peak_rss_gb():.2f}GB")
    return dict(blocks=n_phys, applied=n_apply, nnz=nnz, size_mb=size_mb,
                dim=L.D, build_s=build_s)


if __name__ == "__main__":
    # The full-op-set model always has DIV/MOD; the only shape knob is
    # recurrent_divmod (unrolled 262-block span vs the 21-block reused body).
    which = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if which == "unrolled":
        measure("DIVMOD-UNROLLED")
    elif which == "recurrent":
        measure("DIVMOD-RECURRENT", recurrent_divmod=True)
    else:   # compare
        measure("DIVMOD-UNROLLED")
        measure("DIVMOD-RECURRENT", recurrent_divmod=True)
    print(f"[final] peak_rss={peak_rss_gb():.2f}GB")
