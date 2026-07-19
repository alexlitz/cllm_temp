"""Build + save the SPARSE pure-forward artifacts (lean + divmod) for the
aggressive-speculation full-1096 verify runner, via the STREAMING builder so the
peak build RSS is ~one dense block (never the 79 GB dense divmod).

The saved artifact bakes ALL weights (incl. the EFF=500000 memory-CAM) + the
layout, so ``--load-sparse`` reloads a byte-identical model with NO rebuild.

Usage:
    python -m c4_min._build_artifacts lean  /tmp/c4_lean_sparse_fs.pt
    python -m c4_min._build_artifacts divmod /tmp/c4_divmod_sparse_fs.pt
"""
from __future__ import annotations

import resource
import sys
import time

from c4_min.compact_alloc import (build_compact_sparse_streaming,
                                   save_sparse_transformer)
from c4_min import blogspec_memory as _MEM

_CODE_SIZE = 48   # corpus max instr count is 41 (id=900); 48 gives margin.


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> int:
    which = sys.argv[1]
    path = sys.argv[2]
    include_divmod = (which == "divmod")
    assert abs(_MEM.EFF - 500000.0) < 1e-6, f"EFF must be 500000, got {_MEM.EFF}"
    t0 = time.time()
    sparse, L, stats = build_compact_sparse_streaming(
        code_size=_CODE_SIZE, include_bitwise=True,
        include_divmod=include_divmod, compute_mode="dense_kernel")
    st = sparse.stats()
    save_sparse_transformer(sparse, L, stats, path)
    print(f"SAVED {which} -> {path} | blocks={len(sparse.blocks)} dim={sparse.dim} "
          f"nnz={st.total_nnz} sparse_mb={st.sparse_mb:.2f} EFF={_MEM.EFF} "
          f"code_size={_CODE_SIZE} peakRSS={_peak_gb():.2f}GB t={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
