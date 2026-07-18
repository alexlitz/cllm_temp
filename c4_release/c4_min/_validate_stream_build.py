"""Validation harness for the streaming sparse build (the OOM fix).

Each build runs in a SEPARATE process invocation so ``ru_maxrss`` reports that
one build's peak.  Usage:

    python -m c4_min._validate_stream_build lean-peak-stream
    python -m c4_min._validate_stream_build lean-peak-old
    python -m c4_min._validate_stream_build lean-byteid
    python -m c4_min._validate_stream_build divmod-peak-stream
    python -m c4_min._validate_stream_build divmod-battery-stream
"""
from __future__ import annotations

import resource
import sys
import time

import torch

from c4_min import blogspec_vocab as V


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _decode_bands(x, L):
    """Return a flat tensor of the DECODE bands the driver argmaxes."""
    row = x[0, -1]
    vals = []
    for nm in ("PC_VAL", "AX_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED"):
        vals.append(row[getattr(L, nm)].reshape(-1))
    for k in range(8):
        vals.append(row[L.AX + k].reshape(-1))
    return torch.cat(vals)


def _probe_streams(include_bitwise, include_divmod):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import _build_frame, SP_INIT
    srcs = ["int main(){ return 500 + 700; }",
            "int main(){ int x; x = 1000; return x; }",
            "int main(){ return 100 * 10; }",
            "int identity(int x){ return x; } int main(){ return identity(1000); }"]
    if include_bitwise:
        srcs.append("int main(){ return 12 | 3; }")
        srcs.append("int main(){ return 12 & 10; }")
    if include_divmod:
        srcs.append("int main(){ return 720 / 6; }")
        srcs.append("int main(){ return 84 % 5; }")
    out = []
    for src in srcs:
        code = bytecode_to_isa(compile_c(src)[0])
        stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        for _ in range(4):
            stream += _build_frame(1, 7, SP_INIT - 4, SP_INIT, 3)
        out.append((code, stream))
    return out


def peak_stream(include_bitwise, include_divmod):
    from c4_min.compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    sparse, L, stats = build_compact_sparse_streaming(
        code_size=44, include_bitwise=include_bitwise,
        include_divmod=include_divmod, compute_mode="dense_kernel")
    peak = _peak_gb()
    st = sparse.stats()
    print(f"STREAM build: n_blocks={len(sparse.blocks)} dim_after={stats.dim_after} "
          f"nnz={st.total_nnz} sparse_mb={st.sparse_mb:.2f} "
          f"dense_gb={st.dense_gb:.2f} peakRSS={peak:.2f}GB t={time.time()-t0:.1f}s")
    return sparse, L, stats


def peak_old(include_bitwise, include_divmod):
    from c4_min.compact_alloc import build_compact_pure_forward_model
    from c4_min.sparse_forward import SparseTransformer
    t0 = time.time()
    compact, L, stats = build_compact_pure_forward_model(
        code_size=44, include_bitwise=include_bitwise,
        include_divmod=include_divmod)
    sparse = SparseTransformer(compact, compute_mode="dense_kernel")
    peak = _peak_gb()
    print(f"OLD build: n_blocks={len(compact.blocks)} dim_after={stats.dim_after} "
          f"peakRSS={peak:.2f}GB t={time.time()-t0:.1f}s")
    return sparse, L, stats


def byteid(include_bitwise, include_divmod):
    """L-inf between the OLD (compact->SparseTransformer) and STREAM sparse builds
    on the decode bands, across probe streams."""
    from c4_min.compact_alloc import (build_compact_pure_forward_model,
                                       build_compact_sparse_streaming)
    from c4_min.sparse_forward import SparseTransformer
    from c4_min.nibble_pure_forward_complete import make_overlay_complete

    compact, Lo, _ = build_compact_pure_forward_model(
        code_size=44, include_bitwise=include_bitwise, include_divmod=include_divmod)
    old = SparseTransformer(compact, compute_mode="dense_kernel")
    new, Ln, _ = build_compact_sparse_streaming(
        code_size=44, include_bitwise=include_bitwise,
        include_divmod=include_divmod, compute_mode="dense_kernel")

    worst = 0.0
    for code, stream in _probe_streams(include_bitwise, include_divmod):
        toks = torch.tensor([stream])
        ov_o = make_overlay_complete(code, Lo)
        ov_n = make_overlay_complete(code, Ln)
        with torch.no_grad():
            xo = old.embed[toks].clone(); ov_o(xo)
            for blk in old.blocks:
                xo = blk(xo)
            xn = new.embed[toks].clone(); ov_n(xn)
            for blk in new.blocks:
                xn = blk(xn)
        worst = max(worst, (_decode_bands(xo, Lo) - _decode_bands(xn, Ln)).abs().max().item())
    print(f"BYTE-IDENTITY L-inf(old sparse vs stream sparse) = {worst:.3e}")
    assert worst < 1e-9, f"NOT byte-identical: {worst}"
    print("PASS: streaming build is byte-identical to old sparse build (L-inf=0)")


def battery(sparse, L, cases):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import (
        run_pure_forward_complete, ref_interpret)
    ok = True
    for name, src, exp in cases:
        code = bytecode_to_isa(compile_c(src)[0])
        cap = len(ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
        tr = run_pure_forward_complete(sparse, L, code, max_steps=cap, mask=0xFFFFFFFF)
        got = tr[-1] & 0xFFFFFFFF if tr else None
        flag = "OK" if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(f"  {flag} {name}: got {got} want {exp}")
    print("BATTERY", "PASS" if ok else "FAIL")
    return ok


DIVMOD_CASES = [
    ("div", "int main(){ return 720 / 6; }", 120),
    ("mod", "int main(){ return 84 % 5; }", 4),
    ("gcd", "int gcd(int a, int b){ if (b == 0) return a; return gcd(b, a % b); } "
            "int main(){ return gcd(48, 36); }", 12),
    ("divzero", "int main(){ return 5 / 0; }", 0),
]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "lean-byteid"
    if cmd == "lean-peak-stream":
        peak_stream(True, False)
    elif cmd == "lean-peak-old":
        peak_old(True, False)
    elif cmd == "lean-byteid":
        byteid(True, False)
    elif cmd == "divmod-peak-stream":
        peak_stream(True, True)
    elif cmd == "divmod-byteid":
        byteid(True, True)
    elif cmd == "divmod-battery-stream":
        sparse, L, stats = peak_stream(True, True)
        battery(sparse, L, DIVMOD_CASES)
    elif cmd == "save-divmod":
        path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/divmod_sparse.pt"
        from c4_min.compact_alloc import (build_compact_sparse_streaming,
                                          save_sparse_transformer)
        sparse, L, stats = build_compact_sparse_streaming(
            code_size=44, include_bitwise=True, include_divmod=True,
            compute_mode="dense_kernel")
        save_sparse_transformer(sparse, L, stats, path)
        print(f"SAVED divmod sparse artifact -> {path} peakRSS={_peak_gb():.2f}GB")
    else:
        print("unknown cmd", cmd)
        sys.exit(2)
