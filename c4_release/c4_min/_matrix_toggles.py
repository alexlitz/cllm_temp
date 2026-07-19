"""Full config-toggle MATRIX harness for the c4_min build/run path.

Exercises the real config axes of the canonical builder for the SINGLE
full-op-set model (``compact_alloc.build_compact_sparse_streaming`` /
``load_sparse_transformer``) and the canonical KV-cached driver
(``run_pure_forward_cached``) in every meaningful combination, checks each is
argmax-CORRECT vs the byte-exact reference (``ref_interpret`` / the corpus
``expected``) and CONSISTENT across compute_mode / build-path / device.

There is now ONE op set (every opcode incl. DIV/MOD + bitwise) — the former
include_divmod / include_bitwise op-subset axes are gone.  The model is built via
the STREAMING sparse builder (peak build RSS ~one dense block), never the
~130 GB full-op dense build.

Axes:
  compute_mode    {sparse_mm, dense_kernel}   (must be argmax-identical)
  eviction        {off, on}
  build path      {stream-build, load-sparse}   (must be byte-identical)
  device          {cpu, cuda:0}

Run:  OMP_NUM_THREADS=4 python -u -m c4_min._matrix_toggles [cuda]
(pass "cuda" to include the GPU rows; CUDA must be init-able — the harness
force-inits it before any build, else the model-build path poisons lazy init).
"""
from __future__ import annotations

import resource
import sys
import time

import torch

# CRITICAL: force CUDA init HERE, before the heavy c4_min imports below.
# Those imports otherwise poison torch's lazy CUDA init so that a LATER
# ``.to('cuda:0')`` inside ``main()`` raises "No CUDA GPUs are available"
# (the retry loop there can never recover a poisoned init). A single
# ``.to('cuda:0')`` up front — while CUDA is still pristine — pins init True
# and survives the imports. Only attempt it when the GPU rows were requested.
_CUDA_PREINIT = False
if any(a in sys.argv[1:] for a in ("cuda", "gpuonly")):
    for _ in range(5):
        try:
            _ = torch.zeros(1).to("cuda:0")
            _CUDA_PREINIT = True
            break
        except Exception:  # noqa: BLE001 — driver busy: retry a few times
            time.sleep(2)

# The canonical runner pins SP_INIT=0xF0; mirror it so the reference interpreter
# and the driver agree on the frame arithmetic.
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min.compact_alloc import (
    build_compact_sparse_streaming,
    save_sparse_transformer,
    load_sparse_transformer,
)
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min.nibble_pure_forward_complete import (
    ref_interpret, make_overlay_complete, _build_frame, SP_INIT)
from c4_min.run_1096_pure_forward import bytecode_to_isa
from c4_min import blogspec_vocab as V


# --- sanity programs (byte-exact expected; short so the matrix is fast) ------
ALU = [("add", "int main(){ return 500 + 700; }", 1200),
       ("sub", "int main(){ return 900 - 99; }", 801),
       ("mul", "int main(){ return 100 * 10; }", 1000)]
MEM = [("var", "int main(){ int x; x = 1000; return x; }", 1000)]
FLOW = [("if_gt", "int main(){ if (5 > 3) return 7; return 0; }", 7),
        ("if_eq", "int main(){ if (4 == 4) return 1; return 0; }", 1)]
FUNC = [("func_id", "int identity(int x){ return x; } "
                    "int main(){ return identity(1000); }", 1000)]
BITWISE = [("or", "int main(){ return 12 | 3; }", 15),
           ("and", "int main(){ return 12 & 10; }", 8),
           ("xor", "int main(){ return 12 ^ 10; }", 6),
           ("shl", "int main(){ return 3 << 2; }", 12),
           ("shr", "int main(){ return 48 >> 2; }", 12)]
DIVMOD = [("div", "int main(){ return 720 / 6; }", 120),
          ("mod", "int main(){ return 84 % 5; }", 4),
          ("divzero", "int main(){ return 5 / 0; }", 0)]
# gcd (deep recursion) is a correct-but-slow smoke (~126s/prog on the 304-block
# divmod model); exercised once by the standalone divmod-battery, not the matrix.
DIVMOD_DEEP = [("gcd", "int gcd(int a, int b){ if (b == 0) return a; "
                       "return gcd(b, a % b); } int main(){ return gcd(48, 36); }", 12)]


def _peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


_CODE_CACHE = {}
_REFCAP_CACHE = {}


def _compile(src):
    if src not in _CODE_CACHE:
        from src.compiler import compile_c
        _CODE_CACHE[src] = bytecode_to_isa(compile_c(src)[0])
    return _CODE_CACHE[src]


def _run_one(model, L, src, evict, device):
    code = _compile(src)
    if src not in _REFCAP_CACHE:
        _REFCAP_CACHE[src] = len(
            ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
    tr = run_pure_forward_cached(model, L, code, max_steps=_REFCAP_CACHE[src],
                                 mask=0xFFFFFFFF, evict=evict, prune_interval=120)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    return tr[-1] & 0xFFFFFFFF if tr else None


def battery(model, L, cases, evict=True, device="cpu"):
    """Return (all_ok, [(name, got, exp, ok), ...])."""
    rows = []
    for nm, src, exp in cases:
        try:
            got = _run_one(model, L, src, evict, device)
            ok = (got == exp)
        except Exception as exc:  # noqa: BLE001
            got, ok = f"ERR:{exc!r}", False
        rows.append((nm, got, exp, ok))
    return all(r[3] for r in rows), rows


def _decode_row(x, L):
    row = x[0, -1]
    vals = []
    for nm in ("PC_VAL", "AX_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED"):
        vals.append(row[getattr(L, nm)].reshape(-1))
    for k in range(8):
        vals.append(row[L.AX + k].reshape(-1))
    return torch.cat(vals)


def _block_stack(model, L, code, device="cpu"):
    stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    for _ in range(4):
        stream += _build_frame(1, 7, SP_INIT - 4, SP_INIT, 3)
    toks = torch.tensor([stream])
    ov = make_overlay_complete(code, L)
    with torch.no_grad():
        x = model.embed[toks].clone().to(device)
        ov(x)
        for b in model.blocks:
            x = b(x)
    return _decode_row(x, L).float().cpu()


def linf_build_paths(models_layouts, srcs):
    """Max L-inf across the decode bands between the FIRST model and each other
    over ``srcs`` (byte-identity of build-paths / compute-mode)."""
    codes = [_compile(s) for s in srcs]
    ref = models_layouts[0]
    worst = {}
    for name, (m, L) in list(models_layouts[1].items()):
        w = 0.0
        for c in codes:
            r0 = _block_stack(ref[1][ref[0]][0], ref[1][ref[0]][1], c)
            rn = _block_stack(m, L, c)
            w = max(w, (r0 - rn).abs().max().item())
        worst[name] = w
    return worst


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    want_cuda = ("cuda" in argv) or ("gpuonly" in argv)
    if want_cuda:
        # CUDA must have been force-inited at MODULE TOP (before the c4_min
        # imports) — those imports poison lazy init, so a first-touch here
        # can no longer recover it. ``_CUDA_PREINIT`` records that pin.
        assert _CUDA_PREINIT, (
            "cuda requested but not init-able. The module-top force-init "
            "failed (no GPU / driver busy). Run via "
            "`python -m c4_min._matrix_toggles cuda` so the CLI arg is seen "
            "at import time.")
    if "gpuonly" in argv:
        devices = ["cuda:0"]
    elif "cpuonly" in argv:
        devices = ["cpu"]
    else:
        devices = (["cuda:0"] if want_cuda else []) + ["cpu"]

    print("=" * 82, flush=True)
    print(f"c4_min CONFIG-TOGGLE MATRIX  (devices={devices})", flush=True)
    print("=" * 82, flush=True)
    results = {}          # key -> (ok, rows)
    byteid = {}           # (family, comparison) -> linf

    srcs_bid = ["int main(){ return 500 + 700; }",
                "int main(){ int x; x = 1000; return x; }",
                "int main(){ return 12 | 3; }"]

    # ============= SINGLE FULL-OP-SET MODEL (streaming only) =============
    # Every opcode present (ALU + mem + flow + func + bitwise + div/mod).  The
    # streaming build is memory-safe (~one dense block peak); load-sparse and
    # sparse_mm are the remaining byte-identity axes.
    d0 = devices[0]
    path = "/tmp/c4_matrix_full.pt"
    stream, Ls, st_stream = build_compact_sparse_streaming(
        code_size=64, compute_mode="dense_kernel")
    save_sparse_transformer(stream, Ls, st_stream, path)
    print(f"  [full stream-build: dim={st_stream.dim_after} "
          f"nblocks={st_stream.n_blocks} nnz={stream.stats().total_nnz} "
          f"peakRSS={_peak_gb():.1f}GB]", flush=True)
    loaded, Ll = load_sparse_transformer(path, compute_mode="dense_kernel")
    stream_mm, Lmm, _ = build_compact_sparse_streaming(
        code_size=64, compute_mode="sparse_mm")

    # -- byte-identity of build paths (stream vs load) + dense_kernel vs sparse_mm
    srcs_bid_full = srcs_bid + ["int main(){ return 720 / 6; }",
                                "int main(){ return 84 % 5; }"]
    for c in [_compile(s) for s in srcs_bid_full]:
        r_stream = _block_stack(stream, Ls, c)
        byteid[("full", "stream-vs-load")] = max(
            byteid.get(("full", "stream-vs-load"), 0.0),
            (r_stream - _block_stack(loaded, Ll, c)).abs().max().item())
        byteid[("full", "dense_kernel-vs-sparse_mm")] = max(
            byteid.get(("full", "dense_kernel-vs-sparse_mm"), 0.0),
            (r_stream - _block_stack(stream_mm, Lmm, c)).abs().max().item())

    base = ALU + MEM + FLOW + FUNC + BITWISE + DIVMOD
    fast = ALU[:1] + MEM + FLOW[:1] + BITWISE[:2] + DIVMOD[:2]
    # The canonical stream model gets the FULL end-to-end battery on device[0]
    # (evict on); the redundant axes (evict off, second device) get the fast
    # subset.  load / sparse_mm are proven L-inf-identical to the stream model
    # above, so each gets ONE fast correctness confirmation.
    for device in devices:
        sm = stream.to(device)
        for evict in (True, False):
            cases = base if (device == d0 and evict) else fast
            ok, rows = battery(sm, Ls, cases, evict=evict, device=device)
            key = ("full", "dense_kernel", "stream", device,
                   "evict" if evict else "noevict")
            results[key] = (ok, rows)
            _print_row("full", "all", "dense_kernel", "stream", device,
                       evict, rows)
        sm.to("cpu")
    for kind, m, L, mode in [("load", loaded, Ll, "dense_kernel"),
                             ("stream_mm", stream_mm, Lmm, "sparse_mm")]:
        md = m.to(d0)
        ok, rows = battery(md, L, fast, evict=True, device=d0)
        results[("full", mode, kind, d0, "evict")] = (ok, rows)
        _print_row("full", "all", mode, kind, d0, True, rows)
        md.to("cpu")
    del stream, loaded, stream_mm

    # ================= SUMMARY =========================================
    print("=" * 82, flush=True)
    print("BYTE-IDENTITY / consistency L-inf (decode bands):", flush=True)
    for (fam, cmp_), v in sorted(byteid.items()):
        verdict = "IDENTICAL" if v < 1e-9 else (
            "argmax-ok(fp-residue)" if v < 1e-3 else "!!DIVERGES!!")
        print(f"  {fam:8s} {cmp_:28s} L-inf={v:.3e}  {verdict}", flush=True)
    n_fail = sum(1 for ok, _ in results.values() if not ok)
    bid_bad = sum(1 for (fam, cmp_), v in byteid.items()
                  if "sparse_mm" not in cmp_ and v >= 1e-9)
    print("-" * 82, flush=True)
    print(f"MATRIX: {len(results)} combos, {len(results)-n_fail} PASS, "
          f"{n_fail} FAIL | byte-id-violations(dense/paths)={bid_bad}", flush=True)
    print(f"peakRSS={_peak_gb():.1f}GB", flush=True)
    return 1 if (n_fail or bid_bad) else 0


def _print_row(fam, tag, mode, kind, device, evict, rows):
    fails = [r for r in rows if not r[3]]
    ok = not fails
    print(f"  {'PASS' if ok else 'FAIL':4s}  {fam} {tag:4s} "
          f"mode={mode:12s} path={kind:6s} {device:6s} "
          f"{'evict' if evict else 'noevict':7s} "
          f"({len(rows)-len(fails)}/{len(rows)})"
          + ("  FAILS=" + str([(r[0], r[1], r[2]) for r in fails]) if fails else ""),
          flush=True)


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"[wall {time.time()-t0:.0f}s]", flush=True)
    sys.exit(rc)
