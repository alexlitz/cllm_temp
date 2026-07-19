"""Full config-toggle MATRIX harness for the c4_min build/run path.

Exercises the real config axes of the canonical builder
(``compact_alloc.build_compact_sparse_streaming`` /
``build_compact_pure_forward_model`` / ``load_sparse_transformer``) and the
canonical KV-cached driver (``run_pure_forward_cached``) in every meaningful
combination, checks each is argmax-CORRECT vs the byte-exact reference
(``ref_interpret`` / the corpus ``expected``) and CONSISTENT across
compute_mode / build-path / device.

Axes:
  include_divmod  {False, True}   (True => stream-build / load-sparse, never 79GB dense)
  include_bitwise {False, True}
  compute_mode    {sparse_mm, dense_kernel}   (must be argmax-identical)
  eviction        {off, on}
  build path      {fresh, stream-build, load-sparse}   (must be byte-identical)
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

# The canonical runner pins SP_INIT=0xF0; mirror it so the reference interpreter
# and the driver agree on the frame arithmetic.
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min.compact_alloc import (
    build_compact_pure_forward_model,
    build_compact_sparse_streaming,
    save_sparse_transformer,
    load_sparse_transformer,
)
from c4_min.sparse_forward import SparseTransformer
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
    want_cuda = "cuda" in argv
    if want_cuda:
        # CRITICAL: force CUDA init BEFORE any model build. The build path
        # (fork/thread interplay) otherwise poisons torch's lazy CUDA init and
        # `.to('cuda:0')` then raises "No CUDA GPUs are available".
        # `is_available()` CACHES its first result, so a transient
        # driver-busy at startup would pin it False — retry a few times.
        ok = False
        for _ in range(5):
            try:
                _ = torch.zeros(1).to("cuda:0")
                ok = True
                break
            except Exception:  # noqa: BLE001
                torch.cuda.is_available.cache_clear() if hasattr(
                    torch.cuda.is_available, "cache_clear") else None
                time.sleep(2)
        assert ok, "cuda requested but not init-able (driver busy?)"
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

    # ================= LEAN family (no divmod) ==========================
    for include_bitwise in (False, True):
        tag = "bw" if include_bitwise else "nobw"
        fresh_c, Lf, _ = build_compact_pure_forward_model(
            code_size=64, include_bitwise=include_bitwise, include_divmod=False)
        fresh = SparseTransformer(fresh_c, compute_mode="dense_kernel")
        del fresh_c
        path = f"/tmp/c4_matrix_lean_{tag}.pt"
        stream, Ls, st_stream = build_compact_sparse_streaming(
            code_size=64, include_bitwise=include_bitwise,
            include_divmod=False, compute_mode="dense_kernel")
        save_sparse_transformer(stream, Ls, st_stream, path)
        loaded, Ll = load_sparse_transformer(path, compute_mode="dense_kernel")
        stream_mm, Lmm, _ = build_compact_sparse_streaming(
            code_size=64, include_bitwise=include_bitwise,
            include_divmod=False, compute_mode="sparse_mm")

        # -- byte-identity of build paths (fresh vs stream vs load) + sparse_mm
        for c in [_compile(s) for s in srcs_bid]:
            r_fresh = _block_stack(fresh, Lf, c)
            for nm, (m, L) in [("stream", (stream, Ls)), ("load", (loaded, Ll))]:
                d = (r_fresh - _block_stack(m, L, c)).abs().max().item()
                byteid[(tag, f"fresh-vs-{nm}")] = max(
                    byteid.get((tag, f"fresh-vs-{nm}"), 0.0), d)
            d = (r_fresh - _block_stack(stream_mm, Lmm, c)).abs().max().item()
            byteid[(tag, "dense_kernel-vs-sparse_mm")] = max(
                byteid.get((tag, "dense_kernel-vs-sparse_mm"), 0.0), d)

        base = ALU + MEM + FLOW + FUNC + (BITWISE if include_bitwise else [])
        fast = ALU[:1] + MEM + FLOW[:1] + (BITWISE[:2] if include_bitwise else [])
        # The canonical stream model gets the FULL end-to-end battery on device[0]
        # (evict on); the redundant axes (evict off, second device) get the fast
        # subset.  fresh / load / sparse_mm are proven L-inf-identical to the
        # stream model above, so each gets ONE fast correctness confirmation.
        d0 = devices[0]
        for device in devices:
            sm = stream.to(device)
            for evict in (True, False):
                cases = base if (device == d0 and evict) else fast
                ok, rows = battery(sm, Ls, cases, evict=evict, device=device)
                key = (f"divmod=F/{tag}", "dense_kernel", "stream", device,
                       "evict" if evict else "noevict")
                results[key] = (ok, rows)
                _print_row("divmod=F", tag, "dense_kernel", "stream", device,
                           evict, rows)
            sm.to("cpu")
        for kind, m, L, mode in [("fresh", fresh, Lf, "dense_kernel"),
                                 ("load", loaded, Ll, "dense_kernel"),
                                 ("stream_mm", stream_mm, Lmm, "sparse_mm")]:
            md = m.to(d0)
            ok, rows = battery(md, L, fast, evict=True, device=d0)
            results[(f"divmod=F/{tag}", mode, kind, d0, "evict")] = (ok, rows)
            _print_row("divmod=F", tag, mode, kind, d0, True, rows)
            md.to("cpu")

        if not include_bitwise:
            _, negrows = battery(stream, Ls, BITWISE, evict=True, device="cpu")
            nwrong = sum(1 for r in negrows if not r[3])
            print(f"  NOTE  divmod=F nobw: bitwise ops unsupported — "
                  f"{nwrong}/{len(negrows)} diverge from correct (as expected)",
                  flush=True)
        del fresh, stream, loaded, stream_mm

    # ================= DIVMOD family (stream-build + load-sparse) =========
    print("-" * 82, flush=True)
    dm_path = "/tmp/c4_matrix_divmod.pt"
    sp_dm, L_dm, st_dm = build_compact_sparse_streaming(
        code_size=64, include_bitwise=True, include_divmod=True,
        compute_mode="dense_kernel")
    save_sparse_transformer(sp_dm, L_dm, st_dm, dm_path)
    print(f"  [divmod stream-build: dim={st_dm.dim_after} nblocks={st_dm.n_blocks} "
          f"nnz={sp_dm.stats().total_nnz} peakRSS={_peak_gb():.1f}GB]", flush=True)
    loaded_dm, L_dm2 = load_sparse_transformer(dm_path, compute_mode="dense_kernel")
    dm_mm, L_dm3 = load_sparse_transformer(dm_path, compute_mode="sparse_mm")

    # byte-identity: stream vs load, dense_kernel vs sparse_mm
    for c in [_compile(s) for s in
              ["int main(){ return 720 / 6; }", "int main(){ return 84 % 5; }"]]:
        r_stream = _block_stack(sp_dm, L_dm, c)
        byteid[("divmod", "stream-vs-load")] = max(
            byteid.get(("divmod", "stream-vs-load"), 0.0),
            (r_stream - _block_stack(loaded_dm, L_dm2, c)).abs().max().item())
        byteid[("divmod", "dense_kernel-vs-sparse_mm")] = max(
            byteid.get(("divmod", "dense_kernel-vs-sparse_mm"), 0.0),
            (r_stream - _block_stack(dm_mm, L_dm3, c)).abs().max().item())

    dm_base = DIVMOD + ALU[:1] + BITWISE[:1]
    dm_fast = DIVMOD[:2]              # div + mod (fast) for the redundant axes
    d0 = devices[0]
    # canonical stream-built divmod model: full battery on d0/evict, subset else.
    for device in devices:
        sm = sp_dm.to(device)
        for evict in (True, False):
            cases = dm_base if (device == d0 and evict) else dm_fast
            ok, rows = battery(sm, L_dm, cases, evict=evict, device=device)
            key = ("divmod=T/bw", "dense_kernel", "stream", device,
                   "evict" if evict else "noevict")
            results[key] = (ok, rows)
            _print_row("divmod=T", "bw", "dense_kernel", "stream", device,
                       evict, rows)
        sm.to("cpu")
    # load-sparse (dense_kernel) + sparse_mm: one fast confirmation on device[0].
    for kind, m, L, mode in [("load", loaded_dm, L_dm2, "dense_kernel"),
                             ("load_mm", dm_mm, L_dm3, "sparse_mm")]:
        md = m.to(d0)
        ok, rows = battery(md, L, dm_fast, evict=True, device=d0)
        results[("divmod=T/bw", mode, kind, d0, "evict")] = (ok, rows)
        _print_row("divmod=T", "bw", mode, kind, d0, True, rows)
        md.to("cpu")

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
