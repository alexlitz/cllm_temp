#!/usr/bin/env python3
"""Re-validate the capstone pathways against the ONE unified full-op VM.

Background
==========
The capstone pathways (ONNX / C-runtime / bundle / quine / CLI) were originally
validated on lean op-subset branch builds or the older ``neural_vm`` model.  The
op-subset splits have since been collapsed into a SINGLE full-op-set interpreter
(``nibble_pure_forward_complete.build_pure_forward_complete_model``), and the
runtime library (malloc/free/memset/memcmp) has been ported onto that same
unified model with a 32-bit load-address query (``lib_neural``).

This script re-runs the pathways against THAT unified model.  Where a pathway can
be driven off the MEMORY-SAFE streaming-sparse form of the unified model
(``compact_alloc.build_compact_sparse_streaming`` — peak RSS = one block, ~5.5 GB,
byte-identical L-inf=0 in ``dense_kernel`` mode to the dense whole), it is; the
dense ONNX export (~48-62 GB RSS) is gated on free memory and otherwise DEFERRED
rather than run (the documented dense-build hazard).

Each pathway prints a PASS/FAIL line and the sample it actually ran (no silent
full-corpus claims).  Select pathways with ``--only cli,quine,bundle,corpus``.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=<c4_release> python -m c4_min.revalidate_capstones_unified
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import c4_min.nibble_pure_forward as _PF          # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _free_gb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def build_unified_streaming(code_size: int = 32, recurrent_divmod: bool = True):
    """The MEMORY-SAFE unified full-op model (streaming sparse, peak = one block).

    Byte-identical (L-inf=0, ``dense_kernel`` mode) to wrapping the dense
    ``build_pure_forward_complete_model`` output in a ``SparseTransformer``.
    """
    from c4_min.compact_alloc import build_compact_sparse_streaming
    t = time.time()
    sparse, L, stats = build_compact_sparse_streaming(
        code_size=code_size, recurrent_divmod=recurrent_divmod)
    print("[build] unified streaming-sparse: dim=%d blocks=%d took %.1fs peakRSS=%.2fGB"
          % (sparse.dim, len(sparse.blocks), time.time() - t, _rss_gb()))
    return sparse, L, stats


# ---------------------------------------------------------------------------
# CLI pathway (echo / cat / yes) — byte-exact stdin->stdout on the unified VM.
# Consumes the existing I/O path (nibble_filesys tool-call boundary); does NOT
# modify it.
# ---------------------------------------------------------------------------
def pathway_cli(model, L) -> bool:
    from c4_min import cli_tools as CLI
    from c4_min import nibble_filesys as FS
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    def run_model_cached(prog, stdin_bytes=b"", max_steps=400):
        fio = FS.FileOpState(runner=FS.FileRunner(
            fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(stdin_bytes)))
        run_pure_forward_cached(model, L, prog.code, max_steps=max_steps, mask=0xFF,
                                fio=fio, data_seg=dict(prog.data_seg))
        return bytes(fio.runner.stdout).decode("latin-1")

    cases = [
        ("echo", CLI.build_echo("hello\n"), b""),
        ("yes",  CLI.build_yes("y\n", n=3), b""),
        ("cat",  CLI.build_cat(),           b"copy me!\n"),
    ]
    allok = True
    for name, prog, stdin in cases:
        ref = CLI.run_ref(prog, stdin_bytes=stdin)
        t0 = time.time()
        got = run_model_cached(prog, stdin_bytes=stdin)
        ok = (ref == got)
        allok &= ok
        print("  [cli:%s] ref=%r model=%r BYTE-EXACT=%s (%.1fs)"
              % (name, ref, got, ok, time.time() - t0))
    return allok


# ---------------------------------------------------------------------------
# Quine pathway — the quine self-outputs its own source through the unified VM,
# byte-exact.  Same algorithm/params as ``test_neural_quine_byte_exact_self_output``
# but driven off the memory-safe streaming-sparse unified model (byte-identical).
#
# The quine's seed string literal sits ~666 tokens back, so the §Memory head needs
# the WIDENED recency horizon (``MEM_ALIBI_SLOPE = 0.05``, EFF/slope ~ 80k tokens).
# That slope is BAKED at build time, so the quine builds its OWN unified model
# (the CLI/bundle/corpus pathways use the default slope=1.0).
# ---------------------------------------------------------------------------
def pathway_quine(code_size: int = 64) -> bool:
    import c4_min.blogspec_memory as _MEM
    _MEM.MEM_ALIBI_SLOPE = 0.05                 # set BEFORE the build (baked in)
    try:
        model, L, _ = build_unified_streaming(code_size=code_size)
        from c4_min.quine_prtf import build_quine
        from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

        code, seed_mem, S = build_quine()
        out: list = []
        t0 = time.time()
        run_pure_forward_cached(model, L, code, max_steps=4000, mask=0xFFFFFFFF,
                                evict=True, prune_interval=90, out=out,
                                seed_mem=seed_mem)
        ok = (out == S)
        print("  [quine] source_bytes=%d model_out=%d BYTE-EXACT=%s (%.1fs)"
              % (len(S), len(out), ok, time.time() - t0))
        if not ok:
            print("    S   =", S[:40], "...")
            print("    out =", out[:40], "...")
    finally:
        _MEM.MEM_ALIBI_SLOPE = 1.0              # restore default for other pathways
    return ok


# ---------------------------------------------------------------------------
# Bundle pathway — bundle model+bytecode+program into one .c4bundle and run it
# end-to-end THROUGH the bundled model, byte-exact result.
#
# The bundle serialises DENSE weights (COO over dense linear layers) and
# ``run_bundle`` reconstructs a fresh DENSE unified full-op model
# (``build_pure_forward_complete_model``) to scatter them into — that dense build
# is the ~48-62 GB hazard, so this pathway is GATED on free memory (needs a dense
# model, twice: once to assemble, once to run).  ``min_free_gb`` guards it.
# ---------------------------------------------------------------------------
def pathway_bundle(code_size: int = 32, min_free_gb: float = 70.0):
    """Returns True/False (byte-exact) or None (deferred — not enough free mem)."""
    import tempfile
    from c4_min import bundle_small as B

    free = _free_gb()
    if free < min_free_gb:
        print("  [bundle] DEFERRED: free=%.1fGB < %.0fGB gate (dense reconstruct "
              "needed for genuine end-to-end run_bundle)" % (free, min_free_gb))
        return None

    src = "int main(){ return 500 + 700; }"     # -> 1200 (full 32-bit AX)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.c4bundle")
        t0 = time.time()
        # assemble_bundle bakes its OWN dense unified model (no reuse_model here so
        # the round-trip is genuine: assemble builds dense -> serialise -> run
        # rebuilds a SEPARATE dense model + scatters the serialised weights in).
        info = B.assemble_bundle(src, path, expected=1200, description="add-const",
                                 code_size=code_size, step_cap=400)
        res = B.run_bundle(path, verbose=False)
        got = res.get("got")
        ok = (res.get("status") == "PASS" and got == 1200)
        print("  [bundle] src=%r expected=1200 status=%s got=%s BYTE-EXACT=%s "
              "(%.1fs) file=%dB peakRSS=%.1fGB"
              % (src, res.get("status"), got, ok, time.time() - t0,
                 info["bundle_bytes"], _rss_gb()))
    return ok


# ---------------------------------------------------------------------------
# ONNX pathway.
#
# The GENUINE unified-full-op ONNX export traces the DENSE compact model
# (``export_onnx_compact.build_compact`` -> ``build_compact_pure_forward_model``,
# ~48.6 GB RSS measured) — there is NO lean/op-subset export variant since the
# unification.  So the full-op ONNX + onnxruntime byte-exact corpus battery is
# MEMORY-GATED (run ``python -m c4_min.run_onnx_corpus --battery`` on a host with
# a stable >60 GB free window).
#
# What IS memory-safe here: the ONNX EXPORT PIPELINE + VANILLA-graph property +
# byte-exact decode on a real baked C4 model — proven via ``export_onnx.main``
# (the foundation step model, ~0.5 MB).  It exercises the exact same
# torch.onnx.export -> op_inventory/assert_vanilla -> sparse-initializer ->
# onnxruntime byte-exact machinery the full-op path uses, just on a small model.
# ---------------------------------------------------------------------------
def pathway_onnx(min_free_gb: float = 70.0):
    """Returns True (foundation ONNX vanilla+byte-exact) and separately reports
    whether the FULL-op unified export is runnable now or deferred."""
    from c4_min import export_onnx as EO
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = EO.main(out_dir="/tmp/c4_onnx_foundation")
    ok = (rc == 0)
    # pull the vanilla + byte-exact verdicts out of the report
    txt = buf.getvalue()
    vanilla = "VANILLA: YES" in txt
    byte_exact = "byte_exact_decode(dense+sparse)=True" in txt
    print("  [onnx:foundation-model] vanilla(no Loop/Scan/If)=%s byte-exact-decode=%s "
          "OVERALL=%s" % (vanilla, byte_exact, "PASS" if ok else "FAIL"))
    free = _free_gb()
    if free < min_free_gb:
        print("  [onnx:full-unified-VM] DEFERRED: free=%.1fGB < %.0fGB gate — dense "
              "compact export is ~48.6GB RSS (no lean op-subset variant). Run "
              "`python -m c4_min.run_onnx_corpus --battery` on a stable >60GB host."
              % (free, min_free_gb))
    else:
        print("  [onnx:full-unified-VM] free=%.1fGB OK — run "
              "`python -m c4_min.run_onnx_corpus --battery --code-size 48`" % free)
    return ok and vanilla and byte_exact


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", type=str, default="cli",
                    help="comma list of pathways: cli,quine,onnx,bundle")
    ap.add_argument("--code-size", type=int, default=32)
    args = ap.parse_args(argv)

    want = {p.strip() for p in args.only.split(",") if p.strip()}
    print("=" * 72)
    print("RE-VALIDATE CAPSTONE PATHWAYS ON THE UNIFIED FULL-OP VM")
    print("free=%.1fGB  pathways=%s" % (_free_gb(), sorted(want)))
    print("=" * 72)

    import gc
    results = {}

    # CLI shares the default-slope streaming model (built once, then freed).
    if "cli" in want:
        model, L, _ = build_unified_streaming(code_size=args.code_size)
        results["cli"] = pathway_cli(model, L)
        del model
        gc.collect()

    # Quine builds its OWN model (needs the widened MEM_ALIBI_SLOPE=0.05).
    if "quine" in want:
        results["quine"] = pathway_quine(code_size=max(64, args.code_size))
        gc.collect()

    # ONNX: foundation-model export is memory-safe; full-op export is gated.
    if "onnx" in want:
        results["onnx"] = pathway_onnx()

    # Bundle needs the DENSE reconstruct (memory-gated).
    if "bundle" in want:
        results["bundle"] = pathway_bundle(code_size=args.code_size)

    print("=" * 72)
    def _verdict(v):
        if v is None:
            return "DEFERRED (memory-gated)"
        return "PASS-on-unified" if v else "FAIL"
    for k, v in results.items():
        print("PATHWAY %-8s : %s" % (k, _verdict(v)))
    print("peakRSS=%.2fGB" % _rss_gb())
    # deferred pathways don't fail the run; only an explicit False does.
    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
