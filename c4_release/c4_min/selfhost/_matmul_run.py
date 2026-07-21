"""Run the fixed-point dot-product / MatMul kernel through the ACTUAL c4_min
``model.forward``, byte-exact vs the reference interpreter / numpy — the
ONE-LAYER SELF-EMULATION demo (a piece of a transformer layer, computed by the
transformer itself).

BOUNDED SCOPE (honest): this is the SMALLEST genuine self-hosting compute run
THROUGH the neural forward — a fixed-point dot product (the atomic matmul op) over
a slice of the model's OWN weights.  It is NOT full self-hosting / self-emulation;
the full self-forward is the ~2.4M-VM-step / ~88-day wall in
``docs/SELFHOST_3LAYER_FEASIBILITY.md`` and
``docs/ONE_LAYER_SELF_EMULATION_2026_07_20.md``.

The model is built ONCE: the SAME streaming ``model.forward`` both (a) supplies the
weight-row operand (a slice of its own embedding matrix) and (b) executes the
compiled c4 bytecode that multiplies it — so it genuinely "emulates itself".

    OMP_NUM_THREADS=4 python -m c4_min.selfhost._matmul_run [--device=cuda:0] \
        [--kind=dot|matmul] [--fixed]
"""
from __future__ import annotations
import os
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

from c4_min import isa
from c4_min.selfhost._matmul_src import (
    dot_c, dot_reference, matvec_c, matvec_reference,
    matmul_c, matmul_reference, SCALE)


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _own_weight_row(sparse):
    """Find a length-2 row of small non-negative integers in the model's OWN
    embedding weight matrix (the nibble/value one-hot encodings, 0..15) — so the dot
    product is genuinely a weight-row of the transformer itself.  Returns
    ``(w, provenance)``."""
    w = sparse.embed.detach().float()
    for r0 in range(w.shape[0]):
        for c0 in range(w.shape[1] - 1):
            vals = [int(round(float(w[r0, c0]))), int(round(float(w[r0, c0 + 1])))]
            if all(1 <= v <= 6 for v in vals) and len(set(vals)) >= 1 and sum(vals) >= 2:
                prov = (f"c4_min embedding weight row slice "
                        f"embed[{r0}, {c0}:{c0 + 2}] = {vals} "
                        f"(model's OWN weights, integer-valued, quantized to "
                        f"fixed-point scale {SCALE})")
                return vals, prov
    return [1, 2], "FIXED fallback row [1, 2] (no clean own-weights row found)"


def _own_weight_block(sparse):
    """Find a 2x2 block of small non-negative integers in the OWN embedding matrix."""
    w = sparse.embed.detach().float()
    for r0 in range(w.shape[0] - 1):
        for c0 in range(w.shape[1] - 1):
            blk = w[r0:r0 + 2, c0:c0 + 2]
            vals = [int(round(float(x))) for x in blk.flatten().tolist()]
            if all(1 <= v <= 6 for v in vals) and len(set(vals)) >= 2:
                A = [[vals[0], vals[1]], [vals[2], vals[3]]]
                prov = (f"c4_min embedding weight block embed[{r0}:{r0 + 2}, "
                        f"{c0}:{c0 + 2}] = {A} (model's OWN weights, "
                        f"quantized to fixed-point scale {SCALE})")
                return A, prov
    A = [[1, 1], [2, 1]]
    return A, f"FIXED fallback block {A} (no clean own-weights block found)"


def run(kind="dot", device="cpu", verbose=False, use_own_weights=True):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import ref_interpret
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    # ---- build the streaming/lean model ONCE (KV-cached, eviction ON) --------
    # This one model both SUPPLIES the own-weights operand and EXECUTES the
    # bytecode that multiplies it.
    print(f"kind={kind}  RSS before build: {_rss_gb():.2f} GB", flush=True)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(
        code_size=256, recurrent_divmod=True, addr32=True)
    t_build = time.time() - t0
    print(f"build wall: {t_build:.1f}s  RSS after build: {_rss_gb():.2f} GB", flush=True)

    # ---- operands from the model's OWN weights (self-emulation) --------------
    if kind == "dot":
        x = [2, 3]
        if use_own_weights:
            w, prov = _own_weight_row(sparse)
        else:
            w, prov = [1, 2], "FIXED weight row [1, 2]"
        print(f"weight row w: {prov}")
        print(f"input vector x: FIXED {x}  (fixed-point scale = {SCALE})")
        C_ref = dot_reference(w, x, SCALE)
        src = dot_c(w, x, SCALE)
        op_desc = f"dot product w . x = {w} . {x}"
        operands = dict(w=w, x=x)
    elif kind == "matvec":
        x = [2, 3]
        if use_own_weights:
            A, prov = _own_weight_block(sparse)
        else:
            A, prov = [[1, 1], [2, 1]], "FIXED block [[1,1],[2,1]]"
        print(f"weight matrix A: {prov}")
        print(f"input vector x: FIXED {x}  (fixed-point scale = {SCALE})")
        C_ref = matvec_reference(A, x, SCALE)
        src = matvec_c(A, x, SCALE)
        op_desc = f"matrix-vector A @ x = {A} @ {x}"
        operands = dict(A=A, x=x)
    elif kind == "matmul":
        B = [[1, 2], [3, 1]]
        if use_own_weights:
            A, prov = _own_weight_block(sparse)
        else:
            A, prov = [[1, 1], [2, 1]], "FIXED block [[1,1],[2,1]]"
        print(f"operand A: {prov}")
        print(f"operand B: FIXED {B}  (fixed-point scale = {SCALE})")
        C_ref = matmul_reference(A, B, SCALE)
        src = matmul_c(A, B, SCALE)
        op_desc = f"2x2 matmul A @ B = {A} @ {B}"
        operands = dict(A=A, B=B)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    bytecode, data = compile_c(src)                # the REAL c4_min C compiler
    code = bytecode_to_isa(bytecode)
    big = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not big, f"IMM>255 present (would diverge from ref): {big}"
    assert len(code) + 2 <= 256, f"kernel {len(code)} instrs exceeds code_size 256"

    ref_out = []
    ref_tr = ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF, out=ref_out)
    n_ref_steps = len(ref_tr)
    assert ref_out == C_ref, f"reference VM {ref_out} != numpy {C_ref}"
    wrapped = [v for v in ref_tr if v >= 2 ** 31]
    assert not wrapped, f"{len(wrapped)} wrapped-negative AX values"

    print(f"{op_desc}: n_instrs={len(code)} ref_steps={n_ref_steps}")
    print(f"numpy/reference fixed-point result bytes: {C_ref}", flush=True)

    if device and device != "cpu":
        sparse = sparse.to(device)
        print(f"moved model to {device}", flush=True)

    # ---- run each VM instruction as one model.forward -----------------------
    out, stats = [], {}
    print(f"running {n_ref_steps} steps through model.forward ...", flush=True)
    t0 = time.time()
    run_pure_forward_cached(sparse, L, code, max_steps=n_ref_steps + 6,
                            mask=0xFFFFFFFF, evict=True, prune_interval=60,
                            out=out, stats=stats, data_seg=data, verbose=verbose)
    t_run = time.time() - t0
    n = stats.get("steps", 0)

    print(f"run wall: {t_run:.1f}s  neural_steps={n}  "
          f"per-step={t_run / max(1, n):.2f}s  peak RSS={_rss_gb():.2f} GB")
    print(f"max_seq={stats.get('max_seq_len')} max_cache={stats.get('max_cache_size')} "
          f"evicted={stats.get('total_evicted')}")
    print(f"--- reference (numpy fixed-point) result bytes: {C_ref}")
    print(f"--- neural model.forward PRTF result bytes:     {out}")
    match = out == C_ref == ref_out
    print(f"BYTE-EXACT MATCH: {match}  (neural == numpy == reference-VM)")
    info = dict(kind=kind, provenance=prov, n_instrs=len(code),
                ref_steps=n_ref_steps, neural_steps=n, build_s=t_build,
                run_s=t_run, rss_gb=_rss_gb(), device=device, stats=stats)
    info.update(operands)
    return match, out, C_ref, info


if __name__ == "__main__":
    DEV = "cpu"
    KIND = "dot"
    VERB = "-v" in sys.argv or "--verbose" in sys.argv
    USE_OWN = "--fixed" not in sys.argv
    for a in sys.argv:
        if a.startswith("--device="):
            DEV = a.split("=", 1)[1]
        if a.startswith("--kind="):
            KIND = a.split("=", 1)[1]
    ok, *_ = run(kind=KIND, device=DEV, verbose=VERB, use_own_weights=USE_OWN)
    sys.exit(0 if ok else 1)
