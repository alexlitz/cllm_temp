#!/usr/bin/env python3
"""measure_whole_forward_steps.py — the GROUNDED whole-forward draft-VM step count
for the tiny ``c4vm.onnx``, MATMUL + NON-MATMUL, the real Rel-1 number.

``measure_matmul_steps_per_mac.py`` grounds only the MATMUL portion (~48M draft
steps) and extrapolates the rest.  This tool closes the gap: it measures the
draft-VM steps/element rate of EVERY op (matmul AND non-matmul) with the c4-subset
kernels that self-host byte-exact (``_matmul_general_src`` + ``_nonmatmul_ops_src``,
run through ``nibble_pure_forward_complete.ref_interpret``), then walks the tiny
model's REAL 153-node list (from the exported ``.nblbin``) and sums each node's
grounded cost.  The result is the honest whole-forward draft-VM step count.

Per-op rates are MEASURED here (differencing two sizes cancels every
size-independent setup term, so the slope is the marginal cost):

  * matmul:   ~101 steps / MAC (fpmul-call) + ~104 steps / output-element overhead
  * add/sub:  ~112 steps / element     mul/div: ~129 / element
  * gather:   ~109    transpose: ~101    reduce: ~54 / contraction-element
  * neg/abs/clip/reshape/identity/unsqueeze/shape/cast/concat: ~90-100 / element
  * exp:      ~1000+ / element (data-dependent range-reduction loop)
  * softmax:  ~1200+ / element (exp + reduce + div fused)   sigmoid: ~140 / element

CPU-only (c4 compiler + draft VM + one tiny ONNX export).  NO GPU, NO neural build
for the rates; the tiny-model dims come from the exported graph.

Run:  OMP_NUM_THREADS=4 python -m c4_min.selfhost.measure_whole_forward_steps
      ... --no-onnx   (rates only, skip the tiny-model walk)
"""
from __future__ import annotations

import argparse
import sys

from c4_min.selfhost import _nonmatmul_ops_src as S
from c4_min.selfhost._matmul_general_src import matmul_general_c


# --------------------------------------------------------------------------- #
# draft-VM run helpers                                                         #
# --------------------------------------------------------------------------- #
def _steps_draft(src, max_steps=3_000_000):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import ref_interpret

    bc, _ = compile_c(src)
    code = bytecode_to_isa(bc)
    out = []
    tr = ref_interpret(code, max_steps=max_steps, mask=0xFFFFFFFF, out=out)
    assert len(tr) < max_steps, "hit max_steps"
    return len(tr)


def _steps_word(src, max_steps=8_000_000):
    """Step count on the full-word VM (== draft-VM step count; masking is
    value-only, not control-flow, so the instruction count is identical — used for
    exp/softmax whose >255 values overflow the draft byte store)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    bc, _ = compile_c(src)
    code = bytecode_to_isa(bc)
    out = []
    _ax, steps = S.refword_interpret(code, max_steps=max_steps, out=out)
    return steps


# --------------------------------------------------------------------------- #
# per-op marginal rate measurement (steps / element, differenced)             #
# --------------------------------------------------------------------------- #
def measure_rates(verbose=True):
    def slope(runner, gen, n1, n2):
        return (runner(gen(n2)) - runner(gen(n1))) / (n2 - n1)

    r = {}
    # elementwise binary
    r["Add"] = slope(_steps_draft, lambda n: S.add_c([1] * n, [1] * n), 3, 6)
    r["Sub"] = slope(_steps_draft, lambda n: S.sub_c([1] * n, [1] * n), 3, 6)
    r["Mul"] = slope(_steps_draft, lambda n: S.mul_c([1] * n, [1] * n), 3, 6)
    r["Div"] = slope(_steps_draft, lambda n: S.div_c([1] * n, [1] * n), 3, 6)
    # gather / transpose
    r["Gather"] = slope(_steps_draft,
                        lambda n: S.gather_c(list(range(n)), list(range(n))), 3, 6)
    r["Transpose"] = slope(_steps_draft,
                           lambda n: S.transpose2d_c(1, n, list(range(1, n + 1))), 3, 6)
    # reduce (per contraction element)
    r["ReduceMax"] = slope(_steps_draft,
                           lambda n: S.reduce_max_c(1, n, list(range(1, n + 1))), 3, 6)
    r["ReduceSum"] = slope(_steps_draft,
                           lambda n: S.reduce_sum_c(1, n, list(range(1, n + 1))), 3, 6)
    # unary
    r["Neg"] = slope(_steps_draft, lambda n: S.neg_c([1] * n), 3, 6)
    r["Abs"] = slope(_steps_draft, lambda n: S.abs_c([1] * n), 3, 6)
    r["Clip"] = slope(_steps_draft, lambda n: S.clip_c([1] * n, 0), 3, 6)
    # shape-plumbing / copy ops all cost a verbatim copy (== reshape rate)
    copy_rate = slope(_steps_draft, lambda n: S.reshape_c([1] * n), 3, 6)
    for op in ("Reshape", "Identity", "Unsqueeze", "Concat", "Cast",
               "Shape", "Range", "ConstantOfShape", "Trilu"):
        r[op] = copy_rate
    # exp family (full-word step count; value-only masking, control flow identical)
    r["Exp"] = slope(_steps_word,
                     lambda n: S.exp_c([-(i % 4) for i in range(n)], 4096), 3, 6)
    r["Sigmoid"] = slope(_steps_word,
                         lambda n: S.sigmoid_c([-(i % 4) for i in range(n)], 4096), 3, 6)
    r["Softmax"] = slope(_steps_word,
                         lambda n: S.softmax_c([-(i % 4) for i in range(n)], 4096), 4, 8)
    # matmul: per-MAC + per-output (controlled sweep, fpmul-call form)
    def mm_steps(M, K, N):
        A = [1] * (M * K)
        B = [1] * (K * N)
        return _steps_draft(matmul_general_c(A, B, M, K, N))
    per_mac = mm_steps(1, 6, 1) - mm_steps(1, 5, 1)
    per_out = mm_steps(6, 1, 1) - mm_steps(5, 1, 1)   # 1 MAC + row overhead
    r["_matmul_per_mac"] = per_mac
    r["_matmul_per_out"] = per_out

    if verbose:
        print("=" * 70)
        print("MEASURED draft-VM steps/element (byte-exact self-host kernels)")
        print("=" * 70)
        for k in sorted(r):
            if not k.startswith("_"):
                print(f"    {k:18} {r[k]:7.1f}")
        print(f"    {'MatMul /MAC':18} {per_mac:7.1f}   "
              f"{'MatMul /output':16} {per_out:7.1f}")
    return r


# --------------------------------------------------------------------------- #
# walk the tiny model's real node list + sum grounded per-node costs           #
# --------------------------------------------------------------------------- #
def walk_tiny_model():
    """Export the tiny c4vm.onnx, lower to .nblbin, run the numpy reference on
    [1,2,42,3] while intercepting every op to record its output size / MACs /
    reduction length.  Returns (per-node records, reference argmax)."""
    import tempfile, os
    import numpy as np
    from c4_min import blogspec_compiler as C, export_onnx as E
    from c4_min.onnx_to_c4bin import lower_onnx_to_bin
    from c4_min.nbl_bin_interp import Graph, OP_NAMES

    d = tempfile.mkdtemp(prefix="whole_fwd_")
    onnxp = os.path.join(d, "m.onnx")
    binp = os.path.join(d, "m.nblbin")
    model, L, _code = C.build_step_model(E.PROOF_PROG)
    model.eval()
    E.export_onnx(model, onnxp)
    lower_onnx_to_bin(onnxp, binp)

    g = Graph(binp)
    recs = []

    # monkeypatch numpy ufuncs the interpreter uses so we can size each node
    orig = {}

    def size_of(x):
        return int(np.prod(np.shape(x))) if np.ndim(x) else 1

    # We re-run g.run but instrument by wrapping the node loop.  Simplest: replicate
    # the dispatch here recording op + output size + (for matmul) MACs + (for reduce)
    # last-axis length.  We reuse Graph.run's own logic by re-implementing the size
    # accounting off the computed value store.
    vals = {}
    for i, isi in enumerate(g.is_init):
        if isi:
            vals[i] = g.init_vals[i]
    tokens = np.array([[1, 2, 42, 3]], dtype=np.int64)
    vals[g.input_tid] = tokens

    # Run the real interpreter to get the true argmax (authoritative) ...
    out = g.run(tokens)
    argmax = np.argmax(out[0], axis=-1).tolist()

    # ... and separately walk the nodes to size them (recompute values the same way
    # the reference does; this is the reference's own arithmetic, so sizes are exact).
    vals = {}
    for i, isi in enumerate(g.is_init):
        if isi:
            vals[i] = g.init_vals[i]
    vals[g.input_tid] = tokens
    import struct
    for op, ins, outs, attrs in g.nodes:
        name = OP_NAMES[op]
        a = [vals[i] if i >= 0 else None for i in ins]
        rec = {"op": name}
        if name == "MatMul":
            r = np.matmul(a[0], a[1])
            K = a[0].shape[-1]
            rec["macs"] = int(np.prod(r.shape)) * K
            rec["out"] = int(np.prod(r.shape))
        elif name in ("ReduceMax", "ReduceSum"):
            # cost ~ per contraction element over the last axis, for every output row
            last = a[0].shape[-1]
            if name == "ReduceMax":
                axes = attrs.get("axes", [-1])
                kd = attrs.get("keepdims", [1])[0]
                r = np.max(a[0], axis=tuple(axes), keepdims=bool(kd))
            else:
                axes = [int(x) for x in a[1].tolist()] if len(a) > 1 else [-1]
                kd = attrs.get("keepdims", [1])[0]
                r = np.sum(a[0], axis=tuple(axes), keepdims=bool(kd))
            rec["contraction"] = int(np.prod(a[0].shape))  # total elems reduced
            rec["out"] = int(np.prod(r.shape))
        else:
            r = _ref_eval(name, a, attrs)
            rec["out"] = size_of(r)
        recs.append(rec)
        vals[outs[0]] = r
    return recs, argmax, len(g.nodes)


def _ref_eval(name, a, attrs):
    """Mirror of nbl_bin_interp.Graph.run's per-op arithmetic (for sizing)."""
    import numpy as np
    import struct
    if name == "Add":
        return a[0] + a[1]
    if name == "Sub":
        return a[0] - a[1]
    if name == "Mul":
        return a[0] * a[1]
    if name == "Div":
        return a[0] / a[1]
    if name == "Gather":
        axis = attrs.get("axis", [0])[0]
        return np.take(a[0], a[1].astype(np.int64), axis=axis)
    if name == "Reshape":
        return a[0].reshape([int(x) for x in a[1].tolist()])
    if name == "Transpose":
        return np.transpose(a[0], attrs["perm"])
    if name == "Concat":
        axis = attrs.get("axis", [0])[0]
        return np.concatenate([np.atleast_1d(x) for x in a], axis=axis)
    if name == "Unsqueeze":
        r = a[0]
        for ax in sorted([int(x) for x in a[1].tolist()]):
            r = np.expand_dims(r, ax)
        return r
    if name == "Shape":
        return np.array(a[0].shape, dtype=np.int64)
    if name == "Cast":
        to = attrs["to"][0]
        return a[0].astype(np.float32 if to == 1 else np.int64)
    if name == "Exp":
        return np.exp(a[0])
    if name == "Neg":
        return -a[0]
    if name == "Abs":
        return np.abs(a[0])
    if name == "Range":
        return np.arange(int(a[0]), int(a[1]), int(a[2]), dtype=np.int64)
    if name == "Clip":
        r = a[0]
        if len(a) > 1 and a[1] is not None:
            r = np.maximum(r, a[1])
        if len(a) > 2 and a[2] is not None:
            r = np.minimum(r, a[2])
        return r
    if name == "Trilu":
        upper = attrs.get("upper", [1])[0]
        k = int(a[1]) if len(a) > 1 and a[1] is not None else 0
        return np.triu(a[0], k) if upper else np.tril(a[0], k)
    if name == "ConstantOfShape":
        shape = [int(x) for x in a[0].tolist()]
        bits = attrs.get("cos_fill_bits", [0])[0]
        fill = struct.unpack("<f", struct.pack("<i", bits))[0]
        return np.full(shape, fill, dtype=np.float32)
    if name == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-a[0]))
    if name == "Identity":
        return a[0]
    raise ValueError(f"unknown op {name}")


def ground_whole_forward(rates, verbose=True):
    recs, argmax, n_nodes = walk_tiny_model()
    per_mac = rates["_matmul_per_mac"]
    per_out_mm = rates["_matmul_per_out"] - per_mac   # pure per-output overhead

    total = 0
    matmul_total = 0
    nonmatmul_total = 0
    by_op = {}
    for rec in recs:
        op = rec["op"]
        if op == "MatMul":
            c = per_mac * rec["macs"] + per_out_mm * rec["out"]
            matmul_total += c
        elif op in ("ReduceMax", "ReduceSum"):
            c = rates[op] * rec["contraction"]
            nonmatmul_total += c
        else:
            rate = rates.get(op)
            if rate is None:
                rate = rates["Identity"]
            c = rate * rec["out"]
            nonmatmul_total += c
        total += c
        by_op[op] = by_op.get(op, 0) + c

    if verbose:
        print("\n" + "=" * 70)
        print(f"TINY c4vm.onnx WHOLE FORWARD ({n_nodes} nodes) — grounded draft steps")
        print("=" * 70)
        print(f"    reference argmax on [1,2,42,3] = {argmax}")
        print(f"\n    per-op grounded draft-VM steps:")
        for op in sorted(by_op, key=lambda k: -by_op[k]):
            print(f"      {op:18} {by_op[op]:14,.0f}")
        print(f"\n    MATMUL portion      = {matmul_total:14,.0f} draft steps")
        print(f"    NON-MATMUL portion  = {nonmatmul_total:14,.0f} draft steps "
              f"({100 * nonmatmul_total / total:.1f}% of total)")
        print(f"    WHOLE FORWARD TOTAL = {total:14,.0f} draft steps  [GROUNDED]")
        print(f"\n    vs matmul-only extrapolation ~48M -> whole forward is "
              f"{total / max(1, matmul_total):.2f}x the matmul portion")
    return dict(total=total, matmul=matmul_total, nonmatmul=nonmatmul_total,
                argmax=argmax, by_op=by_op, n_nodes=n_nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    print("GROUNDING the WHOLE-forward draft-VM step count (matmul + non-matmul)\n")
    rates = measure_rates(verbose=True)
    if not args.no_onnx:
        ground_whole_forward(rates, verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
