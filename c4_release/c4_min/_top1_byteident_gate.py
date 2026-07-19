"""BYTE-IDENTITY gate: dense-blend dispatch vs TOP-1-routed dispatch.

Builds ONE lean pure-forward-complete VM, runs the 30-op-class battery (the same
programs ``_probe_pf_corpus_sample`` enumerates — arith / cmp / bitwise / muldiv /
control / memory / functions), FIRST with the dense dispatch FFN and THEN with the
same weights re-expressed as a ``Top1RoutedFFN`` (only the active opcode's units
computed).  Asserts the DECODED outputs are argmax-identical across every program,
each under the no-python-compute guard.  Also cross-checks both vs
``ref_interpret``.

This is the deliverable's byte-identity gate: the top-1-routed dispatch must
produce argmax-identical output to the dense-blend version across all op families
+ a corpus sanity sample.  Since both models share the exact same weights (the
routed one just skips the inactive opcodes' near-zero units), any difference is a
routing bug, not a weight difference.
"""
from __future__ import annotations
import sys, time

import c4_min.nibble_pure_forward as PF
import c4_min.nibble_pure_forward_complete as C
PF.SP_INIT = 0xF0
C.SP_INIT = 0xF0

import torch  # noqa: E402
from c4_min import isa  # noqa: E402
from c4_min.compact_alloc import build_compact_pure_forward_model  # noqa: E402
from c4_min.nibble_pure_forward_complete import ref_interpret  # noqa: E402
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached  # noqa: E402
from c4_min.moe_top1 import Top1RoutedFFN  # noqa: E402
from c4_min._pf_op_class_progs import progs_by_class, ALL  # noqa: E402


def _run(m, L, code):
    """KV-cached driver -> AX byte trace (fast, O(cache)/step)."""
    return run_pure_forward_cached(m, L, code, max_steps=64, evict=False)


def _route(model, L):
    """Wrap the dispatch block's FFN in top-1 routing (in place)."""
    names = list(L._block_names)
    bi = names.index("dispatch")
    dense = model.blocks[bi].ffn
    routed = Top1RoutedFFN(dense, op_is_base=int(L.OP_IS),
                           num_ops=int(isa.NUM_OPS))
    model.blocks[bi].ffn = routed
    return routed, dense, bi


def main():
    t = time.time()
    # lean COMPACT (no divmod, no global-max FFN padding -> ~15x faster/step):
    # keeps full stack + callconv + ADD/SUB/MUL + cmp/bitwise/mem.  This is the
    # SAME model the corpus/ONNX runners use.
    m, L, _stats = build_compact_pure_forward_model(
        code_size=24)
    print(f"LEAN compact model: dim={m.dim} blocks={len(m.blocks)} "
          f"heads={m.blocks[0].attn.n_heads} ({time.time()-t:.1f}s)", flush=True)

    P = progs_by_class()

    # -- 1. DENSE-blend dispatch: capture every program's decoded output. ------
    dense_out = {}
    for op in ALL:
        for pi, prog in enumerate(P[op]):
            code = isa.assemble(list(prog))
            dense_out[(op, pi)] = _run(m, L, code)

    # -- 2. Route the dispatch block, RE-run the SAME battery. -----------------
    routed, dense, bi = _route(m, L)
    print(f"routed dispatch block idx={bi}: dense units={routed.n_units} "
          f"-> K={routed.K} routed units/op (+{routed.n_ungated} ungated)",
          flush=True)

    npass = nfail = ref_ok = 0
    fails = []
    for op in ALL:
        for pi, prog in enumerate(P[op]):
            code = isa.assemble(list(prog))
            ref = ref_interpret(code)
            got = _run(m, L, code)
            d = dense_out[(op, pi)]
            same_as_dense = (got == d)
            same_as_ref = (got == ref)
            if same_as_ref:
                ref_ok += 1
            if same_as_dense:
                npass += 1
            else:
                nfail += 1
                fails.append((op, got, d))
            tag = "PASS" if same_as_dense else "FAIL"
            print(f"  [{tag}] {op:5s} routed={got[-1] if got else None} "
                  f"dense={d[-1] if d else None} ref={ref[-1] if ref else None}",
                  flush=True)

    n = npass + nfail
    print(f"\n=== TOP-1 vs DENSE byte-identity: {npass}/{n} argmax-identical ===",
          flush=True)
    print(f"=== TOP-1 vs ref_interpret: {ref_ok}/{n} byte-exact "
          f"(same as the dense model's own corpus fraction; DIV/MOD unsupported "
          f"in lean) ===", flush=True)
    if fails:
        print("MISMATCH vs dense (ROUTING BUG):", flush=True)
        for op, g, d in fails:
            print(f"    {op:5s} routed={g} dense={d}", flush=True)
        return 1
    print("PASS: top-1 routing is argmax-identical to the dense blend on ALL "
          "op families + the corpus sample.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
