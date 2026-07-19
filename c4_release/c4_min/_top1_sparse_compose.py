"""COMPOSITION probe: top-1 routing (expert sparsity) + sparse_mm (weight
sparsity), together, byte-identical.

Builds the compact VM, routes the dispatch block (``Top1RoutedFFN``), then wraps
the whole model in ``SparseTransformer`` with ``compute_mode='sparse_mm'`` — the
routed dispatch block is kept AS-IS (it already computes only the active opcode's
rows; sparsifying its padded weights would break the router and buy nothing),
while every OTHER block's weights go through the sparse (nnz-proportional)
mat-mul.  Runs the op-family battery through:

    plain dense  |  routed-only  |  routed + sparse_mm

and asserts all three are argmax-identical (the routing shrinks the row set; the
sparse_mm shrinks per-row cost on the rest of the stack — orthogonal, composable).
"""
from __future__ import annotations
import sys, time

import c4_min.nibble_pure_forward as PF
import c4_min.nibble_pure_forward_complete as C
PF.SP_INIT = 0xF0
C.SP_INIT = 0xF0

from c4_min import isa  # noqa: E402
from c4_min.compact_alloc import build_compact_pure_forward_model  # noqa: E402
from c4_min.nibble_pure_forward_complete import ref_interpret  # noqa: E402
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached  # noqa: E402
from c4_min.moe_top1 import Top1RoutedFFN  # noqa: E402
from c4_min.sparse_forward import SparseTransformer  # noqa: E402
from c4_min._probe_pf_corpus_sample import progs_by_class, ALL  # noqa: E402


def _run(m, L, code):
    return run_pure_forward_cached(m, L, code, max_steps=48, evict=False)


def main():
    t = time.time()
    m, L, _ = build_compact_pure_forward_model(
        code_size=24, include_bitwise=True, include_divmod=False)
    print(f"compact model ({time.time()-t:.1f}s): dim={m.dim} blocks={len(m.blocks)}")
    P = progs_by_class()

    # -- 1. plain dense (baseline) --------------------------------------------
    dense_out = {op: _run(m, L, isa.assemble(list(P[op][0]))) for op in ALL}

    # -- 2. route the dispatch block ------------------------------------------
    names = list(L._block_names)
    bi = names.index("dispatch")
    routed = Top1RoutedFFN(m.blocks[bi].ffn, op_is_base=int(L.OP_IS),
                           num_ops=int(isa.NUM_OPS))
    m.blocks[bi].ffn = routed
    routed_out = {op: _run(m, L, isa.assemble(list(P[op][0]))) for op in ALL}

    # -- 3. routed + sparse_mm on all OTHER blocks ----------------------------
    t = time.time()
    sm = SparseTransformer(m, density_thresh=0.25, min_numel=4096,
                           compute_mode="sparse_mm")
    st = sm.stats()
    print(f"SparseTransformer built ({time.time()-t:.1f}s): "
          f"{st.n_sparsified} weights sparsified, "
          f"{st.dense_gb:.3f}GB dense -> {st.sparse_mb:.1f}MB sparse; "
          f"dispatch block kept routed (expert-sparse).")
    both_out = {op: _run(sm, L, isa.assemble(list(P[op][0]))) for op in ALL}

    npass = n = 0
    fails = []
    for op in ALL:
        n += 1
        d, r, b = dense_out[op], routed_out[op], both_out[op]
        ok = (r == d) and (b == d)
        if ok:
            npass += 1
        else:
            fails.append((op, d, r, b))
        print(f"  [{'PASS' if ok else 'FAIL'}] {op:5s} "
              f"dense={d[-1] if d else None} routed={r[-1] if r else None} "
              f"routed+sparse={b[-1] if b else None}")
    print(f"\n=== routed  == dense AND routed+sparse_mm == dense : "
          f"{npass}/{n} argmax-identical ===")
    if fails:
        for op, d, r, b in fails:
            print(f"    {op}: dense={d} routed={r} both={b}")
        return 1
    print("PASS: top-1 routing composes with sparse_mm, byte-identical "
          "end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
