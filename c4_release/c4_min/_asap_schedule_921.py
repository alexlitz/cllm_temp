"""#921 ASAP (as-soon-as-possible) SCHEDULE critical path — the HONEST minimum depth a
free-reorder scheduler achieves, respecting only true data dependencies (RAW/WAR/WAW
on residual places).  This is the "width via earlier-FFN + step-reorder" strategy's
best-case: every block runs at the earliest step its inputs are ready, unlimited width.

Unlike the greedy same-order collapse (_measure_critpath_921, a conservative UPPER
bound), this is a proper longest-path over the dependency DAG (a tighter, honest LOWER
bound on the reorder-achievable depth).  A block B depends on block A (A before B) iff
they share a place with a RAW/WAR/WAW hazard AND A is emitted before B in source order
(the source order fixes the write-versioning; we honor it as the intended dataflow).

Golden 174ece66 untouched (off build path).
"""
from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

import torch

from c4_min._measure_critpath_921 import _block_rw


def asap_depth(specs, one_dim: int) -> Tuple[int, List[int]]:
    """Longest-path depth over the hazard DAG.  level[i] = 1 + max(level[j]) over all
    j < i that i truly depends on; total depth = max level.  Dependencies are the
    version-respecting hazards in SOURCE ORDER (so the dataflow the build intends is
    preserved — a later block that reads place p depends on the LATEST earlier writer
    of p; a later writer of p depends on earlier readers/writers of p)."""
    n = len(specs)
    rw = []
    for name, spec in specs:
        r, w = _block_rw(spec)
        r.discard(one_dim); w.discard(one_dim)
        rw.append((r, w))
    # last writer of each place seen so far, and readers since last write
    level = [1] * n
    last_writer: Dict[int, int] = {}
    readers_since_write: Dict[int, List[int]] = {}
    for i in range(n):
        r, w = rw[i]
        deps: Set[int] = set()
        # RAW: read p -> depends on last writer of p
        for p in r:
            if p in last_writer:
                deps.add(last_writer[p])
        # WAW + WAR: write p -> depends on last writer of p AND all readers since
        for p in w:
            if p in last_writer:
                deps.add(last_writer[p])
            for rd in readers_since_write.get(p, []):
                deps.add(rd)
        if deps:
            level[i] = 1 + max(level[d] for d in deps)
        # update state
        for p in r:
            readers_since_write.setdefault(p, []).append(i)
        for p in w:
            last_writer[p] = i
            readers_since_write[p] = []
    return (max(level) if level else 0), level


def run():
    import c4_min.qwen_full_vm as Q
    from c4_min.nibble_logsink_blocks import (
        extend_layout_for_logsink, compile_bm1, compile_log_query, _recip_attn_ffn,
        compile_newton_br, compile_newton_step, compile_qf, compile_seed_rem,
        compile_qb_products, compile_qb_split, compile_qb_carry, compile_qb_recombine,
        compile_rem, compile_correct, compile_snap_nibbles, compile_finalize,
        _msb_extract_block, compile_refine, compile_refine_add)
    os.environ["C4_LOGSINK_DIV"] = "1"
    subset = Q.SUBSET_FULL
    QL = Q.QwenFullLayout(24, subset, efficient_alu=True, recurrent_divmod=False,
                          code_from_memory=True, shift_via_mul=True, div_logsink=True)
    L = QL.L
    dim = L.D
    specs = Q._block_specs(L, 24, subset, efficient_alu=True, recurrent_divmod=False,
                           code_from_memory=True, shift_via_mul=QL.shift_via_mul,
                           div_logsink=QL.div_logsink)
    names = [n for n, _ in specs]
    base = [(n, s) for n, s in specs if not n.startswith("ls")]
    ext = L.LOGSINK

    def pdec(prefix, rem, nib):
        b = [(f"{prefix}-ext{c}", _msb_extract_block(L, dim, rem, nib, c,
                                                     round_low=(c == 0))) for c in range(8)]
        b.append((f"{prefix}-snap", compile_snap_nibbles(L, dim, nib, 8)))
        return b

    def schoolbook(pfx):
        return [(f"{pfx}-qb-products", compile_qb_products(L, dim)),
                (f"{pfx}-qb-split", compile_qb_split(L, dim)),
                (f"{pfx}-qb-carry0", compile_qb_carry(L, dim, ext.QB_COL, ext.QB_C1)),
                (f"{pfx}-qb-carry1", compile_qb_carry(L, dim, ext.QB_C1, ext.QB_COL)),
                (f"{pfx}-qb-carry2", compile_qb_carry(L, dim, ext.QB_COL, ext.QB_C1)),
                (f"{pfx}-qb-recombine", compile_qb_recombine(L, dim, ext.QB_C1)),
                (f"{pfx}-rem", compile_rem(L, dim)), (f"{pfx}-correct", compile_correct(L, dim))]

    core = [("ls-bm1", compile_bm1(L, dim)), ("ls-logq", compile_log_query(L, dim)),
            ("ls-recip-attn", _recip_attn_ffn(L, dim)),
            ("ls-newton-br1", compile_newton_br(L, dim, ext.RECIP)),
            ("ls-newton-st1", compile_newton_step(L, dim, ext.RECIP, ext.RECIP2)),
            ("ls-newton-br2", compile_newton_br(L, dim, ext.RECIP2)),
            ("ls-newton-st2", compile_newton_step(L, dim, ext.RECIP2, ext.RECIP2)),
            ("ls-qf", compile_qf(L, dim))]
    fp64 = core + [("ls-q-seed", compile_seed_rem(L, dim, ext.QF, ext.QREM, offset=-0.5))] + \
        pdec("ls-q", ext.QREM, ext.Q) + schoolbook("ls")
    fp64 += [("ls-d-seed", compile_seed_rem(L, dim, ext.QSC2, ext.DREM, offset=-0.5))] + \
        pdec("ls-d", ext.DREM, ext.DIV_RES)
    fp64 += [("ls-m-seed", compile_seed_rem(L, dim, ext.REM2, ext.MREM, offset=-0.5))] + \
        pdec("ls-m", ext.MREM, ext.MOD_RES) + [("ls-finalize", compile_finalize(L, dim))]
    fp32 = core + [("ls-q-seed", compile_seed_rem(L, dim, ext.QF, ext.QREM, offset=-0.5))] + \
        pdec("ls-q", ext.QREM, ext.Q) + schoolbook("ls")
    fp32 += [("ls-refine", compile_refine(L, dim)), ("ls-refine-add", compile_refine_add(L, dim)),
             ("ls-r-seed", compile_seed_rem(L, dim, ext.QSC, ext.QREM, offset=-0.5))] + \
        pdec("ls-r", ext.QREM, ext.Q) + schoolbook("ls2")
    fp32 += [("ls-d-seed", compile_seed_rem(L, dim, ext.QSC2, ext.DREM, offset=-0.5))] + \
        pdec("ls-d", ext.DREM, ext.DIV_RES)
    fp32 += [("ls-m-seed", compile_seed_rem(L, dim, ext.REM2, ext.MREM, offset=-0.5))] + \
        pdec("ls-m", ext.MREM, ext.MOD_RES) + [("ls-finalize", compile_finalize(L, dim))]

    base_names = [n for n, _ in base]
    mux_pos = base_names.index("alu-ax-mux")

    out = {}
    # divmod-only ASAP
    out["divmod_fp64_asap"] = asap_depth(fp64, L.ONE)[0]
    out["divmod_fp32_asap"] = asap_depth(fp32, L.ONE)[0]
    out["base_asap"] = asap_depth(base, L.ONE)[0]
    # whole-ISA ASAP (divmod spliced before ax-mux)
    for lab, dm in [("fp64", fp64), ("fp32", fp32)]:
        whole = base[:mux_pos] + dm + base[mux_pos:]
        out[f"whole_isa_{lab}_asap"] = asap_depth(whole, L.ONE)[0]
        out[f"whole_isa_{lab}_stored"] = len(whole)
    # also the existing serial builds for calibration
    ser = [(n, s) for n, s in specs]
    out["logsink_serial_asap"] = asap_depth(ser, L.ONE)[0]
    out["logsink_serial_stored"] = len(ser)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
