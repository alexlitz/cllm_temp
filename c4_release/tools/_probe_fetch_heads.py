"""Byte-identity check: derived ``fetch()`` heads vs hand-built L5 fetch heads.

Builds the 6 production L5 fetch head specs (``_fetch_head_specs``) and the 6
``fetch(FetchSpec(...))`` derived specs against the SAME dim_positions, then
asserts every resolved (slot, dim, weight) Q/K/V/O cell is identical.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def cells(spec):
    q = {(w.slot, w.dim): w.weight for w in spec.q}
    k = {(w.slot, w.dim): w.weight for w in spec.k}
    v = {(w.slot, w.dim): w.weight for w in spec.v}
    o = {(w.out_dim, w.slot): w.weight for w in spec.o}
    # detect accidental cell collisions (two AP writing the same cell): the
    # dict would silently drop one; count the raw list length vs dict length.
    for tag, raw, d in (("q", spec.q, q), ("k", spec.k, k), ("v", spec.v, v)):
        keys = [(w.slot, w.dim) for w in raw]
        if len(keys) != len(set(keys)):
            dup = [x for x in set(keys) if keys.count(x) > 1]
            raise AssertionError(f"{tag} DUPLICATE cells {dup}")
    okeys = [(w.out_dim, w.slot) for w in spec.o]
    if len(okeys) != len(set(okeys)):
        raise AssertionError("o DUPLICATE cells")
    return q, k, v, o


def main():
    from c4_release.neural_vm.unified_compiler.ops.l5_ops import (
        _fetch_head_specs, _derived_fetch_specs,
    )
    from c4_release.neural_vm.unified_compiler.ops.shared import _as_setdim_proxy

    names = [
        "TEMP", "EMBED_LO", "EMBED_HI", "FETCH_LO", "FETCH_HI",
        "ADDR_KEY", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
        "OPCODE_BYTE_LO", "OPCODE_BYTE_HI",
        "MARK_AX", "MARK_PC", "CONST", "HAS_SE",
    ]
    dim_positions = {nm: (100 + 1000 * i) for i, nm in enumerate(names)}
    BD = _as_setdim_proxy(dim_positions)

    hand = _fetch_head_specs(BD)
    derived_bundles = _derived_fetch_specs()

    assert len(hand) == len(derived_bundles), (
        f"count mismatch {len(hand)} vs {len(derived_bundles)}"
    )

    all_ok = True
    for h in hand:
        bundle = derived_bundles[h.head_idx]
        d = bundle.head_spec_builder(dim_positions, h.head_idx)
        hq, hk, hv, ho = cells(h)
        dq, dk, dv, do = cells(d)
        ok = (hq == dq and hk == dk and hv == dv and ho == do
              and h.alibi_slope == d.alibi_slope)
        status = "OK " if ok else "FAIL"
        print(f"head {h.head_idx}: {status}")
        if not ok:
            all_ok = False
            for tag, hh, dd in (("Q", hq, dq), ("K", hk, dk),
                                ("V", hv, dv), ("O", ho, do)):
                only_h = {c: hh[c] for c in hh if hh.get(c) != dd.get(c)}
                only_d = {c: dd[c] for c in dd if dd.get(c) != hh.get(c)}
                if only_h or only_d:
                    print(f"  {tag} hand-only/diff: {only_h}")
                    print(f"  {tag} derived-only/diff: {only_d}")
            if h.alibi_slope != d.alibi_slope:
                print(f"  alibi hand={h.alibi_slope} derived={d.alibi_slope}")

    print("ALL IDENTICAL" if all_ok else "MISMATCH")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
