"""Compare the #221 consumer-lookahead opcode-fetch head to a fetch() FetchSpec.

Dumps the lookahead builder's cells so we can design the FetchSpec variant that
reproduces it byte-identically (single top-0 K match + symmetric marker confirm).
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
    return q, k, v, o


def main():
    from c4_release.neural_vm.unified_compiler.ops.l5_ops import _ARITH_CONSUMER_GATE
    from c4_release.neural_vm.unified_compiler.isa_semantics_dsl import fetch, FetchSpec

    names = [
        "ADDR_KEY", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
        "NEXT_OPCODE_LO", "NEXT_OPCODE_HI", "LOOKAHEAD_PC_LO", "LOOKAHEAD_PC_HI",
        "MARK_AX", "CONST", "HAS_SE",
    ]
    dp = {nm: (100 + 1000 * i) for i, nm in enumerate(names)}

    hand = _ARITH_CONSUMER_GATE.opcode_fetch_head_spec_builder(dp, 6)

    # Candidate: dynamic addr from LOOKAHEAD_PC, static single top-0, symmetric
    # marker confirm, non-first step gate.
    spec = FetchSpec(
        name="lookahead_opcode_fetch", marker="MARK_AX",
        addr_mode="dynamic",
        addr_source_lo="LOOKAHEAD_PC_LO", addr_source_hi="LOOKAHEAD_PC_HI",
        top_mode="static_zero_single", marker_confirm_mode="symmetric",
        step_gate="non_first",
        target_lo="NEXT_OPCODE_LO", target_hi="NEXT_OPCODE_HI",
        alibi_slope=0.0,
    )
    derived = fetch(spec).head_spec_builder(dp, 6)

    hq, hk, hv, ho = cells(hand)
    dq, dk, dv, do = cells(derived)
    ok = (hq == dq and hk == dk and hv == dv and ho == do
          and hand.alibi_slope == derived.alibi_slope)
    print("OK" if ok else "MISMATCH")
    if not ok:
        for tag, hh, dd in (("Q", hq, dq), ("K", hk, dk), ("V", hv, dv), ("O", ho, do)):
            oh = {c: hh[c] for c in hh if hh.get(c) != dd.get(c)}
            od = {c: dd[c] for c in dd if dd.get(c) != hh.get(c)}
            if oh or od:
                print(f"  {tag} hand-only/diff:", oh)
                print(f"  {tag} derived-only/diff:", od)
        print("alibi hand", hand.alibi_slope, "derived", derived.alibi_slope)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
