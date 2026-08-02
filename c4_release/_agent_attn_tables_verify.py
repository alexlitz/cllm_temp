"""C4_ATTN_TABLES — one-shot verify: byte-exact proof + efficiency + doom compose.

Run:  PYTHONPATH=<c4_release> python _agent_attn_tables_verify.py

Prints PASS/FAIL for every byte-exact claim, the measured efficiency, the
reciprocal-divide error, and the doom isin/icos composition — then reminds that
golden 069cc32f is unchanged (all files here are additive, gated, off the build
path; verified separately with ``python -m c4_min._fingerprint_build``).
"""
from __future__ import annotations

from _agent_attn_tables import AttentionTable, init_sin_reference
from _agent_attn_tables_measure import (
    prove_byte_exact, measure_init_steps_saved, measure_store_log_reduction,
    measure_lookup_cost, measure_residency_tradeoff, measure_reciprocal_divide,
)

CIRC = 256


def doom_isin_compose() -> dict:
    """isin(a)/icos(a) via the attention-baked sintab, byte-exact incl wrap."""
    sintab = init_sin_reference(CIRC, 1024)
    t = AttentionTable(sintab, name="sintab", signed=True).bake()

    def _norm(a):
        while a < 0:
            a += CIRC
        while a >= CIRC:
            a -= CIRC
        return a

    def isin_ref(a):
        return sintab[_norm(a)]

    def isin_attn(a):
        return t.lookup(_norm(a))

    def icos_ref(a):
        return isin_ref(a + CIRC // 4)

    def icos_attn(a):
        return isin_attn(a + CIRC // 4)

    bad = [a for a in range(-300, 600)
           if isin_ref(a) != isin_attn(a) or icos_ref(a) != icos_attn(a)]
    return {"angles_tested": "-300..599 (incl +/- wraparound)",
            "isin_icos_byte_exact": len(bad) == 0, "mismatches": bad[:5]}


def main():
    print("=" * 74)
    print("C4_ATTN_TABLES — attention-baked static lookup tables (gated, OFF)")
    print("=" * 74)

    print("\n[1] BYTE-EXACT LOOKUP PROOF (LOOKUP(table,i) == table[i]):")
    p = prove_byte_exact()
    for name in ("sintab", "palette", "reciprocal"):
        d = p[name]
        print(f"  {name:11s}: {d['entries']} entries  byte_exact={d['byte_exact']}"
              f"  mismatches={d['mismatches']}")
    z = p["zfod_unbaked_index"]
    print(f"  ZFOD       : lookup(unbaked {z['index']}) = {z['value']} "
          f"(expect 0) ok={z['ok']}")

    print("\n[2] INIT-STEPS SAVED (init_sin compute vs baked 0-init):")
    s = measure_init_steps_saved()
    print(f"  runtime init_sin VM steps : {s['runtime_init_sin_vm_steps']} "
          f"(DIV={s['runtime_init_sin_DIV_ops']}, MUL={s['runtime_init_sin_MUL_ops']}, "
          f"SI={s['runtime_init_sin_SI_stores']})")
    print(f"  baked init VM steps       : {s['baked_init_vm_steps']} "
          f"(bake wall {s['baked_bake_wall_s']}s)")
    print(f"  STEPS SAVED               : {s['steps_saved']}  "
          f"[DIV note: {s['div_note']}]")

    print("\n[3] STORE-LOG REDUCTION (table not in dynamic per-step log):")
    sl = measure_store_log_reduction()
    print(f"  runtime store-log rows added : {sl['runtime_store_log_rows_added']}")
    print(f"  baked   store-log rows added : {sl['baked_store_log_rows_added']}")
    print(f"  runtime eviction working set : {sl['runtime_eviction_working_set']}")
    print(f"  baked   eviction working set : {sl['baked_eviction_working_set']}")

    print("\n[4] LOOKUP COST (O(1) attention gather):")
    lc = measure_lookup_cost()
    print(f"  device                       : {lc['device']}")
    print(f"  attention gather 256q wall   : "
          f"{lc['attention_gather_256queries_wall_ms']} ms "
          f"({lc['per_lookup_attention_us']} us/lookup)")

    print("\n[5] RESIDENCY TRADEOFF (large tables, honest VRAM):")
    rt = measure_residency_tradeoff()
    for r in rt["per_table"]:
        print(f"  {r['entries']:5d} entries -> {r['kv_MB_fp32']} MB fp32 KV "
              f"(bake {r['bake_wall_s']}s)")
    print(f"  {rt['note']}")

    print("\n[6] RECIPROCAL-DIVIDE (a/b -> a*LOOKUP(recip,b)>>S, vs 168-block DIV):")
    rd = measure_reciprocal_divide()
    for tag in ("shift16_Q16", "shift24_Q24"):
        d = rd[tag]
        print(f"  {tag}: exact={d['exact_fraction']*100:.1f}% "
              f"max_abs_err={d['max_abs_error']}  worst={d['worst_case']}  "
              f"({d['pairs']} pairs)")
    print("  power-of-two divisor (FP=1024=2^10): reciprocal-multiply is EXACT")

    print("\n[7] DOOM COMPOSITION (isin/icos via attention-baked sintab):")
    dc = doom_isin_compose()
    print(f"  {dc['angles_tested']}: byte_exact={dc['isin_icos_byte_exact']} "
          f"mismatches={dc['mismatches']}")

    print("\n" + "=" * 74)
    all_ok = (all(p[n]["byte_exact"] for n in ("sintab", "palette", "reciprocal"))
              and p["zfod_unbaked_index"]["ok"]
              and dc["isin_icos_byte_exact"])
    print(f"ALL BYTE-EXACT CLAIMS: {'PASS' if all_ok else 'FAIL'}")
    print("golden 069cc32f UNCHANGED (additive files, gated OFF, off build path;")
    print("verify: PYTHONPATH=. python -m c4_min._fingerprint_build)")
    print("=" * 74)
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
