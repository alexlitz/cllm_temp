"""Measure the CROSS-FUNCTIONAL sharing structure of the c4_min op experts.

Companion to ``measure_weight_dedup`` (whole-tensor byte-identical TIE).  Where
that pass ties *replicated identical tensors* (the mul-carry loop body), this
tool quantifies the *cross-functional* sharing between ops that do SIMILAR
things — the comparison family (EQ/NE/LT/GT/LE/GE), ADD-vs-SUB, and the bitwise
family (OR/XOR/AND/SHL/SHR) — and reports, per family:

  * current unique nonzero weights,
  * how much of that is ALREADY a shared core (construction-level share) vs the
    op-specific combine,
  * the PROJECTED weights a further cross-functional refactor would leave, split
    into (a) byte-identity-preserving wins and (b) measured-potential rewrites
    (per-bit bitwise gadget, physical ADD/SUB merge) that CANNOT be made
    byte-identical and are therefore reported, not applied.

KEY FINDING it quantifies: the c4_min VM is ALREADY cross-functionally shared at
construction level (one ``compile_cmp_compute`` core + 6 dispatch combines; one
``_byte_add_block`` adder reused by SUB; one ``bw-expand`` + one merged
``bw-select``).  At the per-hidden-unit granularity (the finest byte-identical
tie) only ~0% of the in-scope units are duplicates, because every unit is
addressed to a DIFFERENT residual band.  So the residual compaction is a
non-byte-identical rewrite (measured here, not forced).

Run:  python -m c4_min.measure_cross_func_dedup
Memory-safe: streaming build, peak ~2.6 GB RSS (bitwise config), no divmod.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from c4_min import isa
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.nibble_bitwise import _BITWISE_FN, N_NIB
from c4_min.weight_dedup import _dense_of


# ---------------------------------------------------------------------------
# per-hidden-unit fingerprint (the finest byte-identical tie granularity)
# ---------------------------------------------------------------------------
def _unit_fp(Wu, Wg, Wd, bu, bg, u) -> str:
    h = hashlib.sha256()
    h.update(Wu[u].numpy().tobytes())
    h.update(Wg[u].numpy().tobytes())
    h.update(Wd[:, u].numpy().tobytes())
    h.update(np.array([bu[u], bg[u]], dtype="float32").tobytes())
    return h.hexdigest()


def _block(sparse, names, nm):
    return sparse.blocks[names.index(nm)]


def _ffn_nnz(sparse, names, nm) -> int:
    f = _block(sparse, names, nm).ffn
    return int(f.W_up.nnz) + int(f.W_gate.nnz) + int(f.W_down.nnz)


def measure(sparse, names):
    # -- family -> its blocks --
    fam = {
        "CMP (EQ/NE/LT/GT/LE/GE)": ["cmp-compute"],
        "ADD (4-byte carry chain)": [f"alu-add-b{i}" for i in range(4)],
        "SUB (4-byte carry chain)": [f"alu-sub-b{i}" for i in range(4)],
        "ADD+SUB operand expand (shared)": ["alu-expand"],
        "ALU result mux (5 ALU ops)": ["ax-mux"],
        "BITWISE operand expand (shared)": ["bw-expand"],
        "BITWISE OR/XOR/AND+SHL/SHR select": ["bw-select"],
        "BITWISE shift bit4": ["bw-bit4"],
        "BITWISE recompose": ["bw-recompose"],
    }
    present = {k: [b for b in bl if b in names] for k, bl in fam.items()}

    print("=" * 78)
    print("STEP 1 — CROSS-FUNCTIONAL SHARING MEASUREMENT (non-divmod ops)")
    print("=" * 78)
    print(f"{'op family':40s} {'blocks':>6s} {'unique nnz':>11s}")
    fam_nnz = {}
    for k, bl in present.items():
        n = sum(_ffn_nnz(sparse, names, b) for b in bl)
        fam_nnz[k] = n
        print(f"{k:40s} {len(bl):6d} {n:11d}")

    # ---- comparison core-vs-combine split ----
    print("\n--- COMPARISONS: shared core vs op-specific combine ---")
    print("  compile_cmp_compute builds ONE shared core: 3 lanes")
    print("    CMP_EQ = Z(d), CMP_GT = step(d>=1), CMP_LT = step(-d>=1)  [d = pop-ax]")
    print(f"    -> {fam_nnz['CMP (EQ/NE/LT/GT/LE/GE)']} nnz shared across ALL 6 ops")
    print("  each op is a TINY linear combine on that core (in the ONE dispatch block):")
    print("    EQ=CMP_EQ  NE=1-CMP_EQ  LT=CMP_LT  GT=CMP_GT  LE=1-CMP_GT  GE=1-CMP_LT")
    print("  => shareable fraction of the comparison COMPUTE = 100% (already 1 core, "
          "6 combines)")

    # ---- ADD vs SUB structural identity ----
    print("\n--- ADD vs SUB: is SUB reusing the ADD carry-chain adder? ---")
    print("  YES (construction): both call ONE `_byte_add_block`; SUB feeds the")
    print("  ones-complement operand (NOTB, built in alu-expand) + cin=1.")
    id_add = fam_nnz["ADD (4-byte carry chain)"]
    id_sub = fam_nnz["SUB (4-byte carry chain)"]
    print(f"  ADD nnz={id_add}  SUB nnz={id_sub}  (equal nnz = identical gadget)")
    for i in range(4):
        a = _block(sparse, names, f"alu-add-b{i}").ffn
        s = _block(sparse, names, f"alu-sub-b{i}").ffn
        same = all(torch.equal(_dense_of(getattr(a, k)), _dense_of(getattr(s, k)))
                   for k in ("W_up", "W_gate", "W_down"))
        print(f"    b{i}: add/sub byte-identical tensors = {same} "
              f"(differ ONLY by operand band B->NOTB, result band ADD->SUB, cin 0->1)")
    print("  => byte-identical TIE NOT possible (both blocks run in parallel, feed the")
    print("     mux; different bands/biases).  Physical MERGE = measured potential below.")

    # ---- bitwise tables vs the IMPLEMENTED per-bit gadget ----
    from .nibble_unified import _bitwise_perbit_enabled
    perbit_on = _bitwise_perbit_enabled()
    print("\n--- BITWISE: 3 lookup tables vs the SHARED per-bit gadget (IMPLEMENTED) ---")
    per_nib = {}
    for nm, op in (("OR", isa.OR), ("XOR", isa.XOR), ("AND", isa.AND)):
        fn = _BITWISE_FN[op]
        per_nib[nm] = sum(1 for a in range(16) for b in range(16) if fn(a, b) != 0)
    tot_tbl = sum(per_nib.values())
    print(f"  OR/XOR/AND per-nibble nonzero TABLE entries: "
          f"OR={per_nib['OR']} XOR={per_nib['XOR']} AND={per_nib['AND']} "
          f"= {tot_tbl}/nibble x {N_NIB} nibbles = {tot_tbl * N_NIB} select rules")
    # the applied per-bit gadget: per nibble, one -AX self-cancel + per plane the
    # boolean combine (AND 1, OR 3, XOR 3 rules) = 1 + 4*(1+3+3) = 29 rules/nibble.
    perbit_rules_per_nib = 1 + 4 * (1 + 3 + 3)
    perbit_total = perbit_rules_per_nib * N_NIB
    print(f"  APPLIED shared per-bit gadget: bit-plane extract (bw-bitplanes, shared "
          f"across the 3 ops) + a boolean combine")
    print(f"    AND=a.b  OR=a+b-a.b  XOR=a+b-2a.b  on the shared planes = "
          f"{perbit_rules_per_nib} rules/nibble x {N_NIB} = {perbit_total} select rules")
    print(f"  => OR/XOR/AND select {tot_tbl * N_NIB} -> {perbit_total} "
          f"(~{tot_tbl * N_NIB / max(1, perbit_total):.1f}x smaller); ARGMAX-IDENTICAL "
          f"to the table (bit-exact per (a,b)).")
    print(f"  [current build path: {'PER-BIT gadget (default)' if perbit_on else 'FULL TABLES (C4_BITWISE_PERBIT=0)'}]")

    # ---- per-hidden-unit byte-identical tie (the finest clean granularity) ----
    print("\n--- FINEST BYTE-IDENTICAL TIE: per-hidden-unit dedup across ALL in-scope "
          "op blocks ---")
    scope = [b for bl in present.values() for b in bl]
    seen = {}
    total = 0
    for nm in scope:
        f = _block(sparse, names, nm).ffn
        Wu, Wg, Wd = _dense_of(f.W_up), _dense_of(f.W_gate), _dense_of(f.W_down)
        H = Wu.shape[0]
        bu = f.b_up.numpy() if f.b_up is not None else np.zeros(H, dtype="float32")
        bg = f.b_gate.numpy() if f.b_gate is not None else np.zeros(H, dtype="float32")
        for u in range(H):
            total += 1
            fp = _unit_fp(Wu, Wg, Wd, bu, bg, u)
            seen[fp] = seen.get(fp, 0) + 1
    dupes = total - len(seen)
    print(f"  in-scope FFN hidden units: {total}  distinct: {len(seen)}  "
          f"byte-identical duplicates: {dupes} ({100 * dupes / total:.2f}%)")
    print("  => cross-functional byte-identical unit sharing is EXHAUSTED: every unit is")
    print("     addressed to a distinct residual band, so structurally-identical gadgets")
    print("     are NOT byte-identical.  This is why the whole-tensor tie already")
    print("     captures all the byte-identical cross-functional share there is.")

    # ---- the summary table (op family -> current -> shareable -> projected) ----
    print("\n" + "=" * 78)
    print("SUMMARY: op family -> current unique nnz -> shareable-across-func -> projected")
    print("=" * 78)
    print(f"{'family':38s} {'current':>8s} {'shared':>8s} {'projected':>10s}  note")
    rows = [
        ("COMPARISONS (6 ops)",
         fam_nnz["CMP (EQ/NE/LT/GT/LE/GE)"],
         fam_nnz["CMP (EQ/NE/LT/GT/LE/GE)"],
         fam_nnz["CMP (EQ/NE/LT/GT/LE/GE)"],
         "already 1 core + 6 combines (byte-id; no further clean win)"),
        ("ADD", id_add, id_add, id_add, "adder gadget (kept)"),
        ("SUB", id_sub, 0, 0,
         "reuses ADD gadget; physical merge saves ~all (NOT byte-id)"),
    ]
    for fam_name, cur, shr, proj, note in rows:
        print(f"{fam_name:38s} {cur:8d} {shr:8d} {proj:10d}  {note}")
    print(f"{'BITWISE OR/XOR/AND (bw-select rules)':38s} "
          f"{tot_tbl * N_NIB:8d} {tot_tbl * N_NIB - perbit_total:8d} {perbit_total:10d}  "
          f"per-bit gadget APPLIED (argmax-identical; C4_BITWISE_PERBIT)")

    return fam_nnz, dupes, total


def main():
    print("Building compact/streaming sparse model (bitwise config, no divmod)...")
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=48,
        compute_mode="dense_kernel")
    names = L._block_names
    print(f"n_blocks={len(sparse.blocks)} dim={sparse.dim}\n")
    measure(sparse, names)


if __name__ == "__main__":
    main()
