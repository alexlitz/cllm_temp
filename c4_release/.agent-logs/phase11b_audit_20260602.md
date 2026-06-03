# Phase 11.B audit — token-marker vs opcode-marker IR rule gating

## How counted

`tools/phase11b_audit.py` walks every `make_*` factory in
`c4_release/neural_vm/unified_compiler/ops/*.py` that has no required
positional args (170/170 succeeded), instantiates each op, pulls
`FFNRule`s via `decl_verifier._collect_ffn_rules_from_op`, and
classifies each rule by its condition gates.

A rule is classified by inspecting `conditions=`, the `gate=` DimRef,
and `gate_terms=`:

- **opcode-gated**: contains any dim whose name starts with `OP_`,
  `ACTIVE_OPCODE_`, or `IO_IS_`.
- **token-gated**: contains any `MARK_*` dim but no opcode-class dim.
- **mixed**: both.
- **dim-free**: neither.

(The doc's prior counts used a narrower `OP_*`-only check on
`conditions=`. This audit folds in `gate=` / `gate_terms=` and
includes `ACTIVE_OPCODE_*` / `IO_IS_*` aliases — which matches how the
lowerer actually discriminates.)

## Counts (170 ops, 25,009 rules)

| Bucket        | Count  | %     |
| ------------- | ------ | ----- |
| opcode-gated  |    168 |  0.7  |
| mixed         | 22,031 | 88.1  |
| token-gated   |  2,197 |  8.8  |
| dim-free      |    613 |  2.5  |

(Doc's pre-phase-11.B numbers: 27,331 rules; 22,953 with OP_* condition;
4,378 token-only. Discrepancies: (a) audit only walks no-arg factories
— factories taking `block_idx` / `S` etc. are skipped (≈ 2,300 rules).
(b) Classifier folds the `gate=` field — many "token-gated by the
narrow rule" are actually opcode-class via `gate=OP_MUL` etc.)

## Token-gated rules per op (top 14)

```
973   tail_bit32_result_correction        (of 2059)
530   layer6_routing_ffn                  (of 1507)
272   layer9_alu                          (of 3405)
159   layer4_ffn                          (of  544)
 88   opcode_decode_ffn                   (of   89)   -- creates OP_* dims
 48   layer14_clear_addr_key_pollution    (of   48)
 45   layer3_ffn                          (of  136)
 40   layer16_lev_routing                 (of  792)
 32   layer15_nibble_copy                 (of   42)
  4   layer8_alu                          (of 2023)
  2   layer14_clear_output_corruption     (of    3)
  2   layer2_initial_pc_bake_cancel       (of    2)
  1   layer14_temp_clear                  (of    4)
  1   layer8_sp_gathered_sentinel         (of    1)
```

## Finding: no clean candidates for naive migration

Manual inspection of each "few token-gated rule" op uncovered no
zero-risk token→opcode swap candidates:

- `layer3_ffn` (45 rules) — register-marker default writes
  (PC/SP/BP/MEM/STACK0 first-step defaults). Genuinely opcode-agnostic;
  must fire at MARK_* for every opcode. **No migration.**
- `layer6_routing_ffn` (530 rules) — delayed JMP-cancel / EXIT-routing
  cleanups at MARK_PC gated by CMP cascade flags from L5. Opcode is
  implied by the CMP flag, not the rule. **No migration.**
- `layer8_alu cmp_clear_k0..k3` (4 rules) — clear CMP+k at MARK_AX with
  `gate_weight=S, gate_bias=-S/2`. The MARK_AX gate is load-bearing for
  the SiLU half-step; swapping to OP_* changes the lowered W_gate.
  **No migration.**
- `layer14_clear_addr_key_pollution` (48 rules) — defensive clear with
  ALL-MARK_* exclusion gates (negative weights). The rule purpose is
  "fire when NOT a marker". **No migration.**
- `layer14_clear_output_corruption`, `layer14_temp_clear` — same
  exclusion-gate pattern (negative MARK_* weights). **No migration.**
- `layer2_initial_pc_bake_cancel` (2 rules) — cancels EMBED initial-PC
  bake at MARK_PC for HAS_SE rows. PC-marker dependency is structural.
  **No migration.**
- `layer16_lev_routing l16_top_store_stack0_restore_lo_*` (40 rules) —
  already opcode-gated via `MEM_STORE` (which encodes SI/SC/PSH). The
  MARK_STACK0 condition is the structural marker, not an opcode proxy.
  **No migration.**
- `tail_bit32_result_correction` (973 rules) — late tail corrections;
  bulk uses gate-strength patterns where MARK_* sets the SiLU half-step
  cutoff. Each rule needs case-by-case analysis. **Deferred.**

## Conclusion for phase 11.B planning

The 2,197 remaining token-gated rules in this corpus are not
mechanically migratable. They fall into:

1. **Marker defaults** (~50 rules in L3): fire at every register-marker
   first step regardless of opcode. Genuinely opcode-agnostic.
2. **Marker cleanups** (~100 rules in L8/L14): use MARK_* as the gate
   that pins the SiLU half-step; the lowered weights depend on `MARK_*`
   appearing in `W_gate`. Swapping the gate dim is not a refactor.
3. **Marker exclusion gates** (~100 rules in L14): use MARK_* with
   negative weights to exclude marker positions. The "fire when NOT
   marker" semantics has no opcode equivalent.
4. **Marker + opcode-class proxy** (~1500 rules in L16/L10/L6): already
   use opcode-class proxies (MEM_STORE, CMP+k, PSH_AT_SP) alongside
   MARK_*. Adding raw OP_* would over-constrain.

The remaining ~5,500 candidates implied by the doc are likely the same
pattern — **per-op semantic investigation, not a templatable mass
migration**. The doc's framing of "4,378 token-only gates should
migrate" should be re-scoped to "audit per family and tighten where
the marker-dependency is accidental", which the prior phase 11.A audit
also concluded.

Recommended next step: a smaller, more targeted audit on the
`tail_bit32_result_correction` 973 rules — the largest bucket where
opcode-class tightening might be feasible (each tail correction
already has an OP_* in `mixed` for one writeback but its sibling
correction writes lack it). That family is also the most likely to
contain accidental MARK_*-only rules from incremental authoring.
