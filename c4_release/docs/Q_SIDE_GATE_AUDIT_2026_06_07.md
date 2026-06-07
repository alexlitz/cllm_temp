# Q-side single-condition gate audit (2026-06-07)

Audit of `c4_release/neural_vm/unified_compiler/ops/*_ops.py` (and
sibling files in that directory) for the anti-pattern documented in
`docs/DSL_ATTENTION_GATE_FINDING_2026_06_07.md` (commit `bf11d69d`).
Based off `main` HEAD (`bf11d69d`; the brief referenced `6eaa6ef6`
which is not present in this clone — the design doc is identical at
that ref).

## Method

Static AST scan via `c4_release/tools/q_side_gate_audit.py`.

The audit understands two weight-author dialects:

* **Declarative DSL** — `AP(slot, BD.<DIM>, weight)` inside `q=` / `k=`
  tuples of a `DeclarativeAttentionHeadSpec` (the main path on
  current `main`). The scanner walks `BinOp` concatenations, list
  comprehensions, starred unpacks, and module-local helper calls
  that return AP/AO tuples (so `_addr_key_match_writes(BD, L)` is
  expanded inline).
* **Imperative writes** — `attn.W_q[slot_expr, BD.<DIM>] = ...` /
  `attn.W_k[...] = ...` (legacy / not-yet-migrated bakes).

For each Q-side write whose dim starts with `MARK_`, `OP_`, `HAS_`,
or `IS_`, the scanner groups by `(scope, slot_expr)` and inspects
the K-side at the same `(scope, slot_expr)`:

* **Safe** — K-side at the slot writes at least one non-`CONST` dim
  (a real K-side discriminator: MARK complement, address bit,
  byte-index marker, etc.).
* **No-op gate** — either no K-side write at the slot (Q-only —
  literal dead weight per the design doc), or only K-side `CONST`
  (the canonical silent-leak case).

The "K only writes CONST" classifier is independent of sign of the
Q-side weight — both positive ("require X") and negative ("blocker
on X") variants are silent no-ops when K is uniform, because the
softmax-cancelling argument is symmetric.

## Per-file summary

| file | Q-cond writes | safe | no-op gate |
| --- | ---: | ---: | ---: |
| flag_gated_ops.py | 3 | 3 | 0 |
| l3_ops.py | 13 | 8 | 5 |
| l4_ops.py | 7 | 4 | 3 |
| l5_ops.py | 11 | 0 | 11 |
| l6_ops.py | 10 | 10 | 0 |
| l7_ops.py | 20 | 14 | 6 |
| l8_ops.py | 30 | 17 | 13 |
| l9_ops.py | 10 | 8 | 2 |
| l10_ops.py | 31 | 19 | 12 |
| l14_ops.py | 28 | 15 | 13 |
| l15_ops.py | 5 | 3 | 2 |
| model_ops.py | 13 | 10 | 3 |
| other files (l0/l1/l2/l11/l12/l13/l16, alu, all_core, user_input, control_flow_heads) | 0 | 0 | 0 |
| **total** | **181** | **111** | **70** |

No-op gate breakdown:
- 60 are "K-side only writes CONST" — the doc's canonical case.
- 10 are "no K-side write at slot" — pure dead weight (slot 32 in
  L5 fetch heads; slot 0 in `_layer10_stack0_persistence_head_spec`).

Caveats:
- The L5 "dead weight" slot 32 is *exactly* the L5 head 0 pattern
  the design doc calls out as "works because the FETCH register has
  active FFN cleanup at non-AX rows". The slot leaks; downstream FFN
  masks it. The audit cannot detect downstream cleanup.
- L9 heads 0 and 1 still bake via `_set_layer9_lev_addr_relay` /
  `_set_layer9_lev_bp_to_pc_relay` in `vm_step.py` (delegated through
  `make_layer9_lev_addr_relay_op`'s `bake_fn`). The 2 no-op gate
  cases in `l9_ops.py` are the legacy GATE=33 cases from
  `_carry_forward_head_spec` siblings (`_layer9_lev_addr_relay_head_spec`
  L1471 and `_layer9_lev_bp_to_pc_relay_head_spec` L1507) — declarative
  rewrites of the carry-forward primitive's GATE=33 anti-leakage
  pattern.

## High-volume hot-spots

### `_carry_forward_head_spec` (L3 + L9, declarative re-baked primitive)

The historical `Primitives.carry_forward_attention` has been replaced
by `_carry_forward_head_spec` (l3_ops.py L1257-1308). The new spec
still emits the same GATE=33 no-op:

```python
GATE = 33
q = [AP(0, marker_dim, L),
     AP(GATE, marker_dim, L),    # ← no-op gate Q-side
     AP(GATE, BD.CONST, -L / 2)]
k = [AP(0, BD.L1H1 + l1h1_idx, L),
     AP(0, BD.L1H0 + l1h0_idx, -L),
     AP(GATE, BD.CONST, L)]      # ← only CONST on K-side at GATE
```

Used by:
- L3 heads 0–3 (PC/AX/SP/BP carry-forward, `layer3_carry_forward_attn`,
  always-on) — 4 GATE=33 no-ops.
- L3 head 4 stack0 carry (`_stack0_carry_head_spec`) — 1.
- L3 head 5 AX_FULL relay (`_ax_full_relay_head_spec`) — 1.
- L3 head 6 LEV BP→PC (`_lev_bp_to_pc_head_spec`) — 1.
- L3 head 7 PC byte1 prev (`_pc_byte1_prev_head_spec`) — 2 (slot
  33 IS_BYTE and MARK_PC).
- L9 head 0 (`_layer9_lev_addr_relay_head_spec`) — 1.
- L9 head 1 (`_layer9_lev_bp_to_pc_relay_head_spec`) — 1.
- L4 heads 0–1 (`_layer4_pc_relay_head_specs`) — 2.

The GATE=33 anti-pattern is structurally identical in all of them and
the count alone is ≥ 12 of the 60 K=CONST cases.

### `l8_ops.py:bake` (legacy imperative, `make_layer8_mem_to_alu_op`)

L2318-2342, slot=`base`: 6 K=CONST no-ops (MARK_AX positive +
MARK_PC/SP/BP/MEM/STACK0 negative blockers). Plus dynamic OP_* loop
no-ops not directly counted (the audit reports the 6 literal MARK_*
writes; the additional OP_ADD/SUB/MUL/.../EXIT writes on the same
slot are also no-ops but live in a Python `for op_dim in (...)` loop).
**Currently `enable=False`** in `all_core_ops.py:209` — risk is
contained until that flag flips.

### `l14_ops.py:_layer14_mem_generation_head_specs`

8 heads × multiple slots, 13 no-op gate slots. Notably:
- Slot 33 ("Position gate"): Q-side has 4–5 condition writes,
  K-side `AP(33, BD.CONST, 5.0)` only. Repeated for heads 0–3 and
  4–7.
- Slot 38 ("shared non-MEM target blocker"): Q-side has 8 negative
  MARK_*/H1+* blockers, K-side `AP(38, BD.CONST, 5.0)` only.

`make_layer14_mem_generation_op()` is unconditionally registered in
`all_core_ops.py:303` — these no-ops are always live.

### `l10_ops.py:_layer10_stack0_persistence_head_spec`

10 no-op gate writes across slots 0 and 33 — slot 0 has Q-only
negative gates (PSH_AT_SP, OP_PSH, CMP+0/1/2/4, OP_LEV), and slot 33
mirrors them with stronger negatives plus HAS_SE / STACK0_BYTE0–3
positives.

K-side: slot 0 has no AP entries; slot 33 has only `AP(33, BD.CONST,
100.0)`. The substantive byte-relay logic lives at slots 4–11 with
proper K-side STACK0_BYTE*/BYTE_INDEX_* discriminators, which are
correctly classified safe.

## Top-5 risk: silent leaks most likely masking bugs

Risk = (live in default config) × (downstream consumer sensitive to
leak rows) × (number of Q-side gates apparently doing real work).

### 1. `l14_ops.py` `_layer14_mem_generation_head_specs` slot 38 — **HIGH**

`make_layer14_mem_generation_op()` is always-on. The slot-38 "shared
non-MEM target blocker" tries to suppress MEM-generation contribution
at non-MEM marker rows (PC/AX/BP/STACK0/H1+*). With K[38]=CONST only,
the per-Q-row -target_block_s offset is uniform across K → softmax
cancels → the blocker doesn't blocker. MEM-generation V/O routes
CLEAN_EMBED/OUTPUT into OUTPUT_LO/HI at every Q row, including
non-MEM marker rows. The L14 MEM-generation output is what
`_inject_mem_metadata` would otherwise write — leakage onto PC/AX/BP
rows aliases MEM content into register markers. Probable interaction
with PSH/SI/LEV byte-index regressions.

Locations: L512-515 and L608-611 (slot 38), L501 and L597 (slot 33).

### 2. `l10_ops.py` `_layer10_stack0_persistence_head_spec` slot 33 — **HIGH**

Always-on (`make_layer10_stack0_persistence_op` registered in
all_core_ops.py). Slot 33 has the multi-condition "active step"
gate: `HAS_SE` requires step ≥ 1, OP_PSH/CMP+0/1/2/4/OP_LEV all
block STACK0 byte-relay on those opcodes, STACK0_BYTE0/1/2 positive
to require byte-index alignment. K-side at 33 is only
`AP(33, BD.CONST, 100.0)`. None of the OP_* blockers fire. This op
is on the STACK0 critical path — the documented `1096_var_*` and
PSH-related regression clusters (see MEMORY.md `var_failure_mode_shifted`
note) are plausibly downstream of the slot-33 leak.

Locations: L1838-1850 (slot 33), plus 2 entries at slot 0 (L1793,
L1798 — Q-only OP_PSH / OP_LEV blockers, no K-side at all → dead
weight rather than active leak).

### 3. L3 carry-forward primitive GATE=33 (every register relay) — **MEDIUM**

`_carry_forward_head_spec` (l3_ops.py L1284-1294) bakes a GATE=33
no-op gate. The historical comment is "Anti-leakage gate at slot 33"
— the comment was *aspirational*, not descriptive. It does not
suppress leakage.

Heads affected: L3 head 0 (PC), 1 (AX → AX_CARRY), 2 (SP), 3 (BP),
plus head 4 (stack0 carry retirer), head 5 (AX_FULL relay),
head 6 (LEV BP→PC), head 7 (PC byte1 prev). The downstream
consumers are AX_CARRY (FFN ALU input), AX_FULL (multi-byte AX
arithmetic), and the LEV return-address path — exactly the residual
bands implicated by the documented `EQ(17,17) byte-1 corruption at
L6` and `var_failure` clusters.

The doc explicitly flagged commit `9e4711e7` ("Var fix v2 H1+4
blocker") as "uses K-side blocker, should be fine" — but the
carry-forward GATE=33 is the *general* version of that same
anti-pattern across every register relay. Worth a runtime probe at
every register marker.

### 4. `l8_ops.py` `_layer8_sp_gather_head_specs` — **MEDIUM**

`make_layer8_sp_gather_op()` is always-on. Slot 33 has Q-side
MARK_STACK0 (L1884) and MARK_SP (L1946) gates with K-side CONST
only. This op stages SP-byte → ADDR_KEY content for the L8 ALU
path (per the docstring on `make_layer4_sp_to_addr_key_op`). Leak
at slot 33 = SP-byte content potentially routed onto non-STACK0
markers, contaminating ADDR_KEY at PC/AX/BP rows. Lower priority
than #1/#2 because the substantive address-match logic at other
slots is safe.

### 5. `l14_ops.py` `_layer14_alu_high_byte_relay_spec` slot 34 — **MEDIUM**

`AP(34, BD.OP_LI_RELAY, -10000.0)` and
`AP(34, BD.OP_LC_RELAY, -10000.0)` (L834-835) try to block the
ALU-high-byte relay on LI/LC steps. K[34] = CONST only → blockers
fire on no row. If LI/LC steps relay ALU_HI through this head when
they shouldn't, that's an ALU contamination path.

Also: L827 slot 33 `IS_BYTE` gate has the same defect.

### Honorable mentions

- `l7_ops.py` `_layer7_operand_gather_head_specs` slot 33 (L247-251):
  Q-side blockers on OP_LEA / OP_ADJ / OP_ENT with K=CONST. The
  doc's `ff4edb61` ("L7 head 5 K-side OP_IMM blocker — claimed to
  work, but did it really?") — this audit can confirm the L7 head
  in question is structurally identical to other no-op blockers.
- `model_ops.py` `_function_call_l5_head_specs` slot 33 / 34
  (MARK_STACK0, MARK_BP, OP_ENT) — function-call routing path; if
  leaking, may interact with JSR/ENT/LEV identity regressions.

## Suggested next steps

1. **Add an IR static check.** Inside
   `compare_symbolic_to_lowered_attn` or a dedicated lint, walk every
   `DeclarativeAttentionHeadSpec` and flag any `(slot, dim)` where
   `dim` starts with `MARK_` / `OP_` / `HAS_` / `IS_` and the K-side
   has no non-`CONST` writes at the same slot. With 60 hits across
   12 files this could land as a warning today and ratchet to error.
2. **Runtime probe top-2 risks.** Build a probe that fires each
   affected head against a hand-curated grid of Q rows (varying
   marker, opcode, has_se) and checks that the head's output is zero
   at the "should be blocked" rows. The probe surface is small:
   - `_layer14_mem_generation_head_specs` heads 0–7 at non-MEM
     markers.
   - `_layer10_stack0_persistence_head_spec` at OP_PSH / OP_LEV / CMP
     rows.
3. **Decide on the GATE=33 anti-leakage pattern.** It appears in 12+
   places. Either delete it everywhere (it's already a no-op so
   removal is a no-op too) or rewrite it with a real K-side
   blocker (the design doc's option 3: "at non-target K positions
   write a large negative for the Q-gated slot").
4. **Out of scope for ops/ but related** — `setup_helpers.py:238`,
   `primitives.py` `carry_forward_attention`, and the
   `_set_layer9_*` helpers in `vm_step.py` all contain or originate
   the same GATE=33 / Q-gate-only patterns.
