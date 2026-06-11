# Opcode-broadcast-defeats-intent-blocker — class audit (2026-06-11)

Date: 2026-06-11. Path: spec_k=0, hook-free (`tools/probe_groundtruth.py`).
Dims read via `probe.model.dim_positions` (NOT `dim_registry_dynamic`, which is
stale). READ-ONLY pass — no ops/ weight edits (the JSR→ENT prologue chain agent
owns l16/l10/l9). Tooling added: `tools/audit_opcode_broadcast.py` (broadcast
magnitudes, spec_k=0) and `tools/scan_opcode_broadcast_rules.py` (static
algebra scan, no model build).

## TL;DR

* The bug class is **real and confined to frame/control opcodes**. ALU opcodes
  (ADD/SUB/OR/AND/MUL/EQ/LT/…) broadcast **0.0** in-step → they cannot defeat a
  blocker. The class is OP_JSR / OP_ENT / OP_LEA / OP_LEV / OP_ADJ only.
* **OP_JSR is the worst broadcaster**: ~15.5 at register markers, **~17.2 at
  byte rows**, ~11.5 at STEP_END. OP_ENT is the second driver, ~7.3-11.4 (peaks
  at the ENT-step AX/SP marker). OP_LEA ~5.2. **OP_LEV / OP_ADJ measured ≈0** at
  the sampled rows — they are NOT practical drivers. **JSR + ENT are the only
  two opcodes that drive the class in practice.**
* **87 distinct rule families** reference a broadcasting frame/control opcode as
  a positive condition (696 rules incl. 16×/32× band expansion). After the
  co-firing-blocker algebra: **~2-4 genuinely-vulnerable, 14 latent, 71 safe.**
* **All 5 already-fixed rules + every high-weight (OP×50…1000) rule are now SAFE
  via the hard `-1e6`/`-1e9` NOT-blocker pattern.** The remaining risk is in
  **low-weight (OP×1-5) nudge rules with thin margins** — the latent class.
* **Proposed systematic fix:** promote the IS_BYTE + non-target-marker blockers
  on the 14 latent + 2-4 (a) families to hard NOT-blockers (`-1e6`), exactly the
  pattern the chain used on the 5 fixed rules. This neutralizes the class
  permanently regardless of how large the opcode broadcast grows. Cluster
  leverage: these families gate the JSR→ENT frame prologue shared by the
  ~525 func/var/loop/rec/gcd programs.

---

## Part A — broadcast magnitude table (spec_k=0)

Residual magnitude of each `OP_<X>` dim read at the **input to the L20 FFN
block** (physical block 28 output) — the block where the l16 frame/control
rules read the opcode flag. Identical at block 33 (the flag persists). Rows
grouped by type; `mark` = the PC/AX/SP/BP register markers, `byte` = the 4 byte
rows after each marker, `SE` = STEP_END.

### var_simple_12 (id 262, `int x; x=28; return x;` — JSR→ENT prologue)

| opcode  | mark_max | mark_mean | byte_max | byte_mean | SE_max |
|---------|---------:|----------:|---------:|----------:|-------:|
| OP_JSR  |  **15.500** | 6.960 | **17.220** | 5.895 | **11.454** |
| OP_ENT  |    2.072 | 0.589 |    1.912 | 0.507 |  4.064 |
| OP_LEV  |    0.000 |  −0.000 |  0.000 | −0.000 | 0.000 |
| OP_ADJ  |    0.000 |  −0.000 |  0.000 | −0.000 | 0.000 |
| OP_LEA  |    5.233 | 0.694 |    0.221 | 0.061 |  2.990 |
| OP_ADD…OP_LT (all ALU) | **0.000** | 0.000 | **0.000** | 0.000 | 0.000 |

Per-step peak at register-marker rows (id 262):
`step0: OP_JSR=+15.5  step1: OP_JSR=+12.9 OP_ENT=+2.1 OP_LEA=+5.2`

### ENT-bearing program (id 651, `CALL; LI; LEV; EXIT`) and ADJ program (id 951)

Same probe, reading at L20-input. On both, the sampled register-marker / byte
rows are dominated by the **JSR→ENT prologue** the program spends most of its
rows in, so the persistent broadcasters are OP_JSR and OP_ENT:

| opcode | mark_max | byte_max | SE_max |
|--------|---------:|---------:|-------:|
| OP_JSR | 15.49 | 17.20 | 11.45 |
| OP_ENT | **7.28** | **7.61** | 7.04 |
| OP_LEV | 0.000 | 0.000 | 0.000 |
| OP_ADJ | 0.000 | 0.000 | 0.000 |
| OP_LEA | 0.000 | 0.000 | 0.000 |

Per-step: `step0 OP_JSR=+15.5  step1 OP_JSR=+12.9 OP_ENT=+7.3`.

**Key correction to the scanner's pin:** OP_ENT's persistent broadcast measures
**7.3-11.4** depending on the row (7.3 at the LI/LEV-program rows here, 11.4 at
the dedicated ENT-step AX/SP marker in commit f04e9f02). OP_LEV / OP_ADJ /
OP_LEA do **not** show a persistent broadcast at the sampled L20-input rows
(0.0) — their flag is only briefly asserted at their own opcode's step, which is
a single step late in these programs. The scanner pins OP_LEV=OP_ADJ=5.0 as a
**conservative upper bound**; the true sampled magnitude is ≈0, so every
LEV/ADJ-gated rule classed (b)-latent below is in practice **even safer** than
reported. **OP_JSR (15.5/17.2) and OP_ENT (7.3-11.4) are the only two opcodes
that drive the class in practice.**

### Reading

* The broadcast is **in-step and addressed to every row**, not just the
  opcode's own marker — confirming the class premise. OP_JSR's byte-row peak
  (17.2) is even higher than its marker peak (15.5), which is why the byte-row
  mis-fires (REG_BP byte0/1/3 = 0x0A) were the original symptom.
* The intent-blockers in the original rules were sized for `OP_<X>≈1` (one-hot
  at its own marker). At 15-17× that, a `−10`/`−100`/`−300` blocker no longer
  vetoes — exactly the 5 fixed instances.
* **ALU opcodes broadcast 0** → the ~hundreds of ALU FFN rules with `("OP_ADD",
  W)` etc. are inert for this class and need no audit.

---

## Part B — at-risk rule list (static scan, co-firing-blocker algebra)

`tools/scan_opcode_broadcast_rules.py` imports every ops factory, collects all
**26,998** resolved `FFNRule`s, and for each rule whose `conditions`/`gate_terms`
carry a POSITIVE `("OP_<broadcaster>", W)` term computes the realistic mis-fire
score: `Σ(OP_<X>·broadcast) − Σ(co-firing blockers at the wrong row)` vs
`threshold`. Co-firing blockers = `IS_BYTE` (byte rows) + the single largest
`MARK_*`/`H*` (one wrong-register marker) + `BYTE_INDEX_{1,2,3}`. A blocker
`≥1e5` is treated as a hard NOT-blocker (immune).

**87 distinct rule families** touch a broadcasting opcode (band/lane-collapsed
from 696 rules). Classification:

| class | meaning | family count |
|-------|---------|-------------:|
| (a) | genuinely-vulnerable (mis-fires now under the co-fire model) | 2 strict, up to 4 if a wrong-register marker does NOT co-fire at the mis-fire row |
| (b) | latent (blocker wins now, thin margin / no hard blocker) | 14 |
| (c) | safe (hard NOT-blocker, or blocker decisively vetoes) | 71 |

### (a) genuinely-vulnerable

| family (×band) | op term | thr | mis-fire score | note |
|---|---|---:|---:|---|
| `l6_adj_ax_to_output` (×32) | OP_ADJ×1 (≈5) | 4 | 5−1 = **4 ≥ 4** | ADJ-step AX route; only a −1 IS_BYTE blocker. Mis-fires onto AX byte rows. |
| `l6_jsr_sp_fixup` (×2) | OP_JSR×0 + bias | 2 | 3−1 = **2 ≥ 2** | JSR SP-fixup; weakest single −1 blocker. Marginal. |
| `l6_jsr_ax_to_output` (×32) | OP_JSR×1 (≈17) | 4 | depends* | *at the AX **byte** row only IS_BYTE(−10) co-fires (no wrong MARK), giving 17−10 = 7 ≥ 4 → vulnerable. The co-fire model conservatively also subtracts a MARK(−8) → 17−18 = −1 (safe). Needs the empirical check below; structurally borderline (a)/(b). |

These three are **all in the L6 AX/SP-route family** — the same `make_layer6_*`
factory whose JSR-leg leak was the genesis of the whole chain (commit
2e91979c / `l6_jsr_*`). They write a routed AX_CARRY/SP value into the OUTPUT
band; mis-firing onto a register byte row corrupts that byte's emission. They
are LOW weight (OP×0-1) so the margin is small, but the broadcast (17) is large
enough to clear a single weak (−1 or −10) blocker.

### (b) latent (14 families)

| family | op term | thr | why latent |
|---|---|---:|---|
| `l9_alu_lo_clear` / `l9_alu_hi_clear` (×16) | JSR+ENT+ADJ+LEV ×1 each = **39** | 2 | no marker/IS_BYTE blocker at all; safe only because it co-requires a non-opcode positive (the ALU-clear marker). If that positive ever leaks, 39 ≫ 2 fires everywhere. |
| `l16_lev_stack0_byte0_preserve` (×32) | OP_LEV×5 (≈25) | 7.5 | co-fire blockers 18, score 7 vs thr 7.5 — wins by 0.5. One extra co-asserted opcode or a bigger LEV broadcast flips it. |
| `l6_ent_after_jsr_*` (sp/bp/stack, 6 families) | OP_ENT×1 (≈11) | 6-8 | no IS_BYTE/marker blocker; relies on a per-byte/per-value positive. ENT broadcast 11 > thr. |
| `l6_ent_sp_writeback` (×32), `l6_adj_sp_writeback` (×32), `l16_lev_pc_temp` (×32) | ENT/ADJ/LEV×1 | 2-4 | no hard blocker; scoped only by a co-required positive. |
| `l8_alu_lev_b1/b2_step_end` (×1 each) | OP_LEV×1 | 2 | no blocker; co-required positive only. |

### (c) safe (71 families) — includes the 5 fixed instances

The fixed rules land in (c), validating the classifier:

| fixed rule (commit) | op term | blocker | verdict |
|---|---|---|---|
| `l16_jsr_initial_stack0_marker_0a` (ccfe8ffd) | OP_JSR×50 = 860 | **−1e6** | hard NOT-blocker |
| `l16_ent_nested_bp_byte0_d8` (e2e334c8) | OP_ENT×100 = 1140 | **−1e9** | hard NOT-blocker |
| `l16_ent_nested_sp_byte0_d8` (f17b3a14) | OP_ENT×10 = 114 | **−1e6** | hard NOT-blocker |
| L9 ENT hi-nibble band (f04e9f02) | ENT×1 | HAS_SE gate | scoped to first step |
| `tail_sp_marker_byte0_f8` (0dd14aff) | OP_JSR/ENT | **−1e9/−1e11** | hard NOT-blocker |

The other (c) families are safe by the same mechanism (hard NOT-blocker present)
or because the broadcast simply cannot clear threshold past the co-fire blockers
(e.g. `l16_lev_ax_full` OP_LEV×1=5 vs blockers 18, thr 12 → −13).

---

## Cross-check (spec_k=0, representative programs)

1. **The 5 fixed rules land in (c)-safe** — the classifier reproduces every
   already-resolved instance as safe via its hard `≥1e5` NOT-blocker. This is
   the strongest validation: the same algebra that classes the fixed rules safe
   classes the unfixed low-weight L6 routes vulnerable. (All 5 commits report
   smoke unchanged 24/8-of-8, confirming the hard-blocker promotion is
   byte-identical at the legit firing row.)

2. **OP_JSR/OP_ENT broadcast is row-uniform and persistent** — measured 15.5/7.3
   at L20-input AND identical at block 33 (the late residual). The flag is not a
   transient; it is read at the same magnitude wherever a frame/control rule
   fires, so a vulnerable rule mis-fires deterministically (matching the
   "value-independent across the var/func prologue cluster" observation in
   commit f04e9f02).

3. **`l6_jsr_ax_to_output` is the borderline case** — the VAR baseline doc
   already empirically traced the step-0 JSR→BP prologue leak to the L6 JSR
   route family (genesis "physical block 6 / L6, first amplification past unity
   at the REG_PC marker row"). That family is the same `make_layer6_*` factory
   carrying `l6_jsr_ax_to_output` / `l6_jsr_sp_fixup`. Whether each individual
   L6 route currently mis-fires depends on whether `MARK_AX` bleeds onto the AX
   byte rows; the static model brackets it (a)/(b). Recommend the fix agent
   confirm with `probe_var_genesis.py` at the AX byte rows before/after the
   hard-blocker promotion.

---

## Part C — proposed systematic fix

**Pattern (the one the chain already proved on 5 rules):** promote the
`IS_BYTE` + every non-target `MARK_*` / `H*` / wrong-`BYTE_INDEX` blocker on each
(a) and (b) family to a hard NOT-blocker `−1e6` (and bump `max_abs_weight` to
`1e6`+ on those rules). At the rule's *legitimate* firing row those blockers are
0, so the bake is byte-identical there; at every other row they decisively veto
regardless of the opcode broadcast magnitude. This makes the rule's scope
robust to **any** future growth of the opcode flag, not just the currently
measured 15-17.

Where a rule legitimately MUST fire on every step of its opcode at a specific
marker (the (b) families with "co-requires non-opcode positive" and **no**
blocker at all — `l9_alu_*_clear`, `l6_ent_sp_writeback`, `l6_adj_sp_writeback`,
`l16_lev_pc_temp`, `l8_alu_lev_b*`), the fix is instead to **add** the missing
hard `IS_BYTE`/non-target-`MARK_*` NOT-blockers so the co-required positive is
no longer the *only* thing keeping the broadcast off the wrong rows.

### Alternative (not recommended as the primary)

Scope the opcode condition to its own marker row (`OP_<X>` AND `MARK_<own>`).
This is more surgical but must be done per-rule and risks desyncing the frame
(the chain's 3 prior writer-based ENT-SP attempts desynced — see f17b3a14). The
hard-NOT-blocker promotion is the proven, low-risk path and is subtractive only.

### Rule count + cluster leverage

* **2-4 (a)** + **14 (b)** families = **16-18 families** to harden, expanding to
  ~**110-150 lowered rules** (16×/32× bands). All live in the L6 AX/SP-route, L9
  ALU-clear, L16 LEV/ENT-frame, and L8 ALU-LEV clusters.
* **Leverage:** every one of these gates the **JSR→ENT frame prologue** that
  opens all ~525 func/var/loop/rec/gcd programs (clusters A/B/F/G/J in the 1096
  triage). The 5 already-fixed links sit in the same prologue; hardening the
  remaining 16-18 families closes the class so the prologue chain cannot reopen
  it at a new link. Flat (no-frame) programs are unaffected (opcode broadcast 0
  for their ALU/IMM/IO opcodes).
* **Priority order** (by blast radius): (1) the 6 `l6_ent_after_jsr_*` + 2
  `l6_ent/adj_sp_writeback` families (ENT/ADJ broadcast 11/5, no blocker, sit on
  the live prologue); (2) `l16_lev_stack0_byte0_preserve` (wins by only 0.5);
  (3) `l9_alu_*_clear` (op_total 39, no blocker — highest absolute exposure if
  the co-required positive ever leaks); (4) the 3 L6 (a) routes.

---

## Tooling (committed, read-only)

* `tools/audit_opcode_broadcast.py <ids…>` — spec_k=0 broadcast-magnitude table
  per opcode × row type for a program. Reads dims via `model.dim_positions`.
* `tools/scan_opcode_broadcast_rules.py [--families] [--csv]` — static algebra
  scan (no model build): collects all FFNRules, computes the mis-fire score vs
  threshold under the measured broadcast, classifies (a)/(b)/(c). `--families`
  collapses band/lane expansion to rule families; `--csv` emits machine-readable
  rows. Broadcast magnitudes are pinned at the top (`BROADCAST = {…}`) from the
  audit tool — update them if re-probed.
