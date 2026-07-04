# Semantic Spec — the DECODE family

**Status:** DESIGN / READ-only inventory (100%-derivable architecture prep, parallel
to the generic-lowering engine pilot #391).
**Golden gate:** no weight change. `tools/_isa_golden_hash.py` ==
`81557d21422f3eada0a87c677b00dced41cc26c3ee3bfb094c5eeb71c9b4d3cb` (`81557d21`)
before and after this doc — this file only READS ops and WRITES documentation.

This document maps the CURRENT hand-authored lowering that DECODES each opcode
byte and sets its `OP_<NAME>` marker (logical **L5**, `neural_vm/unified_compiler/ops/l5_ops.py`),
expresses that behavior as **semantic-spec DATA** (a table), and produces the
**derivation GAP-LIST** — what (if anything) about decode would resist a generic
"table → FFN rules" lowering engine.

TL;DR: **Decode is a lookup.** The main decode table is a mechanical opcode-byte
→ two-nibble one-hot AND → write-`OP_<NAME>` mapping; ALL 34 main rules are
programmatically verified to equal `(opcode & 0xF, (opcode >> 4) & 0xF)`. A
generic engine reproduces the entire main band from the ISA opcode table with
zero per-opcode special-casing. The residue that resists a naive table is
**structural, not per-opcode**: three EXTRA decode bands (first-step-at-PC,
all-step-at-PC, TEMP-clear) and one JSR quirk (JSR sets a scratch `TEMP[0]`
IS_JSR flag, not `OP_JSR`, at PC), which are attributes of the *marker/step
context* the decode fires in, not of any single opcode.

---

## 0. Where decode lives (block / layer)

* **Logical layer:** L5. L5 maps **1:1** to **physical block 5** (only L8/L14/L25
  expand into extra physical blocks; see `docs/PROBE_GROUNDTRUTH_2026_06_10.md`).
* **File:** `neural_vm/unified_compiler/ops/l5_ops.py`.
* Decode has **two cooperating stages** in that one block:
  * **L5 attention (`layer5_fetch`, 8 heads, block[5].attn)** — reads the opcode
    *byte* out of immutable CODE memory (by `ADDR_KEY` content-match on PC / PC+K)
    and writes its two nibbles into `OPCODE_BYTE_LO` / `OPCODE_BYTE_HI`. This is
    the *fetch* half; it does NOT set any `OP_<NAME>` marker.
  * **L5 FFN (`opcode_decode_ffn`, block[5].ffn)** — the actual DECODE: reads
    `OPCODE_BYTE_LO/HI` one-hot nibbles and writes the `OP_<NAME>` marker (and a
    couple of scratch flags). This is the subject of this spec.

The FFN owns the entire L5 FFN footprint: **89 hidden units** (90 with the
default-ON `C4_NESTED_JSR_PC_FIX`). Layout table: `_FETCH_FFN_UNIT_LAYOUT`
(l5_ops.py:180).

---

## 1. INVENTORY — which rules/heads recognize the opcode byte + set each marker

### 1.a Fetch attention (produces the opcode byte) — `layer5_fetch`

* **Op factory:** `make_fetch_op`; head specs in `_fetch_head_specs`, now DERIVED
  from `_l5_fetch_specs()` via the generic `fetch(FetchSpec(...))` FETCH primitive
  (see gap-list G6 — RESOLVED). Zero hand-authored Q/K/V/O.
* **8 attention heads (0..5 production + 6/7 deleted 2026-05-11).** Each head
  content-matches an address (PC or PC+K, per `ADDR_KEY`) against the immutable
  per-CODE-position `ADDR_KEY` and copies that CODE slot's `CLEAN_EMBED_LO/HI`
  (the opcode/immediate byte nibbles). Roles:
  * Head 0 — non-first-step immediate fetch @AX from `TEMP` (=PC+1) → `FETCH_LO/HI`.
  * Head 1 — non-first-step **opcode** fetch @AX from relayed PC → `OPCODE_BYTE_LO/HI`.
  * Head 2 — first-step **opcode** fetch @PC (`HAS_SE==0`) → `OPCODE_BYTE_LO/HI`.
  * Head 3 — dynamic immediate fetch @PC → `FETCH_LO/HI`.
  * Head 4 — first-step **opcode** fetch @AX → `OPCODE_BYTE_LO/HI`.
  * Head 5 — non-first-step **opcode** fetch @PC → `OPCODE_BYTE_LO/HI`.
* **Not per-opcode.** The fetch is opcode-*agnostic*: it copies whatever byte is
  at the matched CODE position. No `OP_<NAME>` is produced here. (Constants:
  `PC_OFFSET=2`, `INSTR_WIDTH=8`.)

### 1.b Decode FFN (sets the `OP_<NAME>` markers) — `opcode_decode_ffn`

* **Op factory:** `make_opcode_decode_ffn_op` (l5_ops.py:611); full rule sequence
  `_opcode_decode_ffn_rules` (l5_ops.py:1022). Reads
  `OPCODE_BYTE_LO(.prev)/OPCODE_BYTE_HI, MARK_AX, MARK_PC, HAS_SE`; writes the 34
  `OP_*` dims + `TEMP`.
* Four rule sub-banks + one reserved blank, in this exact unit order:

| Units | Sub-bank | Builder (l5_ops.py) | #rules | Gate / context | Writes |
|-------|----------|---------------------|--------|----------------|--------|
| 0..33 | main-at-AX | `_opcode_decode_main_rules` (766) | 34 | `gate=MARK_AX` (mult.) | `OP_<NAME>` (JSR→`OP_JSR`) |
| 34..51 | first-step-at-PC | `_opcode_decode_first_step_rules` (836) | 18 | conds incl. `MARK_PC +1`, `HAS_SE −1` | `OP_<NAME>`, **JSR→`TEMP+0`** |
| 52 | reserved blank | `_opcode_decode_jsr_temp0_blank_rule` (989) | 1 | — (no-op unit) | none |
| 53..83 | temp-clear-at-PC | `_opcode_decode_temp_clear_rules` (896) | 31 | `gate=TEMP+k` (−1), cond `MARK_PC` | `TEMP+k` (k=1..31) |
| 84..88 | all-step-at-PC | `_opcode_decode_all_step_pc_rules` (921) | 5 | conds incl. `MARK_PC +1` | `OP_<NAME>` (BZ/BNZ/LEV/EXIT/JMP) |
| 89 | all-step-JSR-at-PC | `_opcode_decode_all_step_jsr_rules` (958) | 0 or 1 | conds incl. `MARK_PC +1`; flag `C4_NESTED_JSR_PC_FIX` | **`TEMP+0`** (JSR only) |

  Only the **main-at-AX** bank (units 0..33) is the actual "opcode-byte → marker"
  decode. The other three OP-writing banks re-decode a *subset* at the PC marker
  for step-framing reasons (see gap-list §3). Rule-name templates:
  `l5_decode_{op}_at_ax`, `l5_first_step_decode_{op}`,
  `l5_all_step_decode_{op}_at_pc`, `all_step_decode_jsr_temp0_at_pc`.

* **Dep anchors** (topology only, no weights): `make_fetch_dep_anchor_op`
  (`_layer5_fetch_dep_anchor`), `make_opcode_decode_ffn_dep_anchor_op`
  (`_opcode_decode_ffn_dep_anchor`).

* **Not part of decode:** the `#221` consumer-lookahead ops at the bottom of
  l5_ops.py (`make_lookahead_*`, `make_next_arith_*`, `make_prior_arith_latch_op`,
  `make_dump_block_flag_op`, flag `C4_STACK0_NEXT_ARITH`) REUSE the two-nibble AND
  decode pattern to classify the *next* instruction, but they feed a STACK0-dump
  gate, not the `OP_<NAME>` markers. They are out of scope for the DECODE family
  (listed here so a future engine doesn't double-count them).

### 1.c Per-opcode LOC (main-at-AX decode)

Every opcode's main rule is generated by ONE loop over the `opcodes` list at
`_opcode_decode_main_rules` (l5_ops.py:780-815). There is **no per-opcode LOC** —
all 34 rules share the identical `multi_way_and_rule(...)` template
(l5_ops.py:818-833); the only per-opcode data is the `(Opcode, lo, hi)` triple
and the derived `OP_<NAME>` write. LOC per opcode = one table row.

---

## 2. SEMANTIC-SPEC DATA — decode as a table

### 2.a Schema

Minimal fields a generic engine needs to reproduce the **main-at-AX** decode:

| Field | Meaning | Source of truth |
|-------|---------|-----------------|
| `opcode_byte` | the 8-bit opcode value | `neural_vm.embedding.Opcode.<NAME>` |
| `lo_nibble` | `opcode_byte & 0xF` (derived) | — |
| `hi_nibble` | `(opcode_byte >> 4) & 0xF` (derived) | — |
| `marker_dim` | output marker to set | `dim_ref("opcode_flag", NAME)` → `OP_<NAME>` |
| `gate` | marker context this fires under | `dim_ref("marker", "AX")` |
| `block` | physical block | 5 (logical L5) |
| `threshold` | AND threshold | `1.5` (two +1 conds; "one missing" = 1.0 < 1.5) |
| `write_weight` | marker strength | `10.0 / S` |

The `lo/hi/threshold/gate/block/write_weight` columns are **constant across all
34 opcodes** — the ONLY per-row data is `(opcode_byte, NAME)`. So the semantic
spec collapses to the ISA opcode table itself:

```
DECODE_MAIN[op] :=  (OPCODE_BYTE_LO + (op & 0xF)) AND (OPCODE_BYTE_HI + (op>>4 & 0xF))
                    gated MARK_AX  →  write OP_<name(op)> = 10/S     @ block 5
```

### 2.b Table (main-at-AX; programmatically verified — see §4)

| opcode | byte | lo | hi | marker_dim | gate | thr | unit |
|--------|-----:|---:|---:|-----------|------|----:|-----:|
| LEA | 0 | 0 | 0 | OP_LEA | MARK_AX | 1.5 | 0 |
| IMM | 1 | 1 | 0 | OP_IMM | MARK_AX | 1.5 | 1 |
| JMP | 2 | 2 | 0 | OP_JMP | MARK_AX | 1.5 | 2 |
| JSR | 3 | 3 | 0 | OP_JSR | MARK_AX | 1.5 | 3 |
| BZ  | 4 | 4 | 0 | OP_BZ  | MARK_AX | 1.5 | 4 |
| BNZ | 5 | 5 | 0 | OP_BNZ | MARK_AX | 1.5 | 5 |
| ENT | 6 | 6 | 0 | OP_ENT | MARK_AX | 1.5 | 6 |
| ADJ | 7 | 7 | 0 | OP_ADJ | MARK_AX | 1.5 | 7 |
| LEV | 8 | 8 | 0 | OP_LEV | MARK_AX | 1.5 | 8 |
| LI  | 9 | 9 | 0 | OP_LI  | MARK_AX | 1.5 | 9 |
| LC  | 10 | 10 | 0 | OP_LC  | MARK_AX | 1.5 | 10 |
| SI  | 11 | 11 | 0 | OP_SI  | MARK_AX | 1.5 | 11 |
| SC  | 12 | 12 | 0 | OP_SC  | MARK_AX | 1.5 | 12 |
| PSH | 13 | 13 | 0 | OP_PSH | MARK_AX | 1.5 | 13 |
| OR  | 14 | 14 | 0 | OP_OR  | MARK_AX | 1.5 | 14 |
| XOR | 15 | 15 | 0 | OP_XOR | MARK_AX | 1.5 | 15 |
| AND | 16 | 0 | 1 | OP_AND | MARK_AX | 1.5 | 16 |
| EQ  | 17 | 1 | 1 | OP_EQ  | MARK_AX | 1.5 | 17 |
| NE  | 18 | 2 | 1 | OP_NE  | MARK_AX | 1.5 | 18 |
| LT  | 19 | 3 | 1 | OP_LT  | MARK_AX | 1.5 | 19 |
| GT  | 20 | 4 | 1 | OP_GT  | MARK_AX | 1.5 | 20 |
| LE  | 21 | 5 | 1 | OP_LE  | MARK_AX | 1.5 | 21 |
| GE  | 22 | 6 | 1 | OP_GE  | MARK_AX | 1.5 | 22 |
| SHL | 23 | 7 | 1 | OP_SHL | MARK_AX | 1.5 | 23 |
| SHR | 24 | 8 | 1 | OP_SHR | MARK_AX | 1.5 | 24 |
| ADD | 25 | 9 | 1 | OP_ADD | MARK_AX | 1.5 | 25 |
| SUB | 26 | 10 | 1 | OP_SUB | MARK_AX | 1.5 | 26 |
| MUL | 27 | 11 | 1 | OP_MUL | MARK_AX | 1.5 | 27 |
| DIV | 28 | 12 | 1 | OP_DIV | MARK_AX | 1.5 | 28 |
| MOD | 29 | 13 | 1 | OP_MOD | MARK_AX | 1.5 | 29 |
| EXIT | 38 | 6 | 2 | OP_EXIT | MARK_AX | 1.5 | 30 |
| NOP | 39 | 7 | 2 | OP_NOP | MARK_AX | 1.5 | 31 |
| PUTCHAR | 65 | 1 | 4 | OP_PUTCHAR | MARK_AX | 1.5 | 32 |
| GETCHAR | 64 | 0 | 4 | OP_GETCHAR | MARK_AX | 1.5 | 33 |

Every `lo == byte & 0xF` and `hi == (byte >> 4) & 0xF` (checked in §4).

### 2.c The three EXTRA decode bands (context re-decode)

These re-decode a subset of opcodes at the **PC** marker instead of AX, for
step-framing reasons (they are not "extra opcodes" — they are the same opcodes in
a different marker context). Same schema, different `gate`/`context` and threshold:

* **first-step-at-PC** (18 rules, units 34..51): conditions
  `OPCODE_BYTE_LO+lo, OPCODE_BYTE_HI+hi, MARK_PC +1, HAS_SE −1`, `threshold=2.5`.
  Covers `{JMP, JSR→TEMP+0, IMM, LEA, EXIT, NOP, ADD, SUB, MUL, DIV, MOD, OR,
  XOR, AND, EQ, LT, SHL, SHR}`. **JSR here writes `TEMP+0` (IS_JSR scratch), not
  `OP_JSR`.**
* **all-step-at-PC** (5 rules, units 84..88): conditions
  `OPCODE_BYTE_LO+lo, OPCODE_BYTE_HI+hi, MARK_PC +1`, `threshold=2.5`. Covers
  `{BZ, BNZ, LEV, EXIT, JMP}` (the opcodes whose PC-marker copy must also fire on
  the AX step).
* **all-step-JSR-at-PC** (1 rule, unit 89, flag `C4_NESTED_JSR_PC_FIX` default-ON):
  `OPCODE_BYTE_LO+3, OPCODE_BYTE_HI+0, MARK_PC +1`, `threshold=2.5`, writes
  `TEMP+0`. Root-B nested-JSR fix (mirrors all-step for JSR's IS_JSR flag on every
  step, not just first).
* **temp-clear-at-PC** (31 rules, units 53..83): NOT a decode — `gate=TEMP+k`
  (weight −1), cond `MARK_PC`, writes `TEMP+k` (k=1..31). Scratch-slot hygiene,
  opcode-independent. Reserved blank unit 52 keeps `TEMP+0` for the JSR IS_JSR flag.

Opcodes present ONLY at main-at-AX (never re-decoded at PC): `ADJ, BNZ*, BZ*,
ENT, GE, GETCHAR, GT, LC, LE, LEV*, LI, NE, PSH, PUTCHAR, SC, SI` (`*` = also in
all-step-at-PC, so those three ARE at PC via the all-step bank; the true
AX-only-forever set is `ADJ, ENT, GE, GETCHAR, GT, LC, LE, LI, NE, PSH, PUTCHAR,
SC, SI`).

---

## 3. GAP-LIST — what resists generic derivation

Classification convention: **[spec-expressiveness]** = a field the spec table
must carry to reproduce the behavior; **[lowering-generality]** = whether the
generic "table → FFN rules" engine can emit it without opcode-specific code.

### G1. Main-at-AX decode — FULLY DERIVABLE. [lowering-generality: clean]
The 34 main rules are a pure lookup: `(opcode & 0xF, (opcode>>4)&0xF)` two-nibble
one-hot AND, `gate=MARK_AX`, `write OP_<name> = 10/S`, `threshold=1.5`. Every
per-row field except `(byte, NAME)` is constant, and `(byte, NAME)` comes straight
from `Opcode`. **A generic engine reproduces all 34 with zero special-casing** —
this is the decisive "decode is a lookup" result.

### G2. Marker-context is a spec field, not per-opcode. [spec-expressiveness]
There are 3 decode *contexts* — `at_ax` (all 34), `first_step_at_pc`
(18 subset, `HAS_SE==0`), `all_step_at_pc` (5 subset). The engine derives these
from a small **band table** (context → {gate terms, threshold, HAS_SE polarity,
opcode-subset}), NOT from per-opcode logic. The subsets themselves need to be
declared (which opcodes re-decode at PC and in which band) — this is genuine
spec DATA, but it is ~3 short lists, not 34 special cases. It does reflect a
real semantic asymmetry (first-step vs all-step vs AX-only) that the engine
cannot *infer* from the ISA opcode table alone; it must be given.

### G3. JSR writes a scratch flag, not its `OP_` marker, at PC. [spec-expressiveness]
The one genuine per-opcode quirk: at the AX marker JSR writes `OP_JSR` (uniform),
but at the PC contexts (first-step + all-step-JSR) JSR writes **`TEMP+0`**
(the IS_JSR scratch flag the model_ops JSR-PC-override reads), NOT `OP_JSR`.
The spec must carry a per-(opcode, context) `write_override` for this one cell
(`JSR @ *_at_pc → TEMP+0`). Everything else in the band uses the uniform
`OP_<NAME>` write. A generic engine handles it with a single optional
`write_dim` override column (default `OP_<NAME>`); it is expressible, but it is
the one place the naive "write OP_<NAME>" rule is wrong.

### G4. TEMP-clear band is hygiene, not decode. [lowering-generality: separate primitive]
Units 53..83 (`TEMP[1..31]` clear at MARK_PC, `gate=TEMP+k` weight −1) and the
reserved blank unit 52 are scratch-slot maintenance, opcode-independent. They are
NOT part of the opcode→marker map and should be modeled by a separate generic
primitive ("clear scratch band k=1..31 at PC, reserving slot 0"), not the decode
table. A DECODE engine that only emits the OP-marker banks must still coexist with
this band in the same FFN block (unit-layout ordering matters for byte-identity —
see `_FETCH_FFN_UNIT_LAYOUT`).

### G5. Unit ORDER / layout is byte-identity-load-bearing. [lowering-generality]
The 4 OP-banks + blank + temp-clear must lower in EXACTLY the declared unit order
(`_FETCH_FFN_UNIT_LAYOUT`, asserted in the bake at l5_ops.py:651) for the golden
hash to hold. A generic engine reproduces byte-identity ONLY if it emits banks in
this order with the reserved blank at unit 52. This is a lowering constraint, not
a spec-expressiveness gap — the *behavior* is order-independent (rules are gated
disjointly), but the *weights* are not.

### G6. Fetch attention is a separate (non-decode) primitive. [scoping] — RESOLVED
Producing `OPCODE_BYTE_LO/HI` (the byte the decode reads) is done by 6 fetch
attention heads with per-role Q/K/V wiring (first-step vs non-first-step, PC vs
AX marker, immediate vs opcode). These are NOT opcode-parameterized and are NOT
derivable from the opcode table — they are addressing/fetch machinery shared by
ALL opcodes. A DECODE engine should treat `OPCODE_BYTE_LO/HI` as a given INPUT
and leave fetch to a separate "instruction fetch" spec family.

**RESOLVED (FETCH primitive):** that separate "instruction fetch" spec family
now EXISTS — `isa_semantics_dsl.fetch(FetchSpec(...))`. Every fetch head is a
CONTENT-ADDRESS COPY: Q builds a byte-address (`addr_mode="dynamic"` from a
per-nibble residual band, or `"static"` from the compile-time `PC_OFFSET`), K
matches the immutable per-CODE-position `ADDR_KEY`, and V/O copy that slot's
`CLEAN_EMBED` nibbles into a target byte band. The only VARYING data is the
address source, the marker (`MARK_AX`/`MARK_PC`), the top-nibble match mode
(`"dynamic"` / `"static"` / `"static_zero_single"`), the per-step `HAS_SE` gate
(`"first"` / `"non_first"` / `None`), and the target band + O-scale — carried on
`FetchSpec`. All 6 L5 fetch heads (`l5_ops._fetch_head_specs`, via
`_l5_fetch_specs()`) are DERIVED from it with ZERO hand-authored Q/K/V/O
construction, and the same primitive re-expresses the #221 consumer-lookahead
opcode-fetch head (7 heads total, one primitive). Byte-identical: golden
`91f55411` unchanged; standing test `test_fetch_reexpresses_live_l5_fetch_heads`
(+ lookahead) in `tests/test_isa_semantics_dsl.py`.

The L4 `pc_relay` heads (0/1) are NOT fetch heads and do NOT fit this primitive:
they are a MARKER-TO-MARKER band relay (Q anchors the AX marker, K matches the PC
marker positionally — NO `ADDR_KEY` content match, NO address projection), copying
`EMBED_LO/HI` + `ADDR_KEY`-top from the PC row to the AX marker/byte rows. That is
a distinct ISA-semantic family (a positional band relay, sibling to
`marker_broadcast` but firing at a marker row rather than byte positions); L4 has
no opcode-fetch / threshold / lookback attention head to derive via `fetch`.

### G7. Weight scalars are uniform constants. [spec-expressiveness: trivial]
`write_weight = 10.0/S`, condition weight `1.0`, `threshold ∈ {1.5 (AX), 2.5
(PC)}`, `HAS_SE = −1` (first-step only). All are context constants, not
per-opcode. The spec carries them once per band. No gap.

---

## 4. Verdict + reproducibility

**Decode IS cleanly, generically derivable.** The opcode→marker core (the 34
main-at-AX rules) is a pure lookup with zero per-opcode special-casing; a generic
engine reproduces it from the `Opcode` table plus a one-line rule template. The
only residue is:
* a **band table** (3 contexts × opcode-subset + gate/threshold constants) — G2, small DATA;
* **one** per-(opcode,context) write override (**JSR @ PC → TEMP+0**) — G3;
* co-tenant primitives that are NOT decode (TEMP-clear G4, fetch G6) and a
  byte-identity unit-ordering constraint (G5).

None of these is a per-opcode *arithmetic* or *control* special case — decode
carries no opcode-specific behavior beyond "this byte lights this marker in this
context." This is the strongest-case member of the derivability program: if the
generic lowering engine can express (a) a two-nibble one-hot AND from a scalar
opcode value, (b) a per-context gate/threshold band, and (c) a single write-dim
override cell, it reproduces the **entire** DECODE family byte-identically.

### Verification performed (READ-only; no weights touched)
* All 34 main rows satisfy `lo == byte & 0xF` and `hi == (byte>>4)&0xF`
  (programmatic check over `Opcode` + the l5_ops table → ALL MATCH).
* Rule counts confirmed live: `_opcode_decode_ffn_rules(100.0)` → 90 rules
  (34 main + 18 first-step + 1 blank + 31 temp-clear + 5 all-step-pc + 1
  all-step-jsr, `C4_NESTED_JSR_PC_FIX` default-ON).
* Example rule shape confirmed: `l5_decode_op_lea_at_ax` → conds
  `(OPCODE_BYTE_LO+0, OPCODE_BYTE_HI+0)`, `threshold=1.5`, `gate=MARK_AX`,
  `writes=(OP_LEA, 0.1)` (= 10/S at S=100).
* Golden hash unchanged: `81557d21` (this doc reads ops + writes markdown only).

### Feeds
This spec is the input to the generic lowering engine (pilot #391 / the CONTROL
family map #392). The DECODE band table + write-override column is the reference
schema the engine's opcode-family lowering must satisfy.
