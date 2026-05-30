# L10 tail_mem_store_addr0_* Family — Architectural Refactor Plan

Branch base: `speedup-cache-and-buckets` @ `4d069f7`
Target file: `c4_release/neural_vm/unified_compiler/ops/l10_ops.py`
Family location: lines ~3450-4100 (block returned by `_tail_bit32_result_correction_rules`)
Sentinel: 3/32 baseline preserved.

------------------------------------------------------------------
## 1. Family inventory

Eleven rules participate in the `tail_mem_store_addr0_*` / `tail_sp_marker_*`
discrimination problem. They are co-located in
`_tail_bit32_result_correction_rules` and all fire on `MARK_MEM` (or `MARK_SP`)
marker rows after the post-op tail. Each one writes a single byte value into
OUTPUT byte 0 using `Primitives.byte_value_writes(value, strength=...)`.

| # | Rule name                                                       | Line  | Output byte | Strength    | Helper                | Notes |
|---|-----------------------------------------------------------------|-------|-------------|-------------|-----------------------|-------|
| A | tail_sp_marker_byte0_f8_from_initial_stack_exact                | 3465  | 0xF8 (SP)   | active_value=4 / max=1e9 | exact_output_byte_rules | SP marker; B2-B noted it fires when MARK_SP residual >1.0 |
| B | tail_sp_byte1_ff_from_initial_stack_exact                       | 3503  | 0xFF (SP b1)| active=50   | exact_output_byte_rules | SP marker byte 1 — sibling |
| C | tail_mem_store_addr0_f8_exact                                   | 3540  | 0xF8        | strength=10_000 | constant_write    | PSH-store address byte 0=0xF8 (stack top, no SE evidence required) |
| D | tail_mem_store_addr1_ff_from_stack_store_exact                  | 3569  | 0xFF (b1)   | active=500  | exact_output_byte_rules | MEM byte 1 — uses CLEAN_EMBED gating |
| E | tail_mem_store_addr0_f8_initial_jsr_exact                       | 3604  | 0xF8        | active=5000 / max=1e9 | exact_output_byte_rules | OP_JSR variant, no HAS_SE; pairs with F |
| F | tail_mem_store_addr0_f8_initial_jsr_authority                   | 3636  | 0xF8        | strength=50_000 | constant_write    | OP_JSR sibling authority — same conditions as E |
| G | tail_mem_store_addr0_f0_exact                                   | 3666  | 0xF0        | strength=1_000_000 | constant_write | distinguishes 0xF0 from 0xF8 by `ALU_LO+14` rather than `ALU_LO+2` |
| H | tail_mem_store_addr0_00_from_global_exact                       | 3696  | 0x00        | strength=1_000_000_000 | constant_write | Global address, blocks by `PSH_AT_SP`/`OP_JSR`. B2-A/B3-η noted false-fire at step5 |
| I | tail_mem_store_addr2_zero_from_global_exact                     | 3731  | 0x00 (b2)   | active=500/max=1e9 | exact_output_byte_rules | Sibling byte 2 for global addr |
| J | tail_mem_store_addr3_zero_from_global_exact                     | 3774  | 0x00 (b3)   | active=500/max=1e9 | exact_output_byte_rules | Sibling byte 3 for global addr |
| K | tail_mem_store_addr0_f8_from_mod_local_exact                    | 3817  | 0xF8        | active=500  | exact_output_byte_rules | Local-mod variant: ALU_LO+7 positive; ALU_LO+10/14 negative |
| L | tail_mem_store_addr0_e0_from_local_offset_exact                 | 3853  | 0xE0        | active=50000 | exact_output_byte_rules | Local-offset variant: ALU_LO+8 positive |
| M | tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority       | 3889  | 0xE0        | strength=5_000_000_000 | constant_write | PSH-at-SP variant; B3-η noted overpowers ENT-main wrongly |
| N | tail_mem_store_addr0_e0_from_jsr_local_exact                    | 3924  | 0xE0        | active=5000 | exact_output_byte_rules | JSR-local variant (uses CMP+4 positive, OP_ENT negative) |
| O | tail_mem_store_addr0_e0_from_jsr_local_strong                   | 3958  | 0xE0        | active=5000 | exact_output_byte_rules | Identical conditions to N — pure strength sibling |
| P | tail_mem_store_addr0_e8_from_nested_local_exact                 | 3992  | 0xE8        | active=500  | exact_output_byte_rules | ALU_LO+10 positive, ALU_LO+7/14 negative |
| Q | tail_mem_store_addr0_e8_from_local_frame_addr_exact             | 4028  | 0xE8        | active=5000 | exact_output_byte_rules | **Uses ADDR_B0_LO+8 / ADDR_B0_HI+14** — the canonical evidence-based prototype |
| R | tail_mem_store_addr0_e8_from_local_frame_output_exact           | 4062  | 0xE8        | strength=500 | gated_write       | Gated by OUTPUT_HI+14; OP_JSR negative |

Effectively four byte values get written by ten "addr0" rules
(0x00 / 0xE0 / 0xE8 / 0xF0 / 0xF8) plus three SP/byte-1/2/3 cousins. Each rule
encodes its own ad-hoc proof for *why this MEM_STORE row should resolve to this
byte right now*, and the proofs overlap badly:

* All rules share the same `MARK_MEM` / `HAS_SE` / `H1+4` / `H1+10` / `MEM_STORE`
  base.
* Discrimination amongst E/F (0xF8 from JSR), G (0xF0), H (0x00), K (0xF8 mod
  local), L (0xE0 local), M (0xE0 PSH SP), N/O (0xE0 JSR local), P (0xE8),
  Q (0xE8 frame addr), R (0xE8 frame output) is done by **disjoint subsets of
  ALU_LO / CMP / PSH_AT_SP / OP_JSR / MEM_ADDR_SRC / OUTPUT_HI**.
* The intended discrimination is "which lane of `ADDR_B0_LO` and `ADDR_B0_HI`
  is one-hot active?" L13's `make_layer13_mem_addr_gather_op` already gathers
  this address into `ADDR_B0_LO / ADDR_B0_HI / ADDR_B1_LO / ...` for SI/SC/LI/LC.
  Rule Q (the lone ADDR-based rule) is the design exemplar.

------------------------------------------------------------------
## 2. Per-rule analysis (legitimate fire vs. wrong fire)

| Rule | Should fire on … | Wrongly fires on … (per evidence) | Truly discriminating evidence |
|------|------------------|-----------------------------------|-------------------------------|
| A (SP 0xF8) | First-step SP marker after initial stack frame is staged | Any step where `MARK_SP` >1.0 residual (B2-B) | `ADDR_B0_LO+8` & `ADDR_B0_HI+15` — but A predates L13 and uses OUTPUT/ALU witnesses |
| C (MEM 0xF8) | PSH/SI/SC store whose computed address is `…F8` (top of stack) | shares evidence with E/F (JSR variant) and K (mod-local) | `ADDR_B0_LO+8`,`ADDR_B0_HI+15` (positive) and the missing ones (negative) |
| E/F (MEM 0xF8 JSR) | JSR push of return PC, addr=…F8 | overlaps with C on JSR boundary | `OP_JSR` + `ADDR_B0_LO+8`,`ADDR_B0_HI+15` |
| G (MEM 0xF0) | Store of full-frame address (lo nibble 0) | distinguished from F8 only by ALU_LO+14 vs ALU_LO+2 | `ADDR_B0_LO+0`,`ADDR_B0_HI+15` |
| H (MEM 0x00) | Global address store (lo=0,hi=0) | B2-A: false-fires at step5 because `OUTPUT_LO+0`,`OUTPUT_HI+0` proxies hi nibble | `ADDR_B0_LO+0`,`ADDR_B0_HI+0` |
| K (MEM 0xF8 mod local) | Modified-local store, addr…F8 | overlaps with C/L by ALU_LO+7 vs +8 vs +10 | `ADDR_B0_LO+8`,`ADDR_B0_HI+15`, OP_ENT/OP_JSR negative |
| L (MEM 0xE0 local) | Local-offset store, addr…E0 (b0 lo=0 hi=14) | overlaps with M on PSH_AT_SP boundary | `ADDR_B0_LO+0`,`ADDR_B0_HI+14` |
| M (MEM 0xE0 PSH SP) | PSH at SP where addr…E0 | **B3-η: fires on ENT-main wrongly with strength 5e9 outvoting siblings** | `ADDR_B0_LO+0`,`ADDR_B0_HI+14` + (OP_PSH ∧ ¬OP_ENT) |
| N/O (MEM 0xE0 JSR local) | JSR push, addr…E0 | Identical pair — pure strength escalation | `OP_JSR`, `ADDR_B0_LO+0`,`ADDR_B0_HI+14` |
| P (MEM 0xE8 nested local) | Nested-local addr…E8 | overlaps with Q/R | `ADDR_B0_LO+8`,`ADDR_B0_HI+14` |
| Q (MEM 0xE8 frame addr) | Local frame addr…E8 | the canonical, sound rule | already correct |
| R (MEM 0xE8 frame output) | Same scenario, late fire | gated by OUTPUT_HI+14 residue | already partially evidence-based |

**Pattern**: rules added in chronological order each pick a "novel" combination
of ALU_LO/CMP/PSH_AT_SP/OP_JSR/MEM_ADDR_SRC to discriminate, then crank
`strength` until they win against the residual produced by previously-added
siblings. Rule Q shows the correct paradigm: read the L13 ADDR_B0 lanes
directly.

------------------------------------------------------------------
## 3. Proposed architectural refactor (recommended path)

### 3.1 Principle
Replace ten "addr0 → byte X" decisions with **one** dispatcher that linearly
combines (`ADDR_B0_LO+lo`, `ADDR_B0_HI+hi`) evidence with the
opcode/MEM-source-context evidence. Each address byte 0 in {0x00, 0xE0, 0xE8,
0xF0, 0xF8} maps to a (lo, hi) pair:

```
0x00 → (LO+0,  HI+0)
0xE0 → (LO+0,  HI+14)
0xE8 → (LO+8,  HI+14)
0xF0 → (LO+0,  HI+15)
0xF8 → (LO+8,  HI+15)
```

### 3.2 Concrete rewrite — five address-only rules

Replace rules C, E, F, G, H, K, L, M, N, O, P, Q, R with five
"addr0 byte X from ADDR_B0" constant-write rules with bounded strength
(target ≤ 1e6) and a uniform structural base:

```
tail_mem_store_addr0_from_ADDR_B0_xxx
  conditions:
    MARK_MEM         +1.0
    HAS_SE           +1.0  # (or OP_JSR for the JSR variants; see 3.3)
    H1+4             +20.0
    H1+1 / +2 / +3   -1e6     # not other family heads
    MEM_STORE        +5.0
    ADDR_B0_LO+<lo>  +50.0
    ADDR_B0_HI+<hi>  +50.0
    ADDR_B0_LO+<other lo lanes>   -200.0   # winner-take-all
    ADDR_B0_HI+<other hi lanes>   -200.0
    MARK_AX/PC/SP/BP/STACK0  -1e6
    NEXT_*           -1e6
  threshold:   140.0
  writes:      byte_writes(value, strength=10_000)
```

That single template (plus a `(lo, hi, value)` tuple per row) generates all
five address-byte variants. Strength is bounded at 10k because the
ADDR_B0_LO/HI lanes ARE the correct address and don't need to "outvote" — the
evidence is decisive.

### 3.3 JSR / PSH-at-SP variants
For the few cases where the ADDR_B0 gather hasn't completed (initial JSR push
through PSH-at-SP), retain a slim version of E/M with three changes:

1. Cap strength at 1e6.
2. Add **positive** `ADDR_B0_LO+lo / ADDR_B0_HI+hi` evidence as a SOFT signal
  (weight +5.0) — these arrive late from L13 but still strengthen the proof
  when present.
3. Add a hard `OP_ENT` negative gate (-1e6) on the PSH-at-SP rules. B3-η's
  evidence shows M wrongly fires on ENT-main because it doesn't disambiguate
  ENT from PSH.

### 3.4 SP marker rule A (`tail_sp_marker_byte0_f8_from_initial_stack_exact`)
Rule A is structurally similar but operates on `MARK_SP` instead of
`MARK_MEM`. The L13 ADDR_B0 gather does not run for SP-marker rows, so A
cannot be re-routed through ADDR_B0 directly. Instead, apply two minimal
fixes:

1. Add a hard `MARK_SP > 0.5` floor: `("MARK_SP", 10.0)` is already present;
  raise the threshold so residual MARK_SP <1.0 cannot fire (B2-B's regression).
2. Add evidence-based positive: when ADDR_B0_LO+8 / ADDR_B0_HI+15 ARE present
  on the row (which they will be after L13 completes for SI/SC), boost
  conviction; when absent, fall back to the OUTPUT_LO+8 / OUTPUT_HI+15
  witnesses currently used.

Rule B (`tail_sp_byte1_ff_from_initial_stack_exact`) is a pure byte-1 sibling
that uses BYTE_INDEX_1 — it is structurally distinct and can remain
untouched.

### 3.5 Byte-1/2/3 siblings (D, I, J)
Once ADDR_B0/B1/B2 are routed correctly for byte 0, the cousins for byte 1
(rule D), byte 2 (rule I), byte 3 (rule J) should be moved to use
`ADDR_B1_LO/HI`, `ADDR_B2_LO/HI` evidence with the same template.

### 3.6 Why this resolves the regressions
* **B3-η** ENT-main misfire of M (5e9 strength): with bounded strength and an
  explicit `OP_ENT -1e6` blocker, the rule cannot outvote the legitimate ENT
  pathway.
* **B2-A** step5 misfire of H: H currently uses `OUTPUT_LO+0 / OUTPUT_HI+0` as
  a *proxy* for the address byte. After step5 those OUTPUT lanes happen to
  fire for unrelated reasons. Switching to `ADDR_B0_LO+0 / ADDR_B0_HI+0`
  removes the proxy ambiguity.
* **B2-B** MARK_SP >1.0 misfire of A: tightening the MARK_SP floor and
  threshold removes residual-driven activation.
* **N/O** redundant siblings: one survives, the other is deleted.

------------------------------------------------------------------
## 4. Minimal implementation (concrete diff sketch)

Inside `_tail_bit32_result_correction_rules`, define a helper near the top of
the family block:

```python
def addr_from_l13_rules(
    *,
    name_prefix: str,
    target_byte: int,
    lo_lane: int,
    hi_lane: int,
    extra_conditions: tuple = (),
    threshold: float = 140.0,
    strength: float = 10_000.0,
) -> tuple[FFNRule, ...]:
    """Emit byte 0 for MEM-store rows directly from L13 ADDR_B0 lanes.

    Replaces the strength-based ad-hoc family by reading the L13 mem-addr
    gather one-hot. Strength bounded at 10k.
    """
    other_lo = tuple((f"ADDR_B0_LO+{k}", -200.0) for k in range(16) if k != lo_lane)
    other_hi = tuple((f"ADDR_B0_HI+{k}", -200.0) for k in range(16) if k != hi_lane)
    base = (
        ("MARK_MEM", 1.0),
        ("HAS_SE", 1.0),
        ("H1+4", 20.0),
        ("H1+1", -1_000_000.0),
        ("H1+2", -1_000_000.0),
        ("H1+3", -1_000_000.0),
        ("H1+10", -1_000_000.0),
        ("MEM_STORE", 5.0),
        (f"ADDR_B0_LO+{lo_lane}", 50.0),
        (f"ADDR_B0_HI+{hi_lane}", 50.0),
        ("IS_BYTE", -100.0),
        ("MARK_AX", -1_000_000.0),
        ("MARK_PC", -100.0),
        ("MARK_SP", -100.0),
        ("MARK_BP", -100.0),
        ("MARK_STACK0", -100.0),
        ("NEXT_PC", -1_000_000.0),
        ("NEXT_AX", -1_000_000.0),
        ("NEXT_SP", -1_000_000.0),
        ("NEXT_BP", -1_000_000.0),
        ("NEXT_STACK0", -1_000_000.0),
        ("NEXT_MEM", -1_000_000.0),
        ("NEXT_SE", -1_000_000.0),
    ) + other_lo + other_hi + extra_conditions
    return (
        FFNRule.constant_write(
            name=f"{name_prefix}_{target_byte:02x}",
            conditions=base,
            threshold=threshold,
            writes=byte_writes(target_byte, strength=strength),
        ),
    )
```

Use it to emit five replacement rules. Then **delete** rules C, G, H, K, L,
M, N, O, P (and downgrade strength on E/F to 1e5). Rules Q and R are already
ADDR-based and can stay as-is. Rule D/I/J get a similar ADDR_B1/B2 helper.

Approximate net change: ~600 lines replaced by ~80 lines + 5 rule instances
≈ 200 lines.

------------------------------------------------------------------
## 5. Validation strategy

### 5.1 Pre-implementation baseline (sentinel preservation)
Capture the 3/32 sentinel count using the spec command:

```bash
for spec in "0 32" "200 50" "425 25" "550 25" "822 50"; do
  ...
done | tee .agent-logs/l10-tail-family-refactor/baseline.log
```

### 5.2 Post-implementation
Rerun the same command. Acceptance criteria:
* 3/32 sentinel must hold or improve.
* B3-η's failing ENT-main case must improve to "pass" (M no longer
  out-strengths legitimate ENT pathway).
* B2-A's step5 misfire of H must improve (ADDR_B0_LO+0/HI+0 evidence is
  honest at step5).
* B2-B's MARK_SP residual misfire of A must remain fixed.

### 5.3 Smoke / unit
The smoke tests bound to `layer13_mem_addr_gather` exercise the
ADDR_B0 path:
* `TestSmokeMemory::test_sc_lc_roundtrip`
* `TestSmokeMemory::test_si_li_roundtrip`

Run these plus the broader tail-rule unit tests under
`c4_release/tests/` looking for any `test_tail_*` regression.

### 5.4 Decl-verifier
Use `decl_verifier.py` (per MEMORY.md feedback brief default) to inspect any
declarations-only regression.

------------------------------------------------------------------
## 6. Risk register

| Risk | Mitigation |
|------|-----------|
| ADDR_B0 lanes may not be reliably populated for non-SI/SC/LI/LC paths (PSH, JSR push of return PC) | Keep retained E/F/M variants with positive ADDR_B0 boost but fallback proofs. |
| Bounding strength to 10k may let some upstream tail rule outvote the new dispatcher | Audit other `tail_*` rules that write OUTPUT byte 0; the 10k strength is comparable to existing siblings like the 10_000 in current rule C. Adjust to 100_000 if necessary, still 4-5 orders of magnitude lower than the 5e9 outliers. |
| Behaviour for SP marker rule A diverges from MEM (no L13 ADDR for MARK_SP) | Keep A's OUTPUT/ALU witnesses; only tighten threshold. |
| Cascading reliance: changing E/F/H may shift downstream stale-OUTPUT cleanup rules | Each new rule keeps the same `byte_writes` interface, so downstream OUTPUT consumers see the same one-hot pattern. |
| Spec-generated declarative authority disallows arbitrary edits | All `_tail_bit32_result_correction_rules` are already authored in-tree (not spec-generated); declarative authority is `spec_generated` for the OWNING ops only. The tail FFN rules are owner-internal. |

------------------------------------------------------------------
## 7. Why this is the right architectural fix (per U13 recommendation)

U13's recommendation was: "the L10 tail rules should depend on L13's
ADDR_B0/ADDR_B1/ADDR_B2 as positive evidence, rather than hardcoded byte
values." This plan implements that recommendation exactly.

The L13 layer already computes the address via attention over STACK0/AX_CARRY
and writes one-hot lanes into ADDR_B0_LO/HI etc. (`make_layer13_mem_addr_gather_op`
in `c4_release/neural_vm/unified_compiler/ops/l13_ops.py`). The L10 tail
should therefore READ that one-hot rather than re-derive the byte from
disjoint ALU/CMP/PSH/OP witnesses. The current family is a historical
accretion that pre-dates the L13 gather being trustworthy.

Rule Q (`tail_mem_store_addr0_e8_from_local_frame_addr_exact`) at line 4028
is the proof of concept: it already reads `ADDR_B0_LO+8 / ADDR_B0_HI+14` and
behaves correctly. The refactor generalises that pattern to all sibling
byte values.

------------------------------------------------------------------
## 8. Delivery

This document is the deliverable for Path 1. No code is changed; the
sentinel baseline is preserved by construction. Implementation can be
sequenced as:

1. Land helper `addr_from_l13_rules` and one rule (the new 0x00 variant
  replacing H). Validate sentinel + B2-A step5 fix.
2. Land 0xE0 / 0xE8 / 0xF0 / 0xF8 replacements. Validate sentinel + B3-η ENT
  fix.
3. Delete redundant N/O sibling.
4. Tighten A's MARK_SP threshold. Validate B2-B fix.
5. Migrate D/I/J to ADDR_B1/B2 helper.

Each step is an independent commit that holds the sentinel.
