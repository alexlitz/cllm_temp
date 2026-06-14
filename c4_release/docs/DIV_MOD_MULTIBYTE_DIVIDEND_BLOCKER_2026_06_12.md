# Multi-byte-dividend DIV/MOD — root cause, boundary, and the multi-byte-division design (2026-06-12)

> **RESOLVED 2026-06-14 (commit `83ba5602`, flag `C4_DIV_MULTIBYTE`, default
> OFF — flag-off byte-identical).** The "architecture wall" below was WRONG on
> three counts (the brief's lead about a static-dim probe artifact was correct):
>
> 1. **"STACK0_BYTE_VAL_1 written nowhere" was a static-dim / wrong-band /
>    wrong-row artifact.** At BUILT `layout.dim_positions` (d_model=981,
>    spec_k=0) the high dividend byte IS stored: `1162/37` →
>    `STACK0_BYTE_VAL_1` (built dim **602**, NOT the static 734) = `0x04` at the
>    PSH-frame rows. The prior probes read the static registry dim and the wrong
>    `CLEAN_EMBED` band. Case **(a)**: stored, just not relayed into the operand
>    band the divide read.
> 2. **The long-division pipeline is ALREADY multi-byte capable.** Lookup-mode
>    `FlattenedDivMod` (`alu/ops/divmod_longdiv.py::LongDivisionModule`) reads the
>    dividend as a full 8-nibble GE vector and does a real MSB→LSB long division
>    (up to 32 bits). "Wall B" (no carry cascade in one FFN) does not apply — the
>    cascade lives inside the module.
> 3. **The actual root**: the smoke/groundtruth path builds `alu_mode='efficient'`
>    (`trust_neural_alu=True`), whose divmod install lowers
>    `wide_div_rules_ge_format(width_bytes=1)` — a flat 256×256 **single-byte**
>    lookup reading only `ALU_LO/HI` (byte 0). THAT is the single-byte truncation
>    (the 16.7M-rule "Wall A" was about extending *this lookup*, never necessary).
>
> **Fix (two flag-gated parts).** (i) `make_alu_divmod_composite_ops`: under
> `C4_DIV_MULTIBYTE`, build+install the multi-byte `FlattenedDivMod` long-division
> composite even in efficient mode (the install co-locates at the
> `layer10_carry_relay` anchor where the 3 stage bakes assemble it, so
> `builder.composite` is populated before append). (ii) `BDToGEConverter`: route
> the DIV/MOD operand-A byte-1 gather to `STACK0_BYTE_VAL_1_LO/HI` (the carrier
> `layer10_psh_ax_broadcast` populates at the cummax-picked `STACK0_BYTE1` row)
> instead of `CLEAN_EMBED_LO/HI` (= 0x00 there).
>
> **Result (spec_k=0):** `1162/37→31`, `300/5→60`, `280%6→4` (were 3/8/0);
> single-byte `84/2→42` unchanged. `run_1096_canonical --ids 150-249
> --criterion exit_code`: **div 21/50→48/50, mod 24/50→48/50** (45/100→96/100).
> `tests/test_smoke.py` with `C4_DIV_MULTIBYTE=1`: **51 passed / 0 failed**.
> Flag-off param hash == HEAD (byte-identical). The 4 residual fails are the
> documented operand-cleanliness outliers (the `~5.56` cell-0 artifact), not the
> multi-byte truncation. The rest of this doc is the SUPERSEDED pre-fix analysis.

Status: ~~**architecture wall** (operand-pipeline single-byte gather + flat-lookup
width wall)~~ **RESOLVED (see banner above)**. Companion to
[`DSL_W5_MULDIV_LIMIT.md`](DSL_W5_MULDIV_LIMIT.md),
[`LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md`](LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md),
and [`DIV_GE_FORMAT_INSTALL_BLOCKER_2026_06_10.md`](DIV_GE_FORMAT_INSTALL_BLOCKER_2026_06_10.md).

This doc records the **exact mechanism** behind the ~55 failing 1096 div/mod
programs (the large-dividend cluster), the precise pass/fail boundary, and what a
multi-byte-division extension would actually require. No code change ships — the
fix hits two compounding architecture walls, and the single-byte path that
already passes must not regress.

## The cluster and the exact boundary

`tools/run_1096_canonical.py --ids 150-249` (spec_k=0, suite-exact):
**div 21/50 pass, mod 24/50 pass** — 55 failures total (29 div + 26 mod).

Corpus generation (`tests/test_suite_1000.py`, seeded `Random(42)`):
- **div** (ids 150-199): `b ∈ [1,50]`, `a = b·[1..50] + [0..b-1]` → dividend `a`
  up to **2549**; quotient ≤ 50, divisor ≤ 50.
- **mod** (ids 200-249): `a ∈ [10,500]`, `b ∈ [2,20]` → dividend up to 500;
  remainder ≤ 19, divisor ≤ 20.

The boundary is **crisp and structural**:

| dividend `a` | bytes | verdict |
| --- | --- | --- |
| `a ≤ 255` (single byte) | 1 | **PASS** (20/21 div, 21/22 mod) |
| `a > 255` (multi-byte) | 2+ | **FAIL** (all 29 div, all 28 mod multi-byte) |

The fail set is **exactly** the multi-byte-dividend ids
(`div` = {150,151,153,154,158,160,164,165,166,167,168,169,170,173,174,176,181,
182,183,184,185,186,188,189,192,194,195,196,198};
`mod` analogous). The divisor, quotient, and remainder all fit in one byte for
the whole corpus — **only the dividend overflows the single byte the install can
see.**

## The truncation mechanism (verified, 52/55 exact)

For **52 of 55 failures** the neural output equals **`low_byte(dividend) op
divisor` exactly**:

```
id150  1162 / 37  exp=31  neural=3   ;  138 / 37 = 3    (138 = 0x048A & 0xFF)
id151   843 / 31  exp=27  neural=2   ;   75 / 31 = 2
id200   390 % 19  exp=10  neural=1   ;  134 % 19 = 1    (134 = 0x0186 & 0xFF)
```

The remaining 3 (id170, id201, id203) deviate by 1-2 cells — minor
operand-cleanliness perturbations of the `~5.56` cell-0 artifact
(`DIV_GE_FORMAT_INSTALL_BLOCKER`), not a different mechanism. id201 (`89 % 10`)
is the **only single-byte failure** (a≤255) and is in that secondary class.

### Where the dividend is truncated — two stacked causes

The L10 efficient-mode DIV/MOD install
(`alu_ops.py make_alu_divmod_composite_ops::make_install`) lowers
`wide_div_rules_ge_format(width_bytes=1)` — a flat **256×256** `(a, b) → (q, r)`
cross-product (131,072 rules for DIV+MOD). It reads the dividend from
`ALU_LO`/`ALU_HI` (16+16 dims = **one byte**: low nibble + high nibble) and the
divisor from `AX_CARRY_LO`/`AX_CARRY_HI`, all at the `MARK_AX` row.

Two compounding facts make the high byte unreachable (verified live at HEAD
`5b42e26a`, spec_k=0, real compiled `1162/37` bytecode):

1. **The operand pipeline only delivers byte 0.** `ALU_LO`/`ALU_HI` are pinned
   `"mark == AX OR (is_byte AND byte_index == 0)"` (16 dims each = a single
   byte). The L7 `operand_gather` head 0 reads **only `CLEAN_EMBED_LO/HI`
   (STACK0 byte 0)** into `ALU_LO/HI` (`l7_ops.py:154`). The dividend's high
   byte (`0x04` for 1162) lives at the STACK0 byte-1 row and is **gathered
   nowhere** at the AX row. Probe of `1162/37` at the DIV decode row (pos 140,
   block 22):
   ```
   ALU_LO  cell10=5.82   (0x8A low nibble = 0xA = 10)   ← low byte only
   ALU_HI  cell 8=5.82   (0x8A high nibble = 0x8 = 8)
   CARRY_LO cell5, CARRY_HI cell2  (divisor 0x25 = 37)
   STACK0_BYTE_VAL_1_LO/HI = []   ← high byte 0x04 absent everywhere
   ```
   (`STACK0_BYTE_VAL_1_*`, pos 734-765, are scaffolded but unwritten — see
   `dim_registry_dynamic.py:66`.)

2. **The lookup is single-byte by construction.** `wide_div_rules_ge_format`
   raises `NotImplementedError` for `width_bytes > 1`. Its `a` index runs 0..255
   only.

So even before the lookup-width wall, **the high dividend byte never enters the
AX-row operand band.** The 256×256 table then divides the only byte it sees
(`a & 0xFF`) → `low_byte/divisor`, the observed result.

## Why multi-byte DIV is an architecture wall (not a single-rule fix)

A correct multi-byte div needs BOTH a wider operand relay AND a wider divide,
and the wider divide hits a hard flat-lookup ceiling.

### Wall A — flat lookup explodes past one byte

A *general* 2-byte-dividend ÷ 1-byte-divisor byte-accurate lookup is a
**3-input** table `(dividend_lo, dividend_hi, divisor) → (q, r)` =
256·256·256 = **16,777,216 rules**. The current single-byte install already
lowers a dedicated 131,072-unit `PureFFN`; 16.7M units is a ~16.7M×872 fp32
`W_up` (≈58 GB) — **impossible**. A full 16-bit-dividend / 8-bit-divisor table
is 33.5M (DIV+MOD).

**No flat-lookup factoring escapes this.** Long division by one byte across two
dividend bytes is:
- Stage A (high byte): `(hi, divisor) → (q_hi, r_hi)` = 65,536 rules — tractable.
- Stage B (low byte): reduce `r_hi·256 + lo` by `divisor` → `(q_lo, r)`. Its
  inputs are `(r_hi, lo, divisor)` = 256·256·256 = **16.7M** — Stage B is itself
  an irreducible 2-byte/1-byte division. The remainder-carry `r_hi → Stage B` is
  a *cross-block* dependency (expressible via the existing multi-post_op
  staging), but Stage B's table is the wall.

### Wall B — single FFN layer can't pipe carries; needs multi-pass

`FlattenedDivMod` does wide div as an 8-outer × 3-inner bit-serial
shift-and-subtract on the GE workspace. The cross-position remainder/borrow
cascade (output of one rule = input of the next) **cannot live in one FFN
layer** (`LONG_DIVISION_FFN_RULE_INFEASIBILITY`).

### The minimal multi-byte-division design (deferred — structural)

Bit-serial long division as a `multi_pass` cascade is the only tractable general
form. For a 16-bit dividend ÷ 8-bit divisor:

1. **Operand relay (new attention head).** Extend L7 `operand_gather` (or add an
   L8/L9 head) to gather STACK0 bytes 1..n into a new multi-byte dividend band at
   the AX row (the `STACK0_BYTE_VAL_h_*` slots already reserved at 734-829 are
   the natural target — they need a writer). Without this, no fix is possible:
   the high byte simply is not in the residual.
2. **Per-bit remainder-carry band.** A new `DIV_PARTIAL_REM` residual band
   (≤ 9 bits) carried across passes.
3. **16 staged FFN post_op blocks.** One per dividend bit, each a 131,072-rule
   `(r_partial_8b, next_bit, divisor_8b) → (new_r, q_bit)` lookup. The model is
   already 37 physical blocks; +16 is a ~43% block increase plus ~2.1M new FFN
   units. This is the `multi_pass_rules` IR primitive (Path 1) from
   `DSL_W5_MULDIV_LIMIT.md §"Two paths forward"` — several sessions of DSL +
   verifier work and a structural model change.

This is the **same class of wall as 16-bit MUL** (the flat cross-product /
carry-cascade ceiling): the per-byte primitive is fine, but composing it across
bytes needs an inter-pass cascade the band-offset DSL cannot express in one
layer, plus an operand relay the pipeline does not yet have.

## Why no safe partial ships

- A corpus-overfit table (exploiting quotient ≤ 50 / divisor ≤ 50) would violate
  declarative generality and is not the VM's operand contract.
- Any genuine fix requires the operand relay (Wall A prerequisite) + the
  multi-pass cascade (Wall B) — both structural. There is no single rule or
  install tweak that flips any multi-byte case without that infrastructure, and
  per `feedback_single_rule_fixes_are_zero_sum` such stacked partials net zero.
- `FlattenedDivMod` (lookup mode) remains the authoritative multi-byte path; the
  `width_bytes=1` GE-format install is the correct, regression-free single-byte
  surface and must stay.

## Verification (HEAD 5b42e26a, this worktree, spec_k=0)

- `--ids 150-249`: div 21/50, mod 24/50 (55 fail = exactly the multi-byte
  dividends; 52/55 = `low_byte/divisor`).
- Authoritative smoke `pytest tests/test_smoke.py -q`: **45 passed, 4 failed,
  2 xfailed** (the 4 fails — mul_basic, eq_true, eq_false, sub_16bit — are
  pre-existing CMP/mul/sub-16bit walls, owned elsewhere). `test_div_basic`
  (84/2=42) and `test_mod_basic` (43%10=3) **PASS** — single-byte DIV/MOD healthy.
- No source change in this commit → zero regression by construction.

## Cross-references

- `neural_vm/unified_compiler/wide_alu_dsl.py:972` — `wide_div_rules_ge_format`
  (`width_bytes>1` → `NotImplementedError`).
- `neural_vm/unified_compiler/ops/alu_ops.py:1694` — `make_install` (the
  single-byte GE-format install).
- `neural_vm/unified_compiler/ops/l7_ops.py:119` — `operand_gather` (byte-0-only
  gather; the operand-relay prerequisite lives here).
- `neural_vm/dim_registry_dynamic.py:234,647` — `ALU_LO/HI` (one byte) and the
  reserved `STACK0_BYTE_VAL_h_*` band (the multi-byte relay target).

---

## ADDENDUM (2026-06-13): divmod-compute L14 PLACEMENT DRIFT — found + fixed

Independent of the multi-byte wall above, the efficient-mode DIV/MOD compute was
**mis-placed onto the wrong physical block** — the exact same drift class as the
MUL L11 fix (`docs/MUL_BASIC_L11_PLACEMENT_FIX_2026_06_13.md`). This was found
while re-checking the brief's "is the divmod wrap mis-bound like MUL?" lead.

### The drift

`make_alu_divmod_composite_ops`'s install op (`l10_alu_divmod_install`, the only
one of the 4 ops that bakes in efficient mode) bound via
`target_op_name="layer10_carry_relay"`. That L10 attn anchor is itself **placed
at pre-exp layer 14** (the dep-scheduler floats it downstream of the L10 op
family — same mechanism that pushed `_layer11_ffn_dep_anchor` to L15 for MUL).
So the two efficient-mode divmod post_ops it appends to `block.post_ops` —

| FFN | hidden | landed (before) | landed (after fix) |
| --- | --- | --- | --- |
| operand-cleanup | 4 | phys 24 = logical **L14** | phys 12 = logical **L10** |
| `wide_div_rules_ge_format` lookup | 131072 | phys 25 = logical **L14** | phys 13 = logical **L10** |

Verified with `tools/probe_divmod_assignment.py` (spec_k=0, disk_cache=False):
`layer10_carry_relay` is at pre-exp layer 14, not 11; before the fix the GE_FFN
sat at physical block 25 (= logical L14), four logical layers past its intended
L10 install point — where the operand bands the install reads are no longer the
clean MARK_AX state the cleanup+lookup were tuned against.

### The fix (one line + comment, `alu_ops.py make_install`)

```python
-   target_op_name="layer10_carry_relay",   # mis-resolves to L14
+   layer_idx=10,                            # pin to L10 (physical ~11)
    requires={"after": "l10_alu_divmod_getobd"},   # KEPT
```

Mirrors the MUL fix exactly: `layer_idx` takes precedence over `target_op_name`
for block ops (`layer_compiler.py:1505`); `requires['after']` is kept so the
efficient install still fires after the (no-op in efficient mode) lookup-mode
GE→BD stage chain.

### Result — byte-identity-safe structural correctness, NO pass-set change

- `tools/probe_divmod_assignment.py`: GE_FFN now lands at physical block 13 =
  **logical L10** (was L14). Confirmed.
- `pytest tests/test_smoke.py -q`: **49 passed / 2 failed** (same two pre-existing
  fails: `test_simple_function`, `test_mul_overflow`). `test_div_basic` /
  `test_mod_basic` stay green; zero ALU/bitwise/shift/cmp/32-bit regressions.
- `pytest tests/test_div_mode.py -q`: **11 passed**.
- `run_1096_canonical.py --criterion full_trace --ids 150-249`: **div 21/50,
  mod 22/50 — UNCHANGED**, with byte-identical divergences (same fail ids, same
  expected/got AX).

### Why the placement fix recovers ZERO failing programs

The full-trace divergences are **all at the dividend-LOAD step (step 1), BEFORE
the divide executes** — `expected ax=<full dividend>, got ax=<low_byte(dividend)>`
(e.g. id150 `1162/37`: expected ax=1162, got ax=138=1162&0xFF). Every single
failing program has a multi-byte dividend (`a > 255`); every passing one has a
single-byte dividend. The divmod compute — now correctly placed — never sees a
correct multi-byte operand because the truncation is upstream (the byte-0-only
operand gather + the AX byte-1 DUMP truncation, the latter owned by the
in-flight task #220 / HEAD commit `bf2ce4f7`). The placement fix is real and
structurally correct (the compute is no longer baked onto L14 garbage state),
byte-identity safe, and a prerequisite-cleanup for the eventual multi-byte
relay, but it cannot flip any case alone. The remaining path is the
multi-session bit-serial cascade + multi-byte operand relay described above —
HARD STOP per the div/mod fix brief.

---

## ADDENDUM (2026-06-13): re-verified at HEAD ac9177d1 — AX_BYTE1_DUMP default-ON does NOT touch the operand relay; wall stands

Re-diagnosed the WHOLE div/mod/gcd cluster at HEAD `ac9177d1` (spec_k=0,
`full_trace`). HEAD is materially newer than the doc's `5b42e26a`: it now
carries the **AX byte-1 register-dump carry DEFAULT-ON** (`C4_AX_BYTE1_DUMP`,
commit `f080a46e`) plus the L10 divmod placement fix (`a30e1048`, the addendum
above). The lead worth checking was: *does the now-default-on AX byte-1 carry
make the multi-byte dividend survive into the divide?* **It does not** — those
are two different residual paths (register-emit vs operand-gather).

### Cluster boundary at HEAD (unchanged from the doc body)

`run_1096_canonical.py --criterion full_trace --spec-k 0`:
**div 21/50, mod 24/50** (single-byte dividends pass; multi-byte fail).
The gcd cluster (ids 900-949) is 0/50 — gcd's `while(b) { b = a%b; a = temp; }`
loop is built on the same `%` op, so every gcd inherits the div/mod multi-byte
truncation (and most gcd inputs reach a multi-byte intermediate remainder).

### The compute-error mechanism, quantified (52/55 + airtight)

Of the 55 div+mod failures, **49 are byte-exactly `low_byte(dividend) op
divisor`**; verified on the fresh slice (`--ids 150-159,200-209`):

```
id150 1162/37: 1162&0xFF=138 -> 138/37=3  == got_ax 3
id151  843/31:  843&0xFF=75  ->  75/31=2  == got_ax 2
id200  390%19:  390&0xFF=134 -> 134%19=1  == got_ax 1
id208  434%10:  434&0xFF=178 -> 178%10=8  == got_ax 8
```

The 6 non-truncation outliers are NOT a different div/mod root: id170/173/183
are the same truncation caught one step earlier at the dividend-LOAD step
(`got_ax = a&0xFF` itself); id184/203 are the AX byte-1 **0xFF / leak** family
(`got=0xFFxx` / `0x1xx`), owned by the AX-byte1 task, not the divide; id201
(`89%10`, single-byte) is a lone **operand-cleanliness** miss — the dividend
low nibble 0x9 is decoded as 0x7 (89→87, 87%10=7=got), an L8 operand-gather
noise perturbation of the `~5.56` cell-0 artifact, NOT a divmod-rule bug, and
high-risk to chase (the `_ARTIFACT=5.56` cleanup constant is load-bearing for
the 42 single-byte div/mod cases that pass).

### The structural smoking gun (re-probed at HEAD, AX_BYTE1_DUMP ON)

For `1162/37` (high byte 0x04), at block 13 (post-L10-divmod-install):

```
REG_AX register tokens decode = 1162 = 0x048A  (byte0=0x8A, byte1=0x04 PRESENT)
   ^ the AX byte-1 DUMP carry DOES preserve the full dividend in the REGISTER.
operand band ALU_LO/HI (what the 256x256 lookup reads) = a SINGLE byte only.
reserved high-byte relay STACK0_BYTE_VAL_1 written ANYWHERE? -> False.
```

So the register-emit carry and the operand-gather are **independent paths**:
AX_BYTE1_DUMP fixed the former; the latter (the band the divide reads) is
still single-byte, and the reserved `STACK0_BYTE_VAL_1_*` relay slot is still
unwritten. The `wide_div_rules_ge_format(width_bytes=1)` lookup is
byte-accurate by construction (`out_value = a//b` / `a%b` per (a,b) pair —
read at `wide_alu_dsl.py:1206`); it simply never sees `a`'s high byte, so it
divides `a&0xFF`. **No declarative single-rule / install tweak can flip a
multi-byte case** without first writing the high dividend byte into the
operand band — which is the new attention head (Wall-A prereq) + the bit-serial
cascade (Wall-B), both multi-session structural. HARD STOP confirmed; no source
change shipped (smoke stays 51/51 by construction).
