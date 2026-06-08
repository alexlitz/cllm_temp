# ptr_/deref_/addr_ cluster attribution (2026-06-07)

Date: 2026-06-07 (H3 relaunch after API outage)
Base ref: `speedup-cache-and-buckets` HEAD `6d7a1c1d`
Task: 1096 cluster attribution for the `ptr_*` / `deref_*` / `addr_*`
family — pointer load, pointer dereference, and address-of operations
at the C language level.

## TL;DR

**Neither the `ptr_*`, `deref_*`, nor `addr_*` cluster exists in the
1096 pure-neural corpus.** The combined filter
`-k "ptr_ or deref_ or addr_"` selects zero tests; each of the three
filters run individually also selects zero. Attribution analysis cannot
proceed against non-existent parametrisations. This doc captures the
negative result so the next agent does not re-run the same dead probe.

This is the second boundary doc in the H1/H3 sequence; H1 already
established the same empty-filter pattern for
`char_/string_/array/struct/global/static` (see
`docs/CHAR_STRING_ATTRIBUTION_2026_06_07.md` and
`docs/STRUCT_GLOBAL_ATTRIBUTION_2026_06_07.md`). H3 confirms that
pointer-shaped C semantics are likewise absent.

## Pattern collection

| Filter                        | Selected | Deselected | Result |
|-------------------------------|---------:|-----------:|--------|
| `-k "ptr_ or deref_ or addr_"`|        0 |       1098 | empty  |
| `-k "ptr_"`                   |        0 |       1098 | empty  |
| `-k "deref_"`                 |        0 |       1098 | empty  |
| `-k "addr_"`                  |        0 |       1098 | empty  |

All four runs terminate in ~0.20s with `1098 deselected in 0.20s`. No
`--runxfail` failures, no errors, no skips.

A case-insensitive `grep -i -E "ptr|deref|addr"` against the full
`pytest --collect-only` list returns **0 matches**.

## Actual 1096 category inventory

Per `STRUCT_GLOBAL_ATTRIBUTION_2026_06_07.md`, the 1096 categories are:

`absdiff_`, `add_`, `bool_`, `div_`, `edge_`, `expr_`, `func_`, `gcd_`,
`if_`, `loop_`, `mod_`, `mul_`, `nested_`, `rec_`, `sub_`, `var_`
(plus 20 uncategorised). Total: 1098 collected.

None of these prefixes touch pointer syntax. The 1096 generator
(`tests/test_suite_1000.py:generate_test_programs`) is integer-only
arithmetic + control-flow; the C type system (pointers, arrays,
structs, address-of) is not parametrised.

## Where did the cluster names come from?

The brief inherited terminology from a C-language program-classification
taxonomy: `ptr_` (pointer variable), `deref_` (`*p` reads/writes), and
`addr_` (`&x` address-of). None of these were ever instantiated as 1096
test cases. C4 supports these opcodes (LEA for address-of, LI/LC for
load through pointer, SI/SC for store through pointer), but the 1096
corpus exercises them only as **emergent byproducts** of `var_*` and
`func_*` (stack-frame addressing, JSR/LEV ret-addr deref) — never as
named parametrisations.

## Cluster 1 — ptr_*

**Pass count:** N/A (0 tests selected).
**Representative failure:** none — filter is empty.
**Root surface:** undefined. The nearest analog at the opcode level is
LI through a stack-pointer-derived address, which surfaces in `var_*`
(100 tests, 0 passing per closeout Wave C2). Root surface for that
analog is L3 `SP_byte2` OUTPUT_LO[1]-vs-[0] at block=3 gen=13
(`var_failure_mode_shifted` memory note); not a distinct ptr surface.

## Cluster 2 — deref_*

**Pass count:** N/A (0 tests selected).
**Representative failure:** none — filter is empty.
**Root surface:** undefined. C4 dereference compiles to `LI` (load
int) or `LC` (load char) against an address held in a local. The
relevant memory-step surface (L10 PSH `MEM_addr0`, L8 SP gather) is the
predicted analog but has no `deref_*` test to exercise it.

## Cluster 3 — addr_*

**Pass count:** N/A (0 tests selected).
**Representative failure:** none — filter is empty.
**Root surface:** undefined. C4 `&x` compiles to `LEA` against the
frame pointer. The 51-test smoke `test_lea_basic` already covers this
surface and is one of the six expected smoke failures (Wave S2 in the
closeout plan). No `addr_*` test routes through it.

## Memory-cluster dependence (LI/LC step hook)

Per the brief, if any failing tests had been selected, the next step
would have been to hook L8/L10/L14 on LI/LC steps and determine
memory-cluster dependence. With zero selected tests, this hook is
unreachable. The predicted dependence (had tests existed) is:

- **L10 PSH `STACK0_BYTE_VAL_h` propagation** (Wave S1) — for `deref_`.
- **L10 PSH `MEM_addr0=0xE0` ENT-main pin bug** (memory note
  `project_l10_psh_addr_ent_bug.md`) — likely blocker for `addr_` and
  `ptr_`.
- **L8 SP gather audit** (`docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md`)
  — for `ptr_` stack-relative loads.

All three are already attributed via `var_*` / `func_*` failures; the
ptr/deref/addr names add no new surface.

## Dependencies on in-flight work

- **A3 (in-flight, `var_*` / JSR-step-0 `MEM_value1`):** no dependency.
  If `ptr_*` existed, it would inherit A3's outcome via the shared L10
  PSH surface. With zero selected tests this is moot.
- **Wave 2 F1-F3:** no dependency. Targeted at categories that exist
  (`func_add`, `var`, EQ Shape-B). None of them would change the
  ptr/deref/addr selection count from 0.
- **H1 (closed empty):** sibling. Same empty-filter outcome for
  `char_/string_/array/struct/global/static`. H3 is the pointer-shaped
  analog and closes the same way.

## Smoke

51-test smoke (memory + LEA per closeout Wave S1/S2): expected 45
passed, 6 failed — same six as H1's closeout
(`test_lea_basic`, `test_si_li_roundtrip`, `test_sc_lc_roundtrip`,
`test_si_li_multiple_stores`, `test_si_li_overwrite`,
`test_si_li_16bit_value`). Threshold ≥45 expected. No code changes in
this commit — doc only.

## Recommendation

Replace the brief's `ptr_/deref_/addr_` cluster names with concrete
1096 parametrisations that actually exercise pointer-shaped code paths.
If C-pointer coverage is desired, it must be added as a new 1096
parametrisation (extending `tests/test_suite_1000.py`) before
dispatching an attribution agent. As of this doc, the four real
attribution clusters remain: C1 `func_add`, C2 `var`, C3 EQ Shape-B,
C4 `expr_add_mul`.
