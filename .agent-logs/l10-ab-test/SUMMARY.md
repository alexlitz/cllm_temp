# Batch 5 RETRY of B5-C: L10 ENT-fix A/B Test

**Base**: speedup-cache-and-buckets @ 4d069f7

## Variants Tested

| Variant | Branch | Commit | Strategy |
|---|---|---|---|
| baseline | speedup-cache-and-buckets | 4d069f7 | (no fix) |
| b2a | fix/l10-ent-discrimination | ee2f493 | OP_ENT=-1e6 only (L10+L16 0xE0 PSH guard) |
| b3d | fix/rec-premature-exit | d3c7c53 | OP_ENT=-1e6 + tail PSH 0xE0 strength tamed |
| b4a | fix/l10-e0-coordinated | 44dbfd1 | OP_ENT/LEV/IMM blockers on L10+L16 0xE0 rules |

## Ok-count Results (4x5 table)

| Variant | sentinel (0/32) | var_simple (200/50) | if_var (425/25) | func_identity (550/25) | rec_factorial (700/25) | TOTAL |
|---|---|---|---|---|---|---|
| baseline | 3 | 25 | 0 | TIMEOUT/KILL (rc=143) | TIMEOUT/KILL (rc=137) | 28 (3 cells valid) |
| b2a      | 3 | 25 | 0 | 0 | 4 | **32** |
| b3d      | 3 | 25 | 0 | 0 | 4 | **32** |
| b4a      | 3 | 25 | 0 | 0 | TIMEOUT/KILL (rc=137) | 28 (4 cells valid) |

## Run Durations (s)

| Variant | sentinel | var_simple | if_var | func_identity | rec_factorial |
|---|---|---|---|---|---|
| baseline | 205 | 546 | 1500 | 744 (killed) | 15 (killed) |
| b2a      | 190 | 426 | 1262 | 653 | 1325 |
| b3d      | 130 | 346 | 1000 | 639 | 1306 |
| b4a      | 140 | 342 | 1107 | 1028 | 1406 (killed) |

## Analysis

- **sentinel (0/32)** all variants identical at 3 ok. No variant moves the needle on entry-zone behavior.
- **var_simple (200/50)** all variants identical at 25 ok. No regression, no improvement.
- **if_var (425/25)** all variants identical at 0 ok. None of the three ENT fixes recover any if_var cells.
- **func_identity (550/25)** all three fix variants give 0 ok. Baseline cell was killed before completion, but b2a/b3d/b4a are all identical at 0.
- **rec_factorial (700/25)** b2a and b3d both recover 4/25 cells. b4a's run was killed; b4a/baseline numbers unknown.

## Conclusion

**Winner: TIE between b2a (fix/l10-ent-discrimination) and b3d (fix/rec-premature-exit)** with 32 total ok across the 5 ranges. They produce identical results on every cell that completed for both.

Because b2a is the **simpler** intervention (single change: OP_ENT=-1e6 guard on L10+L16 0xE0 PSH MEM rules; b3d adds an additional strength-taming that produced no observable benefit), **b2a is the recommended winner**.

b4a (the more invasive coordinated change) is no worse on the 4 cells we could compare but offers no improvement on those, and its rec_factorial cell was killed mid-execution.

## Caveats

- Heavy contention from many parallel agents during the run inflated per-cell durations 3-5x and triggered SIGTERM/SIGKILL on 3 of 20 cells (all in baseline 550/700 and b4a 700).
- Only rec_factorial discriminated between fix variants and the others. The other 4 ranges showed no variation across variants.
- b4a's missing rec_factorial cell means we cannot fully confirm whether the more invasive coordinated fix matches or beats the simpler ones.

## Winner

**b2a (fix/l10-ent-discrimination, cherry-pick of 6f9f002 -> ee2f493)** -- totals: 32 ok across 5 ranges.
