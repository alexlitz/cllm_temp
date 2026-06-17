# Multi-local-frame root (var_mul / var_three / if_var ~75): LEA guard + the remaining BP-relay wall

Date: 2026-06-17. spec_k=0, BUILT `layout.dim_positions`, SwiGLU `silu(up)*gate`.
CPU-only (no GPU smoke / full_trace this session — deferred-verify). Vehicle:
teacher-forced single-forward CPU probe (`tools/probe_var_tf_full.py`, run with
`CUDA_VISIBLE_DEVICES=""`) + the re-anchored CPU gate
(`tools/interp_oracle_gate.py`) + the interpreter rule-attribution
(`FaithfulInterpreter.attribute_runtime_contribution`).

Builds on diag commit 18365452 (which committed tooling only) and the now-merged
L15 fix eddad334 (`Fix L15 nibble-copy writer overwhelming L16 BP_byte1
override`).

## What landed (this lane): the multi-local LEA 0xE8 over-fire guard

`tail_lea_local_ax_marker_byte0_e8` (l10 `_tail_bit32_result_correction` bank,
physical block 41 = L25 tail, unit 1791) hardcodes the BP-8 effective-address
low byte **0xE8** onto the LEA AX-marker row. Its FETCH discriminator
(`FETCH_LO+8` w=2.0, `FETCH_HI+15` w=0.2) is only ADDITIVE — the 9-pt non-FETCH
positive sum (MARK_AX+HAS_SE+OP_LEA+CMP+7+MEM_ADDR_SRC) clears threshold=7
ALONE, so the 0xE8 writer over-fires on the SECOND local (BP-16, imm=-16,
correct low byte 0xE0) and stamps 0xE8 → the two locals ALIAS.

FETCH band signatures (clean one-hots at block-40 residual, the tail rule's
input; `tools/probe_var_fetch_band.py`):

| local | imm | FETCH_LO nib | FETCH_HI nib | correct AX low byte |
|-------|-----|--------------|--------------|---------------------|
| 1st   | -8  | 8            | 15 (F)       | 0xE8                |
| 2nd   | -16 | 0            | 15 (F)       | 0xE0                |
| 3rd   | -24 | 8            | 14 (E)       | 0xD8                |

The guard (flag `C4_LEA_LOCAL_E8_MULTILOCAL_GUARD`, default **ON**) appends two
NOT-blockers — `FETCH_LO+0` (-10.0, the imm=-16 low-nibble) and `FETCH_HI+14`
(-10.0, the imm=-24 high-nibble) — so the 0xE8 stamp is a genuine imm=-8
requirement. On a 2nd/3rd-local LEA the competing nibble is lit (≈1.0 clean),
the -10 blocker (×S=100 → -1000) drives the score far below threshold, and the
correct L8 effective-address byte survives.

### CPU verification (teacher-forced single forward, spec_k=0)

`tools/probe_var_tf_full.py 275` (var_mul), guard ON vs OFF — PC/AX columns:

```
            OFF                         ON
step 6  AX 0xffe8/0xffe0 XX     AX 0xffe0/0xffe0 OK   <-- the flip
```

Every other PC/AX byte is correct in BOTH; the guard changes ONLY the 2nd-local
LEA AX byte0. Confirmed byte-identical / inert where it must be:

* **flag-OFF is byte-identical** by construction (the conditions tuple appends an
  empty tuple when off; a full-model build scan finds 0 units carrying the
  guard signature flag-off, 1 unit — block 41 unit 1791 — flag-on).
* **var_update (id 325, imm=-8 only): ON ≡ OFF** byte-for-byte (the guard never
  fires on a single-local program). LEA-basic and all imm=-8 LEAs are
  unaffected — the guard touches ONLY imm=-16/-24, exclusive to multi-local var.

## Why the guard alone does NOT flip the full_trace verdict (the remaining wall)

The diag (18365452) already established the LEA guard is a per-step fix, not a
verdict flip — confirmed again here on CPU. Two distinct cross-step roots remain;
under teacher forcing every PC/AX is correct, but the BP/SP/STACK0 frame VALUE
bytes (which poison the autoregressive decode) are still wrong:

### Root A — the universal BP byte1=0xFF cross-step poison (ENT step)

The re-anchored gate flags var_mul / var_three / if_var as **CROSS-STEP, value
poison @ step 1** (the ENT BP-frame establishment). At the step-1 BP-byte1 row
the OUTPUT band is correct (`OUTPUT_HI+15` favored = 0xff) through block 49,
then **block 50 (logical L25 tail)** slams `OUTPUT_LO+0` / `OUTPUT_HI+0` to
~+3.4e9 / +4.8e9 (strength-2000 units, W_down≈+2e5), flipping byte1 → 0x00. The
interpreter attributes the wrong cell to the
`layer16_lev_routing::l16_stack0_e8_output_authoritative_*` family (256 rules,
strength 2000), which carries `MARK_STACK0=+1e9` / `OP_ENT=-1e9` / `IS_BYTE=-1e9`
— it should be vetoed on the ENT BP-byte1 row (MARK_STACK0=0, OP_ENT≈16,
IS_BYTE=1) yet the L25-tail amplification still lands the 0x00 crush. This is the
**L20+L25 `layer16_lev_routing` 0x00-default crush** documented in the memory
note `project_var_fulltrace_stack0_frame_desync.md` — a width-sensitive
(`project_l10_tail_bank_width_sensitive.md`), memory-smoke-load-bearing,
multi-lane surface where single-rule changes are zero-sum
(`feedback_single_rule_fixes_are_zero_sum.md`). NOT a solo corrector; it needs
the coordinated two-part L20+L25 build the var-fulltrace note specifies. The L15
half (eddad334, the nibble-copy BP blocker) is already merged and necessary; it
clears the L15 wide-writer but does not reach the L25-tail crush.

### Root B — var_three 3rd-local LEA computes BP-8, not BP-24 (upstream L8)

var_three's 3rd local (imm=-24, expected AX low byte 0xD8) reads **0xE8 already
at block 40** — i.e. BEFORE the tail LEA rule — in BOTH guard ON and OFF
(`OUTPUT_LO+8`=49, `OUTPUT_HI+14`=87 at block 40). The tail 0xE8 stamp is NOT the
source here; the L8 effective-address compute / FETCH→ALU relay produces BP-8 for
the -24 offset. The guard correctly leaves this untouched (it only suppresses the
tail stamp, which is not what wins). This is a separate upstream root from the
2nd-local case (where the L8 value IS correct, 0xE0, and the guard lets it
survive). var_mul / var_update / if_var have ≤2 locals so Root B does not affect
them; it gates only var_three's deeper local.

## Net

* **Landed**: `C4_LEA_LOCAL_E8_MULTILOCAL_GUARD` (default ON) — fixes the
  multi-local 2nd-local LEA AX (var_mul step-6 0xFFE8→0xFFE0), byte-identical
  flag-off, inert on imm=-8 / single-local programs. This is the load-bearing
  per-step component the diag asked this lane to land once the BP relay lands.
* **Remaining verdict blocker (all ~75)**: Root A, the cross-step BP-byte1=0xFF
  L20+L25 crush — the documented multi-lane, width-sensitive surface. Plus Root B
  (var_three 3rd-local upstream L8 BP-24 compute). Neither is a single-rule fix
  and both overlap the STACK0 frame surface (the if/bool pop-discriminator lane
  works the same `layer16_lev_routing` / STACK0 band).

## Overlap note

Root A is the SAME `layer16_lev_routing` 0x00-default + L25-tail crush that the
if/bool pop-discriminator lane bottoms out on (the STACK0 OUTPUT-band surface).
Sequence the BP-byte1 relay with that lane — a coordinated L20+L25 build, not
parallel solo correctors.
