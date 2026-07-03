# AX high-byte register-DUMP truncation — definitive root (2026-06-13)

**Status:** ROOT CONFIRMED. It is a **multi-byte carry-forward (compute) gap**, NOT
a late-pipeline emission-overwrite. The fix is a multi-head / multi-rule
architectural addition (a multi-byte AX register carry-forward), overlapping the
in-flight convergent multi-byte tasks (#206 PSH multibyte store, #213 multi-byte
operand relay, #217/#218 ADD multi-byte carry). Flagged multi-session per the
fix brief; this doc records the exact evidence so the next pass can build it.

## What grades us: full-trace

`tools/run_1096_canonical.py --criterion full_trace` decodes each VM step's
(PC, AX) from the model's emitted 35-token step (`REG_AX` marker at step-offset 6,
then 4 LE value-byte tokens at offsets 6..9 = AX[0..3]) and fails on the first
divergent step. 452/982 fails diverge at step 1; 620 fails are PC-correct but
AX-wrong; ~139 of those have AX low byte matching the oracle → clean high-byte
truncation.

## Canonical reproduction: program id 0 (`add_0: 654 + 114`)

Compiled bytecode runs: `IMM 654 ; PSH ; IMM 114 ; ADD ; EXIT`.
Ground-truth probe (`tools/probe_groundtruth.py`, spec_k=0, hook-free) emits:

| step | opcode | pc | AX bytes [b0,b1,b2,b3] | decoded AX | oracle AX | verdict |
|------|--------|----|------------------------|------------|-----------|---------|
| 0    | IMM654 | 10 | [142, **2**, 0, 0]     | **654** ✅ | 654       | correct |
| 1    | PSH    | 18 | [142, **0**, 0, 0]     | **142** ❌ | 654       | byte1 dropped |
| 2    | IMM114 | 26 | [114, 0, 0, 0]         | 114 ✅     | 114       | correct |

The truncation is on the **PSH step (step 1)**, not the IMM step. PSH does not
modify AX, so AX=654 must *persist*. The model emits byte 1 = 0x02 correctly on
the IMM step (which freshly produces AX) but drops it to 0x00 on the very next
non-AX-writing step. Any value ≥ 256 that survives a PSH/non-AX step truncates →
the dominant full-trace step-1 divergence.

## Why it is COMPUTE, not emission-overwrite

Probed the byte-1 dump row residual across **all 40 physical blocks** at
spec_k=0. The driving residual is fully determined **before block 29** and is
**byte-identical across blocks 29..39** (the L19/L20/L25 tail the brief
suspected). So no late corrector zeros byte 1 — there is simply nothing to zero;
the value was never carried to that row.

### The byte-1 value lives in STACK0_BYTE_VAL_* on the producing step, and vanishes on the carried step

Full-residual diff id0 (byte1=0x02) vs id6 (`913+558`, byte1=0x03), byte-1 row,
block 39:

* **IMM step (correct):** the dims that differ by the program's byte-1 value are
  `STACK0_BYTE_VAL_2_HI` / `STACK0_BYTE_VAL_3_LO/HI` at strength **45.74** (the
  PSH multi-byte broadcast band, task #206). The conventional dump dims
  `OUTPUT_LO/OUTPUT_HI` at the byte-1 row are **identical** for id0 and id6
  (OUTPUT_LO=14, OUTPUT_HI=10) even though the emitted byte differs (0x02 vs
  0x03) — i.e. the output head reads the high-byte value from the
  STACK0_BYTE_VAL band, not from OUTPUT_LO/HI, for the high dump rows.
* **PSH step (wrong):** those `STACK0_BYTE_VAL_*` 45.74 signals are **gone**.
  Only weak (≤1.0) control-flag differences remain plus `AX_CARRY_*` residual
  bleed at ~0.6.

### AX_CARRY is NOT a usable byte-1 source

`AX_CARRY_LO/HI` at the byte-1 dump row decodes to the right byte only by
coincidence:

| id  | program   | exp byte1 | AX_CARRY decode |
|-----|-----------|-----------|-----------------|
| 0   | 654+114   | 0x02      | 0x02 (match, luck) |
| 4   | 754+104   | 0x02      | 0x24 (no)       |
| 6   | 913+558   | 0x03      | 0xb3 (no)       |

The ~0.6-strength `AX_CARRY` at the high dump rows is bleed from the marker row
(which itself carries only prev-step **byte 0**), not a real byte-1 value.

## Mechanism: the L3 carry-forward is single-byte (byte 0) only

`make_layer3_carry_forward_attn_op` (l3_ops.py:1138) head 1 carries AX across
steps: `Q[0]=MARK_AX·L`, `K[0]=L1H1+AX_I·L / L1H0+AX_I·-L` selects the **prev
step's byte-0 row**, V copies prev `EMBED_LO/HI` → `AX_CARRY_LO/HI`. It fires
only at the `MARK_AX` marker row and only retrieves **byte 0**. There is no
analogous head for AX bytes 1..3, and the AX-byte-0 → OUTPUT projection family
(`l16_lev_ax_carry_*`, l16_ops.py:269; the PUTCHAR `AX_CARRY→OUTPUT` units in
model_ops.py:120) all carry `("IS_BYTE", -10.0)` — i.e. they explicitly fire
only at the marker row and are blocked on the value-byte rows (IS_BYTE=1).

Precedent that the equivalent for PC already exists and is hard:
`_layer3_pc_byte1_output_rules` + L3 head 7 ("PC byte1 prev → TEMP") carry PC
byte 1, but they are heavily corpus-tuned (`CLEAN_EMBED_HI 0..4`, "corpus never
runs past 0x14a", explicit wrap-token detection). AX is unbounded (0..0xFFFF+),
so that enumeration approach does not transfer.

## Why a single rule/head cannot fix it (and what would)

* No clean reliable byte-1 dim exists at the carried step's byte-1 row to project
  from (AX_CARRY is bleed; STACK0_BYTE_VAL is gone; OUTPUT_LO/HI is a constant
  control signal there).
* The real fix is a **multi-byte AX register carry-forward subsystem**: new L3
  carry-forward heads (free head slots) keyed on `MARK_AX + IS_BYTE +
  BYTE_INDEX_1/2/3` that attend to the **prev step's byte-1/2/3 rows** (selected
  with the `L1H2 vs L1H1` / higher discriminators, mirroring the byte-0
  `L1H1 vs L1H0` selector) and copy each high byte's value into a fresh per-byte
  carry band, **plus** a projection of that band into whatever dim the output
  head reads for the high dump rows (the `STACK0_BYTE_VAL` band, per the IMM-step
  evidence above — NOT `OUTPUT_LO/HI`). This must be gated to fire on
  non-AX-writing steps without disturbing the fresh-AX path (IMM/ADD/ALU), and
  byte-identity-gated to keep smoke at 48/3.
* This overlaps tasks #206/#213/#217/#218 ("the convergent multi-byte root").
  Build it there, not as a standalone whack-a-mole rule (memory:
  `feedback_single_rule_fixes_are_zero_sum`).

## Baselines measured this session (HEAD f92783ac, GPU 1, spec_k=0, full_trace)

* add (ids 0-49): **12/50**
* sub (ids 50-99): **5/50**  (NB: the brief's "100-149" is the mul cluster, 0/50)
* edge (ids 1000-1045): **28/46**  (the brief's "300-345" is var_three/var_update)
* smoke: **48 passed / 3 failed** (test_mul_basic, test_mul_overflow,
  test_simple_function — the known arch-blocks)

No production weights were changed (display-only fixes are impossible here — the
high bytes are never carried to the dump row to begin with), so all baselines
above are unchanged.

## Repro / probe recipe for the next pass

```
cd c4_release   # the python package root (neural_vm/ lives here)
CUDA_VISIBLE_DEVICES=1 python tools/run_1096_canonical.py \
    --fail-fast --criterion full_trace --ids 0-49 --output /tmp/add.json
```
Probe: build `tools.probe_groundtruth.GroundTruthProbe`, `compile_c(src)` first
(test programs are C source, not bytecode), then `probe(bytecode, max_steps=...)`
for per-step bytes and `residual_at(bytecode, block, pos, dim_names)` for
per-block dims. Step `s` byte-`b` dump-row token is at
`prompt_len + s*35 + 6 + b`; `prompt_len = len(probe._build_context(bytecode))`.
Block↔layer: `probe.print_block_layer_map()` (40 phys / 27 logical; L19=phys30,
L20=phys31, L25 tail=phys36..38).
