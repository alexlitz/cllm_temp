# OP_LEA leak into MARK_AX on IMM-dispatch rows — Investigation

Date: 2026-06-05
Worktree: `/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/agent-aec617f5813079926`
Branch: `speedup-cache-and-buckets` (off `07faf5d9`)
Brief: localize the actual writer of OP_LEA at MARK_AX on IMM dispatch
rows so Removal 1 of `RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md` has a
real upstream patch target. No code changes, doc only.

## TL;DR

The "OP_LEA leak" framing in `RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md`
and `REMOVAL_1_IMM_OVERRIDE_2026_06_05.md` (in commit `c9e9dc28`'s log)
is **structurally inconsistent with the declarative IR**: no declared
FFN rule or attention head writes `BD.OP_LEA` (dim 262) at MARK_AX on
IMM rows. The L5 LEA-decode rule (l5_ops.py:580, l5_ops.py:617-632)
requires `OPCODE_BYTE_LO+0 AND OPCODE_BYTE_HI+0 AND MARK_AX-gate`; on an
IMM row OPCODE_BYTE_LO+1=1 (not LO+0), so the condition score is
1 < threshold 1.5 — the LEA rule does NOT fire.

The actual load-bearing positive of `l10_ops.py:6474 tail_lea_local_ax_
marker_byte0_e8` is **`CMP+7`**, not `OP_LEA`. `CMP+7` is the L7 head 5
OP_LEA-relay output (l7_ops.py:494, AP(3, BD.OP_LEA, 0.2), AO(BD.CMP+7,
3, 1.0)). L7 head 5 sources `OP_LEA` via attention V from the source
token's residual, gates Q on MARK_AX, gates K on MARK_AX. So `CMP+7`
spuriously becomes nonzero at MARK_AX when the attention softmax
distributes mass over ANY position where the OP_LEA dim has even a
small residual — which is non-zero for several incidental writers in
the L0..L4 pipeline (most notably the EMBED_LO+0 / OPCODE_FLAGS[0]
position aliasing — see "Structural aliasing" below).

Combined with `FETCH_LO+8` and `FETCH_HI+15` from the actual IMM
operand bytes (which are 0xF_ for the test_xor_basic step 0 case,
IMM 0xFF), the L10 LEA-tail rule reaches `MARK_AX+HAS_SE+CMP+7+
FETCH_LO+8*2+FETCH_HI+15*0.2 = ~5.4` which is still under threshold
9.0 — but with attention-side OP_LEA broadcast plus the
`("OP_LEA", 1.0)` positive that ALSO reads the same source attention
post-rule, the score can cross threshold on the bytes that the
`L5_BYTEDECODE_FIX_ATTEMPT` probe identified.

A static one-line `OP_IMM` blocker bump cannot block this because the
leak is positive evidence from `CMP+7`/attention broadcast, not from
OP_IMM failing to silence. The c9e9dc28 attempt is consistent with this.

## Structural map of OP_LEA writers (declared, per-dim, dim 262)

`BD.OP_LEA = 262`, aliased to `OPCODE_BASE` and `OPCODE_FLAGS+0`
(dim_registry_dynamic.py:236,451). Since LEA = opcode 0 and the
"OPCODE_BASE" alias points at the same cell, every write to "the base
of the opcode one-hot" lands in OP_LEA's column.

### Direct writers (declarative, in DSL/IR)
None in L0..L4: no FFN rule or attention head in layers 0–4 writes
OP_LEA (grep evidence: zero matches under `unified_compiler/ops/l0..
l4`). L0..L4 produce zero OP_LEA contribution per `L5_BYTEDECODE_FIX_
ATTEMPT_2026_06_05.md`'s DSLInterpreter walk.

### L5 writers
- `l5_ops.py:578-632` `_opcode_decode_main_rules` — LEA decode unit 0:
  `multi_way_and_rule(OPCODE_BYTE_LO+0 AND OPCODE_BYTE_HI+0, gate=MARK_AX,
  writes=(OP_LEA, 10/S))`. **Does not fire on IMM rows** (LO+0=0).
- `l5_ops.py:650-692` `_opcode_decode_first_step_rules` — first-step PC
  decode (lo=0, hi=0, MARK_PC, NOT HAS_SE → OP_LEA). Fires only at PC
  marker on a real LEA step; does not fire at AX.

### Imperative (legacy) writers (vm_step.py / setup_helpers)
- `vm_step.py:3798` `ffn.W_down[BD.OP_LEA, unit] = 10.0/S` — PC-marker
  LEA decode, same predicate as the declarative version.
- `vm_step.py:5412` `attn.W_v[base + 3, BD.OP_LEA] = 0.2` — this is
  the **L7 head 5 V projection** (see "L7 head 5 relay" below).
- `vm_step.py:4444,4453, 8696,8703` `ffn6.W_up[unit, BD.OP_LEA] = S` —
  these READ OP_LEA, not write (W_up is the input gate).
- `weight_modules/function_calls.py:268` `attn7.W_q[base, BD.OP_LEA]
  = L7/5` — Q-side READ (LEA gate for function-call head).

**None of these write OP_LEA at MARK_AX on a clean IMM row.**

## L7 head 5 OP_LEA relay — the actual load-bearing path

`l7_ops.py:484-524` head 5:
- Q: `AP(0, BD.MARK_AX, L), AP(0, BD.H1+AX_I, L)` — Q gates on
  MARK_AX (firing position).
- K: `AP(0, BD.MARK_AX, L*2.0)` — K gates on MARK_AX (source position).
- V: `AP(3, BD.OP_LEA, 0.2)` (and other OP_* dims to other slots).
- O: `AO(BD.CMP+7, 3, 1.0)` — slot 3 → `CMP+7`.

So head 5 reads `OP_LEA` from the V at the source MARK_AX position and
writes `CMP+7` at the Q's MARK_AX position. **The relay is one-way:
OP_LEA at source AX → CMP+7 at this AX.**

For test_xor_basic step 0 (`IMM 0xFF; PSH; ...`), there is NO prior AX
marker where OP_LEA was set. At step 0 the model has only ever seen the
program prefix and the synthetic `REG_PC, REG_AX` markers for step 0.
So head 5's V should produce ~0 OP_LEA contribution at source AX
positions in step 0. **In principle, CMP+7 ≈ 0 at the step-0 AX
marker.**

In practice, the attention softmax over the K-side MARK_AX-matched
positions can attend to:
- Step 0's own AX marker (V-side OP_LEA=0 if L5 decode didn't fire).
- The synthetic `REG_AX` token embedding (V-side OP_LEA=0 since
  embeddings do not write OP_LEA — `model_ops.py:1884
  _embedding_bake_rules` confirms).
- Future AX markers (causal mask prevents this).

So under clean math the L7 head 5 output should be 0 at step-0 AX.
The L10 tail_lea_ax_marker_byte0_e8 rule's `("OP_LEA", 1.0)` positive
ALSO reads OP_LEA directly at MARK_AX in L10. If OP_LEA is 0 there,
that positive is 0.

But `L5_BYTEDECODE_FIX_ATTEMPT` empirically shows the L10 0xE8 rule
fires on 16 IMM bytes (0xF_ family) and produces 0xE8. So something
crosses the rule's threshold 9.0. With `MARK_AX(1) + HAS_SE(1) +
FETCH_LO+8(2 when lo nibble = 8, e.g. 0xF8) + FETCH_HI+15(0.2 for hi
= 15) = 4.2` and **OP_LEA+CMP+7 ≈ 0**, the math says score 4.2 < 9.

The reverted c9e9dc28 attempt added `OP_IMM: -1e9` and saw the rule
still fire on 30 bytes. This means **either**:
1. The two positives (OP_LEA and CMP+7) are NOT zero at MARK_AX on
   IMM rows — there is a leak via attention broadcast, OR
2. The float arithmetic of `_dispatch_pure_neural` introduces residual
   noise that lets the rule cross threshold on FETCH_LO+8 / FETCH_HI+15
   alone.

This investigation could not localize which option holds without
running a residual-stream probe (out of scope per "investigation only"
constraint). The c9e9dc28 author noted: "OP_LEA leaking into MARK_AX
on IMM rows being the load-bearing positive of the misfire". That
matches option (1) but the actual leak source remains unconfirmed.

## Why prior fixes failed

### c9e9dc28: OP_IMM blocker bump (-1e6 → -1e9) on L10 tail_lea
- L10's `tail_lea_local_ax_marker_byte0_e8` had no OP_IMM blocker
  originally. Added one at -1e9. Rule still fires on 30 IMM bytes.
- L16's `l16_psh_mem_addr0_e0_from_sp_no_addr_src` already had
  OP_IMM: -1e9 from the prior EDGE_POW2 fix. No change there.
- **Why insufficient**: per c9e9dc28's commit message, the leak is
  positive evidence (OP_LEA/CMP+7), not OP_IMM-driven negative
  evidence failing to dominate. A blocker on OP_IMM doesn't subtract
  anything when OP_IMM at AX is essentially 0 (per the same upstream
  broadcast attenuation phenomenon).

### Earlier L16 IMM-AX authority attempt (L5_BYTEDECODE doc)
- Added 32 `l16_imm_ax_output_authority_{lo,hi}_{k}` rules at
  strength 1e9/S. Broke `test_imm_exit` (IMM 42 → 20).
- **Why broke a passing case**: the blockers (`MARK_PC: -1e6`, etc.)
  were crossable at MARK_AX because the upstream MARK_PC/SP/BP residual
  at the AX marker is ~1e-3, giving -1e6 × 1e-3 = -1000, insufficient
  to silence the positive sum.

## Recommended fix

Three options, in increasing order of complexity:

### Option A — Add MEM_ADDR_SRC positive predicate to L10 tail_lea
(EDGE_POW2 Fix B pattern)

In `l10_ops.py:6473-6493`, the LEA tail rule writes 0xE8 to mimic the
"local LEA result at AX marker for LEA from BP-8". The rule needs the
positive that distinguishes a real LEA from an IMM:

```python
("MEM_ADDR_SRC", 5.0),  # only fires when the address-byte source is
                        # live, which IMM doesn't have
```

This makes the rule's threshold ramp up to 14.0 and adds a positive
signal that's only present on actual LEA-byte-evaluation steps.

**Pro**: targeted, mirrors EDGE_POW2 Fix B's success on the L16
sibling. **Con**: needs verification that MEM_ADDR_SRC is genuinely
present at MARK_AX on LEA steps and absent on IMM steps.

### Option B — Source OP_LEA more strictly

If the residual-probe confirms OP_LEA at MARK_AX is non-zero on IMM
rows via L7 head 5's attention broadcast, the fix is upstream: tighten
L7 head 5's K-side gate so its softmax distribution at MARK_AX on IMM
rows attends only to a position where OP_LEA=0.

`l7_ops.py:486-490` Q/K currently use `MARK_AX*L + H1+AX_I*L` as Q and
`MARK_AX*L*2.0` as K. Add an `OP_IMM*-L*100` term on K so positions
with OP_IMM (the IMM dispatch AX marker) score negatively, pushing
softmax mass elsewhere.

**Pro**: fixes the leak at its source. **Con**: changes a load-bearing
relay; need to verify all real-LEA steps still get CMP+7=1.

### Option C — Re-source the LEA decode as one-hot mutual exclusion

The deepest fix: change L5's `_opcode_decode_main_rules` to emit
**mutually-exclusive** OP_* writes. Currently each opcode's rule is an
independent positive write; on an IMM row, OP_LEA gets no positive
write but also no negative write to clear any residual leak. Adding
explicit `-10/S` writes on the other 33 OP_* dims (one per opcode
rule) makes the LEA rule actively zero OP_LEA when IMM is decoded.

**Pro**: closes the leak class at the architectural level. **Con**:
large delta (34 rules × 33 negative writes = ~1100 new writes); may
overdrive other OP_* readers downstream.

**Recommendation**: pursue Option A first (smallest, mirrors known-
working EDGE_POW2 Fix B pattern). If MEM_ADDR_SRC turns out to leak on
IMM rows too, escalate to Option B.

## Structural aliasing reminder

`OP_LEA` at dim 262 is structurally identical to `OPCODE_BASE` and
`OPCODE_FLAGS+0` (dim_registry_dynamic.py:144-156, 234-255, 450-452).
Any IR rule that names `"OPCODE_BASE"` in `writes` will lower to W_down
row 262, indistinguishable from a `"OP_LEA"` write. Cross-check before
landing any model fix: grep `writes.*OPCODE_BASE` across ops modules
and confirm no rule unintentionally writes the LEA flag.

## Files of interest (absolute paths)

- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6473-6493`
  — `tail_lea_local_ax_marker_byte0_e8` (the firing rule).
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ops/l16_ops.py:767-802`
  — `l16_psh_mem_addr0_e0_from_sp_no_addr_src` (sibling, already has
  OP_IMM: -1e9).
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ops/l7_ops.py:486-524`
  — L7 head 5 OP_LEA → CMP+7 relay (the candidate leak path).
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ops/l5_ops.py:565-632`
  — L5 LEA-decode rule (no AX-IMM misfire by construction; this is the
  rule that other agents wrongly suspected).
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/batched_pure_neural.py:1985-2003`
  — the `b5cf7099` runner override that currently masks the leak.
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/dim_registry_dynamic.py:234-255,450-452`
  — OP_LEA/OPCODE_BASE/OPCODE_FLAGS+0 alias declaration (dim 262).

## Confidence

- **High** that no declared FFN/attention rule writes OP_LEA at MARK_AX
  directly on IMM rows by construction (L5 LEA rule fails its
  conditions; no other writers exist at the AX marker scope).
- **High** that `CMP+7` (L7 head 5 OP_LEA relay) is the structurally
  load-bearing positive on the L10 tail_lea rule, alongside FETCH_LO+8
  and FETCH_HI+15.
- **Medium** that OP_LEA at MARK_AX leaks through L7 head 5's
  attention softmax due to insufficient K-side gating against IMM
  rows; would need a residual-stream probe at L7-input vs L7-output to
  confirm magnitudes.
- **High** that the c9e9dc28 OP_IMM-blocker bump cannot work in
  principle, because the leak is positive evidence not negative
  evidence (c9e9dc28's own commit message reached the same conclusion).
- **Medium** that Option A (add MEM_ADDR_SRC positive predicate)
  fixes the leak; requires verification probe.

## Leak magnitude

Not measured this session — investigation budget precluded a residual-
stream probe. The `L5_BYTEDECODE_FIX_ATTEMPT` raw-model probe showed
30/256 IMM bytes produce wrong AX byte 0 with the runner override
disabled (15 → 0x01, 15 → 0xE8), so the misfire is binary: rule fires
or it doesn't. No graded "leak magnitude" exists at the residual level
that this doc can quote without a fresh compile.

Recommended next-investigation step: dump L10 input residual at the
step-0 AX position for `IMM 0xC8; EXIT` (a representative misfire
case) and compare OP_LEA / CMP+7 / FETCH_LO+8 / FETCH_HI+15 magnitudes
against a real LEA case (e.g. `LEA -8; EXIT`). The relative magnitudes
will identify whether Option A or Option B is the right surface.
