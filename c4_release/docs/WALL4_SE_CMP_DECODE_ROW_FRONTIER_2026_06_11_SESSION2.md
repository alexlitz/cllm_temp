# Wall-4 SE CMP — SESSION 2 correction: the SE pipeline is DEAD, not "mis-routed"

Status: **architecturally blocked; prior diagnosis materially corrected.**
Stopped at pristine HEAD `8ad47bf4` with **zero behavioural code change**
(full smoke 31p/10f intact, all 8 guardrails green). This doc
**supersedes the framing of** `WALL4_SE_CMP_DECODE_ROW_FRONTIER_2026_06_11.md`
and the task brief's premise. New, decisive spec_k=0 ground-truth probes
(hook-free, cached build) prove the situation is different — and worse —
than "the SE comparison is correct but reads the wrong row."

## Tools added (spec_k=0, hook-free, cached build)

- `tools/probe_wall4_full.py` — CMP / OUTPUT_LO / OP_* / raw operands
  (ALU_LO/HI, AX_CARRY_LO/HI) and the SE_-tagged mirrors at BOTH the
  binop AX row and the SE row, across all 39 blocks.
- `tools/probe_wall4_cmp_origin.py` — first block that writes CMP at the
  binop AX row + CMP_GROUP / OP_* there.

Run: `CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -u tools/probe_wall4_full.py eq_true eq_false lt_true`

## Authoritative pristine baseline (HEAD 8ad47bf4, `tools/run_full_smoke.py`)

- **Full smoke: 31 passed / 10 failed.** All 8 GUARDRAILS pass
  (`cmp_and_branch, ge_true, gt_true, le_true, lt_true, ne_true, shl,
  shr`). NOTE: `cmp_and_branch` PASSES at this HEAD (prior Wall-3 doc
  claimed it fails at pristine — stale).
- Focused subset (`comparison bitwise basic bit32 shift integration`):
  20 passed / 9 failed.
- TARGETS passing: 2/6 (`and_basic, and_16bit`). FAIL:
  `eq_true (got 0), eq_false (got 17), mul_basic, add_carry_cascade`.
- **CORRECTION to prior docs: `eq_false` FAILS at this HEAD** (got 17),
  it does NOT pass. The "eq_false is the only passing TARGET" claim in
  WALL2/3/4 is stale.

## Decisive finding 1 — the SE cmp pipeline is COMPLETELY DEAD

At the binop SE row (`probe_wall4_full.py`, eq_true 5==5):

```
blk10 SErow SE_operands: SE_ALU_LO=[] SE_ALU_HI=[] SE_AXC_LO=[] SE_AXC_HI=[]
blk11 SErow SE_operands: SE_ALU_LO=[all 16 ≈ -1.41]  SE_ALU_HI=[all 16 ≈ -1.41]
                         SE_AXC_LO=[] SE_AXC_HI=[]
blk11..38 SErow CMP=[]            (SE_CMP never set)
blk11..38 SErow OUTPUT_LO = uniform -240 floor at every index
```

The Wave-A `layer9_step_end_operand_relay` does NOT transmit (Wall 2 was
never landed on main): `SE_AX_CARRY_*` is empty, `SE_ALU_*` is a uniform
negative band (not the operand one-hots), `SE_CMP_GROUP≈0.28` (a weak
leak, not 1.0). The migrated L9 cmp cascade (`_layer9_cmp_rules`,
gated `MARK_SE_ONLY` + `SE_CMP_GROUP`) therefore **never fires** — SE_CMP
is empty at every block. The SE-row cmp_combine writes land on the
uniform -240 OUTPUT_LO floor and are inert.

**So there is no "correct SE comparison result" to re-route.** The
brief's and prior Wall-4 doc's premise ("Wave B moved COMPUTE to SE;
correct SE flags never reach the AX-row decode") is FALSE: the SE
compute produces nothing. Wall 4 as specified (route SE→AX, or re-point
L3 head 5 at SE) cannot help — the SE row carries no signal.

## Decisive finding 2 — lt/le/gt/ge PASS by ACCIDENT at the AX row

The exit code is the binop step's AX-row OUTPUT_LO, relayed by L3 head 5
(`_ax_full_relay_head_spec`, `l3_ops.py:1444`: K@`MARK_AX`, V@`OUTPUT_LO`
→ `AX_FULL`) into the next (EXIT) step. At the binop AX row (eq_true vs
lt_true), the comparison flags are:

```
eq_true (5==5): blk11 AXrow CMP = {CMP+0(hi_lt): 7.03, CMP+1(hi_eq): 0.90}
lt_true (10<20): blk11 AXrow CMP = {CMP+0(hi_lt): 7.23, CMP+1(hi_eq): 0.70}
```

`eq_true` and `eq_false` have **byte-identical CMP at the AX row**
(`{hi_lt 7.03, hi_eq 0.90}`), and lt_true is essentially the same. CMP+0
(hi_lt) is spuriously HOT for ALL comparisons; CMP+1 (hi_eq) is uniformly
WEAK (~0.9). Origin (`probe_wall4_cmp_origin.py`): CMP is empty at the AX
row through block 10 (logical L9) and **first appears at block 11
(logical L10), MARK_AX row** — it is NOT a real per-operand cascade
output (L9's cascade is SE-dead). It is residual leak that the migrated
L8 `cmp_clear` (Wave B → MARK_SE_ONLY, `l8_ops.py:799`) no longer wipes
at the AX row ("L6 attention relay heads write JMP/EXIT/PSH/POP flags
into CMP[0..3] at every position" — that comment is now load-bearing).

Result of the OUTPUT_LO resolution at the AX row (block 22, logical L14):

```
lt_true: AXrow OUTPUT_LO[1]=+140, [0]=-130  -> argmax 1 == LT  (PASS)
eq_true: AXrow OUTPUT_LO[0]=+9.56            -> argmax 0 == default (FAIL: want 1)
eq_false:AXrow OUTPUT_LO[0]=+9.56            -> default 0 (right value, but got=17 downstream)
```

LT/LE/GT/GE pass because the spurious-hot CMP+0 (hi_lt) drives their
`cmp_override_2way(... "CMP+0" ...)` overrides; EQ/NE need
`CMP+1 (hi_eq) AND CMP+2 (lo_eq)`, which are never hot at the AX row, so
EQ falls through to default 0 — wrong for eq_true.

**The guardrails lt/le/gt/ge are therefore a knife-edge accident**, not a
robust path: they rely on CMP+0 being spuriously hot for every
comparison. Any change that "cleans up" the AX-row CMP leak (e.g.
restoring `cmp_clear` at MARK_AX) risks zeroing CMP+0 and regressing all
four ordered guardrails. This is why every prior Wall-2/3 attempt to
activate the "correct" path regressed lt/le.

## Decisive finding 3 — the raw operands at the AX row ARE good enough for EQ

The operands the L9 cascade WOULD need are present and correct at the
binop AX row (block 10, logical L9 output):

```
eq_true  (5,5): ALU_HI={0:11.38} AXC_HI={0:1.26} ALU_LO={0:5.39(artifact),5:6.00} AXC_LO={0:0.32,5:0.94}
eq_false (5,7): ALU_HI={0:11.38} AXC_HI={0:1.26} ALU_LO={0:5.39(artifact),5:5.99} AXC_LO={0:0.32,7:0.94}
```

So `hi_eq` (ALU_HI+k AND AXC_HI+k) is satisfiable at k=0 for both;
`lo_eq` (ALU_LO+k AND AXC_LO+k) is satisfiable at k=5 for eq_true and is
NOT satisfiable for eq_false (A.lo=5, B.lo=7). A per-nibble equality
engine reading these raw operands at the AX row WOULD compute EQ
correctly. The hazards are (a) the index-0 magnitude artifact in
`ALU_LO+0=5.39` (operand-gather hybrid encoding —
`project_operand_gather_hybrid_encoding_is_cmp_alu_root`) and (b) the
~6–11 over-amplification of ALU vs the clean ~0.9–1.3 AX_CARRY, which
makes a balanced AND threshold delicate.

## The real fix (a co-design, NOT a single corrective op)

The only path that can fix EQ without regressing the accidental ordered
guardrails is to compute the EQ/NE result at the binop **AX row** from
the **raw operands**, scoped to OP_EQ/OP_NE, and write OUTPUT_LO there —
WITHOUT touching the spurious CMP+0 the ordered ops depend on. Concretely:

1. **New L10 (logical) FFN op, scoped `OP_EQ + MARK_AX`** (and a separate
   `OP_NE + MARK_AX`), a 16×16 nibble-pair cross-product (≈256 units per
   opcode): each unit fires on
   `ALU_HI+h AND AX_CARRY_HI+h AND ALU_LO+l AND AX_CARRY_LO+l` and writes
   `OUTPUT_LO[1] += , OUTPUT_LO[0] -= ` (EQ) at the AX row; the OP_EQ
   default writes OUTPUT_LO[0]. NE is the complement. This reads CLEAN
   inputs (not stacked on the broken CMP), so it is NOT whack-a-mole.
   - Threshold MUST reject the `AX_CARRY_LO+0=0.32` artifact and the
     `ALU_LO+0=5.39` index-0 artifact while passing the true ~0.9 / ~6
     matches. Use a balanced 4-way AND with per-term weights that
     normalise the ALU over-amplification (e.g. ALU terms ×(1/6),
     AX_CARRY terms ×1.0, threshold ~3.3).
   - Scope `OP_EQ`/`OP_NE` ONLY → structurally cannot touch
     lt/le/gt/ge (the knife-edge guardrails) or the shared CMP dims.
2. **Verify NE stays green**: `ne_true` currently PASSES via the AX-row
   default (NE default = 1, no hi_eq/lo_eq → stays 1). An NE override
   must fire ONLY when the operands are truly equal; gate carefully or
   leave NE untouched if `ne_true` is the only NE smoke (it is) and
   `eq` is the sole target.
3. **Downstream block-37 (logical L25) band**: eq_false's `got=17`
   (not 0) shows a downstream OUTPUT_LO corruption AFTER the AX-row
   default-0 is already correct. Trace `tools/probe_wall4_full.py`
   blk36-38: the L25 post-op writes a uniform `OUTPUT_LO[1..15]=238,
   [0]=-221` band at the AX row. The EQ op's OUTPUT_LO[0] write must be
   strong enough to survive (or the L25 band must be gated off CMP/EXIT
   rows). This is a SECOND, independent obstacle for eq_false that the
   eq_true fix does not address — budget for it.

## Why this was not landed this session

- Cold build cost: every `neural_vm/**` source byte is hashed into the
  disk-cache key (`_legacy_redirect._hash_source_bytes`), so ANY op edit
  forces a full ~10–12 min cold bake; a pristine tree is a ~10 s cache
  load. The 256-unit threshold tuning against over-amplified, artifact-
  laden operands realistically needs 2–3 cold cycles plus a full-smoke
  verify — beyond a single session's safe budget without risking a
  landed regression.
- The brief's premise (route SE→AX) is invalidated by Finding 1, so the
  cheap "small attention copy SE→AX" option is a dead end (nothing to
  copy). The viable fix is a new from-operands AX-row EQ engine
  (Findings 2+3) plus the eq_false L25-band obstacle (Finding 3.3).

Per the brief's STOP clause, held at the green pristine baseline rather
than landing a guardrail regression.

## SESSION 3 (2026-06-12): the EQ engine landed; efficient-mode re-home + build nondeterminism

The per-nibble EQ engine of Findings 2+3 is now implemented and PROVEN to
flip `eq_true` -> 1 and keep `eq_false` -> 0 with ALL 8 guardrails green.

**What landed.** `_layer10_alu_eq_engine_rules` (l10_ops.py): 256 units, a
4-way AND on `(ALU_HI+h, AX_CARRY_HI+h, ALU_LO+l, AX_CARRY_LO+l)` gated
`OP_EQ` at MARK_AX. The matching unit writes `CMP+1` (hi_eq) and `CMP+2`
(lo_eq) -- the two flags the LIVE `ComparisonCombine` EQ override reads --
so EQ flips to 1 through the SAME proven path lt/le use via CMP+0. CMP+0
(the load-bearing accidental hi_lt leak) and CMP+3 are NEVER written, so
lt/le/gt/ge are structurally untouched. Threshold/weights tuned offline
(`tools/tune_eq_engine.py`) against the probed AX-row operand bands for the
widest clean margin (true match sum 6.47 vs the index-0 artifact ceiling
5.27, threshold 5.872 -> 0.60 headroom each side).

**The keystone correction to Findings (efficient-mode discards layer10_alu's
FFN).** The smoke gate runs `trust_neural_alu=True` => `alu_mode='efficient'`.
In efficient mode `make_efficient_l10_andorxor_wrap_op` (alu_ops.py) does
`block.ffn = cleanup_ffn`, REPLACING L10's FFN -- so the eq engine baked into
the lookup-mode `layer10_alu` FFN is silently DROPPED (a CMP-write search
across all 39 blocks finds zero eq units). The engine MUST be re-homed into
the efficient wrap, MERGED into `cleanup_ffn` (NOT appended as a post_op,
which adds a passthrough block and shifts every downstream block index ->
breaks lea/cmp_and_branch). With the merge, CMP+1/CMP+2 appear at ~block 15
(`6-8` magnitude) and ComparisonCombine flips EQ -> exit 1. eq_false's
operands don't match so no unit fires; CMP stays clean -> exit 0. (The doc's
Finding 3.3 "eq_false L25-band obstacle / got 17" is a STALE-CACHE artifact:
at the correct disk_cache=False config eq_false already decodes 0 pristine.)

**TWO probe-path traps that wasted the prior sessions' analysis:**
1. `compile_full_vm_dynamic(disk_cache=False)` with DEFAULT args builds a
   STRUCTURALLY DIFFERENT model (lea_basic/cmp_and_branch baseline FAIL,
   eq_false PASS) than production. Any no-cache probe MUST pass the runner's
   exact args: `alu_mode='efficient', n_heads=DEFAULT_N_HEADS,
   ffn_hidden=DEFAULT_FFN_HIDDEN, max_seq_len=4096`. With those it matches the
   cached production baseline (35 pass, 8 guardrails, eq_false PASS).
2. The shared on-disk compiled-vm cache (`~/.cache/c4_release/compiled_vm`) is
   polluted by concurrent agents; `tools/run_full_smoke.py` /
   `build_groundtruth_probe` repeatedly served a STALE model (eq_false=17,
   no eq engine) for the SAME source hash. Use `disk_cache=False` +
   correct args (`tools/run_full_smoke_nocache.py`) for any trustworthy
   measurement.

**Remaining obstacle = BUILD NONDETERMINISM (environmental, not the engine).**
Across repeated disk_cache=False cold builds of the IDENTICAL source, the
eq_true *decode trace* is intermittent: HEALTHY builds (ax_row=155, the normal
EQ trace) give eq_true=1 with CMP+1/CMP+2 written and 8/8 guardrails;
occasional builds decode a different trace (ax_row=140, CMP leak sometimes
entirely absent) and give eq_true=0. The guardrails (lt/le/gt/ge via CMP+0)
are stable in BOTH, so ComparisonCombine placement is stable -- the variance
is in the autoregressive EQ decode itself (eq sits on a decision boundary),
likely GPU-contention corruption during the bake (3+ agents share the GPUs).
Widening the firing margin (0.43 -> 0.60) did NOT remove the intermittency,
confirming it is not the unit-firing threshold. Probes:
`tools/probe_eq_correct_config.py` (CMP/OLO trajectory),
`tools/run_full_smoke_nocache.py` (production-config smoke),
`tools/tune_eq_engine.py` (offline threshold tuner).
