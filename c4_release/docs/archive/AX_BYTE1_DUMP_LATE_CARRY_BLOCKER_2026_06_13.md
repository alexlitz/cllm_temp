# AX byte-1 DUMP carry — late-layer path found, blocked by L13 build perturbation (2026-06-13)

**Status:** NOT LANDED. Approach 1 (the cheap partial) was attempted as a
late-layer (L13 head 6) carry of the `H1` byte-1 one-hot and REVERTED: the
head is provably inert at runtime yet inserting the op catastrophically
regressed smoke (48/3 → 11/40) via a build-level perturbation of the L13
attention block's ALiBi state. This doc records the NEW, actionable findings so
the next pass builds the right thing without re-deriving them.

Supersedes the "Approach 1 = L3 carry" framing in the brief and in
`AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md`: an L3/L4 carry is
**impossible** (no byte-1 value exists pre-L9), and the only carriable signal
(the `H1` one-hot itself) is born at L9. The viable path is a **late-layer
back-attention carry**, blocked only by the L13 build fragility documented below.

## Confirmed root (re-validated this session, spec_k=0, hook-free)

`tools/probe_ax_logit_attrib.py` on `add_0` (654+114): the byte-1 emission at the
byte-1 predictor row (= AX marker + 1, the AX byte-0 token row) is driven >99% by
the `H1` band (dims 67..73) as a one-hot `H1+(byte_value+2)`. IMM step: `H1+4 =
12.14, contrib +60.7` → emits 0x02. PSH step: `H1+4 = 0.0` → emits 0x00. Exactly
as the prior doc states.

## NEW finding 1 — the `H1` one-hot is BORN at L9 (block 10), not carriable earlier

`tools/probe_h1_carry_design.py` (block-by-block trace of the byte-1 predictor
row's `H1` band, id 0):

| block | IMM-step (fresh) H1[2..5]   | PSH-step (carried) H1[2..5] |
|-------|-----------------------------|------------------------------|
| 3..9  | `[0.94, 0, 0, 0]`           | `[0.94, 0, 0, 0]`            |
| **10**| `[2.82, 0, 12.14, 0]`       | `[0.94, 0, 0, 0]`            |
| 11..39| `[2.82, 0, 12.14, 0]` (held)| `[0.94, 0, 0, 0]` (held)     |

The one-hot appears at **block 10 = logical L9** on the fresh step and is
byte-identical from block 10 → block 39 (the LM head). On the carried step it is
NEVER born.

## NEW finding 2 — there is NO value-faithful byte-1 band pre-L9 to carry

At block 4 (L4) the prev AX **marker** row holds only `CLEAN_EMBED_LO=(slot 4)`
(the constant REG_AX token embedding) — NO `OUTPUT`/`AX_FULL`/`AX_CARRY` value.
The decoded AX value only materialises at L8+. `AX_CARRY` at the byte-1 predictor
row decodes to **byte 0** (e.g. 432+32: AX_CARRY → 0xd2≈208=byte0), confirming the
prior doc: `AX_CARRY` is the byte-0 carry, not a byte-1 source. So **any carry
head at a layer ≤ L8 has nothing byte-1-faithful to read.** The brief's
"L3 head 1 mirror for byte 1" cannot work.

## NEW finding 3 — the viable path: late-layer back-attention to the prev step's predictor row

On a carried (PSH) step, the **previous step's byte-1 predictor row** — which DID
receive the `H1` one-hot at L9 — sits at an earlier position and IS visible to
causal attention. Verified: the prev step's m+1 row holds `H1=[_,_,2.8,_,12.1,_,
1.5]` from block 10 through block 39; the current carried step's m+1 row holds only
`[_,_,0.9,...]`. So a head AFTER block 10 and BEFORE the LM head can attend BACK to
the prev predictor row and V-copy its `H1` one-hot forward.

Required gating data (all measured stable across the add/sub corpus at the L13
read point = block 12):

* **Byte-1 predictor row signature:** `IS_BYTE=1` AND `L1H1+2 = 1.0` (the L1
  fine-threshold head's slot-2 marker-distance class). NOT `H1+AX_I` (that slot
  is ~0 here — an early mis-identification that broke the first build attempt).
  This signature is shared by EVERY step's byte-1 predictor row, so it both
  fires the Q (current carried step) and matches the K (prior step's predictor
  row holding the one-hot).
* **Carried-vs-fresh gate:** the 32-wide `AX_CARRY` band SUM at the byte-1
  predictor row is **~ -988 on a FRESH-AX step** and **~ +2.7 on a CARRIED
  step**. A Q term `+w · Σ AX_CARRY` (w≈2) thus drives Q strongly negative on
  fresh steps (head off → existing one-hot untouched) and ~0 on carried steps
  (head fires). This is a clean, value-independent discriminator.
* **ALiBi:** runtime applies `-slope·|q−k|` (vm_step.py:478). POSITIVE slope
  (the L13 default 0.5) prefers the NEAREST prior predictor row = the previous
  VM step's. (The L13 relay heads 4/5 use -1.0 to prefer the OLDEST; the carry
  needs the opposite.)

## THE BLOCKER — L13 is a fragile build surface; inserting any op perturbs it

The attempt added `make_layer13_ax_byte1_dump_carry_op` at L13 head 6 (heads 6/7
are free: 0-2 mem_addr_gather, 3 bitwise_byte1_gather, 4 sub_minuend_relay,
5 add_addend_relay). With the corrected `L1H1+2` gating the head became INERT —
`tools/probe_h1_carry_design.py` showed **0 rows** where it changed the `H1`
band at block 13. Yet smoke regressed 48/3 → 11/40 (notably ALL of SHL/SHR,
add/sub 16-bit, cmp_and_branch), and the emitted token stream diverged (marker
positions shifted 141 → 156 on id 0).

Because the head is provably inert at runtime, the regression is **purely a
build-level perturbation**, almost certainly the L13 attention block's ALiBi
slope state:

* `make_layer13_mem_addr_gather_op`'s bake does `attn.alibi_slopes.fill_(0.5)`,
  then heads 4/5 (`sub_minuend_relay`/`add_addend_relay`) override their slots to
  `-1.0`. The L13 FFN shift family (`layer13_shifts`, units 0..4095) routes
  through `AX_CARRY_LO[s]+AX_CARRY_HI[0]` and is highly sensitive to this block.
* Inserting a new op into the L13 sequence (even one that only sets
  `alibi_slopes[6]=0.5` and writes 7 `H1` cells) reordered the bake / slope
  writes enough to break the -1.0 overrides and/or the shift FFN routing.
  This is the exact ALiBi-clobber class in memory note
  `project_attention_dsl_alibi_slope_gap` (alibi_slope=None lets ops write
  slopes imperatively with no compile-time ownership check).

## THE SECOND BLOCKER — the clean gate signals are NOT available at a build-safe host

The L11 re-host (head 2, declared `alibi_slope=`, no shift-FFN co-tenant) was
ALSO attempted and reverted. It did NOT perturb the build (markers stayed at
49/85/120/156, no cascade), but the head **cannot be gated correctly at L11's
read point (block 10)**:

* The clean carried-vs-fresh `ΣAX_CARRY` separation (~ -988 vs ~ +2.7) only
  exists at **block 12** (L13's input). At **block 10** (L11's input) the fresh
  IMM step's `ΣAX_CARRY` is **+62**, the fresh ADD step is ~+2, and the carried
  PSH step is **+2.7** — fresh and carried are NOT separable there. The gate
  signal is created by blocks 11-12.
* The `H1[3:7]`-band-sum (a value-independent "one-hot present?" gate) is +13.7
  on a fresh IMM step but **0.0 on BOTH the carried PSH step AND the fresh ADD
  step** at block 10 — so it cannot distinguish carried from fresh-ADD either.
* The `L1H1+2` row signature is **not program-stable** at block 10: it is 1.0
  for `654+114` step 0/1 but 0.0 for `754+104` / `913+558` step 0. So it is not
  a reliable Q-fire / K-match discriminator across the corpus at this read
  point.

So there is a genuine **read-point trade-off**:
* **block 12 (L13 input)** has the clean gate + stable signature, but L13 is a
  build-fragile host (ALiBi/shift-FFN — see the first blocker).
* **block 10 (L11 input)** is a build-safe host, but the gate signals haven't
  crystallised yet.

The clean signals first appear at **block 12**. The only build-safe host that
reads block 12 would be an op at **block 13 that does NOT touch the L13
attention block's ALiBi/shift state** — i.e. the real fix is to FIRST decouple
the L13 attention block (move the shift-FFN's AX_CARRY dependence or the ALiBi
overrides off the shared block) so a clean head can be added there, OR find/
create a free-head attention block whose read point is exactly block 12.

## Recommended next pass (genuinely multi-step)

1. **Decouple L13 first.** The L13 attention block co-hosts `fill_(0.5)` +
   per-head `-1.0` ALiBi imperative overrides (heads 4/5) and an AX_CARRY-routed
   4096-unit shift FFN. Migrate those ALiBi writes to declared `alibi_slope=`
   spec fields (memory `project_attention_dsl_alibi_slope_gap`) and confirm the
   block tolerates an added inert head (the smoke-neutral precondition this
   session lacked). THEN add the carry head there with the clean block-12 gate.
2. **OR** add a new attention block whose read point is block 12 with free
   heads, if the architecture admits one.
3. Use the verified gating: `IS_BYTE + L1H1+2` row signature **at block 12** (it
   IS stable there — re-confirm), `+2·ΣAX_CARRY` carried gate (clean at block
   12), positive ALiBi slope (nearest-prior). The first L13 attempt's only
   semantic bug was `H1+AX_I` vs `L1H1+2`; fixing it made the head inert/clean
   at block 12 — the residual blocker was purely the L13 build perturbation.
4. Ensure a TRUE sink when off: K slot-33 `CONST` sink with V=0.
5. Byte-identity gate with `compare_symbolic_to_lowered_attn`; verify
   `tools/probe_ax_carry.py` recovers id-0 byte 1 on step 1; measure
   `tools/run_1096_canonical.py --criterion full_trace --ids 0-49`
   (add baseline 12/50) and `--ids 50-99` (sub baseline 5/50); smoke must hold
   48/3.

**Bottom line:** this is NOT a cheap single-head partial. The carry head design
is correct, but every build-safe host reads too early for the gate, and the
only host that reads late enough (L13) must be structurally decoupled first.
True multi-session, overlapping the L13/ALiBi-migration and the convergent
multi-byte work.

This is still the brittle, corpus-bounded partial (covers byte1 ≤ 4 because
`H1` is 7-wide → `H1+(v+2)` caps at v=4). The value-general fix (re-point the
dump to a real 8-bit band) remains Approach 2 / a separate multi-session task,
overlapping the convergent multi-byte root (#206/#213/#217).

## Baselines (HEAD 030d08e1, GPU 0, spec_k=0, full_trace)

* add (ids 0-49): **12/50**
* sub (ids 50-99): **5/50**
* smoke: **48 passed / 3 failed** (mul_basic / mul_overflow / simple_function)

## Tools added this session (kept, reusable)

* `tools/probe_h1_carry_design.py` — block-by-block `H1`-band trace at the
  byte-1 predictor row + byte-1-predictor row-signature dump (the
  `L1H1+2` discovery) + the `AX_CARRY`-sum carried-vs-fresh gate measurement.
  spec_k=0, single-forward-per-block (fast).
