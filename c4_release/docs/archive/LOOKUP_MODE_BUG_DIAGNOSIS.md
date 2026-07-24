# Diagnosis: `migrate-lookup-mode-hybrid-alu` smoke regression

Branch: `migrate-lookup-mode-hybrid-alu`
Commit: `e280342cda7012cf86ce87243420c828d1d3d541` (2026-05-10 14:34:48 -0400)
Base: `dd71fe73b2212584a4605541b40d3bd517c3c41d` (2026-05-10 13:00:48 -0400)
Investigation context: Unit 9 of `moonlit-sleeping-swan` migration plan.
Investigated by: Wave-1 Unit 9 (read-only diagnosis).

## TL;DR

The `migrate-lookup-mode-hybrid-alu` branch does **not** break smoke in
isolation — `test_imm_exit` and `test_add_basic` **both pass** when the
branch is checked out at `e280342`. The regression only manifests **after**
the branch is merged with current `main`, because `main` integrated two
other branches that day that the diff does not anticipate:

1. **`cleanup-alu-aliases`** (`356e460`, parent commit `22a4828`) — deleted
   the `EfficientDivMod_Neural = ALUDivMod` alias from
   `c4_release/neural_vm/efficient_alu_neural.py:903`.
2. **`vanilla-flattened-divmod`** (`c3c150d`) and `flatten-efficient-divmod`
   — introduced `make_alu_divmod_composite_ops()` in
   `c4_release/neural_vm/unified_compiler/migrated_ops.py`, which installs
   a `FlattenedDivMod` composite onto `model.blocks[10].post_ops` at phase
   `10.8` via a shared `_FlattenedDivModBuilder`.

The branch's only **substantively breaking** change after rebase onto main
is its rewrite of `make_l10_post_op_attach_op(alu_mode='lookup')` so the op
appends `EfficientDivMod_Neural(S, _SetDim)` to `block.post_ops` at
phase `10.7`. Once main has both the alias removed **and** the
FlattenedDivMod composite installed at phase `10.8`, this branch's append
produces either:

- **ImportError** (the literal text of e280342, since
  `EfficientDivMod_Neural` no longer resolves from `vm_step` or
  `efficient_alu_neural`), or
- **`RuntimeError: FlattenedDivMod: missing stages [...]`** (if you rebase
  the literal alias to `FlattenedDivMod`, because two separate
  `FlattenedDivMod` instances now end up on `block.post_ops` — one
  empty/uninstalled at phase 10.7, one fully populated at phase 10.8 — and
  the empty one's forward asserts on the missing-stages guard).

The L8/L9/L10/L11/L12/L13 **op-pinning** half of the branch (flipping
`make_layer{8,9,10}_alu_op` and friends to `kind="block", layer_idx=N,
migrated=True` and removing their inline calls in `set_vm_weights`)
is **NOT** the cause of the regression. Applied to current main without
the post_op change, smoke still passes.

## Method

Reproduction worktree: `/home/alexlitz/Documents/misc/c4_release/` on
branch `migrate-everything-unit9-diagnosis` (created off `main` at
`7f64bd4`). Smoke command:

```
CUDA_VISIBLE_DEVICES=0 timeout 200 python -m pytest \
  c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit \
  -v --tb=short --timeout=180
```

### Step 1 — confirm the branch passes in isolation

Checked out `e280342` (detached HEAD). `test_imm_exit` and `test_add_basic`
both **PASS** (~31s + ~75s = 105s). The branch is internally consistent
with its own base `dd71fe7`.

### Step 2 — simulate the merge

`git cherry-pick --no-commit e280342` onto current main produced merge
conflicts in `c4_release/neural_vm/unified_compiler/migrated_ops.py` and
`c4_release/neural_vm/vm_step.py`. Most of these are docstring conflicts;
the **substantive** post-rebase diff reduces to:

| Change | Status on main HEAD `7f64bd4` |
|---|---|
| `make_layer8_alu_op`: `kind="ffn"` -> `kind="block", layer_idx=8, migrated=True` | not yet migrated |
| `make_layer9_alu_op`: same, plus combined with `_set_layer9_marker_suppress` | not yet migrated |
| `make_layer10_alu_op`: same, `kind="block", layer_idx=10, migrated=True` | not yet migrated |
| `make_layer11_mul_partial_op`: pin to layer 11 block | **already done** on main (since `c3c150d`) |
| `make_layer12_mul_combine_op`: pin to layer 12 block | **already done** on main |
| `make_layer13_shifts_op`: pin to layer 13 block | **already done** on main |
| `make_l10_post_op_attach_op(alu_mode='lookup')`: append `EfficientDivMod_Neural(S, _SetDim)` after `BitwiseBytePropagationPostOp` | conflicts with new `make_alu_divmod_composite_ops` install at phase 10.8 |
| `vm_step.py`: remove inline `_set_layer8_alu` / `_set_layer9_alu`+`_set_layer9_marker_suppress` / `_set_layer10_alu` calls | three deletions, all in the `alu_mode == 'lookup'` branch |
| `vm_step.py`: remove inline `model.blocks[10].post_ops[-1] = EfficientDivMod_Neural(S, BD)` | the swap line is already a comment on main (since `c3c150d`) |

### Step 3 — bisect the substantive deltas

I applied the **op-pinning changes alone** to main (the L8/L9/L10 flips
plus the three inline removals in `vm_step.py`) and re-ran smoke:

- `test_imm_exit` **PASSED** in 31.93s.

Then I additionally applied the `make_l10_post_op_attach_op` change
(append `EfficientDivMod_Neural` -> with the alias gone, I aliased it to
`FlattenedDivMod` for an apples-to-apples runtime test) and re-ran smoke:

- `test_imm_exit` **FAILED** with:
  ```
  RuntimeError: FlattenedDivMod: missing stages ['bd_to_ge', 'div_layers',
  'mod_layers', 'ge_to_bd']. All 3 stage compiler ops
  (phase 10.0/10.1/10.2) plus the install op (phase 10.8) must run before
  forward().
  ```

This pinpoints the regression cleanly. Two `FlattenedDivMod` instances
land on `block.post_ops`:

- The **empty** instance constructed inline at phase 10.7 by the broken
  branch's `block.post_ops.append(EfficientDivMod_Neural(S, _SetDim))`.
  Its forward asserts on missing stages (see
  `efficient_alu_divmod_split.py:414-424`).
- The **populated** instance built by `_FlattenedDivModBuilder.composite`
  (used by all 4 of `make_alu_divmod_composite_ops()` so the stage bakes at
  phases 10.0/10.1/10.2 mutate the composite and the install at phase 10.8
  appends it).

Even without the runtime-guard error, forwarding the model would now run
both DivMod modules sequentially as `block.post_ops[i]`, double-counting
their writes to `OUTPUT_LO/HI`.

### Step 4 — why the branch's `EfficientDivMod_Neural` import resolved at the time of commit

At commit `e280342` the alias still existed:

```python
# efficient_alu_neural.py, e280342 (at line 903):
EfficientDivMod_Neural = ALUDivMod
```

`vm_step.py` re-exports `EfficientDivMod_Neural` from
`efficient_alu_neural`, so `from ..vm_step import EfficientDivMod_Neural`
worked inside the branch. That alias was removed at `22a4828` (2026-05-10
14:17:43, integrated as `356e460`), which is 17 minutes **before**
`e280342` was authored but only landed on `main` via the integration
commit. Since the branch was forked from `dd71fe7` (earlier), it didn't
see the deletion. The import name is therefore stale on today's main.

## Why today's Unit E (L9 LEV attn) succeeded but L9 ALU here failed

Both branches use the same migration pattern in concept — convert an
existing op to `kind="block", layer_idx=N, migrated=True` so it bakes on
the right block before legacy_bake.

**Unit E (`migrate-l9-lev-attn-bakes`, integrated `a43c83a`)** succeeded
because:

1. It migrated **attention** ops (`_set_layer9_lev_addr_relay` and
   `_set_layer9_lev_bp_to_pc_relay`), each writing to a different
   attention head. No shared dependency on a builder/composite.
2. The original inline call sites in `set_vm_weights` were the **only**
   callers — nothing else in the codebase was doubling them or doing
   late mutation on the same `attn` block.
3. The migrated op's bake operates on `block.attn` and uses
   `attn.W_q.shape[0] // attn.num_heads` for HD — no class-import
   dependency that could go stale.

**This branch's L8/L9/L10 ALU migrations** failed at the merge because:

1. The migrated `make_l10_post_op_attach_op(alu_mode='lookup')` rewrote a
   different op (the L10 post_op attach) AND introduced a fresh
   `EfficientDivMod_Neural` instance at phase 10.7 — directly competing
   with an unrelated, simultaneous migration (`make_alu_divmod_composite_ops`)
   that ALSO targets `model.blocks[10].post_ops`. Two migrations editing
   the same `block.post_ops` list, registered in different branches that
   both passed CI independently, fight at merge time.
2. The `EfficientDivMod_Neural` import name became stale 17 minutes before
   commit time without the agent realizing — a coordination problem
   between two parallel agents (`cleanup-alu-aliases` and
   `migrate-lookup-mode-hybrid-alu`) rather than a design flaw in the
   migration pattern itself.
3. Bonus issue (latent, NOT a smoke regression): the branch's expanded
   `make_layer9_alu_op` reads-set adds 6 dims (`MARK_SP, MARK_BP,
   MARK_STACK0, OP_MUL, OP_DIV, OP_MOD`) without symmetric writes. Since
   the op is now `kind="block", layer_idx=9, migrated=True`, the
   dep-graph layer assignment ignores them, so this is **benign**. But
   it would matter if any future refactor flipped the op back to
   `kind="ffn"`.

## Fix plan for Wave 2

Migrating L8/L9/L10 ALU FFN bakes is mostly straightforward — apply the
in-place placeholder -> migrated pattern. **Do NOT touch
`make_l10_post_op_attach_op`** — leave that op alone and let
`make_alu_divmod_composite_ops` own the DIV/MOD module.

### Recipe

For **L8 ALU FFN** (Unit 10), **L9 ALU FFN** (Unit 11), and **L10 ALU
FFN** (Unit 12), each agent:

1. **Edit `migrated_ops.py:make_layer{8,9,10}_alu_op`** — flip
   `kind="ffn"` -> `kind="block"`, add `layer_idx=N`, add
   `migrated=True`. Change the `bake_fn` signature from
   `def bake(ffn, dim_positions, S)` to `def bake(block, dim_positions, S)`
   and replace `ffn` with `block.ffn` inside the body. Keep the existing
   reads/writes set as-is (do not expand it as the broken branch did —
   that change is unnecessary for `kind="block"` ops). Phase stays at
   `N` (8/9/10).

2. **For L9 specifically (Unit 11)** — combine
   `_set_layer9_alu` and `_set_layer9_marker_suppress` into one bake_fn
   per the broken branch's pattern, BUT keep the bake_fn body small and
   capture `n9` from `_set_layer9_alu` inside:

   ```python
   def bake(block, dim_positions, S):
       from ..vm_step import _set_layer9_alu, _set_layer9_marker_suppress
       proxy = _as_setdim_proxy(dim_positions)
       n9 = _set_layer9_alu(block.ffn, S, proxy)
       _set_layer9_marker_suppress(block.ffn, S, proxy, n9)
   ```

   Leave the existing standalone `make_layer9_marker_suppress_op` as-is
   (it's an unused dep anchor at `migrated_ops.py:2011`; its `bake_fn`
   with `start_unit=0` would clobber the units written by
   `make_layer9_alu_op`, but it never runs in production because it's
   `kind="ffn"` without `migrated=True` and `legacy_bake` is present).

3. **In `vm_step.py:set_vm_weights`** (the `alu_mode == 'lookup'`
   branch), delete the inline calls:
   - L8: line 2046, `_set_layer8_alu(ffn8, S, BD)`. Leave the
     `_set_layer8_multibyte_routing(ffn8, S, BD)` call at line 2053 alone
     (its internal `_set_layer8_alu` re-call is idempotent, but the
     migrated op fires first so weights are already in place when the
     re-call lands).
   - L9: lines 2076-2077, the `n9 = _set_layer9_alu(...)` and
     `_set_layer9_marker_suppress(...)` pair.
   - L10: line 2093, `_set_layer10_alu(ffn10, S, BD)`.

   Replace each deletion with a 2-3 line `# MIGRATED ... see
   make_layer{N}_alu_op` comment. Do **NOT** delete the
   `_set_layer8_multibyte_routing` call.

4. **Do NOT modify `make_l10_post_op_attach_op`.** Do **not** add
   `EfficientDivMod_Neural` / `FlattenedDivMod` appends here. The
   `make_alu_divmod_composite_ops` install op (phase 10.8) already owns
   that responsibility on main.

5. **Do NOT modify the L11/L12/L13 op factories.** They are already
   block-pinned migrations on main (`c3c150d` integrated those).

### Pattern selection

Per the playbook in `moonlit-sleeping-swan.md`:

- **Use pattern 1 — "in-place placeholder -> migrated"** for all three
  Wave 2 units. The existing `make_layer{8,9,10}_alu_op` factories are
  the placeholders today; flipping them in place is byte-equivalent.
- **No dep_anchor companion needed.** Block ops bypass the dep-graph
  layer assignment, so removing the `kind="ffn"` declaration does not
  shift downstream attn/ffn placements (block ops carry their own
  `layer_idx`).
- **No `kind="model"` phase 998-999 ordering needed.** The migrated
  block ops at phase N fire before legacy_bake at phase 999, which is
  exactly the order legacy_bake expects (it reads
  `model.blocks[N].ffn.W_up` after the migrated bake has written them).

### Concretely: smallest viable diff for Unit 10 (L8 ALU FFN)

```diff
 def make_layer8_alu_op() -> Operation:
     """L8 FFN: ADD/SUB lo nibble + carry/borrow + LEA + CMP_GROUP."""
-    def bake(ffn, dim_positions, S):
+    def bake(block, dim_positions, S):
         from ..vm_step import _set_layer8_alu
-        _set_layer8_alu(ffn, S, _as_setdim_proxy(dim_positions))
+        _set_layer8_alu(block.ffn, S, _as_setdim_proxy(dim_positions))

     return Operation(
         name="layer8_alu",
         phase=8,
         reads={"MARK_AX", "MARK_PC", "ALU_LO", "AX_CARRY_LO", "FETCH_LO",
                "OP_ADD", "OP_SUB", "OP_LEA",
                "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE"},
         writes={"OUTPUT_LO", "CARRY", "CMP_GROUP"},
-        kind="ffn",
+        kind="block",
         bake_fn=bake,
+        layer_idx=8,
+        migrated=True,
     )
```

And in `vm_step.py`:

```diff
     if alu_mode == 'lookup':
         # Use full lookup tables (pure FFN, many parameters)
         ffn8 = model.blocks[8].ffn
-        _set_layer8_alu(ffn8, S, BD)
+        # MIGRATED: _set_layer8_alu is baked by the migrated block op
+        # make_layer8_alu_op (kind="block", layer_idx=8, migrated=True).
+        # NOTE: _set_layer8_multibyte_routing below internally re-invokes
+        # _set_layer8_alu to discover its unit count; that call is
+        # idempotent.
         _set_layer8_multibyte_routing(ffn8, S, BD)
```

I confirmed locally that this exact diff applied to `7f64bd4` keeps
`test_imm_exit` PASSING in ~32 s on GPU 0.

## Should we discard or fix-up-and-re-apply `migrate-lookup-mode-hybrid-alu`?

**Discard the branch.** The branch's six op-pinning changes are mostly
either (a) already done on main (L11/L12/L13 — 3 ops) or (b) sufficiently
small that re-doing them via Wave-2 Units 10/11/12 with a clean, byte-
minimum diff is faster than reconciling the merge conflicts. The seventh
change (the `make_l10_post_op_attach_op` `EfficientDivMod_Neural` append)
is **actively wrong** on today's main and must NOT be carried forward.

Recommendation:

1. **Delete the branch** (`git branch -D migrate-lookup-mode-hybrid-alu`
   in the agent's workspace once Wave 2 lands; the integration ref on
   `main` doesn't exist since the branch was never merged).
2. **Spawn Wave 2** Units 10/11/12 per the recipe above. Each is a ~10-
   line diff and should pass smoke 2/2 in isolation. Expect coordinator-
   level merge conflicts in `set_vm_weights` (3 different agents
   removing 3 different lines from the same `if alu_mode == 'lookup':`
   block) — trivial to resolve at integration time.
3. **No Wave 2 unit should touch `make_l10_post_op_attach_op` or any
   file under `c4_release/neural_vm/efficient_alu_divmod_split.py`.**
   DIV/MOD installation is already owned end-to-end by
   `make_alu_divmod_composite_ops`.

## File map

| File | Role |
|---|---|
| `c4_release/neural_vm/unified_compiler/layer_compiler.py` (lines 50-104, 417-504) | Defines `Operation`, `ModelLayout`, and `build_model_from_layout`'s dispatch order (per-layer migrated -> block migrated -> all model ops). |
| `c4_release/neural_vm/unified_compiler/full_vm_compiler.py` (lines 59-213) | `compile_full_vm` entry point; registers `make_legacy_bake_op` at phase 999 and the FlattenedDivMod composite ops. |
| `c4_release/neural_vm/unified_compiler/migrated_ops.py:670-1292` | The L8-L13 ALU FFN op factories (`make_layer{8,9,10,11,12,13}_alu_op`, `make_layer13_shifts_op`). |
| `c4_release/neural_vm/unified_compiler/migrated_ops.py:2043-2147` | `make_l10_post_op_attach_op` — DO NOT modify in Wave 2. |
| `c4_release/neural_vm/unified_compiler/migrated_ops.py:2907-2990` | `make_alu_divmod_composite_ops` — installs `FlattenedDivMod` at phase 10.8 via shared builder. |
| `c4_release/neural_vm/vm_step.py:2043-2133` | `set_vm_weights` `alu_mode == 'lookup'` branch — where the L8/L9/L10 inline calls live (the lines Wave 2 will delete). |
| `c4_release/neural_vm/vm_step.py:5663-6564` | `_set_layer8_alu`, `_set_layer8_multibyte_routing`, `_set_layer9_alu`, `_set_layer9_marker_suppress` definitions. |
| `c4_release/neural_vm/efficient_alu_divmod_split.py` (class `FlattenedDivMod`, lines 241-430) | The composite that replaces `EfficientDivMod_Neural`. Its forward asserts on missing stages — this is what trips when the broken branch's empty append fires. |
| `c4_release/neural_vm/efficient_alu_neural.py:903` (at commit `e280342` only) | The `EfficientDivMod_Neural = ALUDivMod` alias that disappeared at `22a4828`. |
| `c4_release/neural_vm/hybrid_alu.py:21-48` | `HybridALUBlock` (wraps lookup_ffn + efficient_alu); installed by `_dispatch_migrated_block_ops` inside `set_vm_weights` itself, so it fires AFTER legacy_bake's inline `_set_layer*` calls (i.e., AFTER the migrated block ops at phase N). |

## References

- Plan: `/home/alexlitz/.claude/plans/moonlit-sleeping-swan.md` (Unit 9
  brief on lines 125-133, Wave 2 brief on lines 135-141).
- Broken branch: `migrate-lookup-mode-hybrid-alu` @ `e280342`.
- Branch base: `dd71fe7`.
- Concurrent integrations on `main` not in the branch base:
  `356e460` (cleanup-alu-aliases), `c3c150d` (vanilla-flattened-divmod),
  `dd71fe7` (flatten-efficient-divmod / nibble long division),
  `a43c83a` (migrate-l9-lev-attn-bakes — the Unit E reference), and
  six more migration integrations.
- Reproduction branch: `migrate-everything-unit9-diagnosis` (this PR).
