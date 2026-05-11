# Model Regression Bisect — STEP_END / HALT Emission

## TL;DR

The model regression that Wave A identified — model emitting `REG_PC + value +
REG_AX + value + zeros` instead of the full 35-token STEP sequence ending in
`STEP_END` — was introduced by:

- **Offending commit**: `881cfbcc` "Integrate worktree-agent-a02bd3117fa17040f (A2 L1 migration)"
  - Dev-branch tip: `6ab853bc` "Migrate L1 setup ops to compiler dispatch"
- **Parent (HEALTHY)**: `ccd93c6` "Integrate worktree-agent-a6893efcef9837dd3 (Phase 7 heap/DIV tests)"

The regression is structural: an op pinned to layer 1 in the legacy `set_vm_weights`
pipeline was migrated to the compiler dispatch and got mis-anchored to layer 0
because the `LayerCompiler` uses **dep-based earliest-feasible assignment**,
not layer-name-based pinning. The same root cause has compounded across the
~50 subsequent migration commits; multiple `migrated=True` ops at HEAD are
similarly mis-pinned.

## How the bisect was performed

- Worktree: `/tmp/d2-bisect` (detached HEAD)
- Reproducer: `/tmp/bisect_check.py`. Builds the model via
  `AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)`, then
  autoregressively generates 35 tokens from the bare CODE+DATA prefix for
  bytecode `[(Opcode.IMM<<8)|5, Opcode.EXIT]`. Verdict:
  - HEALTHY = position 34 == `Token.STEP_END`
  - BROKEN  = position 34 != `Token.STEP_END` (typically zeros from ~pos 11
    onward, never see REG_SP/REG_BP/STACK0/MEM markers)
- Bisect range: `b0037c1` (good) … `f993b6d` (bad / HEAD when bisect started)
- ~7 steps, plus one `git bisect skip` for `6ab853b` which fails to build
  (`ModuleNotFoundError: No module named 'neural_vm.unified_compiler.builder'`).
  After the skip, the bisect terminates with `6ab853bc` and `881cfbcc` as the
  candidates; both carry the same `vm_step.py` + `migrated_ops.py` diff and
  `881cfbcc` is the integrate of `6ab853bc`, so the latter is the *origin* of
  the offending diff and the former is the *commit on `main` that first
  exposed it*.

### Bisect timeline

| Commit    | Subject (truncated)                                       | Verdict |
|-----------|-----------------------------------------------------------|---------|
| `b0037c1` | Establish pure-neural mode                                | HEALTHY |
| `5bdd226` | Add diag_pure_neural.py for phase 1-7 status summary      | HEALTHY |
| `ccd93c6` | Integrate worktree-agent-a6893efcef9837dd3 (Phase 7)      | HEALTHY |
| `881cfbc` | Integrate worktree-agent-a02bd3117fa17040f (A2 L1)        | **BROKEN** |
| `6ab853b` | Migrate L1 setup ops to compiler dispatch                 | (build fail; skipped) |
| `da68706` | Integrate worktree-agent-ad8cacfb7dde392dc (A3 L2)        | BROKEN  |
| `49dcea5` | Integrate worktree-agent-a1deec502a1291340 (B3 post-pass) | BROKEN  |
| `5a05614` | Add neural PUTCHAR/GETCHAR semantics for pure_neural      | BROKEN  |
| `f993b6d` | HEAD (Integrate Unit 9 diagnosis report)                  | BROKEN  |

## The offending diff

`881cfbc` ≡ `6ab853b` flips two ops to `migrated=True` and **removes** the
direct calls in `set_vm_weights`:

```
diff --git a/c4_release/neural_vm/unified_compiler/migrated_ops.py …
@@ make_layer1_ffn_op …
         kind="ffn",
         bake_fn=bake,
+        migrated=True,
     )

@@ make_layer1_threshold_attn_op …
         kind="attn",
         bake_fn=bake,
+        migrated=True,
     )

diff --git a/c4_release/neural_vm/vm_step.py …
@@ in set_vm_weights …
-    # ===== LAYER 1: Fine thresholds + STEP_END detection =====
-    attn1 = model.blocks[1].attn
-    if hasattr(attn1, 'alibi_slopes') and attn1.alibi_slopes is not None:
-        attn1.alibi_slopes.fill_(ALIBI_S)
-        attn1.alibi_slopes[3] = 0.0
-    _set_threshold_attn(attn1, [0.5, 1.5, 2.5], [BD.L1H0, BD.L1H1, BD.L1H2], …)
-    base = 3 * HD
-    attn1.W_q[base, BD.CONST] = 10.0
-    attn1.W_k[base, BD.MARK_SE_ONLY] = 10.0
-    attn1.W_v[base + 1, BD.MARK_SE_ONLY] = 1.0
-    attn1.W_o[BD.HAS_SE, base + 1] = 1.0
-    _set_threshold_attn(attn1, [6.5], [BD.L1H4], ALIBI_S, HD, heads=[4])
-    ffn1 = model.blocks[1].ffn
-    _set_layer1_ffn(ffn1, S, BD)
+    # Migrated: L1 threshold attention (heads 0-4 incl. STEP_END global head)
+    # and L1 FFN (STACK0_BYTE0 + BYTE_INDEX flags) are now baked via the
+    # compiler dispatch (see make_layer1_threshold_attn_op, make_layer1_ffn_op).
```

The L1 op declares:

```python
return Operation(
    name="layer1_threshold_attn",
    phase=1,
    reads={"IS_MARK", "MARK_SE_ONLY", "CONST"},
    writes={"L1H0", "L1H1", "L1H2", "L1H4", "HAS_SE"},
    kind="attn",          # ← problem: dep-anchored, not layer-pinned
    bake_fn=bake,
    migrated=True,
)
```

`kind="attn"` makes the LayerCompiler assign this op the **earliest layer at
which all reads are satisfied**, modulo phase tie-breaking. The reads
(`IS_MARK`, `MARK_SE_ONLY`, `CONST`) are written by the embedding (layer-equivalent
0), so the op gets assigned to **block 0**, not block 1. `phase=1` only breaks
topological ties; it does not pin the op to layer 1.

## What is unintentionally lost

The L1 threshold-attention layer programs:

1. **Heads 0-2**: fine ALiBi thresholds at 0.5 / 1.5 / 2.5 writing `L1H0/L1H1/L1H2`.
2. **Head 3 (global)**: `STEP_END` existence detection — `alibi_slopes[3]=0.0`
   plus a `(Q=CONST, K=MARK_SE_ONLY, V=MARK_SE_ONLY, O→HAS_SE)` quad.
3. **Head 4**: threshold 6.5 head writing `L1H4`.

Together these are the **STEP_END detection chain**. With the bake hitting
block 0 instead of block 1:

- Block 0's threshold heads (`H0`..`H7` from `make_layer0_threshold_attn_op`,
  which IS correctly pinned via `kind="block", layer_idx=0`) get **clobbered**
  by the L1 bake (overwriting heads 0-4 of block 0 with the L1 thresholds,
  and forcing `alibi_slopes[3]=0.0` which the L0 op did not request).
- Block 1's attention is left with zero weights for the threshold heads and
  for the `HAS_SE` writer head, so the model loses the ability to:
  - Emit `STEP_END` at position 34 of every step (Head 3 is gone)
  - Identify STACK0 byte 0 (Head 4 is gone)
  - Detect step-section boundaries (Heads 0-2 are gone)

That matches Wave A's symptom exactly: the model emits `REG_PC` + bytes,
`REG_AX` + bytes, then zeros — it never gets the "this is the end of the step"
signal that Block 1's Head 3 used to provide.

## Verifying the diagnosis

A monkey-patched run with `make_layer1_threshold_attn_op` and
`make_layer1_ffn_op` returning `kind="block", layer_idx=1, migrated=True`
versions of the op (so the bake_fn receives `model.blocks[1]` and writes into
the right attention/FFN) restores the healthy output at `881cfbcc`:

```
STEP0_TOKENS: REG_PC 10 0 0 0 REG_AX 5 0 0 0 REG_SP 0 0 1 0 REG_BP 0 0 1 0 STACK0 0 0 0 0 MEM 0 0 0 0 0 0 0 0 STEP_END
POS34_IS_STEP_END: True
VERDICT: HEALTHY
```

## State at HEAD

At HEAD (`f993b6d`), the L1 op is still mispinned (Layer 0). Several
*subsequently* migrated ops are also mispinned because the same
`kind="attn"|"ffn"` dep-anchor pattern was used without `layer_idx`:

| Op (migrated=True) | Assigned layer | Intended layer |
|---|---|---|
| `layer1_threshold_attn` | 0 | 1 |
| `layer2_threshold_attn` | 1 | 2 |
| `layer3_carry_forward_attn` | 2 | 3 |
| `layer14_mem_generation` | 11 | 14 |
| `layer15_memory_lookup` | 12 | 15 |
| `layer16_lev_routing` | 6 | 16 |

(`layer1_ffn`, `layer2_mem_byte_flags` happen to be correctly slotted at
layer 1 and 2 by coincidence of the dep graph, so they are not lost — but
that is luck, not design.)

Note: applying the L1 fix alone at HEAD does NOT restore healthy output —
the model produces a degenerate "REG_PC 255 REG_PC 255 …" pattern, indicating
that subsequent migrations (L2 threshold, L3 carry-forward, L14/15/16) need
the same fix before HEAD will emit STEP_END.

## Recommended fix

Do **NOT** simply revert `881cfbcc`. The migration is intentional and the
correct path forward. Instead, the migrated L1 ops (and all other
`migrated=True` ops with `kind="attn"|"ffn"` and a hand-set target layer
implied by their name/phase) must pin to a specific block:

### Option A — minimal (only fix the offender at b0037c1+1)

In `c4_release/neural_vm/unified_compiler/migrated_ops.py`, change both L1
factories to mirror the pattern already used by `make_layer0_threshold_attn_op`:

```python
return Operation(
    name="layer1_threshold_attn",
    phase=1,
    reads={"IS_MARK", "MARK_SE_ONLY", "CONST"},
    writes={"L1H0", "L1H1", "L1H2", "L1H4", "HAS_SE"},
    kind="block",          # ← was "attn"
    layer_idx=1,           # ← new
    bake_fn=block_bake,    # ← wraps the existing bake_fn so it receives
                           #   block and unpacks block.attn internally
    migrated=True,
)
```

`block_bake`:
```python
def block_bake(block, dim_positions, S):
    orig_bake(block.attn, dim_positions, S)
```

Same for `make_layer1_ffn_op` with `block.ffn`.

### Option B — root-cause (fix the dispatcher)

Teach `LayerCompiler._assign_layers` to honor an explicit `layer_idx` even
when `kind` is "attn"/"ffn":

```python
for op in topo:
    if op.layer_idx is not None and op.kind in ("attn", "ffn"):
        layer = op.layer_idx
    else:
        # existing earliest-feasible logic
        ...
```

This lets the migrated ops keep `kind="attn"|"ffn"` (cleaner — no boilerplate
block_bake wrapper) and just add `layer_idx=1`. The other mispinned ops
listed above (`layer14_mem_generation` etc.) get fixed by adding `layer_idx=N`
to their definitions.

**Option B is the preferred fix** because:
1. It avoids the "wrap bake_fn in block_bake" boilerplate at every migration.
2. It surfaces the pinning intent at the op declaration (one line) rather
   than in the wrapper.
3. It documents that *the dep-graph-only assignment is insufficient when an op
   has a hand-set layer index from the legacy pipeline*.

After applying Option B + `layer_idx` annotations to every `migrated=True`
op with `kind in ("attn", "ffn")`, HEAD should restore the healthy STEP_END
emission and Phase 1 should reach 13/13 without the `run_vm.py` synthesize-
STEP_END workaround.

## Files touched by the regression (informational, no revert)

- `c4_release/neural_vm/unified_compiler/migrated_ops.py` — added
  `migrated=True` to two ops (lines ~115, ~783)
- `c4_release/neural_vm/vm_step.py` — removed the L1 threshold/FFN bake from
  `set_vm_weights` (lines ~1826-1856 in the parent)

## Reproducer (canonical)

```python
import sys
sys.path.insert(0, "/path/to/c4_release")
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode

runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
bc = [(5 << 8) | Opcode.IMM, Opcode.EXIT]
runner._memory = {}; runner._mem_history = {}; runner._mem_access_order = []
runner._last_bp = 0x10000; runner._last_sp = 0x10000
runner._last_pc = None; runner._last_ax = 0
context = runner._build_context(bc, b"", [], "")
gen_ctx = list(context)
toks = []
for _ in range(35):
    nxt = runner.model.generate_next(gen_ctx, use_incremental=False)
    toks.append(nxt); gen_ctx.append(nxt)
print("STEP_END at pos 34:", toks[34] == Token.STEP_END)
```

At a HEALTHY commit (e.g., `b0037c1`, `ccd93c6`): emits `STEP_END` at pos 34.
At BROKEN (`881cfbcc`+): emits zeros from ~pos 11 onward.
