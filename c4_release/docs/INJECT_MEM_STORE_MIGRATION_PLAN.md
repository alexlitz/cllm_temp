# Migration Plan: `_inject_mem_store` → Pure Neural

**Author:** scoping agent
**Date:** 2026-05-11
**Branch:** `i5-mem-store-analysis`
**Status:** scoping only, no code modified

---

## TL;DR — Verdict

**Defer this migration.** Unlike the other four injection methods (`_add_code_addr_keys`,
`_inject_thinking_markers`, `_inject_initial_pc`, `_inject_mem_exec*`) which inject
*position-deterministic* metadata, `_inject_mem_store` (when active) propagates
**runner shadow memory state** into the residual. A pure-neural replacement is not a
single op rebind — it requires either (a) a new memory-attention sub-architecture
that gathers values across an unbounded history, or (b) accepting that the existing
"current-step" MEM_STORE path from `L6 head 6 → L7 head 7` already suffices, and
this injection only needs to be re-applied to *retained* prior steps after KV-cache
eviction. Option (b) is what the codebase is converging on (`_inject_mem_exec_autoregressive`
now sets `MEM_STORE=2.0` on **all** MEM markers including history), making
`_inject_mem_store` largely **redundant** — but only when KV-cache eviction is
disabled. Real elimination requires either:

1. A 1-line removal once it can be proven `_inject_mem_exec_autoregressive` always
   covers the same positions (a cleanup, not a migration), **or**
2. A full residual-stream rework to replace runner-driven shadow memory with
   neural attention over prior PSH steps (multi-week effort, requires training).

This document picks apart both options below.

---

## 1. Exact Injection State

### Source method
`c4_release/neural_vm/neural_embedding.py:312-337` — `_inject_mem_store`.

### What it writes
| Aspect | Value |
|---|---|
| **BD dim (logical)** | `BD.MEM_STORE = 34` |
| **Default residual dim** | 459 (single dim) |
| **Compiler-mode dim** | resolved by `self._dim("MEM_STORE")` |
| **Value written** | `x[b, i, mem_store] = 1.0` |
| **Positions selected** | tokens with `token_ids[b, i] == Token.MEM` and `start_pos <= i < min(_mem_history_end, S)` |

### Geometry
- One scalar dimension; one value (1.0); applied to the MEM-marker token only (not
  to the 4 addr bytes or 4 val bytes — those are handled by L7 head 7 broadcast).
- Operates **only in the retained-history region** `[0, _mem_history_end)`. The
  current step's MEM marker is **not** touched by `_inject_mem_store` — L6 head 6
  + `_set_opcode_relay_head` produce it from the active opcode (`OP_SI / OP_SC /
  OP_PSH / OP_JSR / OP_ENT`) flowing through the neural network.

### Why the injection exists
- L15 attention heads 0-3 (memory lookup for LI/LC/STACK0) gate K-side on
  `BD.MEM_STORE` with weight 100.0 (see `vm_step.py:6069`,
  `_set_layer15_memory_lookup`). Without `MEM_STORE >= 1.0` at a MEM marker, the
  K side scores `-312.5` in dim 1 and `0` in dim 2, producing net suppression and
  ZFOD (zero-fill-on-demand) attention output.
- L14 attention heads also gate Q on `MEM_STORE` (dim 34, weight 500.0;
  `vm_step.py:5771, 5875`) so they only fire at the *current* step's MEM section
  during store ops.
- The KV cache retains MEM sections from prior steps. The raw token at retained
  positions is `Token.MEM` but the residual at those positions, after embedding +
  augmentation, lacks the `MEM_STORE` flag — because `MEM_STORE` is normally
  injected at the *current* step by the runtime opcode flag flowing through
  `_set_opcode_relay_head` (L6 head 6 V[6]) → `_set_layer7_memory_heads` (head 7
  broadcast). Retained MEM positions were emitted in **prior** steps when the
  opcode was active *then*, so they need the flag re-asserted from outside.

---

## 2. Runner-Side Data Source

### Trigger chain
`c4_release/neural_vm/run_vm.py`:

1. **Per-step shadow-memory update** (lines 696-723 for `PSH`/`JSR`/`ENT`,
   lines 789-799 for `SI`/`SC`): the runner observes the opcode just executed
   and calls:
   - `self._mem_store_word(addr, value)` — updates `self._memory` (dict
     `addr → byte`).
   - `self._inject_mem_section(addr, value)` (line 1288) — synthesises a fresh
     9-token MEM section `[Token.MEM, addr_b0..3, val_b0..3]` and passes it to
     `self._track_mem_access(addr, mem_section)` (line 1315).
2. **History tracking with LRU eviction** (`_track_mem_access`, line 1315):
   - `self._mem_history[addr] = mem_section` (dict, latest-value-wins).
   - `self._mem_access_order` — LRU list; eviction at
     `len(_mem_history) > max_mem_history` (default 64).
3. **Context rebuild** (lines 872-877, repeated at 650-655):
   - Flatten all surviving `_mem_history` values to `mem_flat`.
   - `context[prefix_len:] = mem_flat + list(last_step)`.
   - `self.model.embed.set_mem_history_end(prefix_len + len(mem_flat))` — this
     boundary is what `_inject_mem_store` reads.

### Implication
The runner's `_mem_history` dict **is** the shadow memory state, and the only
reason the embedding needs the boundary is so it can re-flag the
runner-reconstructed MEM section tokens. The MEM tokens themselves are also
runner-synthesised (not produced by the neural network's outputs from prior
steps — they are *fresh tokens* built from `_inject_mem_section` after each
store). The injection is thus the tail end of a long impure pipeline:

```
opcode executes (Python)
  → _mem_store_word / _memory dict updated (Python)
  → _inject_mem_section builds 9-token MEM section (Python)
  → _track_mem_access stores in _mem_history (Python, LRU)
  → next step's context rebuild emits MEM sections (Python)
  → set_mem_history_end (Python)
  → _inject_mem_store flips MEM_STORE=1.0 (Python, in the embedding forward)
  → L15 attention reads MEM_STORE on K-side (neural)
```

Eliminating only `_inject_mem_store` does not buy us purity: the MEM tokens
upstream are themselves a runner artefact.

---

## 3. Relationship to `_inject_mem_exec_autoregressive`

Reading `neural_embedding.py:509-604` reveals a critical fact: when
`_autoregressive_exec=True` (the default — see `__init__.py:55-57`),
`_inject_mem_exec_autoregressive` sets `x[b, i, mem_store] = 2.0` on **every**
MEM marker it walks (line 579), and also on **every** MEM val-byte position
(line 590). It scans the entire token stream from `start_pos`, not just the
`_mem_history_end` boundary.

**Therefore in autoregressive mode `_inject_mem_store` is largely a no-op.**
The flag is already set at strength 2.0 (matching the L6 head 6 + L7 head 7
output) at every MEM position by the autoregressive path. `_inject_mem_store`
would only add 1.0 on top — and only in `[0, _mem_history_end)`, a subset of
the positions the autoregressive method already covers.

`_inject_mem_store` retains an effect only when `_autoregressive_exec=False`
(set externally via `set_autoregressive_exec(False)` and `add_exec_addr()` —
used in tests / hint-driven mode), where `_inject_mem_exec` (the non-auto
variant) is gated on `_exec_addrs` and skips MEM sections that are not in
executable regions. In that mode, `_inject_mem_store` was the safety net.

---

## 4. Proposed Neural Replacement

### Architectural pattern: "Stack-attend"

The clean replacement: have step N's MEM lookup attend back to the most-recent
PSH/SI/SC step that wrote to the same address, and read the value there. This
removes:
- `_inject_mem_store` (this method)
- `_inject_mem_section` (runner-side MEM token synthesis)
- `_track_mem_access` + `_mem_history` + LRU eviction
- The whole context rebuild + `mem_flat` machinery

### Mechanism sketch

The address used as the "key" must already live in the residual at the *prior*
step (because that's the step we attend back to). Stack ops have this naturally
(SP at the time of the PSH is in the step's OUTPUT). Heap ops (SI/SC) place
the address in STACK0/AX at the time of the store. The current step's load
side (LI/LC at AX, *SP at STACK0) already builds an address key by L15 time.

```
Step W (PSH at SP=0x1000):
  position p_W = OUTPUT bytes of SP/STACK0 in step W
  K-encoding: ADDR_KEY = 0x1000, MEM_STORE = 1.0 (set by L6/L7 from OP_PSH)
  V-encoding: AX value at step W

Step R (LI at AX=0x1000), W steps later:
  position p_R = AX marker in step R
  Q-encoding: ADDR_KEY = 0x1000, OP_LI_RELAY = 1.0
  Causal attention pattern matches p_R → p_W (most recent matching ADDR_KEY
  with MEM_STORE=1.0 among all prior positions).
  V at p_W is the pushed value.
```

This is essentially what L15 already does, **but** today it reads from
runner-synthesised MEM tokens that share the current step's window. The
proposed change is to drop the MEM tokens entirely and have L15 attend to
the *originating PSH/SI/SC step's OUTPUT positions* directly.

### Required machinery

1. **Persistent ADDR_KEY at PSH/SI/SC OUTPUT positions** — the SP/STACK0/AX
   bytes in step W's OUTPUT need to be enriched with an ADDR_KEY encoding of
   the store address. Today ADDR_KEY exists only for code byte positions
   (`_add_code_addr_keys`). We would need an analogous L_n FFN that, when a
   store opcode is active, copies the address bytes into a 12-bit ADDR_KEY
   (3 nibbles × 16 one-hot) on the value-byte positions.
2. **Most-recent-match selection** — vanilla softmax with ALiBi will prefer
   positions close to Q (small distance). With ALiBi slope ≈ 5 and step length
   35, going back N steps costs `5 × 35 × N = 175N` in score. To make the
   *most recent matching* store win, the address match bonus must dominate
   the ALiBi distance penalty across the entire KV-cache window (say 4096
   tokens / 117 steps), i.e. address-match score ≈ 20000+. Current L15 uses
   ~+300 for a 24-bit match (`vm_step.py:6003`), well below.
3. **Address aliasing risk** — `ADDR_B0_LO` overlaps `OPCODE_BYTE` in the
   residual (see `_set_layer15_memory_lookup` comment line 5995, "up to +1200
   spurious score from residual opcode nibbles"). Extending the address-match
   K signal across long history multiplies this risk; the worst-case score
   at non-target positions rises with the window. Pure neural mem-attention
   would need a cleaner dedicated address dim that does not alias with
   opcode encoding.
4. **Carrying state across KV-cache eviction** — even if the attention pattern
   works, KV cache eviction will eventually drop the originating PSH step.
   With runner shadow memory, the runner re-emits a MEM section on every step
   so even after eviction the value survives. Without shadow memory, eviction
   = data loss. A genuine pure-neural mode would need either:
   - **No KV-cache eviction** (limits programs to ~117 steps),
   - **Learned eviction** (a separate model picks which steps to drop),
   - **State compression** (PSH steps written to a fixed-size scratch region
     via attention).

   Compiler-style hand-set weights cannot easily do (b) or (c).

### Architecture sketch (option A: no eviction)

For programs short enough to fit entirely in the context window:

| Layer | Role | Required change |
|---|---|---|
| Embedding | Inject ADDR_KEY on the *push address* at PSH/SI/SC output value-byte positions | new FFN, ~50 lines |
| L6/L7 (already exists) | Set MEM_STORE on the same positions | reuse current logic; minor retargeting |
| New L_n attention | Stack-attend: Q at load position → K at most-recent matching store position | full new layer or 4 heads, ~200 lines |
| L15 | Continue to consume the attention output | minor retargeting of K/V sources |

### Architecture sketch (option B: keep eviction, use compression)

Replace `_mem_history` LRU with a learned **memory bank** in the residual: a
fixed-size set of "memory slots" implemented as a separate attention layer
that writes on store and reads on load. This is essentially a transformer-
memory layer (à la TransformerXL, RMT, or scratchpad attention). **Not
hand-settable.** Requires training.

---

## 5. Risk Assessment

| Risk | Severity | Notes |
|---|---|---|
| **Behavioural regression on existing tests** | HIGH | The 5-injection set is currently load-bearing for >100 tests. Removing `_inject_mem_store` without confirming `_inject_mem_exec_autoregressive` coverage is identical will break LI/LC/LEV on long-running programs. |
| **ADDR_KEY ↔ OPCODE_BYTE aliasing** | HIGH | Worst-case score collision +1200 today; extending across 4096 tokens makes the suppression budget marginal. May require new BD layout. |
| **KV-cache eviction blocks pure mode** | CRITICAL | Without runner shadow memory, eviction loses store data. Any program longer than ~117 steps will fail until either eviction is disabled or a learned memory mechanism is added. |
| **Training required** | HIGH | Option B (compression) cannot be hand-baked. Option A (no eviction) is bakable but constrains program length. |
| **Cross-test interaction with `trust_neural_alu` and `pure_neural` modes** | MEDIUM | Both modes are recent additions (Phase 6/7). Adding a third orthogonal axis ("trust_neural_memory") for incremental rollout is feasible but adds combinatorial test surface. |
| **Reduction in debuggability** | MEDIUM | Today, `_mem_history` is human-readable. A pure attention path makes "what did the model think was at addr 0x1000?" harder to introspect. |

---

## 6. Estimated Complexity

### Cleanup-only path (remove `_inject_mem_store` after proving redundancy)
- Verify `_inject_mem_exec_autoregressive` covers all positions touched by
  `_inject_mem_store` for every supported KV-cache + eviction configuration.
- Add a regression test in non-autoregressive (`set_autoregressive_exec(False)`)
  mode to ensure history MEM_STORE is still applied — or migrate hint-based
  callers to autoregressive mode.
- Delete `_inject_mem_store`, `set_mem_history_end`, `_mem_history_end`.

**Effort:** ~1 engineer-day. Does **not** make the system pure_neural — it just
removes a redundant injection. The runner shadow memory pipeline
(`_mem_history`, `_inject_mem_section`, `_track_mem_access`,
`set_mem_history_end`) survives and continues to feed
`_inject_mem_exec_autoregressive` via the rebuilt MEM tokens.

### True pure_neural path
| Component | Effort |
|---|---|
| ADDR_KEY-on-output FFN | 2-3 days |
| Stack-attend layer (4 heads, new K/V plumbing) | 5-7 days |
| BD layout rework to separate ADDR_KEY from OPCODE_BYTE | 3-5 days |
| Eviction-free or learned-eviction memory bank | 2-4 **weeks** |
| Re-baking L6/L7/L14/L15 weights to drop MEM-token assumptions | 1-2 weeks |
| Test recovery (LI/LC/LEV/PSH/JSR/ENT regressions) | 1-2 weeks |
| Training (if option B chosen) | 4-8 weeks of compute + tuning |

**Total: 6-14 engineer-weeks** for a hand-baked option A; **multi-month** for
option B. Comparable in scope to the original L14+L15 MEM design effort.

---

## 7. Recommendation

**Defer the true migration. Pursue the cleanup-only path now.**

Rationale:
1. `_inject_mem_store` is functionally subsumed by
   `_inject_mem_exec_autoregressive` in the default mode. The remaining cost
   it imposes on purity is symbolic, not behavioural.
2. The *real* impurity is the runner's shadow memory pipeline
   (`_mem_history`, `_inject_mem_section`, `_track_mem_access`, context
   rebuild). `_inject_mem_store` is the last 6 lines of a 200-line pipeline.
   Migrating only the leaf without the upstream gives no purity win.
3. A genuine pure-neural memory requires either reduced program length
   (option A) or a learned attention layer (option B). Neither is
   appropriate at this stage of the project (Phase 7), which is still
   stabilising opcode-level neural behaviour.
4. The other 4 injection methods (ADDR_KEY, THINKING, INITIAL_PC, MEM_EXEC)
   are all position-deterministic and migrate trivially with a one-shot FFN
   bake. `_inject_mem_store` is fundamentally different in class — it is the
   tip of a runner-state propagation pipeline, not a positional embedding.

### Concrete next actions (if any)
1. **(Now)** Audit all callers of `set_autoregressive_exec(False)` and confirm
   `_inject_mem_store` is the only thing keeping LI/LC working there. If yes,
   migrate those callers to autoregressive mode.
2. **(Then)** Remove `_inject_mem_store`, `set_mem_history_end`, and the
   `_mem_history_end` attribute. Verify all tests pass.
3. **(Future, separate ticket)** If pure_neural becomes a stronger requirement,
   open a dedicated multi-week effort to migrate the entire
   `_mem_history → MEM-token → injection` pipeline to a stack-attend
   architecture as sketched in §4.

### What this plan explicitly does **not** do
- Modify any code.
- Promise that `pure_neural=True` mode currently works for memory ops
  (it does not — see line 632 in `run_vm.py`, the `_MEM_STORE_OPS` shim that
  *manually persists MEM sections* even in pure_neural mode).
- Provide weight diffs.

---

## Appendix A — File Map

| Path | Lines | Role |
|---|---|---|
| `c4_release/neural_vm/neural_embedding.py` | 312-337 | `_inject_mem_store` body |
| `c4_release/neural_vm/neural_embedding.py` | 339-347 | `set_mem_history_end` setter |
| `c4_release/neural_vm/neural_embedding.py` | 49 | `_mem_history_end = 0` init |
| `c4_release/neural_vm/neural_embedding.py` | 576-590 | `_inject_mem_exec_autoregressive` MEM_STORE redundant writes |
| `c4_release/neural_vm/run_vm.py` | 65 | `_MEM_STORE_OPS = {SI, SC, PSH, ENT, JSR}` |
| `c4_release/neural_vm/run_vm.py` | 252-253 | `_mem_history` + `_mem_access_order` init |
| `c4_release/neural_vm/run_vm.py` | 294-296 | per-run reset |
| `c4_release/neural_vm/run_vm.py` | 632-655 | pure_neural MEM-section persistence shim |
| `c4_release/neural_vm/run_vm.py` | 696-799 | PSH/JSR/ENT/SI/SC handlers calling `_inject_mem_section` |
| `c4_release/neural_vm/run_vm.py` | 862-877 | post-step rebuild + `set_mem_history_end` |
| `c4_release/neural_vm/run_vm.py` | 1288-1334 | `_inject_mem_section`, `_track_mem_access`, LRU eviction |
| `c4_release/neural_vm/vm_step.py` | 1772 | `MEM_STORE = 459` BD index |
| `c4_release/neural_vm/vm_step.py` | 4387-4424 | L7 head 7 MEM-marker → byte broadcast |
| `c4_release/neural_vm/vm_step.py` | 4332-4379, 6906-6911 | L6 head 6 OP_*→MEM_STORE relay |
| `c4_release/neural_vm/vm_step.py` | 5642-5900 | L14 attention (uses `MEM_STORE` gate dim 34) |
| `c4_release/neural_vm/vm_step.py` | 5976-6100 | L15 memory lookup (uses `MEM_STORE` on K-side dim 1+2) |

## Appendix B — Comparison Across Injection Methods

| Method | Type | State source | Migration difficulty |
|---|---|---|---|
| `_add_code_addr_keys` | Positional | Token IDs only | Trivial — already position-deterministic |
| `_inject_thinking_markers` | Positional | Token IDs only | Trivial |
| `_inject_initial_pc` | Bootstrap | Constant (`PC_OFFSET=2`) | Trivial |
| `_inject_mem_exec_autoregressive` | Positional | Token IDs only | Trivial |
| **`_inject_mem_store`** | **Runtime state** | **Runner shadow memory** | **Multi-week, training-adjacent** |

The fifth injection is qualitatively different in kind, not degree.
