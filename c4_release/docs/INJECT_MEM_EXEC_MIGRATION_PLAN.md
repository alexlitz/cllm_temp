# INJECT_MEM_EXEC Migration Plan

Scoping document for migrating the `NeuralVMEmbedding._inject_mem_exec`
and `_inject_mem_exec_autoregressive` injection methods out of the
embedding hot path.

Status: scoping only — no code changes.

Source files referenced:

- `c4_release/neural_vm/neural_embedding.py` (the two methods, plus
  `_extract_jump_targets_autoregressive` and the call site at L202-205).
- `c4_release/neural_vm/vm_step.py` (dim layout, `_set_layer5_fetch`,
  `_set_layer15_memory_lookup`).
- `c4_release/neural_vm/setup_helpers.py` (`_set_layer5_fetch`,
  `_set_layer8_multibyte_fetch`).
- `c4_release/neural_vm/unified_compiler/ops/shared.py` (declared dim
  list + the `IO_FORMAT_POS @468 aliases MEM_EXEC` aliasing comment).
- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py`,
  `l15_ops.py` (consumer ops that read ADDR_KEY).
- `c4_release/tests/test_autoregressive_exec.py` (the only test that
  asserts on `_SetDim.MEM_EXEC`).

---

## 1. What the injections actually do

Both methods walk the input `token_ids` and locate MEM sections. A MEM
section is the 9-token pattern
`[Token.MEM, addr_b0, addr_b1, addr_b2, addr_b3, val_b0, val_b1, val_b2, val_b3]`.

For each MEM section found, they write **two distinct kinds of residual
state**:

### 1a. MEM_EXEC flag (dim 468, 1 dim)

Written only on the `Token.MEM` marker position when the address falls
in an "executable" region.

- `_inject_mem_exec` (external-hints path, `start_pos`..`S`):
  if `(addr & ~3) in self._exec_addrs`, sets `x[b, i, MEM_EXEC] = 1.0`.
- `_inject_mem_exec_autoregressive` (default path):
  if `addr < 0x10000` (code region) or `(addr & ~3)` appears as a
  jump target in any in-context `REG_PC` section, sets
  `x[b, i, MEM_EXEC] = 1.0`.

Jump targets are computed by `_extract_jump_targets_autoregressive`,
which scans `REG_PC` sections, decodes the 4-byte little-endian PC
value, and flags any non-sequential PC change (i.e., `pc !=
prev_pc + INSTR_WIDTH`) as a jump target.

### 1b. Memory-key metadata (ADDR_KEY / MEM_STORE / MEM_VAL_Bn)

Only written by the autoregressive method (the external-hints method
writes ADDR_KEY *only when the address is executable*, which is the
old behavior the FIX 2026-04-16 comment supersedes).

Per MEM section, the autoregressive method **unconditionally**:

| Position           | Dim writes                                                    | Value |
| ------------------ | ------------------------------------------------------------- | ----- |
| MEM marker (i)     | `MEM_STORE` (459)                                             | 2.0   |
| Val byte 0 (i+5)   | `MEM_VAL_B0` (461), `MEM_STORE` (459)                         | 1.0 / 2.0 |
| Val byte 1 (i+6)   | `MEM_VAL_B1` (462), `MEM_STORE` (459)                         | 1.0 / 2.0 |
| Val byte 2 (i+7)   | `MEM_VAL_B2` (463), `MEM_STORE` (459)                         | 1.0 / 2.0 |
| Val byte 3 (i+8)   | `MEM_VAL_B3` (464), `MEM_STORE` (459)                         | 1.0 / 2.0 |
| Val bytes (i+5..8) | `ADDR_KEY + {lo, 16+hi, 32+top}` of `addr+byte_off` (206-253) | 1.0   |

So the heart of the autoregressive method is the per-val-byte
ADDR_KEY nibble encoding plus MEM_STORE / MEM_VAL_Bn flags. The
MEM_EXEC flag itself is just a single dim and is gated by
`is_code_region or is_jump_target`.

---

## 2. Who reads MEM_EXEC (dim 468)?

**Nobody.** Across the entire repository:

- `grep -rn "BD\.MEM_EXEC"` — zero hits.
- `grep -rn "_SetDim\.MEM_EXEC"` — only `tests/test_autoregressive_exec.py`.
- `grep -rn "MEM_EXEC"` — five non-test hits:
  - `neural_embedding.py` lines 453, 469, 531, 569 (the writes, in the
    two methods we're migrating).
  - `vm_step.py:1783` (the dim declaration; comment says "for L5
    fetch" but no L5 head reads it).
  - `unified_compiler/ops/shared.py:504, 527, 588, 591` (documentation
    + the auto-declared dim list).

No attention head, FFN op, or bake function in `setup_helpers.py`,
`vm_step.py`, or `unified_compiler/ops/*` reads dim 468 as MEM_EXEC.
Furthermore, dim 468 is **aliased** as `IO_FORMAT_POS` for
conversational I/O (see `vm_step.py:1710` and L8 ops at
`unified_compiler/ops/l8_ops.py:38-64`). The IO_FORMAT_POS write path
(an L8 FFN that increments the format-string pointer) is the *only*
consumer of dim 468 anywhere in the model.

In other words: **the comment "MEM_EXEC = 468 ... for L5 fetch" in
`vm_step.py:1783` is stale**. The L5 fetch (`_set_layer5_fetch`,
`setup_helpers.py:256-385`) keys exclusively on `ADDR_KEY` nibbles
(plus standard marker gates). It does not consume MEM_EXEC.

### Real consumers of the *other* dims written by these methods

These all read the metadata that the autoregressive injector also
writes alongside MEM_EXEC. They are not MEM_EXEC consumers, but a
migration plan must preserve their inputs:

- **ADDR_KEY (206-253)** — read by:
  - `_set_layer5_fetch` heads 0/1/2 (L5 opcode + immediate fetch),
    via K-side projection at `setup_helpers.py:282-285, 323-326,
    367-371`.
  - `_set_layer8_multibyte_fetch` (L8 multi-byte LI/LC), via the
    `reads={"ADDR_KEY", ...}` contract at
    `unified_compiler/ops/l8_ops.py:90, 121`.
  - `_set_layer15_memory_lookup` (L15 stack/memory read), implicitly
    via the 24-bit binary address match at `vm_step.py:5976+`.
- **MEM_STORE (459)** — read by L15 head 0 (target/store
  discrimination, score budget at `vm_step.py:5988-6002`).
- **MEM_VAL_B0..B3 (461-464)** — read by L15 byte-select dim 3
  (`vm_step.py:5992`).

These are required for memory reads to work. None of them is the
MEM_EXEC dim itself.

---

## 3. Can the bytecode be read directly from the token embedding?

Two distinct sources of "bytecode visible to the network" exist; they
must be migrated independently.

### 3a. Prefix code bytes (CODE_START..CODE_END)

These are static. ADDR_KEY for them is written by `_add_code_addr_keys`
(`neural_embedding.py:248-310`) — a *separate* method from the two
under analysis. It is already cached via `_populate_prefix_cache`
(`neural_embedding.py:224-246`). The prefix path is not part of this
migration.

### 3b. Runtime MEM sections (the focus here)

A MEM section is emitted at inference time **after** a store opcode
(SI/SC/PSH) executes. The token embedding for the 4 val-byte positions
already contains the byte value in `CLEAN_EMBED_LO/HI` (read by L5/L8
fetch heads via the V-side projection — see
`setup_helpers.py:305-311, 343-351, 379-385`).

What is **not** in the un-augmented token embedding is the *address*
the bytes belong to. Address nibbles must be derived from the
preceding 4 addr-byte tokens (`i+1..i+4`) of the same MEM section.
Today this derivation happens in Python at embed-time
(`_inject_mem_exec_autoregressive` writes `ADDR_KEY + {lo, hi, top}`
into the val-byte positions). The val-byte tokens themselves have
*no* knowledge of their address from the token embedding alone.

So the question becomes:

> Can the model itself synthesize ADDR_KEY at MEM val-byte positions
> by attending from `(i+5..i+8)` back to the addr bytes at
> `(i+1..i+4)`?

The answer is **yes in principle, but it requires a non-trivial
"address-gather" attention head/FFN to translate (4 little-endian
addr-byte values + byte offset) → 3-nibble one-hot ADDR_KEY**. That
is roughly the inverse of `_set_layer13_mem_addr_gather`
(`setup_helpers.py:1191+`), which already gathers MEM-section addr
nibbles into ADDR_B*_HI dims at register markers.

This is the **real cost** of the migration. The MEM_EXEC flag is
free to drop (it has no consumer). The hard part is moving the
**ADDR_KEY + MEM_STORE + MEM_VAL_Bn** writes to a baked layer.

### 3c. Could we drop the autoregressive heuristic entirely?

The `is_code_region or is_jump_target` decision in
`_inject_mem_exec_autoregressive` is *only* used to gate the
MEM_EXEC=1.0 write. Since nothing reads MEM_EXEC, this entire branch
is dead. The `_extract_jump_targets_autoregressive` helper (the
non-sequential-PC scan) exists only to feed this dead branch and can
be deleted along with MEM_EXEC. The remaining "always write ADDR_KEY
+ MEM_STORE + MEM_VAL_Bn" body is what actually matters.

---

## 4. Per-step state injection beyond static bytecode?

Yes — by design, **runtime-emitted MEM sections** are exactly that:
state that did not exist when `run()` started but accumulates as the
program executes stores.

- The CODE prefix is static and cached
  (`_prefix_cache_*`). It survives across forward passes within one
  `run()`.
- MEM sections are emitted into the autoregressive token stream by the
  runner (`run_vm.py`) whenever the model emits an `IO_STATE_*`/SC/SI
  token sequence. Each new MEM section adds 9 tokens past the cached
  prefix.
- On every forward pass after the prefix grows, the injection
  re-scans the entire post-prefix region (`start_pos = cache_len`)
  and re-writes ADDR_KEY/MEM_STORE/MEM_VAL_Bn on every MEM section it
  finds. (See the `start_pos` plumbing in `forward()` at
  `neural_embedding.py:174-205`.)
- There is no "dynamically modified .text" pattern in c4: the CODE
  region is read-only; SI/SC writes go to data addresses (>= 0x10000).
  Even though the autoregressive heuristic *could* mark a low-address
  MEM as executable, no production code path exercises that. The
  external-hints API (`add_exec_addr`, `set_exec_addrs`) is unused in
  production — only `tests/test_autoregressive_exec.py` calls it.

So the only "per-step" state is the growing list of MEM sections from
program stores. A migration must keep ADDR_KEY/MEM_STORE/MEM_VAL_Bn
re-derived at every new MEM section, either at embed time or via a
baked transformer layer.

---

## 5. Proposed migration approach

**Two phases, increasing in scope. Phase A is fully safe and unlocks
the eventual Phase B.**

### Phase A — Drop MEM_EXEC entirely (single-op, low-risk)

The MEM_EXEC writes have zero consumers. The full Phase A diff:

1. Delete `_inject_mem_exec` (the external-hints method).
2. Delete `_extract_jump_targets_autoregressive` (only used by the
   autoregressive method's now-dead `is_jump_target` branch).
3. In `_inject_mem_exec_autoregressive`, remove the
   `is_code_region`/`is_jump_target` logic and the `x[b, i, mem_exec]
   = 1.0` write. Rename the method to `_inject_mem_metadata` (or
   merge into `_inject_mem_store`).
4. Delete the public API: `set_exec_addrs`, `add_exec_addr`,
   `clear_exec_addrs`, `set_autoregressive_exec`,
   `self._exec_addrs`, `self._autoregressive_exec`.
5. Remove `MEM_EXEC` from `_SetDim` and the declared-dim list in
   `unified_compiler/ops/shared.py:527, 588`. The dim slot at 468 is
   already aliased as `IO_FORMAT_POS`; once `MEM_EXEC` is gone the
   alias note in `shared.py:591` can be cleaned up.
6. Delete `tests/test_autoregressive_exec.py` (the only test that
   asserts MEM_EXEC behavior — and every assertion in it is about
   `_SetDim.MEM_EXEC` being set to 1.0, which is no longer a property
   of the system). Replace with a single test that asserts ADDR_KEY /
   MEM_STORE / MEM_VAL_Bn are correctly written on a synthetic MEM
   section.

Behavioral risk: zero, because no model layer reads MEM_EXEC. The
production path uses autoregressive mode (default), which always
writes the metadata; existing tests that depend on memory reads
(L8/L15) only require ADDR_KEY/MEM_STORE/MEM_VAL_Bn, which are
preserved.

**Estimated cost: single agent, half-day.** Most of the work is
removing now-unused API surface and updating the one test file.

### Phase B — Move address-key gather into a baked attention layer

After Phase A, the only embed-time injection on MEM sections is
ADDR_KEY + MEM_STORE + MEM_VAL_Bn. This is genuinely position +
content dependent (the val-byte position needs to *see* the addr
bytes 1..4 tokens back and combine them with its own byte offset),
so it is the right candidate for a baked attention/FFN.

Outline:

1. Add a new attention head (likely sharing layer L4 or a fresh slot,
   ahead of L5 fetch) that fires at MEM val-byte positions. The head
   gathers the 4 addr bytes from positions `i-{0..3}` relative to the
   MEM marker (or equivalently `j-{1..4}` for val-byte at offset j
   from MEM) and writes their nibble decomposition + offset into
   ADDR_KEY.
2. Add the MEM_STORE / MEM_VAL_Bn flag-relays as a FFN over IS_BYTE
   gated by a "this is a MEM val-byte" marker. The marker itself can
   be derived from a threshold head over MARK_MEM at distance ≤ 8.
3. Update consumer ops in `setup_helpers.py` (L5 fetch, L8 multibyte,
   L13 addr gather, L15 memory lookup) — none of them need to
   change, since they already read the same dims. Only the *source*
   of those dim writes moves from embed-time Python to a baked
   layer.
4. Delete `_inject_mem_metadata` from `neural_embedding.py`. The
   embedding becomes a pure `nn.Embedding` plus only the static
   prefix-cacheable augmentations (CODE ADDR_KEY, initial PC,
   thinking markers, mem_store on historical markers — those are
   independent migrations).
5. The address-derivation head is a one-shot bake (no per-step state)
   and is fully autoregressive: it derives ADDR_KEY at MEM val-byte
   `j` purely from positions `≤ j`.

Behavioral risk: medium-high. Address nibbles need to be exact for
L5/L8/L15 attention matching to score correctly (the score budgets in
`_set_layer5_fetch` and `_set_layer15_memory_lookup` are tight —
single-bit address errors collapse the attention's discrimination by
hundreds of score-units). The bake needs careful validation against
the existing embed-time output on a corpus of MEM sections.

**Estimated cost: multi-day, single agent**, dominated by:

- Designing the address-gather attention pattern (probably 4 heads,
  one per addr byte, with Q-side gating on byte-offset-from-MEM).
- Validating bit-exact equivalence with embed-time injection across
  the L5/L8/L15 attention score budgets.
- Updating `dim_registry.py` contracts and the purity-guard checks.
- Likely a new debug script to compare per-MEM-section ADDR_KEY
  tensors pre/post migration.

---

## 6. Risk assessment

| Concern                                            | Phase A | Phase B |
| -------------------------------------------------- | ------- | ------- |
| Breaks production runs                             | None    | Medium  |
| Breaks `tests/test_autoregressive_exec.py`         | Yes (by design — replace it) | N/A |
| Breaks L5 opcode fetch / L8 multibyte / L15 lookup | None    | High if bake is bit-imprecise |
| Breaks conversational I/O (IO_FORMAT_POS @ 468)    | Cleans up the alias note | Same as Phase A |
| Touches dim_positions / dim_registry contracts     | Small (one dim removed) | Small (only sources change) |
| Backward compat with externally constructed `AutoregressiveVM` | None — no callers of `add_exec_addr` exist | None |
| Performance regression                             | Slight improvement (less Python in embed) | Larger improvement (fewer embed-time loops) |

### Specific risks for Phase B

1. **Address overlap with OPCODE_BYTE.** The score-budget comment at
   `vm_step.py:5995` ("ADDR_B0_LO overlaps OPCODE_BYTE") and the
   `MARK_PC = -25000` fix from 2026-04-16 indicate that wrong
   ADDR_KEY writes have historically caused +1200 score leaks into
   L15. Any bake bug in the address-gather head could surface as
   subtle L15 attention mis-targeting on JSR/LEV.
2. **start_pos / cache interaction.** The current embed-time path
   re-scans positions ≥ start_pos on every forward pass. A baked
   layer reuses KV cache, so addr-derivation will *not* be redone for
   already-attended positions — which is actually what we want. But
   the cache must not have been populated from a state where ADDR_KEY
   was wrong. Phase B must clear KV cache on any run that previously
   used the embed-time injector.
3. **MEM section spanning chunked attention windows.** If a MEM
   section straddles the boundary of an attention window the new
   head must still be able to see the addr bytes from the val-byte
   position. Today the embed-time loop has no such constraint.

### Phase A residual risk

Effectively none, but worth noting that any external script (outside
the repo) that constructed a `NeuralVMEmbedding` and called
`set_autoregressive_exec(False) + add_exec_addr(...)` would break.
No such call exists in `c4_release/` (verified by `grep -rn`).

---

## 7. Estimated complexity summary

- **Phase A** (drop MEM_EXEC dim and external-hints API): single
  agent, ~half-day. Pure cleanup; mechanical changes to one method,
  one dim declaration, one test file. Worth doing immediately because
  it removes a misleading dim/comment, simplifies the embedding, and
  unblocks reasoning about Phase B.

- **Phase B** (move ADDR_KEY + MEM_STORE + MEM_VAL_Bn writes into a
  baked attention/FFN): single agent, ~3-5 days. The address-gather
  bake is novel — no existing op derives ADDR_KEY from
  preceding-token addr bytes — and the L15 score budgets demand
  bit-exact equivalence with the current Python injection. Validation
  is the long pole.

The two phases are independent. Phase A is recommended unconditionally.
Phase B is conditional on whether the broader project goal (i4) needs
the embedding to be a pure `nn.Embedding`; if other injections
(`_inject_mem_store`, `_inject_initial_pc`, `_inject_thinking_markers`,
`_inject_active_opcode`) also need to migrate to baked layers, the
address-gather work in Phase B fits naturally alongside them.
