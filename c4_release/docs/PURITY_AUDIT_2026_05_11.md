# Purity Audit — 2026-05-11

**Author:** purity-audit agent (read-only research, no code modification)
**Branch:** `purity-audit`
**Base:** `main` @ 9d9965b

---

## AMENDMENT 2026-05-11 (post-canonical-spec correction)

The original audit cited `old/BLOG_POST.md` (now deleted) as the architectural
authority. That was the wrong file. The **canonical spec** is now
[`BLOG_SPEC.md`](BLOG_SPEC.md) in this directory. Key directive (line 3 of
the spec):

> "...100% pure transformer with no exotic architecture choices...
> **no encoder-decoder no python loop other than the standard generation
> loop, no auxiliary memory or python variables, no special memory
> management or special masking**, just a standard auto regressive decode
> only transformer and generation loop and things that are actually done
> in real LLMs for optimizations."

And on I/O (line 851):

> "...In neural I/O mode (the default), stdin and stdout are handled
> **purely through transformer attention with no runner intervention**."

This rewrites the **"blog-spec intentional" categorization** below. Per
the canonical spec, the only acceptable Python infrastructure is:
1. The standard autoregressive generation loop.
2. Tool-calling mode (opt-in, NOT the default): `Token.TOOL_CALL` boundary
   to host syscalls (V8/V16) — `c4_release/docs/BLOG_SPEC.md:851` carves
   this out as the opt-in mode.

Everything else, including the items the old audit marked "deferred-
architectural / pragmatic per blog", is a real violation per the canonical:

- **V2** `_inject_mem_metadata` ADDR_KEY decode — NOT pragmatic, must be
  neural (the spec line 830 describes the binary-key + per-byte AND + MoE
  one-hot mask mechanism explicitly).
- **V5** `_memory` dict — NOT pragmatic, must be replaced by neural memory
  via KV cache + ALiBi mem-attention.
- **V6** `_mem_history` — same; replace via KV cache retention.
- **V9** pure_neural shims (`_neural_prtf_emit`, `_neural_open_emit`,
  `_neural_clos_emit`, `_neural_read_emit`, `_inject_getchar`) — NOT
  intentional in neural I/O mode. Spec line 851 says stdout/stdin via THINK
  tag protocol: model emits THINK_END, a visible char byte, then re-enters
  THINK_START. Input bytes injected between USER_INPUT_START/END markers
  and read via position-tracking heads + nibble cascade.
- **V10** pure_neural MEM-persistence shim — context-rewriting Python loop;
  violates "no python loop other than the standard generation loop". Must
  go.
- **V11** stdin buffer — acceptable only inside tool-calling mode; in
  neural I/O mode the bytes must arrive via USER_INPUT_START/END markers
  in the token stream.
- **V17** `_decode_exit_code` — debatable. Result extraction is on the
  generation-loop boundary, similar to reading sampled tokens; probably
  OK but flag for review.

**What stays acceptable:**
- V8/V16: tool-calling mode opt-in (opcode → TOOL_CALL → host syscall →
  AX override). Per spec line 842-849, this is an intentional second mode.
- V20: prefix-embedding cache. It's a memoisation of deterministic
  computation, not Python state.

**Net headline (corrected):** the *pure_neural-blocking* set is now closer
to ~10 items, not 2. The "blog-spec intentional" pile collapses to ~2
(V8/V16 in tool-calling mode only). See the
[`STACK0_VIA_MEM_ATTENTION_PLAN.md`](STACK0_VIA_MEM_ATTENTION_PLAN.md) and
the in-flight ALiBi mem-attention work for the V3/V5/V6/V10 path.

---

This document inventories the **remaining Python state that is folded back
into the neural model** after today's purity migrations land (I1 bake of
initial PC, I2 positional-encoding of ADDR_KEY, I3 thinking marker bake, B2
deletion of MEM_EXEC, and the in-flight alibi-mem-attention migration for
`_inject_mem_store`). It is intended as a complete TODO map for future
purity work, cross-referenced against the blog post architectural intent
(now [`BLOG_SPEC.md`](BLOG_SPEC.md), not the deleted `old/BLOG_POST.md`).

The audit covers four surfaces:

1. **`NeuralVMEmbedding.forward`** — token-time augmentations.
2. **`AutoregressiveVMRunner._dispatch_step`** — per-step Python overrides.
3. **Runner-state** — `_memory` dict, `_mem_history`, `_last_*` shadow regs,
   stdin buffer.
4. **MoE infrastructure** — `set_active_opcode` and `_activate_moe`.

---

## 0. Summary table

| # | Violation | Where | Severity | Blocks pure_neural flip? | Est. fix size |
|---|---|---|---|---|---|
| V1 | `_inject_active_opcode` (5 dims, all positions) | `neural_embedding.py:452-474` | deferred-architectural | **NO** (already gated off in pure_neural) | M (1-2 days; needs L5 all-step PC decode + L6 head 6 re-enable + L5 conv-io re-bake) |
| V2 | `_inject_mem_metadata` (ADDR_KEY/MEM_STORE/MEM_VAL_Bn for current step) | `neural_embedding.py:538-608` | deferred-architectural (per blog: "pragmatic infrastructure") | **NO** (it's positional, not Python state — see §V2) | XL (5-10 days; needs neural addr decode + value broadcast) |
| V3 | `_inject_mem_store` (history MEM markers) | `neural_embedding.py:415-440` | deferred-architectural | NO once alibi-mem-attention lands; YES today | S after alibi (1 line delete); already mostly dead |
| V4 | `set_active_opcode` MoE swap | `vm_step.py:1217-1237`, `base_layers.py:272-294` | cosmetic in pure_neural (skipped); blocking for batch/fast runners | **NO** in pure_neural; YES for batch/fast | M (rework MoE routing to neural gate, ~2-3 days) |
| V5 | Runner `_memory` shadow dict | `run_vm.py:227,313,917,933,1336,1422,1455,...` | deferred-architectural (per blog: explicitly pragmatic) | YES today (PSH/JSR/SI/SC/LI/LC, PRTF/READ/OPEN) | XL (attention-as-memory rework) |
| V6 | Runner `_mem_history` dict + KV-cache eviction | `run_vm.py:252,294,713-723,1043-1046,1484-1503` | infrastructure | NO once alibi-mem-attention lands | M (1-2 days, depends on V3) |
| V7 | `_dispatch_step` REQUIRED overrides (11 blocks) | `run_vm.py:594-1047` | blocking | **YES** for handler-mode; pure_neural already skips them | XL (Phase 2-7 multi-week neural work) |
| V8 | `_dispatch_step` EXTERNAL (`_syscall_handlers`) | `run_vm.py:202-207,740-747,1613-1753` | external I/O | NO (intentional per `PURE_NEURAL_POLICY.md`) | N/A (boundary) |
| V9 | pure_neural shim: `_neural_prtf_emit`, `_neural_open_emit`, `_neural_clos_emit`, `_neural_read_emit`, `_inject_getchar` | `run_vm.py:1137-1345` | external I/O | NO (intentional) | N/A |
| V10 | pure_neural MEM-persistence shim (`_MEM_STORE_OPS` block) | `run_vm.py:700-723` | blocking | YES today (PSH/SI/SC neural store broken) | M (Phase 7, depends on neural L14 store) |
| V11 | Runner stdin buffer (`_stdin_buffer`/`_stdin_pos`) | `run_vm.py:287-288,1146-1151,1322-1327` | external I/O | NO (intentional) | N/A |
| V12 | `set_active_opcode` initial+per-step call sites (handler-mode) | `run_vm.py:337-341,1024-1029` | cosmetic | NO (gated by `not self.pure_neural`) | S (deletes after handler-mode retires) |
| V13 | `_inject_initial_pc` (I1 in flight) | `neural_embedding.py:500-536` | deferred-architectural | NO (purely position-deterministic) | already in flight (I1) |
| V14 | `_inject_thinking_markers` (I3 in flight) | `neural_embedding.py:476-498` | deferred-architectural | NO (token-deterministic, like positional encoding) | already in flight (I3) |
| V15 | `_add_code_addr_keys` (I2 landed) | `neural_embedding.py:337-413` | LANDED — now a precomputed buffer | NO | done |
| V16 | TOOL_CALL host shim path (handler-mode side) | `run_vm.py:468-498,1613-1753` | external I/O | NO (intentional) | N/A |
| V17 | `_decode_exit_code` post-run | `run_vm.py:1780-1788` | allowed per policy | NO (result extraction) | N/A |
| V18 | Conversational-I/O THINKING_END handling | `run_vm.py:384-432` | gated off in pure_neural | NO (handler-mode only) | M (separate convo-io migration) |
| V19 | `_func_call_handlers` dict | `run_vm.py:255,756-758` | dead by default | NO (empty default) | S (delete) |
| V20 | Prefix-embedding cache | `neural_embedding.py:51-70,162-264` | optimisation, not violation | NO | N/A |

**Headline counts (post-I1/I3/B2/alibi-mem):**

- **Pure_neural-blocking violations: 2** (V7 partial — 11 REQUIRED handler-mode overrides that pure_neural skips entirely but cause incorrect output; V10 — MEM persistence shim for store ops).
- **Deferred-architectural (intentional per blog): 3** (V2 mem metadata, V5 `_memory` dict, V6 `_mem_history`).
- **Cosmetic / dead-code-after-handler-retires: 5** (V4, V12, V18, V19, and the REMOVABLE-NOW subset of V7).
- **External / always-Python: 4** (V8, V9, V11, V16, V17).

---

## V1 — `_inject_active_opcode` (5 residual dims, all positions)

**File/lines:** `c4_release/neural_vm/neural_embedding.py:452-474`

**Caller chain:** `AutoregressiveVM.forward` → `NeuralVMEmbedding.forward(..., active_opcode=...)` → `_inject_active_opcode` when `active_opcode is not None`.

**What Python state is injected:** Five residual dims at **all** sequence positions, value 5.0, keyed off `self._active_opcode` (which is set externally by `set_active_opcode`):

| Opcode | Dim name |
|---|---|
| 33 (PRTF) | `ACTIVE_OPCODE_PRTF` |
| 31 (READ) | `ACTIVE_OPCODE_READ` |
| 8 (LEV) | `OP_LEV` |
| 4 (BZ) | `OP_BZ` |
| 5 (BNZ) | `OP_BNZ` |

**Severity:** deferred-architectural. **Already gated off in pure_neural mode.**

`AutoregressiveVMRunner._dispatch_step` (lines 1019-1029) and `run()` init
(lines 337-341) only call `set_active_opcode` when `not self.pure_neural`.
So in pure_neural runs, `self._active_opcode` stays `None` and the embedding
short-circuits at line 237:

```python
if active_opcode is not None:
    self._inject_active_opcode(token_ids, x, active_opcode)
```

**Blocks pure_neural flip?** **No.** The pure_neural smoke (`test_pure_neural_pc.py` 12/13 PASS) confirms the model decodes BZ/BNZ via L5 FFN + L6 head 4 relay without injection. The one gap is OP_LEV at the PC marker for non-first-step LEV invocations (see `SET_ACTIVE_OPCODE_PURITY.md` §3); this is Phase 5 LEV scope, not pure_neural Phase 8 gating.

**Est. fix size:** M (1-2 days). Per `SET_ACTIVE_OPCODE_PURITY.md` recommendation:

1. Add an all-step PC-marker OP_LEV decode to L5 FFN (1 unit).
2. Re-enable L6 head 6 V[0] OP_LEV relay (`unified_compiler/compiler.py:791,833`).
3. Convo-I/O migration (separate scope): replace `ACTIVE_OPCODE_PRTF`/`READ` injection with a baked broadcast head, OR keep injection but mark as handler-mode-only.

---

## V2 — `_inject_mem_metadata` (ADDR_KEY/MEM_STORE/MEM_VAL_Bn for *current step's* MEM section)

**File/lines:** `c4_release/neural_vm/neural_embedding.py:538-608`

**Caller chain:** `AutoregressiveVM.forward` → `NeuralVMEmbedding.forward` → `_inject_mem_metadata(token_ids, x, start_pos=start_pos)` unconditionally.

**What Python computation happens during forward:**

For each `Token.MEM` marker in the token stream:

1. **Reads 8 token bytes from the input** (`token_ids[b, i+1..i+8]`) to extract a 32-bit address.
2. **Writes** `MEM_STORE = 2.0` on the MEM marker and on each of the 4 value-byte positions.
3. **Writes** `MEM_VAL_B0..B3 = 1.0` on val-byte positions.
4. **Writes** `ADDR_KEY + {lo, 16+hi, 32+top} = 1.0` on val-byte positions, computed from `addr + byte_off` (a 3-nibble decomposition of the per-byte address).

This is per-token Python work (B × number-of-MEM-markers × constant). On a typical run, MEM markers appear ≥ once per step (whenever PSH/JSR/ENT/SI/SC executes), so this loop runs at every step.

**Severity:** deferred-architectural. **Intentional per the blog post:**

> "memory is where the computation lives, not what computes" (BLOG_POST.md:283)
> "Could the indexing be neural too? Yes—with attention over memory slots. But that's a different (and larger) project." (BLOG_POST.md:381)

The blog explicitly classifies memory **storage** as "pragmatic infrastructure" (line 381) and **address decoding** as "Not neural" (line 376-379). `_inject_mem_metadata` is exactly this: a deterministic function from the address bytes in the MEM section to the ADDR_KEY one-hot decomposition that L5/L8/L15 attention reads.

**Crucially:** unlike `_inject_active_opcode`, this is **not Python *state* injected into the model** — it is a **deterministic function of the input tokens**. The same token sequence always produces the same residual modifications. This is morally equivalent to a content-addressed positional encoding (read addr bytes from tokens, decode into one-hot ADDR_KEY).

**Blocks pure_neural flip?** **No.** This injection runs in pure_neural today and is the only reason LI/LC and L15 memory lookup work autoregressively. Removing it without a neural replacement would break the smoke `test_pure_neural_pc.py` IMM/EXIT test (which doesn't touch memory, so this is academic for the current Phase 1 gate, but it would break Phase 7).

**Est. fix size:** XL (5-10 days) to make fully neural. The "natural" replacement is:

1. **Address decode head**: L4/L5 attention head that reads bytes 1-4 of every MEM section and produces an ADDR_KEY one-hot decomposition at val-byte positions. This is a content-attention pattern: Q="I'm a val byte 5+k after a MEM marker", K="I'm byte (1+k) after a MEM marker, holding address byte k". The byte-to-nibble decomposition then needs an FFN.
2. **MEM_STORE / MEM_VAL_Bn flags**: simpler — these are token-local flags that fire on `Token.MEM` and the 4 subsequent val bytes. A single L1/L2 FFN unit per flag would suffice.

The address-decode path is the hard part. The current Python implementation is ~30 lines of bit-fiddling; a neural replacement needs roughly 4 heads × 4 bytes × 16 nibble units = ~256 FFN units plus 4-8 attention heads.

**Recommendation:** keep as deferred-architectural. The blog explicitly endorses pragmatic memory infrastructure. Reclassify the audit allowlist to document this as "intentional positional decode" rather than a violation.

---

## V3 — `_inject_mem_store` (history MEM markers in retained KV cache region)

**File/lines:** `c4_release/neural_vm/neural_embedding.py:415-440`

**Caller chain:** `NeuralVMEmbedding.forward` → `_inject_mem_store(token_ids, x, start_pos=start_pos)` — always called, but no-op when `self._mem_history_end == 0`.

**What Python state is injected:** For every position `i < self._mem_history_end` where `token_ids[b, i] == Token.MEM`, sets `x[b, i, MEM_STORE] = 1.0`.

**Why it exists** (from `INJECT_MEM_STORE_MIGRATION_PLAN.md` §1): when the KV cache retains MEM tokens from prior steps, the `MEM_STORE` flag (normally generated by L6 head 6 → L7 head 7 from the *current step's* opcode flag) is missing at those retained positions. L15 memory lookup gates K-side on MEM_STORE with weight 100.0; missing flag → -312.5 penalty → ZFOD attention output.

**Python state used:** `self._mem_history_end`, an integer set by `set_mem_history_end()` from runner code (`run_vm.py:723, 1046, 296`). This integer is computed from `prefix_len + len(mem_flat)`, where `mem_flat` is the flattened `_mem_history` dict — itself Python state.

**Severity:** deferred-architectural. **Target of the in-flight alibi-mem-attention agent.**

**Blocks pure_neural flip?** Today, **yes** — if any test runs long enough to trigger KV-cache eviction (>64 unique mem addresses), retained MEM positions stop scoring in L15. But for Phase 1 smoke (IMM/EXIT), `_mem_history` stays empty and `_mem_history_end == 0`, so this is a no-op.

**Est. fix size:** S after alibi-mem-attention lands. Per `INJECT_MEM_STORE_MIGRATION_PLAN.md`:

- The current step's MEM section already has MEM_STORE=2.0 set by L6 head 6 + L7 head 7 (the "Option B" in the doc).
- ALiBi over the MEM-marker sequence would let L15 attend to *historical* MEM markers without needing the residual flag.
- Once alibi-mem-attention lands and KV cache eviction is verified to work neurally, this method can be deleted (1 line).

---

## V4 — `set_active_opcode` MoE weight swap

**File/lines:** `c4_release/neural_vm/vm_step.py:1217-1237`, `c4_release/neural_vm/base_layers.py:272-294`

**What it does (effect B per `SET_ACTIVE_OPCODE_PURITY.md`):**

```python
def _activate_moe(self, opcode_dim):
    if opcode_dim is None:
        c = self._moe_combined['_full']
    else:
        c = self._moe_combined.get(opcode_dim, self._moe_combined.get('_shared'))
    object.__setattr__(self, 'W_up', c['W_up'])
    ...
```

The runner peeks at the next bytecode instruction and swaps each FFN's
`W_up/W_gate/W_down` pointers to the active opcode's pre-concatenated
shared+expert sub-matrices. This is a Python attribute swap that bypasses
`nn.Module`'s parameter management.

**Severity:** cosmetic in pure_neural mode; **blocking for `fast_runner` / `batch_runner` / `batch_runner_v2` / `transformer_first_runner`** because those call `compact_moe()` and rely on `set_active_opcode` to select experts.

**Python state used:** the active opcode value, set from `bytecode[next_exec] & 0xFF` in `run_vm.py:1027`.

**Blocks pure_neural flip?** **No.** The pure_neural runner never calls `compact_moe()`, so `ffn._moe_combined is None` and `_activate_moe` is a no-op even if `set_active_opcode` were called.

**Est. fix size:** M (2-3 days). The blog post is silent on MoE; the architectural intent appears to be that MoE is a runtime optimisation, not a correctness requirement. Two paths:

1. **Replace MoE swap with neural gate**: have each FFN attend to the bytecode at the current PC, decode the opcode, and gate the expert with `silu(W_up @ x)` masking. This costs more compute per step but makes the runtime fully autoregressive.
2. **Delete MoE entirely** from pure_neural runners and live with the full-matrix FFN cost. `fast_runner`/`batch_runner` stay handler-mode-only.

**Recommendation:** path (2) for now. MoE is an optimisation orthogonal to purity.

---

## V5 — Runner `_memory` shadow dict

**File/lines:** `c4_release/neural_vm/run_vm.py:227,294,313-316,...` (~30 sites)

**What it is:** `self._memory: dict[int, int]` — addr → uint8 byte. Populated by:

- Data section load at run start (`run_vm.py:313`).
- `_track_memory_write` from MEM sections after PSH/JSR/ENT/SI/SC (line 1408).
- `_mem_store_word` from `_dispatch_step` ENT/PSH/JSR/SI handlers (lines 787, 800, 814, 925).
- `_neural_read_emit` for READ syscall (line 1336).
- `_syscall_read` for handler-mode READ (line 1681).
- `_syscall_open` (path string read, not write — uses `_read_string`).

Consumed by:

- `_mem_load_word`, `_read_string` (compiler-emit dispatch reads).
- `_dispatch_step` LC handler (line 917: `val = self._memory.get(addr, 0)`).
- `_dispatch_step` SC handler (line 933).
- `_neural_prtf_emit` for fmt string + varargs (lines 1217, 1239).
- `_neural_open_emit` (line 1258, 1260 — heuristic path-vs-mode detection).
- `_neural_read_emit` (line 1342).

**Severity:** deferred-architectural. **Explicitly pragmatic per the blog post:**

> "For this implementation, I made a pragmatic choice: memory is a Python dictionary." (BLOG_POST.md:267)
> "The address→value mapping itself (the dictionary)" is **not neural** (BLOG_POST.md:376-379)

**Blocks pure_neural flip?** **Today, yes** — but only for ops that need to read memory back across the KV-cache eviction boundary. The `_MEM_STORE_OPS` shim at `run_vm.py:700-723` (V10) persists current-step MEM sections into `_mem_history` so the *next step's* L15 lookup sees them. This works for the "memory written in one step, read in the next" pattern. The dict is also consumed by the runner-side syscall shims (`_neural_prtf_emit` etc.), which is intentional per V8/V9.

**Est. fix size:** XL (multi-week). The blog explicitly punts on this: "I didn't implement these because they're complex and my goal was demonstrating that *computation* can be neural." A neural replacement = Neural Turing Machine-style attention over an unbounded memory buffer. Out of scope for any near-term purity work.

**Recommendation:** keep, document as "intentional per blog spec". The shim at V10 is the only piece that actually blocks pure_neural progress; once L14 store generation lands neurally and L15 LRU eviction is replaced with ALiBi over MEM history (V3 alibi-mem-attention), the runner-side `_memory` dict can be reduced to **only** the syscall I/O staging area (V9 needs PRTF/READ/OPEN paths).

---

## V6 — Runner `_mem_history` dict + KV-cache eviction

**File/lines:** `c4_release/neural_vm/run_vm.py:252,294,713-723,1043-1046,1484-1503`

**What it is:** `self._mem_history: dict[int, list[int]]` — addr → 9-token MEM section. LRU-evicted at `max_mem_history=64`. Used to flatten MEM sections into a "retained memory region" between the prefix and the current step:

```python
last_step = context[-(Token.STEP_TOKENS):]
mem_flat = []
for tokens in self._mem_history.values():
    mem_flat.extend(tokens)
context[prefix_len:] = mem_flat + list(last_step)
self.model.embed.set_mem_history_end(prefix_len + len(mem_flat))
```

This is a runtime-managed sparse memory layer that the neural network reads
back via L15 attention.

**Severity:** infrastructure. Tightly coupled to V3 (`_inject_mem_store`) and V5 (`_memory`).

**Blocks pure_neural flip?** Today, yes — V10's pure_neural MEM-persistence shim (`run_vm.py:700-723`) explicitly maintains `_mem_history` during pure_neural runs. If V3 lands (alibi-mem-attention), the *Python-side flag injection* goes away but the dict itself stays as the storage mechanism.

**Est. fix size:** M (1-2 days, depends on V3 first). After V3 lands, this dict can be replaced by direct context retention (no flattening), letting the KV cache itself hold prior MEM positions and attention pick them up by ALiBi.

---

## V7 — `_dispatch_step` REQUIRED overrides (11 blocks, handler-mode only)

**File/lines:** `c4_release/neural_vm/run_vm.py:594-1047`

**Inventory from annotation agent's classification (verified at lines 622-628):**

| # | Block | Lines | Phase | Why required |
|---|---|---|---|---|
| 1 | PSH SP-=8 + STACK0 store | 780-789 | Phase 2 | Model emits garbage AX after MEM-store sequence |
| 2 | JSR SP-=8 + return-addr push + PC=target | 790-806 | Phase 5 | Nested/AX-preservation paths xfail |
| 3 | ENT BP=SP-=8 + imm offset | 807-823 | Phase 5 | `_set_layer8_alu` doesn't subtract imm from SP yet |
| 4 | LEV restore BP/PC from frame | 824-839 | Phase 5 | `_set_layer9_lev_bp_to_pc_relay` doesn't restore PC |
| 5 | JMP target resolution | 840-849 | Phase 4 | Operand-specific JMP target resolution missing |
| 6 | BZ taken/not-taken | 850-862 | Phase 4 | `_set_layer4_ffn` PC carry-forward bug |
| 7 | BNZ taken/not-taken | 863-873 | Phase 4 | Same blocker as BZ |
| 8 | ADJ SP+=imm | 874-884 | Phase 2 | Likely removable but unverified |
| 9 | _BINARY_POP_OPS SP+=8 + ALU + STACK0 pop | 885-905 | Phase 2/3 | `_set_layer9/10_alu` doesn't read prev STACK0 + AX |
| 10 | LI/LC heap read | 906-919 | Phase 7 | `_set_layer15_memory_lookup` word/char-wide LI/LC broken |
| 11 | SI/SC heap write | 920-934 | Phase 7 | `_set_layer14_mem_generation` SI/SC broken |
| 12 | LEA BP+imm | 978-986 | Phase 5 | ENT doesn't establish BP correctly in pure_neural |
| 13 | AX merge (multi-byte ALU high-byte) | 987-993 | Phase 3 | Multi-byte AX merge for ADD/SUB/AND/OR/XOR/SHL/SHR |

**Severity:** Each individually blocks a Phase 2-7 test. **In aggregate they are the entire pure_neural Phase 1-8 progression.**

**Blocks pure_neural flip?** **The pure_neural path (lines 638-726) skips all 13 blocks entirely.** So a "blanket flip" of pure_neural would succeed on Phase 1 (IMM/EXIT) but xfail every other test until each block's neural replacement lands.

**Est. fix size:** XL (multi-week per phase). See `PURE_NEURAL_GAP_ANALYSIS.md`.

**REMOVABLE-NOW subset (per annotation agent, 7 blocks):**

| # | Block | Lines | Why removable |
|---|---|---|---|
| R1 | PC auto-increment (non-control ops) | 760-771 | Phase 1 13/13 PASS confirms pure_neural emits PC+=INSTR_WIDTH |
| R2 | BP preservation (non-ENT/LEV) | 773-777 | Phase 1 confirms BP preserved neurally |
| R3 | STACK0 mirror (non-stack ops) | 944-950 | Phase 1 IMM/NOP confirms STACK0 preserved |
| R4 | AX preservation (non-AX-modifying ops) | 994-1002 | Phase 1 NOP/EXIT cases |
| R5 | SP passthrough (non-stack-mod ops) | 935-942 | Phase 1 IMM/LEA/EXIT/NOP |
| R6 | IMM AX override | 961-974 | Phase 1 13/13 emits IMM into AX byte 0 |
| R7 | `_func_call_handlers` dispatch | 753-758 | Empty by default (V19) |

These 7 blocks fire only in handler-mode and can be deleted as soon as handler-mode is retired entirely. They are pure overhead (the model already produces the correct output and the override re-asserts it).

---

## V8 — `_dispatch_step` EXTERNAL (`_syscall_handlers` dispatch)

**File/lines:** `c4_release/neural_vm/run_vm.py:202-207,740-747,1613-1753`

**What it is:** Dispatch table for `Opcode.{CLOS, OPEN, READ, PRTF}` to `_syscall_*` Python methods that perform real `os.read/write/open/close` calls, formatted printf, and AX-override with the result.

**Severity:** external I/O. **Explicitly allowed per `PURE_NEURAL_POLICY.md`** §1.2:

> "Detecting `Token.TOOL_CALL` to surface a tool request to the host."

This is the VM/host boundary. The blog implicitly supports this (the C4 OS-level syscalls are real OS calls in the reference implementation).

**Blocks pure_neural flip?** **No.** Pure_neural mode has parallel `_neural_*_emit` shims (V9) that perform the same I/O via the same boundary.

---

## V9 — pure_neural shim methods (`_neural_prtf_emit`, `_neural_open_emit`, `_neural_clos_emit`, `_neural_read_emit`, `_inject_getchar`)

**File/lines:** `c4_release/neural_vm/run_vm.py:1137-1345`

**What they do (each):**

- `_inject_getchar`: Read next byte from `_stdin_buffer`, override REG_AX with it.
- `_neural_prtf_emit`: Walk format string from `_memory`, substitute %d/%s/%c/%x/%% from stack args (read via `_mem_load_word`), `output.append(formatted)`, override AX with byte count.
- `_neural_open_emit`: Heuristic path-vs-mode detection from stack slots, `os.open()`, override AX.
- `_neural_clos_emit`: Pop fd from `_memory[SP]`, `os.close()`, override AX=0.
- `_neural_read_emit`: Heuristic (buf_ptr, fd, count) detection from stack, `os.read()` or stdin chunk, write to `_memory`, also `_inject_mem_section` for each affected word, override AX with byte count.

**Severity:** external I/O. Per policy, the I/O boundary is allowed. **However**, the methods read from `self._memory` (V5) and the stack (`_mem_load_word`) — which means the format string and varargs *do* come from the model's emitted state, just via the runner-side shadow dict.

**Blocks pure_neural flip?** **No** for the I/O boundary itself. The reliance on `_memory` (V5) is the deeper architectural issue (per blog: pragmatic).

**Note:** `_neural_read_emit` also calls `_inject_mem_section` (line 1343) — this is the runner injecting fresh MEM sections into `_mem_history` so the model can read back the bytes that READ stored. That is a Python-state-back-into-model loop that breaks purity if you take a strict view.

**Recommendation:** keep as the intentional I/O boundary. Document the `_memory`-dependent paths as "memory infrastructure boundary" per blog spec.

---

## V10 — pure_neural MEM-persistence shim (`_MEM_STORE_OPS` block)

**File/lines:** `c4_release/neural_vm/run_vm.py:700-723`

**What it does (in pure_neural mode):** After each step where `exec_op in {SI, SC, PSH, ENT, JSR}`, extracts the MEM section the model emitted, persists it into `self._memory + self._mem_history`, and rebuilds context so the next step's L15 attention can find it.

```python
if exec_op in _MEM_STORE_OPS:
    mem_section = self._extract_mem_section(context)
    if mem_section is not None:
        addr = ...
        value = ...
        for j in range(4):
            self._memory[(addr + j) & 0xFFFFFFFF] = (value >> (j * 8)) & 0xFF
        self._track_mem_access(addr, mem_section)
        last_step = context[-(Token.STEP_TOKENS):]
        mem_flat = []
        for tokens in self._mem_history.values():
            mem_flat.extend(tokens)
        context[prefix_len:] = mem_flat + list(last_step)
        self.model.embed.set_mem_history_end(prefix_len + len(mem_flat))
```

**Severity:** blocking. This is the pure_neural side of V5/V6 and is the reason `test_pure_neural_psh_add` and Phase 7 LI/LC tests almost work.

**Python state injected:** the rebuild rewrites `context[prefix_len:]` (the active context window) with a flattened history, and `set_mem_history_end` updates the embedding's `_mem_history_end`. So the model's next forward sees a rewritten context.

**Blocks pure_neural flip?** **Yes** for any test that writes then reads memory across more than ~1 step.

**Est. fix size:** M-L. Depends on V3 (alibi-mem-attention) for the read side; the write side needs neural emission of MEM sections (already partially works for PSH/JSR per `_dispatch_step:780+`). Once L14 store generation is verified neural for SI/SC/ENT/JSR/PSH and L15 lookup attends to historical MEM markers via ALiBi, this block can be deleted.

---

## V11 — Runner stdin buffer

**File/lines:** `c4_release/neural_vm/run_vm.py:287-288,1146-1151,1322-1327`

**What it is:** `self._stdin_buffer = list(stdin)`, `self._stdin_pos = 0`. Consumed by `_inject_getchar` (GETCHAR override) and `_neural_read_emit` (READ stdin path).

**Severity:** external I/O. **Per blog/policy: this is the user input boundary and MUST remain Python.** The model cannot read from external state without the runner mediating.

**Blocks pure_neural flip?** **No** (intentional).

---

## V12 — `set_active_opcode` initial + per-step call sites (handler-mode)

**File/lines:** `c4_release/neural_vm/run_vm.py:337-341,1024-1029`

**What it is:** Two call sites that peek at bytecode and call `model.set_active_opcode(...)`. Both gated by `if not self.pure_neural`.

**Severity:** cosmetic. Both already disabled in pure_neural.

**Blocks pure_neural flip?** **No.**

**Est. fix size:** S (immediate deletion once handler-mode retires).

---

## V13 — `_inject_initial_pc` (target of I1)

**File/lines:** `c4_release/neural_vm/neural_embedding.py:500-536`

**What it does:** For the first step's REG_PC marker (the one with no preceding STEP_END), writes `EMBED_LO[lo] = 1.0` and `EMBED_HI[hi] = 1.0` for `initial_pc = PC_OFFSET`.

**Severity:** deferred-architectural. Already targeted by I1 bake.

**Python state used:** none — fully token-deterministic. The "Python computation" is just scanning for the first REG_PC without a preceding STEP_END, which is a deterministic function of `token_ids`.

**Blocks pure_neural flip?** **No.** Pure-positional encoding.

---

## V14 — `_inject_thinking_markers` (target of I3)

**File/lines:** `c4_release/neural_vm/neural_embedding.py:476-498`

**What it does:** For every `Token.THINKING_START` and `Token.THINKING_END` token in the input, sets `MARK_THINKING_START = 1.0` / `MARK_THINKING_END = 1.0` at that position.

**Severity:** deferred-architectural. Targeted by I3 bake.

**Python state used:** none — fully token-deterministic.

**Blocks pure_neural flip?** **No.** Token-deterministic.

---

## V15 — `_add_code_addr_keys` (LANDED — I2)

**File/lines:** `c4_release/neural_vm/neural_embedding.py:337-413`

**Status:** Already migrated to a precomputed `[max_seq_len, 48]` positional-encoding buffer (`_ensure_addr_key_pos_encoding`, lines 266-335). The per-position Python loop now only runs once at construction time to fill the buffer; forward uses a single broadcast-add into the residual stream.

**Note:** the position-only loop still runs in Python at table-fill time (line 311). This is one-shot per `max_seq_len` and is morally equivalent to a learnable positional encoding initialised at construction. **Not a violation.**

---

## V16 — TOOL_CALL host shim path (handler-mode side)

**File/lines:** `c4_release/neural_vm/run_vm.py:468-498`

**What it does:** When the model emits `Token.TOOL_CALL` and `not self.pure_neural`, looks up the opcode, dispatches to `_syscall_handlers`, and tracks memory writes.

**Severity:** external I/O / gated off in pure_neural.

**Blocks pure_neural flip?** **No** (intentional).

---

## V17 — `_decode_exit_code` post-run

**File/lines:** `c4_release/neural_vm/run_vm.py:1780-1788`

**Severity:** **Explicitly allowed per `PURE_NEURAL_POLICY.md`** §1.1 ("Result extraction"). Reads REG_AX bytes from final context. No Python state injected back into model.

---

## V18 — Conversational-I/O THINKING_END handling

**File/lines:** `c4_release/neural_vm/run_vm.py:384-432`

**What it does:** When `conversational_io=True` (handler-mode only) and model emits `Token.THINKING_END`, walks shadow `_memory` for the format string, appends formatted output, injects `Token.THINKING_START`, and synthesises a new step.

**Severity:** gated off in pure_neural (since pure_neural never has `conversational_io=True` per `SET_ACTIVE_OPCODE_PURITY.md` §4).

**Blocks pure_neural flip?** **No.** Separate migration scope.

---

## V19 — `_func_call_handlers` dict

**File/lines:** `c4_release/neural_vm/run_vm.py:255,756-758`

**What it is:** `self._func_call_handlers = {}` — empty by default, populated only by debug fixtures. Dispatched at line 757 unconditionally if any entry exists.

**Severity:** dead code in the default path.

**Blocks pure_neural flip?** **No.**

**Est. fix size:** S (delete the dict + dispatch).

---

## V20 — Prefix-embedding cache

**File/lines:** `c4_release/neural_vm/neural_embedding.py:51-70,162-264`

**What it does:** Caches the augmentation delta for the immutable CODE/DATA prefix region across calls within a single `run()`. On cache hit, applies the delta as a single in-place add and skips per-position Python loops for cached positions.

**Severity:** optimisation, not a violation. The cached values are exactly what `_inject_*` would produce on every call; this is just memoisation.

**Note:** the cache invalidates on different bytecode (`reset_prefix_cache()` is called at the top of every `run()`). It does **not** survive `_inject_active_opcode` writes (line 226-230 comment): the active-opcode dim writes are excluded from the cache because they can change between forwards within a run.

---

## Cross-reference with blog post

| Violation | Blog stance |
|---|---|
| V1 (`_inject_active_opcode`) | Not explicitly addressed. Blog doesn't discuss MoE or opcode hints. Implicit intent: deterministic from bytecode → should be neural. |
| V2 (`_inject_mem_metadata` addr decode) | **EXPLICITLY PRAGMATIC** (BLOG_POST.md:376-381). "Could the indexing be neural too? Yes—with attention over memory slots. But that's a different (and larger) project." |
| V3 (`_inject_mem_store` history) | Same as V2 — memory infrastructure. |
| V4 (`set_active_opcode` MoE swap) | Not addressed. Pure optimisation. |
| V5 (`_memory` dict) | **EXPLICITLY PRAGMATIC** (BLOG_POST.md:267-279). "For this implementation, I made a pragmatic choice: memory is a Python dictionary." |
| V6 (`_mem_history`) | Memory infrastructure per blog. |
| V7 REQUIRED handler-mode overrides | All target ALU/control flow that the blog says SHOULD be neural ("computation can be neural"). These are the **unintended** violations — phase work. |
| V7 REMOVABLE-NOW subset | Cosmetic — model already produces correct output; overrides re-assert it. |
| V8/V9/V11/V16/V17 (I/O boundary) | Implicitly allowed — Turing-tape analogy (BLOG_POST.md:285): "you don't insist that the tape be made of logic gates. The tape is storage." |
| V10 (pure_neural MEM persistence) | Hybrid: the storage is pragmatic (V5/V6) but the "persist current step → next step" cross-step glue is the kind of thing the blog hand-waves around ("the values flowing in and out are in neural format" — line 307). |
| V18 (convo-I/O) | Not addressed; separate I/O architecture. |

**Unintended violations** (per blog, should be neural eventually):

- V7 REQUIRED handler-mode overrides (Phase 2-7 work)
- V10 pure_neural MEM persistence (Phase 7 + alibi-mem-attention)

**Intended pragmatic** (per blog, fine to keep):

- V2 mem metadata
- V5 `_memory` dict
- V6 `_mem_history`
- V8/V9/V11/V16/V17 I/O boundary

**Cosmetic / dead-code-after-handler-retires**:

- V1 active opcode (with the L5/L6 work)
- V4 MoE swap
- V12 set_active_opcode call sites
- V18 conversational-I/O
- V19 `_func_call_handlers`
- V20 prefix cache (already an optimisation, not a violation)

---

## Recommended priority order for future cleanup

1. **Land alibi-mem-attention** (in-flight): unblocks V3 deletion, simplifies V6, reduces V10 to write-only.
2. **Phase 2 (PSH+ADD)**: deletes V7 blocks 1, 8, 9. **Pure_neural Phase 2 unblocks 18 ops** (ADD/SUB/AND/OR/XOR/SHL/SHR/MUL/DIV/MOD/EQ/NE/LT/GT/LE/GE/SI/SC binary-pop + ADJ + PSH).
3. **Phase 3 (multi-byte ALU)**: deletes V7 block 13 (AX merge).
4. **Phase 4 (BZ/BNZ)**: deletes V7 blocks 6, 7.
5. **Phase 5 (JSR/ENT/LEV/LEA)**: deletes V7 blocks 2, 3, 4, 12. Also closes the OP_LEV gap in V1 (`SET_ACTIVE_OPCODE_PURITY.md` §7 recommendation).
6. **Phase 7 (memory)**: deletes V7 blocks 10, 11; reduces V10 to just write persistence.
7. **Handler-mode retirement** (after pure_neural is the sole runtime): deletes V1, V4, V12, V18, V19, and all REMOVABLE-NOW blocks in V7.
8. **(Optional) V2 neural address decode**: blog-deferred; consider as research project, not purity work.
9. **(Optional) V5 attention-as-memory**: blog-deferred; out of scope for normal purity work.

The headline: after today's I1/I3/B2 + alibi-mem-attention land,
**only V7 (REQUIRED) and V10 are pure_neural-blocking**, and all the
intended-pragmatic violations (V2/V5/V6/V8/V9/V11/V16/V17) match the blog
spec.

---

## Appendix A — Files examined (read-only)

- `c4_release/old/BLOG_POST.md` lines 255-385 (memory + spec sections)
- `c4_release/neural_vm/neural_embedding.py` (full file)
- `c4_release/neural_vm/run_vm.py` lines 1-1879
- `c4_release/neural_vm/vm_step.py` lines 1076-1304, 6900-7000
- `c4_release/neural_vm/base_layers.py` lines 220-300
- `c4_release/neural_vm/unified_compiler/compiler.py` lines 780-860
- `c4_release/neural_vm/purity_guard.py` (full file)
- `c4_release/neural_vm/purity_check.py` (full file)
- `c4_release/docs/PURE_NEURAL_POLICY.md` (full file)
- `c4_release/docs/SET_ACTIVE_OPCODE_PURITY.md` (full file)
- `c4_release/docs/INJECT_MEM_STORE_MIGRATION_PLAN.md` lines 1-100

## Appendix B — Headline counts (post-I1/I3/B2 + alibi)

- Pure_neural-blocking violations: **2** (V7 REQUIRED + V10)
- Deferred-architectural (intentional per blog): **3** (V2, V5, V6)
- Cosmetic / dead-code-after-handler-retires: **5** (V1, V4, V12, V18, V19)
- External / always-Python: **5** (V8, V9, V11, V16, V17)
- LANDED / in-flight: **4** (V3 alibi-in-flight, V13 I1-in-flight, V14 I3-in-flight, V15 I2-landed)
- Optimisation, not violation: **1** (V20)

**Total residual surface tracked: 20 items.**
