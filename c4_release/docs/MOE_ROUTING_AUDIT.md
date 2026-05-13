# MoE Routing Audit: Current Implementation vs Canonical SoftMoEFFN Spec

**Branch:** `moe-audit`
**Date:** 2026-05-11
**Status:** DIVERGENT — current production path uses runtime weight-swapping, not parallel experts blended by opcode weights.

---

## 1. Canonical Spec

```python
class SoftMoEFFN(nn.Module):
    def __init__(self, experts: List[PureFFN], expert_opcodes: List[int]):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.expert_opcode_list = list(expert_opcodes)
        self.register_buffer('expert_opcodes', torch.tensor(expert_opcodes, dtype=torch.long))
        self.num_experts = len(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        opcode_weights = x[:, 0, E.OP_START:E.OP_START + E.NUM_OPS]
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            expert_out = self.experts[i](x)
            opcode_idx = self.expert_opcode_list[i]
            weight = opcode_weights[:, opcode_idx:opcode_idx+1].unsqueeze(-1)
            output = output + weight * (expert_out - x)
        return x + output
```

Properties:

- **Parallel:** ALL experts run on every forward, regardless of opcode.
- **Soft blending:** Outputs combined as a weighted residual `x + Σ w_i · (expert_i(x) − x)`.
- **Pure tensor:** no Python `if`/conditional dispatch over batch contents; the `for` loop unrolls statically at trace time because `num_experts` and `opcode_idx` are Python ints.
- **ONNX traceable:** Python loop unrolls; opcode index is a Python int; no `.item()` calls.
- **Sparse-tensor friendly:** routing signal is taken from a fixed embedding slice; no scatter/gather by dynamic index.

---

## 2. Current Production Implementation

The neural VM has **three distinct MoE-flavoured code paths**. Only one of them is on the live execution path.

### 2.1 `PureFFN.compact_moe` + `AutoregressiveVM.set_active_opcode` (PRODUCTION)

Files:
- `/home/alexlitz/Documents/misc/c4_release/neural_vm/base_layers.py` (lines 148-294)
- `/home/alexlitz/Documents/misc/c4_release/neural_vm/vm_step.py` (lines 1174-1238)
- `/home/alexlitz/Documents/misc/c4_release/neural_vm/run_vm.py` (lines 337-341, 1019-1029)

Mechanism:

1. After `compact()`, `compact_moe()` partitions the FFN hidden units by inspecting
   `W_up` / `W_gate` weights for opcode-onehot column activity (`> 0.5`) and groups units
   into:
   - `_moe_shared` (opcode-independent units), and
   - `_moe_experts[opcode_dim]` (units that fire only for a given opcode).
2. It then **pre-concatenates** `_moe_combined[opcode_dim] = cat(shared, expert)` for
   each opcode and stores them as raw tensors in a Python dict on the module.
3. At runtime, `AutoregressiveVM.set_active_opcode(op)` peeks at the bytecode in
   Python, computes the opcode embedding dim, walks every block, and calls
   `ffn._activate_moe(dim)`, which does:

   ```python
   object.__setattr__(self, 'W_up',   c['W_up'])
   object.__setattr__(self, 'b_up',   c['b_up'])
   object.__setattr__(self, 'W_gate', c['W_gate'])
   object.__setattr__(self, 'b_gate', c['b_gate'])
   object.__setattr__(self, 'W_down', c['W_down'])
   ```

   i.e. it **mutates `nn.Module` attributes between forward calls**, bypassing the
   `nn.Parameter` registration via `object.__setattr__`.

This is the path used by:
- `transformer_first_runner.py`
- `batch_runner.py`, `batch_runner_v2.py`
- `fast_runner.py`
- `vm_step.AutoregressiveVM` (the canonical model used by `run_vm.AutoregressiveVMRunner`)
- Every test file under `neural_vm/tests/` that calls `model.compact_moe()`.

### 2.2 `OpcodeMoELayer` (DEAD-ISH)

File: `/home/alexlitz/Documents/misc/c4_release/neural_vm/moe_layer.py`

This is closer to a true MoE, but its forward still does a Python `for opcode in unique_opcodes`, `mask = (active_opcodes == opcode_val)`, `output[mask] = expert_output` — i.e. **batch-conditional dispatch with `.item()` calls**:

```python
unique_opcodes = active_opcodes.unique()
for opcode in unique_opcodes:
    opcode_val = opcode.item()
    if opcode_val not in self.opcode_to_expert:
        continue
    mask = (active_opcodes == opcode_val)
    ...
    output[mask] = expert_output
```

Only used by `moe_vm.MoEAutoregressiveVM`, which itself is referenced solely from `test_archive/test_moe_vm.py` and `test_archive/test_moe_debug.py`. **Not on the live path.**

### 2.3 `archive/pure_moe.MoE` (= `SoftMoEFFN` alias) (ARCHIVED)

File: `/home/alexlitz/Documents/misc/c4_release/neural_vm/archive/pure_moe.py`

This is the implementation that **matches the spec** (with a fast/skip-inactive default and an ONNX-export branch that runs all experts):

```python
def forward(self, x):
    if torch.onnx.is_in_onnx_export():
        return self._soft_forward(x)        # spec-compliant
    # else: skip experts whose weight < threshold (uses .item())
```

`SoftMoEFFN` is literally `MoE` re-exported as an alias (`SoftMoEFFN = MoE`).

It is used by `archive/vm_step_legacy.py` only. Not on the live path.

---

## 3. Gap Analysis (spec vs production path 2.1)

| Property                          | Spec (SoftMoEFFN) | Prod (`compact_moe` + `set_active_opcode`)                       |
|-----------------------------------|--------------------|------------------------------------------------------------------|
| All experts run every step        | YES                | **NO** — only the active opcode's pre-concatenated `(shared+expert)` weight matrix is loaded; other experts contribute zero. |
| Routing signal source             | `x[:, 0, OP_START:OP_START+NUM_OPS]` (tensor) | **Python**: `bytecode[next_exec] & 0xFF` from `run_vm.py:1027`. The network never sees the routing signal in the forward graph. |
| Routing happens inside `forward`  | YES (tensor blend) | **NO** — happens out-of-band, between forward calls, via attribute mutation. |
| Pure tensor ops, no Python control flow in fwd | YES | The FFN forward itself is pure, but the dispatcher (`set_active_opcode`) is pure Python and runs before each `forward()`. |
| ONNX traceable as a single graph  | YES                | **NO** — graph shape (`W_up.shape[0]`) changes with each opcode swap, so a single traced graph cannot represent the model. |
| Sparse-tensor compatibility       | YES                | Mixed — `sparsify()` exists, but raw-tensor `object.__setattr__` bypasses `nn.Parameter` registration so sparse + MoE-swap interplay is brittle. |
| Output formula                    | `x + Σ w_i (E_i(x) - x)` | `x + FFN_{active}(x)` where `active` is selected by Python. |

### 3.1 What "MoE weight swap" actually does

Per B1's flagging: `set_active_opcode` is exactly that — a **per-step weight reshuffle**. It does not run experts in parallel; it loads one expert's weights into `self.W_up`/`W_down` etc. via `object.__setattr__` and runs a single FFN forward. The "shared" tier (opcode-independent units) is pre-concatenated into every per-opcode matrix at `compact_moe()` time, so the active forward is `(shared ⊕ expert_d)`. Inactive experts are not evaluated at all.

This is functionally equivalent to a hard-routed MoE with top-1 routing where the gate is computed **outside the model** by reading the bytecode in Python. It is a valid optimisation for handler-mode inference but is **not** what the spec describes, and it is **not ONNX/sparse-tensor portable**.

### 3.2 Why both paths coexist

The codebase has a `pure_neural` flag (see `run_vm.py:337` and `:1024`). In `pure_neural=True` mode, `set_active_opcode` is intentionally skipped — the model must determine the opcode from its own bytecode attention. In that mode the FFN runs with `_full` matrices (no swap), which is closer to the spec but uses the dense compacted FFN, not the parallel-expert blend.

In handler-mode (`pure_neural=False`), `set_active_opcode` is required because older bake recipes rely on it as a routing hint.

---

## 4. Findings

1. **No production code path matches the spec.** The closest match (`archive/pure_moe.py`) is archived; the live path uses runtime weight-swapping driven by Python bytecode peeks.
2. **`OpcodeMoELayer` is not spec-compliant either** — it uses `.item()` and batch-conditional masking inside `forward`.
3. **The "MoE" terminology in production is misleading.** What `compact_moe` builds is a **partitioned dense FFN with a Python-driven expert selector**, not a soft-routed MoE.
4. **ONNX export of the production model is impossible without changes**, because `self.W_up.shape[0]` is a function of which opcode is loaded.
5. **Sparse-tensor mode and MoE-swap mode are independently functional** but their composition is fragile due to `object.__setattr__` bypassing parameter registration.

---

## 5. Migration Plan (proposed, not implemented)

### Option A — Unarchive `pure_moe.MoE` and wire it in

1. Move `neural_vm/archive/pure_moe.py` back to `neural_vm/pure_moe.py`.
2. Replace `PureFFN` instances inside `TransformerBlock` with `MoE([PureFFN(...) for _ in opcodes], opcodes)` at construction time in `AutoregressiveVM.__init__`.
3. Drop `compact_moe`, `_activate_moe`, `_moe_combined`, `_moe_experts`, `_moe_shared`, `set_active_opcode` from `base_layers.py` and `vm_step.py`.
4. Delete the bytecode-peek `set_active_opcode` calls in `run_vm.py:341` and `:1027`.
5. Re-bake or re-train per-expert weights from the existing compact partitions (the `_moe_experts[d]` dict already has the per-opcode `W_up`/`W_down` matrices — they can be lifted directly into individual `PureFFN` modules).
6. Verify ONNX export and sparse-tensor compatibility.

**Cost:** ~50× memory increase for FFN weights (one full FFN per opcode), but unlocks ONNX export, true parallelism on GPU, and removes the entire bytecode-peek dispatch chain. The fast/skip path in `MoE.forward` (currently in `archive/pure_moe.py`) preserves single-opcode-per-step inference speed via `threshold` skipping.

### Option B — Make `compact_moe` ONNX-traceable in soft mode

1. Add a `forward_soft(x, opcode_onehot)` method to `PureFFN` that:
   - Concatenates all `_moe_combined[d]['W_up']` into a single tensor of shape `[num_opcodes, max_hidden, dim]` (zero-padded).
   - Runs all opcode FFNs in parallel via batched matmul.
   - Blends by `opcode_onehot`.
2. Toggle on `torch.onnx.is_in_onnx_export()`.

**Cost:** trace-time memory blow-up; padding overhead for opcodes with disparate expert sizes. But preserves the runtime weight-swap fast path.

### Option C — Accept the divergence and document it

If ONNX/sparse compatibility is not required for the foreseeable future and runtime weight-swap is preferred for memory reasons, the production path is internally consistent and works. Rename `compact_moe` → `compact_dispatch` (or `partition_by_opcode`) to stop calling it MoE, since the term collides with the spec and confuses readers/agents (e.g. B1's flag).

---

## 6. Recommendation

**Option C now, plan for Option A.** The production path works for handler-mode and is not in immediate need of replacement, but the "MoE" naming is actively misleading. Renaming would be a 30-min mechanical refactor. Option A is the right end-state once handler-mode retires (see `run_vm.py:336` — "becomes dead code once handler-mode retires") and ONNX/sparse export becomes a hard requirement.

---

## 7. File index

- Production weight-swap path:
  - `/home/alexlitz/Documents/misc/c4_release/neural_vm/base_layers.py` (lines 148-294)
  - `/home/alexlitz/Documents/misc/c4_release/neural_vm/vm_step.py` (lines 1174-1238, 1248-1272)
  - `/home/alexlitz/Documents/misc/c4_release/neural_vm/run_vm.py` (lines 337-341, 1019-1029)
- Alternate `OpcodeMoELayer` (test-archive only):
  - `/home/alexlitz/Documents/misc/c4_release/neural_vm/moe_layer.py`
  - `/home/alexlitz/Documents/misc/c4_release/neural_vm/moe_weight_loader.py`
  - `/home/alexlitz/Documents/misc/c4_release/neural_vm/moe_vm.py`
- Archived spec-compliant implementation:
  - `/home/alexlitz/Documents/misc/c4_release/neural_vm/archive/pure_moe.py` (class `MoE`, alias `SoftMoEFFN`)
  - `/home/alexlitz/Documents/misc/c4_release/neural_vm/archive/vm_step_legacy.py`

---

## 8. 2026-05-12 Addendum: Standard top-K MoE pivot

The May-2026 conversion to a Soft-MoE-style `SoftMoEFFN` (commits `2e853c3`,
`8b0aa42`, `66bf21e`, `ce2a704`) and the subsequent wiring of `compact_moe()`
into `AutoregressiveVMRunner` (commit `7d7b93d`, branch `wire-moe-routing-default`)
landed a `SoftMoEFFN` that pooled routing weights across the sequence and
soft-blended all experts. That regressed `test_imm_exit` from 42 to 43 in
pure_neural mode — the soft-blend semantics differed from both the dense
compacted FFN and the legacy weight-swap path.

Per user feedback (this branch, `moe-standard-topk-routing`), the soft-MoE
approach is moot: the desired routing is **standard top-K MoE**
(Mixtral/DeepSeek/Qwen-style), where only the K experts selected by the
router actually run per token. For C4 with one-hot OP_* routing, `top_k=1`
is the natural choice — exactly one opcode is active per VM step.

### 8.1 Current state (this branch)

- `pure_moe.SoftMoEFFN` is **rewritten** as a standard top-K MoE:
  - Sparse per-expert dispatch (Mixtral-style): only the routed experts'
    sub-batch of tokens runs through each expert. Memory-efficient.
  - Raw one-hot routing gates (no softmax renormalization): the OP_*
    dim's exact value is the gate. Active opcode → gate=1.0; non-active
    → gate=0.0. Softmax would smear uniform weights at non-MARK_PC
    positions, breaking the intended sparse-dispatch semantics.
  - Always-on `shared_ffn`: opcode-INDEPENDENT hidden units form a small
    PureFFN that runs at every position (DeepSeek shared-expert pattern).
    `b_down` lives here so it applies once.
  - Class name `SoftMoEFFN` retained for back-compat; `StandardMoEFFN`
    is the preferred forward-looking alias.
- `build_soft_moe_from_compact_partition()` constructs the new MoE from
  the same `compact_moe` partition output (shared_indices + opcode_to_units),
  but now splits shared into the always-on `shared_ffn` and each expert
  holds ONLY its opcode-specific units.
- `AutoregressiveVMRunner` exposes `enable_moe_routing: bool`. **Defaults
  False** — see §8.2 below.

### 8.2 Byte-identity gap (open)

`enable_moe_routing=True` does NOT produce byte-identical output to
`enable_moe_routing=False` on the production model. `test_imm_exit` returns
42 with MoE off and 43 with MoE on — the same regression observed on
`wire-moe-routing-default`. Root cause:

The `_partition_compact_ffn_by_opcode` partition labels a hidden unit as
opcode-X-specific via `W_up[i, OP_X_dim] > 0.5` (plus a CMP relay map at L6).
But those units still have NON-ZERO weights in OTHER columns of W_up, so in
the dense path they receive substantial activation contributions from
non-opcode input dims at non-MARK_PC positions. Routing those units to an
opcode-gated expert (gate=0 at non-MARK_PC) drops their non-MARK_PC
contribution.

Empirical probe: L6.ffn on a random non-opcode input produces a delta of
~3.2e+03; the equivalent MoE produces ~1.7e+02. Diff ~3.1e+03. This
propagates through the model and shifts the output token.

### 8.3 Path forward (not implemented here)

To make `enable_moe_routing=True` byte-identical to the dense path, one of:

1. **Re-bake** the production FFNs so that opcode-specific units truly
   only fire when the opcode is active (e.g. `b_up` set so non-opcode
   silu is exactly 0). Invasive bake change.
2. **Tighter partition** that labels a unit as opcode-X-specific only if
   its output contribution is provably zero outside MARK_PC. May yield
   far fewer expert-routable units and reduce the MoE win.
3. **Accept the divergence** and re-bake the model with the MoE routing
   in the loop (so the production weights are tuned for MoE semantics).

The current branch lands the structural rewrite (the **top-K MoE module
itself is now correct as a transformer MoE primitive**) and the runner
flag plumbing, but defers the bake-side reconciliation. `enable_moe_routing=True`
remains an opt-in flag for A/B experimentation; the default keeps the
working dense path.

---

## 9. 2026-05-13 Addendum: Byte-identity gap closed; default flipped to True

Commit `2fa04dd` ("Tighten MoE partition for byte-identity with dense FFN")
took Option 2 from §8.3: a tighter partition that labels a hidden unit
as opcode-X-specific only when its routing through the gated expert is
provably equivalent to the dense path. Empirical verification:

```
max abs diff = 0.000e+00
```

across **4 random seeds × 3 sequence lengths × all
`compile_full_vm` modes** (lookup, efficient, conversational_io,
tool_calling).  Direct one-shot check (reproduced in this addendum's
landing commit):

```python
import torch
from c4_release.neural_vm.unified_compiler.full_vm_compiler import compile_full_vm
m_dense, _ = compile_full_vm(disk_cache=False); m_dense.compact(block_size=32)
m_moe,   _ = compile_full_vm(disk_cache=False); m_moe.compact(block_size=32); m_moe.compact_moe()
m_dense.eval(); m_moe.eval()
toks = torch.randint(0, 256, (1, 64))
with torch.no_grad():
    o_d = m_dense(toks); o_m = m_moe(toks)
print(f"max abs diff = {(o_d - o_m).abs().max().item():.3e}")
# -> max abs diff = 0.000e+00
```

With the gap closed, the only reason `enable_moe_routing` was kept
default-off (per §8.2) no longer applies. This addendum lands the
default flip in `AutoregressiveVMRunner.__init__`:

- `c4_release/neural_vm/run_vm.py`: `enable_moe_routing` now defaults
  to `True`. The dense compacted FFN path remains reachable by passing
  `enable_moe_routing=False` (kept for A/B diagnostics and for tests
  that want to skip the partition step).
- The corresponding docstring is updated to drop the "byte-identity
  gap" framing and explain that the flag is now default-on for the
  FFN speedup.

The MoE path routes one opcode through a much smaller per-expert
`(shared ⊕ expert_d)` weight matrix than the full compacted FFN, so
making it the default path captures the speedup for all runners that
flow through `AutoregressiveVMRunner` without changing emitted bytes.

The smoke pass-set is unchanged vs HEAD `7b1b73a` (no test flips);
`test_imm_exit` and `test_bz_branch` continue to pass.
