# Ground-truth probe (`tools/probe_groundtruth.py`) — 2026-06-10

A residual / logit probe that runs the **exact** execution path the smoke
gate measures: `BatchedPureNeuralRunner` with `spec_k=0`. Prior probe-based
diagnosis this session was wrong because the probes ran a *different* path
(`spec_k=8`) than the smoke gate. This probe is the corrected, validated
ground truth.

## The spec_k=0-vs-8 pitfall (read this first)

| | Smoke gate (GROUND TRUTH) | Old probe tooling / conftest default |
|---|---|---|
| Env | `C4_SMOKE_SPEC_K=0` | `C4_TEST_SPEC_K=8` |
| Executor | `BatchedPureNeuralRunner` (`C4_SMOKE_EXECUTOR=batched`) | same class, different `spec_k` |
| Decode | **raw, one token per forward**, no DraftVM, no KV speculation | speculative decode + KV cache |
| Code path | `_run_unspeculative` → `model.forward(padded)` → argmax | `_run_speculative` (DraftVM proposes K·35 tokens, verify+accept) |
| `test_ne_true` | **PASSES** (returns 1) | agents saw it "fail" (returned 16) |

The earlier "forward hooks change model behaviour" claim was a
**misdiagnosis** — the real cause was spec_k=8-vs-0. The production model is
hook-free (hooks live only in `debug_archive/`). See memory note
`project_probe_path_spec_k_not_hooks.md`.

**Rule:** all probing MUST use `spec_k=0`. This probe pins both
`C4_SMOKE_SPEC_K=0` and `C4_TEST_SPEC_K=0` at import and asserts
`use_kv_cache is False`. It never touches the speculative path.

## How it reads internals without hooks

* **Logits / emitted token** — the probe replays the runner's own
  `_run_unspeculative` step loop and calls `model.forward` directly on the
  same single-row padded context the runner builds (the runner's fresh-forward
  branch is literally `logits = self.model.forward(padded)` then `argmax`).
  The argmax is the emitted token. NO `register_forward_hook`.
* **Residuals** — `AutoregressiveVM.forward` grew an opt-in, probe-only
  `stop_after_block=<int>` kwarg. Default `None` ⇒ production behaviour is
  byte-identical (proven: `tests/test_batched_pure_neural.py` 17/17 pass and
  the 4-test byte-identity validation passes). When set, `forward` returns the
  model's own post-block hidden state instead of LM-head logits. The probe
  re-runs the prefix truncated so the target block is the last executed block
  and reads that normally-returned tensor. NO hooks, NO weight overrides.

## API

```python
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.embedding import E            # residual DimPosition enum

p = build_groundtruth_probe()                # bakes model once (slow)

# {position: {token, token_name, top_k_logits, forward_iter, context_len_before}}
trace = p.probe(program_bytes, top_k=8)

# {dim_name: residual_value} after physical block `block_idx`
res = p.residual_at(program_bytes, block_idx=36, position=-1,
                    dim_names={"AX_LO": E.AX_BASE, "OPCODE": E.OPCODE})

p.print_block_layer_map()                    # the table below
p.emitted_result(program_bytes)              # ("", exit_code) — == runner bytes
```

`position` indexes the full context (prompt + emissions); negative indexes
from the end. `dim_names` maps labels to residual-dim indices from the `E` /
`DimPosition` enum in `neural_vm/embedding.py`.

## Validation (acceptance gate)

`python tools/probe_groundtruth.py` (or `validate()`) runs each of the 4 named
programs through BOTH the real `run_batch(..., spec_k=0,
bucket_by_predicted_length=False)` (the smoke gate's exact call) AND the
probe's replay, and asserts `(output, exit_code)` are identical:

| Test | Smoke status | Bytes (runner == probe) |
|---|---|---|
| `TestSmokeComparison::test_ne_true` | PASSES | `('', 1)` |
| `TestSmokeComparison::test_eq_true` | FAILS (expected 1) | `('', 0)` |
| `TestSmokeBitwise::test_or_basic` | FAILS (expected 0x3F) | `('', 16777023)` = `0x00FFFFFF` |
| `TestSmokeMemory::test_si_li_roundtrip` | PASSES | `('', 42)` |

The probe reproduces the runner's emitted bytes **exactly** — including the
*wrong* values on the two failing tests. That is the point: the probe is a
faithful mirror of the ground-truth path, not an oracle.

## 37-physical-block ↔ logical-layer mapping (VALIDATED)

`_expand_wrapper_blocks` (`neural_vm/vm_step.py`) splits each logical block's
`post_ops` into successive **passthrough blocks** (zero-init attention =
residual identity, the post_op as the FFN), inserted immediately after the
originating block. The result is **37 physical blocks** from **27 logical
layers** (NOT 16 — the "~16-layer" figure in `CLAUDE.md` is an approximation;
the true pre-expansion count is 27).

Provenance is read from `_logical_layer` / `_is_post_op_expansion` tags the
expansion now stamps on every emitted block (metadata only, no behaviour
change). Mapping (`d_model=872`, vocab=276):

| phys | logical | expanded | attn | ffn |
|---:|---:|:---:|---|---|
| 0 | 0 | no | AutoregressiveAttention | PureFFN |
| 1 | 1 | no | AutoregressiveAttention | PureFFN |
| 2 | 2 | no | AutoregressiveAttention | PureFFN |
| 3 | 3 | no | AutoregressiveAttention | PureFFN |
| 4 | 4 | no | AutoregressiveAttention | PureFFN |
| 5 | 5 | no | AutoregressiveAttention | PureFFN |
| 6 | 6 | no | AutoregressiveAttention | PureFFN |
| 7 | 7 | no | AutoregressiveAttention | PureFFN |
| 8 | 8 | no | AutoregressiveAttention | PureFFN |
| 9 | 8 | **yes** | AutoregressiveAttention | AddSub5StageBlock |
| 10 | 9 | no | AutoregressiveAttention | PureFFN |
| 11 | 10 | no | AutoregressiveAttention | PureFFN |
| 12 | 11 | no | AutoregressiveAttention | PureFFN |
| 13 | 12 | no | AutoregressiveAttention | PureFFN |
| 14 | 13 | no | AutoregressiveAttention | PureFFN |
| 15 | 14 | no | AutoregressiveAttention | PureFFN |
| 16 | 14 | **yes** | AutoregressiveAttention | PureFFN |
| 17 | 14 | **yes** | AutoregressiveAttention | PureFFN |
| 18 | 14 | **yes** | AutoregressiveAttention | PureFFN |
| 19 | 14 | **yes** | AutoregressiveAttention | PureFFN |
| 20 | 14 | **yes** | AutoregressiveAttention | PureFFN |
| 21 | 14 | **yes** | AutoregressiveAttention | PureFFN |
| 22 | 14 | **yes** | AutoregressiveAttention | PureFFN |
| 23 | 14 | **yes** | AutoregressiveAttention | PureFFN |
| 24 | 15 | no | AutoregressiveAttention | PureFFN |
| 25 | 16 | no | AutoregressiveAttention | PureFFN |
| 26 | 17 | no | AutoregressiveAttention | ALUShiftComposite |
| 27 | 18 | no | AutoregressiveAttention | PureFFN |
| 28 | 19 | no | AutoregressiveAttention | PureFFN |
| 29 | 20 | no | AutoregressiveAttention | PureFFN |
| 30 | 21 | no | AutoregressiveAttention | PureFFN |
| 31 | 22 | no | AutoregressiveAttention | PureFFN |
| 32 | 23 | no | AutoregressiveAttention | PureFFN |
| 33 | 24 | no | AutoregressiveAttention | PureFFN |
| 34 | 25 | no | AutoregressiveAttention | PureFFN |
| 35 | 25 | **yes** | AutoregressiveAttention | PureFFN |
| 36 | 26 | no | AutoregressiveAttention | PureFFN |

Expansions: logical **L8** (+1, AddSub 5-stage), logical **L14** (+8 post_op
passthroughs), logical **L25** (+1). All other logical layers map 1:1. A
logical layer's residual at the boundary that historical "L_N" notes refer to
is the residual *after that logical layer's LAST physical block* — e.g.
logical L14 ends at **physical block 23**, not 15.

To probe "after logical layer N", use the largest physical block whose
`logical == N` (call `block_layer_map()` and filter).

## Constraints honoured

* No `register_forward_hook` / `register_forward_pre_hook` anywhere in the
  probe or the model path.
* No weight overrides — `stop_after_block` only changes the *return point*,
  not any parameter.
* `spec_k=0` throughout; the speculative path is never invoked.
