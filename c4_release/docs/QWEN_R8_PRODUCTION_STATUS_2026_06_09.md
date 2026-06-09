# Qwen R8 Production VM byte-identity status (2026-06-09)

Snapshot of `test_r8_full_production_roundtrip` after the five
[`QWEN_R8_E2E_2026_06_07.md`](QWEN_R8_E2E_2026_06_07.md) blockers were
nominally addressed (the most recent landing was `1ab7b8e5` —
"fix(qwen-export): use max(ffn_dims) for intermediate_size").

## Base

* HEAD: `ed7450de` ("Revert ir(l10): F2 Edit C — rekey stack0_pop_loaded
  to ALU_LO/HI")
* `1ab7b8e5` is in history (one of `ed7450de`'s ancestors via `c0b02e7d`),
  so the `intermediate_size = max(ffn_dims)` fix is live in this branch.

## Production roundtrip result

Command:

```
cd c4_release && C4_QWEN_EXPORT_COMPAT=1 timeout 600 \
  python -m pytest tests/test_qwen_r8_e2e.py::test_r8_full_production_roundtrip \
  --runslow --tb=short -v
```

State: **XFAIL** (1 xfailed in 14.62s).

Failing step: **Step 4 — `AutoModelForCausalLM.from_pretrained`** load.

Pytest's xfail message (verbatim — line breaks normalised):

```
AutoModelForCausalLM.from_pretrained failed after export:
RuntimeError: Error(s) in loading state_dict for Linear:
  size mismatch for weight: copying a param with shape
  torch.Size([1200, 800]) from checkpoint, the shape in current model is
  torch.Size([800, 800]).
Likely blockers: D1/D2/I3 (state_dict shape mismatch on flattened
composite blocks, sink token id, or embedding augmentations).
```

### Interpretation

Steps 1–3 (compile production VM, R5 `flatten_post_ops_for_qwen_export`,
`export_qwen3_dense`) now complete cleanly. The
`max(ffn_dims)` fix in `1ab7b8e5` got us past the ZIP-record error in
`torch.save`; the artefact is written to disk and the HF auto-loader is
reached.

The new failure is a Linear-layer **state_dict ↔ model-config shape
mismatch** at HF load time: at least one Linear in the state_dict has
shape `(1200, 800)` while the config-derived module expects `(800, 800)`.
`800` matches the composite FFN max-internal-width target
(`max(ffn_dims)`); `1200` is `1.5 × 800`. The 1.5× ratio is the SwiGLU
gate+up stack (`2 × intermediate_size`-style concat would be `1600`;
`1.5×` is consistent with an `up` packed with the first gate-segment
when `intermediate_size` was previously misread). The likely root cause
is that `_build_qwen3_dense_config.intermediate_size` and the
state_dict-construction code are not consistently pulling the same
value — config gets `800` but at least one layer writes its W_up/W_gate
at the per-layer composite width that the config zero-pad pass thought
it had erased.

This is a downstream sibling of Blocker 1 (composite FFNs flattened
into Qwen layers): the export now produces a state_dict, but the
flatten + zero-pad pass is **not yet uniform** across all blocks /
projections. Blocker 1 is therefore "partially closed" (export
writes), not "closed" (HF loads).

## argmax match on `int main(){return 42;}`

Not measurable yet. The test xfails at Step 4 before any forward pass
runs (Steps 6 and 7 require a loaded `qmodel`). The R8 ≥99 % gate
remains unobserved on the production VM.

## Other blockers — status

| Blocker | Address | Verified end-to-end? |
|---|---|---|
| D1 composite FFN flattener | `1ab7b8e5` `max(ffn_dims)` for `intermediate_size`; export now writes | No — Linear shape mismatch at HF load |
| D2 softmax1 sink runtime | landed as runtime hint (`3d02dcc0`) | No — load gate doesn't reach it |
| I1 per-head q_norm / k_norm | tiny VM closed via per-head compensator bake; production deferred per E2E doc | No |
| I2 ALiBi → RoPE | export reads `rope_base`; production VM still compiled with ALiBi | No |
| I3 `NeuralVMEmbedding` augmentations | `NeuralVMEmbeddingWrapper` runtime protocol landed | No — load gate doesn't reach it |

The 5-blocker checklist is "code present, integration unverified" — the
HF-load gate is the new front line and reveals that D1's flatten/zero-pad
pass is not uniform across all Linear projections.

## Smoke baseline

Smoke (`tests/test_smoke.py`, flag OFF): **46 / 51 passed**, 1 deselected.
Above the ≥45 floor. The 5 failures are all
`TestSmokeMemory::test_si_li_*` / `test_sc_lc_roundtrip` — same family,
unrelated to R8 / qwen_compat.

## Next step

Identify which Linear is `(1200, 800)` in the exported state_dict. The
1.5× factor strongly suggests a per-block `intermediate_size` that
wasn't padded to `max(ffn_dims)` during state_dict materialisation, or
a SwiGLU repack that concatenated gate+up into a `(intermediate × 1.5,
d_model)` weight. Both candidates live in `extract_composite_ffn_weights`
+ `_swiglu_repack_and_fold_bias` in `neural_vm/qwen_compat.py`.
