# Qwen R8 Production VM byte-identity status (2026-06-09)

Snapshot of `test_r8_full_production_roundtrip` after the
[`QWEN_R8_E2E_2026_06_07.md`](QWEN_R8_E2E_2026_06_07.md) blocker
fixes — most recently `c07c1236`
("fix(qwen-export): pad attention to max num_heads for non-uniform layers")
and `1ab7b8e5` ("use max(ffn_dims) for intermediate_size"), which
together cleared the HF state_dict shape mismatch that previously
xfailed Step 4.

## Production roundtrip result

Command:

```
cd c4_release && C4_QWEN_EXPORT_COMPAT=1 CUDA_VISIBLE_DEVICES="" \
  python -m pytest tests/test_qwen_r8_e2e.py --runslow -v
```

State: **HARD FAIL at Step 8 — argmax assertion** (1 failed, 4 passed
in ~16s).

Pytest tail:

```
c4_release/tests/test_qwen_r8_e2e.py:517: in test_r8_full_production_roundtrip
    assert match >= R8_ARGMAX_TARGET, (
E   AssertionError: R8 argmax match 0.0500 below plan §R8 target 0.99.
E   Window: 20 positions. This is the live ≥99 % gate — if all blockers
E   are landed the gap is a regression in the export pipeline.
E   assert 0.05000000074505806 >= 0.99
```

### Interpretation

Steps 1–7 all complete cleanly:

1. Compile production VM
2. R5 `flatten_post_ops_for_qwen_export`
3. `export_qwen3_dense`
4. `AutoModelForCausalLM.from_pretrained` — **now PASSES** (was xfail on
   `1200×800` vs `800×800` Linear shape mismatch; cleared by
   `c07c1236` + `1ab7b8e5`)
5. Tokenize / build inputs
6. Reference forward (native VM)
7. Exported `qmodel` forward

Step 8 is the live ≥99 % gate: argmax-token match between the native VM
and the exported HF model, over the 20-position decode window on
`int main(){return 42;}`. Observed match is **5.00 %** (1 / 20) —
indistinguishable from chance for an 8-token sentinel vocabulary.
The exported `qmodel` runs, but its forward distribution has no
recognisable correlation with the native VM's; the export pipeline is
producing a syntactically valid Qwen3 checkpoint whose semantics are
disconnected from the source weights.

The chance-level match strongly implicates the three unfinished
production-wiring blockers (I1, I2, I3) rather than a single localised
regression: those are exactly the export-time substitutions where the
production VM's behaviour diverges from what a stock Qwen3
runtime can execute, and none of them are end-to-end wired yet.

## argmax match on `int main(){return 42;}`

**5.00 %** (1 / 20). Far below the R8 ≥99 % gate.

## Blockers — status

| Blocker | Address | Verified end-to-end? |
|---|---|---|
| D1 composite FFN flattener | `1ab7b8e5` `max(ffn_dims)` for `intermediate_size` | YES — HF load passes |
| D2 attention head padding | `c07c1236` `max(num_heads)` for non-uniform layers | YES — HF load passes |
| D3 softmax1 sink runtime | runtime hint landed (`3d02dcc0`) | Unverified — Step 8 fails before this can be isolated |
| I1 per-head `q_norm` / `k_norm` | **NOT landed in production VM** — tiny-VM closed via per-head compensator bake only | NO |
| I2 ALiBi → RoPE | **NOT landed in production VM** — `f7af0777` is an export-time stub (doc-level Blocker 4); production VM is still compiled with ALiBi | NO |
| I3 `NeuralVMEmbedding` augmentations | wrapper landed (`94072b3b`, "export NeuralVMEmbedding augmentations via runtime wrapper") | NO — not verified end-to-end through Step 8 |

D1 and D2 are now genuinely closed: the export pipeline produces a
state_dict that HF's `AutoModelForCausalLM.from_pretrained` accepts
without shape errors. I1, I2, and I3 are the open front line.

## Recommended next actions

The path from 5 % to ≥99 % argmax match is to finish wiring I1, I2,
and I3 into the **production** VM (not just tiny-VM or export-time
stubs):

1. **I1 — per-head `q_norm` / `k_norm` in production.** The tiny-VM
   path bakes a per-head compensator that closes the gap; production
   currently has neither the compensator nor a real `q_norm` /
   `k_norm` layer wired into the exported Qwen3 attention block.
   Without per-head Q/K normalisation the exported attention scores
   diverge from the native VM's almost immediately, which alone is
   sufficient to drive the argmax match to chance.
2. **I2 — ALiBi → RoPE production wiring.** `f7af0777` only documented
   the transform and added an export-time stub; the production VM is
   still compiled with ALiBi slopes, and the export does not yet
   rewrite per-head positional biases into a RoPE base + per-head
   inv_freq table that matches the `Qwen3Config.rope_theta` /
   `rope_scaling` that `export_qwen3_dense` declares. Until this
   lands, the exported model's positional behaviour is
   fundamentally a different function of `position_ids` than the
   native VM's.
3. **I3 — `NeuralVMEmbedding` augmentation verification.** The
   `NeuralVMEmbeddingWrapper` runtime protocol landed at `94072b3b`,
   but it has not been gated end-to-end through Step 8. Confirm the
   wrapper's augmentation tensors are actually being written into
   `model.embed_tokens.weight` in the exported state_dict, and that
   the HF load path reads them back identically. If the augmentations
   are silently dropped at export, every input token starts the
   forward pass at a different residual than the native VM.

After each of I1 / I2 / I3 lands, re-run the command above and record
the new argmax match in this doc. The expected progression is
monotone-up; if any one of them lands without moving the needle,
that is itself a signal worth investigating before stacking the next.

## Smoke baseline

Smoke (`tests/test_smoke.py`, flag OFF) is unchanged from the
2026-06-07 snapshot: 46 / 51 passed, 1 deselected, 5 failures in the
`TestSmokeMemory::test_si_li_*` / `test_sc_lc_roundtrip` family
(unrelated to R8 / qwen_compat).
