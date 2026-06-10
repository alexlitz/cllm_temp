# Serial-mode MUL / SI-LI 16-bit failures (2026-06-09)

Date: 2026-06-09
Brief: fix `TestSmoke32Bit::test_mul_overflow` and
`TestSmokeMemory::test_si_li_16bit_value` in `C4_SMOKE_EXECUTOR=serial`.

## TL;DR

The two failures are **NOT** the multi-byte wide-MUL DSL gap described
in `docs/DSL_W5_MULDIV_LIMIT.md` and they are **NOT** in
`wide_alu_dsl.wide_mul_rules` / `l11_ops.py` / `l12_ops.py`. The
declarative L11/L12 lookup path is only active when `alu_mode="lookup"`;
the smoke test runner uses `alu_mode="efficient"` (via
`trust_neural_alu=True` → `efficient` branch in `run_vm.py:349`), so the
authoritative MUL implementation in the failing run is
`FlattenedALUMul` (multi-byte schoolbook with carry passes — see
`efficient_alu_neural.py:1072` and `alu/ops/mul.py`).

The actual gap is a **serial-vs-batched runner divergence** at the
forward path:

| Runner | MUL result | SI/LI result |
| --- | --- | --- |
| `BatchedPureNeuralRunner.run_batch` (spec_k=0) | 500 (correct) | 512 |
| `AutoregressiveVMRunner.run` (spec_k=0, serial) | 0 | 52 (=0x34) |
| `AutoregressiveVMRunner.run` (spec_k=8, serial) | 0 | 52 (=0x34) |

batched MUL passes; batched SI/LI still fails (different bug — LI
returning 0x200, the address). Serial MUL truncates to 0; serial
SI/LI truncates to byte 0 (52 = 0x34). The serial path emits the
correct AX for the IMM/PSH/SI steps but corrupts AX during the
MUL / LI step's autoregressive token emission.

## Reproduction + localization

```
CUDA_VISIBLE_DEVICES=1 C4_SMOKE_EXECUTOR=serial python -m pytest \
  c4_release/tests/test_smoke.py -k "mul_overflow or 16bit_value" -v
# 2 failed: expected 500 got 0; expected 4660 got 52
```

Direct probe (bypasses pytest fixtures):

```python
runner = AutoregressiveVMRunner(pure_neural=True,
                                trust_neural_alu=True, spec_k=0)
batched = BatchedPureNeuralRunner(model_runner=runner)
# Same compiled model, same alu_mode='efficient', same weights.
batched.run_batch([mul_bytecode], spec_k=0)  # -> [('', 500)]
runner.run(mul_bytecode, b'')                # -> ('', 0)
```

Both paths use the same `FlattenedALUMul`-installed model. The
divergence is in `AutoregressiveVMRunner.run`'s per-step forward
loop, not in any FFN/attention weight.

### Per-step AX-byte trace (serial spec_k=0, MUL test)

Tracing the AX block emitted into context after each `_dispatch_step`:

```
op=0x1 (IMM 100): step ends with AX = [100, 0, 0, 0] = 100        ok
op=0xd (PSH):     step ends with AX = [100, 0, 0, 0] = 100        ok
op=0x1 (IMM 5):   step ends with AX = [5, 0, 0, 0]   = 5          ok
                  -- but a SECOND step is autoregressively pre-emitted
                     in the same dispatch with AX = [0, 0, 0, 0] = 0
op=0x1b (MUL):    runner reads last REG_AX block (the [0,0,0,0]
                  block from the pre-emission) -> _last_ax = 0
```

The model autoregressively emits the next step's tokens **inside**
the dispatch that closes the current step. The MUL step's emitted AX
is `[0,0,0,0]` despite the prior STACK0 / PC context being consistent
with what the batched forward sees. The same FlattenedALUMul produces
500 in the batched (full-prefix) forward.

### Wide-MUL DSL — not the gap here

`wide_alu_dsl.wide_mul_rules` is `width_bytes ∈ {1, 2}`-only and only
ever lowered through `make_efficient_l11_alumul_wrap_op` (which
`return`s a no-op in lookup mode and replaces `block.ffn` with a
256-rule POC `PureFFN` in efficient mode — see `alu_ops.py:781-834`).
The 256-rule POC writes only `OUTPUT_LO+lo_nib` / `OUTPUT_HI+hi_nib`
(low byte only). But the W5 wrap op `requires={"after":
"l12_alu_mul_getobd"}` so it runs **after** the 9-stage
`FlattenedALUMul` is fully assembled, and the FlattenedALUMul
composite (multi-byte) is the live MUL path in efficient mode. The
W5 POC's structural deferral (intractable past width_bytes=2) does
not block these tests because they go through the legacy composite,
not the DSL wrap.

The L11/L12 declarative `make_layer11_mul_partial_op` /
`make_layer12_mul_combine_op` (lookup-mode low-byte path) are
declarative (`compiler_ir=_layer11_mul_partial_ir()` /
`compiler_ir=_layer12_mul_combine_ir()`, `declarative_authority=
"spec_generated"`), but unused in this test run because
`alu_mode='efficient'`.

## Oracle replay

`replay_expected_diff.py` (extended in commits 4a354699 / 1c370089
to cover `MUL_ACCUM` / `AX_FULL_LO/HI`):

```
python tools/replay_expected_diff.py \
  --program "IMM 100; PSH; IMM 5; MUL; EXIT" \
  --dim MUL_ACCUM --declarations-only
# First divergence at block 0 step=3 pos=3 dim=MUL_ACCUM+4 -- writer
# did not fire; suggested op: _layer0_threshold_attn_dep_anchor
```

`MUL_ACCUM+4` at step=3 (the MUL step) is the byte-0 lo nibble of
the MUL product (oracle `_emit_mul_accum_at_marker`, dim_oracle.py:691).
The oracle expects writer activity here; the **declarations-only**
replay does not localize the writer because `MUL_ACCUM` is written
only by the FlattenedALUMul GE→BD writeback, which is a runtime
`nn.Module` post_op (`make_l12_alu_postop_attach_op`), not a
declarative FFNRule.

For `OUTPUT_LO` / `OUTPUT_HI` / `AX_FULL_HI` the first divergence is
the standard block-3 step-0 `layer3_carry_forward_attn` zero —
not informative (same surface as every program's warmup; see
`docs/LONG_DIVISION_BUG36_2026_06_09.md` §"Oracle replay-diff
localization").

## Why no structural fix lands from this brief

1. **The DSL gap is not the gap.** `wide_mul_rules(width_bytes > 2)`
   intractability (16M+ rules) and W5 multi-byte deferral don't apply
   to a test that goes through `FlattenedALUMul` (the legacy
   composite handling multi-byte natively, per
   `docs/DSL_W5_MULDIV_LIMIT.md` §"Why the legacy composite doesn't
   have this limit").
2. **The actual gap is in the serial autoregressive forward.** The
   batched (full-prefix) forward of the same compiled model emits
   the correct MUL=500 AX bytes. The serial token-by-token forward
   emits AX=0. This is a runner integration issue, not a weight or
   DSL issue. Diagnosing it requires comparing the two forward
   paths' attention masks / KV-cache state / position embeddings at
   the MUL step — outside the wide-MUL DSL surface.
3. **Worktree constraint: do not touch L14/L15 memory or batched
   runner.** The SI/LI 16-bit failure is downstream of L14/L15
   memory plumbing (the SI/LI handlers in `vm_step.py` for word-wide
   stores/loads — TODO(phase-7) comments at `run_vm.py:2356/2370`
   explicitly mark them as legacy fallback while
   `_set_layer14_mem_generation` / `_set_layer15_memory_lookup`
   word-wide bake migrations are pending). The constraint excludes
   the structural fix surface for the SI/LI half.
4. **The MUL half** could in principle be fixed by aligning the
   serial-runner forward path to the batched runner's prefix
   forward, but that touches `AutoregressiveVMRunner.run` /
   `_run_speculative` / `_generate_next_cached` / KV cache state —
   far outside the wide-MUL surface the brief targeted, and
   single-rule fixes at L11-L13 are documented zero-sum per
   `feedback_single_rule_fixes_are_zero_sum.md` (0/5 historical
   net-positive).

## Cross-references

- `docs/DSL_W5_MULDIV_LIMIT.md` — wide-MUL DSL POC + deferral (W5).
- `docs/C6_WIDE_MUL_ZERO_OPERAND_2026_06_07.md` — the partial fix
  landed in commit 83d26f48 (OP_MUL/DIV/MOD guards on L10 SP-marker
  tail rules); same surface as Bug #34 but operand-0 not 16-bit.
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:741-877` —
  `make_efficient_l11_alumul_wrap_op` (DSL wrap).
- `c4_release/neural_vm/efficient_alu_neural.py:1072+` —
  `FlattenedALUMul` 9-stage composite.
- `c4_release/neural_vm/run_vm.py:1068-1588` — serial
  `AutoregressiveVMRunner.run` (non-speculative path).
- `c4_release/neural_vm/batched_pure_neural.py:972` —
  `BatchedPureNeuralRunner.run_batch`.
- Memory note `feedback_single_rule_fixes_are_zero_sum.md` — 0/5
  historical record for L11-L13 / L16 single-rule corrective fixes;
  rationale for documenting rather than patching.

## Status

- `test_mul_overflow` in serial: **deferred** — not a wide-MUL DSL
  gap; serial-runner forward divergence vs batched.
- `test_si_li_16bit_value` in serial **and** batched:
  **deferred** — L14/L15 memory cluster (outside worktree perimeter).
