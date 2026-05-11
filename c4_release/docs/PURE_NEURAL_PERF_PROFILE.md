# Pure Neural Performance Profile

Date: 2026-05-11
Branch: `h-pure-neural-profile`
GPU: CUDA (single-GPU run on a quiet device)
Test profiled: `tests/test_pure_neural_pc.py::TestPureNeuralSingleInstruction::test_imm_then_exit`
Program: `[IMM 5, EXIT]` — the simplest pure_neural test.

## TL;DR

`test_imm_then_exit` runs in ~30 s. **84 % of that time (~25 s) is spent inside one FFN block — `block_18` (`FlattenedDivMod`) — even though the program never executes DIV/MOD.** The next-biggest chunks are `block_19` (`FlattenedALUMul`, ~1.9 %) and the embedding layer (~4.2 %, dominated by repeated `_add_code_addr_keys` and `_inject_mem_exec_autoregressive` work that recomputes per call). Everything else is in the noise.

A single optimization — make `LongDivisionModule` only run when the active opcode is DIV/MOD (it already has a per-opcode "skip" affordance via `opcode_w[:, 0, OP_DIV/MOD]`, but the per-element Python-loop body runs regardless) — would cut per-test runtime by **~5×** for non-DIV/MOD tests, which is the vast majority of Phase 1 tests.

## How the test is structured

```
test_imm_then_exit:
  bytecode = [IMM 5, EXIT]
  runner.run(bc, max_steps=30)
```

In pure_neural mode the loop generates one token per `model.generate_next` call and breaks at `EXIT`. For this program:

| Metric | Value |
| --- | --- |
| VM steps executed (logical) | 2 (`IMM`, `EXIT`) |
| `model.forward` calls | **70**  (= 2 steps × 35 tokens/step) |
| First-call context length | 20 tokens |
| Last-call context length | 89 tokens |
| Total e2e wall time | **30.12 s** |
| Total forward-pass time | 30.05 s  (99.8 % of e2e) |
| Avg per-forward latency | **429 ms** |
| Python-side overhead (runner loop, dispatch) | ~70 ms (0.2 %) |

The Python runner loop is **not** the bottleneck. The forward pass is.

### Per-forward latency vs context length

| ctx_len bucket | n | avg ms | total ms |
| --- | --- | --- | --- |
|  20 | 10 | 422.5 |  4225 |
|  30 | 10 | 423.1 |  4231 |
|  40 | 10 | 413.4 |  4134 |
|  50 | 10 | 430.6 |  4306 |
|  60 | 10 | 430.2 |  4302 |
|  70 | 10 | 452.8 |  4528 |
|  80 | 10 | 432.5 |  4325 |

Latency is **almost flat across context length** — it grows from ~420 ms at ctx=20 to ~450 ms at ctx=80. That confirms the cost is dominated by per-block compute that does not scale with sequence length (i.e. the DivMod and Mul ALUs), not attention.

## Per-block breakdown

Captured via `forward_pre_hook` + `forward_hook` on `model.embed`, every `TransformerBlock`, and `model.head`. (CUDA-synchronized before/after each measurement.)

### Top blocks by total time (70 calls, 26 blocks)

| Rank | Block | Component | Total (ms) | Avg ms / call | % of forward |
| ---:|---|---|---:|---:|---:|
| 1 | `block_18` | **FlattenedDivMod** | 26 030 | 371.86 | **84.2 %** |
|   | `block_18.ffn` |   | 25 982 | 371.18 | 84.1 % |
| 2 | `block_19` | FlattenedALUMul | 577 | 8.25 | 1.9 % |
|   | `block_19.ffn` |   | 515 | 7.36 | 1.7 % |
| 3 | `block_09` | AddSub5StageBlock | 414 | 5.92 | 1.3 % |
| 4 | `block_11` | ALUAndOrXor | 409 | 5.85 | 1.3 % |
| 5 | `block_21` | ALUShiftComposite | 398 | 5.68 | 1.3 % |
| 6 | `embed` | NeuralVMEmbedding | 1 299 | 18.56 | 4.2 % |
| 7..26 | all remaining 21 blocks | (PureFFN + small post_ops) | ~1 700 *combined* | ~1.2 each | ~5.5 % |
| —  | `head` (Linear→vocab) | | 33.6 | 0.48 | 0.1 % |

### Key takeaway

The compiled model has 26 sequential blocks. ALU blocks 18 (DivMod) and 19 (Mul) together account for **86 %** of every forward call, but the test program touches **neither** of them. Every other ALU block (Add/Sub, And/Or/Xor, Shift, Compare, etc.) is ≤ 1.3 % individually because they are tiny right-sized FFNs; only DivMod and Mul are still implemented as **Python-loop nibble-pipeline modules**.

## Hottest Python-level functions (cProfile, ranked by `tottime`)

```
ncalls   tottime  cumtime  function
  16800    9.825   15.419  alu/ops/divmod_longdiv.py:172  _compare_le         (LongDivisionModule)
  17920    9.433   10.621  alu/ops/divmod_longdiv.py:121  _trial_multiply     (LongDivisionModule)
 313040    2.442    2.442  Tensor.to                       (dtype churn inside divmod)
 302680    2.291    2.291  torch.rsub                      (1.0 - x in _compare_le)
    350    0.994    1.264  efficient_alu_neural.py:81      _ALUInstrBlock.forward (Mul shared)
 161280    0.905    0.905  torch.floor                     (carry-resolve in _trial_multiply)
     70    0.801    0.833  neural_embedding.py:105         _add_code_addr_keys
   1120    0.593    0.702  alu/ops/divmod_longdiv.py:207   _subtract           (LongDivisionModule)
  57960    0.463    0.463  torch.zeros                     (per-call buffer allocs)
  17430    0.397    0.397  torch._C._nn.linear             (true matmuls — tiny share!)
```

`_compare_le` (9.8 s) + `_trial_multiply` (9.4 s) + `_subtract` (0.6 s) + their tensor-churn (`Tensor.to` 2.4 s, `rsub` 2.3 s, `floor` 0.9 s) ≈ **25.4 s of the 30 s total**. That matches the block-hook number exactly (`block_18.ffn` = 25.98 s).

Note `aten::_C._nn.linear` (real GEMMs) is just **0.4 s** — *all* the actual matrix multiplies across all 26 blocks combined are <1.5 % of runtime. **This is not a GEMM-bound workload; it is a Python-loop / elementwise-kernel-launch-bound workload.**

## Hottest CUDA kernels (`torch.profiler`)

```
                Self CUDA   % of CUDA  # of Calls
aten::copy_      877 ms     19.1 %      558 992
aten::sub        851 ms     18.5 %      643 930
aten::mul        828 ms     18.0 %      659 470
aten::add        718 ms     15.6 %      534 170
aten::gt         244 ms      5.3 %      181 300
aten::lt         220 ms      4.8 %      161 420
aten::floor      209 ms      4.6 %      161 280
aten::div        203 ms      4.4 %      163 100
aten::mm         118 ms      2.6 %       10 220   ← actual GEMMs
aten::addmm       78 ms      1.7 %        7 210   ← linear bias adds
aten::bmm         42 ms      0.9 %        3 640
```

- Self-CUDA time totals **4.59 s** (versus 30 s wall) — i.e. the GPU is **idle ~85 % of the time** while the CPU dispatches micro-ops.
- The `copy_/sub/mul/add` quartet has been called **~2.4 million times** in a 30 s run. Each tiny ([B=1, 9] or [B=1] nibble) op launches its own CUDA kernel — pure launch overhead, ~1-2 µs per kernel.
- **Real GEMMs (`mm + addmm + bmm`) total ~240 ms.** If we removed every other op, the model would run in well under a second per test.

This is the classic "Python-loop on the GPU" anti-pattern: 8 outer division iters × 15 trial values × 9 nibble loops × N forward calls = ~75 000 Python iterations per test, each spawning a handful of CUDA kernels on [B=1, 9]-shaped tensors.

## Where the time goes (estimates)

| Component | est. ms / forward | est. % | Notes |
| ---:|---:|---:|--- |
| `LongDivisionModule.forward` (block_18) | ~371 | **~86 %** | 8×15 nibble trial-multiply + compare-le + subtract loops |
| `FlattenedALUMul` (block_19) | ~7.4 | ~1.7 % | similar elementwise nibble structure, but smaller (one stage, not 8×15) |
| `embed` (`_add_code_addr_keys`, `_inject_mem_exec_autoregressive`, RoPE work) | ~19 | ~4.3 % | recomputes per-call work that could be cached/incremental |
| All 25 other blocks combined | ~30 | ~6 % | mostly small dense ALU FFNs + 1 attention each |
| Python runner loop + dispatch | <1 | ~0.2 % | not the issue |
| **Total** | **~429** | **100 %** | matches measured avg |

## Optimization recommendations

### 1. (Biggest win) Gate `LongDivisionModule.forward` on active opcode

Block 18 currently runs the 8-outer × 15-inner nibble long-division Python loop on **every forward call regardless of opcode**, then zeroes the contribution at the end via `opcode_w` (the OP_DIV/OP_MOD one-hot). Cost: ~371 ms/call, dominates the test. The mask is applied **after** the work is done, so all 25 s of compute is wasted on this test (and every non-DIV/MOD test).

**Fix options, in order of preference:**
- **Short-circuit early-out in `forward`:** check `opcode_w.max().item() < 0.5` (i.e. neither DIV nor MOD active) and return `x` unmodified. Estimated saving: ~371 ms/call → ~5× speedup for almost every Phase 1 test. (Caveat: `.item()` is a CPU/GPU sync, which costs a few µs; still vastly cheaper than the loop. Or use `opcode_w.any()` with `if opcode_w.any():` plus careful tensor handling.)
- **Vectorize the trial-multiply over k=1..15 instead of looping in Python:** if early-out is not acceptable for whatever pipeline-purity reason, the 15 trial multipliers can be batched in a single `[B, 15, 9]` op, and the per-nibble `_compare_le` and `_subtract` loops (`for j in range(9):`) can be replaced with vectorized prefix-product / cumulative comparisons. This would cut even the worst-case (DIV/MOD) cost by ~10×.
- **Lookup-mode for div/mod:** the original `DivModModule` had a `mode='lookup'` path (see `vm_step.py:813 _init_lookup_mode`). On modern GPUs a 2¹⁶ entry LUT is trivial; flipping the alu_mode default for tests where speed matters more than the precision audit would also work.

File: `c4_release/neural_vm/alu/ops/divmod_longdiv.py:121-297`.

### 2. Same treatment for `FlattenedALUMul` (block_19)

Smaller absolute share (~7 ms/call), but again the work runs unconditionally despite IMM not needing multiply. The kernel-launch count dominates here too — `_ALUInstrBlock.forward` shows up at 350 calls × 70 forwards but each ALU op breakdown loop creates ~30 tiny kernels. Same fix as above: early-out when `opcode != MUL`, or vectorize the byte-loop. Estimated extra saving: ~5–7 ms/call.

File: `c4_release/neural_vm/efficient_alu_neural.py:81` (`_ALUInstrBlock.forward`) and `efficient_alu_neural.py:881` (`FlattenedALUMul`).

### 3. Cache per-context-prefix embedding work

`_add_code_addr_keys` (0.80 s tottime, 70 calls, 11.4 ms each) and `_inject_mem_exec_autoregressive` (0.31 s, 70 calls, 5.3 ms each) both run **fully on each `generate_next` call** even though the bytecode prefix never changes within a single `run()` invocation. Cache the prefix-side augmentations once at the start of `run()` and reuse. Estimated saving: ~13 ms/call → ~3 % of e2e (small in absolute terms because of #1, but free).

Files: `c4_release/neural_vm/neural_embedding.py:105`, `:358`.

### 4. Use KV-cache in pure_neural mode

`generate_next` is called with `use_incremental=False` from `run_vm.py:354`, forcing the model to re-process the entire context on every token. With a (modest, single-step) KV cache, attention/FFN K and V tensors for the immutable code+data prefix could be reused. This won't matter much for short tests (the per-call avg is already nearly flat at ~430 ms across ctx 20→80, because the per-step cost is mostly **opcode-independent constant work in DivMod**), but once recommendation #1 lands and per-call time drops to ~60 ms, KV cache becomes the next bottleneck.

Files: `c4_release/neural_vm/run_vm.py:354`, `c4_release/neural_vm/vm_step.py:1307`.

### 5. Reduce tiny-tensor allocations inside the divmod loops

Even if we keep the long-division algorithm:
- `_trial_multiply`, `_compare_le`, `_subtract` each allocate a fresh `torch.zeros(B, 9)` on **every** call (15 × 8 × 70 = 8400 allocations of a 9-element tensor per test).
- `torch.full((B,), float(k), ...)` allocates a 1-element tensor 15 times per outer iteration (~75 000 per test).
- `(d < -0.5).to(dtype)` triggers a temporary plus a dtype cast each time.

Reusable pre-allocated buffers (per-module) and `dtype=`-aware kernel writes would eliminate ~470 ms of `aten::zeros` and ~440 ms of `aten::full` shown in the CUDA op profile. Small per-fix, additive after #1.

### Quick wins summary

| # | Change | Est. saving / test | Effort |
| ---|---|---:|---|
| 1 | Early-out `LongDivisionModule.forward` for non-DIV/MOD opcodes | **~25 s** (84 %) | 1 line + opcode check |
| 2 | Early-out `FlattenedALUMul.forward` for non-MUL | ~0.5 s (1.7 %) | 1 line |
| 3 | Cache embedding prefix work across steps in `run()` | ~0.9 s (3 %) | a few lines in `embed.forward` |
| 4 | Enable KV cache in pure_neural | small now, **~2-3×** after #1 | medium |
| 5 | Pre-allocate `_compare_le/_subtract/_trial_multiply` buffers | ~1 s (3 %) | small |

After #1 + #2 + #3, the same test should drop from ~30 s to roughly **3–4 s** (one-time build + 70 forward calls at ~40 ms each instead of 430 ms).

## Reproduction

```bash
cd /tmp/h-profile/c4_release
CUDA_VISIBLE_DEVICES=$PICK_GPU python scripts/profile_pure_neural.py
```

The profile script writes a structured copy of the timing breakdown to `docs/_profile_data.json` and prints all four views above (per-block, cProfile cumtime, cProfile tottime, torch.profiler op table). Total run is ~2 min on a quiet GPU.
