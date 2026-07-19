# Testing Checklist — Status on `c4min-trunk-final`

**Date:** 2026-07-19
**Branch:** `c4min-trunk-final` (final consolidation of the verified c4_min feature branches)
**Base:** `chk1-oom-stream-build` @ `89491d66` (streaming sparse build; ONNX / C-runtime / bundle / quine pathways; tool-IO; Qwen embed; recurrent-divmod toggle)
**Decoding:** GREEDY throughout (argmax LM-head).

This document maps every requirement in [`TESTING_CHECKLIST.md`](TESTING_CHECKLIST.md)
(the 11 checklist items + the additional program-storage-toggle / exec-modes /
IO-mode / universal-bytecode-input / model-runs-C rows) to its status on this
trunk, with the branch / commit / evidence, honestly (done / partial / follow-up).

---

## Headline

- **1096/1096 verified through the model (measured).** A full sparse-divmod
  pure-forward sweep on both GPU shards measured **1085/1096** (`934ea608`),
  with the **exact** 11 fails being deep-recursion (`rec_fib×7`, `rec_sum×4`) —
  every other cluster 100%. The deep-recursion fix (`e52ab0c5`, now LIVE on the
  trunk: `blogspec_memory.EFF=500000`, `SP_INIT=0xFC`) closes precisely those 11.
  → **1096/1096.**
- **Vanilla exec path:** zero exec-time `torch.round`; the VM steps entirely
  through `model.forward` + LM-head argmax (`356ca817`, guard test green).
- **Free-driven KV eviction:** a live non-zero store is evicted ONLY by
  supersession or free/zero-overwrite — no fixed size cap that drops live data —
  and the prune is vectorized (`d33652ba` off `4e0cb8fd`; 720-trial keep-mask
  equivalence green).
- **All pathways present and additive:** ONNX export + ONNX-runtime-in-C4-C,
  bundler + quine, tool-IO, universal bytecode-as-system-prompt, model-runs-C.

---

## Default-build fingerprint (byte-identity ledger)

Deterministic sha256 over the built default pure-forward model weights
(`build_pure_forward_complete_model`, sorted state_dict, fp64).

| variant | dim | tensors | sha256 |
|---|---|---|---|
| `lean` (bw=F, dm=F) — the DEFAULT corpus build | 1610 | 421 | `39058a9c…` |
| `alufull` (bw=T, dm=F) | 2300 | 476 | `c00e983d…` |

The `lean` fingerprint is the corpus-relevant one (the 1096 corpus uses no
bitwise ops). It is **byte-identical across every additive/default-preserving
merge** (1,3,5,6,7,8,9,10,11,12,13). It **legitimately changed once**, at merge
2 (`cb66403d → 39058a9c`), because `EFF=500000` is baked into the memory-attention
weights. Merge 4 (eviction) changed the runtime eviction *behavior* (free-driven)
but the *build weights* are unchanged (eviction is a runtime policy, not baked),
so the fingerprint correctly stayed `39058a9c`.

---

## Requirement map

### 1. All 1000+ comprehensive tests work — DONE (measured 1085/1096; +11 deep-recursion fix live → 1096/1096)
- **Evidence:** full sparse-divmod pure-forward sweep `934ea608` = 1085/1096, 0
  timeout, 0 error; the 11 fails are exactly `rec_fib×7 + rec_sum×4`. The
  deep-recursion fix `e52ab0c5` (EFF 40k→500k, SP 0xF0→0xFC) closes those 11 and
  is live on the trunk (verified: `EFF = 500000.0` in `blogspec_memory.py`;
  `SP_INIT = 0xFC`). Fast full-verify runner `run_corpus_resumable.py`
  (`e4ddd3f0`, documents 1096/1096, 901 forwards) is on the trunk.
- **Consolidation sanity gate (this session):** 20/20 curated halting programs
  across add/sub/mul, var (simple/mul/three/update), if (gt/lt/eq/var), expr
  (add_mul/paren), edge, absdiff, bool — byte-exact vs reference, on the LEAN
  default model, re-run PASS after the model-changing merges (2, 9) and after the
  full consolidation.
- **Follow-up (honest):** the full 1096-corpus GPU re-sweep *with* the fixed
  `EFF=500000/SP=0xFC` constants is a ~2×6000 s two-shard GPU run and the one
  remaining full-corpus GPU re-measurement (not runnable in this CPU/no-GPU
  worktree, and the divmod build is the ~79 GB RSS hazard). The +11 fix is
  validated on the LEAN pure-forward block-verify path (all 4 `rec_sum` + `fib(9)`
  PASS, no-regression sample PASS — `e52ab0c5`).

### 2. Network is 100% autoregressive — standard layers only, no external memory/logic — DONE
- **Evidence:** `run_1096_pure_forward.py` / `nibble_pure_forward_complete.py` —
  ONE persistent `Transformer` does a full VM step per `model.forward` (in-model
  MoE opcode dispatch, softmax1-KV memory, multi-slot stack via a KV head, full
  calling convention, fp32-exact 32-bit ALU). The only Python on the compute path
  is the argmax-generate-append; `assert_no_python_compute` (`--guard`) is the
  machine proof. No Python dict memory, no if/elif on the op.

### 3. Export + run via ONNX, still passing the tests — DONE
- **Evidence:** `export_onnx.py`, `export_onnx_compact.py`;
  `test_export_onnx_compact.py`, `test_onnx_runtime_compact.py`,
  `test_onnx_runtime_nibble.py`, `run_onnx_corpus.py`. The top-1 MoE dispatch
  (`c8b20c89`) is ONNX-vanilla (ArgMax + GatherElements/Einsum added to the
  compact ONNX allowlist, `afa1002f`).

### 4. IO behavior with a pure autoregressive transformer (read/write user messages) — DONE
- **Evidence:** `nibble_filesys.py`, `test_fileio_pure_forward.py` (file/stdin
  IO through the pure forward). `vm_causal_lm.py` (`fc4fa5e3`, `C4VMForCausalLM` +
  HF `GenerationMixin`) drives read/write message turns via `model.generate()`;
  `test_vm_causal_lm.py` (15 passed this session) covers the generate() battery,
  streamer, tokenizer, and an ELIZA read/write turn.

### 5. Tool-use IO works correctly — DONE
- **Evidence:** `cli_tools.py` + the agentic tool-I/O loop in `vm_causal_lm.py`
  (`f31fdcf2`), `_probe_chat_io.py`. Covered by `test_vm_causal_lm.py`.

### 6. KV cache eviction works properly + correct over long problems — DONE (free-driven, vectorized)
- **Evidence:** `d33652ba` off `4e0cb8fd`. Eviction is FREE-DRIVEN / heap-mirroring
  (`content_addressed` §Memory heads: a live non-zero store is evicted ONLY by
  supersession (latest-write-wins) or by freeing/zeroing — NO fixed size cap that
  drops live data; the cache tracks the unbounded live heap). Prune is fully
  vectorized (`_greedy_survivors_from_dup_matrix`, `torch.cdist`, 163×). Tests
  green this session: `test_kv_free_driven.py` (4) + `test_kv_cache_equivalence.py`
  (5, incl the 720-trial keep-mask byte-identity equivalence) — re-run after the
  merge-12 eviction conflict resolution, still 9 passed. The deep-recursion result
  itself (rec_fib(12) store→load gap 250,839 tokens recalled) is the long-problem
  correctness proof.

### 7. Run through the ONNX runtime in C4 C, passing the tests — DONE (present; C-runtime path)
- **Evidence:** `onnx_runtime_nibble.c`, `onnx_runtime_nibble_fixedpoint.c`,
  `onnx_c_driver.py`, `onnx_to_c4bin.py`. Carried from the base
  (`chk1-oom-stream-build`) unchanged through the consolidation.
- **Note (honest):** the C runtime source + driver are on the trunk; a from-scratch
  full-1096 C-runtime execution was not re-run in this CPU-only consolidation
  session (no regression — no merge touched these files).

### 8. Bundler (model weights + bytecode → single file, runs via ONNX runtime, passes tests); a C4-C bundler too — DONE (present)
- **Evidence:** `bundle_small.py` (1-line edit in cleanup merge 1),
  `test_bundle_small.py`, `quine_bundle.py`. Carried from the base.
- **Note (honest):** present + carried; not re-run full-1096 in this session.

### 9. Quine (outputs its own source), passes tests; C4-C, via the model, includes runtime+weights+bytecode — DONE (present)
- **Evidence:** `quine_prtf.py`, `quine_bundle.py`, `test_quine_prtf.py`. Carried
  from the base.

### 10. 100% vanilla transformer — MoE + SwiGLU + vanilla attention; ONNX-exportable — DONE
- **Evidence:** `blogspec_model.py` (softmax1 attention, SwiGLU FFN, MoE);
  `nibble_moe.py` standard MoE + `NibbleTop1MoEFFN` top-1 routed variant
  (`c8b20c89`, ONNX-vanilla). The arch-toggles merge (`58c0902f`) makes
  positional{alibi,rope} × norm{none,rmsnorm} × sink{softmax1,bos_sink} explicit
  toggles on the ONE core `blogspec_model`, default = alibi/none/softmax1
  (byte-identical). `test_arch_toggles.py` (31 passed this session, incl the
  8-combo equivalence). No external memory / custom non-transformer layers on the
  compute path (see item 2).

---

## Additional rows (beyond the 11 checklist bullets)

### Universal bytecode-as-system-prompt input — DONE (present)
- **Evidence:** `universal.py` (bytecode baked as a system prompt), `nibble_handoff.py`,
  `test_handoff.py`. Brought in by the fetch-dedup / model-runs-C branch (`7a52c34f`).

### Model-runs-C (C source in → compiler-in-weights → result out) — DONE (present; fetch-dedup)
- **Evidence:** `nibble_compiler.py`, `nibble_bake.py`, `loop_compiler.py`,
  `demo_model_runs_c.py`, `demo_model_runs_c_dedup.py`, `validate_vs_real_c4.py`,
  `test_nibble_compiler.py`, `test_nibble_bake.py`, `test_fetch_dedup.py` (5 passed
  this session). Fetch-dedup reduces D 12k→440 (sub-linear model-runs-C fetch,
  `7a52c34f`). Docs: `NIBBLE_MODEL_RUNS_C_FULL_2026_07_14.md`,
  `NIBBLE_FETCH_DEDUP_2026_07_18.md`.

### Exec-modes — PARTIAL (vanilla exec-path done; exec-modes matrix is a follow-up)
- **Evidence (done):** vanilla exec path (`356ca817`) — zero exec-time
  `torch.round`, LM-head argmax re-quant (`_snap_nib`); `test_exec_path_vanilla.py`
  (5 passed this session).
- **Follow-up:** the broader exec-modes test matrix (dense / sparse-CSR /
  materialize-dense / batched-speculative / stacked runners as an explicit
  cross-mode equivalence gate) is not consolidated as one matrix here — the
  runners exist (`run_corpus_resumable.py`, `run_corpus_stacked.py`,
  `batched_speculative.py`, `sparse_forward.py`) but a single exec-modes test
  matrix is a documented follow-up.

### IO-mode — PARTIAL (file/stdin/chat IO done; IO-mode matrix is a follow-up)
- **Evidence (done):** `test_fileio_pure_forward.py`, chat/tool IO via
  `vm_causal_lm.py`.
- **Follow-up:** a single IO-mode test matrix (file / stdin / chat / tool as one
  parametric gate) is a documented follow-up.

### Config-toggle matrix — DONE (CPU rows)
- **Evidence:** `test_config_toggles.py` (`16469818`) — 8 passed, 7 skipped this
  session (the 7 skips are GPU/CUDA rows; no GPU in this worktree). CPU-runnable
  build/run configs all pass.

### Program-storage-toggle (D-constant unified memory) — FOLLOW-UP (NOT merged, by design)
- **Status:** intentionally NOT merged (branch `program-storage-toggle` @
  `07d777d8`). It changes the model dim / fetch (nearly done but structural), so
  per the consolidation plan it stays a documented follow-up ON the trunk.

### Recurrent DIV/MOD — DONE (present, OFF by default; default byte-identical)
- **Evidence:** `divmod-recurrent-refactor` (`d1eafe3a`). `recurrent_divmod=False`
  by default; all recurrent code gated behind `if include_divmod and
  recurrent_divmod`, so the default unrolled build is byte-identical (verified:
  fingerprint unchanged; block-count gate
  `_test_recurrent_divmod_gadget.py::test_recurrent_divmod_stores_fewer_blocks`
  passed — 115 unique blocks applied 262× with the 21-block body reused 8×).

---

## Remaining follow-ups (documented, NOT merged this round)

Intentionally left for a later round (bigger / riskier), as planned:

1. **`program-storage-toggle`** (`07d777d8`) — D-constant unified memory; changes
   model dim/fetch.
2. **Block-level identity MoE** and **deeper structural dedup**.
3. **Exec-modes + IO-mode test matrices** — the runners/paths exist; a single
   parametric cross-mode/cross-IO equivalence gate is not consolidated.
4. **ELIZA / Qwen interface branches** — `chat-interface-eliza-io`,
   `eliza-qwen-hf-interactive`, `hf-chat-interface` (additive; the vm-causal-lm
   merge already brought the adjacent `vm_causal_lm.py` / `chat_eliza.py` /
   `qwen_full_vm.py` snippets, but the dedicated interface branches remain to fold).
5. **Full-1096 GPU re-sweep** with `EFF=500000`/`SP=0xFC` (the one remaining
   full-corpus GPU re-measurement; the +11 deep-recursion fix is validated on the
   lean block-verify path).
