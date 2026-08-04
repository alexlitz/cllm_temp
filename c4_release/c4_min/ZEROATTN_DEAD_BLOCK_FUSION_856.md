# #856 — ZERO-ATTENTION-COMPUTE via dead-block-fusion (composed doom forward)

Resume of #856 (throttle-killed). Model-side, lean streaming, isolated worktree.
Golden byte-neutral (fingerprint unchanged with/without the change).

## THE AUDIT (where the residual attention arithmetic lived)

Composed doom forward = direct-CAM (#834) + direct-local (#843) + banded-local +
frozen-skip + live-local (#845), on the `verify_blocks` batched path.

`_agent_attn_audit.py` instrumented `softmax1` / banded einsum per block on a 402-step
malloc verify (all_matched=True). Finding:

- Only **4 blocks have live attention**: blk0 `ingest+recompose` (20 ingest-local
  heads), blk2 `code-select` (1 CAM), blk7 `mem-cam` (1 CAM), blk11 `stack-pop-cam`
  (2 CAM). These are ALREADY fully lookup-resolved by direct-local / direct-CAM —
  ZERO score/softmax on them.
- The residual attention ARITHMETIC (33 GFLOP / 1666 softmax1 calls, 100% of it) was
  on the **~238 DEAD-attention blocks** (`pc-fetch`, `opcode-decode`, `alu-*`, the
  divmod span, …). `install_local_attention` WINDOWS those blocks' zero-value local
  heads and routes them through the banded/masked softmax — but never SKIPS them. Each
  dead block pays a full O(S·W) banded score+softmax+ctx over heads whose output is
  provably 0. Pure waste.

Root cause: the `install_dead_block_fusion` / `install_live_head_attention` levers
(`live_head_attention.py`, #845-era) that bypass a dead block's whole attention
sublayer (output = x, no Q/K/V/W_o, no softmax, no KV — proven L-inf=0) were NEVER
wired into the `verify_blocks` composition.

## THE SHIM (the fix)

`pf_speculative.py`: new gated lever `C4_DEAD_BLOCK_FUSION` (default OFF,
`_dead_block_fusion_enabled()`), installed in `verify_blocks` BEFORE direct-CAM /
direct-local:

    if _dead_block_fusion_enabled():
        install_live_head_attention(model)   # score only live head-slots
        install_dead_block_fusion(model)      # 0-live-head blocks -> output x, no attn

The 4 live blocks carry live heads so they are never fused; direct-CAM / direct-local
then install their own lookup forward on top. Net: the composed step's attention
compute -> EXACTLY ZERO (FFN + O(1) gathers).

## RESULTS

BYTE-EXACT + ZERO-ATTN-COMPUTE (`_agent_zeroattn_verify.py`, K=64, 4 programs
loop_countdown/nested_call/malloc_heap/malloc_free):

    prog             base_match lookup_match fuse_match out==base sm(fuse) band(fuse)
    ALL 4:              True         True        True      True       0         0     OK
    => ALL BYTE-EXACT + ZERO-ATTN-COMPUTE

Per-step re-embed path byte-exact (`_agent_zeroattn_perstep_verify.py`): L-inf(base
vs fused) = 0.000e+00 at S=91/200/350. Fuses 238/242 dead blocks, live-head slots
24/5808.

#846 FLOP GAUGE (`_flop_gauge.py --dead-fusion`, S=301), attention FLOP collapse:

| component        | WITHOUT fusion | WITH fusion | change |
|------------------|---------------:|------------:|-------:|
| attn_score_ctx   |     111.487 G  |    2.312 G  |  48x lower |
| TOTAL exec FLOP  |     111.574 G  |    2.400 G  |  46x lower |
| DIV ms/step      |      170.9 ms  |     49.2 ms |  3.5x |
| DIV 400K %peak   |    136,576 %   |      812 %  |  168x closer |

(The residual 2.312 G attention in the gauge is the CAM blocks scoring their live
heads over full S — on the VERIFY path those go to zero too via direct-CAM, proven
softmax1=0 above. The gauge's per-step re-embed path has no direct-CAM.)

STEP TIME (`_agent_zeroattn_bench.py`, verify_blocks, malloc K=256, all_matched=True,
clean GPU): lookup-no-fuse 19.59 ms/step (51 steps/s) -> lookup+DEAD_FUSION 4.19
ms/step (239 steps/s) = **4.67x**. (An earlier 1.33x measurement was under GPU
contention from an orphan process; 4.67x is the clean number.) malloc has a growing
heap so the CAM-gather cost grows; the attention-COMPUTE is gone (zero), the residual
4.19 ms is FFN + the growing-CAM O(1) gather. Fixed-S doom-render is lighter.

## GOLDEN

`_fingerprint_build.py` = 7d19cdc3 WITH and WITHOUT the change (byte-neutral; the
change is a runtime forward-orchestration flag in pf_speculative.py, default OFF, no
weight write). The documented 069cc32f is a build-config/torch-version baseline delta
present on the base too — the authoritative gate (unchanged-relative-to-base) passes.

## FILES

- `c4_min/pf_speculative.py` — the lever (`_dead_block_fusion_enabled` + install in
  `verify_blocks`). ONLY production change.
- `c4_min/_flop_gauge.py` — `--dead-fusion` option + `_attn_flops` fused/live-mask aware.
- `c4_min/_agent_attn_audit.py` — the audit tool.
- `c4_min/_agent_zeroattn_verify.py` — byte-exact + zero-attn battery.
- `c4_min/_agent_zeroattn_perstep_verify.py` — per-step L-inf=0 check.
- `c4_min/_agent_zeroattn_bench.py` — ms/step timing.

## FEEDS

#851 (megakernel) — the op-class graph now has NO attention score/softmax on dead
blocks (static identity), so more of the block stack is a fixed-shape graph.
#855 (FFN-only) — with attention compute at zero, the composed step IS the FFN-only
rate + O(1) gathers.

## HONEST / NOT-DONE

- Every live head IS draft-resolvable: the 4 live blocks are register-ingest
  (content-addressed frame roles, direct-local) + memory/stack/LEV/code CAM
  (address-resolved, direct-CAM). No content-addressed-blend head remains on the doom
  path (#853's self-emu content-addressed case is a DIFFERENT program, not doom).
- The step is not yet 1.0 ms on malloc because of the growing-heap CAM gather + the
  FFN; the ATTENTION-compute lever (this task) is fully realized (zero). The remaining
  distance to 1s/frame is FFN + gather + the divmod span (#855 / #851 territory).
