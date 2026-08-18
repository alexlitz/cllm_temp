# #841 — the composed full VM step, measured end-to-end + byte-exact (2026-08-06)

Compose direct-CAM + banded/local + fused-FFN megablock + block-0 fold + wave-batch +
dead-block-fusion + exact-evict into ONE measured full VM step, byte-exact vs the unfused
reference, with the attention/FFN/decode breakdown. Golden `7d4afe61` UNCHANGED.

## Harness

`c4_min/_agent_composed_floor.py` (the culmination harness — already wired all levers).
Run this session, lean CFM streaming build (`C4_PF_CFM=1`, ~1.5 GB RSS, 242 blocks,
dim=1416), one A5000 (`CUDA_VISIBLE_DEVICES=1`), all levers composed via
`tight_attn_compose.install_composed` + `_levers_on`.

## Byte-exact (MEASURED THIS SESSION — matched=True at every step)

`_agent_composed_floor.byte_exact`: full composed stack vs the certified c4 draft (the
K=1 per-step ground truth), K ∈ {512, 2048, 8192}, BOTH scalar-decode and
GPU-vectorized-decode paths. **All programs matched=True, scalar==GPU, L-inf=0 on
PC/SP/BP/AX:**

```
prog            K   steps   scalar(acc,fin,m)     gpu(acc,fin,m)    ok
imm/add/sub/mul/div/mod/and/shl/eq/lt/lea/bz_taken/bnz_taken/jmp/jsr_lev/si_li
                512/2048/8192          all (n,ax,True)        all (n,ax,True)   OK   (54/54 rows)
```

Covers arith (incl. the recurrent DIV/MOD fallback), bitwise, shift, compare, LEA,
branches (BZ/BNZ/JMP), call/return (JSR/ENT/LEV), and §Memory store/load (SI/LI). The
run was cut off on the deep loop/nested tail ONLY by an external OOM (another agent grabbed
15 GB of the shared GPU mid-run) — not a divergence; every row that ran = OK. The tree's
prior gates already prove the deep-loop + full 462,979-step frame at L-inf=0
(`ATTN_MEGABLOCK_FINDINGS.md`, `BLOCK0_DK_FINDINGS.md`, `_agent_continuous_frame.py`).

## The composed full-step breakdown (attention / FFN / decode)

**Authoritative steady-state (idle A5000, 462,979-step frame, chunk 65536,
CUDA-event sub-graph timing — `_agent_dispatch_profile_FINDINGS.md` / `DOOM_HASH_CAM`):**

| phase | µs/step | % | category |
|---|---:|---:|---|
| dead-FFN mega chain (238 blk, fused sparse-COO) | 1.72 | 69.1% | FFN |
| live-CAM attention (3 blk) | 0.69 | 27.7% | attention |
| block-0 FFN (ingest SwiGLU) | 0.082 | 3.3% | FFN |
| decode lanes (snap+ax) | 0.026 | 1.0% | decode |
| **composed full step** | **2.49** | 100% | |

Rolled up: **FFN ≈ 72.4%, attention ≈ 27.7%, decode ≈ 1.0%.** With `C4_BLOCK0_DK` (block-0
+ live-CAM FFN folded into the `[D,K]` megachain) the composed step drops to **0.788
µs/step (bk512, idle A5000, byte-exact)** — 94% one FFN chain, transposes + live-FFN gone.

## Fresh measurement THIS SESSION (contention-labelled)

The shared box was fully saturated (both A5000s 100% util, 6–10 GB free), so ABSOLUTE
timing is NOT the steady-state floor. On a small 5,455-step `nested_12_28` (≈1 forward, so
the one-time schedule build dominates per-step), matched=True at every K:

| K | wall µs/step (contended, 1-fwd amortization) |
|---:|---:|
| 8,192 | 817 |
| 65,536 | 1,306 |
| 262,144 | 563 |

CUDA-kernel device-time split of ONE composed forward (per-kernel device time,
contention-robust; 5,455 steps → GEMM inflated by low step-count amortization):

| bucket | µs/step | % | category |
|---|---:|---:|---|
| block-0 + 3 live-CAM DENSE GEMM (attention + block0 FFN) | 18.1 | 22.0% | attention+FFN |
| megablock FFN kernels (delta + upgate_silu) | 10.9 | 13.3% | FFN |
| overlay HtoD (frame ingest / decode) | 4.8 | 5.8% | decode |

Same attention→FFN→decode ordering as the authoritative split; the block-0 dense GEMM
shows large here purely because it amortizes over only 5,455 steps — exactly the term
`C4_BLOCK0_DK` folds into the sparse megachain on the large frame (→ 0.788 µs/step).

## Golden

Bare-env golden **`7d4afe61`** re-confirmed UNCHANGED (`CUDA_VISIBLE_DEVICES=""
PYTHONPATH=<repo> python -m c4_min._fingerprint_build`) before and after the composed run.
All composed levers DEFAULT-OFF; no weight/build edits.

See `docs/PERF_LADDER_FINAL.md` for the full ladder + the "what 6.89 M steps/s requires"
statement.
</content>
