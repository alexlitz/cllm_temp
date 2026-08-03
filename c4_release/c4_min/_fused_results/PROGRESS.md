# Fused segmented sparse-FFN kernel (#806 follow-up) — progress log
# STATUS: COMPLETE — delta kernel clears 1 min (0.0074 ms/tok, 0.85 min/frame @ K=1024),
#   byte-exact (snapped nibble, doom slice), golden 069cc32f untouched.
#   See FUSED_SPARSE_FFN_RESULTS.md for the full writeup.


GPU: RTX A5000, torch 2.10.0+cu128, triton 3.6.0. CUDA_VISIBLE_DEVICES=0,1.

## Baseline reproduced (this worktree)
- COO-SpMM (reference #806): **0.0321 ms/token** @ K=1024, graph REPLAYS, 3.69 min/frame. Matches #806's 0.0328.
- dense-padded #804: 0.0699 ms/token, 8.03 min/frame.

## Model structure (probed)
- 242 FFN blocks, all share input dim D=1679. Dff in [2..4320], sum=41926, median 25.
- W_up nnz=60812 (~1.45/row), W_gate nnz=41617, W_down nnz=59804. Total 162233.
- b_down ALL ZERO (no down bias). b_up 33833 nz, b_gate 513 nz.
- 242 distinct ffn objects (no recurrent tie in this build).
- 3 live-attn blocks (0,7,11); other 239 are dead-attn -> block = x + FFN_delta(x).

## CRITICAL FINDING — blocks are SEQUENTIAL
- Dependency probe: 241/242 blocks READ a residual dim an EARLIER block WROTE.
- 204 dependency levels; largest parallel wave = only 3 blocks.
- => CANNOT batch all blocks' up-projections into ONE shared-X grid. The residual
  chain is a deep sequential dependency (block i+1 needs block i's output).

## Achievable fusion (given sequentiality)
- Collapse the 3 launches/block (up,gate,down) + silu/transpose glue -> ONE fused
  kernel per block. 726 -> 242 launches, NO [Dff,K] HBM round-trip, no transpose glue.
- The 137us/block is NOT launch overhead (242*5us=1.2ms << 33ms); it's serial kernel
  EXECUTION with poor occupancy (tiny grids underfill 64 SMs) + the silu glue
  (134us on the big block) + 3x the launch/sync barriers.
- Fusing up+gate+silu+down removes: (a) 2/3 of launches, (b) the silu*mul elementwise
  glue kernel, (c) the [Dff,K] read+write to HBM between up/gate and down.

## Plan
1. Build fused per-block CSR kernel: up+gate+silu+down in one launch, one grid over
   K-tiles, hidden kept in SRAM/registers (tile over Dff for big blocks).
2. Verify byte-exact (snapped nibble) vs dense-padded on doom slice + golden untouched.
3. Measure ms/token K=1024/4096 vs 0.0321/0.069; throughput; VRAM; frame projection.

## RESULTS v1 (block_k=128, K=1024) — BIG WIN
| form   | eager ms | graph ms | ms/tok  | vsCOO | vsDense | %peak | frame  |
|--------|---------:|---------:|--------:|------:|--------:|------:|-------:|
| coo    | 38.3     | 32.8     | 0.0320  | 1.02  | 2.18    | 0.67% | 3.68min|
| upgate | 19.9     | 12.6     | 0.0123  | 2.66  | 5.68    | 1.75% | 1.41min|
| full   | 15.2     | 14.9     | 0.0145  | 2.26  | 4.81    | 1.48% | 1.67min|
- Byte-exact per-block: worst rel L-inf 8.3e-7 (up/gate ~exact), all graph-REPLAY.
- upgate (up+gate+silu fused=1 launch + down w/ fused residual=1 launch; 484 total)
  is the winner: 2.66x over COO. Still not <1min (needs 0.0087 ms/tok = 60s/6.89M).
- full (single-launch, recompute hidden) slightly worse: recompute on the dense
  Dff=4320 blocks costs more than it saves in launches.
- NEXT: tune BLOCK_K; hybrid (full for tiny blocks, upgate for big); reduce launches.

## ATTRIBUTION of the upgate 12.3ms (block_k=256, K=1024)
- upgate-silu kernel: 5022 us (242 launches, 20.8 us/launch)
- down kernel:        6545 us (242 launches, 27.0 us/launch) <- BIGGER
- The down kernel writes a FULL [D=1679,K] output per block but W_down has mean
  0.15 nnz/row -> ~99% of the 1679 rows are a PURE RESIDUAL COPY (HBM waste).

## DELTA down kernel — THE FIX (clears 1 min)
- down_delta_inplace: launch a program ONLY for the residual rows W_down writes
  (n_active: Dff=4320 block writes 135 rows not 1679; Dff=288 writes 9). Update the
  residual IN PLACE. Eliminates the [D,K] copy on untouched rows.
- Byte-exact: worst rel L-inf 3.3e-7 (delta form), all graph-REPLAY.

## RESULTS v2 (delta, block_k=512) — CLEARS 1 MINUTE
| form   | K    | graph ms | ms/tok  | vsCOO | vsDense | %peak | frame    | <1min |
|--------|------|---------:|--------:|------:|--------:|------:|---------:|:-----:|
| coo    | 1024 | 33.5     | 0.0328  | 1.00  | 2.13    | 0.67% | 3.76 min | NO    |
| upgate | 1024 | 12.7     | 0.0124  | 2.65  | 5.65    | 1.77% | 1.42 min | NO    |
| DELTA  | 1024 |  7.7     | 0.0075  | 4.37  | 9.31    | 2.93% | 0.86 min | YES   |
| coo    | 4096 | 170.5    | 0.0416  | 1.00  | 1.73    | 1.97% | 4.78 min | NO    |
| upgate | 4096 | 84.8     | 0.0207  | 2.01  | 3.48    | 3.96% | 2.38 min | NO    |
| DELTA  | 4096 | 65.1     | 0.0159  | 2.63  | 4.53    | 5.17% | 1.82 min | NO    |
- hybrid (full-for-small + upgate) LOSES vs upgate: recompute cost > launch savings
  (GPU not purely launch-bound; extra arithmetic hurts). Delta is the winner.
- K=1024 is the operating point: 0.0075 ms/tok -> 51.7 s/frame < 60 s. CLEARS.
- TODO: byte-exact end-to-end vs dense-padded (doom slice) + golden 069cc32f untouched.

## BYTE-EXACTNESS (end-to-end, snapped-nibble) — CONFIRMED
- 9 real C programs run through the VM (add/sub/mul/div/mod/cmp/var/func + a 42-step
  doomslice arithmetic chain 320*200 /16 %7 + compare + branch).
- REF (unmodified sparse_mm FFN) vs FUSED (delta kernel): EVERY program's full per-step
  AX trace is IDENTICAL (trace_eq=True). doomslice: both decode 67997, 42 steps.
- Model state hash UNCHANGED before/after fused swap (f1cc6a90...) -> kernels only READ
  stored W_up/W_gate/W_down; no stored weight modified. GOLDEN 069cc32f UNTOUCHED.
- The W_down fp accum-order residual (~6e-2 raw) is absorbed by the integer nibble-snap,
  exactly as the task specifies ("match the SNAPPED nibble, not raw fp").


