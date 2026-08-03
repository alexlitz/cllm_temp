# Self-emu O(1) memory (direct-CAM) — CHECKPOINT

Task: generalize Doom's O(1) direct-CAM + banded local attention to the SELF-EMULATED
c4 VM bounded-KV path so it stops doing O(S²)/O(n_store) softmax over the growing store set.

## STATE OF THE ART (already landed before this task)
- The 296-step self-emu matmul blocker is ALREADY FIXED: `C4_LEA_Q_SNAP` (commit 6aad41c3).
  Root was NOT address-CAM aliasing — it was a LEA_Q immediate-fetch sub-permille residue
  amplified 64x. The address CAM was proven CORRECT (ZFOD-returns 0 for wrong addr).
- Bounded-KV (#746 `pos_sparse_bounded.BoundedPosSparseRunner`): attention is bounded to
  (local window W) + (STORE rows ≤ q). FLAT in S for the LOCAL heads.
- K-batching (#747 `pf_kbatch.KBatchBoundedRunner`): forwards = steps/K. Projected ~0.29hr.
- direct_cam_batched.py: direct-CAM EXISTS but is wired ONLY into `pf_speculative.verify_blocks`
  (SparseAttn.forward) = the DOOM path. NOT into the self-emu Bounded/KBatch path.

## THE REAL RESIDUAL (the gap this task fills)
`pos_sparse_bounded.BoundedBlock._store_rows` / `pf_kbatch.KBatchBoundedBlock._store_layout`:
the GLOBAL CAM heads STILL do `softmax1` over ALL store rows ≤ q_idx. For self-emu the
emulated VM writes memory heavily -> n_store GROWS ~O(S). That growing softmax IS the
"92x-wasted attention" (only 1 store row is the winner; the rest are dead softmax mass).

`_apply_direct_cam` in bench_pf_kbatch.py ALREADY resolves reads to exact values — but it
patches the DECODED STATE post-forward (correctness overlay), NOT the attention. The model
forward still scores all store rows.

## THE FIX (this task)
Wire `resolve_load_rows` (address -> exact store row, O(stores+reads)) into the
Bounded/KBatch global CAM heads: at each query row, DIRECT-GATHER the resolved store row's V
vector (reconstruct nibbles), skip the softmax over store rows. Gated `C4_SELFEMU_DIRECT_CAM`
default OFF -> byte-identical. Also route local ingest heads through banded_local_attn.

## Key files
- c4_min/pos_sparse_bounded.py  (BoundedBlock: `_store_rows`, `_attend`, `forward`)
- c4_min/pf_kbatch.py            (KBatchBoundedBlock: `_store_layout`, `_attend_dense`, forward)
- c4_min/bench_pf_kbatch.py      (drive_kbatch, drive_seq_bounded, _apply_direct_cam, _make_timing_stream)
- c4_min/nibble_evict_schedule.py (resolve_load_rows, ResolvedRead)
- c4_min/direct_cam_batched.py   (the DOOM direct-CAM reference: _head_out_vec, cam_head_map)
- c4_min/_run_complete_self_forward_kbatch.py (the end-to-end self-emu runner)

## GOLDEN GATE
c4_min `_fingerprint_build` golden = 069cc32f (family b). Verify unchanged flags-OFF.

## MEMORY DISCIPLINE
LEAN streaming build only (build_compact_sparse_streaming, ~1.5GB). NEVER load_sparse_transformer.
Check MemAvailable > 25GB before + during. CUDA_VISIBLE_DEVICES=0,1.

## PROGRESS LOG
- (start) diagnosis complete; building the direct-CAM-in-bounded-block fix next.
- DONE: selfemu_direct_cam.py built (DirectCamTable, block_global_kinds, vectorized
  gather_global_ctx). Wired into KBatchBoundedBlock (global heads: O(1) gather when armed)
  + KBatchBoundedRunner.arm_direct_cam + drive_kbatch (arms when C4_SELFEMU_DIRECT_CAM=1).
- BYTE-EXACT: battery 17/17 + nested_deep(60) OK, K=1/8/32, ON==OFF==seq reference.
- END-TO-END: _run_complete_self_forward_kbatch with flag ON = byte-exact (matvec 512/512
  every step, dot/matmul all intermediates); 2761 VM steps to completion, all byte-exact
  (far past the historical step-296 divergence, which was already fixed by C4_LEA_Q_SNAP).
- MEASURED (cuda:1, A5000-class, K=32) store-row-growth attention OFF (softmax over
  stores, O(n_store)) vs ON (O(1) gather):
    n_store   OFF ms/fwd   ON ms/fwd   speedup
    32          4.64         3.94        1.18x
    512         5.12         3.90        1.31x
    8192       14.69         4.17        3.52x
    32768      48.21         5.35        9.01x
    131072      OOM          9.60        (OFF OOMs; ON runs)  <- O(n_store) VRAM ceiling gone
  OFF grows O(n_store) then OOMs; ON stays ~flat. The 92x-wasted global-CAM softmax is
  COLLAPSED to O(1). ON ms/step 0.12-0.30ms even at 131072 stores.
- GOLDEN unchanged flags-OFF: default 069cc32f, CFM 7d19cdc3 (both byte-identical).

## eff_K CEILING (the real self-emu wall cut), cuda:1, n_store=16384 working set:
    K       OFF ms/fwd    ON ms/fwd    OFF->ON
    8         19.36         3.87        5.0x
    32        25.94         4.54        5.7x
    128       46.64         7.52        6.2x
    512        OOM         23.00        (OFF OOMs; ON runs)
    1024       OOM         40.78        (OFF OOMs; ON runs)
  OFF (softmax over stores) caps at K=128 then the O(K*n_store) score matrix OOMs.
  ON (O(1) gather) runs K=1024+ -> eff_K ceiling lifted 8x = 8x fewer forwards +
  5-6x faster per forward where both run. 23.2M-step self-forward: K=128->1024 = 8x
  fewer forwards.

## CONTINUOUS STRESS: nested(5,6) 200-step store/load-heavy loop = byte-exact
  (ON==draft==OFF), no intermediate PC/SP divergence.

## PRE-EXISTING (out of scope): the --graph (CUDA-graph megakernel) path FAILS with
  the flag OFF too (vanilla bz_taken/bnz_taken/jmp/jsr_lev/si_li) -- a CUDA-graph
  capture issue on this GPU/torch, independent of direct-CAM. The EAGER path (default,
  production) is fully byte-exact both ON and OFF.

## STATUS: DONE. Byte-exact eager path, golden unchanged, measured 5-9x + eff_K 8x.
