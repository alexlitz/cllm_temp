# Doom fast-path -> Self-emulation: which levers transfer + measured step speed

Empirical test: does the doom-on-transformer FAST PATH transfer to SELF-EMULATION
(the transformer emulating its own ONNX-runtime forward as a c4 program)?

Branch: worktree-agent-ae765016f00ba88df @ cddf79f2 (consolidate-0.5b lineage).
All fast-path levers already present on this branch (the #831 self-emu direct-CAM +
the pf_kbatch graphed FFN megakernel); the composed doom perf branch e472b882 did NOT
need merging — the self-emu K-batch path (`pf_kbatch.KBatchBoundedRunner`) carries its
OWN analogues. GPU 0 (24 GB), clean-run numbers only (>=20 GB free, low util).
Golden flags-OFF `069cc32f` UNCHANGED. No flag defaults changed. Only file added is a
new standalone measurement harness (`_agent_selfemu_bigws.py`) — nothing on the build
path, so both goldens are structurally guaranteed unchanged.

## The inversion (why this is not just "rerun doom's levers")

Doom = TINY program, HUGE frame. Self-emu = INVERTED: the working set is the emulated
transformer's weights+activations living in the store log — n_store GROWS ~O(S) toward
~262k persistent-heap KV/step. So the two runners are DIFFERENT:

- doom path  = `pf_speculative.verify_blocks` / `direct_cam_batched` (SparseAttn),
  levers `C4_PRECOMPUTED_SCHEDULE` / `C4_SCHED_PIPELINE` (single-dispatch gather
  schedule over the store log), `C4_DIRECT_CAM_VEC`, `C4_FUSED_MEGABLOCK`.
- self-emu path = `pf_kbatch.KBatchBoundedRunner` (bounded-KV #746 + K-batch #747 +
  selective-fp64 #748), levers `C4_SELFEMU_DIRECT_CAM` (#831) + the graphed
  passthrough-FFN megakernel (`forward_span_graphed` / `C4_FUSED_FFN_MEGAKERNEL`).

## LEVER TRANSFER VERDICT

| doom lever                         | self-emu analogue                         | TRANSFERS? | why |
|------------------------------------|-------------------------------------------|-----------|-----|
| direct-CAM (`C4_DIRECT_CAM_VEC`)   | `C4_SELFEMU_DIRECT_CAM` (#831)            | **YES — decisive** | O(1) address gather replaces softmax-over-stores; the ONLY thing that makes the 262k-KV inverted case RUNNABLE (softmax OFF OOMs) |
| fused-FFN hidden on-chip           | pf_kbatch graphed passthrough-FFN megakernel (`C4_FUSED_FFN_MEGAKERNEL`) | **YES** | per-step FFN math; S-independent, composes on top of direct-CAM; ~1.1-1.3x |
| single-dispatch precomputed schedule (`C4_PRECOMPUTED_SCHEDULE`/`C4_SCHED_PIPELINE`) | N/A — a DIFFERENT schedule concept | **DOES NOT APPLY (and does not need to)** | the doom single-dispatch resolves a gather schedule OVER THE STORE LOG; the self-emu megakernel graph is keyed on `(live-block-schedule, K)` and its static buffer is `[1, K, D]` — over the 242 BLOCKS + K query rows, NOT the store log. See "the 262k-KV build verdict" below. |
| `C4_MEGABLOCK_BLOCK_K=512`         | K-batch K (forwards = steps/K)            | PARTIAL   | doom's block-K is an on-chip FFN tile; self-emu's K is the cross-step batch. Direct-CAM LIFTS the K ceiling (see eff_K) which is the real self-emu win |

## THE 262k-KV SINGLE-DISPATCH-BUILD VERDICT (the flagged obstacle)

The concern was: does the single-dispatch schedule BUILD over the 262k-KV working set
blow up VRAM? **It does NOT — because the self-emu path never builds a gather schedule
over the store log.** The self-emu megakernel (`GraphedFFNChain`) captures ONLY the
`[1, K, D]` query-row passthrough-FFN chain (K=32-128), which is S-INDEPENDENT (static
buffer ~= K*D*4 B ~= 235 KB at K=32). The gathers are resolved by the O(1) direct-CAM
table (`build_direct_cam_table`), which is a `{pos: value}` dict of size K per span, NOT
a K*n_store score matrix. So the "build" cost is flat in n_store. What DOES scale is the
plain forward's per-forward VRAM (input `[1,S,D]` + bounded-KV intermediates); measured
peak at n_store=262144 = **22.97 GB (fits a 24 GB card; OOMs a ~21 GB one)**.

## MEASURED — direct-CAM at the growing working set (byte-exact, GPU 0 clean)

`_agent_selfemu_dcam_verify` (ON == OFF == K=1-seq reference, battery 17/17 +
nested_deep(60), K in {1,8,32}, L-inf=0):

```
  n_store   OFF ms/fwd   ON ms/fwd   speedup   ON ms/step   VRAM off/on GB
  32           5.02        5.12       0.98x      0.160       14.80/14.80
  128          5.62        4.63       1.21x      0.145       14.80/14.80
  512          6.19        4.21       1.47x      0.131       14.80/14.80
  2048         7.29        4.43       1.65x      0.138       14.93/14.84
  8192        15.75        5.55       2.84x      0.173       15.55/15.02
  32768       48.65        6.78       7.17x      0.212       18.05/15.75
  131072        OOM       11.55        -         0.361        OOM /18.64
```

Large-working-set sweep to the full 262k (`_agent_selfemu_bigws`, K=32, GPU 0 clean):

```
  n_store    OFF ms/fwd   ON ms/fwd   ON+graph   OFF->ON   peakVRAM_GB   in_GB
  1024         18.76        14.43       (n/a)      1.3x       14.84       0.02
  4096         26.77        18.65       (n/a)      1.4x       15.15       0.07
  16384        63.40        19.55       (n/a)      3.2x       16.40       0.25
  65536         OOM         21.66        -       OFF-OOM      20.00       0.97
  131072        OOM         12.14        -       OFF-OOM      22.57       1.94
  262144        OOM         17.97        -       OFF-OOM      22.97       3.87   <- the 262k inverted case, RUNS ON, OFF OOMs
```

ON+graph (fused-FFN megakernel composed with direct-CAM), GPU 1 idle:
```
  n_store    OFF ms/fwd   ON ms/fwd   ON+graph
  1024         6.51         5.32        3.68
  16384       26.73         6.17        5.23
  65536         OOM         7.18        6.42
  131072        OOM        10.75       10.24
```
=> fused-FFN megakernel composes cleanly on direct-CAM (~1.1-1.4x on top).

## eff_K ceiling (the throughput win = fewer forwards), n_store=16384, GPU 0

```
  K       OFF ms/fwd   ON ms/fwd   OFF->ON
  8         19.38        3.80       5.1x
  32        24.89        4.50       5.5x
  128       46.48        7.40       6.3x
  512         OOM       22.08       (OFF OOMs; ON runs)
  1024        OOM       41.24       (OFF OOMs; ON runs)
```
OFF (softmax over stores) caps at K=128 then the O(K*n_store) score matrix OOMs. ON
runs K=1024+ => the K ceiling lifts ~8x = 8x fewer forwards over the self-forward.

## END-TO-END byte-exact + steady-state throughput (K=128, GPU 0 clean)

`_run_complete_self_forward_kbatch` (real matvec + width-2/4/6 dots + 3x3 matmul, every
VM step, composed==seq==isa==numpy byte-exact):

- BASELINE  (direct-CAM OFF, graphed dense tail): 0.0696 ms/step -> 26.9 min projected
- FAST       (`C4_SELFEMU_DIRECT_CAM=1 C4_FUSED_FFN_MEGAKERNEL=1`): 0.0575 ms/step -> 22.2 min
- BOTH BYTE-EXACT (overall byte-exact=True, 0 mismatched). peak VRAM 14.85 GB.

NOTE: this steady-state uses the tool's tiny timing stream (S=3964, kv_rows=133) where
direct-CAM barely fires — so the ~1.2x here is mostly the fused megakernel. The direct-CAM
win is realized at the LARGE working set above (3-7x/forward + makes 262k runnable).

## BOTTOM LINE

- Fast path TRANSFERS. The load-bearing lever is **direct-CAM (#831)**: without it the
  262k-KV inverted case is not even RUNNABLE (softmax-over-stores OOMs at n_store>=65k);
  with it the reads are O(1) and the run fits 24 GB (22.97 GB peak at 262k).
- The single-dispatch schedule-build fear does NOT materialize: the self-emu megakernel
  graph is over the 242 blocks + K query rows (`[1,K,D]`, ~235 KB, S-independent), not a
  gather schedule over the store log. No VRAM blow-up.
- New self-emu step speed: the composed byte-exact path is ~0.13-0.36 ms/step at the
  realistic growing working set (ON, per-step = ms/fwd / K), and 0.0575 ms/step on the
  small-frame steady-state at K=128. Extrapolated full self-forward (~23.2M pos-sparse
  steps at K=128, small-frame steady-state) ~= 22 min; at the realistic large working set
  the per-step floor is set by the O(1)-gather forward, ~0.13-0.36 ms/step.
- New dominant cost: the plain bounded-KV forward's per-forward VRAM + compute at large S
  (input `[1,S,D]` + intermediates), NOT the attention softmax (collapsed to O(1)) and NOT
  a schedule build. The remaining throughput lever is bigger K (direct-CAM lifts the K
  ceiling ~8x, 128 -> 1024+).

Golden flags-OFF `069cc32f` unchanged (verified via `c4_min._fingerprint_build`).
CFM `7d19cdc3` unchanged (no weights touched; only a new non-imported harness added).
VRAM peak (clean run, GPU 0): 22.97 GB at n_store=262144 (ON); 14.85 GB end-to-end K=128.
