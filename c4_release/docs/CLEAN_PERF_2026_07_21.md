# CLEAN (uncontended) perf + fast #681/#668 re-verify — 2026-07-21

Branch `combine-fastpath-evict-lea-2026-07-21`.  The GPU (`cuda:0`) was **fully
idle** (0% util, 15 MiB) for this **serialized** run, so the ms/step numbers here
are the honest uncontended figures — the earlier "1.3 s/step" was the ~20x
inflation from 8 concurrent agents sharing one card, NOT the model.

This branch SURGICALLY combines three divergent-lineage branches (a naive
`git merge` deletes work — #669) with BOTH deliverables verified to survive:

* **FAST PATH (base)** — `big-k-spec-worktree-2026-07-20 @ 6b77b63b`
  (`pf_speculative` big-K speculation + OOM-adaptive K + `evict_interval_steps`
  + `bench_fast_path.py`).  Sentinels present: `collect_out` / `prtf_steps` /
  `block_moe` / `_forward_hidden_cached_skip`.
* **EVICTION BATCH** — `evict-fused-work @ d133c4cf` (#670): the per-block
  host-synced `caches[b].evict` loop is replaced by ONE fused on-GPU
  `evict_all_blocks_fused` (batched keep-mask over all 306 blocks) +
  `apply_keep_mask` compaction.  Byte-identical survivor set; de-syncs the
  deep-loop prune so the GPU stays busy.  Merged onto the fast-path cadence
  (`steps_since_evict` / `evict_every`, evict-once-per-block), timing added
  (`t_evict`).
* **#681 FIDELITY** — `fix-668-lea-integer-domain @ b52f2160` (#668/#680): the
  integer-domain LEA decode (`compile_lea_q_reduce` → `LEA_Q`, then
  `compile_lea_addr_nib` decodes the exact-integer `LEA_Q`).  Brought in as the
  LEA_Q block ONLY — NOT the whole file (the fast-path branch is on a lineage
  that lacks the Memory-CAM / `IMM_CLEAN` work; a whole-file copy would drag
  that in / delete fast-path assumptions).  `compile_lea_q_reduce` +
  `compile_lea_addr_nib` (integer-domain) present.

Byte-exactness of the fast path vs the naive path is preserved everywhere below
(`fast output == naive prefix (model==model): True`, `all_matched=N/N`).

Setup for every run: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (needed —
without it the OOM-backoff fragments and can't recover), `--device cuda:0`,
streaming/lean sparse build (~4-6 GB RSS), K capped at ~48-64 (a single verify
forward over K steps × ~35 tokens × 307 blocks is the VRAM wall on a 24 GB card;
K=128 OOMs even with backoff).

---

## 1. The honest uncontended ms/step (naive vs fast)

| program                     | steps | K  | matmul | forwards | naive ms/step | fast ms/step | wall speedup | GPU util | peak VRAM |
|-----------------------------|-------|----|--------|----------|---------------|--------------|--------------|----------|-----------|
| loop n=64                   | 975   | 48 | sparse | 21 (46x) | 349           | **82.0**     | 4.3x         | 91%      | 12.6 GB   |
| loop n=64                   | 975   | 48 | dense  | 21 (46x) | 562           | 102.0        | 5.5x         | 92%      | 12.6 GB   |
| nested 3x100                | 4618  | 64 | dense  | 73 (63x) | 617           | 126.5        | 4.9x         | 92%      | 17.0 GB   |
| nested 6x100 (deep headline)| 9217  | 48 | sparse | 193 (48x)| 427           | **83.9**     | 5.1x         | 90%      | 12.6 GB   |
| malloc_printf (fidelity)    | 227   | —  | dense  | per-step | ~600          | 822 (per-stp)| —            | 33%      | 1.7 GB    |
| matvec `--fixed` (fidelity) | 168   | —  | dense  | per-step | ~600          | 822 (per-stp)| —            | low      | 6.3 GB    |

Naive ms/step is a stable **~350-620 ms/step uncontended** (sparse-CSR ~350,
dense ~560-620) — i.e. **the honest per-step is ~0.5 s, not 1.3 s**; the 1.3 s
was contention.  The fast path amortizes many VM steps into one batched forward:
**46-63x fewer `model.forward` calls** and a **4-5.5x wall-clock speedup** at the
sizes that fit VRAM.

### GPU util before / after
* **Before (naive per-step path):** the per-step cached driver leaves the GPU
  ~33% utilised (see malloc_printf: 33%, 1.7 GB) — one tiny forward per VM step,
  host-bound, the exact "1.3 s/step is insane" problem.
* **After (fast big-K + fused eviction):** the batched verify + on-GPU fused
  eviction run the card at **90-92% mean util** (100% peak) across all loops.

---

## 2. What the amortized win actually is, uncontended (and the honest ceiling)

The fast ms/step decomposes into **forward+overlay** (the batched block forwards
+ O(1) overlay decode) and **eviction** (the fused KV prune):

| program            | fast ms/step | forward+overlay | eviction    | ms/prune | prunes |
|--------------------|--------------|-----------------|-------------|----------|--------|
| loop sparse        | 82.0         | 51.0s (63.8%)   | 29.0s (36%) | 1450     | 20     |
| loop dense         | 102.0        | 70.1s (70.4%)   | 29.4s (30%) | 1470     | 20     |
| nested 3x100 dense | 126.5        | 399.8s (68.4%)  | 184.4s (32%)| 2561     | 72     |
| **nested 6x100 sparse** (headline) | **83.9** | 490.9s (63.5%) | 282.2s (36.5%) | 1470 | 192 |

**The eviction is now the dominant remaining overhead (~30-36% of the fast
wall).**  The #670 fusion made the prune ONE on-GPU op (de-synced, no per-block
host sync — that is why GPU util is 90-92% not 41%), but on a deep loop the prune
STILL costs ~1.5-2.6 s each because it runs an O(S²) near-dup cosine matmul + an
O(S²)-per-pass greedy survivor fixpoint across all 307 blocks.  Note the fused
eviction WORKS as a de-duplicator: on the headline nested 6x100 the cache
collapses to **max_cache=7** (the loop revisits the same frame positions, and the
prune keeps the survivor set near-constant) — VRAM is a flat 12.6 GB over all
9217 steps.  So the honest amortized win is **~5x wall / ~48x fewer forwards** at
**83.9 ms/step** on the deepest program, and the remaining ceiling is the
eviction time-per-prune, not the forward and not the cache size.

### Honest per-program ceiling
* **Deep loops (loop / nested):** the effective ceiling is **eviction-capped**,
  not forward-capped.  Eviction MUST fire per-block (~every 48-64 steps) to keep
  VRAM bounded — evicting every 256 steps (4 blocks) OOMs a 24 GB card because a
  deep loop's cache fills the card in ~4 K-spans.  The pure forward+overlay floor
  is **~50-90 ms/step** (sparse); eviction adds ~30-40 ms/step on top.  The
  10-20 ms/step target is not reachable while the fused-but-still-O(S²) eviction
  fires per block; the real next lever is a cheaper prune (not a cheaper forward).
* **malloc (growing heap) is K-capped:** malloc_printf / matvec run the per-step
  cached driver (not the big-K verify), because a growing-heap / self-emul
  program has no long accepted draft span to batch — the fidelity re-verify is a
  correctness check, so it uses the naive-cached path (~0.8 s/step, 33% GPU).

### dense vs sparse-CSR matmul — use SPARSE
Same loop, same K, same 21 forwards, same cache:

| matmul mode | forward+overlay wall | fast ms/step |
|-------------|----------------------|--------------|
| dense_kernel| 70.1s                | 102.0        |
| **sparse_mm (CSR)** | **51.0s**    | **82.0**     |

**sparse-CSR is ~27% faster** on the block forwards (the compute-rule weights are
genuinely one-hot-sparse), so it is the matmul mode to use.  Eviction cost is
identical between modes (it is independent of the block matmul).

---

## 3. FAST FOUNDATION RE-VERIFY (#681 confirms #668 SOUND)

### matvec `_matmul_run.py --kind=matvec --fixed` → [80, 112] byte-exact
```
matrix-vector A @ x = [[1, 1], [2, 1]] @ [2, 3]: n_instrs=134 ref_steps=168
numpy/reference fixed-point result bytes: [80, 112]
--- neural model.forward PRTF result bytes:     [80, 112]
BYTE-EXACT MATCH: True  (neural == numpy == reference-VM)
run wall: 138.0s  neural_steps=168  per-step=0.82s  peak RSS=6.32 GB
```
The self-emulation matvec is **[80, 112] byte-exact through `model.forward`** —
independently confirming the integer-domain LEA decode (#668) is SOUND (#681).
The old standalone matvec re-verify timed out at 800 s per-step; here it is 138 s
of forward (168 steps) after a ~55 s build — no longer 13 min.  (The a0 byte is
read from BP-2 = addr 236 = 0xEC, adjacent to the 235.5 rounding edge — exactly
the #680 deep-read-back miss the exact-integer `LEA_Q` kills at source.)

### malloc_printf → "byte=72 char=H" byte-exact (first line)
```
malloc_printf model out = 'byte=72 char=H\n\x03'
golden                  = 'byte=72 char=H\nHHH\n'
BYTE-EXACT MATCH (full golden): False
first line 'byte=72 char=H' present: True
```
The **first printf line `byte=72 char=H` is byte-exact** — this is the malloc'd
heap byte `*p` (=72='H') read back through the LEA frame-address decode, which is
exactly what the #648/#680 LEA fix targets, and it is CORRECT.  The second line
`HHH\n` (the `*(p+1)` / `*(p+2)` re-reads) diverges to `\x03` because the FULL
`malloc_printf` golden ALSO needs the **§Memory-CAM role-gate flag-residue fix**
(commits `3fc1f9f9` / `85b56097` / `2bf94d45`), which lives on the lineage the
fast-path base does NOT share and is NOT one of the three branches combined here
(bringing it in touches `blogspec_memory.py` + `nibble_pure_forward.py` — out of
scope; #669 says don't drag in a divergent lineage wholesale).  So the LEA fix's
own deliverable (`byte=72 char=H`) is confirmed; the multi-byte re-read second
line is a separate, un-combined Memory-CAM deliverable.

---

## Surgical-merge integrity (both survive)

| survivor                                | file                             | present |
|-----------------------------------------|----------------------------------|---------|
| `collect_out` / `prtf_steps` / `block_moe` / `_forward_hidden_cached_skip` | `pf_speculative.py` | ✓ |
| `evict_all_blocks_fused` / `apply_keep_mask` | `nibble_pure_forward_cached.py` + `pf_speculative.py` | ✓ |
| `compile_lea_q_reduce` / `compile_lea_addr_nib` (integer-domain `LEA_Q`) | `nibble_pure_forward_complete.py` | ✓ |
| `speculative_run` / `verify_blocks`     | `pf_speculative.py`              | ✓ |
| `selfhost/_matmul_run.py` (matvec re-verify) | `selfhost/`                  | ✓ |

`main` was NOT touched and nothing was merged.
