# Min-flop clever VM memory model at DOOM SCALE — byte-exact + the VRAM floor

2026-08-09. Verifies and measures the min-flop clever VM's memory model at the FULL
Doom heap scale. The min-flop VM reads/writes memory via the **O(1) direct-CAM**
(`examples/clever_honest_attn_realtime.py` `DirectCAMReadHead`; production
`c4_min/direct_cam_batched.py` + `c4_min/selfemu_direct_cam.py`). This doc answers: does
it hold byte-exact at Doom scale, what's the per-access cost, and what's the VRAM /
bandwidth floor.

Harness: `examples/clever_minflop_heap_scale.py` (NEW, off every model build path).
GPU: NVIDIA RTX A5000 (24 GB, ~768 GB/s HBM). Golden **`174ece66`** unchanged (no build
file touched). Numbers MEASURED unless labelled PROJECTED.

Run: `PYTHONPATH=. python examples/clever_minflop_heap_scale.py --verify` (CPU byte-exact),
`--bench --device cuda:0 --iters 200` (GPU cost + VRAM).

---

## The direct-CAM in one line

Each memory read RESOLVES the query address to its exact latest-write-wins store row (a
host-side pointer/dict walk over the write log — `nibble_evict_schedule.resolve_load_rows`,
O(1) `latest.get(addr)`, NOT a GPU softmax over the store) and DIRECT-GATHERS that ONE
row's value into the residual value band via `W_v`/`W_o`. It touches **one row per lane
regardless of S** → S-independent in TIME. This is byte-identical to the reference
softmax1+ALiBi latest-write-wins CAM (the softmax winner IS the latest exact-address row;
unwritten address → the +1 sink → ZFOD 0).

---

## 1. BYTE-EXACT at Doom heap scales (S ∈ {32K, 128K, 262K, 512K})

The O(1) direct gather == the reference O(S) latest-write-wins softmax CAM, **L-inf = 0 at
every S**, for the whole battery (CPU, exactness is S-independent but re-verified at each S
so the claim is measured at 262K/512K, not projected from 8K):

| S | hit L-inf | reconstruct L-inf | miss→ZFOD 0 | latest-write-wins (row/val) | STACK read | FRAMEBUFFER read | verdict |
|---|---|---|---|---|---|---|---|
| 32,768  | 0 | 0 | ✓ | ✓ / ✓ | ✓ | ✓ | **BYTE-EXACT** |
| 131,072 | 0 | 0 | ✓ | ✓ / ✓ | ✓ | ✓ | **BYTE-EXACT** |
| 262,144 | 0 | 0 | ✓ | ✓ / ✓ | ✓ | ✓ | **BYTE-EXACT** |
| 524,288 | 0 | 0 | ✓ | ✓ / ✓ | ✓ | ✓ | **BYTE-EXACT** |

* **hit** — every lane queries a stored address → exact value; nibble reconstruction exact.
* **miss** — never-stored address → 0 (the softmax1 +1 sink), both paths.
* **latest-write-wins** — two writes to the SAME address at different rows; the read returns
  the LATER row's value AND the resolver picks the highest (latest) row index. Byte-exact.
* **STACK / FRAMEBUFFER** — see §3.

---

## 2. Per-read cost + S-INDEPENDENCE in TIME (shared-pool store)

Timed ONE direct-CAM read (gather + nibble unpack + `W_v`/`W_o` value-band write) at batch
B = 65,536, sweeping S over a 16× range, 200 iters (sub-µs cost is launch-jitter-dominated,
so the verdict is a TREND test, not a raw max/min):

| S | µs/lane | ms/batch (B=65536) | peak VRAM | shared-pool store |
|---|---|---|---|---|
| 32,768  | ~0.006 | ~0.4–0.8 | 0.091 GiB | 0.125 MiB |
| 131,072 | ~0.006 | ~0.4–0.8 | 0.092 GiB | 0.500 MiB |
| 262,144 | **0.00577** | ~0.4 | 0.092 GiB | 1.500 MiB |
| 524,288 | ~0.006 | ~0.4–0.8 | 0.093 GiB | 2.500 MiB |

**Read TIME is S-INDEPENDENT (O(1))** — the trend test over the 16× S sweep:

* us/lane vs S **Pearson corr = +0.155** (an O(S) read → ~+1.0; this is ~0).
* cost(max S)/cost(min S) = **1.01×** (an O(S) read over 16× S would be ~16×; this is flat —
  the largest S often times *faster* than the smallest, which is physically impossible for a
  genuine O(S) cost → pure jitter).
* peak VRAM spread over S = **1.02×** (flat — the store touch is one row/lane, not O(S)).

At 262K the per-lane read is **~0.006 µs/lane** and the whole-batch read is **~0.4 ms** for
65,536 lanes. This matches the prior isolated measurements (`clever_honest_attn_realtime.py`:
~0.6% of the composed step, ~0.125–0.254 KB/lane) and the production dispatch profile
(`DOOM_HASH_CAM_2026_08_05.md`: the live-CAM is 0.69 µs/step of FIXED head/FFN machinery with
**no O(S) scan and no gather-over-stores at dispatch** — the resolution is precomputed into a
compact `W_o` scatter-add).

---

## 3. The STACK + FRAMEBUFFER: same address-keyed store, same O(1) read

The Doom program has ONE byte-addressed memory. The zone-alloc HEAP (up to ~262K live
entries), the **SP-addressed STACK** (a contiguous band, grows down from the heap top), and
the **FRAMEBUFFER** (320×200 = 64,000 bytes = 16,000 32-bit words) all live in it. The c4 VM
reads/writes every one of them through the SAME LI/SI/LC/SC/PSH opcodes → the SAME
address-keyed CAM. They are **NOT a separate mechanism** — they are address RANGES inside the
single address-keyed store, held as CAM rows exactly like the heap, and read O(1) identically.

Verified explicitly: a STACK read (address in the ~2²⁴ SP band) and a FRAMEBUFFER read
(address in the fb band, `base + word*4`) both resolve **byte-exact through the same head**
(`stack_value_correct` / `framebuffer_value_correct` = True at every S, table §1). There is no
"framebuffer band" or "stack band" in the residual — the address is the key, and the value's
nibbles are the V. (The framebuffer's 16,000 words are a tiny, fixed sub-range of the 262K
address space; the stack is likewise a bounded live band.)

---

## 4. The BINDING CONSTRAINT: VRAM (dense per-lane heap × batch), NOT read time

The per-STEP GPU work is one gathered row/lane → S-independent in TIME (§2). The binding
constraint is **how the store is HELD in VRAM**. Two regimes:

### DENSE PER-LANE heap — the OOM wall (the thing the direct-CAM exists to avoid)

A literal `(B, S)` int32 store — every lane its own full S-entry heap = **B × S × 4 bytes**.
This is the O(S) materialisation. On a 24 GB card (2 GB headroom for model+activations →
22 GB budget):

| S | per-lane heap | dense OOMs at batch > | B=512 dense | B=4096 dense |
|---|---|---|---|---|
| 32,768  | 0.125 MiB/lane | 180,224 | 0.06 GiB | 0.50 GiB |
| 131,072 | 0.500 MiB/lane | 45,056  | 0.25 GiB | 2.00 GiB |
| **262,144** | **1.000 MiB/lane** | **22,528** | **0.50 GiB** | **4.00 GiB** |
| 524,288 | 2.000 MiB/lane | 11,264  | 1.00 GiB | 8.00 GiB |

MEASURED (closed-form B×S×4 confirmed to the MiB): S=262K B=16384 → **16.029 GiB peak**
(16.0 GiB predicted); S=131K B=16384 → 8.029 GiB; S=524K B=16384 → 32 GiB → PROJECTED-OOM
(exceeds the 24 GB card). **So for the 262K Doom heap, a DENSE per-lane store OOMs a 24 GB
card at batch > ~22,500** (B=512 already needs 0.5 GiB, B=4096 needs 4 GiB).

### SHARED-POOL / EVICTED working set — what actually runs

The per-step marginal GPU work is one gathered row/lane, which a SHARED S-entry pool
(`S × 4 bytes`, **lane-independent**) times identically (`gather_value` supports both forms),
and which the production path holds as a BOUNDED live working set
(`c4_min/qwen_lean_evict.py` / `nibble_evict_schedule`: BOS sink + one register frame +
live-heap rows, evicted to the live footprint — flat VRAM even over millions of steps,
byte-identical to the naive driver in the 45-case eviction battery):

| S | shared-pool / evicted store (TOTAL, lane-independent) |
|---|---|
| 262,144 (Doom heap) | **1.00 MiB** |

The shared pool at 262K is **1 MiB total**, independent of batch — it FITS trivially, and the
gather is byte-identical to the dense form. Peak VRAM was flat at ~0.09 GiB across the entire
16× S sweep at B=65,536 (§2).

---

## VERDICT: the min-flop memory model HOLDS Doom's heap byte-exact; VRAM is the only lever

* **Byte-exact at Doom scale?** YES — L-inf = 0 at S ∈ {32K, 128K, 262K, 512K} for hit /
  miss / latest-write-wins / stack / framebuffer, all through one address-keyed head.
* **Is read TIME the constraint?** NO — the direct-CAM read is O(1) (S-independent: corr
  +0.155, cost ratio 1.01× vs 16× if O(S), VRAM flat 1.02× over the 16× S sweep). ~0.006
  µs/lane at 262K, flat vs S.
* **Is VRAM the constraint?** YES, but ONLY in the DENSE per-lane form: a literal per-lane
  262K heap is 1 MiB/lane → OOMs 24 GB at batch > ~22,500. This is the O(S) materialisation
  the direct-CAM exists to skip.
* **Does eviction fix it?** YES — the production path never holds the dense per-lane heap. It
  holds a SHARED / BOUNDED live working set (BOS + register frame + live-heap rows), 1 MiB
  total at 262K, lane-independent, byte-identical to the dense gather, flat VRAM on long
  programs. The read stays O(1).

### THE MEMORY FLOOR for the min-flop Doom-runner

The floor is the **SHARED / EVICTED live-heap store (~1 MiB total for a 262K heap,
lane-independent) + one gathered row/lane/step (S-independent BANDWIDTH)** — **NOT** the
dense per-lane heap (which would be 262K MiB = 256 GiB at 262K × 262K, or 1 MiB/lane × batch,
the 64 GiB-class wall). The min-flop Doom-runner holds Doom's full heap byte-exact within a
few MiB; VRAM bounds only the naive dense-per-lane materialisation, which eviction / the
shared pool removes. **The read is O(1) in both time and per-step bandwidth; the resident
store is O(live-heap), not O(batch × heap).**

Measured-vs-projected: all §1 byte-exact and §2 cost and the §4 dense measurements
(≤16 GiB) are MEASURED on the A5000; only the 32 GiB (S=524K, B=16384) dense case is
PROJECTED (it physically exceeds the 24 GB card) — and it is projected from the closed-form
B×S×4 that the ≤16 GiB measurements confirmed exactly.
