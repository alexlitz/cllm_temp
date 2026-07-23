# SHIFT-AS-POSITIONAL-ATTENTION bench — reuse the CAM, or keep the mux gadget?

*Does realising a 32-bit shift's coarse whole-nibble step as a **positional-CAM
head** (reusing the machine's existing address-matching attention) beat the
bespoke **FFN mux-tree** the shifter currently uses?*

Code: [`shift_attention_bench.py`](shift_attention_bench.py). Run the table with

```
OMP_NUM_THREADS=2 PYTHONPATH=$(pwd) python -m c4_min.shift_attention_bench
```

It reuses (edits nothing): the nibble mux-tree baseline + peel from
[`shifter_bakeoff.py`](shifter_bakeoff.py), the fp32-exact SwiGLU emitters
(`_empty_spec` / `_mul_gate` / `_floor_div_pow` / RELU_S) from
[`nibble_alu32.py`](nibble_alu32.py), the slow-RoPE address-CAM addressing pattern
from [`qwen_full_vm.py`](qwen_full_vm.py) (`_rope_lane_pair`, `_bake_memory_cam`),
the nibbles-as-tokens layout of [`qwen_vanilla_vm.py`](qwen_vanilla_vm.py), and the
sub-nibble shift math of [`nibble_bitwise.py`](nibble_bitwise.py). Verification is
LEAN — arithmetic sim on the touched bands / token positions (plus a *real* softmax
forward of the CAM head), not a DIM-8192 run.

## The insight (and where it holds)

A shift is `out[j] = in[j − shift]` — a **positional READ**. In the vanilla
nibbles-as-tokens layout a register's nibbles are already separate tokens at known
offsets, so a whole-nibble left-shift by `c = n // 4` is *exactly* "output nibble
`j` attends to the input nibble at position `j − c`". Inject `c` into the query's
slow-RoPE address (the same near-identity address lane `_bake_memory_cam` keys on)
and one head does the coarse shift with **no mux weights** — just q/k/v/o.

So we decompose a bit-shift by `n` into

```
COARSE  =  n // 4  whole-nibble shift   ->  a positional-CAM HEAD
FINE    =  n mod 4  bits  (0..3)         ->  a small per-nibble FFN (nib*2**r + peel)
```

and measure the head's own q/k/v/o cost **fairly**, against the all-FFN mux-tree.

## Results

*(depth = FFN blocks + attention heads; nz = non-zero params; attn vs ffn split for
the attention variant; fp32-exact; byte-exact over the edge grid `x ∈ {0x80000000,
0xFFFFFFFF, 0xDEADBEEF, 0x1, 0x0, 0xF0F0F0F0}`, `n ∈ {0,1,7,15,16,31,32,40}` +
random, vs `ref_interpret(mask=0xFFFFFFFF)` (32-bit) + `isa.interpret` (8-bit
low-byte). All three variants are all-exact.)*

### SHL

| variant | depth | attn nz | ffn nz | total nz | fp32 | byte-exact | isa8 |
|---|---|---:|---:|---:|:---:|:---:|:---:|
| **shift-as-attention** | 1 head + 6 blk | 19 | 3 616 | **3 635** | yes | 198/198 | 40/40 |
| leaner-mux | 8 blk | 0 | 4 089 | 4 089 | yes | 198/198 | 40/40 |
| nibble mux-tree *(baseline)* | 8 blk | 0 | 4 140 | 4 140 | yes | 198/198 | 40/40 |

### SHR

| variant | depth | attn nz | ffn nz | total nz | fp32 | byte-exact | isa8 |
|---|---|---:|---:|---:|:---:|:---:|:---:|
| **shift-as-attention** | 1 head + 6 blk | 19 | 3 559 | **3 578** | yes | 198/198 | 40/40 |
| leaner-mux | 8 blk | 0 | 3 920 | 3 920 | yes | 198/198 | 40/40 |
| nibble mux-tree *(baseline)* | 8 blk | 0 | 3 971 | 3 971 | yes | 198/198 | 40/40 |

The CAM head itself is a real, sink-clean positional gather: a separate **real
softmax forward** (RoPE on the slow lanes, BOS sink at logit 0, per-bit position
agreement + an in-range gate) gathers the correct nibble for **48/48** (pop × shift)
cases per direction, including out-of-range = shifted-in-zero.

## Where the 4,140 actually goes — the reframing that decides it

The prompt's premise was "~480 two-way muxes × ~10 nz". That is the *bit-granular*
log-shifter's cost; in the **nibble** mux-tree the coarse mux is already tiny. The
honest per-phase breakdown of the baseline (SHL) is:

| phase | nz | share |
|---|---:|---:|
| amount_decode (n//4, 2**r, keep, rnz) | 2 708 | 65% |
| **coarse mux (the log-shift tree)** | **308** | **7%** |
| fine_product (nib·2**r) | 72 | 2% |
| fine_peel (carry staircase, kmax=7) | 968 | 23% |
| assemble | 84 | 2% |
| **total** | **4 140** | 100% |

**The coarse mux tree is only 308 nz (7%).** The weight lives in the
**amount-decode (65%)** — turning the scalar `n` into `n//4`, `2**r`, `keep`,
`rnz` via per-value one-hot staircases over `n = 0..63` — and the **fine peel
(23%)**.

### What the attention variant replaces vs keeps (SHL)

| piece | nz | note |
|---|---:|---|
| attn head (q/k/v/o) | 19 | REPLACES the coarse mux (308) **and** the coarse bit-extract (`amount2`, 768) |
| coarse-address FFN | 683 | the coarse amount-decode did NOT vanish — it MOVED here (compute `n//4`, write each output token's `j±c` query address + in-range flag) |
| fine FFN | 2 933 | UNCHANGED and unavoidable: `2**r` decode + `keep`/`rnz` + `nib·2**r` + the peel + assemble |
| **total** | **3 635** | 88% of baseline (a 12% cut) |

## Verdict

**Attention beats the mux-tree, but only by ~12% (4 140 → 3 635 nz, SHL; 3 971 →
3 578, SHR), NOT the order-of-magnitude the "1 head vs 480 muxes" framing
suggests** — and only under the vanilla nibbles-as-tokens layout.

Why the win is small, stated plainly:

1. **The head's own cost is genuinely cheap (19 nz).** A single positional-CAM head
   really does the whole coarse gather, and 48/48 under real softmax. Caveat (a)
   from the brief — "count the head's q/k/v/o fairly" — resolves *in the head's
   favour*: 19 nz is far less than the 308-nz coarse mux + 768-nz coarse
   bit-extract it replaces. **The head is not a wash.**

2. **But the coarse mux was never the expensive part.** It is 7% of the shifter.
   Replacing a 308-nz gadget with a 19-nz head can only ever save ~300 nz on the
   coarse path.

3. **The amount-decode does not disappear — it moves.** The coarse shift still needs
   `c = n // 4` and, for a positional read, that `c` must be turned into each output
   token's query address (`j ± c`) + an in-range flag — a 683-nz FFN. So the coarse
   *mechanism* goes 1 076 nz (mux 308 + `amount2` 768) → 702 nz (head 19 + address
   683): a real ~35% cut on the coarse path, but the coarse path is small.

4. **The fine shift is 100% shared and dominates the remainder.** `2**r`, the
   `nib·2**r` product, the `keep`/`rnz` fold, the fine peel and assemble (~2 933 nz)
   are identical in both designs. Attention does nothing for them, so ~80% of the
   shifter is untouched.

**Leaner-mux barely moves (4 140 → 4 089, −51 nz; SHR 3 971 → 3 920).** Trimming
each 2:1 coarse mux from ~4 units to the minimal single-guarded-delta (`out = same +
n_bit·(neigh − same)`) is byte-identical to the baseline and saves ~50 nz *because
the coarse mux is only 308 nz to begin with*. The "each mux ~10 nz vs ~3 ideal"
overhead the brief flagged is a bit-granular-shifter artifact; at nibble granularity
the mux lowering is already near-minimal.

### Honest caveats surfaced (all three the brief asked for)

* **(a) head cost counted fairly — and it wins.** 19 nz < the 308+768 it replaces.
  The head is *not* as expensive as the muxes; if anything it is dramatically
  cheaper. The reason attention doesn't win *big* is not the head, it is that the
  coarse path it optimises is only ~26% of the shifter.
* **(b) vanilla-path only.** Shift-as-attention needs the nibbles-as-tokens layout
  (`qwen_vanilla_vm`): a register's nibbles must be separate TOKENS for "attend to
  the nibble `c` positions away" to be meaningful. It is **not** a drop-in for the
  residual-band shifter (`nibble_bitwise`, where all 8 nibbles live in ONE token's
  residual — there is no position axis to attend along). On the residual-band path
  the mux-tree (or leaner-mux) is the only option.
* **(c) does the hybrid genuinely beat all-FFN once everything is counted?** Yes,
  by 12% — but the win comes entirely from deleting the coarse mux + coarse
  bit-extract, and it is bounded by the fact that the fine shift (which attention
  cannot help) plus the fine amount-decode is the bulk of the cost. If the coarse
  address FFN could SHARE its `n//4` staircase with the fine block's `n`-decode (not
  done here — counted separately for honesty), the attention total would drop a bit
  further, but the fine peel (968 nz) is a hard floor for any nibble shifter.

### Bottom line

Reusing the CAM for the coarse shift is a **modest, real, layout-constrained win**
(~12% fewer weights, one shared amount-agnostic head, byte- and fp32-exact) —
worth taking *on the vanilla path* where the head is free infrastructure the machine
already has. It is **not** the dramatic "delete 480 muxes" win the shape of the
problem first suggests, because in the nibble representation the mux tree was never
the expensive part: the amount-decode and the fine sub-nibble peel are, and those
survive in every design.
