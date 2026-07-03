# L14 SI/LI Consumer Diagnostic — 2026-06-09

## Setup

* Base: c07c1236 "fix(qwen-export): pad attention to max num_heads for non-uniform layers"
* Producer side (A3.10 OP_IMM discriminator, commit f6ebf6e7) verified working — the L10
  `psh_ax_broadcast` head correctly populates `STACK0_BYTE_VAL_h_LO/HI` at all PSH STACK0 byte-h
  rows (probe_a3_5_l14_consumer: `S0@80 BI1 VAL1_LO=2/3.00` for PSH of `IMM 0x200`).
* Failing test: `TestSmokeMemory::test_si_li_roundtrip` — expects 42, returns 512.

## Program & expected vs actual

Bytecode (constructed by `_make_bytecode`):

```
IMM 0x200    # AX <- 0x200
PSH          # push 0x200 onto stack (address sink for SI/LI)
IMM 42       # AX <- 42
SI           # *(top of stack) = AX = 42
IMM 0x200    # AX <- 0x200 (address for LI)
LI           # AX <- *(AX) = *0x200
EXIT         # return AX
```

`512 == 0x200` is exactly the address used as the SI/LI target — the LI is loading the **address byte**
back as if it were the value byte, not the freshly stored 42.

## Probe results

Probe: `c4_release/tools/probe_l14_li_consumer.py` (new).

Position layout (seq_len=236):

* `OP_PSH active rows: [100]`
* `OP_SI  active rows: [168..172]`
* `OP_LI  active rows: [224]`
* `MARK_AX positions: [65, 100, 133, 168, 203, 224]` — one per instruction (IMM,PSH,IMM,SI,IMM,LI)
* `MARK_STACK0 positions: [80, 114, 148, 183, 218]`
* `MARK_MEM positions: [85, 118, 153, 188]` — one MEM frame per stack-touching op
  * `m@118`  PSH frame: `MEM_STORE=2` at marker.
  * `m@188`  SI frame:  `MEM_STORE=2` at marker; `ADDR_B0_LO[0]=1.79`, `ADDR_B0_HI[0]=2.24`
    (i.e. address byte 0 = 0x00, address byte 1 = 0x02 → 0x200).

## Divergence layer — physical block 12

Layer-by-layer trace of `OUTPUT_LO[0..15]` at `p=193` (SI MEM frame, val byte 1, which should be 0x00):

```
after_L11: [-0.00 -0.00 -0.00 ... -0.00]                      # all near zero — clean
after_L12: [ 0.71 -0.00  1.29 -0.00  0.00  0.00  0.00  ... 0.00]   # << injected here
after_L13: [ 0.71 -0.00  1.29 -0.00 ...]                      # passthrough
after_L26: [ 0.71 -0.00  1.29 -0.00 ...]                      # passthrough through 14 layers
after_L27: [-0.29 -0.00  3.29 -0.00 ...]                      # L14 attn adds another +2.0 at slot 2
```

* `OUTPUT_LO[2] = 1.29` after L12 → decodes to nibble 2, i.e. the low nibble of 0x02 (the
  ADDRESS byte 1).
* Identical pattern at p=194 (val byte 2) and p=195 (val byte 3): `OUT_LO=2/~1.25` jumps in at L12.
* `OUTPUT_HI` follows the same pattern, picking nibble 0 (high nibble of 0x02).

Physical block L12 hosts exactly one declared op (verified via `_build_layout_only`):

```
L12: layer10_psh_ax_broadcast  (kind=attn, num_heads=12, HD=100)
       writes={STACK0_BYTE_VAL_{1,2,3}_{LO,HI}}
       heads 8/9/10 in attn  (via _l10_head_idx)
```

The declared O writes go to `STACK0_BYTE_VAL_h_{LO,HI}` only. **But the empirical write
lands in `OUTPUT_LO/HI`** — the same band that downstream L14/L15/L17 consume. Cross-checked by
inspecting `STACK0_BYTE_VAL_1_LO[0..15]` at p=193: zero across all layers, confirming the head
is depositing into `OUTPUT_LO` instead of (or in addition to) the declared sink.

## Which head is selecting the wrong K row

`_layer10_psh_ax_broadcast_head_spec(byte_h=1)` (heads 8/9/10, the byte-1/2/3 broadcasters):

* **Q gate at slot 0**: `stack0_byte_dim + byte_index_h + IS_BYTE - 2*CONST -
  (neg suppressors at MARK_AX/SP/BP/PC/MEM and other BIs)`.

  At `p=193` (MEM val byte 1): `MARK_MEM` itself is only at `m@188` (d=0); `p=193` carries
  `BYTE_INDEX_1=1, IS_BYTE=1, stack0_byte_dim=0`, no STACK0 marker, no MARK_MEM. Q[slot 0]
  evaluates to ≈0 (`0 + L + L − 2L = 0`). Slot 33 (`stack0_byte_dim + byte_index_h − CONST`) is
  also ≈0.

  → Q at p=193 is **near-zero, not negative**, so softmax1 cannot suppress this query row.
  ALiBi recency + a near-uniform attention distribution still produces a non-trivial V average,
  pulling CLEAN_EMBED from whichever K row dominates softmax. With `IS_BYTE * L` on the K side
  at every byte row, the most recent `BYTE_INDEX_1` K row in context wins — and the only
  BYTE_INDEX_1 K row "near" the SI step is the IMM-step AX byte 1 (rows 134, 204), whose
  CLEAN_EMBED carries `0x02` (the LSB of the 0x200 immediate that ALiBi recency favors most).

* **K side OP discrimination**: `OP_PSH + L` on K row, `OP_SI/SC/JSR/ENT − M_K33` on K, but
  these are slot-33 weights. The Q's slot-33 row at p=193 is ≈0, so the K-side OP_PSH/OP_SI
  scoring contributes ≈0 to the score for THIS query row. K-side scoring only matters when Q[33]
  is nonzero (i.e. at the intended STACK0 byte-h Q row).

* **Why O lands in OUTPUT_LO instead of STACK0_BYTE_VAL_1_LO**:
  Both Q and K are computed via `attn.W_q`/`attn.W_k`; W_q is shared across all heads (it's a
  single Linear projecting d_model → num_heads*HD). At physical block 12 the layer has
  `num_heads=12` (widened by Wave 1 A2), and head 8/9/10 take slot ranges [800..899], [900..999],
  [1000..1099]. But W_v/W_o for this declared head writes its V slots to V[800..899] (head 8).
  Empirical evidence: `STACK0_BYTE_VAL_1_LO[2] = 0` at p=193, while `OUTPUT_LO[2] = 1.29` —
  the head IS firing here but its O column writes into `OUTPUT_LO` (dim 174..189) instead of
  `STACK0_BYTE_VAL_1_LO` (dim 734..749). One of:
    1. The head_idx allocator placed heads 8/9/10 into a layer with `num_heads=8`, wrapping head
       8→head 0 (which is shared with `layer10_carry_relay_bake` whose O writes `OUTPUT_LO`). The
       padded layer-12 attn (num_heads=12) seems to leave heads 0..7 dimensioned but the
       compiled bake may still hit head_0's O column.
    2. O-write target dim was resolved to an offset relative to the broadcast head's own slot
       range but mis-aliased to the global OUTPUT_LO offset.

## Recommended fix

This is a Q-side gate tightness bug compounded by O-routing aliasing. The cleanest fix is
two-part:

1. **Tighten the Q gate at slot 0** so it is strictly negative at every non-target row,
   particularly MEM rows. Add explicit negative suppression for MEM val-byte rows by reading
   `MEM_VAL_B1/B2/B3` (which fire ONLY at MEM val-byte rows):
   ```python
   q.append(AP(0, BD.MEM_VAL_B1, -L))   # for h=1 broadcaster
   q.append(AP(0, BD.MEM_VAL_B2, -L))
   q.append(AP(0, BD.MEM_VAL_B3, -L))
   ```
   This pushes Q[slot 0] at p=193 to `0 + L + L − 2L − L = −L`, deep enough that softmax1
   suppresses the entire row's attention output regardless of K.

2. **Audit the O-write head_idx for heads 8/9/10**. Block 12 is the only num_heads=12 layer in
   the pipeline; the qwen-export padding (commit c07c1236) may have shifted the head-slice
   slicing the broadcast heads expect. Verify that
   `attn.W_o[STACK0_BYTE_VAL_1_LO, head_8 slot 0]` is the only nonzero column for that O write
   at L12, and that **no** column writes to `OUTPUT_LO` from the broadcast head's slot range.
   Use `tools/dim_flow_audit.py STACK0_BYTE_VAL_1_LO` to enumerate declared writers, then
   compare to the empirically nonzero `block.attn.W_o` columns at L12 to identify the mismatch.

The L14 head 5 amplification (+2.0 at L27) is a secondary symptom — it is doing its job of
copying CLEAN_EMBED at the MEM val byte 1 row, but the upstream L12 contamination feeds the
wrong content into the L14 V input. Once L12 is fixed, L14 head 5's correct K-row selection
(AX byte 1 of the SI step, where AX=42 → byte 1 = 0x00) will deliver the correct value.

## Repro

```
python c4_release/tools/probe_l14_li_consumer.py
```

Or the test directly:

```
pytest c4_release/tests/test_smoke.py::TestSmokeMemory::test_si_li_roundtrip -v
```
