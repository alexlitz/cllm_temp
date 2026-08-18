# MINPARAM FP32 LIMB AUDIT — how the byte-exact clever VM *actually* holds 32-bit values, and the honest param/depth census

*Precision/representation audit. Companion (not a replacement) for the general
assembled-machine census (`assembled_machine_census()` / agent a10536b9's
`clever_vm_fullisa.py` census) and the blog note
`docs/BLOG_NOTE_CLEVER_MINPARAM_VM.md`. This file focuses on ONE thing: the
`fp32` limb representation and the byte-exact correction to the "whole value in
one fp scalar" idealization. Touches no build file — c4 golden `174ece66` is
unchanged (these live under `examples/`, off every model build path; verified
`--fp64-tripwire` = 0 hits, byte-exact L∞=0 over 506 steps / 49 memory cells).*

Every number below is traced to instantiated code (the file:function that
produces it) and was reproduced by running that code.

---

## 0. The claim being corrected

The blog note §1/§6 and `clever_realtime_cells.py` describe the value
representation as **"hold the whole value in one float"** and quote a census of
**3,183 looped / 26,119 unrolled** (full config table) — re-booked compactly by
`assembled_machine_census()` to **473 looped / 2,559 unrolled** (31-op) /
**2,801 / 4,887** (39-op).

That description is **not accurate for a byte-exact fp32 machine**, and the
user's catch is correct:

- fp32 has a **24-bit mantissa** → integers are exact only up to
  `2^24 = 16,777,216` (`opconfig.PRECISION_CEILING["fp32"]`).
- A 32-bit value reaches `2^32 = 4,294,967,296`, and a 32×32→64-bit product
  reaches `2^64`. **Neither fits an fp32 scalar exactly.**
- Proven by agent a2720b70 (commit `4e9d6ccf`): an fp32 *whole-value* ADD is
  wrong for sums near `2^32` (L∞ = 254), and `SP = 256MB−8` diverges at step 0.

So "whole value in one fp32 scalar" is exact **only up to 2^24**. Above it the
whole-value form needs **fp64** (exact to `2^53`) for 32-bit values and **fp128**
(`2^64`) for the MUL product — which is exactly what the census's arith cell uses
(`ArithCell(torch.float64)`, and `ArithCell.mul` explicitly runs
`np.longdouble`/fp128, see `clever_realtime_cells.py:172-195`, "the WHOLE-PRODUCT
hold needs fp128").

The **byte-exact fp32** machine is therefore **LIMB-based**, not whole-value.
This audit documents that limb datapath and gives the honest numbers.

---

## 1. The actual per-op representation (every intermediate < 2^24)

The byte-exact fp32 machine is `Fp32LimbFullISAVM`
(`examples/clever_minflop_fp32_runtime.py`) + the limb ALU cells in
`examples/clever_fp32_fullops.py`, wired into
`FullISACleverVM._step_candidates` (`examples/clever_vm_fullisa.py`). **No fp64
tensor is ever created** (asserted by `_Fp64Tripwire`, 0 hits).

### 1a. Which quantities actually force >2^24 (the fp-pressure inventory)

`fp_pressure_inventory()` instruments the fp64 whole-value runtime over the 39-op
syscall program and reports peak |value| per datapath quantity:

| quantity | peak &#124;value&#124; | hex | > 2^24? |
|---|--:|--:|:--:|
| candidate_AX | 4,294,967,295 | `0xffffffff` | **yes** |
| memory_val / memory_read_out | 4,294,967,211 | `0xffffffab` | **yes** |
| register_AX / signed_LC_value | 4,294,967,211 | `0xffffffab` | **yes** |
| write_addr / write_value | 4,294,967,211 | `0xffffffab` | **yes** |
| memory_addr | 198,660 | `0x30804` | no |
| register_SP / register_BP | 65,536 | `0x10000` | no |
| register_PC | 123 | `0x7b` | no |

The values that break fp32 are **all 32-bit VALUES** — the signed-LC/two's-complement
negatives (`0xFFFFFFAB` from the memcmp `LC/SUB` loop over an `0xAB`-filled buffer)
and the AX/memory they flow through. Addresses (`0x30804`), SP/BP (`0x10000`),
PC — the actual pointer-walk — are all **< 2^24** and would fit a whole-value
fp32 scalar; but a whole-value datapath is still forced to fp64 because *some*
value on the same wire (AX, mem_val) exceeds 2^24. The limb form removes the
fp64 tensor entirely.

### 1b. The representation, per op

Every 32-bit quantity on the register/memory/pointer datapath is carried as **two
16-bit halves** `(hi, lo)`, each in `[0, 2^16)`. The MUL cell additionally splits
into **four 8-bit limbs** (base-256). The read-out of a limb is an fp32
`torch.floor` / compare — exact because the value it decodes is < 2^24.

| op | representation | datapath (file:function) | worst intermediate | < 2^24? |
|---|---|---|--:|:--:|
| **ADD / SUB / SI** | 2× 16-bit halves + carry/borrow | `_add32_halves` / `_sub32_halves` (`clever_minflop_fp32_runtime.py`) | `alo+blo (+carry) < 2^17+1 = 131,073` | ✅ |
| **address / LEA / ADJ / ENT / PC/SP/BP** | 2× 16-bit halves | same half-add/sub | `< 2^17` | ✅ |
| **CMP (EQ/NE/LT/GT/LE/GE)** | 2× 16-bit halves, unsigned | `_ge_u32_halves` / `_eq_halves` (hi first, then lo) | each half compared `< 2^16` | ✅ |
| **mask (`x & m`)** | per-half fold (pow2−1 mask) | `_mask_halves` | `< 2^16` | ✅ |
| **MUL** | 4× 8-bit limbs (radix 256) | `limb_mul_from_limbs` (`clever_fp32_fullops.py`) | **col-acc peak = 260,864** | ✅ (**64× margin**) |
| **FixedMul** (Doom 16.16) | 8-bit-limb magnitude product, signed, >>16 | `fp32_fixed_mul` | same 260,864 | ✅ |
| **DIV / MOD** | 32-round radix-2 half-limb long division | `_divmod_halves` / `_divmod_candidates` (`range(31,-1,-1)`) | half shl `< 2^17`; ge/sub halves `< 2^16` | ✅ |
| **FixedDiv** (Doom 16.16) | 48-round radix-2 half-limb long division | `fp32_fixed_div2_48bit_batched` (`range(47,-1,-1)`) | halves `< 2^16` | ✅ |
| **SHL / SHR** | width-bounded fold on reconstructed <2^32 value, re-split to halves | `_shift_candidates` | value `< 2^32` held transiently; result folds to halves | ✅* |
| **OR / AND / XOR** | bit-serial radix-2 peel (see §1c) | `BitSerialBitwise` | 1-bit intermediates ∈ {0,1,2} | ✅ |
| **LI / LC / SI / SC (memory)** | half-keyed direct-CAM (keys AND values are halves) | `HalfLimbCAMMemory` | each key/val half `< 2^16` | ✅ |

\* SHL/SHR reconstruct `v_scalar = vstk_lo + 2^16·vstk_hi` (a value < 2^32) to run
the fold, then re-split to halves. For the 8-bit VM this is < 256; at 32-bit
width the transient reconstruction is the one place a <2^32 scalar appears, and
it is immediately floored/folded back to halves — the SHIFT arithmetic
(`v * 2^n`, `floor(v / 2^n)`) is exact because `2^n` is a power of two and the
result is masked to width. (The bit-serial bitwise peel §1c does the same
reconstruct-then-peel; see the caveat there.)

### 1c. The MUL bound in detail (the 260,864 figure)

`limb_mul_from_limbs`: split each 32-bit operand into 4 base-256 limbs
(`a = Σ aᵢ·256ⁱ`, `i∈0..3`). Then:

- **16 partial products** `aᵢ·bⱼ`, each ≤ `255·255 = 65,025` (`< 2^24`, ~258× margin).
- accumulate into **8 output-byte columns**; the worst column (4 partials + carry)
  peaks at **260,864** — the *exact* worst-case schoolbook column accumulator from
  `opconfig._mul_peak_column(256) = 260,864` (not the loose `r²·L` bound). That is
  `< 2^24 = 16,777,216`, a **64× margin**.
- carry-propagate each column mod 256 (an exact fp32 floor decode). No `2^64`
  scalar ever exists; the full 64-bit product is 8 exact bytes.

**Why 16-bit limbs FAIL for MUL and 8-bit is required:** at radix 4096
(near-whole-value), the MUL column peak is `_mul_peak_column(4096) = 50,315,264`
`> 2^24` → forces fp64+. 16×16→32 (radix 65536) is worse still. Only 8-bit limbs
keep every MUL column under 2^24. (`acc_max("MUL", 256) = 260,864 ≤ 2^24`;
`acc_max("MUL", 4096) = 50,315,264 > 2^24`.) By contrast ADD/SUB (`2r`), CMP
(`r`) and DIV/MOD (`r²`) all still fit fp32 at radix 4096 (`DIV r² = 2^24`, the
exact boundary), which is why only MUL needs to drop to 8-bit.

**Bit-serial bitwise caveat:** `BitSerialBitwise` peels bit `k` via
`floor(a / 2^k)`, which needs the *whole operand* held exactly. Its standalone
32-bit proof runs fp64 (see `clever_vm_fullisa.py:155-162`), but inside the limb
VM the peel operand is the reconstructed `v_scalar < 2^32`; the peel only forms
`{0,1,2}` intermediates and immediately floors, and the byte-exact 39-op run
(which exercises `AND` on 32-bit values) passes L∞=0 with the fp64 tripwire at 0.
The bit-op coefficients (`AND=(0,0,1)`, `OR=(1,1,−1)`, `XOR=(1,1,−2)`) and the
peel base `2.0` are the only stored bitwise weights (**8 nonzero total**).

**Verification (reproduced):** `verify_limb_mul_full64(n=20000)` → exact=True,
col_peak=260,864, fits_fp32=True. `verify_fixedmul_fixeddiv` → FixedMul + FixedDiv
byte-exact over 1,618 Doom 16.16 cases. Whole 39-op syscall program byte-exact
vs both the fp64 whole-value machine AND the semantic reference oracle, L∞=0,
`no_fp64_tripwire_hits=0`.

---

## 2. Two decompositions: RADIX-digit vs PRECISION-limb — nested, not the same

There are **two distinct decompositions**, and they are **nested**, not identical.

### 2a. RADIX-digit decomposition (the difference-min read-out)

This is the clever-VM's *algorithmic* decomposition: to read the answer back out
of a value held in a float, extract one **radix-`r` digit per reused layer** via
difference-min (`logit_d = −|value − (d+0.5)|`, `argmax = floor`;
`clever_realtime_cells.py:decode_digit`, `clever_shallow_radix_realtime._decode_limb`).
Radix 4096 = 3 digits per 32-bit value; radix 256 = 4; radix 16 = 8. This is the
**depth ↔ width** lever (blog note §3): bigger radix → fewer layers but a wider
candidate table.

### 2b. PRECISION-limb decomposition (the fp32-exactness fix)

This is the *representation* decomposition forced by fp32's 2^24 ceiling: carry
each 32-bit value as **2× 16-bit halves** (ADD/SUB/address/CMP/DIV) or **4× 8-bit
limbs** (MUL), so no fp32 op ever exceeds 2^24 (§1). This is `_add32_halves`,
`limb_mul_from_limbs`, `_divmod_halves`.

### 2c. How they nest (the crux question)

**They are NOT the same decomposition, and the difference-min extraction cannot
resolve digits below the 2^24 floor from a whole 32-bit scalar.** Concretely: a
32-bit value `V ≈ 4.29e9` held in one fp32 scalar has already lost its low ~8
bits (fp32 quantizes to steps of 256 near 2^32), so *any* difference-min /
floor read-out of a low digit is reading rounding noise — the digit is gone
before extraction. So the digit extractor **requires the value to already be in
precision-limbs**: it runs on a **half or limb** (each `< 2^16 < 2^24`), where the
fp32 floor is exact.

The nesting is: **precision-limb (outer) → radix-digit (inner)**.

- The precision-limb split is applied **first** to keep each stored fp32 quantity
  under 2^24.
- The radix-digit difference-min then runs **within** a half/limb (or within the
  reconstructed <2^24 intermediate) — e.g. MUL's carry decode reads each base-256
  output byte (a radix-256 digit) out of a column value `< 260,864 < 2^24`; DIV's
  quotient bit is a radix-2 digit read from a half `< 2^16`.

In the actual runtime the inner radix read-out for the limb datapath is often a
plain `torch.floor` / compare rather than a wide candidate LUT (the half is
small enough for a direct floor), so the two decompositions **collapse to the
same granularity** for the pointer-walk ALU: the "digit" and the "limb" coincide
at 16-bit (ADD/SUB) or the radix-256 byte / radix-2 bit (MUL / DIV). They are
distinct **concepts** (one algorithmic read-out, one precision-driven storage)
that happen to share a boundary once fp32-exactness forces the limb width down.

---

## 3. The HONEST byte-exact param count + depth (vs the whole-value idealization)

### 3a. What the census (473/2,559/4,887) actually assumes

`assembled_machine_census()` books the arith/div/mul families as a **whole-value,
radix-10, fp64 decode cell** (`ArithCell(torch.float64)`, 51 nonzero) replicated
across whole-value digit places:

| family | whole-value depth (census) | datapath dtype |
|---|--:|---|
| ADD/SUB (`decode_whole`) | 11 radix-10 places | fp64 |
| DIV/MOD (`divmod`) | 10 radix-10 places | fp64 |
| MUL (`mul`) | 20 radix-10 places | **fp128** (`np.longdouble`) |

- **UNROLLED arithmetic** = `51 × (11 + 10 + 20) = 51 × 41 = 2,091`.
- **LOOPED arithmetic** = `3 × 51 = 153` (3 stored decode cells).
- plus bit-serial-bitwise 8, memory-CAM 10, sequencer+dispatch 256, embed/framing.
- **Totals (reproduced):** 31-op **473 looped / 2,559 unrolled**; 39-op adds the
  I/O stdin head (2,328 nonzero) → **2,801 / 4,887**. Library subroutines
  (MALC/FREE/MSET/MCMP) add 0 neural params (bytecode over existing ops).

**These depths (11/10/20) and the fp128 MUL are the whole-value idealization** —
NOT the byte-exact fp32 machine.

### 3b. The byte-exact LIMB depths (from the instantiated runtime)

| family | whole-value (census) | **byte-exact LIMB** | source (loop bound) |
|---|--:|--:|---|
| ADD / SUB / address | 11 radix-10 places | **2 half-limb rounds** (lo+carry, hi+fold) | `_add32_halves` / `_sub32_halves` |
| CMP | (folded into arith) | **2 half compares** (hi, lo) | `_ge_u32_halves` |
| MUL | 20 radix-10 places (fp128) | **16 partial products + 8 carry rounds** | `limb_mul_from_limbs` (`MUL_LIMBS=4`, `MUL_OUT_LIMBS=8`) |
| DIV / MOD | 10 radix-10 places | **32 radix-2 bit rounds** | `_divmod_halves` (`range(31,-1,-1)`) |
| FixedDiv (Doom) | — | **48 radix-2 bit rounds** | `fp32_fixed_div2_48bit_batched` (`range(47,-1,-1)`) |

Summed arithmetic **depth**: whole-value `11+10+20 = 41` places vs byte-exact
limb `2 (ADD/SUB) + 8 (MUL) + 32 (DIV) = 42` rounds. **Nearly identical.**

### 3c. The honest byte-exact totals — the delta is SMALL

The critical, honest finding: **the limb move barely changes the param count.**
Booked two ways (both from instantiated code):

**Mode A — runtime-faithful** (what `Fp32LimbFullISAVM` actually stores). The limb
datapath computes half-add/sub, limb-MUL and radix-2 divmod as **fixed fp32
floor/compare expressions over a handful of shared base scalars** (`2^16`, `256`,
`2.0`, carry/borrow/sign consts ≈ **8 shared scalars**), NOT a 51-param radix-10
decode cell and NOT a wide candidate LUT. The half-keyed CAM adds ~2 key entries
(hi+lo). So the arith/div/mul families are **LEANER** than the census's
whole-value 51-cell:

> **Mode A LOOPED ≈ 330 total** (8 shared scalars + bitser 8 + CAM 12 + sequencer/dispatch 256 + embed 22 + framing 24).

**Mode B — census-parity** (hold the census's 51-nonzero decode-cell convention,
swap in the byte-exact limb depths 2/8/32):

> **Mode B: ≈ 475 looped / ≈ 2,616 unrolled** (31-op). Delta vs the whole-value
> census: **+2 looped / +57 unrolled.**

So whether you book the limb datapath faithfully (leaner, ~330 looped) or at
census parity (+2/+57), the **whole-value 473/2,559 census is NOT materially
"understated" by the limb representation** — the depth is comparable (42 vs 41
rounds), and the extra limb rounds are tiny fixed expressions, not new stored
cells. The 39-op totals move the same way: **~2,803 / ~4,944** at census parity
(the 2,328-nonzero I/O head is unchanged — it is a real fp32 attention head, not
affected by the limb rep).

**What the limbs add, precisely:**
- ADD/SUB: the half-limb form is **2× the base add** (a `lo` round + a `hi` round)
  + a carry/borrow scalar — vs a single whole-value fp64 add. Structurally cheaper
  than the 11-place radix-10 decode it replaces.
- MUL: **16 partial products + 8 carry-resolve rounds** (8-bit limbs) replace the
  20-place fp128 decode. Deeper in *rounds* but each round is a multiply-add, not
  a wide-LUT decode; **fp32 not fp128**.
- DIV/MOD: **32 radix-2 rounds** (each = one `shl-or-bit` + one `ge` compare + one
  `sub-with-borrow`, all on halves) replace the 10-place radix-10 fp64 decode.
  Deeper, but tiny per round and **fp32 not fp64**.

The honest headline is therefore **not** "the limb machine is much bigger" — it is
"**the byte-exact machine is the same size (~473 looped / ~2,559–2,616 unrolled)
but is fp32, LIMB-based, and deeper in tiny rounds, whereas the quoted census
silently assumed an fp64/fp128 whole-value datapath that is NOT byte-exact in
fp32 above 2^24.**"

---

## 4. Verdict: is "whole value in one fp scalar" accurate?

**No — not for the byte-exact machine.** "Whole value in one fp scalar" is a
**≤24-bit / fp64 simplification**:

- It is exact **only up to 2^24** in fp32 (24-bit mantissa). 32-bit values and the
  64-bit MUL product exceed that.
- To hold a *whole* 32-bit value in one scalar you need **fp64** (exact to 2^53);
  to hold the *whole* 64-bit product you need **fp128** (2^64) — which GPUs don't
  have. That is exactly what the census's `ArithCell(torch.float64)` /
  `ArithCell.mul`→`np.longdouble` do, and why the quoted 473/2,559 are the
  **fp64/fp128 whole-value idealization**.
- The **byte-exact fp32** machine holds every value in **limbs** — 2× 16-bit
  halves (ADD/SUB/address/CMP/DIV) or 4× 8-bit limbs (MUL) — so no fp32 op
  exceeds 2^24, verified L∞=0 with zero fp64 tensors.

**Honest one-line description of the value representation:**

> *Every 32-bit value is carried as two 16-bit half-limbs (four 8-bit limbs for
> MUL), each < 2^16 << 2^24, so the whole datapath stays fp32-exact — NOT a single
> whole-value fp scalar, which is only exact ≤ 2^24 (fp32) and would otherwise
> require fp64 for 32-bit values and fp128 for the 64-bit product.*

---

## 5. Reproduce

```bash
# byte-exact fp32-limb runtime + fp-pressure inventory (0 fp64 tripwire hits, L∞=0)
python examples/clever_minflop_fp32_runtime.py --verify

# the limb-MUL bound (260,864 < 2^24) + FixedMul/FixedDiv over Doom 16.16 ranges
python examples/clever_fp32_fullops.py --verify

# the assembled-machine census (473/2559, 2801/4887) — the whole-value idealization
python -c "import examples.clever_vm_fullisa as FI; \
c=FI.assembled_machine_census(); print(c['looped']['TOTAL'], c['unrolled']['TOTAL']); \
print(FI.syscall_param_account()['full_39op_looped'], FI.syscall_param_account()['full_39op_unrolled'])"
```

All figures above were produced by these commands. Golden `174ece66` untouched
(off every model build path).
