# The WHOLE 39-op min-flop clever-VM runtime in FP32 — byte-exact, zero fp64

**Date:** 2026-08-09 · **Scope:** convert the min-flop clever-VM runtime's 32-bit
**address / pointer / memory** datapath from **fp64 whole-value** to the **fp32
8-bit-LIMB** (here: two 16-bit-HALF) representation, so the WHOLE 39-op syscall
runtime stays fp32 (zero fp64 tensors) while remaining **byte-exact** vs the
a1b82f47 fp64 whole-value machine.

Code: [`examples/clever_minflop_fp32_runtime.py`](../examples/clever_minflop_fp32_runtime.py).
Builds on agent a1b82f47 (commit `df79ea3b`, the 39-op syscall machine, fp64
whole-value) and agent afdd285b (commit `8ae10204`, the fp32 8-bit-limb MUL/DIV
datapath, `clever_fp32_fullops`). Golden `174ece66` untouched (no build file — off
every model build path, verified). GPU numbers MEASURED on RTX A5000 (device 1 free;
timing gated/polled — the 1-GPU cards were contended by other timing agents).

---

## TL;DR verdict

**The WHOLE 39-op min-flop runtime runs FP32 byte-exact, zero fp64.** Every 32-bit
quantity on the address / pointer / memory / register datapath is carried as two
16-bit halves `(hi, lo)`, each `< 2^16 << 2^24`, so **no fp32 scalar ever exceeds
2^24** — the limb form the task asked for.

- **BYTE-EXACT, L-inf = 0:** the 39-op syscall program (MALC → MSET → MCMP → PRTF
  → OPEN/READ/CLOS → neural-READ → FREE) runs on the fp32-limb runtime with
  register L-inf = 0, memory L-inf = 0 (49 touched cells), stdout `sum=42 ok=yes`
  exact, tool-call log exact — vs BOTH the a1b82f47 fp64 whole-value machine AND
  the independent semantic reference oracle. Confirmed for varied stdin / files /
  batch (1/2/4/8), all lanes identical.
- **ZERO fp64:** an fp64 tripwire over the whole step-loop records **0** fp64-tensor
  creations; all 8 registers are `torch.float32`.
- **Full 31-op ISA byte-exact on the limb datapath too** (all 24 opcode families,
  incl. MUL / DIV / MOD / OR / AND / XOR / SHL / SHR). A 32-bit stress program with
  genuinely `> 2^24` addresses (`0x05000000`) and values (`0x0ABCDE12`) is exact —
  where a whole-value fp32 scalar would lose the low bits.

---

## 1. The fp64-pressure inventory (measured)

Instrumenting the a1b82f47 fp64 whole-value runtime on the 39-op program, every
value on the address / pointer / memory datapath, peak `|value|` vs the fp32 2^24
exact-integer ceiling:

| quantity            | peak \|value\| | hex          | > 2^24 ? |
|---------------------|---------------:|--------------|:--------:|
| `candidate_AX`      | 4,294,967,295  | `0xffffffff` | **YES**  |
| `memory_val`        | 4,294,967,211  | `0xffffffab` | **YES**  |
| `memory_read_out`   | 4,294,967,211  | `0xffffffab` | **YES**  |
| `register_AX`       | 4,294,967,211  | `0xffffffab` | **YES**  |
| `signed_LC_value`   | 4,294,967,211  | `0xffffffab` | **YES**  |
| `write_addr`        | 4,294,967,211  | `0xffffffab` | **YES**  |
| `write_value`       | 4,294,967,211  | `0xffffffab` | **YES**  |
| `memory_addr`       | 198,660        | `0x30804`    | no       |
| `register_SP`/`BP`  | 65,536         | `0x10000`    | no       |
| `register_PC`       | 123            | `0x7b`       | no       |

**The pressure is entirely from 32-bit VALUES, not addresses.** The heap / data /
stack **addresses** (0x30804, 0x10000) are all `< 2^24`. What forces fp64 is the
**signed-LC / two's-complement negative** `0xFFFFFFAB` (= `0xAB` signed-char
`-85`, from the **memcmp** `LC/SUB/BZ` loop over the `0xAB`-filled buffer) carried
in AX, pushed to memory, and compared. A whole-value fp32 datapath is forced to
fp64 because SOME value on the wire (AX, mem_val) exceeds 2^24 — even though the
addresses do not.

## 2. The limb conversion

Every 32-bit quantity is two 16-bit halves `(hi, lo)`, each a `(B,)` fp32 in
`[0, 2^16)` — the same limb representation `clever_fp32_fullops` established for
MUL/DIV (col-acc `< 2^24`), reused for the runtime's pointer walk:

- **Registers** PC / SP / BP / AX → half-pairs; every transition is half-limb.
- **Memory** `HalfLimbCAMMemory` keys ON the halves `(addr_hi, addr_lo)` and stores
  values as halves — the latest-write-wins recency gather compares halves, never a
  `> 2^24` scalar.
- **LEA / LI / ADD / SI / SUB** pointer & value arithmetic = half-limb add / sub
  **with carry / borrow** (`_add32_halves` / `_sub32_halves`), each half masked to
  16 bits → exact 32-bit wrap.
- **`fp_mask`** (`x & mask`) = a per-half fold.
- **signed-LC** the two's-complement negative is a half-pair (`0xFFFF`, `0xFFAB`),
  both `< 2^16` — `0xFFFFFFAB` never exists as one fp32 scalar.
- **memcmp** the `LC/SUB` byte-diff is a subtract-with-borrow on halves.
- **CMP** unsigned 32-bit compare on halves (hi first, then lo).

Every fp32 op on this datapath is `< 2^17` (a half + a carry) → **exact**. The
one-hot dispatch, the sequencer, the whole control fabric run fp32. The 8 syscalls
reuse the fp64 machine's own `NeuralIO` + `LibSubroutineLinker` + reference oracle
unchanged (the I/O boundary is a python side-effect, not a datapath tensor).

## 3. Measured — where the fp64 penalty actually lives (honest)

MEASURED on the A5000 (device 1, free):

| measurement                                              | fp64        | fp32-limb   | ratio |
|----------------------------------------------------------|------------:|------------:|------:|
| **(2a) isolated pointer-datapath transition** (elementwise, bandwidth-bound) | 0.505 ns | 0.710 ns | **0.71×** |
| **A5000 COMPUTE-bound (matmul) fp32 / fp64**             | 8.4 TFLOP/s | 0.18 TFLOP/s | **46.3×** |
| **(2b) whole-program per-step** (python-control-bound)   | 17.24 µs    | 17.15 µs    | **1.01×** |

**The honest picture (measured vs the ~1:64 projection):**

- The task's premise — *fp64 ≈ 1/64 fp32 on the A5000, so the min-flop step goes
  0.1 µs → ~6 µs* — is a **projection about COMPUTE-bound (FP-unit-bound) work**.
  Measured, the A5000's compute-bound fp32/fp64 ratio is **46.3×** (matmul; the
  native ratio is ~1:64, fp32 gets some tensor-core assist). **That wall is real**
  and it lives in the **neural model forward + the attention CAM at scale** — the
  matmul-class pieces.
- The **scalar register / pointer datapath itself** is a batched **elementwise**
  computation, so it is **memory-bandwidth-bound, not FP-unit-bound**: the
  elementwise fp64/fp32 ratio is only ~0.7–1.1× (fp64 moves 2× the bytes; the DP-
  unit penalty does not dominate a per-element add / floor / compare). So the
  *pointer walk alone* does NOT pay 46×.
- **Why the limb datapath is still the load-bearing fix:** keeping the WHOLE
  runtime **fp32-only (zero fp64 tensors)** is the precondition for fusing the
  register / pointer walk into the **same fp32 kernel** as the neural forward. A
  mixed fp64/fp32 runtime forces dtype conversions and **separate kernels** at
  every register↔neural boundary, which (a) blocks the fp32-only fusion the min-
  flop step depends on and (b) drags any fused matmul into fp64's 46× regime. The
  limb form removes the fp64 tensor entirely, so the min-flop step stays a single
  fp32 pipeline.

**Verdict:** the whole 39-op min-flop runtime runs **FP32 byte-exact, zero fp64** —
the address / pointer / memory datapath is limb-based (two 16-bit halves). Nothing
on the datapath genuinely requires fp64: the only `> 2^24` values were 32-bit
signed values / memcmp diffs, and each is exactly two `< 2^16` halves. The residual
fp64 cost is **zero** (no fp64 tensor is created). The measured **elementwise**
penalty on the isolated pointer datapath is small (bandwidth-bound); the **46×**
fp64 wall is a compute-bound (matmul) property of the card that the all-fp32
runtime avoids for the neural forward.

## 4. Reproduce

```
python examples/clever_minflop_fp32_runtime.py --verify         # byte-exact + fp-inventory
python examples/clever_minflop_fp32_runtime.py --fp-inventory    # just the fp64-pressure table
python examples/clever_minflop_fp32_runtime.py --bench --device cuda:0   # the measured penalty
```

Golden `174ece66` untouched (this file and the runtime are off every model build
path — verified before and after).
