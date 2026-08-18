# CLEVER_VM_FULLISA — the full c4 ISA assembled into one running machine, with bit-serial bitwise and the honest parameter account

`examples/clever_vm_fullisa.py` does three things and reports one honest number:

1. **collapses the bitwise "floor"** with a BIT-SERIAL (radix-2) OR/AND/XOR — the
   16×16 nibble LUT's ~24K/~3K nonzero becomes **8**;
2. **assembles the FULL c4 ISA** into the Phase-1 running fetch-decode-execute
   machine (`examples/clever_vm_runtime.CleverVM`) — every opcode wired into the
   one-hot dispatch — and runs a real full-ISA program **byte-exact** (L-inf = 0
   at every step vs the reference);
3. reports **the TOTAL non-zero parameter count of the *assembled* running
   machine** (not isolated op-cells), unrolled and looped, with a per-component
   breakdown, and how much lower it is than the LUT-bitwise census.

Golden `174ece66` is untouched — this is a NEW example file, off every model
build path. Run:

```
python examples/clever_vm_fullisa.py --verify     # all three
python examples/clever_vm_fullisa.py --bitwise    # just the bit-serial proof
python examples/clever_vm_fullisa.py --census      # just the parameter account
```

---

## 1. Bit-serial bitwise — the LUT is not a floor

**Why the census's bitwise chunk was big.** The measured `BitwiseCell`
(`examples/clever_realtime_cells.py`) is a RADIX-16 nibble **16×16 LUT**: 256
hidden units per op (one AND-detector per `(na,nb)` pair), ×{OR,AND,XOR}. That is
**2,974 nonzero looped** / **23,792 unrolled** (8 nibble places) — the dominant
non-arithmetic chunk of the whole census.

**Bit-serial (radix 2) makes it trivial.** `BitSerialBitwise` peels each bit and
applies the 1-bit op as ONE arithmetic expression (no table):

| op  | expression on 0/1 inputs `a_k,b_k`        | coefficients `(c1,c2,c3)` s.t. `r_k = c1·a_k + c2·b_k + c3·a_k·b_k` |
|-----|-------------------------------------------|--------------------------------------------------------------------|
| AND | `a_k · b_k`                               | `(0, 0, 1)`   → 1 nonzero |
| OR  | `a_k + b_k − a_k·b_k`                     | `(1, 1, −1)`  → 3 nonzero |
| XOR | `a_k + b_k − 2·a_k·b_k`  (== `(a_k−b_k)²`)| `(1, 1, −2)`  → 3 nonzero |

The bit is peeled with the SAME exact floor the arithmetic decode uses
(`a_k = floor(a/2^k) mod 2`), and the word is recomposed `Σ_k r_k·2^k`. The whole
cell's real nonzero tensors:

```
op_coeffs (OR 3 + AND 1 + XOR 3)          = 7
radix2_base (shared peel / recompose base) = 1
--------------------------------------------------
TOTAL bit-serial bitwise nonzero           = 8      (was 2,974 / 23,792)
```

- **Byte-exact:** OR/AND/XOR all PASS vs `torch.bitwise_*` over 20,000 random
  32-bit operands.
- **Depth cost:** **32 bit-rounds** vs the LUT's 8 nibble-rounds — deeper (radix 2
  = one bit/round), but the per-round machinery is a FIXED 8-nonzero cell reused
  per bit, not a replicated 256-entry table.
- **Precision:** a full 32-bit operand exceeds fp32's 2²⁴ exact-integer ceiling,
  so the peel datapath runs in fp64 (exact to 2⁵³), exactly as the arithmetic
  decode cell does. Inside the 8-bit VM (values < 256) fp32 suffices.

The single reused 8-nonzero cell holds ALL THREE ops (the three coefficient rows
share the peel/recompose base), so it is a single stored member in BOTH modes.

---

## 2. The assembled full-ISA machine

`FullISACleverVM` subclasses the Phase-1 `CleverVM` and wires **every** opcode's
candidate into the one-hot dispatch. The Phase-1 foundation already dispatched
IMM/LEA/LI/LC/SI/SC/PSH/ADD/SUB/CMP×6/JMP/BZ/BNZ/JSR/ENT/ADJ/LEV/HALT; this file
adds the four ALU families the foundation left as a Phase-2 stub:

| family        | datapath                                                                        |
|---------------|---------------------------------------------------------------------------------|
| OR/AND/XOR    | **bit-serial** (`BitSerialBitwise`, this file)                                   |
| SHL/SHR       | `×2^n` / arithmetic `/2^n`, shift-count saturated to the value width (fp32)      |
| MUL           | **limb-MUL** (`clever_fp32_fullops.limb_mul_from_limbs`, 8-bit limbs, fp32)      |
| DIV/MOD       | 16-bit-half **long division** (`clever_fp32_fullops` half-limb datapath, fp32)   |
| memory        | **direct-CAM** read/write (`clever_vm_runtime.DirectCAMMemory`)                  |

**Variable per-op depth** is absorbed by the sequencer's inner unroll: bitwise
runs 32 bit-rounds, MUL 8 limb columns, DIV 32 division rounds, ADD-class 1
combine — every op computes its candidate on the shared datapath, and the SAME
7-channel one-hot commit (`next_R = Σ_k onehot_k · cand_k` for PC/SP/BP/AX +
addr/val/active) routes the winner. Exactly-one-hot ⇒ collision-free.

**One overflow guard was needed to compose the ops honestly:** a non-winning
candidate that overflows to `inf` (e.g. SHL's `2^205` when AX is large) poisons
the commit via `0·inf = NaN`. The shift exponent is saturated to the value width
before exponentiating — finite, and exactly the reference `(v << n) & mask` /
arithmetic `v >> n` result at any shift ≥ width.

### The program it runs byte-exact

`_build_fullisa_program()` (labels computed, no manual off-by-one) exercises
arithmetic + bitwise + shifts + MUL/DIV/MOD + memory + branches + a function
call:

```
main:  6·7=42 ; 42%5=2 ; 2+20=22 ; 22//3=7           (MUL/MOD/ADD/DIV)
       mem[80]:=99 ; load it                          (SI/LI)
       99&15=3 ; 3|48=51 ; 51^255=204                 (AND/OR/XOR bit-serial)
       204>>1=102 ; (102<<1)&255=204                  (SHR/SHL)
       push 204 ; JSR func ; ADJ ; EQ 205 ; BZ else…  (call + compare + branch)
func:  ENT ; LEA 2 ; LI ; PSH ; IMM 1 ; OR ; LEV      (returns arg|1 = 205)
```

- **50 steps, register L-inf = 0, memory L-inf = 0**, lanes-identical across a
  256-lane batch. Final AX = 222.
- **ops exercised:** ADD, ADJ, AND, BZ, DIV, ENT, EQ, HALT, IMM, JSR, LEA, LEV,
  LI, MOD, MUL, OR, PSH, SHL, SHR, SI, XOR (21 distinct).
- **Exhaustive per-opcode proof:** a tiny probe program per op lands **31/31**
  dispatch opcodes (covering the 10 the demo doesn't hit: SUB/NE/LT/GT/LE/GE/
  LC/SC/JMP/BNZ) at full-state L-inf = 0. `EXIT == HALT`.

---

## 3. The real parameter account of the assembled machine

This is the **union of the whole running machine**, not an op-cell subset: all op
weights (bit-serial bitwise) + the sequencer / decode / one-hot-dispatch fabric +
the direct-CAM memory + framing/embed. Per-component real-cell footprints:

| component                       | nonzero | note |
|---------------------------------|--------:|------|
| arith decode cell (fp64)        |      51 | reused per digit place (ADD/SUB/CMP/SHL/SHR/DIV/MOD/MUL) |
| **bit-serial bitwise (OR/AND/XOR)** | **8** | the single reused peel cell |
| memory CAM (LI/LC/SI/SC)        |      10 | shared direct-CAM head |
| **sequencer + one-hot dispatch**|   **256** | opcode one-hot decode (31) + 7-channel commit mux (7·31 = 217) + framing scalars (8) — *the fabric the op-cell census omitted* |
| embed + LM-head framing         |      22 | token embed 12 + LM-head decode 10 |

### UNROLLED (every arith/div/mul place + bitwise a distinct stored layer, replicas counted)

Depths: arith 11 + div 10 + mul 20 (= 41 arith places) + **bitwise 32 bit-serial**
+ memory 1 + trivial 1 → applied depth D = 75. Bit-serial bitwise and the CAM are
each a single reused cell (like the census memory family), so they store once.

| component                     | nonzero |
|-------------------------------|--------:|
| arithmetic (51 × 41 places)   |   2,091 |
| bit-serial bitwise            |       8 |
| memory (direct-CAM)           |      10 |
| sequencer + one-hot dispatch  |     256 |
| framing / embed               |     194 |
| **TOTAL non-zero parameters** | **2,559** |

### LOOPED / Universal-Transformer (one stored cell per member + shared control)

| component                     | nonzero |
|-------------------------------|--------:|
| arithmetic (3 × 51 cells)     |     153 |
| bit-serial bitwise            |       8 |
| memory (direct-CAM)           |      10 |
| sequencer + one-hot dispatch  |     256 |
| framing / embed               |      46 |
| **TOTAL non-zero parameters** |   **473** |

### vs the 16×16-LUT-bitwise census

| mode     | LUT-bitwise census | assembled machine (bit-serial) | reduction |
|----------|-------------------:|-------------------------------:|----------:|
| looped   |              3,183 |                        **473** | **−2,710** |
| unrolled |             26,119 |                      **2,559** | **−23,560** |

The bit-serial collapse alone removes the whole bitwise LUT chunk (−2,966 looped /
−23,784 unrolled); the assembled totals also **add** the 256-nonzero
sequencer/dispatch fabric that the isolated op-cell census never counted — so 473
/ 2,559 is the honest number of a machine that actually *runs* the ISA, not a
subset of op-cells. Even carrying that real control fabric, the assembled machine
is far below the LUT census because bit-serial bitwise erased the dominant table.

---

## Honesty notes

- **The 256 sequencer/dispatch nonzero is a real, load-bearing component**, not a
  fudge: it is the opcode one-hot decode + the 7-channel `Σ onehot·candidate`
  commit + the shared PC/SP/BP/stack-stride/JSR/ENT/branch/LEV framing scalars
  that the running machine computes every step. The op-cell census (`3,183` /
  `26,119`) omitted it because it counted op-cells, not a machine.
- **Depth is the trade for bit-serial:** radix-2 is 32 bit-rounds vs 8
  nibble-rounds. This is the standard depth-vs-width lever — the assembled machine
  pays 4× the bitwise depth to erase the 256-entry table.
- **Byte-exactness is the gate throughout:** bit-serial OR/AND/XOR vs
  `torch.bitwise_*` (20k operands), the full-ISA program (L-inf = 0 on
  PC/SP/BP/AX + stack + memory at every step), and 31/31 dispatch opcodes each
  proven L-inf = 0.
- **Golden `174ece66` untouched** — no build file changed; this is a self-contained
  example + doc.
