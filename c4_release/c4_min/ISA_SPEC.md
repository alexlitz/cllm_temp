# C4-min ISA Reference Spec (bit-exact, 8-bit subset)

Ground-truth reference for the green-field minimal C4 VM (`c4_min/`). The
compiler implements this; the oracle checks against it.

**Authority.** Every rule below is extracted from the *existing* implementation,
which is the sole source of truth. The two Python reference VMs are the
authority for state transitions:

- `neural_vm/verification/dim_oracle.py` → `ReferenceOracle._run()` — the
  **signed / C4-faithful** reference (proper signed DIV/MOD/CMP). This is the
  primary authority.
- `neural_vm/nibble_bytecode_executor.py` → `NibbleBytecodeExecutor.execute()`
  — a second reference that agrees on the integer subset and additionally
  implements `LC`/`SC` (char load/store) which the oracle does not.

Opcode numbers: `neural_vm/opcode_mapper.py::C4Opcode` and
`neural_vm/verification/symbolic_forward.py`. Encoding: `encode_instr` /
`decode_instr` in `symbolic_forward.py`. Prose / calling convention: original
C4 (`docs/BLOG_SPEC.md`, `docs/OPCODE_TABLE.md`).

Where the two references disagree (comparison/DIV/MOD signedness, shift masking)
this spec follows `dim_oracle.py` (signed, C4-faithful) and flags the
divergence explicitly.

---

## 1. Machine model

### 1.1 Registers (register file)

Four architectural registers. In the *full* VM each is **32-bit** and values are
masked into `0 .. 0xFFFFFFFF` after every write. The **minimal 8-bit subset**
operates on values in `0 .. 0xFF`; the register file stays 32-bit-wide but the
subset only exercises the low byte (see §7 for what the 8-bit subset drops).

| Reg | Name            | Width | Initial value            | Role |
|-----|-----------------|-------|--------------------------|------|
| PC  | Program Counter | 32    | `code_base` (default 0)  | Byte address of the next instruction to fetch. |
| AX  | Accumulator     | 32    | 0                        | Sole working register; every ALU/load result lands here. |
| SP  | Stack Pointer   | 32    | `0x100000`               | Top of the descending stack (grows **downward**). |
| BP  | Base Pointer    | 32    | `0x100000` (= initial SP)| Base of the current call frame. |

There is **no separate flags register.** Zero/branch tests read AX directly
(`BZ`/`BNZ`). Comparison opcodes materialize a 0/1 boolean into AX.

`initial_sp = 0x100000` is the reference default (`ReferenceOracle.__init__`,
`nibble_bytecode_executor` line 37). The blog's `0x10000/0x8000` are the
original-C4 defaults; the neural reference uses `0x100000`. **The 8-bit subset
must adopt `0x100000` to be bit-exact with the oracle.**

### 1.2 Instruction encoding

One instruction packs an opcode and a signed/unsigned immediate:

```
word = (op & 0xFF) | ((imm & 0xFFFFFF) << 8)          # encode_instr
op   =  word        & 0xFF                              # decode_instr
imm  = (word >> 8)  & 0xFFFFFF
```

- Opcode: low 8 bits.
- Immediate: next 24 bits (`decode_instr`). In the on-disk **BYTECODE** format
  the instruction is 5 bytes: `[op:1][imm:4 little-endian]` (`docs/BLOG_SPEC.md`).
- **PC stride is 8 bytes per instruction** (`INSTR_WIDTH = 8`). So the
  instruction at PC `p` is `program[(p - code_base) // 8]`, and the next
  sequential PC is `p + 8`. (The on-disk encoding is 5 bytes; the *executed* PC
  advances by 8 — the reference VMs index the program array by `pc // 8`.)

`imm` is treated as an **address / literal** and used verbatim (no sign
extension) except where an opcode interprets AX as signed. Branch/jump targets
are **absolute** byte addresses (`PC = imm`), not PC-relative.

### 1.3 Memory & stack model

Two flat, byte/word-addressable spaces backed by dicts in the reference:

- **`stack`** — word-addressed slots (`stack[addr] = 64-bit word`). Written by
  `PSH`/`JSR`/`ENT` and by `SI`/`SC` to addresses **below `data_base`**. The
  stack grows **downward**: a push does `SP -= 8` then `stack[SP] = value`.
- **`memory_bytes`** — byte-addressed data memory (`memory[addr] = byte`).
  Written by `SI`/`SC` to addresses `>= data_base` (`data_base = 0x10000`), one
  entry per byte, **little-endian** (`SI` writes 8 bytes low-to-high).

Reads (`LI`/`LC`) check the word-`stack` first, then fall back to byte-`memory`
(assembling little-endian for `LI`). **Unwritten memory reads as 0** (zero-fill;
this is the ZFOD contract the neural softmax1 realizes). Endianness is
**little-endian** everywhere (immediates, register byte tokens, `SI`/`LI`).

### 1.4 Fetch–execute cycle

```
loop:
    idx = (PC - code_base) // 8
    if idx < 0 or idx >= len(program):  halt          # PC out of range
    (op, imm) = decode_instr(program[idx])
    pc_next = PC + 8                                   # default sequential
    <execute op; may overwrite pc_next, set halt>
    PC = pc_next
    if halted: break
```

One instruction = one step = (in the neural VM) one forward pass emitting the
register token bundle. The reference caps at `max_cycles` (10000) as a runaway
guard, not an ISA feature.

---

## 2. Opcode map (the ~20 core opcodes)

| # (dec) | # (hex) | Name | Class      | One-liner |
|--------:|--------:|------|------------|-----------|
| 1  | 0x01 | IMM  | literal     | `AX = imm` |
| 25 | 0x19 | ADD  | arithmetic  | `AX = pop + AX` |
| 26 | 0x1A | SUB  | arithmetic  | `AX = pop - AX` |
| 16 | 0x10 | AND  | bitwise     | `AX = pop & AX` |
| 14 | 0x0E | OR   | bitwise     | `AX = pop \| AX` |
| 15 | 0x0F | XOR  | bitwise     | `AX = pop ^ AX` |
| 23 | 0x17 | SHL  | shift       | `AX = pop << (AX & 0x1F)` |
| 24 | 0x18 | SHR  | shift       | `AX = pop >> (AX & 0x1F)` (unsigned) |
| 17 | 0x11 | EQ   | comparison  | `AX = (pop == AX)` (0/1) |
| 18 | 0x12 | NE   | comparison  | `AX = (pop != AX)` (0/1) |
| 19 | 0x13 | LT   | comparison  | `AX = (pop <  AX)` signed (0/1) |
| 21 | 0x15 | LE   | comparison  | `AX = (pop <= AX)` signed (0/1) |
| 20 | 0x14 | GT   | comparison  | `AX = (pop >  AX)` signed (0/1) |
| 22 | 0x16 | GE   | comparison  | `AX = (pop >= AX)` signed (0/1) |
| 9  | 0x09 | LI   | memory      | `AX = *AX` (load word) |
| 11 | 0x0B | SI   | memory      | `*pop = AX` (store word) |
| 10 | 0x0A | LC   | memory      | `AX = *(char*)AX` (load byte) |
| 12 | 0x0C | SC   | memory      | `*(char*)pop = AX` (store byte) |
| 2  | 0x02 | JMP  | control     | `PC = imm` |
| 4  | 0x04 | BZ   | control     | `if AX==0: PC = imm` |
| 5  | 0x05 | BNZ  | control     | `if AX!=0: PC = imm` |
| 3  | 0x03 | JSR  | call        | `push PC_next; PC = imm` |
| 6  | 0x06 | ENT  | call        | `push BP; BP = SP; SP -= imm` |
| 8  | 0x08 | LEV  | call        | `SP = BP; pop BP; pop PC` |
| 38 | 0x26 | EXIT | system      | halt (a.k.a. HALT — see §6) |

Also defined by the full ISA but **outside the minimal subset**: `LEA` (0),
`ADJ` (7), `PSH` (13), `MUL`/`DIV`/`MOD` (27/28/29), the syscall/I-O ops
(30-37, 64-66). `PSH` and `LEA`/`ADJ` are documented in §5 because the calling
convention needs them, but they are not part of the "20 core opcodes" list the
minimal compiler must special-case beyond the table above.

> **Naming note.** The task's opcode list says `HALT`; the ISA's actual
> program-terminate opcode is **`EXIT` (38)**. There is no distinct `HALT`
> opcode — see §6.

Convention shorthand used below: **`pop`** = read the word at `stack[SP]` then
`SP += 8` (only if `SP` is a written stack slot; else the op is a no-op — see
§4.1). **`push v`** = `SP -= 8; stack[SP] = v`. All register writes are masked
`& 0xFFFFFFFF`.

---

## 3. Per-opcode state transitions + worked examples

State tuples below are `(PC, AX, SP, BP)` plus stack/memory notes. Worked
examples use the 8-bit subset (values `0..0xFF`) and start each op at a
representative pre-state; `SP0` denotes some current stack pointer.

### 3.1 IMM — load immediate (op 1)
- **Operands:** immediate.
- **Reads:** imm. **Writes:** `AX = imm & 0xFFFFFFFF`.
- **PC:** `PC += 8`. **Flags:** none.
- **8-bit note:** subset uses `imm` in `0..0xFF`.
- **Example:** pre `AX=0x00`, instr `IMM 0x2A`. → `AX=0x2A`, `PC+=8`.

### 3.2 ADD — add (op 25)
- **Operands:** `AX`, top-of-stack.
- **Reads:** `top = pop`, `AX`. **Writes:** `AX = (top + AX) & 0xFFFFFFFF`.
  `SP += 8`.
- **PC:** `PC += 8`. **Flags:** none (wraps mod 2^32; 8-bit subset wraps mod 2^8
  in effect since operands are ≤ 0xFF, result ≤ 0x1FE, then masked).
- **Example:** stack top `0x05`, `AX=0x03` → pop → `AX = 5+3 = 0x08`, `SP+=8`.

### 3.3 SUB — subtract (op 26)
- **Reads:** `top = pop`, `AX`. **Writes:** `AX = (top - AX) & 0xFFFFFFFF`.
  Order is **`top - AX`** (stack operand minus accumulator).
- **PC:** `PC += 8`. **Flags:** none. Underflow wraps mod 2^32
  (`5 - 8 → 0xFFFFFFFD`).
- **Example:** top `0x0A`, `AX=0x03` → `AX = 10-3 = 0x07`.

### 3.4 AND / OR / XOR — bitwise (ops 16 / 14 / 15)
- **Reads:** `top = pop`, `AX`. **Writes:**
  `AX = (top OP AX) & 0xFFFFFFFF`, `OP ∈ {&, |, ^}`.
- **PC:** `PC += 8`. **Flags:** none. Bit-parallel; identical on any width.
- **Example (XOR):** top `0xF0`, `AX=0x0F` → `AX = 0xFF`.
- **Example (AND):** top `0xF0`, `AX=0x3C` → `AX = 0x30`.

### 3.5 SHL / SHR — shift (ops 23 / 24)
- **Reads:** `top = pop` (value), `AX` (shift count). **Writes:**
  - `SHL`: `AX = (top << (AX & 0x1F)) & 0xFFFFFFFF`.
  - `SHR`: `AX = (top & 0xFFFFFFFF) >> (AX & 0x1F)` — **unsigned/logical** shift.
- **Shift count masked to `AX & 0x1F`** (low 5 bits, per `dim_oracle`; C4
  register-shift semantics).
- **PC:** `PC += 8`. **Flags:** none.
- **Example (SHL):** top `0x01`, `AX=0x04` → `AX = 0x10`.
- **Example (SHR):** top `0x80`, `AX=0x03` → `AX = 0x10`.

> **Reference divergence:** `nibble_bytecode_executor` shifts by the full `AX`
> (`top << ax`, unmasked). Follow `dim_oracle` (`AX & 0x1F`) — it matches C4,
> which uses only the low bits.

### 3.6 EQ / NE / LT / LE / GT / GE — comparison (ops 17 / 18 / 19 / 21 / 20 / 22)
- **Reads:** `top = pop`, `AX`. **Writes:** `AX = 1` if the relation holds, else
  `AX = 0`.
- **Signedness:** `EQ`/`NE` are value-equality (bit-exact). `LT`/`LE`/`GT`/`GE`
  are **signed** 32-bit comparisons: each operand `v` is reinterpreted as
  `v if v < 0x80000000 else v - 0x100000000` before comparing.
- Relation is `top REL AX` (stack operand on the left).
- **PC:** `PC += 8`. **Flags:** none (result is the AX boolean).
- **Example (LT):** top `0x03`, `AX=0x05` → `3 < 5` → `AX = 1`.
- **Example (GE):** top `0x05`, `AX=0x05` → `5 >= 5` → `AX = 1`.
- **Example (LT signed):** top `0xFFFFFFFF` (= −1), `AX=0x01` → `−1 < 1` →
  `AX = 1`.

> **8-bit note:** In the 8-bit subset all values are `0..0xFF`, i.e. non-negative
> in 32-bit space, so signed and unsigned comparisons coincide. The signed rule
> only matters once the full 32-bit range is used.
> **Reference divergence:** `nibble_bytecode_executor` does *unsigned* Python
> `<`. Follow `dim_oracle` (signed) — it is the C4-faithful behavior.

### 3.7 LI — load int/word (op 9)
- **Reads:** `addr = AX`; the value at `addr`. **Writes:** `AX = value`.
- **Resolution:** if `addr` is a written word-`stack` slot → `AX = stack[addr]`;
  else assemble 8 bytes little-endian from `memory_bytes`
  (`val |= memory[addr+i] << (8*i)`, `i=0..7`); unwritten → 0. Result masked to
  32 bits.
- **Address stays in AX.** No `SP` change (address came from AX, not the stack).
- **PC:** `PC += 8`. **Flags:** none.
- **Example:** `AX = 0x10000`, `memory[0x10000]=0x2A` (rest 0) → `AX = 0x2A`.

### 3.8 SI — store int/word (op 11)
- **Operands:** address = **popped** top-of-stack, value = `AX`.
- **Reads:** `addr = pop` (`SP += 8`), `AX`. **Writes:** the 8 bytes of `AX`,
  little-endian, to `addr` (`memory_bytes[addr+i] = (AX >> 8*i) & 0xFF`).
  Stores to `addr < data_base` go to the word-`stack` instead (per the nibble
  executor).
- **AX is unchanged.**
- **PC:** `PC += 8`. **Flags:** none.
- **Example:** stack top `0x10000`, `AX=0x1234` → pop addr; `memory[0x10000]=0x34,
  [0x10001]=0x12`, rest 0. `SP += 8`. `AX` stays `0x1234`.

### 3.9 LC — load char/byte (op 10)
- **Reads:** `addr = AX`; **one byte** at `addr`. **Writes:** `AX = byte & 0xFF`
  (`memory[addr]` or `stack[addr]`, else 0).
- **PC:** `PC += 8`. **Flags:** none. Authority: `nibble_bytecode_executor`
  (the oracle does not implement `LC`).
- **Example:** `AX=0x10000`, `memory[0x10000]=0x41` → `AX = 0x41`.

### 3.10 SC — store char/byte (op 12)
- **Operands:** address = popped top-of-stack, value = low byte of `AX`.
- **Reads:** `addr = pop`, `AX`. **Writes:** `memory[addr] = AX & 0xFF` (single
  byte; `stack[addr]` if `addr < data_base`). `SP += 8`. **AX unchanged.**
- **PC:** `PC += 8`. **Flags:** none. Authority: `nibble_bytecode_executor`.
- **Example:** stack top `0x10000`, `AX=0x0141` → `memory[0x10000]=0x41`.

### 3.11 JMP — unconditional jump (op 2)
- **Operands:** immediate (absolute target).
- **Writes:** `pc_next = imm` (overrides the default `PC += 8`).
- **Flags:** none. No register/stack change.
- **Example:** `JMP 0x40` → `PC = 0x40`.

### 3.12 BZ — branch if zero (op 4)
- **Reads:** `AX`, imm. **Writes:** `pc_next = imm` **iff `AX == 0`**, else
  `pc_next = PC + 8`.
- **Flags:** none (tests AX directly). No register/stack change.
- **Example:** `AX=0x00`, `BZ 0x40` → `PC = 0x40`. With `AX=0x01` → `PC += 8`.

### 3.13 BNZ — branch if not zero (op 5)
- **Reads:** `AX`, imm. **Writes:** `pc_next = imm` **iff `AX != 0`**, else
  `pc_next = PC + 8`.
- **Example:** `AX=0x07`, `BNZ 0x40` → `PC = 0x40`. With `AX=0x00` → `PC += 8`.

### 3.14 JSR — jump to subroutine / call (op 3)
- **Operands:** immediate (callee entry address).
- **Transition:** `SP -= 8; stack[SP] = pc_next (= PC + 8); pc_next = imm`.
  I.e. **push the return address (the address of the *next* instruction), then
  jump to `imm`.**
- **Writes:** SP (−8), one stack slot (return addr), PC. **AX/BP unchanged.**
- **Flags:** none.
- **Example:** `PC=0x00`, `SP=0x100000`, `JSR 0x80` →
  `SP=0xFFFF8`, `stack[0xFFFF8]=0x08`, `PC=0x80`.

### 3.15 ENT — enter frame (op 6)
- **Operands:** immediate = local-frame size (bytes).
- **Transition:** `SP -= 8; stack[SP] = BP; BP = SP; SP = (SP - imm) & 0xFFFFFFFF`.
  I.e. **push the caller's BP, set BP to the new frame base (old SP−8), then
  reserve `imm` bytes of locals.**
- **Writes:** SP, BP, one stack slot (saved BP). **AX/PC** (PC just `+= 8`).
- **Flags:** none.
- **Example:** `SP=0xFFFF8`, `BP=0x100000`, `ENT 0x10` →
  `stack[0xFFFF0]=0x100000`, `BP=0xFFFF0`, `SP=0xFFFE0`.

### 3.16 LEV — leave frame / return (op 8)
- **Operands:** none.
- **Transition (in order):**
  1. `SP = BP` (discard locals).
  2. if `SP` is a written stack slot: `BP = stack[SP] & 0xFFFFFFFF; SP += 8`
     (restore caller BP).
  3. if `SP` is a written stack slot: `pc_next = stack[SP] & 0xFFFFFFFF; SP += 8`
     (pop return address into PC). **else → halt** (return from `main`; see §6).
- **Writes:** SP, BP, PC (or halt). **AX unchanged** (AX carries the return
  value out of `main`).
- **Flags:** none.
- **Example (normal return):** after the `ENT 0x10` frame above, at `LEV`:
  `SP=BP=0xFFFF0`; pop BP=`0x100000`, SP=`0xFFFF8`; pop PC=`0x08` (the JSR return
  addr), SP=`0x100000`. Frame fully unwound; execution resumes after the call.
- **Example (return from main):** `BP == initial SP` and the slot at `SP` after
  the BP-pop is unwritten → `halted = True`, program ends with exit code `AX`.

### 3.17 EXIT — halt (op 38)
- **Operands:** none (exit code is `AX`).
- **Transition:** `halted = True`; program terminates; exit code = `AX & 0xFFFFFFFF`.
- **Writes:** halt flag only. **PC/AX/SP/BP unchanged.**
- **Example:** `AX=0x00`, `EXIT` → program halts, exit code 0.

---

## 4. Corner cases & exact-match rules

### 4.1 Empty-stack pops
Every stack-consuming op (`ADD/SUB/AND/OR/XOR/SHL/SHR/EQ/NE/LT/LE/GT/GE/SI/SC`)
is guarded by `if SP in stack:` in the reference. **If the current `SP` is not a
written stack slot, the op is a silent no-op** (AX unchanged, SP unchanged, no
memory write). Well-formed compiler output always has a matching `PSH` before
each consumer, so this is a robustness guard, not a normal path — but the oracle
compares against it, so the 8-bit VM must reproduce the no-op behavior on
underflow.

### 4.2 Divide/modulo by zero (full ISA, not in the core-20)
`DIV`/`MOD` with `AX == 0` yield `AX = 0` (no trap). Signed, truncate-toward-zero
(`q = -(|a|//|b|)` when signs differ; `r = a - q*b`). Listed for completeness;
`MUL/DIV/MOD` are **not** in the minimal subset (§7).

### 4.3 PC out of range
If the fetched instruction index is `< 0` or `>= len(program)`, the VM halts
(same as EXIT, exit code = current AX). This is how a program that "falls off the
end" terminates.

### 4.4 Masking & endianness
- All register writes: `& 0xFFFFFFFF`.
- All multi-byte memory access: **little-endian**.
- Comparison/DIV/MOD signedness: reinterpret 32-bit as two's complement.
- Shift count: `AX & 0x1F`.

---

## 5. Calling convention (JSR / ENT / LEV) — end to end

C4 uses a classic descending-stack frame. A call site and callee cooperate as
follows (operands `PSH`/`LEA`/`ADJ` shown for context though only `JSR/ENT/LEV`
are in the core list):

**Caller (call site):**
1. Evaluate & `PSH` each argument (args end up on the stack, deepest first).
2. `JSR target` — pushes the return address (`PC+8`), jumps to `target`.
3. After return, `ADJ n` — pop the `n` argument bytes off the stack
   (`SP += n`). (`ADJ` = op 7; SP += imm.)

**Callee (function prologue):**
4. `ENT locals` — push caller BP, set `BP = SP`, reserve `locals` bytes.
   The frame now looks like (high→low address):
   ```
   ... args ...            (pushed by caller, at BP+16, BP+24, ... via LEA)
   [ return address ]      (pushed by JSR)      @ BP+8
   [ saved caller BP ]     (pushed by ENT)      @ BP         <- BP points here
   [ local var slots ]     (reserved by ENT)    below BP
   ```
5. Locals/args are addressed via `LEA` (op 0: `AX = BP + imm`) then `LI`/`SI`.
   Positive offsets from BP reach args + return addr; negative reach locals.

**Callee (epilogue):**
6. Return value in `AX`.
7. `LEV` — `SP = BP`; pop saved BP into BP; pop return address into PC. Control
   returns to the caller instruction after `JSR`.

**Frame teardown ordering is exactly:** `SP=BP` → restore BP → restore PC. This
mirror-images `ENT`'s `push BP → BP=SP → SP-=imm`, so a matched `ENT`/`LEV` pair
restores SP, BP, and PC to their pre-call values (minus the args, which the
caller's `ADJ` reclaims).

---

## 6. Program start / HALT contract

**Start.** Execution begins with `PC = code_base` (default 0),
`SP = BP = 0x100000`, `AX = 0`. The first instruction is `program[0]`. The
program image is the BYTECODE section; DATA is preloaded at `data_base = 0x10000`
(`memory[data_base + i] = data[i]`).

**Halt.** The program terminates on **any** of:
1. `EXIT` (op 38) — explicit halt; exit code = `AX`.
2. `LEV` that returns from `main` — i.e. after `SP = BP` and the BP-pop, the next
   stack slot (the return-address slot) is unwritten → halt; exit code = `AX`.
3. PC out of range (fall off the end of the code) — halt; exit code = `AX`.

There is **no distinct `HALT` opcode.** In the neural VM, halting emits the
end-of-stream / halt token (`docs/BLOG_SPEC.md` §"until completion"); at the ISA
level "HALT" ≡ `EXIT`. The exit code is always the final `AX` masked to 32 bits.

---

## 7. What the minimal 8-bit subset drops vs the full ISA

The full VM is 32-bit and has 46 opcodes. The minimal subset keeps the ~24
opcodes in §2 and operates on **8-bit values** (`0..0xFF`). Explicit reductions:

| Aspect | Full ISA | 8-bit minimal subset |
|--------|----------|----------------------|
| Value width | 32-bit registers, mask `& 0xFFFFFFFF` | values `0..0xFF`; results still masked to 32 bits but never exceed the low byte in practice |
| Signed CMP/shift | 32-bit two's-complement signed `LT/LE/GT/GE`; `SHR` unsigned; `AX & 0x1F` shift count | with operands `0..0xFF` (always ≥0 in 32-bit space) signed==unsigned, so CMP is plain value compare; shifts still mask count to low 5 bits |
| Multi-byte memory | `LI`/`SI` move a full 8-byte little-endian word | subset uses `LC`/`SC` (single byte) for data; `LI`/`SI` still available but only the low byte is meaningful |
| Arithmetic ops | `MUL` (27), `DIV` (28), `MOD` (29) — signed, multi-layer | **dropped** (not in core-20; add later if needed) |
| Stack/addr helpers | `LEA` (0), `ADJ` (7), `PSH` (13) | `PSH`/`LEA`/`ADJ` are needed for the calling convention (§5) but are not "compute" ops; keep them, they are trivially 8-bit-safe |
| Syscalls / I-O | `OPEN/READ/CLOS/PRTF/MALC/FREE/MSET/MCMP` (30-37), `GETCHAR/PUTCHAR/PRINTF2` (64-66) | **dropped** entirely |
| Neural extensions | `NOP` (39), `POP` (40), `BLT` (41), `BGE` (42) — defined in the neural VM, **never emitted by the C4 compiler** | **dropped** |

Because 8-bit operands are non-negative in the 32-bit space, the only places the
"drop to 8-bit" changes an *answer* are: (a) `ADD`/`SHL` overflow past `0xFF`
(the low byte still matches the masked-32 result's low byte, so a pure-8-bit
implementation is bit-exact for the low byte), and (b) `SUB` underflow (an 8-bit
VM wraps to `0..0xFF`; the 32-bit oracle wraps to `0xFFFFFF..`; **the low byte
agrees**). The oracle compares the register **byte tokens**, so an 8-bit
implementation that reproduces the low byte of each masked-32 result is
byte-identical for the subset programs.

---

## 8. Coverage note

**Fully specified (encoding + state transition + example, both references
agree):** IMM, ADD, SUB, AND, OR, XOR, SHL, SHR, EQ, NE, LT, LE, GT, GE, LI, SI,
JMP, BZ, BNZ, JSR, ENT, LEV, EXIT — 23 of the requested 24. Plus the
calling-convention helpers LEA/ADJ/PSH and the register/memory/start/halt model.

**LC / SC (10 / 12):** specified from `nibble_bytecode_executor` only — the
primary `dim_oracle` reference does **not** implement char load/store. If the
green-field oracle is built on `dim_oracle`, LC/SC will be untested there;
validate them against `nibble_bytecode_executor` (or extend the oracle).

**HALT:** resolved to `EXIT` (op 38); no distinct opcode exists (§6).

**Ambiguities / divergences found in the reference (all resolved in favor of
`dim_oracle` = C4-faithful):**
1. **Comparison signedness** — `dim_oracle` signed vs `nibble_bytecode_executor`
   unsigned. Spec follows signed. (Irrelevant inside the 8-bit subset; matters at
   full 32-bit.)
2. **Shift count masking** — `dim_oracle` masks `AX & 0x1F`;
   `nibble_bytecode_executor` shifts by full AX. Spec follows the mask.
3. **`SHR` sign** — both do logical (unsigned) right shift; C4 int is signed but
   the reference VMs implement logical shift. Documented as logical.
4. **Initial SP** — reference VMs use `0x100000`; the blog prose says
   `0x10000/0x8000`. Spec follows the reference (`0x100000`) since the oracle
   uses it. **This must match or PC/SP byte tokens diverge.**
5. **`SI`/`SC` stack-vs-data split** — writes to `addr < data_base` (`0x10000`)
   go to the word-`stack`; `>=` go to byte `memory`. Only `nibble_bytecode_executor`
   models this split; `dim_oracle` always writes byte memory. For subset programs
   that store to legitimate data addresses (`>= 0x10000`) both agree.
6. **DIV/MOD** — signed, truncate-toward-zero, `/0 → 0`. Documented for
   completeness but out of the minimal subset.

**Not applicable to the minimal subset:** MUL/DIV/MOD, all syscalls/I-O, and the
neural-only NOP/POP/BLT/BGE.
