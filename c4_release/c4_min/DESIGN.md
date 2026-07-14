# c4_min — green-field minimal-ISA compiler + model (contract)

This document is the contract between the three green-field pieces. The
oracle harness (`oracle.py` + `run_oracle.py`) is built against the
**Model interface** section below; when the real `compile.py` / `model.py`
land they must satisfy exactly that interface and the harness plugs straight
in with no changes.

> NOTE: this file is authored by the oracle-harness lane to pin down the
> interface it depends on. The compiler lane owns the rest of the design;
> if that lane already shipped a `DESIGN.md`, MERGE the "Model interface"
> section into it verbatim — the harness imports against these signatures.

## The ISA (reference semantics)

The minimal ISA is the c4 instruction set. The authoritative reference
behaviour lives in
`neural_vm/verification/symbolic_program.py::SymbolicDeclarativeProgramRunner`
— a pure-Python bytecode interpreter. The oracle harness reuses it for the
**expected** side (ground truth) only; the model-under-test side is
c4_min-only.

Instruction encoding (the c4 encoding, one 64-bit slot per instruction):

    instr = opcode | (imm << 8)

Opcode ids (subset the harness exercises):

    LEA=0 IMM=1 JMP=2 JSR=3 BZ=4 BNZ=5 ENT=6 ADJ=7 LEV=8
    LI=9 LC=10 SI=11 SC=12 PSH=13
    OR=14 XOR=15 AND=16
    EQ=17 NE=18 LT=19 GT=20 LE=21 GE=22
    SHL=23 SHR=24
    ADD=25 SUB=26 MUL=27 DIV=28 MOD=29
    EXIT=38

A program is `(bytecode, data)`:
  * `bytecode`: `list[int]` of encoded instructions.
  * `data`:     `bytes` loaded at the data segment base `0x10000`.

Registers: `AX` (accumulator / exit code), `PC`, `SP`, `BP`. Stack grows
down from `STACK_INIT=0x10000`, 8-byte slots. The program's observable
result is the **exit code** = `AX` at the time `EXIT` halts (masked to
32 bits).

## Model interface (the harness depends on THIS)

```python
# c4_min/compile.py
def compile_program(prog: Program) -> dict:
    """Lower a program into the green-field transformer's state_dict.

    ``prog`` is a ``c4_min.oracle.Program`` (``.bytecode: list[int]``,
    ``.data: bytes``). Returns a state_dict (any mapping the model's ``run``
    understands — the harness treats it as opaque).
    """

# c4_min/model.py
def run(state_dict: dict, prog: Program, *, max_steps: int = 64) -> Decoded:
    """Autoregressively decode ``prog`` under ``state_dict``.

    Returns a ``c4_min.oracle.Decoded`` with:
      * ``.exit_code: int``            — AX (mod 2**32) when EXIT halted, or None
                                          if it never halted within max_steps.
      * ``.steps: int``                — number of executed instructions.
      * ``.trace: list[tuple[int,int]]`` OPTIONAL — per-step (pc, ax) the model
                                          decoded. If provided, the harness does
                                          a full-trace comparison (like the
                                          reference full_trace gate); if omitted
                                          it compares exit_code + steps only.
      * ``.halted: bool``.
    """
```

The harness (`oracle.py`) provides `Program`, `Decoded`, and a
`decoded_from_bytes(...)` adapter so a model that returns a flat byte stream
can be wrapped into `Decoded`. A model that only produces an exit code is a
valid (weaker) conformer — the harness degrades to exit-code comparison.

## Verdict

For each generated program the harness:
  1. computes the EXPECTED `(exit_code, steps[, trace])` via the reference
     ISA VM (`oracle.expected_for_program`);
  2. `compile_program(prog)` then `model.run(state_dict, prog)`;
  3. PASS iff the model's `exit_code` (and `trace`, when present) equals the
     expected one.

Output is a per-op-class PASS/FAIL table mirroring
`tools/run_per_op_oracle.py`.
