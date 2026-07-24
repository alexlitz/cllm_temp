#!/usr/bin/env python3
"""native_fp32_vm.py — a REAL, RUNNABLE native-fp32-scalar VM + MAC/matmul kernel.

This is the *different interpreter* that ``ablate_matmul_buckets.part_b`` says it
would take to actually run an fp32 MAC (there it is only ANALYTICALLY counted at
~7 TIGHT / ~12 STACK steps/MAC; ``ref_interpret`` — the byte-faithful INTEGER VM —
cannot run fp32 at all: ``IMM``/``LEA`` mask to ``0xFF``, ``SI``/``SC`` store one
byte, AX is a 32-bit INTEGER).

What is different from ``nibble_pure_forward_complete.ref_interpret``
--------------------------------------------------------------------
* **Registers + memory hold fp32 scalars** (one IEEE-754 ``float`` value per
  slot, NOT nibble-decomposed, NOT byte-masked). AX is the single scalar
  register; it holds an fp32 for the fp ops and a plain Python int for index /
  loop-control math (a real VM keeps a separate integer file — modelled here by
  the same register being int-tagged after an ``IMM``/``ADD``/``SUB``/``LT``).
* **32-bit DIRECT addressing.** ``LEA``/``LI``/``FLI``/``SI``/``FSI`` use the
  full index — NO ``& 0xFF``, NO 4× stride, NO paged/windowed local-array walk.
  A slot address is just an index into a flat fp32 array. This removes the
  ~60/MAC byte-safe paging + 2-D-index bucket the draft VM pays.
* **Native ``FMUL`` / ``FADD`` opcodes** — real ``float`` multiply / add on the
  fp32 register, NOT an ``a*b/scale`` fixed-point ``fpmul`` function call and NOT
  a nibble reassembly. This removes the ~13/MAC call frame and the ~1/MAC DIV
  rescale. ``FMUL`` = ``AX <- AX * pop()``; ``FADD`` = ``AX <- AX + pop()`` — a
  two-input fp stack op, exactly parallel to the draft VM's ``MUL``/``ADD``.

Everything is value-exact vs ``numpy.float32`` (tolerance ~1e-5), NOT
byte-exact-integer — that is the RIGHT precision for emulating an fp32 model, and
we say so. Steps are counted one-per-executed-instruction, exactly like
``ref_interpret`` (``steps += 1`` at the top of the dispatch loop), so the
measured ``steps/MAC`` is directly comparable to the draft VM's 101.

CPU-only; the interpreter uses ``struct`` to fold each fp op to true IEEE-754
single precision (identical to ``numpy.float32`` for finite values). numpy is
used only by the tests / references.

The vanilla BAKE of these ops (``FADD``/``FMUL``/``FLI``/``FSI`` as REAL weights
inside the genuine ``blogspec_model.Transformer`` — softmax1 + ALiBi + SwiGLU +
residual — so an fp32 MAC runs BYTE-THROUGH ``model.forward``) lives in
``c4_min.native_fp32_baked`` (gated by ``C4_FP32_ALU``, default OFF; the integer
VM golden is unaffected). This module remains the interpreter that MEASURES the
steps/MAC; ``native_fp32_baked`` is the load-bearing "native fp32 is vanilla"
proof.
"""
from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

Imm = Union[int, float]


# --------------------------------------------------------------------------- #
# fp32 rounding helper                                                         #
# --------------------------------------------------------------------------- #
def f32(x: float) -> float:
    """Round a Python float to IEEE-754 single precision (what a native fp32
    register / FMUL / FADD would hold). ``struct`` pack/unpack is the standard
    exact fp32 fold — identical to ``numpy.float32(x)`` for finite values."""
    return struct.unpack("f", struct.pack("f", x))[0]


# --------------------------------------------------------------------------- #
# fp32 ISA — the minimal set to run a matmul loop (task item 1)               #
# --------------------------------------------------------------------------- #
# Integer / control opcodes reuse the c4 mnemonics; the fp opcodes (FIMM/FLI/
# FSI/FMUL/FADD) are the native-fp32 additions. Opcode numbers are LOCAL to
# this VM (not the byte-faithful isa.py numbering).
(IMM, FIMM, LEA, LI, FLI, FLIX, SI, FSI, PSH, FPSH,
 ADD, SUB, FMUL, FADD, FMACC, LT, BZ, BNZ, JMP, PRTF, HALT) = range(21)

NAMES = {
    IMM: "IMM", FIMM: "FIMM", LEA: "LEA", LI: "LI", FLI: "FLI", FLIX: "FLIX",
    SI: "SI", FSI: "FSI", PSH: "PSH", FPSH: "FPSH", ADD: "ADD", SUB: "SUB",
    FMUL: "FMUL", FADD: "FADD", FMACC: "FMACC", LT: "LT", BZ: "BZ", BNZ: "BNZ",
    JMP: "JMP", PRTF: "PRTF", HALT: "HALT",
}
BY_NAME = {v: k for k, v in NAMES.items()}

_FP_IMM_OPS = {FIMM}  # opcodes whose immediate is an fp32 literal


@dataclass
class Instr:
    op: int
    imm: Imm = 0

    def __repr__(self) -> str:
        return f"{NAMES.get(self.op, self.op)} {self.imm}"


def assemble(prog: Sequence[tuple[str, Imm] | str]) -> list[Instr]:
    """Turn ``[(name, imm), ...]`` (or bare ``name``) into a code table of Instr.

    ``FIMM`` immediates are fp32 literals (folded through :func:`f32`); every
    other immediate is a full-width Python int (slot index / PC target / loop
    count). No masking — direct 32-bit addressing.
    """
    out: list[Instr] = []
    for entry in prog:
        name, imm = (entry if isinstance(entry, tuple) else (entry, 0))
        op = BY_NAME[name]
        imm = f32(float(imm)) if op in _FP_IMM_OPS else imm
        out.append(Instr(op, imm))
    return out


# --------------------------------------------------------------------------- #
# the native-fp32 interpreter (mirrors ref_interpret's dispatch structure)    #
# --------------------------------------------------------------------------- #
def native_fp32_interpret(
    code: Sequence[Instr],
    mem: dict[int, Imm] | None = None,
    max_steps: int = 5_000_000,
    out: list[float] | None = None,
    count_ops: bool = False,
) -> tuple[list[Imm], int, dict[str, int] | None]:
    """Run the native-fp32 VM. Returns ``(trace, steps, hist)``.

    ``trace`` is the AX value after each executed step. ``steps`` counts one per
    executed instruction (``steps += 1`` at the loop top, exactly like the draft
    VM ``ref_interpret``). ``hist`` (when ``count_ops``) is the per-opcode
    execution histogram, used by the steps/MAC ablation.

    ISA
    ---
    ==========  =========================================================
    IMM  k      AX <- int k                       (index / count literal)
    FIMM x      AX <- fp32 x                       (fp literal)
    LEA  k      AX <- int k                        (base slot index, DIRECT)
    LI   s      AX <- int mem[s]                   (direct int load, imm addr)
    FLI  s      AX <- fp32 mem[s]                  (direct fp load, imm addr)
    FLIX        AX <- fp32 mem[AX]                 (indexed fp load, AX addr)
    SI   s      mem[s] <- AX
    FSI  s      mem[s] <- fp32 AX
    PSH         push AX (int)
    FPSH        push fp32 AX
    ADD  k      AX <- int AX + k                   (index math)
    SUB  k      AX <- int AX - k
    FMUL        AX <- fp32 AX * pop()              (native fp multiply)
    FADD        AX <- fp32 AX + pop()              (native fp add)
    FMACC       AC <- fp32 AC + AX * pop()         (fused mul-acc; AC reg)
                AC is initialised to 0.0 at start and moved to AX by the kernel
                via a following ``FADD`` of a pushed AC — but simplest: FMACC
                keeps the running sum in AC and the kernel reads it back with a
                dedicated move (imm=1 -> AX <- AC ; imm=2 -> AC <- 0.0).
    LT   k      AX <- (int AX < k) ? 1 : 0
    BZ   t      if AX == 0: pc <- t
    BNZ  t      if AX != 0: pc <- t
    JMP  t      pc <- t
    PRTF        emit fp32 AX to ``out`` (visible output; AX unchanged)
    HALT        stop
    ==========  =========================================================

    Memory (``mem``) maps a **direct 32-bit slot index** -> value (a real flat
    array; a sparse dict here only to avoid pre-sizing). NO byte mask, NO
    4×-stride, NO paging.
    """
    mem = {} if mem is None else mem
    vstk: list[Imm] = []
    ax: Imm = 0
    ac: float = 0.0          # dedicated fp32 accumulator register (for FMACC)
    pc = 0
    steps = 0
    trace: list[Imm] = []
    hist: dict[str, int] = {}

    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        pc += 1
        if count_ops:
            hist[NAMES[op]] = hist.get(NAMES[op], 0) + 1

        if op == IMM:                       # AX <- int immediate
            ax = int(imm)
        elif op == FIMM:                    # AX <- fp32 immediate
            ax = f32(imm)
        elif op == LEA:                     # AX <- base slot index (DIRECT)
            ax = int(imm)
        elif op == LI:                      # AX <- int mem[imm]  (direct addr)
            ax = int(mem.get(int(imm), 0))
        elif op == FLI:                     # AX <- fp32 mem[imm]  (direct addr)
            ax = f32(mem.get(int(imm), 0.0))
        elif op == FLIX:                    # AX <- fp32 mem[AX]  (indexed addr)
            ax = f32(mem.get(int(ax), 0.0))
        elif op == SI:                      # mem[imm] <- AX
            mem[int(imm)] = ax
        elif op == FSI:                     # mem[imm] <- fp32 AX
            mem[int(imm)] = f32(ax)
        elif op == PSH:                     # push AX (int)
            vstk.append(ax)
        elif op == FPSH:                    # push fp32 AX
            vstk.append(f32(ax))
        elif op == ADD:                     # int index add
            ax = int(ax) + int(imm)
        elif op == SUB:
            ax = int(ax) - int(imm)
        elif op == FMUL:                    # AX <- AX * pop()   (native fp)
            ax = f32(f32(ax) * f32(vstk.pop()))
        elif op == FADD:                    # AX <- AX + pop()   (native fp)
            ax = f32(f32(ax) + f32(vstk.pop()))
        elif op == FMACC:                   # dedicated-register fused mul-acc
            if imm == 1:                    #   AX <- AC   (read accumulator)
                ax = ac
            elif imm == 2:                  #   AC <- 0.0  (reset accumulator)
                ac = 0.0
            else:                           #   AC <- AC + AX * pop()  (the FMA)
                ac = f32(ac + f32(f32(ax) * f32(vstk.pop())))
        elif op == LT:                      # AX <- (AX < imm) ? 1 : 0
            ax = 1 if int(ax) < int(imm) else 0
        elif op == BZ:
            pc = int(imm) if ax == 0 else pc
        elif op == BNZ:
            pc = int(imm) if ax != 0 else pc
        elif op == JMP:
            pc = int(imm)
        elif op == PRTF:                    # visible output: emit fp32 AX
            if out is not None:
                out.append(f32(ax))
        elif op == HALT:
            trace.append(ax)
            break
        else:
            raise NotImplementedError(f"op {NAMES.get(op, op)} not in fp32 ISA")
        trace.append(ax)
    return trace, steps, (hist if count_ops else None)


# =========================================================================== #
# fp32 MAC / dot / matvec / matmul KERNELS  (task item 2)                      #
# =========================================================================== #
# Memory layout for the STACK-form dot: [ a[0..K-1] | b[0..K-1] | acc | i ].
# The inner loop is the honest fpmul-free MAC:
#     FLIX a[i]  (AX=A+i ; FLIX -> a[i])   FPSH
#     FLIX b[i]  (AX=B+i ; FLIX -> b[i])   FMUL     -> product in AX
#     acc += product : FLI acc ; FADD(push product first) via stack
# We hand-emit the opcode stream (a tiny assembler would do the same).


def _dot_slots(K: int) -> tuple[int, int, int, int]:
    """Return (A_base, B_base, ACC_slot, I_slot) for a length-K dot."""
    return 0, K, 2 * K, 2 * K + 1


def dot_mem(a: Sequence[float], b: Sequence[float]) -> dict[int, float]:
    """Initial memory for a dot kernel: a at [0..K-1], b at [K..2K-1]."""
    assert len(a) == len(b)
    K = len(a)
    A, B, _ACC, _I = _dot_slots(K)
    mem: dict[int, float] = {}
    for i in range(K):
        mem[A + i] = f32(float(a[i]))
        mem[B + i] = f32(float(b[i]))
    return mem


def dot_kernel_loop(K: int) -> list[Instr]:
    """STACK / looped length-K fp32 dot ``acc = sum_k a[k]*b[k]``, PRTF acc.

    Per-MAC inner-loop body (the honest c4-style stack machine, one scalar reg
    AX + a value stack + direct 32-bit indexed fp load ``FLIX``):

        LI i ; ADD A ; FLIX        AX <- a[i]            (indexed fp load)
        FPSH                       push a[i]
        LI i ; ADD B ; FLIX        AX <- b[i]
        FMUL                       AX <- a[i]*b[i]       (native fp)
        FPSH                       push product
        FLI acc                    AX <- acc
        FADD                       AX <- acc + product   (native fp)
        FSI acc                    acc <- AX
        LI i ; ADD 1 ; SI i        i <- i + 1            (index math)
        LI i ; LT K ; BNZ body     loop test + branch

    Marginal steps/MAC is a MEASURED constant (the K-sweep slope), directly
    comparable to the draft VM's 101.
    """
    A, B, ACC, IVAR = _dot_slots(K)
    code: list[tuple[str, Imm]] = []
    # prologue: acc = 0.0 ; i = 0
    code += [("FIMM", 0.0), ("FSI", ACC)]
    code += [("IMM", 0), ("SI", IVAR)]
    body = len(code)                       # first instr of the loop body
    code += [("LI", IVAR), ("ADD", A), ("FLIX", 0)]   # AX <- a[i]
    code += [("FPSH", 0)]
    code += [("LI", IVAR), ("ADD", B), ("FLIX", 0)]   # AX <- b[i]
    code += [("FMUL", 0)]                              # AX <- a[i]*b[i]
    code += [("FPSH", 0)]                              # push product
    code += [("FLI", ACC), ("FADD", 0)]               # AX <- acc + product
    code += [("FSI", ACC)]                            # acc <- AX
    code += [("LI", IVAR), ("ADD", 1), ("SI", IVAR)]  # i <- i+1
    code += [("LI", IVAR), ("LT", K), ("BNZ", body)]  # if i<K goto body
    code += [("FLI", ACC), ("PRTF", 0)]               # emit acc
    code += [("HALT", 0)]
    return assemble(code)


def dot_kernel_tight(K: int) -> list[Instr]:
    """TIGHT / fully-unrolled register-allocated length-K fp32 dot, PRTF acc.

    No loop control, no memory-resident loop var, accumulator kept LIVE in the
    ACC slot via a straight-line fold. Per MAC:
        FLI a[k] ; FPSH ; FLI b[k] ; FMUL ; FPSH ; FLI acc ; FADD ; FSI acc
    (8 straight-line ops/MAC). A production compiler that keeps acc in a second
    register would shave the acc load/store; we keep the single-scalar-reg model
    honest and MEASURE the real slope. Slots use direct constant addresses (the
    unroll knows every k), so there is NO index math at all.
    """
    A, B, ACC, _I = _dot_slots(K)
    code: list[tuple[str, Imm]] = []
    code += [("FIMM", 0.0), ("FSI", ACC)]             # acc = 0.0
    for k in range(K):
        code += [("FLI", A + k), ("FPSH", 0)]         # push a[k]
        code += [("FLI", B + k), ("FMUL", 0)]         # AX <- a[k]*b[k]
        code += [("FPSH", 0), ("FLI", ACC), ("FADD", 0), ("FSI", ACC)]  # acc += prod
    code += [("FLI", ACC), ("PRTF", 0), ("HALT", 0)]
    return assemble(code)


def dot_kernel_tight_reg(K: int) -> list[Instr]:
    """TIGHT register-allocated length-K fp32 dot using the fused ``FMACC`` op —
    the accumulator lives in a DEDICATED register (``AC``), so there is NO acc
    load/store round-trip. Per MAC:
        FLI a[k] ; FPSH ; FLI b[k] ; FMACC        (4 straight-line ops/MAC)
    ``FMACC`` does ``AC <- AC + AX*pop()``. This is the honest ISA a compiler
    that keeps the accumulator in a register emits — it MEASURES BELOW the
    analytic ~7/MAC (there is no fixed-point rescale, no fpmul call, no paging,
    and the acc never touches memory). Value-exact fp32."""
    A, B, _ACC, _I = _dot_slots(K)
    code: list[tuple[str, Imm]] = []
    code += [("FMACC", 2)]                            # AC <- 0.0  (reset)
    for k in range(K):
        code += [("FLI", A + k), ("FPSH", 0)]         # push a[k]
        code += [("FLI", B + k), ("FMACC", 0)]        # AC <- AC + a[k]*b[k]
    code += [("FMACC", 1), ("PRTF", 0), ("HALT", 0)]  # AX <- AC ; emit
    return assemble(code)


# --------------------------------------------------------------------------- #
# matvec / matmul over the dot kernel                                         #
# --------------------------------------------------------------------------- #
#: kernel builders by mode name. "loop" = STACK looped; "tight" = unrolled with
#: memory accumulator; "tight_reg" = unrolled with dedicated-register FMACC.
KERNELS = {
    "loop": dot_kernel_loop,
    "tight": dot_kernel_tight,
    "tight_reg": dot_kernel_tight_reg,
}


def run_dot(a: Sequence[float], b: Sequence[float], mode: str = "loop"
            ) -> tuple[float, int]:
    """Run a length-K fp32 dot on the VM; return (result, steps).
    ``mode`` in {"loop", "tight", "tight_reg"}."""
    K = len(a)
    code = KERNELS[mode](K)
    mem = dot_mem(a, b)
    out: list[float] = []
    _tr, steps, _h = native_fp32_interpret(code, mem=mem, out=out)
    assert len(out) == 1, out
    return out[0], steps


def run_matvec(mat: Sequence[Sequence[float]], vec: Sequence[float],
               mode: str = "loop") -> tuple[list[float], int]:
    """fp32 matrix-vector product ``y = mat @ vec`` (M rows, K cols). Each output
    element is one dot run; total steps summed. Returns (y, total_steps)."""
    y: list[float] = []
    total = 0
    for row in mat:
        r, s = run_dot(row, vec, mode=mode)
        y.append(r)
        total += s
    return y, total


def run_matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]],
               mode: str = "loop") -> tuple[list[list[float]], int]:
    """fp32 matmul ``C = A @ B``. A is M×K, B is K×N. Each C[i,j] is one length-K
    dot of A row i with B column j. Returns (C, total_steps)."""
    M, K = len(A), len(A[0])
    N = len(B[0])
    assert len(B) == K, (len(B), K)
    C = [[0.0] * N for _ in range(M)]
    total = 0
    for i in range(M):
        for j in range(N):
            col = [B[k][j] for k in range(K)]
            r, s = run_dot(A[i], col, mode=mode)
            C[i][j] = r
            total += s
    return C, total
