"""c4_min COMPILE-THEN-EXECUTE HANDOFF: a transformer that produces bytecode in
memory and then runs it — the mechanism behind "Model that Directly Runs C Code".

THE HEADLINE GOAL (BLOG_SPEC.md §"Model that Directly Runs C Code"): C source in
-> the transformer parses it to bytecode -> the transformer runs that bytecode,
all with no tool calls. The two proven halves are:

  * BAKING (``nibble_bake.py``): a *fixed* program compiled into the WEIGHTS
    (eq-check -> AND -> one-hot -> MoE-value read-only code segment).
  * UNIVERSAL FETCH (``universal.py``): ONE interpreter whose FETCH reads the
    instruction at PC out of DATA MEMORY (``WORD = sum_i PC_IS[i]*CODE_WORD[i]``).

The missing link is the HANDOFF between them: the compiler half must WRITE the
bytecode it produces into memory, and the interpreter half must then FETCH that
freshly-produced bytecode and execute it. This module proves exactly that link.

The mechanism (self-modifying code = compile-then-execute)
---------------------------------------------------------
We extend the universal PACKED interpreter with ONE new opcode, ``EMIT`` (store an
instruction word into CODE MEMORY at the immediate slot address):

    EMIT imm:  CODE_WORD[imm] := AX + 256*STACK0     # assemble + write the word
               PC += 1                               # imm = target code slot

The produced instruction word is ``op | imm<<8``; we assemble it from two 8-bit
values the generator computes — the low byte (opcode) in ``AX`` and the high byte
(the produced instruction's immediate) in ``STACK0`` — because the VM's arithmetic
folds AX mod-256, so no single 8-bit register can hold the full 16-bit word. The
TARGET slot is the EMIT instruction's own immediate (the "output program counter",
exactly like the c4 compiler's ``code_pos``). ``EMIT`` is a store-SELECT — the
mirror image of the fetch-SELECT: where the fetch reads ``WORD = sum_i
PC_IS[i]*CODE_WORD[i]`` (a PC-one-hot dotted with the code bands), ``EMIT`` writes
``CODE_WORD[i] += ADDR_IS[i]*((AX+256*STACK0) - CODE_WORD[i])`` (an
immediate-address one-hot that SETs the addressed code cell). Both are bilinear
reads/writes over the SAME CODE_WORD memory band; fetch and store share the
memory.

With ``EMIT`` in the ISA, a program can:

  1. COMPILE: compute instruction words (op | imm<<8) in AX and ``EMIT`` them into
     empty CODE slots ``>= gen_end`` (the "code we are producing").
  2. HANDOFF: ``JMP gen_end`` — hand control to the freshly-produced code.
  3. EXECUTE: the universal fetch at PC=gen_end reads the words step (1) just wrote
     into memory and runs them. THE FETCH READS THE FRESHLY-PRODUCED BYTECODE.

Proof surfaces (see ``test_handoff.py``):
  * The produced code slots start EMPTY (CODE_WORD == 0) in the initial state —
    the bytecode that runs is NOT present at load time; it is produced at runtime.
  * After the run, those slots hold the produced words, and the trace matches an
    independent reference interpreter run on the *produced* program.
  * The generator can be BAKED (``nibble_bake``-style, program-in-weights) so the
    ENTIRE thing needs no bytecode in the input at all — a "compiler in the
    weights" that writes a program into memory and runs it.

Full C compiler
---------------
The in-repo c4 compiler (``bundler/c4_compile.c``, 964 LOC) emits exactly this
``code[code_pos] = op | (imm<<8)`` packed word encoding into a ``code[]`` array —
i.e. the c4 compiler's code-emission IS a sequence of ``EMIT``s into CODE memory.
Baking that compiler's ~few-thousand-instruction bytecode as the generator (via
``universal.py``'s program-in-data or ``nibble_bake``'s program-in-weights) and
letting it ``EMIT`` the compiled program then ``JMP`` to it is the model-runs-C
path at full scale; feasibility is scoped in NIBBLE_MODEL_RUNS_C_2026_07_14.md.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from .compiler import VOCAB, _zero_attn, _load_ffn, _load_head, head_matrix
from .compile_ffn import compile_ffn, compile_fold, S as FFN_S
from .dsl import FFNRule, LinearExpr
from .layout import Layout
from .model import Transformer
from . import universal as U

# New opcode value for "emit an instruction word into code memory". We pick a
# free slot in the opcode one-hot region (NUM_OPS=40 covers 0..39; 27 is unused
# by the 8-bit subset in isa.py — see isa.NAMES which skips 27).
EMIT = 27
isa.NAMES.setdefault(EMIT, "EMIT")
isa.BY_NAME.setdefault("EMIT", EMIT)


# ------------------------------------------------------------------ layout ----

def build_handoff_layout(code_size: int, n_heads: int = 4) -> Layout:
    """Universal PACKED layout + a store-address one-hot ``ADDR_IS`` for EMIT.

    Identical to ``universal.build_universal_layout(packed=True)`` except we add
    ``ADDR_IS[i]`` (the store-address one-hot the EMIT store-select routes on) and
    materialise ``STACK0`` as the store address. The CODE_WORD band is shared
    read/write memory: the fetch reads it, EMIT writes it.
    """
    L = Layout(n_heads=n_heads)
    L.CODE_SIZE = code_size
    L.PACKED = True
    L.CODE_WORD = [L._band(f"CODE_WORD_{i}", 1) for i in range(code_size)]
    L.WORD = L._band("WORD", 1)
    L.PC_IS = [L._band(f"PC_IS_{i}", 1) for i in range(code_size)]
    L.ADDR_IS = [L._band(f"ADDR_IS_{i}", 1) for i in range(code_size)]  # EMIT store addr
    L.OP_VAL = L._band("OP_VAL", 1)
    L.OP_IS = L._band("OP_IS", isa.NUM_OPS)
    L.OUT_SLOTS = [L._band("OUT_0", 1)]
    L.HALT_SEEN = [L._band("HSEEN_0", 1)]
    while L._off % n_heads != 0:
        L._band(f"_pad2_{L._off}", 1)
    L.D = L._off
    return L


# --------------------------------------- store-address one-hot (ADDR_IS) -------

def compile_store_addr_onehot(L: Layout, dim: int):
    """ADDR_IS[i] = (IMM == i): the store-address one-hot for EMIT.

    Same exact triangular-pulse gadget the fetch uses for the PC one-hot, applied
    to IMM (the EMIT instruction's own immediate == the target slot / "output PC").
    SET semantics (self-clear first).
    """
    from .compile_ffn import RELU_S
    n = L.CODE_SIZE
    silu_S = float(F.silu(torch.tensor(FFN_S)))
    thresholds = list(range(-1, n + 1))
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_relu = len(thresholds)
    clear0 = n_relu
    n_units = clear0 + n
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    for t, j in thr_unit.items():
        W_up[j, L.IMM] = RELU_S; b_up[j] = -RELU_S * t; W_gate[j, L.ONE] = 1.0
    for c in range(n):
        u = clear0 + c
        W_up[u, L.ONE] = FFN_S; W_gate[u, L.ADDR_IS[c]] = 1.0
        W_down[L.ADDR_IS[c], u] += -1.0 / silu_S
    for i in range(n):
        W_down[L.ADDR_IS[i], thr_unit[i - 1]] += 1.0 / RELU_S
        W_down[L.ADDR_IS[i], thr_unit[i]] += -2.0 / RELU_S
        W_down[L.ADDR_IS[i], thr_unit[i + 1]] += 1.0 / RELU_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# --------------------------------------- EMIT store-select into code memory ----

def compile_emit_store(L: Layout, dim: int):
    """EMIT: ``CODE_WORD[IMM] := AX + 256*STACK0``, gated on decoded OP_IS[EMIT].

    Store-SELECT (the write-mirror of the fetch's read-select):

        CODE_WORD[i] += OP_IS[EMIT] * ADDR_IS[i] * ((AX + 256*STACK0) - CODE_WORD[i])

    - ``OP_IS[EMIT]``: fires only when the current opcode is EMIT.
    - ``ADDR_IS[i]``:  the store-address one-hot (== IMM; exactly one hot).
    - ``(AX + 256*STACK0 - CODE_WORD[i])``: SET the cell to the assembled word,
      low byte from AX (opcode), high byte from STACK0 (the produced immediate).

    Three SwiGLU product units per slot: ``+guard*AX``, ``+guard*256*STACK0``, and
    ``-guard*CODE_WORD[i]`` (guard == OP_IS[EMIT] AND ADDR_IS[i]). The guard is a
    2-input AND baked into the sharp-silu ``up`` gate; the value rides the linear
    ``gate``. For every slot NOT addressed (or when the op isn't EMIT) all units are
    ~0, so memory is untouched — a clean single-cell store. Precision: the value
    reaches 65535, so we normalise with the exact power-of-two ``POW2=256`` scale
    (``silu(256)=256`` and ``1/256`` exact in fp32) rather than the generic silu(S).

    We also bump PC += 1 here (unconditional on OP_IS[EMIT]).
    """
    n = L.CODE_SIZE
    BIG = 256.0                                  # exact power-of-two gate scale
    # guard sum in {0,1,2}; up = BIG*(sum-1.5) -> ON=+0.5*BIG=128, OFF<=-128.
    # silu(128)==128 exactly, silu(-128)==0 in fp32, so 1/silu_big==1/128 is exact.
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))   # == 128.0 exactly
    op_emit = L.OP_IS + EMIT
    # 3 store units per slot + 1 PC bump
    n_units = 3 * n + 1
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    def _and_unit(u, addr_band, gate_terms, gate_const, dst, coeff):
        # up = BIG*(OP_IS[EMIT] + ADDR_IS[i] - 1.5): +0.5BIG iff BOTH hold.
        # silu(0.5*BIG)=silu(128)~=128; but we want the ON hidden == BIG so the
        # 1/silu_big==1/256 reciprocal is exact. Use up bias so ON up == BIG.
        W_up[u, op_emit] += BIG
        W_up[u, addr_band] += BIG
        b_up[u] += -BIG * 1.5
        for band, coeff_g in gate_terms:
            W_gate[u, band] += coeff_g
        b_gate[u] += gate_const
        W_down[dst, u] += coeff / silu_big

    u = 0
    for i in range(n):
        # + guard * AX          (low byte: opcode)
        _and_unit(u, L.ADDR_IS[i], [(L.AX, 1.0)], 0.0, L.CODE_WORD[i], +1.0); u += 1
        # + guard * 256*STACK0  (high byte: produced immediate)
        _and_unit(u, L.ADDR_IS[i], [(L.STACK0, 256.0)], 0.0, L.CODE_WORD[i], +1.0); u += 1
        # - guard * CODE_WORD[i]   (SET: clear the old cell value)
        _and_unit(u, L.ADDR_IS[i], [(L.CODE_WORD[i], 1.0)], 0.0, L.CODE_WORD[i], -1.0); u += 1

    # PC += 1 on EMIT (guard = OP_IS[EMIT] only; single-band guard -> S gate).
    W_up[u, op_emit] += FFN_S
    b_up[u] += -FFN_S * 0.5
    W_gate[u, L.ONE] += 1.0
    silu_S = float(F.silu(torch.tensor(0.5 * FFN_S)))
    W_down[L.PC, u] += 1.0 / silu_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ------------------------------------------------------- the handoff step ------

def build_handoff_step(code_size: int, n_heads: int = 4, max_pos: int = 4):
    """Universal PACKED interpreter + EMIT (store-to-code-memory). ONE model.

    Physical FFN sub-blocks (single residual position, no attention):
      1. fetch      : PC -> PC_IS[i] + AX_ZERO.
      2. word_select: WORD = sum_i PC_IS[i]*CODE_WORD[i]   (read code memory).
      3a. word>>8   : IMM = WORD>>8.
      3b. word&0xff : OP_VAL = WORD&0xFF.
      3c. decode    : OP_VAL -> OP_IS[op] one-hot (now includes EMIT).
      3d. addr1hot  : ADDR_IS[i] = (IMM == i)  (EMIT target slot == its immediate).
      4. dispatch   : universal per-opcode rules (IMM/LEA/PSH/ADD/SUB/JMP/HALT).
      5. emit_store : EMIT -> CODE_WORD[IMM] := AX+256*STACK0 (WRITE code mem) + PC+1.
      6. branch     : BZ/BNZ bilinear PC update.
      7. fold+emit  : AX mod-256, AX -> OUT slot, HALTED -> HALT_SEEN.

    Block 5 is the ONLY addition vs ``universal.build_universal_step(packed=True)``;
    everything else is byte-identical universal machinery. CODE_WORD is the shared
    memory both the fetch (read) and EMIT (write) touch. Returns ``(model, L)``.
    """
    L = build_handoff_layout(code_size, n_heads=n_heads)
    dim = L.D

    ffn_specs = [
        U.compile_fetch_select(L, dim),          # 1. PC one-hot + AX_ZERO
        U.compile_word_select(L, dim),           # 2. WORD <- code memory at PC
        U.compile_word_decode_imm(L, dim),       # 3a. IMM = WORD>>8
        U.compile_word_decode_op(L, dim),        # 3b. OP_VAL = WORD&0xFF
        U.compile_opcode_decode(L, dim),         # 3c. OP_VAL -> OP_IS[op]
        compile_store_addr_onehot(L, dim),       # 3d. ADDR_IS[i] = (IMM==i)
        compile_ffn(U.universal_dispatch_rules(L), dim),  # 4. dispatch
        compile_emit_store(L, dim),              # 5. EMIT -> write code memory
        U.compile_branch_delta(L, dim),          # 6. BZ/BNZ bilinear PC
        compile_fold(L.AX, L.ONE, dim, modulus=256),      # 7a. AX mod-256
        compile_ffn([                            # 7b. emit + halt snapshot
            FFNRule([(L.ONE, 0.5, 1.5)], {L.OUT_SLOTS[0]: LinearExpr.of(L.AX, 1.0)
                                          + LinearExpr.of(L.OUT_SLOTS[0], -1.0)}),
            FFNRule([(L.ONE, 0.5, 1.5)], {L.HALT_SEEN[0]: LinearExpr.of(L.HALTED, 1.0)
                                          + LinearExpr.of(L.HALT_SEEN[0], -1.0)}),
        ], dim),
    ]
    n_blocks = len(ffn_specs)
    hidden = max(f["W_up"].shape[0] for f in ffn_specs)
    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, max_pos=max_pos, vocab=VOCAB)
    with torch.no_grad():
        model.embed.zero_()
        model.embed[0, L.ONE] = 1.0
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
        _load_head(model, L)
    return model, L


# ------------------------------------------------------------ EMIT semantics ---

def interpret_with_emit(code: List[isa.Instr], code_size: int,
                        mem_size: int = 256, max_steps: int = 4096):
    """Reference interpreter that mirrors the handoff model exactly, incl. EMIT.

    EMIT imm: ``code[imm] = (AX & 0xFF) | ((STACK0 & 0xFF) << 8)`` — assembles the
    produced instruction word (low byte AX = opcode, high byte STACK0 = immediate)
    and writes it into the CODE table at slot ``imm`` (self-modifying / code-gen).
    The CODE table IS the memory; fetch reads it, EMIT writes it. Returns (per-step
    AX trace, final code table) so tests can assert the trace AND produced bytecode.
    """
    ax = sp = bp = 0
    sp = mem_size
    pc = 0
    stack = [0] * (mem_size + 1)
    # code table as words (op | imm<<8), zero-padded to code_size
    words = [0] * code_size
    for i, ins in enumerate(code):
        words[i] = (ins.op & 0xFF) | ((ins.imm & 0xFF) << 8)
    emitted = []

    def push(v):
        nonlocal sp
        sp -= 1; stack[sp] = v & 0xFF

    def pop():
        nonlocal sp
        v = stack[sp]; sp += 1; return v & 0xFF

    steps = 0
    while pc < code_size and steps < max_steps:
        steps += 1
        w = words[pc]
        op = w & 0xFF
        imm = (w >> 8) & 0xFF
        stack0 = stack[sp] if sp <= mem_size else 0
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + imm) & 0xFF
        elif op == isa.PSH:
            push(ax)
        elif op == isa.ADD:
            ax = (pop() + ax) & 0xFF
        elif op == isa.SUB:
            ax = (pop() - ax) & 0xFF
        elif op == EMIT:
            # CODE_WORD[imm] := (AX low byte) | (STACK0 high byte << 8)
            words[imm % code_size] = (ax & 0xFF) | ((stack0 & 0xFF) << 8)
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.HALT:
            emitted.append(ax); break
        else:
            raise NotImplementedError(f"op {op} not in handoff ISA")
        emitted.append(ax)
    return emitted, words


# ------------------------------------------------------------- run driver ------

def _step_once(model, state):
    x = state.view(1, 1, -1)
    for blk in model.blocks:
        x = blk(x)
    return x[0, 0]


def _requantize(state, L):
    q = torch.round(state)
    q[L.ONE] = 1.0
    return q


def load_program(model, L, code: List[isa.Instr]) -> torch.Tensor:
    """Initial state with ``code`` in CODE_WORD memory (packed words). Slots past
    the program stay 0 (empty code memory the generator will EMIT into)."""
    assert len(code) <= L.CODE_SIZE
    state = model.embed[0].clone()
    for i, ins in enumerate(code):
        state[L.CODE_WORD[i]] = float((ins.op & 0xFF) | ((ins.imm & 0xFF) << 8))
    return state


def run_handoff(model, L, code: List[isa.Instr], max_steps: int = 4096,
                requantize: bool = True, return_code: bool = False):
    """Run the handoff interpreter on ``code`` loaded as data. Returns the per-step
    AX trace; if ``return_code`` also returns the final CODE_WORD memory (so a test
    can read the freshly-EMITted bytecode straight out of the model's state)."""
    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[0], halt_band=None)
    halt_seen = L.HALT_SEEN[0]
    state = load_program(model, L, code)
    trace: List[int] = []
    for _ in range(max_steps):
        state = _step_once(model, state)
        if requantize:
            state = _requantize(state, L)
        logits = F.linear(state, W, b)
        trace.append(int(logits.argmax().item()))
        if float(state[halt_seen]) > 0.5:
            break
    if return_code:
        words = [int(round(float(state[L.CODE_WORD[i]]))) for i in range(L.CODE_SIZE)]
        return trace, words
    return trace


class HandoffMachine:
    """ONE interpreter that supports compile-then-execute: a program can EMIT new
    instructions into CODE memory and JMP to them (the universal fetch then reads
    the freshly-produced bytecode). Realises the model-runs-C handoff mechanism."""

    def __init__(self, code_size: int, n_heads: int = 4):
        self.code_size = code_size
        self.model, self.L = build_handoff_step(code_size, n_heads=n_heads)

    def run(self, prog, max_steps: int = 4096, requantize: bool = True,
            return_code: bool = False):
        code = _assemble_emit(prog)
        return run_handoff(self.model, self.L, code, max_steps=max_steps,
                           requantize=requantize, return_code=return_code)

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)


def _assemble_emit(prog) -> List[isa.Instr]:
    """assemble that also understands the EMIT mnemonic."""
    out = []
    for entry in prog:
        name, imm = (entry if isinstance(entry, tuple) else (entry, 0))
        op = EMIT if name == "EMIT" else isa.BY_NAME[name]
        out.append(isa.Instr(op, imm & 0xFF))
    return out


def make_word(op_name: str, imm: int) -> int:
    """The packed instruction word ``op | imm<<8`` for a mnemonic — what a
    'compiler' program computes in AX before EMITting it into code memory."""
    op = EMIT if op_name == "EMIT" else isa.BY_NAME[op_name]
    return (op & 0xFF) | ((imm & 0xFF) << 8)
