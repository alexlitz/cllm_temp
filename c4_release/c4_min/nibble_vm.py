"""Universal NIBBLE VM — the coherent BLOG_SPEC-faithful base.

ONE fixed-weight transformer step-block that runs ANY program supplied as DATA,
on the spec's nibble representation, emitting the 30-token register frame each
step, re-quantised by the vanilla autoregressive token round-trip (NO
``torch.round``).

It ports three proven scalar-track mechanisms onto the nibble foundation
(``blogspec_layout`` / ``blogspec_vocab`` / ``blogspec_model``):

  1. RUNTIME-PC DISPATCH  (``git show gf-assembled:.../control.py``)
        Each step reads the LIVE PC, builds the exact-integer PC one-hot
        ``PC_IS[i] == (PC == i)`` (triangular-pulse gadget), and the executed
        instruction is selected by that runtime one-hot — a forward branch that
        skips instructions Just Works because the next step dispatches on the new
        PC. Here PC is a NIBBLE band; ``compile_nibble_to_scalar`` recomposes it
        to the scalar ``PC_VAL`` the pulse operates on.

  2. UNIVERSAL FETCH-FROM-DATA  (``git show greenfield-universal:.../universal.py``)
        The program is NOT baked into the weights. It lives in DATA-MEMORY bands
        ``CODE_OP[i]`` / ``CODE_IMM[i]`` (written by ``load_program`` as INPUT).
        Fetch@PC does the bilinear select ``OP_VAL = Σ_i PC_IS[i]·CODE_OP[i]`` /
        ``IMM = Σ_i PC_IS[i]·CODE_IMM[i]`` (SwiGLU ``silu(up)·gate`` product). The
        opcode is DECODED to a one-hot ``OP_IS[op]`` and dispatch executes ONE
        fixed rule per opcode VALUE ⇒ ONE model, MANY programs (swap the data).

  3. VANILLA REQUANT  (``git show greenfield-vanilla-requant:.../recurrent_vanilla.py``)
        Between steps every register round-trips through the LM head:
        ``token = argmax_v (2·v·band − v²)`` (the integer snap, no round) then
        re-embed. On the nibble foundation this IS the 30-token register emission:
        each register byte is the argmax over 256 byte logits read from the scalar
        next-state lane, and the emitted byte token re-enters through the byte
        embedding — writing its two nibbles back into the register's NIBBLE band,
        annihilating the O(1e-6) SwiGLU residue and closing the scalar→nibble loop.

DISPATCH INTERFACE (the contract the fan-out agents target)
-----------------------------------------------------------
An opcode's effect is ONE ``FFNRule`` set gated on the DECODED opcode one-hot
``OP_IS[op]`` (see ``base_dispatch_rules``). It reads the current register value
lanes (``AX_VAL SP_VAL BP_VAL STK_VAL PC_VAL``) + the fetched ``IMM`` and writes
the next-state directly into those lanes (SET semantics: each op cancels the old
image) plus the PC delta. A new gadget (comparison / bitwise / muldiv / memory /
callconv) plugs in by APPENDING its own ``OP_IS[op]``-gated ``FFNRule`` set —
nothing else in the interpreter changes. See ``docs/NIBBLE_SKELETON_2026_07_14.md``.

The persistent VM value is the NIBBLE band; the scalar value lane is that band's
per-step image (recomposed by ``compile_nibble_to_scalar``) so the PROVEN exact
scalar dispatch algebra runs unchanged; the driver's frame round-trip writes the
scalar next-state back into the nibble bands.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .nibble_vm_layout import NibbleVMLayout
from .blogspec_model import Transformer
from .dsl import FFNRule, LinearExpr


# --- exact-integer silu scales (shared with the proven gadgets) -------------
S = 60.0        # silu identity: silu(60)≈60, silu(-60)≈0 (fp32-exact on ints)
RELU_S = 200.0  # relu-via-silu: silu(RELU_S·z)/RELU_S ≈ relu(z), exact on ints
SILU_S = float(F.silu(torch.tensor(S)))          # ≈ 60.0
SILU_HALF = float(F.silu(torch.tensor(0.5 * S)))  # ≈ 30.0

# The opcode VALUES the interpreter decodes + dispatches (the built base subset).
BASE_OPS = [isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB,
            isa.JMP, isa.BZ, isa.BNZ, isa.HALT]


# ---------------------------------------------------------------------------
# FFN spec container + the generic FFNRule compiler (SwiGLU block).
# ---------------------------------------------------------------------------
def _empty_spec(dim: int, n_units: int) -> Dict[str, torch.Tensor]:
    return {
        "W_up": torch.zeros(n_units, dim), "b_up": torch.zeros(n_units),
        "W_gate": torch.zeros(n_units, dim), "b_gate": torch.zeros(n_units),
        "W_down": torch.zeros(dim, n_units), "b_down": torch.zeros(dim),
    }


def compile_ffn(rules: List[FFNRule], dim: int) -> Dict[str, torch.Tensor]:
    """FFNRule list -> SwiGLU tensors (exact-integer gadget, from compile_ffn.py).

    Per (rule, dst): one hidden unit with gate = write-expr, up = S·(guard
    indicator), down = 1/silu(0.5S) routing ``guard·expr`` into dst.
    """
    n_units = sum(max(1, len(r.write)) for r in rules) or 1
    spec = _empty_spec(dim, n_units)
    u = 0
    for rule in rules:
        n_win = len(rule.when)
        for dst, expr in rule.write.items():
            for band, coeff in expr.terms.items():
                spec["W_gate"][u, band] += coeff
            spec["b_gate"][u] += expr.const
            for (band, lo, hi) in rule.when:
                spec["W_up"][u, band] += S
            spec["b_up"][u] += -S * (n_win - 0.5)
            spec["W_down"][dst, u] += 1.0 / SILU_HALF
            u += 1
    return spec


# ---------------------------------------------------------------------------
# (1) NIBBLE -> SCALAR recompose. value_lane = Σ_j 16^j · nibble_j.
# ---------------------------------------------------------------------------
def compile_nibble_to_scalar(L: NibbleVMLayout, dim: int,
                             hi_nibbles: int = 5) -> Dict[str, torch.Tensor]:
    """Recompose each register's nibble band into its scalar value lane (SET).

    ``VAL = Σ_{j<hi_nibbles} 16^j · nibble_j`` via silu-identity reads
    (``silu(S·nib_j)/S ≈ nib_j``, exact for nibbles 0..15) weighted by ``16^j`` in
    the down-projection. ``hi_nibbles=5`` covers the foundation range: 8-bit
    AX/STACK0 (nibbles 0..1), small PC, and SP/BP = 0x10000 (nibble 4);
    ``16^4 = 65536 < 2^24`` so the fp32 recompose is exact.

    This is the bridge that lets the PROVEN scalar dispatch algebra run on the
    spec's canonical nibble state: the nibble band is canonical; the scalar lane
    is its per-step image.
    """
    # (nibble_base, value_lane, n_read): full registers read hi_nibbles; BP_LOW
    # reads only nibbles 0,1 (the frame-pointer's low byte for the 8-bit LEA).
    pairs = [(L.PC, L.PC_VAL, hi_nibbles), (L.AX, L.AX_VAL, hi_nibbles),
             (L.SP, L.SP_VAL, hi_nibbles), (L.BP, L.BP_VAL, hi_nibbles),
             (L.STACK0, L.STK_VAL, hi_nibbles), (L.BP, L.BP_LOW, 2)]
    n_units = sum(n + 1 for _, _, n in pairs)    # per lane: 1 self-clear + n reads
    spec = _empty_spec(dim, n_units)
    u = 0
    for reg_base, val_lane, n_read in pairs:
        spec["W_up"][u, L.ONE] = S               # self-clear the old lane (SET)
        spec["W_gate"][u, val_lane] = 1.0
        spec["W_down"][val_lane, u] += -1.0 / SILU_S
        u += 1
        for j in range(n_read):
            spec["W_up"][u, L.ONE] = S            # gate passes nibble via silu-id
            spec["W_gate"][u, reg_base + j] = 1.0
            spec["W_down"][val_lane, u] += (16.0 ** j) / SILU_S
            u += 1
    return spec


# ---------------------------------------------------------------------------
# (2) FETCH: PC one-hot + AX_ZERO (block a); code-select from DATA (block b).
# ---------------------------------------------------------------------------
def compile_pc_fetch(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Block (a): PC one-hot ``PC_IS[i] = (PC_VAL == i)`` + ``AX_ZERO = (AX==0)``.

    PC one-hot is the exact-integer triangular pulse
    ``tri_i(x) = relu(x-(i-1)) - 2·relu(x-i) + relu(x-(i+1))`` (1 at x==i, 0 at
    every other integer), sharing one relu unit per threshold. ``AX_ZERO =
    relu(1 - AX_VAL)`` (1 iff AX==0). SETs (self-clears keep them idempotent
    across recurrent steps). Ported from ``control.compile_pc_fetch``.
    """
    n = L.code_size
    clear_bands = list(L.PC_IS) + [L.AX_ZERO]
    thresholds = list(range(-1, n + 1))
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_pc_relu = len(thresholds)
    az_unit = n_pc_relu
    clear0 = az_unit + 1
    n_units = clear0 + len(clear_bands)
    spec = _empty_spec(dim, n_units)

    for t, j in thr_unit.items():
        spec["W_up"][j, L.PC_VAL] = RELU_S
        spec["b_up"][j] = -RELU_S * t
        spec["W_gate"][j, L.ONE] = 1.0
    spec["W_up"][az_unit, L.AX_VAL] = -RELU_S
    spec["b_up"][az_unit] = RELU_S * 1.0
    spec["W_gate"][az_unit, L.ONE] = 1.0
    for c, band in enumerate(clear_bands):
        uu = clear0 + c
        spec["W_up"][uu, L.ONE] = S
        spec["W_gate"][uu, band] = 1.0
        spec["W_down"][band, uu] += -1.0 / SILU_S
    for i, band in enumerate(L.PC_IS):
        spec["W_down"][band, thr_unit[i - 1]] += 1.0 / RELU_S
        spec["W_down"][band, thr_unit[i]] += -2.0 / RELU_S
        spec["W_down"][band, thr_unit[i + 1]] += 1.0 / RELU_S
    spec["W_down"][L.AX_ZERO, az_unit] += 1.0 / RELU_S
    return spec


def compile_code_select(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Block (b): fetch the code cell at PC out of DATA MEMORY.

        OP_VAL = Σ_i PC_IS[i]·CODE_OP[i]      IMM = Σ_i PC_IS[i]·CODE_IMM[i]

    Bilinear product-select ``silu(S·PC_IS[i])·CODE_x[i]/silu(S)``: passes
    ``CODE_x[i]`` iff PC==i (silu(S)=S, silu(0)=0). The weights only know "PC
    one-hot times whatever data band" — the data IS the program. Ported from
    ``universal.compile_code_select``.
    """
    n = L.code_size
    n_units = 2 * n + 2
    spec = _empty_spec(dim, n_units)

    def product(u, sel_band, data_band, dst_band):
        spec["W_up"][u, sel_band] = S
        spec["W_gate"][u, data_band] = 1.0
        spec["W_down"][dst_band, u] += 1.0 / SILU_S

    u = 0
    for i in range(n):
        product(u, L.PC_IS[i], L.CODE_OP[i], L.OP_VAL); u += 1
    for i in range(n):
        product(u, L.PC_IS[i], L.CODE_IMM[i], L.IMM); u += 1
    for band in (L.OP_VAL, L.IMM):               # self-clear (SET)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, band] = 1.0
        spec["W_down"][band, u] += -1.0 / SILU_S
        u += 1
    return spec


# ---------------------------------------------------------------------------
# (3) DECODE: OP_VAL scalar -> opcode one-hot OP_IS[op].
# ---------------------------------------------------------------------------
def compile_opcode_decode(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """``OP_IS[op] = (OP_VAL == op)`` — the SAME triangular-pulse gadget as the PC
    one-hot, applied to the fetched ``OP_VAL``. Turns the dynamically-fetched
    scalar opcode into the one-hot the dispatch gates on. Only the decoded VALUES
    in ``BASE_OPS`` are materialised (others stay 0). Ported from
    ``universal.compile_opcode_decode``."""
    ops = sorted(BASE_OPS)
    thresholds = sorted({t for op in ops for t in (op - 1, op, op + 1)})
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_relu = len(thresholds)
    n_units = n_relu + len(ops)
    spec = _empty_spec(dim, n_units)

    for t, j in thr_unit.items():
        spec["W_up"][j, L.OP_VAL] = RELU_S
        spec["b_up"][j] = -RELU_S * t
        spec["W_gate"][j, L.ONE] = 1.0
    clear0 = n_relu
    for c, op in enumerate(ops):                  # self-clear each OP_IS lane
        uu = clear0 + c
        spec["W_up"][uu, L.ONE] = S
        spec["W_gate"][uu, L.OP_IS + op] = 1.0
        spec["W_down"][L.OP_IS + op, uu] += -1.0 / SILU_S
    for op in ops:
        band = L.OP_IS + op
        spec["W_down"][band, thr_unit[op - 1]] += 1.0 / RELU_S
        spec["W_down"][band, thr_unit[op]] += -2.0 / RELU_S
        spec["W_down"][band, thr_unit[op + 1]] += 1.0 / RELU_S
    return spec


# ===========================================================================
# THE DISPATCH INTERFACE — one FFNRule set per OPCODE VALUE, gated on OP_IS[op].
# Reads the current value lanes + fetched IMM, writes them (SET) + PC delta.
# A fan-out gadget plugs in by APPENDING its own OP_IS[op]-gated rules.
# ===========================================================================
def base_dispatch_rules(L: NibbleVMLayout) -> List[FFNRule]:
    """The base-op transition table on the value lanes (§dispatch interface).

    Every rule is gated on the DECODED opcode one-hot ``OP_IS[op]`` (universal —
    not a baked PC position), reads the pre-op value lanes + fetched ``IMM``, and
    writes the next-state directly into the value lanes (SET semantics: the write
    includes ``-old`` where it replaces, so the additive residual lands on exactly
    the new value). PC is updated as a DELTA (``+1`` sequential, ``IMM-PC`` for
    JMP; BZ/BNZ deferred to the bilinear ``compile_branch_delta``).
    """
    ax, sp, bp, stk, pc = L.AX_VAL, L.SP_VAL, L.BP_VAL, L.STK_VAL, L.PC_VAL
    IMM = L.IMM

    def G(op):
        return [(L.OP_IS + op, 0.5, 1.5)]         # fires iff decoded opcode == op

    rules: List[FFNRule] = []
    # IMM: AX = imm ; PC += 1
    rules.append(FFNRule(G(isa.IMM), {
        ax: LinearExpr.of(IMM, 1.0) + LinearExpr.of(ax, -1.0), pc: LinearExpr.c(1.0)}))
    # LEA: AX = (BP + imm) & 0xFF ; PC += 1  (8-bit op: add BP's LOW byte, then
    # the downstream mod-256 fold keeps AX a byte).
    rules.append(FFNRule(G(isa.LEA), {
        ax: LinearExpr.of(L.BP_LOW, 1.0) + LinearExpr.of(IMM, 1.0) + LinearExpr.of(ax, -1.0),
        pc: LinearExpr.c(1.0)}))
    # PSH: STACK0 = AX ; SP -= 4 ; PC += 1
    rules.append(FFNRule(G(isa.PSH), {
        stk: LinearExpr.of(ax, 1.0) + LinearExpr.of(stk, -1.0),
        sp: LinearExpr.c(-4.0), pc: LinearExpr.c(1.0)}))
    # ADD: AX = STACK0 + AX ; SP += 4 ; PC += 1   (fold mod 256 downstream)
    rules.append(FFNRule(G(isa.ADD), {
        ax: LinearExpr.of(stk, 1.0), sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}))
    # SUB: AX = STACK0 - AX == stk - 2·AX + 256 ; SP += 4 ; PC += 1 (fold downstream)
    rules.append(FFNRule(G(isa.SUB), {
        ax: LinearExpr.of(stk, 1.0) + LinearExpr.of(ax, -2.0) + LinearExpr.c(256.0),
        sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}))
    # JMP: PC = imm == PC += (imm - PC)
    rules.append(FFNRule(G(isa.JMP), {
        pc: LinearExpr.of(IMM, 1.0) + LinearExpr.of(pc, -1.0)}))
    # BZ / BNZ: PC update deferred to compile_branch_delta (bilinear). No write.
    rules.append(FFNRule(G(isa.BZ), {}))
    rules.append(FFNRule(G(isa.BNZ), {}))
    # HALT / EXIT: latch HALTED, freeze PC (no PC write == PC += 0).
    rules.append(FFNRule(G(isa.HALT), {L.HALTED: LinearExpr.c(1.0)}))
    return rules


def compile_branch_delta(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """The BILINEAR PC update for BZ / BNZ (runs after dispatch; PC untouched).

        BZ  taken (AX_ZERO==1):  PC := IMM   -> delta = IMM - pc_pre
        BZ  not     (AX_ZERO==0): PC := pc_pre+1 -> delta = 1
        BNZ taken (AX_ZERO==0):  PC := IMM   -> delta = IMM - pc_pre
        BNZ not     (AX_ZERO==1): PC := pc_pre+1 -> delta = 1

    Four SwiGLU product units: ``up = BIG·(OP_IS[op] + TAKEN − 1.5)`` fires iff
    the decoded op-hot AND the boolean both hold (each 0/1); ``gate`` carries the
    delta value. Ported from ``universal.compile_branch_delta``.
    """
    pc, imm, azero, one = L.PC_VAL, L.IMM, L.AX_ZERO, L.ONE
    BZ, BNZ = L.OP_IS + isa.BZ, L.OP_IS + isa.BNZ
    spec = _empty_spec(dim, 4)
    BIG = 200.0
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))

    def _and_unit(u, op_band, bool_terms, gate_terms, gate_const):
        spec["W_up"][u, op_band] += BIG
        for band, coeff, const in bool_terms:
            if band is not None:
                spec["W_up"][u, band] += BIG * coeff
            spec["b_up"][u] += BIG * const
        spec["b_up"][u] += -BIG * 1.5
        for band, coeff in gate_terms:
            spec["W_gate"][u, band] += coeff
        spec["b_gate"][u] += gate_const

    _and_unit(0, BZ,  [(azero, 1.0, 0.0)],  [(imm, 1.0), (pc, -1.0)], 0.0)
    spec["W_down"][pc, 0] += 1.0 / silu_big
    _and_unit(1, BZ,  [(azero, -1.0, 1.0)], [(one, 1.0)], 0.0)
    spec["W_down"][pc, 1] += 1.0 / silu_big
    _and_unit(2, BNZ, [(azero, -1.0, 1.0)], [(imm, 1.0), (pc, -1.0)], 0.0)
    spec["W_down"][pc, 2] += 1.0 / silu_big
    _and_unit(3, BNZ, [(azero, 1.0, 0.0)],  [(one, 1.0)], 0.0)
    spec["W_down"][pc, 3] += 1.0 / silu_big
    return spec


def compile_fold(band: int, one_band: int, dim: int, modulus: int = 256
                 ) -> Dict[str, torch.Tensor]:
    """Exact mod-``modulus`` fold on ``band`` in [0, 2M): ``band -= M·(band>=M)``,
    a sharp unit ramp at M-0.5. Ported from ``compile_ffn.compile_fold``."""
    M, w = modulus, 0.2
    lo = M - 0.5
    spec = _empty_spec(dim, 2)
    for i, thr in enumerate((lo, lo + w)):
        spec["W_up"][i, band] = RELU_S
        spec["b_up"][i] = -RELU_S * thr
        spec["W_gate"][i, one_band] = 1.0
    spec["W_down"][band, 0] = -M / (RELU_S * w)
    spec["W_down"][band, 1] = +M / (RELU_S * w)
    return spec


# ===========================================================================
# BUILD the universal step-block (the interpreter — program INDEPENDENT).
# ===========================================================================
def build_step_model(code_size: int, n_heads: int = 4):
    """Bake ONE universal nibble VM step-block. The weights are program-
    INDEPENDENT — ``code_size`` only sizes the PC/data bands.

    Physical FFN sub-blocks (single residual position, attention zeroed=identity):
      1. recompose : nibble bands -> scalar value lanes (§the bridge).
      2. fetch     : PC_VAL -> PC_IS[i] one-hot + AX_ZERO predicate.
      3. code_sel  : fetch OP_VAL/IMM at PC from DATA MEMORY (product-select).
      4. decode    : OP_VAL scalar -> OP_IS[op] decoded opcode one-hot.
      5. dispatch  : per-OPCODE rules gated on OP_IS[op] apply the op's value-lane
                     effect + PC delta (branches deferred).
      6. branch    : bilinear BZ/BNZ PC update (AX_ZERO · IMM · PC).
      7. fold      : AX_VAL mod-256.
    The next-state lives in the scalar value lanes (AX_VAL/SP_VAL/BP_VAL/STK_VAL +
    PC_VAL) after this block; the driver's frame round-trip writes them back into
    the canonical nibble bands. Returns ``(model, L)``.
    """
    L = NibbleVMLayout(code_size, n_heads=n_heads)
    dim = L.D
    ffn_specs = [
        compile_nibble_to_scalar(L, dim),
        compile_pc_fetch(L, dim),
        compile_code_select(L, dim),
        compile_opcode_decode(L, dim),
        compile_ffn(base_dispatch_rules(L), dim),
        compile_branch_delta(L, dim),
        compile_fold(L.AX_VAL, L.ONE, dim, modulus=256),
    ]
    n_blocks = len(ffn_specs)
    hidden = max(f["W_up"].shape[0] for f in ffn_specs)

    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, vocab=V.VOCAB, max_seq_len=64)
    with torch.no_grad():
        model.embed.zero_()
        # NB: the recurrent driver seeds the state directly (load_program); the
        # embedding table still serves the standard token interface for the frame
        # emit/ingest proof (blogspec_run.ingest_ax_lowbyte).
        _bake_frame_embedding(model, L)
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
    return model, L


def _zero_attn(attn) -> None:
    for p in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        p.zero_()


def _load_ffn(ffn, spec: Dict[str, torch.Tensor], hidden: int) -> None:
    """Load a compiled spec into an FFN, zero-padding the hidden dim to ``hidden``."""
    h = spec["W_up"].shape[0]
    ffn.W_up.zero_();   ffn.W_up[:h] = spec["W_up"]
    ffn.b_up.zero_();   ffn.b_up[:h] = spec["b_up"]
    ffn.W_gate.zero_(); ffn.W_gate[:h] = spec["W_gate"]
    ffn.b_gate.zero_(); ffn.b_gate[:h] = spec["b_gate"]
    ffn.W_down.zero_(); ffn.W_down[:, :h] = spec["W_down"]
    ffn.b_down.zero_(); ffn.b_down.copy_(spec["b_down"])


def _bake_frame_embedding(model, L) -> None:
    """Byte tokens embed their two nibbles into CUR_NIB; markers light CTX. (The
    same foundation embedding — used by the ingest/emit frame proof.)"""
    E = torch.zeros(V.VOCAB, model.dim)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    model.embed.copy_(E)


# ===========================================================================
# UNIVERSAL: load a program into DATA memory (INPUT, not baked).
# ===========================================================================
def load_program(model, L: NibbleVMLayout, code: List[isa.Instr]) -> torch.Tensor:
    """The INITIAL STATE with ``code`` written into the DATA-MEMORY code bands.

    This is universality: different programs = different data in these bands; the
    model weights are untouched. Registers start at the spec init (§C4 Registers):
    PC=AX=0, SP=BP=0x10000, written as NIBBLE bands. ONE=1.
    """
    assert len(code) <= L.code_size, f"{len(code)} slots > code_size {L.code_size}"
    state = torch.zeros(L.D)
    state[L.ONE] = 1.0
    _write_reg_nibbles(state, L.PC, 0)
    _write_reg_nibbles(state, L.AX, 0)
    _write_reg_nibbles(state, L.SP, 0x10000)
    _write_reg_nibbles(state, L.BP, 0x10000)
    _write_reg_nibbles(state, L.STACK0, 0)
    for i, ins in enumerate(code):
        state[L.CODE_OP[i]] = float(ins.op)
        state[L.CODE_IMM[i]] = float(ins.imm)
    return state


def _write_reg_nibbles(state: torch.Tensor, reg_base: int, value: int) -> None:
    for j, nv in enumerate(V.nibbles_of_value(value, 16)):
        state[reg_base + j] = float(nv)


# ===========================================================================
# VANILLA REQUANT DRIVER — the frame round-trip re-quantises AND writes the
# scalar next-state back into the nibble bands. NO torch.round.
# ===========================================================================
def _emit_and_reembed(state: torch.Tensor, L: NibbleVMLayout) -> torch.Tensor:
    """The vanilla token round-trip = the 30-token register frame requant.

    For every register, read its scalar next-state lane, EMIT the 4 little-endian
    byte tokens (argmax over the 256-byte value head — the integer snap, no
    round), and RE-EMBED each byte token by writing its two nibbles back into the
    register's NIBBLE band. Returns the fresh nibble-band state for the next step.
    This closes the loop: dispatch computed the next state on scalar lanes, and
    this writes it back into the spec's canonical nibble representation while
    annihilating all fp residue via the argmax.
    """
    new = torch.zeros_like(state)
    new[L.ONE] = 1.0
    # carry the immutable program-in-data bands untouched.
    for i in range(L.code_size):
        new[L.CODE_OP[i]] = state[L.CODE_OP[i]]
        new[L.CODE_IMM[i]] = state[L.CODE_IMM[i]]
    new[L.HALTED] = state[L.HALTED]
    lanes = [(L.PC, L.PC_VAL), (L.AX, L.AX_VAL), (L.SP, L.SP_VAL),
             (L.BP, L.BP_VAL), (L.STACK0, L.STK_VAL)]
    for reg_base, val_lane in lanes:
        # SNAP the lane to an exact integer via the LM-head argmax (the vanilla
        # re-quantiser — annihilates the O(1e-6) SwiGLU residue; NO round), then
        # EMIT its 4 little-endian byte tokens and RE-EMBED each byte's two nibbles
        # into the register's nibble band. The byte split uses the spec's own
        # floor/mod (§Efficient Floor), not python rounding.
        value = _snap_lane(state[val_lane])
        for bi in range(4):
            byte = (value >> (8 * bi)) & 0xFF          # spec floor/mod byte split
            lo, hi = V.nibbles_of_byte(byte)
            new[reg_base + 2 * bi + 0] = float(lo)     # RE-EMBED (byte -> nibbles)
            new[reg_base + 2 * bi + 1] = float(hi)
    return new


# The value vocabulary: token v decodes to the scalar integer v. Sized to cover
# every reachable register value in the foundation (SP/BP reach ~0x10000 as they
# move by ±4 around the init). The LM-head argmax over this vocab is the vanilla
# re-quantiser (``argmax_v(2·v·x − v²) == round(x)`` for v in range) — the same
# emit-token snap the classic C4 neural VM relies on, with NO ``torch.round``.
VALVOCAB = 0x10100   # 0..0x100FF: covers SP/BP = 0x10000 ± small, plus head-room


def _snap_lane(lane: torch.Tensor) -> int:
    """The LM-head requant: emit the value token ``argmax_v (2·v·x − v²)`` over the
    value vocabulary — the exact-integer snap of the lane, NO ``round``. Vectorised
    so the wide vocab is a single argmax (the standard decode-step argmax)."""
    x = float(lane)
    v = torch.arange(VALVOCAB, dtype=torch.float64)
    logits = 2.0 * v * x - v * v            # LM-head value logits
    return int(logits.argmax().item())      # argmax == the emitted value token


def run_program(model, L: NibbleVMLayout, code: List[isa.Instr],
                max_steps: int = 100000, verbose: bool = False
                ) -> Tuple[List[int], List[Dict]]:
    """Run ``code`` on the universal nibble VM, emitting a 30-token register frame
    per step. Returns ``(tokens, frames)``: the flat autoregressive token stream
    and the per-step decoded register dicts.

    Each iteration: apply the ONE baked step-block, then the vanilla frame
    round-trip (``_emit_and_reembed``) re-quantises the scalar next-state into the
    nibble bands. NO torch.round anywhere. Stops at HALT.
    """
    state = load_program(model, L, code)
    tokens: List[int] = [V.BOS]
    frames: List[Dict] = []
    for _ in range(max_steps):
        # apply the step-block (the interpreter) to the current nibble state.
        x = state.view(1, 1, -1)
        for blk in model.blocks:
            x = blk(x)
        out = x[0, 0]
        halted = float(out[L.HALTED]) > 0.5
        # decode this step's registers from the scalar next-state lanes.
        dec = {name: _decode_lane(out, lane)
               for name, lane in (("pc", L.PC_VAL), ("ax", L.AX_VAL),
                                  ("sp", L.SP_VAL), ("bp", L.BP_VAL))}
        frame = V.build_step_frame(dec["pc"], dec["ax"], dec["sp"], dec["bp"])
        tokens += frame
        frames.append({**dec, "op": _op_name_at(out, code, L)})
        if verbose:
            print(f"  step op={frames[-1]['op']:5s} -> pc={dec['pc']} ax={dec['ax']} "
                  f"sp={dec['sp']} bp={dec['bp']}")
        # vanilla requant: emit the frame's byte tokens and re-embed -> nibbles.
        state = _emit_and_reembed(out, L)
        if halted:
            tokens.append(V.HALT)
            break
    return tokens, frames


def _decode_lane(state: torch.Tensor, val_lane: int) -> int:
    """Decode a full register value from its scalar lane via the LM-head value
    argmax (the vanilla re-quant snap; no python round)."""
    return _snap_lane(state[val_lane])


def _op_name_at(state: torch.Tensor, code, L) -> str:
    """The op the step just executed (from the PC one-hot), for the trace."""
    for i in range(L.code_size):
        if float(state[L.PC_IS[i]]) > 0.5 and i < len(code):
            return isa.NAMES.get(code[i].op, str(code[i].op))
    return "?"


def decode_trace(frames) -> List[int]:
    return [f["ax"] for f in frames]


# ---------------------------------------------------------------------------
# Convenience bundle.
# ---------------------------------------------------------------------------
class NibbleVM:
    """ONE universal nibble VM step-block + its layout; ``run`` any program."""

    def __init__(self, code_size: int, n_heads: int = 4):
        self.model, self.L = build_step_model(code_size, n_heads=n_heads)
        self.code_size = code_size

    def run(self, prog, max_steps: int = 100000, verbose: bool = False):
        code = isa.assemble(prog) if prog and isinstance(prog[0], (tuple, str)) else prog
        return run_program(self.model, self.L, code, max_steps=max_steps, verbose=verbose)

    def weight_hash(self) -> str:
        """A hash of the model weights (proves one FIXED model runs many programs)."""
        import hashlib
        h = hashlib.sha256()
        for p in self.model.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]
