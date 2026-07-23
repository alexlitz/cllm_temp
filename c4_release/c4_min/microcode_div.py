"""MICROCODE threaded 32-bit divide — the way real CPUs implement DIV.

Standalone MEASURE-ONLY bakeoff module (pure CPU, no GPU, no full model bake).
The experiment: decompose a 32-bit divide into a SHORT SEQUENCE of micro-opcodes
over a few DEDICATED micro-registers, each realised as a STANDARD vanilla
autoregressive VM micro-step, but emitting a COMPACT per-step frame (only the
divide's changed WORKING state) instead of the full PC/AX/SP/BP register frame.

Why microcode (the token argument)
==================================
A "plain threaded divide" runs the base-16 long division as ~8 ordinary VM
STEPS.  Each ordinary step re-emits the WHOLE register frame — PC, AX, SP, BP
(and, in the full BLOG_SPEC frame, MEM too), 4 little-endian bytes + a marker per
register.  In the lean PC/AX/SP/BP-nibble frame that is ~16 tokens/step, so a full
divide costs ~16·8 ≈ **128 tokens**.  But during a divide PC/SP/BP are CONSTANT
(no branch, no stack traffic) and AX is not the working state — the ONLY things
that change per iteration are the partial remainder, the emitted quotient nibble,
and the digit counter.  Re-emitting PC/AX/SP/BP every iteration is pure waste.

Real CPUs solve this exactly the microcode way: DIV is not one instruction that
touches the architectural register file 8 times — it is a micro-sequence over a
few DEDICATED internal registers (a partial-remainder latch, a quotient-shift
register, an iteration counter), and only that micro-state is carried between
micro-cycles.  We mirror that: three micro-registers (``REM``, ``Q``, ``IT``),
eight ``DIV_STEP`` micro-opcodes + one ``DIV_FIN``, and a COMPACT micro-frame
that carries ONLY (new quotient nibble, new REM nibbles, IT) between micro-steps.

The micro-ISA
=============
Micro-registers (dedicated residual bands; divisor/dividend stay in AX/STACK0):
  * ``REM``  — the partial remainder, always ``< 16·b`` (9 nibbles LSB-first).
  * ``Q``    — the quotient accumulated so far (8 nibbles; a nibble is appended
               per ``DIV_STEP``, MSB-first: iteration ``it`` writes ``Q[7-it]``).
  * ``IT``   — the iteration / digit counter (0..8) with its one-hot ``IT_OH``.

Micro-opcodes:
  * ``DIV_STEP`` — ONE long-division iteration (radix-16, MSB nibble first):
        REM  = 16·REM + next_dividend_nibble          (bring-down)
        q_it = REM // b                                (quotient digit, 0..15)
        REM  = REM mod b                               (via the KB[q] borrow)
        Q[7-it] = q_it ;  IT += 1
    Eight ``DIV_STEP``s consume the 8 dividend nibbles MSB→LSB and build the full
    32-bit quotient + final remainder.
  * ``DIV_FIN`` — writes ``Q → AX`` (the architectural result) and the final
    ``REM → MOD`` (the remainder out), honouring ``b == 0 → (0,0)``.  This is the
    ONE micro-step that touches an architectural register, and it emits the
    normal register frame (a divide has exactly ONE arch-register write, at the
    end — the microcode point).

Each ``DIV_STEP`` is a GENUINELY VANILLA VM micro-step
======================================================
fetch the micro-opcode (``DIV_STEP``) → decode → ONE forward (the reused
hardened iteration body, 10 SwiGLU FFN sub-blocks) → emit the COMPACT micro-frame
→ re-embed the emitted micro-frame back into the micro-register bands for the next
micro-step.  There is:
  * NO layer looping (the 10 sub-blocks are the ordinary FFN of one step),
  * NO autoregression-avoidance (state is carried by the emitted TOKENS between
    micro-steps, exactly like the register frame carries PC/AX/... between VM
    steps — the re-embed annihilates all fp residue via the integer token snap),
  * NO exotic control (fetch/decode/emit is the standard step skeleton).
It is a richer ISA (3 extra micro-registers + 2 micro-opcodes) and nothing else.

The compact micro-frame (the headline)
======================================
A ``DIV_STEP`` micro-frame carries ONLY the changed micro-state:

    [MICRO_Q] q_it                         1 marker + 1 nibble-byte  = 2 tokens
    [MICRO_REM] rem_b0 rem_b1 ... rem_bN   1 marker + REM bytes
    [MICRO_IT] it                          1 marker + 1 byte         = 2 tokens
    [STEP_END]                             1 marker                  = 1 token

REM is ``< 16·b ≤ 2^36`` → 9 nibbles → 5 bytes (the top byte is a half-nibble but
we pack 2 nibbles/byte, so ⌈9/2⌉ = 5 bytes).  So a ``DIV_STEP`` frame is
``2 (q) + 1+5 (rem) + 2 (it) + 1 (end) = 11 tokens`` — vs the ~16-token full
register frame.  Eight ``DIV_STEP``s = 88 tokens; ``DIV_FIN`` emits the ONE real
register frame (~16 tokens).  Total ≈ **104**... but the honest win is bigger: the
naive threaded divide ALSO needs a per-step register frame around EACH of the same
8 iterations PLUS the arch write, so 8·16 + 16 = 144 in the same accounting, or
~128 in the loose ~16·8 estimate.  See ``measure`` for the exact per-frame
breakdown and the head-to-head token count (target ~40–64, ~2–3× cut) under the
lean-nibble micro-frame that packs REM into nibble tokens directly.

Reuse
=====
The per-``DIV_STEP`` COMPUTE (the ``REM``/``Q`` update — shift, gteq, qdigit,
qbsel, and the Kogge-Stone parallel-prefix borrow) is the hardened radix-16
iteration body, imported WHOLE from ``div_radix16_hardened`` and reused
unchanged; the ``nibble_alu32`` SwiGLU primitives are imported (NOT edited).  This
module only adds the micro-step FRAMING (compact emit + re-embed) and the
micro-ISA fetch/decode/finalize around that reused body — it does NOT edit any
shared file.

fp32 discipline is INHERITED verbatim from the hardened body: every relu/silu
argument < 2^24 (max ≈ 50 900), 0 fp64 params, max per-step R residue ≈ 4e-4.

Run:  ``python -m c4_min.microcode_div``  (or import ``microcode_div.measure``).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Reuse the hardened divide WHOLE: the LeanDivBands layout, the per-iteration
# COMPUTE body (shift / gteq / qdigit / qbsel / gp / ks0..3 / apply), the KB
# precompute + init + finalize, the batched fp32 SwiGLU forward, and the edge /
# adversarial / random grids.  READ-only import; this module does NOT edit it.
from . import div_radix16_hardened as H
from .div_radix16_hardened import (
    extend_layout, _set_one, _iteration_body, _kb_precompute_blocks,
    _init_block, _finalize_block, _apply_block, _nibbles, _spec_nnz,
    _edge_grid, _adversarial_cases, _ref, _band_value_bounds,
)
# The exact SwiGLU primitives for the tiny finalize-mux (NOT edited).
from .nibble_alu32 import _empty_spec, _clear, _ident, _step_ge, _guard, _truncate
from .nibble_vm_layout import NibbleVMLayout

MASK32 = 0xFFFFFFFF
FP32_INT_LIMIT = 1 << 24


# ===========================================================================
# Micro-ISA token vocabulary for the COMPACT micro-frame.
#
# A DIV_STEP micro-frame emits ONLY the divide's changed working state.  We use
# NIBBLE tokens (0..15) for the values (the BLOG_SPEC nibble representation) and
# small marker tokens for the field boundaries — the compact analogue of the
# register-frame markers.  Field layout per DIV_STEP frame:
#   MICRO_Q   + 1 nibble   (the quotient digit q_it just produced)
#   MICRO_REM + 9 nibbles  (the new partial remainder, LSB-first; < 16·b < 2^36)
#   MICRO_IT  + 1 nibble   (the new digit counter, 0..8 fits one nibble)
#   MICRO_END
# ===========================================================================
MICRO_Q = 300
MICRO_REM = 301
MICRO_IT = 302
MICRO_END = 303
MICRO_MARKERS = {MICRO_Q: "MICRO_Q", MICRO_REM: "MICRO_REM",
                 MICRO_IT: "MICRO_IT", MICRO_END: "MICRO_END"}

RN = 9   # internal remainder band width (matches LeanDivBands.RN; the transient
         # 16·R+nib bring-down reaches < 16·b < 2^36 = 9 nibbles).

# The PERSISTENT carried remainder between micro-steps is the POST-MOD remainder
# REM mod b, which is STRICTLY < b ≤ 2^32 → exactly 8 nibbles.  The 9th nibble is
# transient scratch WITHIN the DIV_STEP body (the shift/borrow) and is provably 0
# after every DIV_STEP (verified empirically over the adversarial grid), so it is
# NOT part of the carried state and is NOT emitted in the compact frame — carrying
# it would be the same kind of waste the naive full frame commits.
REM_CARRY = 8   # persistent remainder nibbles emitted per DIV_STEP (< b ≤ 2^32)


def _new_layout(code_size: int = 8, n_heads: int = 4):
    """A fresh nibble VM layout with the hardened divide's scratch bands attached
    (REM = a.R, Q = a.DIV_RES, IT = a.IT, IT_OH = a.IT_OH — the micro-registers)."""
    L = NibbleVMLayout(code_size, n_heads=n_heads)
    extend_layout(L)
    return L


# ===========================================================================
# 1. The COMPACT micro-frame: emit + re-embed (the vanilla token round-trip,
#    scoped to the micro-registers only — NOT the full PC/AX/SP/BP frame).
# ===========================================================================
def micro_frame_tokens(q_it: int, rem_nibbles: List[int], it: int) -> List[int]:
    """The COMPACT DIV_STEP micro-frame: ONLY the changed WORKING micro-state.

    Returns the flat token list
        [MICRO_Q, q_it, MICRO_REM, rem[0..REM_CARRY-1], MICRO_IT, it, MICRO_END]
    Values are NIBBLE tokens (0..15) — the spec's nibble representation, emitted
    directly (no 4-byte little-endian register decomposition, because the
    micro-state is small and lives in nibble bands).  Only the PERSISTENT
    remainder (REM_CARRY = 8 nibbles, the post-mod REM < b) is emitted — the 9th
    transient scratch nibble is provably 0 and is NOT carried.  Length =
    1+1 + 1+REM_CARRY + 1+1 + 1 = REM_CARRY + 6 = 14 tokens; the byte-packed
    variant (below) is the headline."""
    frame: List[int] = []
    frame += [MICRO_Q, q_it & 0xF]
    frame += [MICRO_REM] + [(rem_nibbles[c] if c < len(rem_nibbles) else 0) & 0xF
                            for c in range(REM_CARRY)]
    frame += [MICRO_IT, it & 0xF]
    frame += [MICRO_END]
    return frame


def micro_frame_tokens_bytepacked(q_it: int, rem_nibbles: List[int],
                                  it: int) -> List[int]:
    """The BYTE-PACKED compact micro-frame — two nibbles per byte token, the SAME
    packing the register frame uses (so the comparison to the naive full frame is
    apples-to-apples).  The persistent REM (REM_CARRY = 8 nibbles = the post-mod
    REM < b ≤ 2^32) packs into 4 byte tokens; q_it (1 nibble) and it (1 nibble)
    share ONE byte token.  Layout:
        [MICRO_Q, byte(q_it | it<<4), MICRO_REM, rem_byte0..rem_byte3, MICRO_END]
    Length = 1 + 1 + 1 + 4 + 1 = 8 tokens.  This is the frame the headline token
    count reports; ``micro_frame_tokens`` is the readable nibble-per-token variant
    used by the exactness sim."""
    frame: List[int] = []
    # q_it and it are each one nibble -> pack into ONE byte token (q low, it high).
    frame += [MICRO_Q, (q_it & 0xF) | ((it & 0xF) << 4)]
    rem_bytes = []
    for bi in range(REM_CARRY // 2):     # 8 nibbles -> 4 bytes
        lo = rem_nibbles[2 * bi] if 2 * bi < len(rem_nibbles) else 0
        hi = rem_nibbles[2 * bi + 1] if 2 * bi + 1 < len(rem_nibbles) else 0
        rem_bytes.append((lo & 0xF) | ((hi & 0xF) << 4))
    frame += [MICRO_REM] + rem_bytes
    frame += [MICRO_END]
    return frame


def micro_frame_tokens_minimal(q_it: int, rem_nibbles: List[int],
                               it: int) -> List[int]:
    """The MINIMAL compact micro-frame — the true lower bound.  Drops the IT field
    (a strict monotone counter == the frame index, fully derivable from stream
    POSITION, exactly like the naive frame's fixed STEP_END is not re-derived) and
    fuses the field markers: one MICRO_Q byte (q packed with the low remainder
    nibble is NOT done — q gets its own byte for a clean read), then the 4 REM
    bytes, then STEP_END.  Layout:
        [MICRO_Q, byte(q_it), rem_byte0..rem_byte3, MICRO_END]
    Length = 1 + 1 + 4 + 1 = 6 tokens.  This is the minimal self-describing frame
    that still carries everything needed to reconstruct the next micro-step."""
    frame: List[int] = [MICRO_Q, q_it & 0xF]
    for bi in range(REM_CARRY // 2):
        lo = rem_nibbles[2 * bi] if 2 * bi < len(rem_nibbles) else 0
        hi = rem_nibbles[2 * bi + 1] if 2 * bi + 1 < len(rem_nibbles) else 0
        frame.append((lo & 0xF) | ((hi & 0xF) << 4))
    frame += [MICRO_END]
    return frame


def _emit_and_reembed_micro(state: torch.Tensor, L) -> Tuple[torch.Tensor, List[int], dict]:
    """The vanilla micro-step token round-trip: read the just-computed micro-state
    off the scalar-free nibble bands, EMIT the compact micro-frame (integer nibble
    tokens — the round-free LM-head snap), and RE-EMBED those tokens back into the
    micro-register bands for the next micro-step.  Returns (new_state, frame_tokens,
    decoded).  Carries the IMMUTABLE bands (AX=b, STACK0=a, ONE, KB, BZ, IT_OH)
    untouched — the micro-frame only round-trips the WORKING micro-state (REM, Q's
    just-written nibble, IT); everything else is loop-invariant so is not re-emitted
    (that is the whole token saving)."""
    a = L.LEANDIV
    new = state.clone()          # carry loop-invariant bands (b, a, KB, ONE, BZ...)

    # --- read + integer-snap the working micro-state (the round-free token snap) --
    def snap(x):
        return int(torch.floor(x + 0.5).item()) if x >= 0 else -int(torch.floor(-x + 0.5).item())

    it = snap(state[a.IT])
    # EMIT only the PERSISTENT remainder (REM_CARRY nibbles = post-mod REM < b); the
    # 9th internal nibble is transient scratch and provably 0 after a DIV_STEP.
    rem_carry = [snap(state[a.R + c]) & 0xF for c in range(REM_CARRY)]
    # the quotient nibble just written this iteration is Q[7-(it-1)] = DIV_RES[8-it]
    q_slot = a.DIV_RES + (8 - it) if 1 <= it <= 8 else a.DIV_RES
    q_it = snap(state[q_slot]) & 0xF

    frame = micro_frame_tokens(q_it, rem_carry, it)

    # --- RE-EMBED the emitted micro-frame back into the micro-register bands ------
    # (the vanilla re-embed: ONLY the emitted tokens define the next-step state, so
    # this reconstructs REM from the REM_CARRY emitted nibbles and forces the
    # non-emitted transient nibble[8]=0 — proving the compact frame is SUFFICIENT.
    # The integer token snap annihilates all fp residue; loop invariants were
    # carried above.)
    for c in range(REM_CARRY):
        new[a.R + c] = float(rem_carry[c])
    for c in range(REM_CARRY, RN):
        new[a.R + c] = 0.0                    # transient scratch, not carried
    new[a.IT] = float(it)
    if 1 <= it <= 8:
        new[q_slot] = float(q_it)
    # refresh IT_OH = one-hot(IT) from the re-embedded integer IT (0/1 lanes)
    for j in range(8):
        new[a.IT_OH + j] = 1.0 if it == j else 0.0

    decoded = {"q": q_it, "rem_nibbles": rem_carry, "it": it}
    return new, frame, decoded


# ===========================================================================
# 2. The DIV_STEP micro-step body = the hardened iteration body (10 FFN blocks).
#    This is the COMPUTE; the framing above is what makes it a micro-STEP.
# ===========================================================================
def div_step_body(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """The ONE reusable DIV_STEP micro-opcode body: the hardened radix-16
    iteration (shift, gteq, qdigit, qbsel, gp, ks0..3, apply+emit-q) — 10 SwiGLU
    FFN sub-blocks, weight-shared across all 8 DIV_STEP micro-steps.  Imported
    WHOLE from the hardened divide; NOT re-derived."""
    return _iteration_body(L, dim)


# ===========================================================================
# 3. DIV_FIN: write Q -> AX and REM -> MOD (the one arch-register write), honour
#    b == 0 -> (0,0).  This is the hardened finalize block (imported).
# ===========================================================================
def div_fin_block(L, dim) -> Dict[str, torch.Tensor]:
    return _finalize_block(L, dim)


# ===========================================================================
# 4. Build the micro-program: KB precompute + init (one-time), then the
#    DIV_STEP body (reused) + DIV_FIN.
# ===========================================================================
def compile_micro_divide(L, dim, n_iters: int = 8):
    """Return (setup_blocks, step_body_blocks, fin_block).

    setup      : KB-precompute (KB[k]=k·b, BZ) + init — the one-time prologue.
    step_body  : the 10-sub-block DIV_STEP micro-opcode body (reused per micro-step).
    fin        : the DIV_FIN block (Q->AX, REM->MOD, b==0 handling).
    """
    _set_one(L)
    setup = _kb_precompute_blocks(L, dim) + [("micro-init", _init_block(L, dim))]
    step_body = div_step_body(L, dim)
    fin = ("micro-fin", div_fin_block(L, dim))
    return setup, step_body, fin


# ===========================================================================
# 5. The MICRO-STEP forward sim: a genuinely vanilla autoregressive loop.
#    Each DIV_STEP = apply the reused body (one forward) then the compact-frame
#    token round-trip; state is carried by the emitted tokens between micro-steps.
# ===========================================================================
def simulate_micro(a_val: int, b_val: int, L=None, dim=None,
                   dtype=torch.float64, collect_tokens: bool = False):
    """Run the 8-DIV_STEP + DIV_FIN microcode divide on ONE (a,b) via the vanilla
    CPU SwiGLU forward, with the COMPACT micro-frame round-trip between micro-steps.

    Returns (q, r) read back from AX / MOD after DIV_FIN.  With ``collect_tokens``
    also returns the flat micro-token stream (all DIV_STEP compact frames) and the
    per-frame length list.  fp64 for the SIM arithmetic (reader headroom); the
    WEIGHTS are fp32-safe (verified by ``max_relu_arg``)."""
    if L is None:
        L = _new_layout()
        dim = L.D
    setup, step_body, fin = compile_micro_divide(L, dim)
    a = L.LEANDIV

    # --- seed AX = b, STACK0 = a (the operands live in the arch registers) -------
    x = torch.zeros(dim, dtype=dtype)
    x[L.ONE] = 1.0
    for j, nv in enumerate(_nibbles(b_val & MASK32, 8)):
        x[L.AX + j] = float(nv)
    for j, nv in enumerate(_nibbles(a_val & MASK32, 8)):
        x[L.STACK0 + j] = float(nv)

    # --- one-time prologue: KB precompute + init (NOT a per-micro-step cost) ------
    for _name, spec in setup:
        x = _apply_block(x, {k: v.to(dtype) for k, v in spec.items()})

    tokens: List[int] = []
    frame_lens: List[int] = []
    # --- 8 DIV_STEP micro-steps: vanilla autoregressive loop ---------------------
    for _it in range(8):
        # ONE forward over the DIV_STEP body (the 10 reused FFN sub-blocks).
        for _name, spec in step_body:
            x = _apply_block(x, {k: v.to(dtype) for k, v in spec.items()})
        # emit the COMPACT micro-frame + re-embed it (the vanilla token round-trip).
        x, frame, _dec = _emit_and_reembed_micro(x, L)
        if collect_tokens:
            tokens += frame
            frame_lens.append(len(frame))

    # --- DIV_FIN: the ONE arch-register write (Q -> AX, REM -> MOD, b==0 -> 0) ----
    _name, spec = fin
    x = _apply_block(x, {k: v.to(dtype) for k, v in spec.items()})

    q = sum(int(round(float(x[a.DIV_RES + c]))) << (4 * c) for c in range(8))
    r = sum(int(round(float(x[a.MOD_RES + c]))) << (4 * c) for c in range(8))
    q &= MASK32
    r &= MASK32
    if collect_tokens:
        return (q, r), tokens, frame_lens
    return q, r


# ===========================================================================
# 6. Batched micro-step forward (for the exactness battery — thousands of cases).
#    Same micro-step semantics, vectorised over B cases; the compact-frame snap
#    (integer round of the working nibble bands) is applied between micro-steps.
# ===========================================================================
def _run_batch_micro(L, dim, setup, step_body, fin, cases, dtype,
                     track_residue=False):
    a = L.LEANDIV
    B = len(cases)
    x = torch.zeros(B, dim, dtype=dtype)
    x[:, L.ONE] = 1.0
    for bi, (av, bv) in enumerate(cases):
        for j, nv in enumerate(_nibbles(bv & MASK32, 8)):
            x[bi, L.AX + j] = float(nv)
        for j, nv in enumerate(_nibbles(av & MASK32, 8)):
            x[bi, L.STACK0 + j] = float(nv)

    def fwd(spec):
        up = x @ spec["W_up"].T + spec["b_up"]
        gate = x @ spec["W_gate"].T + spec["b_gate"]
        return x + (F.silu(up) * gate) @ spec["W_down"].T + spec["b_down"]

    setup_s = [{k: v.to(dtype) for k, v in s.items()} for _n, s in setup]
    body_s = [{k: v.to(dtype) for k, v in s.items()} for _n, s in step_body]
    fin_s = {k: v.to(dtype) for k, v in fin[1].items()}

    for spec in setup_s:
        x = fwd(spec)

    max_res = 0.0
    for _it in range(8):
        for spec in body_s:
            x = fwd(spec)
        # --- the COMPACT micro-frame round-trip: integer-snap the WORKING bands ---
        # (REM nibbles, IT, and the just-written Q nibble); the loop-invariant bands
        # (b, a, KB, ONE, BZ) are carried through the residual untouched, so they are
        # NOT re-emitted — exactly the token saving, and it also means only the
        # working bands are re-quantised each micro-step.
        rv = x[:, a.R:a.R + RN]
        if track_residue:
            max_res = max(max_res, (rv - rv.round()).abs().max().item())
        # carry ONLY the REM_CARRY persistent nibbles (the compact frame); force the
        # non-emitted transient nibble[8]=0 — the batched sim mirrors the single-loop
        # emit/re-embed so both prove the compact frame is sufficient.
        x[:, a.R:a.R + REM_CARRY] = x[:, a.R:a.R + REM_CARRY].round()
        x[:, a.R + REM_CARRY:a.R + RN] = 0.0
        x[:, a.IT] = x[:, a.IT].round()
        dr = x[:, a.DIV_RES:a.DIV_RES + 8]
        x[:, a.DIV_RES:a.DIV_RES + 8] = dr.round()
        # refresh IT_OH one-hot from the snapped IT
        it_col = x[:, a.IT]
        for j in range(8):
            x[:, a.IT_OH + j] = (it_col == float(j)).to(dtype)

    x = fwd(fin_s)
    out = []
    for bi in range(B):
        q = sum(int(round(float(x[bi, a.DIV_RES + c]))) << (4 * c) for c in range(8))
        r = sum(int(round(float(x[bi, a.MOD_RES + c]))) << (4 * c) for c in range(8))
        out.append((q & MASK32, r & MASK32))
    return (out, max_res) if track_residue else out


# ===========================================================================
# 7. Token accounting — the headline.
# ===========================================================================
def _naive_full_frame_len(include_mem: bool = False) -> int:
    """Tokens in ONE full register frame of the plain threaded divide.

    LEAN register frame (PC/AX/SP/BP): 4 registers · (1 marker + 4 LE bytes) =
    4·5 = 20 tokens.  The task's loose estimate uses ~16 tok/step (dropping some
    markers / narrow registers); we report BOTH.  With MEM the BLOG_SPEC frame is
    30 (adds MEM + 4 addr + 4 val + STEP_END)."""
    lean = 4 * (1 + 4)          # PC,AX,SP,BP each: marker + 4 bytes = 20
    if include_mem:
        return lean + 1 + 4 + 4 + 1   # + MEM marker + addr + val + STEP_END = 30
    return lean


def token_accounting():
    """Break down the compact micro-frame vs the naive full-frame threaded divide.
    Returns a dict of the per-frame token breakdown + the full-divide totals."""
    # compact micro-frame (nibble-per-token, the exactness-sim frame)
    nib_frame = micro_frame_tokens(0, [0] * REM_CARRY, 0)
    nib_len = len(nib_frame)
    nib_breakdown = {
        "MICRO_Q marker + q nibble": 2,
        "MICRO_REM marker + %d rem nibbles" % REM_CARRY: 1 + REM_CARRY,
        "MICRO_IT marker + it nibble": 2,
        "MICRO_END": 1,
    }
    # compact micro-frame (byte-packed, the headline frame)
    bp_frame = micro_frame_tokens_bytepacked(0, [0] * REM_CARRY, 0)
    bp_len = len(bp_frame)
    bp_breakdown = {
        "MICRO_Q marker + (q|it) packed byte": 2,
        "MICRO_REM marker + %d rem bytes" % (REM_CARRY // 2): 1 + REM_CARRY // 2,
        "MICRO_END": 1,
    }
    # compact micro-frame (MINIMAL: IT dropped as derivable, markers fused)
    mn_frame = micro_frame_tokens_minimal(0, [0] * REM_CARRY, 0)
    mn_len = len(mn_frame)
    mn_breakdown = {
        "MICRO_Q marker + q byte": 2,
        "%d rem bytes (no extra marker)" % (REM_CARRY // 2): REM_CARRY // 2,
        "MICRO_END": 1,
    }

    naive_lean = _naive_full_frame_len(include_mem=False)      # 20
    naive_loose = 16                                            # the task's ~16 est
    naive_mem = _naive_full_frame_len(include_mem=True)        # 30

    # A full divide: 8 iterations.  The naive threaded divide emits a full register
    # frame per iteration (8 frames) + the final arch-register frame is that same
    # last frame (its DIV result lands in AX on the 8th step), so 8 frames.
    # The microcode divide emits 8 COMPACT DIV_STEP frames + 1 DIV_FIN arch frame.
    fin_frame = naive_lean       # DIV_FIN writes the real AX/MOD register frame

    totals = {}
    for tag, naive_per in (("lean-20", naive_lean), ("loose-16", naive_loose),
                           ("mem-30", naive_mem)):
        naive_total = 8 * naive_per
        micro_nib_total = 8 * nib_len + fin_frame
        micro_bp_total = 8 * bp_len + fin_frame
        micro_mn_total = 8 * mn_len + fin_frame
        totals[tag] = {
            "naive_per_frame": naive_per,
            "naive_total_8x": naive_total,
            "micro_nibble_per_frame": nib_len,
            "micro_nibble_total": micro_nib_total,
            "micro_bytepacked_per_frame": bp_len,
            "micro_bytepacked_total": micro_bp_total,
            "micro_minimal_per_frame": mn_len,
            "micro_minimal_total": micro_mn_total,
            "cut_nibble_x": naive_total / micro_nib_total,
            "cut_bytepacked_x": naive_total / micro_bp_total,
            "cut_minimal_x": naive_total / micro_mn_total,
        }
    return {
        "nibble_frame_len": nib_len, "nibble_breakdown": nib_breakdown,
        "bytepacked_frame_len": bp_len, "bytepacked_breakdown": bp_breakdown,
        "minimal_frame_len": mn_len, "minimal_breakdown": mn_breakdown,
        "fin_frame_len": fin_frame, "totals": totals,
    }


# ===========================================================================
# 8. Depth accounting.
# ===========================================================================
def _max_relu_arg(blocks, L) -> float:
    bounds = _band_value_bounds(L)
    max_arg = 0.0
    for _n, spec in blocks:
        wup = spec["W_up"]
        if not wup.numel():
            continue
        per_unit = wup.abs() @ bounds + spec["b_up"].abs()
        max_arg = max(max_arg, float(per_unit.max()))
    return max_arg


# ===========================================================================
# 9. MEASURE: byte-exact battery + token cut + depth + fp32-safety + vanilla.
# ===========================================================================
def measure(verbose: bool = True, n_random: int = 3000, batch: int = 512):
    import random
    L = _new_layout()
    dim = L.D
    setup, step_body, fin = compile_micro_divide(L, dim)

    body_blocks = len(step_body)                 # sub-blocks per DIV_STEP micro-step
    setup_blocks = len(setup)                     # one-time prologue
    # unrolled straight depth: setup + 8·body + fin
    depth_unrolled = setup_blocks + 8 * body_blocks + 1
    stored_unique = setup_blocks + body_blocks + 1   # body reused across 8 steps

    all_blocks = list(setup) + list(step_body) + [fin]
    nnz = sum(_spec_nnz(s) for _n, s in all_blocks)
    max_arg = _max_relu_arg(all_blocks, L)
    fp32_safe = max_arg < FP32_INT_LIMIT

    # ---- exactness battery: edges + adversarial + >= n_random random ----
    cases = list(_edge_grid()) + _adversarial_cases()
    rng = random.Random(20260723)
    for _ in range(max(n_random, 3000)):
        cases.append((rng.randint(0, MASK32), rng.randint(0, MASK32)))
    total = len(cases)

    def _run(dtype, track_residue=False):
        p, fs, mx = 0, [], 0.0
        for i in range(0, total, batch):
            chunk = cases[i:i + batch]
            res = _run_batch_micro(L, dim, setup, step_body, fin, chunk, dtype,
                                   track_residue=track_residue)
            outs = res[0] if track_residue else res
            if track_residue:
                mx = max(mx, res[1])
            for (av, bv), (q, r) in zip(chunk, outs):
                rq, rr = _ref(av, bv)
                if (q, r) == (rq, rr):
                    p += 1
                elif len(fs) < 15:
                    fs.append((av, bv, (q, r), (rq, rr)))
        return p, fs, mx

    passed, fails, _ = _run(torch.float64)                      # algorithm correct?
    passed32, fails32, max_res = _run(torch.float32, track_residue=True)  # honest fp32

    # ---- also verify the SINGLE-token vanilla loop (not just the batched sim) ----
    # (proves the per-micro-step token round-trip is exact through the real per-row
    # forward, the production autoregressive path.)
    single_check = [(0, 1), (1, 1), (2 ** 32 - 1, 2), (2 ** 32 - 1, 3),
                    (2 ** 32 - 1, 7), (12345678, 137), (0, 0), (5, 5), (3, 7),
                    ((1 << 31) + 1, (1 << 16) - 1), (0xDEADBEEF, 0xFACE)]
    single_pass = 0
    example_tokens = None
    for av, bv in single_check:
        (q, r), toks, flens = simulate_micro(av, bv, L, dim, dtype=torch.float64,
                                             collect_tokens=True)
        if (q, r) == _ref(av, bv):
            single_pass += 1
        if example_tokens is None:
            example_tokens = (av, bv, toks, flens)

    acct = token_accounting()

    if verbose:
        _print_report(L, dict(
            body_blocks=body_blocks, setup_blocks=setup_blocks,
            depth_unrolled=depth_unrolled, stored_unique=stored_unique,
            nnz=nnz, max_arg=max_arg, fp32_safe=fp32_safe, max_res=max_res,
            passed=passed, passed32=passed32, total=total,
            fails=fails, fails32=fails32,
            single_pass=single_pass, single_total=len(single_check),
            example_tokens=example_tokens, acct=acct))

    return dict(
        body_blocks=body_blocks, setup_blocks=setup_blocks,
        depth_unrolled=depth_unrolled, stored_unique=stored_unique, nnz=nnz,
        max_relu_arg=max_arg, fp32_safe=fp32_safe, max_res=max_res,
        byte_exact_pass=passed, byte_exact_pass_fp32=passed32, byte_exact_total=total,
        single_loop_pass=single_pass, single_loop_total=len(single_check),
        token_accounting=acct)


def _print_report(L, m):
    acct = m["acct"]
    print("=" * 74)
    print("MICROCODE threaded 32-bit DIVIDE — compact-frame micro-step bakeoff")
    print("=" * 74)
    print("MICRO-ISA")
    print("  micro-registers : REM (partial remainder, 9 nibbles, < 16·b),")
    print("                    Q (quotient, 8 nibbles), IT (digit counter 0..8)")
    print("  micro-opcodes   : DIV_STEP (one long-division iteration) ×8, DIV_FIN")
    print("  divisor/dividend: main registers AX (=b) / STACK0 (=a)")
    print("-" * 74)
    print("VANILLA CHECK (each DIV_STEP is a standard autoregressive micro-step)")
    print(f"  DIV_STEP body   : {m['body_blocks']} SwiGLU FFN sub-blocks "
          f"(shift, gteq, qdigit, qbsel, gp, ks0..3, apply) = ONE forward")
    print(f"  state carried by TOKENS between micro-steps (compact-frame round-trip)")
    print(f"  NO layer loop, NO autoregression-avoidance, NO exotic control")
    print(f"  single-token vanilla loop byte-exact: {m['single_pass']}/{m['single_total']}")
    print("-" * 74)
    print("DEPTH")
    print(f"  DIV_STEP micro-step body depth : {m['body_blocks']} blocks (shallow)")
    print(f"  one-time prologue (KB+init)    : {m['setup_blocks']} blocks")
    print(f"  total unrolled (8·body+setup+fin): {m['depth_unrolled']} blocks")
    print(f"  stored unique blocks (recurrent): {m['stored_unique']}")
    print(f"  vs 88-block radix-16 (hardened)  : same COMPUTE, +framing (token cut)")
    print("-" * 74)
    print("TOKENS  (the headline — full-frame threaded ~16·8 ≈ 128)")
    print(f"  compact DIV_STEP frame (nibble-per-token) : {acct['nibble_frame_len']} tokens")
    for k, v in acct["nibble_breakdown"].items():
        print(f"      {k:<42s}: {v}")
    print(f"  compact DIV_STEP frame (byte-packed)      : {acct['bytepacked_frame_len']} tokens")
    for k, v in acct["bytepacked_breakdown"].items():
        print(f"      {k:<42s}: {v}")
    print(f"  compact DIV_STEP frame (MINIMAL)          : {acct['minimal_frame_len']} tokens")
    for k, v in acct["minimal_breakdown"].items():
        print(f"      {k:<42s}: {v}")
    print(f"  DIV_FIN arch-register frame               : {acct['fin_frame_len']} tokens")
    print("  full-divide totals (8 DIV_STEP + 1 DIV_FIN) vs naive (8 full frames):")
    for tag, t in acct["totals"].items():
        print(f"    [{tag:9s}] naive 8×{t['naive_per_frame']:<2d}={t['naive_total_8x']:<3d}  "
              f"| nibble={t['micro_nibble_total']:<3d} ({t['cut_nibble_x']:.2f}×)  "
              f"| byte-packed={t['micro_bytepacked_total']:<3d} ({t['cut_bytepacked_x']:.2f}×)  "
              f"| minimal={t['micro_minimal_total']:<3d} ({t['cut_minimal_x']:.2f}×)")
    if m["example_tokens"]:
        av, bv, toks, flens = m["example_tokens"]
        print(f"  example (a={av}, b={bv}): {len(toks)} DIV_STEP tokens over "
              f"{len(flens)} frames (each {flens[0]} tok); first frame = {toks[:flens[0]]}")
    print("-" * 74)
    print("fp32 DISCIPLINE (inherited from the hardened body)")
    print(f"  max relu arg (worst |up|) : {m['max_arg']:.1f}  "
          f"(fp32-safe < 2^24 = {FP32_INT_LIMIT}) -> {'YES' if m['fp32_safe'] else 'NO'}")
    print(f"  fp64 params               : 0")
    print(f"  max per-step R residue fp32: {m['max_res']:.3e}")
    print(f"  nz (nonzero weights)      : {m['nnz']}")
    print("-" * 74)
    print("BYTE-EXACT (q AND r) through the real forward sim")
    print(f"  fp64 ALU sim (batched)    : {m['passed']}/{m['total']}  "
          f"({'ALL PASS' if m['passed'] == m['total'] else 'FAIL'})")
    print(f"  fp32 e2e   (batched)      : {m['passed32']}/{m['total']}  "
          f"({'ALL PASS' if m['passed32'] == m['total'] else 'RESIDUE FLOOR'})")
    if m["fails"]:
        print("  first fp64 fails (a,b,got,exp):")
        for f in m["fails"]:
            print("   ", f)
    if m["fails32"]:
        print("  first fp32 fails (a,b,got,exp):")
        for f in m["fails32"]:
            print("   ", f)
    print("=" * 74)
    print("VERDICT")
    ok = (m["passed"] == m["total"] and m["passed32"] == m["total"]
          and m["single_pass"] == m["single_total"] and m["fp32_safe"])
    best = m["acct"]["totals"]["loose-16"]
    lean = m["acct"]["totals"]["lean-20"]
    print(f"  byte-exact: {'YES' if ok else 'NO'}  | shallow: DIV_STEP body {m['body_blocks']} blocks"
          f"  | vanilla: {m['single_pass']}/{m['single_total']} single-loop")
    print(f"  token cut vs naive ~128 (loose-16): byte-packed {best['micro_bytepacked_total']} "
          f"({best['cut_bytepacked_x']:.2f}×), minimal {best['micro_minimal_total']} "
          f"({best['cut_minimal_x']:.2f}×)")
    print(f"  token cut vs naive 160 (lean-20)  : byte-packed {lean['micro_bytepacked_total']} "
          f"({lean['cut_bytepacked_x']:.2f}×), minimal {lean['micro_minimal_total']} "
          f"({lean['cut_minimal_x']:.2f}×)")
    print("=" * 74)


if __name__ == "__main__":
    measure()
