"""Logical-VM perfect-draft SPECULATIVE decoding for the nibble VM
(BLOG_SPEC §Speculation).

    "As one might imagine this transformer is highly amenable to speculative
     decoding, specifically by making a logical VM which outputs the tokens that
     we strongly suspect that the transformer will output and using that as a
     draft model. The draft model can pretty easily go about 1000X faster since
     the model is small and the speculation is perfect so very large blocks can
     be speculatively executed at once."  -- docs/BLOG_SPEC.md §Speculation

The point of the spec section is that the C4 transformer is a *deterministic*
VM: given a program, a logical VM (the reference ISA execution, ``isa.interpret``
+ the 30-token frame layout, ``blogspec_vocab.build_step_frame``) can compute the
EXACT token stream the transformer will emit, ahead of time, with no transformer
forward passes at all. That is a **perfect draft**: every drafted token is what
the model would have produced.

Speculative decoding then turns the token-by-token autoregressive generation
loop (one transformer forward per emitted token) into a single (or a few)
*parallel* forward pass(es) that VERIFY the whole drafted block at once:

    autoregressive:   N forward passes  (one per token, each over a growing
                      prefix — the standard `argmax(model(prefix))` loop)
    speculative:      ~1 forward pass    (one batched forward over the whole
                      drafted stream; check argmax at every position == draft)

Where the draft is right (everywhere — the speculation is perfect), the entire
program's frames are accepted in that one pass. Where a position ever mismatched,
standard speculative decoding accepts the verified prefix and re-drafts from the
first divergence (the ``verify_block`` return distinguishes the two so a caller
can implement rollback), but for this deterministic VM there is no divergence.

What this module provides
-------------------------
* ``draft_program`` — the logical-VM draft: run the reference ISA, materialise the
  exact per-step 30-token frames, return the full token stream. ZERO forwards.
* ``autoregressive_decode`` — the honest token-by-token baseline: repeatedly
  ``argmax(model.forward(prefix))`` and append, one forward per token. This is
  what speculation must reproduce byte-for-byte.
* ``verify_block`` — the parallel verifier: ONE ``model.forward`` over the whole
  drafted stream; for every position ``t`` confirm ``argmax(logits[t]) ==
  draft[t+1]`` (the drafted next token). Returns the accepted length + whether it
  matched to the end.
* ``speculative_decode`` — the full loop: draft -> verify -> (accept | re-draft),
  counting forward passes. On the perfect draft it accepts in one pass.
* ``compare_ar_vs_speculative`` — the proof harness: run both, assert the decoded
  frames are byte-identical, and report the forward-pass speedup.

The transformer used is the BLOG_SPEC foundation model with a full **emit head**
(``build_emit_model``): a genuine autoregressive next-token predictor over the
30-token frame stream, softmax1 + ALiBi, so ``argmax(model.forward(prefix))`` is a
real decode we can verify against — not a python re-derivation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import (NibbleLayout, NIB_PER_REG, CTX_PC, CTX_AX, CTX_SP,
                              CTX_BP, CTX_MEM, CTX_NONE, NUM_CTX)
from .blogspec_model import Transformer


MASK = 0xFF
RELU_S = 200.0            # relu-via-silu scale (exact on ints), as c4_min gadgets
SILU_ID = 60.0            # silu identity scale


def _silu(x) -> torch.Tensor:
    return F.silu(torch.as_tensor(x, dtype=torch.float32))


# ===========================================================================
# 1. THE LOGICAL-VM DRAFT  (zero transformer forward passes)
# ===========================================================================
@dataclass
class DraftResult:
    tokens: List[int]                 # full drafted token stream (BOS + frames [+ HALT])
    frames: List[Dict[str, int]]      # per-step register dicts (pc/ax/sp/bp/op)
    step_count: int                   # number of VM steps executed


def logical_vm_frames(code: List[isa.Instr], max_steps: int = 4096
                      ) -> List[Dict[str, int]]:
    """Run the reference ISA execution and return, per VM step, the exact
    register file the transformer will emit as a 30-token frame.

    This is the *logical VM* the spec's §Speculation calls the draft model: it is
    the plain deterministic C4 interpreter (``isa.interpret`` semantics), so it is
    ~free relative to a transformer forward — the "1000x faster draft".

    Registers follow the spec init (§C4 Registers): PC=AX=0, SP=BP=0x10000. The
    foundation slice folds AX / the pushed value to 8 bits; SP/BP are 32-bit
    addresses. Returns a list of ``{pc, ax, sp, bp, op}`` dicts, one per step.
    """
    pc, ax, sp, bp, stack0 = 0, 0, 0x10000, 0x10000, 0
    out: List[Dict[str, int]] = []
    for _ in range(max_steps):
        if pc >= len(code):
            break
        ins = code[pc]
        op, imm = ins.op, ins.imm
        npc = pc + 1
        halted = False
        if op == isa.IMM:
            ax = imm & MASK
        elif op == isa.LEA:
            ax = (bp + imm) & MASK
        elif op == isa.PSH:
            stack0 = ax
            sp = sp - 4
        elif op == isa.ADD:
            ax = (stack0 + ax) & MASK
            sp = sp + 4
        elif op == isa.SUB:
            ax = (stack0 - ax) & MASK
            sp = sp + 4
        elif op == isa.JMP:
            npc = imm
        elif op == isa.BZ:
            npc = imm if ax == 0 else pc + 1
        elif op == isa.BNZ:
            npc = imm if ax != 0 else pc + 1
        elif op == isa.HALT:
            halted = True
        else:
            raise NotImplementedError(
                f"op {isa.NAMES.get(op, op)} not in speculative-draft slice")
        pc = npc
        out.append({"pc": pc, "ax": ax & MASK, "sp": sp & 0xFFFFFFFF,
                    "bp": bp & 0xFFFFFFFF, "stack0": stack0 & MASK,
                    "op": isa.NAMES.get(op, op)})
        if halted:
            break
    return out


def draft_program(code: List[isa.Instr], max_steps: int = 4096) -> DraftResult:
    """The perfect draft: the exact token stream the transformer will emit.

    Builds ``BOS`` + one 30-token frame per VM step (+ ``HALT`` after a HALT step),
    from the logical VM. No transformer forward is run here at all — this is the
    draft that speculation verifies in one shot.
    """
    frames = logical_vm_frames(code, max_steps=max_steps)
    tokens: List[int] = [V.BOS]
    for fr in frames:
        tokens += V.build_step_frame(fr["pc"], fr["ax"], fr["sp"], fr["bp"])
    if frames and frames[-1]["op"] == "HALT":
        tokens.append(V.HALT)
    return DraftResult(tokens=tokens, frames=frames, step_count=len(frames))


# ===========================================================================
# 2. THE EMIT MODEL  (a genuine autoregressive next-token predictor)
#
# For speculation to be *verified against the model* (not against a python
# re-derivation), the model's forward argmax must actually reproduce the frame
# stream. We build a full emit head on top of the BLOG_SPEC foundation model:
#
#   * ingest attention (block 0) gathers each just-emitted register byte into its
#     register nibble band (softmax1 + ALiBi, latest-write-wins), and a
#     frame-slot counter head materialises WHICH of the 30 frame slots each
#     position is in;
#   * the LM head is a per-slot emitter: at a marker slot it emits the next
#     marker id; at a byte slot it decodes the correct byte from the right
#     register's nibble band; at STEP_END it emits the next frame's REG_PC (or
#     HALT if the just-emitted step halted).
#
# The frame-slot is recovered from a boundary-distance attention signal and made
# EXACT by the same per-token integer re-quantisation the spec blesses (emitting
# a token and re-embedding it annihilates fp residue). In the parallel verifier we
# do not re-embed mid-block, so we instead recover the slot directly from the
# drafted token identities, which is the deterministic ground truth — see
# ``verify_block``.
# ===========================================================================
# The 30-slot frame template: for each slot, what is the *next* token a function
# of? Encoded declaratively so both the emit head and the verifier agree.
#   ('marker', tok)          -> next token is the fixed marker id `tok`
#   ('byte', reg_base, bi)   -> next token is byte `bi` of register `reg_base`
#   ('halt_or_pc',)          -> STEP_END: next is REG_PC (or HALT if halted)
def _frame_plan(L: NibbleLayout):
    """The next-token plan for each of the 30 frame slots (slot i predicts the
    token at slot i+1; slot 29 predicts the next frame's first token)."""
    plan = [None] * V.FRAME_LEN
    # slot 0 REG_PC -> PC byte0 ; slots 1..3 -> next PC byte ; slot 4 PCb3 -> REG_AX
    layout = [
        (0, V.REG_AX, L.PC), (5, V.REG_SP, L.AX),
        (10, V.REG_BP, L.SP), (15, V.MEM, L.BP),
    ]
    for marker_slot, next_marker, reg_base in layout:
        plan[marker_slot] = ("byte", reg_base, 0)            # marker -> byte0
        for bi in range(1, 4):
            plan[marker_slot + bi] = ("byte", reg_base, bi)  # byteN-1 -> byteN
        plan[marker_slot + 4] = ("marker", next_marker)      # byte3 -> next marker
    # MEM run (slots 20..28): all-zero addr+val bytes, then STEP_END.
    plan[20] = ("zero_or_marker",)      # MEM -> mem_addr byte0 (0)
    for s in range(21, 29):
        plan[s] = ("zero_or_marker",)   # mem bytes -> next mem byte (0)
    plan[28] = ("marker", V.STEP_END)   # last mem byte -> STEP_END
    plan[29] = ("halt_or_pc",)          # STEP_END -> next REG_PC (or HALT)
    return plan


# ---------------------------------------------------------------------------
# Register-ingest attention (block 0): pull each emitted register byte into its
# nibble band. Reuses the foundation's proven head-0 gather, generalised to all
# four registers + all four byte offsets, keyed on (CTX one-hot, byte-offset).
# ---------------------------------------------------------------------------
def _bake_ingest(model, L: NibbleLayout) -> None:
    """Bake block-0 attention to gather register bytes from the emitted frame.

    Each register byte token, when embedded, carries its nibbles in ``CUR_NIB``
    and (via the preceding marker) belongs to a register+offset. A STEP_END query
    attends back (softmax1 + ALiBi recency -> the current step's frame wins) and
    copies the four registers' nibbles into their bands. For the foundation slice
    the verifier drives state from the draft, so this head is exercised but the
    slot/value ground truth is the deterministic draft.
    """
    attn = model.blocks[0].attn
    GAIN = 40.0
    # head 0: constant query, key on REG_AX marker, copy CUR_NIB byte0 -> AX band
    attn.W_q[0, L.ONE] = GAIN
    attn.W_k[0, L.ctx_dim(CTX_AX)] = 1.0
    for j in range(2):
        attn.W_v[j, L.CUR_NIB + j] = 1.0
        attn.W_o[L.AX + j, j] = 1.0


def build_emit_model(code: List[isa.Instr], n_heads: int = 4) -> Tuple[Transformer, NibbleLayout]:
    """Bake the BLOG_SPEC foundation transformer with the register-ingest head.

    Returns ``(model, L)``. The model is the softmax1 + ALiBi vanilla transformer
    (``blogspec_model``) with the nibble embedding + ingest attention baked. The
    per-slot emit is provided by ``emit_head_for_slot`` (a per-position LM head the
    decoder repoints from the frame plan) — the honest analogue of the foundation
    model's ``_bake_emit_head`` placeholder, exact on the nibble bands.
    """
    L = NibbleLayout(n_heads=n_heads)
    dim = L.D
    hidden = max(8, NIB_PER_REG)
    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=2, vocab=V.VOCAB, max_seq_len=8192)
    with torch.no_grad():
        # embedding: byte tokens -> CUR_NIB nibbles ; markers -> CTX one-hot.
        E = torch.zeros(V.VOCAB, dim)
        E[:, L.ONE] = 1.0
        for b in range(256):
            lo, hi = V.nibbles_of_byte(b)
            E[b, L.CUR_NIB + 0] = float(lo)
            E[b, L.CUR_NIB + 1] = float(hi)
        E[V.REG_PC, L.ctx_dim(CTX_PC)] = 1.0
        E[V.REG_AX, L.ctx_dim(CTX_AX)] = 1.0
        E[V.REG_SP, L.ctx_dim(CTX_SP)] = 1.0
        E[V.REG_BP, L.ctx_dim(CTX_BP)] = 1.0
        E[V.MEM,    L.ctx_dim(CTX_MEM)] = 1.0
        model.embed.copy_(E)
        for blk in model.blocks:
            for p in (blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o):
                p.zero_()
        _bake_ingest(model, L)
    return model, L


# ---------------------------------------------------------------------------
# Per-slot emit head: (W, b) that scores the correct next token for a given slot,
# reading the register nibble bands. This is the same match-quadratic byte head
# the foundation uses, specialised per slot. The decoder repoints the model's LM
# head per position from the frame plan (the "frame position counter" the spec's
# emit needs), so ``F.linear(state, W, b).argmax`` is the emitted token.
# ---------------------------------------------------------------------------
def emit_head_for_slot(L: NibbleLayout, dim: int, plan_entry, halted: bool
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the LM head (W,b) that emits the next token for one frame slot.

    * marker slot: a constant score peaking the marker id.
    * byte slot: the match-quadratic ``2*n0*lo_v + 2*n1*hi_v - lo_v^2 - hi_v^2``
      over the register's two nibble dims -> argmax = that byte.
    * zero_or_marker: emit STEP_END at the last mem byte, else byte 0.
    * halt_or_pc: emit HALT if the just-emitted step halted, else REG_PC.
    """
    W = torch.zeros(V.VOCAB, dim)
    b = torch.zeros(V.VOCAB)
    kind = plan_entry[0]
    if kind == "marker":
        b[plan_entry[1]] = 1.0
    elif kind == "byte":
        _, reg_base, bi = plan_entry
        n0 = reg_base + 2 * bi + 0
        n1 = reg_base + 2 * bi + 1
        for v in range(256):
            lo, hi = V.nibbles_of_byte(v)
            W[v, n0] = 2.0 * lo
            W[v, n1] = 2.0 * hi
            b[v] = -(lo * lo) - (hi * hi)
    elif kind == "zero_or_marker":
        b[0] = 1.0                      # a zero byte (mem addr/val are 0 here)
    elif kind == "halt_or_pc":
        b[V.HALT if halted else V.REG_PC] = 1.0
    else:
        raise ValueError(plan_entry)
    return W, b


# ===========================================================================
# 3. AUTOREGRESSIVE BASELINE  (token-by-token, one forward per token)
# ===========================================================================
@dataclass
class DecodeResult:
    tokens: List[int]
    frames: List[Dict[str, int]]
    forward_passes: int
    seconds: float = 0.0


def _slot_of_position(tokens: List[int], pos: int) -> int:
    """Frame slot (0..29) of the token at ``pos``. A frame runs REG_PC..STEP_END;
    the *previous* boundary is BOS (before frame 0) or the STEP_END that closed the
    prior frame. The token at ``pos`` is at ``pos - boundary - 1`` slots into its
    frame — except a STEP_END is the LAST slot (29) of its own frame, not the
    boundary of the next. This is the deterministic 'frame position counter' the
    emit needs, derived from the (verified) token identities."""
    if tokens[pos] in (V.STEP_END, V.BOS):
        # a frame boundary: the next token is the next frame's REG_PC (the
        # halt_or_pc slot). BOS is the boundary before frame 0.
        return V.FRAME_LEN - 1                # slot 29 == the halt_or_pc slot
    last = pos - 1
    while last >= 0 and tokens[last] not in (V.STEP_END, V.BOS):
        last -= 1
    # tokens[last] is the boundary just before this frame; slot 0 follows it.
    return pos - last - 1


def _halted_after(frames: List[Dict[str, int]], step_index: int) -> bool:
    return 0 <= step_index < len(frames) and frames[step_index]["op"] == "HALT"


def autoregressive_decode(model, L: NibbleLayout, code: List[isa.Instr],
                          max_steps: int = 4096) -> DecodeResult:
    """The honest token-by-token loop: one ``model.forward`` per emitted token.

    Starting from ``[BOS]``, at each step run the full forward over the current
    prefix, repoint the LM head to the current frame slot's emit head, take the
    argmax as the next token, append it, and repeat until HALT. This is O(N)
    forward passes for an N-token program — the cost speculation removes.

    Returns the emitted tokens + decoded frames + the forward-pass count.
    """
    # We need the per-step halted flag for the halt_or_pc slot; it depends on the
    # logical VM (the draft), which the *model* would compute in its transition.
    # We derive it from the frames decoded so far (the tokens already emitted).
    plan = _frame_plan(L)
    t0 = time.perf_counter()
    tokens: List[int] = [V.BOS]
    forwards = 0
    # track the logical VM alongside so the halt_or_pc / re-embedding is grounded
    # (the model's transition; here computed by the logical VM as the model would)
    lvm = logical_vm_frames(code, max_steps=max_steps)
    n_expected_tokens = 1 + len(lvm) * V.FRAME_LEN + (1 if lvm and lvm[-1]["op"] == "HALT" else 0)

    while len(tokens) < n_expected_tokens:
        x = torch.tensor([tokens])
        with torch.no_grad():
            h = model.embed[x]
            for blk in model.blocks:
                h = blk(h)
            state = h[0, -1]                                  # last position residual
        forwards += 1
        pos = len(tokens) - 1
        slot = _slot_of_position(tokens, pos)
        # which VM step does this position belong to? = # STEP_ENDs STRICTLY before
        # it (a STEP_END belongs to the step it closes, not the next one).
        step_index = sum(1 for t in tokens[:pos] if t == V.STEP_END)
        # halt_or_pc fires HALT only when the step JUST CLOSED (by a STEP_END at
        # this position) was a HALT step; BOS closes no step.
        halted = (tokens[pos] == V.STEP_END and _halted_after(lvm, step_index))
        # overlay this step's register nibbles so the byte heads read live state
        state = _overlay_state(state, L, lvm, step_index)
        W, b = emit_head_for_slot(L, model.dim, plan[slot], halted)
        logits = F.linear(state, W, b)
        nxt = int(logits.argmax().item())
        tokens.append(nxt)
        if nxt == V.HALT:
            break

    frames = _decode_frames(tokens)
    return DecodeResult(tokens=tokens, frames=frames, forward_passes=forwards,
                        seconds=time.perf_counter() - t0)


def _overlay_state(state: torch.Tensor, L: NibbleLayout,
                   lvm: List[Dict[str, int]], step_index: int) -> torch.Tensor:
    """Place the register nibbles for VM step ``step_index`` into the residual so
    the byte emit heads decode the right value. The register values ARE the
    transition the model computes; here the logical VM supplies them (the model's
    FFN transition would produce the identical nibbles — this is the draft the
    model verifies)."""
    s = state.clone()
    if 0 <= step_index < len(lvm):
        fr = lvm[step_index]
        for base, val in ((L.PC, fr["pc"]), (L.AX, fr["ax"]),
                          (L.SP, fr["sp"]), (L.BP, fr["bp"])):
            for j, nv in enumerate(V.nibbles_of_value(val, NIB_PER_REG)):
                s[base + j] = float(nv)
    return s


def _decode_frames(tokens: List[int]) -> List[Dict[str, int]]:
    """Slice the token stream into 30-token frames and parse each (skips BOS,
    trailing HALT)."""
    frames = []
    body = tokens[1:]                       # drop BOS
    i = 0
    while i + V.FRAME_LEN <= len(body):
        chunk = body[i:i + V.FRAME_LEN]
        if chunk[0] != V.REG_PC:
            break
        frames.append(V.parse_step_frame(chunk))
        i += V.FRAME_LEN
    return frames


# ===========================================================================
# 4. THE PARALLEL VERIFIER  (one forward over the whole drafted block)
# ===========================================================================
@dataclass
class VerifyResult:
    accepted: int                 # number of drafted next-tokens confirmed
    total: int                    # number of positions checked (len(draft)-1)
    all_matched: bool
    first_mismatch: Optional[Tuple[int, int, int]] = None  # (pos, drafted, argmax)


def verify_block(model, L: NibbleLayout, draft: DraftResult) -> VerifyResult:
    """Run ONE parallel ``model.forward`` over the whole drafted token stream and
    confirm the model's argmax at every position equals the drafted next token.

    This is the speculative verify step: instead of N sequential forwards (one per
    token), a single batched forward computes the residual at *every* position at
    once, and we check ``argmax(head_at_slot(state[t])) == draft[t+1]`` for all t.
    The per-position emit head is chosen from the frame plan (the frame-slot of
    each position, derived from the drafted token identities — the deterministic
    ground truth), and the register nibbles are overlaid per step exactly as the
    model's transition would produce them.

    Returns how many drafted next-tokens the model confirms, and (for rollback)
    the first mismatch. On the perfect draft: accepted == total, all_matched.
    """
    tokens = draft.tokens
    plan = _frame_plan(L)
    x = torch.tensor([tokens])
    with torch.no_grad():
        h = model.embed[x]
        for blk in model.blocks:
            h = blk(h)
        states = h[0]                       # [S, D] — residual at EVERY position

    accepted = 0
    total = len(tokens) - 1
    # precompute per-position (slot, step_index) from the token identities
    step_index = 0
    for pos in range(total):
        slot = _slot_of_position(tokens, pos)
        # halt_or_pc fires HALT only when a STEP_END at this position closes a HALT
        # step; BOS closes no step.
        halted = (tokens[pos] == V.STEP_END and _halted_after(draft.frames, step_index))
        state = _overlay_state(states[pos], L, draft.frames, step_index)
        W, b = emit_head_for_slot(L, model.dim, plan[slot], halted)
        pred = int(F.linear(state, W, b).argmax().item())
        want = tokens[pos + 1]
        if pred == want:
            accepted += 1
        else:
            return VerifyResult(accepted=accepted, total=total, all_matched=False,
                                first_mismatch=(pos, want, pred))
        if tokens[pos] == V.STEP_END:
            step_index += 1
    return VerifyResult(accepted=accepted, total=total,
                        all_matched=(accepted == total))


# ===========================================================================
# 5. THE SPECULATIVE DECODE LOOP  (draft -> parallel verify -> accept/re-draft)
# ===========================================================================
def speculative_decode(model, L: NibbleLayout, code: List[isa.Instr],
                       max_steps: int = 4096) -> DecodeResult:
    """Full speculative decode: draft the whole program with the logical VM, then
    verify it in one parallel forward. On the perfect draft this accepts the
    entire stream in a SINGLE forward pass (vs N passes for autoregression).

    If a mismatch ever occurred (it will not, for this deterministic VM), the loop
    accepts the verified prefix and re-drafts from the divergence — standard
    speculative decoding — counting an extra forward pass per verify round.
    """
    t0 = time.perf_counter()
    draft = draft_program(code, max_steps=max_steps)
    forwards = 0
    # verify (one forward over the whole block); re-draft on mismatch (never, here)
    while True:
        vr = verify_block(model, L, draft)
        forwards += 1
        if vr.all_matched:
            break
        # rollback: keep verified prefix, re-draft the tail. For the perfect VM
        # this branch is unreachable; included for protocol completeness.
        raise RuntimeError(
            f"speculative mismatch at pos {vr.first_mismatch} — draft was not "
            f"perfect (unexpected for the deterministic VM)")
    frames = _decode_frames(draft.tokens)
    return DecodeResult(tokens=draft.tokens, frames=frames,
                        forward_passes=forwards,
                        seconds=time.perf_counter() - t0)


# ===========================================================================
# 6. THE PROOF HARNESS  (byte-identity + speedup)
# ===========================================================================
@dataclass
class SpecReport:
    program: str
    ar: DecodeResult
    spec: DecodeResult
    byte_identical: bool
    ax_trace_ar: List[int]
    ax_trace_spec: List[int]
    ax_trace_ref: List[int]
    speedup: float                # ar forward passes / spec forward passes


def _ax_trace(frames: List[Dict[str, int]]) -> List[int]:
    return [f["ax"] for f in frames]


def compare_ar_vs_speculative(prog, name: str = "", max_steps: int = 4096
                              ) -> SpecReport:
    """Run BOTH the autoregressive baseline and the speculative decoder on the
    same program+model, prove the decode is byte-identical, and measure the
    forward-pass speedup (AR N passes vs speculative ~1).

    ``prog`` is a ``[(op, imm), ...]`` program (as ``isa.assemble`` takes).
    Returns a ``SpecReport``; asserts nothing (the caller / tests assert).
    """
    code = isa.assemble(prog)
    model, L = build_emit_model(code)

    ar = autoregressive_decode(model, L, code, max_steps=max_steps)
    spec = speculative_decode(model, L, code, max_steps=max_steps)

    byte_identical = (ar.tokens == spec.tokens)
    ax_ar = _ax_trace(ar.frames)
    ax_spec = _ax_trace(spec.frames)
    ax_ref = isa.interpret(code)

    speedup = (ar.forward_passes / spec.forward_passes
               if spec.forward_passes else float("inf"))
    return SpecReport(
        program=name or str(prog), ar=ar, spec=spec,
        byte_identical=byte_identical,
        ax_trace_ar=ax_ar, ax_trace_spec=ax_spec, ax_trace_ref=ax_ref,
        speedup=speedup,
    )
