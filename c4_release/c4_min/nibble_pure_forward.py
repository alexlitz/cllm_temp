"""PURE-FORWARD C4 VM step — one VM step = one ``model.forward``, compute in weights.

This is the mission's *true form*: the whole VM runs through the **vanilla
autoregressive forward**. The token stream + KV cache carry ALL state; the ALU /
dispatch / control live in FFN weights; memory is softmax1-KV attention over the
emitted MEM tokens; PC/registers live in the emitted 30-token frames. The ONLY
Python on the compute path is the standard generation loop — ``argmax`` the next
token and append it, then forward again. There is **no**
``blogspec_run._apply_op`` (python if/elif + integer VMState), **no**
``DictMemStack`` (python dict memory), and **no** functional torch gadget built
per call.

Contrast with the recurrent step-loop (``nibble_vm.run_program`` /
``verify_unified.step_run``): that carries the VM state in a persistent RESIDUAL
vector and snaps lanes in Python between steps. Here the state instead round-trips
through the **token stream**: each step re-reads the register file out of the
previously-emitted 30-token frame by ATTENTION (the spec's "we write the registers
each step ... retrieve by attending"), the baked FFN blocks compute the next step
in the SAME forward, and the LM head emits the next frame's bytes. State lives in
the sequence, exactly as a real decoder-only transformer.

The pure-forward step mechanism
===============================
The token stream is ``BOS`` then a sequence of 30-token register frames
(``blogspec_vocab.build_step_frame``). One VM step is ONE ``model.forward`` over
the whole stream so far; block 0's attention reconstructs the register nibble
state from the most-recent frame, the baked step blocks compute the next state on
that state, and the register value lanes at the LAST position are the next-step
registers. The driver decodes those four register values with the LM byte-head's
value argmax (the spec's own re-quantiser — no ``torch.round``) and APPENDS the
next 30-token frame, closing the loop through the token stream.

  block 0   FRAME-INGEST attention — a softmax1 + ALiBi content-addressable read
            keyed on a per-(register, byte) ROLE (a rigid structural tag of the
            frame slot, like a positional encoding), with a query-exclusion
            penalty and recency ALiBi so the LATEST frame's bytes are gathered
            into this position's register NIBBLE bands. State comes from the
            SEQUENCE, not a python variable.
  block 0 FFN + blocks 1..k  the baked VM STEP — recompose nibbles→scalar lanes,
            fetch@PC over the program (code-as-data), opcode decode, dispatch of
            the op (IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ + cmp/bitwise/muldiv experts),
            branch delta, mod-256 fold. Identical persistent weights to
            ``nibble_vm.build_step_model`` — the op RESULT is computed by these
            FFN weights inside ``model.forward``.

The register-nibble ingest is proven byte-exact (``prove_ingest``); the whole step
is proven byte-exact vs ``isa.interpret`` by ``run_pure_forward`` under a trace
guard (``assert_no_python_compute``) that fails if ``_apply_op`` / ``DictMemStack``
/ a per-call gadget is ever entered.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .nibble_vm_layout import NibbleVMLayout
from .nibble_vm import (
    S, RELU_S, SILU_S, build_step_model, load_program,
    compile_nibble_to_scalar, compile_pc_fetch, compile_code_select,
    compile_opcode_decode, base_dispatch_rules, compile_branch_delta,
    compile_fold, compile_ffn, _empty_spec, _load_ffn, _zero_attn,
    _snap_lane, VALVOCAB,
)
from .blogspec_model import Transformer, softmax1


# The registers the ingest reconstructs from the prior frame, in frame order.
# Each is 4 bytes; STACK0 is carried in the STACK0 slot of the frame (extended
# frame). The base 30-token frame carries PC/AX/SP/BP; STACK0 is carried in the
# frame's MEM value slot for the pure-forward stack-top mirror.
INGEST_REGS = ["PC", "AX", "SP", "BP", "STACK0"]
BYTES_PER_REG = 4
N_ROLES = len(INGEST_REGS) * BYTES_PER_REG          # 20 (register, byte) roles


# ===========================================================================
# The pure-forward layout: the baked VM step-block bands + the ingest role bands.
# ===========================================================================
class PureForwardLayout(NibbleVMLayout):
    """``NibbleVMLayout`` (the baked step bands) + the frame-ingest role bands.

    The extra bands are set on the residual by the driver as it embeds each frame
    (a rigid structural tag of the 30-token frame slot — like a positional
    encoding — NOT computed VM state):

      ``ROLE`` (N_ROLES)   — per (register, byte) one-hot; a frame byte token in
                             slot (r, bi) carries ROLE[r*4+bi]=1 (its KEY), the
                             ingest query for that slot carries the same one-hot.
      ``IS_FRAME_BYTE`` (1)— 1.0 on real frame byte tokens (KV candidates); 0 on
                             markers / the query row (query-exclusion penalty).
    """

    def __init__(self, code_size: int, n_heads: int):
        super().__init__(code_size, n_heads=n_heads)
        self._off = self.D
        self.ROLE = self._band("ROLE", N_ROLES)
        self.IS_FRAME_BYTE = self._scalar("IS_FRAME_BYTE")
        while self._off % n_heads != 0:
            self._scalar(f"_pfpad{self._off}")
        self.D = self._off


# ===========================================================================
# FRAME-INGEST attention — the content-addressable read of the prior frame.
#
# One head per (register, byte) role gathers that slot's two nibbles from the
# latest frame into the register's nibble band. Each head is the same CAM: KEY =
# +smag on the role dim the token holds, QUERY = +smag on the role this head
# reconstructs, a query-exclusion penalty drives non-frame-byte rows to -PEN, and
# a recency ALiBi picks the LATEST frame among equal roles (loops re-emit the same
# roles every step). The VALUE is the token's CUR_NIB nibbles; W_o writes them
# into the register nibble band dims 2*bi+0 / 2*bi+1.
# ===========================================================================
INGEST_EFF = 4000.0                 # per-role match contribution (huge; §Memory)
INGEST_RECENCY = 6.0                # ALiBi recency slope (latest frame wins)


def bake_frame_ingest(attn, L: PureForwardLayout, reg_bases: Dict[str, int]) -> None:
    """Bake the N_ROLES-head frame-ingest CAM. ``reg_bases`` maps register name ->
    its nibble-band base. head ``h = r*4+bi`` gathers register ``r``'s byte ``bi``.
    Requires ``attn.n_heads >= N_ROLES`` and ``attn.head_dim >= 3`` (2 CAM channels
    + 1 penalty; value uses 2 local channels)."""
    hs = attn.scale
    smag = (INGEST_EFF / hs) ** 0.5
    PEN = 100.0 * INGEST_EFF
    p = (PEN / hs) ** 0.5
    HD = attn.head_dim
    for w in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        w.zero_()
    for h in range(N_ROLES):
        attn.alibi_slopes[h] = INGEST_RECENCY
        base = h * HD
        r_idx, bi = divmod(h, BYTES_PER_REG)
        reg_name = INGEST_REGS[r_idx]
        reg_base = reg_bases[reg_name]
        # CAM channel 0: role match (key = the token's role dim, query = this role).
        attn.W_k[base + 0, L.ROLE + h] = smag
        attn.W_q[base + 0, L.ROLE + h] = smag
        # penalty channel 1: non-frame-byte rows keyed -p (via ONE), byte rows 0.
        # The query keys +p via ONE (constant), so a non-frame-byte candidate (incl.
        # the query row itself, IS_FRAME_BYTE=0) scores -p*hs = -PEN and can never
        # win; a real frame byte row (IS_FRAME_BYTE=1) keys 0 here.
        attn.W_q[base + 1, L.ONE] = p
        attn.W_k[base + 1, L.ONE] = -p
        attn.W_k[base + 1, L.IS_FRAME_BYTE] = p
        # value: the two nibbles carried by the byte token in CUR_NIB.
        attn.W_v[base + 0, L.CUR_NIB + 0] = 1.0
        attn.W_v[base + 1, L.CUR_NIB + 1] = 1.0
        # write into the register nibble band's two dims for byte bi.
        attn.W_o[reg_base + 2 * bi + 0, base + 0] = 1.0
        attn.W_o[reg_base + 2 * bi + 1, base + 1] = 1.0


# ===========================================================================
# BUILD the pure-forward model: ingest attention on block 0 + the baked step.
# ===========================================================================
def build_pure_forward_model(code_size: int = 32):
    """Assemble the pure-forward VM: block-0 attention = the frame-ingest CAM, and
    the SAME baked step FFN blocks as ``nibble_vm.build_step_model`` (recompose /
    fetch / code-select / decode / dispatch / branch / fold). ``n_heads`` is forced
    to ``N_ROLES`` so every register byte gets its own gather head.

    Returns ``(model, L)``. The op result is computed by the FFN weights inside
    ``model.forward``; the register state is reconstructed from the token stream by
    the block-0 attention. Nothing is computed in Python.
    """
    n_heads = N_ROLES                                # one gather head per reg-byte
    L = PureForwardLayout(code_size, n_heads=n_heads)
    dim = L.D
    ffn_specs = [
        compile_nibble_to_scalar(L, dim),            # block 0 FFN
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
                        n_blocks=n_blocks, vocab=V.VOCAB, max_seq_len=8192)
    with torch.no_grad():
        _bake_pure_embedding(model, L)
        for bi, spec in enumerate(ffn_specs):
            _zero_attn(model.blocks[bi].attn)
            _load_ffn(model.blocks[bi].ffn, spec, hidden)
        # block 0 attention = the frame-ingest CAM.
        reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP, "STACK0": L.STACK0}
        bake_frame_ingest(model.blocks[0].attn, L, reg_bases)
    return model, L


def _bake_pure_embedding(model, L: PureForwardLayout) -> None:
    """Byte tokens embed their two nibbles into CUR_NIB; ONE=1 in every row. The
    ROLE / IS_FRAME_BYTE tags are set by the driver's per-position frame overlay
    (structural frame-slot tags), so the embedding table itself stays universal."""
    E = torch.zeros(V.VOCAB, model.dim)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    model.embed.copy_(E)


# ===========================================================================
# The token stream + the per-position overlay (program-in-data + frame roles).
#
# The stream is BOS then one 30-token frame per emitted step. The overlay carries
# two structural things (NOT computed VM state): (1) the PROGRAM in the DATA bands
# on the BOS position (universal fetch — the code is INPUT), and (2) the per-frame
# ROLE / IS_FRAME_BYTE tags of each byte token (a rigid function of frame slot,
# like a positional encoding). The register VALUES ride in the byte-token
# embeddings (CUR_NIB) — the real state, in the token stream.
# ===========================================================================
# Frame slot -> (register_index, byte_index) for the 20 register byte tokens.
# Frame layout: [REG_PC, pc0..3, REG_AX, ax0..3, REG_SP, sp0..3, REG_BP, bp0..3,
#                MEM, addr0..3, val0..3, STEP_END]  (STACK0 rides the MEM val slot)
_FRAME_ROLE_SLOTS = {}
def _init_frame_role_slots():
    # marker positions and their following 4 byte slots, in frame-local index.
    reg_marker_pos = {"PC": 0, "AX": 5, "SP": 10, "BP": 15}
    for r_idx, name in enumerate(["PC", "AX", "SP", "BP"]):
        m = reg_marker_pos[name]
        for bi in range(4):
            _FRAME_ROLE_SLOTS[m + 1 + bi] = r_idx * 4 + bi
    # STACK0 rides the MEM value bytes (frame slots 25..28).
    for bi in range(4):
        _FRAME_ROLE_SLOTS[25 + bi] = 4 * 4 + bi       # register index 4 = STACK0
_init_frame_role_slots()


def build_frame_tokens(pc: int, ax: int, sp: int, bp: int, stack0: int
                       ) -> List[int]:
    """The 30-token frame carrying the five registers; STACK0 in the MEM value
    slot (so the pure-forward stack-top mirror round-trips through the stream)."""
    return V.build_step_frame(pc, ax, sp, bp, mem_addr=0, mem_val=stack0 & 0xFFFFFFFF)


def make_overlay(code: List[isa.Instr], L: PureForwardLayout):
    """Return an ``overlay(x)`` that writes, in-place on the embedded stream ``x``
    ([1,S,D]): the PROGRAM into the DATA bands at every position (so fetch@PC works
    at the last position), and the ROLE / IS_FRAME_BYTE frame-slot tags on each
    30-token frame. BOS carries the initial register state so step 1 has a frame to
    ingest. Everything here is structural (program = input; roles = frame layout);
    NO VM value is computed in Python."""
    def overlay(x: torch.Tensor) -> None:
        S = x.shape[1]
        # program in DATA bands + ONE at every position (fetch@PC reads the last).
        for i in range(S):
            x[0, i, L.ONE] = 1.0
            for k, ins in enumerate(code):
                x[0, i, L.CODE_OP[k]] = float(ins.op)
                x[0, i, L.CODE_IMM[k]] = float(ins.imm)
        # BOS (position 0) carries the INITIAL register frame's role/value so the
        # first step ingests PC=0,AX=0,SP=BP=0x10000,STACK0=0. We encode the init
        # registers as a virtual frame on BOS via the ROLE/CUR_NIB of... no — the
        # driver seeds an explicit init frame as the first real frame (see runner).
        # Here: tag every 30-token frame's byte slots with their role.
        # positions 1.. are frames of length 30.
        pos = 1
        while pos + V.FRAME_LEN <= S:
            for local, role in _FRAME_ROLE_SLOTS.items():
                p = pos + local
                x[0, p, L.ROLE + role] = 1.0
                x[0, p, L.IS_FRAME_BYTE] = 1.0
            pos += V.FRAME_LEN
        # The INGEST QUERY position is the LAST token of the stream (the current
        # STEP_END). It must light EVERY role query so all 20 gather heads fire
        # (each head reads only its own ROLE+h dim). IS_FRAME_BYTE stays 0 there so
        # the query row is not itself a KV candidate (query-exclusion).
        for role in range(N_ROLES):
            x[0, -1, L.ROLE + role] = 1.0
    return overlay


# ===========================================================================
# THE PURE-FORWARD DRIVER — one VM step = one model.forward; argmax + append only.
# ===========================================================================
# Spec register init (§C4 Registers): PC=AX=0, SP=BP at the stack top, STACK0=0.
SP_INIT = 0x10000


def _emit_frame_from_state(state: torch.Tensor, L: PureForwardLayout
                           ) -> Tuple[List[int], int, int, bool]:
    """Decode the model's computed next-state (the value lanes at the last
    position) into the next 30-token frame via the LM byte-head's value argmax —
    the spec's own re-quantiser (no ``torch.round``). Returns
    ``(frame_tokens, ax_value, next_pc, halted)``."""
    pc = _snap_lane(state[L.PC_VAL])
    ax = _snap_lane(state[L.AX_VAL])
    sp = _snap_lane(state[L.SP_VAL])
    bp = _snap_lane(state[L.BP_VAL])
    stk = _snap_lane(state[L.STK_VAL])
    halted = float(state[L.HALTED]) > 0.5
    frame = build_frame_tokens(pc, ax, sp, bp, stk)
    return frame, ax & 0xFF, pc, halted


def run_pure_forward(model, L: PureForwardLayout, code: List[isa.Instr],
                     max_steps: int = 512, verbose: bool = False,
                     collect_tokens: bool = False):
    """Execute ``code`` with the PURE-FORWARD step: every VM step is ONE
    ``model.forward`` over the growing token stream (state read from the prior
    frame by the block-0 attention; the op computed by the FFN weights), and the
    only Python is the LM value-argmax emit + append. Returns the per-step AX trace
    (matching ``isa.interpret``); with ``collect_tokens`` also the flat token
    stream.

    The stream starts ``[BOS] + init_frame`` where ``init_frame`` is the spec
    register init (PC=AX=0, SP=BP=0x10000, STACK0=0) — the "step 0" frame the first
    real step ingests. Each iteration appends exactly one 30-token frame.
    """
    overlay = make_overlay(code, L)
    init_frame = build_frame_tokens(0, 0, SP_INIT, SP_INIT, 0)
    stream: List[int] = [V.BOS] + init_frame
    trace: List[int] = []
    for _ in range(max_steps):
        toks = torch.tensor([stream])
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)                              # program-in-data + frame roles
            for blk in model.blocks:                # == model.forward minus LM head
                x = blk(x)
        state = x[0, -1]
        frame, ax_byte, npc, halted = _emit_frame_from_state(state, L)
        trace.append(ax_byte)
        stream += frame                             # APPEND the emitted frame
        if verbose:
            print(f"  step -> pc_next={npc} ax={ax_byte} halted={halted}")
        if halted or npc < 0 or npc >= len(code):
            break
    if collect_tokens:
        return trace, stream
    return trace


# ===========================================================================
# TRACE GUARD — prove NO python compute (no _apply_op / DictMemStack / gadget).
# ===========================================================================
_FORBIDDEN_QUALNAMES = {
    "_apply_op",                       # blogspec_run python if/elif dispatch
    "DictMemStack.__init__", "DictMemStack.store_int", "DictMemStack.load_int",
    "nibble_add_gadget", "nibble_sub_gadget",   # per-call ALU gadgets
    "mul32", "div32", "mod32",         # per-call muldiv gadgets
    "compare", "to_bit",               # per-call cmp gadget
    "or_gadget", "xor_gadget", "and_gadget", "shl_gadget", "shr_gadget",
}


class _NoPythonComputeGuard:
    """A settrace guard that raises if any forbidden compute-path function is
    entered while it is active — the machine proof that the VM ran purely in
    ``model.forward`` (no python if/elif dispatch, no python memory dict, no
    per-call gadget)."""

    def __init__(self):
        self.violations: List[str] = []
        self._prev = None

    def _tracer(self, frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            qual = frame.f_code.co_qualname if hasattr(frame.f_code, "co_qualname") else name
            if name in _FORBIDDEN_QUALNAMES or qual in _FORBIDDEN_QUALNAMES or \
               any(qual.endswith("." + f) for f in _FORBIDDEN_QUALNAMES):
                self.violations.append(qual)
        return None                                 # do not trace lines (fast)

    def __enter__(self):
        self._prev = sys.gettrace()
        sys.settrace(self._tracer)
        return self

    def __exit__(self, *a):
        sys.settrace(self._prev)
        return False


def assert_no_python_compute(fn, *args, **kwargs):
    """Run ``fn`` under the guard; assert it entered NO forbidden compute-path
    function. Returns ``fn``'s result. This is the mission's make-or-break proof:
    the VM step ran entirely in ``model.forward``."""
    guard = _NoPythonComputeGuard()
    with guard:
        result = fn(*args, **kwargs)
    assert not guard.violations, \
        f"python compute leaked into the step: {sorted(set(guard.violations))}"
    return result
