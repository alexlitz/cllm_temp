"""Spec-faithful vocabulary + 30-token register frame (BLOG_SPEC §Registers).

The model's token stream is bytes plus a handful of structural marker tokens.
Each VM step the model emits EXACTLY 30 tokens — the register frame:

    | 1  REG_PC  | 4 PC bytes  |
    | 1  REG_AX  | 4 AX bytes  |
    | 1  REG_SP  | 4 SP bytes  |
    | 1  REG_BP  | 4 BP bytes  |
    | 1  MEM     | 8 addr+val  |
    | 1  STEP_END|
    = 30 tokens  (little-endian bytes)

This emit->re-embed of the register bytes IS the re-quantization: a freshly
emitted byte token is an *exact* integer that re-enters the residual through the
(exact-integer) embedding table at the next position, annihilating the O(1e-6)
SwiGLU fp residue — replacing ``torch.round`` with the vanilla autoregressive
loop (BLOG_SPEC §Tokenization: "we write the up to date values each step ...
this ... is the re-quantization").
"""
from __future__ import annotations

from typing import List

# --- token id space -------------------------------------------------------
# 0..255  byte values
BYTE_LO, BYTE_HI = 0, 255
# structural markers
REG_PC   = 256
REG_AX   = 257
REG_SP   = 258
REG_BP   = 259
MEM      = 260
STEP_END = 261
HALT     = 262   # program-halt terminator (EXIT); ends generation (§Exiting)
BOS      = 263   # attention sink / position anchor
SEP      = 264   # system-prompt section separator (token 256 in the spec)

VOCAB = 265

MARKER_NAMES = {
    REG_PC: "REG_PC", REG_AX: "REG_AX", REG_SP: "REG_SP", REG_BP: "REG_BP",
    MEM: "MEM", STEP_END: "STEP_END", HALT: "HALT", BOS: "BOS", SEP: "SEP",
}

# order of the four registers in a step frame
REG_MARKERS = [REG_PC, REG_AX, REG_SP, REG_BP]

FRAME_LEN = 30   # tokens emitted per VM step (§Registers)


def bytes_le(value: int, n: int = 4) -> List[int]:
    """Little-endian byte decomposition of ``value`` into ``n`` bytes."""
    return [(value >> (8 * i)) & 0xFF for i in range(n)]


def value_from_bytes_le(bs: List[int]) -> int:
    v = 0
    for i, b in enumerate(bs):
        v |= (b & 0xFF) << (8 * i)
    return v


def build_step_frame(pc: int, ax: int, sp: int, bp: int,
                     mem_addr: int = 0, mem_val: int = 0) -> List[int]:
    """The 30-token register frame for one VM step (little-endian bytes)."""
    frame: List[int] = []
    frame += [REG_PC] + bytes_le(pc)
    frame += [REG_AX] + bytes_le(ax)
    frame += [REG_SP] + bytes_le(sp)
    frame += [REG_BP] + bytes_le(bp)
    frame += [MEM] + bytes_le(mem_addr) + bytes_le(mem_val)
    frame += [STEP_END]
    assert len(frame) == FRAME_LEN, len(frame)
    return frame


def parse_step_frame(frame: List[int]) -> dict:
    """Inverse of ``build_step_frame``: decode a 30-token frame back to register
    integers (used by the decoder / oracle to read what the model emitted)."""
    assert len(frame) == FRAME_LEN, len(frame)
    assert frame[0] == REG_PC and frame[5] == REG_AX and frame[10] == REG_SP \
        and frame[15] == REG_BP and frame[20] == MEM and frame[29] == STEP_END, frame
    return {
        "pc": value_from_bytes_le(frame[1:5]),
        "ax": value_from_bytes_le(frame[6:10]),
        "sp": value_from_bytes_le(frame[11:15]),
        "bp": value_from_bytes_le(frame[16:20]),
        "mem_addr": value_from_bytes_le(frame[21:25]),
        "mem_val": value_from_bytes_le(frame[25:29]),
    }


# --- nibble helpers (BLOG_SPEC §Internal Representation) ------------------
def nibbles_of_byte(b: int) -> List[int]:
    """The two 4-bit nibbles of a byte, little-endian: [low, high]."""
    return [b & 0xF, (b >> 4) & 0xF]


def nibbles_of_value(v: int, n_nibbles: int = 16) -> List[int]:
    """The ``n_nibbles`` little-endian 4-bit nibbles of a value."""
    return [(v >> (4 * j)) & 0xF for j in range(n_nibbles)]


# NB: nibbles are decoded back to integers ONLY via the LM byte-head's argmax
# (``blogspec_run._decode_byte_from_nibbles``) — there is deliberately no
# python-round nibble->int helper on the exec path, so the model's own head is
# the sole quantizer (the spec's vanilla re-quantization).
