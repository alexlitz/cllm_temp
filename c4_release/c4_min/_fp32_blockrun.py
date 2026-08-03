"""Standalone FP32 megablock runner + a minimal layout, for bit-exact testing.

Applies a list of (name, spec) SwiGLU FFN blocks to a residual vector so a float
op's block stack can be validated against ``isa.f32_op_bits`` WITHOUT baking the
whole VM.  The residual carries the register nibble bands; a helper seeds a raw
32-bit float bit pattern into a register's 8 low nibbles and reads a result band
back to an int.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

os.environ.setdefault("C4_FLOAT_OPS", "1")

from . import isa
from .nibble_vm_layout import NibbleVMLayout


def make_layout(n_heads: int = 8, code_size: int = 4) -> NibbleVMLayout:
    return NibbleVMLayout(code_size, n_heads=n_heads)


def apply_block(x: torch.Tensor, spec: Dict[str, torch.Tensor]) -> torch.Tensor:
    up = F.linear(x, spec["W_up"]) + spec["b_up"]
    gate = F.linear(x, spec["W_gate"]) + spec["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, spec["W_down"], spec["b_down"])


def apply_blocks(x: torch.Tensor, blocks: List[Tuple[str, Dict]]) -> torch.Tensor:
    for _name, spec in blocks:
        x = apply_block(x, spec)
    return x


def seed_bits(x: torch.Tensor, reg_base: int, bits: int) -> None:
    """Write the 8 low nibbles of a 32-bit value into a register band."""
    for j in range(8):
        x[reg_base + j] = float((bits >> (4 * j)) & 0xF)


def read_nibbles(x: torch.Tensor, base: int, n: int = 8) -> int:
    """Read ``n`` nibbles at ``base`` (rounded) back into an int."""
    v = 0
    for j in range(n):
        nib = int(round(float(x[base + j]))) & 0xF
        v |= nib << (4 * j)
    return v


def new_residual(L, dim: int) -> torch.Tensor:
    x = torch.zeros(dim)
    x[L.ONE] = 1.0
    return x


def set_op_is(x: torch.Tensor, L, op: int) -> None:
    x[L.OP_IS + op] = 1.0
