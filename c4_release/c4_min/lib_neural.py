"""Neural integration of the c4_min runtime library onto the ONE unified model.

The runtime library (``nibble_runtime``: malloc / free / memset / memcmp) is pure
base-ISA BYTECODE, so it needs NO new neural op — the unified full-op interpreter
(``nibble_pure_forward_complete``) already decodes + executes every op the library
composes from (IMM/LEA/PSH/ADD/SUB/MUL/LI/LC/SI/SC/LT/BZ/JMP).  The ONE thing the
library needs that the default corpus does NOT is a **32-bit LOAD ADDRESS**: the
heap lives at ``0x30008`` (bump cell ``0x30000``), far above the low-byte window
the base corpus uses.

The unified §Memory KV head ALREADY keys on the full 32-bit address
(``ADDR_BITS = 32``): the store KEY (``ADDR_BIN``) is expanded from the store
frame's 4 address bytes, and the store VALUE carries all 4 bytes.  The ONLY 8-bit
bottleneck is the LI/LC address QUERY: ``compile_mem_prep`` expands the query from
the 8-bit scalar ``AX_VAL`` (``compile_addr_expand`` ``n_bits=8``), so a load from
``0x30000`` queries ``0x00`` and mismatches every 32-bit store key.

:func:`compile_mem_prep_addr32` supplies the fix: expand the LI/LC query from ALL
8 register nibbles of the ingested AX band (32 bits), reusing the SAME
``compile_nibble_addr_expand`` the stack-pop head already uses for the 32-bit SP
address.  :func:`build_lib_model` builds the canonical model and SWAPS the
``mem-prep`` block's FFN for the widened one — an ADDITIVE, post-build patch that
does not touch the shared build function.  For an address that fits a byte (the
whole base corpus) the widened query produces the IDENTICAL low 8 bits and zeros
bits 8..31, so the base corpus stays byte-identical.
"""
from __future__ import annotations

from typing import Dict

import torch

from . import isa
from .nibble_vm import _load_ffn, S, SILU_S
from .nibble_pure_forward import (
    _flag_from_ops, _clear_band_gated, _concat_specs,
)
from .nibble_pure_forward_complete import (
    build_pure_forward_complete_model, compile_nibble_addr_expand,
    PureForwardCompleteLayout,
)


def compile_mem_prep_addr32(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Like ``nibble_pure_forward.compile_mem_prep`` but the LI/LC load-address
    QUERY (``QRY_BIN``) is expanded from ALL 8 AX register nibbles (32 bits),
    not the low byte — so a load from a 32-bit heap address (``0x30008``) matches
    the 32-bit store key.  Byte-identical low 8 bits for a byte-sized address.

    Components (same as ``compile_mem_prep`` except the address expand):
      * ``IS_LOAD`` = OP_IS[LI] + OP_IS[LC]  (SET, opcode-gated).
      * ``QRY_BIN`` <- 32 bits of the AX nibble band (ungated; read only when
        IS_LOAD gates the CAM).
      * clear the AX nibble band on a load (so the CAM's additive write is a SET).
    """
    specs = []
    specs.append(_flag_from_ops(L, L.IS_LOAD, [isa.LI, isa.LC], dim))
    # QRY_BIN <- bits of the AX nibble band (all 8 nibbles = 32 bits).  The AX band
    # holds the full 32-bit load address at LI/LC time (ingested from the prior
    # frame's 4 AX bytes), so this recovers the whole heap address.
    specs.append(compile_nibble_addr_expand(L, L.AX, L.QRY_BIN, dim, n_nibbles=8))
    specs.append(_clear_band_gated(L, L.AX, 8, dim, gate_ops=[isa.LI, isa.LC]))
    return _concat_specs(specs, dim)


def build_lib_model(code_size: int = 32, recurrent_divmod: bool = True,
                    addr32: bool = True):
    """Build the canonical unified full-op model, then (if ``addr32``) SWAP the
    ``mem-prep`` block's FFN for the 32-bit-address-query widening so the runtime
    library's heap addresses (0x30008) survive the §Memory CAM.

    Returns ``(model, L)``.  With ``addr32=False`` this is exactly
    ``build_pure_forward_complete_model`` (the base corpus model).
    """
    model, L = build_pure_forward_complete_model(
        code_size=code_size, recurrent_divmod=recurrent_divmod)
    if not addr32:
        return model, L
    # Locate the mem-prep block (records live in L._block_names, in apply order for
    # the recurrent build; the physical block list is model._phys_blocks when the
    # recurrence remapped model.blocks).
    names = L._block_names
    phys = getattr(model, "_phys_blocks", model.blocks)
    idx = names.index("mem-prep")
    hidden = phys[idx].ffn.W_up.shape[0]
    spec = compile_mem_prep_addr32(L, L.D)
    assert spec["W_up"].shape[0] <= hidden, (
        f"widened mem-prep hidden {spec['W_up'].shape[0]} > block hidden {hidden}")
    with torch.no_grad():
        _load_ffn(phys[idx].ffn, spec, hidden)
    return model, L
