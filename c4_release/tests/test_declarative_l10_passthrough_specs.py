"""Tests for declarative L10 byte-passthrough attention specs."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _bake_layer10_byte_passthrough_head,
    _bake_layer10_bp_byte_passthrough_head,
    _bake_layer10_carry_relay_head,
    _bake_layer10_psh_stack0_passthrough_head,
    _bake_layer10_sp_byte_passthrough_head,
    _bake_layer10_stack0_byte_relay_head,
)
from neural_vm.vm_step import (  # noqa: E402
    AutoregressiveAttention,
    _SetDim,
    _set_layer10_byte_passthrough,
    _set_layer10_bp_byte_passthrough,
    _set_layer10_carry_relay,
    _set_layer10_psh_stack0_passthrough,
    _set_layer10_sp_byte_passthrough,
    _set_layer10_stack0_byte_relay,
)


def _new_attention():
    d_model = 512
    num_heads = 8
    return AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=10, use_flash_attention=False
    )


def _assert_attention_equal(legacy, generated):
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name
    assert torch.equal(legacy.alibi_slopes, generated.alibi_slopes)


def test_l10_ax_byte_passthrough_spec_matches_legacy_helper():
    legacy = _new_attention()
    generated = _new_attention()
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_layer10_byte_passthrough(legacy, 100.0, _SetDim, hd)
        _bake_layer10_byte_passthrough_head(generated, _SetDim, 100.0, hd)

    _assert_attention_equal(legacy, generated)


def test_l10_carry_relay_spec_matches_legacy_helper():
    legacy = _new_attention()
    generated = _new_attention()
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_layer10_carry_relay(legacy, 100.0, _SetDim, hd)
        _bake_layer10_carry_relay_head(generated, _SetDim, 100.0, hd)

    _assert_attention_equal(legacy, generated)


def test_l10_sp_byte_passthrough_spec_matches_legacy_helper():
    legacy = _new_attention()
    generated = _new_attention()
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_layer10_sp_byte_passthrough(legacy, 100.0, _SetDim, hd)
        base = 2 * hd
        for slot in (34, 35):
            legacy.W_q[base + slot, _SetDim.CMP + 4] = -600.0
            legacy.W_q[base + slot, _SetDim.OP_JSR] = -600.0
        _bake_layer10_sp_byte_passthrough_head(generated, _SetDim, 100.0, hd)

    _assert_attention_equal(legacy, generated)


def test_l10_sp_byte_passthrough_generic_chain_blocks_sp_marker_rows():
    attn = _new_attention()
    hd = attn.W_q.shape[0] // attn.num_heads
    base = 2 * hd

    with torch.no_grad():
        _bake_layer10_sp_byte_passthrough_head(attn, _SetDim, 100.0, hd)

    q0 = attn.W_q[base]
    assert q0[_SetDim.IS_BYTE] == 100.0
    assert q0[_SetDim.HAS_SE] == 200.0
    assert q0[_SetDim.PSH_AT_SP] == -200.0
    assert q0[_SetDim.MARK_SP] == -200.0

    marker = torch.zeros(512)
    marker[_SetDim.CONST] = 1.0
    marker[_SetDim.HAS_SE] = 1.0
    marker[_SetDim.MARK_SP] = 1.0
    marker[_SetDim.H1 + 2] = 1.0
    marker[_SetDim.CMP + 3] = 4.0
    assert torch.dot(q0, marker) < 0

    sp_byte = marker.clone()
    sp_byte[_SetDim.MARK_SP] = 0.0
    sp_byte[_SetDim.IS_BYTE] = 1.0
    assert torch.dot(q0, sp_byte) > 0

    assert attn.W_q[base + 34, _SetDim.MARK_SP] == 300.0
    assert attn.W_q[base + 35, _SetDim.MARK_SP] == 300.0
    assert attn.W_q[base + 34, _SetDim.CMP + 4] == -600.0
    assert attn.W_q[base + 35, _SetDim.OP_JSR] == -600.0

    jsr_marker = torch.zeros(512)
    jsr_marker[_SetDim.CONST] = 1.0
    jsr_marker[_SetDim.HAS_SE] = 1.0
    jsr_marker[_SetDim.MARK_SP] = 1.0
    jsr_marker[_SetDim.CMP + 4] = 1.0
    jsr_marker[_SetDim.OP_JSR] = 5.0
    assert torch.dot(attn.W_q[base + 34], jsr_marker) < 0
    assert torch.dot(attn.W_q[base + 35], jsr_marker) < 0


def test_l10_bp_byte_passthrough_spec_matches_legacy_helper():
    legacy = _new_attention()
    generated = _new_attention()
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_layer10_bp_byte_passthrough(legacy, 100.0, _SetDim, hd)
        _bake_layer10_bp_byte_passthrough_head(generated, _SetDim, 100.0, hd)

    _assert_attention_equal(legacy, generated)


def test_l10_psh_stack0_passthrough_spec_matches_legacy_helper():
    legacy = _new_attention()
    generated = _new_attention()
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_layer10_psh_stack0_passthrough(legacy, 100.0, _SetDim, hd)
        _bake_layer10_psh_stack0_passthrough_head(generated, _SetDim, 100.0, hd)

    _assert_attention_equal(legacy, generated)


def test_l10_stack0_byte_relay_spec_matches_legacy_helper():
    legacy = _new_attention()
    generated = _new_attention()
    hd = legacy.W_q.shape[0] // legacy.num_heads

    with torch.no_grad():
        _set_layer10_stack0_byte_relay(legacy, 100.0, _SetDim, hd)
        _bake_layer10_stack0_byte_relay_head(generated, _SetDim, 100.0, hd)
        # The compiler-owned nonbitwise head intentionally strengthens the
        # STACK0 high-byte relay so L10 ADD/SUB base propagation sees the same
        # ALU amplitude it was calibrated for in the full compact model.
        base = 5 * hd
        for k in range(16):
            legacy.W_o[_SetDim.ALU_LO + k, base + 1 + k] = 6.0
            legacy.W_o[_SetDim.ALU_HI + k, base + 17 + k] = 6.0

    _assert_attention_equal(legacy, generated)


def test_l10_stack0_persistence_store_route_requires_mem_store():
    attn = _new_attention()
    hd = attn.W_q.shape[0] // attn.num_heads
    base = 6 * hd

    with torch.no_grad():
        _bake_layer10_stack0_byte_relay_head(attn, _SetDim, 100.0, hd)

    q7 = attn.W_q[base + 7]
    marker = torch.zeros(512)
    marker[_SetDim.CONST] = 1.0
    marker[_SetDim.MARK_STACK0] = 1.0
    marker[_SetDim.HAS_SE] = 1.0
    marker[_SetDim.CMP + 3] = 4.0

    assert torch.dot(q7, marker) < 0
    marker[_SetDim.MEM_STORE] = 1.0
    assert torch.dot(q7, marker) > 0
    marker[_SetDim.MEM_ADDR_SRC] = 1.0
    assert torch.dot(q7, marker) < 0
