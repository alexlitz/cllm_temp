"""Tests for declarative attention bake specs."""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.ops.l0_ops import make_layer0_threshold_attn_op  # noqa: E402
from neural_vm.unified_compiler.ops.l7_ops import _layer7_memory_head_specs  # noqa: E402
from neural_vm.unified_compiler.ops.l4_ops import (  # noqa: E402
    _layer4_pc_relay_head_specs,
)
from neural_vm.unified_compiler.ops.l3_ops import (  # noqa: E402
    _stack0_carry_head_spec,
)
from neural_vm.unified_compiler.ops.l5_ops import (  # noqa: E402
    _fetch_head_specs,
    make_fetch_op,
)
from neural_vm.unified_compiler.ops.l6_ops import (  # noqa: E402
    _bake_layer6_attn_spec,
    _bake_layer6_relay_heads_spec,
    _layer6_bz_bnz_relay_head_spec,
)
from neural_vm.unified_compiler.ops.l8_ops import (  # noqa: E402
    _layer8_multibyte_fetch_head_spec,
    _layer8_sp_gather_head_specs,
)
from neural_vm.unified_compiler.ops.l9_ops import (  # noqa: E402
    _layer9_lev_addr_relay_head_spec,
    _layer9_lev_bp_to_pc_relay_head_spec,
)
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.setup_helpers import (  # noqa: E402
    _set_layer9_lev_addr_relay,
    _set_layer9_lev_bp_to_pc_relay,
)
from neural_vm.vm_step import (  # noqa: E402
    AutoregressiveAttention,
    _SetDim,
    _set_threshold_attn,
    _set_bz_bnz_relay,
    _set_stack0_carry_attn,
    _set_layer4_pc_relay,
    _set_layer5_fetch,
    _set_layer6_attn,
    _set_layer6_relay_heads,
    _set_layer7_memory_heads,
    _set_layer8_multibyte_fetch,
    _set_layer8_sp_gather,
)


def _setdim_positions():
    return {
        name: value
        for name, value in vars(_SetDim).items()
        if name.isupper() and isinstance(value, int)
    }


def test_layer0_threshold_attn_production_bake_preserves_stack0_byte0_cutoff():
    """L0 H1 must not be sharpened past the STACK0 byte-0 cutoff."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads
    slope = 10.0
    thresholds = [3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5]
    out_bases = [
        _SetDim.H0,
        _SetDim.H1,
        _SetDim.H2,
        _SetDim.H3,
        _SetDim.H4,
        _SetDim.H5,
        _SetDim.H6,
        _SetDim.H7,
    ]

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=0, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=0, use_flash_attention=False
    )

    with torch.no_grad():
        _set_threshold_attn(legacy, thresholds, out_bases, slope, hd, BD=_SetDim)
        op = make_layer0_threshold_attn_op()
        op.bake_fn(SimpleNamespace(attn=generated), _setdim_positions(), 100.0)

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name

    h1_base = hd
    assert generated.W_k[h1_base, _SetDim.IS_MARK].item() == 4.5
    assert generated.W_k[h1_base, _SetDim.IS_MARK].item() < 6.0


def test_layer7_memory_heads_declarative_byte_identical_to_legacy_helper():
    """The declarative L7 memory-head specs reproduce legacy matrix writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=7, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=7, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer7_memory_heads(legacy, 100.0, _SetDim, hd)
        # Keep this test aligned with make_layer7_memory_heads_op(): that op
        # preserves the post-helper softmax-sharpness fix for head 5 by doubling
        # its K row after the legacy bake.
        legacy.W_k.data[5 * hd] *= 2.0

        Primitives.generate_attention_heads(
            generated, _layer7_memory_head_specs(_SetDim), hd
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        legacy_tensor = getattr(legacy, name)
        generated_tensor = getattr(generated, name)
        assert torch.equal(legacy_tensor, generated_tensor), name


def test_layer4_pc_relay_declarative_byte_identical_to_legacy_helper():
    """The declarative L4 PC relay specs reproduce legacy matrix writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=4, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=4, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer4_pc_relay(legacy, 100.0, _SetDim, hd)
        Primitives.generate_attention_heads(
            generated,
            _layer4_pc_relay_head_specs(_SetDim),
            hd,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer3_stack0_carry_declarative_byte_identical_to_legacy_helper():
    """The declarative L3 STACK0 carry spec reproduces legacy head 4 writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=3, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=3, use_flash_attention=False
    )

    with torch.no_grad():
        _set_stack0_carry_attn(legacy, 4, hd, BD=_SetDim)
        Primitives.generate_attention_head(
            generated,
            _stack0_carry_head_spec(_SetDim),
            hd,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer5_fetch_declarative_byte_identical_to_legacy_helper():
    """The declarative L5 fetch specs reproduce legacy matrix writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=5, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=5, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer5_fetch(legacy, 100.0, _SetDim, hd)
        Primitives.generate_attention_heads(
            generated,
            _fetch_head_specs(_SetDim),
            hd,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer5_fetch_pc_opcode_head_uses_exact_address_without_recency_bias():
    """PC opcode fetch must choose exact ADDR_KEY over later aliases."""

    d_model = 512
    num_heads = 8
    attn = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=5, use_flash_attention=False
    )

    make_fetch_op().declarative_bake_fn(
        SimpleNamespace(attn=attn),
        _setdim_positions(),
        100.0,
    )

    assert attn.alibi_slopes is not None
    assert torch.allclose(attn.alibi_slopes, torch.zeros(num_heads))


def test_layer8_sp_gather_declarative_byte_identical_to_legacy_helper():
    """The declarative L8 SP-gather specs reproduce legacy matrix writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=8, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=8, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer8_sp_gather(legacy, 100.0, _SetDim, hd)
        Primitives.generate_attention_heads(
            generated,
            _layer8_sp_gather_head_specs(_SetDim),
            hd,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer8_multibyte_fetch_declarative_matches_production_bake():
    """The declarative L8 multibyte-fetch spec matches the production bake."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=8, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=8, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer8_multibyte_fetch(legacy, 100.0, _SetDim, hd)
        # Production intentionally adds an AX-marker K-side exclusion after
        # the original helper so the fetch head cannot self-match staged keys.
        legacy.W_q.data[3 * hd + 35, _SetDim.H1 + 1] = 100.0
        legacy.W_q.data[3 * hd + 35, _SetDim.IS_BYTE] = 100.0
        legacy.W_q.data[3 * hd + 35, _SetDim.CONST] = -150.0
        legacy.W_k.data[3 * hd + 35, _SetDim.MARK_AX] = -50.0

        Primitives.generate_attention_head(
            generated,
            _layer8_multibyte_fetch_head_spec(_SetDim),
            hd,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer9_lev_addr_relay_declarative_byte_identical_to_legacy_helper():
    """The declarative L9 LEV SP-address relay spec reproduces legacy writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=9, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=9, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer9_lev_addr_relay(legacy, 100.0, _SetDim, hd)
        Primitives.generate_attention_head(
            generated,
            _layer9_lev_addr_relay_head_spec(_SetDim),
            hd,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer9_lev_bp_to_pc_relay_declarative_byte_identical_to_legacy_helper():
    """The declarative L9 LEV BP-to-PC relay spec reproduces legacy writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=9, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=9, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer9_lev_bp_to_pc_relay(legacy, 100.0, _SetDim, hd)
        Primitives.generate_attention_head(
            generated,
            _layer9_lev_bp_to_pc_relay_head_spec(_SetDim),
            hd,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer6_attn_spec_declarative_byte_identical_to_legacy_helper():
    """The declarative L6 attention spec reproduces legacy matrix writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=6, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=6, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer6_attn(legacy, 100.0, _SetDim, hd)
        legacy.W_k.data[5 * hd] *= 10.0

        _bake_layer6_attn_spec(generated, _SetDim, hd)
        generated.W_k.data[5 * hd] *= 10.0

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer6_first_step_fetch_relay_blocks_ax_byte_rows():
    """The byte0 FETCH relay must not overwrite L4 multibyte address staging."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads
    attn = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=6, use_flash_attention=False
    )

    with torch.no_grad():
        _bake_layer6_attn_spec(attn, _SetDim, hd)

    row = 5 * hd + 53
    assert attn.W_q[row, _SetDim.H1 + 1].item() == -6500.0
    assert attn.W_q[row, _SetDim.IS_BYTE].item() == -6500.0
    assert attn.W_q[row, _SetDim.MARK_AX].item() == 6500.0
    assert attn.W_q[row, _SetDim.H1 + 0].item() == 6500.0
    assert attn.W_k[row, _SetDim.CONST].item() == 5.0

    branch_row = 5 * hd + 52
    assert attn.W_q[branch_row, _SetDim.IS_BYTE].item() == 300.0
    assert attn.W_q[branch_row, _SetDim.H1 + 0].item() == 300.0
    assert attn.W_q[branch_row, _SetDim.BYTE_INDEX_0].item() == 300.0
    assert attn.W_q[branch_row, _SetDim.CONST].item() == 0.0
    assert attn.W_v[5 * hd + 34, _SetDim.OP_JSR].item() == 1.0
    assert attn.W_o[_SetDim.OP_JSR, 5 * hd + 34].item() == 1.0
    assert attn.W_v[5 * hd + 35 + 3, _SetDim.OPCODE_BYTE_LO + 3].item() == 1.0
    assert attn.W_v[5 * hd + 51, _SetDim.OPCODE_BYTE_HI].item() == 1.0
    assert attn.W_o[_SetDim.OPCODE_BYTE_LO + 3, 5 * hd + 35 + 3].item() == 1.0
    assert attn.W_o[_SetDim.OPCODE_BYTE_HI, 5 * hd + 51].item() == 1.0


def test_layer6_relay_heads_spec_declarative_byte_identical_to_legacy_helper():
    """The declarative L6 relay-head spec reproduces legacy matrix writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=6, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=6, use_flash_attention=False
    )

    with torch.no_grad():
        _set_layer6_relay_heads(legacy, 100.0, _SetDim, hd)
        _bake_layer6_relay_heads_spec(generated, _SetDim, hd)

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name


def test_layer6_bz_bnz_relay_spec_declarative_byte_identical_to_legacy_helper():
    """The declarative L6 BZ/BNZ relay spec reproduces legacy matrix writes."""

    d_model = 512
    num_heads = 8
    hd = d_model // num_heads

    legacy = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=6, use_flash_attention=False
    )
    generated = AutoregressiveAttention(
        d_model, num_heads=num_heads, layer_idx=6, use_flash_attention=False
    )

    with torch.no_grad():
        _set_bz_bnz_relay(legacy, 100.0, _SetDim, hd)
        Primitives.generate_attention_head(
            generated,
            _layer6_bz_bnz_relay_head_spec(_SetDim),
            hd,
        )

    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(legacy, name), getattr(generated, name)), name
