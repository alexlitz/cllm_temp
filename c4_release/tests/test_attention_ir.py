import pytest
import torch

from c4_release.neural_vm.base_layers import PureAttention
from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    SymbolicResidualState,
)
from c4_release.neural_vm.unified_compiler.primitives import (
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
    Primitives,
)
from c4_release.neural_vm.unified_compiler.ops.l7_ops import (
    make_layer7_memory_heads_op,
)


def _probe_attention_head_spec() -> DeclarativeAttentionHeadSpec:
    return DeclarativeAttentionHeadSpec(
        head_idx=1,
        q=(AP(0, 2, 3.5), AP(2, 7, -1.25)),
        k=(AP(0, 3, 4.0),),
        v=(AP(1, 8, 1.0), AP(2, 9, -2.0)),
        o=(AO(10, 1, 1.0), AO(11, 2, -0.5)),
    )


def test_attention_spec_can_be_carried_in_compiler_ir():
    spec = _probe_attention_head_spec()
    ir = CompilerIR()

    ir.layer(0).attention.append(
        spec,
        name="probe_head",
        metadata={"purpose": "compiler-ir-attention-prototype"},
    )

    head = ir.layer(0).attention.heads[0]
    assert head.spec is spec
    assert head.name == "probe_head"
    assert head.metadata["purpose"] == "compiler-ir-attention-prototype"


def test_attention_ir_debug_report_exposes_qkvo_writes():
    spec = _probe_attention_head_spec()
    ir = CompilerIR()
    ir.layer(2).attention.append(spec, name="probe_head")

    report = ir.attention_debug_report(HD=8, layer_idx=2)

    assert report.matrix_write_counts == {
        "W_q": 2,
        "W_k": 1,
        "W_v": 2,
        "W_o": 2,
    }
    assert report.heads[0].name == "probe_head"
    assert report.heads[0].head_idx == 1
    assert report.heads[0].q[0].matrix == "W_q"
    assert report.heads[0].q[0].row == 8
    assert report.heads[0].q[0].col == 2
    assert report.heads[0].q[0].slot == 0
    assert report.heads[0].q[0].source_dim == 2
    assert report.heads[0].q[0].output_dim is None
    assert report.heads[0].o[0].matrix == "W_o"
    assert report.heads[0].o[0].row == 10
    assert report.heads[0].o[0].col == 9
    assert report.heads[0].o[0].source_dim is None
    assert report.heads[0].o[0].output_dim == 10


def test_attention_ir_lowering_matches_direct_primitive_matrix_writes():
    spec = _probe_attention_head_spec()
    d_model = 32
    num_heads = 4
    hd = d_model // num_heads
    direct = PureAttention(d_model, num_heads=num_heads)
    lowered = PureAttention(d_model, num_heads=num_heads)
    ir = CompilerIR()
    ir.layer(0).attention.append(spec)

    with torch.no_grad():
        Primitives.generate_attention_head(direct, spec, hd)
        lowered_count = ir.lower_attention(lowered, hd)

    assert lowered_count == 1
    for name in ("W_q", "W_k", "W_v", "W_o"):
        assert torch.equal(getattr(direct, name), getattr(lowered, name)), name


def test_symbolic_attention_hardmax_copies_best_causal_source():
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, 0, 4.0),),
        k=(AP(0, 1, 4.0),),
        v=(AP(1, 2, 1.0),),
        o=(AO(3, 1, 1.0),),
    )
    ir = CompilerIR()
    ir.layer(0).attention.append(spec, name="copy_prior_value")
    state = SymbolicResidualState([
        {1: 1.0, 2: 7.0},
        {1: 1.0, 2: 11.0},
        {0: 1.0},
    ])

    out, choices = ir.symbolic_attention_positions(
        state,
        HD=16,
        layer_idx=0,
        alibi_slopes={0: 1.0},
    )

    assert out.get(2, 3) == 11.0
    assert choices[-1].query_pos == 2
    assert choices[-1].key_pos == 1
    assert not choices[-1].sink_selected


def test_symbolic_attention_hardmax_uses_softmax1_sink_when_score_nonpositive():
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, 0, 1.0),),
        k=(AP(0, 1, -1.0),),
        v=(AP(1, 2, 1.0),),
        o=(AO(3, 1, 1.0),),
    )
    ir = CompilerIR()
    ir.layer(0).attention.append(spec)
    state = SymbolicResidualState([{1: 1.0, 2: 5.0}, {0: 1.0}])

    out, choices = ir.symbolic_attention_positions(state, HD=1)

    assert out.get(1, 3) == 0.0
    assert choices[-1].sink_selected


def test_symbolic_attention_softmax1_blends_sources_and_sink():
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AP(0, 0, 1.0),),
        k=(AP(0, 1, 1.0),),
        v=(AP(1, 2, 1.0),),
        o=(AO(3, 1, 1.0),),
    )
    ir = CompilerIR()
    ir.layer(0).attention.append(spec)
    state = SymbolicResidualState([{1: 1.0, 2: 3.0}, {0: 1.0}])

    out, _ = ir.symbolic_attention_positions(state, HD=1, mode="softmax1")

    expected = torch.exp(torch.tensor(1.0)).item()
    # Causal attention includes the current query row. That row has score 0
    # and V=0 here, so it participates in the denominator alongside the
    # softmax1 sink but contributes no value.
    expected = expected / (2.0 + expected) * 3.0
    assert out.get(1, 3) == pytest.approx(expected)


def _l7_memory_head_test_dims():
    dims = {}
    cursor = 0

    def alloc(name, size=1):
        nonlocal cursor
        dims[name] = cursor
        cursor += size

    for name, size in (
        ("CONST", 1),
        ("MARK_AX", 1),
        ("MARK_MEM", 1),
        ("MARK_STACK0", 1),
        ("MARK_SP", 1),
        ("H1", 8),
        ("H3", 8),
        ("H4", 8),
        ("BYTE_INDEX_0", 1),
        ("BYTE_INDEX_1", 1),
        ("BYTE_INDEX_2", 1),
        ("CLEAN_EMBED_LO", 16),
        ("CLEAN_EMBED_HI", 16),
        ("ADDR_B0_LO", 16),
        ("ADDR_B0_HI", 16),
        ("ADDR_B1_LO", 16),
        ("ADDR_B1_HI", 16),
        ("ADDR_B2_LO", 16),
        ("ADDR_B2_HI", 16),
        ("OP_LI", 1),
        ("OP_LC", 1),
        ("OP_LEA", 1),
        ("OP_AND", 1),
        ("OP_OR", 1),
        ("OP_XOR", 1),
        ("OP_JSR", 1),
        ("OP_SHR", 1),
        ("OP_SI", 1),
        ("OP_SC", 1),
        ("OP_ADD", 1),
        ("OP_SUB", 1),
        ("OP_PSH", 1),
        ("OP_ENT", 1),
        ("OP_LI_RELAY", 1),
        ("OP_LC_RELAY", 1),
        ("PSH_AT_SP", 1),
        ("MEM_STORE", 1),
        ("MEM_ADDR_SRC", 1),
        ("TEMP", 16),
        ("CMP", 8),
    ):
        alloc(name, size)
    return dims, cursor


@pytest.mark.parametrize(
    "op_name,relay_key",
    (("OP_ADD", "TEMP+8"), ("OP_SUB", "TEMP+9")),
)
def test_l7_add_sub_relay_declaration_matches_lowered_attention(op_name, relay_key):
    dim_positions, d_model = _l7_memory_head_test_dims()
    # Keep a real multi-head shape so head 5 slot numbers map exactly as they do
    # in production.
    num_heads = 8
    head_dim = 64
    d_model = max(d_model, num_heads * head_dim)
    op = make_layer7_memory_heads_op()
    ir = op.compiler_ir_factory(dim_positions, head_dim)

    attn = PureAttention(d_model, num_heads=num_heads)
    with torch.no_grad():
        ir.lower_attention(attn, head_dim)

    marker = {
        dim_positions["MARK_AX"]: 1.0,
        dim_positions[op_name]: 5.0,
    }
    ax_byte = {
        dim_positions["H1"] + 1: 1.0,
    }
    symbolic_in = SymbolicResidualState([marker, ax_byte])
    symbolic_out, _choices = ir.symbolic_attention_positions(
        symbolic_in,
        head_dim,
        layer_idx=0,
        mode="hardmax",
    )

    x = torch.zeros(1, 2, d_model)
    for dim, value in marker.items():
        x[0, 0, dim] = value
    for dim, value in ax_byte.items():
        x[0, 1, dim] = value
    y = attn(x)

    relay_dim = relay_key.split("+")
    expected_dim = dim_positions[relay_dim[0]] + int(relay_dim[1])
    assert symbolic_out.get(1, expected_dim) == pytest.approx(1.0)
    assert float(y[0, 1, expected_dim].item()) == pytest.approx(1.0, abs=1e-4)
