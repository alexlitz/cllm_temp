"""Tests for ``TokenEmbeddingRule`` and friends (Phase 7.D.1)."""

import pytest
import torch
import torch.nn as nn

from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    DimRef,
    TokenEmbeddingRule,
    WriteTerm,
    compare_symbolic_to_lowered_embedding,
)


def _synthetic_model(vocab_size: int, d_model: int):
    """Replicate the minimal duck shape ``lower_token_embeddings`` writes into.

    Tests use this rather than the production ``NeuralVMEmbedding`` /
    ``AutoregressiveVM`` stack to keep the IR-level tests self-contained.
    """

    class _Embed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            with torch.no_grad():
                self.embed.weight.zero_()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = _Embed()
            self.head = nn.Linear(d_model, vocab_size)
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.bias.zero_()

    return _Model()


def test_embed_rule_lowers_into_embed_weight_table():
    dim_positions = {"CONST": 0, "EMBED_LO": 4, "EMBED_HI": 20}
    ir = CompilerIR()
    ir.embeddings.append(
        TokenEmbeddingRule.embed_write(
            token_ids=[42],
            writes=(("CONST", 1.0), ("EMBED_LO+2", 1.0), ("EMBED_HI+5", 1.0)),
            name="byte_42",
        )
    )

    model = _synthetic_model(vocab_size=64, d_model=64)
    applied = ir.lower_token_embeddings(model, dim_positions)

    assert applied == 1
    w = model.embed.embed.weight.detach()
    assert float(w[42, 0]) == 1.0       # CONST
    assert float(w[42, 4 + 2]) == 1.0   # EMBED_LO+2
    assert float(w[42, 20 + 5]) == 1.0  # EMBED_HI+5
    # No bleed into other rows.
    assert float(w[0, 0]) == 0.0
    assert float(w[41, 0]) == 0.0


def test_head_weight_and_bias_rules_lower_into_model_head():
    dim_positions = {"OUTPUT_LO": 0, "OUTPUT_HI": 16, "NEXT_PC": 40}
    ir = CompilerIR()
    ir.embeddings.append(
        TokenEmbeddingRule.head_weight_write(
            token_ids=[5, 6, 7],
            writes=(("OUTPUT_LO+1", 5.0), ("NEXT_PC", -80.0)),
            name="head_byte_block",
        )
    )
    ir.embeddings.append(
        TokenEmbeddingRule.head_bias_write(
            token_ids=[5, 6, 7],
            value=-5.0,
            name="head_byte_bias",
        )
    )

    model = _synthetic_model(vocab_size=64, d_model=64)
    applied = ir.lower_token_embeddings(model, dim_positions)

    assert applied == 6  # 3 tokens * 2 rules
    hw = model.head.weight.detach()
    hb = model.head.bias.detach()
    for tok in (5, 6, 7):
        assert float(hw[tok, 1]) == 5.0      # OUTPUT_LO+1
        assert float(hw[tok, 40]) == -80.0   # NEXT_PC
        assert float(hb[tok]) == -5.0
    # Untouched rows stay zero.
    assert float(hw[8, 1]) == 0.0
    assert float(hb[8]) == 0.0


def test_multiple_rules_accumulate_on_same_cell():
    """Mirror of FFN lowering's ``+=`` semantics: two rules → summed write."""
    dim_positions = {"X": 0}
    ir = CompilerIR()
    ir.embeddings.append(TokenEmbeddingRule.embed_write(
        token_ids=[3], writes=(("X", 0.25),), name="rule_a",
    ))
    ir.embeddings.append(TokenEmbeddingRule.embed_write(
        token_ids=[3], writes=(("X", 0.5),), name="rule_b",
    ))
    ir.embeddings.append(TokenEmbeddingRule.head_bias_write(
        token_ids=[3], value=-1.0, name="bias_a",
    ))
    ir.embeddings.append(TokenEmbeddingRule.head_bias_write(
        token_ids=[3], value=-2.5, name="bias_b",
    ))

    model = _synthetic_model(vocab_size=8, d_model=4)
    ir.lower_token_embeddings(model, dim_positions)

    assert float(model.embed.embed.weight[3, 0]) == 0.75
    assert float(model.head.bias[3]) == -3.5


def test_head_bias_rule_rejects_non_bias_writes_at_construction():
    with pytest.raises(ValueError):
        # head_bias rules must use the empty DimRef; non-empty must reject.
        TokenEmbeddingRule(
            target="head_bias",
            token_ids=(3,),
            writes=(WriteTerm(DimRef("OUTPUT_LO", 0), 1.0),),
        )

    with pytest.raises(ValueError):
        TokenEmbeddingRule(target="bogus", token_ids=(0,), writes=())


def test_compare_symbolic_to_lowered_embedding_flags_undeclared_dim_and_drift():
    """The comparison tool must catch both declaration and lowering drift."""
    dim_positions = {"CONST": 0, "OUTPUT_LO": 4}

    # 1) Declaration semantics: ``EMBED_LO`` is not in dim_positions.
    bad_decl_rule = TokenEmbeddingRule.embed_write(
        token_ids=[1], writes=(("EMBED_LO+3", 1.0),), name="bad_decl",
    )
    report = compare_symbolic_to_lowered_embedding(
        bad_decl_rule, dim_positions, vocab_size=8, d_model=8,
    )
    assert not report.ok
    assert report.primary_failure_kind == "declaration_semantics"

    # 2) Token-id out of vocab range.
    oob_rule = TokenEmbeddingRule.head_bias_write(
        token_ids=[999], value=-1.0, name="oob_token",
    )
    report = compare_symbolic_to_lowered_embedding(
        oob_rule, dim_positions, vocab_size=8, d_model=8,
    )
    assert not report.ok
    assert report.primary_failure_kind == "declaration_semantics"

    # 3) Lowering drift: pre-populate the model with the wrong cell value,
    #    pass ``lower=False`` so the helper only validates rather than baking.
    ir = CompilerIR()
    ir.embeddings.append(TokenEmbeddingRule.embed_write(
        token_ids=[2], writes=(("CONST", 1.0),), name="want_one",
    ))
    model = _synthetic_model(vocab_size=8, d_model=8)
    # Wrong! Pretend a buggy lowering wrote -1.0 here.
    with torch.no_grad():
        model.embed.embed.weight[2, 0] = -1.0
    report = compare_symbolic_to_lowered_embedding(
        ir, dim_positions, model=model, lower=False,
    )
    assert not report.ok
    assert report.primary_failure_kind == "lowering"

    # 4) Happy path with synthetic model + actual lowering.
    ir = CompilerIR()
    ir.embeddings.append(TokenEmbeddingRule.embed_write(
        token_ids=[2, 3], writes=(("CONST", 1.0), ("OUTPUT_LO+2", 0.5)),
    ))
    ir.embeddings.append(TokenEmbeddingRule.head_bias_write(
        token_ids=[2, 3], value=-5.0,
    ))
    report = compare_symbolic_to_lowered_embedding(
        ir, dim_positions, vocab_size=8, d_model=8,
    )
    assert report.ok
    assert report.issues == []
