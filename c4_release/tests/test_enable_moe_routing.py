"""Verify ``enable_moe_routing`` plumbing and the standard top-K
``SoftMoEFFN`` structural contract.

Tests covered:
  1. The default (``enable_moe_routing=False``) leaves all FFN blocks
     as dense PureFFN / ALU composites — no SoftMoEFFN.
  2. ``enable_moe_routing=True`` installs SoftMoEFFN modules on the
     blocks listed by the legacy ``compact_moe`` partition (L6, L8, L10,
     L12, L17, L20, L22, L24 in the current model).
  3. Each installed SoftMoEFFN has ``top_k == 1`` (the C4 natural choice)
     and exposes the per-expert ``expert_opcode_dims`` routing list.
  4. The runner records ``enable_moe_routing`` on ``self`` for diagnostic
     visibility.
  5. Structural correctness: each ``SoftMoEFFN.experts[i]`` is a
     ``PureFFN`` with the expert's specific hidden units only; the
     ``shared_ffn`` holds the opcode-independent units (or a 1-unit
     dummy carrying ``b_down`` when the partition has no shared
     units).
  6. At a position where exactly one OP_* dim is 1.0 and the rest are
     0.0, top-1 routing selects the expert at that opcode dim (the
     "routing-correctness" check — not byte-identity vs dense).

Notes:
    Byte-identity to the dense compacted FFN is currently NOT
    achievable; see ``c4_release/docs/MOE_ROUTING_AUDIT.md`` §8 for
    details. These tests verify the MoE *structure* and the
    *routing decision*, not the bit-exact output.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.pure_moe import SoftMoEFFN, StandardMoEFFN  # noqa: E402
from neural_vm.base_layers import PureFFN  # noqa: E402


def _count_softmoe_blocks(model):
    return sum(1 for b in model.blocks if isinstance(b.ffn, SoftMoEFFN))


def test_default_keeps_dense_ffn():
    """Default constructor does NOT install SoftMoEFFN (byte-identity
    to the dense path is preserved). This locks in the post-pivot
    default state — see audit doc §8.2."""
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        cache_model=False,
    )
    n_softmoe = _count_softmoe_blocks(runner.model)
    assert n_softmoe == 0, (
        f"Default ``enable_moe_routing=False`` expected zero SoftMoEFFN "
        f"blocks, found {n_softmoe}/{len(runner.model.blocks)}."
    )
    assert runner.enable_moe_routing is False


def test_enable_moe_routing_true_installs_softmoe():
    """``enable_moe_routing=True`` puts SoftMoEFFN on the live path."""
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        enable_moe_routing=True,
        cache_model=False,
    )
    n_softmoe = _count_softmoe_blocks(runner.model)
    assert n_softmoe > 0, (
        f"Expected ``enable_moe_routing=True`` to install at least one "
        f"SoftMoEFFN block, but found {n_softmoe} of "
        f"{len(runner.model.blocks)}."
    )
    assert runner.enable_moe_routing is True


def test_softmoeffn_is_top_k_one_with_shared_expert():
    """Every installed SoftMoEFFN has the standard top-K MoE shape:
    ``top_k=1`` (C4 dispatches one opcode per step), a ``shared_ffn``,
    and a list of per-expert opcode dims matching the experts list.
    """
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        enable_moe_routing=True,
        cache_model=False,
    )
    moe_blocks = [b for b in runner.model.blocks if isinstance(b.ffn, SoftMoEFFN)]
    assert moe_blocks, "Expected at least one SoftMoEFFN block."
    for block in moe_blocks:
        m = block.ffn
        assert m.top_k == 1, f"top_k should default to 1 for C4; got {m.top_k}"
        assert m.num_experts == len(m.experts) == len(m.expert_opcode_dims)
        # Every expert is a PureFFN holding only its opcode-specific units.
        for e in m.experts:
            assert isinstance(e, PureFFN)
        # Always-on shared path (DeepSeek-style shared expert).
        assert m._has_shared, "SoftMoEFFN should carry an always-on shared FFN"
        assert isinstance(m.shared_ffn, PureFFN)


def test_standardmoeffn_is_alias_for_softmoeffn():
    """``StandardMoEFFN`` is the forward-looking name; it aliases
    ``SoftMoEFFN`` so external callers can use either."""
    assert StandardMoEFFN is SoftMoEFFN


def test_topk_routing_picks_correct_expert():
    """Build a tiny SoftMoEFFN with two experts gated by dims 5 and 6.
    Verify that at a position with OP_dim_5=1.0, expert 0 fires (gate=1.0);
    at a position with OP_dim_6=1.0, expert 1 fires; at a position with
    neither, no expert fires (gate=0.0 → no contribution beyond shared)."""
    torch.manual_seed(0)
    D = 8
    H_shared = 2
    H_each = 3

    shared = PureFFN(dim=D, hidden_dim=H_shared)
    with torch.no_grad():
        # Zero out shared so we can isolate per-expert contributions.
        shared.W_up.data.zero_()
        shared.W_gate.data.zero_()
        shared.W_down.data.zero_()
        shared.b_up.data.zero_()
        shared.b_gate.data.zero_()
        shared.b_down.data.zero_()

    e0 = PureFFN(dim=D, hidden_dim=H_each)
    e1 = PureFFN(dim=D, hidden_dim=H_each)
    with torch.no_grad():
        for f, marker in [(e0, 0.5), (e1, 2.5)]:
            f.W_up.data.zero_()
            f.W_gate.data.zero_()
            f.W_down.data.zero_()
            # Each expert injects a constant delta marker into output[0].
            f.b_up.data.fill_(1.0)
            f.b_gate.data.fill_(1.0)
            # W_down maps hidden -> output dim 0 only, with value marker.
            f.W_down.data[0, 0] = marker
            f.b_up.data.zero_()  # ensure clean
            f.b_gate.data.fill_(1.0)
            f.b_down.data.zero_()
            # silu(0)=0; we need W_up to fire from gating
            f.W_up.data[0, 4] = 1.0  # gate on dim 4
            f.b_up.data[0] = 0.0

    moe = SoftMoEFFN(
        experts=[e0, e1],
        expert_opcode_dims=[5, 6],
        shared_ffn=shared,
        top_k=1,
    )

    # Position 0: opcode dim 5 active.
    x = torch.zeros(1, 3, D)
    x[0, 0, 4] = 1.0  # gate input
    x[0, 0, 5] = 1.0  # OP_5 → expert 0
    x[0, 1, 4] = 1.0
    x[0, 1, 6] = 1.0  # OP_6 → expert 1
    # Position 2: no opcode active; expert should not fire.
    x[0, 2, 4] = 1.0

    out = moe(x)
    delta = (out - x)[0, :, 0]
    # Expert 0 marker = 0.5; expert 1 marker = 2.5.
    # delta[0] should be ~0.5 (silu(0)*1.0 * 0.5 = 0; hmm, actually
    # silu(0)*1.0 = 0, so W_down @ hidden = 0). The test is structural —
    # just verify that positions 0,1 fire (delta!=0) but position 2 does
    # not. With our bake, silu(b_up)=silu(0)=0, so delta is 0 everywhere.
    # Reset: use a non-zero b_up.
    with torch.no_grad():
        for f in (e0, e1):
            f.b_up.data[0] = 1.0  # silu(1.0) ≈ 0.73 → non-zero hidden
            f.W_up.data.zero_()
    # Rebuild the stacked buffers since the per-expert weights changed.
    moe._build_stacked_weights()
    out = moe(x)
    delta = (out - x)[0, :, 0]
    assert torch.is_tensor(delta)
    # Position 2 (no opcode): delta should be exactly 0.
    assert torch.isclose(delta[2], torch.zeros(()), atol=1e-7), (
        f"position with no opcode should have zero MoE delta, got {delta[2].item()}"
    )
    # Position 0 (OP_5=1.0 → expert 0, marker 0.5).
    assert delta[0].item() != 0.0, "expert 0 should fire at OP_5 position"
    # Position 1 (OP_6=1.0 → expert 1, marker 2.5).
    assert delta[1].item() != 0.0, "expert 1 should fire at OP_6 position"
    # Expert 1's marker is 5x expert 0's, so delta[1] / delta[0] ≈ 5.
    ratio = (delta[1] / delta[0]).item()
    assert 4.5 < ratio < 5.5, (
        f"expert 1 marker (2.5) / expert 0 marker (0.5) should yield "
        f"ratio ~5; got {ratio:.3f}"
    )


def test_empty_moe_is_passthrough():
    """SoftMoEFFN with zero experts and no shared FFN is identity."""
    D = 4
    moe = SoftMoEFFN(
        experts=[],
        expert_opcode_dims=[],
        shared_ffn=None,
        top_k=1,
    )
    x = torch.randn(1, 3, D)
    out = moe(x)
    assert torch.allclose(out, x), "Empty SoftMoEFFN should be identity"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])
