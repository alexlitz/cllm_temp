"""Verify that ``enable_moe_routing`` flag actually installs SoftMoEFFN.

Tests the wire-moe-routing-default plumbing:
  1. With ``enable_moe_routing=True`` (default), at least some block.ffn
     modules are SoftMoEFFN instances.
  2. With ``enable_moe_routing=False``, no block.ffn is SoftMoEFFN — all
     remain PureFFN or ALU composites (the previous default behavior).

This is the test that proves the MoE path is actually wired into the
production runner — not just a code-graveyard module.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.pure_moe import SoftMoEFFN  # noqa: E402
from neural_vm.base_layers import PureFFN  # noqa: E402


def _count_softmoe_blocks(model):
    return sum(1 for b in model.blocks if isinstance(b.ffn, SoftMoEFFN))


def test_enable_moe_routing_default_installs_softmoe():
    """Default ``enable_moe_routing=True`` puts SoftMoEFFN on the live path."""
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        cache_model=False,
    )
    n_softmoe = _count_softmoe_blocks(runner.model)
    assert n_softmoe > 0, (
        f"Expected ``enable_moe_routing=True`` (the default) to install at "
        f"least one SoftMoEFFN block, but found {n_softmoe} of "
        f"{len(runner.model.blocks)}."
    )


def test_enable_moe_routing_false_keeps_dense_ffn():
    """``enable_moe_routing=False`` leaves all FFN blocks dense (no SoftMoEFFN)."""
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        enable_moe_routing=False,
        cache_model=False,
    )
    n_softmoe = _count_softmoe_blocks(runner.model)
    assert n_softmoe == 0, (
        f"Expected ``enable_moe_routing=False`` to leave all FFN blocks "
        f"as dense PureFFN, but {n_softmoe}/{len(runner.model.blocks)} "
        f"are SoftMoEFFN instances."
    )


def test_enable_moe_routing_flag_is_recorded_on_runner():
    """The runner exposes the flag for diagnostic visibility."""
    r_on = AutoregressiveVMRunner(
        pure_neural=True, trust_neural_alu=True,
        enable_moe_routing=True, cache_model=False,
    )
    r_off = AutoregressiveVMRunner(
        pure_neural=True, trust_neural_alu=True,
        enable_moe_routing=False, cache_model=False,
    )
    assert r_on.enable_moe_routing is True
    assert r_off.enable_moe_routing is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])
