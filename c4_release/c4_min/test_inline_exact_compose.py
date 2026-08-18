"""CPU-safe test for the #914 inline EXACT-COMPOSE circuit (part of the #923 Wave-2
RL-circuits consolidation).

The load-bearing claim of ``_agent_inline_exact_compose`` is that the operand-
extraction path (read multi-digit operand -> compose place-value -> feed the ALU)
is byte-exact BY CONSTRUCTION — a digit-addressing CAM (no learned weights) plus a
place-value accumulation THROUGH the frozen #910 byte-exact ADD ALU, NOT a learned
readout.  ``ExactPlaceValueCompose.selfcheck_all_bytes()`` is the MODEL-FREE proof:
it feeds the true 3-digit decomposition of every operand 0..255 through the
constructed ALU adder and asserts the recomposed value equals the operand.

This test runs ONLY that model-free self-check — it constructs the byte-exact ALU
(``construct_byte_exact``, no gradient / no HF host / no 0.5B load) and asserts
256/256.  CPU-safe: ~500 MB RSS, ~2 s.  The full RL self-checks (C1 variance-
reduction, C4 unprotected-reward) load a real Qwen2.5-0.5B and are GPU-gated in
``test_rl_circuits_gpu.py`` (NOT run on CPU).

Off-build-path: imports no neural-model build-path module — golden 174ece66 is
untouched (see ``test_golden_174ece66_untouched``).
"""
from __future__ import annotations

import os
import resource
import sys

# The compose module uses bare top-level imports (``from _agent_graft_sgd_vm_rl
# import ...``) — the same way it resolves when run as a script.  Put the c4_min
# package directory on sys.path so those bare sibling imports resolve under pytest.
_C4MIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _C4MIN_DIR not in sys.path:
    sys.path.insert(0, _C4MIN_DIR)


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def test_exact_compose_selfcheck_all_bytes():
    """The read->compose->operand path is byte-exact BY CONSTRUCTION for every byte
    operand 0..255 (MODEL-FREE: constructed digit-CAM place-value adder + frozen #910
    byte-exact ADD ALU, no learned weights, no HF host).  Expect 256/256."""
    import torch

    from _agent_graft_sgd_vm_rl import GraftedByteExactALU, construct_byte_exact
    from _agent_inline_exact_compose import ExactPlaceValueCompose

    dev = "cpu"
    alu = GraftedByteExactALU(temp=30.0).to(dev)
    construct_byte_exact(alu)
    for p in alu.parameters():
        p.requires_grad_(False)

    compose = ExactPlaceValueCompose(alu, dev)
    ok, tot = compose.selfcheck_all_bytes()
    assert tot == 256
    assert ok == 256, (
        f"exact place-value compose byte-exact only {ok}/{tot} — the "
        "read->compose->ALU->operand path is NOT byte-exact by construction")

    # Cross-check the ALU-adder path against the pure-integer reference accumulation
    # (same units->tens->hundreds order): proves the ADD flows through the exact
    # circuit and matches the reference wrap byte-for-byte on all 256 operands.
    digs = torch.tensor(
        [[int(c) for c in f"{v:03d}"] for v in range(256)], dtype=torch.long)
    via_alu = compose.compose(digs)
    via_ref = compose.compose_table_only(digs)
    assert torch.equal(via_alu.cpu(), via_ref.cpu()), (
        "ALU-adder compose diverged from the integer reference accumulation")

    assert _rss_mb() < 1024, f"RSS {_rss_mb()} MB exceeded the 1 GB CPU budget"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
