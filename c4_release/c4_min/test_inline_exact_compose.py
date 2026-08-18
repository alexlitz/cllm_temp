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
import sys

# The compose module uses bare top-level imports (``from _agent_graft_sgd_vm_rl
# import ...``) — the same way it resolves when run as a script.  Put the c4_min
# package directory on sys.path so those bare sibling imports resolve under pytest.
_C4MIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _C4MIN_DIR not in sys.path:
    sys.path.insert(0, _C4MIN_DIR)


def _current_rss_mb():
    """CURRENT (not peak) whole-process RSS in MB, or ``None`` if psutil absent.

    Used only for a DELTA check on THIS test's OWN allocation.  The absolute
    ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` peak is a whole-process
    high-water mark that is co-scheduling-contaminated: when the full CPU suite
    runs, earlier torch-import tests in the SAME process already lift the peak to
    ~2 GB, so an absolute ``ru_maxrss < 1 GB`` guard false-fails here even though
    this model-free self-check itself allocates only ~500 MB.  Whole-process
    memory is guarded by the harness watchdog; this test asserts only that its
    own footprint stays bounded.
    """
    try:
        import psutil  # noqa: PLC0415
    except Exception:
        return None
    return psutil.Process().memory_info().rss // (1024 * 1024)


def test_exact_compose_selfcheck_all_bytes():
    """The read->compose->operand path is byte-exact BY CONSTRUCTION for every byte
    operand 0..255 (MODEL-FREE: constructed digit-CAM place-value adder + frozen #910
    byte-exact ADD ALU, no learned weights, no HF host).  Expect 256/256."""
    import torch

    from _agent_graft_sgd_vm_rl import GraftedByteExactALU, construct_byte_exact
    from _agent_inline_exact_compose import ExactPlaceValueCompose

    _rss_before_mb = _current_rss_mb()

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

    # DELTA-based memory guard: assert THIS test's OWN allocation stayed bounded,
    # not the (co-scheduling-contaminated) whole-process peak.  When run alone the
    # body allocates ~500 MB; when co-scheduled after torch-import tests the
    # ru_maxrss peak is already ~2 GB, so the old absolute `ru_maxrss < 1 GB`
    # assert false-failed even though the byte-exact 256/256 self-check passed.
    _rss_after_mb = _current_rss_mb()
    if _rss_before_mb is not None and _rss_after_mb is not None:
        delta_mb = _rss_after_mb - _rss_before_mb
        assert delta_mb < 1024, (
            f"inline-compose self-check allocated {delta_mb} MB (before "
            f"{_rss_before_mb} MB -> after {_rss_after_mb} MB), exceeding the "
            "1 GB per-test CPU budget")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
