"""Byte-identity gate for the compact-gather inference optimisation.

Per the 2026-06-06 sparse-inference benchmark
(``c4_release/docs/SPARSE_INFERENCE_BENCHMARK_2026_06_06.md``), applying
``model.compact(block_size=1, compact_attn=True)`` to the compiled
``AutoregressiveVM`` yields a 1.17x CUDA forward-pass speed-up at
**max abs diff 0.0 and 100% argmax match** vs the dense baseline.

That zero-diff result is the load-bearing claim: the compact-gather
path only removes rows/columns whose dot-product contribution is
already zero, so the resulting logits are **bit-identical** — directly
verifiable with :func:`torch.equal`.

This test compiles ONE dense and ONE compact-gather runner, runs the
same 5 random-token batches through each, and asserts every logit
tensor is :func:`torch.equal`. Five inputs is small enough to keep the
test under the per-test budget while still covering different
token-id positions / per-block opcode paths.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from neural_vm.run_vm import AutoregressiveVMRunner


# Five fixed seeds covering distinct token-id sequences. The model's
# embedding lookup is the only ingest path that depends on token IDs, so
# any per-row gather diff would surface immediately at the head.
SMOKE_SEEDS = [0, 1, 7, 42, 1234]

# Modest shape — large enough to exercise every block but small enough
# the per-input forward is sub-second on the target GPU. The bench used
# (B=8, T=128); we use the same to track its measurement.
BATCH_SIZE = 4
SEQ_LEN = 64


def _has_cuda() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(scope="module")
def dense_runner() -> AutoregressiveVMRunner:
    """Build the dense baseline once per module."""
    return AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        csr_inference=False,
        compact_gather=False,
    )


@pytest.fixture(scope="module")
def compact_runner() -> AutoregressiveVMRunner:
    """Build the compact-gather runner once per module.

    Lives in its own cache slot (cache_key includes ``compact_gather``)
    so the dense fixture above is untouched.
    """
    return AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        csr_inference=False,
        compact_gather=True,
    )


def _make_tokens(model, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        0, model.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=gen,
    ).to(device)


@pytest.mark.skipif(not _has_cuda(),
                    reason="compact-gather production path targets CUDA")
@pytest.mark.parametrize("seed", SMOKE_SEEDS)
def test_compact_gather_byte_identical(seed, dense_runner, compact_runner):
    """``torch.equal`` on the logits of the dense and compact-gather paths.

    The compact-gather transform is a pure row/column gather of dense
    weights where the removed rows/columns are all-zero. The matmul
    kernel, summation order, and floating-point ops are unchanged. The
    output therefore must be bit-identical, not just close. We use
    :func:`torch.equal` (exact bitwise) rather than ``allclose`` so any
    accidental reordering or summation-order change surfaces.
    """
    dense_model = dense_runner.model
    compact_model = compact_runner.model
    assert compact_runner._compact_gather_info is not None
    assert compact_runner._compact_gather_info["converted"], (
        "compact_gather=True did not actually shrink any weight matrix; "
        f"info={compact_runner._compact_gather_info!r}"
    )

    device = next(dense_model.parameters()).device
    tokens = _make_tokens(dense_model, seed, device)

    with torch.no_grad():
        dense_logits = dense_model(tokens)
        compact_logits = compact_model(tokens)

    assert dense_logits.shape == compact_logits.shape, (
        f"shape mismatch: dense={dense_logits.shape} "
        f"compact={compact_logits.shape}"
    )
    assert torch.equal(dense_logits, compact_logits), (
        f"compact-gather logits differ from dense on seed={seed}; "
        f"max abs diff = "
        f"{(dense_logits - compact_logits).abs().max().item():.6e}, "
        f"argmax-match-frac = "
        f"{(dense_logits.argmax(-1) == compact_logits.argmax(-1)).float().mean().item():.6f}"
    )


@pytest.mark.skipif(not _has_cuda(),
                    reason="compact-gather production path targets CUDA")
def test_compact_gather_default_is_off():
    """Default behaviour must keep the dense model unchanged.

    The brief specifies opt-in only: every existing test path must keep
    seeing the dense weights so legacy byte-identity gates stay valid.
    """
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        csr_inference=False,
    )
    assert getattr(runner, "_compact_gather_info", None) is None, (
        "AutoregressiveVMRunner default constructed a compact-gather "
        "model; the flag must be opt-in only."
    )


@pytest.mark.skipif(not _has_cuda(),
                    reason="compact-gather production path targets CUDA")
def test_compact_gather_mutually_exclusive_with_csr():
    """Combining the two inference modes is unsupported and must raise."""
    with pytest.raises(ValueError):
        AutoregressiveVMRunner(
            pure_neural=True,
            trust_neural_alu=True,
            csr_inference=True,
            compact_gather=True,
        )
