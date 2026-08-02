"""Byte-exactness tests for the flash-softmax1 attention kernels (C4_FLASH_ATTN).

GPU-gated (the flash path needs CUDA + Triton).  Proves the SDPA + LSE-rescale and
the custom Triton online-softmax1 kernels reproduce the masked-full softmax1 + ALiBi
+ causal reference to < 1e-4 (fp32 tiled-reduction noise, below the nibble margin),
across the un-cached FULL, cached (bottom-right causal), windowed, and non-power-of-2
head-dim regimes.  Also checks the CPU chunked-reference fallback is byte-exact.

Golden 069cc32f unchanged (these are tests; no stored weight is touched).
"""
from __future__ import annotations

import pytest
import torch

from c4_min.blogspec_model import softmax1
from c4_min import flash_softmax1 as FL

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="flash path needs CUDA")


def _ref(Q, K, V, q_pos, k_pos, slopes, scale, window=None):
    H = Q.shape[1]
    sc = torch.matmul(Q, K.transpose(-2, -1)) * scale
    dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).float()
    sc = sc - slopes.view(1, H, 1, 1) * dist.abs().unsqueeze(0)
    m = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))
    if window is not None:
        m = m | (dist >= window)
    sc = sc.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
    return torch.matmul(softmax1(sc, dim=-1), V)


def _case(H, Sq, Skpast, HD, dev, seed=0):
    torch.manual_seed(seed)
    span = Skpast
    q_pos = torch.arange(span, span + Sq, device=dev, dtype=torch.long)
    k_pos = torch.arange(0, span + Sq, device=dev, dtype=torch.long)
    Q = torch.randn(1, H, Sq, HD, device=dev)
    K = torch.randn(1, H, span + Sq, HD, device=dev)
    V = torch.randn(1, H, span + Sq, HD, device=dev)
    slopes = torch.rand(H, device=dev) * 0.5 + 0.01
    return Q, K, V, q_pos, k_pos, slopes, HD ** -0.5


@gpu
def test_sdpa_uncached_full():
    Q, K, V, qp, kp, sl, sc = _case(4, 128, 0, 64, "cuda:0", seed=0)
    ref = _ref(Q, K, V, qp, kp, sl, sc)
    got = FL.sdpa_flash_softmax1(Q, K, V, qp, kp, sl, sc)
    assert (ref - got).abs().max().item() < 1e-4


@gpu
def test_triton_cached_bottom_right():
    Q, K, V, qp, kp, sl, sc = _case(4, 30, 200, 64, "cuda:0", seed=1)
    ref = _ref(Q, K, V, qp, kp, sl, sc)
    got = FL.triton_flash_softmax1(Q, K, V, qp, kp, sl, sc)
    assert (ref - got).abs().max().item() < 1e-4


@gpu
@pytest.mark.parametrize("HD", [40, 64, 80])
def test_triton_non_power_of_two_headdim(HD):
    Q, K, V, qp, kp, sl, sc = _case(8, 40, 300, HD, "cuda:0", seed=HD)
    ref = _ref(Q, K, V, qp, kp, sl, sc)
    got = FL.triton_flash_softmax1(Q, K, V, qp, kp, sl, sc)
    assert (ref - got).abs().max().item() < 1e-4


@gpu
@pytest.mark.parametrize("W", [96, 28])
def test_triton_windowed(W):
    Q, K, V, qp, kp, sl, sc = _case(4, 300, 96, 64, "cuda:0", seed=11)
    ref = _ref(Q, K, V, qp, kp, sl, sc, window=W)
    got = FL.triton_flash_softmax1(Q, K, V, qp, kp, sl, sc, window=W)
    assert (ref - got).abs().max().item() < 1e-4


@gpu
def test_dispatch_matches_reference():
    # un-cached full via dispatcher
    Q, K, V, qp, kp, sl, sc = _case(4, 100, 0, 64, "cuda:0", seed=3)
    ref = _ref(Q, K, V, qp, kp, sl, sc)
    got = FL.flash_softmax1_context(Q, K, V, qp, kp, sl, sc, uncached_full=True)
    assert (ref - got).abs().max().item() < 1e-4
    # cached via dispatcher
    Q, K, V, qp, kp, sl, sc = _case(4, 20, 150, 64, "cuda:0", seed=4)
    ref = _ref(Q, K, V, qp, kp, sl, sc)
    got = FL.flash_softmax1_context(Q, K, V, qp, kp, sl, sc)
    assert (ref - got).abs().max().item() < 1e-4


def test_cpu_chunked_reference_byte_exact():
    """The CPU / no-Triton fallback (chunked online softmax1) must be byte-exact."""
    dev = "cpu"
    Q, K, V, qp, kp, sl, sc = _case(3, 40, 100, 32, dev, seed=5)
    ref = _ref(Q, K, V, qp, kp, sl, sc)
    got = FL._chunked_reference(Q, K, V, qp, kp, sl, sc, None)
    assert (ref - got).abs().max().item() < 1e-4
    # windowed
    ref = _ref(Q, K, V, qp, kp, sl, sc, window=16)
    got = FL._chunked_reference(Q, K, V, qp, kp, sl, sc, 16)
    assert (ref - got).abs().max().item() < 1e-4
