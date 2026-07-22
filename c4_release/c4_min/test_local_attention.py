"""Unit tests for local (sliding-window) attention on the non-memory heads.

Memory-safe: uses a TINY synthetic ``SparseAttn`` (dim 24, 4 heads) — no full model
build, no GPU — so it runs in the default conftest RSS ceiling.  Proves
``windowed_forward`` is byte-identical (L∞=0) to the global forward whenever the
window covers every head's true span, for the cached (q_positions given) path the
fast path actually uses.
"""
from __future__ import annotations

import torch

from c4_min import sparse_forward as _SF
from c4_min import local_attention as LA
from c4_min.blogspec_model import Attn


def _mk_attn(dim=24, nh=4, seed=0):
    torch.manual_seed(seed)
    a = Attn(dim, nh, max_seq_len=512, positional="alibi")
    with torch.no_grad():
        for w in (a.W_q, a.W_k, a.W_v, a.W_o):
            w.data.copy_(torch.randn(dim, dim) * 0.1)
    return _SF.SparseAttn(a, density_thresh=1.0, min_numel=10 ** 9, log={},
                          compute_mode="dense_kernel"), nh, dim


def _cached_ctx(sa, S, dim, seed=1):
    torch.manual_seed(seed)
    x_ctx = torch.randn(1, S, dim) * 0.1
    _, kv = LA._global_forward(sa, x_ctx, None, torch.arange(S), True)
    return kv


def test_windowed_equals_global_cached_all_local():
    """Cached path (q_positions given — the fast-path branch), all heads LOCAL, window
    covering the whole span -> byte-identical (this is the fast path's exact usage)."""
    sa, nh, dim = _mk_attn()
    S = 40
    K, Vv, kpos = _cached_ctx(sa, S, dim)
    xq = torch.randn(1, 3, dim) * 0.1
    qpos = torch.arange(S, S + 3)
    g = LA._global_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    sa._local_window = 1000
    sa._global_head_mask = torch.zeros(nh, dtype=torch.bool)
    l = LA.windowed_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    assert float((g - l).abs().max()) == 0.0


def test_windowed_equals_global_all_global():
    """All heads flagged GLOBAL -> the windowed forward is the global forward."""
    sa, nh, dim = _mk_attn(seed=2)
    S = 30
    K, Vv, kpos = _cached_ctx(sa, S, dim, seed=3)
    xq = torch.randn(1, 4, dim) * 0.1
    qpos = torch.arange(S, S + 4)
    g = LA._global_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    sa._local_window = 8
    sa._global_head_mask = torch.ones(nh, dtype=torch.bool)
    l = LA.windowed_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    assert float((g - l).abs().max()) == 0.0


def test_windowed_equals_global_mixed():
    """Mixed global/local heads in the SAME block, window >= span.  Arithmetically
    identical; the windowed forward computes the global and local heads in SEPARATE
    matmuls (different fp32 reduction order) so the L∞ is fp32 epsilon (~1e-9), NOT
    exact zero.  NOTE: this MIXED-in-one-block case does NOT occur in the real model
    — block 0 is all-LOCAL, the mem-cam/stack-pop blocks are all-GLOBAL (their local
    heads are zero-value), so the production path is exact (verified L∞=0.0 e2e)."""
    sa, nh, dim = _mk_attn(seed=4)
    S = 35
    K, Vv, kpos = _cached_ctx(sa, S, dim, seed=5)
    xq = torch.randn(1, 5, dim) * 0.1
    qpos = torch.arange(S, S + 5)
    g = LA._global_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    sa._local_window = 500
    sa._global_head_mask = torch.tensor([True, False, True, False])
    l = LA.windowed_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    assert float((g - l).abs().max()) < 1e-6   # fp32 split-matmul epsilon


def test_window_actually_cuts_keys():
    """A window SMALLER than the true span DOES change the output (sanity: the
    windowing machinery is really active, not a no-op)."""
    sa, nh, dim = _mk_attn(seed=6)
    S = 40
    K, Vv, kpos = _cached_ctx(sa, S, dim, seed=7)
    xq = torch.randn(1, 3, dim) * 0.1
    qpos = torch.arange(S, S + 3)
    g = LA._global_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    sa._local_window = 5
    sa._global_head_mask = torch.zeros(nh, dtype=torch.bool)
    l = LA.windowed_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    assert float((g - l).abs().max()) > 0.0


def test_window_boundary_inclusive():
    """A window EXACTLY as wide as the span is byte-identical (boundary check):
    W = Sk covers distances 0..Sk-1, so the oldest key (dist Sk-1 < W) is kept."""
    sa, nh, dim = _mk_attn(seed=8)
    S = 20
    K, Vv, kpos = _cached_ctx(sa, S, dim, seed=9)   # cache positions 0..19
    xq = torch.randn(1, 1, dim) * 0.1
    qpos = torch.tensor([S])                          # query at pos 20
    g = LA._global_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    sa._local_window = S + 1                          # covers dist 0..S (oldest = S)
    sa._global_head_mask = torch.zeros(nh, dtype=torch.bool)
    l = LA.windowed_forward(sa, xq, (K, Vv, kpos), qpos, True)[0]
    assert float((g - l).abs().max()) == 0.0


def test_install_uninstall_restores_global():
    """install_local_attention swaps in the windowed forward; uninstall restores the
    ORIGINAL class forward (the per-instance override is removed)."""
    sa, nh, dim = _mk_attn(seed=10)

    class _M:
        pass
    m = _M()

    class _Blk:
        def __init__(self, at):
            self.attn = at
    m.blocks = [_Blk(sa)]
    # classification needs alibi slopes; give the non-memory default slopes.
    la = LA.install_local_attention(m, window=32)
    assert "forward" in sa.__dict__            # per-instance override present
    assert la["window"] == 32
    LA.uninstall_local_attention(m)
    assert "forward" not in sa.__dict__        # reverted
    assert not hasattr(sa, "_local_window")
