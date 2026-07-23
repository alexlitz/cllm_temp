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


# ---------------------------------------------------------------------------
# CONTENT-BOUND global-head retention: the store-role gate keep-predicate.
# ---------------------------------------------------------------------------
def _mk_split_cache(Hg=3, S=10, HD=8, cR=5, p=1e5, store_rows=(1, 4, 7)):
    """A synthetic split cache whose GLOBAL keys carry the store-role gate: store
    rows key ~0 at channel ``cR``, non-store rows key ``-p`` (as the real §Memory /
    stack / LEV CAM heads do)."""
    from c4_min.nibble_pure_forward_cached import BlockKVCacheBatched
    slopes = torch.ones(Hg + 2)                # arbitrary; only slice used by cache
    c = BlockKVCacheBatched(n_heads=Hg + 2, head_dim=HD, slopes=slopes)
    c.set_head_groups(list(range(Hg)), local_window=4,
                      content_bound=True, content_cR=cR)
    K = torch.randn(1, Hg, S, HD)
    for s in range(S):
        gate = 0.0 if s in store_rows else -p
        K[:, :, s, cR] = gate
    V = torch.randn(1, Hg, S, HD)
    pos = torch.arange(S)
    return c, K, V, pos, set(store_rows)


def test_content_keep_predicate_selects_store_rows():
    """``_global_content_keep`` keeps EXACTLY the store rows (gate ~0) and drops the
    non-store rows (gate -p) — the provably-inert frames a global head weights 0."""
    c, K, V, pos, store = _mk_split_cache()
    keep = c._global_content_keep(K[0])         # [S] bool
    kept = set(int(i) for i in torch.nonzero(keep).flatten().tolist())
    assert kept == store, (kept, store)


def test_content_commit_keeps_only_store_rows():
    """A content-bound split commit routes ONLY the store rows into the global cache;
    the global cache size == #store rows, not the span length."""
    c, K, V, pos, store = _mk_split_cache(S=12, store_rows=(2, 5, 9))
    # commit a full-H new span (the split routes local vs global internally); build a
    # full-H K/V whose GLOBAL-head slice matches K/V above.
    H = c.n_heads
    Kf = torch.zeros(1, H, K.shape[2], K.shape[3])
    Vf = torch.zeros(1, H, K.shape[2], K.shape[3])
    g = c._global_head_idx
    Kf[:, g] = K
    Vf[:, g] = V
    c.commit(Kf, Vf, pos)
    assert c.size() == len(store), (c.size(), len(store))
    # the surviving positions are exactly the store rows.
    assert set(int(p) for p in c.pos.tolist()) == store


def test_content_off_keeps_full_history():
    """With ``content_bound=False`` the global cache keeps EVERY committed row (the
    plain drop-KV split), so a non-store frame is retained."""
    from c4_min.nibble_pure_forward_cached import BlockKVCacheBatched
    Hg, S, HD = 2, 8, 8
    c = BlockKVCacheBatched(n_heads=Hg + 1, head_dim=HD, slopes=torch.ones(Hg + 1))
    c.set_head_groups(list(range(Hg)), local_window=4, content_bound=False)
    H = c.n_heads
    Kf = torch.randn(1, H, S, HD)
    Vf = torch.randn(1, H, S, HD)
    c.commit(Kf, Vf, torch.arange(S))
    assert c.size() == S           # full history kept (no content drop)
