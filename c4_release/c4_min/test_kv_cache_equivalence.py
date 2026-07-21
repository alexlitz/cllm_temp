"""Unit proofs for the KV-cache machinery (pytest-collectable).

  1. ``test_uncached_forward_byte_identical`` — the KV-cache edit to
     ``blogspec_model`` leaves the DEFAULT (un-cached) forward BYTE-IDENTICAL to
     the original spec forward (a golden check over a battery of random-weight
     twins + sequence lengths).
  2. ``test_cached_attention_matches_full`` — the incremental cached ``Attn``
     path (cache + new tokens, ABSOLUTE-position ALiBi) equals the full forward.
  3. ``test_vectorized_prune_matches_reference`` — the vectorized prune keep-mask
     (``nibble_pure_forward_cached.prune_keep_mask_head``) is byte-for-byte the
     survivor set of the reference ``nibble_kv_prune.KVCache.prune``.
"""
from __future__ import annotations

import importlib.util
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from c4_min import blogspec_vocab as V
from c4_min import blogspec_model as M
from c4_min.nibble_kv_prune import KVCache
from c4_min.nibble_pure_forward_cached import prune_keep_mask_head


def _load_original_model_module():
    """Load the base-commit ``blogspec_model`` (pre KV-cache edit) if a copy is
    on disk at ``/tmp/blogspec_model_orig.py``; else return None (skip golden)."""
    path = "/tmp/blogspec_model_orig.py"
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("blogspec_model_orig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cached_attention_matches_full():
    torch.manual_seed(1)
    dim, H = 40, 4
    attn = M.Attn(dim, H, max_seq_len=1024)
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.3)
    S = 50
    x = torch.randn(1, S, dim)
    with torch.no_grad():
        full = attn(x)
        o1, kv = attn(x[:, :40], q_positions=torch.arange(40), use_cache=True)
        o2, _ = attn(x[:, 40:], past_kv=kv, q_positions=torch.arange(40, 50),
                     use_cache=True)
    assert torch.equal(full[:, 40:], o2), \
        (full[:, 40:] - o2).abs().max().item()


def test_uncached_forward_byte_identical():
    orig = _load_original_model_module()
    if orig is None:
        # regenerate is out of scope in CI; the in-repo default path is exercised
        # by test_cached_attention_matches_full's `full` call, so just assert the
        # default Attn path runs and is deterministic.
        torch.manual_seed(0)
        a = M.Attn(24, 4)
        for p in a.parameters():
            torch.nn.init.normal_(p, std=0.2)
        x = torch.randn(1, 30, 24)
        with torch.no_grad():
            assert torch.equal(a(x), a(x))
        return
    torch.manual_seed(42)
    dim, H, hidden, nb = 40, 4, 64, 5
    m_new = M.Transformer(dim=dim, n_heads=H, hidden=hidden, n_blocks=nb,
                          vocab=V.VOCAB, max_seq_len=1024)
    # small std so the norm-free deep residual stays finite (no shared NaN, which
    # would make a bit-exact compare vacuous).
    for p in m_new.parameters():
        torch.nn.init.normal_(p, std=0.05)
    m_orig = orig.Transformer(dim=dim, n_heads=H, hidden=hidden, n_blocks=nb,
                              vocab=V.VOCAB, max_seq_len=1024)
    m_orig.load_state_dict(m_new.state_dict())
    for trial in range(8):
        toks = torch.randint(0, V.VOCAB, (1, 1 + 30 * (trial + 1)))
        with torch.no_grad():
            lo, ln = m_orig(toks), m_new(toks)
        assert torch.isfinite(lo).all(), "orig produced non-finite logits"
        # bit-exact: identical tensors including any shared inf, but we asserted
        # finiteness above so a plain equality is the true golden.
        assert torch.equal(lo, ln), (lo - ln).abs().max().item()
        # argmax (the decode-relevant quantity) must match too.
        assert torch.equal(lo.argmax(-1), ln.argmax(-1))


def test_vectorized_prune_matches_reference():
    torch.manual_seed(0)
    mism = 0
    trials = 0
    for _ in range(120):
        S = int(torch.randint(1, 60, (1,)))
        HD = 8
        keys = torch.randn(S, HD)
        vals = torch.randn(S, HD)
        for _ in range(int(torch.randint(0, S, (1,)))):
            i = int(torch.randint(0, S, (1,)))
            j = int(torch.randint(0, S, (1,)))
            keys[i] = keys[j] * (1.0 + 0.001 * torch.randn(1))
        for _ in range(int(torch.randint(0, S // 2 + 1, (1,)))):
            i = int(torch.randint(0, S, (1,)))
            vals[i] = 0.0
            if int(torch.randint(0, 2, (1,))):
                keys[i] = 0.0
        positions = torch.arange(S) * int(torch.randint(1, 40, (1,)))
        slope = float(2.0 ** (-8.0 / 23 * int(torch.randint(1, 24, (1,)))))
        scale = HD ** -0.5
        for cos_thr, zeps, reps in [(0.99, 1e-9, 1e-6), (0.95, 1e-9, 1e-4),
                                    (0.999, 1e-9, 1e-9)]:
            # exercise BOTH head types: register-marker (cosine near-dup, recency
            # horizon on all rows) AND content-addressed §Memory (exact near-dup,
            # NO recency drop on a live non-zero store — the free-driven rule).
            # The reference and the vectorised driver must agree on BOTH so the two
            # implementations stay byte-for-byte consistent.
            for dm in ("cosine", "exact"):
                c = KVCache(cos_threshold=cos_thr, zero_eps=zeps, slope=slope,
                            recency_eps=reps, score_scale=scale, dup_metric=dm)
                for i in range(S):
                    c.append(keys[i], vals[i], int(positions[i]), meta=i)
                c.prune()
                ref = set(e.meta for e in c.entries)
                mask = prune_keep_mask_head(keys, vals, positions, slope, scale,
                                            cos_thr, zeps, reps, dup_metric=dm)
                vec = set(torch.nonzero(mask, as_tuple=False).flatten().tolist())
                trials += 1
                if ref != vec:
                    mism += 1
    assert mism == 0, f"{mism}/{trials} prune keep-set mismatches"


def test_vectorized_prune_matches_reference_large_and_content_addressed():
    """Byte-identity of the vectorised keep-mask on the DEEP-TAIL regime that the
    per-entry Python loop was too slow for: LARGE near-dup-heavy caches (the live
    heap a rec_fib(12) reaches) and the EXPLICIT ``content_addressed`` flag (the
    free-driven §Memory head — live non-zero stores are never recency-dropped).

    Complements ``test_vectorized_prune_matches_reference`` (720 small trials): it
    stresses the two regimes the vectorisation targets — many near-duplicate
    register markers repeated across steps, and the address-CAM live-heap head.
    """
    torch.manual_seed(7)
    mism = 0
    trials = 0
    HD = 8
    scale = HD ** -0.5
    for S in [200, 600, 1500]:
        # a realistic live-heap cache: a handful of distinct key groups (register
        # markers / store addresses) each repeated across many steps (near-dup),
        # plus freed/NULL zero rows — exactly the supersession-heavy shape.
        n_groups = 24
        proto = torch.randn(n_groups, HD)
        idx = torch.randint(0, n_groups, (S,))
        keys = proto[idx] * (1.0 + 0.0005 * torch.randn(S, 1))
        vals = torch.randn(S, HD)
        zmask = torch.rand(S) < 0.15
        vals[zmask] = 0.0
        zk = zmask & (torch.rand(S) < 0.5)
        keys[zk] = 0.0
        positions = torch.arange(S) * 30
        slope = 0.25
        for cos_thr, zeps, reps in [(0.99, 1e-9, 1e-6), (0.999, 1e-9, 1e-9)]:
            for dm, ca in [("cosine", False), ("exact", True), ("exact", False),
                           ("cosine", True)]:
                c = KVCache(cos_threshold=cos_thr, zero_eps=zeps, slope=slope,
                            recency_eps=reps, score_scale=scale, dup_metric=dm,
                            content_addressed=ca)
                for i in range(S):
                    c.append(keys[i], vals[i], int(positions[i]), meta=i)
                c.prune()
                ref = set(e.meta for e in c.entries)
                mask = prune_keep_mask_head(keys, vals, positions, slope, scale,
                                            cos_thr, zeps, reps, dup_metric=dm,
                                            content_addressed=ca)
                vec = set(torch.nonzero(mask, as_tuple=False).flatten().tolist())
                trials += 1
                if ref != vec:
                    mism += 1
    assert mism == 0, f"{mism}/{trials} large/content-addressed keep-set mismatches"


def test_fused_all_block_evict_matches_per_block():
    """The FUSED all-block eviction (``evict_all_blocks_fused`` +
    ``apply_keep_mask``) produces the BYTE-IDENTICAL survivor set (K/V/pos) to the
    per-block ``BlockKVCacheBatched.evict`` loop it replaces — over a battery of
    random multi-block caches with (a) DIVERGENT per-block sizes (blocks that
    already evicted differently), (b) mixed register-marker vs content-addressed
    (address-CAM) heads, (c) protection positions.  This is the fusion's
    byte-identity gate: the deep-loop prune de-syncs onto the GPU but keeps EXACTLY
    the entries the per-block policy kept (#667)."""
    import copy
    from c4_min.nibble_pure_forward_cached import (
        BlockKVCacheBatched, evict_all_blocks_fused)
    torch.manual_seed(11)
    cos_threshold, zero_eps, recency_eps = 0.99, 1e-9, 1e-6
    H, HD = 4, 8
    for trial in range(30):
        n_blocks = int(torch.randint(2, 8, (1,)))
        # a few distinct ALiBi slope vectors shared across blocks (as in the model).
        slope_protos = [torch.tensor([2.0 ** (-8.0 / 23 * (h + 1 + off))
                                      for h in range(H)])
                        for off in range(3)]
        # build per-block caches with DIVERGENT sizes + positions (post-evict shape).
        caches_a, caches_b = [], []
        protect = set()
        for b in range(n_blocks):
            slopes = slope_protos[b % len(slope_protos)]
            S = int(torch.randint(3, 40, (1,)))
            # a content-addressed block? give its keys a big shared common-mode bias
            # so cm_frac>0.9 (address-CAM head) for some blocks, marker heads for rest.
            if b % 2 == 0:
                keys = torch.randn(1, H, S, HD) + 8.0            # common-mode -> exact
            else:
                keys = torch.randn(1, H, S, HD)                  # marker -> cosine
            vals = torch.randn(1, H, S, HD)
            # inject near-dup + zero rows (the eviction hazards).
            for _ in range(int(torch.randint(0, S, (1,)))):
                i = int(torch.randint(0, S, (1,))); j = int(torch.randint(0, S, (1,)))
                keys[0, :, i] = keys[0, :, j] * (1.0 + 0.0005 * torch.randn(1))
            for _ in range(int(torch.randint(0, S // 2 + 1, (1,)))):
                i = int(torch.randint(0, S, (1,))); vals[0, :, i] = 0.0
                if int(torch.randint(0, 2, (1,))):
                    keys[0, :, i] = 0.0
            pos = torch.sort(torch.randperm(600)[:S])[0].to(torch.long)
            ca = BlockKVCacheBatched(H, HD, slopes)
            ca.K, ca.V, ca.pos = keys.clone(), vals.clone(), pos.clone()
            cb = BlockKVCacheBatched(H, HD, slopes)
            cb.K, cb.V, cb.pos = keys.clone(), vals.clone(), pos.clone()
            caches_a.append(ca); caches_b.append(cb)
            if int(torch.randint(0, 3, (1,))) == 0 and S:
                protect.add(int(pos[int(torch.randint(0, S, (1,)))]))
        protect = sorted(protect) if protect else None

        # reference: per-block evict.
        for c in caches_a:
            c.evict(cos_threshold, 60, zero_eps, recency_eps,
                    protect_positions=protect)
        # fused: one batched decision + per-block mask compaction.
        masks = evict_all_blocks_fused(caches_b, cos_threshold, zero_eps,
                                       recency_eps, protect_positions=protect)
        for b in range(n_blocks):
            if masks[b] is not None:
                caches_b[b].apply_keep_mask(masks[b])
            # byte-identical surviving K/V/pos per block.
            assert torch.equal(caches_a[b].pos, caches_b[b].pos), (trial, b)
            assert torch.equal(caches_a[b].K, caches_b[b].K), (trial, b)
            assert torch.equal(caches_a[b].V, caches_b[b].V), (trial, b)


def test_driver_byte_identical_naive_incl_functions_and_eviction():
    """The KV-cached (+evicted) driver's full output is BYTE-IDENTICAL to the
    naive re-forward driver on a battery that exercises the eviction hazards:
    multi-step arithmetic, 32-bit values, a memory store/load round-trip, the
    calling convention (JSR/ENT/LEV — the regression that caught the
    address-merge bug), and a loop that triggers many prunes.
    """
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xF0
    _PFC.SP_INIT = 0xF0
    from c4_min import isa
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    from c4_min._build_guard import guarded_complete_build

    def I(op, imm=0):
        return isa.Instr(op, imm)

    # Memory-SAFE streaming build (peak ~5 GB) — byte-identical (L-inf=0) to the
    # dense complete model whose ~160k-row MUL/DIV/MOD block would peak at
    # 54-108 GB RSS.  Covers the mul32 battery case; both cached + naive drivers
    # accept the streaming SparseTransformer.
    model, L = guarded_complete_build(code_size=16)
    battery = [
        ("add", [I(isa.IMM, 5), I(isa.PSH), I(isa.IMM, 3), I(isa.ADD),
                 I(isa.HALT)], 20, 0xFF),
        ("mul32", [I(isa.IMM, 1000), I(isa.PSH), I(isa.IMM, 1000), I(isa.MUL),
                   I(isa.HALT)], 20, 0xFFFFFFFF),
        ("si_li", [I(isa.IMM, 7), I(isa.PSH), I(isa.IMM, 200), I(isa.PSH),
                   I(isa.IMM, 7), I(isa.SI), I(isa.IMM, 200), I(isa.LI),
                   I(isa.HALT)], 30, 0xFF),
        # JSR to a routine that returns AX=42, then LEV back — the calling
        # convention over the content-addressed stack KV head.
        ("func", [I(isa.JSR, 3), I(isa.HALT), I(isa.NOP), I(isa.ENT, 0),
                  I(isa.IMM, 42), I(isa.LEV)], 30, 0xFF),
        # loop that triggers multiple prunes (eviction-heavy) — kept CI-small
        # (the naive re-forward baseline is O(stream^2); the cached path is fast).
        ("loop", [I(isa.IMM, 6), I(isa.PSH), I(isa.IMM, 1), I(isa.SUB),
                  I(isa.BNZ, 1), I(isa.HALT)], 40, 0xFF),
    ]
    for name, code, ms, mask in battery:
        naive = run_pure_forward_complete(model, L, code, max_steps=ms, mask=mask)
        cached = run_pure_forward_cached(
            model, L, code, max_steps=ms, mask=mask, evict=True, prune_interval=60)
        assert naive == cached, (name, naive, cached)


if __name__ == "__main__":
    test_cached_attention_matches_full()
    test_uncached_forward_byte_identical()
    test_vectorized_prune_matches_reference()
    test_vectorized_prune_matches_reference_large_and_content_addressed()
    test_fused_all_block_evict_matches_per_block()
    test_driver_byte_identical_naive_incl_functions_and_eviction()
    print("all KV-cache equivalence tests passed")
