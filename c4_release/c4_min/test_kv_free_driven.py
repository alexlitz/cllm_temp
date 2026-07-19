"""FREE-DRIVEN, HEAP-MIRRORING KV eviction proof (BLOG_SPEC §Memory 410-412 +
§Memory Allocation and Freeing 689-691 + line 708).

This is the policy-validation harness for the free-driven / no-fixed-cap eviction
model.  It runs the REAL softmax1+ALiBi §Memory head (``blogspec_memory.KVMemory``
/ ``build_memory_model``) and drives the REFERENCE eviction policy
(``nibble_kv_prune.KVCache``) over that head's actual K/V projections, proving:

  1. BYTE-IDENTICAL — a load over the BOUNDED (evicted) cache decodes the exact
     same value as over the UNBOUNDED (never-evicted) cache, for every load in a
     malloc/use/free program AND a deep memory-heavy program.  Eviction drops
     ONLY nil-effect entries (freed / superseded / NULL rows).

  2. FREE-DRIVEN — the cache GROWS as distinct addresses are malloc'd, and SHRINKS
     when they are FREED (zero-overwritten).  A later load of a freed address reads
     0 (ZFOD) while a live address still reads its stored value.  There is NO fixed
     recency window / size cap dropping a live address — a live store survives
     until it is freed or superseded.

  3. UNBOUNDED — a memory-heavy program (many distinct live addresses, no frees)
     grows the cache to hold the LIVE HEAP (tracks it, is not capped).  No live
     data is dropped; every address still reads its value after eviction.

  4. ALL FOUR EVICT CATEGORIES fire: (a) FREED (zero-overwritten) rows, (b)
     SUPERSEDED same-address writes (older when a newer write exists), (c) NULL
     memory writes, (d) dead / zero-value rows.

Run: PYTHONPATH=<repo> python -m c4_min.test_kv_free_driven
 (or: python -m pytest c4_min/test_kv_free_driven.py -q)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from c4_min import blogspec_vocab as V
from c4_min.blogspec_memory import (
    build_memory_model, KVMemory, MemoryLayout, address_bits, MEM_ALIBI_SLOPE,
)
from c4_min.blogspec_layout import NIB_PER_REG
from c4_min import nibble_kv_prune as P


# ---------------------------------------------------------------------------
# A KVMemory subclass that projects each STORE position through the real memory
# head's W_k / W_v and mirrors it into a reference ``nibble_kv_prune.KVCache``,
# so we can (a) run the SPEC eviction policy on the actual head projections and
# (b) verify a load over the BOUNDED cache equals the load over the FULL stream.
# ---------------------------------------------------------------------------
class EvictingKVMemory(KVMemory):
    """KVMemory that maintains a bounded reference cache of the §Memory head.

    Every ``store`` appends the store row's real (W_k, W_v) projection to a
    ``KVCache`` built with ``content_addressed=True`` (a §Memory address-CAM
    head).  ``prune_now()`` applies the spec's free/superseded/NULL/dead-value
    eviction.  ``bounded_load`` decodes a load's value from the softmax1+ALiBi
    mix over the LIVE (post-eviction) cache — to be compared byte-for-byte with
    the ordinary (full-stream) ``load``.
    """

    def __init__(self, n_heads: int = 4, prune_interval: int = 4):
        model, L = build_memory_model(n_heads=n_heads)
        super().__init__(model, L)
        self.attn = model.blocks[0].attn
        self.scale = self.attn.scale
        self.slope = float(self.attn.alibi_slopes[0])       # MEM head slope
        # content_addressed=True: the §Memory head's non-zero store rows are LIVE
        # HEAP, protected from the recency horizon (free-driven, no fixed cap).
        self.cache = P.KVCache(slope=self.slope, score_scale=self.scale,
                               dup_metric="exact", content_addressed=True,
                               prune_interval=prune_interval)
        # a PARALLEL never-pruned reference cache (the UNBOUNDED baseline) over the
        # SAME head projections + absolute positions.  ``bounded_load`` (over
        # ``self.cache``) vs ``unbounded_load`` (over ``self.full``) is an
        # apples-to-apples eviction test: identical isolated-head decode, differing
        # ONLY by which entries the policy dropped — position-preserving (§712).
        self.full = P.KVCache(slope=self.slope, score_scale=self.scale,
                              dup_metric="exact", content_addressed=True,
                              prune_interval=10**9)           # never prunes
        self._pos = 1                                        # BOS at 0
        self.evict_hist: List[Tuple[str, int, int]] = []     # (event, addr, size)

    # -- overridden store: also mirror into the reference cache ---------------
    def store(self, addr: int, value: int, *, char: bool = False) -> None:
        super().store(addr, value, char=char)               # append to stream
        self._pos += 1
        # project this store row exactly as the head does.
        ov = self._stream[-1][1]
        x = self.model.embed[torch.tensor([[V.MEM]])].clone()[0, 0]
        for dim, val in ov.items():
            x[dim] = val
        k = torch.nn.functional.linear(x, self.attn.W_k).view(
            self.attn.n_heads, self.attn.head_dim)[0]
        v = torch.nn.functional.linear(x, self.attn.W_v).view(
            self.attn.n_heads, self.attn.head_dim)[0]
        self.cache.append(k, v, self._pos, meta=(addr, value))
        self.full.append(k, v, self._pos, meta=(addr, value))

    def null_write(self) -> None:
        """A NULL memory write (addr==val==0, IS_STORE=0) — the per-step row the
        VM emits every step when it is NOT storing.  Mirrored into the cache as a
        zero/near-zero row (mechanism-3 zero-value + free-zero territory)."""
        self._stream.append((V.MEM, {}))                    # no store overlay
        self._pos += 1
        x = self.model.embed[torch.tensor([[V.MEM]])].clone()[0, 0]
        k = torch.nn.functional.linear(x, self.attn.W_k).view(
            self.attn.n_heads, self.attn.head_dim)[0]
        v = torch.nn.functional.linear(x, self.attn.W_v).view(
            self.attn.n_heads, self.attn.head_dim)[0]
        self.cache.append(k, v, self._pos, meta=("NULL", 0))
        self.full.append(k, v, self._pos, meta=("NULL", 0))
        self._stream.pop()                                  # not a real store row

    def prune_now(self) -> int:
        n = self.cache.prune()
        return n

    def cache_size(self) -> int:
        return len(self.cache)

    # -- a load decoded over a given cache (spec attention) -------------------
    def _decode_from_cache(self, cache: P.KVCache, addr: int, char: bool) -> int:
        """Decode ``AX = *addr`` from the softmax1+ALiBi mix over ``cache``.

        Builds the load query exactly as the head does, mixes over the cache's
        entries via ``KVCache.attention_output`` (the spec math, position-
        preserving), then decodes the value nibbles W_o writes into the AX band.
        Used with the pruned cache (``bounded_load``) and the never-pruned
        reference (``unbounded_load``) so the two differ ONLY by eviction."""
        from c4_min.blogspec_memory import _decode_byte
        L = self.L
        x = self.model.embed[torch.tensor([[V.MEM]])].clone()[0, 0]
        ov = self._load_overlay(addr, char)
        for dim, val in ov.items():
            x[dim] = val
        q = torch.nn.functional.linear(x, self.attn.W_q).view(
            self.attn.n_heads, self.attn.head_dim)[0]
        qpos = self._pos + 1
        head_out = cache.attention_output(q, qpos, self.slope, self.scale)
        # W_o maps the head's local value slots -> AX nibble band, matching the
        # real head (bake_memory_head): AX_nib_j <- head_local_slot_j.
        state = torch.zeros(L.D)
        base = 0                                             # head 0 local base
        for j in range(NIB_PER_REG):
            state[L.AX + j] += head_out[base + j]
        val = 0
        for bi in range(4):
            val |= _decode_byte(state, L, L.AX, bi) << (8 * bi)
        return (val & 0xFF) if char else val

    def bounded_load(self, addr: int, *, char: bool = False) -> int:
        """Load over the PRUNED (bounded) cache — the eviction result."""
        return self._decode_from_cache(self.cache, addr, char)

    def unbounded_load(self, addr: int, *, char: bool = False) -> int:
        """Load over the never-pruned reference cache — the UNBOUNDED baseline."""
        return self._decode_from_cache(self.full, addr, char)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _n_live_stores(mem: EvictingKVMemory) -> int:
    """Count NON-zero-value entries in the cache = the live heap footprint."""
    return sum(1 for e in mem.cache.entries if float(e.value.norm()) > 1e-6)


# ===========================================================================
# 1 + 2.  FREE-DRIVEN: malloc grows, free shrinks, freed reads 0, byte-exact.
# ===========================================================================
def test_free_driven_grow_shrink_zfod():
    mem = EvictingKVMemory(prune_interval=1)
    # -- malloc phase: store distinct addresses; cache GROWS with the live heap --
    addrs = [0, 4, 8, 12, 16]
    vals = {a: 10 + a for a in addrs}
    sizes_after_alloc = []
    for a in addrs:
        mem.store(a, vals[a])
        mem.null_write()                     # a per-step NULL row (evictable)
        mem.prune_now()
        sizes_after_alloc.append(_n_live_stores(mem))
    # each malloc added exactly one live store; the NULL rows were evicted.
    assert sizes_after_alloc == [1, 2, 3, 4, 5], sizes_after_alloc
    peak = _n_live_stores(mem)
    assert peak == len(addrs), (peak, len(addrs))

    # every live address reads its value, BOUNDED == UNBOUNDED (byte-exact).
    for a in addrs:
        ub = mem.unbounded_load(a)           # never-pruned reference
        bd = mem.bounded_load(a)             # bounded (post-evict) cache
        assert ub == vals[a] == bd, (a, ub, vals[a], bd)

    # -- free phase: zero-overwrite frees; cache SHRINKS ------------------------
    freed = [0, 8, 16]
    live = [a for a in addrs if a not in freed]
    for a in freed:
        mem.free(a)                          # store(addr, 0): zero-overwrite
        mem.null_write()
        mem.prune_now()
    after_free = _n_live_stores(mem)
    # the live heap shrank to the still-allocated addresses (freed rows + their
    # zero-overwrite both evicted with nil effect).
    assert after_free == len(live), (after_free, len(live), freed)
    assert after_free < peak, (after_free, peak)

    # -- BYTE-IDENTICAL: bounded (freed rows evicted) == unbounded (kept), for
    #    EVERY address.  This is the spec's core claim (line 691): evicting BOTH
    #    the old value AND the zero-overwrite has NIL effect on the read.
    for a in addrs:
        assert mem.bounded_load(a) == mem.unbounded_load(a), (
            a, mem.bounded_load(a), mem.unbounded_load(a))
    # -- ZFOD: a freed address now reads 0; a live address still reads its value.
    for a in freed:
        assert mem.bounded_load(a) == 0, (a, mem.bounded_load(a))  # bounded ZFOD
    for a in live:
        assert mem.bounded_load(a) == vals[a], (a, mem.bounded_load(a))


# ===========================================================================
# 3.  UNBOUNDED: a memory-heavy program (big live array, no frees) grows the
#     cache to track the live heap — NOT capped, no live data dropped.
# ===========================================================================
def test_unbounded_heap_tracked_no_cap():
    mem = EvictingKVMemory(prune_interval=1)
    N = 40                                   # a 40-element live array (no frees)
    addrs = [4 * i for i in range(N)]
    vals = {a: (a * 3 + 1) & 0xFFFFFFFF for a in addrs}
    for a in addrs:
        mem.store(a, vals[a])
        mem.null_write()                     # NULL rows interleaved, all evict
        mem.prune_now()
    live = _n_live_stores(mem)
    # the cache GREW to hold ALL N distinct live addresses — it tracks the heap,
    # is NOT capped at some fixed size.
    assert live == N, (live, N)
    # sanity: growth is monotone with distinct mallocs (no cap truncation).
    assert live > 32, live                   # would fail if a 32-cap silently hit
    # NO live data dropped: every one of the N addresses still reads its value,
    # BOUNDED == UNBOUNDED, even the FIRST-written (oldest, most recency-penalised)
    # address, which a fixed recency window would have evicted.
    for a in addrs:
        ub = mem.load(a)
        bd = mem.bounded_load(a)
        assert ub == vals[a] == bd, (a, ub, vals[a], bd)
    # the OLDEST address specifically (max ALiBi distance) survives — the direct
    # refutation of a recency-window cap on live heap.
    assert mem.bounded_load(addrs[0]) == vals[addrs[0]]


# ===========================================================================
# 4.  All four evict categories fire.
# ===========================================================================
def test_all_four_evict_categories():
    slope, scale = MEM_ALIBI_SLOPE, 1.0

    # (a) FREED (zero-overwrite): old value row + the zeroing write both evict.
    ca = P.KVCache(slope=slope, score_scale=scale, dup_metric="exact",
                   content_addressed=True, recency_eps=1e-6)
    kA = torch.zeros(8); kA[0] = 5.0                     # address-A key
    ca.append(kA.clone(), torch.tensor([9.0] + [0.0] * 7), position=1)   # value 9
    ca.append(kA.clone(), torch.zeros(8), position=40)   # zero-overwrite (free)
    ca.append(torch.zeros(8), torch.zeros(8), position=80)  # a later NULL row
    freed = ca.prune()
    assert freed >= 2, freed                             # both A-rows gone (freed)
    assert all(float(e.value.norm()) < 1e-6 or e.position != 1 for e in ca.entries)

    # (b) SUPERSEDED (older same-address write dropped, latest-write-wins).
    cb = P.KVCache(slope=slope, score_scale=scale, dup_metric="exact",
                   content_addressed=True)
    k = torch.zeros(8); k[1] = 5.0
    cb.append(k.clone(), torch.tensor([3.0] + [0.0] * 7), position=1)     # old
    cb.append(k.clone(), torch.tensor([7.0] + [0.0] * 7), position=40)    # new
    sup = cb.prune()
    assert sup == 1, sup
    assert cb.entries[0].position == 40                 # newest survives

    # (c) NULL memory writes (addr==val==0) evict once recency-stale.
    cc = P.KVCache(slope=slope, score_scale=scale, dup_metric="exact",
                   content_addressed=True, recency_eps=1e-6)
    live = torch.zeros(8); live[2] = 5.0
    cc.append(live.clone(), torch.tensor([4.0] + [0.0] * 7), position=1)  # live
    for p in range(2, 60):                               # many NULL rows
        cc.append(torch.zeros(8), torch.zeros(8), position=p)
    nulls = cc.prune()
    assert nulls >= 55, nulls                            # NULL rows evicted
    assert any(float(e.value.norm()) > 1e-6 for e in cc.entries)   # live kept

    # (d) DEAD / zero-value head evicts everything (mechanism 2a).
    cd = P.KVCache(slope=slope, score_scale=scale, dup_metric="exact",
                   content_addressed=True)
    for p in range(1, 20):
        cd.append(torch.randn(8), torch.zeros(8), position=p)   # all zero-value
    dead = cd.prune()
    assert dead == 19 and len(cd) == 0, (dead, len(cd))


def test_no_recency_cap_on_live_store():
    """The KEY no-fixed-cap regression: on a content-addressed §Memory head the
    recency HORIZON must NOT drop a live (non-zero-value, non-superseded) store —
    even one far behind the frontier that a register head's horizon WOULD evict.
    This is the direct refutation of a fixed recency/size window on live heap."""
    slope, scale = MEM_ALIBI_SLOPE, 1.0
    # A live store with a SMALL key (ceil~0) at distance 99 — the recency horizon
    # (exp(ceil - slope*99) < recency_eps) fires on a register head but must be a
    # NO-OP on a content-addressed head.
    def _run(content_addressed):
        c = P.KVCache(slope=slope, score_scale=scale, dup_metric="exact",
                      content_addressed=content_addressed)
        ksmall = torch.zeros(34); ksmall[0] = 0.5
        v = torch.zeros(34); v[0] = 4.0                  # live non-zero value
        c.append(ksmall.clone(), v.clone(), position=1)          # old live store
        c.append(ksmall.clone() + 0.01, v.clone(), position=100)  # newer distinct store
        c.prune()
        return any(e.position == 1 for e in c.entries)
    assert _run(content_addressed=True) is True     # live heap: kept (no cap)
    assert _run(content_addressed=False) is False   # register head: recency drops
    # Unbounded distance: a live store at dist 2,000,000 (>> EFF/slope=500000, the
    # old fixed recall-horizon cap) STILL survives — the cache is UNBOUNDED.
    from c4_min.blogspec_memory import EFF
    c = P.KVCache(slope=slope, score_scale=scale, dup_metric="exact",
                  content_addressed=True)
    k = torch.zeros(34)
    for b in range(32):
        k[b] = (EFF / scale) ** 0.5 * (1 if b % 2 else -1)
    v = torch.zeros(34); v[0] = 4.0
    c.append(k.clone(), v.clone(), position=1)
    c.append(torch.full((34,), 0.01), torch.zeros(34), position=1 + 2_000_000)
    c.prune()
    assert any(e.position == 1 for e in c.entries), "live store dropped past EFF!"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} free-driven KV-evict tests passed")
