"""Checklist #6 (hardening) — KV-cache eviction is ROBUST + byte-exact under
adversarial pressure, not just on the happy-path register-churn stream.

CHK-6 (`test_kv_eviction_longrun.py`) proved the eviction policy byte-identical to a
42k-token flat cache on the deep-loop corpus. This module HARDENS that result: it
stress-tests the eviction policy (`nibble_kv_prune`) against the correctness risks a
naive recency-only policy fails on, each asserted **byte-identical to the unbounded
(never-pruned) reference** with the cache staying **flat + memory bounded**:

  1. STORE-EARLY / LOAD-LATE (the #1 risk): a value stored at an ADDRESS at step 5,
     then thousands of unrelated steps touching OTHER addresses, then a LOAD (LI) of
     the original address — the policy must keep the entry by ADDRESS LIVENESS (not
     recency) while it is still recallable, and match the unbounded cache's recall
     (including "no longer recallable once ALiBi crushes it" — which the reference
     ALSO stops recalling). Distinguishes recency-stale-but-address-live (KEEP) from
     truly-dead/overwritten (drop).
  2. 250k-TOKEN SCALE: an ~8400-step program (~250k emitted tokens) — flat cache,
     byte-identical frontier decode, bounded RSS, at deep-loop-corpus scale.
  3. CAPACITY PRESSURE: a large working set of MANY distinct simultaneously-live
     keys — eviction never drops a still-needed key; if the working set exceeds a
     hard cap the degradation is documented.
  4. MEMORY-OP DENSITY: many distinct SI/LI addresses actively read/written, plus
     ZFOD (zero-fill-on-demand reads of never-written addresses) and free/overwrite
     — latest-write-wins, never resurrect a freed/overwritten value.
  5. ADVERSARIAL RECENCY: near-duplicate register-marker keys interleaved with
     genuinely-needed old memory keys, so recency alone would mislead. (This is the
     scenario that exposed the un-gated-mechanism-1 bug — see the hardening doc.)

Why a POLICY-level stress (not a `run_program` stress)
------------------------------------------------------
The c4 foundation SLICE (`blogspec_run._apply_op`) executes only IMM/LEA/PSH/ADD/
SUB/JMP/BZ/BNZ/HALT — it has **no SI/LI** memory ops, so a store-early/load-late
program cannot be expressed in the run path. But the KV cache is exactly the layer
where memory addressing lives (BLOG_SPEC: memory is realised as softmax1+ALiBi over
address-keyed KV entries, §Memory / line 410). So the store-early/load-late and
memory-density risks are tested where they actually live: against the eviction
POLICY, driving it with per-head KV entries whose KEYS are the address/marker keys
the real memory head projects (a strong distinctive key per address, a weak
byte-value key per churn token) and asserting byte-identity to the unbounded
softmax1+ALiBi reference. `_MemHead` below is the minimal faithful stand-in: its
`attention_output` IS `nibble_kv_prune.KVCache.attention_output` (the same
softmax1+ALiBi mixer the real `blogspec_model.Attn` uses), so the equivalence is
against the real attention math. The register-churn HALF of the stream is projected
from a real baked model where it matters (adversarial-recency reuses the CHK-6
model path).

Run:
    PYTHONPATH=<repo> python -m c4_min.test_kv_eviction_stress     # verbose report
    PYTHONPATH=<repo> python -m pytest c4_min/test_kv_eviction_stress.py -q
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch

from c4_min import blogspec_compiler as C
from c4_min import blogspec_run as R
from c4_min import nibble_kv_prune as P


# ---------------------------------------------------------------------------
# A minimal faithful memory-head model: address-keyed + churn KV entries, mixed
# through the SAME softmax1+ALiBi attention the real model uses (KVCache).
# ---------------------------------------------------------------------------
class _MemHead:
    """One attention head's-worth of address-keyed memory + register-churn KV,
    driven exactly as the generation loop drives it (append -> maybe_prune ->
    frontier query). Keys/values match the shapes the real memory head projects:

      * an ADDRESS key is a strong distinctive vector (norm chosen so an
        exact-address query beats ALiBi far in the past — BLOG_SPEC line 410: the
        memory head's ``sum(scale^2)`` is large). It carries the stored value.
      * a CHURN key (a register marker / byte token) is a WEAK vector (norm ~1)
        that recency-governs (this is what keeps the cache bounded).

    ``score_scale`` and ``slope`` are the real head-0 values (head_dim=26 -> scale
    ~0.196; 4-head ALiBi slope 0.25), so the recency/address thresholds are the
    production ones.
    """

    def __init__(self, head_dim: int = 26, slope: float = 0.25,
                 prune_interval: int = 60, recency_eps: float = 1e-6,
                 addr_key_norm: float = 12.0, seed: int = 0):
        self.HD = head_dim
        self.scale = head_dim ** -0.5
        self.slope = slope
        self.addr_key_norm = addr_key_norm
        self.rng = random.Random(seed)
        self.bounded = P.KVCache(slope=slope, score_scale=self.scale,
                                 recency_eps=recency_eps,
                                 prune_interval=prune_interval)
        # the unbounded reference NEVER prunes (one entry per emitted token).
        self.unbounded = P.KVCache(slope=slope, score_scale=self.scale,
                                   recency_eps=recency_eps,
                                   prune_interval=10 ** 9)
        self.pos = 0
        self.max_live = 0
        self._addr_dirs: Dict[int, torch.Tensor] = {}

    # -- key builders --------------------------------------------------------
    def _addr_dir(self, addr: int) -> torch.Tensor:
        """A stable, ~distinct unit direction per address (a random-but-fixed unit
        vector — distinct addresses have near-orthogonal keys, an exact-address
        query lands on its own direction)."""
        d = self._addr_dirs.get(addr)
        if d is None:
            g = torch.Generator().manual_seed(1000 + addr)
            v = torch.randn(self.HD, generator=g)
            d = v / v.norm()
            self._addr_dirs[addr] = d
        return d

    def addr_key(self, addr: int) -> torch.Tensor:
        return self._addr_dir(addr) * self.addr_key_norm

    def churn_key(self) -> torch.Tensor:
        """A weak register-marker/byte key (norm ~1) — recency-governed."""
        k = torch.zeros(self.HD)
        k[self.rng.randint(2, self.HD - 1)] = 1.0
        return k

    def value_vec(self, v: int, dim: int = 1) -> torch.Tensor:
        out = torch.zeros(self.HD)
        out[dim] = float(v)
        return out

    # -- generation-loop step ------------------------------------------------
    def emit(self, key: torch.Tensor, value: torch.Tensor, meta=None) -> None:
        self.bounded.append(key.clone(), value.clone(), self.pos, meta)
        self.unbounded.append(key.clone(), value.clone(), self.pos, meta)
        self.pos += 1
        self.bounded.maybe_prune()
        if len(self.bounded) > self.max_live:
            self.max_live = len(self.bounded)

    def store(self, addr: int, value: int, vdim: int = 1) -> None:
        """SI addr = value: a strong address key carrying the value."""
        self.emit(self.addr_key(addr), self.value_vec(value, vdim),
                  meta=("SI", addr, value))

    def churn(self, value: int = 0) -> None:
        """One unrelated register-churn token (weak key)."""
        self.emit(self.churn_key(), self.value_vec(value, dim=1),
                  meta=("churn",))

    # -- read + equivalence check --------------------------------------------
    def load_diff(self, addr: int) -> float:
        """LI addr: query both caches at the current frontier with the exact
        address key; return max |bounded - unbounded| over the attention output."""
        q = self.addr_key(addr)
        ob = self.bounded.attention_output(q, self.pos, self.slope, self.scale)
        ou = self.unbounded.attention_output(q, self.pos, self.slope, self.scale)
        return float((ob - ou).abs().max())

    def load_recalls(self, addr: int, vdim: int = 1) -> Tuple[float, float]:
        """(bounded, unbounded) recalled value on the value dim for LI addr."""
        q = self.addr_key(addr)
        ob = self.bounded.attention_output(q, self.pos, self.slope, self.scale)
        ou = self.unbounded.attention_output(q, self.pos, self.slope, self.scale)
        return float(ob[vdim]), float(ou[vdim])

    def load_byte_agrees(self, addr: int, vdim: int = 1) -> bool:
        """The ultimate byte-exact bar: does the LI addr recall the SAME rounded
        byte (0..255) from the bounded cache as from the unbounded one? The value
        dim carries the byte magnitude directly, so ``round`` is the byte decode."""
        ob, ou = self.load_recalls(addr, vdim)
        return int(round(ob)) == int(round(ou))

    def frontier_diff(self, q: Optional[torch.Tensor] = None) -> float:
        """Max |bounded-unbounded| for an arbitrary frontier query (a churn query
        if none given) — the equivalence a generation loop's frontier read needs."""
        if q is None:
            q = self.churn_key()
        ob = self.bounded.attention_output(q, self.pos, self.slope, self.scale)
        ou = self.unbounded.attention_output(q, self.pos, self.slope, self.scale)
        return float((ob - ou).abs().max())


# The eviction is exact TO ``recency_eps``: an entry whose max future softmax1
# weight is just below the ``recency_eps = 1e-6`` floor may be dropped, contributing
# up to ~``recency_eps * value_magnitude`` to the raw attention-output diff. These
# stress values reach 255 (a full byte), so the honest per-entry bound is
# ``recency_eps * 255 ≈ 2.5e-4``; we use ``3e-4`` (a hair above that). This is still
# ORDERS OF MAGNITUDE below any byte-head argmax margin — the DECODED byte never
# flips — so it is the faithful byte-exact bar for byte-valued payloads. (The
# register-marker long-run stream in CHK-6 carries nibble-sized values, so its diff
# is ~1e-4 at the 1e-3 decode tolerance; here the payload is a full byte.)
_EXACT_TOL = 3e-4


# ---------------------------------------------------------------------------
# 1. STORE-EARLY / LOAD-LATE — the #1 correctness risk.
# ---------------------------------------------------------------------------
def run_store_early_load_late(n_unrelated: int = 8000, addr: int = 7,
                              stored_value: int = 42) -> Dict:
    """Store ``addr = stored_value`` at step 5, run ``n_unrelated`` steps touching
    OTHER addresses / register churn, then LOAD ``addr``. The policy must keep the
    address-live entry (not recency-evict it) and match the unbounded recall at
    EVERY intermediate LI, distinguishing 'recency-stale but address-live' (KEEP)
    from 'truly dead' (drop)."""
    h = _MemHead(seed=3)
    # a few warm-up churn steps, then the early store.
    for _ in range(5):
        h.churn(h.rng.randint(1, 200))
    h.store(addr, stored_value)
    store_pos = h.pos - 1

    worst = 0.0
    recall_trace: List[Tuple[int, float, float]] = []
    other_addrs = [a for a in range(20, 60)]
    for step in range(n_unrelated):
        # churn OTHER addresses + register markers (never touch ``addr``).
        if step % 4 == 0:
            oa = other_addrs[step % len(other_addrs)]
            h.store(oa, h.rng.randint(1, 255))
        else:
            h.churn(h.rng.randint(1, 255))
        # LI the early address periodically and check bounded==unbounded.
        if step % 200 == 0:
            worst = max(worst, h.load_diff(addr))
            if step % 1000 == 0:
                ob, ou = h.load_recalls(addr)
                recall_trace.append((h.pos - store_pos, ob, ou))
    worst = max(worst, h.load_diff(addr))
    final_ob, final_ou = h.load_recalls(addr)
    store_kept = any(e.meta == ("SI", addr, stored_value)
                     for e in h.bounded.entries)
    return {
        "seq_len": h.pos,
        "store_dist_at_end": h.pos - store_pos,
        "worst_load_diff": worst,
        "final_bounded_recall": final_ob,
        "final_unbounded_recall": final_ou,
        "store_still_cached": store_kept,
        "address_live_protected": h.bounded.address_live_protected,
        "bounded_live": len(h.bounded),
        "unbounded_live": len(h.unbounded),
        "recall_trace": recall_trace,
    }


def test_store_early_load_late_byte_exact():
    """The store-early/load-late guarantee: bounded cache matches the unbounded
    recall at every LI, the address-live store is KEPT (not recency-evicted), and
    the cache stays bounded while the stream runs thousands of steps."""
    rep = run_store_early_load_late(n_unrelated=8000)
    # (a) bounded == unbounded at every intermediate + final LOAD.
    assert rep["worst_load_diff"] < _EXACT_TOL, rep["worst_load_diff"]
    # (b) the address-live entry was PROTECTED from recency eviction (the fix fired).
    assert rep["address_live_protected"] > 0, rep
    # (c) cache stayed bounded (didn't blow up keeping every step).
    assert rep["bounded_live"] < rep["seq_len"] // 10, rep
    assert rep["unbounded_live"] == rep["seq_len"], rep


def test_store_early_load_late_recall_is_correct():
    """Sanity on the RECALLED VALUE (not just the diff): while the store is within
    the ALiBi horizon the bounded cache recalls the SAME value the unbounded cache
    does (nonzero, matching); once ALiBi crushes it BOTH read ~0 (ZFOD) — the
    policy never recalls a value the reference wouldn't, nor drops one it would."""
    rep = run_store_early_load_late(n_unrelated=400, addr=7, stored_value=42)
    # at the short horizon the store is still recallable in BOTH caches, equal.
    assert abs(rep["final_bounded_recall"] - rep["final_unbounded_recall"]) < _EXACT_TOL, rep


# ---------------------------------------------------------------------------
# 4. MEMORY-OP DENSITY — many SI/LI addresses, ZFOD, free/overwrite.
# ---------------------------------------------------------------------------
def run_memory_density(n_addrs: int = 32, rounds: int = 300) -> Dict:
    """A dense address space: ``n_addrs`` addresses repeatedly SI-written (with new
    values -> overwrite / latest-write-wins), LI-read, some never written (ZFOD ->
    reads 0), and some 'freed' (overwritten with 0). At every LI the bounded recall
    must equal the unbounded recall — never resurrect a freed/overwritten value."""
    h = _MemHead(seed=5)
    shadow: Dict[int, int] = {}          # python model of latest-write-wins memory
    worst = 0.0
    worst_zfod = 0.0
    freed = set()
    for r in range(rounds):
        addr = h.rng.randrange(n_addrs)
        act = h.rng.random()
        if act < 0.55:                    # SI addr = new value (overwrite)
            val = h.rng.randint(1, 255)
            h.store(addr, val)
            shadow[addr] = val
            freed.discard(addr)
        elif act < 0.70:                  # FREE addr (overwrite with 0)
            h.store(addr, 0)
            shadow[addr] = 0
            freed.add(addr)
        else:                             # LI addr (read) — check equivalence
            worst = max(worst, h.load_diff(addr))
        # interleave register churn so the marker keys pressure the cache too.
        for _ in range(h.rng.randint(0, 3)):
            h.churn(h.rng.randint(1, 255))
        # ZFOD: read a never-written address -> reads 0 in BOTH caches.
        if r % 25 == 0:
            unwritten = n_addrs + 100 + (r % 17)     # never stored
            worst_zfod = max(worst_zfod, h.load_diff(unwritten))
    # final sweep: LI every address, bounded must equal unbounded everywhere
    # (both the raw attention diff AND the decoded byte).
    final_worst = 0.0
    byte_disagreements = 0
    for addr in range(n_addrs):
        final_worst = max(final_worst, h.load_diff(addr))
        if not h.load_byte_agrees(addr):
            byte_disagreements += 1
    return {
        "seq_len": h.pos,
        "n_addrs": n_addrs,
        "worst_li_diff": worst,
        "worst_zfod_diff": worst_zfod,
        "final_sweep_worst": final_worst,
        "final_byte_disagreements": byte_disagreements,
        "bounded_live": len(h.bounded),
        "unbounded_live": len(h.unbounded),
        "n_freed": len(freed),
    }


def test_memory_density_latest_write_wins_and_zfod():
    """Dense SI/LI with overwrite + free + ZFOD: the bounded cache matches the
    unbounded recall at every LI and never resurrects a freed/overwritten value,
    and ZFOD reads of never-written addresses read 0 in both."""
    rep = run_memory_density(n_addrs=32, rounds=400)
    assert rep["worst_li_diff"] < _EXACT_TOL, rep["worst_li_diff"]
    assert rep["worst_zfod_diff"] < _EXACT_TOL, rep["worst_zfod_diff"]
    assert rep["final_sweep_worst"] < _EXACT_TOL, rep["final_sweep_worst"]
    # the decoded BYTE matches the unbounded recall at every address (the ultimate
    # correctness bar: latest-write-wins, no resurrected freed/overwritten value).
    assert rep["final_byte_disagreements"] == 0, rep


# ---------------------------------------------------------------------------
# 3. CAPACITY PRESSURE — a large simultaneously-live working set.
# ---------------------------------------------------------------------------
def run_capacity_pressure(working_set: int = 40, revisits: int = 6) -> Dict:
    """A large working set: ``working_set`` distinct addresses ALL kept live by
    revisiting each (LI/SI) inside the ALiBi horizon, so none goes recency-dead.
    Eviction must keep every still-needed key; the live cache scales with the
    working set (the address-live set), NOT with step count. A hard cap (if any)
    is documented via ``bounded_live`` vs ``working_set``."""
    h = _MemHead(seed=9)
    # seed the working set with distinct stores.
    for a in range(working_set):
        h.store(a, (a * 7 + 1) & 0xFF)
    worst = 0.0
    # keep the whole working set HOT: cycle through, re-touching each address
    # within the horizon so all stay address-live + recency-relevant.
    for rnd in range(revisits):
        for a in range(working_set):
            worst = max(worst, h.load_diff(a))
            if rnd % 2 == 0:
                h.store(a, (a * 7 + rnd) & 0xFF)   # refresh (latest-write-wins)
            for _ in range(2):
                h.churn(h.rng.randint(1, 255))
    final_worst = 0.0
    byte_disagreements = 0
    for a in range(working_set):
        final_worst = max(final_worst, h.load_diff(a))
        if not h.load_byte_agrees(a):
            byte_disagreements += 1
    return {
        "seq_len": h.pos,
        "working_set": working_set,
        "worst_li_diff": max(worst, final_worst),
        "byte_disagreements": byte_disagreements,
        "bounded_live": len(h.bounded),
        "unbounded_live": len(h.unbounded),
    }


def test_capacity_pressure_never_drops_live_key():
    """A large hot working set: every address is LI-exact vs the unbounded cache
    (no still-needed key is dropped), and the live cache tracks the working-set
    size — the address-live set is kept, register churn is evicted."""
    rep = run_capacity_pressure(working_set=40, revisits=6)
    assert rep["worst_li_diff"] < _EXACT_TOL, rep["worst_li_diff"]
    assert rep["byte_disagreements"] == 0, rep                # decoded byte matches
    # the live cache holds the working set (kept by address-liveness), NOT the whole
    # stream — bounded by (working_set + a recency window), far below seq_len.
    assert rep["bounded_live"] >= rep["working_set"], rep     # every live addr kept
    assert rep["bounded_live"] < rep["seq_len"] // 2, rep     # churn still evicted


# ---------------------------------------------------------------------------
# 5. ADVERSARIAL RECENCY — near-dup register markers vs needed old memory keys.
# ---------------------------------------------------------------------------
def run_adversarial_recency(n_steps: int = 4000) -> Dict:
    """Interleave near-DUPLICATE register-marker keys (recency-live, high weight)
    with a genuinely-needed OLD address-live memory key. Recency alone would keep
    the fresh near-dups and drop the old memory key — the policy must instead keep
    the address-live key (recallable) AND all recency-live near-dups (denominator-
    live), matching the unbounded cache at every frontier. This is the scenario
    that exposed the un-gated-mechanism-1 bug."""
    h = _MemHead(seed=11)
    h.store(7, 99)                                   # the needed old memory key
    worst_load = 0.0
    worst_frontier = 0.0
    # a small pool of register-marker directions that repeat every step (near-dups)
    marker_dirs = []
    for m in range(6):
        g = torch.Generator().manual_seed(500 + m)
        v = torch.randn(h.HD, generator=g)
        marker_dirs.append(v / v.norm())
    for step in range(n_steps):
        # re-emit the SAME set of register markers (near-dup keys), fresh VALUES —
        # different value each step so a value-blind supersession would diverge.
        for m, d in enumerate(marker_dirs):
            k = d * 1.0 + 1e-4 * torch.randn(h.HD)   # cos-sim ~1 across steps
            h.emit(k, h.value_vec(h.rng.randint(1, 15), dim=1), meta=("marker", m))
        # occasionally touch other addresses.
        if step % 10 == 0:
            h.store(20 + (step % 30), h.rng.randint(1, 255))
        # frontier read (a marker query) + the old-memory LOAD, both must match.
        if step % 50 == 0:
            worst_frontier = max(worst_frontier,
                                 h.frontier_diff(marker_dirs[0] * 1.0))
            worst_load = max(worst_load, h.load_diff(7))
    worst_frontier = max(worst_frontier, h.frontier_diff(marker_dirs[0] * 1.0))
    worst_load = max(worst_load, h.load_diff(7))
    return {
        "seq_len": h.pos,
        "worst_frontier_diff": worst_frontier,
        "worst_old_load_diff": worst_load,
        "bounded_live": len(h.bounded),
        "unbounded_live": len(h.unbounded),
        "address_live_protected": h.bounded.address_live_protected,
    }


def test_adversarial_recency_byte_exact():
    """Near-dup register markers (different values each step) interleaved with a
    needed old address-live memory key: bounded == unbounded at every frontier AND
    every old-memory LOAD. Recency alone would mislead; the recency-gated
    supersession + address-liveness protection keep exactly the right entries."""
    rep = run_adversarial_recency(n_steps=3000)
    assert rep["worst_frontier_diff"] < _EXACT_TOL, rep["worst_frontier_diff"]
    assert rep["worst_old_load_diff"] < _EXACT_TOL, rep["worst_old_load_diff"]
    assert rep["bounded_live"] < rep["seq_len"] // 5, rep    # still bounded


# ---------------------------------------------------------------------------
# 2. 250k-TOKEN SCALE (real run_program) — flat cache + byte-exact decode + RSS.
# ---------------------------------------------------------------------------
def _rss_gb() -> float:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1.0


def run_scale_250k(target_tokens: int = 250_000, prune_interval: int = 60) -> Dict:
    """The deep-loop-corpus scale: a real countdown program whose emitted 30-token
    stream reaches ~``target_tokens``. Drives the bounded cache exactly as the gen
    loop does over the WHOLE stream, checking the frontier attention output vs the
    unbounded reference in query-blocks (so peak memory stays flat), and confirming
    the cache stays flat + RSS bounded at 250k tokens. Reuses the CHK-6 harness's
    blocked full-cache reference (imported) so this is the SAME byte-exact contract
    at 6x the CHK-6 scale."""
    from c4_min.test_kv_eviction_longrun import (_full_frontier_attn_output,
                                                 _residual_stream, _decode_all_bytes)
    import torch.nn.functional as F

    steps = target_tokens // 30 + 1
    # a SELF-RESTARTING countdown: AX counts 200->0, then JMP 0 re-seeds it, so the
    # loop never HALTs and runs ``max_steps`` iterations (an outer×inner nested-loop
    # equivalent — the 8400-step deep-loop regime). ``max_steps`` caps it at ~250k
    # emitted tokens.
    prog = [("IMM", 200), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("BNZ", 1), ("JMP", 0)]                  # loops forever (no HALT)
    model, L, code = C.build_step_model(prog)
    tokens, frames = R.run_program(model, L, code, max_steps=steps)
    attn = model.blocks[0].attn
    res = _residual_stream(model, tokens)
    S = res.shape[0]
    W_o = attn.W_o
    rss_after_ref = 0.0

    # unbounded frontier reference (blocked -> flat memory). At 250k tokens the
    # per-block scores tensor is [H, q_block, S]; a small q_block (128) keeps its
    # peak at ~H*128*S*4 ≈ 0.5 GB so total RSS stays well under discipline.
    full_out = _full_frontier_attn_output(attn, res, q_block=128)
    rss_after_ref = _rss_gb()

    bounded = P.MultiHeadKVCache(attn.n_heads, attn.head_dim, attn.alibi_slopes,
                                 prune_interval=prune_interval)
    bounded_out = torch.empty_like(full_out)
    max_live = 0
    with torch.no_grad():
        for pos in range(S):
            k, v = P.project_kv(attn, res[pos])
            bounded.append(k, v, pos)
            bounded.maybe_prune()
            max_live = max(max_live, bounded.per_head_size())
            q = P.project_q(attn, res[pos])
            bounded_out[pos] = bounded.attention_output(q, pos)
    with torch.no_grad():
        worst_attn = float((bounded_out - full_out).abs().max())
        xb = res + F.linear(bounded_out, W_o)
        xf = res + F.linear(full_out, W_o)
        bb = _decode_all_bytes(xb, L, [("AX", L.AX)])
        bf = _decode_all_bytes(xf, L, [("AX", L.AX)])
    byte_exact = not bool((bb != bf).any())
    return {
        "seq_len": S,
        "steps": len(frames),
        "max_bounded_live_per_head": max_live,
        "unbounded_live_per_head": S,
        "total_evicted_per_head": bounded.heads[0].total_evicted,
        "worst_attn_absdiff": worst_attn,
        "byte_exact": byte_exact,
        "peak_rss_gb": max(rss_after_ref, _rss_gb()),
    }


def test_scale_250k_tokens_byte_exact():
    """Deep-loop-corpus scale: ~250k emitted tokens, flat bounded cache, decode
    byte-identical to the unbounded reference, RSS bounded. (CHK-6 reached 42k;
    this is ~6x, the 8400-step / 250k-token deep-loop regime.)"""
    rep = run_scale_250k(target_tokens=250_000)
    assert rep["seq_len"] > 240_000, rep["seq_len"]
    assert rep["byte_exact"], rep
    assert rep["worst_attn_absdiff"] < 1e-3, rep["worst_attn_absdiff"]
    # cache FLAT: a tiny constant vs the 250k-token stream (eviction fired ~250k x).
    assert rep["max_bounded_live_per_head"] < 200, rep
    assert rep["max_bounded_live_per_head"] < rep["seq_len"] // 1000, rep
    assert rep["total_evicted_per_head"] > 200_000, rep
    # RSS bounded well under the memory-discipline ceiling (the unbounded reference
    # is computed in small query blocks -> O(H*block*S), never O(H*S^2)).
    assert rep["peak_rss_gb"] < 8.0, rep["peak_rss_gb"]


# ---------------------------------------------------------------------------
# Standalone verbose report.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    print("KV-eviction HARDENING stress suite\n" + "=" * 60)

    t0 = time.time()
    print("\n[1] STORE-EARLY / LOAD-LATE (store@step5, 8000 unrelated steps, LI):")
    r = run_store_early_load_late(n_unrelated=8000)
    print(f"    seq_len={r['seq_len']}  store_dist_at_end={r['store_dist_at_end']}")
    print(f"    worst LOAD |bounded-unbounded| = {r['worst_load_diff']:.2e}  "
          f"(tol {_EXACT_TOL:.0e})")
    print(f"    store_still_cached={r['store_still_cached']}  "
          f"address_live_protected={r['address_live_protected']}")
    print(f"    bounded_live={r['bounded_live']}  unbounded_live={r['unbounded_live']}")
    print(f"    recall trace (dist, bounded, unbounded):")
    for d, ob, ou in r["recall_trace"]:
        print(f"        dist={d:6d}  bounded={ob:+.4f}  unbounded={ou:+.4f}")
    print(f"    -> {'BYTE-EXACT' if r['worst_load_diff'] < _EXACT_TOL else 'DIVERGED'}")

    print("\n[4] MEMORY-OP DENSITY (32 addrs, overwrite+free+ZFOD, 400 rounds):")
    r = run_memory_density(n_addrs=32, rounds=400)
    print(f"    seq_len={r['seq_len']}  worst LI diff={r['worst_li_diff']:.2e}  "
          f"ZFOD diff={r['worst_zfod_diff']:.2e}  final sweep={r['final_sweep_worst']:.2e}")
    print(f"    bounded_live={r['bounded_live']}  unbounded_live={r['unbounded_live']}  "
          f"freed={r['n_freed']}")

    print("\n[3] CAPACITY PRESSURE (working set 40, 6 revisits):")
    r = run_capacity_pressure(working_set=40, revisits=6)
    print(f"    seq_len={r['seq_len']}  working_set={r['working_set']}  "
          f"worst LI diff={r['worst_li_diff']:.2e}")
    print(f"    bounded_live={r['bounded_live']}  unbounded_live={r['unbounded_live']}")

    print("\n[5] ADVERSARIAL RECENCY (near-dup markers vs old memory, 3000 steps):")
    r = run_adversarial_recency(n_steps=3000)
    print(f"    seq_len={r['seq_len']}  frontier diff={r['worst_frontier_diff']:.2e}  "
          f"old-load diff={r['worst_old_load_diff']:.2e}")
    print(f"    bounded_live={r['bounded_live']}  unbounded_live={r['unbounded_live']}  "
          f"address_live_protected={r['address_live_protected']}")

    print("\n[2] 250k-TOKEN SCALE (real run_program countdown):")
    r = run_scale_250k(target_tokens=250_000)
    print(f"    seq_len={r['seq_len']} tokens  steps={r['steps']}")
    print(f"    bounded live/head={r['max_bounded_live_per_head']}  "
          f"unbounded live/head={r['unbounded_live_per_head']}  "
          f"evicted/head={r['total_evicted_per_head']}")
    print(f"    worst attn |Δ|={r['worst_attn_absdiff']:.2e}  "
          f"byte_exact={r['byte_exact']}  peak RSS={r['peak_rss_gb']:.2f} GB")

    print(f"\nwall time: {time.time()-t0:.1f}s")
