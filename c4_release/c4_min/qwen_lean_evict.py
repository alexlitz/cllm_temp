"""BOUNDED KV EVICTION on the LEAN compacted forward — memory-bounded long programs.

The lean forward (:mod:`qwen_lean_forward`) already accepts a per-layer KV cache
(``LeanQwenVM.forward(x, past, q_positions)``, the async-KV hook from the
foundation #695).  The *naive* driver (:func:`qwen_lean_forward.run_program_lean`)
re-builds a FRESH single-frame window every VM step (``past=None``), so it is O(1)
in cache but re-embeds the whole window every step.  The **persistent-cache**
driver here keeps ONE growing per-layer KV cache and appends only the new frame's
rows per step — turning the re-embed into an append — and PRUNES that cache so it
stays **BOUNDED** even when a loop runs to millions of steps.

Why the lean/qwen model needs eviction differently from the deep ALiBi backend
------------------------------------------------------------------------------
The compacted CAM (``qwen_full_vm._bake_register_cam``) was designed for a
SINGLE-frame window: within one window each register appears on exactly ONE token,
so a content-match (``CONTENT_GAIN=26``) alone selects it — "the window IS the
recency" (``qwen_full_vm`` §417-424).  RoPE recency (``RECENCY_GAIN=3``) is a mild
tie-break, NOT a strong ALiBi ``-slope*dist`` suppressor.  So on a PERSISTENT cache
a stale prior register frame's ROLE marker is a same-strength content competitor to
the current one, and the read is corrupted by even ONE stale frame (measured:
n_stale=1 -> wrong AX).  This is the honest structural fact that shapes the policy:

  * **Register / STEP_END rows are LATEST-WRITE-WINS and MUST be superseded every
    step.**  Each step re-emits all 5 register markers, so the new frame SUPERSEDES
    the previous one exactly (same ROLE keys, newer position).  Dropping the prior
    frame's rows keeps EXACTLY ONE frame in the cache -> the read is byte-identical to
    the naive fresh-window forward (verified).  This supersession is STRUCTURAL and
    per-step (it is what makes the persistent cache correct at all), not a "prune".

  * **Store rows are the LIVE HEAP (address-CAM).**  A store row is retrieved by
    ADDRESS at an arbitrary future step (``_bake_memory_cam``: exact address-match +
    RoPE latest-write-wins), so a live non-zero store is NOT recency-decayed — it is
    evicted ONLY by SUPERSESSION (a newer same-address store) or by FREEING (its
    value overwritten with 0 -> zero-value row, nil under softmax's BOS sink).  These
    two split by whether the eviction is read-tolerable, which is the honest finding:

      - SUPERSESSION (a newer same-address store) is **structural / mandatory**: the
        lean RoPE memory-CAM does NOT reliably pick the latest among two PHYSICALLY
        PRESENT same-address rows (measured: an lww load reads the WRONG value with
        both rows in cache), exactly like the register-CAM.  So a superseded row is
        dropped at store-append time (:meth:`LeanKVCache.supersede_store_addr`,
        mirroring the naive driver's Python ``store_log`` compaction) — it cannot
        wait for a periodic prune.

      - FREEING / zero-value rows ARE **read-tolerable** (nil under the BOS sink,
        verified: a freed cell can persist un-pruned and later reads are still
        byte-identical).  THESE are the periodic / async / watermark-triggered prune
        targets (BLOG_SPEC §KV-Cache-Pruning / §Memory) — they accumulate transiently
        and the prune reclaims them, so the prune is genuinely DECOUPLE-safe.

  * **BOS sink (position 0) is kept forever** — the content-free ZFOD sink row
    (``_bake_memory_cam``: an unwritten address reads 0 through the BOS logit-0 sink).

So the cache is ``BOS + one register frame (6 rows) + live-heap store rows``.  On a
register-only loop that is a CONSTANT ~7 rows forever; on a heap program it tracks
the LIVE heap footprint (grows with distinct live addresses, shrinks on free).

Decoupling the prune from the step loop
---------------------------------------
The register-frame supersession is per-step and cheap (a boolean mask over ~13
rows).  The HEAP prune (superseded-same-address / freed / zero-value store rows) is
what can be DECOUPLED: an evicted store row already LOSES the attention (a superseded
store loses via RoPE latest-write-wins; a zero-value store is nil under the BOS
sink), so pruning it now, later, or never (VRAM permitting) is BYTE-IDENTICAL — the
transient un-pruned rows change no decode.  We therefore run the heap prune on a
SEPARATE CUDA STREAM (:class:`AsyncPruner`), watermark-triggered, so the step never
blocks on it; the pruned cache is swapped in at the next step boundary.

Entry points
------------
  * :func:`run_program_lean_evict` — the persistent-cache + bounded-eviction driver.
    Byte-identical AX trace to :func:`qwen_lean_forward.run_program_lean`.  Modes:
    ``evict="off"`` (grow unbounded — the un-pruned baseline for equivalence),
    ``evict="sync"`` (prune inline every ``prune_interval`` tokens),
    ``evict="async"`` (prune on a decoupled CUDA stream, watermark-triggered).
  * :class:`LeanKVCache` — the per-layer ``(K,V,pos)`` cache + the supersession /
    heap-prune keep-mask.
  * :class:`AsyncPruner` — the decoupled-stream heap prune with a VRAM watermark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .nibble_pure_forward import SP_INIT
from .qwen_full_vm import (
    CAM_REGS, _REG_TOKEN, _snap, _address_bits, _signed_imm,
)
from .qwen_lean_forward import LeanQwenVM


# Default heap prune cadence (BLOG_SPEC §816: eviction runs every ~120 tokens).
PRUNE_INTERVAL_TOKENS = 120
# VRAM watermark (cache rows): above this the async prune is force-triggered so the
# transient un-pruned window can never blow the memory budget.
DEFAULT_WATERMARK_ROWS = 4096


# ===========================================================================
# Row tags — every cached row carries a semantic tag so the keep-mask can apply the
# right rule (register-frame supersession vs heap latest-write-wins/free).  The tags
# live in a parallel python list on the cache (NOT in the tensor), so the tensor
# forward is untouched (byte-identical to qwen_lean_forward).
# ===========================================================================
TAG_BOS = 0        # the position-0 ZFOD sink — kept forever
TAG_REG = 1        # a register/STEP_END frame row — latest-write-wins, per-step
TAG_STORE = 2      # a §Memory store row — live heap, address-CAM


@dataclass
class _RowMeta:
    tag: int
    # for TAG_STORE: the 8-bit address and whether the value is zero (freed/NULL).
    addr: Optional[int] = None
    is_zero: bool = False
    # for TAG_REG: which frame (step) it belongs to — the newest frame's rows are
    # kept, all older frames are superseded.
    frame: int = -1


# ===========================================================================
# Per-layer KV cache with the lean-forward eviction policy.
# ===========================================================================
class LeanKVCache:
    """The persistent per-layer ``(K,V,pos)`` KV cache for the lean forward, with
    the supersession + heap-prune keep-mask.

    Holds ONE ``(K,V,pos)`` tuple PER LAYER (the shape ``LeanQwenVM.forward``'s
    ``past`` expects) plus a parallel ``_RowMeta`` list (shared across layers — the
    rows are the same token positions in every layer).  ``append`` extends every
    layer; ``keep_mask`` computes the survivors; ``apply_keep`` gathers them.
    """

    def __init__(self, n_layers: int, device: torch.device):
        self.n_layers = n_layers
        self.device = device
        # per-layer (K [1,Hkv,S,HD], V [1,Hkv,S,HD], pos [1,S]); None until seeded.
        self.past: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = \
            [None] * n_layers
        self.meta: List[_RowMeta] = []
        self.total_appended = 0
        self.total_evicted = 0

    # -- size / VRAM bookkeeping --------------------------------------------
    def size(self) -> int:
        """Number of cached rows (one axis, shared across layers)."""
        return len(self.meta)

    def bytes(self) -> int:
        """Total bytes held by the K/V tensors across all layers (the VRAM footprint
        of the cache — the quantity the bound demo tracks)."""
        tot = 0
        for p in self.past:
            if p is not None:
                tot += p[0].numel() * p[0].element_size()
                tot += p[1].numel() * p[1].element_size()
        return tot

    def as_past(self):
        """The ``None``-or-list form ``LeanQwenVM.forward`` expects for ``past``."""
        if self.past is None or all(p is None for p in self.past):
            return None
        return self.past

    # -- append (called once per forward, with that forward's new rows) ------
    def append(self, new_past: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
               row_meta: List[_RowMeta]) -> None:
        """Adopt the ``new_past`` returned by ``LeanQwenVM.forward`` (which already
        concatenated the new rows onto the passed-in cache) and record the new rows'
        tags.  ``new_past`` is authoritative for the tensors; we only extend meta."""
        self.past = list(new_past)
        self.meta.extend(row_meta)
        self.total_appended += len(row_meta)

    # -- the keep-mask -------------------------------------------------------
    def keep_mask(self, heap_prune: bool = True) -> torch.Tensor:
        """Boolean ``[S]`` keep-mask under the lean-forward eviction policy.

        ALWAYS applied (structural, per-step for correctness):
          * BOS (tag BOS) — kept forever.
          * register/STEP_END rows (tag REG) — keep ONLY the NEWEST frame; every
            older frame is SUPERSEDED (same ROLE keys, newer position -> the current
            frame's content-match + RoPE recency wins, so the stale rows are dead).

        Applied when ``heap_prune`` (the periodic / async prune — decoupled-safe):
          * FREED/zero-value store rows (nil under the BOS sink) — the read-tolerable
            eviction class: a zero-value store contributes 0 to the softmax numerator
            and is indistinguishable from never-attended, so keeping it now vs later
            vs never is BYTE-IDENTICAL (this is why the prune can be DECOUPLED).
          * superseded same-address stores — belt-and-braces (they are already
            dropped STRUCTURALLY at store-append via :meth:`supersede_store_addr`,
            which correctness requires; this keeps the mask a complete statement of
            "the latest live store per address survives").
          A live non-zero store to a DISTINCT address is NEVER recency-dropped (it is
          the live heap, retrieved by address at an arbitrary future step).
        """
        S = len(self.meta)
        keep = torch.ones(S, dtype=torch.bool)
        # newest register frame index.
        newest_frame = max((m.frame for m in self.meta if m.tag == TAG_REG),
                           default=-1)
        # latest live store position per address (for supersession).
        latest_store_idx: Dict[int, int] = {}
        if heap_prune:
            for i, m in enumerate(self.meta):
                if m.tag == TAG_STORE and not m.is_zero:
                    latest_store_idx[m.addr] = i      # later index = newer store
        for i, m in enumerate(self.meta):
            if m.tag == TAG_BOS:
                continue                              # keep the sink forever
            if m.tag == TAG_REG:
                if m.frame != newest_frame:
                    keep[i] = False                   # superseded register frame
            elif m.tag == TAG_STORE and heap_prune:
                if m.is_zero:
                    keep[i] = False                   # freed / NULL store row
                elif latest_store_idx.get(m.addr, -1) != i:
                    keep[i] = False                   # superseded same-address store
        return keep

    def keep_mask_drop_reg_frames(self) -> torch.Tensor:
        """Boolean ``[S]`` keep-mask that drops EVERY register/STEP_END row, keeping
        BOS + all store rows.  Called at the TOP of a step (before the new frame is
        appended) so the incoming frame attends over ``BOS + live-heap stores`` only
        — no stale register frame.  (After the append, the incoming frame is the ONLY
        register frame in the cache, so this is exactly the per-step supersession.)"""
        keep = torch.ones(len(self.meta), dtype=torch.bool)
        for i, m in enumerate(self.meta):
            if m.tag == TAG_REG:
                keep[i] = False
        return keep

    def supersede_store_addr(self, addr: int) -> int:
        """STRUCTURAL same-address supersession: physically drop every EXISTING store
        row for ``addr`` (a newer store to the same address is about to be appended).

        This is the exact analogue of the naive driver's Python ``store_log``
        compaction (one live row per address), and it is MANDATORY for correctness on
        the lean RoPE forward: two physically-present same-address store rows corrupt
        the memory-CAM's load read (RoPE recency does not reliably pick the latest),
        so the superseded row must be gone before the next load — it cannot wait for a
        periodic prune.  (Freed/zero rows are a separate, read-tolerable class the
        periodic/async prune reclaims.)  Returns #rows dropped."""
        addr &= 0xFF
        has = any(m.tag == TAG_STORE and m.addr == addr for m in self.meta)
        if not has:
            return 0
        keep = torch.ones(len(self.meta), dtype=torch.bool)
        for i, m in enumerate(self.meta):
            if m.tag == TAG_STORE and m.addr == addr:
                keep[i] = False
        return self.apply_keep(keep)

    def apply_keep(self, keep: torch.Tensor) -> int:
        """Gather the ``keep`` survivors in place across every layer + meta.
        Returns #rows dropped."""
        S = len(self.meta)
        n_keep = int(keep.sum().item())
        if n_keep >= S:
            return 0
        idx = torch.nonzero(keep, as_tuple=False).flatten().to(self.device)
        new_past = []
        for p in self.past:
            if p is None:
                new_past.append(None)
                continue
            K, Vv, pos = p
            new_past.append((K[:, :, idx, :], Vv[:, :, idx, :], pos[:, idx]))
        self.past = new_past
        self.meta = [self.meta[j] for j in idx.tolist()]
        dropped = S - n_keep
        self.total_evicted += dropped
        return dropped


# ===========================================================================
# Async (decoupled-stream) heap prune with a VRAM watermark.
# ===========================================================================
class AsyncPruner:
    """Run the HEAP keep-mask on a SEPARATE CUDA stream so the step loop never
    blocks on the prune.

    The keep decision (which store rows are superseded / freed) is pure bookkeeping
    over the python ``meta`` list (fast), and the actual eviction is a ``gather`` on
    the cache tensors — that gather is what we place on the side stream, overlapped
    with the next step's forward on the default stream.  Because an evicted store row
    already LOSES the attention (a superseded store loses via RoPE latest-write-wins;
    a zero-value store contributes 0 under the BOS sink), pruning it a step late is
    BYTE-IDENTICAL — the transient un-pruned rows change no decode.  So we can
    pipeline it freely and only SYNC the side stream at the next step boundary
    (before the cache is read).

    Watermark: if the cache exceeds ``watermark_rows`` we force a prune regardless of
    the cadence, so the transient un-pruned window can never exceed the VRAM budget.
    """

    def __init__(self, device: torch.device, prune_interval: int,
                 watermark_rows: int):
        self.device = device
        self.prune_interval = prune_interval
        self.watermark_rows = watermark_rows
        self._is_cuda = device.type == "cuda"
        self._stream = torch.cuda.Stream(device=device) if self._is_cuda else None
        self._tokens_since_prune = 0
        self._pending = False
        self._pending_past = None
        self._pending_meta = None
        self._pending_dropped = 0
        self.n_prunes = 0

    def note_tokens(self, n: int) -> None:
        self._tokens_since_prune += n

    def should_prune(self, cache: "LeanKVCache") -> bool:
        return (self._tokens_since_prune >= self.prune_interval
                or cache.size() > self.watermark_rows)

    def launch(self, cache: "LeanKVCache") -> None:
        """Compute the heap keep-mask (host bookkeeping) and issue the gather on the
        side CUDA stream so it overlaps the next forward.  The gather RESULT is
        stashed and swapped into the cache at :meth:`sync` (the next step boundary),
        so the step's forward runs concurrently with the eviction copy."""
        keep = cache.keep_mask(heap_prune=True)
        S = len(cache.meta)
        n_keep = int(keep.sum().item())
        if n_keep >= S:
            self._tokens_since_prune = 0
            return
        idx = torch.nonzero(keep, as_tuple=False).flatten().to(self.device)
        if self._is_cuda:
            # The gather runs on the side stream; it reads the CURRENT cache tensors
            # (which the default stream is done writing — a prune is issued at a step
            # boundary after the forward completed), and produces the compacted
            # tensors.  We record the survivor meta immediately (host side).
            self._stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._stream):
                new_past = []
                for p in cache.past:
                    if p is None:
                        new_past.append(None)
                        continue
                    K, Vv, pos = p
                    new_past.append((K[:, :, idx, :].contiguous(),
                                     Vv[:, :, idx, :].contiguous(),
                                     pos[:, idx].contiguous()))
            self._pending_past = new_past
        else:
            new_past = []
            for p in cache.past:
                if p is None:
                    new_past.append(None)
                    continue
                K, Vv, pos = p
                new_past.append((K[:, :, idx, :], Vv[:, :, idx, :], pos[:, idx]))
            self._pending_past = new_past
        self._pending_meta = [cache.meta[j] for j in idx.tolist()]
        self._pending_dropped = S - n_keep
        self._pending = True
        self._tokens_since_prune = 0
        self.n_prunes += 1

    def sync(self, cache: "LeanKVCache") -> int:
        """At the next step boundary, wait for the side-stream gather to finish and
        swap the compacted cache in.  Returns #rows dropped (0 if none pending)."""
        if not self._pending:
            return 0
        if self._is_cuda:
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
        cache.past = self._pending_past
        cache.meta = self._pending_meta
        cache.total_evicted += self._pending_dropped
        dropped = self._pending_dropped
        self._pending = False
        self._pending_past = None
        self._pending_meta = None
        return dropped


# ===========================================================================
# Frame builder — append ONE step's rows to the persistent cache.
#
# Unlike the naive fresh-window builder, this does NOT re-emit the store log every
# step: each store row is appended ONCE (when its SI/SC executes) and PERSISTS in the
# cache (the live heap).  The register markers + STEP_END query ARE re-emitted every
# step (that is the register file's per-step frame).
# ===========================================================================
def _append_bos(lean: LeanQwenVM, code: List[isa.Instr]
               ) -> Tuple[torch.Tensor, torch.Tensor, List[_RowMeta]]:
    """Build the BOS sink row (position 0)."""
    L = lean.QL.L
    toks = torch.tensor([[V.BOS]], device=lean.device)
    x = lean.embed[toks].clone()
    x[0, 0, L.ONE] = 1.0
    for k, ins in enumerate(code):
        x[0, 0, L.CODE_OP[k]] = float(ins.op)
        x[0, 0, L.CODE_IMM[k]] = float(_signed_imm(ins.imm))
    return x, torch.tensor([0], device=lean.device), [_RowMeta(TAG_BOS)]


def _append_store(lean: LeanQwenVM, code: List[isa.Instr], addr: int, val: int,
                  pos: int) -> Tuple[torch.Tensor, torch.Tensor, List[_RowMeta]]:
    """Build one §Memory store row (appended once when the SI/SC executes)."""
    from .blogspec_memory import ADDR_BITS
    L = lean.QL.L
    toks = torch.tensor([[V.MEM]], device=lean.device)
    x = lean.embed[toks].clone()
    x[0, 0, L.ONE] = 1.0
    for k, ins in enumerate(code):
        x[0, 0, L.CODE_OP[k]] = float(ins.op)
        x[0, 0, L.CODE_IMM[k]] = float(_signed_imm(ins.imm))
    x[0, 0, L.IS_STORE] = 1.0
    for b, bit in enumerate(_address_bits(addr, ADDR_BITS)):
        x[0, 0, L.ADDR_BIN + b] = bit
    for j, nv in enumerate(V.nibbles_of_value(val, NIB_PER_REG)):
        x[0, 0, L.VAL_NIB + j] = float(nv)
    is_zero = (int(val) & 0xFF) == 0
    return (x, torch.tensor([pos], device=lean.device),
            [_RowMeta(TAG_STORE, addr=addr & 0xFF, is_zero=is_zero)])


def _append_reg_frame(lean: LeanQwenVM, code: List[isa.Instr], reg_state: dict,
                      load_addr: Optional[int], pos: int, frame: int
                      ) -> Tuple[torch.Tensor, torch.Tensor, List[_RowMeta]]:
    """Build the step's 5 register markers + STEP_END query row."""
    from .blogspec_memory import ADDR_BITS
    QL, L = lean.QL, lean.QL.L
    stream = [_REG_TOKEN[r] for r in CAM_REGS] + [V.STEP_END]
    toks = torch.tensor([stream], device=lean.device)
    x = lean.embed[toks].clone()
    Sn = x.shape[1]
    for i in range(Sn):
        x[0, i, L.ONE] = 1.0
        for k, ins in enumerate(code):
            x[0, i, L.CODE_OP[k]] = float(ins.op)
            x[0, i, L.CODE_IMM[k]] = float(_signed_imm(ins.imm))
    for hh, reg in enumerate(CAM_REGS):
        p = hh
        for j, nv in enumerate(V.nibbles_of_value(reg_state[reg], NIB_PER_REG)):
            x[0, p, QL.TOK_NIB + j] = float(nv)
        x[0, p, QL.ROLE + hh] = 1.0
        x[0, p, QL.IS_TOK] = 1.0
    for hh in range(len(CAM_REGS)):
        x[0, -1, QL.ROLE + hh] = 1.0
    if lean.subset.memory and load_addr is not None:
        x[0, -1, L.IS_LOAD] = 1.0
        for b, bit in enumerate(_address_bits(load_addr, ADDR_BITS)):
            x[0, -1, L.QRY_BIN + b] = float(bit)
    meta = [_RowMeta(TAG_REG, frame=frame) for _ in range(Sn)]
    positions = torch.arange(pos, pos + Sn, device=lean.device)
    return x, positions, meta


# ===========================================================================
# The persistent-cache + bounded-eviction driver.
# ===========================================================================
@dataclass
class EvictStats:
    exact: bool
    steps: int
    ax_trace: List[int]
    ref_trace: List[int]
    max_cache_rows: int
    final_cache_rows: int
    max_cache_bytes: int
    total_evicted: int
    n_prunes: int
    cache_size_trace: List[int] = field(default_factory=list)
    cache_bytes_trace: List[int] = field(default_factory=list)


def run_program_lean_evict(lean: LeanQwenVM, code: List[isa.Instr], *,
                           max_steps: int = 64, evict: str = "async",
                           prune_interval: int = PRUNE_INTERVAL_TOKENS,
                           watermark_rows: int = DEFAULT_WATERMARK_ROWS,
                           sample_every: int = 1,
                           verbose: bool = False) -> EvictStats:
    """Execute ``code`` on the lean fused VM with a PERSISTENT per-layer KV cache and
    BOUNDED eviction — the memory-bounded long-program driver.

    Decodes the SAME AX trace as :func:`qwen_lean_forward.run_program_lean` (the
    naive fresh-window driver): the per-step register-frame supersession keeps
    EXACTLY the current frame (byte-identical read to a fresh window), and the heap
    prune only drops store rows that already lose the attention.

    ``evict``:
      * ``"off"``  — grow the cache unbounded (only the mandatory register
        supersession; NO heap prune).  The un-pruned baseline for the equivalence
        proof and the "cache would grow linearly" demo.
      * ``"sync"`` — prune the heap inline every ``prune_interval`` tokens
        (blocking).
      * ``"async"`` — prune the heap on a DECOUPLED CUDA stream, watermark-triggered
        (``watermark_rows``); the step never blocks on the prune.

    Returns :class:`EvictStats` (byte-exactness + the cache-size / VRAM trace).
    """
    subset = lean.subset
    L = lean.QL.L
    # the golden reference trace (for the ``exact`` field only); a never-halting spin
    # loop or an out-of-slice op has no finite reference — the byte-identity gate is
    # vs the naive lean driver, not vs isa.interpret, so a missing ref is fine.
    try:
        # match the oracle's step budget to the driver's so a long loop is not
        # truncated at the default 256-step cap (#691 BUG 2).
        ref_trace = isa.interpret(code, max_steps=max_steps)
    except Exception:
        ref_trace = []

    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    ax_trace: List[int] = []
    cur_pc = 0
    call_stack: List[Tuple[Optional[int], Optional[int]]] = []

    cache = LeanKVCache(lean.n_layers, lean.device)
    pruner = AsyncPruner(lean.device, prune_interval, watermark_rows) \
        if evict == "async" else None

    # seed the cache with the BOS sink (kept forever).
    xb, pb, mb = _append_bos(lean, code)
    with torch.no_grad():
        _, new_past = lean.forward(xb, past=None, q_positions=pb)
    cache.append(new_past, mb)
    next_pos = 1
    tokens_since_prune = 0

    max_rows = cache.size()
    max_bytes = cache.bytes()
    size_trace: List[int] = []
    bytes_trace: List[int] = []
    frame_idx = 0

    for step in range(max_steps):
        # if an async prune is pending, swap it in at this step boundary (before the
        # cache is read) — the step's forward overlapped the eviction copy.
        if pruner is not None:
            pruner.sync(cache)

        # -- MANDATORY register-frame supersession (BEFORE the decode) ----------
        # Drop the PRIOR step's register/STEP_END rows so the new frame attends over
        # EXACTLY ``BOS + live-heap stores`` (no stale frame).  This must happen
        # BEFORE the forward: the compacted CAM's content-match ties across frames
        # (RoPE recency is too weak to suppress a stale ROLE marker), so a single
        # stale frame in the cache corrupts the register read (measured n_stale=1 →
        # wrong AX).  Removing it first makes the read byte-identical to the naive
        # fresh-window forward.  This is structural (correctness), not a heap prune.
        cache.apply_keep(cache.keep_mask_drop_reg_frames())

        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF

        # append THIS step's register frame (5 markers + STEP_END query) and decode.
        x, positions, meta = _append_reg_frame(
            lean, code, reg_state, load_addr, next_pos, frame_idx)
        with torch.no_grad():
            hidden, new_past = lean.forward(x, past=cache.as_past(),
                                            q_positions=positions)
        cache.append(new_past, meta)
        next_pos += x.shape[1]
        tokens_since_prune += x.shape[1]
        frame_idx += 1

        state = hidden[0, -1]
        pc = _snap(state[L.PC_VAL])
        ax = _snap(state[L.AX_VAL]) & 0xFF
        sp = _snap(state[L.SP_VAL])
        bp = _snap(state[L.BP_VAL])
        stk = _snap(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5

        # calling convention (mirrors run_program_lean).
        if op == isa.JSR:
            call_stack.append((cur_pc + 1, bp))
        elif op == isa.ENT:
            call_stack.append((None, prev["BP"]))
        elif op == isa.LEV:
            saved_bp = ret_pc = None
            if call_stack:
                _, saved_bp = call_stack.pop()
            if call_stack:
                ret_pc, _ = call_stack.pop()
            if saved_bp is not None:
                bp = saved_bp
            if ret_pc is not None:
                pc = ret_pc
        elif subset.memory and op in (isa.SI, isa.SC):
            store_addr = _snap(state[L.STK_VAL]) & 0xFF
            store_val = ax if op == isa.SI else (ax & 0xFF)
            # STRUCTURAL same-address supersession (mirrors the naive driver's Python
            # ``store_log`` latest-write-wins compaction).  A newer store to the SAME
            # address makes the older one dead — and, crucially, the lean RoPE
            # memory-CAM (like the register-CAM) does NOT reliably pick the latest
            # among two PHYSICALLY-PRESENT same-address store rows (RoPE recency is a
            # mild tie-break, not a strong suppressor: measured lww load reads the
            # WRONG value with both rows in cache).  So the superseded row must be
            # dropped BEFORE the next load that could read this address — this is
            # correctness, not just memory.  (Freed/zero rows are a SEPARATE,
            # read-tolerable class handled by the periodic/async prune below.)
            cache.supersede_store_addr(store_addr)
            xs, ps, ms = _append_store(lean, code, store_addr, store_val, next_pos)
            with torch.no_grad():
                _, new_past = lean.forward(xs, past=cache.as_past(), q_positions=ps)
            cache.append(new_past, ms)
            next_pos += 1
            tokens_since_prune += 1

        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": stk}
        ax_trace.append(ax)

        # (The prior frame is superseded at the TOP of the NEXT step, before its
        # decode — see the mandatory supersession above.  This keeps the current
        # frame + its store row live for a same-step load's read.)

        # -- the HEAP prune (decoupled-safe: superseded / freed / zero stores) --
        if evict == "sync" and tokens_since_prune >= prune_interval:
            cache.apply_keep(cache.keep_mask(heap_prune=True))
            tokens_since_prune = 0
        elif evict == "async":
            pruner.note_tokens(x.shape[1] + (1 if op in (isa.SI, isa.SC) else 0))
            if pruner.should_prune(cache):
                pruner.launch(cache)          # runs on the side stream, overlapped
                tokens_since_prune = 0

        if step % sample_every == 0:
            size_trace.append(cache.size())
            bytes_trace.append(cache.bytes())
        max_rows = max(max_rows, cache.size())
        max_bytes = max(max_bytes, cache.bytes())

        if verbose and step < 8:
            print(f"  step {step} op={isa.NAMES.get(op, op):5s} -> ax={ax} pc={pc} "
                  f"cache_rows={cache.size()}")
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(code):
            break

    # drain any pending async prune.
    if pruner is not None:
        pruner.sync(cache)

    n_prunes = pruner.n_prunes if pruner is not None else 0
    return EvictStats(
        exact=(ax_trace == ref_trace), steps=len(ax_trace), ax_trace=ax_trace,
        ref_trace=ref_trace, max_cache_rows=max_rows, final_cache_rows=cache.size(),
        max_cache_bytes=max_bytes, total_evicted=cache.total_evicted,
        n_prunes=n_prunes, cache_size_trace=size_trace, cache_bytes_trace=bytes_trace)
