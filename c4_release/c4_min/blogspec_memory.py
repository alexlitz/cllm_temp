"""Spec-faithful softmax1 KV memory subsystem (BLOG_SPEC §Memory).

This module builds the program memory the blog post specifies, *on the nibble
foundation* (``blogspec_layout`` / ``blogspec_model``), and implements the four
memory opcodes LI / LC / SI / SC. It is a direct transcription of BLOG_SPEC
§Memory (lines 408-412, 687-691):

    "The program memory works by having a store instruction which attends to its
     registers to get the address and value it is storing, it represents the
     address in binary and sets the key to +scale for ones, -scale for zeros.
     Which allows us to retrieve this position in memory by attending with a
     query identical to the key. The scale should be large enough that other
     keys differing in one or more position have trivial weights, the positional
     bias should be such that earlier writes to the position have trivial
     weight ... [ALiBi] strictly prioritized exact address match and subject to
     that having priority to the most recently written store."

    "all addresses are helpfully initialized to zero by softmax1."   (ZFOD)

    "Overwrite the value with zero. ... with softmax1 zero is something of a
     default value for attention, so we get ZFOD for free! ... an eviction
     policy that recognizes that ... evicting both the old value and the zero
     overwriting it would have nil effect."                          (free)

The mechanism (one dedicated softmax1 + ALiBi attention head)
=============================================================
Every VM step emits a 30-token frame that ends with a ``MEM`` marker followed by
``addr(4) + val(4)`` little-endian bytes (``blogspec_vocab.build_step_frame``).
A store instruction (SI/SC) emits the real ``addr``/``val`` in that slot; every
other step emits a NULL memory write (``addr=val=0``) — exactly the spec's "we
could avoid the NULL writes but it keeps it simple" (§461). So the KV cache of
the memory head, over the whole token stream, is precisely the append-only log
of memory writes.

The head is baked once (position-independent weights). Let ``n = ADDR_BITS`` and
``EFF`` be the per-bit match contribution *after* the model's ``head_dim**-0.5``
score scale (see the constants block for the numerics):

  * **K (key) — the binary address, ±key.**  A store position carries the written
    address as 32 per-bit dims (``ADDR_BIN``, bit=0/1). ``W_k`` maps bit dim ``b``
    -> key channel ``b`` as ``2·smag·bit − smag·ONE``, i.e. ``+smag`` for a 1-bit,
    ``-smag`` for a 0-bit (the spec's exact ±scale key, with ``smag²·head_scale =
    EFF``).

  * **Q (query) — identical to the key for the queried address.**  A LI/LC load
    position carries the address to load as the same 32 per-bit dims (``QRY_BIN``)
    with the *identical* ``±smag`` map. Per bit Q·K = ``+EFF`` (agree) / ``-EFF``
    (disagree) after ×head_scale, so an exact address scores ``+n·EFF`` and ``k``
    differing bits cost ``2k·EFF`` — one differing bit already makes the row lose
    (§410 "differing in one or more position have trivial weights").

  * **ZFOD bias + role gate — the two extra CAM channels.**  A store-only channel
    subtracts a constant ``BIAS = (n-1)·EFF`` from every load↔store pair, so an
    exact match nets ``+EFF`` (positive) and a 1-bit-off row nets ``-EFF``
    (negative). Because mismatches are *negative*, softmax1's ``+1`` sink wins
    when no exact row exists ⇒ ZFOD (§412). A second store-role channel drives
    NON-store rows (BOS/load rows, whose ``ADDR_BIN=0`` would otherwise read as a
    store of address ``0x0``) to a large negative ``-PEN`` so only real stores are
    ever candidates.

  * **V (value) — the stored nibbles.**  ``W_v`` copies the store row's value
    nibbles; ``W_o`` writes them into the load target (AX) nibble band, whence the
    LM byte-head decodes the loaded value.

  * **ALiBi (recency) — latest-write-wins.**  The head's ``MEM_ALIBI_SLOPE`` biases
    earlier positions down by ``-slope·dist``; among exact-address matches the
    most recent store wins. ``EFF`` is large enough that a far-back exact match
    still reads weight ~1 (the address match dominates the ALiBi spread, §410), so
    recency only breaks ties among *equal* addresses.

  * **softmax1 (ZFOD) — unwritten reads 0.**  If no store matches, every candidate
    score is ``≤ -EFF`` (or ``-PEN``) and the ``+1`` sink dominates: the head
    contributes ~0, so the AX band keeps its ZFOD 0. Overwriting a value with 0
    (free) writes a matching entry whose value nibbles are 0 — the newest match
    returns 0 (§691).

The whole thing runs on ``blogspec_model`` (real softmax1 + ALiBi) via
``model.forward``; the proofs below feed genuine token streams and read the
result out of the AX nibble band with the LM byte-head — no python dict.

Interface for the stack agents
==============================
``KVMemory`` (built by :func:`build_memory_model`) exposes:
    ``.store(addr, value, *, char=False)``  -> append a SI/SC store token block
    ``.load(addr, *, char=False)``          -> append a LI/LC load, return value
Both drive the real transformer forward. See :class:`KVMemory` and the module
doc block ``INTERFACE`` for the residual-band contract the stack builder uses.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from . import blogspec_vocab as V
from .blogspec_layout import NibbleLayout, NIB_PER_REG
from .blogspec_model import Transformer


# ---------------------------------------------------------------------------
# Constants (BLOG_SPEC §Memory: "scale large enough that other keys differing in
# one or more position have trivial weights ... sum of scale^2 large enough that
# the 1 in softmax1 is trivial even with extremely significant positional bias").
# ---------------------------------------------------------------------------
# The design decouples ADDRESS MATCH from RECENCY exactly as §410 requires. Let
# ``EFF`` be the *post-model-scale* per-bit match contribution and ``n=ADDR_BITS``:
#
#   * per-bit query/key are ``±sqrt(EFF/head_scale)`` so an agreeing bit scores
#     ``+EFF`` and a disagreeing bit ``-EFF`` (after the model's ``head_dim**-0.5``
#     scale). An exact address is ``+n·EFF`` raw; ``k`` differing bits cost
#     ``2k·EFF`` — the spec's ±scale binary key.
#   * a constant ``BIAS = (n-1)·EFF`` is subtracted from every load↔store pair
#     (a dedicated store-flag channel), so an exact match nets ``+EFF`` and a
#     1-bit-off row nets ``-EFF``: matches stay positive, EVERY mismatch is
#     negative ⇒ the softmax1 ``+1`` sink gives ZFOD when no exact row exists.
#   * ``EFF`` is large so ``+EFF`` still dominates the ALiBi penalty ``slope·dist``
#     for a far-back store (the spec's "sum of scale² large enough ... even with
#     extremely significant positional bias"), while the small ``MEM_ALIBI_SLOPE``
#     only has to separate stores to the SAME address, which are ≥ one 30-token
#     frame apart, so ``slope·30`` decisively favours the newer (latest-write-wins).
ADDR_BITS = 32           # 4-byte aligned 32-bit addresses (§Memory)
EFF = 500000.0           # post-scale per-bit match contribution (huge, §410); raised
                         # 40k->500k (deep-recursion fix e52ab0c5) so the final LEV of a
                         # deep recursion (rec_fib(12) store->load gap 250,839 tokens)
                         # reads weight ~1 instead of fading to ZFOD 0.
BIAS = (ADDR_BITS - 1) * EFF   # constant subtracted from load↔store pairs (ZFOD)
# Recency slope. Two constraints (§410 decoupling): (a) ``slope·Δ`` must decisively
# prefer the newer of two same-address stores — the driver spaces store rows one
# 30-token FRAME apart, so ``slope·30 = 30`` gives an exp(30)≈1e13× preference (far
# past any byte-decode argmax margin); (b) ``slope·dist`` must never overwhelm an
# exact match's ``+EFF``, i.e. a far store still reads weight ~1 while
# ``dist < EFF/slope = 500000`` tokens — the RECALL HORIZON.
#
# DEEP-LOOP FIX (CHK-1): the horizon must EXCEED the longest store→load recall gap
# a program produces.  Measured across the deep-loop corpus (gcd, loop_sum, nested,
# rec_*), the ordinary-loop max gap is ~700 tokens and most recursion needs a few
# thousand; the previous ``EFF=4000 slope=6`` horizon of only 667 tokens sat BELOW
# the loop_sum (669) / gcd (699) recall gaps, so the newest exact-address store fell
# just past the horizon (``EFF - slope·dist < 0``) and softmax1's +1 sink won — the
# load faded to ZFOD 0 (the LI→0 deep-loop divergence).
#
# DEEP-RECURSION FIX (CHK-1, 2026-07-18): the FINAL LEV of a deep-recursion program
# (rec_fib, rec_sum) returns from ``main``'s OUTERMOST call — it recalls the saved
# BP + return-PC that were stored at the VERY START of the run (frame ~5/6), spanning
# nearly the whole program.  Measured max store→load recall gap: rec_sum(14) = 8649,
# rec_fib(9) = 58599, rec_fib(10) = 95319, rec_fib(11) = 154719, rec_fib(12) = 250839
# tokens (the deepest in the corpus).  The store SURVIVES eviction (the CAM head's
# mechanism-3 recency horizon is |k|²·scale/slope ≈ 130M tokens, far past any gap),
# but the ATTENTION READ at recall time gave the single far-back exact-address store
# a NEGATIVE score (``EFF - slope·dist = 40000 - 250839 < 0``), so softmax1's +1 sink
# won and the LEV recalled ZFOD 0 — the saved BP/return-PC collapsed (bp=0, pc=0) and
# the frame desynced.  This is a pure ATTENTION recall-horizon limit, NOT an eviction
# or truncation bug (proven: raising EFF alone flips fib(9) 34→PASS).  Raising EFF
# 40000→500000 (slope unchanged at 1.0) lifts the horizon 40000→500000 tokens — 2×
# the deepest corpus gap (250839) — while KEEPING the exp(30) latest-write-wins
# recency, exactly the spec's "sum of scale² large enough ... even with extremely
# significant positional bias" (§410).  NOTE: the horizon is finite, so a recursion
# deeper than ~500k tokens would still fade — a fundamental property of the
# stack-in-KV-memory-with-ALiBi design (§410 decouples match from recency but a
# finite match magnitude can always be out-run by a far enough store); 500000 covers
# the entire 1096 corpus with margin.
MEM_ALIBI_SLOPE = 1.0

# ---------------------------------------------------------------------------
# GATE-CHANNEL penalty scale (store-role + load/pop/lev-enable channels).
# ---------------------------------------------------------------------------
# The CAM's ROLE-GATE channels drive an INELIGIBLE candidate/query to a large
# negative score so softmax1's +1 sink wins.  This penalty MUST stay huge: it has
# to dominate the WORST-CASE partial address match of a load query against a
# zero-address (non-store / query-row) key, which is O(ADDR_BITS·EFF) — so PEN_GATE
# stays at ``100·ADDR_BITS·EFF``.
#
# The bug the frame-pointer read-back exposed was NOT this magnitude but the gate's
# LINEAR dependence on the query FLAG.  The load/pop/lev-ENABLE channel contributes
# ``-PEN_GATE·(1 - IS_LOAD)`` to the score; a query FLAG carrying a sub-permille
# opcode-decode RESIDUE (``IS_LOAD = 1-ε`` at a large PC, where the CODE_OP nibble
# ramp leaves ~1e-3 of slack) then contributes ``-PEN_GATE·ε`` — which, with
# PEN_GATE huge, swamps the exact-address match ``+EFF`` and sinks the load to ZFOD
# 0 (the malloc_printf memset frame-pointer read-back; the deep-frame ceiling on
# mandelbrot ~136 / self-emulation ~296, where PC/SP grow large).  The FIX is to
# THRESHOLD the query flag to a clean 0/1 at authoring time — ``_flag_from_ops``
# (nibble_pure_forward) now writes a saturating relu-ramp STEP so ``IS_LOAD /
# IS_POP / IS_LEV = 1.0`` exactly for any ``g > 0.6`` (residue-immune) — so the
# huge gate can stay huge without amplifying flag residue.
PEN_GATE = 100.0 * ADDR_BITS * EFF


class MemoryLayout(NibbleLayout):
    """``NibbleLayout`` + the bands the KV-memory head needs.

    Extra bands (all baked, position-independent):
      ``ADDR_BIN`` (32)  — per-bit (0/1) expansion of the MEM-slot store address
                           (the KEY source).  Set on MEM-marker store positions.
      ``QRY_BIN``  (32)  — per-bit (0/1) expansion of the AX address to load
                           (the QUERY source). Set on LI/LC load positions.
      ``VAL_NIB``  (16)  — the store value's nibbles (the VALUE source), copied
                           into AX on a matching load.
      ``IS_STORE`` (1)   — 1.0 on a real memory-store position (SI/SC), keys the
                           store-flag channel so only stores become KV entries.
      ``IS_LOAD``  (1)   — 1.0 on a memory-load position (LI/LC), queries.
      ``IS_CHAR``  (1)   — 1.0 for the char (byte) variants LC/SC (informational;
                           the byte/char paths share the same KV head — a char is
                           just a 1-byte value, still 4-byte aligned per §Memory).
    """

    def __init__(self, n_heads: int = 4):
        # allocate the base foundation bands first, then extend before padding.
        # Re-run the base allocation logic but keep going.
        self._off = 0
        self._names = {}

        self.PC = self._band("PC", NIB_PER_REG)
        self.AX = self._band("AX", NIB_PER_REG)
        self.SP = self._band("SP", NIB_PER_REG)
        self.BP = self._band("BP", NIB_PER_REG)
        self.STACK0 = self._band("STACK0", NIB_PER_REG)

        self.CUR_NIB = self._band("CUR_NIB", NIB_PER_REG)
        from .blogspec_layout import NUM_CTX
        self.CTX = self._band("CTX", NUM_CTX)
        self.BYTE_OFS = self._scalar("BYTE_OFS")
        self.ONE = self._scalar("ONE")

        # --- KV-memory bands ---------------------------------------------
        self.ADDR_BIN = self._band("ADDR_BIN", ADDR_BITS)  # store-addr bits (KEY)
        self.QRY_BIN = self._band("QRY_BIN", ADDR_BITS)    # load-addr bits (QUERY)
        self.VAL_NIB = self._band("VAL_NIB", NIB_PER_REG)  # store value nibbles
        self.IS_STORE = self._scalar("IS_STORE")
        self.IS_LOAD = self._scalar("IS_LOAD")
        self.IS_CHAR = self._scalar("IS_CHAR")

        # RMSNorm compensator lane (only carries K when norm="rmsnorm"; stays 0
        # otherwise, so it is inert for the default norm-free path). A single
        # large-K lane whose energy dominates ``mean(x^2)`` makes RMSNorm an
        # identity on the real dims (``blogspec_model.Transformer
        # .set_norm_compensator`` / ``qwen_embed`` §3).
        self.NORM_COMP = self._scalar("NORM_COMP")

        while self._off % n_heads != 0:
            self._scalar(f"_pad{self._off}")
        self.D = self._off


# ---------------------------------------------------------------------------
# Binary-address expansion: value -> 32 per-bit dims (0.0 / 1.0).
# ---------------------------------------------------------------------------
def address_bits(addr: int) -> List[float]:
    """The 32 little-endian bits of ``addr`` as floats (0.0 / 1.0)."""
    return [float((addr >> b) & 1) for b in range(ADDR_BITS)]


# ---------------------------------------------------------------------------
# The KV memory head (one softmax1 + ALiBi attention head, baked once).
# ---------------------------------------------------------------------------
def bake_memory_head(attn, L: MemoryLayout, head: int = 0) -> None:
    """Bake attention head ``head`` as the spec's §Memory KV head.

    Implements the binary-address CAM with the decoupled ``EFF``/``BIAS``/slope
    numerics documented at the top of the module. Let ``hs = head_dim**-0.5`` be
    the model's per-head score scale and ``n = ADDR_BITS``.

    Channels of head ``head`` (the head owns dims ``[head*head_dim : ...]``; the
    CAM lives on the head's LOCAL channels 0..n+1):

      * **address channels 0..n-1** — per-bit ``±sqrt(EFF/hs)``. K reads the store
        address bits ``ADDR_BIN``, Q reads the load address bits ``QRY_BIN``; both
        map bit ``b`` -> ``±smag`` via ``2·smag·bit - smag·ONE`` (so ``+smag`` for
        a 1-bit, ``-smag`` for a 0-bit). Q·K per bit = ``+EFF`` (agree) / ``-EFF``
        (disagree) after ``×hs``. Exact address ⇒ ``+n·EFF``; ``k`` differing bits
        ⇒ ``+(n-2k)·EFF`` — every differing bit costs ``2·EFF`` (§410).

      * **channel n — the ZFOD bias (store-only).** A load queries this channel
        ``-qb`` (``IS_LOAD``); a store keys it ``+kb`` (``IS_STORE``) with
        ``qb·kb·hs = BIAS = (n-1)·EFF``. So a load↔store pair carries a constant
        ``-BIAS`` — turning the exact match into net ``+EFF`` and a 1-bit-off row
        into net ``-EFF`` (mismatch < 0 ⇒ softmax1 sink gives ZFOD, §412). A
        load↔non-store pair gets 0 here (``IS_STORE=0``).

      * **channel n+1 — the store-role penalty.** Non-store rows (BOS anchor,
        load-query rows) carry ``ADDR_BIN=0`` and would otherwise masquerade as a
        store of address ``0x0``. Keying ``-p·ONE + p·IS_STORE`` makes a store
        row score ``0`` and a NON-store row ``-p`` here; a load queries ``+p`` so
        a non-store candidate is driven to ``-p²·hs = -PEN`` (far below any CAM
        score) and can never win — only real stores are memory entries.

      * **value relay.** V copies the store row's ``VAL_NIB`` nibbles; O writes
        them into the AX nibble band, where the LM byte-head decodes the loaded
        value. Free = value-0 store ⇒ nibbles 0 ⇒ relayed value 0 (§687-691).

    The head's ALiBi slope is set to ``MEM_ALIBI_SLOPE`` (small): recency only
    breaks ties among exact-address matches (§410 latest-write-wins).
    """
    hs = attn.scale                              # head_dim**-0.5 (model score scale)
    smag = (EFF / hs) ** 0.5                     # per-bit key/query magnitude
    # bias so a load↔store pair contributes exactly -BIAS after ×hs.
    qb = (BIAS / hs) ** 0.5
    kb = (BIAS / hs) ** 0.5
    # store-role penalty: non-store rows driven to -PEN_GATE (>> any CAM score) so a
    # BOS/load row (ADDR_BIN=0, i.e. "address 0") can never win a load.  PEN_GATE is
    # the shared role-gate scale (see module-top note); the query flags feeding these
    # channels are THRESHOLDED clean (``_flag_from_ops`` step) so the huge penalty
    # cannot amplify a flag residue (frame-pointer read-back / deep-frame fix).
    PEN = PEN_GATE
    p = (PEN / hs) ** 0.5

    # recency slope for this head only (small; §410 latest-write-wins).
    attn.alibi_slopes[head] = MEM_ALIBI_SLOPE

    # head ``head`` owns the local channel range [head*head_dim, ...); write the
    # CAM on this head's LOCAL channels so multi-head models stay isolated.
    HD = attn.head_dim
    base = head * HD

    # --- address CAM: symmetric ±smag key & query -----------------------
    for b in range(ADDR_BITS):
        # KEY <- store address bits (±smag): +smag for a 1-bit, -smag for 0.
        attn.W_k[base + b, L.ADDR_BIN + b] = 2.0 * smag
        attn.W_k[base + b, L.ONE] = -smag
        # QUERY <- load address bits (±smag), identical to the key (§410).
        attn.W_q[base + b, L.QRY_BIN + b] = 2.0 * smag
        attn.W_q[base + b, L.ONE] = -smag

    # --- channel n: ZFOD bias, store-only (load -qb ; store +kb) ---------
    cB = base + ADDR_BITS
    attn.W_q[cB, L.IS_LOAD] = -qb
    attn.W_k[cB, L.IS_STORE] = kb

    # --- channel n+1: store-role penalty (load +p ; key -p*ONE + p*STORE) -
    # non-store rows key -p here (via ONE), stores key 0 (ONE cancels STORE);
    # a load queries +p so non-stores get -p^2*hs = -PEN and can never win.
    cR = base + ADDR_BITS + 1
    attn.W_q[cR, L.IS_LOAD] = p
    attn.W_k[cR, L.ONE] = -p
    attn.W_k[cR, L.IS_STORE] = p

    # --- VALUE: the store value nibbles -> AX band ----------------------
    # V/O use this head's LOCAL value slots (base + 0..NIB_PER_REG-1).
    for j in range(NIB_PER_REG):
        attn.W_v[base + j, L.VAL_NIB + j] = 1.0
        attn.W_o[L.AX + j, base + j] = 1.0


NORM_K = 1.0e6     # RMSNorm compensator constant (>> residual & CAM mags; the
                   # KV head's ±smag CAM score ~n·EFF is huge, so the compensator
                   # must dominate strongly enough that per-row rms differences
                   # (VAL_NIB nibble energy varies per store) don't perturb the
                   # score past the tiny recency tie-break; qwen_embed §3).


def build_memory_model(n_heads: int = 4, positional: str = "alibi",
                       norm: str = "none", sink: str = "softmax1",
                       norm_K: float = NORM_K):
    """Bake a ``blogspec_model`` transformer with the KV-memory head on block 0.

    Returns ``(model, L)``. The remaining heads/blocks are identity; block 0
    head 0 is the real softmax1+ALiBi §Memory head. The embedding carries the
    ONE lane (needed for the ±smag key bias). Store/load positions are set by
    :class:`KVMemory` overlaying the residual bands, then run through
    ``model.forward``.

    ``positional`` / ``norm`` / ``sink`` are the three ``blogspec_model``
    architectural toggles (default = ALiBi / norm-free / softmax1, byte-identical
    to the historical head). Under ``norm="rmsnorm"`` a NORM_COMP lane holding
    ``norm_K`` is baked onto every embedding row and every RMSNorm ``weight`` set
    to ``norm_K/√dim`` so RMSNorm is an identity on the real dims (the
    compensator-lane trick).
    """
    L = MemoryLayout(n_heads=n_heads)
    model = Transformer(dim=L.D, n_heads=n_heads, hidden=max(8, NIB_PER_REG),
                        n_blocks=1, vocab=V.VOCAB, max_seq_len=8192,
                        positional=positional, norm=norm, sink=sink)
    with torch.no_grad():
        E = torch.zeros(V.VOCAB, L.D)
        E[:, L.ONE] = 1.0
        model.embed.copy_(E)
        for blk in model.blocks:
            for p in (blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o):
                p.zero_()
        bake_memory_head(model.blocks[0].attn, L, head=0)
    # RMSNorm compensator: identity gamma + K on every row's NORM_COMP lane.
    model.set_norm_compensator(L.NORM_COMP, K=norm_K)
    return model, L


# ---------------------------------------------------------------------------
# The KV memory interface: build a token stream of store/load blocks and run the
# REAL model.forward over it, reading loaded values from the AX nibble band.
# ---------------------------------------------------------------------------
class KVMemory:
    """A softmax1 KV memory over the real ``blogspec_model`` transformer.

    Usage::

        mem = KVMemory()
        mem.store(0x200, 42)          # SI: *0x200 = 42
        assert mem.load(0x200) == 42  # LI: AX = *0x200

    Each ``store``/``load`` appends one position to an internal token stream and
    re-runs ``model.forward`` over the whole stream (the KV cache is recomputed
    from scratch each call — a tiny model, this is exact and simple). The store
    positions carry the ±smag binary-address key; the load positions carry the
    identical query. ZFOD, latest-write-wins and free (zero-overwrite) all fall
    out of softmax1 + ALiBi as the spec describes — no python memory dict.
    """

    def __init__(self, model=None, L: Optional[MemoryLayout] = None,
                 n_heads: int = 4, positional: str = "alibi",
                 norm: str = "none", sink: str = "softmax1"):
        if model is None:
            model, L = build_memory_model(n_heads=n_heads, positional=positional,
                                          norm=norm, sink=sink)
        self.model = model
        self.L = L
        # Each entry is a (token_id, residual_overlay) pair. The residual overlay
        # is a dict {dim: value} applied on top of the token embedding, encoding
        # the store/load bands for that position (what the ingest FFN would set
        # from the MEM/addr/val tokens — here we set it directly, the ingest of
        # the frame bytes is proven separately in the foundation).
        self._stream: List[tuple] = [(V.BOS, {})]

    # -- position builders --------------------------------------------------
    def _store_overlay(self, addr: int, value: int, char: bool) -> dict:
        L = self.L
        ov = {L.IS_STORE: 1.0, L.IS_CHAR: 1.0 if char else 0.0}
        for b, bit in enumerate(address_bits(addr)):
            ov[L.ADDR_BIN + b] = bit
        for j, nv in enumerate(V.nibbles_of_value(value & 0xFFFFFFFF, NIB_PER_REG)):
            ov[L.VAL_NIB + j] = float(nv)
        return ov

    def _load_overlay(self, addr: int, char: bool) -> dict:
        L = self.L
        ov = {L.IS_LOAD: 1.0, L.IS_CHAR: 1.0 if char else 0.0}
        for b, bit in enumerate(address_bits(addr)):
            ov[L.QRY_BIN + b] = bit
        return ov

    def _forward(self) -> torch.Tensor:
        """Embed the stream, apply the per-position band overlays, run forward."""
        L, model = self.L, self.model
        toks = torch.tensor([[tid for tid, _ in self._stream]])
        with torch.no_grad():
            x = model.embed[toks].clone()             # [1, S, D]
            for i, (_, ov) in enumerate(self._stream):
                for dim, val in ov.items():
                    x[0, i, dim] = val
            for blk in model.blocks:
                x = blk(x)
        return x[0]                                   # [S, D]

    # -- public API ---------------------------------------------------------
    def store(self, addr: int, value: int, *, char: bool = False) -> None:
        """SI/SC: ``*addr = value`` — append a store position (KV entry)."""
        assert addr % 4 == 0 or char, "non-char stores are 4-byte aligned (§Memory)"
        self._stream.append((V.MEM, self._store_overlay(addr, value, char)))

    def free(self, addr: int) -> None:
        """Free ``addr`` by zero-overwrite (BLOG_SPEC §687-691). A store of value
        0 whose value nibbles are all 0, so the newest (this) matching entry
        relays 0 — the address reads ZFOD again, exactly as if the entry were
        evicted ("evicting both the old value and the zero overwriting it would
        have nil effect on the attention computation")."""
        self.store(addr, 0)

    def load(self, addr: int, *, char: bool = False) -> int:
        """LI/LC: ``AX = *addr`` — append a load position, return the value.

        Runs the real softmax1+ALiBi model.forward and decodes AX from the
        nibble band the memory head wrote (ZFOD 0 if the address is unwritten).
        """
        assert addr % 4 == 0 or char, "non-char loads are 4-byte aligned (§Memory)"
        self._stream.append((V.MEM, self._load_overlay(addr, char)))
        state = self._forward()[-1]                   # load position residual
        # decode the value from the AX nibble band via the LM byte-head algebra.
        val = 0
        for bi in range(4):
            val |= _decode_byte(state, self.L, self.L.AX, bi) << (8 * bi)
        # a char load is a single byte.
        self._stream.pop()                            # loads are queries, not KV
        return (val & 0xFF) if char else val

    def peek_weights(self, addr: int, *, char: bool = False):
        """Diagnostic: the memory head's attention weights for a load of ``addr``
        over the current stream (used by the proofs to show address-match +
        ALiBi recency directly)."""
        L, model = self.L, self.model
        self._stream.append((V.MEM, self._load_overlay(addr, char)))
        toks = torch.tensor([[tid for tid, _ in self._stream]])
        with torch.no_grad():
            x = model.embed[toks].clone()
            for i, (_, ov) in enumerate(self._stream):
                for dim, val in ov.items():
                    x[0, i, dim] = val
            attn = model.blocks[0].attn
            B, S, D = x.shape
            H, HD = attn.n_heads, attn.head_dim
            Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
            K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale
            pos = torch.arange(S)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
            scores = scores - attn.alibi_slopes.view(1, H, 1, 1) * dist
            causal = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
            scores = scores + causal
            from .blogspec_model import softmax1
            w = softmax1(scores, dim=-1)[0, 0, -1]    # head 0, load pos over keys
        self._stream.pop()
        return w


# ---------------------------------------------------------------------------
# Byte decode (LM head algebra) — shared with the foundation, kept local so the
# memory module has no torch.round on its path either.
# ---------------------------------------------------------------------------
def _decode_byte(state: torch.Tensor, L: MemoryLayout, reg_base: int,
                 byte_index: int) -> int:
    n0 = reg_base + 2 * byte_index + 0
    n1 = reg_base + 2 * byte_index + 1
    W = torch.zeros(V.VOCAB, L.D)
    b = torch.zeros(V.VOCAB)
    for v in range(256):
        lo, hi = V.nibbles_of_byte(v)
        W[v, n0] = 2.0 * lo
        W[v, n1] = 2.0 * hi
        b[v] = -(lo * lo) - (hi * hi)
    logits = F.linear(state, W, b)
    return int(logits[:256].argmax().item())
