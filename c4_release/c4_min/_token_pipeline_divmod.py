"""TOKEN-PIPELINED divmod — the SEQUENCE-axis lever (distinct from the #921
width-across-LAYERS work).

The #921 verdict (docs SHALLOW_WIDE_FP32_FP64_VERDICT) found the ISA WIDTH
(hidden >= 2624 fp64 / 3008 fp32) is the binder: a literal 0.5B checkpoint
(hidden 896) "cannot host the ISA at ANY depth" when the whole divmod runs in
ONE token's forward.  #921 attacked WIDTH by packing independent FFN blocks into
ONE wide super-layer (earlier FFNs WITHIN a token) and DEPTH by ASAP reorder --
both still inside a SINGLE token's forward.

THIS module attacks the OTHER axis: pipeline the divmod across the VM step's
TOKEN / SEQUENCE dimension.  A blogspec VM step emits ~30 register tokens, EACH
a full forward through all layers, with softmax1+ALiBi attention ACROSS them
(the same KV head measured at coherence-horizon 500k).  The lever:

  * ONE-WAY (independent) sub-computations -> computed at EARLIER tokens in
    parallel, READ BACK by the KV-memory attention CAM at the token where the
    result is needed.
  * SERIAL digit-recurrence -> chained ACROSS tokens (token t reads token t-1's
    remainder from the KV cache) so per-token depth stays shallow.
  * WIDE banks -> SLICED across tokens (each slice <= 896 hidden); attention
    concatenates.

MEASURED, off-build-path.  Golden 174ece66 UNTOUCHED (NEW file, no build-path
change).  Sparse-resident tiny model (hidden <= 896, <= 24 blocks); poll RSS,
abort > 4 GB.

The reference is ``nibble_muldivmod.divmod32`` (== ``isa.interpret`` DIV/MOD).
"""
from __future__ import annotations

import os
import resource
import threading
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

MASK32 = 0xFFFFFFFF


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def start_watchdog(limit_mb: int = 3800):
    def watch():
        while True:
            if rss_mb() > limit_mb:
                print(f"RSS ABORT {rss_mb()} MB > {limit_mb}", flush=True)
                os._exit(3)
            time.sleep(0.25)
    threading.Thread(target=watch, daemon=True).start()


# ===========================================================================
# COMPACT 896-dim LAYOUT for the token-pipelined divmod.
#
# The whole-ISA residual is 2569 dims because ALL divmod scratch bands (~950
# LOGSINK dims) coexist in ONE token's forward.  The token-pipeline slices those
# bands across tokens: at any single token only a SMALL scratch slice is live
# (measured floor 226 dims via liveness).  This layout parks:
#   * PERSISTENT bands (always live at every token): a, b, ONE, the KV-CAM
#     channels (address/value/store/load) used for cross-token gather.
#   * A reusable SCRATCH POOL that each token's block(s) write into; the pool is
#     large enough for the busiest single token (ls-qb-split needs QB_PP 45 +
#     QB_COL 9 + QB_C1 9 = 63 live scratch dims), then reused by the next token.
# ===========================================================================
class TPLayout:
    """A compact residual layout (<= 896) for the token-pipelined divmod.

    All the LOGSINK bands are RE-MAPPED into a single 896-dim residual, but the
    SCRATCH bands are packed so the max simultaneously-live footprint is small.
    We keep the SAME band *names* the logsink emitters use (they read
    ``L.LOGSINK.<BAND>`` and ``L.AX``/``L.STACK0``/``L.ONE``), so the byte-exact
    emitters wire unchanged into this compact layout.
    """

    def __init__(self, d_model: int = 896):
        self._off = 0
        self._names = {}
        # --- persistent VM state (a=STACK0, b=AX, ONE) ---
        self.PC = self._band("PC", 16)
        self.AX = self._band("AX", 16)         # divisor b nibbles (b = AX in a OP b)
        self.SP = self._band("SP", 16)
        self.BP = self._band("BP", 16)
        self.STACK0 = self._band("STACK0", 16)  # dividend a nibbles
        self.ONE = self._scalar("ONE")
        # --- KV-CAM cross-token gather channels (binary addr CAM + value relay) ---
        self.ADDR_BIN = self._band("ADDR_BIN", 32)  # store-slot addr bits (KEY)
        self.QRY_BIN = self._band("QRY_BIN", 32)    # gather-query addr bits (QUERY)
        self.VAL_NIB = self._band("VAL_NIB", 16)    # gathered value nibbles
        self.IS_STORE = self._scalar("IS_STORE")
        self.IS_LOAD = self._scalar("IS_LOAD")
        self.NORM_COMP = self._scalar("NORM_COMP")
        # --- LOGSINK scratch (the divmod bands, all mapped compactly) ---
        # The logsink emitters address these via an object with .BAND attrs; we
        # provide the SAME attribute names.  They are packed contiguously; the
        # liveness slicing is realised by which token WRITES/READS which band.
        self.LOGSINK = _LogSinkView(self)
        # pad up to d_model
        while self._off % 8 != 0:
            self._scalar(f"_pad{self._off}")
        self.D = self._off
        assert self.D <= d_model, (self.D, d_model)
        self.d_model = d_model

    def _scalar(self, name):
        return self._band(name, 1)

    def _band(self, name, size):
        base = self._off
        self._names[name] = (base, size)
        self._off += size
        return base


class _LogSinkView:
    """Mirror of ``nibble_logsink_blocks.LogSinkBands`` band names, packed into the
    compact TPLayout.  The DIV_RES/MOD_RES result bands + all divmod scratch."""

    def __init__(self, L: "TPLayout"):
        b = L._band
        s = L._scalar
        self.DIV_RES = b("LS_DIV_RES", 8)
        self.MOD_RES = b("LS_MOD_RES", 8)
        self.BM1 = b("LS_BM1", 8)
        self.BZ = s("LS_BZ")
        self.LOGQ = b("LS_LOGQ", 8)
        self.LOGKEY = b("LS_LOGKEY", 8)
        self.IS_RECIPQ = s("LS_IS_RECIPQ")
        self.IS_SINK = s("LS_IS_SINK")
        self.IS_RECIP_ROW = s("LS_IS_RECIP_ROW")
        self.IS_RECIP_Q = s("LS_IS_RECIP_Q")
        self.RECIP = s("LS_RECIP")
        self.RECIP2 = s("LS_RECIP2")
        self.QF = s("LS_QF")
        self.QSC = s("LS_QSC")
        self.Q = b("LS_Q", 8)
        self.QB_PP = b("LS_QB_PP", 45)
        self.QB_COL = b("LS_QB_COL", 9)
        self.QB_C1 = b("LS_QB_C1", 9)
        self.QB = s("LS_QB")
        self.REM = s("LS_REM")
        self.REM_NEG = s("LS_REM_NEG")
        self.REM_GEB = s("LS_REM_GEB")
        self.QSC2 = s("LS_QSC2")
        self.REM2 = s("LS_REM2")
        self.QREM = s("LS_QREM")
        self.DREM = s("LS_DREM")
        self.MREM = s("LS_MREM")


# ===========================================================================
# The token-pipelined divmod block list (parallel-form fp64), built against the
# COMPACT layout.  Reuses the BYTE-EXACT logsink FFN emitters.
# ===========================================================================
def build_parallel_divmod_blocks(L: TPLayout, refine: bool = True):
    """Return the parallel-form fp64 divmod as a list of (name, ffn_spec) blocks,
    plus the ASAP stage assignment (which token each block runs at).

    This is the #921 parallel form: parallel per-nibble decompose (all 8 extracts
    independent) + direct schoolbook.  The recip-attn block is realised as a plain
    reciprocal FFN here (b is a bounded scalar, so 1/b is computed by a small
    Newton refinement seeded from a coarse FFN reciprocal) — NOTE we build the
    reciprocal directly with an exact fp64 division in the *seed* unit so the
    downstream Newton/qf/decompose/schoolbook chain (the part the token-pipeline
    slices) is exercised byte-exact end-to-end without re-baking the RoPE-lane
    sink CAM.
    """
    from c4_min import nibble_logsink_blocks as LB
    dim = L.D
    ext = L.LOGSINK

    def pdec(prefix, rem, nib, clean=False):
        """The BYTE-EXACT decompose: MSB-first extract -> snap -> reduce running rem.
        This is the actually-runnable form (the #921 'parallel' extract of a 2^32
        scalar needs O(2^32) thresholds or fp32-chunking; the reduce chain is the
        genuine byte-exact construction).  For the token pipeline this serial reduce
        chain lives WITHIN one decompose-token (measured per-token depth below); the
        CROSS-TOKEN boundaries sit at the INTEGER stage outputs (Q, QSC2, REM2,
        DIV_RES, MOD_RES), which the CAM carries byte-exact."""
        blist = []
        for c in range(7, -1, -1):
            blist.append((f"{prefix}-ext{c}",
                          LB._msb_extract_block(L, dim, rem, nib, c, round_low=(c == 0))))
            npass = 1 if clean else (3 if c == 0 else 1)
            for sidx in range(npass):
                blist.append((f"{prefix}-snap{c}_{sidx}",
                              LB.compile_snap_nibbles(L, dim, nib + c, 1)))
            if c > 0:
                blist.append((f"{prefix}-red{c}",
                              LB._msb_reduce_block(L, dim, rem, nib, c)))
        return blist

    def schoolbook(pfx):
        return [
            (f"{pfx}-qb-products", LB.compile_qb_products(L, dim)),
            (f"{pfx}-qb-split", LB.compile_qb_split(L, dim)),
            (f"{pfx}-qb-carry0", LB.compile_qb_carry(L, dim, ext.QB_COL, ext.QB_C1)),
            (f"{pfx}-qb-carry1", LB.compile_qb_carry(L, dim, ext.QB_C1, ext.QB_COL)),
            (f"{pfx}-qb-carry2", LB.compile_qb_carry(L, dim, ext.QB_COL, ext.QB_C1)),
            (f"{pfx}-qb-recombine", LB.compile_qb_recombine(L, dim, ext.QB_C1)),
            (f"{pfx}-rem", LB.compile_rem(L, dim)),
            (f"{pfx}-correct", LB.compile_correct(L, dim)),
        ]

    # reciprocal seed (exact fp64 1/b into RECIP) + 2 Newton refines (the serial chain)
    seed = _compile_recip_seed(L, dim)
    core = [
        ("ls-bm1", LB.compile_bm1(L, dim)),
        ("ls-recip-seed", seed),
        ("ls-newton-br1", LB.compile_newton_br(L, dim, ext.RECIP)),
        ("ls-newton-st1", LB.compile_newton_step(L, dim, ext.RECIP, ext.RECIP2)),
        ("ls-newton-br2", LB.compile_newton_br(L, dim, ext.RECIP2)),
        ("ls-newton-st2", LB.compile_newton_step(L, dim, ext.RECIP2, ext.RECIP2)),
        ("ls-qf", LB.compile_qf(L, dim)),
    ]
    blocks = core
    blocks += [("ls-q-seed", LB.compile_seed_rem(L, dim, ext.QF, ext.QREM, offset=-0.5))]
    blocks += pdec("ls-q", ext.QREM, ext.Q)
    blocks += schoolbook("ls")
    if refine:
        # REFINE (only needed for the FP32 in-model reciprocal, #921): QSC = QSC2 +
        # round(REM2/b), re-decompose + re-schoolbook.  With the EXACT fp64 reciprocal
        # floor(a*(1/b)-0.5) is within ±1 (measured 250k), so the single ±1 in
        # `correct` suffices and refine is UNNEEDED -> depth halves.
        blocks += [("ls-refine", LB.compile_refine(L, dim)),
                   ("ls-refine-add", LB.compile_refine_add(L, dim)),
                   ("ls-r-seed", LB.compile_seed_rem(L, dim, ext.QSC, ext.QREM, offset=-0.5))]
        blocks += pdec("ls-r", ext.QREM, ext.Q)
        blocks += schoolbook("ls2")
    blocks += [("ls-d-seed", LB.compile_seed_rem(L, dim, ext.QSC2, ext.DREM, offset=-0.5))]
    blocks += pdec("ls-d", ext.DREM, ext.DIV_RES, clean=True)
    blocks += [("ls-m-seed", LB.compile_seed_rem(L, dim, ext.REM2, ext.MREM, offset=-0.5))]
    blocks += pdec("ls-m", ext.MREM, ext.MOD_RES, clean=True)
    blocks += [("ls-finalize", LB.compile_finalize(L, dim))]
    return blocks


def _run_prefix(x, blocks, prefixes):
    """Run all blocks whose name starts with any of ``prefixes`` (in order)."""
    for name, spec in blocks:
        if any(name.startswith(p) for p in prefixes):
            x = _ffn_forward(x, spec)
    return x


def _compile_parallel_extract(L, dim, src_band, nib_dst, c):
    """TRUE parallel-form nibble extract (the #921 rewrite): each nibble reads the
    CLEAN integer scalar ``src_band`` directly, NO running remainder --

        nib_c = floor(src / 16^c) - 16 * floor(src / 16^(c+1))

    so all 8 extracts are genuinely INDEPENDENT (parallelizable across tokens).
    ``src`` is a clean integer (the seed offsets it by -0.5 so floor is exact).
    Runs in fp64 (src up to 2^32 exceeds fp32's 2^24 -- the #921 precision crux).
    """
    from c4_min.nibble_logsink_blocks import (
        _empty64, _truncate, _clear, _floor_div_pow)
    import c4_min.nibble_logsink_blocks as LB
    LB._ONE = L.ONE
    # kmax must cover floor(src/16^c) up to ~16 (one nibble) but floor(src/16^c) for
    # the low nibbles can be huge; we only need the LOW 4 bits, so:
    #   nib_c = floor(src/16^c) mod 16 = floor(src/16^c) - 16*floor(src/16^(c+1)).
    # floor(src/16^c) needs kmax up to 16^(8-c); use the two-floor difference form.
    spec = _empty64(dim, 2 + 2 * 16 * (2 + 2))  # generous; _grow expands as needed
    u = 0
    u = _clear(spec, u, nib_dst + c)
    # + floor(src / 16^c): sum_{k>=1}[src >= k*16^c], capped since nib in 0..15 after
    #   subtracting 16*floor(src/16^(c+1)); but floor(src/16^c) itself can be large.
    # Use kmax = 15 on the RATIO by reading floor(src/16^c) - 16*floor(src/16^(c+1))
    # directly: [src >= k*16^c] for k=1..15 gives (floor(src/16^c) mod 16) ONLY when
    # floor(src/16^c) < 16, i.e. the top nibble.  For lower nibbles we must subtract
    # the higher part.  The clean way: nib_c = sum_{k=1..15}[ (src mod 16^(c+1)) >= k*16^c ].
    # src mod 16^(c+1) = src - 16^(c+1)*floor(src/16^(c+1)); but that needs the higher
    # floor.  Simplest EXACT parallel form: nib_c = floor(src/16^c) - 16*floor(src/16^(c+1)).
    m = 16 ** c
    mp1 = 16 ** (c + 1)
    kmax = (2 ** 32) // m + 2  # enough thresholds for floor(src/16^c)
    kmax = min(kmax, 16 ** (8 - c) + 2)
    u = _floor_div_pow(spec, u, {src_band: 1.0}, 0.0, m, kmax, nib_dst + c, 1.0)
    kmaxp = (2 ** 32) // mp1 + 2
    kmaxp = min(kmaxp, 16 ** (8 - c - 1) + 2)
    u = _floor_div_pow(spec, u, {src_band: 1.0}, 0.0, mp1, kmaxp, nib_dst + c, -16.0)
    return _truncate(spec, u, dim)


def _compile_recip_seed(L, dim):
    """Seed RECIP = 1/b exactly (fp64) from the b nibbles.  A single SiLU-gated unit
    cannot divide, so we bake the reciprocal as a data-independent LOOKUP is not
    possible for a 32-bit b; instead this block computes 1/b by the same softmax1
    sink mechanism the real build uses — but here, to exercise the DOWNSTREAM
    token-sliced chain byte-exact without the RoPE-lane CAM, we mark RECIP as a
    HOST-SEEDED band (the runner writes the exact fp64 1/b into RECIP before this
    block).  The block itself is an identity passthrough on RECIP (keeps the spec
    shape valid and lets the Newton refine read it)."""
    from c4_min.nibble_logsink_blocks import _empty64, _truncate, _ident, S
    import c4_min.nibble_logsink_blocks as LB
    LB._ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, 1)
    # identity: RECIP += 0 (a valid no-op unit; RECIP is host-seeded)
    spec["W_up"][0, L.ONE] = S
    spec["W_gate"][0, L.ONE] = 0.0
    return _truncate(spec, 1, dim)


# ===========================================================================
# A minimal fp64 FFN-block runner + the cross-token KV gather.
#
# We run the divmod blocks as SwiGLU FFN layers on the compact residual, and at
# STAGE BOUNDARIES route the carried intermediates through a REAL softmax1+ALiBi
# KV-memory attention gather (byte-exact address CAM).  This exercises the
# load-bearing cross-token read-back: an intermediate written at token t is
# gathered by the token where it is next needed.
# ===========================================================================
def _ffn_forward(x, spec):
    """One SwiGLU FFN block (fp64), additive residual, matching blogspec FFN."""
    from c4_min.nibble_logsink_blocks import SILU_S  # noqa
    Wup = spec["W_up"]; bup = spec["b_up"]
    Wg = spec["W_gate"]; bg = spec["b_gate"]
    Wd = spec["W_down"]; bd = spec["b_down"]
    up = F.linear(x, Wup) + bup
    gate = F.linear(x, Wg) + bg
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, Wd, bd)


def _run_blocks(x, blocks, names=None):
    """Run a list of (name, spec) FFN blocks sequentially on residual x."""
    for name, spec in blocks:
        if names is not None and name not in names:
            continue
        x = _ffn_forward(x, spec)
    return x


# ---------------------------------------------------------------------------
# The cross-token KV gather (the load-bearing byte-exact read-back).
#
# Exactly the blogspec_memory §Memory CAM, in fp64: a producer token stores
# (addr, value-nibbles) as a KV entry (binary-address key, value relay); a
# consumer token gathers it by querying the identical address.  softmax1 + ALiBi
# so an unwritten address reads ZFOD 0 and the newest write wins.  We use it to
# carry BOTH the wide nibble banks (Q, DIV_RES, MOD_RES: 8 nibbles each) AND the
# serial scalars (encoded as 8 nibbles of a fixed-point scaled integer) token to
# token.
# ---------------------------------------------------------------------------
ADDR_BITS = 32
EFF = 500000.0
BIAS = (ADDR_BITS - 1) * EFF
SLOPE = 1.0


def _addr_bits(addr):
    return [float((addr >> b) & 1) for b in range(ADDR_BITS)]


def kv_gather(store_rows, query_addr, dtype=torch.float64):
    """One softmax1+ALiBi CAM read.  ``store_rows`` = list of (position, addr,
    value_nibbles[16]) producer entries; returns the 16 gathered value nibbles at
    the query position (ZFOD 0 if the address was never stored).  Byte-exact: the
    per-bit ±key/query gives an exact-match score n·EFF that dominates the +1 sink
    and every 1-bit-off row, and ALiBi picks the most recent among equal addrs."""
    # build the score of the query against each stored row
    qbits = _addr_bits(query_addr)
    qpos = max((p for p, _, _ in store_rows), default=0) + 1
    scores = []
    vals = []
    for (pos, addr, nibs) in store_rows:
        kbits = _addr_bits(addr)
        # per-bit agree/disagree contribution: sum (2b-1)(2q-1) * EFF  (±EFF each)
        s = sum(EFF * (2 * kb - 1) * (2 * qb - 1) for kb, qb in zip(kbits, qbits))
        s -= BIAS                      # ZFOD bias: exact match -> +EFF, 1-off -> -EFF
        s -= SLOPE * abs(qpos - pos)   # ALiBi recency (latest-write-wins)
        scores.append(s)
        vals.append(nibs)
    # softmax1: implicit +1 sink at score 0
    t = torch.tensor(scores + [0.0], dtype=dtype)
    w = torch.softmax(t, dim=0)        # the +0 sink is the last entry (==softmax1)
    out = torch.zeros(16, dtype=dtype)
    for i, nibs in enumerate(vals):
        out += w[i] * torch.tensor([float(n) for n in nibs], dtype=dtype)
    return out  # gathered nibbles (ZFOD ~0 if no exact match)


# ===========================================================================
# The token-pipelined divmod END-TO-END: run the parallel-form divmod as FFN
# transformer blocks on the compact 896-dim residual, SLICED across tokens, with
# the carried intermediates transported token-to-token by the real KV gather.
#
# Token stages (each a separate token's forward; carried scalars cross via CAM):
#   T0  reciprocal + Newton + qf  -> carry QF (fractional scalar) via CAM addr A0
#   T1  q-seed + q-decompose (8 parallel ext + snap) -> carry Q nibbles addr A1
#   T2  schoolbook q*b + rem + correct -> carry QSC2, REM2 (integers) addr A2/A3
#   T3  d-decompose (DIV_RES) -> result nibbles
#   T4  m-decompose (MOD_RES) -> result nibbles + finalize
#
# The carried value at each boundary is written into the producer token's
# VAL_NIB band and gathered into the consumer token's AX/scratch band.  We drive
# the FFN blocks on the SAME compact residual to keep them byte-exact, and splice
# the CAM read-back at the boundary so the cross-token path is genuinely
# exercised (not a single-token stitch).
# ===========================================================================
STAGE_ADDR = {"QF": 0x10, "Q": 0x20, "QSC2": 0x30, "REM2": 0x40, "RECIP2": 0x50}


def run_pipeline_divmod(a: int, b: int, L: TPLayout, blocks, verbose=False):
    """Run the divmod across 5 token-stages with the CAM carrying boundary values.

    Returns (q, r) decoded from the DIV_RES/MOD_RES nibbles of the final residual.
    Each stage runs the relevant FFN blocks on a FRESH compact residual seeded
    from the CAM gather of the previous stage's carried intermediates -- so the
    cross-token read-back is on the critical path.
    """
    import c4_min.nibble_logsink_blocks as LB
    a &= MASK32
    b &= MASK32
    ext = L.LOGSINK
    dim = L.D
    D = dim
    bname = {n: s for n, s in blocks}

    def seed_ab(x):
        # a -> STACK0 nibbles, b -> AX nibbles, ONE=1
        for c in range(8):
            x[L.STACK0 + c] = float((a >> (4 * c)) & 0xF)
            x[L.AX + c] = float((b >> (4 * c)) & 0xF)
        x[L.ONE] = 1.0

    store_rows = []  # (pos, addr, nibs) CAM log across the whole step
    pos = [0]

    def store(addr, value_band_vals):
        pos[0] += 30           # one 30-token frame apart (spec spacing)
        nibs = list(value_band_vals) + [0.0] * (16 - len(value_band_vals))
        store_rows.append((pos[0], addr, nibs))

    def gather(addr):
        pos[0] += 1
        return kv_gather(store_rows, addr)

    # ---- STAGE T0: reciprocal + Newton + qf (all FRACTIONAL, stays in one token) ----
    x = torch.zeros(D, dtype=torch.float64)
    seed_ab(x)
    # fp64 sink reciprocal: the softmax1 sink weight 1/(1+sum 16^j d_j) = 1/b, which
    # in fp64 is 1.0/b exactly (the log-sink build's fp64 reciprocal).  b==0 -> 0.
    recip = 0.0 if b == 0 else (1.0 / b)
    x[ext.RECIP] = recip
    # run bm1 (b-1 nibbles + BZ) then Newton refine + qf on this token
    x = _ffn_forward(x, bname["ls-bm1"])
    for nm in ("ls-newton-br1", "ls-newton-st1", "ls-newton-br2", "ls-newton-st2", "ls-qf"):
        x = _ffn_forward(x, bname[nm])
    QF = float(x[ext.QF])
    # carry QF across the token boundary via the CAM (as a raw fractional scalar)
    store(STAGE_ADDR["QF"], [QF])
    if verbose:
        print(f"    T0: QF={QF!r}")

    have_refine = "ls-refine" in bname
    # ---- STAGE T1: q-decompose (needs QF).  The serial extract->snap->reduce
    #      chain lives INSIDE this one token; only the boundary output crosses. ----
    x = torch.zeros(D, dtype=torch.float64)
    seed_ab(x)
    x[ext.QF] = float(gather(STAGE_ADDR["QF"])[0])   # <-- cross-token read-back
    x = _ffn_forward(x, bname["ls-q-seed"])
    x = _run_prefix(x, blocks, ["ls-q-ext", "ls-q-snap", "ls-q-red"])
    x = _run_prefix(x, blocks, ["ls-qb", "ls-rem", "ls-correct"])  # schoolbook+correct
    if have_refine:
        # refine: QSC = QSC2 + round(REM2/b) (fixes >±1 quotient error for large q)
        x = _ffn_forward(x, bname["ls-refine"])
        x = _ffn_forward(x, bname["ls-refine-add"])
        QSC = float(x[ext.QSC])
        store(STAGE_ADDR["Q"], [QSC])   # carry the refined integer quotient scalar
        if verbose:
            print(f"    T1: QSC(refined)={QSC!r}")
        # ---- STAGE T2: re-decompose refined QSC -> re-schoolbook -> correct ----
        x = torch.zeros(D, dtype=torch.float64)
        seed_ab(x)
        x[ext.QSC] = float(gather(STAGE_ADDR["Q"])[0])    # <-- cross-token read-back
        x = _ffn_forward(x, bname["ls-r-seed"])
        x = _run_prefix(x, blocks, ["ls-r-ext", "ls-r-snap", "ls-r-red"])
        x = _run_prefix(x, blocks, ["ls2-qb", "ls2-rem", "ls2-correct"])
    QSC2 = float(x[ext.QSC2]); REM2 = float(x[ext.REM2])
    store(STAGE_ADDR["QSC2"], [QSC2])
    store(STAGE_ADDR["REM2"], [REM2])
    if verbose:
        print(f"    T{'2' if have_refine else '1'}: QSC2={QSC2!r} REM2={REM2!r}")

    # ---- STAGE T3: DIV_RES decompose (needs QSC2) ----
    x = torch.zeros(D, dtype=torch.float64)
    seed_ab(x)
    x[ext.QSC2] = float(gather(STAGE_ADDR["QSC2"])[0])  # <-- cross-token read-back
    x = _ffn_forward(x, bname["ls-d-seed"])
    x = _run_prefix(x, blocks, ["ls-d-ext", "ls-d-snap", "ls-d-red"])
    div_nibs = [float(x[ext.DIV_RES + c]) for c in range(8)]

    # ---- STAGE T4: MOD_RES decompose (needs REM2) ----
    x2 = torch.zeros(D, dtype=torch.float64)
    seed_ab(x2)
    x2[ext.REM2] = float(gather(STAGE_ADDR["REM2"])[0])  # <-- cross-token read-back
    x2 = _ffn_forward(x2, bname["ls-m-seed"])
    x2 = _run_prefix(x2, blocks, ["ls-m-ext", "ls-m-snap", "ls-m-red"])
    mod_nibs = [float(x2[ext.MOD_RES + c]) for c in range(8)]

    # finalize (b==0 gate) — copy div/mod nibbles into one residual and run it
    xf = torch.zeros(D, dtype=torch.float64)
    seed_ab(xf)
    # BZ = [b==0]: recompute via bm1
    xf = _ffn_forward(xf, bname["ls-bm1"])
    for c in range(8):
        xf[ext.DIV_RES + c] = div_nibs[c]
        xf[ext.MOD_RES + c] = mod_nibs[c]
    xf = _ffn_forward(xf, bname["ls-finalize"])
    q = 0
    r = 0
    for c in range(8):
        q |= (int(round(float(xf[ext.DIV_RES + c]))) & 0xF) << (4 * c)
        r |= (int(round(float(xf[ext.MOD_RES + c]))) & 0xF) << (4 * c)
    return q & MASK32, r & MASK32


def run_packed_divmod(a: int, b: int, L: TPLayout, blocks, cap: int = 23,
                      verbose=False):
    """Run the divmod as a GENERIC token pipeline: pack the block list into tokens
    of <= ``cap`` FFN layers each; at EVERY token boundary carry the FULL live
    scratch across via the CAM (each live band its own address), then re-seed the
    next token from the gather.  This is the honest per-token<=24-layer construction
    -- the cross-token CAM read-back is exercised at every cut, not just the coarse
    stage boundaries.

    Returns (q, r).  The reciprocal seed is written host-side into the first token
    (the sink-attention head is a single one-way block, verified separately)."""
    a &= MASK32
    b &= MASK32
    ext = L.LOGSINK
    D = L.D
    bname_order = list(blocks)

    def seed_ab(x):
        for c in range(8):
            x[L.STACK0 + c] = float((a >> (4 * c)) & 0xF)
            x[L.AX + c] = float((b >> (4 * c)) & 0xF)
        x[L.ONE] = 1.0

    # Every non-persistent band that is ever written -> a CAM address to carry it.
    from c4_min._measure_critpath_921 import _block_rw
    persistent = set()
    for nm in ("PC", "AX", "SP", "BP", "STACK0", "ONE"):
        base = getattr(L, nm)
        for k in range(16 if nm in ("PC", "AX", "SP", "BP", "STACK0") else 1):
            persistent.add(base + k)
    # partition blocks into tokens of <= cap
    tokens = []
    cur = []
    for nb in bname_order:
        if len(cur) >= cap:
            tokens.append(cur)
            cur = []
        cur.append(nb)
    if cur:
        tokens.append(cur)

    store_rows = []
    pos = [0]

    def carry_out(x, dims):
        """Store each live scratch dim as its own CAM entry (addr = dim index)."""
        for d in sorted(dims):
            pos[0] += 30
            store_rows.append((pos[0], 0x1000 + d, [float(x[d])] + [0.0] * 15))

    def carry_in(x, dims):
        for d in sorted(dims):
            pos[0] += 1
            x[d] = float(kv_gather(store_rows, 0x1000 + d)[0])

    # run token by token; between tokens, carry the union of (written so far & read later)
    # conservatively carry ALL non-persistent dims written up to this token.
    x = torch.zeros(D, dtype=torch.float64)
    seed_ab(x)
    recip = 0.0 if b == 0 else (1.0 / b)
    x[ext.RECIP] = recip
    written = set()
    for ti, tok in enumerate(tokens):
        if ti > 0:
            # carry live scratch across the token boundary via the CAM
            live = written - persistent
            carry_out(x, live)
            xn = torch.zeros(D, dtype=torch.float64)
            seed_ab(xn)
            carry_in(xn, live)
            x = xn
        for name, spec in tok:
            r, w = _block_rw(spec)
            x = _ffn_forward(x, spec)
            written |= (w - {L.ONE})
    q = 0
    r = 0
    for c in range(8):
        q |= (int(round(float(x[ext.DIV_RES + c]))) & 0xF) << (4 * c)
        r |= (int(round(float(x[ext.MOD_RES + c]))) & 0xF) << (4 * c)
    return q & MASK32, r & MASK32, len(tokens)
