"""CONST-DIVISOR DIVMOD as an ATTENTION-CAM AUTOMATON (the transition table as a
content-addressable KV memory, one attention lookup per step).

Where ``const_divmod_automaton.py`` realises the divisibility-DFA transition
``(q, R_new) = divmod(16·R + d, b)`` as a per-step FFN **one-hot** (block EQ builds
the equality pulses, block RD guards the ``b·16``-row table — 2 transition blocks
+ 1 commit = **3 blocks/step, 25 blocks**), this module realises the SAME baked
transition table as an **attention KV memory** and does a **content-addressable
lookup per step** — collapsing the two FFN transition blocks into ONE attention
block.

The idea
========
Because ``b`` is a compile-time constant, the transition table is a fixed set of
``b·16`` rows ``(R, d) -> (q, R_new)`` — a perfect candidate for a CAM:

  * **Keys** — one baked key row per reachable ``(R, d)`` pair, encoding ``(R, d)``
    as a per-bit ``±1`` agreement code on a RESERVED indicator band (same binary
    per-bit-agreement key as ``qwen_full_vm._bake_memory_cam`` and
    ``shift_attention_bench``'s coarse-CAM head).  The reserved band means program
    state never cross-matches a key.
  * **Query** — the CURRENT ``(R, d)`` in the same ±1 code.
  * **Value** — the baked ``(q, R_new)`` for that entry (as nibble scalars).

One attention head (softmax1) retrieves the matching row: with a per-bit gain
``G`` the exact-match net score is ``+0.5·G²/√d`` (softmax weight ≈ 1.0 in fp32)
and every 1-bit mismatch is ``≥ 2·G²/√d`` BELOW it (weight ≈ 0), so
``value = Σ_i w_i·v_i ≈ v_matched`` is **byte-exact** — exactly the sharp-CAM
regime the memory CAM runs in (§Memory latest-write-wins minus the recency; here
each ``(R,d)`` key is unique so no tie-break is needed).

Per-step structure (the attention automaton cell)
=================================================
Over a tiny residual whose bands are exactly the DFA planes:

  * **Block QENC** (FFN, ~½ block) — write the ±1 binary query code for the current
    ``(R, d)`` from the state nibbles ``R`` and the streamed input nibble ``d``.  A
    nibble ``x`` bit ``bit(x,i)`` is ``[floor(x/2^i) is odd] = [x & 2^i]``, realised
    as one ``_step_ge`` staircase per bit (the bit is ``Σ_{k odd} [x >= k·2^i] -
    [x >= (k+1)·2^i]``).  The query lane is ``G·(2·bit − 1)`` (i.e. +G if the bit is
    1, −G if 0) plus the ``−B`` bias term on the bias lane.
  * **Block CAM** (ATTENTION) — the softmax1 content-addressable lookup: query the
    ``b·16`` baked key rows, retrieve the matched row's value ``(q, R_new)`` into a
    value band ``V_OUT``.  **This is the single attention block that replaces the two
    FFN transition blocks (EQ + RD).**
  * **Block COMMIT** (FFN, ~½ block) — thread ``R := R_new`` (SET) and scatter the
    emitted quotient digit ``q = V_OUT[0]`` into the assembled quotient's MSB-first
    place.  As in the FFN automaton, the commit CANNOT fold into CAM: the CAM writes
    ``V_OUT`` reading the block INPUT, so ``R := R_new`` must be a subsequent block.

So the cell is **1 attention block + 2 trivial FFN blocks** per step (vs the FFN
automaton's 2 FFN transition blocks + 1 commit).  The attention-lookups-per-step is
**1** (the target), and the total depth is ``3·in_nibs + 1`` blocks — same block
count as the FFN automaton, but the heavy ``b·16``-row transition table now lives in
the ATTENTION KV cache (fixed weight, position-invariant) instead of ``b·16``
FFN guard units per step.  The point: the per-step transition LOOKUP is ONE
attention head, and the ``b·16`` table is a shared KV memory, not replicated FFN
one-hots every step.

The KV rows are BAKED as a fixed prepended context (the transition table is
compile-time constant, so its key/value rows are permanent tokens the query
attends to — the ``_bake_memory_cam`` "store frames ride the token stream"
mechanism, here with the store log frozen at build time).

fp32 discipline
===============
The QENC staircase forms a nibble bit (``x & 2^i``): each ``_step_ge`` argument is a
nibble (0..15) scaled by ``RELU_S`` → ``≤ RELU_S·16 ≈ 3200``, five orders below the
``2^24`` fp32-integer limit — NO ``16·R`` amplification, no accumulation.  The
attention scores are per-bit ``±G²`` dots (``G=16`` → net ~16 logits, e^16 ≫ the
sink), so the CAM is a hardmax in fp32 (exact-match softmax weight rounds to 1.0).
Values are baked integer nibbles.  Zero fp64 anywhere.

Feasible ``b`` range
====================
The KV cache holds ``b·16`` rows (one per reachable ``(R,d)``), each a
``(n_bits + control)``-dim key + a ``(1 + state_nibs)``-dim value.  The state
occupies ``n_bits_R = ⌈log₂ b⌉`` query bits; a 32-bit dividend nibble is 4 bits.
This stays sharp and small for **small-to-moderate ``b``** (the softmax stays a
clean hardmax up to thousands of rows; verified to ``b=1024`` → 16384 rows).  Past
that the ``b·16`` KV table dominates the memory (as the FFN automaton's ``b·16``
guard table did) and the ``b``-independent digit-recurrence wins.

READ/import-only from the shared ALU: ``_step_ge``, ``_ident``, ``_clear``,
``_truncate``, ``_empty_spec``, ``RELU_S``, ``S``, ``SILU_S`` — this module edits no
shared file.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import nibble_alu32 as alu
from .nibble_vm import RELU_S, S, SILU_S, _empty_spec

MASK32 = 0xFFFFFFFF
FP32_INT_LIMIT = 1 << 24            # fp32 loses unit precision above 2**24.

# --- CAM sharpness constants (mirroring _bake_memory_cam) ------------------
_CAM_G = 16.0                       # per-bit agreement gain: net exact match ~16 logits


# ===========================================================================
# Residual-plane SwiGLU forward (identical to the landed ALU blocks + the FFN
# automaton's runner): x + Linear(silu(up)*gate, W_down).
# ===========================================================================
def _apply_ffn(x: torch.Tensor, w: Dict[str, torch.Tensor]) -> torch.Tensor:
    up = F.linear(x, w["W_up"]) + w["b_up"]
    gate = F.linear(x, w["W_gate"]) + w["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, w["W_down"], w["b_down"])


def _nz(w: Dict[str, torch.Tensor]) -> int:
    return sum(int((w[k] != 0).sum())
               for k in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"))


# ===========================================================================
# The attention-CAM lookup — a baked softmax1 KV memory over the transition table.
# Structurally the _bake_memory_cam / coarse-CAM head: per-bit ±G agreement key +
# query, a load-bias lane, softmax1 sink, value copy.  Here the "store frames" are
# the b*16 baked transition rows (frozen at build time), and the "load query" is the
# current (R, d).  Because every (R,d) key is UNIQUE, no recency tie-break is needed.
# ===========================================================================
class _AttnCAM:
    """The baked KV memory realising the divmod transition table as a CAM.

    Keys/queries live on a per-bit ±G agreement code (matching bits ADD +G^2,
    mismatching CANCEL -G^2) plus ONE load-bias lane so an EXACT match nets
    +0.5*G^2 (softmax1 weight ~1) and any 1-bit mismatch nets <=-1.5*G^2 (weight
    ~0).  Values carry the baked (q, R_new) nibble scalars.  The softmax1 sink (a
    zero key/value) guarantees ZFOD if no row matches.
    """

    def __init__(self, b: int, state_nibs: int):
        self.b = b
        self.state_nibs = state_nibs
        self.n_bits_R = max(1, (b - 1).bit_length()) if b > 1 else 1
        self.n_bits_d = 4
        self.n_bits = self.n_bits_R + self.n_bits_d
        self.dim = self.n_bits + 1                 # + bias lane
        self.G = _CAM_G
        self.B = math.sqrt(self.n_bits - 0.5) * self.G if self.n_bits > 0 else 0.0
        # value width: q (1 nibble) + R_new (state_nibs nibbles).
        self.val_dim = 1 + state_nibs
        # bake the KV rows (keys, values) for all reachable (R, d).
        keys: List[List[float]] = []
        vals: List[List[float]] = []
        for R in range(b):
            for d in range(16):
                v = 16 * R + d
                q = v // b
                r_new = v % b
                keys.append(self._code(R, d, is_key=True))
                vals.append([float(q)] + [float((r_new >> (4 * j)) & 0xF)
                                          for j in range(state_nibs)])
        self.K = torch.tensor(keys, dtype=torch.float32) if keys else torch.zeros(0, self.dim)
        self.V = torch.tensor(vals, dtype=torch.float32) if vals else torch.zeros(0, self.val_dim)
        self.n_rows = len(keys)

    def _bits(self, x: int, n: int) -> List[int]:
        return [(x >> i) & 1 for i in range(n)]

    def _code(self, R: int, d: int, is_key: bool) -> List[float]:
        """The ±G per-bit agreement code for (R, d); key and query differ only on the
        bias lane sign (key +B, query -B) so their product is -B^2 = -(n_bits-0.5)G^2
        on the bias lane, subtracting the constant that pushes non-exact matches below
        the sink."""
        bits = self._bits(R, self.n_bits_R) + self._bits(d, self.n_bits_d)
        code = [self.G * (2 * bit - 1) for bit in bits]
        code.append(self.B if is_key else -self.B)
        return code

    def query(self, R: int, d: int) -> torch.Tensor:
        return torch.tensor(self._code(R, d, is_key=False), dtype=torch.float32)

    def lookup(self, R: int, d: int) -> Tuple[int, int]:
        """The REAL softmax1 attention forward: retrieve (q, R_new) for state (R,d).

        scores = (K @ query) / sqrt(dim); softmax1 weights = e^s / (1 + Σ e^s) (the
        +1 is the BOS sink); value = Σ w_i · V_i.  Then round each value nibble."""
        return self.lookup_qvec(self.query(R, d))

    def lookup_qvec(self, q: torch.Tensor) -> Tuple[int, int]:
        """The softmax1 attention forward over an ARBITRARY query vector ``q`` — the
        one the QENC FFN block actually wrote into the QCODE band (so the CAM
        consumes the FFN's output, not a re-derived ideal query).  Byte-exact
        because the QENC ±G code is fp32-exact to ~1e-3 and the CAM margin is ~16
        logits (e^16)."""
        scores = (self.K @ q) / math.sqrt(self.dim)
        exps = torch.exp(scores)
        weights = exps / (1.0 + exps.sum())        # softmax1 (sink at score 0)
        out = (weights.unsqueeze(-1) * self.V).sum(0)
        q_dig = int(round(float(out[0]))) & 0xF
        r_new = 0
        for j in range(self.state_nibs):
            r_new |= (int(round(float(out[1 + j]))) & 0xF) << (4 * j)
        return q_dig, r_new

    def worst_exact_weight(self) -> float:
        """The MINIMUM (over all rows) exact-match softmax1 weight — the sharpness of
        the CAM.  Must round to 1.0 in fp32 for byte-exact retrieval."""
        w_min = 1.0
        for R in range(self.b):
            for d in range(16):
                q = self.query(R, d)
                scores = (self.K @ q) / math.sqrt(self.dim)
                exps = torch.exp(scores)
                weights = exps / (1.0 + exps.sum())
                w_min = min(w_min, float(weights.max()))
        return w_min

    def key_query_magnitude(self) -> float:
        """The max |key| / |query| entry — the sharp-CAM regime flag (per-bit ±G, so
        ~G; the bias lane ~B ~ sqrt(n_bits)·G).  Reported to confirm we stay in the
        _bake_memory_cam regime (G=16, well within fp32)."""
        m = float(self.K.abs().max()) if self.n_rows else 0.0
        for R in range(min(self.b, 4)):
            m = max(m, float(self.query(R, 0).abs().max()))
        return m

    def nz(self) -> int:
        """Non-zero KV weights (the attention-CAM's parameter cost): the K + V rows.
        The projections (q/k/v/o) are near-identity onto the reserved band, so the
        real cost is the KV cache — b*16 rows of (n_bits+1) key + (1+state_nibs)
        value entries."""
        return int((self.K != 0).sum() + (self.V != 0).sum())


# ===========================================================================
# Layout — a tiny residual whose bands are exactly the planes the DFA touches.
# ===========================================================================
class _AttnAutomatonLayout:
    def __init__(self, b: int, in_width: int = 32):
        self.b = b
        self.in_width = in_width
        self.in_nibs = in_width // 4
        self.state_nibs = max(1, math.ceil(math.log(max(b, 2), 16)))
        if b > 0 and 16 ** self.state_nibs <= (b - 1):
            self.state_nibs += 1
        self.q_nibs = self.in_nibs
        self.r_nibs = self.state_nibs

        self.cam = _AttnCAM(b if b > 0 else 1, self.state_nibs)

        off = 0

        def band(sz):
            nonlocal off
            b0 = off
            off += sz
            return b0

        self.ONE = band(1)                         # constant-1 lane
        self.D_IN = band(1)                        # current input nibble d (0..15)
        self.R = band(self.state_nibs)             # state R as nibbles (LSB first)
        # the ±G binary QUERY code lanes fed to the attention CAM (n_bits + bias).
        self.QCODE = band(self.cam.dim)
        # the attention CAM's retrieved value (q digit + R_new nibbles).
        self.V_OUT = band(self.cam.val_dim)
        self.Q_STEP = band(1)                      # emitted quotient digit this step
        self.Q = band(self.q_nibs)                 # assembled quotient (LSB..MSB order)
        self.BZ = band(1)                          # [b == 0] divisor-zero predicate
        self.D = off

    def load(self, a: int) -> torch.Tensor:
        x = torch.zeros(self.D)
        x[self.ONE] = 1.0
        x[self.BZ] = 1.0 if self.b == 0 else 0.0
        return x                                   # R := 0, Q := 0

    def input_nibble(self, a: int, step: int) -> int:
        a &= (1 << self.in_width) - 1
        shift = 4 * (self.in_nibs - 1 - step)      # MSB-first
        return (a >> shift) & 0xF

    def decode_q(self, x: torch.Tensor) -> int:
        val = 0
        for j in range(self.q_nibs):
            val |= (int(round(float(x[self.Q + j]))) & 0xF) << (4 * j)
        return val

    def decode_r(self, x: torch.Tensor) -> int:
        val = 0
        for j in range(self.r_nibs):
            val |= (int(round(float(x[self.R + j]))) & 0xF) << (4 * j)
        return val


# ===========================================================================
# Block QENC (FFN) — write the ±G binary query code for the current (R, d).
# ===========================================================================
def _qenc_block(L: _AttnAutomatonLayout) -> Dict[str, torch.Tensor]:
    """Emit the ±G per-bit agreement query code into QCODE from state R and input d.

    Bit ``i`` of a nibble ``x`` is ``[x & 2^i]``; realised as the odd-parity sum of
    ``2^i``-steps: ``Σ_{k odd, k·2^i <= 15}([x >= k·2^i] - [x >= (k+1)·2^i])``.  Each
    step is one ``_step_ge`` (two relu units).  We SET the QCODE lane to
    ``G·(2·bit − 1) = 2G·bit − G``: baseline ``−G`` (a constant) + ``2G·bit``.  The
    bias lane gets the constant ``−B`` (query side).  SET-cleared at the head so
    re-running the block each step is idempotent."""
    cam = L.cam
    alu._ONE = L.ONE
    G = cam.G
    n_lanes = cam.dim
    # generous unit budget: per bit ~ (clear + const + odd-step staircase).
    spec = _empty_spec(L.D, n_lanes * (2 + 16) + 8)
    u = 0
    # clear all query-code lanes first (SET).
    for c in range(n_lanes):
        u = alu._clear(spec, u, L.QCODE + c)
    # helper: write bit(x, i) * 2G into lane, and the -G baseline.
    def emit_bit(src_band: int, i: int, lane: int):
        nonlocal u
        # baseline -G (constant, via ONE).
        u = alu._ident(spec, u, {L.ONE: 1.0}, 0.0, lane, -G)
        # + 2G * bit(x, i) = 2G * Σ_{k odd}([x>=k·2^i] - [x>=(k+1)·2^i]).
        m = 1 << i
        k = 1
        while k * m <= 15:
            # odd k contributes +1, and the -[x>=(k+1)m] pairs it (even boundary).
            u = alu._step_ge(spec, u, {src_band: 1.0}, 0.0, k * m, lane, 2.0 * G)
            u = alu._step_ge(spec, u, {src_band: 1.0}, 0.0, (k + 1) * m, lane, -2.0 * G)
            k += 2
    # R bits -> first n_bits_R lanes.
    bit_lane = 0
    for s in range(L.state_nibs):
        # each state nibble contributes 4 bits, but only up to n_bits_R total.
        for i in range(4):
            if bit_lane >= cam.n_bits_R:
                break
            emit_bit(L.R + s, i, L.QCODE + bit_lane)
            bit_lane += 1
    # d bits -> next n_bits_d lanes.
    for i in range(cam.n_bits_d):
        emit_bit(L.D_IN, i, L.QCODE + cam.n_bits_R + i)
    # bias lane: query side = -B (a constant).
    u = alu._ident(spec, u, {L.ONE: 1.0}, 0.0, L.QCODE + cam.n_bits, -cam.B)
    return alu._truncate(spec, u, L.D)


# ===========================================================================
# Block CAM (ATTENTION) — the softmax1 content-addressable transition lookup.
# Runs the real attention forward on the baked KV table (via _AttnCAM.lookup) and
# writes the retrieved (q, R_new) into V_OUT.  This is the single attention block
# that replaces the FFN automaton's two transition blocks (EQ + RD).
# ===========================================================================
def _cam_block_forward(x: torch.Tensor, L: _AttnAutomatonLayout) -> torch.Tensor:
    """Apply the attention CAM: read the ±G query code the QENC FFN block WROTE into
    QCODE, run the REAL softmax1 attention lookup over the baked KV table, and write
    the retrieved ``(q, R_new)`` into V_OUT.  The lookup consumes the ACTUAL FFN
    output (``QCODE``), so this is the genuine end-to-end chain (not a re-derived
    ideal query).  SET semantics on V_OUT (overwrite each step)."""
    x = x.clone()
    qvec = x[L.QCODE:L.QCODE + L.cam.dim]
    q_dig, r_new = L.cam.lookup_qvec(qvec)
    x[L.V_OUT + 0] = float(q_dig)
    for j in range(L.state_nibs):
        x[L.V_OUT + 1 + j] = float((r_new >> (4 * j)) & 0xF)
    return x


# ===========================================================================
# Block COMMIT (FFN) — thread R := R_new and scatter q MSB-first.
# ===========================================================================
def _commit_block(L: _AttnAutomatonLayout, step: int) -> Dict[str, torch.Tensor]:
    """R := V_OUT[1..] (SET) and scatter V_OUT[0] into the assembled quotient at its
    MSB-first place (step 0 = high nibble)."""
    alu._ONE = L.ONE
    S_ = L.state_nibs
    spec = _empty_spec(L.D, 2 * S_ + 6)
    u = 0
    for s in range(S_):
        u = alu._clear(spec, u, L.R + s)
        u = alu._ident(spec, u, {L.V_OUT + 1 + s: 1.0}, 0.0, L.R + s, 1.0)
    j = L.in_nibs - 1 - step
    u = alu._ident(spec, u, {L.V_OUT + 0: 1.0}, 0.0, L.Q + j, 1.0)
    return alu._truncate(spec, u, L.D)


def _bz_gate_block(L: _AttnAutomatonLayout) -> Dict[str, torch.Tensor]:
    """Final b==0 -> (q, r) = (0, 0) gate (ISA_SPEC §4.2)."""
    alu._ONE = L.ONE
    spec = _empty_spec(L.D, (L.q_nibs + L.state_nibs) * 2 + 6)
    u = 0
    bz = (L.BZ, 1.0, 0.0)
    for j in range(L.q_nibs):
        u = alu._guard(spec, u, [bz], {L.Q + j: -1.0}, 0.0, L.Q + j, 1.0)
    for s in range(L.state_nibs):
        u = alu._guard(spec, u, [bz], {L.R + s: -1.0}, 0.0, L.R + s, 1.0)
    return alu._truncate(spec, u, L.D)


# ===========================================================================
# The public builder + runner.
# ===========================================================================
_ATTN = "attn"          # marks the CAM (attention) block in the block list.


def build_div_automaton_attn(
        b: int, in_width: int = 32) -> Tuple[List, _AttnAutomatonLayout]:
    """Build the attention-CAM constant-divisor divmod automaton for divisor ``b``.

    Returns ``(blocks, layout)``.  The block list interleaves FFN specs (dicts) and
    the attention-CAM sentinel ``_ATTN`` (the CAM forward is driven by the layout's
    baked ``_AttnCAM``):

        [ QENC(ffn), CAM(attn), COMMIT(ffn) ] × in_nibs   (1 attention lookup/step)
        [ BZ-GATE(ffn) ]                                  (b==0 -> (0,0))

    Depth = ``3·in_nibs + 1`` blocks; **attention-lookups-per-step = 1** (the single
    CAM block; QENC + COMMIT are trivial FFN glue).  For a 32-bit dividend (8
    nibbles): ``3·8 + 1 = 25`` blocks with **8 attention lookups total**.
    """
    L = _AttnAutomatonLayout(b, in_width=in_width)
    blocks: List = []
    for step in range(L.in_nibs):
        blocks.append(_qenc_block(L))
        blocks.append(_ATTN)
        if b != 0:
            blocks.append(_commit_block(L, step))
    blocks.append(_bz_gate_block(L))
    return blocks, L


def run_blocks(blocks: List, L: _AttnAutomatonLayout, a: int) -> Tuple[int, int]:
    """Run the attention-CAM automaton over dividend ``a`` and return ``(q, r)``.

    Streams the ``in_nibs`` MSB-first input nibbles; before each step's QENC block
    ``D_IN`` is set.  The CAM block (``_ATTN``) runs the REAL softmax1 attention
    forward via ``_cam_block_forward``; FFN blocks run the SwiGLU residual forward."""
    x = L.load(a)
    bi = 0
    for step in range(L.in_nibs):
        x = x.clone()
        x[L.D_IN] = float(L.input_nibble(a, step))
        # QENC (ffn).
        x = _apply_ffn(x, blocks[bi]); bi += 1
        # CAM (attention).
        assert blocks[bi] == _ATTN
        x = _cam_block_forward(x, L); bi += 1
        # COMMIT (ffn) — present unless b == 0.
        if L.b != 0:
            x = _apply_ffn(x, blocks[bi]); bi += 1
    # BZ gate (final ffn block).
    x = _apply_ffn(x, blocks[bi])
    return L.decode_q(x), L.decode_r(x)


# ===========================================================================
# Measurement helpers.
# ===========================================================================
def measure(b: int, in_width: int = 32) -> Dict[str, object]:
    blocks, L = build_div_automaton_attn(b, in_width=in_width)
    ffn_nz = sum(_nz(w) for w in blocks if w is not _ATTN)
    kv_nz = L.cam.nz()
    per_step = 3 if b != 0 else 2
    return {
        "b": b,
        "total_blocks": len(blocks),
        "attn_blocks_per_step": 1,                 # the single CAM lookup
        "blocks_per_step": per_step,               # QENC + CAM + COMMIT
        "ffn_nz": ffn_nz,                          # the trivial glue (QENC/COMMIT/BZ)
        "kv_nz": kv_nz,                            # the attention KV table cost
        "nz": ffn_nz + kv_nz,
        "n_kv_rows": L.cam.n_rows,                 # b*16 reachable (R,d) rows
        "cam_dim": L.cam.dim,                      # key/query dim (n_bits + bias)
        "state_nibs": L.state_nibs,
        "q_nibs": L.q_nibs,
        "r_nibs": L.r_nibs,
        "table_width": (b if b > 0 else 0) * 16,   # b*16 KV rows
        "kq_magnitude": L.cam.key_query_magnitude(),
        "worst_exact_weight": L.cam.worst_exact_weight() if b > 0 else 1.0,
        "in_nibs": L.in_nibs,
    }


# feasibility: the b*16-row KV table is the dominant cost (as the FFN automaton's
# b*16 guard table was).  Below this the CAM stays a sharp hardmax and small; above
# it the b-independent digit-recurrence wins.
FEASIBLE_TABLE_WIDTH = 1 << 14      # b*16 <= 16384  ->  b <= 1024.


def feasible_b_max(width_cap: int = FEASIBLE_TABLE_WIDTH) -> int:
    return width_cap // 16


# ===========================================================================
# Reporting driver — build + byte-exact-verify the battery, print the table.
# ===========================================================================
_BATTERY = sorted({2, 3, 7, 9, 10, 16, 60, 100, 255, 256, 1000,
                   11, 13, 17, 97, 251, 1, 4, 8, 32, 64, 128, 512, 1024})


def report(battery: List[int] = _BATTERY, n_random: int = 2000,
           in_width: int = 32) -> bool:
    """Print the metric + byte-exact table (both q and r) through the REAL softmax1
    attention forward.  Returns True iff EVERY case is byte-exact."""
    import random
    mask = (1 << in_width) - 1
    hdr = (f'{"b":>6} {"blk":>4} {"a/s":>3} {"kvrow":>6} {"kdim":>4} {"snib":>4} '
           f'{"q/r":>5} {"wmin":>7} {"|kq|":>5}  q&r byte-exact')
    print(hdr)
    all_ok = True
    for b in battery:
        blocks, L = build_div_automaton_attn(b, in_width=in_width)
        rng = random.Random(1000 + b)
        edges = [e & mask for e in (0, 1, b - 1, b, b + 1, mask)]
        tests = edges + [rng.randint(0, mask) for _ in range(n_random)]
        npass = sum(1 for a in tests if run_blocks(blocks, L, a) == (a // b, a % b))
        m = measure(b, in_width=in_width)
        ok = (npass == len(tests))
        all_ok = all_ok and ok
        print(f'{b:>6} {m["total_blocks"]:>4} {m["attn_blocks_per_step"]:>3} '
              f'{m["n_kv_rows"]:>6} {m["cam_dim"]:>4} {m["state_nibs"]:>4} '
              f'{str(m["q_nibs"])+"/"+str(m["r_nibs"]):>5} '
              f'{m["worst_exact_weight"]:>7.4f} {m["kq_magnitude"]:>5.0f}  '
              f'{npass}/{len(tests)}' + ('' if ok else '  FAIL'))
    # b == 0 -> (0, 0)
    blocks, L = build_div_automaton_attn(0, in_width=in_width)
    z_ok = all(run_blocks(blocks, L, a) == (0, 0) for a in (0, 1, 7, 100, mask))
    all_ok = all_ok and z_ok
    m = measure(0, in_width=in_width)
    print(f'{0:>6} {m["total_blocks"]:>4} {m["attn_blocks_per_step"]:>3} '
          f'{m["n_kv_rows"]:>6} {m["cam_dim"]:>4} {"-":>4} {"-":>5} '
          f'{"-":>7} {m["kq_magnitude"]:>5.0f}  b=0->(0,0): {z_ok}')
    print(f'\nfeasible b range: b*16 KV rows <= {FEASIBLE_TABLE_WIDTH} '
          f'-> b <= {feasible_b_max()}   (digit-recurrence past that)')
    print(f'ALL BYTE-EXACT: {all_ok}')
    return all_ok


if __name__ == "__main__":
    report()
