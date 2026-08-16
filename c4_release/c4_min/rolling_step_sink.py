"""Rolling-per-step sink for the blogspec KV-memory CAM (BLOG_SPEC §Memory).

Problem (measured, not re-derived)
==================================
The spec's ZFOD default is softmax1's ``+1`` — an attention sink at logit 0 that
is *ALiBi-EXEMPT*.  If instead you use a single GLOBAL BOS sink token at absolute
position 0 under plain softmax + ALiBi (``softmax1_via_bos_sink`` places it at a
FIXED position 0), its effective logit is ``C_sink - slope*(pos_q - 0)``, which
DECAYS ``-> -inf`` as the query position ``pos_q`` grows into a long program.
Once ``slope*pos_q > EFF`` the sink drops below a mediocre recent 1-bit-off store
and ZFOD BREAKS (a no-match read returns a spurious value).

The rolling-per-step sink (this module)
=======================================
Designate a sink token at the START of every VM step.  Under plain softmax +
ALiBi its effective logit is ``C_sink - slope*(pos_q - step_start)`` — and since
every in-step query is at most ``W_step`` (=30) tokens past its step's start, the
sink's penalty is BOUNDED by ``slope*W_step`` (a CONSTANT, independent of program
depth).  So the sink never decays: ZFOD holds at any program position.  This is
PLAIN softmax + ALiBi throughout — NO softmax1, NO ALiBi exemption.

Realisation
===========
We keep the real ``blogspec_model`` §Memory head verbatim (the ±smag address CAM,
the ZFOD-bias channel, the store-role gate, the value relay).  The ONLY change is
the SINK: instead of softmax1's implicit +1 column, we prepend a real *sink key
row* into the KV stream at each step start.  The sink row:
    * carries a CONTENT score ``C_sink`` against every load query (a dedicated
      sink channel: query keys +qs, sink row keys +qs so Q.K = C_sink),
    * is a real token at position ``step_start`` so ALiBi penalises it
      ``-slope*(pos_q - step_start)`` like any other key,
    * has value 0 (so when it wins, the read is 0 = ZFOD),
    * loses to every EXACT in-horizon address match (``+EFF`` content dominates).

Because there is a fresh sink row at every step, the query's nearest sink is at
most ``W_step`` tokens back -> bounded penalty.  Plain softmax over
``[...real keys..., sink_row_for_this_step]`` reproduces the ZFOD default the way
softmax1 does, but WITHOUT the position-0 decay.

Calibration
===========
``C_sink`` is the sink's content logit.  ZFOD (no exact match -> 0) needs the
sink to BEAT the worst competitor — the most-recent 1-bit-off store, whose score
is ``-EFF - slope*d_near`` (``d_near`` >= 1).  The sink's in-step score is
``>= C_sink - slope*W_step``.  So ZFOD holds when

        C_sink - slope*W_step  >  -EFF - slope*d_near        (d_near >= 0)
   <=>  C_sink                 >  -EFF + slope*W_step        (worst case d_near=0)

and an exact match must still beat the sink:

        +EFF - slope*d_match   >  C_sink - slope*W_step
   <=>  C_sink                 <  EFF - slope*(d_match - W_step)

so any ``C_sink in ( -EFF + slope*W_step , EFF - slope*(d_match - W_step) )``
works.  ``C_sink = 0`` (matching softmax1's sink level) satisfies the ZFOD bound
with margin ``EFF - slope*W_step`` and the match bound out to ``d_match < EFF/slope
+ W_step`` — i.e. the SAME recall horizon as softmax1 (§the horizon is set by EFF,
not by the sink).  ``C_sink = 0`` is the calibration; NO exemption is introduced.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .blogspec_memory import (
    ADDR_BITS, EFF, MEM_ALIBI_SLOPE, PEN_GATE, MemoryLayout,
    address_bits, bake_memory_head, build_memory_model, _decode_byte,
)

W_STEP = V.FRAME_LEN            # 30 tokens per VM step (the sink horizon bound)


# ---------------------------------------------------------------------------
# The rolling-sink layout: base MemoryLayout + a dedicated SINK channel + a
# SINK-ROW flag (marks a row as the per-step sink so W_k/W_v build the sink key).
# ---------------------------------------------------------------------------
class RollingSinkLayout(MemoryLayout):
    def __init__(self, n_heads: int = 4):
        super().__init__(n_heads=n_heads)
        # Re-open the layout to append two bands, then re-pad to n_heads.
        # (MemoryLayout finished with D aligned + _pad dims; we append AFTER D by
        #  reusing the _band machinery on a fresh tail.)
        self._off = self.D
        self.IS_SINK = self._scalar("IS_SINK")     # 1.0 on a per-step sink row
        self.SINK_Q = self._scalar("SINK_Q")       # query drives the sink channel
        while self._off % n_heads != 0:
            self._scalar(f"_pad2_{self._off}")
        self.D = self._off


def bake_rolling_sink_head(attn, L: RollingSinkLayout, head: int = 0,
                           C_sink: float = 0.0) -> None:
    """Bake head ``head`` as the §Memory CAM with a ROLLING per-step sink.

    Identical to ``bake_memory_head`` (address CAM + ZFOD bias + store-role gate
    + value relay) PLUS a dedicated sink channel: a load query keys ``+qs`` on
    the sink channel and a sink row keys ``+ks`` with ``qs*ks*hs = C_sink`` (the
    sink's content logit).  The sink row's value is 0 (ZFOD).  The store-role
    gate would normally drive the sink row (a non-store) to ``-PEN``; we EXEMPT
    the sink row from that gate (it is a legitimate default candidate, not a
    masquerading store) by having the sink row also assert IS_STORE=0 but key the
    role channel to 0 via a sink-specific term.  Because the sink is a real KV
    row at the step-start position, ALiBi penalises it from THAT position.
    """
    # First lay down the exact §Memory CAM.
    bake_memory_head(attn, L, head=head)

    # Read EFF/PEN_GATE DYNAMICALLY from the module (not the import-time snapshot)
    # so a rebake under a custom EFF (the small-EFF verification regime) uses the
    # SAME smag/p as bake_memory_head just did — otherwise the exemption cancels
    # against a stale magnitude and leaks a huge address-CAM score.
    import c4_min.blogspec_memory as _BM
    _EFF = _BM.EFF
    _PEN_GATE = _BM.PEN_GATE

    hs = attn.scale
    HD = attn.head_dim
    base = head * HD
    smag = (_EFF / hs) ** 0.5                 # per-bit key/query magnitude (§CAM)

    # --- ADDRESS-CAM EXEMPTION for the sink row. ------------------------------
    # bake_memory_head keys every address channel b as  2*smag*ADDR_BIN[b]
    # - smag*ONE.  A SINK row carries ADDR_BIN=0 and ONE=1, so its key is -smag
    # on every bit — which (against a load query) leaks a large partial-address
    # match (up to +n*EFF).  Cancel the -smag*ONE term on a sink row by keying
    # +smag*IS_SINK on each address channel, so a sink row's address key is
    # exactly 0 on every bit -> zero address contribution.  Its content is then
    # ONLY the dedicated sink channel below (= C_sink).
    for b in range(ADDR_BITS):
        attn.W_k[base + b, L.IS_SINK] = smag

    # --- sink content channel: a dedicated CAM channel for C_sink. ------------
    # A load queries +qs (via SINK_Q), the sink row keys +ks (via IS_SINK), so
    # Q.K on this channel = qs*ks*hs = C_sink.  C_sink can be 0 (level = softmax1
    # sink); the channel still exists so the sink row is a *real* candidate.
    cS = base + ADDR_BITS + 2               # channels 0..n-1 addr, n bias, n+1 role
    if C_sink != 0.0:
        mag = (abs(C_sink) / hs) ** 0.5
        sign = 1.0 if C_sink > 0 else -1.0
        attn.W_q[cS, L.SINK_Q] = mag
        attn.W_k[cS, L.IS_SINK] = sign * mag
    # If C_sink == 0 the sink row's content logit is exactly 0 (softmax1 level);
    # no channel weight needed — the ZERO content is the default.

    # --- ZFOD-bias EXEMPTION for the sink row. --------------------------------
    # bake_memory_head keys channel n as +kb*IS_STORE (the -(n-1)EFF bias vs a
    # load's -qb*IS_LOAD).  A sink row is a non-store (IS_STORE=0) so it gets 0
    # here already — no action needed (documented for completeness).

    # --- store-role gate EXEMPTION for the sink row. --------------------------
    # bake_memory_head keyed channel n+1 as  -p*ONE + p*IS_STORE  so a non-store
    # row scores -p (=> -PEN after the load's +p query).  The sink row is a
    # non-store, so it would be gated OUT.  Add +p*IS_SINK on that key channel so
    # a sink row nets 0 there (it is a legitimate default, not a fake store).
    cR = base + ADDR_BITS + 1
    p = (_PEN_GATE / hs) ** 0.5
    attn.W_k[cR, L.IS_SINK] = p

    # The sink row's VALUE is 0 on the AX relay: bake_memory_head copies VAL_NIB
    # -> AX; the sink row carries VAL_NIB=0, so a winning sink relays 0 (ZFOD).
    # (No extra weights needed — VAL_NIB defaults to 0 on the sink overlay.)


def build_rolling_sink_model(n_heads: int = 4, C_sink: float = 0.0):
    """Bake a blogspec_model with the rolling-sink §Memory head on block 0.

    Uses ``sink="softmax1"``? NO — the whole point is PLAIN softmax.  We use
    ``sink="bos_sink"``? Also no — that prepends a FIXED position-0 column.  The
    rolling sink is a real KV ROW we inject at each step start, so the model runs
    PLAIN softmax over the real key set (softmax1 with NO +1: we pass
    ``sink="softmax1"`` but the +1 must be OFF).  We therefore run the attention
    in PLAIN-softmax mode via a tiny shim: the rolling sink row IS the sink, so
    there must be no additional implicit +1.  We achieve plain softmax by using
    the model's ``sink="bos_sink"`` machinery but supplying our OWN sink row and
    turning the model's auto-sink off (handled in RollingSinkMemory._forward).
    """
    from .blogspec_model import Transformer
    L = RollingSinkLayout(n_heads=n_heads)
    model = Transformer(dim=L.D, n_heads=n_heads, hidden=max(8, NIB_PER_REG),
                        n_blocks=1, vocab=V.VOCAB, max_seq_len=8192,
                        positional="alibi", norm="none", sink="softmax1")
    with torch.no_grad():
        E = torch.zeros(V.VOCAB, L.D)
        E[:, L.ONE] = 1.0
        model.embed.copy_(E)
        for blk in model.blocks:
            for pp in (blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o):
                pp.zero_()
        bake_rolling_sink_head(model.blocks[0].attn, L, head=0, C_sink=C_sink)
    return model, L
