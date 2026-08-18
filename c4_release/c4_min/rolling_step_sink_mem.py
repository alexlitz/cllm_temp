"""RollingSinkMemory — the KV memory driver over the rolling-per-step sink.

Runs the REAL blogspec_model §Memory CAM head, but with the SINK realised as a
per-step KV row (plain softmax + ALiBi, NO softmax1 / NO exemption) instead of
softmax1's implicit +1.  The three sink modes are all driven through ONE forward
so the head-to-head is byte-exact on identical machinery:

    sink="softmax1"   : reference — softmax1's ALiBi-exempt +1 (no injected row)
    sink="global_bos" : NEGATIVE CONTROL — a single sink row at absolute pos 0,
                        plain softmax + ALiBi (decays with depth)
    sink="rolling"    : THE FIX — a fresh sink row at each step start, plain
                        softmax + ALiBi (bounded penalty, no decay)

The store/load overlays reuse blogspec_memory's exact ±smag CAM bands; only the
sink handling differs.  We compute the attention score matrix directly (as
KVMemory.peek_weights does) so we can apply plain softmax with the injected sink
row, keeping the CAM weights byte-identical to the baked §Memory head.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .blogspec_memory import ADDR_BITS, EFF, MEM_ALIBI_SLOPE, address_bits, _decode_byte
from .rolling_step_sink import (
    RollingSinkLayout, W_STEP, bake_rolling_sink_head, build_rolling_sink_model,
)


class RollingSinkMemory:
    """KV memory whose ZFOD sink is a rolling per-step row (plain softmax+ALiBi).

    ``sink_mode`` selects the head-to-head variant. Each entry in ``self._rows``
    is ``(pos, kind, overlay)`` where ``kind`` in {"store","sink"} and ``pos`` is
    the absolute token position of that row.  A store consumes W_STEP token
    positions per step (one full frame) so positions match the real driver; the
    sink row for step s sits at that step's START position.  Loads are queries,
    not persisted rows.
    """

    def __init__(self, sink_mode: str = "rolling", C_sink: float = 0.0,
                 n_heads: int = 4, eff: Optional[float] = None,
                 slope: Optional[float] = None):
        assert sink_mode in ("softmax1", "global_bos", "rolling"), sink_mode
        self.sink_mode = sink_mode
        self.C_sink = C_sink
        self.model, self.L = build_rolling_sink_model(n_heads=n_heads,
                                                       C_sink=C_sink)
        self.attn = self.model.blocks[0].attn
        if eff is not None or slope is not None:
            self._rebake(eff, slope)
        self.eff = eff if eff is not None else EFF
        self.slope = slope if slope is not None else MEM_ALIBI_SLOPE
        # persisted KV rows (stores + injected sinks). Each: (pos, kind, overlay)
        self._rows: List[Tuple[int, str, dict]] = []
        self._step = 0                   # VM-step counter (drives step_start)
        # global-BOS sink row lives once at position 0.
        if sink_mode == "global_bos":
            self._rows.append((0, "sink", self._sink_overlay()))

    # -- rebake for a custom EFF/slope (for the small-EFF verification regime) --
    def _rebake(self, eff, slope):
        import c4_min.blogspec_memory as BM
        old_eff, old_slope = BM.EFF, BM.MEM_ALIBI_SLOPE
        old_bias, old_pen = BM.BIAS, BM.PEN_GATE
        if eff is not None:
            BM.EFF = float(eff)
            BM.BIAS = (BM.ADDR_BITS - 1) * BM.EFF
            BM.PEN_GATE = 100.0 * BM.ADDR_BITS * BM.EFF
        if slope is not None:
            BM.MEM_ALIBI_SLOPE = float(slope)
        with torch.no_grad():
            for pp in (self.attn.W_q, self.attn.W_k, self.attn.W_v, self.attn.W_o):
                pp.zero_()
            bake_rolling_sink_head(self.attn, self.L, head=0, C_sink=self.C_sink)
        BM.EFF, BM.MEM_ALIBI_SLOPE = old_eff, old_slope
        BM.BIAS, BM.PEN_GATE = old_bias, old_pen

    # -- overlays ----------------------------------------------------------
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
        # the load query drives the sink channel so the sink row is a candidate.
        ov[L.SINK_Q] = 1.0
        return ov

    def _sink_overlay(self) -> dict:
        L = self.L
        # a sink row: IS_SINK=1 (keys the sink channel + role-gate exemption),
        # value nibbles 0 (VAL_NIB defaults to 0), NOT a store.
        return {L.IS_SINK: 1.0}

    # -- stepping ----------------------------------------------------------
    def _advance_step(self) -> int:
        """Advance to a new VM step; return its START absolute position.

        Each step occupies W_STEP token positions.  For rolling mode we inject a
        fresh sink row at the step start.
        """
        step_start = self._step * W_STEP
        if self.sink_mode == "rolling":
            self._rows.append((step_start, "sink", self._sink_overlay()))
        self._step += 1
        return step_start

    def store(self, addr: int, value: int, *, char: bool = False) -> None:
        """SI/SC: persist a store row inside a fresh step frame."""
        step_start = self._advance_step()
        # the store's MEM marker sits near the frame end (offset 20 in the frame),
        # matching build_step_frame; we place the store row at step_start+20.
        pos = step_start + 20
        self._rows.append((pos, "store", self._store_overlay(addr, value, char)))

    def free(self, addr: int) -> None:
        self.store(addr, 0)

    # -- the forward: plain softmax (or softmax1) over rows + a load query -----
    def _forward_load(self, addr: int, char: bool) -> Tuple[int, torch.Tensor]:
        step_start = self._advance_step()
        q_pos = step_start + 5            # a load fires early in its frame
        return self._run_load(addr, char, q_pos)

    def _run_load(self, addr: int, char: bool,
                  q_pos: int) -> Tuple[int, torch.Tensor]:
        """Plain-softmax (rolling/global_bos) or softmax1 (reference) load forward
        over the current ``self._rows`` with an EXPLICIT query position ``q_pos``.

        Placing the query at an explicit absolute position lets a caller model an
        INTRA-STEP gather (q_pos a few tokens after a same-step key) or a deep
        recall (q_pos far past the store) without going through _advance_step —
        the position is what ALiBi sees, so this is the single byte-exact forward
        both roles share.  Returns (decoded value, attention weights over rows)."""
        L, attn = self.L, self.attn
        # assemble rows (positions, overlays) + the query row.
        rows = list(self._rows)
        # Build residual for [rows..., query].
        S = len(rows) + 1
        x = torch.zeros(1, S, L.D)
        x[:, :, L.ONE] = 1.0             # the ONE lane (needed for ±smag bias)
        positions = []
        for i, (pos, kind, ov) in enumerate(rows):
            for dim, val in ov.items():
                x[0, i, dim] = val
            positions.append(pos)
        # query row
        qov = self._load_overlay(addr, char)
        for dim, val in qov.items():
            x[0, S - 1, dim] = val
        positions.append(q_pos)
        pos_t = torch.tensor(positions, dtype=torch.float)

        with torch.no_grad():
            B, _, D = x.shape
            H, HD = attn.n_heads, attn.head_dim
            Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
            K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
            Vv = F.linear(x, attn.W_v).view(B, S, H, HD).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale  # [B,H,S,S]
            # ALiBi over ABSOLUTE positions (head 0 slope = MEM_ALIBI_SLOPE).
            dist = (pos_t.unsqueeze(1) - pos_t.unsqueeze(0)).abs()       # [S,S]
            scores = scores - attn.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(0).unsqueeze(0)
            # causal mask over absolute positions (query attends to pos<=q_pos).
            mask = (pos_t.unsqueeze(0) > pos_t.unsqueeze(1))             # [S,S]
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            # -- the sink: plain softmax over rows (rolling / global_bos), or
            #    softmax1's +1 (reference).  The query row attends to itself with
            #    score ~0 (no CAM channels align) so we MASK the self row out to
            #    keep the sink the sole ZFOD default (a self-attend would leak the
            #    query's own residual; the real head relies on softmax1 dropping
            #    it via the +1 — here the sink row plays that role).
            self_idx = S - 1
            scores[..., self_idx, self_idx] = float("-inf")
            if self.sink_mode == "softmax1":
                from .blogspec_model import softmax1
                w = softmax1(scores, dim=-1)
            else:
                w = torch.softmax(scores, dim=-1)
            out = torch.matmul(w, Vv)                    # [B,H,S,HD]
            out = out.transpose(1, 2).contiguous().view(B, S, D)
            state = (x + F.linear(out, attn.W_o))[0, -1]   # query row residual
        val = 0
        for bi in range(4):
            val |= _decode_byte(state, L, L.AX, bi) << (8 * bi)
        return ((val & 0xFF) if char else val), w[0, 0, -1]

    def load(self, addr: int, *, char: bool = False) -> int:
        v, _ = self._forward_load(addr, char)
        return v

    def load_at(self, addr: int, q_pos: int, *, char: bool = False) -> int:
        """Load ``addr`` with an EXPLICIT query token position ``q_pos`` (does NOT
        advance the step counter).  Used to model an intra-step gather or a deep
        recall at a chosen program depth over the current rows."""
        v, _ = self._run_load(addr, char, q_pos)
        return v

    def load_weights(self, addr: int, *, char: bool = False):
        """Return (value, attention_weights_over_rows) for diagnostics."""
        return self._forward_load(addr, char)
