"""DIRECT-LOCAL-CAM (C4_DIRECT_LOCAL_CAM): O(1) direct gather for the block-0
register-ingest heads — the LAST attention floor of the composed doom step.

#841 profiled the composed doom step (2.748 ms with direct-CAM + banded-local +
fused-FFN) and found ATTENTION is ~97% of the wall.  direct-CAM already resolves the
GLOBAL memory heads (LI/LC, stack-pop, LEV, code-fetch) O(1) via the perfect draft
(``resolve_load_rows``).  What REMAINS is the ~20 LOCAL register-ingest heads on
block 0, still doing a windowed softmax (banded kernel, O(Sq*W)) over the last-W rows.

THE KEY INSIGHT (the user's "use the structure to optimize attention"): the draft (the
perfect c4 VM) knows the EXACT register frame each ingest head reads.  Ingest head
``h = r*4 + bi`` (register r in [PC,AX,SP,BP,STACK0], byte bi) content-addresses the
role-(r,bi) token of the LATEST emitted frame — the ALiBi recency (slope 6.0) + the
huge match logit (INGEST_EFF=4000) make the softmax1 a HARD SINGLE-ROW WINNER (probed:
760/760 heads, weight exactly 1.0 on one key, worst margin 1.000000).  So the head's
attention output is EXACTLY that one token's value nibbles — which the draft knows: the
register byte of the frame the query row follows.  We DIRECT-GATHER it (write the
head's value slots) and skip the softmax.

WHY BYTE-EXACT (by construction)
--------------------------------
The ingest head's value projection reads ONLY ``CUR_NIB[0/1]`` (the winner token's two
nibbles) into head-value slots 0 and 1 (``bake_frame_ingest``: ``W_v[base+0,CUR_NIB+0]``
= ``W_v[base+1,CUR_NIB+1]`` = 1), and ``W_o`` maps them into the register nibble band
``reg_base + 2*bi + {0,1}``.  With softmax1 weight ~=1 on the single winner and ~=0
elsewhere (the +1 sink -> 0 for a role that is always present), the head output is
EXACTLY ``[lo_nib, hi_nib, 0, ...]`` for that winner byte.  We reconstruct that vector
from the draft's emitted frame byte and set the head's ``out`` slot — so ``x + W_o(out)``
is bit-identical to the banded/softmax path.  Every OTHER head on block 0 (there are
none live-local besides these 20; the zero-value heads output 0 regardless) runs its
normal path.

THE RESOLVED FRAME (what byte each head reads)
----------------------------------------------
The query row of step ``s`` sits at ``draft.win_starts[s]`` and follows the frame
EMITTED by step ``s-1`` (the driver appends step s's frame AFTER recording its query
row), or the INIT frame for ``s == 0`` (pc=ax=0, sp=bp=SP_INIT, STACK0=0).  The emitted
frame's role tokens carry: PC/AX/SP/BP = ``frames[s-1]{pc,ax,sp,bp}``; the STACK0 role
tokens (frame slots 25..28) carry the frame's emitted MEM_VAL — which is the STORE value
``s_val`` on a store step, else the STACK0 mirror ``stk``.  We mirror ``_build_frame``
exactly so the STACK0 byte is the EMITTED one.

Composition: this is the block-0 counterpart to ``direct_cam_batched`` (which handles the
global CAM blocks).  Both install a forward wrapper; direct-local defers no head to a
windowed forward (block 0 has no genuine global head), so it fully computes block 0's
attention output by gather.  Gated ``C4_DIRECT_LOCAL_CAM`` default OFF -> the banded /
softmax ingest path (golden 069cc32f unchanged).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch

from . import blogspec_vocab as V
from .nibble_pure_forward import (
    INGEST_REGS, BYTES_PER_REG, INGEST_RECENCY, _FRAME_ROLE_SLOTS)


def direct_local_cam_enabled() -> bool:
    """C4_DIRECT_LOCAL_CAM (DEFAULT OFF): direct-index gather for the block-0
    register-ingest heads in the batched verify span.  OFF -> the banded/softmax ingest
    (byte-exact golden path).  ON -> each ingest head direct-gathers its resolved frame
    byte per query row (no windowed score) -> the last local-attention floor collapses."""
    return os.environ.get("C4_DIRECT_LOCAL_CAM", "0") not in ("0", "", "false", "False")


def _block0_sq_chunk() -> int:
    """C4_BLOCK0_SQ_CHUNK (default 0 == OFF -> whole-span forward, byte-identical to the
    prior direct-local path).  When > 0, the block-0 direct-local forward processes the
    span's ``S = K*30`` rows in CHUNKS of this many rows, so it NEVER materialises the
    full ``[1, H, S, HD]`` K/V/out tensors (nor the ``[1, S, D]`` out2 / W_o output) at
    once — the ONLY per-forward tensors whose VRAM scales as O(K*30).

    WHY THIS IS THE CAP-LIFTER.  Under ``frozen_skip`` (C4_FROZEN_ROW_SKIP) block 0 is the
    SOLE block that runs over all S = K*30 rows (``_frozen_skip_cut`` == 1 on the doom
    config): every OTHER block runs over the K query rows only (O(K)).  So block 0's
    whole-span Q/K/V/out is the LAST O(K*30) tensor on the forward, and it is what caps
    eff_K at ~8k on a 24 GB card (measured: 12.2 GB peak at K=8192, OOM at K=16384).
    Chunking block 0's S axis holds block-0 peak at O(chunk*D) REGARDLESS of K, so the
    per-forward VRAM stops scaling with K*30 and K can grow far past the ceiling.

    BYTE-EXACT (by construction).  The direct-local forward is a PER-ROW independent map:
    row ``i``'s residual is ``x[i] + W_o(out[i])`` where ``out[i]`` is the gathered
    ingest-head value (query rows) or 0 (frozen rows), and its K/V is ``W_{k,v}(x[i])`` —
    NONE of which reads any other row (the ingest heads are resolved by a direct gather,
    NOT by scoring the span, so there is no cross-row attention).  Splitting the contiguous
    S axis into chunks and reassembling the per-chunk residual / K / V is therefore
    bit-identical to the whole-span forward (validated L-inf == 0 in the verify battery).
    512 rows/chunk keeps block-0 peak flat.
    """
    try:
        return int(os.environ.get("C4_BLOCK0_SQ_CHUNK", "0"))
    except ValueError:
        return 0


def _block0_drop_dead_kv() -> bool:
    """C4_BLOCK0_DROP_DEAD_KV (default 1 == ON when Sq-chunking): return a ``None`` cache
    for block 0 in the chunked path instead of the full ``[1, H, S, HD]`` K/V.

    Block 0's KV is PROVABLY DEAD under direct-local-CAM: (1) block 0's OWN future
    attention is the direct GATHER (``direct_local_forward`` ignores ``past_kv`` entirely
    — it resolves the ingest heads from the draft, never by scoring the cache), and (2) NO
    OTHER block reads block-0's cache (the driver's per-block cache is strictly positional,
    ``caches[b]`` <-> ``blocks[b]``).  So dropping it is byte-safe — the EXACT contract
    ``dead_block_forward`` / the block-MoE skip already rely on (``_commit_span`` tolerates
    a ``None`` new_kv; only ``caches[0].size()`` is read, and that feeds a cosmetic
    ``max_cache_size`` STAT, never a score).  Dropping the dead K/V is the OTHER half of
    the VRAM win: with it OFF the chunked path still holds the full K/V (only out/out2/W_o
    are chunked); with it ON no O(K*30) K/V is held at all.  Set to 0 to force the full
    (byte-identical) K/V cache in the chunked path."""
    return os.environ.get("C4_BLOCK0_DROP_DEAD_KV", "1") not in ("0", "", "false", "False")


# ===========================================================================
# 1. THE INGEST-HEAD MAP: which heads on block 0 are the register-ingest heads
#    (recency slope == INGEST_RECENCY, a live value head), and which (register,
#    byte) each reads.  Mirrors ``bake_frame_ingest`` (h = r*4 + bi).
# ===========================================================================
def ingest_head_map(attn, slope_tol: float = 1e-3) -> Dict[int, Tuple[int, int]]:
    """Return ``{head_idx: (r_idx, byte_idx)}`` for block-0's register-ingest heads.

    A head is an ingest head iff its ALiBi recency slope is ``INGEST_RECENCY`` (the
    proven-local frame-ingest signature — same test ``local_attention.classify_heads``
    uses).  ``bake_frame_ingest`` assigns head ``h`` role ``h`` => ``(h//4, h%4)``."""
    out: Dict[int, Tuple[int, int]] = {}
    for h in range(attn.n_heads):
        if abs(float(attn.alibi_slopes[h]) - INGEST_RECENCY) <= slope_tol:
            r_idx, bi = divmod(h, BYTES_PER_REG)
            if r_idx < len(INGEST_REGS):
                out[h] = (r_idx, bi)
    return out


# ===========================================================================
# 2. THE RESOLVED FRAME per ABSOLUTE query-row position: the 5 register values of
#    the frame the query row FOLLOWS (the exact bytes the ingest heads read).
# ===========================================================================
# Frame-local slot of role (r_idx, byte_idx), inverted from ``_FRAME_ROLE_SLOTS``.
_ROLE_TO_LOCAL: Dict[int, int] = {role: local for local, role in _FRAME_ROLE_SLOTS.items()}


class ResolvedFrames:
    """Per query-row resolved ingest bytes, in a VECTORIZED form for O(1) gather.

    * ``pos``            : LongTensor [n_query]  absolute stream position of each query row
    * ``pos_to_local``   : Dict[abs_pos -> index into ``pos``] (for span-local mapping)
    * ``nib_lo/nib_hi``  : Float [n_query, N_ROLES]  the low/high nibble each role-h ingest
                           head reads at that query row (role index == head index).
    The forward gathers, for the query rows present in a span, the matching rows of
    ``nib_lo/nib_hi`` and scatters them into the ingest heads' value slots — a single
    vectorized index op, NO Python per-row loop (the earlier loop was 4x SLOWER than the
    banded kernel; the gather must be a tensor op to win)."""

    def __init__(self, pos, nib_lo, nib_hi):
        self.pos = pos                      # [n_query] long
        self.nib_lo = nib_lo                # [n_query, N_ROLES] float
        self.nib_hi = nib_hi                # [n_query, N_ROLES] float
        self.pos_to_local = {int(p): i for i, p in enumerate(pos.tolist())}


def build_resolved_frames(draft, device="cpu") -> ResolvedFrames:
    """Resolve, per step's query row, the ``N_ROLES`` register-role BYTES the ingest
    heads read, by decoding the LATEST complete 30-token frame ending before the query
    row directly from ``draft.tokens`` — as VECTORIZED nibble tensors.

    Ground truth (matches the softmax winner proven by ``_agent_dlc_probe``): head ``h``
    content-addresses the role-``h`` token of the latest emitted frame; its value is that
    byte token's two nibbles (byte tokens 0..255 embed their own value == the token id).
    Role index == head index (``bake_frame_ingest``: head ``h`` -> role ``h``).
    """
    import torch as _t
    toks = draft.tokens
    FL = V.FRAME_LEN
    NR = len(_ROLE_TO_LOCAL)
    # role -> frame-local slot, as a fixed vector for a vectorized frame decode.
    local_of_role = _t.tensor([_ROLE_TO_LOCAL[r] for r in range(NR)], dtype=_t.long)
    positions = []
    bytes_rows = []                          # each: [NR] int byte per role
    tk = _t.tensor(toks, dtype=_t.long)
    for s in range(draft.step_count):
        qpos = draft.win_starts[s]
        fstart = qpos - FL + 1
        if fstart >= 0 and toks[fstart] == V.REG_PC:
            row = tk[fstart + local_of_role]         # [NR] token ids (== byte values)
            row = _t.where((row >= 0) & (row <= 255), row, _t.zeros_like(row))
        else:
            row = _t.zeros(NR, dtype=_t.long)
        positions.append(qpos)
        bytes_rows.append(row)
    if bytes_rows:
        B = _t.stack(bytes_rows, 0)                  # [n_query, NR]
    else:
        B = _t.zeros(0, NR, dtype=_t.long)
    nib_lo = (B & 0xF).to(dtype=_t.float32)
    nib_hi = ((B >> 4) & 0xF).to(dtype=_t.float32)
    pos = _t.tensor(positions, dtype=_t.long)
    return ResolvedFrames(pos.to(device), nib_lo.to(device), nib_hi.to(device))


# ===========================================================================
# 3. THE DIRECT-LOCAL windowed forward: a drop-in for block-0 ``.forward`` that
#    DIRECT-GATHERS each ingest head's output at query rows (no windowed score).
# ===========================================================================
def _install_direct_local_forward(model, block_idx: int,
                                   head_map: Dict[int, Tuple[int, int]],
                                   rf: ResolvedFrames):
    attn = model.blocks[block_idx].attn
    ing_heads = sorted(head_map)                     # role index == head index
    ing_heads_t = torch.tensor(ing_heads, dtype=torch.long)
    # A global absolute-position -> resolved-row index map as a dense LongTensor (default
    # -1 = "not a query row").  Built once; indexed by q_pos in the forward (O(1) gather,
    # NO Python loop).  Sized to the max resolved position + 1.
    max_pos = int(rf.pos.max().item()) if rf.pos.numel() else -1
    pos_map = torch.full((max_pos + 2,), -1, dtype=torch.long)
    if rf.pos.numel():
        pos_map[rf.pos.cpu()] = torch.arange(rf.pos.numel(), dtype=torch.long)

    def _gather_out_chunk(out, q_pos_c, span_off, dev):
        """Fill the per-chunk ingest-head output ``out`` [1,H,Sc,HD] by direct gather.
        ``q_pos_c`` is the chunk's absolute positions; ``span_off`` maps a query row's
        chunk-local index.  Identical gather math to the whole-span path, restricted to
        the chunk's rows (per-row independent, so byte-identical)."""
        pm = pos_map.to(dev)
        clamped = q_pos_c.clamp(max=pm.numel() - 1)
        row_idx = pm[clamped]                                 # [Sc], -1 for non-query
        qmask = row_idx >= 0
        if bool(qmask.any()):
            sel = row_idx[qmask]                              # [nq] resolved-row indices
            span_q = qmask.nonzero(as_tuple=False).flatten()  # [nq] chunk-local positions
            lo = rf.nib_lo.to(dev).index_select(0, sel)       # [nq, N_ROLES]
            hi = rf.nib_hi.to(dev).index_select(0, sel)       # [nq, N_ROLES]
            heads = ing_heads_t.to(dev)                        # [n_ing] ingest head idx
            h_idx = heads.unsqueeze(0).expand(sel.numel(), -1)      # [nq, n_ing]
            r_idx2 = span_q.unsqueeze(1).expand(-1, heads.numel())  # [nq, n_ing]
            out[0, h_idx, r_idx2, 0] = lo[:, heads]
            out[0, h_idx, r_idx2, 1] = hi[:, heads]

    def direct_local_forward(self, x, past_kv=None, q_positions=None, use_cache=False):
        H, HD = self.n_heads, self.head_dim
        B, S, D = x.shape
        dev = x.device
        if q_positions is None:
            q_pos = torch.arange(S, device=dev)
        else:
            q_pos = q_positions.to(device=dev, dtype=torch.long)

        # ---- Sq-CHUNKED PATH (C4_BLOCK0_SQ_CHUNK > 0): the big-K VRAM cap lifter -----
        # Process the span's S rows in chunks so we NEVER hold the full [1,H,S,HD]
        # K/V/out (nor the [1,S,D] out2 / W_o output) at once.  Under frozen_skip block 0
        # is the ONLY block over all S rows, so this is what stops the per-forward VRAM
        # scaling as O(K*30).  Byte-exact: the forward is a per-row independent map.
        chunk = _block0_sq_chunk()
        if chunk > 0 and S > chunk:
            res = x + 0.0                                     # residual buffer [1,S,D]
            drop_kv = _block0_drop_dead_kv()
            K_parts = [] if (use_cache and not drop_kv) else None
            V_parts = [] if (use_cache and not drop_kv) else None
            for lo0 in range(0, S, chunk):
                hi0 = min(lo0 + chunk, S)
                xc = x[:, lo0:hi0, :]                         # [1,Sc,D] view
                Sc = hi0 - lo0
                outc = xc.new_zeros(B, H, Sc, HD)
                _gather_out_chunk(outc, q_pos[lo0:hi0], lo0, dev)
                out2c = outc.transpose(1, 2).contiguous().view(B, Sc, D)
                res[:, lo0:hi0, :] = xc + self.W_o.linear(out2c)
                if K_parts is not None:
                    K_parts.append(self.W_k.linear(xc).view(B, Sc, H, HD)
                                   .transpose(1, 2))
                    V_parts.append(self.W_v.linear(xc).view(B, Sc, H, HD)
                                   .transpose(1, 2))
                del outc, out2c
            if not use_cache:
                return res
            if drop_kv:
                # block 0's KV is DEAD under direct-local: its own future attention is the
                # gather (ignores past_kv) and no other block reads block-0's cache, so a
                # None cache is byte-safe (same contract as dead_block_forward).  This is
                # the second half of the VRAM win — no full [1,H,S,HD] K/V is held at all.
                return res, None
            Knew = torch.cat(K_parts, dim=2)
            Vnew = torch.cat(V_parts, dim=2)
            return res, (Knew, Vnew, q_pos)

        # ---- WHOLE-SPAN PATH (chunk OFF): byte-identical to the prior direct-local -----
        # K/V still projected + committed for the cache contract (the non-ingest heads
        # and a fallback); the ingest heads' OUTPUT is a pure vectorized gather.
        Knew = self.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)
        Vnew = self.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)
        out = x.new_zeros(B, H, S, HD)
        _gather_out_chunk(out, q_pos, 0, dev)

        out2 = out.transpose(1, 2).contiguous().view(B, S, D)
        res = x + self.W_o.linear(out2)
        if use_cache:
            return res, (Knew, Vnew, q_pos)
        return res

    def gather_ingest_out(self, x, q_positions):
        """Fill + return the ingest-head gather ``out`` [1, H, S, HD] for the rows ``x``
        [1, S, D] at absolute positions ``q_positions`` — the SAME direct-CAM scatter
        the forward does, exposed so a CUDA-graph (block0_graph.Block0ChunkGraph) can
        feed it as the graph's fixed-shape input (the graph does the W_o + FFN GEMMs).
        Byte-identical to the gather inside ``direct_local_forward`` (same
        ``_gather_out_chunk`` closure, same ``pos_map`` index math)."""
        H, HD = self.n_heads, self.head_dim
        B, S, _ = x.shape
        dev = x.device
        q_pos = q_positions.to(device=dev, dtype=torch.long)
        out = x.new_zeros(B, H, S, HD)
        _gather_out_chunk(out, q_pos, 0, dev)
        return out

    attn.forward = direct_local_forward.__get__(attn, type(attn))
    attn.gather_ingest_out = gather_ingest_out.__get__(attn, type(attn))
    attn._direct_local_installed = True
    # EXPOSE the resolved-frame gather internals so the whole-step graph (RUNG 2,
    # C4_WHOLE_STEP_GRAPH) can precompute the per-chunk ingest nibbles + fold the gather
    # into its CUDA graph (same rf / pos_map / ingest heads -> byte-identical).
    attn._direct_local_rf = rf
    attn._direct_local_pos_map = pos_map
    attn._direct_local_ing_heads = ing_heads


# ===========================================================================
# 4. INSTALL.
# ===========================================================================
def install_direct_local_cam(model, L, draft, verbose: bool = False
                             ) -> Optional[ResolvedFrames]:
    """Install the direct-local-CAM forward on block 0 (the register-ingest block).

    Returns the ``ResolvedFrames`` (kept alive by the caller) or None if disabled.
    Call it AFTER ``install_local_attention`` (it OVERRIDES block-0's windowed forward
    with the direct gather; every other block keeps its windowed/global forward)."""
    if not direct_local_cam_enabled():
        return None
    # block 0 is the ingest block ("ingest+recompose"); locate it by name for safety.
    names = list(getattr(L, "_block_names", []))
    b0 = names.index("ingest+recompose") if "ingest+recompose" in names else 0
    attn = model.blocks[b0].attn
    head_map = ingest_head_map(attn)
    if not head_map:
        if verbose:
            print("[direct-local-cam] no ingest heads found on block 0 -> skip",
                  flush=True)
        return None
    dev = getattr(model, "embed", None)
    dev = dev.device if dev is not None and hasattr(dev, "device") else "cpu"
    rf = build_resolved_frames(draft, device=dev)
    _install_direct_local_forward(model, b0, head_map, rf)
    if verbose:
        print(f"[direct-local-cam] installed on block {b0} "
              f"({len(head_map)} ingest heads) resolved {rf.pos.numel()} query rows",
              flush=True)
    return rf
