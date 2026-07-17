"""GPU-BATCHED pure-forward C4 VM driver on the SMALL (compact + sparse) model.

The CPU KV-cached driver
(:func:`nibble_pure_forward_cached.run_pure_forward_cached`) runs ONE program at
a time, forwarding a FIXED 31-row window against a per-block incremental KV cache
every VM step.  The programs in the 1096 corpus are INDEPENDENT, so the
embarrassingly-parallel win is to STACK ``B`` programs' windows into one
``[B, W, D]`` batched forward and step them together, terminating (masking) each
slot as its program HALTs.

This module keeps the SAME per-step arithmetic as the single-program cached
driver — same windowed overlay, same softmax1+ALiBi attention over absolute
positions, same last-row register decode (``_snap_lane`` / nibble argmax) — so
the emitted byte trace of program ``p`` is **byte-identical** to
``run_pure_forward_cached(..., evict=False)`` for that program (up to the
documented saturated-tie fp handful where the model itself is
fp-accumulation-order divergent between a CPU-serial and a GPU-parallel matmul
reduction — those are isolated and reported, never hidden).

It runs on the NOW-SMALL model: ``compact_alloc.build_compact_pure_forward_model``
+ ``sparse_forward.SparseTransformer`` (dense_kernel mode -> L-inf=0), whose
weights load in ~0.034 GB VRAM.  Because the whole model is tiny, the batch B can
be LARGE (hundreds).  Every weight goes through ``SparseWeight.linear`` (which
already accepts ``[B, W, D]``), so NO separate ``to_sparse_ffn`` / ``to_sparse_attn``
monkeypatch is needed — the SparseTransformer IS the sparse model.

WHAT IS BATCHED (device-resident, one forward for the whole batch):
  * the embedding gather + overlay of each slot's window (``[B, W, D]``),
  * the whole block stack + per-block KV attention (``[B, H, S, HD]``).
Everything else (the ~5 scalar-lane argmax decodes + the next 30-token frame
construction per active slot) is per-program Python, exactly as the CPU driver
does it — those are O(1) scalar ops off the last window row, not model compute.

KV CACHES are kept per-program in the batch axis, PER BLOCK.  Because programs
step asynchronously (different halt steps) AND eviction prunes each block
independently, the per-block cache length can differ across slots AND across
blocks — the batched cache is therefore PADDED to the block's batch-max length
(computed PER BLOCK from that block's live caches, not from block 0), with pad
rows given a sentinel absolute position LARGER than any real query position so the
causal mask (``k_pos > q_pos``) makes every query ignore them — a no-op, so the
padded attention is byte-identical to the ragged per-program attention.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .nibble_pure_forward import (
    N_ROLES, _FRAME_ROLE_SLOTS, _MEM_MARKER_LOCAL, _address_bits, SP_INIT,
    VALVOCAB,
)
from .nibble_pure_forward_complete import (
    PureForwardCompleteLayout, IMM_NIBS, _build_frame, _mem_top,
)
from .blogspec_layout import NIB_PER_REG


# ===========================================================================
# Vectorised register decode.  ``_snap_lane`` / ``_decode_reg_from_nibbles`` in
# the reference are per-scalar Python argmaxes; here we evaluate the SAME
# ``argmax_v (2·v·x − v²)`` requant over a whole batch of lanes at once (on the
# same device as ``state``) so the per-step decode is a couple of tensor ops.
# The math is bit-identical to the reference (an integer argmax over the same
# candidate set), so the decoded value is the same.
# ===========================================================================
def _snap_lane_batch(lanes: torch.Tensor) -> torch.Tensor:
    """Batched ``argmax_v (2·v·x − v²)`` over the value vocab. ``lanes`` [N] ->
    [N] long. Matches ``nibble_pure_forward._snap_lane`` per element."""
    v = torch.arange(VALVOCAB, device=lanes.device, dtype=torch.float64)
    logits = 2.0 * lanes.to(torch.float64).unsqueeze(-1) * v - v * v
    return logits.argmax(dim=-1)


def _snap_nib_batch(lanes: torch.Tensor) -> torch.Tensor:
    """Batched ``argmax_n (2·n·x − n²)`` over n in 0..15. ``lanes`` [N] -> [N] long.
    Matches ``nibble_pure_forward_complete._snap_nib`` per element."""
    n = torch.arange(16, device=lanes.device, dtype=torch.float64)
    logits = 2.0 * lanes.to(torch.float64).unsqueeze(-1) * n - n * n
    return logits.argmax(dim=-1)


def _decode_reg_batch(states: torch.Tensor, reg_base: int) -> torch.Tensor:
    """Decode a 4-byte register from its 8 nibble dims for a BATCH of last-row
    states (``[B, D]``).  Returns a ``[B]`` long tensor of the register values.
    Matches ``_decode_reg_from_nibbles`` per row."""
    vals = torch.zeros(states.shape[0], dtype=torch.long, device=states.device)
    for bi in range(4):
        lo = _snap_nib_batch(states[:, reg_base + 2 * bi + 0])
        hi = _snap_nib_batch(states[:, reg_base + 2 * bi + 1])
        vals |= (lo + 16 * hi) << (8 * bi)
    return vals


# ===========================================================================
# Per-program state carried across steps (mirrors the locals of the single-program
# cached driver: stream, store_log, cur_pc/sp/bp/ax, window bounds, per-block cache).
# ===========================================================================
class _ProgState:
    __slots__ = ("code", "stream", "trace", "store_log", "cur_pc", "cur_sp",
                 "cur_bp", "cur_ax", "frame_idx", "win_start", "win_len",
                 "K", "V", "pos", "halted", "max_seq", "steps", "step_cap",
                 "tokens_since_prune")

    def __init__(self, code: List[isa.Instr], n_blocks: int, step_cap: int):
        self.code = code
        init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        self.stream: List[int] = [V.BOS] + init_frame
        self.trace: List[int] = []
        self.store_log: Dict[int, Tuple[int, int]] = {}
        self.cur_pc = 0
        self.cur_sp = self.cur_bp = SP_INIT
        self.cur_ax = 0
        self.frame_idx = 0
        self.win_start = 0
        self.win_len = len(self.stream)
        # per-block KV cache: lists indexed by block. None until first commit.
        self.K: List[Optional[torch.Tensor]] = [None] * n_blocks
        self.V: List[Optional[torch.Tensor]] = [None] * n_blocks
        self.pos: List[Optional[torch.Tensor]] = [None] * n_blocks
        self.halted = False
        self.max_seq = len(self.stream)
        self.steps = 0
        self.step_cap = step_cap
        self.tokens_since_prune = 0

    def cache_len(self, b: int = 0) -> int:
        """Cached-row count of block ``b`` (PER BLOCK: eviction prunes each block
        independently, so block ``b`` may differ from block 0)."""
        return 0 if self.pos[b] is None else int(self.pos[b].shape[0])


# ===========================================================================
# Overlay one program's window into a [W, D] slice (host tensor). Same math as
# ``nibble_pure_forward_cached.apply_overlay_window`` but writing a [W, D] view.
# ===========================================================================
def _overlay_window_into(x_slot: torch.Tensor, w_start: int, code, L,
                         store_log: Dict[int, Tuple[int, int]]) -> None:
    """In-place overlay of one program's window residual ``x_slot`` ([W, D]).
    ``w_start`` is the absolute stream position of ``x_slot[0]``.  The LAST row is
    always the query row (all-ROLE one-hots)."""
    W = x_slot.shape[0]
    for wi in range(W):
        p = w_start + wi
        x_slot[wi, L.ONE] = 1.0
        for k, ins in enumerate(code):
            x_slot[wi, L.CODE_OP[k]] = float(ins.op)
            x_slot[wi, L.CODE_IMM[k]] = float(ins.imm)
            for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
                x_slot[wi, L.CODE_IMM_NIB[k] + j] = float(nv)
        if p == 0:
            continue
        f = (p - 1) // V.FRAME_LEN
        local = (p - 1) % V.FRAME_LEN
        if local in _FRAME_ROLE_SLOTS:
            role = _FRAME_ROLE_SLOTS[local]
            x_slot[wi, L.ROLE + role] = 1.0
            x_slot[wi, L.IS_FRAME_BYTE] = 1.0
        if local == _MEM_MARKER_LOCAL and f in store_log:
            addr, val = store_log[f]
            x_slot[wi, L.IS_STORE] = 1.0
            x_slot[wi, L.IS_FRAME_BYTE] = 0.0
            for b, bit in enumerate(_address_bits(addr)):
                x_slot[wi, L.ADDR_BIN + b] = bit
            for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                x_slot[wi, L.VAL_NIB + j] = float(nv)
    for role in range(N_ROLES):
        x_slot[-1, L.ROLE + role] = 1.0


# ===========================================================================
# Batched softmax1 + ALiBi attention over a PADDED per-program KV cache.
#
# ``SparseWeight.linear`` already accepts ``[B, W, D]`` (dense_kernel -> L-inf=0),
# so the sparse model's blocks are used verbatim; only the attention *math* is
# re-expressed batched with a padded K/V + a per-slot validity mask.  This
# mirrors ``sparse_forward.SparseAttn.forward``'s cached path.
# ===========================================================================
def _attn_batched(attn, x, past_K, past_V, past_pos, past_valid, q_pos):
    """x [B,W,D]; past_* [B,H,Sc,HD]/[B,Sc]; past_valid [B,Sc] bool (real cache row);
    q_pos [B,W] absolute positions of the query rows.  Returns (out[B,W,D],
    (K_all[B,H,Sc+W,HD], V_all, pos_all[B,Sc+W], valid_all[B,Sc+W]))."""
    B, W, D = x.shape
    H, HD = attn.n_heads, attn.head_dim
    Q = attn.W_q.linear(x).view(B, W, H, HD).transpose(1, 2)
    Knew = attn.W_k.linear(x).view(B, W, H, HD).transpose(1, 2)
    Vnew = attn.W_v.linear(x).view(B, W, H, HD).transpose(1, 2)
    new_valid = torch.ones(B, W, dtype=torch.bool, device=x.device)
    if past_K is not None:
        K = torch.cat([past_K, Knew], dim=2)            # [B,H,Sc+W,HD]
        Vv = torch.cat([past_V, Vnew], dim=2)
        k_pos = torch.cat([past_pos, q_pos], dim=1)     # [B,Sc+W]
        valid = torch.cat([past_valid, new_valid], dim=1)  # [B,Sc+W]
    else:
        K, Vv, k_pos, valid = Knew, Vnew, q_pos, new_valid

    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale   # [B,H,W,Sk]
    dist = (q_pos.unsqueeze(2) - k_pos.unsqueeze(1)).abs().float()  # [B,W,Sk]
    scores = scores - attn.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(1)
    # causal over absolute positions + drop padding cache rows (mark invalid).
    fut = (k_pos.unsqueeze(1) > q_pos.unsqueeze(2))              # [B,W,Sk]
    fut = fut | (~valid).unsqueeze(1)                           # invalid pad -> masked
    scores = scores.masked_fill(fut.unsqueeze(1), float("-inf"))
    # softmax1 (ZFOD): exp(x-m)/(exp(-m)+sum exp(x-m)), m=clamp(max,0).
    m = scores.max(dim=-1, keepdim=True)[0]
    m = torch.clamp(m, min=0.0)
    exp_x = torch.exp(scores - m)
    denom = torch.exp(-m) + exp_x.sum(dim=-1, keepdim=True)
    a = exp_x / denom
    out = torch.matmul(a, Vv).transpose(1, 2).contiguous().view(B, W, D)
    out = x + attn.W_o.linear(out)
    return out, (K, Vv, k_pos, valid)


# A few blocks have a WIDE FFN hidden (up to the global-max 21544), whose
# ``silu(up)*gate`` activation ``[B, W, hidden]`` is the batched-forward VRAM peak.
# Chunk the (B*W) rows so that transient stays bounded regardless of B — the FFN
# is applied per-row independently, so a row-chunked pass is byte-identical.
_FFN_ROW_CHUNK = 2048


def _ffn_forward_chunked(ffn, a):
    """``ffn.forward(a)`` for ``a`` [B,W,D], row-chunked to bound the wide-hidden
    activation.  Byte-identical (the FFN is a per-row map + residual)."""
    B, W, D = a.shape
    rows = B * W
    if rows <= _FFN_ROW_CHUNK:
        return ffn.forward(a)
    flat = a.reshape(rows, 1, D)
    outs = []
    for s in range(0, rows, _FFN_ROW_CHUNK):
        outs.append(ffn.forward(flat[s:s + _FFN_ROW_CHUNK]))
    return torch.cat(outs, dim=0).reshape(B, W, D)


def _block_forward_batched(blk, x, past_K, past_V, past_pos, past_valid, q_pos):
    a, kv = _attn_batched(blk.attn, x, past_K, past_V, past_pos, past_valid, q_pos)
    out = _ffn_forward_chunked(blk.ffn, a)   # SparseFFN.forward accepts [*,W,D]
    return out, kv


# ===========================================================================
# THE GPU-BATCHED DRIVER.
# ===========================================================================
def run_batch_gpu(model, L: PureForwardCompleteLayout,
                  codes: List[List[isa.Instr]], step_caps: List[int],
                  device: str = "cuda", mask: int = 0xFFFFFFFF,
                  verbose: bool = False, evict: bool = False,
                  prune_interval: int = 120, cos_threshold: float = 0.99,
                  zero_eps: float = 1e-9, recency_eps: float = 1e-6,
                  stats: Optional[dict] = None) -> List[List[int]]:
    """Run a BATCH of independent programs on ``device`` together, one batched
    forward per VM step, terminating slots as they HALT.

    ``codes``      list of per-program ``[isa.Instr]`` (the compiled programs).
    ``step_caps``  list of per-program max-step caps (halts count against these).

    ``model`` is a ``sparse_forward.SparseTransformer`` (dense_kernel -> L-inf=0)
    already ``.to(device)``.

    ``evict`` (default OFF) applies the spec softmax1+ALiBi bounded eviction per
    program (the SAME ``nibble_pure_forward_cached.prune_keep_mask_head`` policy) so
    deep loops keep a FLAT cache.  OFF by default for byte-identity to the naive
    ground-truth; turn ON only for the deep-loop path where an unbounded cache is
    the memory hazard.

    Returns a list of per-program AX traces (masked to ``mask``); with
    ``evict=False`` byte-identical to ``run_pure_forward_cached(..., evict=False)``
    (== naive) on the same device (up to the saturated-tie fp handful).
    """
    from .nibble_pure_forward_cached import evict_keep_index
    n_blocks = len(model.blocks)
    H = model.blocks[0].attn.n_heads
    HD = model.blocks[0].attn.head_dim
    B = len(codes)
    dev = torch.device(device)

    # Host mirror of the embedding table so the per-step window residual (embed
    # gather + structural overlay) is built on CPU and copied to device ONCE per
    # step — the many tiny scalar writes are far cheaper on the host than as
    # per-element CUDA kernel launches (the deep-tail bottleneck).
    embed_cpu = model.embed.detach().to("cpu")

    progs = [_ProgState(c, n_blocks, sc) for c, sc in zip(codes, step_caps)]
    active = list(range(B))            # indices of slots still running
    max_seq = len(progs[0].stream)
    max_cache = 0
    n_forwards = 0

    while active:
        Bact = len(active)
        W = progs[active[0]].win_len   # all active slots share W (31)
        # -- build the batched overlaid window residual [Bact, W, D] -----------
        win_toks = torch.zeros(Bact, W, dtype=torch.long)
        q_positions = torch.zeros(Bact, W, dtype=torch.long)
        for i, p in enumerate(active):
            ps = progs[p]
            seg = ps.stream[ps.win_start:ps.win_start + W]
            win_toks[i, :len(seg)] = torch.tensor(seg, dtype=torch.long)
            q_positions[i] = torch.arange(ps.win_start, ps.win_start + W)
        with torch.no_grad():
            # Build the overlaid window residual ON CPU (fast scalar writes) then
            # move to device ONCE (writing ~1500 structural scalars per row into a
            # CUDA tensor is a kernel launch each; a host build + single H2D copy
            # is ~100x cheaper and byte-identical).
            x_cpu = embed_cpu[win_toks].clone()      # [Bact, W, D] on host
            for i, p in enumerate(active):
                ps = progs[p]
                _overlay_window_into(x_cpu[i], ps.win_start, ps.code, L, ps.store_log)
            x = x_cpu.to(dev)                        # single H2D transfer
            q_positions = q_positions.to(dev)

            # -- assemble padded per-block batched caches --------------------
            # For each block, stack this batch's slot caches for THAT BLOCK,
            # padding to that block's Sc_max with sentinel positions + invalid
            # flags.  Lengths are read PER BLOCK (eviction prunes blocks
            # independently), fixing the block-0-length cache-shape bug.
            #
            # MEMORY: we do NOT keep the full [B,H,Sc+W,HD] K/V of every block
            # alive (that is n_blocks * full-cache and OOMs at large B) — only the
            # newly-computed WINDOW tail ([B,H,W,HD]) is needed to commit, so we
            # slice it out immediately and let the full padded K/V free.
            hidden = x
            win_kv_per_block = []                 # (K_win[B,H,W,HD], V_win, pos_win[B,W])
            sentinel = int(q_positions.max().item()) + 1_000_000
            for b in range(n_blocks):
                lens = [progs[p].cache_len(b) for p in active]
                Sc = max(lens) if lens else 0
                if Sc == 0:
                    past_K = past_V = past_pos = past_valid = None
                else:
                    past_K = torch.zeros(Bact, H, Sc, HD, device=dev)
                    past_V = torch.zeros(Bact, H, Sc, HD, device=dev)
                    past_pos = torch.full((Bact, Sc), sentinel, dtype=torch.long,
                                          device=dev)
                    past_valid = torch.zeros(Bact, Sc, dtype=torch.bool, device=dev)
                    for i, p in enumerate(active):
                        ps = progs[p]
                        n = lens[i]
                        if n:
                            past_K[i, :, :n, :] = ps.K[b]
                            past_V[i, :, :n, :] = ps.V[b]
                            past_pos[i, :n] = ps.pos[b].to(dev)
                            past_valid[i, :n] = True
                out, kv = _block_forward_batched(
                    model.blocks[b], hidden, past_K, past_V, past_pos,
                    past_valid, q_positions)
                hidden = out
                # keep ONLY the freshly-computed window tail (last W rows), cloned
                # off the big cat'd K/V so the [B,H,Sc+W,HD] tensor can free now.
                K_all, V_all, pos_all, _valid_all = kv
                win_kv_per_block.append((
                    K_all[:, :, -W:, :].contiguous(),
                    V_all[:, :, -W:, :].contiguous(),
                    pos_all[:, -W:].contiguous()))
                del kv, K_all, V_all, pos_all, past_K, past_V, past_pos, past_valid
            n_forwards += 1

        # -- per-slot decode + frame emit (scalar, off the last window row) ----
        states = hidden[:, -1, :]                 # [Bact, D]
        pc_b = _snap_lane_batch(states[:, L.PC_VAL])
        sp_b = _snap_lane_batch(states[:, L.SP_VAL])
        bp_b = _snap_lane_batch(states[:, L.BP_VAL])
        stk_b = _snap_lane_batch(states[:, L.STK_VAL])
        ax_b = _decode_reg_batch(states, L.AX)
        halted_b = (states[:, L.HALTED] > 0.5)
        pc_l = pc_b.tolist(); sp_l = sp_b.tolist(); bp_l = bp_b.tolist()
        stk_l = stk_b.tolist(); ax_l = ax_b.tolist()
        halt_l = halted_b.tolist()

        still_active = []
        for i, p in enumerate(active):
            ps = progs[p]
            pc = pc_l[i]; sp = sp_l[i]; bp = bp_l[i]; stk = stk_l[i]
            ax = ax_l[i]; halted = halt_l[i]
            op = ps.code[ps.cur_pc].op if 0 <= ps.cur_pc < len(ps.code) else None
            s_addr = s_val = 0
            is_store = False
            if op in (isa.SI, isa.SC):
                is_store = True; s_addr = _mem_top(ps.store_log, ps.cur_sp); s_val = ax & mask
            elif op == isa.PSH:
                is_store = True; s_addr = ps.cur_sp - 4; s_val = ax & mask
            elif op == isa.JSR:
                is_store = True; s_addr = ps.cur_sp - 4; s_val = (ps.cur_pc + 1) & 0xFFFFFFFF
            elif op == isa.ENT:
                is_store = True; s_addr = ps.cur_sp - 4; s_val = ps.cur_bp & 0xFFFFFFFF
            frame = _build_frame(pc, ax, sp, bp, stk,
                                 mem_addr=(s_addr if is_store else 0),
                                 mem_val=(s_val if is_store else 0))
            ps.trace.append(ax & mask)
            ps.frame_idx += 1
            if is_store:
                ps.store_log[ps.frame_idx] = (s_addr, s_val)
            ps.steps += 1
            ps.cur_pc, ps.cur_sp, ps.cur_bp, ps.cur_ax = pc, sp, bp, ax

            done = halted or pc < 0 or pc >= len(ps.code) or ps.steps >= ps.step_cap
            if not done:
                prev_query_pos = ps.win_start + ps.win_len - 1
                ps.stream += frame
                ps.win_start = prev_query_pos
                ps.win_len = 1 + V.FRAME_LEN
                ps.max_seq = max(ps.max_seq, len(ps.stream))
                max_seq = max(max_seq, len(ps.stream))
                still_active.append(p)
            else:
                ps.halted = True

        # -- COMMIT frozen window rows to each active slot's per-block cache ----
        # window's last row is the query row (NOT frozen); rows [0:W-1] freeze.
        n_commit = W - 1
        if n_commit > 0:
            for b in range(n_blocks):
                K_win, V_win, pos_win = win_kv_per_block[b]  # already the W-tail
                for i, p in enumerate(active):
                    ps = progs[p]
                    if ps.halted:
                        continue                  # halted slot: cache frozen forever
                    Kc = K_win[i:i+1, :, :n_commit, :]
                    Vc = V_win[i:i+1, :, :n_commit, :]
                    if ps.K[b] is None:
                        ps.K[b] = Kc.clone()
                        ps.V[b] = Vc.clone()
                        ps.pos[b] = pos_win[i, :n_commit].clone()
                    else:
                        ps.K[b] = torch.cat([ps.K[b], Kc], dim=2)
                        ps.V[b] = torch.cat([ps.V[b], Vc], dim=2)
                        ps.pos[b] = torch.cat([ps.pos[b], pos_win[i, :n_commit]], dim=0)

        # -- bounded eviction (deep-loop path): per program, prune every
        # ``prune_interval`` tokens using the SAME spec softmax1+ALiBi policy as the
        # CPU cached driver (per head; union across heads keeps any needed row).
        # After a prune the surviving row set of block ``b`` differs from block 0 —
        # this is exactly why ``cache_len`` and the padded-cache assembly above are
        # PER BLOCK.
        if evict:
            scale = HD ** -0.5
            for p in still_active:
                ps = progs[p]
                ps.tokens_since_prune += V.FRAME_LEN
                if ps.tokens_since_prune < prune_interval:
                    continue
                ps.tokens_since_prune = 0
                for b in range(n_blocks):
                    if ps.K[b] is None:
                        continue
                    Kb, Vb, posb = ps.K[b], ps.V[b], ps.pos[b]
                    S = int(posb.shape[0])
                    # SHARED spec eviction policy (SAME ``evict_keep_index`` the
                    # single-program cached driver / block-verifier use).  Per BLOCK
                    # slopes (not block-0's), and — crucially — the EXACT dup metric
                    # on content-addressed §Memory store heads, without which distinct
                    # stored values are cosine-merged (~0.999 ADDR_BIN common-mode) and
                    # a deep-loop LI/LC recalls nothing (got=0, ZFOD — the batched-path
                    # deep-loop memory bug).
                    keep_idx = evict_keep_index(
                        Kb, Vb, posb, model.blocks[b].attn.alibi_slopes, scale, H,
                        cos_threshold=cos_threshold, zero_eps=zero_eps,
                        recency_eps=recency_eps)
                    if keep_idx is not None and int(keep_idx.numel()) < S:
                        keep_dev = keep_idx.to(Kb.device)
                        ps.K[b] = Kb[:, :, keep_dev, :]
                        ps.V[b] = Vb[:, :, keep_dev, :]
                        ps.pos[b] = posb[keep_dev]

        for p in active:
            max_cache = max(max_cache, progs[p].cache_len(0))
        active = still_active
        if verbose:
            print(f"  batched step: {Bact} active, forwards={n_forwards}, "
                  f"maxcache={max_cache}")

    if stats is not None:
        stats["max_seq_len"] = max_seq
        stats["max_cache_size"] = max_cache
        stats["n_forwards"] = n_forwards
        stats["batch"] = B

    return [ps.trace for ps in progs]
