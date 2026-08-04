"""#741 — GPU KERNEL-FUSION for the composed POSITION-SPARSE fast path.

The composed pos-sparse forward (block-skip + query-row-only + direct-CAM) cut the
DIV nnz-MACs 245-537x but only ~1.9x latency: the forward is LAUNCH-BOUND.  Per live
block the eager path dispatches ~8 tiny kernels (W_k[S], W_v[S], W_q[1], scores,
softmax, W_o[1], then the FFN's W_up[1]/W_gate[1]/silu*gate/W_down[1]); DIV's 186
live blocks => ~1500 sequential GPU dispatches, each doing trivially little work at
1 query row.  A CUDA graph amortises the CPU launch latency but not the GPU-side
SEQUENTIAL dispatch; a MEGAKERNEL (fuse the per-block schedule into far fewer, larger
kernels) is the lever.

This module builds three ascending fusion levels, each verified BYTE-EXACT against
the full-238 dense-over-positions reference by ``bench_pos_sparse_composed``:

  LEVEL 1 — PER-BLOCK GEMM FUSION (``FusedBlock``).  Concat the two S-row attention
    projections ``[W_k; W_v]`` into ONE [2D,D] GEMM (the S-row cost, the dominant
    term) and the two 1-row FFN projections ``[W_gate; W_up]`` into ONE [2H,D] GEMM.
    Q stays a 1-row GEMM (Q is 1 row, K/V are S rows — different M).  ~8 -> ~5
    kernels/block.  Byte-exact: a concatenated linear is the same per-row dot
    products, just co-scheduled.  The FFN runs in fp64 on the single query row (the
    ``lea-addr-nib`` fp-fragile-decode fix, see ``pos_sparse_forward``) — OR, under
    ``C4_LEA_INT_SNAP``, the fragile block uses an EXACT-INTEGER nibble decode and
    every other block runs fp32, so ZERO fp64 remains (byte-exact; #870).

  LEVEL 2 — WHOLE-STEP CUDA GRAPH (``PosSparseStepGraph``).  Capture the ENTIRE
    per-op live-block schedule (all FusedBlocks) into ONE CUDA graph per op-class
    live shape.  Collapses ~1500 CPU launches to ONE graph launch/step; the GPU-side
    kernels still run sequentially inside the graph but with zero per-kernel launch
    overhead.  This is the practical "megakernel" on this stack — a genuine single-
    kernel persistent megakernel would need a hand-written Triton/CUDA C kernel that
    the SwiGLU + softmax1 + ALiBi schedule does not currently have; the CUDA graph is
    the byte-exact, buildable form of the same win.

Gate: ``C4_POS_SPARSE`` (the composed path's flag); these fusions are drop-in
runners the bench selects.  The golden flag-OFF build is untouched.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .blogspec_model import softmax1
from .step_block_skip import build_live_index


# ---------------------------------------------------------------------------
# LEVEL 1 — per-block fusion.  Precompute the concatenated weights ONCE.
# ---------------------------------------------------------------------------
def _resident(sw):
    """Return a block's dense-resident weight (materialize_dense'd) or its dense."""
    if getattr(sw, "dense_resident", None) is not None:
        return sw.dense_resident
    return sw.dense


class FusedBlock:
    """One pos-sparse live block with fused GEMMs (query-row-only heavy compute).

    Precomputes:
      * ``W_kv`` = ``cat([W_k, W_v], 0)`` — one [2D, D] S-row GEMM (was two).
      * ``W_gu`` = ``cat([W_gate, W_up], 0)`` — one [2H, D] 1-row GEMM (was two),
        in BOTH fp64 (the default fragile-decode fix) and fp32 (used when
        ``C4_LEA_INT_SNAP`` is on).
    Attention Q / scores / softmax / W_o stay 1-row (fp32).

    FFN dtype:
      * default (``C4_LEA_INT_SNAP`` off): the query-row SwiGLU runs fp64 (the
        ``lea-addr-nib`` fragile-decode fix — see ``pos_sparse_forward``).
      * ``C4_LEA_INT_SNAP`` on (``_int_snap_mode``): the fragile ``lea-addr-nib``
        block is computed with the EXACT-INTEGER nibble split of ``LEA_Q & 0xFF``
        (``apply_int_lea_addr_nib``) and every OTHER block runs fp32 — ZERO fp64,
        byte-exact to the fp64 path.
    A routed (Top-1 MoE) FFN is NOT fused (its dispatch picks <=K dense rows); it
    keeps its own forward.
    """

    def __init__(self, blk, lea_snap_dims=None):
        self.attn = blk.attn
        self.ffn = blk.ffn
        self.routed = blk._routed
        # INTEGER LEA-address snap (C4_LEA_INT_SNAP, #870): a tuple of residual-dim
        # indices set by the runner for the ``lea-addr-nib`` block; when armed the
        # query-row FFN is the exact-integer nibble split of ``LEA_Q & 0xFF`` (no
        # fp64, no ~768-step silu decode).  None -> the fp64 SwiGLU.
        self._lea_snap_dims = lea_snap_dims
        # when the int-snap is armed on ANY block, the whole runner is in int-snap
        # mode -> the fragile block uses the int path, every other block runs fp32
        # (no fp64 anywhere).  Set by the runner build loop.
        self._int_snap_mode = False
        A = blk.attn
        self.H, self.HD, self.D = A.n_heads, A.head_dim, A.dim
        self.scale = A.scale
        self.alibi = A.alibi_slopes
        Wk, Wv = _resident(A.W_k), _resident(A.W_v)
        self.W_kv = torch.cat([Wk, Wv], dim=0).contiguous()          # [2D, D]
        self.W_q = _resident(A.W_q)
        self.W_o = _resident(A.W_o)
        self.W_gu32 = None
        if not self.routed:
            F_ = blk.ffn
            Wg, Wu = _resident(F_.W_gate), _resident(F_.W_up)
            gu = torch.cat([Wg, Wu], dim=0).contiguous()                     # [2H, D]
            b_gu = torch.cat([F_.b_gate, F_.b_up], dim=0).contiguous()
            Wd = _resident(F_.W_down).contiguous()
            self.W_gu64 = gu.double().contiguous()
            self.b_gu64 = b_gu.double().contiguous()
            self.W_down64 = Wd.double().contiguous()
            self.b_down64 = F_.b_down.double().contiguous()
            # fp32 fused weights (for the non-fragile blocks when C4_LEA_INT_SNAP is on).
            self.W_gu32 = gu
            self.b_gu32 = b_gu
            self.W_down32 = Wd
            self.b_down32 = F_.b_down
            self.Hdim = F_.b_up.shape[0]

    def to(self, device):
        self.W_kv = self.W_kv.to(device)
        self.W_q = self.W_q.to(device)
        self.W_o = self.W_o.to(device)
        self.alibi = self.alibi.to(device)
        if not self.routed:
            self.W_gu64 = self.W_gu64.to(device)
            self.b_gu64 = self.b_gu64.to(device)
            self.W_down64 = self.W_down64.to(device)
            self.b_down64 = self.b_down64.to(device)
            self.W_gu32 = self.W_gu32.to(device)
            self.b_gu32 = self.b_gu32.to(device)
            self.W_down32 = self.W_down32.to(device)
            self.b_down32 = self.b_down32.to(device)
        return self

    def forward(self, x: torch.Tensor, q_idx: int,
                pos: torch.Tensor, causal_bias: torch.Tensor) -> torch.Tensor:
        """Byte-exact fused query-row-only block.  ``pos``/``causal_bias`` are the
        precomputed per-S position/causal tensors (shared across the schedule)."""
        B, S, D = x.shape
        H, HD = self.H, self.HD
        # -- ATTENTION -----------------------------------------------------
        # fused K/V over ALL positions: one [2D,D] GEMM.
        kv = F.linear(x, self.W_kv)                                   # [B,S,2D]
        K = kv[..., :D].view(B, S, H, HD).transpose(1, 2)            # [B,H,S,HD]
        V = kv[..., D:].view(B, S, H, HD).transpose(1, 2)
        xq = x[:, q_idx:q_idx + 1]                                    # [B,1,D]
        Q = F.linear(xq, self.W_q).view(B, 1, H, HD).transpose(1, 2)  # [B,H,1,HD]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # [B,H,1,S]
        dist = (pos - q_idx).abs().to(scores.dtype)                  # [S]
        scores = scores - self.alibi.view(1, H, 1, 1) * dist.view(1, 1, 1, S)
        scores = scores + causal_bias.view(1, 1, 1, S)
        a = softmax1(scores, dim=-1)
        ctx = torch.matmul(a, V).transpose(1, 2).contiguous().view(B, 1, D)
        aout = xq + F.linear(ctx, self.W_o)                          # [B,1,D]
        out = x.clone()
        out[:, q_idx:q_idx + 1] = aout
        # -- FFN (query row) -----------------------------------------------
        if self.routed:
            fq = self.ffn(aout)
        elif self._int_snap_mode:
            # C4_LEA_INT_SNAP: the fragile ``lea-addr-nib`` block uses the exact
            # INTEGER nibble split (no fp64); every other block runs fp32.
            if self._lea_snap_dims is not None:
                from .pos_sparse_forward import apply_int_lea_addr_nib
                fq = apply_int_lea_addr_nib(aout, self._lea_snap_dims)
            else:
                gu = F.linear(aout, self.W_gu32) + self.b_gu32      # [B,1,2H]
                gate = gu[..., :self.Hdim]
                up = gu[..., self.Hdim:]
                hidden = F.silu(up) * gate
                fq = aout + F.linear(hidden, self.W_down32) + self.b_down32
        else:
            xq64 = aout.double()
            gu = F.linear(xq64, self.W_gu64) + self.b_gu64          # [B,1,2H]
            gate = gu[..., :self.Hdim]
            up = gu[..., self.Hdim:]
            hidden = F.silu(up) * gate
            fq = (xq64 + F.linear(hidden, self.W_down64) + self.b_down64).to(x.dtype)
        out[:, q_idx:q_idx + 1] = fq
        return out


class FusedPosSparseRunner:
    """LEVEL-1 per-block-fused pos-sparse runner (drop-in for PositionSparseRunner)."""

    def __init__(self, model, L):
        self.model = model
        self.L = L
        self.live_index = build_live_index(model, L)
        self.n_blocks = len(model.blocks)
        dev = model.embed.device
        # INTEGER LEA-address snap (C4_LEA_INT_SNAP, #870): under the flag the fragile
        # ``lea-addr-nib`` block is computed in exact integer arithmetic and every
        # other block runs fp32 -> ZERO fp64.  Default OFF -> the fp64 path.
        from .pos_sparse_forward import (lea_int_snap_enabled, resolve_lea_snap_dims,
                                         LEA_ADDR_NIB_BLOCK_NAME)
        int_snap = lea_int_snap_enabled()
        dims = resolve_lea_snap_dims(L) if int_snap else None
        names = list(getattr(L, "_block_names", []))
        self.fused = []
        for bi, b in enumerate(model.blocks):
            snap = dims if (dims is not None and bi < len(names)
                            and names[bi] == LEA_ADDR_NIB_BLOCK_NAME
                            and not b._routed) else None
            fb = FusedBlock(b, lea_snap_dims=snap).to(dev)
            fb._int_snap_mode = int_snap and dims is not None
            self.fused.append(fb)

    def live_count(self, op) -> int:
        return len(self.live_index.get(op, self.live_index[None]))

    def forward(self, x: torch.Tensor, op) -> torch.Tensor:
        live = self.live_index.get(op)
        S = x.shape[1]
        q = S - 1
        pos = torch.arange(S, device=x.device)
        causal = torch.where(pos > q, torch.full((S,), float("-inf"), device=x.device),
                             torch.zeros(S, device=x.device))
        with torch.no_grad():
            if live is None:
                for fb in self.fused:
                    x = fb.forward(x, q, pos, causal)
                return x
            live_set = set(live)
            for bi, fb in enumerate(self.fused):
                if bi in live_set:
                    x = fb.forward(x, q, pos, causal)
        return x


# ---------------------------------------------------------------------------
# LEVEL 2 — whole-step CUDA graph over the op-class live schedule.
# ---------------------------------------------------------------------------
class PosSparseStepGraph:
    """Capture + replay ONE CUDA graph per distinct op-class live schedule using the
    LEVEL-1 FusedBlocks.  Static shapes: S fixed, query row q=S-1, the live-block set
    fixed per op-class.  Replaying collapses the whole per-step schedule's launches to
    a single graph launch.  Byte-exact: the graphed arithmetic is the SAME FusedBlock
    chain (same fp64 FFN, same softmax1/ALiBi), verified L-inf=0 vs eager."""

    def __init__(self, runner: FusedPosSparseRunner, S: int, device):
        self.runner = runner
        self.S = S
        self.device = device
        self.q = S - 1
        D = runner.model.embed.shape[1]
        self.D = D
        self.static_in = torch.zeros(1, S, D, device=device,
                                     dtype=runner.model.embed.dtype)
        self.pos = torch.arange(S, device=device)
        self.causal = torch.where(
            self.pos > self.q, torch.full((S,), float("-inf"), device=device),
            torch.zeros(S, device=device))
        # op -> live-schedule key; distinct keys share a graph.
        self.op_to_key: Dict[int, Tuple[int, ...]] = {}
        seen: Dict[Tuple[int, ...], List[int]] = {}
        for op, live in runner.live_index.items():
            if op is None:
                continue
            key = tuple(live)
            self.op_to_key[op] = key
            seen.setdefault(key, []).append(op)
        self.distinct_keys = list(seen.keys())
        self._seen = seen
        self.graphs: Dict[Tuple[int, ...], Tuple] = {}

    def _apply(self, x, live: List[int]):
        for bi in live:
            x = self.runner.fused[bi].forward(x, self.q, self.pos, self.causal)
        return x

    def try_capture(self, key: Tuple[int, ...]) -> str:
        live = list(key)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    _ = self._apply(self.static_in, live)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        try:
            with torch.no_grad():
                with torch.cuda.graph(g):
                    static_out = self._apply(self.static_in, live)
            torch.cuda.synchronize()
        except Exception as e:
            return f"skip:{type(e).__name__}:{str(e).splitlines()[0][:70]}"
        self.graphs[key] = (g, static_out)
        return "ok"

    def replay_ms(self, key, iters=50, warmup=10) -> Optional[float]:
        import time
        ent = self.graphs.get(key)
        if ent is None:
            return None
        g, _ = ent
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1e3


__all__ = ["FusedBlock", "FusedPosSparseRunner", "PosSparseStepGraph"]
