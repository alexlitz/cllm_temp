"""CUDA-graph capture of the LEAN 7-14 layer forward (``qwen_lean_forward``).

The lean forward (``LeanQwenVM.forward``) is a pure-``torch`` block stack — RoPE +
RMSNorm + softmax + SwiGLU over 7-14 layers, 6 CAM heads, no ``transformers``
machinery.  On a deterministic single-token-window VM step that stack is 7-10
*sequential* attention+MLP sub-layers, each firing a fistful of tiny GEMM /
softmax / RMSNorm kernels.  On these ~120-200 MB compacted models the kernels are
so small that the **per-kernel launch + the Python block-loop dispatch** is a real
slice of the per-step wall-clock — exactly what a CUDA graph removes: capture the
whole stack ONCE, then replay the entire fused kernel schedule with a single
``graph.replay()`` (zero Python, zero per-launch CPU overhead).

Design
======
A CUDA graph requires **static input shapes AND static memory addresses**: the
capture records the exact kernel sequence over fixed buffers, and replay re-runs it
against those same buffers.  The lean forward's shape varies two ways:

  * the **naive** driver calls ``forward(x=[1,S,H])`` where ``S`` grows with the
    store-log length (BOS + n_store + 5 reg frames + STEP_END);
  * the **speculative** driver calls ``forward(x=[B,Smax,H])`` — B = drafted steps
    in the block (<= ``block_steps``), Smax the padded window — the IDEAL
    fixed-shape replay target (one big forward per block).

We therefore **bucket by the exact ``(B, S)`` shape**: one captured graph per shape,
lazily on first sight, cached in a dict.  In practice a program touches only a
handful of shapes (S in {7,8,...}; B in {block_steps, tail}), so a couple of graphs
cover a whole run.  An optional ``pad_window`` mode instead pads every ``S`` up to a
single ``max_window`` (one graph covers all naive steps) — a coarser bucket that
trades a little wasted compute on short windows for a single capture; the pad rows
sit at a far causal position so the decode is byte-identical.

Byte-identity
=============
Replay runs the SAME kernels over the SAME weights as the eager lean forward, so the
graphed forward is **byte-for-byte identical** to ``LeanQwenVM.forward`` (and thus to
the HF ``Qwen2Model`` and to ``isa.interpret`` where the model is correct).  The
graph path is a pure evaluator swap — ZERO change to the bake, the weights, or the
decode.  ``test_qwen_lean_cuda_graph.py`` pins this on the full arith/cmp/branch/
loop/memory battery for BOTH the naive and speculative drivers.

This wraps ``LeanQwenVM.forward`` WITHOUT the KV cache (``past=None``): both
production drivers rebuild the whole window each step / block and call the forward
with ``past=None``, so the fixed-shape full-window forward is the correct capture
target.  (The incremental KV path — verified Linf~6e-8 — is a separate replay target
for the async-KV agent; a per-bucket graph over a fixed cache length drops in the
same way against this scaffolding.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from .qwen_lean_forward import (
    LeanQwenVM, CAM_REGS, SP_INIT,
    _build_stream_and_overlay, _build_spec_batch,
    draft_program_lean, run_program_lean,
    LeanSpecResult,
)
from .qwen_full_vm import _snap


# ===========================================================================
# One captured graph over a fixed (B, S) window shape.
# ===========================================================================
@dataclass
class _CapturedGraph:
    graph: "torch.cuda.CUDAGraph"
    static_x: torch.Tensor          # [B, S, H] input buffer (copy_ into it, then replay)
    static_pos: torch.Tensor        # [B, S] position buffer
    static_out: torch.Tensor        # [B, S, H] output buffer (clone out after replay)
    B: int
    S: int


class GraphedLeanForward:
    """CUDA-graph replay wrapper around ``LeanQwenVM.forward`` (``past=None``).

    Lazily captures one CUDA graph per distinct ``(B, S)`` window shape and replays
    it for every subsequent forward of that shape.  ``__call__(x, q_positions)``
    returns ``hidden [B,S,H]`` (byte-identical to ``lean.forward(x, None,
    q_positions)[0]``).  On CPU / no-CUDA it transparently falls back to the eager
    forward, so the same driver code runs everywhere.

    ``pad_window``: if set, every forward is padded up to ``(B, pad_window)`` so a
    SINGLE graph per B covers all naive windows (the pad rows sit at a far causal
    position -> byte-identical decode).  Otherwise bucket by the exact shape.
    """

    def __init__(self, lean: LeanQwenVM, *, pad_window: Optional[int] = None,
                 warmup_iters: int = 3):
        self.lean = lean
        self.pad_window = pad_window
        self.warmup_iters = warmup_iters
        self._graphs: Dict[Tuple[int, int], _CapturedGraph] = {}
        self._pool = None                          # shared graph memory pool
        self.enabled = (lean.device.type == "cuda")
        # a far causal position for pad rows (mirrors _build_spec_batch's PAD_POS).
        self._pad_pos = 10_000_000
        # stats
        self.n_capture = 0
        self.n_replay = 0
        self.n_eager = 0

    # ------------------------------------------------------------------
    @property
    def shapes(self) -> List[Tuple[int, int]]:
        """The (B, S) shapes captured so far."""
        return sorted(self._graphs.keys())

    def _pad(self, x: torch.Tensor, pos: torch.Tensor,
             target_S: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad ``x [B,S,H]`` / ``pos [B,S]`` up to ``target_S`` rows.  The new rows
        are zero-embedding at a far causal position (causally invisible -> the
        decoded state on the real rows is byte-identical)."""
        B, S, H = x.shape
        if S == target_S:
            return x, pos
        assert S < target_S, (S, target_S)
        xp = x.new_zeros(B, target_S, H)
        xp[:, :S] = x
        pp = pos.new_full((B, target_S), self._pad_pos)
        pp[:, :S] = pos
        # give distinct far positions so no two pad rows collide (harmless either way).
        pp[:, S:] = self._pad_pos + torch.arange(target_S - S, device=pos.device)
        return xp, pp

    def _normalize_pos(self, q_positions: Optional[torch.Tensor],
                       B: int, S: int, device) -> torch.Tensor:
        if q_positions is None:
            return torch.arange(S, device=device).unsqueeze(0).expand(B, S).contiguous()
        pos = q_positions.to(device=device, dtype=torch.long)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0).expand(B, S).contiguous()
        return pos.contiguous()

    def _capture(self, B: int, S: int) -> _CapturedGraph:
        """Capture a CUDA graph of ``lean.forward`` over a fixed ``[B,S,H]`` window."""
        lean = self.lean
        H = lean.hidden_size
        dev = lean.device
        # Pin the CURRENT device to the model's device for the whole capture: the
        # warmup side-stream, ``torch.cuda.graph`` (which captures the current stream
        # of the current device) and the buffers must all agree, else replay hits
        # cudaErrorStreamCaptureInvalidated when the caller's current device differs
        # (e.g. capturing on cuda:1 while cuda:0 is current).
        with torch.cuda.device(dev):
            static_x = torch.zeros(B, S, H, device=dev, dtype=lean.dtype)
            static_pos = torch.zeros(B, S, device=dev, dtype=torch.long)

            # WARMUP on a side stream — required before capture so cuBLAS/cuDNN pick
            # their kernels and all lazy allocations happen OUTSIDE the graph (the
            # standard torch CUDA-graph capture protocol).
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(self.warmup_iters):
                    with torch.no_grad():
                        out, _ = lean.forward(static_x, past=None, q_positions=static_pos)
            torch.cuda.current_stream(dev).wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            pool = self._pool
            with torch.no_grad():
                if pool is None:
                    with torch.cuda.graph(graph):
                        out, _ = lean.forward(static_x, past=None, q_positions=static_pos)
                    self._pool = graph.pool()
                else:
                    with torch.cuda.graph(graph, pool=pool):
                        out, _ = lean.forward(static_x, past=None, q_positions=static_pos)
        cg = _CapturedGraph(graph=graph, static_x=static_x, static_pos=static_pos,
                            static_out=out, B=B, S=S)
        self._graphs[(B, S)] = cg
        self.n_capture += 1
        return cg

    # ------------------------------------------------------------------
    def __call__(self, x: torch.Tensor,
                 q_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Graphed forward.  Returns ``hidden [B,S,H]`` (the last-layer normed
        residual), byte-identical to ``lean.forward(x, None, q_positions)[0]``."""
        lean = self.lean
        B, orig_S, H = x.shape
        pos = self._normalize_pos(q_positions, B, orig_S, x.device)

        if not self.enabled:
            self.n_eager += 1
            with torch.no_grad():
                hidden, _ = lean.forward(x, past=None, q_positions=pos)
            return hidden

        S = orig_S
        if self.pad_window is not None and S < self.pad_window:
            x, pos = self._pad(x, pos, self.pad_window)
            S = self.pad_window

        key = (B, S)
        cg = self._graphs.get(key)
        if cg is None:
            cg = self._capture(B, S)
        # copy the real inputs into the static capture buffers, replay, clone out.
        with torch.cuda.device(lean.device):
            cg.static_x.copy_(x)
            cg.static_pos.copy_(pos)
            cg.graph.replay()
        self.n_replay += 1
        # strip the pad rows (causally invisible -> the real rows are byte-identical);
        # the decode reads the real query row (hidden[..., -1] / hidden[..., qrow]).
        return cg.static_out[:, :orig_S].clone()

    def stats(self) -> Dict[str, object]:
        return {"enabled": self.enabled, "shapes": self.shapes,
                "n_capture": self.n_capture, "n_replay": self.n_replay,
                "n_eager": self.n_eager, "pad_window": self.pad_window}


# ===========================================================================
# Graphed drivers — drop-in variants of the naive + speculative drivers that route
# the lean forward through a captured CUDA graph.  Byte-identical to the eager
# drivers (the graph replay == eager forward).
# ===========================================================================
def run_program_lean_graphed(lean: LeanQwenVM, code: List[isa.Instr],
                             max_steps: int = 64, graphed: "GraphedLeanForward" = None,
                             pad_window: Optional[int] = None,
                             verbose: bool = False) -> Dict[str, object]:
    """Naive one-forward-per-VM-step driver, but each forward is a CUDA-graph replay.

    Mirrors ``run_program_lean`` EXACTLY — same window build, same control loop, same
    decode — only ``lean.forward`` is swapped for the graphed replay.  Byte-identical
    to the eager naive driver.  ``graphed`` may be reused across programs to amortise
    the capture; otherwise one is built here."""
    QL, L = lean.QL, lean.QL.L
    subset = lean.subset
    g = graphed or GraphedLeanForward(lean, pad_window=pad_window)
    ref_trace = isa.interpret(code)

    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    store_log: List[dict] = []
    ax_trace: List[int] = []
    cur_pc = 0
    call_stack: List[Tuple[Optional[int], Optional[int]]] = []

    for _ in range(max_steps):
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF

        x, positions = _build_stream_and_overlay(lean, code, reg_state, store_log, load_addr)
        hidden = g(x, positions)
        state = hidden[0, -1]

        pc = _snap(state[L.PC_VAL])
        ax = _snap(state[L.AX_VAL]) & 0xFF
        sp = _snap(state[L.SP_VAL])
        bp = _snap(state[L.BP_VAL])
        stk = _snap(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5

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
            store_addr = _snap(state[L.STK_VAL])
            store_val = ax if op == isa.SI else (ax & 0xFF)
            store_log = [s for s in store_log
                         if (s["addr"] & 0xFF) != (store_addr & 0xFF)]
            store_log.append({"addr": store_addr, "val": store_val})

        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": stk}
        ax_trace.append(ax)
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):5s} -> "
                  f"pc={pc} ax={ax} sp={sp} bp={bp} stk={stk} halt={halted}")
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(code):
            break
    return {"ax_trace": ax_trace, "ref_trace": ref_trace,
            "exact": ax_trace == ref_trace, "steps": len(ax_trace),
            "graph_stats": g.stats()}


def speculative_run_lean_graphed(lean: LeanQwenVM, code: List[isa.Instr], *,
                                 block_steps: int = 32, max_steps: int = 4096,
                                 graphed: "GraphedLeanForward" = None,
                                 verbose: bool = False) -> LeanSpecResult:
    """Perfect-draft speculation with each block's batched forward run as a CUDA-graph
    replay.

    Mirrors ``speculative_run_lean`` EXACTLY — the deterministic VM drafts the whole
    trace, the lean forward verifies ``block_steps`` steps per forward by batching the
    per-step windows into one ``[B,Smax,H]`` forward — only that batched forward is a
    graph replay (the IDEAL fixed-shape replay target).  Byte-identical to the eager
    speculative driver.  Falls back to the graphed naive driver for out-of-slice
    programs (functions)."""
    g = graphed or GraphedLeanForward(lean)
    draft = draft_program_lean(lean, code, max_steps=max_steps)
    ref_trace = draft.ref_trace
    if not draft.steps:
        r = run_program_lean_graphed(lean, code, max_steps=max_steps,
                                     graphed=g, verbose=verbose)
        n = r["steps"]
        return LeanSpecResult(
            status="PASS" if r["exact"] else "FAIL", ax_trace=r["ax_trace"],
            ref_trace=r["ref_trace"], exact=r["exact"], steps=n, forwards=n,
            naive_forwards=n, speedup=1.0, accepted=n, detail="naive-fallback")

    QL, L = lean.QL, lean.QL.L
    n_steps = len(draft.steps)
    ax_trace: List[int] = []
    forwards = 0
    accepted = 0
    for s0 in range(0, n_steps, block_steps):
        slab = draft.steps[s0:s0 + block_steps]
        x, positions = _build_spec_batch(lean, code, slab)
        hidden = g(x, positions)
        forwards += 1
        for i, st in enumerate(slab):
            n_store = len(st["store_log"]) if lean.subset.memory else 0
            qrow = (1 + n_store) + len(CAM_REGS)          # BOS + stores + 5 regs + STEP_END
            state = hidden[i, qrow]
            ax = _snap(state[L.AX_VAL]) & 0xFF
            ax_trace.append(ax)
            accepted += 1
    exact = ax_trace == ref_trace
    speedup = (n_steps / forwards) if forwards else 0.0
    return LeanSpecResult(
        status="PASS" if exact else "FAIL", ax_trace=ax_trace, ref_trace=ref_trace,
        exact=exact, steps=n_steps, forwards=forwards, naive_forwards=n_steps,
        speedup=speedup, accepted=accepted,
        detail="" if exact else "spec trace != isa.interpret")
