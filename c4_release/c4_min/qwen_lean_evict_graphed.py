"""CUDA-GRAPHED + VECTORIZED bounded-KV incremental decode (the ~1.74 ms/step target).

The eager bounded-KV driver (:func:`qwen_lean_evict.run_program_lean_evict`) is DEAD
FLAT in S (the register-frame supersession + heap prune bound the attention to ~a
handful of rows), but each VM step still costs ~10 ms = a ~7 ms 11-layer forward
(LAUNCH-bound over the bounded cache: 10-11 sequential attn+MLP sub-layers, each a
fistful of tiny GEMM/softmax/RMSNorm kernels ~60 us to LAUNCH) + ~3 ms Python
bookkeeping (the per-step evict keep-mask + ``apply_keep`` gather + ``torch.cat``
append + nibble decode).  This module lands the two levers that collapse both:

  1. **CUDA graph over the incremental forward.**  In steady state the cache going
     INTO the forward is a CONSTANT prefix (BOS sink + the immutable CODE frames + the
     live heap store rows) — the prior register frame is SUPERSEDED (dropped) at the
     top of every step, so the forward always sees ``prefix`` + the NEW 6-row register
     frame.  That is a FIXED (B,S) shape against FIXED prefix-cache addresses, so we
     capture ONE CUDA graph per prefix length and ``replay()`` it per step (zero
     Python, zero per-kernel launch overhead) — the measured 4.3x (7 -> ~1.6 ms).
     The register-frame ``x`` + its growing RoPE positions are the only inputs that
     change: they are ``copy_``'d into the graph's static input buffers each step.

  2. **Vectorized per-step bookkeeping.**  The steady-state cache is FIXED-length, so
     the per-step supersession is a no-op copy (the prefix is untouched; the new frame
     overwrites the same static slots), the "append" is the graph writing the fixed
     prefix K/V (never re-``cat``'d on the Python hot path), and the register/nibble
     decode is a single fused vectorised argmax read off the graph's static output.
     There is NO per-step ``keep_mask`` python loop, NO ``torch.nonzero`` gather, NO
     ``torch.cat`` — they are hoisted to prefix-change boundaries only (a store append
     / a heap prune), which are rare (a spin loop never hits them).

Byte-identity
=============
The graph replay runs the SAME kernels over the SAME weights as the eager lean
forward, and the fixed-prefix cache carries the SAME K/V rows the eager evict driver
would (BOS + CODE + live heap), at the SAME growing RoPE positions.  So the decoded
AX trace is BYTE-IDENTICAL to :func:`qwen_lean_evict.run_program_lean_evict` (and thus
to the naive fresh-window driver and ``isa.interpret``).  ``test_qwen_lean_evict_graphed``
pins this on the arith/cmp/branch/loop/memory/lww battery; the bench
(:mod:`bench_bounded_kv_incremental`) re-verifies vs the naive driver before timing.

The prefix-change path (a store append / a heap prune changes the prefix length or
its rows) rebuilds the fixed prefix cache and re-captures the graph for the new prefix
length — lazily, cached per length.  A register-only spin loop touches ONE prefix
length forever (one capture), a bounded heap program a small handful.

Entry point: :func:`run_program_lean_evict_graphed` — a drop-in for
``run_program_lean_evict`` returning the same :class:`qwen_lean_evict.EvictStats`.
Falls back to the eager driver transparently on CPU / no-CUDA (so tests run off-GPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .nibble_pure_forward import SP_INIT
from .qwen_full_vm import (
    CAM_REGS, _REG_TOKEN, _snap,
)
from .qwen_lean_forward import LeanQwenVM
from . import qwen_lean_evict as EV


# ===========================================================================
# Vectorized register-frame builder — the constant template is precomputed ONCE
# (embed + ONE + ROLE + IS_TOK + the query row's roles) and only the per-step data
# (the 5 registers' nibbles + the fetch@PC / load-address query bits) is scattered in
# a single vectorised ``index_put`` — off the per-scalar-write Python hot path.
# ===========================================================================
class VectorizedFrameBuilder:
    """Build the step's register frame residual ``x`` [1,F,H] with a precomputed
    constant template + a batched per-step nibble/query scatter.

    ``_append_reg_frame`` writes ~100 individual scalar cells into ``x`` each step
    (``x[0,i,ONE]=1``, per-register nibbles, roles, PC/load bits), each a tiny CUDA
    kernel — ~1.3 ms/step.  Here the frame token embed + ONE + per-register ROLE +
    IS_TOK + the query row's ROLE one-hots are STEP-INVARIANT, so they are baked into a
    template tensor once; per step we only overwrite the changing lanes (the 5×16
    register nibbles, the fetch@PC address bits, the load-address bits) with ONE
    vectorised scatter.  Byte-identical residual to ``_append_reg_frame``."""

    def __init__(self, lean: LeanQwenVM, code: List[isa.Instr]):
        self.lean = lean
        self.QL, self.L = lean.QL, lean.QL.L
        self.cfm = lean.code_from_memory
        self.mem = lean.subset.memory
        dev = lean.device
        L, QL = self.L, self.QL
        from .qwen_full_vm import CODE_ADDR_BITS
        from .blogspec_memory import ADDR_BITS
        self.ADDR_BITS = ADDR_BITS
        self.CODE_ADDR_BITS = CODE_ADDR_BITS
        stream = [_REG_TOKEN[r] for r in CAM_REGS] + [V.STEP_END]
        toks = torch.tensor([stream], device=dev)
        # -- constant template: everything that does NOT change per step -----------
        tmpl = lean.embed[toks].clone()                         # [1,F,H]
        F = tmpl.shape[1]
        self.F = F
        for i in range(F):
            tmpl[0, i, L.ONE] = 1.0
        for hh in range(len(CAM_REGS)):
            tmpl[0, hh, QL.ROLE + hh] = 1.0
            tmpl[0, hh, QL.IS_TOK] = 1.0
            tmpl[0, -1, QL.ROLE + hh] = 1.0                     # query row roles
        self.template = tmpl
        # -- the flat (row, col) indices of the per-step nibble lanes (5 regs × 16) --
        rows, cols = [], []
        for hh in range(len(CAM_REGS)):
            for j in range(NIB_PER_REG):
                rows.append(hh)
                cols.append(QL.TOK_NIB + j)
        self.nib_rows = torch.tensor(rows, device=dev)
        self.nib_cols = torch.tensor(cols, device=dev)
        # the register order for gathering values from a reg_state dict, fast.
        self._regs = list(CAM_REGS)
        # precompute the little-endian nibble shift table (16 nibbles).
        self._shifts = torch.arange(NIB_PER_REG, device=dev) * 4      # [16]
        # precompute the per-bit shift tables for the fetch@PC + load-address queries
        # (vectorise ``_overlay_fetch_query`` / the load-bit loop: ONE slice write each).
        self._code_shifts = torch.arange(CODE_ADDR_BITS, device=dev)      # [12]
        self._addr_shifts = torch.arange(ADDR_BITS, device=dev)          # [8]
        self._code_mask = (1 << CODE_ADDR_BITS) - 1
        # persistent scratch frame buffer (reused each step; the mutable lanes are
        # fully overwritten, so no clone/reset is needed — see ``build``).  The load
        # lanes are cleared only when a prior step wrote them (``_load_dirty``).
        self._buf = self.template.clone()
        self._load_dirty = False

    def build(self, reg_state: dict, load_addr: Optional[int], pos: int
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x [1,F,H], positions [F])`` for the given register state — the
        persistent scratch buffer with only the per-step nibble/query lanes overwritten
        (all vectorised: ONE nibble scatter + ONE fetch-bit slice + at most ONE
        load-bit slice).  Byte-identical to ``_append_reg_frame``.

        The buffer is reused across steps (the caller — ``g.forward_frame`` — copies it
        into the graph's static input BEFORE the next ``build``), and every mutable lane
        is FULLY overwritten each step, so no per-step clone/reset is needed except
        clearing the load lanes when a prior LOAD step wrote them and this step is not a
        load (``_load_dirty``)."""
        L = self.L
        dev = self.lean.device
        x = self._buf
        # register values -> [5,16] little-endian nibbles, in ONE vectorised op.
        vals = torch.tensor([reg_state[r] for r in self._regs],
                            device=dev, dtype=torch.long)            # [5]
        nibs = ((vals.unsqueeze(1) >> self._shifts.unsqueeze(0)) & 0xF).to(x.dtype)  # [5,16]
        x[0, self.nib_rows, self.nib_cols] = nibs.reshape(-1)
        if self.cfm:
            # fetch@PC query on the last (STEP_END) row: IS_FETCH=1 + CODE_QRY_BIN bits.
            x[0, -1, L.IS_FETCH] = 1.0
            pc_bits = ((int(reg_state["PC"]) & self._code_mask) >> self._code_shifts) & 1
            x[0, -1, L.CODE_QRY_BIN:L.CODE_QRY_BIN + self.CODE_ADDR_BITS] = pc_bits.to(x.dtype)
        if self.mem and load_addr is not None:
            x[0, -1, L.IS_LOAD] = 1.0
            bits = ((int(load_addr) >> self._addr_shifts) & 1).to(x.dtype)
            x[0, -1, L.QRY_BIN:L.QRY_BIN + self.ADDR_BITS] = bits
            self._load_dirty = True
        elif self._load_dirty:
            # a prior step wrote the load lanes; this step is not a load -> clear them
            # so the residual matches a fresh ``_append_reg_frame`` (which never sets
            # IS_LOAD/QRY_BIN on a non-load step).
            x[0, -1, L.IS_LOAD] = 0.0
            x[0, -1, L.QRY_BIN:L.QRY_BIN + self.ADDR_BITS] = 0.0
            self._load_dirty = False
        positions = torch.arange(pos, pos + self.F, device=dev)
        return x, positions


# ===========================================================================
# Batched register decode — the 5 register lanes' value-argmax requant in ONE pass.
# ===========================================================================
class BatchedSnap:
    """Fuse the 5 per-register ``_snap`` argmax requants into ONE batched fp64 argmax.

    ``_snap`` builds an ``arange(VALVOCAB)`` fp64 grid + argmaxes ``2·v·x − v²`` PER
    register — 5 separate ~0.2 ms calls.  Here the value grid is built ONCE (cached on
    the model device) and all 5 register lanes are argmaxed together: ``logits[r,v] =
    2·v·x[r] − v²`` over the shared ``v`` grid, one ``argmax(dim=1)``.  Byte-identical
    to five ``_snap`` calls (same fp64 arithmetic, same tie-break)."""

    def __init__(self, device: torch.device):
        from .nibble_vm import VALVOCAB
        self.v = torch.arange(VALVOCAB, dtype=torch.float64, device=device)  # [Vv]
        self.v2 = self.v * self.v

    def snap_many(self, xs: torch.Tensor) -> List[int]:
        """``xs`` [N] fp lane values -> the N argmax-requant integers (as a python
        list).  ``logits[n,v] = 2·v·xs[n] − v²``; argmax over v per row."""
        xs64 = xs.to(torch.float64).unsqueeze(1)                    # [N,1]
        logits = 2.0 * self.v.unsqueeze(0) * xs64 - self.v2.unsqueeze(0)  # [N,Vv]
        return logits.argmax(dim=1).tolist()


# ===========================================================================
# One captured graph over a fixed (prefix_len + frame_len) incremental forward.
# ===========================================================================
@dataclass
class _CapturedIncGraph:
    graph: "torch.cuda.CUDAGraph"
    static_x: torch.Tensor          # [1, F, H] new register-frame input (copy_ in)
    static_pos: torch.Tensor        # [1, F] new-frame absolute positions (copy_ in)
    static_out: torch.Tensor        # [1, F, H] hidden output (read decode off [-1])
    # the fixed prefix cache the graph reads (per-layer (K,V,pos)); captured by ref.
    prefix_past: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    prefix_len: int
    frame_len: int


class GraphedIncrementalForward:
    """CUDA-graph replay of the bounded-KV INCREMENTAL forward against a FIXED prefix.

    Holds the persistent prefix cache (BOS + CODE + live heap store rows) in static
    tensors, and lazily captures one graph per (prefix_len, frame_len) shape.  A step
    is: ``copy_`` the new register frame's residual + positions into the static input
    buffers, ``graph.replay()``, read the decode off ``static_out[-1]`` — no Python
    cat / gather / keep-mask on the hot path.

    ``set_prefix(past, prefix_len)`` installs / updates the fixed prefix cache (called
    at seed time and whenever a store append / heap prune changes the prefix); it
    invalidates the graphs so they re-capture against the new prefix addresses.
    """

    def __init__(self, lean: LeanQwenVM, *, warmup_iters: int = 3):
        self.lean = lean
        self.warmup_iters = warmup_iters
        self.enabled = (lean.device.type == "cuda")
        self._graphs: Dict[Tuple[int, int], _CapturedIncGraph] = {}
        self._prefix_past: Optional[List] = None
        self._prefix_len = 0
        self.n_capture = 0
        self.n_replay = 0
        self.n_eager = 0

    # ------------------------------------------------------------------
    def set_prefix(self, past: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                   prefix_len: int) -> None:
        """Install the fixed prefix cache (clone to stable, contiguous buffers) and
        invalidate the captured graphs (they must re-capture against the new prefix
        tensor addresses)."""
        if not self.enabled:
            self._prefix_past = past
            self._prefix_len = prefix_len
            return
        stable = []
        for p in past:
            if p is None:
                stable.append(None)
                continue
            K, Vv, pos = p
            stable.append((K.contiguous().clone(), Vv.contiguous().clone(),
                           pos.contiguous().clone()))
        self._prefix_past = stable
        self._prefix_len = prefix_len
        # a prefix change moves the addresses the graph reads -> drop stale graphs.
        self._graphs.clear()

    def _capture(self, frame_len: int) -> _CapturedIncGraph:
        lean = self.lean
        H = lean.hidden_size
        dev = lean.device
        with torch.cuda.device(dev):
            static_x = torch.zeros(1, frame_len, H, device=dev, dtype=lean.dtype)
            static_pos = torch.zeros(1, frame_len, device=dev, dtype=torch.long)
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(self.warmup_iters):
                    with torch.no_grad():
                        out, _ = lean.forward(static_x, past=self._prefix_past,
                                              q_positions=static_pos)
            torch.cuda.current_stream(dev).wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            # Each capture gets its OWN private pool: a prefix change
            # (``set_prefix``) drops the old graphs, and reusing a pool whose buffers
            # are still referenced by a live captured graph trips the caching
            # allocator's use-count assert.  A private pool per graph is clean and the
            # count of distinct (prefix_len, frame_len) shapes is tiny (a spin loop = 1;
            # a bounded heap = a small handful), so pool proliferation is a non-issue.
            with torch.no_grad():
                with torch.cuda.graph(graph):
                    out, _ = lean.forward(static_x, past=self._prefix_past,
                                          q_positions=static_pos)
        cg = _CapturedIncGraph(
            graph=graph, static_x=static_x, static_pos=static_pos, static_out=out,
            prefix_past=self._prefix_past, prefix_len=self._prefix_len,
            frame_len=frame_len)
        self._graphs[(self._prefix_len, frame_len)] = cg
        self.n_capture += 1
        return cg

    # ------------------------------------------------------------------
    def forward_frame(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Incremental forward of the new register frame ``x`` [1,F,H] at absolute
        ``positions`` [F] against the installed fixed prefix.  Returns ``hidden``
        [1,F,H] byte-identical to ``lean.forward(x, past=prefix, q_positions=positions)``
        — via a CUDA-graph replay (fixed shape) on GPU, eager on CPU."""
        lean = self.lean
        F = x.shape[1]
        if not self.enabled:
            self.n_eager += 1
            with torch.no_grad():
                hidden, _ = lean.forward(x, past=self._prefix_past,
                                         q_positions=positions)
            return hidden
        key = (self._prefix_len, F)
        cg = self._graphs.get(key)
        if cg is None:
            cg = self._capture(F)
        pos = positions if positions.dim() == 2 else positions.unsqueeze(0)
        with torch.cuda.device(lean.device):
            cg.static_x.copy_(x)
            cg.static_pos.copy_(pos)
            cg.graph.replay()
        self.n_replay += 1
        return cg.static_out

    def stats(self) -> Dict[str, object]:
        return {"enabled": self.enabled, "n_capture": self.n_capture,
                "n_replay": self.n_replay, "n_eager": self.n_eager,
                "prefix_len": self._prefix_len, "shapes": sorted(self._graphs.keys())}


# ===========================================================================
# The graphed + vectorized bounded-KV driver.
# ===========================================================================
def _seed_prefix(lean: LeanQwenVM, code: List[isa.Instr]
                 ) -> Tuple[List, int]:
    """Build the fixed prefix cache (BOS sink + CODE frames) via ONE eager forward.
    Returns ``(prefix_past, prefix_len)`` — the immutable rows every step attends over
    (positions 0..prefix_len-1)."""
    cache = EV.LeanKVCache(lean.n_layers, lean.device)
    xb, pb, mb = EV._append_bos(lean, code)
    with torch.no_grad():
        _, new_past = lean.forward(xb, past=None, q_positions=pb)
    cache.append(new_past, mb)
    next_pos = 1
    if lean.code_from_memory:
        xc, pc_, mc = EV._append_code_frames(lean, code, next_pos)
        with torch.no_grad():
            _, new_past = lean.forward(xc, past=cache.as_past(), q_positions=pc_)
        cache.append(new_past, mc)
        next_pos += xc.shape[1]
    return cache.past, cache.size()


def run_program_lean_evict_graphed(
        lean: LeanQwenVM, code: List[isa.Instr], *, max_steps: int = 64,
        graphed: Optional["GraphedIncrementalForward"] = None,
        sample_every: int = 1, verbose: bool = False) -> EV.EvictStats:
    """Bounded-KV incremental decode with a CUDA-GRAPHED forward + VECTORIZED
    bookkeeping — the ~1.74 ms/step driver.

    Byte-identical AX trace to :func:`qwen_lean_evict.run_program_lean_evict` (and thus
    to the naive fresh-window driver).  Structure:

      * The prefix (BOS + CODE frames [+ live heap store rows]) is a FIXED cache the
        register-frame forward attends over; it changes ONLY on a store append (SI/SC)
        or a heap prune (rare), which rebuild it + re-capture the graph.
      * Each step overwrites the SAME static input slots with the new register frame
        (no cat / gather / keep-mask on the hot path) and replays the graph.
      * A SI/SC store appends its row to the prefix (supersede same-address first) and
        re-seeds the fixed prefix cache; a load reads it via the persistent CODE/heap
        rows exactly as the eager driver.

    Falls back to the eager evict driver on CPU (``graphed`` disabled) — used by the
    off-GPU byte-identity tests.  Returns :class:`qwen_lean_evict.EvictStats`."""
    subset = lean.subset
    L = lean.QL.L
    try:
        ref_trace = isa.interpret(code, max_steps=max_steps)
    except Exception:
        ref_trace = []

    g = graphed or GraphedIncrementalForward(lean)
    # vectorized per-step bookkeeping: a precomputed frame template (only the changing
    # nibble/query lanes scattered per step) + a batched 5-register value-argmax decode.
    fb = VectorizedFrameBuilder(lean, code)
    bs = BatchedSnap(lean.device)

    # -- the fixed prefix cache (BOS + CODE) + a mirror EV.LeanKVCache for the heap.
    # The heap store rows are appended to a growing "heap cache" that, together with
    # the BOS+CODE seed, forms the prefix the graph reads.  For a spin loop the heap
    # is empty and the prefix is a single constant length forever (one capture).
    heap = EV.LeanKVCache(lean.n_layers, lean.device)          # BOS + CODE + stores
    xb, pb, mb = EV._append_bos(lean, code)
    with torch.no_grad():
        _, np_ = lean.forward(xb, past=None, q_positions=pb)
    heap.append(np_, mb)
    next_pos = 1
    if lean.code_from_memory:
        xc, pc_, mc = EV._append_code_frames(lean, code, next_pos)
        with torch.no_grad():
            _, np_ = lean.forward(xc, past=heap.as_past(), q_positions=pc_)
        heap.append(np_, mc)
        next_pos += xc.shape[1]
    g.set_prefix(heap.past, heap.size())

    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    ax_trace: List[int] = []
    cur_pc = 0
    call_stack: List[Tuple[Optional[int], Optional[int]]] = []
    frame_idx = 0

    max_rows = heap.size() + len(CAM_REGS) + 1
    max_bytes = 0
    size_trace: List[int] = []
    bytes_trace: List[int] = []
    total_evicted = 0

    for step in range(max_steps):
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF

        # -- build the register frame (vectorised template scatter) at the growing pos.
        x, positions = fb.build(reg_state, load_addr, next_pos)
        hidden = g.forward_frame(x, positions)
        next_pos += x.shape[1]
        frame_idx += 1

        # -- batched value-argmax decode: the 5 register lanes in ONE fp64 argmax.
        state = hidden[0, -1]
        pc, ax, sp, bp, stk = bs.snap_many(
            state[[L.PC_VAL, L.AX_VAL, L.SP_VAL, L.BP_VAL, L.STK_VAL]])
        ax &= 0xFF
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
            store_addr = _snap(state[L.STK_VAL]) & 0xFF
            store_val = ax if op == isa.SI else (ax & 0xFF)
            # STRUCTURAL same-address supersession, then append the store row to the
            # prefix (heap cache) and RE-SEED the fixed prefix (which re-captures the
            # graph for the new prefix length).  This is the ONLY hot-path cat/gather,
            # and it fires only on a store (not on the spin/arith/branch inner loop).
            dropped = heap.supersede_store_addr(store_addr)
            total_evicted += dropped
            xs, ps, ms = EV._append_store(lean, code, store_addr, store_val, next_pos)
            with torch.no_grad():
                _, np_ = lean.forward(xs, past=heap.as_past(), q_positions=ps)
            heap.append(np_, ms)
            next_pos += 1
            # prune freed / superseded zero-value store rows off the prefix (bounded).
            dropped = heap.apply_keep(heap.keep_mask(heap_prune=True))
            total_evicted += dropped
            g.set_prefix(heap.past, heap.size())

        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": stk}
        ax_trace.append(ax)

        cur_rows = heap.size() + x.shape[1]
        if step % sample_every == 0:
            size_trace.append(cur_rows)
            bytes_trace.append(heap.bytes())
        max_rows = max(max_rows, cur_rows)
        max_bytes = max(max_bytes, heap.bytes())

        if verbose and step < 8:
            print(f"  step {step} op={isa.NAMES.get(op, op):5s} -> ax={ax} pc={pc} "
                  f"prefix={heap.size()} rows={cur_rows}")
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(code):
            break

    return EV.EvictStats(
        exact=(ax_trace == ref_trace), steps=len(ax_trace), ax_trace=ax_trace,
        ref_trace=ref_trace, max_cache_rows=max_rows,
        final_cache_rows=heap.size() + len(CAM_REGS) + 1,
        max_cache_bytes=max_bytes, total_evicted=total_evicted, n_prunes=0,
        cache_size_trace=size_trace, cache_bytes_trace=bytes_trace)
