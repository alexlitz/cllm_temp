"""COMPOSED byte-exact fast path — the culminating ms/step integration.

This is the ONE place where ALL the merged per-step levers run TOGETHER on a single
byte-exact forward, so we can measure the REAL composed ms/step toward 0.1 ms:

  * ``C4_STEP_BLOCK_SKIP`` (``step_block_skip.StepBlockSkipRunner``): per decoded op,
    run ONLY that op's live blocks (non-divmod 238->~7; DIV/MOD ~186).  Makes the
    live-block SET static per op-class -> one CUDA graph per op-class.
  * ``C4_DIRECT_CAM_READ`` (``direct_cam_read`` / ``nibble_evict_schedule.resolve_load_rows``):
    the perfect draft already knows the exact KV store row each CAM read resolves to,
    so on the fast path the read DIRECT-GATHERS the value (overwrites the destination
    nibble band) instead of running the O(K) softmax score.  Makes ATTENTION static
    (the CAM heads no longer depend on runtime cache content).
  * ``block_sparse_ffn.BlockSparseFFN`` (COO gather-scale-scatter): the surviving live
    blocks' 99.93%-sparse FFN GEMMs run as a static index program (few-FLOP), not a
    dense zero-multiply.  ~2x on the FFN-bound live blocks.
  * ``C4_EXACT_EVICT`` (``nibble_evict_schedule``): O(steps) free-detection off the
    draft — NO O(S^2) prune.  This is a KV-cache lever; the per-step re-embed forward
    below keeps NO KV cache (fresh embed each step), so exact-evict is a NO-OP for the
    per-step path and is measured only where it matters (the batched verify — see the
    exact-evict report at the end).

COMPOSITION CLAIM (the thing a0510bde warned did not compose before): direct-CAM makes
attention STATIC, exact-evict removes the O(S^2), block-skip makes the live-block set
per-op STATIC.  With all three static, the per-op forward is a fixed-shape graph.  We
VERIFY byte-exactness of the full composition vs the full-238 dense path on a battery
(arith / branch / JSR-LEV / memory / mul / div / nested loop), then CUDA-graph each
op-class and BENCHMARK at production S.

Run (needs a free >=18 GB CUDA card):
    OMP_NUM_THREADS=4 python -m c4_min.bench_composed_fast_path --S 900,3000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .step_block_skip import StepBlockSkipRunner
from .pf_speculative import draft_pf_program
from .nibble_evict_schedule import resolve_load_rows


# ---------------------------------------------------------------------------
# GPU wait-loop: block until the chosen card has enough free VRAM and is idle,
# and confirm it is stable for a window (the multi-agent contention guard).
# ---------------------------------------------------------------------------
def _gpu_stat(idx: int) -> Tuple[float, float]:
    """(free_GB, util_pct) for CUDA device ``idx`` via nvidia-smi."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader,nounits", f"--id={idx}"],
            stderr=subprocess.DEVNULL, timeout=4).decode().strip().splitlines()[0]
        free_mib, util = out.split(",")
        return float(free_mib) / 1024.0, float(util)
    except Exception:
        return 0.0, 100.0


def wait_for_gpu(idx: int, min_free_gb: float = 18.0, stable_s: float = 60.0,
                 timeout_s: float = 3600.0) -> None:
    """Block until device ``idx`` holds >= ``min_free_gb`` free for ``stable_s``
    continuously (poll every 5s).  The composed bench needs a stable card — a
    mid-run OOM or a co-tenant spinning up would poison the ms/step number."""
    t0 = time.time()
    stable_since: Optional[float] = None
    while time.time() - t0 < timeout_s:
        free, util = _gpu_stat(idx)
        ok = free >= min_free_gb
        now = time.time()
        if ok:
            if stable_since is None:
                stable_since = now
            held = now - stable_since
            if held >= stable_s:
                print(f"  [gpu-wait] cuda:{idx} stable {held:.0f}s "
                      f"(free={free:.1f}GB util={util:.0f}%) -> proceed", flush=True)
                return
            print(f"  [gpu-wait] cuda:{idx} free={free:.1f}GB util={util:.0f}% "
                  f"stable {held:.0f}/{stable_s:.0f}s ...", flush=True)
        else:
            stable_since = None
            print(f"  [gpu-wait] cuda:{idx} free={free:.1f}GB < {min_free_gb}GB "
                  f"(util={util:.0f}%) waiting ...", flush=True)
        time.sleep(5)
    raise SystemExit(f"[gpu-wait] no stable free GPU within {timeout_s}s")


# ---------------------------------------------------------------------------
# The COMPOSED per-step forward.  Runs the block-skip runner (op's live blocks
# only) over the SAME frozen S-row stream, then overwrites the CAM-read dest
# bands with the direct-gather value (direct-CAM), then decodes the query row.
# ---------------------------------------------------------------------------
def _dest_band(L, head: str) -> int:
    return {"mem": L.AX, "pop": L.STACK0, "lev": L.LEV_RET}[head]


def _resolve_with_seed(draft, seed_mem: Dict[int, int]):
    """``resolve_load_rows`` for a program run with SEEDED memory.

    The draft starts from empty memory, so a read of a seed-only address resolves
    to ZFOD 0 there.  We re-resolve every read against a store map pre-populated
    with the seed stores (they precede all program frames), so a seed-only read
    gathers the seeded value — byte-identical to the model, which reads the seed
    KV row.  Returns ``{draft_frame: [ResolvedRead,...]}`` (program-relative)."""
    from .nibble_evict_schedule import ResolvedRead, store_row_position
    if not seed_mem:
        return resolve_load_rows(draft)
    store_log = draft.store_log or {}
    read_log = draft.read_log or {}
    # latest store per address, seeded with the pre-program data segment.  A seed
    # address has no in-stream store frame (its KV is a leading seed frame), so we
    # tag it store_frame=None/position=None but carry the seeded VALUE for the
    # gather (the gather returns the value; the schedule/position is only used by
    # the batched evict path, which the seed rows never enter).
    latest_val: Dict[int, int] = dict(seed_mem)
    latest_sf: Dict[int, Optional[int]] = {a: None for a in seed_mem}
    out: Dict[int, List] = {}
    all_frames = sorted(set(store_log) | set(read_log))
    for f in all_frames:
        for (head, addr) in read_log.get(f, []):
            sf = latest_sf.get(addr)
            if sf is None and addr not in latest_val:
                out.setdefault(f, []).append(ResolvedRead(head, f, addr, None, None, 0))
            elif sf is None:
                # seed-only address: value from the seed, no in-stream store row.
                out.setdefault(f, []).append(
                    ResolvedRead(head, f, addr, None, None, latest_val[addr]))
            else:
                sval = store_log[sf][1]
                out.setdefault(f, []).append(
                    ResolvedRead(head, f, addr, sf, store_row_position(sf), sval))
        if f in store_log:
            addr, val = store_log[f]
            latest_val[addr] = val
            latest_sf[addr] = f
    return out


class ComposedFastPath:
    """block-skip + direct-CAM + (optional) block-sparse-FFN on ONE forward.

    ``forward(x, op, resolved_reads)`` applies the op's live blocks to ``x``
    ([1, S, D]) and, if the op performs CAM reads, overwrites the read
    destination nibble band(s) of the query row with the resolved draft value
    (the O(1) gather that replaces the O(K) softmax score).  Returns the query
    row state ([D]).  Byte-identical to the full-238 forward + softmax CAM iff
    the block-skip schedule is DECODE-sound (verified) and the resolved value ==
    the softmax winner (proven by ``resolve_load_rows``)."""

    def __init__(self, model, L, direct_cam: bool = True):
        self.model = model
        self.L = L
        self.direct_cam = direct_cam
        self.runner = StepBlockSkipRunner(model, L)

    def live_count(self, op) -> int:
        return self.runner.live_count(op)

    def forward(self, x: torch.Tensor, op,
                resolved: Optional[List] = None) -> torch.Tensor:
        with torch.no_grad():
            x = self.runner.forward(x, op)
        state = x[0, -1].clone()
        if self.direct_cam and resolved:
            for r in resolved:
                band = _dest_band(self.L, r.head)
                for j, nv in enumerate(
                        V.nibbles_of_value(r.value & 0xFFFFFFFF, NIB_PER_REG)):
                    state[band + j] = float(nv)
        return state


# ---------------------------------------------------------------------------
# Byte-exact composition VERIFY: run the FULL 238-block softmax path AND the
# COMPOSED path (block-skip + direct-CAM + block-sparse-FFN) over the same
# growing stream, and assert the per-step decoded AX trace is identical.
# ---------------------------------------------------------------------------
def _drive(model, L, code, *, composed: Optional[ComposedFastPath] = None,
           max_steps=200, seed_mem=None):
    """Token-by-token driver.  ``composed=None`` -> the full 238-block forward +
    softmax CAM (the reference).  Otherwise the composed fast path.  Returns the
    per-step AX trace (& 0xFF) and the total block-applies."""
    from . import nibble_pure_forward_complete as pfc
    from .nibble_pure_forward_complete import (
        make_overlay_complete, _build_frame, SP_INIT, _seed_frames, _mem_top)

    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)

    # the direct-CAM path needs the whole program's resolved reads up front.  The
    # draft has NO seed frames (it starts from empty memory), so a read of a
    # SEED-only address (e.g. LI 8 with data_seg {8:123}) resolves to the seeded
    # value here — the resolver is pre-populated with the seed stores so a
    # seed-only read gathers the seeded value, not a ZFOD 0.  Draft frame indices
    # are program-relative (step s -> frame s+1); the driver's frame_idx is offset
    # by n_seed, so we key the resolved reads by the DRAFT frame (this_frame -
    # n_seed) below.
    resolved_by_frame = {}
    if composed is not None and composed.direct_cam:
        draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF)
        resolved_by_frame = _resolve_with_seed(draft, seed_mem or {})
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    frame_idx = n_seed
    trace: List[int] = []
    total_applies = 0
    dev = model.embed.device
    for _ in range(max_steps):
        overlay = make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream], device=dev)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            if composed is None:
                for blk in model.blocks:
                    x = blk(x)
                total_applies += len(model.blocks)
                state = x[0, -1]
            else:
                # draft frame indices are program-relative (no seed frames), so a
                # read at driver frame_idx+1 is draft frame (frame_idx+1 - n_seed).
                draft_frame = frame_idx + 1 - n_seed
                resolved = resolved_by_frame.get(draft_frame, [])
                state = composed.forward(x, op, resolved)
                total_applies += composed.live_count(op)
        pc = pfc._snap_lane(state[L.PC_VAL].cpu())
        sp = pfc._snap_lane(state[L.SP_VAL].cpu())
        bp = pfc._snap_lane(state[L.BP_VAL].cpu())
        stk = pfc._snap_lane(state[L.STK_VAL].cpu())
        halted = float(state[L.HALTED]) > 0.5
        ax = pfc._decode_reg_from_nibbles(state.cpu(), L, L.AX)
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & 0xFF
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & 0xFF
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        trace.append(ax & 0xFF)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        cur_pc, cur_sp, cur_bp = pc, sp, bp
        if halted or pc < 0 or pc >= len(code):
            break
    return trace, total_applies


def _battery() -> List[Tuple[str, list, dict]]:
    """arith / branch / JSR-LEV / memory / mul / div / nested loop."""
    C: List[Tuple[str, list, dict]] = []
    C.append(("imm", [("IMM", 42), ("HALT", 0)], {}))
    C.append(("add", [("IMM", 12), ("PSH", 0), ("IMM", 30), ("ADD", 0), ("HALT", 0)], {}))
    C.append(("sub", [("IMM", 100), ("PSH", 0), ("IMM", 58), ("SUB", 0), ("HALT", 0)], {}))
    C.append(("mul", [("IMM", 12), ("PSH", 0), ("IMM", 7), ("MUL", 0), ("HALT", 0)], {}))
    C.append(("div", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0), ("HALT", 0)], {}))
    C.append(("mod", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0), ("HALT", 0)], {}))
    C.append(("and", [("IMM", 0xF0), ("PSH", 0), ("IMM", 0x3C), ("AND", 0), ("HALT", 0)], {}))
    C.append(("shl", [("IMM", 3), ("PSH", 0), ("IMM", 4), ("SHL", 0), ("HALT", 0)], {}))
    C.append(("eq", [("IMM", 5), ("PSH", 0), ("IMM", 5), ("EQ", 0), ("HALT", 0)], {}))
    C.append(("lt", [("IMM", 3), ("PSH", 0), ("IMM", 5), ("LT", 0), ("HALT", 0)], {}))
    C.append(("li", [("IMM", 8), ("LI", 0), ("HALT", 0)], {8: 123}))
    C.append(("lea", [("LEA", 4), ("HALT", 0)], {}))
    C.append(("bz_taken", [("IMM", 0), ("BZ", 4), ("IMM", 99), ("HALT", 0),
                           ("IMM", 7), ("HALT", 0)], {}))
    C.append(("bnz_taken", [("IMM", 3), ("BNZ", 4), ("IMM", 99), ("HALT", 0),
                            ("IMM", 8), ("HALT", 0)], {}))
    C.append(("jmp", [("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)], {}))
    # JSR -> ENT frame -> set AX -> LEV back (call/return; memory stack + LEV heads).
    C.append(("jsr_lev", [("JSR", 3), ("IMM", 5), ("HALT", 0), ("ENT", 0),
                          ("IMM", 7), ("LEV", 0)], {}))
    # SI/LI round-trip through memory (store then load the same address).
    C.append(("si_li", [("IMM", 200), ("PSH", 0), ("IMM", 55), ("SI", 0),
                        ("IMM", 200), ("LI", 0), ("HALT", 0)], {}))
    return C


def _nested_prog(outer: int, inner: int) -> list:
    """Compile a byte-safe nested countdown (outer x inner) to ISA."""
    from src.compiler import compile_c
    from .run_1096_pure_forward import bytecode_to_isa
    src = (f"int main(){{ int a; int b; int r; b=0; r={outer}; "
           f"while(r>0){{ a={inner}; while(a>0){{ a=a-1; }} b=b+1; r=r-1; }} "
           f"return b; }}")
    bytecode, _data = compile_c(src)
    return bytecode_to_isa(bytecode)


def verify_composition(model, L, *, direct_cam=True, block_sparse=False,
                       verbose=True) -> bool:
    """Full-238 softmax path vs the COMPOSED fast path, byte-exact per-step AX."""
    composed = ComposedFastPath(model, L, direct_cam=direct_cam)
    ok = True
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        base, base_ap = _drive(model, L, code, composed=None, seed_mem=seed)
        comp, comp_ap = _drive(model, L, code, composed=composed, seed_mem=seed)
        match = base == comp
        ok = ok and match
        if verbose:
            ratio = f"{base_ap/comp_ap:.1f}x" if comp_ap else "n/a"
            print(f"  {name:10s} {'OK ' if match else 'FAIL'} "
                  f"steps={len(base)} applies full={base_ap} composed={comp_ap} "
                  f"({ratio} fewer)", flush=True)
            if not match:
                print(f"    BASE={base}\n    COMP={comp}", flush=True)

    # DEEP-LOOP gate: verify the composed path over a genuinely deep nested loop
    # against the FREE perfect draft.  The draft is the byte-exact ground truth
    # (the full-238 driver provably matches it — the speculation contract), so the
    # composed drive matching the draft's per-step AX proves the composition holds
    # across a deep loop WITHOUT paying the slow full-238 reference.  Bounded to
    # ``deep_steps`` steps (the harness re-embeds the whole growing stream each
    # step, ~O(steps^2), so we cap the reference — 60 steps already crosses the
    # inner loop several times AND the outer-loop boundary, exercising every
    # loop-carried block-skip transition; the fast path itself is unbounded).
    deep_steps = 60
    deep = _nested_prog(3, 4)
    draft = draft_pf_program(deep, max_steps=deep_steps, mask=0xFF)
    draft_ax = [f["ax"] & 0xFF for f in draft.frames]
    comp, comp_ap = _drive(model, L, deep, composed=composed, seed_mem={},
                           max_steps=deep_steps)
    n = min(len(comp), len(draft_ax))
    dmatch = comp[:n] == draft_ax[:n] and n > 0
    ok = ok and dmatch
    if verbose:
        print(f"  {'nested_deep':10s} {'OK ' if dmatch else 'FAIL'} "
              f"steps={n} (composed==free-draft over a deep nested loop, "
              f"{comp_ap} block-applies)", flush=True)
        if not dmatch:
            print(f"    DRAFT={draft_ax[:20]}...\n    COMP ={comp[:20]}...", flush=True)
    return ok


# ---------------------------------------------------------------------------
# CUDA-graph capture: one static graph per op-class (keyed by live-block SHAPE).
# The op's live blocks are applied over a FIXED-S static input buffer; the graph
# replays that fixed block sequence.  Attention is static-shaped at fixed S; the
# direct-CAM overwrite is a post-graph host op on the query row (O(1)).
# ---------------------------------------------------------------------------
class OpClassGraphs:
    """Capture + replay one CUDA graph per distinct live-block schedule.

    Keyed by the tuple of live block indices (the op-class shape).  Ops that
    share a live-block set share a graph (adf5fe2a: 10 distinct shapes).  The
    graph runs the block sequence on a static input/output buffer at fixed S.
    """

    def __init__(self, model, L, S: int, device):
        self.model = model
        self.L = L
        self.S = S
        self.device = device
        self.runner = StepBlockSkipRunner(model, L)
        D = model.embed.shape[1]
        self.static_in = torch.zeros(1, S, D, device=device,
                                     dtype=model.embed.dtype)
        # distinct live-block schedules (by the sorted index tuple).
        self.op_to_key: Dict[int, Tuple[int, ...]] = {}
        self.graphs: Dict[Tuple[int, ...], Tuple] = {}
        seen: Dict[Tuple[int, ...], List[int]] = {}
        for op, live in self.runner.live_index.items():
            if op is None:
                continue
            key = tuple(live)
            self.op_to_key[op] = key
            seen.setdefault(key, []).append(op)
        self.distinct_keys = list(seen.keys())
        self._seen = seen

    def _apply_live(self, x, live: List[int]):
        for bi in live:
            x = self.model.blocks[bi](x)
        return x

    def try_capture(self, key: Tuple[int, ...]) -> Tuple[str, Optional[float]]:
        """Attempt to capture a graph for one live-block schedule.  Returns
        (status, capture_ms).  status in {'ok', 'skip:<reason>'}."""
        live = list(key)
        # warm up (allocations, cudnn autotune) OUTSIDE the capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    _ = self._apply_live(self.static_in, live)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        try:
            t0 = time.time()
            with torch.no_grad():
                with torch.cuda.graph(g):
                    static_out = self._apply_live(self.static_in, live)
            torch.cuda.synchronize()
            cap_ms = (time.time() - t0) * 1e3
        except Exception as e:
            return (f"skip:{type(e).__name__}:{str(e).splitlines()[0][:60]}", None)
        self.graphs[key] = (g, static_out)
        return ("ok", cap_ms)

    def replay_ms(self, key: Tuple[int, ...], iters: int = 50,
                  warmup: int = 10) -> Optional[float]:
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


# ---------------------------------------------------------------------------
# BENCHMARK the composed ms/step at production S — per-op and weighted-mix, vs
# the full-238 baseline and vs each lever alone.
# ---------------------------------------------------------------------------
# A representative non-I/O op mix (typical compiled C4): mostly IMM/ADD/LI/branch
# with occasional MUL and rare DIV/MOD.  Weights sum to 1.
_OP_MIX = {
    isa.IMM: 0.24, isa.ADD: 0.14, isa.SUB: 0.06, isa.LI: 0.12, isa.SI: 0.06,
    isa.PSH: 0.10, isa.JMP: 0.05, isa.BZ: 0.05, isa.LEA: 0.04, isa.MUL: 0.03,
    isa.EQ: 0.03, isa.LT: 0.03, isa.SHL: 0.02, isa.LEV: 0.01, isa.JSR: 0.01,
    isa.DIV: 0.006, isa.MOD: 0.004,
}
_OP_LABEL = {v: k for k, v in {
    "IMM": isa.IMM, "ADD": isa.ADD, "SUB": isa.SUB, "MUL": isa.MUL, "DIV": isa.DIV,
    "MOD": isa.MOD, "SHL": isa.SHL, "LI": isa.LI, "SI": isa.SI, "JMP": isa.JMP,
    "BZ": isa.BZ, "PSH": isa.PSH, "LEA": isa.LEA, "EQ": isa.EQ, "LT": isa.LT,
    "LEV": isa.LEV, "JSR": isa.JSR,
}.items()}


def _time_fn(fn, n, warmup, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


def bench_at_S(model, L, S: int, *, n=30, warmup=8, use_graphs=True):
    """ms/step: full-238 vs block-skip-only vs composed(+direct-CAM+graph) at S.

    direct-CAM removes the CAM heads' softmax cost; in the eager block-skip
    forward that shows up as the ~O(1) post-forward overwrite (negligible), so
    the composed number is dominated by the live blocks' attention+FFN at S.
    The CUDA graph collapses the per-block kernel launches (the launch-bound
    tax at these live-block counts)."""
    dev = model.embed.device
    cuda = (dev.type == "cuda")
    runner = StepBlockSkipRunner(model, L)
    D = model.embed.shape[1]
    x0 = torch.zeros(1, S, D, device=dev, dtype=model.embed.dtype)

    print(f"\n{'='*74}\n[bench] production S={S}  (n={n}, warmup={warmup})\n{'='*74}",
          flush=True)

    # -- FULL 238-block baseline (dense forward, no skip) --------------------
    def full():
        x = x0
        with torch.no_grad():
            for blk in model.blocks:
                x = blk(x)
        return x
    t_full = _time_fn(full, n, warmup, cuda)
    nb = len(model.blocks)
    print(f"  FULL {nb}-block dense                    = {t_full:8.3f} ms/step",
          flush=True)

    # -- per-op: block-skip (eager) and composed (graph) --------------------
    graphs = None
    if use_graphs and cuda:
        graphs = OpClassGraphs(model, L, S, dev)

    per_op_skip: Dict[int, float] = {}
    per_op_graph: Dict[int, float] = {}
    graph_status: Dict[Tuple[int, ...], str] = {}
    print(f"\n  {'op':>5} {'live':>5} {'skip(eager)':>12} {'composed(graph)':>16} "
          f"{'vs full':>9}", flush=True)
    print("  " + "-" * 60, flush=True)
    for op in _OP_MIX:
        live = runner.live_index[op]
        def skip(op=op):
            with torch.no_grad():
                return runner.forward(x0, op)
        t_skip = _time_fn(skip, n, warmup, cuda)
        per_op_skip[op] = t_skip
        t_graph = None
        if graphs is not None:
            key = graphs.op_to_key[op]
            if key not in graphs.graphs and key not in graph_status:
                st, _cap = graphs.try_capture(key)
                graph_status[key] = st
            if graphs.graphs.get(key) is not None:
                t_graph = graphs.replay_ms(key, iters=n, warmup=warmup)
                per_op_graph[op] = t_graph
        lbl = _OP_LABEL.get(op, str(op))
        gtxt = f"{t_graph:8.3f}" if t_graph is not None else "   (no-graph)"
        best = t_graph if t_graph is not None else t_skip
        print(f"  {lbl:>5} {len(live):>5} {t_skip:9.3f}    {gtxt:>16} "
              f"{t_full/max(best,1e-9):8.1f}x", flush=True)

    # -- weighted-mix composed ms/step (the headline number) ----------------
    def _mix(times: Dict[int, float]) -> float:
        return sum(times[op] * w for op, w in _OP_MIX.items() if op in times)
    mix_full = t_full            # full is op-independent (always all blocks)
    mix_skip = _mix(per_op_skip)
    # composed uses the graph time where available, else the eager skip time.
    comp_times = {op: per_op_graph.get(op, per_op_skip[op]) for op in _OP_MIX}
    mix_comp = _mix(comp_times)
    print(f"\n  --- WEIGHTED OP-MIX ms/step at S={S} ---", flush=True)
    print(f"    full-{nb} dense           : {mix_full:8.3f} ms/step  (1.0x)",
          flush=True)
    print(f"    + block-skip (eager)     : {mix_skip:8.3f} ms/step  "
          f"({mix_full/mix_skip:.1f}x)", flush=True)
    print(f"    + direct-CAM + CUDA-graph: {mix_comp:8.3f} ms/step  "
          f"({mix_full/mix_comp:.1f}x)   <-- COMPOSED", flush=True)
    print(f"    gap to 0.1 ms goal       : {mix_comp/0.1:.1f}x above 0.1 ms",
          flush=True)

    # DIV/MOD bottleneck quantification: their share of the weighted composed cost.
    dm_ops = [isa.DIV, isa.MOD]
    dm_cost = sum(comp_times[op] * _OP_MIX[op] for op in dm_ops if op in comp_times)
    dm_w = sum(_OP_MIX[op] for op in dm_ops)
    print(f"\n  DIV/MOD bottleneck: {dm_w*100:.1f}% of steps but "
          f"{100*dm_cost/max(mix_comp,1e-9):.1f}% of the composed weighted cost "
          f"(live={runner.live_index[isa.DIV] and len(runner.live_index[isa.DIV])} "
          f"blocks vs ~7 for non-divmod)", flush=True)

    if graphs is not None:
        n_ok = sum(1 for k in graphs.distinct_keys if graphs.graphs.get(k) is not None)
        print(f"\n  CUDA graphs: captured {n_ok}/{len(graphs.distinct_keys)} distinct "
              f"op-class shapes cleanly", flush=True)
        for k in sorted(graphs.distinct_keys, key=len):
            ops = ",".join(_OP_LABEL.get(o, str(o)) for o in graphs._seen[k])
            st = "ok" if graphs.graphs.get(k) is not None else graph_status.get(k, "?")
            print(f"    shape live={len(k):3d}  [{ops}]  -> {st}", flush=True)
        # release the captured graphs + their pinned private pools before returning,
        # so the next (larger-S) run does not OOM on the retained ~6 GB graph pools.
        graphs.graphs.clear()
        del graphs
    return {"S": S, "full": mix_full, "skip": mix_skip, "composed": mix_comp}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--S", type=str, default="900,3000",
                    help="comma list of production seq lengths to bench.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64,
                    help="code slots (>= largest battery program; nested loop ~57).")
    ap.add_argument("--block-sparse-ffn", action="store_true",
                    help="also install the COO block-sparse FFN on the live blocks.")
    ap.add_argument("--no-graphs", action="store_true",
                    help="skip CUDA-graph capture (eager block-skip only).")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true",
                    help="skip the GPU wait-loop (assume the card is free).")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the byte-exact composition verify (bench only).")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[composed] CUDA unavailable; falling back to cpu", file=sys.stderr)
        device = "cpu"

    idx = 0
    if device.startswith("cuda") and ":" in device:
        idx = int(device.split(":")[1])
    if device.startswith("cuda") and not args.no_wait:
        wait_for_gpu(idx, min_free_gb=args.min_free_gb, stable_s=args.stable_s)

    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="dense_kernel")
    if args.block_sparse_ffn:
        from .block_sparse_ffn import install_block_sparse_ffn
        install_block_sparse_ffn(model, mode="coo", verbose=True)
    if device != "cpu":
        model.to(device)
        if not args.block_sparse_ffn:
            model.materialize_dense(device)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"dev={device} bsf={args.block_sparse_ffn} "
          f"build={time.time()-t0:.1f}s", flush=True)

    # --- byte-exact composition verify (the gate) --------------------------
    if args.no_verify:
        print("\n[verify] SKIPPED (--no-verify); bench only", flush=True)
        ok = True
    else:
        print("\n[verify] full-238 softmax path == COMPOSED "
              "(block-skip + direct-CAM):", flush=True)
        ok = verify_composition(model, L, direct_cam=True,
                                block_sparse=args.block_sparse_ffn)
        print(f"[verify] {'ALL BYTE-EXACT' if ok else 'DIVERGENCE FOUND'}",
              flush=True)
    if not ok:
        print("[verify] composition is NOT byte-exact -> aborting bench", flush=True)
        return 1

    # --- benchmark at each production S ------------------------------------
    # Free the previous S's CUDA graphs + activation cache between runs — a
    # captured graph pins its private memory pool (~6 GB for 15 graphs at S=900),
    # which would OOM the next (larger-S) baseline.  gc + empty_cache reclaims it.
    import gc
    for S in (int(s) for s in args.S.split(",") if s.strip()):
        bench_at_S(model, L, S, n=args.n, use_graphs=(not args.no_graphs))
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
