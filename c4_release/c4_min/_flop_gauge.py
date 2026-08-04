#!/usr/bin/env python3
"""_flop_gauge.py — the REUSABLE roofline gauge for the doom-transformer forward.

Measures the ACTUAL per-step FLOPs the doom c4_min pure-forward model EXECUTES,
so #845 (the 400K-steps/sec megakernel push) has a real roofline instead of the
long-standing "~1 MFLOP/step" ESTIMATE.  Re-runnable after each optimization.

WHAT IT MEASURES, per opcode and per component (FFN / attention / direct-CAM /
register-emit), THREE numbers so the overhead is visible:

  (a) dense-equiv FLOP  — what a FULLY-DENSE forward would cost (every weight
      multiplied, no sparsity / block-skip): 2*out*in per weight over N rows.
  (b) actual-executed FLOP — the REAL number: only the op's live blocks run
      (block-skip), each sparse weight charges 2*nnz*N (the COO / CSR kernel
      does exactly nnz multiply-accumulates per row), attention scores/context
      charge only over the ACTUAL KV rows, and the direct-CAM gather is ~0 FLOP
      (bytes-moved only).
  (c) theoretical-useful FLOP — the c4 INSTRUCTION's real arithmetic (~1-100
      FLOP: an ADD is 1 add, a DIV is ~a few divides, a PSH is ~0).

The custom sparse kernels (COO `CooLinear`, CSR `SparseWeight`, the direct-CAM
gather) are INVISIBLE to a plain torch aten FLOP counter, so we instrument them
directly (monkeypatched `.linear` accumulators that count nnz x N = real MACs)
AND cross-check the aten GEMM/BMM/einsum via `torch.utils.flop_counter`.

Then: FLOP-UTIL = actual-executed-FLOP / (measured ms/step x GPU-peak-FLOP/s),
and the 400K-feasibility = actual-FLOP x 400K vs the A5000 peak.

MEMORY: lean streaming build (C4_PF_CFM=1), ~1.5 GB RSS; NEVER densifies the
non-CFM ~108 GB path.  Golden 069cc32f byte-identical (measurement-only: no
weight write, all flags default).

Run:
    cd c4_release
    C4_PF_CFM=1 OMP_NUM_THREADS=4 python -m c4_min._flop_gauge --device cuda:0
    # narrow / options:
    python -m c4_min._flop_gauge --device cuda:0 --ops IMM,ADD,DIV --no-aten
    python -m c4_min._flop_gauge --device cpu   # CPU: counts only (no util/timing peak)
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# LEAN STREAMING is MANDATORY: code-from-memory keeps dim fixed (~1392) so the
# build stays ~1.5 GB.  Set before any c4_min import touches the build path.
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG

# A5000 vendor peaks (FLOP/s).
A5000_FP32 = 27.8e12
A5000_TF32 = 55.6e12
DEFAULT_PEAK = A5000_FP32


# =========================================================================== #
# Memory guard — the session-long OOM lesson: check /proc/meminfo, stop < 25 GB.
# =========================================================================== #
def _mem_avail_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return 1e9  # unknown -> don't block


def _mem_guard(stop_gb: float = 25.0, where: str = "") -> None:
    avail = _mem_avail_gb()
    if avail < stop_gb:
        raise SystemExit(
            f"[MEM-GUARD] MemAvailable={avail:.1f} GB < {stop_gb} GB "
            f"({where}) — STOP (OOM-kills the box + all agents).")


# =========================================================================== #
# (1) KERNEL INSTRUMENTATION — count the ACTUAL MACs the custom sparse kernels do.
#
# The COO `CooLinear.linear` and the CSR `SparseWeight.linear` each do exactly
# `nnz` multiply-accumulates per input row (one MAC per stored nonzero).  A plain
# torch FLOP counter sees only `index_select`/`scatter_add` (0 FLOP) or a
# to_dense()+F.linear (dense-equiv, WRONG for the real cost).  So we wrap `.linear`
# with an accumulator keyed by the weight's role — the ground-truth executed MACs.
# =========================================================================== #
@dataclass
class KernelTally:
    """Accumulated executed MACs and dense-equiv MACs, per component tag."""
    exec_macs: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    dense_macs: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    cam_bytes: float = 0.0            # direct-CAM gather: bytes moved (0 FLOP)
    n_weight_calls: int = 0

    def reset(self):
        self.exec_macs = defaultdict(float)
        self.dense_macs = defaultdict(float)
        self.cam_bytes = 0.0
        self.n_weight_calls = 0

    def add_weight(self, tag: str, nnz: int, out_dim: int, in_dim: int,
                   n_rows: int, is_sparse: bool):
        self.n_weight_calls += 1
        self.dense_macs[tag] += float(out_dim) * float(in_dim) * float(n_rows)
        # a sparse weight (COO/CSR) does nnz MACs/row; a kept-dense weight does the
        # full product (out*in) — but the kept-dense ones ARE the tiny/dense tail,
        # so charge them their true nnz too (== out*in for a fully-dense weight,
        # <= for a mostly-zero dense-kept weight, which is the honest executed cost).
        macs = float(nnz) * float(n_rows)
        self.exec_macs[tag] += macs


# The single active tally the wrappers write to (None -> passthrough, no count).
_ACTIVE_TALLY: Optional[KernelTally] = None
# Component tag for the block currently executing (set by the driver around blk()).
_ACTIVE_TAG: str = "other"


def _tag_of_block_name(name: str) -> str:
    """Map a block NAME to a coarse component bucket for the per-component table."""
    n = name or ""
    if n.startswith("alu-div"):
        return "ffn_divmod"
    if n.startswith("alu-") or n.startswith("bw-") or n.startswith("cmp-"):
        return "ffn_alu"
    if n in ("mem-cam", "stack-pop-cam", "lev-addr4", "mem-prep", "stack-prep",
             "pop-addr"):
        return "attn_cam"
    if n in ("ingest+recompose",):
        return "attn_ingest"
    if n in ("pc-fetch", "code-select"):
        return "attn_fetch"
    return "ffn_other"


@contextlib.contextmanager
def _instrument_kernels(tally: KernelTally):
    """Monkeypatch the custom sparse kernels' `.linear` to tally executed MACs.

    Wraps: SparseWeight.linear (CSR / dense_kernel), CooLinear.linear (COO),
    DenseActiveLinear.linear (compact dense).  Restored on exit.  Byte-identical:
    the wrapper calls the ORIGINAL kernel unchanged and only records shapes/nnz.
    """
    from . import sparse_forward as SF
    patched = []

    def wrap(cls, nnz_attr="nnz"):
        if not hasattr(cls, "linear"):
            return
        orig = cls.linear

        def linear(self, x, _orig=orig):
            global _ACTIVE_TALLY, _ACTIVE_TAG
            out = _orig(self, x)
            t = _ACTIVE_TALLY
            if t is not None:
                n_rows = 1
                for d in x.shape[:-1]:
                    n_rows *= int(d)
                nnz = int(getattr(self, "nnz", 0) or 0)
                is_sparse = bool(getattr(self, "is_sparse", True))
                if nnz == 0:
                    # COO/dense_active carry no `.nnz` on older builds; recover it.
                    if hasattr(self, "vals"):
                        nnz = int(self.vals.numel())
                    elif hasattr(self, "wsub"):
                        nnz = int((self.wsub != 0).sum().item())
                    else:
                        d = getattr(self, "dense", None)
                        if d is not None:
                            nnz = int((d != 0).sum().item())
                t.add_weight(_ACTIVE_TAG, nnz, int(self.out_dim), int(self.in_dim),
                             n_rows, is_sparse)
            return out

        cls.linear = linear
        patched.append((cls, orig))

    wrap(SF.SparseWeight)
    try:
        from . import block_sparse_ffn as BSF
        wrap(BSF.CooLinear)
        wrap(BSF.DenseActiveLinear)
    except Exception:
        pass
    try:
        yield
    finally:
        for cls, orig in patched:
            cls.linear = orig


# =========================================================================== #
# (2) ATTENTION — the score/context einsum is NOT a `.linear`; count it directly.
#
# For every live attention block we count the ACTUAL attention arithmetic over the
# ACTUAL KV rows the query attends to: scores = Q@Kᵀ (2*Sq*Sk*HD per head), context
# = attn@V (2*Sq*Sk*HD per head), plus the softmax/ALiBi elementwise (~a few *
# Sq*Sk*H, small).  We read S (rows) from the block's forward input, not a bound.
# =========================================================================== #
def _attn_flops(block, B, Sq, Sk) -> Tuple[float, float]:
    """(executed_attn_flop, dense_equiv_attn_flop) for one attention block.

    executed = over ACTUAL Sk (the causal / stored rows the step really has);
    dense-equiv = the same shapes (attention is already O(S*S), no weight-sparsity
    win in the score matmul) — so the attention exec≈dense-equiv, the sparsity win
    is entirely in the Q/K/V/O PROJECTIONS (counted as weights above).
    """
    attn = getattr(block, "attn", None)
    if attn is None:
        return 0.0, 0.0
    H = int(getattr(attn, "n_heads", 0) or 0)
    HD = int(getattr(attn, "head_dim", 0) or 0)
    if H == 0 or HD == 0:
        return 0.0, 0.0
    # scores Q@Kᵀ : [B,H,Sq,HD]x[B,H,HD,Sk] -> 2*B*H*Sq*Sk*HD
    # context attn@V: [B,H,Sq,Sk]x[B,H,Sk,HD] -> 2*B*H*Sq*Sk*HD
    macs = 2.0 * float(B) * H * Sq * Sk * HD          # score + context (2*  each side)
    flop = 2.0 * macs                                  # MAC -> 2 FLOP
    return flop, flop


# =========================================================================== #
# (3) USEFUL FLOP — the c4 instruction's real arithmetic (the denominator of the
# overhead ratio).  A hand table of the minimal scalar ops each opcode performs.
# =========================================================================== #
_USEFUL_FLOP: Dict[int, float] = {
    isa.IMM: 0.0, isa.LEA: 1.0, isa.JMP: 0.0, isa.JSR: 1.0, isa.ENT: 1.0,
    isa.ADJ: 1.0, isa.LEV: 2.0, isa.PSH: 0.0, isa.LI: 0.0, isa.LC: 0.0,
    isa.SI: 0.0, isa.SC: 0.0,
    isa.ADD: 1.0, isa.SUB: 1.0, isa.MUL: 1.0, isa.DIV: 1.0, isa.MOD: 1.0,
    isa.OR: 1.0, isa.XOR: 1.0, isa.AND: 1.0, isa.SHL: 1.0, isa.SHR: 1.0,
    isa.EQ: 1.0, isa.NE: 1.0, isa.LT: 1.0, isa.GT: 1.0, isa.LE: 1.0, isa.GE: 1.0,
}


# =========================================================================== #
# (4) THE MEASURE-ONE-OP CORE — force the opcode-under-test and run EXACTLY its
# block-skip live set over a representative stream, tallying per-block executed /
# dense / attention FLOPs.
#
# WHY FORCE THE OP (not decode-navigate to it): the whole point of block-skip is
# that the live-block SET is STATIC per op-class — so the FLOPs a DIV step executes
# are entirely determined by `runner._live_masks[DIV]`, independent of the decoded
# operands.  Forcing the op measures precisely that static set (the #845 megakernel
# will be one graph PER op-class), and decouples the FLOP gauge from decode drift
# (the per-step re-embed driver's decode fidelity is a SEPARATE concern the
# byte-exact tests own — irrelevant to how many FLOPs the op's blocks run).
# The blocks are called UNCHANGED (byte-identical); the tally only reads shapes.
# =========================================================================== #
def _make_stream(L, S_tokens: int, seed_mem=None):
    """Build a representative embedded-input stream of ~`S_tokens` tokens: a BOS +
    a growing sequence of register frames (the exact frame layout the driver emits).
    Returns (stream_tokens, store_log)."""
    from . import nibble_pure_forward_complete as pfc
    from .nibble_pure_forward_complete import _build_frame, SP_INIT, _seed_frames
    seed_frames, store_log = _seed_frames(seed_mem or {})
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    sp = SP_INIT
    # append plausible frames until we reach the target token length.
    pc = 1
    while len(stream) < S_tokens:
        sp -= 4
        stream += _build_frame(pc, pc * 3 + 7, sp, SP_INIT, pc, mem_addr=sp,
                               mem_val=(pc & 0xFF))
        store_log[len(store_log) + 1] = (sp, pc & 0xFF)
        pc += 1
    return stream, store_log


def _measure_one_op(model, L, runner, op, S_tokens, *, seed_mem=None,
                    aten=False) -> dict:
    """Run op's live-block set over a representative stream; return a FLOP record.

    rec = {op, S, live_blocks, exec_macs (FFN weights), dense_macs, attn_exec,
    attn_dense, exec_by_tag, dense_by_tag, attn_by_tag, aten}."""
    global _ACTIVE_TALLY, _ACTIVE_TAG
    from . import nibble_pure_forward_complete as pfc
    from .nibble_pure_forward_complete import make_overlay_complete

    dev = model.embed.device
    names = list(getattr(L, "_block_names", []))
    stream, store_log = _make_stream(L, S_tokens, seed_mem)
    # a dummy 1-op code so the overlay has a code frame to write (op is FORCED below).
    code = isa.assemble([("IMM", 0), ("HALT", 0)])
    overlay = make_overlay_complete(code, L, store_log=store_log)
    toks = torch.tensor([stream], device=dev)
    S = len(stream)

    rec = {"op": op, "S": S, "exec_macs": 0.0, "dense_macs": 0.0,
           "attn_exec": 0.0, "attn_dense": 0.0,
           "exec_by_tag": defaultdict(float), "dense_by_tag": defaultdict(float),
           "attn_by_tag": defaultdict(float), "live_blocks": 0, "aten": None}

    with torch.no_grad():
        x = model.embed[toks].clone()
        overlay(x)
        mask = runner._live_masks.get(op)
        live_bi = [i for i in range(runner.n_blocks) if (mask is None or mask[i])]
        rec["live_blocks"] = len(live_bi)

        step_tally = KernelTally()
        _ACTIVE_TALLY = step_tally
        aten_ctx = _maybe_aten(aten)
        with aten_ctx as fcm:
            for bi in live_bi:
                blk = runner.blocks[bi]
                nm = names[bi] if bi < len(names) else ""
                _ACTIVE_TAG = _tag_of_block_name(nm)
                # per-step re-embed: no KV cache, every block sees all S rows causal.
                a_exec, a_dense = _attn_flops(blk, 1, S, S)
                rec["attn_exec"] += a_exec
                rec["attn_dense"] += a_dense
                rec["attn_by_tag"][_ACTIVE_TAG] += a_exec
                x = blk(x)
            _ACTIVE_TALLY = None
        if aten and fcm is not None:
            rec["aten"] = _sum_aten(fcm)

    for tag, m in step_tally.exec_macs.items():
        rec["exec_macs"] += 2.0 * m            # MAC -> 2 FLOP (nonzero-only)
        rec["exec_by_tag"][tag] += 2.0 * m
    for tag, m in step_tally.dense_macs.items():
        rec["dense_macs"] += 2.0 * m
        rec["dense_by_tag"][tag] += 2.0 * m
    return rec


def _maybe_aten(active: bool):
    if not active:
        return contextlib.nullcontext()
    try:
        from torch.utils.flop_counter import FlopCounterMode
        return FlopCounterMode(display=False)
    except Exception:
        return contextlib.nullcontext()


def _sum_aten(fcm) -> float:
    try:
        return float(fcm.get_total_flops())
    except Exception:
        return 0.0


# =========================================================================== #
# (5) PER-OP BATTERY — the representative opcode set.  Each op's FLOPs are fully
# determined by its STATIC block-skip live set (forced, decode-independent), so
# the battery is just the opcodes to measure.
# =========================================================================== #
def _battery() -> List[Tuple[str, int]]:
    """(label, opcode) — the representative op battery (DIV/MOD heaviest via the
    179-block divmod span; IMM/PSH lightest)."""
    names = ["IMM", "LEA", "JMP", "PSH", "LI", "LC", "SI", "SC",
             "JSR", "ENT", "LEV",
             "ADD", "SUB", "MUL", "DIV", "MOD",
             "AND", "OR", "XOR", "SHL", "SHR",
             "EQ", "NE", "LT", "GT"]
    return [(nm, getattr(isa, nm)) for nm in names]


# =========================================================================== #
# (6) REPORTING
# =========================================================================== #
def _fmt(x: float) -> str:
    if x >= 1e9:
        return f"{x/1e9:8.3f}G"
    if x >= 1e6:
        return f"{x/1e6:8.3f}M"
    if x >= 1e3:
        return f"{x/1e3:8.3f}K"
    return f"{x:9.1f}"


def _time_op_step(model, L, runner, op, S_tokens, seed_mem, *, reps=20,
                  warmup=5) -> float:
    """Wall-clock ms for ONE forced-op forward (its static block-skip live set over
    an `S_tokens`-token stream).  Same op / same stream the FLOP count measured."""
    from . import nibble_pure_forward_complete as pfc
    from .nibble_pure_forward_complete import make_overlay_complete

    dev = model.embed.device
    cuda = dev.type == "cuda"
    stream, store_log = _make_stream(L, S_tokens, seed_mem)
    code = isa.assemble([("IMM", 0), ("HALT", 0)])
    overlay = make_overlay_complete(code, L, store_log=store_log)
    toks = torch.tensor([stream], device=dev)

    def fwd():
        with torch.no_grad():
            x = model.embed[toks].clone(); overlay(x)
            return runner.forward(x, op)

    for _ in range(warmup):
        fwd()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fwd()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=48)
    ap.add_argument("--ops", default="",
                    help="comma list to restrict (default: full battery)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--stream-len", type=int, default=91,
                    help="representative stream token length S (the per-step "
                         "re-embed forward sees the whole growing stream; doom "
                         "mid-render S is a few hundred — try --stream-len 300)")
    ap.add_argument("--peak-tflops", type=float, default=None,
                    help="GPU peak TFLOP/s (default: A5000 fp32 27.8)")
    ap.add_argument("--target-steps-per-sec", type=float, default=400_000.0)
    ap.add_argument("--no-aten", action="store_true",
                    help="skip the torch FlopCounterMode aten cross-check")
    ap.add_argument("--no-time", action="store_true",
                    help="skip wall-clock timing (FLOP counts only)")
    ap.add_argument("--block-sparse", dest="block_sparse", action="store_true",
                    default=True, help="install COO block-sparse FFN (default on)")
    ap.add_argument("--no-block-sparse", dest="block_sparse", action="store_false")
    a = ap.parse_args(argv)

    _mem_guard(where="startup")
    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> cpu (no util/timing peak)", flush=True)
        device = "cpu"
    cuda = device.startswith("cuda")
    peak = (a.peak_tflops * 1e12) if a.peak_tflops else DEFAULT_PEAK
    if cuda and a.peak_tflops is None:
        try:
            nm = torch.cuda.get_device_name(0).lower()
            if "a5000" in nm:
                peak = A5000_FP32
        except Exception:
            pass

    print("=" * 100, flush=True)
    print("DOOM-TRANSFORMER FLOP GAUGE — actual per-step FLOPs (the #845 roofline)",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  flags: C4_PF_CFM={os.environ.get('C4_PF_CFM')} (lean streaming)  "
          f"block_sparse_ffn={'COO' if a.block_sparse else 'off'}  device={device}",
          flush=True)

    # ---- build (lean streaming) ----
    _mem_guard(where="pre-build")
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _stats = build_compact_sparse_streaming(
        code_size=a.code_size, compute_mode="dense_kernel")
    if cuda:
        model.to(device)
        model.materialize_dense(device)
    _mem_guard(where="post-build")
    if a.block_sparse:
        from .block_sparse_ffn import install_block_sparse_ffn
        install_block_sparse_ffn(model, mode="coo", verbose=False)
        if cuda:
            model.to(device)
    _mem_guard(where="post-install")

    from .step_block_skip import StepBlockSkipRunner
    runner = StepBlockSkipRunner(model, L)
    D = model.embed.shape[1]
    n_blocks = len(model.blocks)
    print(f"  built: {n_blocks} blocks  dim={D}  build={time.time()-t0:.1f}s  "
          f"MemAvail={_mem_avail_gb():.1f}GB", flush=True)
    if cuda:
        print(f"  GPU peak (fp32) = {peak/1e12:.1f} TFLOP/s  "
              f"(TF32 {A5000_TF32/1e12:.1f})", flush=True)

    battery = _battery()
    if a.ops:
        want = {s.strip().upper() for s in a.ops.split(",") if s.strip()}
        battery = [b for b in battery if b[0].upper() in want]

    print(f"  stream_len S = {a.stream_len} tokens (per-step re-embed; attention "
          f"cost is O(S^2))", flush=True)

    # ---- per-op measurement (force the opcode, count its static live set) ----
    rows = []
    comp_accum = defaultdict(lambda: defaultdict(float))  # component -> exec/dense
    for label, opcode in battery:
        _mem_guard(where=f"op {label}")
        aten = (not a.no_aten and cuda)
        with _instrument_kernels(KernelTally()):
            rec = _measure_one_op(model, L, runner, opcode, a.stream_len, aten=aten)
        exec_flop = rec["exec_macs"] + rec["attn_exec"]
        dense_flop = rec["dense_macs"] + rec["attn_dense"]
        useful = _USEFUL_FLOP.get(opcode, 1.0)

        # per-component accumulation (for the component table): FFN weights + attn.
        for tag, v in rec["exec_by_tag"].items():
            comp_accum[tag]["exec"] += v
            comp_accum[tag]["dense"] += rec["dense_by_tag"].get(tag, 0.0)
        for tag, v in rec["attn_by_tag"].items():
            comp_accum["attn_score_ctx"]["exec"] += v
            comp_accum["attn_score_ctx"]["dense"] += v

        ms = None
        if not a.no_time and cuda:
            ms = _time_op_step(model, L, runner, opcode, a.stream_len, None,
                               reps=a.reps)
        util = None
        req_frac = None
        if ms is not None and ms > 0:
            achieved = exec_flop / (ms / 1e3)
            util = achieved / peak * 100.0
            req = exec_flop * a.target_steps_per_sec
            req_frac = req / peak * 100.0

        rows.append({
            "label": label, "op": opcode, "S": rec["S"],
            "live": rec["live_blocks"],
            "dense": dense_flop, "exec": exec_flop, "useful": useful,
            "attn_exec": rec["attn_exec"], "ffn_exec": rec["exec_macs"],
            "ms": ms, "util": util, "aten": rec["aten"], "req_frac": req_frac,
        })
        _print_op_row(rows[-1], peak)

    # ---- TABLES ----
    _print_op_table(rows, peak, a.target_steps_per_sec)
    _print_component_table(comp_accum)
    _print_verdict(rows, peak, a.target_steps_per_sec, D)
    return 0


def _print_op_row(r, peak):
    ratio_de = r["dense"] / r["exec"] if r["exec"] else 0.0
    ratio_uo = r["exec"] / r["useful"] if r["useful"] else 0.0
    ms = f"{r['ms']:.3f}" if r["ms"] is not None else "   -  "
    util = f"{r['util']:.4f}%" if r["util"] is not None else "  -   "
    aten = _fmt(r["aten"]) if r["aten"] is not None else "   -    "
    print(f"  {r['label']:5s} live={r['live']:3d} S={r['S']:3d} | "
          f"dense={_fmt(r['dense'])} exec={_fmt(r['exec'])} useful={r['useful']:5.0f} | "
          f"D/E={ratio_de:7.1f}x E/U={ratio_uo:11.0f}x | "
          f"ms={ms} util={util} aten={aten}", flush=True)


def _print_op_table(rows, peak, tgt):
    print("\n" + "=" * 100, flush=True)
    print("PER-OP FLOP TABLE  (dense-equiv | actual-executed | useful; ms/step; util)",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  {'op':5s} {'live':>4s} {'dense-eq':>10s} {'ACTUAL':>10s} "
          f"{'useful':>7s} {'D/E':>7s} {'ms/step':>8s} {'util%':>9s} "
          f"{'400K-req%peak':>13s}", flush=True)
    for r in sorted(rows, key=lambda x: -x["exec"]):
        ms = f"{r['ms']:.3f}" if r["ms"] is not None else "-"
        util = f"{r['util']:.4f}" if r["util"] is not None else "-"
        reqf = f"{r['req_frac']:.2f}" if r["req_frac"] is not None else "-"
        de = r["dense"] / r["exec"] if r["exec"] else 0.0
        print(f"  {r['label']:5s} {r['live']:4d} {_fmt(r['dense']):>10s} "
              f"{_fmt(r['exec']):>10s} {r['useful']:7.0f} {de:6.0f}x "
              f"{ms:>8s} {util:>9s} {reqf:>13s}", flush=True)


def _print_component_table(comp_accum):
    print("\n" + "=" * 100, flush=True)
    print("PER-COMPONENT FLOP  (summed executed over the battery — where FLOPs GO)",
          flush=True)
    print("-" * 100, flush=True)
    total = sum(c["exec"] for c in comp_accum.values()) or 1.0
    order = sorted(comp_accum.items(), key=lambda kv: -kv[1]["exec"])
    for tag, c in order:
        pct = c["exec"] / total * 100.0
        de = c["dense"] / c["exec"] if c["exec"] else 0.0
        print(f"  {tag:18s} exec={_fmt(c['exec']):>10s} "
              f"({pct:5.1f}% of total)  dense-eq={_fmt(c['dense']):>10s}  "
              f"D/E={de:6.0f}x", flush=True)
    print(f"  {'TOTAL':18s} exec={_fmt(total):>10s}", flush=True)


def _print_verdict(rows, peak, tgt, D):
    print("\n" + "=" * 100, flush=True)
    print("VERDICT — roofline for #845", flush=True)
    print("-" * 100, flush=True)
    light = [r for r in rows if r["label"] in ("IMM", "PSH")]
    heavy = [r for r in rows if r["label"] in ("DIV", "MOD")]
    if light:
        lo = min(r["exec"] for r in light)
        print(f"  lightest ops (IMM/PSH): {_fmt(lo)} FLOP/step "
              f"({'BELOW' if lo < 1e6 else 'AT/ABOVE'} the 1-MFLOP estimate)",
              flush=True)
    if heavy:
        hi = max(r["exec"] for r in heavy)
        print(f"  heaviest ops (DIV/MOD): {_fmt(hi)} FLOP/step  (the 179-block divmod)",
              flush=True)
    utilrows = [r for r in rows if r["util"] is not None]
    if utilrows:
        umin = min(r["util"] for r in utilrows)
        umax = max(r["util"] for r in utilrows)
        print(f"  measured FLOP-UTIL range: {umin:.4f}% .. {umax:.4f}%  "
              f"(the ~0.2%% estimate {'CONFIRMED' if umin<=0.5 else 'CORRECTED'})",
              flush=True)
    # where the FLOPs go: attention (score+ctx, O(S^2)) vs FFN weights (sparse).
    tot_attn = sum(r["attn_exec"] for r in rows)
    tot_ffn = sum(r["ffn_exec"] for r in rows)
    tot = tot_attn + tot_ffn
    if tot:
        print(f"  where the FLOPs GO: attention(score+ctx) = {tot_attn/tot*100:.1f}%,"
              f"  FFN weights(sparse COO) = {tot_ffn/tot*100:.2f}%  "
              f"-> attention is O(blocks x 24heads x S^2 x HD), the dominant term",
              flush=True)

    import statistics
    mean_exec = statistics.mean(r["exec"] for r in rows)
    req = mean_exec * tgt
    print(f"  400K-steps/sec feasibility (mean-op {_fmt(mean_exec)} FLOP/step "
          f"@ S={rows[0]['S']}):", flush=True)
    print(f"    required = {_fmt(req)} FLOP/s = {req/peak*100:.1f}% of the "
          f"A5000 {peak/1e12:.1f} TFLOP/s peak  -> "
          f"{'FLOP-FEASIBLE' if req < peak else 'FLOP-BOUND at this S'}",
          flush=True)
    heavy_exec = max((r["exec"] for r in heavy), default=mean_exec)
    reqh = heavy_exec * tgt
    print(f"    worst-op (DIV) 400K required = {_fmt(reqh)} FLOP/s = "
          f"{reqh/peak*100:.1f}% of peak", flush=True)

    # PROJECTED FLOOR for #845: the per-step re-embed computes attention for ALL S
    # query rows, but only the LAST (decode) row is used.  Computing attention for
    # Sq=1 (the decode row only — the obvious kernel win) drops the O(S^2) term to
    # O(S), i.e. attention/S.  Show the resulting mean-op FLOP + 400K-feasibility.
    S = rows[0]["S"]
    q1_attn = tot_attn / max(S, 1)                 # Sq=1: attention/S
    q1_mean = statistics.mean(
        (r["attn_exec"] / max(S, 1) + r["ffn_exec"]) for r in rows)
    q1_req = q1_mean * tgt
    print(f"  PROJECTED FLOOR (Sq=1 decode-row-only attention, the #845 lever):",
          flush=True)
    print(f"    mean-op -> {_fmt(q1_mean)} FLOP/step  (~{tot_attn/max(q1_attn,1):.0f}x "
          f"less attention); 400K required = {q1_req/peak*100:.1f}% of peak  -> "
          f"{'FLOP-FEASIBLE' if q1_req < peak else 'still FLOP-bound (shrink S / heads / divmod)'}",
          flush=True)
    print("\n  NOTE: util is low because a c4 VM step is ~1-100 USEFUL FLOP but the "
          "transformer\n  executes ~M FLOP of overhead (framing / decode / the "
          "179-block divmod), NOT because\n  FLOPs are wasted on zeros — the "
          "block-skip + COO sparse path already removed those\n  (see the D/E "
          "column: dense-equiv is ~10^3-10^4x the executed number).", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
