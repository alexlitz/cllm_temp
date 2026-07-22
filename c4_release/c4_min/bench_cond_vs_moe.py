"""END-TO-END ms/step: DENSE vs BLOCK-MoE (#628) vs CONDITIONAL fine-grained.

This is the HONEST net-win bench the conditional-sparsity story needs.  The prior
``bench_conditional_sparse`` / ``bench_conditional_full`` measured conditional vs
*dense*, but the model ALREADY ships block-MoE (#628, ``block_moe_divmod``): a
COARSE per-op route that skips the whole DIV/MOD megablock span when the step's op
is not DIV/MOD.  Much of the 79-379x "conditional beats dense" number OVERLAPS with
what block-MoE already gives for free.  The NET-NEW win of FINE-GRAINED (within an
active layer's units) conditional sparsity is the number that matters -- so this
bench wires all THREE configs into the SAME live spec driver and reports both
``conditional / dense`` (the headline the prior bench gave) AND ``conditional /
block-MoE`` (the honest net win over the baseline the model already has).

The three configs, all as a per-layer active-unit set fed to ``ConditionalBlockLean``
(a gather + dense-GEMM, no torch BSR):

  (a) DENSE       -- every layer keeps ALL ``I`` units (== ``LeanQwenVM.forward``).
  (b) BLOCK-MoE   -- the lean-forward dual of #628.  A layer is MoE-ACTIVE for the
                    program's op-cluster iff ANY of its ``I`` units fires for ANY
                    step in the batch; a MoE-active layer keeps ALL ``I`` units (the
                    coarse "run the whole block" route), a MoE-dead layer keeps 0
                    (the "skip the whole block" route == #628's identity expert).
                    This is EXACTLY the #628 granularity mapped onto the compacted
                    lean layers: it skips the op-specific megablock layer (FULL L9,
                    I=160465, fires only DIV/MOD) when the program avoids that op,
                    but keeps every touched layer fully dense.
  (c) CONDITIONAL -- every layer keeps ONLY the units that fire (the fine-grained
                    within-layer block).

Byte-identity: (a)==(b)==(c) decode the same register trace (thr=0 active sets drop
only units whose ``silu(up)·gate`` is exactly 0 for every row -> L-inf-0 vs dense).
``verify_conditional_decode`` remains the load-bearing gate; this bench re-asserts
the decoded ax-trace of all three == ``isa.interpret`` on every program.

Run:
    python -m c4_min.bench_cond_vs_moe --device cuda:0 --subset mem+cmp
    python -m c4_min.bench_cond_vs_moe --device cuda:0 --subset full  # dense OOMs
"""
from __future__ import annotations

import argparse
import time
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import perlayer_conditional_sparse as PC


# ---------------------------------------------------------------------------
# Active-set builders for the three configs.
# ---------------------------------------------------------------------------
def blockmoe_active_set(lean, cond_active: Sequence[torch.Tensor]
                        ) -> Tuple[List[torch.Tensor], List[bool]]:
    """(b) BLOCK-MoE: the #628 granularity on the lean layers.

    Given the CONDITIONAL firing set per layer, a layer is MoE-ACTIVE iff it has
    ANY firing unit; a MoE-active layer keeps ALL its I units (run the whole block),
    a MoE-dead layer keeps 0 (skip the whole block == #628 identity expert).

    Returns (active_sets, is_active): ``active_sets[li]`` is arange(I) for a
    MoE-active layer and an empty tensor for a skipped one.
    """
    sets: List[torch.Tensor] = []
    is_active: List[bool] = []
    for l, fires in zip(lean.layers, cond_active):
        I = l.gate_w.shape[0]
        if fires.numel() > 0:
            sets.append(torch.arange(I))
            is_active.append(True)
        else:
            sets.append(torch.empty(0, dtype=torch.long))
            is_active.append(False)
    return sets, is_active


# ---------------------------------------------------------------------------
class SkipAwareCondLean(PC.ConditionalBlockLean):
    """``ConditionalBlockLean`` that, for a layer with 0 active units, skips the
    ENTIRE layer (attn + FFN) as identity -- the block-MoE route.  A layer with
    active units runs the standard attn + sliced-FFN.

    Byte-identity for the block-MoE config: the FULL model's divmod megablock
    layer (L9) has all-zero attention (identity) and its FFN writes only scratch
    bands that a non-divmod step's ax-mux never reads (#628 proof), so skipping the
    whole layer on a non-divmod op is L-inf-0.  For dense/conditional no layer is
    ever empty, so this reduces to the base forward.
    """

    def __init__(self, lean, active_units, skip_empty: bool = True):
        self._skip_empty = skip_empty
        self._empty = [a.numel() == 0 for a in active_units]
        super().__init__(lean, active_units)

    def forward(self, x, past=None, q_positions=None):
        B, S, H = x.shape
        if q_positions is None:
            q_pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)
            if q_pos.dim() == 1:
                q_pos = q_pos.unsqueeze(0).expand(B, S)
        h = x
        for li, layer in enumerate(self.layers):
            if self._skip_empty and self._empty[li]:
                continue  # block-MoE identity route: skip attn + FFN entirely
            xn = self._rmsnorm(h, layer.ln1)
            a, _ = self._attn(layer, xn, None, q_pos)
            h = h + a
            xn2 = self._rmsnorm(h, layer.ln2)
            if layer.n_active > 0:
                mlp = F.linear(
                    F.silu(F.linear(xn2, layer.gate_w)) * F.linear(xn2, layer.up_w),
                    layer.down_w)
                h = h + mlp
        return self._rmsnorm(h, self.final_norm), None


# ---------------------------------------------------------------------------
def _bench_forward(fwd: Callable, x, pos, n: int, warmup: int, cuda: bool) -> float:
    if cuda:
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            fwd(x, pos)
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            fwd(x, pos)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def _fwd_callable(model):
    if isinstance(model, LF.LeanQwenVM):
        def f(x, pos):
            return model.forward(x, past=None, q_positions=pos)[0]
        return f

    def f(x, pos):
        return model.forward(x, q_positions=pos)[0]
    return f


def _model_device(model):
    return model.device


def _vram_ok(device: str, need_bytes: float, headroom_gb: float = 2.0) -> bool:
    if not device.startswith("cuda"):
        return True
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return need_bytes + headroom_gb * (1024 ** 3) < free


# ---------------------------------------------------------------------------
# Representative programs.
# ---------------------------------------------------------------------------
def _prog_countdown(n=200):
    return isa.assemble([("IMM", n), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                         ("BNZ", 1), ("HALT", 0)])


def _prog_mul_accum(n=120):
    return isa.assemble([
        ("IMM", 1), ("PSH", 0), ("IMM", 3), ("MUL", 0), ("PSH", 0),
        ("IMM", 5), ("ADD", 0), ("PSH", 0), ("IMM", 1), ("SUB", 0),
        ("BNZ", 1), ("HALT", 0)])


def _prog_divmod(n=100):
    return isa.assemble([
        ("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0), ("PSH", 0),
        ("IMM", 1), ("ADD", 0), ("PSH", 0), ("IMM", 1), ("SUB", 0),
        ("BNZ", 1), ("HALT", 0)])


def _prog_mixed():
    """A MIXED-op loop hitting many distinct ops (add/sub/mul/cmp) -- the worst
    case for conditional (large union)."""
    return isa.assemble([
        ("IMM", 30), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("PSH", 0),
        ("IMM", 3), ("MUL", 0), ("PSH", 0), ("IMM", 2), ("SUB", 0),
        ("PSH", 0), ("IMM", 10), ("LT", 0), ("PSH", 0),
        ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])


# ---------------------------------------------------------------------------
def _run_spec_via(lean, model, code, max_steps, block_steps=64):
    """Run the lean spec driver but route its batched forward through ``model``.
    Returns the decoded ax-trace (list of ints), using the SAME window builder +
    decode as ``speculative_run_lean`` so traces are directly comparable."""
    from .qwen_lean_forward import (draft_program_lean, _build_spec_batch,
                                     CAM_REGS, _snap, run_program_lean)
    draft = draft_program_lean(lean, code, max_steps=max_steps)
    if not draft.steps:
        return run_program_lean(lean, code, max_steps=max_steps)["ax_trace"]
    L = lean.QL.L
    fwd = _fwd_callable(model)
    dev = _model_device(model)
    ax_trace: List[int] = []
    for s0 in range(0, len(draft.steps), block_steps):
        slab = draft.steps[s0:s0 + block_steps]
        x, positions = _build_spec_batch(lean, code, slab)
        x = x.to(dev); positions = positions.to(dev)
        with torch.no_grad():
            hidden = fwd(x, positions)
        for i, st in enumerate(slab):
            n_store = len(st["store_log"]) if lean.subset.memory else 0
            qrow = (1 + n_store) + len(CAM_REGS)
            ax = _snap(hidden[i, qrow][L.AX_VAL]) & 0xFF
            ax_trace.append(ax)
    return ax_trace


# ---------------------------------------------------------------------------
def run(device="cuda:0", subset_name="mem+cmp",
        batches=(512, 2048, 8192), n=20, warmup=5, thr=0.0, max_steps=400):
    subsets = {"base": Q.SUBSET_BASE, "mem+cmp": Q.SUBSET_MEM_CMP,
               "bitwise": Q.SUBSET_BITWISE, "full": Q.SUBSET_FULL}
    subset = subsets[subset_name]
    cuda = device.startswith("cuda")
    print(f"# COND-vs-MoE end-to-end bench subset={subset_name} device={device} "
          f"n={n} warmup={warmup}", flush=True)
    print("# building fused VM ...", flush=True)
    vm = Q.build(code_size=24, subset=subset)
    # FULL dense is 51.6 GB (infeasible on GPU); build the lean bundle on CPU and
    # only materialise the (small) conditional/MoE-active blocks on GPU.
    dev_build = device if subset_name != "full" else "cpu"
    lean = LF.LeanQwenVM.from_full_vm(vm, device=dev_build)
    H = lean.hidden_size
    I = lean.layers[0].gate_w.shape[0]
    nL = lean.n_layers
    dense_gb = sum(l.gate_w.numel() + l.up_w.numel() + l.down_w.numel()
                   for l in lean.layers) * 4 / (1024 ** 3)
    print(f"# n_layers={nL} H={H} I={I}  DENSE FFN wt = {dense_gb:.2f} GB "
          f"-> dense {'FITS' if dense_gb < 18 else 'DOES NOT FIT'} on GPU", flush=True)

    progs = [("countdown (SUB/BNZ loop)", _prog_countdown()),
             ("mul_accum (MUL/ADD loop)", _prog_mul_accum())]
    if subset.muldiv:
        progs.append(("divmod (MOD loop, fires megablock)", _prog_divmod()))
    progs.append(("mixed (add/mul/sub/cmp loop)", _prog_mixed()))

    for label, code in progs:
        print(f"\n{'='*78}\n## PROGRAM: {label}", flush=True)
        try:
            xw, posw, opc = PC.repetitive_program_windows(lean, code, max_steps=max_steps)
        except ValueError as e:
            print(f"  SKIP ({e})")
            continue
        xw_d = xw.to(dev_build); posw_d = posw.to(dev_build)
        info = PC.conditional_active_units(lean, xw_d, posw_d, thr=thr)
        cond_active = info["active_units"]
        cond_counts = [a.numel() for a in cond_active]
        moe_sets, moe_flags = blockmoe_active_set(lean, cond_active)
        moe_counts = [s.numel() for s in moe_sets]
        n_moe_active = sum(moe_flags)
        opmix = ", ".join(f"{isa.NAMES[o]}:{c}" for o, c in
                          sorted(opc.items(), key=lambda kv: -kv[1]))
        print(f"  steps={xw.shape[0]}  op mix: {opmix}")
        print(f"  conditional active/layer: {cond_counts}  total={sum(cond_counts)} "
              f"({sum(cond_counts)/(I*nL)*100:.4f}% of dense)")
        print(f"  block-MoE  active layers: {n_moe_active}/{nL} "
              f"(dead/skipped: {[i for i,a in enumerate(moe_flags) if not a]})  "
              f"MoE units total={sum(moe_counts)} ({sum(moe_counts)/(I*nL)*100:.4f}% of dense)")
        cond_of_moe = (sum(cond_counts) / sum(moe_counts) * 100) if sum(moe_counts) else 0.0
        print(f"  -> conditional keeps {cond_of_moe:.3f}% of the block-MoE units "
              f"(the fine-grained NET reduction WITHIN active layers)", flush=True)

        # ---- build the three models on GPU --------------------------------------
        cond_m = SkipAwareCondLean(lean, list(cond_active), skip_empty=True).to(device)
        moe_m = SkipAwareCondLean(lean, moe_sets, skip_empty=True).to(device)
        dense_fits = dense_gb < 18 and _vram_ok(device, dense_gb * (1024 ** 3))
        dense_m = None
        if dense_fits:
            # DENSE uses the IDENTICAL forward machinery as MoE/cond (a
            # SkipAwareCondLean holding ALL I units, no skip) so the ONLY difference
            # measured is the active-unit SET, not the forward implementation.  This
            # isolates the sparsity effect from any per-layer python-loop overhead.
            dense_full = [torch.arange(l.gate_w.shape[0]) for l in lean.layers]
            dense_m = SkipAwareCondLean(lean, dense_full, skip_empty=False).to(device)

        # ---- byte-identity gate: all configs decode == isa.interpret ------------
        ref = isa.interpret(code, max_steps=max_steps)
        traces = {"conditional": _run_spec_via(lean, cond_m, code, max_steps),
                  "block-moe": _run_spec_via(lean, moe_m, code, max_steps)}
        if dense_m is not None:
            traces["dense"] = _run_spec_via(lean, dense_m, code, max_steps)
        byte_ok = all(t == ref for t in traces.values())
        agree = len(set(tuple(t) for t in traces.values())) == 1
        print(f"  BYTE-IDENTITY: all configs ax-trace == isa.interpret: {byte_ok}  "
              f"(configs mutually agree: {agree})  [ref len={len(ref)}]", flush=True)

        # ---- END-TO-END ms/step at spec batch sizes -----------------------------
        base_x = xw_d[:1].to(device).contiguous()
        base_pos = posw_d[:1].to(device).contiguous()
        S = base_x.shape[1]
        hdr = f"  {'B':>6} {'M':>8} |"
        if dense_m is not None:
            hdr += f" {'dense/stp':>10}"
        hdr += f" {'moe/stp':>10} {'cond/stp':>10} | {'c/dense':>8} {'c/moe':>7} {'moe/dense':>9}"
        print(hdr)
        for B in batches:
            if not _vram_ok(device, 12 * B * S * H * 4):
                print(f"  {B:>6}  -- SKIP (VRAM guard) --")
                continue
            xb = base_x.expand(B, -1, -1).contiguous()
            pb = base_pos.expand(B, -1).contiguous()
            t_cond = _bench_forward(_fwd_callable(cond_m), xb, pb, n, warmup, cuda)
            t_moe = _bench_forward(_fwd_callable(moe_m), xb, pb, n, warmup, cuda)
            t_dense = (_bench_forward(_fwd_callable(dense_m), xb, pb, n, warmup, cuda)
                       if dense_m is not None else None)
            row = f"  {B:>6} {B*S:>8} |"
            if t_dense is not None:
                row += f" {t_dense/B*1000:9.4f}u"
            row += f" {t_moe/B*1000:9.4f}u {t_cond/B*1000:9.4f}u |"
            c_dense = (t_dense / t_cond) if t_dense else float('nan')
            c_moe = t_moe / t_cond if t_cond else float('nan')
            moe_dense = (t_dense / t_moe) if t_dense else float('nan')
            row += f" {c_dense:7.2f}x {c_moe:6.2f}x {moe_dense:8.2f}x"
            print(row, flush=True)
            del xb, pb
            if cuda:
                torch.cuda.empty_cache()
        del cond_m, moe_m
        if dense_m is not None and dense_m is not lean:
            del dense_m
        if cuda:
            torch.cuda.empty_cache()


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", default="mem+cmp",
                    choices=["base", "mem+cmp", "bitwise", "full"])
    ap.add_argument("--batch", default="512,2048,8192")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=400)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, subset_name=a.subset,
        batches=tuple(int(b) for b in a.batch.split(",")),
        n=a.n, warmup=a.warmup, thr=a.thr, max_steps=a.max_steps)
