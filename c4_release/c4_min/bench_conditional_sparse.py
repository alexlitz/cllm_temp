"""HONEST bench of PER-LAYER CONDITIONAL (per-op active-block) sparsity.

Answers the 6 tasks of the conditional-sparsity brief on the LEAN fused C4 VM
(``qwen_lean_forward``), cuda:0, VRAM-guarded:

  1. PER-LAYER static block density (each layer's OWN matrices) vs the aggregate.
  2. per-OP active-unit fraction per layer (silu(up)·gate fires), repetitive vs
     mixed.
  3. (block-MoE #628 already skips whole blocks per op; this extends it to the
     fine-grained within-layer active-unit block.)
  4. the custom GATHER + DENSE-GEMM per-layer kernel (NOT torch BSR).
  5. ms/step conditional-block vs dense on a REPETITIVE program at B=512..16384.
  6. HONEST both-regime verdict (mixed = large union = little win; repetitive =
     small union = win?).

Run:
    python -m c4_min.bench_conditional_sparse --device cuda:0 --subset full
    python -m c4_min.bench_conditional_sparse --device cuda:0 --subset mem+cmp
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import perlayer_conditional_sparse as PC


# ---------------------------------------------------------------------------
def _bench(fwd, x, pos, n: int, warmup: int, cuda: bool) -> float:
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


def _vram_ok(device: str, B: int, S: int, H: int, headroom_gb: float = 3.0) -> bool:
    if not device.startswith("cuda"):
        return True
    free, _tot = torch.cuda.mem_get_info(torch.device(device))
    est = 40 * B * S * H * 4
    return est + headroom_gb * (1024 ** 3) < free


# ---------------------------------------------------------------------------
# Representative programs.  REPETITIVE = tiny dynamic op-set; MIXED = many ops.
# ---------------------------------------------------------------------------
def _prog_countdown(n=200):
    """A long ADD/BNZ loop: dec a counter to 0.  Dynamic op-set = {IMM,PSH,SUB,BNZ}
    (no div/mod) — the L9 divmod megablock never fires."""
    # AX starts 0; IMM n; loop: PSH; IMM 1; SUB; BNZ loop; HALT
    return isa.assemble([
        ("IMM", n),            # 0
        ("PSH", 0),            # 1  loop:
        ("IMM", 1),            # 2
        ("SUB", 0),            # 3  ax = stk - 1
        ("BNZ", 1),            # 4  if ax != 0 goto loop
        ("HALT", 0),           # 5
    ])


def _prog_mul_accumulate(n=120):
    """A MUL-heavy loop (mandelbrot-inner-like: repeated MUL/ADD, no div)."""
    return isa.assemble([
        ("IMM", 1),            # 0
        ("PSH", 0),            # 1  loop:
        ("IMM", 3),            # 2
        ("MUL", 0),            # 3  ax = stk*3
        ("PSH", 0),            # 4
        ("IMM", 5),            # 5
        ("ADD", 0),            # 6  ax = stk+5
        ("PSH", 0),            # 7
        ("IMM", 1),            # 8
        ("SUB", 0),            # 9
        ("BNZ", 1),            # 10
        ("HALT", 0),           # 11
    ])


def _prog_divmod(n=100):
    """A DIV/MOD loop (Euclid-like): the ONLY cluster that fires the L9 megablock."""
    return isa.assemble([
        ("IMM", 100),          # 0
        ("PSH", 0),            # 1  loop:
        ("IMM", 7),            # 2
        ("MOD", 0),            # 3  ax = stk % 7
        ("PSH", 0),            # 4
        ("IMM", 1),            # 5
        ("ADD", 0),            # 6
        ("PSH", 0),            # 7
        ("IMM", 1),            # 8
        ("SUB", 0),            # 9
        ("BNZ", 1),            # 10
        ("HALT", 0),           # 11
    ])


# ---------------------------------------------------------------------------
def run(device="cuda:0", subset_name="full",
        batches=(512, 2048, 4096, 8192, 16384), n=20, warmup=5,
        thr=0.0):
    subsets = {"base": Q.SUBSET_BASE, "mem+cmp": Q.SUBSET_MEM_CMP,
               "bitwise": Q.SUBSET_BITWISE, "full": Q.SUBSET_FULL}
    subset = subsets[subset_name]
    cuda = device.startswith("cuda")
    print(f"# CONDITIONAL-sparse bench subset={subset_name} device={device} "
          f"n={n} warmup={warmup} thr={thr}")
    print("# building fused VM ...", flush=True)
    vm = Q.build(code_size=24, subset=subset)
    lean_cpu = LF.LeanQwenVM.from_full_vm(vm, device="cpu")
    H = lean_cpu.hidden_size
    I = lean_cpu.layers[0].gate_w.shape[0]
    print(f"# n_layers={lean_cpu.n_layers} H={H} I={I} "
          f"n_heads={lean_cpu.n_heads} kv={lean_cpu.n_kv_heads} hd={lean_cpu.head_dim}")

    # ---- TASK 1: PER-LAYER static block density vs aggregate -----------------
    print("\n## TASK 1 — PER-LAYER static block density (own matrices, pack perm)")
    rep = PC.static_perlayer_density(lean_cpu, tiles=(16,))
    t = 16
    print(f"  tile={t}: per-layer FFN (gate/up/down) active-tile skip% and fill%")
    print(f"  {'L':>3} {'ffn_skip%':>10} {'ffn_fill%':>10} {'ffn_flop':>9} "
          f"{'attn_skip%':>10} {'attn_fill%':>10}")
    for li, rec in enumerate(rep["per_layer"]):
        f = rec[t]["ffn"]; a = rec[t]["attn"]
        print(f"  {li:>3} {f.skip_frac*100:>10.3f} {f.tile_fill_frac*100:>10.3f} "
              f"{f.block_flops_ratio:>9.4f} {a.skip_frac*100:>10.3f} {a.tile_fill_frac*100:>10.3f}")
    agg = rep["aggregate"][t]
    print(f"  AGGREGATE tile={t}: ffn skip={agg['ffn'].skip_frac*100:.3f}% "
          f"fill={agg['ffn'].tile_fill_frac*100:.3f}% flop_ratio={agg['ffn'].block_flops_ratio:.5f}  "
          f"|  all skip={agg['all'].skip_frac*100:.3f}% fill={agg['all'].tile_fill_frac*100:.3f}%")

    # ---- TASK 2: per-OP active-unit fraction per layer -----------------------
    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    print("\n## TASK 2 — per-OP active-unit fraction per layer (silu(up)·gate fires)")
    probe_ops = [isa.IMM, isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
                 isa.PSH, isa.BNZ, isa.LI, isa.EQ, isa.LT, isa.AND]
    per_op_active: Dict[int, List[int]] = {}
    for op in probe_ops:
        x, pos = PC.op_window(lean, op)
        x = x.to(device); pos = pos.to(device)
        info = PC.conditional_active_units(lean, x, pos.unsqueeze(0), thr=thr)
        counts = [au.numel() for au in info["active_units"]]
        per_op_active[op] = counts
    # print a per-layer x per-op table of active-unit counts.
    hdr = "  " + f"{'op':>5} " + "".join(f"L{li:<5}" for li in range(lean.n_layers)) + " TOTAL"
    print(hdr)
    for op in probe_ops:
        counts = per_op_active[op]
        print("  " + f"{isa.NAMES[op]:>5} " +
              "".join(f"{c:<6}" for c in counts) + f" {sum(counts)}")
    # union across ALL ops (the mixed-batch active set) per layer.
    union_all = [set() for _ in range(lean.n_layers)]
    for op in probe_ops:
        x, pos = PC.op_window(lean, op)
        x = x.to(device); pos = pos.to(device)
        info = PC.conditional_active_units(lean, x, pos.unsqueeze(0), thr=thr)
        for li, au in enumerate(info["active_units"]):
            union_all[li] |= set(au.tolist())
    union_counts = [len(s) for s in union_all]
    print("  " + f"{'UNION':>5} " + "".join(f"{c:<6}" for c in union_counts) +
          f" {sum(union_counts)}   (mixed-batch active set, all {len(probe_ops)} ops)")
    print(f"  dense per-layer I = {I}  ->  UNION active frac = "
          f"{sum(union_counts)/(I*lean.n_layers)*100:.4f}% of dense")

    # ---- TASK 5/6: timing on a REPETITIVE vs MIXED batch ---------------------
    programs = {
        "countdown (SUB/BNZ loop, no divmod)": _prog_countdown(),
        "mul_accum (MUL/ADD loop, no divmod)": _prog_mul_accumulate(),
        "divmod   (MOD loop, fires L9 megablock)": _prog_divmod(),
    }
    for label, code in programs.items():
        print(f"\n## PROGRAM: {label}")
        # active units = union over the program's OWN step windows (the repetitive set).
        xw, posw, op_counts = PC.repetitive_program_windows(lean, code, max_steps=400)
        xw = xw.to(device); posw = posw.to(device)
        opmix = ", ".join(f"{isa.NAMES[o]}:{c}" for o, c in
                          sorted(op_counts.items(), key=lambda kv: -kv[1]))
        info = PC.conditional_active_units(lean, xw, posw, thr=thr)
        act = info["active_units"]
        act_counts = [a.numel() for a in act]
        print(f"  drafted steps={xw.shape[0]}  op mix: {opmix}")
        print(f"  per-layer active units: {act_counts}  total={sum(act_counts)} "
              f"({sum(act_counts)/(I*lean.n_layers)*100:.4f}% of dense)")

        cond = PC.ConditionalBlockLean(lean, act).to(device)
        # byte-identity gate on the program's own batch (should be L-inf 0 at thr=0).
        with torch.no_grad():
            rd = lean.forward(xw, past=None, q_positions=posw)[0]
            rc = cond.forward(xw, q_positions=posw)[0]
        linf = (rd - rc).abs().max().item()
        scale = rd.abs().max().item() + 1e-30
        # decode-space check: compare argmax/register decode at query rows.
        print(f"  BYTE-IDENTITY vs dense (own batch): L-inf={linf:.3e} rel={linf/scale:.3e}")

        # timing at speculation batch sizes (replicate one window to B rows).
        base_x = xw[:1].contiguous()
        base_pos = posw[:1].contiguous()
        print(f"  {'B':>6} {'M':>8} | {'dense/fwd':>10} {'dense/stp':>10} | "
              f"{'cond/fwd':>10} {'cond/stp':>10} | {'speedup':>8} {'Linf_rel':>10}")
        S = base_x.shape[1]
        for B in batches:
            if not _vram_ok(device, B, S, H):
                print(f"  {B:>6}  -- SKIP (VRAM guard) --")
                continue
            xb = base_x.expand(B, -1, -1).contiguous()
            pb = base_pos.expand(B, -1).contiguous()
            td = _bench(lambda xx, pp: lean.forward(xx, past=None, q_positions=pp),
                        xb, pb, n, warmup, cuda)
            tc = _bench(lambda xx, pp: cond.forward(xx, q_positions=pp),
                        xb, pb, n, warmup, cuda)
            with torch.no_grad():
                r1 = lean.forward(xb[:1], q_positions=pb[:1])[0]
                r2 = cond.forward(xb[:1], q_positions=pb[:1])[0]
            rel = (r1 - r2).abs().max().item() / (r1.abs().max().item() + 1e-30)
            sp = td / tc if tc else 0.0
            print(f"  {B:>6} {B*S:>8} | {td:9.3f}m {td/B*1000:9.4f}u | "
                  f"{tc:9.3f}m {tc/B*1000:9.4f}u | {sp:7.2f}x {rel:10.1e}")
            del xb, pb
            if cuda:
                torch.cuda.empty_cache()
        del cond, xw
        if cuda:
            torch.cuda.empty_cache()

    # ---- MIXED batch: union of MANY distinct ops (the honest no-win regime) ---
    print("\n## MIXED batch (union of all 12 probe ops = large active set)")
    mixed_x_list = []
    mixed_pos_list = []
    for op in probe_ops:
        x, pos = PC.op_window(lean, op)
        mixed_x_list.append(x.to(device))
        mixed_pos_list.append(pos.to(device).unsqueeze(0))
    Smax = max(x.shape[1] for x in mixed_x_list)
    # pad windows to Smax so they stack (pad rows sit at far positions).
    def _pad(x, pos):
        B, s, Hh = x.shape
        if s == Smax:
            return x, pos
        xp = torch.zeros(B, Smax, Hh, device=x.device, dtype=x.dtype)
        xp[:, :s] = x
        pp = torch.full((B, Smax), 10_000_000, device=pos.device, dtype=pos.dtype)
        pp[:, :s] = pos
        return xp, pp
    mx = []; mp = []
    for x, pos in zip(mixed_x_list, mixed_pos_list):
        xp, pp = _pad(x, pos)
        mx.append(xp); mp.append(pp)
    mixed_x = torch.cat(mx, dim=0)
    mixed_pos = torch.cat(mp, dim=0)
    info = PC.conditional_active_units(lean, mixed_x, mixed_pos, thr=thr)
    act = info["active_units"]
    act_counts = [a.numel() for a in act]
    print(f"  per-layer active units (12-op union): {act_counts} total={sum(act_counts)} "
          f"({sum(act_counts)/(I*lean.n_layers)*100:.4f}% of dense)")
    cond = PC.ConditionalBlockLean(lean, act).to(device)
    base_x = mixed_x[:1].contiguous(); base_pos = mixed_pos[:1].contiguous()
    S = base_x.shape[1]
    print(f"  {'B':>6} {'M':>8} | {'dense/stp':>10} {'cond/stp':>10} | {'speedup':>8}")
    for B in (512, 4096, 16384):
        if not _vram_ok(device, B, S, H):
            print(f"  {B:>6}  -- SKIP (VRAM guard) --")
            continue
        xb = base_x.expand(B, -1, -1).contiguous(); pb = base_pos.expand(B, -1).contiguous()
        td = _bench(lambda xx, pp: lean.forward(xx, past=None, q_positions=pp),
                    xb, pb, n, warmup, cuda)
        tc = _bench(lambda xx, pp: cond.forward(xx, q_positions=pp), xb, pb, n, warmup, cuda)
        sp = td / tc if tc else 0.0
        print(f"  {B:>6} {B*S:>8} | {td/B*1000:9.4f}u {tc/B*1000:9.4f}u | {sp:7.2f}x")
        del xb, pb
        if cuda:
            torch.cuda.empty_cache()


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", default="full",
                    choices=["base", "mem+cmp", "bitwise", "full"])
    ap.add_argument("--batch", default="512,2048,4096,8192,16384")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.0)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, subset_name=a.subset,
        batches=tuple(int(b) for b in a.batch.split(",")),
        n=a.n, warmup=a.warmup, thr=a.thr)
