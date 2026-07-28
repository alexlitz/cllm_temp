"""#751 — GROUPED / band-structured GEMM bakeoff for the K-row query FFN.

The band-grouped GEMM plan: decompose each live block's FFN into (input-band ->
hidden-slice -> output-band) dense sub-blocks and GROUP them across
(bands x live-blocks x K rows) into a few big tensor-core GEMMs captured in one CUDA
graph, at big K on TF32 tensor cores.

The band-structure probe (_agent_band_structure) already MEASURED that the premise
is false: each FFN hidden unit reads a MEDIAN of 1 residual dim (max 3), W_up density
~0.08%, and a 16x16 tile is only ~4-6% useful (94-96% padding) — the nonzeros do NOT
cluster into dense band sub-blocks.  This bench turns that into a TIMING verdict: it
runs the K-row query FFN over ALL live blocks of a representative op in FOUR forms and
measures ms + achieved/useful FLOP util, on TF32 tensor cores (the #751 unlock):

  (1) PER-BLOCK DENSE   — the current path: one small F.linear per block (fp32/TF32).
  (2) GROUPED-DENSE BMM — pad every block's [K,D]@[D,Dff] to a common (Dmax,Dffmax)
      and run ONE batched bmm over the live blocks (the "band-grouped dense GEMM").
      Measures the padding trap: it does n_blocks x K x Dmax x Dffmax MACs, most zero.
  (3) COO SCATTER       — block_sparse_ffn's gather-scale-scatter (the ~1-nnz-per-row
      matched form): few-FLOP static index program, byte-exact for 1-nnz rows.
  (4) FUSED CONCAT GEMM — concat all live blocks' W along the Dff axis into ONE
      [D, sum(Dff)] GEMM per projection (no per-block launch, no inter-block padding):
      K rows x D x sum(Dff) — the honest "group the useful work" GEMM.

Reports ms/forward, executed vs useful GFLOP, and util vs the A5000 TF32 peak.
Read-only w.r.t. golden weights.

Run:  OMP_NUM_THREADS=4 python -m c4_min._agent_grouped_gemm_bench --device cuda:1 --K 128
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import torch.nn.functional as F

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from .pf_speculative import draft_pf_program

A5000_FP32 = 27.8e12
A5000_TF32 = 55.6e12


def _dense(w):
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    return w


def _live_ops(runner, code, max_steps=200):
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF)
    cur_pc, ops = 0, []
    for f in draft.frames:
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        ops.append(op)
        cur_pc = f["pc"]
    return runner.live_union(ops)


class FFNPack:
    """Precompute the four FFN execution forms for a set of live blocks."""

    def __init__(self, runner, live: List[int], device, dtype):
        self.blocks = []          # (Wg, Wu, Wd, bg, bu, bd) fp32 dense per block
        self.coo = []             # (rows,cols,vals) per proj per block
        self.dev = device
        self.dtype = dtype
        Wg_list, Wu_list, Wd_list = [], [], []
        bg_list, bu_list, bd_list = [], [], []
        for bi in live:
            b = runner.kblocks[bi].b
            if b.routed or getattr(b.ffn, "W_up", None) is None:
                continue
            Wg = _dense(b.ffn.W_gate).to(device, dtype).contiguous()
            Wu = _dense(b.ffn.W_up).to(device, dtype).contiguous()
            Wd = _dense(b.ffn.W_down).to(device, dtype).contiguous()
            bg = b.ffn.b_gate.to(device, dtype)
            bu = b.ffn.b_up.to(device, dtype)
            bd = b.ffn.b_down.to(device, dtype)
            self.blocks.append((Wg, Wu, Wd, bg, bu, bd))
            Wg_list.append(Wg); Wu_list.append(Wu); Wd_list.append(Wd)
            bg_list.append(bg); bu_list.append(bu); bd_list.append(bd)
        self.D = self.blocks[0][0].shape[1]
        self.Dffs = [Wg.shape[0] for (Wg, *_1) in self.blocks]
        self.n = len(self.blocks)
        # ---- (2) grouped-dense bmm: pad to common (D, Dffmax) ----
        # This is the "band-grouped DENSE GEMM": each block padded to the max Dff and
        # stacked into one bmm.  For a heterogeneous-Dff live set (DIV: 18..4320) the
        # padded weight tensor is n*Dffmax*D floats — the padding trap made concrete.
        # Guard the allocation: if it would exceed ~4GB, mark grouped-bmm INFEASIBLE
        # (report it as the trap rather than OOM).
        self.Dffmax = max(self.Dffs)
        pad_bytes = self.n * self.Dffmax * self.D * 4 * 3   # Wg+Wu+Wd padded
        self.bmm_feasible = pad_bytes < 4.0e9
        self.pad_gb = pad_bytes / 1e9
        if self.bmm_feasible:
            Wg_pad = torch.zeros(self.n, self.Dffmax, self.D, device=device, dtype=dtype)
            Wu_pad = torch.zeros(self.n, self.Dffmax, self.D, device=device, dtype=dtype)
            Wd_pad = torch.zeros(self.n, self.D, self.Dffmax, device=device, dtype=dtype)
            for i, (Wg, Wu, Wd, *_2) in enumerate(self.blocks):
                f = Wg.shape[0]
                Wg_pad[i, :f] = Wg; Wu_pad[i, :f] = Wu; Wd_pad[i, :, :f] = Wd
            self.Wg_pad, self.Wu_pad, self.Wd_pad = Wg_pad, Wu_pad, Wd_pad
        # ---- (4) fused concat along Dff ----
        self.Wg_cat = torch.cat([b[0] for b in self.blocks], dim=0).contiguous()  # [sumDff, D]
        self.Wu_cat = torch.cat([b[1] for b in self.blocks], dim=0).contiguous()
        # W_down: block-diagonal would be needed to write per-block outputs; instead
        # concat down-inputs and keep per-block down GEMMs summed via segment.  For the
        # residual add we sum each block's contribution; use a [D, sumDff] with a
        # per-block column offset — a single [K,sumDff]@[sumDff,D] gives the SUM of all
        # blocks' down outputs (each block writes the full D residual, they add), which
        # matches applying blocks in sequence only if blocks are independent.  We report
        # the concat GEMM as the FLOP/latency of the up/gate projections (the dominant
        # cost); down is measured separately per-block.
        self.bg_cat = torch.cat([b[3] for b in self.blocks], dim=0)
        self.bu_cat = torch.cat([b[4] for b in self.blocks], dim=0)
        self.sumDff = self.Wg_cat.shape[0]
        # ---- (3) COO ----
        from .block_sparse_ffn import CooLinear
        for (Wg, Wu, Wd, bg, bu, bd) in self.blocks:
            self.coo.append((
                CooLinear.from_dense(Wg).to(device),
                CooLinear.from_dense(Wu).to(device),
                CooLinear.from_dense(Wd).to(device),
                bg, bu, bd))

    # --- useful FLOP (the real per-block MACs) ---
    def useful_flops(self, K):
        f = 0.0
        for Wg, Wu, Wd, *_3 in self.blocks:
            Dff, D = Wg.shape
            f += 2.0 * K * D * Dff * 3   # gate, up, down
        return f

    def dense_padded_flops(self, K):
        # grouped bmm: n * K * D * Dffmax * 3.
        return 2.0 * self.n * K * self.D * self.Dffmax * 3

    # --- (1) per-block dense ---
    def run_per_block(self, xq):
        outs = []
        for (Wg, Wu, Wd, bg, bu, bd) in self.blocks:
            gate = F.linear(xq, Wg) + bg
            up = F.linear(xq, Wu) + bu
            h = F.silu(up) * gate
            outs.append(xq + F.linear(h, Wd) + bd)
        return outs

    # --- (2) grouped-dense bmm (padded) ---
    def run_grouped_bmm(self, xq):
        # xq [K,D] -> [n,K,D] broadcast, bmm with [n,Dffmax,D]^T.
        K = xq.shape[0]
        xb = xq.unsqueeze(0).expand(self.n, K, self.D)          # [n,K,D]
        gate = torch.bmm(xb, self.Wg_pad.transpose(1, 2))       # [n,K,Dffmax]
        up = torch.bmm(xb, self.Wu_pad.transpose(1, 2))
        h = F.silu(up) * gate                                   # [n,K,Dffmax]
        down = torch.bmm(h, self.Wd_pad.transpose(1, 2))        # [n,K,D]
        return down

    # --- (3) COO scatter ---
    def run_coo(self, xq):
        outs = []
        for (cg, cu, cd, bg, bu, bd) in self.coo:
            gate = cg.linear(xq) + bg
            up = cu.linear(xq) + bu
            h = F.silu(up) * gate
            outs.append(xq + cd.linear(h) + bd)
        return outs

    # --- (4) fused concat up/gate GEMM ---
    def run_concat(self, xq):
        gate = F.linear(xq, self.Wg_cat) + self.bg_cat          # [K, sumDff]
        up = F.linear(xq, self.Wu_cat) + self.bu_cat
        h = F.silu(up) * gate                                   # [K, sumDff]
        # down: one [K,sumDff]@[sumDff,D] via block-diagonal concat of W_down^T is not
        # right (would mix). Report up/gate concat only + per-block down.
        outs = []
        off = 0
        for (Wg, Wu, Wd, bg, bu, bd) in self.blocks:
            f = Wg.shape[0]
            hd = h[:, off:off + f]
            outs.append(xq + F.linear(hd, Wd) + bd)
            off += f
        return outs


def _time(fn, n=50, warmup=10, cuda=True):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=20.0)
    a = ap.parse_args(argv)

    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not a.no_wait:
        from .bench_composed_fast_path import wait_for_gpu
        wait_for_gpu(idx, min_free_gb=a.min_free_gb, stable_s=a.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True     # the #751 unlock (byte-exact verified)
    torch.backends.cudnn.allow_tf32 = True

    from .compact_alloc import build_compact_sparse_streaming
    model, L, _ = build_compact_sparse_streaming(code_size=a.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device); model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=a.window, selective_fp64=True)
    dev = torch.device(device)
    dtype = torch.float32
    cuda = device != "cpu"
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} TF32=ON", flush=True)

    add = isa.assemble([("IMM", 12), ("PSH", 0), ("IMM", 30), ("ADD", 0), ("HALT", 0)])
    div = isa.assemble([("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0), ("HALT", 0)])

    K = a.K
    for tag, code, ms in (("ADD", add, 40), ("DIV", div, 200)):
        live = sorted(_live_ops(runner, code, ms))
        pack = FFNPack(runner, live, dev, dtype)
        xq = torch.randn(K, pack.D, device=dev, dtype=dtype) * 0.1
        useful = pack.useful_flops(K)
        padded = pack.dense_padded_flops(K)

        # correctness sanity: concat & coo down-summed == per-block (up to fp order).
        # NOTE: per-block returns each block's OWN residual output (they are applied in
        # SEQUENCE in the real forward, sharing the residual); here we compare each
        # form's per-block FFN CONTRIBUTION (down proj) rather than a fused residual,
        # to validate the arithmetic of each form independently.
        with torch.no_grad():
            pb = pack.run_per_block(xq)
            co = pack.run_coo(xq)
            cc = pack.run_concat(xq)
            err_co = max((a_ - b_).abs().max().item() for a_, b_ in zip(pb, co))
            err_cc = max((a_ - b_).abs().max().item() for a_, b_ in zip(pb, cc))

        def t_pb():
            with torch.no_grad(): pack.run_per_block(xq)
        def t_coo():
            with torch.no_grad(): pack.run_coo(xq)
        def t_cc():
            with torch.no_grad(): pack.run_concat(xq)

        ms_pb = _time(t_pb, cuda=cuda)
        ms_coo = _time(t_coo, cuda=cuda)
        ms_cc = _time(t_cc, cuda=cuda)
        if pack.bmm_feasible:
            def t_bmm():
                with torch.no_grad(): pack.run_grouped_bmm(xq)
            ms_bmm = _time(t_bmm, cuda=cuda)
        else:
            ms_bmm = None

        def util(f, msv, peak):
            return 100.0 * (f / (msv / 1e3)) / peak

        print(f"\n{'='*92}\n[{tag}] {pack.n} live FFN blocks, K={K}, D={pack.D}, "
              f"Dff in [{min(pack.Dffs)},{max(pack.Dffs)}] sum={pack.sumDff}\n"
              f"  useful GFLOP/fwd={useful/1e9:.3f}  grouped-padded GFLOP/fwd={padded/1e9:.3f} "
              f"({padded/max(useful,1):.0f}x padding, pad-weight={pack.pad_gb:.2f}GB "
              f"{'feasible' if pack.bmm_feasible else 'INFEASIBLE->skip'})\n"
              f"  sanity: max|per_block - coo|={err_co:.2e}  max|per_block - concat|={err_cc:.2e}\n"
              f"{'='*92}", flush=True)
        print(f"  {'form':24s} {'ms/fwd':>9} {'useful-util%':>13} {'exec-util%':>12}", flush=True)
        print(f"  {'(1) per-block dense':24s} {ms_pb:9.4f} {util(useful, ms_pb, A5000_TF32):13.4f} "
              f"{util(useful, ms_pb, A5000_TF32):12.4f}", flush=True)
        if ms_bmm is not None:
            print(f"  {'(2) grouped-dense bmm':24s} {ms_bmm:9.4f} {util(useful, ms_bmm, A5000_TF32):13.4f} "
                  f"{util(padded, ms_bmm, A5000_TF32):12.4f}", flush=True)
        else:
            print(f"  {'(2) grouped-dense bmm':24s} {'SKIP':>9}  (pad-weight {pack.pad_gb:.1f}GB "
                  f">4GB — the padding trap; Dffmax={pack.Dffmax} vs median~{sorted(pack.Dffs)[len(pack.Dffs)//2]})",
                  flush=True)
        print(f"  {'(3) COO scatter':24s} {ms_coo:9.4f} {util(useful, ms_coo, A5000_TF32):13.4f} "
              f"{util(useful, ms_coo, A5000_TF32):12.4f}", flush=True)
        print(f"  {'(4) fused concat GEMM':24s} {ms_cc:9.4f} {util(useful, ms_cc, A5000_TF32):13.4f} "
              f"{util(useful, ms_cc, A5000_TF32):12.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
