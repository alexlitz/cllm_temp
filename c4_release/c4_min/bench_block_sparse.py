"""HONEST head-to-head benchmark: DENSE vs COO/CSR vs BLOCK-SPARSE (BSR) forward
of the LEAN compacted C4 VM, on the SAME clustering permutation and device.

Measures, per VM-step-window (and per spec-batch of B windows), the ms/forward and
the FLOP-utilization (achieved-work / scheduled-work) of each evaluator, plus the
BLOCK DENSITY the clustering permutation buys.  The point is the truthful tradeoff:
does the tensor-core-friendly block matmul actually beat COO/CSR + dense on a model
whose nonzeros are hand-placed (not naturally block-structured), given the
partial-block padding waste?

Run:
    python -m c4_min.bench_block_sparse --device cuda:0
    python -m c4_min.bench_block_sparse --device cuda:0 --subset full     # 86k nnz
    python -m c4_min.bench_block_sparse --device cuda:0 --batch 1,8,32     # spec-batch scaling
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import block_sparse_analysis as BSA
from . import block_sparse_forward as BSF


# ---------------------------------------------------------------------------
# COO/CSR evaluator (the unstructured-sparse baseline the block form must beat).
# ---------------------------------------------------------------------------
class _CSRWeight:
    """A CSR (unstructured) sparse linear — the tensor-core-UNfriendly baseline."""

    def __init__(self, w: torch.Tensor, min_numel=4096, density_thresh=0.25):
        self.out_dim, self.in_dim = w.shape
        numel = w.numel()
        nnz = int((w != 0).sum())
        self.is_sparse = numel >= min_numel and (nnz / numel) < density_thresh and nnz > 0
        if self.is_sparse:
            self.csr = w.to_sparse_csr()
            self.dense = None
        else:
            self.dense = w.contiguous()
            self.csr = None

    def to(self, dev):
        if self.dense is not None:
            self.dense = self.dense.to(dev)
        if self.csr is not None:
            self.csr = self.csr.to(dev)
        return self

    def linear(self, x):
        if not self.is_sparse:
            return F.linear(x, self.dense)
        orig = x.shape
        x2d = x.reshape(-1, self.in_dim)
        try:
            out = torch.sparse.mm(self.csr, x2d.transpose(0, 1).contiguous())
        except (RuntimeError, NotImplementedError):
            out = self.csr.to_dense() @ x2d.transpose(0, 1).contiguous()
        return out.transpose(0, 1).contiguous().reshape(*orig[:-1], self.out_dim)


class _CSRLean:
    """A CSR mirror of the lean forward (same math, unstructured-sparse GEMMs)."""

    def __init__(self, lean):
        self._proto = lean
        for a in ("hidden_size", "n_layers", "n_heads", "n_kv_heads", "head_dim",
                  "rope_theta", "rms_eps", "device", "dtype", "QL", "subset"):
            setattr(self, a, getattr(lean, a))
        self.inv_freq = lean.inv_freq.clone()
        self.final_norm = lean.final_norm.clone()
        self.embed = lean.embed.clone()
        self.layers = []
        for l in lean.layers:
            self.layers.append({
                "ln1": l.ln1.clone(), "ln2": l.ln2.clone(),
                "q": _CSRWeight(l.q_w), "k": _CSRWeight(l.k_w),
                "v": _CSRWeight(l.v_w), "o": _CSRWeight(l.o_w),
                "q_b": l.q_b, "k_b": l.k_b, "v_b": l.v_b,
                "gate": _CSRWeight(l.gate_w), "up": _CSRWeight(l.up_w),
                "down": _CSRWeight(l.down_w)})

    def to(self, dev):
        self.device = torch.device(dev)
        self.inv_freq = self.inv_freq.to(dev)
        self.final_norm = self.final_norm.to(dev)
        self.embed = self.embed.to(dev)
        for l in self.layers:
            l["ln1"] = l["ln1"].to(dev); l["ln2"] = l["ln2"].to(dev)
            for k in ("q", "k", "v", "o", "gate", "up", "down"):
                l[k].to(dev)
            for k in ("q_b", "k_b", "v_b"):
                if l[k] is not None:
                    l[k] = l[k].to(dev)
        return self

    # reuse the lean math helpers by binding them off the LeanQwenVM class.
    _rmsnorm = LF.LeanQwenVM._rmsnorm
    _rope_cos_sin = LF.LeanQwenVM._rope_cos_sin
    _rotate_half = staticmethod(LF.LeanQwenVM._rotate_half)
    _apply_rope = LF.LeanQwenVM._apply_rope

    def forward(self, x, past=None, q_positions=None):
        B, S, H = x.shape
        if q_positions is None:
            q_pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)
            if q_pos.dim() == 1:
                q_pos = q_pos.unsqueeze(0).expand(B, S)
        h = x
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        scale = hd ** -0.5
        for l in self.layers:
            xn = self._rmsnorm(h, l["ln1"])
            q = l["q"].linear(xn) + (l["q_b"] if l["q_b"] is not None else 0)
            k = l["k"].linear(xn) + (l["k_b"] if l["k_b"] is not None else 0)
            v = l["v"].linear(xn) + (l["v_b"] if l["v_b"] is not None else 0)
            q = q.view(B, S, nh, hd).transpose(1, 2)
            k = k.view(B, S, nkv, hd).transpose(1, 2)
            v = v.view(B, S, nkv, hd).transpose(1, 2)
            cos_q, sin_q = self._rope_cos_sin(q_pos)
            q = self._apply_rope(q, cos_q, sin_q)
            k = self._apply_rope(k, cos_q, sin_q)
            n_rep = nh // nkv
            if n_rep != 1:
                Sk = k.shape[2]
                Kr = k[:, :, None, :, :].expand(B, nkv, n_rep, Sk, hd).reshape(B, nh, Sk, hd)
                Vr = v[:, :, None, :, :].expand(B, nkv, n_rep, Sk, hd).reshape(B, nh, Sk, hd)
            else:
                Kr, Vr = k, v
            scores = torch.matmul(q, Kr.transpose(-2, -1)) * scale
            mask = (q_pos.unsqueeze(1) > q_pos.unsqueeze(2))
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
            out = torch.matmul(attn, Vr).transpose(1, 2).contiguous().view(B, S, nh * hd)
            h = h + l["o"].linear(out)
            xn2 = self._rmsnorm(h, l["ln2"])
            h = h + l["down"].linear(F.silu(l["gate"].linear(xn2)) * l["up"].linear(xn2))
        return self._rmsnorm(h, self.final_norm), None


# ---------------------------------------------------------------------------
def _make_window(lean, device, B: int):
    """One (or B replicated) VM-step window(s) — the residual the forward runs on."""
    code = isa.assemble([("IMM", 100), ("PSH", 0), ("IMM", 27), ("ADD", 0), ("HALT", 0)])
    x, pos = LF._build_stream_and_overlay(
        lean, code, {"PC": 0, "AX": 0, "SP": 252, "BP": 252, "STACK0": 0}, [], None)
    x = x.to(device); pos = pos.to(device)
    if B > 1:
        x = x.expand(B, -1, -1).contiguous()
        pos = pos.unsqueeze(0).expand(B, -1).contiguous()
    else:
        pos = pos.unsqueeze(0)
    return x, pos


def _bench(fwd, x, pos, n=100, warmup=10, cuda=True):
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


def run(device="cuda:0", subset_name="mem+cmp", batches=(1, 8, 32),
        blocksizes=(8, 16, 32), method="pack"):
    subsets = {"base": Q.SUBSET_BASE, "mem+cmp": Q.SUBSET_MEM_CMP,
               "bitwise": Q.SUBSET_BITWISE, "full": Q.SUBSET_FULL}
    subset = subsets[subset_name]
    cuda = device.startswith("cuda")
    print(f"# build subset={subset_name} device={device} method={method}")
    vm = Q.build(code_size=24, subset=subset)
    lean_cpu = LF.LeanQwenVM.from_full_vm(vm, device="cpu")
    perm = BSA.build_clustering_permutation(lean_cpu, method=method)

    # block-density report before/after.
    print(f"\n## BLOCK DENSITY ({subset_name}, method={method})")
    for mname in ("identity", method):
        p2 = BSA.build_clustering_permutation(lean_cpu, method=mname)
        m2 = BSA.apply_permutation(lean_cpu, p2)
        rep = BSA.model_tile_report(m2, tiles=blocksizes)
        for tile in blocksizes:
            ts = rep[tile]["all"]
            print(f"  {mname:>8} tile={tile:2d}: active={ts.active_tiles:>5}/{ts.total_tiles:<7} "
                  f"skip={ts.skip_frac*100:6.3f}% fill={ts.tile_fill_frac*100:5.2f}% "
                  f"flop_ratio={ts.block_flops_ratio:.4f}")

    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    csr = _CSRLean(lean_cpu).to(device)
    print(f"\n## ms/forward (n=100), decode-space L-inf vs dense")
    for B in batches:
        x, pos = _make_window(lean, device, B)
        t_dense = _bench(lambda xx, pp: lean.forward(xx, past=None, q_positions=pp),
                         x, pos, cuda=cuda)
        t_csr = _bench(lambda xx, pp: csr.forward(xx, q_positions=pp), x, pos, cuda=cuda)
        r_dense = lean.forward(x, q_positions=pos)[0]
        line = [f"B={B:<3} dense={t_dense:7.3f}ms  csr={t_csr:7.3f}ms"]
        for bsz in blocksizes:
            bs = BSF.PermutedBlockSparseVM(lean_cpu, perm, blocksize=bsz,
                                           min_active_tiles=2).to(device)
            t_bs = _bench(lambda xx, pp: bs.forward(xx, q_positions=pp), x, pos, cuda=cuda)
            r_bs = bs.forward(x, q_positions=pos)[0]
            linf = (r_dense - r_bs).abs().max().item()
            rel = linf / (r_dense.abs().max().item() + 1e-30)
            line.append(f"bsr{bsz}={t_bs:7.3f}ms(rel={rel:.1e})")
        print("  " + "  ".join(line))

    # FLOP-utilization: useful (nnz) work vs scheduled work per method.  The GEMM
    # MACs each method processes: dense = full numel; csr = nnz (one MAC per stored
    # value); block = active_tiles·tile² (dense tiles, padding included).  The
    # useful-work fraction (achieved/scheduled) is nnz / scheduled.
    print(f"\n## FLOP utilization (per-forward MACs over the 7 GEMM families)")
    total_nnz = 0
    dense_macs = 0
    for l in lean_cpu.layers:
        for nm in ("q_w", "k_w", "v_w", "o_w", "gate_w", "up_w", "down_w"):
            w = getattr(l, nm)
            total_nnz += int((w != 0).sum())
            dense_macs += w.numel()
    print(f"  useful MACs (nnz)         = {total_nnz:>12,}")
    print(f"  dense MACs (full)         = {dense_macs:>12,}"
          f"  util={total_nnz/dense_macs*100:.4f}%")
    print(f"  csr  MACs (=nnz)          = {total_nnz:>12,}"
          f"  util=100.0000% (no padding, but scalar-gather bound)")
    for bsz in blocksizes:
        p2 = BSA.build_clustering_permutation(lean_cpu, method=method)
        m2 = BSA.apply_permutation(lean_cpu, p2)
        rep = BSA.model_tile_report(m2, tiles=(bsz,))
        ts = rep[bsz]["all"]
        block_macs = ts.active_tiles * bsz * bsz
        util = total_nnz / block_macs if block_macs else 0.0
        print(f"  bsr{bsz:<2} MACs (active·tile²) = {block_macs:>12,}"
              f"  util={util*100:.4f}%  (tensor-core MMA; {(1-util)*100:.2f}% padding)")


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", default="mem+cmp",
                    choices=["base", "mem+cmp", "bitwise", "full"])
    ap.add_argument("--batch", default="1,8,32")
    ap.add_argument("--blocks", default="8,16,32")
    ap.add_argument("--method", default="pack", choices=["identity", "pack", "rcm"])
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, subset_name=a.subset,
        batches=tuple(int(b) for b in a.batch.split(",")),
        blocksizes=tuple(int(b) for b in a.blocks.split(",")),
        method=a.method)
