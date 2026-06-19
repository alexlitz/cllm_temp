#!/usr/bin/env python3
"""Dump blk16 (L11) attention head-0 V/O wiring: which residual dims it reads
into the head (V = W_v columns) and which it writes out (O = W_o rows), so we
know what head0 copies from its attended source row to the byte-1 Q row.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

BLK = int(os.environ.get("PROBE_BLK", "16"))
HEAD = int(os.environ.get("PROBE_HEAD", "0"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def name_for(dp_inv, idx):
    return dp_inv.get(idx, f"?{idx}")


def main():
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    # build inverse map dim->name (band base names)
    inv = {}
    for nm, base in dp.items():
        inv[base] = nm
    attn = p.model.blocks[BLK].attn
    H = attn.num_heads
    HD = attn.head_dim
    Wv = td(attn.W_v)  # [H*HD, D] (or compact)
    Wo = td(attn.W_o)  # [D, H*HD]
    compact = getattr(attn, "_is_compact", False)
    print(f"=== blk{BLK} head{HEAD} H={H} HD={HD} compact={compact} ===")
    if compact:
        in_idx = list(attn._compact_in_idx)
        out_idx = list(attn._compact_out_idx)
        # Wv: [n_out, n_in]; head h owns rows [h*HD_c:(h+1)*HD_c] where HD_c=n_out//H
        HDc = len(out_idx) // H
        rows = range(HEAD * HDc, (HEAD + 1) * HDc)
        print(f"  V (W_v) reads per slot (compact, HDc={HDc}):")
        for s in rows:
            r = Wv[s]
            nz = [(in_idx[j], float(r[j])) for j in range(len(r)) if abs(r[j]) > 0.01]
            if nz:
                desc = " ".join(f"{_nm(inv, d)}={v:.1f}" for d, v in nz[:6])
                print(f"    slot{s - HEAD*HDc}: {desc}")
        print(f"  O (W_o) writes per slot (compact):")
        # Wo: [n_out_full?, H*HDc] -> map out_idx
        for s in rows:
            col = Wo[:, s]
            nz = [(out_idx[i], float(col[i])) for i in range(len(col)) if abs(col[i]) > 0.01]
            if nz:
                desc = " ".join(f"{_nm(inv, d)}={v:.1f}" for d, v in nz[:6])
                print(f"    slot{s - HEAD*HDc}: {desc}")
    else:
        rows = range(HEAD * HD, (HEAD + 1) * HD)
        print("  V (W_v) reads per slot:")
        for s in rows:
            r = Wv[s]
            nz = [(j, float(r[j])) for j in range(len(r)) if abs(r[j]) > 0.01]
            if nz:
                desc = " ".join(f"{_nm(inv, d)}={v:.1f}" for d, v in nz[:6])
                print(f"    slot{s - HEAD*HD}: {desc}")
        print("  O (W_o) writes per slot:")
        for s in rows:
            col = Wo[:, s]
            nz = [(i, float(col[i])) for i in range(len(col)) if abs(col[i]) > 0.01]
            if nz:
                desc = " ".join(f"{_nm(inv, d)}={v:.1f}" for d, v in nz[:6])
                print(f"    slot{s - HEAD*HD}: {desc}")


def _nm(inv, idx):
    # find nearest band base <= idx
    best = None
    for base in inv:
        if base <= idx and (best is None or base > best):
            best = base
    if best is None:
        return f"d{idx}"
    return f"{inv[best]}+{idx - best}"


if __name__ == "__main__":
    main()
