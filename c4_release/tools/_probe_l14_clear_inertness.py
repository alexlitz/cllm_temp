#!/usr/bin/env python3
"""L14 clear/guard family inertness probe.

For each L14 clear op (temp_clear u0-3, addr_key_pollution u4-51,
output_corruption u52-69, mem_marker_output u70-133) measure the actual
per-unit W_down contribution to the block output on a representative set of
memory-generation programs. A unit whose max |W_down[:,u] * hidden[u]| == 0
across every row of every program is INERT (a free delete).

Usage: python tools/_probe_l14_clear_inertness.py
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch, torch.nn.functional as F  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

# clear-op unit ranges within the L14 FFN (per the make_* docstrings)
RANGES = {
    "temp_clear":            (0, 4),
    "addr_key_pollution":    (4, 52),
    "output_corruption":     (52, 70),
    "mem_marker_output":     (70, 134),
}


def dense(w):
    if w.is_sparse or (hasattr(w, "layout") and "sparse" in str(w.layout)):
        w = w.to_dense()
    return w.float().contiguous()


@torch.no_grad()
def main():
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    # find the clear-op FFN block: the ADDR_KEY-pollution clear writes a diagonal
    # W_down[ADDR_KEY+k, unit 4+k] for k in 0..47 (post-op expansion places the
    # L14 cleanup chain on a later physical block, NOT a logical-14 block).
    dp = probe.model.embed._dim_positions
    def dimpos(name):
        v = dp.get(name)
        return int(v) if v is not None else None
    addr_key = dimpos("ADDR_KEY")
    target_blk = None
    for phys, blk in enumerate(probe.model.blocks):
        ffn = getattr(blk, "ffn", None)
        if ffn is None or not hasattr(ffn, "W_down"):
            continue
        Wd = dense(ffn.W_down)
        if Wd.shape[1] < 134 or addr_key is None:
            continue
        diag = sum(1 for k in range(48) if abs(float(Wd[addr_key + k, 4 + k])) > 0)
        if diag >= 40:
            target_blk = phys
            break
    print(f"selected clear-op FFN block: physical {target_blk}")
    if target_blk is None:
        print("COULD NOT LOCATE clear-op block; aborting")
        return

    block = probe.model.blocks[target_blk]
    ffn = block.ffn
    Wu, bu = dense(ffn.W_up), dense(ffn.b_up)
    Wg, bg = dense(ffn.W_gate), dense(ffn.b_gate)
    Wd = dense(ffn.W_down)

    per_range_max = {k: 0.0 for k in RANGES}
    per_unit_max = {}

    def _accumulate(x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        up = x @ Wu.T + bu
        gate = x @ Wg.T + bg
        hidden = F.silu(up) * gate           # [B,S,H]
        for name, (lo, hi) in RANGES.items():
            hslice = hidden[..., lo:hi]
            wd_inf = Wd[:, lo:hi].abs().max(dim=0).values
            contrib = hslice.abs() * wd_inf
            m = float(contrib.max()) if contrib.numel() else 0.0
            per_range_max[name] = max(per_range_max[name], m)
            if contrib.numel():
                flat = contrib.reshape(-1, hi - lo)
                umax = flat.max(dim=0).values
                for u in range(hi - lo):
                    gu = lo + u
                    per_unit_max[gu] = max(per_unit_max.get(gu, 0.0), float(umax[u]))

    captured = {}
    def pre_hook(m, inp):
        _accumulate(inp[0].detach().float())
    h = ffn.register_forward_pre_hook(pre_hook)

    # representative memory-generation programs across the corpus.
    progs = generate_test_programs()
    n_sample = int(os.environ.get("N_SAMPLE", "12"))
    step = max(1, 1096 // n_sample)
    sample_ids = list(range(0, 1096, step))[:n_sample]
    n_ran = 0
    for pid in sample_ids:
        try:
            src, exp, desc = progs[pid]
            bc = compile_c(src)[0]
        except Exception:
            continue
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _ = probe._final_context(bc, max_steps=8)
            n_ran += 1
            print(f"  ran pid {pid} ({n_ran}/{len(sample_ids)})", flush=True)
        except Exception as e:
            print(f"  skip pid {pid}: {type(e).__name__}", flush=True)
            continue
    h.remove()
    print(f"\nprograms run: {n_ran}")
    print(f"{'clear op':<24}{'max |W_down contrib|':>22}   verdict")
    for name, (lo, hi) in RANGES.items():
        m = per_range_max[name]
        verdict = "INERT (0)" if m == 0.0 else ("near-0" if m < 1e-4 else "ACTIVE")
        print(f"{name:<24}{m:>22.6g}   {verdict}   units[{lo}:{hi}]")
    # report any fully-inert units
    inert_units = sorted(u for u, m in per_unit_max.items() if m == 0.0)
    active_units = sorted(u for u, m in per_unit_max.items() if m > 0.0)
    print(f"\nfully-inert units (0 contrib on sample): {len(inert_units)} / {len(per_unit_max)}")
    if active_units:
        print(f"ACTIVE units sample (u:maxcontrib): "
              + ", ".join(f"{u}:{per_unit_max[u]:.3g}" for u in active_units[:20]))


if __name__ == "__main__":
    main()
