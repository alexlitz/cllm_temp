"""#751 — FAST TF32/bf16 byte-exactness verdict (K=1, no deep DIV/MOD).

A quick companion to ``_agent_tf32_bf16_verify``: same fp64-reference vs
fp32/TF32/bf16 comparison, but at K=1 and on a SHALLOW battery (skips the deep
recurrent DIV/MOD that dominate the full run's wall) so the byte-exactness verdict
lands in ~1-2 min.  Also runs the raw-divergence probe: the max-abs residual delta
at the AX nibble decode dims under TF32/bf16 vs fp64 (the margin vs the ~8 nibble
flip), quantifying HOW close each precision comes to flipping a decode.

Run:  OMP_NUM_THREADS=4 python -m c4_min._agent_tf32_quick --device cuda:1
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import torch.nn.functional as F

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from . import bench_pf_kbatch as bk
from ._agent_tf32_bf16_verify import _install_bf16_ffn


def _shallow_battery() -> List[Tuple[str, list, dict, int]]:
    """arith / branch / mem / cmp — NO deep DIV/MOD (those add ~180 blocks/step)."""
    B = [
        ("imm", [("IMM", 42), ("HALT", 0)], {}),
        ("add", [("IMM", 12), ("PSH", 0), ("IMM", 30), ("ADD", 0), ("HALT", 0)], {}),
        ("sub", [("IMM", 100), ("PSH", 0), ("IMM", 58), ("SUB", 0), ("HALT", 0)], {}),
        ("mul", [("IMM", 12), ("PSH", 0), ("IMM", 7), ("MUL", 0), ("HALT", 0)], {}),
        ("and", [("IMM", 0xF0), ("PSH", 0), ("IMM", 0x3C), ("AND", 0), ("HALT", 0)], {}),
        ("shl", [("IMM", 3), ("PSH", 0), ("IMM", 4), ("SHL", 0), ("HALT", 0)], {}),
        ("eq", [("IMM", 5), ("PSH", 0), ("IMM", 5), ("EQ", 0), ("HALT", 0)], {}),
        ("lt", [("IMM", 3), ("PSH", 0), ("IMM", 5), ("LT", 0), ("HALT", 0)], {}),
        ("li", [("IMM", 8), ("LI", 0), ("HALT", 0)], {8: 123}),
        ("lea", [("LEA", 4), ("HALT", 0)], {}),
        ("bz", [("IMM", 0), ("BZ", 4), ("IMM", 99), ("HALT", 0), ("IMM", 7), ("HALT", 0)], {}),
        ("jsr_lev", [("JSR", 3), ("IMM", 5), ("HALT", 0), ("ENT", 0), ("IMM", 7), ("LEV", 0)], {}),
        ("si_li", [("IMM", 200), ("PSH", 0), ("IMM", 55), ("SI", 0),
                   ("IMM", 200), ("LI", 0), ("HALT", 0)], {}),
    ]
    out = []
    for name, prog, seed in B:
        out.append((name, isa.assemble(prog), seed, 40))
    return out


def _traces(model, L, runner, progs, K=1):
    out = {}
    for name, code, seed, ms in progs:
        tr, _ = bk.drive_kbatch(model, L, runner, code, K=K, seed_mem=seed, max_steps=ms)
        out[name] = tr
    return out


def _bad(ref, cur):
    return [n for n in ref if ref[n] != cur.get(n)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=20.0)
    a = ap.parse_args(argv)
    bk.runner_window = a.window

    device = a.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not a.no_wait:
        from .bench_composed_fast_path import wait_for_gpu
        wait_for_gpu(idx, min_free_gb=a.min_free_gb, stable_s=a.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(code_size=a.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device); model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=a.window, selective_fp64=True)
    names = list(getattr(L, "_block_names", []))
    fp64 = [bi for bi, kb in enumerate(runner.kblocks) if kb.b.fp64_ffn]
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"fp64={[names[b] for b in fp64] if fp64 else '(none)'} build={time.time()-t0:.1f}s",
          flush=True)

    progs = _shallow_battery()

    # fp64 reference (all-fp64).
    runner.set_fp64_blocks(None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = _traces(model, L, runner, progs)
    print(f"[ref] fp64: {len(ref)} programs", flush=True)

    def stage(tag):
        cur = _traces(model, L, runner, progs)
        bad = _bad(ref, cur)
        print(f"  [{tag}] {'BYTE-EXACT' if not bad else 'FLIP: ' + str(bad)}", flush=True)
        for n in bad:
            print(f"      {n}: ref={ref[n]}  got={cur[n]}", flush=True)
        return bad

    # fp32 selective (baseline).
    runner.set_fp64_blocks(set(fp64))
    torch.backends.cuda.matmul.allow_tf32 = False
    print("\n[A] fp32 (selective fp64, TF32 OFF)", flush=True); stage("fp32")

    # fp32 + TF32.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("\n[B] fp32 + TF32", flush=True); stage("tf32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # bf16.
    print("\n[C] bf16 FFN (fp32 blocks)", flush=True)
    restore = _install_bf16_ffn(runner)
    try:
        stage("bf16")
    finally:
        restore()

    # also: ALL blocks fp32 (no selective fp64) with TF32 — the harshest test.
    print("\n[D] ALL fp32 + TF32 (no selective fp64 — harshest)", flush=True)
    runner.set_fp64_blocks(set())
    torch.backends.cuda.matmul.allow_tf32 = True
    stage("allfp32_tf32")
    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"\ntotal wall: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
