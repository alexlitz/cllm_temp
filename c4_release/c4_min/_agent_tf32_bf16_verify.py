"""#751 — TF32 / bf16 tensor-core BYTE-EXACTNESS probe on the KBatch bounded path.

The KEY unlock hypothesis (#755): the VM's nibble argmax re-quantizer
argmax_v(2*v*nib - v^2) ABSORBS approximate precision — SiLU->GeLU was byte-exact
with a 6e-6 residual (~6 orders below the ~8.0 nibble-flip margin).  IMPLICATION:
TF32 / bf16 tensor cores (reduced mantissa = same class as an approximate transform)
MAY be byte-exact for the VM -> tensor cores unlocked = the enabler for high util.

This probe runs the ``pf_kbatch.KBatchBoundedRunner`` battery + a deep nested loop
under:
  (0) fp64 reference (the a39ae2c all-fp64 byte-exact ground truth)
  (A) fp32 default   (selective fp64; the current #748 path)
  (B) fp32 + TF32    (allow_tf32=True on the fp32 GEMMs)
  (C) bf16 matmul    (an env-gated bf16 FFN GEMM on the fp32 blocks)
and reports byte-exact PASS/FAIL + which programs flip.  If a decode flips we
identify which block needs to stay fp32/fp64 (selective, like the fp64 approach).

Additive experiment file (golden 069cc32f unchanged: imports only, builds nothing
on the golden path; C4_POS_SPARSE gates the composed path).

Run:  OMP_NUM_THREADS=4 python -m c4_min._agent_tf32_bf16_verify --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import torch.nn.functional as F

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from .pf_speculative import draft_pf_program
from . import bench_pf_kbatch as bk
from .bench_composed_fast_path import wait_for_gpu, _battery, _nested_prog


def _progs() -> List[Tuple[str, list, dict, int]]:
    out: List[Tuple[str, list, dict, int]] = []
    for name, prog, seed in _battery():
        code = prog if (prog and isinstance(prog[0], isa.Instr)) else isa.assemble(prog)
        out.append((name, code, seed, 200))
    out.append(("nested_deep", _nested_prog(3, 4), {}, 60))
    return out


def _all_traces(model, L, runner, K: int, progs) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for name, code, seed, ms in progs:
        tr, _ = bk.drive_kbatch(model, L, runner, code, K=K, seed_mem=seed, max_steps=ms)
        out[name] = tr
    return out


def _mismatches(ref: Dict[str, List[int]], cur: Dict[str, List[int]]) -> List[str]:
    return [name for name in ref if ref[name] != cur.get(name)]


# ---------------------------------------------------------------------------
# bf16 FFN patch: run the fp32-path SwiGLU GEMMs in bf16 on the NON-fp64 blocks.
# Monkeypatches BoundedBlock._ffn_qrow so bf16 only affects the blocks whose
# fp64_ffn is False (the fp32 blocks); the fp64-fragile blocks stay fp64.
# ---------------------------------------------------------------------------
def _install_bf16_ffn(runner):
    from .pos_sparse_bounded import BoundedBlock
    orig = BoundedBlock._ffn_qrow

    def bf16_qrow(self, aout):
        if self.routed or self.fp64_ffn or self._W_gu32 is None:
            return orig(self, aout)
        # bf16 the fp32 fused GEMM path.
        F_ = self.ffn
        a16 = aout.to(torch.bfloat16)
        gu = F.linear(a16, self._W_gu32.to(torch.bfloat16)) + self._b_gu32.to(torch.bfloat16)
        gate = gu[..., :self._Hdim]
        up = gu[..., self._Hdim:]
        hidden = F.silu(up) * gate
        down = F.linear(hidden, self._W_down32.to(torch.bfloat16))
        return aout + down.to(aout.dtype) + F_.b_down

    BoundedBlock._ffn_qrow = bf16_qrow
    return lambda: setattr(BoundedBlock, "_ffn_qrow", orig)


def _first_flip_block(model, L, runner, ref: Dict[str, List[int]], bad: List[str],
                      K: int) -> Optional[int]:
    """For a failing config, find which specific block(s), if forced back to fp64,
    restore byte-exactness — the block(s) that the reduced precision broke."""
    progs = [p for p in _progs() if p[0] in bad]
    # live-block pool over the failing programs.
    pool: Set[int] = set()
    for name, code, seed, ms in progs:
        draft = draft_pf_program(code, max_steps=ms, mask=0xFFFFFFFF)
        cur_pc, ops = 0, []
        for f in draft.frames:
            op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
            ops.append(op)
            cur_pc = f["pc"]
        pool |= set(runner.live_union(ops))
    # try forcing each live block to fp64 (leaving the reduced-precision elsewhere).
    base_fp64 = [bi for bi, kb in enumerate(runner.kblocks) if kb.b.fp64_ffn]
    for b in sorted(pool):
        if b in base_fp64:
            continue
        runner.set_fp64_blocks(set(base_fp64) | {b})
        cur = {p[0]: _all_traces(model, L, runner, K, [p])[p[0]] for p in progs}
        runner.set_fp64_blocks(set(base_fp64))
        if not _mismatches({p[0]: ref[p[0]] for p in progs}, cur):
            return b
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--K", type=str, default="1,32")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args(argv)
    bk.runner_window = args.window

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not args.no_wait:
        wait_for_gpu(idx, min_free_gb=args.min_free_gb, stable_s=args.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=args.window, selective_fp64=True)
    names = list(getattr(L, "_block_names", []))
    fp64 = [bi for bi, kb in enumerate(runner.kblocks) if kb.b.fp64_ffn]
    print(f"[built] blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"build={time.time()-t0:.1f}s", flush=True)
    print(f"[fp64] SELECTIVE: {len(fp64)}/{len(runner.kblocks)} fp64 "
          f"{[names[b] for b in fp64] if fp64 else '(none)'}", flush=True)

    Ks = [int(k) for k in args.K.split(",") if k.strip()]
    progs = _progs()

    print(f"\n{'='*84}\n[0] fp64 reference (a39ae2c all-fp64 byte-exact ground truth)\n"
          f"{'='*84}", flush=True)
    runner.set_fp64_blocks(None)              # ALL fp64
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    refs = {K: _all_traces(model, L, runner, K, progs) for K in Ks}
    for K in Ks:
        print(f"  K={K}: {len(refs[K])} programs decoded (reference)", flush=True)

    results = {}

    def run_stage(tag, K):
        cur = _all_traces(model, L, runner, K, progs)
        bad = _mismatches(refs[K], cur)
        results[(tag, K)] = bad
        verd = "BYTE-EXACT" if not bad else f"FLIP: {bad}"
        print(f"  [{tag}] K={K}: {verd}", flush=True)
        return bad

    # (A) fp32 default (selective fp64) — the current #748 path, TF32 OFF.
    print(f"\n{'='*84}\n[A] fp32 default (selective fp64, TF32 OFF)\n{'='*84}", flush=True)
    runner.set_fp64_blocks(set(fp64))
    torch.backends.cuda.matmul.allow_tf32 = False
    for K in Ks:
        run_stage("fp32", K)

    # (B) fp32 + TF32 tensor cores on the fp32 GEMMs.
    print(f"\n{'='*84}\n[B] fp32 + TF32 (allow_tf32=True; tensor cores on fp32 GEMMs)\n"
          f"{'='*84}", flush=True)
    runner.set_fp64_blocks(set(fp64))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for K in Ks:
        bad = run_stage("tf32", K)
        if bad:
            b = _first_flip_block(model, L, runner, refs[K], bad, K)
            nm = names[b] if (b is not None and b < len(names)) else str(b)
            print(f"      -> TF32 flip repaired by forcing block {b} ({nm}) to fp64",
                  flush=True)
            runner.set_fp64_blocks(set(fp64))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # (C) bf16 matmul on the fp32 blocks.
    print(f"\n{'='*84}\n[C] bf16 matmul (bf16 FFN GEMM on the fp32 blocks)\n{'='*84}",
          flush=True)
    runner.set_fp64_blocks(set(fp64))
    restore = _install_bf16_ffn(runner)
    try:
        for K in Ks:
            bad = run_stage("bf16", K)
            if bad:
                b = _first_flip_block(model, L, runner, refs[K], bad, K)
                nm = names[b] if (b is not None and b < len(names)) else str(b)
                print(f"      -> bf16 flip repaired by forcing block {b} ({nm}) to fp64",
                      flush=True)
                runner.set_fp64_blocks(set(fp64))
    finally:
        restore()

    # ---- verdict matrix ----
    print(f"\n{'='*84}\nVERDICT MATRIX (byte-exact vs fp64 reference)\n{'='*84}", flush=True)
    print(f"  {'config':10s} " + " ".join(f"K={K:<4d}" for K in Ks), flush=True)
    for tag in ("fp32", "tf32", "bf16"):
        row = []
        for K in Ks:
            bad = results.get((tag, K), [])
            row.append("exact" if not bad else f"FLIP({len(bad)})")
        print(f"  {tag:10s} " + " ".join(f"{c:<6s}" for c in row), flush=True)
    print(f"\ntotal wall: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
