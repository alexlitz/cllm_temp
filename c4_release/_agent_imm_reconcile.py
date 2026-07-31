"""RECONCILE: qwen_full_vm's IMM 1000->232 (8-bit fold) vs pure-forward's full-literal IMM.

PART-1 step 1.  Builds the PURE-FORWARD streaming model (build_lib_model_streaming,
addr32, the 20-bit-IMM path) and runs IMM literals (300/1000/1024/100000) through
BOTH the byte-exact cached driver AND the speculative verify_blocks (big-K), decoding
AX from all 8 nibbles.  Compares to the model's own draft transition (imm & _IMM_MASK).

CONCLUSION we expect: the pure-forward model already materializes the full 20-bit
literal (IMM_NIBS=5, compile_imm_ax_nibbles writes all fetched nibbles), so IMM 1000
-> 1000 and 1024 -> 1024 through the SAME fast speculation used by bench_fast_path.
The qwen_full_vm 232 is a DIFFERENT model family (scalar IMM + a global mod-256 fold).
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from c4_min import isa
from c4_min.pf_speculative import draft_pf_program, verify_blocks, _IMM_MASK
from c4_min.nibble_pure_forward_complete import _decode_reg_from_nibbles


def imm_prog(v: int):
    return isa.assemble([("IMM", v), ("HALT", 0)])


def run():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[reconcile] device={device}  IMM_NIBS band mask=_IMM_MASK=0x{_IMM_MASK:X} "
          f"({_IMM_MASK.bit_length()} bits)", flush=True)

    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=64, recurrent_divmod=True,
                                             addr32=True, compute_mode="dense_kernel")
    print(f"[reconcile] built pure-forward streaming model: blocks={len(sparse.blocks)} "
          f"dim={sparse.embed.shape[1]} in {time.time()-t0:.1f}s", flush=True)
    if device != "cpu":
        sparse = sparse.to(device)
        sparse.materialize_dense(device=device)

    print(f"\n  {'IMM literal':>12} {'draft_ax':>10} {'cached_ax':>10} "
          f"{'specK8_ax':>10} {'qwen_fold(&0xFF)':>16} {'match':>6}", flush=True)
    print("  " + "-" * 74, flush=True)
    all_ok = True
    for v in (42, 255, 256, 300, 1000, 1024, 100000, 133653):
        code = imm_prog(v)
        # (1) the MODEL's draft transition (what verify checks against)
        draft = draft_pf_program(code, max_steps=8, mask=0xFFFFFFFF)
        draft_ax = draft.frames[0]["ax"]
        # (2) the byte-exact cached driver: decode AX from all 8 nibbles
        trace = run_pure_forward_cached(sparse, L, code, max_steps=8, mask=0xFFFFFFFF)
        # the cached trace is AX&mask per step; step 0 is the IMM
        cached_ax = trace[0] if trace else None
        # (3) the speculative verify (big-K = the FAST path)
        stats = {}
        vr = verify_blocks(sparse, L, code, draft, block_steps=8, device=device,
                           evict=False, mask=0xFFFFFFFF, stats=stats, fast=True)
        spec_ax = vr.decoded_final_ax
        qwen_fold = v & 0xFF
        want = v & _IMM_MASK
        ok = (draft_ax == want and cached_ax == want and spec_ax == want)
        all_ok = all_ok and ok
        print(f"  {v:>12} {draft_ax:>10} {str(cached_ax):>10} {str(spec_ax):>10} "
              f"{qwen_fold:>16} {'OK' if ok else 'DIFF':>6}", flush=True)

    print(f"\n[reconcile] all IMM literals survive full 20-bit through pure-forward "
          f"fast path: {all_ok}", flush=True)
    print(f"[reconcile] doom max literal 0x20a15={0x20a15} = {0x20a15.bit_length()} bits "
          f"< 20 -> {'COVERED' if 0x20a15 < (1<<20) else 'NOT COVERED'} by 20-bit IMM",
          flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
