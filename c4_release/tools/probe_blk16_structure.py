#!/usr/bin/env python3
"""Inspect physical block 16's structure (attn zero-init? ffn post_ops?) and
isolate whether the OUTPUT_LO cmp-result write at the SE row is attention or FFN.

Per docs/PROBE_GROUNDTRUTH_2026_06_10.md, physical block 16 is a logical-L14
post_op EXPANSION passthrough (zero-init attn = residual identity, post_op as
ffn). This probe verifies that for the BUILT model and measures the OUTPUT_LO
cell0/cell1 delta contributed by (a) block-16 attention only, (b) block-16 ffn.

  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_varupd2 python tools/probe_blk16_structure.py
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

PROGS = {
    "ifVAR_66gt24_T": ("int main() { int x; x = 66; if (x > 24) return 1; return 0; }", 10),
    "ifGT_66gt24_T":  ("int main() { if (66 > 24) return 1; return 0; }", 3),
}


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    se_mark = dp.get("MARK_SE_ONLY")
    olo = dp["OUTPUT_LO"]

    # --- structure of block 16 ---
    blocks = model.blocks
    blk = blocks[16]
    attn = getattr(blk, "attn", None)
    print(f"=== block 16 structure ===", flush=True)
    print(f"  type={type(blk).__name__} attn={type(attn).__name__ if attn else None}", flush=True)
    for nm in ("_logical_layer", "_is_post_op_expansion"):
        print(f"  {nm}={getattr(blk, nm, '<none>')}", flush=True)
    if attn is not None:
        for wn in ("W_q", "W_k", "W_v", "W_o"):
            w = getattr(attn, wn, None)
            if w is not None:
                print(f"  attn.{wn}: shape={tuple(w.shape)} absmax={w.abs().max().item():.3e} nnz={(w.abs()>1e-9).sum().item()}", flush=True)
    ffn = getattr(blk, "ffn", None)
    print(f"  ffn={type(ffn).__name__ if ffn else None}", flush=True)
    postops = getattr(blk, "post_ops", None)
    print(f"  post_ops={len(postops) if postops is not None else 0}", flush=True)

    # --- residual cmp-result delta: block 15 -> 16, attn-only vs full ---
    for pname, (src, res_step) in PROGS.items():
        bc, data = compile_c(src)
        ctx = p._final_context(bc, max_steps=25)
        prefix_len = len(p._build_context(bc))
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(padded)[0]
        stp = res_step
        step_start = prefix_len + stp * STEP
        # find SE row of result step
        se_row = None
        for off in range(STEP):
            r = step_start + off
            if r >= len(ctx):
                break
            if se_mark is not None and emb[r, se_mark].abs().item() > 0.5:
                se_row = r
                break
        if se_row is None:
            print(f"\n{pname}: no SE row at step {stp}", flush=True)
            continue
        with torch.no_grad():
            r15 = model.forward(padded, stop_after_block=15)[0][se_row]
            r16 = model.forward(padded, stop_after_block=16)[0][se_row]
        c0_15, c1_15 = r15[olo + 0].item(), r15[olo + 1].item()
        c0_16, c1_16 = r16[olo + 0].item(), r16[olo + 1].item()
        print(f"\n=== {pname} SE row {se_row} (step {stp}) ===", flush=True)
        print(f"  after blk15: OUTPUT_LO[0]={c0_15:+.4f} [1]={c1_15:+.4f} (argmax cell{0 if c0_15>c1_15 else 1})", flush=True)
        print(f"  after blk16: OUTPUT_LO[0]={c0_16:+.4f} [1]={c1_16:+.4f} (argmax cell{0 if c0_16>c1_16 else 1})", flush=True)
        print(f"  blk16 delta: [0]={c0_16-c0_15:+.4f} [1]={c1_16-c1_15:+.4f}", flush=True)


if __name__ == "__main__":
    main()
