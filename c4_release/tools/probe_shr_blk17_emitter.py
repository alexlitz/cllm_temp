#!/usr/bin/env python3
"""Localize the block-17 SHR OUTPUT byte-0 emitter (campaign config, spec_k=0).

The leak probe showed OUTPUT_{LO,HI}+0 = 2.0 first appears at physical block 17
on the SHR MARK_AX row (empty at block 16, and empty for SHL). This probe:
  1. Prints the physical->logical block map around 16-18.
  2. Dumps the attn/ffn class names + post_ops for block 17.
  3. Shows whether the attn OR the ffn (or a post_op) plants OUTPUT+0 by reading
     the residual just-before vs just-after each sub-component isn't possible
     without hooks, so instead we compare: resid@blk16 (in) vs resid@blk17 (out)
     and report the OP_SHR/OP_SHL discriminator dims on the row.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_SHIFT_OUTPUT_B0_CLEAR", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(items):
    out = []
    for it in items:
        if isinstance(it, tuple):
            out.append(int(it[0])); out.append(int(it[1]))
        else:
            out.append(int(it))
    return out


SHR_PROG = _mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHR, Opcode.EXIT])


def main():
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)

    # Block map around 14-20.
    rows = p.block_layer_map()
    print("phys -> logical | attn | ffn | n_post_ops")
    for r in rows:
        if 14 <= r["physical"] <= 20:
            blk = model.blocks[r["physical"]]
            npost = len(getattr(blk, "post_ops", []) or [])
            post_types = [type(po).__name__ for po in (getattr(blk, "post_ops", []) or [])]
            print(f"  {r['physical']:>3} -> L{r['logical']:<3} | "
                  f"{r['attn']:<26} | {r['ffn']:<26} | post={npost} {post_types}")

    _, oracle_tokens = runner._oracle_pc_ax_steps(
        SHR_PROG, b"", "", expected_steps=None, with_tokens=True)
    prompt = runner._build_element(SHR_PROG, b"", [], "", spec_k=1,
                                   adaptive_start_k=0, expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in oracle_tokens:
        tape.extend(stp)
    padded = torch.tensor([tape], dtype=torch.long, device=dev)
    ax_base = dp["MARK_AX"]
    with torch.no_grad():
        emb = model.embed(padded)[0]
        resid11 = model.forward(padded, stop_after_block=11)[0]

    shift_row = None
    for stp_idx in range(len(oracle_tokens)):
        ss = len(prefix) + stp_idx * STEP
        for off in range(STEP):
            r = ss + off
            if r < len(tape) and emb[r, ax_base].abs().item() > 0.5 \
               and resid11[r, dp["OP_SHR"]].item() > 0.5:
                shift_row = r
                break
        if shift_row is not None:
            break
    print(f"\nSHR MARK_AX row = {shift_row}")

    # Discriminator dims present on the row at block 16 (input to block 17).
    with torch.no_grad():
        resid16 = model.forward(padded, stop_after_block=16)[0]
    row = resid16[shift_row]
    interesting = ["MARK_AX", "OP_SHR", "OP_SHL", "OP_LI", "OP_IMM", "OP_PSH",
                   "IS_BYTE", "MARK_SE", "MARK_SE_ONLY", "MARK_STACK0",
                   "HAS_SE", "AX_CARRY_LO", "AX_CARRY_HI"]
    print("\nrow dims @blk16 (input to block 17):")
    for nm in interesting:
        if nm in dp:
            v = row[dp[nm]].item()
            if abs(v) > 0.2:
                print(f"    {nm} = {v:.2f}")
    # Full nonzero scan of the row at blk16 (to find what block-17 attn reads).
    print("\n  ALU_LO@16:", [f"{row[dp['ALU_LO']+i].item():.1f}@{i}"
                             for i in range(16) if abs(row[dp['ALU_LO']+i].item()) > 0.3])
    print("  ALU_HI@16:", [f"{row[dp['ALU_HI']+i].item():.1f}@{i}"
                           for i in range(16) if abs(row[dp['ALU_HI']+i].item()) > 0.3])


if __name__ == "__main__":
    main()
