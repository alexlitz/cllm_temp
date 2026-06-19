#!/usr/bin/env python3
"""Inc-3 if_gt: the AX-marker row (off=5 of the branch step) PREDICTS the AX
value-byte (byte0=0x00 for a not-taken branch). It loses to a spurious REG_PC
marker. This probe (a) decomposes the LM-head logit for byte0 vs the register
markers at that row, and (b) does a per-block residual scan of OUTPUT_LO+0 /
OUTPUT_HI+0 (the AX value-byte cells) to find the lev_routing crush block.

Runs on the ORACLE clean tape (stable positions). Uses the PROBE model's OWN
dim_positions.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_ifgt_axcrush.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
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
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")
TARGET_STEP = int(os.environ.get("PROBE_STEP", "4"))
# off=5 = the AX register marker row, which predicts the AX byte-0 value token.
OFF = int(os.environ.get("PROBE_OFF", "5"))


def main():
    STEP = int(Token.STEP_TOKENS)
    bytecode, data = compile_c(SRC)
    p = build_groundtruth_probe()
    runner = p.runner
    dp = dict(p.model.dim_positions)
    _, oracle_tokens = runner._oracle_pc_ax_steps(
        bytecode, data or b"", "", expected_steps=None, with_tokens=True)
    prompt = runner._build_element(bytecode, data or b"", [], "", spec_k=1,
                                   adaptive_start_k=0, expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in oracle_tokens:
        tape.extend(stp)
    padded = torch.tensor([tape], dtype=torch.long, device=p._device)
    row = len(prefix) + TARGET_STEP * STEP + OFF
    print(f"=== step{TARGET_STEP} off={OFF} (AX-marker predictor) row={row} src={SRC!r} ===")

    # (a) Final LM-head logit for byte0 vs the register markers.
    with torch.no_grad():
        logits = p.model.forward(padded)[0, row]
    print("  LM-head logits at this row:")
    for nm, tk in [("byte0", 0), ("byte1", 1), ("byte2", 2),
                   ("REG_PC", 257), ("REG_AX", 258), ("REG_SP", 259),
                   ("REG_BP", 260)]:
        print(f"    {nm:8s}(tok{tk:3d}) = {float(logits[tk]):+.2f}")
    print(f"    argmax = tok{int(logits.argmax())}")

    # (b) per-block residual scan of OUTPUT_LO+0 / OUTPUT_HI+0.
    bl_map = p.block_layer_map()
    nblk = len(p.model.blocks)
    for nib in ("OUTPUT_LO", "OUTPUT_HI", "OUTPUT_HI_THIS_STEP"):
        if nib not in dp:
            continue
        track = dp[nib] + 0
        print(f"  --- per-block {nib}+0 = dim{track} ---")
        prev = 0.0
        with torch.no_grad():
            for blk in range(nblk):
                v = float(p.model.forward(padded, stop_after_block=blk)[0, row, track].item())
                if abs(v - prev) > 50.0:
                    lm = bl_map[blk]
                    print(f"   blk{blk:2d} (L{lm['logical']:2d} {lm['ffn'][:34]:34s}) "
                          f"{prev:+14.2f} -> {v:+14.2f}")
                prev = v
        print(f"   FINAL {nib}+0 = {prev:+.2f}")


if __name__ == "__main__":
    main()
