#!/usr/bin/env python3
"""Inc-3 if_gt: find the BLOCK that writes the OUTPUT_LO+1 leak (~+6985) at the
branch-step PC-marker row, on the ORACLE clean tape (so positions are stable, not
drifted). Per-block residual scan of dim OUTPUT_LO+1 using the PROBE model's OWN
dim_positions. Localizes the op to gate.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_ifgt_leaksrc.py
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
# Which OUTPUT nibble to track (default OUTPUT_LO+1, the low-nibble-1 leak).
NIB = os.environ.get("PROBE_NIB", "OUTPUT_LO+1")


def main():
    STEP = int(Token.STEP_TOKENS)
    bytecode, data = compile_c(SRC)
    p = build_groundtruth_probe()
    runner = p.runner
    dp = dict(p.model.dim_positions)
    base_dim, off = (NIB.split("+", 1) + ["0"])[:2]
    track = dp[base_dim] + int(off)
    _, oracle_tokens = runner._oracle_pc_ax_steps(
        bytecode, data or b"", "", expected_steps=None, with_tokens=True)
    prompt = runner._build_element(bytecode, data or b"", [], "", spec_k=1,
                                   adaptive_start_k=0, expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in oracle_tokens:
        tape.extend(stp)
    padded = torch.tensor([tape], dtype=torch.long, device=p._device)
    # The PC-marker row of step TARGET_STEP predicts PC byte0.
    row = len(prefix) + TARGET_STEP * STEP + 0
    bl_map = p.block_layer_map()
    nblk = len(p.model.blocks)
    print(f"=== {NIB}=dim{track} at step{TARGET_STEP} PC-marker row={row} "
          f"(blocks={nblk}) ===")
    prev = 0.0
    with torch.no_grad():
        for blk in range(nblk):
            v = float(p.model.forward(padded, stop_after_block=blk)[0, row, track].item())
            if abs(v - prev) > 50.0:
                lm = bl_map[blk]
                print(f" blk{blk:2d} (L{lm['logical']:2d} {lm['ffn'][:30]:30s}) "
                      f"{NIB} {prev:+12.2f} -> {v:+12.2f}")
            prev = v
    print(f" FINAL {NIB} = {prev:+.2f}")


if __name__ == "__main__":
    main()
