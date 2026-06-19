#!/usr/bin/env python3
"""Inc-3 ROOT A — what's DIFFERENT in the Q-row (blk16 head1 input) across configs?

The AX[1] source row's raw K-score against the step5 byte-1 predictor Q-row
COLLAPSES +155M (golden) -> -4.2M (campaign), so the PC_marker (MEM_STORE) row
wins. K-side (the AX[1] row signature) is byte-identical across configs, so the
collapse is on the Q SIDE: the predictor row's residual differs. This dumps the
blk16-INPUT residual at the step5 byte-1 predictor row (off=6) for the dims the
head's Q conditions on -- BOTH configs -- to find which Q gate flipped.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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

SRC = "int main() { int x; x = 990; return x; }"
BLK = 16


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30)" if nostk else "GOLDEN(35)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], device=p._device)
    x_in = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])
    off = pl + 5 * STEP + 6
    row = x_in[off]
    # Q conditions: IS_BYTE, HAS_SE, OP_LI_RELAY, OP_IMM, OP_LC_RELAY, TEMP+3, CMP+3,
    # BYTE_INDEX_0..3, H1+AX, MARK_AX, MEM_STORE, MEM_VAL_B*, OP_SI/OP_SC.
    NAMES = ["IS_BYTE", "HAS_SE", "OP_LI_RELAY", "OP_IMM", "OP_LC_RELAY",
             "OP_LI", "OP_SI", "OP_SC",
             "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
             "MARK_AX", "MARK_MEM", "MEM_STORE", "MEM_ADDR_SRC",
             "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
             "TEMP", "CMP", "H1", "H2", "H3", "H4"]
    print(f"=== {cfg} STEP={STEP} blk{BLK} head1 Q-row (off={off}, step5 byte1 predictor) ===")
    out = []
    for nm in NAMES:
        if nm not in dp:
            continue
        base = dp[nm]
        # for marker bands H1..H4, TEMP, CMP show offsets 0..5
        if nm in ("H1", "H2", "H3", "H4", "TEMP", "CMP"):
            for i in range(6):
                v = float(row[base + i])
                if abs(v) > 0.2:
                    out.append(f"{nm}+{i}={v:+.2f}")
        else:
            v = float(row[base])
            if abs(v) > 0.05:
                out.append(f"{nm}={v:+.2f}")
    print("  " + "  ".join(out))


if __name__ == "__main__":
    main()
