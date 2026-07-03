#!/usr/bin/env python3
"""Inc-3 cont.: probe the step-5 LI byte-1 load for var_simple x=990.

step4 SI stores x=990 (0x03DE) into local mem; step5 `return x` -> LI reloads.
In the 30-tok campaign config AX byte-1 (0x03) is DROPPED -> AX=0x00DE (222).
This probe captures, at the step-5 AX[0] predictor row (offset 6, which PREDICTS
the byte-1 token at offset 7) the residual + the OUTPUT_HI/LO that decodes the
byte-1 token -- in BOTH configs.

FINDING (GPU-confirmed 2026-06-18, this session): the byte-1 (0x03) is delivered
in GOLDEN at the L13 attention block (physical block 16 in this build) by a head
that CROSS-STEP attends the PRIOR step's AX[1] row (step4 AX held 990=0x03DE, so
its byte-1=3 is carried forward to the LI reload). See probe_inc3_l11h1.py for the
per-head decomposition + the K-score dims (OP_IMM+IS_BYTE+BYTE_INDEX_1+H1.*.-1).
It is NOT L15 (L15 writes OUTPUT at the byte-1 token row off7, not the off6
predictor) and NOT the L8 mem-CAM. In CAMPAIGN that L13 carry's K-match shifts to
the wrong row (PC marker via a MEM_STORE leak) -> byte-1 -> 0. The head identity
also differs across the two widths (block16/head1 is a different op flag-ON vs
flag-OFF) so a direct in-place patch of the existing head is layout-fragile.

Run TWICE (clear cache between!):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_li_byte1.py   # golden, byte-1 delivered
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_li_byte1.py  # campaign, dropped
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token, _step_offset_field  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 990; return x; }"

# Dims that govern the L15 LI-load + the AX byte-1 OUTPUT.
SIG = [
    ("MARK_AX", 0), ("MARK_MEM", 0), ("MEM_STORE", 0),
    ("MEM_VAL_B1", 0), ("MEM_VAL_B2", 0), ("MEM_VAL_B3", 0),
    ("BYTE_INDEX_0", 0), ("BYTE_INDEX_1", 0), ("BYTE_INDEX_2", 0),
    ("OP_LI", 0), ("OP_LI_RELAY", 0), ("OP_LC_RELAY", 0),
    ("L2H0", 4), ("H1", 4),
    ("CLEAN_EMBED_LO", 0), ("CLEAN_EMBED_HI", 0),
    ("OUTPUT_LO", 0), ("OUTPUT_LO", 14), ("OUTPUT_LO", 13),
    ("OUTPUT_HI_THIS_STEP", 0), ("OUTPUT_HI_THIS_STEP", 3),
    ("ALU_LO", 0), ("ALU_HI", 0),
    ("AX_FULL_LO", 0), ("AX_FULL_HI", 0),
    ("STACK0_BYTE_VAL_1_LO", 0), ("STACK0_BYTE_VAL_1_HI", 0),
]


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_layout.dim_positions)

    dim_names = {}
    for base, off in SIG:
        if base in dp:
            dim_names[f"{base}+{off}" if off else base] = dp[base] + off

    STEP = int(Token.STEP_TOKENS)
    last_block = len(p.runner.model.blocks) - 1
    prompt_len = len(p._build_context(bytecode))
    print(f"=== {cfg}  STEP_TOKENS={STEP}  last_block={last_block} ===")
    print(f"src: {SRC}  (x=990=0x03DE; byte0=0xDE=222, byte1=0x03)")

    # Dump the full step-5 emitted tokens so we SEE the AX bytes.
    ctx = p._final_context(bytecode, max_steps=9)
    s5 = prompt_len + 5 * STEP
    print(f"step5 tokens [{s5}:{s5+STEP}]: {ctx[s5:s5+STEP]}")
    # AX field starts at offset 5; bytes at 6,7,8,9.
    print(f"  AX bytes: {ctx[s5+6:s5+10]}  (want [222, 3, 0, 0])")

    for off in (5, 6, 7):
        pos = s5 + off
        vals = p.residual_at(bytecode, last_block, pos, dim_names, max_steps=9)
        padded = torch.tensor([ctx[:pos + 1]], dtype=torch.long, device=p._device)
        logits = p.model.forward(padded)[0, pos]
        pred = int(logits.argmax().item())
        tag = _step_offset_field(off)
        sig = " ".join(
            f"{k}={v:+.1f}" for k, v in vals.items() if abs(v) > 0.3)
        print(f" step5 off{off:2d} [{tag:10s}] pred_next=0x{pred:02x}({pred})")
        print(f"     SIG: {sig}")


if __name__ == "__main__":
    main()
