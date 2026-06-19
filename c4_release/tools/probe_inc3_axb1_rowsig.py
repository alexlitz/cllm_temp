#!/usr/bin/env python3
"""Inc-3: capture the AX-byte-1 row signature in the campaign 30-tok config vs golden.

The 90 (really ~287) AX_wrong fails are 0xFFE8 -> 0x00E8: PC ok, AX byte-1 (0xFF)
dropped to 0x00. The `tail_sp_pop_byte1_ff_after_{e0,d8,f8,e8}` +
`tail_stack0_pushed_addr_byte1_ff_after_e8` emitters (l10_ops.py:5253-5394) write the
0xFF on a 35-tok-frame ROW SIGNATURE that SHIFTS on the 30-tok frame so they no longer
fire. This probe reads, at the LAST physical block, the AX[1] row (step 2 offset 7) of
var_simple_0, dumping the exact signature dims those rules condition on -- in BOTH
configs -- so we see which condition flipped.

Run TWICE:
  C4_NO_STACK0_EMIT=0 ... python tools/probe_inc3_axb1_rowsig.py   # golden, rule FIRES
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 ... (campaign, rule SILENT)
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

# Signature dims used by the byte-1 emitters (band base + offset -> single dim).
SIG = [
    ("IS_BYTE", 0), ("HAS_SE", 0),
    ("H1", 1), ("H1", 2), ("H1", 3),
    ("BYTE_INDEX_0", 0), ("BYTE_INDEX_1", 0), ("BYTE_INDEX_2", 0),
    ("CMP", 3),
    ("CLEAN_EMBED_LO", 0), ("CLEAN_EMBED_LO", 8),
    ("CLEAN_EMBED_HI", 13), ("CLEAN_EMBED_HI", 14), ("CLEAN_EMBED_HI", 15),
    ("STACK0_BYTE0", 0), ("MARK_STACK0", 0), ("MARK_AX", 0),
    ("MARK_MEM", 0), ("MEM_STORE", 0),
    ("MEM_VAL_B0", 0), ("MEM_VAL_B1", 0),
    # cross-step + AX-discriminator dims
    ("H1_PREV_STEP", 0), ("H3_PREV_STEP", 4), ("H3_PREV_STEP", 5),
    ("SE_REG_AX_PRESENT", 0), ("AX_CARRY_OVERFLOW", 0),
    ("ALU_LO", 0), ("ALU_HI", 0),
    ("OUTPUT_LO", 0), ("OUTPUT_HI_THIS_STEP", 0),
]


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    dp = dict(p.runner.model_runner.layout.dim_positions) if hasattr(
        p.runner, "model_runner") else None
    # Fallback: pull dim_positions off the built layout.
    if dp is None:
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
    print(f"src: {SRC}")
    # The byte-1 token lands at off 7; its PREDICTOR row is off 6 (AX[0] row).
    # Probe off 6 (predictor of byte-1) at the divergent step 2 AND the working
    # step 3, plus the AX[1] row (off 7) itself.
    import torch
    for step in (2, 3):
        base = prompt_len + step * STEP
        for off in (6, 7):
            pos = base + off
            try:
                vals = p.residual_at(bytecode, last_block, pos, dim_names,
                                     max_steps=6)
            except Exception as e:
                print(f" step{step} off{off} ({_step_offset_field(off)}): ERR {e}")
                continue
            # Predicted byte at this position (LM-head argmax of the FULL forward).
            ctx = p._final_context(bytecode, max_steps=6)
            padded = torch.tensor([ctx[:pos + 1]], dtype=torch.long,
                                  device=p._device)
            logits = p.model.forward(padded)[0, pos]
            pred = int(logits.argmax().item())
            tag = _step_offset_field(off)
            sig = " ".join(
                f"{k}={v:+.1f}" for k, v in vals.items()
                if abs(v) > 0.5 and k not in ("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"))
            print(f" step{step} off{off:2d} [{tag:8s}] pred_next=0x{pred:02x}"
                  f"({pred}) OUT_LO={vals.get('OUTPUT_LO',0):+7.1f} "
                  f"OUT_HI={vals.get('OUTPUT_HI_THIS_STEP',0):+7.1f}")
            print(f"     SIG: {sig}")


if __name__ == "__main__":
    main()
