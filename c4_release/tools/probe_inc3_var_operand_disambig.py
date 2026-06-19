#!/usr/bin/env python3
"""Inc-3 var frontier (var_mul/three/update + if_var, ids 275-449) — the UNIFIED
root probe: the 30-tok campaign frame breaks LEA-address vs LI-value operand-row
DISAMBIGUATION in the L8-L12 operand-gather / ALU relay.

THE FINDING (2026-06-19, this probe is the evidence):
  All four still-0/25 var clusters fail at the FIRST multi-LEA load step with
  got_ax = 0xffe8 (an SP-relative LEA *address*, e.g. -0x18) where the oracle
  wants the loaded *value* (or a DIFFERENT local's address). The compare-branch
  (if_gt/lt/eq) clusters are ALREADY FIXED by the L20 AX-crush (b94c77d2); they
  are NOT this root.

  The divergence localizes to ~blk12 (logical L9/L10, the operand->ALU relay):
  at a WORKING LI step the ALU carries the value (var_update step4 ALU=0x32=50);
  at the BROKEN multi-LEA LI step the ALU carries the address high byte
  (var_update step9 ALU_HI=0xf, the 0xffe8 address). The mem[SP] operand CAM /
  L7-L8 operand-gather selects the LEA-ADDRESS row instead of the LI-VALUE row
  because in the 5-token-shorter frame the address-row and value-row signatures
  collide (the same break the multi-local var_three step6 e8-vs-e0 aliasing and
  the var_mul step6 operand-load show).

  CONSEQUENCE: this is a from-scratch operand-row-disambiguation build in the
  L7/L8 operand CAM + L9/L10 relay (the documented multi-session frontier;
  see memory project_l7_operand_gather_not_broken / _operand_gather_hybrid /
  project_l15_li_stack0_byte_attribution). It is NOT a single-rule fix and
  every prior single-rule attempt on this surface was zero-sum.

  NOTE on var_three step0 (campaign-only regression, blk32/L18 unit 3): an
  OUTPUT_HI zero-default darken unit (W_down=-50 on OUTPUT_HI+1..15) FLIPS SIGN
  in the 30-tok frame (golden silu=positive -> darkens byte1 high nibble to 0;
  campaign silu=-0.18 -> +9 BRIGHTENS -> byte1=0x10 -> got_ax=0x1000). Driven by
  two 30-tok leak terms into its up: AX_CARRY_HI.*.-1+15 (x~3.6e-8 * -1e9 =
  -36.6) + BYTE_INDEX_1+0 (x=0.0133 * -1000 = -13.3) pushing up past the silu
  knee. Hardening it would only move var_three's divergence step0->step6
  (matching golden); it is NOT a program-level pass.

Usage (campaign config; GPU is the sole authority):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    python tools/probe_inc3_var_operand_disambig.py

Tooling only (no build path) -> model byte-identical (golden 4958b35b18108745).
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

# (src, working-LI step, broken-LI step) per cluster. The working step proves the
# operand relay CAN deliver the value; the broken step is the multi-LEA collision.
CASES = [
    ("var_update", "int main() { int x; x = 50; x = x + 7; return x; }", 4, 9),
    ("var_mul",    "int main() { int a; int b; a = 23; b = 47; return a * b; }", 4, 6),
    ("var_three",  "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }", 4, 6),
]


def main():
    STEP = int(Token.STEP_TOKENS)
    p = build_groundtruth_probe()
    dp = dict(p.model.dim_positions)
    alo, ahi = dp["ALU_LO"], dp["ALU_HI"]
    print(f"STEP_TOKENS={STEP} (campaign frame: {STEP == 30})")
    for name, src, ok_step, bad_step in CASES:
        bc, _ = compile_c(src)
        ctx = p._final_context(bc, max_steps=18)
        prompt_len = len(p._build_context(bc))
        padded = torch.tensor([list(ctx)], dtype=torch.long, device=p._device)
        print(f"\n=== {name}: ALU value at OK step{ok_step} vs BROKEN step{bad_step} "
              f"(off5) per block ===")
        for blk in (8, 10, 12, 13):
            with torch.no_grad():
                full = p.model.forward(padded, stop_after_block=blk)
                if full.is_sparse:
                    full = full.to_dense()
            cells = []
            for ts in (ok_step, bad_step):
                r = full[0, prompt_len + ts * STEP + 5]
                lo = int(torch.argmax(r[alo:alo + 16]))
                hi = int(torch.argmax(r[ahi:ahi + 16]))
                cells.append(f"step{ts}=0x{hi * 16 + lo:02x}"
                             f"(LO{lo}@{float(r[alo:alo+16].max()):+.0f}/"
                             f"HI{hi}@{float(r[ahi:ahi+16].max()):+.0f})")
            print(f"  blk{blk:2d}: " + "   ".join(cells))


if __name__ == "__main__":
    main()
