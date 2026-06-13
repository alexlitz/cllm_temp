"""MUL efficient-mode diagnostic + fix validation (2026-06-13).

Confirms the mul_basic root + fix on the production smoke path
(trust_neural_alu=True => alu_mode='efficient', spec_k=0):

ROOT (refutes the lookup-mode "dead dispatch" brief premise):
  The smoke gate runs EFFICIENT mode, not lookup. In efficient mode the
  MUL compute IS installed (``make_efficient_l11_alumul_wrap_op`` bakes a
  256-rule ``wide_mul_rules`` PureFFN), BUT before the fix it landed on the
  WRONG block: ``target_op_name="_layer11_ffn_dep_anchor"`` resolved to
  pre-expansion layer 15 (physical block 26 = logical L15) instead of L11
  (physical block 12), because the L10 op family stacks across pre-exp
  layers 9..14 and the anchor (``requires after: layer10_carry_relay``)
  floated to 15. So the MUL operands — present + clean at the MARK_AX row
  through L11 (ALU_LO one-hot for A, AX_CARRY_LO one-hot for B) — were never
  multiplied, and the L15-resident wide_mul wrote a NOISY OUTPUT band (the
  "L15 OUTPUT materialiser" prior docs blamed IS this misplaced wrap) that
  the L20 tail spike amplified into a wrong decode (mul_basic -> 1).

FIX (this run validates it on the real compiled model):
  1. ``layer_idx=11`` pin (replaces the mis-resolving ``target_op_name``).
  2. Operand-magnitude-matched AND thresholds (ALU_LO one-hot ~6, not 1.0).

mul_overflow stays arch-blocked: it needs an 8x8 (width=2) product, whose
flat lookup is poisoned by the L8 operand-gather cell-0 hybrid encoding
artifact (Wall-1) and whose extra result lanes collide with CLEAN_EMBED.

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/probe_mul_full_fix_validate.py
"""
import os
os.environ.setdefault("C4_TEST_SPEC_K", "0")
import torch
torch.set_grad_enabled(False)

from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.embedding import Opcode
from tests.test_smoke import _make_bytecode


def main():
    p = build_groundtruth_probe()
    # Block placement: the wide_mul 256-rule PureFFN must be at logical L11.
    for phys, blk in enumerate(p.model.blocks):
        lg = getattr(blk, "_logical_layer", phys)
        ffn = getattr(blk, "ffn", None)
        if lg == 11:
            print(f"L11 = physical block {phys}: ffn={type(ffn).__name__} "
                  f"hidden={getattr(ffn, 'hidden_dim', None)} "
                  f"(expect PureFFN hidden=256 after fix)")

    cases = [
        ("mul_basic", [(Opcode.IMM, 6), Opcode.PSH, (Opcode.IMM, 7), Opcode.MUL, Opcode.EXIT], 42),
        ("mul_overflow", [(Opcode.IMM, 100), Opcode.PSH, (Opcode.IMM, 5), Opcode.MUL, Opcode.EXIT], 500),
    ]
    for label, prog, exp in cases:
        bc = _make_bytecode(prog)
        _out, ec = p.emitted_result(bc, max_steps=20)
        print(f"{label}: exit={ec} expected={exp} "
              f"{'PASS' if ec == exp else 'FAIL'}")


if __name__ == "__main__":
    main()
