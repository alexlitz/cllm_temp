"""
Debug what's present at position 20 (REG_PC marker) that's causing OUTPUT_HI contamination.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

# Build model
print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

# Build initial context
value = 42
bytecode = [
    Opcode.IMM | (value << 8),
    Opcode.EXIT,
]

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Add REG_PC to context (step 1)
context.append(Token.REG_PC)

# Forward pass
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Check embedding at position 20 (REG_PC)
    pos = len(context) - 1
    print(f"Position {pos} (REG_PC):")
    print(f"  Token: {context[pos]}")

    # Check all opcode flags
    print(f"\nOpcode flags at embedding:")
    for opname in ['OP_IMM', 'OP_EXIT', 'OP_NOP', 'OP_JMP', 'OP_BZ', 'OP_BNZ', 'OP_PSH']:
        if hasattr(BD, opname):
            val = x[0, pos, getattr(BD, opname)].item()
            if abs(val) > 0.1:
                print(f"  {opname}: {val:.3f}")

    # Check marker flags
    print(f"\nMarker flags:")
    for markname in ['MARK_PC', 'MARK_AX', 'MARK_SP', 'MARK_BP', 'MARK_STACK0', 'IS_BYTE']:
        if hasattr(BD, markname):
            val = x[0, pos, getattr(BD, markname)].item()
            print(f"  {markname}: {val:.3f}")

    # Check relevant input dimensions for FFN activation
    print(f"\nInput dimensions for FFN:")
    print(f"  HAS_SE: {x[0, pos, BD.HAS_SE].item():.3f}")

    # Check FETCH dimensions (used by IMM units)
    fetch_lo_vals = [x[0, pos, BD.FETCH_LO + k].item() for k in range(16)]
    fetch_hi_vals = [x[0, pos, BD.FETCH_HI + k].item() for k in range(16)]
    if any(abs(v) > 0.1 for v in fetch_lo_vals + fetch_hi_vals):
        print(f"\n  FETCH values present:")
        for k in range(16):
            if abs(fetch_lo_vals[k]) > 0.1 or abs(fetch_hi_vals[k]) > 0.1:
                print(f"    FETCH_LO[{k}]: {fetch_lo_vals[k]:.2f}, FETCH_HI[{k}]: {fetch_hi_vals[k]:.2f}")

    # Now forward through to Layer 6 FFN only
    for i in range(7):  # Layers 0-6
        x = model.blocks[i](x, kv_cache=None)

    print(f"\nAfter blocks[6] (Layer 6):")
    print(f"  OUTPUT_HI[2]: {x[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    # Check CMP dimensions (used by JMP/BZ/BNZ units at MARK_PC)
    print(f"\nCMP dimensions (before Layer 6 FFN):")
    for k in range(16):
        cmp_val = x[0, pos, BD.CMP + k].item()
        if abs(cmp_val) > 0.1:
            print(f"  CMP[{k}]: {cmp_val:.3f}")

    # Check which FFN units might be activating
    S = 16.0
    mark_pc = x[0, pos, BD.MARK_PC].item()

    print(f"\nActivation checks for MARK_PC-gated units:")

    # IMM unit (MARK_AX + OP_IMM > T=5.5)
    T = 5.5
    op_imm = x[0, pos, BD.OP_IMM].item()
    mark_ax = x[0, pos, BD.MARK_AX].item()
    activation = S * op_imm + S * mark_ax - S * T
    print(f"  IMM: OP_IMM({op_imm:.3f}) + MARK_AX({mark_ax:.3f}) - T({T}) = act({activation:.3f}) → {activation > 0}")

    # JMP PC override (MARK_PC + CMP[0] > T_jmp=5.5)
    T_jmp = 5.5
    cmp0 = x[0, pos, BD.CMP + 0].item()
    activation = S * mark_pc + S * cmp0 - S * T_jmp
    print(f"  JMP override: MARK_PC({mark_pc:.3f}) + CMP[0]({cmp0:.3f}) - T({T_jmp}) = act({activation:.3f}) → {activation > 0}")

    # First-step JMP (MARK_PC + OP_JMP - HAS_SE > 5.0)
    op_jmp = x[0, pos, BD.OP_JMP].item()
    has_se = x[0, pos, BD.HAS_SE].item()
    activation = S * mark_pc + S * op_jmp - S * has_se - S * 5.0
    print(f"  First-step JMP: MARK_PC({mark_pc:.3f}) + OP_JMP({op_jmp:.3f}) - HAS_SE({has_se:.3f}) - 5.0 = act({activation:.3f}) → {activation > 0}")

    # BZ override (MARK_PC + CMP[2] + CMP[4] + CMP[5] > T_bz=3.5)
    T_bz = 3.5
    cmp2 = x[0, pos, BD.CMP + 2].item()
    cmp4 = x[0, pos, BD.CMP + 4].item()
    cmp5 = x[0, pos, BD.CMP + 5].item()
    activation = S * mark_pc + S * cmp2 + S * cmp4 + S * cmp5 - S * T_bz
    print(f"  BZ override: MARK_PC({mark_pc:.3f}) + CMP[2]({cmp2:.3f}) + CMP[4]({cmp4:.3f}) + CMP[5]({cmp5:.3f}) - T({T_bz}) = act({activation:.3f}) → {activation > 0}")

    # BNZ override group A (MARK_PC + CMP[3] - CMP[4] > T_bnz=1.5)
    T_bnz = 1.5
    cmp3 = x[0, pos, BD.CMP + 3].item()
    activation = S * mark_pc + S * cmp3 - S * cmp4 - S * T_bnz
    print(f"  BNZ override A: MARK_PC({mark_pc:.3f}) + CMP[3]({cmp3:.3f}) - CMP[4]({cmp4:.3f}) - T({T_bnz}) = act({activation:.3f}) → {activation > 0}")
