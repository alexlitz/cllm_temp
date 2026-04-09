"""Check actual dimension values when PRTF executes."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

c_code = '''
int main() {
    printf("H\\n");
    return 0;
}
'''

code, data = compile_c(c_code)
runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track when we reach PRTF instruction
prtf_pc = 5 * 6  # instruction 5, PC = 0x1e

# Detailed activation inspection
original_forward = runner.model.forward
step_count = [0]

def inspect_forward(token_ids):
    result = original_forward(token_ids)

    # Check every forward pass, not just step boundaries
    step_count[0] += 1

    # Get activations after L6
    with torch.no_grad():
        if len(token_ids.shape) == 1:
            token_ids = token_ids.unsqueeze(0)

        context = token_ids[0].cpu().tolist()

        # Find PC value
        pc_pos = -1
        for j in range(len(context) - 1, max(0, len(context) - 40), -1):
            if context[j] == Token.REG_PC:
                pc_pos = j
                break

        pc_val = None
        if pc_pos >= 0 and pc_pos + 4 < len(context):
            pc_val = 0
            for k in range(4):
                pc_val |= (context[pc_pos + 1 + k] & 0xFF) << (k * 8)

        instr_idx = pc_val // 6 if pc_val is not None else -1

        # Only print when at PRTF instruction
        if instr_idx == 5:
            x = runner.model.embed(token_ids)
            for i in range(6):  # Through L6
                x = runner.model.blocks[i](x)

            # Find ALL PC markers to check for duplicates
            all_pc_positions = []
            for j in range(len(context)):
                if context[j] == Token.REG_PC:
                    if j + 4 < len(context):
                        pc_bytes = [context[j + 1 + k] & 0xFF for k in range(4)]
                        pc_value = sum(pc_bytes[k] << (k*8) for k in range(4))
                        all_pc_positions.append((j, pc_value))

            # Find PC and AX positions (most recent)
            pc_pos = -1
            ax_pos = -1
            ax_val = None
            pc_val_check = None
            for j in range(len(context) - 1, max(0, len(context) - 60), -1):
                if context[j] == Token.REG_PC and pc_pos < 0:
                    pc_pos = j
                    # Also extract PC bytes to verify
                    if j + 4 < len(context):
                        pc_val_check = 0
                        for k in range(4):
                            pc_val_check |= (context[j + 1 + k] & 0xFF) << (k * 8)
                if context[j] == Token.REG_AX and ax_pos < 0:
                    ax_pos = j
                    # Extract 4 bytes after AX marker
                    if j + 4 < len(context):
                        ax_val = 0
                        for k in range(4):
                            ax_val |= (context[j + 1 + k] & 0xFF) << (k * 8)
                # Keep searching until we find both
                if pc_pos >= 0 and ax_pos >= 0:
                    break

            last_pos = x.shape[1] - 1

            # Check opcode bytes at PC position (where they should be set)
            opcode_lo_1_pc = x[0, pc_pos, BD.OPCODE_BYTE_LO + 1].item() if pc_pos >= 0 else 0.0
            opcode_hi_2_pc = x[0, pc_pos, BD.OPCODE_BYTE_HI + 2].item() if pc_pos >= 0 else 0.0

            # Check which opcode nibbles are active at PC
            opcode_lo_sum_pc = 0.0
            opcode_hi_sum_pc = 0.0
            opcode_lo_which_pc = -1
            opcode_hi_which_pc = -1
            if pc_pos >= 0:
                for k in range(16):
                    lo_val = x[0, pc_pos, BD.OPCODE_BYTE_LO + k].item()
                    hi_val = x[0, pc_pos, BD.OPCODE_BYTE_HI + k].item()
                    opcode_lo_sum_pc += abs(lo_val)
                    opcode_hi_sum_pc += abs(hi_val)
                    if abs(lo_val) > 0.5:
                        opcode_lo_which_pc = k
                    if abs(hi_val) > 0.5:
                        opcode_hi_which_pc = k

            # Extract values at PC position
            io_is_prtf_pc = x[0, pc_pos, BD.IO_IS_PRTF].item() if pc_pos >= 0 else 0.0
            mark_pc = x[0, pc_pos, BD.MARK_PC].item() if pc_pos >= 0 else 0.0

            # Extract values at AX and SE positions
            io_is_prtf_ax = x[0, ax_pos, BD.IO_IS_PRTF].item() if ax_pos >= 0 else 0.0
            mark_ax = x[0, ax_pos, BD.MARK_AX].item() if ax_pos >= 0 else 0.0
            cmp3_last = x[0, last_pos, BD.CMP + 3].item()
            next_se = x[0, last_pos, BD.NEXT_SE].item()
            next_thinking_end = x[0, last_pos, BD.NEXT_THINKING_END].item()

            pc_str = f"0x{pc_val:04x}" if pc_val is not None else "None"
            pc_check_str = f"0x{pc_val_check:04x}" if pc_val_check is not None else "None"
            ax_str = f"0x{ax_val:08x}" if ax_val is not None else "None"

            # Show all PC markers in context
            print(f"\n[Check #{step_count[0]}] All PC markers in context:")
            for pos, val in all_pc_positions:
                print(f"  Position {pos}: PC=0x{val:04x} (instr {val//6})")

            # Show actual tokens around most recent PC marker
            if pc_pos >= 0 and pc_pos + 5 < len(context):
                pc_tokens = [context[pc_pos + i] for i in range(6)]
                print(f"\n  Using most recent PC={pc_str}, instr={instr_idx}, AX={ax_str}")
                print(f"  PC marker tokens: {pc_tokens} (should be [REG_PC, b0, b1, b2, b3, ...])")
                print(f"  PC bytes: {[f'0x{b:02x}' for b in pc_tokens[1:5]]}")
            else:
                print(f"\n  Using PC={pc_str}, instr={instr_idx}, AX={ax_str}")

            print(f"  Positions: PC pos={pc_pos}, AX pos={ax_pos}, Last pos={last_pos}")
            if pc_val != pc_val_check:
                print(f"  ⚠️ PC MISMATCH! Earlier scan found {pc_str}, but marker has {pc_check_str}")
            print(f"  At PC: MARK_PC={mark_pc:.2f}")
            print(f"  At PC: Opcode bytes LO[1]={opcode_lo_1_pc:.2f}, HI[2]={opcode_hi_2_pc:.2f}")
            print(f"  At PC: Opcode totals: LO_sum={opcode_lo_sum_pc:.2f} (nibble {opcode_lo_which_pc}), HI_sum={opcode_hi_sum_pc:.2f} (nibble {opcode_hi_which_pc})")
            print(f"  Expected: PRTF=0x21 → lo=1, hi=2")
            print(f"  At PC: IO_IS_PRTF={io_is_prtf_pc:.2f}")
            print(f"  At AX: MARK_AX={mark_ax:.2f}, IO_IS_PRTF={io_is_prtf_ax:.2f}")
            print(f"  At Last (SE): CMP[3]={cmp3_last:.2f}, NEXT_SE={next_se:.2f}, NEXT_THINKING_END={next_thinking_end:.2f}")

            if instr_idx == 5:
                print(f"  >>> THIS IS THE PRTF INSTRUCTION! <<<")
                if opcode_lo_1_pc > 0.5 and opcode_hi_2_pc > 0.5:
                    print(f"  ✓ Opcode bytes match PRTF (0x21) at PC")
                else:
                    print(f"  ✗ Opcode bytes DON'T match PRTF at PC")

                if io_is_prtf_pc > 0.5:
                    print(f"  ✓ IO_IS_PRTF set at PC marker")
                else:
                    print(f"  ✗ IO_IS_PRTF NOT set at PC marker")

                if cmp3_last > 0.5:
                    print(f"  ✓ CMP[3] set at SE position (relay worked)")
                else:
                    print(f"  ✗ CMP[3] NOT set at SE position (relay failed)")

                if next_thinking_end > 0.5:
                    print(f"  ✓ NEXT_THINKING_END set!")
                else:
                    print(f"  ✗ NEXT_THINKING_END NOT set")

    return result

runner.model.forward = inspect_forward

print("Running VM...\n")
output_str, exit_code = runner.run(code, data, [], max_steps=10)
print(f"\n✓ Execution complete, exit code: {exit_code}")
