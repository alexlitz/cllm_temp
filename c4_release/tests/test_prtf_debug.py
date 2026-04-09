"""Debug PRTF detection and THINKING_END emission."""

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

print("Compiling...")
code, data = compile_c(c_code)
print(f"✓ Compiled: {len(code)} instructions")

# Print disassembly to find PRTF instruction
print("\nDisassembly:")
for i, instr in enumerate(code):
    op = instr & 0xFF
    imm = (instr >> 8) & 0xFFFFFF
    print(f"  {i}: op={op:3d} (0x{op:02x}) imm=0x{imm:06x}")
    if op == 33:  # PRTF
        print(f"    ^^^ PRTF at instruction {i}")

print("\nCreating runner...")
runner = AutoregressiveVMRunner(conversational_io=True)

if torch.cuda.is_available():
    print("Moving to CUDA...")
    runner.model = runner.model.cuda()

print("✓ Runner created")

# Monkey-patch to inspect activations
original_forward = runner.model.forward

def debug_forward(token_ids):
    result = original_forward(token_ids)

    # Get last position activations after each layer
    if len(token_ids.shape) == 1:
        token_ids = token_ids.unsqueeze(0)

    with torch.no_grad():
        x = runner.model.embed(token_ids)

        # Check after L5 (PRTF detection)
        for i in range(5):
            x = runner.model.blocks[i](x)

        # Check IO_IS_PRTF - scan for AX marker and PC in recent positions
        last_pos = x.shape[1] - 1
        context = token_ids[0].cpu().tolist()

        # Find most recent PC and AX marker
        ax_pos = -1
        pc_pos = -1
        for j in range(len(context) - 1, max(0, len(context) - 40), -1):
            if context[j] == Token.REG_AX and ax_pos < 0:
                ax_pos = j
            if context[j] == Token.REG_PC and pc_pos < 0:
                pc_pos = j
            if ax_pos >= 0 and pc_pos >= 0:
                break

        # Extract PC value (4 bytes after PC marker)
        pc_val = None
        if pc_pos >= 0 and pc_pos + 4 < len(context):
            pc_val = 0
            for k in range(4):
                pc_val |= (context[pc_pos + 1 + k] & 0xFF) << (k * 8)

        instr_idx = pc_val // 6 if pc_val is not None else -1

        io_is_prtf_at_last = x[0, last_pos, BD.IO_IS_PRTF].item()
        mark_ax_at_ax = x[0, ax_pos, BD.MARK_AX].item() if ax_pos >= 0 else 0.0
        io_is_prtf_at_ax = x[0, ax_pos, BD.IO_IS_PRTF].item() if ax_pos >= 0 else 0.0

        # Check opcode bytes at AX marker (PRTF = 0x21, lo=1, hi=2)
        opcode_lo_1 = x[0, ax_pos, BD.OPCODE_BYTE_LO + 1].item() if ax_pos >= 0 else 0.0
        opcode_hi_2 = x[0, ax_pos, BD.OPCODE_BYTE_HI + 2].item() if ax_pos >= 0 else 0.0

        # Print PC and instruction info every few steps
        step_num = len(context) // 35
        if step_num <= 10 or (opcode_lo_1 > 0.5 and opcode_hi_2 > 0.5):
            pc_str = f"0x{pc_val:04x}" if pc_val is not None else "None"
            print(f"[Step {step_num}] PC={pc_str}, instr_idx={instr_idx}, LO[1]={opcode_lo_1:.1f}, HI[2]={opcode_hi_2:.1f}")

        # Print when we detect PRTF opcode bytes (LO[1]=1, HI[2]=1)
        if (opcode_lo_1 > 0.5 and opcode_hi_2 > 0.5):
            print(f"[L5] PRTF DETECTED! IO_IS_PRTF at AX={io_is_prtf_at_ax:.2f}, at SE={io_is_prtf_at_last:.2f}")

        # Check after L6 (relay + state machine)
        x = runner.model.blocks[5](x)

        # The SE position is where we're about to generate STEP_END (current last position)
        se_pos = x.shape[1] - 1

        # Check at SE position (where we're generating next token)
        cmp3_at_se = x[0, se_pos, BD.CMP + 3].item()
        next_se_at_se = x[0, se_pos, BD.NEXT_SE].item()
        next_thinking_end_at_se = x[0, se_pos, BD.NEXT_THINKING_END].item()

        # Also check at AX position (where IO_IS_PRTF was set)
        cmp3_at_ax = x[0, ax_pos, BD.CMP + 3].item() if ax_pos >= 0 else 0.0

        if io_is_prtf_at_ax > 0.1 or abs(cmp3_at_se) > 0.1 or abs(next_se_at_se) > 0.1:
            print(f"[L6] SE pos {se_pos}: CMP[3]={cmp3_at_se:.2f}, NEXT_SE={next_se_at_se:.2f}, NEXT_THINKING_END={next_thinking_end_at_se:.2f}")
            if ax_pos >= 0:
                print(f"      AX pos {ax_pos}: CMP[3]={cmp3_at_ax:.2f}, distance={(se_pos - ax_pos)}")

    return result

runner.model.forward = debug_forward

print("\nRunning VM (max 20 steps)...")
try:
    output_str, exit_code = runner.run(code, data, [], max_steps=20)
    print(f"\n✓ Execution complete")
    print(f"Exit code: {exit_code}")
    print(f"Output: {repr(output_str)}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
