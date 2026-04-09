"""Check activation values during actual PRTF step generation."""

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

runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Track when we generate the PRTF step
prtf_step_checked = [False]
original_forward = runner.model.forward

def check_prtf_step(token_ids):
    result = original_forward(token_ids)

    # Only check when active_opcode is PRTF
    if runner.model._active_opcode == 33 and not prtf_step_checked[0]:
        prtf_step_checked[0] = True

        print(f"\n{'='*60}")
        print(f"CHECKING PRTF STEP (active_opcode=33)")
        print(f"{'='*60}")
        print(f"Context length: {token_ids.shape[1]}")

        with torch.no_grad():
            # Get activations through L6
            x = runner.model.embed(token_ids, active_opcode=33)

            for i in range(7):  # Through L6
                x = runner.model.blocks[i](x)

                if i == 4:  # After L5
                    last_pos = x.shape[1] - 1
                    io_is_prtf = x[0, last_pos, BD.IO_IS_PRTF].item()
                    print(f"\nAfter L5 FFN:")
                    print(f"  Last position: {last_pos}")
                    print(f"  IO_IS_PRTF at last pos: {io_is_prtf:.2f}")

                elif i == 5:  # After L6 attention (relay)
                    last_pos = x.shape[1] - 1
                    # Check last few positions
                    print(f"\nAfter L6 Attention (relay):")
                    for offset in range(5, 0, -1):
                        pos = last_pos - offset + 1
                        if pos >= 0:
                            tok = token_ids[0, pos].item() if pos < token_ids.shape[1] else -1
                            next_se = x[0, pos, BD.NEXT_SE].item()
                            io_is_prtf = x[0, pos, BD.IO_IS_PRTF].item()
                            cmp3 = x[0, pos, BD.CMP + 3].item()

                            tok_name = "?"
                            if tok == Token.STEP_END:
                                tok_name = "STEP_END"
                            elif tok == Token.REG_PC:
                                tok_name = "REG_PC"
                            elif tok == Token.MEM:
                                tok_name = "MEM"

                            if next_se > 0.5 or io_is_prtf > 0.5 or cmp3 > 0.5:
                                print(f"  Pos {pos} ({tok_name}): NEXT_SE={next_se:.2f}, IO_IS_PRTF={io_is_prtf:.2f}, CMP[3]={cmp3:.2f}")

                elif i == 6:  # After L6 FFN (state machine)
                    last_pos = x.shape[1] - 1
                    print(f"\nAfter L6 FFN (state machine):")
                    for offset in range(5, 0, -1):
                        pos = last_pos - offset + 1
                        if pos >= 0:
                            next_se = x[0, pos, BD.NEXT_SE].item()
                            next_thinking_end = x[0, pos, BD.NEXT_THINKING_END].item()
                            cmp3 = x[0, pos, BD.CMP + 3].item()

                            if next_se > 0.5 or next_thinking_end > 0.5:
                                print(f"  Pos {pos}: NEXT_SE={next_se:.2f}, CMP[3]={cmp3:.2f}, NEXT_THINKING_END={next_thinking_end:.2f}")

    return result

runner.model.forward = check_prtf_step

print("\nRunning VM...")
output_str, exit_code = runner.run(code, data, [], max_steps=10)

print(f"\n✓ Execution complete, exit code: {exit_code}")

if prtf_step_checked[0]:
    print("\n✓ PRTF step was checked")
else:
    print("\n⚠️ PRTF step was never reached")
