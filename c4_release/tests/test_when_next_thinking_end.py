"""Find at what context length NEXT_THINKING_END is set."""

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

# Check at every generation point when opcode is PRTF
original_generate = runner.model.generate_next

def check_every_prtf(context):
    if runner.model._active_opcode == 33:
        with torch.no_grad():
            token_ids = torch.tensor([context], device=runner.model.embed.embed.weight.device)

            # Get activations
            x = runner.model.embed(token_ids, active_opcode=33)
            for i in range(16):
                x = runner.model.blocks[i](x)

            last_pos = len(context) - 1
            next_thinking_end = x[0, last_pos, BD.NEXT_THINKING_END].item()
            next_se = x[0, last_pos, BD.NEXT_SE].item()
            cmp3 = x[0, last_pos, BD.CMP + 3].item()

            if next_thinking_end > 0.5 or next_se > 0.5:
                print(f"Context len {len(context)}, pos {last_pos}: NEXT_THINKING_END={next_thinking_end:.2f}, NEXT_SE={next_se:.2f}, CMP[3]={cmp3:.2f}")

    return original_generate(context)

runner.model.generate_next = check_every_prtf

print("\nRunning VM (showing activations when PRTF active)...")
output_str, exit_code = runner.run(code, data, [], max_steps=10)

print(f"\n✓ Execution complete")
