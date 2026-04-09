"""Check activations at the exact point where token 113 is generated."""

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

# Intercept at generation point
checked = [False]
original_generate = runner.model.generate_next

def check_at_generation(context):
    # Check when we're about to generate token for position that will become 113+
    # and active_opcode is PRTF
    if runner.model._active_opcode == 33 and len(context) == 116 and not checked[0]:
        checked[0] = True

        print(f"\n{'='*60}")
        print(f"Generating token at context length {len(context)}")
        print(f"active_opcode = {runner.model._active_opcode} (PRTF)")
        print(f"{'='*60}")

        # Get the actual forward pass result
        with torch.no_grad():
            token_ids = torch.tensor([context], device=runner.model.embed.embed.weight.device)
            logits = runner.model(token_ids, kv_cache=None)  # [1, seq, vocab]

            last_pos = len(context) - 1

            # Get activations (re-run to inspect intermediate layers)
            x = runner.model.embed(token_ids, active_opcode=33)
            for i in range(16):  # All layers
                x = runner.model.blocks[i](x)

            # Check key dimensions at last position
            next_thinking_end = x[0, last_pos, BD.NEXT_THINKING_END].item()
            next_se = x[0, last_pos, BD.NEXT_SE].item()
            io_is_prtf = x[0, last_pos, BD.IO_IS_PRTF].item()

            print(f"\nActivations at position {last_pos} (last):")
            print(f"  NEXT_THINKING_END: {next_thinking_end:.2f}")
            print(f"  NEXT_SE: {next_se:.2f}")
            print(f"  IO_IS_PRTF: {io_is_prtf:.2f}")

            # Get top 5 token logits
            last_logits = logits[0, last_pos, :]
            top5_logits, top5_tokens = torch.topk(last_logits, 5)

            print(f"\nTop 5 tokens by logit:")
            for logit, tok in zip(top5_logits, top5_tokens):
                tok = tok.item()
                logit = logit.item()
                tok_name = "?"
                if tok == Token.THINKING_END:
                    tok_name = "THINKING_END ★"
                elif tok == Token.STEP_END:
                    tok_name = "STEP_END"
                elif tok == Token.REG_PC:
                    tok_name = "REG_PC"
                elif 0 <= tok < 256:
                    tok_name = f"byte_{tok:02x}"
                else:
                    tok_name = f"tok_{tok}"

                print(f"  {tok:3d} ({tok_name:20s}): logit = {logit:7.2f}")

            # Check THINKING_END logit specifically
            thinking_end_logit = logits[0, last_pos, Token.THINKING_END].item()
            print(f"\nTHINKING_END token ({Token.THINKING_END}) logit: {thinking_end_logit:.2f}")

            if thinking_end_logit < -50:
                print(f"  ⚠️ THINKING_END is being strongly suppressed!")

    return original_generate(context)

runner.model.generate_next = check_at_generation

print("\nRunning VM...")
output_str, exit_code = runner.run(code, data, [], max_steps=10)

print(f"\n✓ Execution complete")

if checked[0]:
    print("✓ Generation point was checked")
else:
    print("⚠️ Generation point was not reached")
