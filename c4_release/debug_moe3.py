"""Debug MoE: instrument the actual runner to trace every step."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Build model with MoE
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Monkey-patch model.set_active_opcode to trace calls
orig_set_active = model.set_active_opcode
def traced_set_active(opcode_value):
    if opcode_value is not None:
        dim = _SetDim.opcode_dim(opcode_value)
        print(f"  [MoE] set_active_opcode({opcode_value}) -> dim={dim}")
    else:
        print(f"  [MoE] set_active_opcode(None) -> full weights")
    orig_set_active(opcode_value)
model.set_active_opcode = traced_set_active

# Monkey-patch generate_next to trace tokens
orig_gen = model.generate_next
token_count = [0]
def traced_gen(context):
    tok = orig_gen(context)
    token_count[0] += 1
    # Print on milestones
    if tok >= 256:  # markers and special tokens
        names = {
            Token.REG_PC: 'REG_PC', Token.REG_AX: 'REG_AX',
            Token.REG_SP: 'REG_SP', Token.REG_BP: 'REG_BP',
            Token.STEP_END: 'STEP_END', Token.HALT: 'HALT',
        }
        name = names.get(tok, f'TOKEN_{tok}')
        print(f"  [gen] #{token_count[0]:3d} = {name} ({tok})")
    return tok
model.generate_next = traced_gen

runner = AutoregressiveVMRunner()
runner.model = model

# Trace _override_register_in_last_step
orig_override = runner._override_register_in_last_step
def traced_override(context, marker, value):
    names = {Token.REG_PC: 'PC', Token.REG_AX: 'AX', Token.REG_SP: 'SP', Token.REG_BP: 'BP'}
    name = names.get(marker, f'REG_{marker}')
    print(f"  [override] {name} = {value} (0x{value:08X})")
    orig_override(context, marker, value)
runner._override_register_in_last_step = traced_override

# Trace _extract_register
orig_extract = runner._extract_register
def traced_extract(context, marker):
    val = orig_extract(context, marker)
    names = {Token.REG_PC: 'PC', Token.REG_AX: 'AX', Token.REG_SP: 'SP', Token.REG_BP: 'BP'}
    name = names.get(marker, f'REG_{marker}')
    if val is not None:
        print(f"  [extract] {name} = {val} (0x{val:08X})")
    return val
runner._extract_register = traced_extract

print("=== Running IMM 42; EXIT with MoE ===")
output, exit_code = runner.run(bytecode, max_steps=5)
print(f"\nFinal: exit_code={exit_code}")
