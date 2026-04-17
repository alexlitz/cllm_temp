"""Debug draft tokens."""
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.vm_step import Token

bytecode = [Opcode.IMM | (42 << 8)]
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Draft tokens (35 tokens for one VM step):")
token_names = (
    ['REG_PC'] + [f'PC_b{j}' for j in range(4)] +
    ['REG_AX'] + [f'AX_b{j}' for j in range(4)] +
    ['REG_SP'] + [f'SP_b{j}' for j in range(4)] +
    ['REG_BP'] + [f'BP_b{j}' for j in range(4)] +
    ['STACK0'] + [f'ST_b{j}' for j in range(4)] +
    ['MEM'] + [f'MEM_a{j}' for j in range(4)] + [f'MEM_v{j}' for j in range(4)] +
    ['END']
)

for i in range(min(20, len(draft_tokens))):
    name = token_names[i] if i < len(token_names) else f'Token{i}'
    print(f"  {i:2d} ({name:8s}): {draft_tokens[i]:3d}")

print(f"\nChecking specific positions:")
print(f"  REG_SP should be {Token.REG_SP}")
print(f"  Position 10 (REG_SP): {draft_tokens[10]}")
print(f"  Position 11 (SP_b0): {draft_tokens[11]}")
print(f"  Position 12 (SP_b1): {draft_tokens[12]}")
print(f"  Position 13 (SP_b2): {draft_tokens[13]}")
