"""Debug which Layer 6 FFN units are activating at position 29."""
import sys
if 'neural_vm.vm_step' in sys.modules:
    del sys.modules['neural_vm.vm_step']

import torch
import torch.nn.functional as F
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context[:]
for i in range(21):
    current_context.append(draft_tokens[i])

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
pos = 29  # position of token 21

# Hook to capture Layer 6 FFN intermediate states
ffn_input = None
ffn_up = None
ffn_gate = None
ffn_hidden = None

def hook_ffn(module, input, output):
    global ffn_input, ffn_up, ffn_gate, ffn_hidden
    ffn_input = input[0].detach().clone()

    # Manually compute FFN internals
    x = input[0]
    up = F.linear(x, module.W_up, module.b_up)
    gate = F.linear(x, module.W_gate, module.b_gate)
    hidden = F.silu(up) * gate

    ffn_up = up.detach().clone()
    ffn_gate = gate.detach().clone()
    ffn_hidden = hidden.detach().clone()

model.blocks[6].ffn.register_forward_hook(hook_ffn)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("Layer 6 FFN Analysis at Position 29 (Token 21)")
print("=" * 70)
print()

# Check key input dimensions
if ffn_input is not None:
    print("Input flags at position 29:")
    print(f"  MARK_STACK0: {ffn_input[0, pos, BD.MARK_STACK0]:.3f}")
    print(f"  MARK_SP:     {ffn_input[0, pos, BD.MARK_SP]:.3f}")
    print(f"  MARK_BP:     {ffn_input[0, pos, BD.MARK_BP]:.3f}")
    print(f"  MARK_AX:     {ffn_input[0, pos, BD.MARK_AX]:.3f}")
    print(f"  IS_BYTE:     {ffn_input[0, pos, BD.IS_BYTE]:.3f}")
    print(f"  OP_JMP:      {ffn_input[0, pos, BD.OP_JMP]:.3f}")
    print(f"  OP_NOP:      {ffn_input[0, pos, BD.OP_NOP]:.3f}")
    print(f"  OP_EXIT:     {ffn_input[0, pos, BD.OP_EXIT]:.3f}")
    print(f"  OP_PSH:      {ffn_input[0, pos, BD.OP_PSH]:.3f}")
    print(f"  CMP[0]:      {ffn_input[0, pos, BD.CMP + 0]:.3f}")
    print(f"  CMP[1]:      {ffn_input[0, pos, BD.CMP + 1]:.3f}")
    print()

# Find which hidden units are contributing to OUTPUT_HI[15]
if ffn_hidden is not None:
    ffn = model.blocks[6].ffn
    W_down = ffn.W_down.data

    # Get units that write to OUTPUT_HI[15]
    writers = torch.abs(W_down[BD.OUTPUT_HI + 15, :])  # absolute contribution weights
    hidden_vals = ffn_hidden[0, pos, :]

    # Find top contributing units
    contributions = writers * torch.abs(hidden_vals)
    top_units = torch.argsort(contributions, descending=True)[:10]

    print("Top 10 units contributing to OUTPUT_HI[15] at position 29:")
    print()
    for i, unit in enumerate(top_units):
        unit = unit.item()
        w = W_down[BD.OUTPUT_HI + 15, unit].item()
        h = hidden_vals[unit].item()
        contrib = w * h

        print(f"  Unit {unit:4d}: W_down={w:7.3f}, hidden={h:7.3f}, contrib={contrib:7.3f}")

        # Check up value for this unit
        if ffn_up is not None:
            up_val = ffn_up[0, pos, unit].item()
            gate_val = ffn_gate[0, pos, unit].item()
            print(f"              up={up_val:7.3f}, gate={gate_val:7.3f}")

        print()
