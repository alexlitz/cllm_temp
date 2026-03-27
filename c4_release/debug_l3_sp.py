"""Debug Layer 3 FFN at SP_b1."""
import torch
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

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context + draft_tokens[:13]  # Up to SP_b1
ctx_tensor = torch.tensor([current_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    sp_b1_pos = len(current_context) - 1

    print(f"=== BEFORE LAYER 3 at SP_b1 (pos {sp_b1_pos}) ===")
    print(f"H1[SP]: {x[0, sp_b1_pos, BD.H1 + 2].item():.3f}")
    print(f"BYTE_INDEX_1: {x[0, sp_b1_pos, BD.BYTE_INDEX_1].item():.3f}")
    print(f"HAS_SE: {x[0, sp_b1_pos, BD.HAS_SE].item():.3f}")

    # Run through Layer 3
    for layer_idx in range(4):
        x = model.blocks[layer_idx](x)

    print(f"\n=== AFTER LAYER 3 at SP_b1 ===")
    print(f"OUTPUT_LO values:")
    for k in range(16):
        val = x[0, sp_b1_pos, BD.OUTPUT_LO + k].item()
        if abs(val) > 0.01:
            print(f"  OUTPUT_LO[{k}] = {val:.3f}")

    # Check FFN weights
    ffn = model.blocks[3].ffn
    print(f"\n=== LAYER 3 FFN WEIGHTS (checking SP init unit) ===")
    # The SP init unit should be around unit 4-5 (after PC init units)
    # Let me search for units that write to OUTPUT_LO[1]
    for unit in range(100):
        w_down = ffn.W_down[BD.OUTPUT_LO + 1, unit].item()
        if abs(w_down) > 0.01:
            print(f"\nUnit {unit} writes to OUTPUT_LO[1] with W_down={w_down:.3f}")
            print(f"  W_up[H1+2]: {ffn.W_up[unit, BD.H1 + 2].item():.3f}")
            print(f"  W_up[BYTE_INDEX_1]: {ffn.W_up[unit, BD.BYTE_INDEX_1].item():.3f}")
            print(f"  b_up: {ffn.b_up[unit].item():.3f}")
            print(f"  b_gate: {ffn.b_gate[unit].item():.3f}")

            # Compute unit activation manually
            up_val = (x[0, sp_b1_pos, BD.H1 + 2].item() * ffn.W_up[unit, BD.H1 + 2].item() +
                      x[0, sp_b1_pos, BD.BYTE_INDEX_1].item() * ffn.W_up[unit, BD.BYTE_INDEX_1].item() +
                      ffn.b_up[unit].item())
            gate_val = ffn.b_gate[unit].item()
            print(f"  Up value: {up_val:.3f}, Gate value: {gate_val:.3f}")
            print(f"  Activation: silu({up_val}) * {gate_val}")
