"""
Debug why head predicts byte_01 when OUTPUT_LO[10]=6, OUTPUT_HI[2]=6 (should be 0x2a).
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

# Generate first 6 tokens to get to REG_AX
for _ in range(6):
    tok = model.generate_next(context)
    context.append(tok)

# Forward pass
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    pos = len(context) - 1  # REG_AX marker position

    # Full forward pass
    for i, block in enumerate(model.blocks):
        x = block(x, kv_cache=None)

    print(f"At position {pos} (REG_AX marker):")

    # Check IS_BYTE value
    is_byte = x[0, pos, BD.IS_BYTE].item()
    mark_ax = x[0, pos, BD.MARK_AX].item()
    mark_pc = x[0, pos, BD.MARK_PC].item()
    op_imm = x[0, pos, BD.OP_IMM].item()
    print(f"  IS_BYTE: {is_byte:.3f}")
    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  MARK_PC: {mark_pc:.3f}")
    print(f"  OP_IMM: {op_imm:.3f}")

    # Compute expected activation with current guards
    S = 16.0
    T = 6.0
    activation = S * op_imm + S * mark_ax - S * mark_pc + S * is_byte - S * T
    print(f"\n  Expected IMM unit activation:")
    print(f"    {S}*{op_imm:.1f} + {S}*{mark_ax:.1f} - {S}*{mark_pc:.1f} + {S}*{is_byte:.1f} - {S}*{T}")
    print(f"    = {activation:.3f}")
    print(f"    Should fire: {activation > 0}")

    # Check all OUTPUT dimensions
    print(f"\nAll OUTPUT dimensions:")
    print(f"  OUTPUT_LO:")
    for k in range(16):
        val = x[0, pos, BD.OUTPUT_LO + k].item()
        if abs(val) > 0.5:
            print(f"    [{k}]: {val:.3f}", end='')
            if k == 10:
                print(" ← nibble A (correct!)")
            else:
                print()

    print(f"  OUTPUT_HI:")
    for k in range(16):
        val = x[0, pos, BD.OUTPUT_HI + k].item()
        if abs(val) > 0.5:
            print(f"    [{k}]: {val:.3f}", end='')
            if k == 2:
                print(" ← nibble 2 (correct!)")
            else:
                print()

    # Predict
    logits = model.head(x)
    probs = torch.softmax(logits[0, -1, :], dim=-1)

    # Top 10 predictions
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nTop {top_k} predictions:")
    for i in range(top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        if idx < 256:
            marker = " ← CORRECT!" if idx == 0x2a else ""
            print(f"  {i+1}. byte_{idx:02x} (prob={prob:.4f}){marker}")
        else:
            for name, val in vars(Token).items():
                if isinstance(val, int) and val == idx and name.isupper():
                    print(f"  {i+1}. {name} (prob={prob:.4f})")
                    break

    # Check what byte_2a and byte_01 logits are
    logit_2a = logits[0, -1, 0x2a].item()
    logit_01 = logits[0, -1, 0x01].item()

    print(f"\nLogit comparison:")
    print(f"  byte_2a (correct): {logit_2a:.3f}")
    print(f"  byte_01 (predicted): {logit_01:.3f}")
    print(f"  Difference: {logit_2a - logit_01:.3f}")

    # Check if there are other dimensions with high values
    print(f"\nAll dimensions with |value| > 1.0:")
    for dim_name in dir(BD):
        if dim_name.isupper() and not dim_name.startswith('_'):
            try:
                dim_val = getattr(BD, dim_name)
                if isinstance(dim_val, int) and 0 <= dim_val < x.shape[-1]:
                    val = x[0, pos, dim_val].item()
                    if abs(val) > 1.0:
                        print(f"  {dim_name} (dim {dim_val}): {val:.3f}")
            except:
                pass
