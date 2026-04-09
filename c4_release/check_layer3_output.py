#!/usr/bin/env python3
"""Check Layer 3 FFN units writing to OUTPUT_LO[1]."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

# Generate up to position 37
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if len(context) == 38:
        break

pred_pos = 37

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(3):
        x = model.blocks[i](x, kv_cache=None)

    # x is now after Layer 2
    x_before_l3 = x.clone()

    # Apply Layer 3
    x_after_l3 = model.blocks[3](x, kv_cache=None)

    print(f'Layer 3 at position {pred_pos}:')
    print(f'  OUTPUT_LO[1] before: {x_before_l3[0, pred_pos, BD.OUTPUT_LO + 1].item():.6f}')
    print(f'  OUTPUT_LO[1] after:  {x_after_l3[0, pred_pos, BD.OUTPUT_LO + 1].item():.6f}')

    # Check Layer 3 FFN
    block3 = model.blocks[3]
    ffn = block3.ffn

    # Check which units write to OUTPUT_LO[1]
    writers = (ffn.W_down[BD.OUTPUT_LO + 1, :].abs() > 0.001).nonzero(as_tuple=True)[0]
    print(f'\nLayer 3 FFN units writing to OUTPUT_LO[1]: {len(writers.tolist())} units')

    if len(writers) > 0:
        # Get FFN activations
        x_ln = block3.ln_2(x_before_l3)
        up = torch.matmul(x_ln, ffn.W_up.T) + ffn.b_up
        gate = torch.matmul(x_ln, ffn.W_gate.T) + ffn.b_gate
        hidden = torch.nn.functional.silu(up) * gate

        print('\nTop contributing units:')
        contribs = []
        for unit in writers.tolist():
            act = hidden[0, pred_pos, unit].item()
            weight = ffn.W_down[BD.OUTPUT_LO + 1, unit].item()
            contrib = act * weight
            if abs(contrib) > 0.001:
                contribs.append((unit, act, weight, contrib))

        contribs.sort(key=lambda x: abs(x[3]), reverse=True)
        for unit, act, weight, contrib in contribs[:10]:
            print(f'  Unit {unit:4d}: act={act:8.6f}, weight={weight:8.6f}, contrib={contrib:+.6f}')

            # Check what activates this unit
            up_val = up[0, pred_pos, unit].item()
            gate_val = gate[0, pred_pos, unit].item()
            print(f'         up={up_val:.6f}, gate={gate_val:.6f}')

            # Check up and gate inputs
            up_inputs = []
            for dim in range(512):
                w = ffn.W_up[unit, dim].item()
                if abs(w) > 0.1:
                    val = x_ln[0, pred_pos, dim].item()
                    if abs(val * w) > 0.01:
                        up_inputs.append((dim, val, w, val*w))

            if up_inputs:
                up_inputs.sort(key=lambda x: abs(x[3]), reverse=True)
                print(f'         Top up inputs: {up_inputs[:3]}')

print('\n' + '=' * 80)
print('This shows which Layer 3 FFN units contribute to OUTPUT_LO[1] = 0.940')
print('at position 37 (AX byte 2).')
