#!/usr/bin/env python3
"""Trace OUTPUT_LO[1] through layers to find its source."""
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
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

# Generate up to position 37 (where we predict byte 2)
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if len(context) == 38:
        break

pred_pos = 37  # Position where we're predicting byte 2

print(f"Tracing OUTPUT_LO[1] at position {pred_pos}")
print(f"This position is predicting AX byte 2")
print(f"Context at this position: ...{context[pred_pos-5:pred_pos+1]}")
print()

token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    print("After Embedding:")
    print(f"  OUTPUT_LO[1]: {x[0, pred_pos, BD.OUTPUT_LO + 1].item():.6f}")

    # Trace through each layer
    for layer_idx in range(16):
        x_before = x.clone()
        x = model.blocks[layer_idx](x, kv_cache=None)

        output_lo_before = x_before[0, pred_pos, BD.OUTPUT_LO + 1].item()
        output_lo_after = x[0, pred_pos, BD.OUTPUT_LO + 1].item()

        if abs(output_lo_after - output_lo_before) > 0.01:
            print(f"\nLayer {layer_idx}:")
            print(f"  OUTPUT_LO[1] before: {output_lo_before:.6f}")
            print(f"  OUTPUT_LO[1] after:  {output_lo_after:.6f}")
            print(f"  Change: {output_lo_after - output_lo_before:+.6f}")

            # Check what else changed to understand the source
            block = model.blocks[layer_idx]

            # Check if it was attention or FFN
            x_attn = block.attn(block.attn_norm(x_before))
            x_after_attn = x_before + x_attn

            output_lo_after_attn = x_after_attn[0, pred_pos, BD.OUTPUT_LO + 1].item()

            if abs(output_lo_after_attn - output_lo_before) > 0.01:
                print(f"  → Changed in ATTENTION")
                print(f"    After attention: {output_lo_after_attn:.6f}")

                # Check attention output
                print(f"    Attention output at OUTPUT_LO[1]: {x_attn[0, pred_pos, BD.OUTPUT_LO + 1].item():.6f}")

            else:
                print(f"  → Changed in FFN")

                # Check which FFN wrote to OUTPUT_LO[1]
                ffn = block.ffn
                x_ffn_norm = ffn.norm(x_after_attn)
                up = torch.matmul(x_ffn_norm, ffn.W_up.T) + ffn.b_up
                gate = torch.matmul(x_ffn_norm, ffn.W_gate.T) + ffn.b_gate
                hidden = torch.nn.functional.silu(up) * gate

                # Find units that write to OUTPUT_LO[1]
                writers = (ffn.W_down[BD.OUTPUT_LO + 1, :] != 0).nonzero(as_tuple=True)[0]
                if len(writers) > 0:
                    print(f"    FFN units writing to OUTPUT_LO[1]: {writers.tolist()[:10]}")

                    # Check top activated units
                    for unit in writers[:5]:
                        act = hidden[0, pred_pos, unit].item()
                        weight = ffn.W_down[BD.OUTPUT_LO + 1, unit].item()
                        contrib = act * weight
                        if abs(contrib) > 0.01:
                            print(f"      Unit {unit}: act={act:.3f}, weight={weight:.3f}, contrib={contrib:.6f}")

    print(f"\nFinal OUTPUT_LO[1]: {x[0, pred_pos, BD.OUTPUT_LO + 1].item():.6f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("This traces where OUTPUT_LO[1] = 0.940 comes from, which causes")
print("the prediction of byte 0x01 at AX byte 2 position.")
