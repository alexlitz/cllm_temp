#!/usr/bin/env python3
import torch
from neural_vm.run_vm import AutoregressiveVMRunner

runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
model = runner.model
print(f"Total blocks: {len(model.blocks)}", flush=True)

# Look at FFN sizes (already in init output) but also non-zero attn dims per block
for i in range(len(model.blocks)):
    attn = model.blocks[i].attn
    W_q = attn.W_q.to_dense() if attn.W_q.is_sparse else attn.W_q
    nz_q = (W_q.abs() > 0.01).sum().item()
    # Find non-zero key dims
    W_k = attn.W_k.to_dense() if attn.W_k.is_sparse else attn.W_k
    nz_k = (W_k.abs() > 0.01).sum().item()
    print(f"Block {i}: attn nz_q={nz_q}, nz_k={nz_k}, num_heads={attn.num_heads}, alibi_slopes={attn.alibi_slopes.tolist() if attn.alibi_slopes is not None else None}", flush=True)
