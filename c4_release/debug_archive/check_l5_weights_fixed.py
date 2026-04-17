#!/usr/bin/env python3
"""Check if L5 head 3 weights are set - using correct layer index."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD
from neural_vm.constants import PC_OFFSET

runner = AutoregressiveVMRunner()
# L5 is at index 5, not 4!
L5_attn = runner.model.blocks[5].attn

HD = 64
head = 3
base = head * HD
L = 20.0

print(f"Checking L5 (blocks[5]) Head 3 weights:")

imm_addr = PC_OFFSET + 1  # = 3
print(f"\nW_q weights (should query for addr={imm_addr}):")
print(f"  W_q[{base + (imm_addr & 0xF)}, BD.CONST={BD.CONST}]: {L5_attn.W_q[base + (imm_addr & 0xF), BD.CONST].item():.2f} (expected {L})")
print(f"  W_q[{base + 16 + ((imm_addr >> 4) & 0xF)}, BD.CONST={BD.CONST}]: {L5_attn.W_q[base + 16 + ((imm_addr >> 4) & 0xF), BD.CONST].item():.2f} (expected {L})")
print(f"  W_q[{base + 32}, BD.MARK_PC={BD.MARK_PC}]: {L5_attn.W_q[base + 32, BD.MARK_PC].item():.2f} (expected {L})")

print(f"\nW_k weights (should match ADDR_KEY):")
for k in range(4):
    print(f"  W_k[{base + k}, BD.ADDR_KEY+{k}={BD.ADDR_KEY + k}]: {L5_attn.W_k[base + k, BD.ADDR_KEY + k].item():.2f} (expected {L})")

print(f"\nW_v weights (should copy CLEAN_EMBED):")
for k in range(4):
    print(f"  W_v[{base + 32 + k}, BD.CLEAN_EMBED_LO+{k}={BD.CLEAN_EMBED_LO + k}]: {L5_attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k].item():.2f} (expected 1.0)")

print(f"\nW_o weights (should write to FETCH):")
for k in range(4):
    print(f"  W_o[BD.FETCH_LO+{k}={BD.FETCH_LO + k}, {base + 32 + k}]: {L5_attn.W_o[BD.FETCH_LO + k, base + 32 + k].item():.2f} (expected 40.0)")
