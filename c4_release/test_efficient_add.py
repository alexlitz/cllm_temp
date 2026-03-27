"""Test efficient ADD parameter count."""
import torch
import torch.nn as nn
from neural_vm.vm_step import _SetDim as BD
from neural_vm.efficient_add import set_efficient_add_stage1, set_efficient_add_stage2

S = 100.0

class DummyFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_up = nn.Parameter(torch.zeros(1000, 512))
        self.b_up = nn.Parameter(torch.zeros(1000))
        self.W_gate = nn.Parameter(torch.zeros(1000, 512))
        self.b_gate = nn.Parameter(torch.zeros(1000))
        self.W_down = nn.Parameter(torch.zeros(512, 1000))

ffn1 = DummyFFN()
ffn2 = DummyFFN()

units1 = set_efficient_add_stage1(ffn1, S, BD)
units2 = set_efficient_add_stage2(ffn2, S, BD)

print(f"\nTotal: {units1 + units2} units for ADD")
print(f"Current: 752 units")
print(f"Savings: {752 - (units1 + units2)} units ({100*(752-(units1+units2))/752:.1f}%)")
