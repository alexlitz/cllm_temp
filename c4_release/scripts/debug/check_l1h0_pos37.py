#!/usr/bin/env python3
import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
runner._func_call_handlers = {}; runner._syscall_handlers = {}
runner._memory = {}; runner._mem_history = {}; runner._mem_access_order = []
model = runner.model
model.eval()
device = next(model.parameters()).device

prog = [(Opcode.IMM, 1), (Opcode.BZ, 4), (Opcode.IMM, 7), Opcode.EXIT]
bytecode = []
for p in prog:
    if isinstance(p, tuple): op, imm = p; bytecode.append((imm<<8)|op)
    else: bytecode.append(p)

context = runner._build_context(bytecode, b"", [], b"")
prefix_len = len(context)
sp0 = 0x10000

def append_step(ctx, pc, ax):
    ctx.append(Token.REG_PC)
    for i in range(4): ctx.append((pc >> (i*8)) & 0xFF)
    ctx.append(Token.REG_AX)
    for i in range(4): ctx.append((ax >> (i*8)) & 0xFF)
    ctx.append(Token.REG_SP)
    for i in range(4): ctx.append((sp0 >> (i*8)) & 0xFF)
    ctx.append(Token.REG_BP); ctx.extend([0]*4)
    ctx.append(Token.STACK0); ctx.extend([0]*4)
    ctx.append(Token.MEM); ctx.extend([0]*8)
    ctx.append(Token.STEP_END)

append_step(context, pc=10, ax=1)
pos = len(context)
context.append(Token.REG_PC)
print(f"Context len: {len(context)}, step1 PC marker pos: {pos}", flush=True)
print(f"prefix_len: {prefix_len}", flush=True)
print(f"Tokens around step0 PC: pos36={context[36]}, pos37={context[37]} pos38={context[38]} pos40={context[40]}", flush=True)
print(f"Token.REG_PC={Token.REG_PC}", flush=True)
token_ids = torch.tensor([context], dtype=torch.long, device=device)

with torch.no_grad():
    x0 = model.embed(token_ids, active_opcode=model._active_opcode)
    x1 = model.blocks[0](x0)
    x2 = model.blocks[1](x1)
    # After block 1, L1H0 should be set

    # Show L1H0[PC] (=BD.L1H0) and L1H1[PC] at positions around step 0
    print(f"\nAfter block 1 (L1 + L1 FFN):", flush=True)
    for p in range(35, 45):
        l1h0 = x2[0, p, BD.L1H0].item()
        l1h1 = x2[0, p, BD.L1H1].item()
        l1h2 = x2[0, p, BD.L1H2].item()
        is_mark = x2[0, p, BD.IS_MARK].item()
        is_byte = x2[0, p, BD.IS_BYTE].item()
        mark_pc = x2[0, p, BD.MARK_PC].item()
        tok = context[p]
        print(f"  pos={p} tok={tok}: L1H0[PC]={l1h0:.3f} L1H1[PC]={l1h1:.3f} L1H2[PC]={l1h2:.3f} IS_MARK={is_mark:.2f} IS_BYTE={is_byte:.2f} MARK_PC={mark_pc:.2f}", flush=True)

