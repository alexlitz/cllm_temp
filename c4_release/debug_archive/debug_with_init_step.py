#!/usr/bin/env python3
"""Debug WITH initial step to see if PC=2 is carried forward."""
import sys
sys.path.insert(0, '.')
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

def _le_bytes(val):
    v = val & 0xFFFFFFFF
    return [(v >> (i * 8)) & 0xFF for i in range(4)]

def build_context_with_init(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        opcode = instr & 0xFF
        imm = (instr >> 8) & 0xFFFFFFFF
        context.append(opcode)
        for i in range(4):
            context.append((imm >> (i * 8)) & 0xFF)
        context.extend([0, 0, 0])
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.append(Token.DATA_END)

    # ADD INITIAL STEP with PC=2
    context.append(Token.REG_PC)
    context.extend(_le_bytes(2))
    context.append(Token.REG_AX)
    context.extend(_le_bytes(0))
    context.append(Token.REG_SP)
    context.extend(_le_bytes(0))
    context.append(Token.REG_BP)
    context.extend(_le_bytes(0))
    context.append(Token.STACK0)
    context.extend(_le_bytes(0))
    context.append(Token.MEM)
    context.extend(_le_bytes(0))
    context.extend(_le_bytes(0))
    context.append(Token.STEP_END)
    return context

def main():
    BD = _SetDim
    print('Building model...')
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
    context = build_context_with_init(bytecode)

    draft = DraftVM(bytecode)
    draft.step()
    draft_tokens = draft.draft_tokens()

    print(f'Context WITH initial step: {len(context)} tokens')
    print(f'Initial step PC bytes at positions 21-24: {context[21:25]}')
    print(f'Draft step 0 PC: {draft_tokens[1]} (should be 7)')
    print(f'Draft step 0 AX: {draft_tokens[6]} (should be 42)')

    full_context = context + draft_tokens

    with torch.no_grad():
        x = torch.tensor([full_context], dtype=torch.long)
        h = model.embed(x)
        model._inject_code_addr_keys(x, h)
        model._inject_mem_store(x, h)

        ax_marker_pos = len(context) + 5
        pc_marker_init = 20  # PC marker of initial step
        pc_b0_init = 21  # PC byte 0 of initial step
        pc_marker_draft = len(context)  # PC marker of draft step 0

        # Check initial step embedding
        print(f'\\n=== Initial step PC at positions {pc_marker_init}-{pc_b0_init} ===')
        print(f'PC marker (pos {pc_marker_init}): token={x[0, pc_marker_init].item()}')
        print(f'PC byte 0 (pos {pc_b0_init}): token={x[0, pc_b0_init].item()}')
        print(f'EMBED_LO at PC byte 0: {h[0, pc_b0_init, BD.EMBED_LO:BD.EMBED_LO+4].tolist()}')

        # L0-L1
        for i in range(2):
            h = model.blocks[i](h)
        print(f'\\n=== After L1 at PC byte 0 of init step (pos {pc_b0_init}) ===')
        print(f'L1H1[PC_I=0]: {h[0, pc_b0_init, BD.L1H1]:.4f}')
        print(f'L1H0[PC_I=0]: {h[0, pc_b0_init, BD.L1H0]:.4f}')
        print(f'Expected: L1H1=1 (d=1 ≤ 1.5), L1H0=0 (d=1 > 0.5)')

        # L2
        h = model.blocks[2](h)
        print(f'\\n=== After L2 at PC marker of draft step (pos {pc_marker_draft}) ===')
        print(f'MARK_PC: {h[0, pc_marker_draft, BD.MARK_PC]:.4f}')
        print(f'EMBED_LO: {h[0, pc_marker_draft, BD.EMBED_LO:BD.EMBED_LO+4].tolist()}')

        # L3
        h = model.blocks[3](h)
        print(f'\\n=== After L3 (PC carry+increment) at PC marker of draft (pos {pc_marker_draft}) ===')
        print(f'HAS_SE: {h[0, pc_marker_draft, BD.HAS_SE]:.4f}')
        print(f'EMBED_LO: {h[0, pc_marker_draft, BD.EMBED_LO:BD.EMBED_LO+16].tolist()}')
        print(f'OUTPUT_LO (all 16): {h[0, pc_marker_draft, BD.OUTPUT_LO:BD.OUTPUT_LO+16].tolist()}')
        print(f'Expected EMBED_LO[2]=1 (PC=2 from init step)')
        print(f'Expected OUTPUT_LO[7]=1 (PC+5=7)')

        print(f'\\n=== After L3 at AX marker pos {ax_marker_pos} ===')
        print(f'AX_CARRY_LO: {h[0, ax_marker_pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+4].tolist()}')
        print(f'Expected: [0, 0, 1.0, 0] relayed from EMBED at PC marker')

        # L4
        h = model.blocks[4](h)
        print(f'\\n=== After L4 (PC+1 relay) at AX marker pos {ax_marker_pos} ===')
        print(f'TEMP: {h[0, ax_marker_pos, BD.TEMP:BD.TEMP+4].tolist()}')
        print(f'Expected: [3, 0, 0, 0] or [0,0,1,0,...] for PC+1=3')

        # L5
        h = model.blocks[5](h)
        print(f'\\n=== After L5 (fetch+decode) at AX marker pos {ax_marker_pos} ===')
        print(f'OPCODE_BYTE_LO: {h[0, ax_marker_pos, BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+4].tolist()}')
        print(f'FETCH_LO: {h[0, ax_marker_pos, BD.FETCH_LO:BD.FETCH_LO+4].tolist()}')
        print(f'OP_IMM: {h[0, ax_marker_pos, BD.OP_IMM]:.4f}')
        print(f'Expected FETCH_LO: [0,0,1,0,...] for immediate value 42 at address 3')

        # Check what's at address 3 in CODE
        print(f'\\n=== CODE section byte at address 3 (pos 4) ===')
        print(f'Token: {x[0, 4].item()} (should be 0, imm byte 2)')
        print(f'ADDR_KEY: {h[0, 4, BD.ADDR_KEY:BD.ADDR_KEY+16].tolist()[:4]}')

if __name__ == "__main__":
    main()
