#!/usr/bin/env python3
"""Debug the context structure to understand step boundaries."""

from neural_vm.vm_step import AutoregressiveVM, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

TOKEN_NAMES = {
    257: "PC",
    258: "AX",
    259: "SP",
    260: "BP",
    261: "MEM",
    262: "STEP_END",
    263: "HALT",
    268: "STACK0",
}

bytecode = [
    Opcode.JSR | (26 << 8),  # Instr 0: Jump to PC=26
    Opcode.EXIT,              # Instr 1: Never reached
    Opcode.NOP,               # Instr 2: Padding
    Opcode.IMM | (42 << 8),   # Instr 3: Target at PC=26
    Opcode.EXIT,              # Instr 4: Exit
]

runner = AutoregressiveVMRunner()
model = runner.model

# Build initial context
context = runner._build_context(bytecode, b"", [], "")
print(f"Initial context length: {len(context)}")
print("\n=== Initial context tokens ===")
for i, t in enumerate(context):
    name = TOKEN_NAMES.get(t, str(t))
    print(f"  {i:3d}: {t:3d} ({name})")

# Generate tokens and show full context
print("\n=== Generating step 1 ===")
step1_count = 0
for _ in range(50):  # Generate more tokens
    token = model.generate_next(context)
    context.append(token)
    step1_count += 1
    name = TOKEN_NAMES.get(token, str(token))
    print(f"  Gen {step1_count} -> idx {len(context)-1}: {token:3d} ({name})")
    if token == Token.STEP_END:
        print("  >>> STEP 1 ENDED <<<")
        break
    if token == Token.HALT:
        print("  >>> HALTED <<<")
        break

# Generate step 2
print("\n=== Generating step 2 ===")
step2_count = 0
for _ in range(50):
    token = model.generate_next(context)
    context.append(token)
    step2_count += 1
    name = TOKEN_NAMES.get(token, str(token))
    print(f"  Gen {step2_count} -> idx {len(context)-1}: {token:3d} ({name})")
    if token == Token.STEP_END:
        print("  >>> STEP 2 ENDED <<<")
        break
    if token == Token.HALT:
        print("  >>> HALTED <<<")
        break

# Parse results
print("\n=== Parsing results ===")
i = 0
step = 0
while i < len(context):
    t = context[i]
    if t == Token.STEP_END:
        step += 1
        print(f"\n--- End of step {step} at idx {i} ---")
        i += 1
    elif t == Token.REG_PC:
        if i + 4 < len(context):
            pc_bytes = context[i+1:i+5]
            pc_val = sum(b << (j*8) for j, b in enumerate(pc_bytes))
            print(f"PC at idx {i}: {pc_val} (bytes: {list(pc_bytes)})")
        i += 5
    elif t == Token.REG_AX:
        if i + 4 < len(context):
            ax_bytes = context[i+1:i+5]
            ax_val = sum(b << (j*8) for j, b in enumerate(ax_bytes))
            print(f"AX at idx {i}: {ax_val} (bytes: {list(ax_bytes)})")
        i += 5
    elif t == Token.REG_SP:
        if i + 4 < len(context):
            sp_bytes = context[i+1:i+5]
            sp_val = sum(b << (j*8) for j, b in enumerate(sp_bytes))
            print(f"SP at idx {i}: {sp_val} (bytes: {list(sp_bytes)})")
        i += 5
    elif t == Token.REG_BP:
        if i + 4 < len(context):
            bp_bytes = context[i+1:i+5]
            bp_val = sum(b << (j*8) for j, b in enumerate(bp_bytes))
            print(f"BP at idx {i}: {bp_val} (bytes: {list(bp_bytes)})")
        i += 5
    elif t == 268:  # STACK0
        if i + 4 < len(context):
            s0_bytes = context[i+1:i+5]
            s0_val = sum(b << (j*8) for j, b in enumerate(s0_bytes))
            print(f"STACK0 at idx {i}: {s0_val} (bytes: {list(s0_bytes)})")
        i += 5
    elif t == 261:  # MEM
        if i + 8 < len(context):
            addr_bytes = context[i+1:i+5]
            val_bytes = context[i+5:i+9]
            addr = sum(b << (j*8) for j, b in enumerate(addr_bytes))
            val = sum(b << (j*8) for j, b in enumerate(val_bytes))
            print(f"MEM at idx {i}: addr={addr}, val={val}")
        i += 9
    else:
        i += 1
