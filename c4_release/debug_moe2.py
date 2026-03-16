"""Debug MoE routing: trace full runner execution step by step."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Build two identical models, one with MoE routing, one without
model_moe = AutoregressiveVM()
set_vm_weights(model_moe)
model_moe.compact(block_size=32)
model_moe.compact_moe()

model_ref = AutoregressiveVM()
set_vm_weights(model_ref)
model_ref.compact(block_size=32)
model_ref.compact_moe()

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Build identical contexts
runner = AutoregressiveVMRunner()
runner.model = model_moe
ctx_moe = runner._build_context(bytecode, b'', [])
ctx_ref = list(ctx_moe)
prefix_len = len(ctx_moe)

print(f"Prefix ({prefix_len} tokens): {ctx_moe}")
print(f"Bytecode: IMM 42 (op={bytecode[0] & 0xFF}), EXIT (op={bytecode[1] & 0xFF})")
print()

# Step 1: Generate with IMM routing for MoE model, full for ref
init_op = bytecode[0] & 0xFF  # IMM = 1
model_moe.set_active_opcode(init_op)
# model_ref stays with full weights (no set_active_opcode call)

print("=== STEP 1 (IMM 42) ===")
print(f"MoE: routed to opcode {init_op}, dim={_SetDim.opcode_dim(init_op)}")
step1_moe = []
step1_ref = []
for j in range(Token.STEP_TOKENS):
    t_moe = model_moe.generate_next(ctx_moe)
    t_ref = model_ref.generate_next(ctx_ref)
    ctx_moe.append(t_moe)
    ctx_ref.append(t_ref)
    step1_moe.append(t_moe)
    step1_ref.append(t_ref)
    if t_moe != t_ref:
        print(f"  Token {j}: MoE={t_moe} Ref={t_ref} MISMATCH")
    if t_moe in (Token.STEP_END, Token.HALT):
        break

print(f"Step 1 match: {step1_moe == step1_ref}")
print(f"Step 1 tokens: {step1_moe}")

# Context pruning (like runner does)
last_step_moe = ctx_moe[-(Token.STEP_TOKENS):]
ctx_moe[prefix_len:] = last_step_moe
last_step_ref = ctx_ref[-(Token.STEP_TOKENS):]
ctx_ref[prefix_len:] = last_step_ref
print(f"\nAfter pruning: contexts match = {ctx_moe == ctx_ref}")
print(f"Context length: {len(ctx_moe)}")

# Step 2: Switch to EXIT opcode
# In runner: _last_pc is extracted from step 1 output, then next_exec = (_last_pc + 5) // 5
# For IMM 42, output PC = 5 (next instruction). So next_exec = (5+5)//5 = 2? No...
# Wait, _last_pc after step 1 should be 5 (the PC output from IMM step is the NEXT PC = 5)
# Then _exec_pc() = _last_pc + 5 = 10. 10 // 5 = 2. But bytecode only has indices 0, 1.
# That means next_exec >= len(bytecode), so set_active_opcode(None)

# Actually let me extract PC from context
def extract_reg(ctx, marker):
    for i in range(len(ctx) - 1, -1, -1):
        if ctx[i] == marker and i + 4 < len(ctx):
            return sum(ctx[i + 1 + j] << (j * 8) for j in range(4))
    return None

pc_step1 = extract_reg(ctx_moe, Token.REG_PC)
ax_step1 = extract_reg(ctx_moe, Token.REG_AX)
print(f"\nStep 1 output: PC={pc_step1}, AX={ax_step1}")

# Runner logic: next_exec = _exec_pc() // 5
# But _exec_pc() uses _last_pc which is the OUTPUT PC from step 1
# For IMM at index 0: output PC should be 5 (PC of instruction 1 = EXIT)
# Then _exec_pc() AFTER step 1 = _last_pc + 5 = 5 + 5 = 10
# Wait no — _exec_pc() computes PC of instruction BEING EXECUTED, not next.
# After step 1: _last_pc = 5 (output PC from step 1 = PC at start of step 2)
# next_exec = _exec_pc() // 5 = (5 + 5) // 5 = 2
# Hmm, bytecode has indices 0 and 1. 2 >= len(bytecode), so set_active_opcode(None)

# BUT the EXIT instruction is at index 1, PC = 5. The runner should route to EXIT.
# The issue: _exec_pc() adds 5 to _last_pc, but _last_pc is already the NEXT PC (5).
# So _exec_pc() returns 10, which points past the program.

# Let me check: what PC does the model output for IMM step?
# In the encoding: PC = 5*I + 2 for instruction I. But the OUTPUT PC should be
# the PC register value after executing IMM, which is just the next sequential PC.

# Actually wait, let me re-read _exec_pc:
# "Adding 5 (instruction width) gives the PC of the instruction that just executed"
# So if _last_pc from step 1 output = 5 (PC of EXIT instruction = 5*1+0 = 5)
# Then for STEP_END at line 367: next_exec = self._exec_pc() // 5
# _exec_pc() = 5 + 5 = 10, 10 // 5 = 2
# This is WRONG for routing to the EXIT step.

# Actually wait — _exec_pc() is meant to compute the EXECUTING instruction for the
# CURRENT step. After step 1 completes, _last_pc is set to 5 (from step 1 output).
# For step 2's routing, we need the instruction AT PC=5, which is EXIT at index 1.
# But next_exec at STEP_END (line 367) uses _exec_pc() which returns _last_pc + 5 = 10.
# This is the PC of the instruction EXECUTING in step 2... wait no.

# Let me re-read the docstring:
# "_last_pc from previous step gives where we WERE. Adding 5 gives PC of instruction
# that just executed in this step"
# So at the END of step 1, _last_pc is set to 5 (output of step 1).
# At STEP_END, we compute NEXT step's opcode:
# next_exec = _exec_pc() // 5 = (5 + 5) // 5 = 2
# But the NEXT step executes EXIT at index 1 (PC=5, not PC=10).
# This is a bug! _exec_pc() is designed for figuring out what was JUST executed,
# not what WILL BE executed next.

# For NEXT step routing, we should use _last_pc (the output PC from this step)
# directly, not _exec_pc() which adds 5.

print(f"\n_exec_pc logic: _last_pc={pc_step1}, _exec_pc={pc_step1 + 5 if pc_step1 else 0}")
print(f"next_exec = {(pc_step1 + 5) // 5 if pc_step1 else 0}")
print(f"bytecode length = {len(bytecode)}")

# The next step should execute EXIT at index 1 (PC=5)
# But next_exec = (5+5)//5 = 2 which is out of range
# So set_active_opcode(None) is called — uses full matrices
# That should be equivalent to no MoE routing...

# Let me check: maybe the issue is earlier. Let me look at the INITIAL routing.
# At line 268-270: init_exec = self._exec_pc() // 5
# _last_pc is None, so _exec_pc() returns 0
# init_exec = 0 // 5 = 0
# bytecode[0] & 0xFF = IMM = 1
# So initial routing is correct: set_active_opcode(IMM)

# After step 1: next_exec = _exec_pc() // 5
# _last_pc = 5 (extracted from step 1 output)
# _exec_pc() = 5 + 5 = 10
# next_exec = 2
# 2 >= len(bytecode)=2 -> set_active_opcode(None)
# This uses _full matrices. So step 2 should use full weights = same as no MoE.

# Hmm, so the routing for step 2 should be fine (full weights).
# The issue might be in step 1 then — IMM routing changes the output.

# Let me actually set MoE to EXIT and compare
print("\n=== Testing with EXIT routing for step 2 ===")
model_moe.set_active_opcode(None)  # Full (what runner would do)
step2_moe = []
for j in range(Token.STEP_TOKENS):
    t = model_moe.generate_next(ctx_moe)
    ctx_moe.append(t)
    step2_moe.append(t)
    if t in (Token.STEP_END, Token.HALT):
        break

step2_ref = []
for j in range(Token.STEP_TOKENS):
    t = model_ref.generate_next(ctx_ref)
    ctx_ref.append(t)
    step2_ref.append(t)
    if t in (Token.STEP_END, Token.HALT):
        break

print(f"Step 2 match: {step2_moe == step2_ref}")
if step2_moe != step2_ref:
    for j in range(min(len(step2_moe), len(step2_ref))):
        if step2_moe[j] != step2_ref[j]:
            print(f"  Token {j}: MoE={step2_moe[j]} Ref={step2_ref[j]}")

ax_moe = extract_reg(ctx_moe, Token.REG_AX)
ax_ref = extract_reg(ctx_ref, Token.REG_AX)
print(f"Final AX: MoE={ax_moe}, Ref={ax_ref}")
print(f"Step 2 tokens: {step2_moe}")
