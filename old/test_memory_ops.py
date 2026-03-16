"""Debug LI/SI memory operations."""

import torch
from c4_vm import Op
from c4_printf_autoregressive import C4AutoregressivePrintf


def instr(op, imm=0):
    return float(op + (imm << 8))


def test_memory():
    print("MEMORY OPERATIONS TEST")
    print("=" * 60)
    print()

    executor = C4AutoregressivePrintf(memory_size=256)

    # Test 1: Store and load
    print("Test 1: SI then LI")
    # Store 42 at address 100, then load it back
    memory = torch.zeros(256)
    code = [
        instr(Op.IMM, 100),    # 0: ax = 100 (address)
        instr(Op.PSH),         # 1: push address
        instr(Op.IMM, 42),     # 2: ax = 42 (value)
        instr(Op.SI),          # 3: mem[100] = 42
        instr(Op.IMM, 100),    # 4: ax = 100
        instr(Op.LI),          # 5: ax = mem[100]
        instr(Op.EXIT),        # 6
    ]
    for i, c in enumerate(code):
        memory[i * 8] = c

    pc = torch.tensor(0.0)
    sp = torch.tensor(200.0)
    bp = torch.tensor(200.0)
    ax = torch.tensor(0.0)

    print("  Tracing execution:")
    for step in range(10):
        if executor.is_exit(memory, pc):
            break
        ctx = executor.ctx.build_context(memory, pc, sp, bp, ax)
        instr_val = executor.ctx.attend(ctx, pc, query_type='memory')
        opcode = instr_val - torch.floor(instr_val / 256) * 256
        imm = torch.floor(instr_val / 256)
        print(f"    Step {step}: PC={int(pc.item())}, op={int(opcode.item())}, imm={int(imm.item())}, AX={int(ax.item())}, mem[100]={int(memory[100].item())}")
        pc, sp, bp, ax, memory, output, out_ptr = executor.step(pc, sp, bp, ax, memory, torch.zeros(64), torch.tensor(0.0))

    print(f"  Final AX: {int(ax.item())} (expected 42)")
    print(f"  mem[100]: {int(memory[100].item())}")
    status = "✓" if int(ax.item()) == 42 else "✗"
    print(f"  {status} LI loaded stored value")
    print()

    # Test 2: Simple variable increment
    print("Test 2: Increment variable at address 100")
    memory2 = torch.zeros(256)
    # mem[100] = 5; mem[100] = mem[100] + 1; return mem[100]
    code2 = [
        # mem[100] = 5
        instr(Op.IMM, 100),    # ax = 100
        instr(Op.PSH),         # push &x
        instr(Op.IMM, 5),      # ax = 5
        instr(Op.SI),          # mem[100] = 5

        # ax = mem[100] + 1
        instr(Op.IMM, 100),    # ax = 100
        instr(Op.LI),          # ax = mem[100] = 5
        instr(Op.PSH),         # push 5
        instr(Op.IMM, 1),      # ax = 1
        instr(Op.ADD),         # ax = 6

        # mem[100] = ax
        instr(Op.PSH),         # push 6
        instr(Op.IMM, 100),    # ax = 100
        instr(Op.PSH),         # push &x
        # Wait, SI pops address and stores ax... let me re-read

        # SI: *(int*)*sp++ = ax
        # So we need: push address, then have value in ax
        # Actually: SI pops address from stack, stores ax there

        instr(Op.EXIT),
    ]

    # Let me redo this more carefully
    print("  Let me trace SI semantics...")

    # SI: pop address, store AX there
    # So: push address, have value in AX, then SI
    memory3 = torch.zeros(256)
    code3 = [
        # mem[100] = 42
        instr(Op.IMM, 100),    # 0: ax = 100
        instr(Op.PSH),         # 1: push 100 (address)
        instr(Op.IMM, 42),     # 2: ax = 42
        instr(Op.SI),          # 3: mem[pop()] = ax -> mem[100] = 42, sp += 8

        # Check mem[100]
        instr(Op.IMM, 100),    # 4: ax = 100
        instr(Op.LI),          # 5: ax = mem[100]
        instr(Op.EXIT),        # 6
    ]
    for i, c in enumerate(code3):
        memory3[i * 8] = c

    pc, sp, bp, ax, memory3, output = executor.run(
        memory3, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0), max_steps=20
    )
    print(f"  After SI and LI: AX = {int(ax.item())} (expected 42)")
    print(f"  mem[100] = {int(memory3[100].item())}")
    print()


def test_loop():
    print("LOOP TEST - sum 1 to 5")
    print("=" * 60)

    def instr(op, imm=0):
        return float(op + (imm << 8))

    executor = C4AutoregressivePrintf(memory_size=512)

    SUM_ADDR = 300
    I_ADDR = 308

    LOOP_START = 8 * 8   # instruction 8
    LOOP_END = 30 * 8    # instruction 30

    code_loop = [
        # sum = 0
        instr(Op.IMM, SUM_ADDR),   # 0
        instr(Op.PSH),              # 1
        instr(Op.IMM, 0),           # 2
        instr(Op.SI),               # 3: mem[SUM_ADDR] = 0

        # i = 1
        instr(Op.IMM, I_ADDR),      # 4
        instr(Op.PSH),              # 5
        instr(Op.IMM, 1),           # 6
        instr(Op.SI),               # 7: mem[I_ADDR] = 1

        # LOOP: if i > 5, exit
        instr(Op.IMM, I_ADDR),      # 8: ax = &i
        instr(Op.LI),               # 9: ax = i
        instr(Op.PSH),              # 10: push i
        instr(Op.IMM, 6),           # 11: ax = 6
        instr(Op.LT),               # 12: ax = (i < 6)
        instr(Op.BZ, LOOP_END),     # 13: if i >= 6, exit loop

        # sum += i
        instr(Op.IMM, SUM_ADDR),    # 14: ax = &sum
        instr(Op.PSH),              # 15: push &sum (for SI)
        instr(Op.LI),               # 16: ax = sum
        instr(Op.PSH),              # 17: push sum
        instr(Op.IMM, I_ADDR),      # 18: ax = &i
        instr(Op.LI),               # 19: ax = i
        instr(Op.ADD),              # 20: ax = sum + i
        instr(Op.SI),               # 21: mem[&sum] = sum + i

        # i += 1
        instr(Op.IMM, I_ADDR),      # 22: ax = &i
        instr(Op.PSH),              # 23: push &i
        instr(Op.LI),               # 24: ax = i
        instr(Op.PSH),              # 25: push i
        instr(Op.IMM, 1),           # 26: ax = 1
        instr(Op.ADD),              # 27: ax = i + 1
        instr(Op.SI),               # 28: mem[&i] = i + 1

        instr(Op.JMP, LOOP_START),  # 29: goto LOOP

        # END: return sum
        instr(Op.IMM, SUM_ADDR),    # 30 (LOOP_END)
        instr(Op.LI),               # 31
        instr(Op.EXIT),             # 32
    ]

    memory = torch.zeros(512)
    for i, c in enumerate(code_loop):
        memory[i * 8] = c

    pc = torch.tensor(0.0)
    sp = torch.tensor(450.0)
    bp = torch.tensor(450.0)
    ax = torch.tensor(0.0)

    print("Tracing loop:")
    for step in range(100):
        if executor.is_exit(memory, pc):
            print(f"  EXIT at step {step}")
            break

        ctx = executor.ctx.build_context(memory, pc, sp, bp, ax)
        instr_val = executor.ctx.attend(ctx, pc, query_type='memory')
        opcode = instr_val - torch.floor(instr_val / 256) * 256
        imm = torch.floor(instr_val / 256)

        sum_val = memory[SUM_ADDR].item()
        i_val = memory[I_ADDR].item()

        if step < 50:
            print(f"  Step {step:2d}: PC={int(pc.item()):3d}, op={int(opcode.item()):2d}, AX={int(ax.item()):3d}, SP={int(sp.item()):3d}, sum={int(sum_val)}, i={int(i_val)}")

        pc, sp, bp, ax, memory, output, out_ptr = executor.step(
            pc, sp, bp, ax, memory, torch.zeros(64), torch.tensor(0.0)
        )

    print(f"\nFinal: AX={int(ax.item())}, sum={int(memory[SUM_ADDR].item())}, i={int(memory[I_ADDR].item())}")
    print(f"Expected: sum = 1+2+3+4+5 = 15")


if __name__ == "__main__":
    test_memory()
    print()
    test_loop()
