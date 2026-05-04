"""
Comprehensive tests for the 8-bit neural VM.
"""

import sys
import torch
import random

sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_release")

from vm_8bit.neural_alu import NeuralALU, E8, N
from vm_8bit.soft_vm import SoftVM, assemble
from vm_8bit.neural_vm import NeuralVMRunner, NeuralStepRunner
from vm_8bit.autoregressive_vm import AutoregressiveRunner, FE
from vm_8bit.config import Op

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


def run_alu(alu, a, b, op_dim):
    x = torch.zeros(1, N, E8.DIM)
    x[0, 0, E8.NIB_A] = float(a & 0xF)
    x[0, 1, E8.NIB_A] = float((a >> 4) & 0xF)
    x[0, 0, E8.NIB_B] = float(b & 0xF)
    x[0, 1, E8.NIB_B] = float((b >> 4) & 0xF)
    x[0, 0, op_dim] = 1.0
    x[0, 1, op_dim] = 1.0
    with torch.no_grad():
        x = alu(x)
    lo = int(round(x[0, 0, E8.RESULT].item()))
    hi = int(round(x[0, 1, E8.RESULT].item()))
    return (lo + (hi << 4)) & 0xFF


# ── Test ALU ────────────────────────────────────────────────────────────

print("=" * 60)
print("NEURAL ALU TESTS")
print("=" * 60)

alu = NeuralALU()
alu.eval()

print("\nADD (neural SwiGLU):")
errors = 0
for a in range(256):
    for b in range(256):
        expected = (a + b) & 0xFF
        result = run_alu(alu, a, b, E8.OP_ADD)
        if result != expected:
            errors += 1
            if errors <= 3:
                print(f"    {a} + {b} = {result}, expected {expected}")
check("ADD all 256x256", errors == 0, f"{errors} errors")

print("\nSUB (neural SwiGLU):")
errors = 0
for a in range(256):
    for b in range(256):
        expected = (a - b) & 0xFF
        result = run_alu(alu, a, b, E8.OP_SUB)
        if result != expected:
            errors += 1
            if errors <= 3:
                print(f"    {a} - {b} = {result}, expected {expected}")
check("SUB all 256x256", errors == 0, f"{errors} errors")

print("\nMUL (neural SwiGLU):")
errors = 0
for a in range(256):
    for b in range(256):
        expected = (a * b) & 0xFF
        result = run_alu(alu, a, b, E8.OP_MUL)
        if result != expected:
            errors += 1
            if errors <= 5:
                print(f"    {a} * {b} = {result}, expected {expected}")
check("MUL all 256x256", errors == 0, f"{errors} errors")

print("\nRandom mixed stress:")
random.seed(42)
stress_ok = True
for _ in range(2000):
    a = random.randint(0, 255)
    b = random.randint(0, 255)
    op = random.choice([E8.OP_ADD, E8.OP_SUB, E8.OP_MUL])
    result = run_alu(alu, a, b, op)
    if op == E8.OP_ADD:
        expected = (a + b) & 0xFF
    elif op == E8.OP_SUB:
        expected = (a - b) & 0xFF
    else:
        expected = (a * b) & 0xFF
    if result != expected:
        stress_ok = False
        print(f"    FAIL: a={a} b={b} op={op} got={result} exp={expected}")
        break
check("Random 2000 stress", stress_ok)

print("\nAND (neural, 2000 random):")
errors = 0
for _ in range(2000):
    a, b = random.randint(0, 255), random.randint(0, 255)
    result = run_alu(alu, a, b, E8.OP_AND)
    if result != (a & b):
        errors += 1
        if errors <= 3:
            print(f"    {a} & {b} = {result}, expected {a & b}")
check("AND random 2000", errors == 0, f"{errors} errors")

print("\nOR (neural, 2000 random):")
errors = 0
for _ in range(2000):
    a, b = random.randint(0, 255), random.randint(0, 255)
    result = run_alu(alu, a, b, E8.OP_OR)
    if result != (a | b):
        errors += 1
        if errors <= 3:
            print(f"    {a} | {b} = {result}, expected {a | b}")
check("OR random 2000", errors == 0, f"{errors} errors")

print("\nXOR (neural, 2000 random):")
errors = 0
for _ in range(2000):
    a, b = random.randint(0, 255), random.randint(0, 255)
    result = run_alu(alu, a, b, E8.OP_XOR)
    if result != (a ^ b):
        errors += 1
        if errors <= 3:
            print(f"    {a} ^ {b} = {result}, expected {a ^ b}")
check("XOR random 2000", errors == 0, f"{errors} errors")

print("\nDIV (neural, 2000 random):")
errors = 0
for _ in range(2000):
    a, b = random.randint(0, 255), random.randint(0, 255)
    expected = (a // b) & 0xFF if b != 0 else 0
    result = run_alu(alu, a, b, E8.OP_DIV)
    if result != expected:
        errors += 1
        if errors <= 5:
            print(f"    {a} // {b} = {result}, expected {expected}")
check("DIV random 2000", errors == 0, f"{errors} errors")

print("\nMOD (neural, 2000 random):")
errors = 0
for _ in range(2000):
    a, b = random.randint(0, 255), random.randint(0, 255)
    expected = (a % b) & 0xFF if b != 0 else 0
    result = run_alu(alu, a, b, E8.OP_MOD)
    if result != expected:
        errors += 1
        if errors <= 5:
            print(f"    {a} % {b} = {result}, expected {expected}")
check("MOD random 2000", errors == 0, f"{errors} errors")

print("\nSHL (neural, 2000 random):")
errors = 0
for _ in range(2000):
    a, b = random.randint(0, 255), random.randint(0, 255)
    expected = (a << b) & 0xFF if b < 8 else 0
    result = run_alu(alu, a, b, E8.OP_SHL)
    if result != expected:
        errors += 1
        if errors <= 3:
            print(f"    {a} << {b} = {result}, expected {expected}")
check("SHL random 2000", errors == 0, f"{errors} errors")

print("\nSHR (neural, 2000 random):")
errors = 0
for _ in range(2000):
    a, b = random.randint(0, 255), random.randint(0, 255)
    expected = (a >> b) & 0xFF if b < 8 else 0
    result = run_alu(alu, a, b, E8.OP_SHR)
    if result != expected:
        errors += 1
        if errors <= 3:
            print(f"    {a} >> {b} = {result}, expected {expected}")
check("SHR random 2000", errors == 0, f"{errors} errors")


# ── Test SoftVM ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SOFT VM TESTS")
print("=" * 60)

print("\nHello World (soft):")
bc = assemble([
    (Op.IMM, ord("H")), (Op.PUTCHAR, 0),
    (Op.IMM, ord("i")), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out, code = vm.run()
check("Hello soft", out == "Hi" and code == 0, f"out={out!r} code={code}")

print("\nADD 42+50=92 (soft):")
bc = assemble([
    (Op.IMM, 42),
    (Op.PSH, 0),
    (Op.IMM, 50),
    (Op.ADD, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out, code = vm.run()
check("ADD soft", out == chr(92), f"out={out!r} expected={chr(92)!r}")

print("\nLoop 3x (soft):")
bc = assemble([
    (Op.IMM, 3),        # counter         addr 0
    (Op.BZ, 16),        # if 0 goto EXIT  addr 2
    (Op.PSH, 0),        # save counter    addr 4
    (Op.IMM, ord("X")), # AX='X'          addr 6
    (Op.PUTCHAR, 0),    # print           addr 8
    (Op.IMM, 1),        # AX=1            addr 10
    (Op.SUB, 0),        # AX=pop_cnt - 1  addr 12
    (Op.JMP, 2),        # back to BZ      addr 14
    (Op.IMM, 0),        #                 addr 16
    (Op.EXIT, 0),       #                 addr 18
])
vm = SoftVM()
vm.load(bc)
out, code = vm.run()
check("Loop 3x soft", out == "XXX", f"out={out!r}")

print("\nJSR/LEV (soft):")
fn_addr = 8
bc = assemble([
    (Op.IMM, ord("K")), # AX='K'           addr 0
    (Op.JSR, fn_addr),  # call fn(addr=8)  addr 2
    (Op.IMM, 0),        #                  addr 4
    (Op.EXIT, 0),       #                  addr 6
    (Op.ENT, 0),        # setup frame      addr 8
    (Op.PUTCHAR, 0),    # print AX         addr 10
    (Op.LEV, 0),        # return           addr 12
])
vm = SoftVM()
vm.load(bc)
out, code = vm.run()
check("JSR/LEV soft", out == "K", f"out={out!r} code={code}")


# ── Test NeuralVM Runner ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("NEURAL VM RUNNER TESTS")
print("=" * 60)

print("\nHello (neural):")
bc = assemble([
    (Op.IMM, ord("H")), (Op.PUTCHAR, 0),
    (Op.IMM, ord("i")), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("Hello neural", out == "Hi", f"out={out!r}")

print("\nADD 42+50 (neural ALU):")
bc = assemble([
    (Op.IMM, 42),
    (Op.PSH, 0),
    (Op.IMM, 50),
    (Op.ADD, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("ADD 42+50 neural", out == chr(92), f"out={out!r} exp={chr(92)!r}")

print("\nMUL 12*10 (neural ALU):")
bc = assemble([
    (Op.IMM, 12),
    (Op.PSH, 0),
    (Op.IMM, 10),
    (Op.MUL, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("MUL 12*10 neural", out == chr(120), f"out={out!r} exp={chr(120)!r}")

print("\nSUB 200-50 (neural ALU):")
bc = assemble([
    (Op.IMM, 200),
    (Op.PSH, 0),
    (Op.IMM, 50),
    (Op.SUB, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("SUB 200-50 neural", out == chr(150), f"out={out!r} exp={chr(150)!r}")

print("\nLoop 3x B (neural):")
bc = assemble([
    (Op.IMM, 3),        # counter         addr 0
    (Op.BZ, 16),        # if 0 goto EXIT  addr 2
    (Op.PSH, 0),        # save counter    addr 4
    (Op.IMM, ord("B")), # AX='B'          addr 6
    (Op.PUTCHAR, 0),    # print           addr 8
    (Op.IMM, 1),        # AX=1            addr 10
    (Op.SUB, 0),        # AX=cnt-1        addr 12
    (Op.JMP, 2),        # back to BZ      addr 14
    (Op.IMM, 0),        #                 addr 16
    (Op.EXIT, 0),       #                 addr 18
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("Loop 3x B neural", out == "BBB", f"out={out!r}")

print("\nCount 0..4 (neural):")
bc = assemble([
    (Op.IMM, 0), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 1), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 2), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 3), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 4), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("Count 0-4 neural", out == "01234", f"out={out!r}")

print("\nAND 0xAA & 0x0F (neural):")
bc = assemble([
    (Op.IMM, 0xAA),
    (Op.PSH, 0),
    (Op.IMM, 0x0F),
    (Op.AND, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("AND neural", out == chr(0x0A), f"out={out!r} exp={chr(0x0A)!r}")

print("\nOR 0xA0 | 0x0F (neural):")
bc = assemble([
    (Op.IMM, 0xA0),
    (Op.PSH, 0),
    (Op.IMM, 0x0F),
    (Op.OR, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("OR neural", out == chr(0xAF), f"out={out!r} exp={chr(0xAF)!r}")

print("\nXOR 0xFF ^ 0x0F (neural):")
bc = assemble([
    (Op.IMM, 0xFF),
    (Op.PSH, 0),
    (Op.IMM, 0x0F),
    (Op.XOR, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("XOR neural", out == chr(0xF0), f"out={out!r} exp={chr(0xF0)!r}")

print("\nDIV 100//7 (neural):")
bc = assemble([
    (Op.IMM, 100),
    (Op.PSH, 0),
    (Op.IMM, 7),
    (Op.DIV, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("DIV neural", out == chr(14), f"out={out!r} exp={chr(14)!r}")

print("\nMOD 100%7 (neural):")
bc = assemble([
    (Op.IMM, 100),
    (Op.PSH, 0),
    (Op.IMM, 7),
    (Op.MOD, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("MOD neural", out == chr(2), f"out={out!r} exp={chr(2)!r}")

print("\nSHL 5<<4 (neural):")
bc = assemble([
    (Op.IMM, 5),
    (Op.PSH, 0),
    (Op.IMM, 4),
    (Op.SHL, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("SHL neural", out == chr(80), f"out={out!r} exp={chr(80)!r}")

print("\nSHR 200>>3 (neural):")
bc = assemble([
    (Op.IMM, 200),
    (Op.PSH, 0),
    (Op.IMM, 3),
    (Op.SHR, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralVMRunner()
runner.load(bc)
out, code = runner.run()
check("SHR neural", out == chr(25), f"out={out!r} exp={chr(25)!r}")

# ── Test NeuralStepRunner (no Python dispatch) ──────────────────────────

print("\n" + "=" * 60)
print("NEURAL STEP RUNNER TESTS (no Python dispatch)")
print("=" * 60)

print("\nHello (neural step):")
bc = assemble([
    (Op.IMM, ord("H")), (Op.PUTCHAR, 0),
    (Op.IMM, ord("i")), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralStepRunner()
runner.load(bc)
out, code = runner.run()
check("Hello neural step", out == "Hi", f"out={out!r}")

print("\nADD 42+50 (neural step):")
bc = assemble([
    (Op.IMM, 42), (Op.PSH, 0), (Op.IMM, 50), (Op.ADD, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralStepRunner()
runner.load(bc)
out, code = runner.run()
check("ADD neural step", out == chr(92), f"out={out!r}")

print("\nSUB 200-50 (neural step):")
bc = assemble([
    (Op.IMM, 200), (Op.PSH, 0), (Op.IMM, 50), (Op.SUB, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralStepRunner()
runner.load(bc)
out, code = runner.run()
check("SUB neural step", out == chr(150), f"out={out!r}")

print("\nMUL 12*10 (neural step):")
bc = assemble([
    (Op.IMM, 12), (Op.PSH, 0), (Op.IMM, 10), (Op.MUL, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralStepRunner()
runner.load(bc)
out, code = runner.run()
check("MUL neural step", out == chr(120), f"out={out!r}")

print("\nLoop 3x B (neural step):")
bc = assemble([
    (Op.IMM, 3),
    (Op.BZ, 16),
    (Op.PSH, 0),
    (Op.IMM, ord("B")),
    (Op.PUTCHAR, 0),
    (Op.IMM, 1),
    (Op.SUB, 0),
    (Op.JMP, 2),
    (Op.IMM, 0),
    (Op.EXIT, 0),
])
runner = NeuralStepRunner()
runner.load(bc)
out, code = runner.run()
check("Loop 3x B neural step", out == "BBB", f"out={out!r}")

print("\nCount 0..4 (neural step):")
bc = assemble([
    (Op.IMM, 0), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 1), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 2), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 3), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 4), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = NeuralStepRunner()
runner.load(bc)
out, code = runner.run()
check("Count 0-4 neural step", out == "01234", f"out={out!r}")

# ── Test AutoregressiveRunner (fully neural, no Python dispatch) ──────────

print("\n" + "=" * 60)
print("AUTOREGRESSIVE RUNNER TESTS (fully neural)")
print("=" * 60)

print("\nHello (autoregressive):")
bc = assemble([
    (Op.IMM, ord("H")), (Op.PUTCHAR, 0),
    (Op.IMM, ord("i")), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("Hello autoregressive", out == "Hi", f"out={out!r}")

print("\nADD 42+50 (autoregressive):")
bc = assemble([
    (Op.IMM, 42), (Op.PSH, 0), (Op.IMM, 50), (Op.ADD, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("ADD 42+50 autoregressive", out == chr(92), f"out={out!r}")

print("\nMUL 12*10 (autoregressive):")
bc = assemble([
    (Op.IMM, 12), (Op.PSH, 0), (Op.IMM, 10), (Op.MUL, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("MUL 12*10 autoregressive", out == chr(120), f"out={out!r}")

print("\nSUB 200-50 (autoregressive):")
bc = assemble([
    (Op.IMM, 200), (Op.PSH, 0), (Op.IMM, 50), (Op.SUB, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("SUB 200-50 autoregressive", out == chr(150), f"out={out!r}")

print("\nLoop 3x B (autoregressive):")
bc = assemble([
    (Op.IMM, 3),
    (Op.BZ, 16),
    (Op.PSH, 0),
    (Op.IMM, ord("B")),
    (Op.PUTCHAR, 0),
    (Op.IMM, 1),
    (Op.SUB, 0),
    (Op.JMP, 2),
    (Op.IMM, 0),
    (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("Loop 3x B autoregressive", out == "BBB", f"out={out!r}")

print("\nAND 0xAA & 0x0F (autoregressive):")
bc = assemble([
    (Op.IMM, 0xAA), (Op.PSH, 0), (Op.IMM, 0x0F), (Op.AND, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("AND autoregressive", out == chr(0x0A), f"out={out!r}")

print("\nOR 0xA0 | 0x0F (autoregressive):")
bc = assemble([
    (Op.IMM, 0xA0), (Op.PSH, 0), (Op.IMM, 0x0F), (Op.OR, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("OR autoregressive", out == chr(0xAF), f"out={out!r}")

print("\nXOR 0xFF ^ 0x0F (autoregressive):")
bc = assemble([
    (Op.IMM, 0xFF), (Op.PSH, 0), (Op.IMM, 0x0F), (Op.XOR, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("XOR autoregressive", out == chr(0xF0), f"out={out!r}")

print("\nDIV 100//7 (autoregressive):")
bc = assemble([
    (Op.IMM, 100), (Op.PSH, 0), (Op.IMM, 7), (Op.DIV, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("DIV autoregressive", out == chr(14), f"out={out!r}")

print("\nMOD 100%7 (autoregressive):")
bc = assemble([
    (Op.IMM, 100), (Op.PSH, 0), (Op.IMM, 7), (Op.MOD, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("MOD autoregressive", out == chr(2), f"out={out!r}")

print("\nSHL 5<<4 (autoregressive):")
bc = assemble([
    (Op.IMM, 5), (Op.PSH, 0), (Op.IMM, 4), (Op.SHL, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("SHL autoregressive", out == chr(80), f"out={out!r}")

print("\nSHR 200>>3 (autoregressive):")
bc = assemble([
    (Op.IMM, 200), (Op.PSH, 0), (Op.IMM, 3), (Op.SHR, 0),
    (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("SHR autoregressive", out == chr(25), f"out={out!r}")

print("\nCount 0..4 (autoregressive):")
bc = assemble([
    (Op.IMM, 0), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 1), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 2), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 3), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 4), (Op.PSH, 0), (Op.IMM, ord("0")), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("Count 0-4 autoregressive", out == "01234", f"out={out!r}")

print("\nJSR/LEV (autoregressive):")
bc = assemble([
    (Op.IMM, ord("K")),
    (Op.JSR, 8),
    (Op.IMM, 0),
    (Op.EXIT, 0),
    (Op.ENT, 0),
    (Op.PUTCHAR, 0),
    (Op.LEV, 0),
])
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("JSR/LEV autoregressive", out == "K", f"out={out!r}")

print("\nADJ (autoregressive):")
bc = assemble([
    (Op.IMM, 10), (Op.PSH, 0),
    (Op.IMM, 20), (Op.PSH, 0),
    (Op.IMM, 30), (Op.PSH, 0),
    (Op.ADJ, 2),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("ADJ autoregressive", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nLI+SI (autoregressive):")
bc = assemble([
    (Op.IMM, 128), (Op.PSH, 0),
    (Op.IMM, 42), (Op.SI, 0),
    (Op.IMM, 128), (Op.LI, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("LI+SI autoregressive", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nENT+ADJ+LEV (autoregressive):")
bc = assemble([
    (Op.IMM, 42),
    (Op.PSH, 0),
    (Op.JSR, 14),
    (Op.ADJ, 1),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0),
    (Op.EXIT, 0),
    (Op.ENT, 2),
    (Op.ADJ, 2),
    (Op.LEV, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("ENT+ADJ+LEV autoregressive", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── Comparison ops (autoregressive) ──────────────────────────────────────

print("\n" + "=" * 60)
print("COMPARISON OP TESTS (autoregressive)")
print("=" * 60)


def _cmp_test(name, op, a, b, expected):
    bc = assemble([
        (Op.IMM, a), (Op.PSH, 0), (Op.IMM, b), (op, 0),
        (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0),
    ])
    runner = AutoregressiveRunner()
    runner.load(bc)
    out, code = runner.run()
    check(name, out == chr(expected), f"out={out!r} exp={chr(expected)!r}")


_cmp_test("EQ 5==5", Op.EQ, 5, 5, 1)
_cmp_test("EQ 5==3", Op.EQ, 5, 3, 0)
_cmp_test("NE 5!=3", Op.NE, 5, 3, 1)
_cmp_test("NE 5!=5", Op.NE, 5, 5, 0)
_cmp_test("LT 3<5", Op.LT, 3, 5, 1)
_cmp_test("LT 5<3", Op.LT, 5, 3, 0)
_cmp_test("LT 5<5", Op.LT, 5, 5, 0)
_cmp_test("GT 5>3", Op.GT, 5, 3, 1)
_cmp_test("GT 3>5", Op.GT, 3, 5, 0)
_cmp_test("GT 5>5", Op.GT, 5, 5, 0)
_cmp_test("LE 3<=5", Op.LE, 3, 5, 1)
_cmp_test("LE 5<=5", Op.LE, 5, 5, 1)
_cmp_test("LE 5<=3", Op.LE, 5, 3, 0)
_cmp_test("GE 5>=3", Op.GE, 5, 3, 1)
_cmp_test("GE 5>=5", Op.GE, 5, 5, 1)
_cmp_test("GE 3>=5", Op.GE, 3, 5, 0)

_cmp_test("EQ 0==0", Op.EQ, 0, 0, 1)
_cmp_test("EQ 255==255", Op.EQ, 255, 255, 1)
_cmp_test("LT 0<255", Op.LT, 0, 255, 1)
_cmp_test("GT 255>0", Op.GT, 255, 0, 1)
_cmp_test("LT 100<200", Op.LT, 100, 200, 1)
_cmp_test("GT 200>100", Op.GT, 200, 100, 1)

# ── LC/SC alias tests (autoregressive) ───────────────────────────────────

print("\n" + "=" * 60)
print("LC/SC ALIAS TESTS (autoregressive)")
print("=" * 60)

print("\nLC alias (autoregressive):")
bc = assemble([
    (Op.IMM, 128), (Op.PSH, 0),
    (Op.IMM, 42), (Op.SC, 0),
    (Op.IMM, 128), (Op.LC, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("LC/SC alias", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nLC reads stored value (autoregressive):")
bc = assemble([
    (Op.IMM, 200), (Op.PSH, 0),
    (Op.IMM, 99), (Op.SI, 0),
    (Op.IMM, 200), (Op.LC, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("LC reads SI value", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── GETCHAR test (autoregressive) ────────────────────────────────────────

print("\n" + "=" * 60)
print("GETCHAR TEST (autoregressive)")
print("=" * 60)

print("\nGETCHAR echo (autoregressive):")
bc = assemble([
    (Op.GETCHAR, 0),
    (Op.PUTCHAR, 0),
    (Op.GETCHAR, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
vm.set_input("AB")
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
runner.set_input("AB")
out, code = runner.run()
check("GETCHAR echo AB", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nGETCHAR empty input (autoregressive):")
bc = assemble([
    (Op.GETCHAR, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("GETCHAR empty -> 255", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nGETCHAR + arithmetic (autoregressive):")
bc = assemble([
    (Op.GETCHAR, 0),
    (Op.PSH, 0),
    (Op.IMM, 1), (Op.ADD, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
vm.set_input("A")
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
runner.set_input("A")
out, code = runner.run()
check("GETCHAR+ADD", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── Stress tests (autoregressive) ────────────────────────────────────────

print("\n" + "=" * 60)
print("STRESS TESTS (autoregressive)")
print("=" * 60)

print("\nFibonacci(7) = 13 (autoregressive):")

def _soft_fib(n):
    bc = assemble([
        (Op.IMM, 0), (Op.PSH, 0),
        (Op.IMM, 1), (Op.PSH, 0),
    ])
    offset = len(bc) // 2
    loop_addr = offset * 2
    bc += assemble([
        (Op.IMM, n),
        (Op.BZ, 0),
        (Op.PSH, 0),
        (Op.IMM, 1), (Op.SUB, 0),
    ])
    bc += assemble([
        (Op.PSH, 0),
        (Op.IMM, 2), (Op.ADJ, 2),
        (Op.ADD, 0),
        (Op.PSH, 0),
        (Op.IMM, 1), (Op.ADJ, 1),
        (Op.JMP, loop_addr),
    ])
    exit_addr = len(bc) // 2 * 2
    bc_list = list(bc)
    exit_imm_pos = (offset + 2) * 2 + 1
    bc_list[exit_imm_pos] = exit_addr & 0xFF
    bc = bytes(bc_list)
    bc += assemble([
        (Op.IMM, 0), (Op.ADJ, 3),
        (Op.PUTCHAR, 0),
        (Op.IMM, 0), (Op.EXIT, 0),
    ])
    vm = SoftVM()
    vm.load(bc)
    out, _ = vm.run()
    return out, bc

expected_fib, fib_bc = _soft_fib(7)
runner = AutoregressiveRunner()
runner.load(fib_bc)
out, code = runner.run()
check("Fibonacci(7)", out == expected_fib, f"out={out!r} exp={expected_fib!r}")

print("\nNested function calls (autoregressive):")
bc = assemble([
    (Op.IMM, 3), (Op.PSH, 0),
    (Op.JSR, 10),
    (Op.ADJ, 1),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
    (Op.ENT, 0),
    (Op.IMM, ord("A")), (Op.ADD, 0),
    (Op.LEV, 0),
    (Op.ENT, 2),
    (Op.LI, 0),
    (Op.PSH, 0),
    (Op.IMM, 1), (Op.SUB, 0),
    (Op.PSH, 0),
    (Op.IMM, 1), (Op.ADJ, 1),
    (Op.ADD, 0),
    (Op.LEV, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("Nested JSR/LEV", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nLoop counting down with BZ+BNZ (autoregressive):")
bc = assemble([
    (Op.IMM, 5),        # 0: AX=5
    (Op.BZ, 14),        # 2: if AX==0 goto EXIT
    (Op.PSH, 0),        # 4: push AX
    (Op.PUTCHAR, 0),    # 6: print AX
    (Op.IMM, 1),        # 8: AX=1
    (Op.SUB, 0),        # 10: AX=pop-1
    (Op.BNZ, 2),        # 12: if AX!=0 goto BZ check
    (Op.IMM, 0),        # 14
    (Op.EXIT, 0),       # 16
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("BZ+BNZ countdown", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nComparison-driven loop (autoregressive):")
bc = assemble([
    (Op.IMM, 0), (Op.PSH, 0),
    (Op.IMM, 1), (Op.PSH, 0),
    (Op.IMM, 48), (Op.ADD, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 1), (Op.PSH, 0),
    (Op.ADD, 0), (Op.PSH, 0),
    (Op.IMM, 5), (Op.LT, 0),
    (Op.BNZ, 14),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("LT-driven loop", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nMemory roundtrip stress (autoregressive):")
bc = assemble([
    (Op.IMM, 0x90), (Op.PSH, 0),
    (Op.IMM, 42), (Op.SI, 0),
    (Op.IMM, 0x91), (Op.PSH, 0),
    (Op.IMM, 99), (Op.SI, 0),
    (Op.IMM, 0x90), (Op.LI, 0),
    (Op.PSH, 0),
    (Op.IMM, 0x91), (Op.LI, 0),
    (Op.ADD, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("Memory roundtrip", out == out_soft, f"out={out!r} soft={out_soft!r}")


# ── PUTCHAR chr(0) ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PUTCHAR CHR(0) TEST")
print("=" * 60)

print("\nPUTCHAR chr(0) (autoregressive):")
bc = assemble([
    (Op.IMM, 0), (Op.PUTCHAR, 0),
    (Op.IMM, ord("Z")), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("PUTCHAR chr(0)", out == out_soft, f"out={out!r} soft={out_soft!r}")
check("PUTCHAR chr(0) is \\x00Z", out == "\x00Z", f"out={out!r}")

# ── PC wrapping at 254/255 ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("PC WRAPPING TEST")
print("=" * 60)

print("\nPC wraps from 254 to 0 (autoregressive):")
bc = bytearray(256)
bc[254] = Op.IMM; bc[255] = ord("X")
bc[0] = Op.PUTCHAR; bc[1] = 0
bc[2] = Op.IMM; bc[3] = 0
bc[4] = Op.EXIT; bc[5] = 0
vm = SoftVM()
vm.load(bytes(bc))
vm.pc = 254
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bytes(bc))
state = torch.zeros(1, FE.DIM)
for i in range(256):
    state[0, FE.MEM + i] = float(bc[i])
state[0, FE.PC] = 254.0
state[0, FE.SP] = float(0xFE)
state[0, FE.BP] = float(0xFE)
state[0, FE.CONST] = 1.0
output = []
with torch.no_grad():
    for _ in range(20):
        state[0, FE.INPUT_VAL] = 255.0
        state = runner.model(state)
        if state[0, FE.OP_PUTCHAR].clamp(0, 1).round().item() > 0.5:
            output.append(chr(int(round(state[0, FE.OUTPUT_CHAR].item())) & 0xFF))
        if state[0, FE.HALT].item() > 0.5:
            break
out = "".join(output)
check("PC wrap 254->0", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── SP wrapping ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SP WRAPPING TEST")
print("=" * 60)

print("\nSP at 0 push wraps (autoregressive):")
bc = assemble([
    (Op.IMM, 0x80), (Op.PSH, 0),
    (Op.IMM, 42), (Op.SI, 0),
    (Op.IMM, 0x80), (Op.LI, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
vm.sp = 1
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
state = torch.zeros(1, FE.DIM)
for i, b in enumerate(bc):
    state[0, FE.MEM + i] = float(b & 0xFF)
state[0, FE.SP] = 1.0
state[0, FE.BP] = 1.0
state[0, FE.CONST] = 1.0
output = []
with torch.no_grad():
    for _ in range(100):
        state[0, FE.INPUT_VAL] = 255.0
        state = runner.model(state)
        if state[0, FE.OP_PUTCHAR].clamp(0, 1).round().item() > 0.5:
            output.append(chr(int(round(state[0, FE.OUTPUT_CHAR].item())) & 0xFF))
        if state[0, FE.HALT].item() > 0.5:
            break
out = "".join(output)
check("SP at 1 push wraps", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── Register preservation ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("REGISTER PRESERVATION TEST")
print("=" * 60)

print("\nNon-modifying ops preserve registers (autoregressive):")
bc = assemble([
    (Op.IMM, 42),
    (Op.PSH, 0),
    (Op.IMM, 7),
    (Op.PSH, 0),
    (Op.IMM, 99),
    (Op.PUTCHAR, 0),
    (Op.ADD, 0),
    (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM()
vm.load(bc)
out_soft, _ = vm.run()
runner = AutoregressiveRunner()
runner.load(bc)
out, code = runner.run()
check("Register preservation", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── IMM edge values ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("IMM EDGE VALUE TESTS")
print("=" * 60)

for val in [0, 1, 127, 128, 254, 255]:
    bc = assemble([
        (Op.IMM, val), (Op.PUTCHAR, 0),
        (Op.IMM, 0), (Op.EXIT, 0),
    ])
    vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
    runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
    check(f"IMM {val}", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── ENT/ADJ various immediates ──────────────────────────────────────────

print("\n" + "=" * 60)
print("ENT/ADJ VARIOUS IMMEDIATES TESTS")
print("=" * 60)

for adj_val in [0, 1, 3, 5, 10]:
    bc = assemble([
        (Op.IMM, 42), (Op.PSH, 0), (Op.IMM, 17), (Op.PSH, 0), (Op.IMM, 99), (Op.PSH, 0),
        (Op.ADJ, adj_val),
        (Op.PUTCHAR, 0),
        (Op.IMM, 0), (Op.EXIT, 0),
    ])
    vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
    runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
    check(f"ADJ {adj_val}", out == out_soft, f"out={out!r} soft={out_soft!r}")

for ent_val in [0, 1, 3, 5]:
    bc = assemble([
        (Op.IMM, 77),
        (Op.JSR, 6),
        (Op.IMM, 0), (Op.EXIT, 0),
        (Op.ENT, ent_val),
        (Op.PUTCHAR, 0),
        (Op.LEV, 0),
    ])
    vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
    runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
    check(f"ENT {ent_val}", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── JMP standalone ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("JMP STANDALONE TEST")
print("=" * 60)

print("\nJMP forward (autoregressive):")
bc = assemble([
    (Op.JMP, 6),            # 0: skip past B
    (Op.IMM, ord("B")),     # 2
    (Op.PUTCHAR, 0),        # 4
    (Op.IMM, ord("A")),     # 6
    (Op.PUTCHAR, 0),        # 8
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
check("JMP forward skip", out == out_soft, f"out={out!r} soft={out_soft!r}")

print("\nJMP backward loop (autoregressive):")
bc = assemble([
    (Op.IMM, 3),        # 0: AX=3
    (Op.PSH, 0),        # 2: push AX
    (Op.PUTCHAR, 0),    # 4: print AX
    (Op.IMM, 1),        # 6: AX=1
    (Op.SUB, 0),        # 8: AX=pop-1
    (Op.BZ, 14),        # 10: if AX==0 goto EXIT
    (Op.JMP, 2),        # 12: loop back to PSH
    (Op.IMM, 0),        # 14
    (Op.EXIT, 0),       # 16
])
vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
check("JMP backward loop", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── GETCHAR buffer exhaustion mid-run ───────────────────────────────────

print("\n" + "=" * 60)
print("GETCHAR BUFFER EXHAUSTION TEST")
print("=" * 60)

print("\nGETCHAR 3 reads with 1 char (autoregressive):")
bc = assemble([
    (Op.GETCHAR, 0), (Op.PUTCHAR, 0),
    (Op.GETCHAR, 0), (Op.PUTCHAR, 0),
    (Op.GETCHAR, 0), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM(); vm.load(bc); vm.set_input("A"); out_soft, _ = vm.run()
runner = AutoregressiveRunner(); runner.load(bc); runner.set_input("A"); out, _ = runner.run()
check("GETCHAR 3 reads 1 char", out == out_soft, f"out={out!r} soft={out_soft!r}")
expected = chr(ord("A")) + chr(255) + chr(255)
check("GETCHAR exhaustion is A+255+255", out == expected, f"out={out!r} exp={expected!r}")

# ── LI/SI boundary addresses ────────────────────────────────────────────

print("\n" + "=" * 60)
print("LI/SI BOUNDARY ADDRESS TESTS")
print("=" * 60)

for addr in [0, 1, 127, 128, 254, 255]:
    bc = assemble([
        (Op.IMM, addr), (Op.PSH, 0),
        (Op.IMM, 77), (Op.SI, 0),
        (Op.IMM, addr), (Op.LI, 0),
        (Op.PUTCHAR, 0),
        (Op.IMM, 0), (Op.EXIT, 0),
    ])
    vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
    runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
    check(f"LI/SI addr {addr}", out == out_soft, f"out={out!r} soft={out_soft!r}")

# ── BZ/BNZ explicit fall-through ────────────────────────────────────────

print("\n" + "=" * 60)
print("BZ/BNZ FALL-THROUGH TESTS")
print("=" * 60)

print("\nBZ fall-through (AX!=0) (autoregressive):")
bc = assemble([
    (Op.IMM, 5),
    (Op.BZ, 8),
    (Op.IMM, ord("Y")), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
check("BZ fall-through AX!=0", out == out_soft, f"out={out!r} soft={out_soft!r}")
check("BZ fall-through is Y", out == "Y", f"out={out!r}")

print("\nBZ taken (AX==0) (autoregressive):")
bc = assemble([
    (Op.IMM, 0),            # 0: AX=0
    (Op.BZ, 8),             # 2: jump to addr 8
    (Op.IMM, ord("N")),     # 4: (skipped)
    (Op.PUTCHAR, 0),        # 6: (skipped)
    (Op.IMM, ord("Y")),     # 8: AX='Y'
    (Op.PUTCHAR, 0),        # 10: print Y
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
check("BZ taken AX==0", out == out_soft, f"out={out!r} soft={out_soft!r}")
check("BZ taken is Y", out == "Y", f"out={out!r}")

print("\nBNZ fall-through (AX==0) (autoregressive):")
bc = assemble([
    (Op.IMM, 0),
    (Op.BNZ, 8),
    (Op.IMM, ord("Y")), (Op.PUTCHAR, 0),
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
check("BNZ fall-through AX==0", out == out_soft, f"out={out!r} soft={out_soft!r}")
check("BNZ fall-through is Y", out == "Y", f"out={out!r}")

print("\nBNZ taken (AX!=0) (autoregressive):")
bc = assemble([
    (Op.IMM, 5),            # 0: AX=5
    (Op.BNZ, 8),            # 2: jump to addr 8
    (Op.IMM, ord("N")),     # 4: (skipped)
    (Op.PUTCHAR, 0),        # 6: (skipped)
    (Op.IMM, ord("Y")),     # 8: AX='Y'
    (Op.PUTCHAR, 0),        # 10: print Y
    (Op.IMM, 0), (Op.EXIT, 0),
])
vm = SoftVM(); vm.load(bc); out_soft, _ = vm.run()
runner = AutoregressiveRunner(); runner.load(bc); out, _ = runner.run()
check("BNZ taken AX!=0", out == out_soft, f"out={out!r} soft={out_soft!r}")
check("BNZ taken is Y", out == "Y", f"out={out!r}")

# ── Random fuzz (SoftVM vs AutoregressiveRunner) ────────────────────────

print("\n" + "=" * 60)
print("RANDOM FUZZ TESTS (SoftVM vs AutoregressiveRunner)")
print("=" * 60)


def _gen_structured_fuzz(rng):
    bc = bytearray()
    prog_type = rng.randint(0, 5)
    if prog_type == 0:
        bc += assemble([(Op.IMM, rng.randint(0, 255)), (Op.PSH, 0),
                        (Op.IMM, rng.randint(0, 255)),
                        (rng.choice([Op.ADD, Op.SUB, Op.MUL, Op.AND, Op.OR, Op.XOR]), 0),
                        (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0)])
    elif prog_type == 1:
        bc += assemble([(Op.IMM, rng.randint(0, 255)), (Op.PSH, 0),
                        (Op.IMM, rng.randint(0, 255)),
                        (rng.choice([Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE]), 0),
                        (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0)])
    elif prog_type == 2:
        val = rng.randint(1, 5)
        bc += assemble([(Op.IMM, val), (Op.BZ, 14), (Op.PSH, 0), (Op.PUTCHAR, 0),
                        (Op.IMM, 1), (Op.SUB, 0), (Op.JMP, 2),
                        (Op.IMM, 0), (Op.EXIT, 0)])
    elif prog_type == 3:
        addr = rng.choice([128, 160, 192, 200])
        val = rng.randint(0, 255)
        bc += assemble([(Op.IMM, addr), (Op.PSH, 0), (Op.IMM, val), (Op.SI, 0),
                        (Op.IMM, addr), (Op.LI, 0), (Op.PUTCHAR, 0),
                        (Op.IMM, 0), (Op.EXIT, 0)])
    elif prog_type == 4:
        n = rng.randint(1, 4)
        for c in range(n):
            bc += bytes([Op.IMM, rng.randint(32, 126), Op.PSH, 0])
        bc += assemble([(Op.ADJ, n), (Op.PUTCHAR, 0), (Op.IMM, 0), (Op.EXIT, 0)])
    else:
        fn_addr = 8
        bc += assemble([(Op.IMM, rng.randint(32, 126)), (Op.JSR, fn_addr),
                        (Op.IMM, 0), (Op.EXIT, 0),
                        (Op.ENT, 0), (Op.PUTCHAR, 0), (Op.LEV, 0)])
    return bytes(bc)


def _gen_valid_fuzz(rng):
    bc = bytearray()
    prog_len = rng.randint(1, 30)
    bc += bytes([Op.IMM, rng.randint(0, 255)])
    bc += bytes([Op.PSH, 0])
    for _ in range(prog_len):
        choice = rng.randint(0, 15)
        if choice <= 3:
            bc += bytes([Op.IMM, rng.randint(0, 255)])
            bc += bytes([Op.PSH, 0])
        elif choice <= 7:
            op = rng.choice([Op.ADD, Op.SUB, Op.MUL, Op.AND, Op.OR, Op.XOR,
                             Op.DIV, Op.MOD, Op.SHL, Op.SHR,
                             Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE])
            bc += bytes([op, 0])
        elif choice <= 9:
            bc += bytes([Op.PUTCHAR, 0])
        elif choice <= 11:
            addr = rng.randint(0x80, 0xFD)
            bc += bytes([Op.IMM, addr])
            bc += bytes([Op.PSH, 0])
            bc += bytes([Op.IMM, rng.randint(0, 255)])
            bc += bytes([Op.SI, 0])
            bc += bytes([Op.IMM, addr])
            bc += bytes([Op.LI, 0])
        elif choice == 12:
            bc += bytes([Op.GETCHAR, 0])
        elif choice == 13:
            bc += bytes([Op.PSH, 0])
        elif choice == 14:
            bc += bytes([Op.ADJ, rng.randint(0, 3)])
        else:
            bc += bytes([Op.IMM, rng.randint(0, 255)])
            bc += bytes([Op.PSH, 0])
    bc += bytes([Op.PUTCHAR, 0])
    bc += bytes([Op.IMM, 0])
    bc += bytes([Op.EXIT, 0])
    if len(bc) > 254:
        bc = bc[:252] + bytes([Op.IMM, 0, Op.EXIT, 0])
    bc += bytes([Op.EXIT, 0]) * ((256 - len(bc)) // 2)
    return bytes(bc)


FUZZ_COUNT = 300
fuzz_errors = 0
fuzz_skipped = 0
rng = random.Random(12345)
for i in range(FUZZ_COUNT):
    if i < FUZZ_COUNT // 3:
        bc = _gen_structured_fuzz(rng)
    else:
        bc = _gen_valid_fuzz(rng)
    vm = SoftVM()
    vm.load(bc)
    out_soft, _ = vm.run()
    vm2 = SoftVM()
    vm2.load(bc)
    min_sp = 256
    for _ in range(500):
        min_sp = min(min_sp, vm2.sp)
        if not vm2.step():
            break
    if min_sp < len(bc):
        fuzz_skipped += 1
        continue
    runner = AutoregressiveRunner()
    runner.load(bc)
    out, _ = runner.run()
    if out != out_soft:
        fuzz_errors += 1
        if fuzz_errors <= 5:
            print(f"    FAIL fuzz #{i}: out={out!r} soft={out_soft!r}")
            print(f"      bc={list(bc[:40])}")
check(f"Random fuzz {FUZZ_COUNT}", fuzz_errors == 0, f"{fuzz_errors} errors ({fuzz_skipped} skipped)")


# ── Summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
