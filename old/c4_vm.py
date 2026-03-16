"""
C4 Virtual Machine - Implementation of C4's bytecode interpreter.

C4 has 39 opcodes organized into:
- Control flow: LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV
- Memory: LI, LC, SI, SC, PSH
- Arithmetic: ADD, SUB, MUL, DIV, MOD
- Bitwise: OR, XOR, AND, SHL, SHR
- Comparison: EQ, NE, LT, GT, LE, GE
- System: OPEN, READ, CLOS, PRTF, MALC, FREE, MSET, MCMP, EXIT

Registers:
- PC: program counter
- SP: stack pointer
- BP: base pointer (frame pointer)
- AX: accumulator (return values, expressions)

Memory layout (1MB total):
- 0x00000 - 0x3FFFF: code segment (256KB)
- 0x40000 - 0x7FFFF: data segment (256KB)
- 0x80000 - 0xBFFFF: stack (256KB, grows down)
- 0xC0000 - 0xFFFFF: heap (256KB)
"""

import struct
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


class Op(IntEnum):
    """C4 Opcodes."""
    # Control flow
    LEA = 0   # load effective address: ax = bp + immediate
    IMM = 1   # load immediate: ax = immediate
    JMP = 2   # jump: pc = immediate
    JSR = 3   # jump to subroutine: push pc, pc = immediate
    BZ = 4    # branch if zero: if ax == 0, pc = immediate
    BNZ = 5   # branch if not zero: if ax != 0, pc = immediate
    ENT = 6   # enter subroutine: push bp, bp = sp, sp -= immediate
    ADJ = 7   # adjust stack: sp += immediate
    LEV = 8   # leave subroutine: sp = bp, bp = pop, pc = pop

    # Memory
    LI = 9    # load int: ax = *(int*)ax
    LC = 10   # load char: ax = *(char*)ax
    SI = 11   # store int: *(int*)*sp++ = ax
    SC = 12   # store char: *(char*)*sp++ = ax
    PSH = 13  # push: *--sp = ax

    # Arithmetic
    ADD = 14  # ax = *sp++ + ax
    SUB = 15  # ax = *sp++ - ax
    MUL = 16  # ax = *sp++ * ax
    DIV = 17  # ax = *sp++ / ax
    MOD = 18  # ax = *sp++ % ax

    # Bitwise
    OR = 19   # ax = *sp++ | ax
    XOR = 20  # ax = *sp++ ^ ax
    AND = 21  # ax = *sp++ & ax
    SHL = 22  # ax = *sp++ << ax
    SHR = 23  # ax = *sp++ >> ax

    # Comparison (result in ax: 1 if true, 0 if false)
    EQ = 24   # ax = *sp++ == ax
    NE = 25   # ax = *sp++ != ax
    LT = 26   # ax = *sp++ < ax
    GT = 27   # ax = *sp++ > ax
    LE = 28   # ax = *sp++ <= ax
    GE = 29   # ax = *sp++ >= ax

    # System calls
    OPEN = 30  # open file
    READ = 31  # read from file
    CLOS = 32  # close file
    PRTF = 33  # printf
    MALC = 34  # malloc
    FREE = 35  # free
    MSET = 36  # memset
    MCMP = 37  # memcmp
    EXIT = 38  # exit program


# Memory layout constants
CODE_BASE = 0x00000
CODE_SIZE = 0x40000  # 256KB
DATA_BASE = 0x40000
DATA_SIZE = 0x40000  # 256KB
STACK_BASE = 0x80000
STACK_SIZE = 0x40000  # 256KB (stack grows down from STACK_BASE + STACK_SIZE)
HEAP_BASE = 0xC0000
HEAP_SIZE = 0x40000  # 256KB

MEMORY_SIZE = 0x100000  # 1MB total


@dataclass
class VMState:
    """Complete state of the C4 VM."""
    pc: int          # program counter
    sp: int          # stack pointer
    bp: int          # base pointer
    ax: int          # accumulator
    memory: bytearray  # 1MB memory
    cycle: int       # instruction count
    halted: bool     # whether EXIT was called
    exit_code: int   # exit code if halted

    def clone(self) -> 'VMState':
        """Create a copy of the state."""
        return VMState(
            pc=self.pc,
            sp=self.sp,
            bp=self.bp,
            ax=self.ax,
            memory=bytearray(self.memory),
            cycle=self.cycle,
            halted=self.halted,
            exit_code=self.exit_code
        )

    def to_tensor(self, device='cpu') -> torch.Tensor:
        """Convert state to tensor for transformer input."""
        # Pack registers into first 32 bytes
        header = struct.pack('<QQQQ', self.pc, self.sp, self.bp, self.ax & 0xFFFFFFFFFFFFFFFF)
        state_bytes = header + bytes(self.memory)
        return torch.tensor(list(state_bytes), dtype=torch.long, device=device)

    @staticmethod
    def from_tensor(t: torch.Tensor) -> 'VMState':
        """Reconstruct state from tensor."""
        data = bytes(t.tolist())
        pc, sp, bp, ax = struct.unpack('<QQQQ', data[:32])
        return VMState(
            pc=pc, sp=sp, bp=bp, ax=ax,
            memory=bytearray(data[32:32+MEMORY_SIZE]),
            cycle=0, halted=False, exit_code=0
        )


class C4VM:
    """C4 Virtual Machine."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset VM to initial state."""
        self.state = VMState(
            pc=CODE_BASE,
            sp=STACK_BASE + STACK_SIZE,  # stack starts at top
            bp=STACK_BASE + STACK_SIZE,
            ax=0,
            memory=bytearray(MEMORY_SIZE),
            cycle=0,
            halted=False,
            exit_code=0
        )
        self.heap_ptr = HEAP_BASE  # simple bump allocator

    def load_code(self, code: List[int], data: bytes = b''):
        """Load bytecode and data into memory."""
        # Load code (as 64-bit words)
        for i, instruction in enumerate(code):
            addr = CODE_BASE + i * 8
            struct.pack_into('<q', self.state.memory, addr, instruction)

        # Load data
        for i, byte in enumerate(data):
            self.state.memory[DATA_BASE + i] = byte

    def read_int(self, addr: int) -> int:
        """Read 64-bit signed int from memory."""
        if addr < 0 or addr + 8 > MEMORY_SIZE:
            raise ValueError(f"Memory read out of bounds: {addr:#x}")
        return struct.unpack_from('<q', self.state.memory, addr)[0]

    def write_int(self, addr: int, value: int):
        """Write 64-bit signed int to memory."""
        if addr < 0 or addr + 8 > MEMORY_SIZE:
            raise ValueError(f"Memory write out of bounds: {addr:#x}")
        struct.pack_into('<q', self.state.memory, addr, value)

    def read_byte(self, addr: int) -> int:
        """Read byte from memory."""
        if addr < 0 or addr >= MEMORY_SIZE:
            raise ValueError(f"Memory read out of bounds: {addr:#x}")
        return self.state.memory[addr]

    def write_byte(self, addr: int, value: int):
        """Write byte to memory."""
        if addr < 0 or addr >= MEMORY_SIZE:
            raise ValueError(f"Memory write out of bounds: {addr:#x}")
        self.state.memory[addr] = value & 0xFF

    def push(self, value: int):
        """Push value onto stack."""
        self.state.sp -= 8
        self.write_int(self.state.sp, value)

    def pop(self) -> int:
        """Pop value from stack."""
        value = self.read_int(self.state.sp)
        self.state.sp += 8
        return value

    def fetch(self) -> Tuple[int, int]:
        """Fetch instruction at PC, return (opcode, immediate)."""
        instruction = self.read_int(self.state.pc)
        opcode = instruction & 0xFF
        immediate = instruction >> 8
        self.state.pc += 8
        return opcode, immediate

    def step(self) -> bool:
        """Execute one instruction. Returns False if halted."""
        if self.state.halted:
            return False

        op, imm = self.fetch()
        s = self.state

        if op == Op.LEA:
            s.ax = s.bp + imm

        elif op == Op.IMM:
            s.ax = imm

        elif op == Op.JMP:
            s.pc = imm

        elif op == Op.JSR:
            self.push(s.pc)
            s.pc = imm

        elif op == Op.BZ:
            if s.ax == 0:
                s.pc = imm

        elif op == Op.BNZ:
            if s.ax != 0:
                s.pc = imm

        elif op == Op.ENT:
            self.push(s.bp)
            s.bp = s.sp
            s.sp -= imm

        elif op == Op.ADJ:
            s.sp += imm

        elif op == Op.LEV:
            s.sp = s.bp
            s.bp = self.pop()
            s.pc = self.pop()

        elif op == Op.LI:
            s.ax = self.read_int(s.ax)

        elif op == Op.LC:
            s.ax = self.read_byte(s.ax)

        elif op == Op.SI:
            self.write_int(self.pop(), s.ax)

        elif op == Op.SC:
            self.write_byte(self.pop(), s.ax)

        elif op == Op.PSH:
            self.push(s.ax)

        elif op == Op.ADD:
            s.ax = self.pop() + s.ax

        elif op == Op.SUB:
            s.ax = self.pop() - s.ax

        elif op == Op.MUL:
            s.ax = self.pop() * s.ax

        elif op == Op.DIV:
            s.ax = self.pop() // s.ax if s.ax != 0 else 0

        elif op == Op.MOD:
            s.ax = self.pop() % s.ax if s.ax != 0 else 0

        elif op == Op.OR:
            s.ax = self.pop() | s.ax

        elif op == Op.XOR:
            s.ax = self.pop() ^ s.ax

        elif op == Op.AND:
            s.ax = self.pop() & s.ax

        elif op == Op.SHL:
            s.ax = self.pop() << s.ax

        elif op == Op.SHR:
            s.ax = self.pop() >> s.ax

        elif op == Op.EQ:
            s.ax = 1 if self.pop() == s.ax else 0

        elif op == Op.NE:
            s.ax = 1 if self.pop() != s.ax else 0

        elif op == Op.LT:
            s.ax = 1 if self.pop() < s.ax else 0

        elif op == Op.GT:
            s.ax = 1 if self.pop() > s.ax else 0

        elif op == Op.LE:
            s.ax = 1 if self.pop() <= s.ax else 0

        elif op == Op.GE:
            s.ax = 1 if self.pop() >= s.ax else 0

        elif op == Op.MALC:
            # Simple bump allocator
            size = s.ax
            s.ax = self.heap_ptr
            self.heap_ptr += size

        elif op == Op.FREE:
            # No-op for bump allocator
            pass

        elif op == Op.MSET:
            # memset(dest, val, size)
            size = self.pop()
            val = self.pop()
            dest = self.pop()
            for i in range(size):
                self.write_byte(dest + i, val)
            s.ax = dest

        elif op == Op.MCMP:
            # memcmp(a, b, size)
            size = self.pop()
            b = self.pop()
            a = self.pop()
            result = 0
            for i in range(size):
                diff = self.read_byte(a + i) - self.read_byte(b + i)
                if diff != 0:
                    result = diff
                    break
            s.ax = result

        elif op == Op.EXIT:
            s.halted = True
            s.exit_code = s.ax
            return False

        elif op == Op.PRTF:
            # Simplified printf - just return 0
            s.ax = 0

        elif op == Op.OPEN:
            # Simplified - return -1 (error)
            s.ax = -1

        elif op == Op.READ:
            # Simplified - return 0 (no data)
            s.ax = 0

        elif op == Op.CLOS:
            # Simplified - return 0
            s.ax = 0

        else:
            raise ValueError(f"Unknown opcode: {op}")

        s.cycle += 1
        return True

    def run(self, max_cycles: int = 1000000) -> int:
        """Run until EXIT or max cycles. Returns exit code."""
        while self.state.cycle < max_cycles:
            if not self.step():
                break
        return self.state.exit_code


def encode_instruction(op: Op, immediate: int = 0) -> int:
    """Encode opcode and immediate into 64-bit instruction."""
    return (immediate << 8) | int(op)


def assemble(instructions: List[Tuple[Op, int]]) -> List[int]:
    """Assemble list of (opcode, immediate) into bytecode."""
    return [encode_instruction(op, imm) for op, imm in instructions]


# =============================================================================
# Example programs
# =============================================================================

def program_factorial(n: int) -> List[int]:
    """
    Generate bytecode for: return factorial(n)

    int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
    """
    # Labels (byte addresses)
    ENTRY = CODE_BASE
    FACTORIAL = CODE_BASE + 8 * 10
    BASE_CASE = CODE_BASE + 8 * 20
    RECURSE = CODE_BASE + 8 * 25

    code = [
        # Entry point: call factorial(n), then exit
        (Op.IMM, n),           # ax = n
        (Op.PSH, 0),           # push n
        (Op.JSR, FACTORIAL),   # call factorial
        (Op.ADJ, 8),           # pop argument
        (Op.PSH, 0),           # push result
        (Op.IMM, 0),           # exit code in ax
        (Op.EXIT, 0),          # exit

        # Padding to FACTORIAL
        (Op.IMM, 0), (Op.IMM, 0), (Op.IMM, 0),

        # factorial(n):
        # [bp+16] = n (argument)
        (Op.ENT, 0),           # enter, no locals
        (Op.LEA, 16),          # ax = &n
        (Op.LI, 0),            # ax = n
        (Op.PSH, 0),           # push n
        (Op.IMM, 1),           # ax = 1
        (Op.GT, 0),            # ax = (n > 1)
        (Op.BZ, BASE_CASE),    # if n <= 1, goto base case

        # Recursive case: return n * factorial(n-1)
        (Op.LEA, 16),          # ax = &n
        (Op.LI, 0),            # ax = n
        (Op.PSH, 0),           # push n (for multiplication)
        (Op.LEA, 16),          # ax = &n
        (Op.LI, 0),            # ax = n
        (Op.PSH, 0),           # push n
        (Op.IMM, 1),           # ax = 1
        (Op.SUB, 0),           # ax = n - 1
        (Op.PSH, 0),           # push (n-1)
        (Op.JSR, FACTORIAL),   # call factorial(n-1)
        (Op.ADJ, 8),           # pop argument
        (Op.MUL, 0),           # ax = n * factorial(n-1)
        (Op.LEV, 0),           # return

        # Base case: return 1
        (Op.IMM, 1),           # ax = 1
        (Op.LEV, 0),           # return
    ]
    return assemble(code)


def program_sum(n: int) -> List[int]:
    """
    Generate bytecode for: sum 1 to n

    Using data segment for variables.
    sum = 0; for i = 1 to n: sum += i
    """
    SUM_ADDR = DATA_BASE
    I_ADDR = DATA_BASE + 8

    # Calculate instruction addresses carefully
    # Each instruction is 8 bytes
    instructions = []

    def emit(op, imm=0):
        addr = CODE_BASE + len(instructions) * 8
        instructions.append((op, imm))
        return addr

    # sum = 0
    emit(Op.IMM, SUM_ADDR)     # 0: ax = &sum
    emit(Op.PSH, 0)            # 1: push &sum
    emit(Op.IMM, 0)            # 2: ax = 0
    emit(Op.SI, 0)             # 3: *pop() = ax -> sum = 0

    # i = 1
    emit(Op.IMM, I_ADDR)       # 4: ax = &i
    emit(Op.PSH, 0)            # 5: push &i
    emit(Op.IMM, 1)            # 6: ax = 1
    emit(Op.SI, 0)             # 7: i = 1

    LOOP_START = CODE_BASE + 8 * 8  # instruction 8

    # LOOP: check i <= n (i.e., i < n+1)
    emit(Op.IMM, I_ADDR)       # 8: ax = &i
    emit(Op.LI, 0)             # 9: ax = i
    emit(Op.PSH, 0)            # 10: push i
    emit(Op.IMM, n + 1)        # 11: ax = n + 1
    emit(Op.LT, 0)             # 12: ax = (i < n+1)

    LOOP_END = CODE_BASE + 8 * 30  # will be instruction 30

    emit(Op.BZ, LOOP_END)      # 13: if !(i <= n), exit

    # sum += i
    emit(Op.IMM, SUM_ADDR)     # 14: ax = &sum
    emit(Op.LI, 0)             # 15: ax = sum
    emit(Op.PSH, 0)            # 16: push sum
    emit(Op.IMM, I_ADDR)       # 17: ax = &i
    emit(Op.LI, 0)             # 18: ax = i
    emit(Op.ADD, 0)            # 19: ax = sum + i
    emit(Op.PSH, 0)            # 20: push new_sum
    emit(Op.IMM, SUM_ADDR)     # 21: ax = &sum
    emit(Op.PSH, 0)            # 22: push &sum
    # SI: *(pop()) = pop() -- no wait, SI is *sp++ = ax
    # So we need: push &sum, then have result in ax
    # Let me re-read SI: *(int*)*sp++ = ax
    # It pops an address and stores ax there

    # Redo sum += i:
    instructions.clear()

    # sum = 0
    emit(Op.IMM, SUM_ADDR)     # ax = &sum
    emit(Op.PSH, 0)            # push &sum
    emit(Op.IMM, 0)            # ax = 0
    emit(Op.SI, 0)             # *&sum = 0

    # i = 1
    emit(Op.IMM, I_ADDR)       # ax = &i
    emit(Op.PSH, 0)            # push &i
    emit(Op.IMM, 1)            # ax = 1
    emit(Op.SI, 0)             # *&i = 1

    LOOP_START = CODE_BASE + len(instructions) * 8

    # check i <= n
    emit(Op.IMM, I_ADDR)       # ax = &i
    emit(Op.LI, 0)             # ax = i
    emit(Op.PSH, 0)            # push i
    emit(Op.IMM, n + 1)        # ax = n+1
    emit(Op.LT, 0)             # ax = (i < n+1) = (i <= n)

    # BZ to end (we'll patch this)
    bz_idx = len(instructions)
    emit(Op.BZ, 0)             # placeholder

    # sum = sum + i
    emit(Op.IMM, SUM_ADDR)     # ax = &sum
    emit(Op.PSH, 0)            # push &sum (for SI later)
    emit(Op.LI, 0)             # ax = sum
    emit(Op.PSH, 0)            # push sum
    emit(Op.IMM, I_ADDR)       # ax = &i
    emit(Op.LI, 0)             # ax = i
    emit(Op.ADD, 0)            # ax = sum + i
    emit(Op.SI, 0)             # *&sum = ax

    # i = i + 1
    emit(Op.IMM, I_ADDR)       # ax = &i
    emit(Op.PSH, 0)            # push &i
    emit(Op.LI, 0)             # ax = i
    emit(Op.PSH, 0)            # push i
    emit(Op.IMM, 1)            # ax = 1
    emit(Op.ADD, 0)            # ax = i + 1
    emit(Op.SI, 0)             # *&i = ax

    emit(Op.JMP, LOOP_START)   # loop back

    LOOP_END = CODE_BASE + len(instructions) * 8

    # Patch BZ
    instructions[bz_idx] = (Op.BZ, LOOP_END)

    # return sum
    emit(Op.IMM, SUM_ADDR)     # ax = &sum
    emit(Op.LI, 0)             # ax = sum
    emit(Op.EXIT, 0)           # exit

    return assemble(instructions)


def program_simple_arithmetic() -> List[int]:
    """Simple: compute (3 + 4) * 5 = 35"""
    code = [
        (Op.IMM, 3),     # ax = 3
        (Op.PSH, 0),     # push 3
        (Op.IMM, 4),     # ax = 4
        (Op.ADD, 0),     # ax = 3 + 4 = 7
        (Op.PSH, 0),     # push 7
        (Op.IMM, 5),     # ax = 5
        (Op.MUL, 0),     # ax = 7 * 5 = 35
        (Op.EXIT, 0),    # exit with 35
    ]
    return assemble(code)


# =============================================================================
# Tests
# =============================================================================

def test_vm():
    """Test the C4 VM."""
    print("=" * 60)
    print("C4 VM TESTS")
    print("=" * 60)

    # Test 1: Simple arithmetic
    print("\n1. Simple arithmetic: (3 + 4) * 5 = 35")
    vm = C4VM()
    vm.load_code(program_simple_arithmetic())
    result = vm.run()
    print(f"   Result: {result}")
    assert result == 35, f"Expected 35, got {result}"
    print("   ✓ PASS")

    # Test 2: Sum 1 to 10 = 55
    print("\n2. Sum 1 to 10 = 55")
    vm = C4VM()
    vm.load_code(program_sum(10))
    result = vm.run()
    print(f"   Result: {result}")
    print(f"   Cycles: {vm.state.cycle}")
    assert result == 55, f"Expected 55, got {result}"
    print("   ✓ PASS")

    # Test 3: Sum 1 to 100 = 5050
    print("\n3. Sum 1 to 100 = 5050")
    vm = C4VM()
    vm.load_code(program_sum(100))
    result = vm.run()
    print(f"   Result: {result}")
    print(f"   Cycles: {vm.state.cycle}")
    assert result == 5050, f"Expected 5050, got {result}"
    print("   ✓ PASS")

    # Test 4: State to tensor and back
    print("\n4. State serialization")
    vm = C4VM()
    vm.load_code(program_simple_arithmetic())
    vm.step()  # Execute one instruction
    vm.step()  # Execute another

    tensor = vm.state.to_tensor()
    print(f"   Tensor shape: {tensor.shape}")
    print(f"   First 40 bytes: {tensor[:40].tolist()}")

    # Reconstruct
    restored = VMState.from_tensor(tensor)
    print(f"   Restored PC: {restored.pc:#x}")
    print(f"   Restored AX: {restored.ax}")
    assert restored.pc == vm.state.pc
    assert restored.ax == vm.state.ax
    print("   ✓ PASS")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


def generate_training_data(num_samples: int = 100):
    """Generate (state, next_state) pairs for transformer training."""
    print(f"\nGenerating {num_samples} training samples...")

    samples = []
    programs = [
        program_simple_arithmetic(),
        program_sum(5),
        program_sum(10),
        program_sum(20),
    ]

    for prog in programs:
        vm = C4VM()
        vm.load_code(prog)

        while not vm.state.halted and vm.state.cycle < 1000:
            state_before = vm.state.clone()
            vm.step()
            state_after = vm.state.clone()

            samples.append((state_before, state_after))

            if len(samples) >= num_samples:
                break

        if len(samples) >= num_samples:
            break

    print(f"Generated {len(samples)} samples")
    return samples


if __name__ == "__main__":
    test_vm()

    # Show opcode table
    print("\n" + "=" * 60)
    print("C4 OPCODES")
    print("=" * 60)
    for op in Op:
        print(f"  {op.value:2d}: {op.name}")
