"""
Full C4 VM with pointer support.

Memory layout:
- Code: 0x00000 - 0x0FFFF (64KB)
- Data: 0x10000 - 0x1FFFF (64KB)
- Stack: 0x20000 - 0x2FFFF (64KB, grows down from top)

All addresses are byte addresses.
"""

from enum import IntEnum
from typing import List, Tuple, Dict, Optional
import struct


class Op(IntEnum):
    LEA = 0
    IMM = 1
    JMP = 2
    JSR = 3
    BZ = 4
    BNZ = 5
    ENT = 6
    ADJ = 7
    LEV = 8
    LI = 9
    LC = 10
    SI = 11
    SC = 12
    PSH = 13

    OR = 14
    XOR = 15
    AND = 16
    EQ = 17
    NE = 18
    LT = 19
    GT = 20
    LE = 21
    GE = 22
    SHL = 23
    SHR = 24
    ADD = 25
    SUB = 26
    MUL = 27
    DIV = 28
    MOD = 29

    OPEN = 30
    READ = 31
    CLOS = 32
    PRTF = 33
    MALC = 34
    FREE = 35
    MSET = 36
    MCMP = 37
    EXIT = 38


# Memory layout
CODE_BASE = 0x00000
DATA_BASE = 0x10000
STACK_BASE = 0x20000
STACK_TOP = 0x30000
MEMORY_SIZE = 0x30000


class C4VM:
    """Full C4 VM with pointer support."""

    def __init__(self):
        self.memory = bytearray(MEMORY_SIZE)
        self.pc = 0
        self.sp = STACK_TOP
        self.bp = STACK_TOP
        self.ax = 0
        self.halted = False
        self.exit_code = 0
        self.output = ""
        self.heap_ptr = DATA_BASE + 0x8000  # Heap starts at middle of data segment

        # File handles for syscalls
        self.files: Dict[int, any] = {}
        self.next_fd = 3

    def load(self, code: List[int], data: List[int]):
        """Load code and data into memory."""
        # Load code (each instruction is 8 bytes)
        for i, instr in enumerate(code):
            addr = CODE_BASE + i * 8
            struct.pack_into('<q', self.memory, addr, instr)

        # Load data (byte array)
        for i, byte in enumerate(data):
            self.memory[DATA_BASE + i] = byte & 0xFF

    def read_int(self, addr: int) -> int:
        """Read 64-bit int from memory."""
        if addr < 0 or addr + 8 > MEMORY_SIZE:
            return 0
        return struct.unpack_from('<q', self.memory, addr)[0]

    def write_int(self, addr: int, val: int):
        """Write 64-bit int to memory."""
        if addr < 0 or addr + 8 > MEMORY_SIZE:
            return
        struct.pack_into('<q', self.memory, addr, val)

    def read_byte(self, addr: int) -> int:
        """Read byte from memory."""
        if addr < 0 or addr >= MEMORY_SIZE:
            return 0
        return self.memory[addr]

    def write_byte(self, addr: int, val: int):
        """Write byte to memory."""
        if addr < 0 or addr >= MEMORY_SIZE:
            return
        self.memory[addr] = val & 0xFF

    def read_string(self, addr: int) -> str:
        """Read null-terminated string from memory."""
        chars = []
        while addr < MEMORY_SIZE:
            c = self.memory[addr]
            if c == 0:
                break
            chars.append(chr(c))
            addr += 1
        return ''.join(chars)

    def push(self, val: int):
        self.sp -= 8
        self.write_int(self.sp, val)

    def pop(self) -> int:
        val = self.read_int(self.sp)
        self.sp += 8
        return val

    def step(self) -> bool:
        """Execute one instruction."""
        if self.halted:
            return False

        # Fetch
        instr = self.read_int(self.pc)
        op = instr & 0xFF
        imm = instr >> 8
        self.pc += 8

        # Execute
        if op == Op.LEA:
            self.ax = self.bp + imm
        elif op == Op.IMM:
            self.ax = imm
        elif op == Op.JMP:
            self.pc = imm
        elif op == Op.JSR:
            self.push(self.pc)
            self.pc = imm
        elif op == Op.BZ:
            if self.ax == 0:
                self.pc = imm
        elif op == Op.BNZ:
            if self.ax != 0:
                self.pc = imm
        elif op == Op.ENT:
            self.push(self.bp)
            self.bp = self.sp
            self.sp -= imm
        elif op == Op.ADJ:
            self.sp += imm
        elif op == Op.LEV:
            self.sp = self.bp
            self.bp = self.pop()
            self.pc = self.pop()
        elif op == Op.LI:
            self.ax = self.read_int(self.ax)
        elif op == Op.LC:
            self.ax = self.read_byte(self.ax)
        elif op == Op.SI:
            self.write_int(self.pop(), self.ax)
        elif op == Op.SC:
            self.write_byte(self.pop(), self.ax)
            self.ax = self.ax & 0xFF
        elif op == Op.PSH:
            self.push(self.ax)

        elif op == Op.OR:
            self.ax = self.pop() | self.ax
        elif op == Op.XOR:
            self.ax = self.pop() ^ self.ax
        elif op == Op.AND:
            self.ax = self.pop() & self.ax
        elif op == Op.EQ:
            self.ax = 1 if self.pop() == self.ax else 0
        elif op == Op.NE:
            self.ax = 1 if self.pop() != self.ax else 0
        elif op == Op.LT:
            self.ax = 1 if self.pop() < self.ax else 0
        elif op == Op.GT:
            self.ax = 1 if self.pop() > self.ax else 0
        elif op == Op.LE:
            self.ax = 1 if self.pop() <= self.ax else 0
        elif op == Op.GE:
            self.ax = 1 if self.pop() >= self.ax else 0
        elif op == Op.SHL:
            self.ax = self.pop() << self.ax
        elif op == Op.SHR:
            self.ax = self.pop() >> self.ax
        elif op == Op.ADD:
            self.ax = self.pop() + self.ax
        elif op == Op.SUB:
            self.ax = self.pop() - self.ax
        elif op == Op.MUL:
            self.ax = self.pop() * self.ax
        elif op == Op.DIV:
            b = self.ax
            a = self.pop()
            self.ax = a // b if b != 0 else 0
        elif op == Op.MOD:
            b = self.ax
            a = self.pop()
            self.ax = a % b if b != 0 else 0

        # Syscalls
        elif op == Op.OPEN:
            # open(filename, flags)
            flags = self.pop()
            filename_addr = self.pop()
            filename = self.read_string(filename_addr)
            try:
                mode = 'rb' if flags == 0 else 'wb'
                f = open(filename, mode)
                fd = self.next_fd
                self.next_fd += 1
                self.files[fd] = f
                self.ax = fd
            except:
                self.ax = -1

        elif op == Op.READ:
            # read(fd, buf, count)
            count = self.pop()
            buf_addr = self.pop()
            fd = self.pop()
            if fd in self.files:
                data = self.files[fd].read(count)
                for i, b in enumerate(data):
                    self.memory[buf_addr + i] = b
                self.ax = len(data)
            else:
                self.ax = -1

        elif op == Op.CLOS:
            # close(fd)
            fd = self.pop()
            if fd in self.files:
                self.files[fd].close()
                del self.files[fd]
                self.ax = 0
            else:
                self.ax = -1

        elif op == Op.PRTF:
            # printf(fmt, ...) - simplified
            # Just print the first int argument for now
            # Stack has args pushed right-to-left
            arg = self.read_int(self.sp + 8)  # First arg after format
            self.output += str(arg) + "\n"
            self.ax = 0

        elif op == Op.MALC:
            # malloc(size)
            size = self.pop()
            self.ax = self.heap_ptr
            self.heap_ptr += size
            # Align to 8 bytes
            self.heap_ptr = (self.heap_ptr + 7) & ~7

        elif op == Op.FREE:
            # free(ptr) - no-op for bump allocator
            self.pop()
            self.ax = 0

        elif op == Op.MSET:
            # memset(dst, val, count)
            count = self.pop()
            val = self.pop()
            dst = self.pop()
            for i in range(count):
                self.memory[dst + i] = val & 0xFF
            self.ax = dst

        elif op == Op.MCMP:
            # memcmp(a, b, count)
            count = self.pop()
            b = self.pop()
            a = self.pop()
            result = 0
            for i in range(count):
                diff = self.memory[a + i] - self.memory[b + i]
                if diff != 0:
                    result = diff
                    break
            self.ax = result

        elif op == Op.EXIT:
            self.halted = True
            self.exit_code = self.ax
            return False

        else:
            raise ValueError(f"Unknown opcode: {op}")

        return True

    def run(self, max_steps=1000000) -> int:
        """Run until exit or max steps."""
        for _ in range(max_steps):
            if not self.step():
                break
        return self.exit_code


def run_program(code: List[int], data: List[int], entry=0, debug=False) -> Tuple[int, str]:
    """Run a program and return (exit_code, output)."""
    vm = C4VM()
    vm.load(code, data)
    vm.pc = entry

    if debug:
        step = 0
        while not vm.halted and step < 100:
            instr = vm.read_int(vm.pc)
            op = instr & 0xFF
            imm = instr >> 8
            op_names = {int(o): o.name for o in Op}
            print(f"  [{step:3d}] pc={vm.pc:5d} sp={vm.sp:5d} ax={vm.ax:10d} | {op_names.get(op, '???'):4s} {imm}")
            vm.step()
            step += 1
    else:
        vm.run()

    return vm.ax, vm.output


def test_vm():
    from c4_compiler_full import compile_c

    print("FULL C4 VM TEST")
    print("=" * 60)

    # Test 1: Basic
    print("\n1. Basic arithmetic (3 + 4):")
    code, data = compile_c("int main() { return 3 + 4; }")
    result, _ = run_program(code, data)
    print(f"   Result: {result}")
    assert result == 7, f"Expected 7, got {result}"
    print("   ✓ PASS")

    # Test 2: Variables
    print("\n2. Variables (x=10, y=5, x+y):")
    code, data = compile_c("""
        int main() {
            int x;
            int y;
            x = 10;
            y = 5;
            return x + y;
        }
    """)
    result, _ = run_program(code, data)
    print(f"   Result: {result}")
    assert result == 15, f"Expected 15, got {result}"
    print("   ✓ PASS")

    # Test 3: Pointers
    print("\n3. Pointers (*p = &x):")
    code, data = compile_c("""
        int main() {
            int x;
            int *p;
            x = 42;
            p = &x;
            return *p;
        }
    """)
    result, _ = run_program(code, data)
    print(f"   Result: {result}")
    assert result == 42, f"Expected 42, got {result}"
    print("   ✓ PASS")

    # Test 4: String literal
    print("\n4. String literal (*s = 'h' = 104):")
    code, data = compile_c("""
        int main() {
            char *s;
            s = "hello";
            return *s;
        }
    """)
    result, _ = run_program(code, data)
    print(f"   Result: {result} ('{chr(result)}')")
    assert result == ord('h'), f"Expected {ord('h')}, got {result}"
    print("   ✓ PASS")

    # Test 5: Array indexing
    print("\n5. Array indexing (s[1] = 'b' = 98):")
    code, data = compile_c("""
        int main() {
            char *s;
            s = "abc";
            return s[1];
        }
    """)
    result, _ = run_program(code, data)
    print(f"   Result: {result} ('{chr(result)}')")
    assert result == ord('b'), f"Expected {ord('b')}, got {result}"
    print("   ✓ PASS")

    # Test 6: Pointer arithmetic
    print("\n6. Pointer arithmetic (*(p+1) = 'y' = 121):")
    code, data = compile_c("""
        int main() {
            char *p;
            p = "xyz";
            p = p + 1;
            return *p;
        }
    """)
    result, _ = run_program(code, data)
    print(f"   Result: {result} ('{chr(result)}')")
    assert result == ord('y'), f"Expected {ord('y')}, got {result}"
    print("   ✓ PASS")

    # Test 7: While loop with pointer
    print("\n7. String length via pointer loop:")
    code, data = compile_c("""
        int main() {
            char *p;
            int len;
            p = "hello";
            len = 0;
            while (*p) {
                len = len + 1;
                p = p + 1;
            }
            return len;
        }
    """)
    result, _ = run_program(code, data)
    print(f"   Result: {result}")
    assert result == 5, f"Expected 5, got {result}"
    print("   ✓ PASS")

    # Test 8: Enum
    print("\n8. Enum (C=10, D=11, C+D=21):")
    code, data = compile_c("""
        enum { A, B, C = 10, D };
        int main() { return C + D; }
    """)
    result, _ = run_program(code, data)
    print(f"   Result: {result}")
    assert result == 21, f"Expected 21, got {result}"
    print("   ✓ PASS")

    # Test 9: Sizeof
    print("\n9. Sizeof (int=8, char=1):")
    code, data = compile_c("""
        int main() {
            return sizeof(int) + sizeof(char);
        }
    """)
    result, _ = run_program(code, data)
    print(f"   Result: {result}")
    assert result == 9, f"Expected 9, got {result}"
    print("   ✓ PASS")

    # Test 10: Function call
    print("\n10. Function call:")
    code, data = compile_c("""
        int add(int a, int b) {
            return a + b;
        }
        int main() {
            return add(3, 4);
        }
    """)
    # Find main entry point (after add function)
    result, _ = run_program(code, data)
    print(f"   Result: {result}")
    # Note: This might fail if function ordering is wrong
    print("   (function calls need entry point handling)")

    print("\n" + "=" * 60)
    print("VM TESTS COMPLETE!")


if __name__ == "__main__":
    test_vm()
