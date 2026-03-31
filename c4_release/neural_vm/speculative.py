"""
Speculative decoding for the autoregressive Neural VM.

DraftVM executes C4 bytecode via Python arithmetic and produces 35 draft
tokens per step — the same format the transformer generates. A single
forward pass through the transformer verifies all 35 tokens at once,
giving a 35x reduction in forward passes per step.

Token format per VM step (35 tokens):
    REG_PC  + 4 value bytes     (5 tokens)
    REG_AX  + 4 value bytes     (5 tokens)
    REG_SP  + 4 value bytes     (5 tokens)
    REG_BP  + 4 value bytes     (5 tokens)
    STACK0  + 4 value bytes     (5 tokens)  — value at *SP (stack top)
    MEM     + 4 addr + 4 value  (9 tokens)
    STEP_END or HALT            (1 token)
"""

from .vm_step import Token
from .constants import PC_OFFSET, INSTR_WIDTH, pc_to_idx, idx_to_pc, STACK_INIT


def _le_bytes(val):
    """32-bit value -> 4 little-endian byte tokens."""
    v = val & 0xFFFFFFFF
    return [(v >> (i * 8)) & 0xFF for i in range(4)]


def _binop(op, a, b):
    """Execute binary op (pop OP AX), return 32-bit result."""
    if op == 14:    # OR
        return (a | b) & 0xFFFFFFFF
    elif op == 15:  # XOR
        return (a ^ b) & 0xFFFFFFFF
    elif op == 16:  # AND
        return (a & b) & 0xFFFFFFFF
    elif op == 17:  # EQ
        return 1 if a == b else 0
    elif op == 18:  # NE
        return 1 if a != b else 0
    elif op == 19:  # LT
        return 1 if a < b else 0
    elif op == 20:  # GT
        return 1 if a > b else 0
    elif op == 21:  # LE
        return 1 if a <= b else 0
    elif op == 22:  # GE
        return 1 if a >= b else 0
    elif op == 23:  # SHL
        return (a << b) & 0xFFFFFFFF
    elif op == 24:  # SHR
        return (a >> b) & 0xFFFFFFFF
    elif op == 25:  # ADD
        return (a + b) & 0xFFFFFFFF
    elif op == 26:  # SUB
        return (a - b) & 0xFFFFFFFF
    elif op == 27:  # MUL
        return (a * b) & 0xFFFFFFFF
    elif op == 28:  # DIV
        return (a // b) & 0xFFFFFFFF if b != 0 else 0
    elif op == 29:  # MOD
        return (a % b) & 0xFFFFFFFF if b != 0 else 0
    else:
        raise ValueError(f"Unknown binary op: {op}")


class DraftVM:
    """Lightweight C4 VM for speculative token prediction.

    Executes one instruction per step(), producing 35 draft tokens
    matching the autoregressive transformer's output format.
    """

    def __init__(self, bytecode):
        self.code = bytecode  # list of packed instructions
        self.idx = 0          # instruction index
        self.pc = PC_OFFSET   # Initial PC uses configured offset
        self.ax = 0
        self.sp = STACK_INIT  # Initialize stack pointer to standard location
        self.bp = STACK_INIT  # Initialize base pointer to standard location
        self.memory = {}      # sparse memory dict (addr -> 32-bit value)
        self._mem_bytes = {}  # byte-level memory for unified memory execution
        self.halted = False
        self._last_mem_addr = 0
        self._last_mem_val = 0
        self.output = []      # Output buffer for printf
        self.stdin_buf = ""   # Input buffer for read
        self.stdin_pos = 0    # Position in stdin buffer

    def load_data(self, data):
        """Load data section into memory at standard address 0x10000."""
        if data:
            for i, b in enumerate(data):
                self.memory[0x10000 + i] = b

    def _mem_read(self, addr):
        """Read 32-bit value from memory (ZFOD: uninitialized -> 0)."""
        return self.memory.get(addr & 0xFFFFFFFF, 0)

    def _mem_write(self, addr, val):
        """Write 32-bit value to memory. Track last write for MEM field."""
        addr = addr & 0xFFFFFFFF
        val = val & 0xFFFFFFFF
        self.memory[addr] = val
        self._last_mem_addr = addr
        self._last_mem_val = val

    def _read_string(self, addr):
        """Read null-terminated string from memory."""
        chars = []
        while True:
            # Read byte at address
            word = self._mem_read(addr)
            byte_val = word & 0xFF
            if byte_val == 0:
                break
            chars.append(chr(byte_val))
            addr += 1
            if len(chars) > 10000:  # Safety limit
                break
        return ''.join(chars)

    def _handle_printf(self):
        """Handle PRTF syscall - parse format string and output.

        Stack layout when PRTF is called:
        SP+0: arg_n
        SP+8: arg_n-1
        ...
        SP+(n-1)*8: arg_1
        SP+n*8: format_string_pointer

        After PRTF, ADJ will clean up the stack.
        """
        # Peek ahead to see if next instruction is ADJ to get argc
        next_idx = self.idx
        argc = 1  # At least format string
        if next_idx < len(self.code):
            next_instr = self.code[next_idx]
            next_op = next_instr & 0xFF
            if next_op == 7:  # ADJ
                next_imm = next_instr >> 8
                if next_imm >= 0x800000:
                    next_imm -= 0x1000000
                argc = next_imm // 8

        # Format string is at SP + (argc-1)*8
        fmt_addr = self._mem_read(self.sp + (argc - 1) * 8)
        fmt = self._read_string(fmt_addr)

        # Parse format string and extract arguments
        result = []
        arg_idx = 1
        i = 0
        while i < len(fmt):
            if fmt[i] == '\\' and i + 1 < len(fmt):
                if fmt[i + 1] == 'n':
                    result.append('\n')
                    i += 2
                elif fmt[i + 1] == 't':
                    result.append('\t')
                    i += 2
                elif fmt[i + 1] == '\\':
                    result.append('\\')
                    i += 2
                else:
                    result.append(fmt[i])
                    i += 1
            elif fmt[i] == '%' and i + 1 < len(fmt):
                spec = fmt[i + 1]
                if spec == '%':
                    result.append('%')
                    i += 2
                    continue

                # Get argument from stack
                if arg_idx < argc:
                    val = self._mem_read(self.sp + (argc - 1 - arg_idx) * 8)
                else:
                    val = 0

                if spec == 'd':
                    # Handle signed integer
                    if val > 0x7FFFFFFF:
                        val = val - 0x100000000
                    result.append(str(val))
                elif spec == 'x':
                    result.append(format(val & 0xFFFFFFFF, 'x'))
                elif spec == 'c':
                    result.append(chr(val & 0xFF))
                elif spec == 's':
                    result.append(self._read_string(val))
                else:
                    result.append('%' + spec)

                arg_idx += 1
                i += 2
            else:
                result.append(fmt[i])
                i += 1

        self.output.append(''.join(result))
        self.ax = 0  # printf returns 0

    def _handle_read(self):
        """Handle READ syscall.

        read(fd, buf, count)
        SP+0: count
        SP+8: buf
        SP+16: fd
        """
        fd = self._mem_read(self.sp + 16)
        buf = self._mem_read(self.sp + 8)
        count = self._mem_read(self.sp)

        if fd == 0:  # stdin
            bytes_read = 0
            for i in range(count):
                if self.stdin_pos < len(self.stdin_buf):
                    byte_val = ord(self.stdin_buf[self.stdin_pos])
                    self._mem_write(buf + i, byte_val)
                    self.stdin_pos += 1
                    bytes_read += 1
                else:
                    break
            self.ax = bytes_read
        else:
            self.ax = -1  # Unsupported fd

    def _fetch_instr_from_memory(self, pc):
        """Fetch instruction from memory at PC address (unified memory).

        Returns (opcode, imm) tuple, or None if memory not initialized.
        Instruction format: opcode byte + 3 immediate bytes (little-endian).
        """
        op = self._mem_bytes.get(pc)
        if op is None:
            return None
        imm0 = self._mem_bytes.get(pc + 1, 0)
        imm1 = self._mem_bytes.get(pc + 2, 0)
        imm2 = self._mem_bytes.get(pc + 3, 0)
        imm = imm0 | (imm1 << 8) | (imm2 << 16)
        # Sign extend 24-bit immediate
        if imm >= 0x800000:
            imm -= 0x1000000
        return (op, imm)

    def write_code_to_memory(self, addr, opcode, imm=0):
        """Write instruction to memory for unified memory execution.

        Args:
            addr: Byte address where instruction starts
            opcode: Opcode byte (0-255)
            imm: 24-bit signed immediate value
        """
        imm_unsigned = imm & 0xFFFFFF
        self._mem_bytes[addr] = opcode & 0xFF
        self._mem_bytes[addr + 1] = imm_unsigned & 0xFF
        self._mem_bytes[addr + 2] = (imm_unsigned >> 8) & 0xFF
        self._mem_bytes[addr + 3] = (imm_unsigned >> 16) & 0xFF

    def step(self):
        """Execute one instruction. Returns True if executed, False if halted/done."""
        if self.halted:
            return False

        # Check if executing from static code or from memory (unified memory)
        _from_memory = False
        if self.idx < len(self.code):
            # Fetch from static code array
            instr = self.code[self.idx]
            op = instr & 0xFF
            imm = instr >> 8
            # Sign extend 24-bit immediate
            if imm >= 0x800000:
                imm -= 0x1000000
            self.idx += 1
            self.pc = idx_to_pc(self.idx)  # default: advance to next instruction
        else:
            # Fetch from memory (unified memory execution)
            result = self._fetch_instr_from_memory(self.pc)
            if result is None:
                return False  # No code at this PC
            op, imm = result
            _from_memory = True
            self.pc += 4  # Advance PC by instruction size (4 bytes in memory)

        self._last_mem_addr = 0
        self._last_mem_val = 0

        if op == 0:    # LEA
            self.ax = (self.bp + imm) & 0xFFFFFFFF
        elif op == 1:  # IMM
            self.ax = imm & 0xFFFFFFFF
        elif op == 2:  # JMP
            # imm is byte address - check if within static code or memory
            target = imm & 0xFFFFFFFF
            target_idx = target // INSTR_WIDTH
            if target_idx < len(self.code):
                self.idx = target_idx
                self.pc = idx_to_pc(self.idx)
            else:
                # Jump to memory - use address directly
                self.idx = len(self.code)  # Mark as "in memory"
                self.pc = target
        elif op == 3:  # JSR
            self.sp = (self.sp - 8) & 0xFFFFFFFF
            self._mem_write(self.sp, self.pc)  # push return address
            # imm is byte address - check if within static code or memory
            target = imm & 0xFFFFFFFF
            target_idx = target // INSTR_WIDTH
            if target_idx < len(self.code):
                self.idx = target_idx
                self.pc = idx_to_pc(self.idx)
            else:
                # Jump to memory - use address directly
                self.idx = len(self.code)
                self.pc = target
        elif op == 4:  # BZ
            if self.ax == 0:
                # imm is byte address - check if within static code or memory
                target = imm & 0xFFFFFFFF
                target_idx = target // INSTR_WIDTH
                if target_idx < len(self.code):
                    self.idx = target_idx
                    self.pc = idx_to_pc(self.idx)
                else:
                    self.idx = len(self.code)
                    self.pc = target
        elif op == 5:  # BNZ
            if self.ax != 0:
                # imm is byte address - check if within static code or memory
                target = imm & 0xFFFFFFFF
                target_idx = target // INSTR_WIDTH
                if target_idx < len(self.code):
                    self.idx = target_idx
                    self.pc = idx_to_pc(self.idx)
                else:
                    self.idx = len(self.code)
                    self.pc = target
        elif op == 6:  # ENT
            self.sp = (self.sp - 8) & 0xFFFFFFFF
            self._mem_write(self.sp, self.bp)
            self.bp = self.sp
            self.sp = (self.sp - imm) & 0xFFFFFFFF
        elif op == 7:  # ADJ
            self.sp = (self.sp + imm) & 0xFFFFFFFF
        elif op == 8:  # LEV
            self.sp = self.bp
            self.bp = self._mem_read(self.sp)
            self.sp = (self.sp + 8) & 0xFFFFFFFF
            ret_addr = self._mem_read(self.sp)
            self.sp = (self.sp + 8) & 0xFFFFFFFF
            self.pc = ret_addr & 0xFFFFFFFF
            self.idx = pc_to_idx(ret_addr)
        elif op == 9:  # LI
            self.ax = self._mem_read(self.ax)
        elif op == 11:  # SI
            addr = self._mem_read(self.sp)
            self.sp = (self.sp + 8) & 0xFFFFFFFF
            self._mem_write(addr, self.ax)
        elif op == 13:  # PSH
            self.sp = (self.sp - 8) & 0xFFFFFFFF
            self._mem_write(self.sp, self.ax)
        # Binary ops that pop stack
        elif 14 <= op <= 29:
            a = self._mem_read(self.sp)
            self.sp = (self.sp + 8) & 0xFFFFFFFF
            self.ax = _binop(op, a, self.ax)
        elif op == 31:  # READ
            self._handle_read()
        elif op == 33:  # PRTF
            self._handle_printf()
        elif op == 38:  # EXIT
            self.halted = True
        # NOP and other unhandled ops: do nothing (advance PC only)

        return True

    def get_output(self):
        """Get accumulated output."""
        return ''.join(self.output)

    def set_stdin(self, data):
        """Set stdin buffer for READ operations."""
        self.stdin_buf = data
        self.stdin_pos = 0

    def draft_tokens(self):
        """Encode current state as 35 tokens."""
        tokens = []
        tokens.append(Token.REG_PC)
        tokens.extend(_le_bytes(self.pc))
        tokens.append(Token.REG_AX)
        tokens.extend(_le_bytes(self.ax))
        tokens.append(Token.REG_SP)
        tokens.extend(_le_bytes(self.sp))
        tokens.append(Token.REG_BP)
        tokens.extend(_le_bytes(self.bp))
        tokens.append(Token.STACK0)
        tokens.extend(_le_bytes(self._mem_read(self.sp)))
        tokens.append(Token.MEM)
        tokens.extend(_le_bytes(self._last_mem_addr))
        tokens.extend(_le_bytes(self._last_mem_val))
        tokens.append(Token.HALT if self.halted else Token.STEP_END)
        assert len(tokens) == 35
        return tokens
