#!/usr/bin/env python3
"""
Tool-Use Based I/O for C4 VM

Implements file I/O and user interaction via "tool calls" that pause
VM execution and request external action.

This enables the VM to:
1. Read/write files via tool calls
2. Get user input interactively
3. Printf formatted output
4. Act as an "agentic VM" that can interact with external systems

Tool Call Architecture:
- VM executes until it hits an I/O opcode
- VM generates a ToolCall request with operation details
- External handler (Python, MCP server, etc.) processes the request
- Handler provides response back to VM
- VM continues execution with the result

This is similar to how LLMs use tool calls - the VM "pauses" and
requests external action, then resumes with the result.
"""

import sys
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')
from neural_vm import Opcode


# Extended opcodes used by the compiler (src/compiler.py)
class ExtendedOpcode:
    """Extended opcodes from the compiler - matches src/compiler.py Op class."""
    # Standard opcodes (same as Opcode)
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

    # System calls (from compiler.py - exact values)
    OPEN = 30
    READ = 31
    CLOS = 32
    PRTF = 33
    MALC = 34
    FREE = 35
    MSET = 36
    MCMP = 37
    EXIT = 38

    # I/O system calls
    GETCHAR = 64
    PUTCHAR = 65


class ToolCallType(Enum):
    """Types of tool calls the VM can make."""
    FILE_OPEN = "file_open"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_CLOSE = "file_close"
    USER_INPUT = "user_input"      # GETC - get single char
    USER_INPUT_LINE = "user_input_line"  # Read line
    PRINTF = "printf"              # Formatted print
    PUTCHAR = "putchar"            # Single char output
    MALLOC = "malloc"              # Memory allocation
    EXIT = "exit"                  # Program exit


@dataclass
class ToolCall:
    """A tool call request from the VM."""
    call_type: ToolCallType
    call_id: int
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "type": self.call_type.value,
            "id": self.call_id,
            "params": self.params
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ToolResponse:
    """Response to a tool call."""
    call_id: int
    success: bool
    result: Any = None
    error: Optional[str] = None


class FileHandle:
    """Virtual file handle."""
    def __init__(self, path: str, mode: str, fd: int):
        self.path = path
        self.mode = mode
        self.fd = fd
        self.position = 0
        self.content: bytes = b""
        self.is_open = True


class ToolUseIOHandler:
    """
    Handles tool calls from the VM.

    Can be extended to:
    - Use actual file system
    - Connect to MCP servers
    - Prompt user for input
    - Send to external APIs
    """

    def __init__(self,
                 allow_file_io: bool = False,
                 file_sandbox: Optional[str] = None,
                 input_callback: Optional[Callable[[], str]] = None,
                 output_callback: Optional[Callable[[str], None]] = None):
        """
        Args:
            allow_file_io: Whether to allow real file I/O
            file_sandbox: Directory to sandbox file operations
            input_callback: Function to get user input (default: input())
            output_callback: Function to output text (default: print())
        """
        self.allow_file_io = allow_file_io
        self.file_sandbox = file_sandbox
        self.input_callback = input_callback or input
        self.output_callback = output_callback or print

        # Virtual file system
        self.files: Dict[int, FileHandle] = {}
        self.next_fd = 3  # 0=stdin, 1=stdout, 2=stderr

        # Output buffer
        self.stdout_buffer: List[int] = []
        self.stderr_buffer: List[int] = []

        # Input buffer for character-by-character reading
        self.input_buffer: List[int] = []

        # Tool call history
        self.call_history: List[ToolCall] = []
        self.response_history: List[ToolResponse] = []

    def handle(self, call: ToolCall) -> ToolResponse:
        """Process a tool call and return response."""
        self.call_history.append(call)

        handlers = {
            ToolCallType.FILE_OPEN: self._handle_file_open,
            ToolCallType.FILE_READ: self._handle_file_read,
            ToolCallType.FILE_WRITE: self._handle_file_write,
            ToolCallType.FILE_CLOSE: self._handle_file_close,
            ToolCallType.USER_INPUT: self._handle_user_input,
            ToolCallType.USER_INPUT_LINE: self._handle_user_input_line,
            ToolCallType.PRINTF: self._handle_printf,
            ToolCallType.PUTCHAR: self._handle_putchar,
            ToolCallType.MALLOC: self._handle_malloc,
            ToolCallType.EXIT: self._handle_exit,
        }

        handler = handlers.get(call.call_type)
        if handler:
            response = handler(call)
        else:
            response = ToolResponse(call.call_id, False, error=f"Unknown call type: {call.call_type}")

        self.response_history.append(response)
        return response

    def _handle_file_open(self, call: ToolCall) -> ToolResponse:
        """Handle file open request."""
        path = call.params.get("path", "")
        mode = call.params.get("mode", "r")

        # Create virtual file handle
        fd = self.next_fd
        self.next_fd += 1

        handle = FileHandle(path, mode, fd)

        if self.allow_file_io and self.file_sandbox:
            # Real file I/O (sandboxed)
            import os
            full_path = os.path.join(self.file_sandbox, path)
            try:
                if "r" in mode:
                    with open(full_path, "rb") as f:
                        handle.content = f.read()
                self.files[fd] = handle
                return ToolResponse(call.call_id, True, result=fd)
            except Exception as e:
                return ToolResponse(call.call_id, False, error=str(e))
        else:
            # Virtual file (in-memory)
            self.files[fd] = handle
            return ToolResponse(call.call_id, True, result=fd)

    def _handle_file_read(self, call: ToolCall) -> ToolResponse:
        """Handle file read request."""
        fd = call.params.get("fd", 0)
        size = call.params.get("size", 1)

        if fd == 0:  # stdin
            # Get input from user
            data = self.input_callback()
            return ToolResponse(call.call_id, True, result=data.encode()[:size])

        handle = self.files.get(fd)
        if not handle or not handle.is_open:
            return ToolResponse(call.call_id, False, error="Invalid file descriptor")

        # Read from position
        data = handle.content[handle.position:handle.position + size]
        handle.position += len(data)
        return ToolResponse(call.call_id, True, result=data)

    def _handle_file_write(self, call: ToolCall) -> ToolResponse:
        """Handle file write request."""
        fd = call.params.get("fd", 1)
        data = call.params.get("data", b"")

        if isinstance(data, str):
            data = data.encode()

        if fd == 1:  # stdout
            self.stdout_buffer.extend(data)
            self.output_callback(data.decode(errors='replace'))
            return ToolResponse(call.call_id, True, result=len(data))
        elif fd == 2:  # stderr
            self.stderr_buffer.extend(data)
            return ToolResponse(call.call_id, True, result=len(data))

        handle = self.files.get(fd)
        if not handle or not handle.is_open:
            return ToolResponse(call.call_id, False, error="Invalid file descriptor")

        # Write to content
        if isinstance(handle.content, bytes):
            handle.content = bytearray(handle.content)
        handle.content[handle.position:handle.position + len(data)] = data
        handle.position += len(data)

        return ToolResponse(call.call_id, True, result=len(data))

    def _handle_file_close(self, call: ToolCall) -> ToolResponse:
        """Handle file close request."""
        fd = call.params.get("fd", 0)

        handle = self.files.get(fd)
        if not handle:
            return ToolResponse(call.call_id, False, error="Invalid file descriptor")

        handle.is_open = False

        # Write back if needed
        if self.allow_file_io and self.file_sandbox and "w" in handle.mode:
            import os
            full_path = os.path.join(self.file_sandbox, handle.path)
            try:
                with open(full_path, "wb") as f:
                    f.write(bytes(handle.content))
            except Exception as e:
                return ToolResponse(call.call_id, False, error=str(e))

        return ToolResponse(call.call_id, True, result=0)

    def _handle_user_input(self, call: ToolCall) -> ToolResponse:
        """Handle single character input (GETC)."""
        prompt = call.params.get("prompt", "")

        if prompt:
            self.output_callback(prompt)

        # If we have buffered input, return next character
        if self.input_buffer:
            char = self.input_buffer.pop(0)
            return ToolResponse(call.call_id, True, result=char)

        # Otherwise, get new input and buffer it
        try:
            line = self.input_callback()
            if line:
                # Buffer all characters from the line
                for c in line:
                    self.input_buffer.append(ord(c))
                # Return first character
                if self.input_buffer:
                    char = self.input_buffer.pop(0)
                    return ToolResponse(call.call_id, True, result=char)
            return ToolResponse(call.call_id, True, result=-1)  # EOF
        except EOFError:
            return ToolResponse(call.call_id, True, result=-1)

    def _handle_user_input_line(self, call: ToolCall) -> ToolResponse:
        """Handle line input."""
        prompt = call.params.get("prompt", "")

        if prompt:
            self.output_callback(prompt)

        try:
            line = self.input_callback()
            return ToolResponse(call.call_id, True, result=line)
        except EOFError:
            return ToolResponse(call.call_id, True, result="")

    def _handle_printf(self, call: ToolCall) -> ToolResponse:
        """Handle printf request."""
        format_str = call.params.get("format", "")
        args = call.params.get("args", [])

        try:
            # Simple printf formatting
            output = self._format_printf(format_str, args)
            self.stdout_buffer.extend(output.encode())
            self.output_callback(output)
            return ToolResponse(call.call_id, True, result=len(output))
        except Exception as e:
            return ToolResponse(call.call_id, False, error=str(e))

    def _format_printf(self, fmt: str, args: List[Any]) -> str:
        """Simple printf-style formatting."""
        result = []
        i = 0
        arg_idx = 0

        while i < len(fmt):
            if fmt[i] == '%' and i + 1 < len(fmt):
                spec = fmt[i + 1]
                if spec == 'd' or spec == 'i':
                    result.append(str(args[arg_idx] if arg_idx < len(args) else 0))
                    arg_idx += 1
                elif spec == 's':
                    result.append(str(args[arg_idx] if arg_idx < len(args) else ""))
                    arg_idx += 1
                elif spec == 'c':
                    val = args[arg_idx] if arg_idx < len(args) else 0
                    result.append(chr(val) if isinstance(val, int) else str(val))
                    arg_idx += 1
                elif spec == 'x':
                    result.append(hex(args[arg_idx] if arg_idx < len(args) else 0)[2:])
                    arg_idx += 1
                elif spec == '%':
                    result.append('%')
                else:
                    result.append('%' + spec)
                i += 2
            elif fmt[i] == '\\' and i + 1 < len(fmt):
                if fmt[i + 1] == 'n':
                    result.append('\n')
                elif fmt[i + 1] == 't':
                    result.append('\t')
                else:
                    result.append(fmt[i + 1])
                i += 2
            else:
                result.append(fmt[i])
                i += 1

        return ''.join(result)

    def _handle_putchar(self, call: ToolCall) -> ToolResponse:
        """Handle putchar request."""
        char = call.params.get("char", 0)

        if isinstance(char, int):
            self.stdout_buffer.append(char & 0xFF)
            self.output_callback(chr(char & 0xFF))
        else:
            self.stdout_buffer.append(ord(char[0]))
            self.output_callback(char[0])

        return ToolResponse(call.call_id, True, result=char)

    def _handle_malloc(self, call: ToolCall) -> ToolResponse:
        """Handle malloc request (returns address in VM's heap)."""
        size = call.params.get("size", 0)
        # This is handled by the VM itself, just acknowledge
        return ToolResponse(call.call_id, True, result=size)

    def _handle_exit(self, call: ToolCall) -> ToolResponse:
        """Handle exit request."""
        code = call.params.get("code", 0)
        return ToolResponse(call.call_id, True, result=code)

    def get_stdout(self) -> bytes:
        """Get accumulated stdout."""
        return bytes(self.stdout_buffer)

    def get_stderr(self) -> bytes:
        """Get accumulated stderr."""
        return bytes(self.stderr_buffer)


class ToolUseVM:
    """
    VM that uses tool calls for I/O operations.

    When an I/O opcode is encountered, the VM pauses and
    generates a ToolCall. The caller must provide the response
    before execution can continue.
    """

    def __init__(self, io_handler: Optional[ToolUseIOHandler] = None):
        self.io_handler = io_handler or ToolUseIOHandler()
        self.reset()

    def reset(self):
        self.memory: Dict[int, int] = {}
        self.sp = 0x10000
        self.bp = 0x10000
        self.ax = 0
        self.pc = 0
        self.halted = False
        self.code: List[tuple] = []
        self.heap = 0x20000

        # Pending tool call (if any)
        self.pending_call: Optional[ToolCall] = None
        self.call_counter = 0

    def load(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load bytecode."""
        self.code = []
        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

        if data:
            for i, b in enumerate(data):
                self.memory[0x10000 + i] = b

    def step(self) -> Optional[ToolCall]:
        """
        Execute one instruction.

        Returns ToolCall if an I/O operation needs handling,
        None otherwise.
        """
        if self.halted or self.pending_call:
            return self.pending_call

        pc_idx = self.pc // 8
        if pc_idx >= len(self.code):
            self.halted = True
            return None

        op, imm = self.code[pc_idx]
        self.pc += 8

        # Standard opcodes (no I/O)
        if op == Opcode.LEA:
            self.ax = self.bp + imm
        elif op == Opcode.IMM:
            self.ax = imm
        elif op == Opcode.JMP:
            self.pc = imm
        elif op == Opcode.JSR:
            self.sp -= 8
            self.memory[self.sp] = self.pc
            self.pc = imm
        elif op == Opcode.BZ:
            if self.ax == 0:
                self.pc = imm
        elif op == Opcode.BNZ:
            if self.ax != 0:
                self.pc = imm
        elif op == Opcode.ENT:
            self.sp -= 8
            self.memory[self.sp] = self.bp
            self.bp = self.sp
            self.sp -= imm
        elif op == Opcode.ADJ:
            self.sp += imm
        elif op == Opcode.LEV:
            self.sp = self.bp
            self.bp = self.memory.get(self.sp, 0)
            self.sp += 8
            self.pc = self.memory.get(self.sp, 0)
            self.sp += 8
        elif op == Opcode.LI:
            self.ax = self.memory.get(self.ax, 0)
        elif op == Opcode.LC:
            self.ax = self.memory.get(self.ax, 0) & 0xFF
        elif op == Opcode.SI:
            addr = self.memory.get(self.sp, 0)
            self.sp += 8
            self.memory[addr] = self.ax
        elif op == Opcode.SC:
            addr = self.memory.get(self.sp, 0)
            self.sp += 8
            self.memory[addr] = self.ax & 0xFF
        elif op == Opcode.PSH:
            self.sp -= 8
            self.memory[self.sp] = self.ax
        elif op == Opcode.AND:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a & self.ax
        elif op == Opcode.OR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a | self.ax
        elif op == Opcode.XOR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a ^ self.ax
        elif op == Opcode.EQ:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = 1 if a == self.ax else 0
        elif op == Opcode.NE:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = 1 if a != self.ax else 0
        elif op == Opcode.LT:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            # Signed comparison - treat as 32-bit signed integers
            a_signed = a if a < 0x80000000 else a - 0x100000000
            b_signed = self.ax if self.ax < 0x80000000 else self.ax - 0x100000000
            self.ax = 1 if a_signed < b_signed else 0
        elif op == Opcode.GT:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            # Signed comparison - treat as 32-bit signed integers
            a_signed = a if a < 0x80000000 else a - 0x100000000
            b_signed = self.ax if self.ax < 0x80000000 else self.ax - 0x100000000
            self.ax = 1 if a_signed > b_signed else 0
        elif op == Opcode.LE:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            # Signed comparison - treat as 32-bit signed integers
            a_signed = a if a < 0x80000000 else a - 0x100000000
            b_signed = self.ax if self.ax < 0x80000000 else self.ax - 0x100000000
            self.ax = 1 if a_signed <= b_signed else 0
        elif op == Opcode.GE:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            # Signed comparison - treat as 32-bit signed integers
            a_signed = a if a < 0x80000000 else a - 0x100000000
            b_signed = self.ax if self.ax < 0x80000000 else self.ax - 0x100000000
            self.ax = 1 if a_signed >= b_signed else 0
        elif op == Opcode.SHL:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a << self.ax) & 0xFFFFFFFF
        elif op == Opcode.SHR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a >> self.ax
        elif op == Opcode.ADD:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a + self.ax) & 0xFFFFFFFF
        elif op == Opcode.SUB:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a - self.ax) & 0xFFFFFFFF
        elif op == Opcode.MUL:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a * self.ax) & 0xFFFFFFFF
        elif op == Opcode.DIV:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a // self.ax if self.ax != 0 else 0
        elif op == Opcode.MOD:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a % self.ax if self.ax != 0 else 0

        # === I/O OPCODES - Generate Tool Calls ===
        elif op == Opcode.OPEN or op == ExtendedOpcode.OPEN:
            # Args on stack: mode, path_ptr
            mode = self.memory.get(self.sp, 0)
            self.sp += 8
            path_ptr = self.memory.get(self.sp, 0)
            self.sp += 8

            # Read path string from memory
            path = self._read_string(path_ptr)
            mode_str = "r" if mode == 0 else "w" if mode == 1 else "a"

            self.pending_call = ToolCall(
                ToolCallType.FILE_OPEN,
                self._next_call_id(),
                {"path": path, "mode": mode_str}
            )
            return self.pending_call

        elif op == Opcode.READ or op == ExtendedOpcode.READ:
            # Args: size, buf_ptr, fd
            size = self.memory.get(self.sp, 0)
            self.sp += 8
            buf_ptr = self.memory.get(self.sp, 0)
            self.sp += 8
            fd = self.memory.get(self.sp, 0)
            self.sp += 8

            self.pending_call = ToolCall(
                ToolCallType.FILE_READ,
                self._next_call_id(),
                {"fd": fd, "size": size, "buf_ptr": buf_ptr}
            )
            return self.pending_call

        elif op == Opcode.CLOS or op == ExtendedOpcode.CLOS:
            fd = self.memory.get(self.sp, 0)
            self.sp += 8

            self.pending_call = ToolCall(
                ToolCallType.FILE_CLOSE,
                self._next_call_id(),
                {"fd": fd}
            )
            return self.pending_call

        elif op == Opcode.PRTF or op == ExtendedOpcode.PRTF:
            # Printf: format_ptr on stack, args follow
            # For simplicity, we'll extract format and first few args
            format_ptr = self.memory.get(self.sp, 0)
            self.sp += 8

            format_str = self._read_string(format_ptr)

            # Count format specifiers to know how many args
            num_args = format_str.count('%') - format_str.count('%%')
            args = []
            for _ in range(num_args):
                args.append(self.memory.get(self.sp, 0))
                self.sp += 8

            self.pending_call = ToolCall(
                ToolCallType.PRINTF,
                self._next_call_id(),
                {"format": format_str, "args": args}
            )
            return self.pending_call

        elif op == Opcode.GETC or op == ExtendedOpcode.GETCHAR:
            self.pending_call = ToolCall(
                ToolCallType.USER_INPUT,
                self._next_call_id(),
                {}
            )
            return self.pending_call

        elif op == Opcode.PUTC or op == ExtendedOpcode.PUTCHAR:
            # Note: Don't pop arguments here - caller will use ADJ to clean up
            char = self.memory.get(self.sp, 0) & 0xFF

            self.pending_call = ToolCall(
                ToolCallType.PUTCHAR,
                self._next_call_id(),
                {"char": char}
            )
            return self.pending_call

        elif op == Opcode.MALC or op == ExtendedOpcode.MALC:
            # Note: Don't pop arguments here - caller will use ADJ to clean up
            size = self.memory.get(self.sp, 0)
            self.ax = self.heap
            self.heap += size
            # No tool call needed, just allocate

        elif op == Opcode.FREE or op == ExtendedOpcode.FREE:
            # No-op for now
            ptr = self.memory.get(self.sp, 0)
            self.sp += 8

        elif op == Opcode.MSET or op == ExtendedOpcode.MSET:
            # memset(ptr, val, size)
            size = self.memory.get(self.sp, 0)
            self.sp += 8
            val = self.memory.get(self.sp, 0) & 0xFF
            self.sp += 8
            ptr = self.memory.get(self.sp, 0)
            self.sp += 8

            for i in range(size):
                self.memory[ptr + i] = val
            self.ax = ptr

        elif op == Opcode.MCPY or op == 37:
            # memcpy(dst, src, size)
            size = self.memory.get(self.sp, 0)
            self.sp += 8
            src = self.memory.get(self.sp, 0)
            self.sp += 8
            dst = self.memory.get(self.sp, 0)
            self.sp += 8

            for i in range(size):
                self.memory[dst + i] = self.memory.get(src + i, 0)
            self.ax = dst

        elif op == Opcode.MCMP or op == ExtendedOpcode.MCMP:
            # memcmp(a, b, size)
            size = self.memory.get(self.sp, 0)
            self.sp += 8
            b_ptr = self.memory.get(self.sp, 0)
            self.sp += 8
            a_ptr = self.memory.get(self.sp, 0)
            self.sp += 8

            result = 0
            for i in range(size):
                a_val = self.memory.get(a_ptr + i, 0)
                b_val = self.memory.get(b_ptr + i, 0)
                if a_val != b_val:
                    result = a_val - b_val
                    break
            self.ax = result

        elif op == Opcode.EXIT or op == ExtendedOpcode.EXIT:
            code = self.memory.get(self.sp, 0) if self.sp in self.memory else self.ax
            self.halted = True
            self.pending_call = ToolCall(
                ToolCallType.EXIT,
                self._next_call_id(),
                {"code": code}
            )
            return self.pending_call

        return None

    def provide_response(self, response: ToolResponse):
        """Provide response to pending tool call."""
        if not self.pending_call or response.call_id != self.pending_call.call_id:
            raise ValueError("Response doesn't match pending call")

        call = self.pending_call
        self.pending_call = None

        if call.call_type == ToolCallType.FILE_OPEN:
            self.ax = response.result if response.success else -1

        elif call.call_type == ToolCallType.FILE_READ:
            if response.success:
                data = response.result
                buf_ptr = call.params["buf_ptr"]
                for i, b in enumerate(data):
                    self.memory[buf_ptr + i] = b if isinstance(b, int) else ord(b)
                self.ax = len(data)
            else:
                self.ax = -1

        elif call.call_type == ToolCallType.FILE_CLOSE:
            self.ax = 0 if response.success else -1

        elif call.call_type == ToolCallType.PRINTF:
            self.ax = response.result if response.success else -1

        elif call.call_type == ToolCallType.USER_INPUT:
            self.ax = response.result if response.success else -1

        elif call.call_type == ToolCallType.PUTCHAR:
            self.ax = response.result if response.success else -1

        elif call.call_type == ToolCallType.EXIT:
            pass  # Already halted

    def run(self, max_steps: int = 100000) -> int:
        """
        Run VM to completion, handling all I/O via tool calls.

        Returns:
            Exit code (value of AX at exit)
        """
        steps = 0

        while steps < max_steps and not self.halted:
            call = self.step()

            if call:
                # Handle the tool call
                response = self.io_handler.handle(call)
                self.provide_response(response)

            steps += 1

        return self.ax

    def _read_string(self, ptr: int, max_len: int = 1024) -> str:
        """Read null-terminated string from memory."""
        chars = []
        for i in range(max_len):
            c = self.memory.get(ptr + i, 0)
            if c == 0:
                break
            chars.append(chr(c))
        return ''.join(chars)

    def _next_call_id(self) -> int:
        self.call_counter += 1
        return self.call_counter


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tool-Use I/O VM Test")
    print("=" * 60)

    # Create VM with I/O handler
    output_buffer = []

    def capture_output(s):
        output_buffer.append(s)

    handler = ToolUseIOHandler(
        output_callback=capture_output,
        input_callback=lambda: "test input\n"
    )

    vm = ToolUseVM(handler)

    # Test 1: Simple computation (no I/O)
    print("\n--- Test 1: Simple computation ---")
    bytecode = [
        (Opcode.IMM << 0) | (5 << 8),   # IMM 5
        (Opcode.PSH << 0),               # PSH
        (Opcode.IMM << 0) | (3 << 8),   # IMM 3
        (Opcode.ADD << 0),               # ADD -> 8
        (Opcode.EXIT << 0),              # EXIT
    ]
    vm.reset()
    vm.load(bytecode)
    result = vm.run()
    print(f"  5 + 3 = {result}")
    assert result == 8, f"Expected 8, got {result}"
    print("  PASS")

    # Test 2: PUTC (output via tool call)
    print("\n--- Test 2: PUTC ---")
    output_buffer.clear()
    bytecode = [
        (Opcode.IMM << 0) | (ord('H') << 8),
        (Opcode.PSH << 0),
        (Opcode.PUTC << 0),
        (Opcode.IMM << 0) | (ord('i') << 8),
        (Opcode.PSH << 0),
        (Opcode.PUTC << 0),
        (Opcode.EXIT << 0),
    ]
    vm.reset()
    vm.load(bytecode)
    vm.run()
    print(f"  Output: {''.join(output_buffer)}")
    assert ''.join(output_buffer) == "Hi", f"Expected 'Hi', got {''.join(output_buffer)}"
    print("  PASS")

    # Test 3: Tool call tracking
    print("\n--- Test 3: Tool call tracking ---")
    print(f"  Total tool calls: {len(handler.call_history)}")
    print(f"  Call types: {[c.call_type.value for c in handler.call_history]}")

    # Test 4: File I/O
    print("\n--- Test 4: File open/close ---")
    handler.call_history.clear()

    # Store filename in memory
    filename = "test.txt"
    data = bytearray(1024)
    for i, c in enumerate(filename):
        data[i] = ord(c)
    data[len(filename)] = 0  # null terminator

    bytecode = [
        # open("test.txt", "r")
        (Opcode.IMM << 0) | (0 << 8),        # mode = 0 (read)
        (Opcode.PSH << 0),
        (Opcode.IMM << 0) | (0x10000 << 8),  # path ptr
        (Opcode.PSH << 0),
        (Opcode.OPEN << 0),                   # OPEN -> fd in AX
        (Opcode.PSH << 0),                    # save fd
        (Opcode.CLOS << 0),                   # CLOSE
        (Opcode.EXIT << 0),
    ]
    vm.reset()
    vm.load(bytecode, bytes(data))
    vm.run()

    file_calls = [c for c in handler.call_history if c.call_type in
                  (ToolCallType.FILE_OPEN, ToolCallType.FILE_CLOSE)]
    print(f"  File operations: {len(file_calls)}")
    for c in file_calls:
        print(f"    {c.call_type.value}: {c.params}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
