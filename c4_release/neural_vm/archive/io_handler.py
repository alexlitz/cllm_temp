"""
I/O Handler for Neural VM.

External handler that monitors embedding slots for I/O requests.
Supports two modes:
1. Streaming Mode: LLM-style with <NEED_INPUT/>, <PROGRAM_END/> markers
2. Tool-Use Mode: Agentic with TOOL_CALL:type:id:{params} protocol

The VM step function checks embedding slots after each instruction.
"""

import torch
import json
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .embedding import E, IOToolCallType


@dataclass
class ToolCall:
    """Represents a tool call from the VM."""
    call_type: str
    call_id: int
    params: Dict[str, Any]

    def to_string(self) -> str:
        """Format as TOOL_CALL:type:id:{params}"""
        return f"TOOL_CALL:{self.call_type}:{self.call_id}:{json.dumps(self.params)}"

    @classmethod
    def from_string(cls, s: str) -> Optional['ToolCall']:
        """Parse TOOL_CALL:type:id:{params} format."""
        if not s.startswith("TOOL_CALL:"):
            return None
        parts = s[10:].split(":", 2)
        if len(parts) != 3:
            return None
        try:
            return cls(
                call_type=parts[0],
                call_id=int(parts[1]),
                params=json.loads(parts[2])
            )
        except (ValueError, json.JSONDecodeError):
            return None


@dataclass
class ToolResponse:
    """Represents a tool response to the VM."""
    call_id: int
    success: bool
    result: Any

    def to_string(self) -> str:
        """Format as TOOL_RESPONSE:id:result"""
        return f"TOOL_RESPONSE:{self.call_id}:{json.dumps(self.result)}"

    @classmethod
    def from_string(cls, s: str) -> Optional['ToolResponse']:
        """Parse TOOL_RESPONSE:id:result format."""
        if not s.startswith("TOOL_RESPONSE:"):
            return None
        parts = s[14:].split(":", 1)
        if len(parts) != 2:
            return None
        try:
            return cls(
                call_id=int(parts[0]),
                success=True,
                result=json.loads(parts[1])
            )
        except (ValueError, json.JSONDecodeError):
            return None


class IOHandler:
    """
    External handler that monitors embedding slots for I/O requests.

    Usage:
        handler = IOHandler(mode='streaming')
        while not handler.finished:
            embedding = vm.step()
            output = handler.check_io(embedding)
            if output:
                if handler.waiting_for_input:
                    user_input = get_user_input()
                    handler.provide_input(ord(user_input[0]), embedding)
                else:
                    print(output, end='')
    """

    TOOL_TYPE_NAMES = {
        IOToolCallType.GETCHAR: "getchar",
        IOToolCallType.PUTCHAR: "putchar",
        IOToolCallType.EXIT: "exit",
        IOToolCallType.OPEN: "open",
        IOToolCallType.READ: "read",
        IOToolCallType.CLOSE: "close",
        IOToolCallType.PRINTF: "printf",
    }

    def __init__(self, mode: str = 'streaming'):
        """
        Initialize I/O handler.

        Args:
            mode: 'streaming' for LLM-style, 'tooluse' for agentic
        """
        self.mode = mode
        self.output_buffer = ""
        self.input_buffer = []
        self.input_pos = 0
        self.tool_call_id = 0
        self.waiting_for_input = False
        self.finished = False
        self.exit_code = 0
        self.pending_tool_call: Optional[ToolCall] = None

    def _decode_io_char(self, embedding: torch.Tensor) -> int:
        """Decode character from IO_CHAR slot."""
        # IO_CHAR contains character value at position 0
        char_val = int(round(embedding[0, 0, E.IO_CHAR].item()))
        return char_val & 0xFF  # Mask to 8 bits

    def _encode_io_char(self, embedding: torch.Tensor, char: int):
        """Write character to IO_CHAR slot."""
        # Write to position 0's IO_CHAR slot
        embedding[0, 0, E.IO_CHAR] = float(char & 0xFF)

    def check_io(self, embedding: torch.Tensor) -> Optional[str]:
        """
        Check embedding slots for I/O requests.

        Args:
            embedding: [batch, positions, dim] tensor

        Returns:
            String output (for streaming) or tool call string (for tooluse),
            or None if no I/O action needed.
        """
        result = None

        # Check IO_OUTPUT_READY (PUTCHAR)
        if embedding[0, 0, E.IO_OUTPUT_READY].item() > 0.5:
            char = self._decode_io_char(embedding)
            if self.mode == 'streaming':
                self.output_buffer += chr(char)
                result = chr(char)
            else:
                self.tool_call_id += 1
                self.pending_tool_call = ToolCall(
                    call_type="putchar",
                    call_id=self.tool_call_id,
                    params={"char": char}
                )
                result = self.pending_tool_call.to_string()
            # Clear the flag
            embedding[0, 0, E.IO_OUTPUT_READY] = 0.0
            embedding[0, 0, E.IO_TOOL_CALL_TYPE] = 0.0

        # Check IO_NEED_INPUT (GETCHAR needs input)
        elif embedding[0, 0, E.IO_NEED_INPUT].item() > 0.5:
            self.waiting_for_input = True
            if self.mode == 'streaming':
                result = "<NEED_INPUT/>"
            else:
                self.tool_call_id += 1
                self.pending_tool_call = ToolCall(
                    call_type="getchar",
                    call_id=self.tool_call_id,
                    params={}
                )
                result = self.pending_tool_call.to_string()

        # Check IO_PROGRAM_END (EXIT)
        elif embedding[0, 0, E.IO_PROGRAM_END].item() > 0.5:
            self.finished = True
            self.exit_code = int(embedding[0, 0, E.IO_EXIT_CODE].item())
            if self.mode == 'streaming':
                result = f"<PROGRAM_END code=\"{self.exit_code}\"/>"
            else:
                self.tool_call_id += 1
                self.pending_tool_call = ToolCall(
                    call_type="exit",
                    call_id=self.tool_call_id,
                    params={"code": self.exit_code}
                )
                result = self.pending_tool_call.to_string()

        # Check for other tool call types (OPEN, READ, CLOSE, PRINTF)
        tool_type = int(embedding[0, 0, E.IO_TOOL_CALL_TYPE].item())
        if tool_type > 0 and tool_type not in {IOToolCallType.GETCHAR,
                                                 IOToolCallType.PUTCHAR,
                                                 IOToolCallType.EXIT}:
            if self.mode == 'tooluse':
                self.tool_call_id += 1
                type_name = self.TOOL_TYPE_NAMES.get(tool_type, f"unknown_{tool_type}")
                self.pending_tool_call = ToolCall(
                    call_type=type_name,
                    call_id=self.tool_call_id,
                    params={}  # Params would be extracted from registers
                )
                result = self.pending_tool_call.to_string()
            # Clear the tool call type
            embedding[0, 0, E.IO_TOOL_CALL_TYPE] = 0.0

        return result

    def provide_input(self, char: int, embedding: torch.Tensor):
        """
        Write input character to embedding slots.

        Args:
            char: Character code (0-255)
            embedding: [batch, positions, dim] tensor
        """
        self._encode_io_char(embedding, char)
        embedding[0, 0, E.IO_INPUT_READY] = 1.0
        embedding[0, 0, E.IO_NEED_INPUT] = 0.0
        self.waiting_for_input = False

    def provide_input_string(self, s: str, embedding: torch.Tensor):
        """
        Buffer a string for input.

        Args:
            s: Input string
            embedding: [batch, positions, dim] tensor
        """
        self.input_buffer = list(s)
        self.input_pos = 0
        if self.input_buffer:
            self.provide_input(ord(self.input_buffer[0]), embedding)
            self.input_pos = 1

    def next_input_char(self, embedding: torch.Tensor) -> bool:
        """
        Provide next buffered input character.

        Returns True if character provided, False if buffer exhausted.
        """
        if self.input_pos < len(self.input_buffer):
            self.provide_input(ord(self.input_buffer[self.input_pos]), embedding)
            self.input_pos += 1
            return True
        return False

    def handle_tool_response(self, response: ToolResponse, embedding: torch.Tensor):
        """
        Handle a tool response from external system.

        Args:
            response: ToolResponse object
            embedding: [batch, positions, dim] tensor
        """
        # Write response value to IO_TOOL_RESPONSE
        if isinstance(response.result, int):
            embedding[0, 0, E.IO_TOOL_RESPONSE] = float(response.result)
        elif isinstance(response.result, str) and len(response.result) == 1:
            embedding[0, 0, E.IO_TOOL_RESPONSE] = float(ord(response.result))
            # Also set as input character for GETCHAR
            self.provide_input(ord(response.result), embedding)

        # Clear pending tool call
        self.pending_tool_call = None


class StreamingIOHandler(IOHandler):
    """
    I/O Handler for streaming (LLM) mode.

    Output appears as continuous text stream.
    Input is requested via <NEED_INPUT/> marker.
    Program end is signaled via <PROGRAM_END/> marker.
    """

    def __init__(self):
        super().__init__(mode='streaming')

    def get_full_output(self) -> str:
        """Get complete output buffer."""
        return self.output_buffer


class NativeIOHandler(IOHandler):
    """
    I/O Handler for native (direct stdio) mode.

    Directly reads from stdin and writes to stdout.
    No markers or tool calls - just raw I/O.

    Also supports printf with format specifiers:
    - %d: integer (decimal)
    - %x: integer (hex)
    - %c: character
    - %s: string from memory
    - %%: literal %
    """

    def __init__(self, blocking_input: bool = True, memory: Optional[Dict[int, int]] = None):
        """
        Initialize native I/O handler.

        Args:
            blocking_input: If True, block on input. If False, return -1 if no input.
            memory: Optional memory dict for reading strings (addr -> byte)
        """
        super().__init__(mode='native')
        self.blocking_input = blocking_input
        self._stdin_buffer = []
        self._stdout_buffer = ""
        self.memory = memory or {}  # For %s format (reading strings from memory)
        self._printf_args = []  # Stack of arguments for printf

    def check_io(self, embedding: torch.Tensor) -> Optional[str]:
        """
        Check embedding slots for I/O requests and handle directly.

        Returns None for native mode (output goes to stdout directly).
        """
        import sys

        # Check for PRINTF (before PUTCHAR, as PRTF also sets IO_OUTPUT_READY)
        tool_type = int(embedding[0, 0, E.IO_TOOL_CALL_TYPE].item())
        if tool_type == IOToolCallType.PRINTF and embedding[0, 0, E.IO_OUTPUT_READY].item() > 0.5:
            self._handle_printf(embedding)
            embedding[0, 0, E.IO_OUTPUT_READY] = 0.0
            embedding[0, 0, E.IO_TOOL_CALL_TYPE] = 0.0
            return None

        # Check IO_OUTPUT_READY (PUTCHAR)
        if embedding[0, 0, E.IO_OUTPUT_READY].item() > 0.5:
            char = self._decode_io_char(embedding)
            # Write directly to stdout
            sys.stdout.write(chr(char))
            sys.stdout.flush()
            self._stdout_buffer += chr(char)
            self.output_buffer += chr(char)
            # Clear the flag
            embedding[0, 0, E.IO_OUTPUT_READY] = 0.0
            embedding[0, 0, E.IO_TOOL_CALL_TYPE] = 0.0
            return None

        # Check IO_NEED_INPUT (GETCHAR needs input)
        if embedding[0, 0, E.IO_NEED_INPUT].item() > 0.5:
            # Read from stdin
            char = self._read_char()
            if char is not None:
                self.provide_input(char, embedding)
            else:
                # No input available, set to -1 (EOF)
                self.provide_input(0xFF, embedding)  # -1 as unsigned byte
            return None

        # Check IO_PROGRAM_END (EXIT)
        if embedding[0, 0, E.IO_PROGRAM_END].item() > 0.5:
            self.finished = True
            self.exit_code = int(embedding[0, 0, E.IO_EXIT_CODE].item())
            return None

        return None

    def _handle_printf(self, embedding: torch.Tensor):
        """
        Handle printf format string.

        Reads format string address from NIB_A and formats output.
        """
        import sys

        # Get format string address from NIB_A (combined 32-bit value)
        fmt_addr = 0
        for pos in range(E.NUM_POSITIONS):
            nibble = int(round(embedding[0, pos, E.NIB_A].item()))
            fmt_addr |= (nibble & 0xF) << (pos * 4)

        # Read format string from memory
        fmt_str = self._read_string(fmt_addr)

        if not fmt_str:
            return

        # Format and output
        output = self._format_string(fmt_str, embedding)
        sys.stdout.write(output)
        sys.stdout.flush()
        self._stdout_buffer += output
        self.output_buffer += output

    def _read_string(self, addr: int) -> str:
        """Read null-terminated string from memory."""
        result = []
        max_len = 1024  # Safety limit
        for i in range(max_len):
            byte = self.memory.get(addr + i, 0)
            if byte == 0:
                break
            result.append(chr(byte))
        return ''.join(result)

    def _format_string(self, fmt: str, embedding: torch.Tensor) -> str:
        """
        Format a printf-style string.

        Supports: %d, %x, %c, %s, %%
        Arguments come from NIB_B (first arg) or _printf_args stack.
        """
        result = []
        i = 0
        arg_idx = 0

        while i < len(fmt):
            if fmt[i] == '%' and i + 1 < len(fmt):
                spec = fmt[i + 1]
                if spec == 'd':
                    # Decimal integer
                    val = self._get_printf_arg(arg_idx, embedding)
                    result.append(str(val))
                    arg_idx += 1
                elif spec == 'x':
                    # Hex integer
                    val = self._get_printf_arg(arg_idx, embedding)
                    result.append(hex(val)[2:])  # Remove '0x' prefix
                elif spec == 'c':
                    # Character
                    val = self._get_printf_arg(arg_idx, embedding)
                    result.append(chr(val & 0xFF))
                    arg_idx += 1
                elif spec == 's':
                    # String
                    addr = self._get_printf_arg(arg_idx, embedding)
                    result.append(self._read_string(addr))
                    arg_idx += 1
                elif spec == '%':
                    # Literal %
                    result.append('%')
                else:
                    # Unknown specifier, pass through
                    result.append('%')
                    result.append(spec)
                i += 2
            else:
                result.append(fmt[i])
                i += 1

        return ''.join(result)

    def _get_printf_arg(self, idx: int, embedding: torch.Tensor) -> int:
        """Get printf argument by index."""
        if idx < len(self._printf_args):
            return self._printf_args[idx]

        # First arg comes from NIB_B
        if idx == 0:
            val = 0
            for pos in range(E.NUM_POSITIONS):
                nibble = int(round(embedding[0, pos, E.NIB_B].item()))
                val |= (nibble & 0xF) << (pos * 4)
            return val

        return 0  # No more args

    def push_printf_arg(self, val: int):
        """Push an argument for the next printf call."""
        self._printf_args.append(val)

    def clear_printf_args(self):
        """Clear printf argument stack."""
        self._printf_args = []

    def _read_char(self) -> Optional[int]:
        """
        Read a single character from stdin.

        Returns character code, or None if no input available (non-blocking).
        """
        import sys
        import select

        # If we have buffered input, use it
        if self._stdin_buffer:
            return self._stdin_buffer.pop(0)

        if self.blocking_input:
            # Blocking read
            try:
                char = sys.stdin.read(1)
                if char:
                    return ord(char)
                return None  # EOF
            except (EOFError, KeyboardInterrupt):
                return None
        else:
            # Non-blocking read (Unix only)
            try:
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    char = sys.stdin.read(1)
                    if char:
                        return ord(char)
                return None
            except (EOFError, KeyboardInterrupt, select.error):
                return None

    def buffer_input(self, s: str):
        """Buffer a string for input (for testing)."""
        self._stdin_buffer.extend(ord(c) for c in s)

    def get_stdout_buffer(self) -> str:
        """Get all stdout output."""
        return self._stdout_buffer


class ToolUseIOHandler(IOHandler):
    """
    I/O Handler for tool-use (agentic) mode.

    Each I/O operation generates explicit tool calls.
    External system responds with tool responses.

    Supports:
    - GETCHAR/PUTCHAR: Character I/O
    - OPEN/READ/CLOS: File I/O
    - EXIT: Program termination
    - PRINTF: Formatted output
    """

    def __init__(self):
        super().__init__(mode='tooluse')
        self.tool_calls: list[ToolCall] = []
        self.open_files: Dict[int, Any] = {}  # fd -> file object
        self.next_fd = 3  # 0=stdin, 1=stdout, 2=stderr

    def check_io(self, embedding: torch.Tensor) -> Optional[str]:
        """Check for all tool call types including file I/O."""
        result = None

        # Check tool call type
        tool_type = int(embedding[0, 0, E.IO_TOOL_CALL_TYPE].item())

        if tool_type == IOToolCallType.OPEN:
            result = self._handle_file_open(embedding)
        elif tool_type == IOToolCallType.READ:
            result = self._handle_file_read(embedding)
        elif tool_type == IOToolCallType.CLOSE:
            result = self._handle_file_close(embedding)
        else:
            # Use parent handler for GETCHAR, PUTCHAR, EXIT, PRINTF
            result = super().check_io(embedding)

        if self.pending_tool_call:
            self.tool_calls.append(self.pending_tool_call)

        return result

    def _handle_file_open(self, embedding: torch.Tensor) -> str:
        """Handle OPEN tool call."""
        self.tool_call_id += 1

        # Get filename from NIB_A (pointer) - would need memory access
        # For now, just generate the tool call
        filename_ptr = self._read_nibbles_as_int(embedding, E.NIB_A)

        self.pending_tool_call = ToolCall(
            call_type="open",
            call_id=self.tool_call_id,
            params={"filename_ptr": filename_ptr}
        )

        # Clear tool call type
        embedding[0, 0, E.IO_TOOL_CALL_TYPE] = 0.0

        return self.pending_tool_call.to_string()

    def _handle_file_read(self, embedding: torch.Tensor) -> str:
        """Handle READ tool call."""
        self.tool_call_id += 1

        # Get fd, buf, count from stack (would need memory access)
        fd = self._read_nibbles_as_int(embedding, E.NIB_A)

        self.pending_tool_call = ToolCall(
            call_type="read",
            call_id=self.tool_call_id,
            params={"fd": fd}
        )

        embedding[0, 0, E.IO_TOOL_CALL_TYPE] = 0.0
        return self.pending_tool_call.to_string()

    def _handle_file_close(self, embedding: torch.Tensor) -> str:
        """Handle CLOS tool call."""
        self.tool_call_id += 1

        fd = self._read_nibbles_as_int(embedding, E.NIB_A)

        self.pending_tool_call = ToolCall(
            call_type="close",
            call_id=self.tool_call_id,
            params={"fd": fd}
        )

        embedding[0, 0, E.IO_TOOL_CALL_TYPE] = 0.0
        return self.pending_tool_call.to_string()

    def _read_nibbles_as_int(self, embedding: torch.Tensor, slot: int) -> int:
        """Read 8 nibbles from a slot as a 32-bit integer."""
        value = 0
        for pos in range(E.NUM_POSITIONS):
            nibble = int(round(embedding[0, pos, slot].item()))
            value |= (nibble & 0xF) << (pos * 4)
        return value

    def get_pending_call(self) -> Optional[ToolCall]:
        """Get the pending tool call, if any."""
        return self.pending_tool_call

    def all_tool_calls(self) -> list[ToolCall]:
        """Get all tool calls made during execution."""
        return self.tool_calls


class ArgvMemorySetup:
    """
    Set up argv memory layout with proper pointer array.

    C4 expects argv to be char** - an array of pointers to strings.

    Memory layout:
        STRING_BASE (0x80000):  "program\0arg1\0arg2\0..."
        ARGV_BASE (0x7F000):    [ptr0, ptr1, ptr2, ...]  (8 bytes each)

    Stack setup (before main):
        push argc
        push argv_base

    Usage:
        setup = ArgvMemorySetup(["program", "arg1", "arg2"])
        writes = setup.get_memory_writes()
        for addr, value in writes:
            memory.write(addr, value)
        # Push argc and argv_base onto stack
    """

    STRING_BASE = 0x80000   # Where strings are stored
    ARGV_BASE = 0x7F000     # Where pointer array is stored

    def __init__(self, argv: list[str], string_base: int = None, argv_base: int = None):
        """
        Initialize with argument list.

        Args:
            argv: List of argument strings (argv[0] = program name)
            string_base: Optional custom base for string storage
            argv_base: Optional custom base for pointer array
        """
        self.argv = argv
        self.argc = len(argv)
        self.string_base = string_base or self.STRING_BASE
        self.argv_base = argv_base or self.ARGV_BASE

        # Compute string offsets
        self._string_offsets = []
        offset = 0
        for arg in argv:
            self._string_offsets.append(self.string_base + offset)
            offset += len(arg) + 1  # +1 for null terminator

        self._total_string_bytes = offset

    def get_string_writes(self) -> list[tuple[int, int]]:
        """
        Get byte-level writes for strings.

        Returns list of (address, byte_value) tuples.
        """
        writes = []
        addr = self.string_base

        for arg in self.argv:
            for char in arg:
                writes.append((addr, ord(char)))
                addr += 1
            writes.append((addr, 0))  # Null terminator
            addr += 1

        return writes

    def get_pointer_writes(self) -> list[tuple[int, int]]:
        """
        Get 8-byte pointer writes for argv array.

        Returns list of (address, pointer_value) tuples.
        Each pointer is stored as 8-byte little-endian.
        """
        writes = []

        for i, string_addr in enumerate(self._string_offsets):
            ptr_addr = self.argv_base + i * 8
            writes.append((ptr_addr, string_addr))

        return writes

    def get_memory_writes(self) -> list[tuple[int, int, int]]:
        """
        Get all memory writes needed.

        Returns list of (address, value, width) tuples.
        width: 1 for byte (strings), 8 for int (pointers)
        """
        writes = []

        # String bytes (1 byte each)
        for addr, byte_val in self.get_string_writes():
            writes.append((addr, byte_val, 1))

        # Pointer array (8 bytes each)
        for addr, ptr_val in self.get_pointer_writes():
            writes.append((addr, ptr_val, 8))

        return writes

    def get_stack_setup(self) -> list[tuple[str, int]]:
        """
        Get stack operations needed before main().

        Returns list of (operation, value) tuples.
        Operations: 'push' for pushing onto stack.
        """
        return [
            ('push', self.argc),
            ('push', self.argv_base),
        ]

    def setup_memory(self, memory_write_fn):
        """
        Set up memory using provided write function.

        Args:
            memory_write_fn: Function(addr, value, width) to write memory
        """
        for addr, value, width in self.get_memory_writes():
            memory_write_fn(addr, value, width)

    def __repr__(self):
        return (f"ArgvMemorySetup(argc={self.argc}, "
                f"argv_base=0x{self.argv_base:X}, "
                f"string_base=0x{self.string_base:X})")


class ArgvHandler:
    """
    Handler for plaintext argc/argv passing via embedding slots.

    NOTE: This streams bytes but does NOT set up the pointer array.
    For proper C4 compatibility, use ArgvMemorySetup instead.

    Instead of setting up memory with string pointers, both argc and argv
    are streamed through embedding slots as characters, just like I/O.

    Input format (plaintext):
        <ARGV>
        program_name
        arg1
        arg2
        </ARGV>

    Stream format (what the program reads):
        "3\0program_name\0arg1\0arg2\0"
         ^-- argc as string, then each argv string

    The program reads:
    1. First string is argc (e.g., "3\0")
    2. Following strings are argv[0], argv[1], etc.
    3. IO_ARGV_END signals end of current string (null terminator)
    4. IO_ALL_ARGV_READ signals all data consumed

    Usage:
        handler = ArgvHandler(["program", "hello", "world"])
        handler.setup(embedding)

        # Program reads: "3\0program\0hello\0world\0"
        # First parse argc string -> 3
        # Then read 3 argv strings
    """

    def __init__(self, argv: list[str]):
        """
        Initialize with argument list.

        Args:
            argv: List of argument strings (argv[0] = program name)
        """
        self.argv = argv
        self.argc = len(argv)

        # Build stream: [argc as 4 bytes LE] + "arg0\0arg1\0arg2\0"
        # argc is 4 bytes little-endian, then null-terminated strings
        argc_bytes = self.argc.to_bytes(4, byteorder='little')
        self._stream = list(argc_bytes)  # [0x04, 0x00, 0x00, 0x00] for argc=4
        for arg in argv:
            self._stream.extend(ord(c) for c in arg)
            self._stream.append(0)  # null terminator

        self._pos = 0  # Current position in stream
        self._reading_argc = True  # True for first 4 bytes
        self._current_string = 0  # Which argv[] we're reading (after argc)
        self._finished = False

    def setup(self, embedding: torch.Tensor):
        """
        Initialize embedding for argv streaming.

        Call this before program execution starts.
        """
        embedding[0, 0, E.IO_ARGV_INDEX] = 0.0
        embedding[0, 0, E.IO_ARGV_READY] = 0.0
        embedding[0, 0, E.IO_ARGV_END] = 0.0
        embedding[0, 0, E.IO_ALL_ARGV_READ] = 0.0
        embedding[0, 0, E.IO_NEED_ARGV] = 0.0

    # Keep old name as alias for compatibility
    def setup_argc(self, embedding: torch.Tensor):
        """Alias for setup() - kept for compatibility."""
        self.setup(embedding)

    def check_argv(self, embedding: torch.Tensor) -> bool:
        """
        Check if program needs argv data and provide it.

        Returns True if data was provided, False otherwise.
        """
        if self._finished:
            return False

        # Check IO_NEED_ARGV
        if embedding[0, 0, E.IO_NEED_ARGV].item() > 0.5:
            self.provide_next_byte(embedding)
            return True

        return False

    def provide_next_byte(self, embedding: torch.Tensor):
        """
        Provide next byte from the stream.

        Stream format: [4 bytes argc LE] + "argv[0]\0argv[1]\0...\0"
        """
        if self._pos >= len(self._stream):
            # All data consumed
            embedding[0, 0, E.IO_ALL_ARGV_READ] = 1.0
            embedding[0, 0, E.IO_NEED_ARGV] = 0.0
            self._finished = True
            return

        byte = self._stream[self._pos]
        self._pos += 1

        # First 4 bytes are argc (no special handling, just bytes)
        if self._pos <= 4:
            # Reading argc bytes
            embedding[0, 0, E.IO_CHAR] = float(byte)
            embedding[0, 0, E.IO_ARGV_END] = 0.0
        elif byte == 0:
            # End of current argv string
            embedding[0, 0, E.IO_CHAR] = 0.0
            embedding[0, 0, E.IO_ARGV_END] = 1.0
            self._current_string += 1
            embedding[0, 0, E.IO_ARGV_INDEX] = float(self._current_string)
        else:
            # Regular character
            embedding[0, 0, E.IO_CHAR] = float(byte)
            embedding[0, 0, E.IO_ARGV_END] = 0.0

        embedding[0, 0, E.IO_ARGV_READY] = 1.0
        embedding[0, 0, E.IO_NEED_ARGV] = 0.0

    def reset(self):
        """Reset to beginning of stream."""
        self._pos = 0
        self._reading_argc = True
        self._current_string = 0
        self._finished = False

    @classmethod
    def from_string(cls, s: str) -> 'ArgvHandler':
        """
        Parse argv from plaintext format.

        Format (argc is first line):
            <ARGV>
            4
            calculator
            add
            5
            3
            </ARGV>

        The first line is argc, followed by that many argv strings.
        argc is stored internally as 4 bytes little-endian.
        """
        s = s.strip()

        # XML format
        if s.startswith('<ARGV>') and s.endswith('</ARGV>'):
            content = s[6:-7].strip()
            lines = [line.strip() for line in content.split('\n') if line.strip()]

            if not lines:
                return cls([])

            # First line is argc
            argc = int(lines[0])
            argv = lines[1:argc+1]
            return cls(argv)

        # Newline-separated (first line is argc)
        if '\n' in s:
            lines = [line.strip() for line in s.split('\n') if line.strip()]
            if not lines:
                return cls([])
            argc = int(lines[0])
            argv = lines[1:argc+1]
            return cls(argv)

        # Space-separated: "argc arg0 arg1 ..."
        parts = s.split()
        if not parts:
            return cls([])
        argc = int(parts[0])
        argv = parts[1:argc+1]
        return cls(argv)


class CombinedIOHandler:
    """
    Combined I/O handler that handles both argv and stdio.

    This is the main handler to use for running programs:
    1. Setup argv via ArgvHandler
    2. Handle stdin/stdout via IOHandler
    3. Coordinate between both

    Usage:
        handler = CombinedIOHandler(
            argv=["program", "arg1"],
            stdin="user input here",
            mode='streaming'
        )
        handler.setup(embedding)

        while not handler.finished:
            embedding = vm.step()
            output = handler.check_io(embedding)
            if output:
                print(output, end='')
    """

    def __init__(self, argv: list[str], stdin: str = "", mode: str = 'streaming'):
        """
        Initialize combined handler.

        Args:
            argv: Command-line arguments
            stdin: Input to provide for GETCHAR
            mode: 'streaming', 'tooluse', or 'native'
        """
        self.argv_handler = ArgvHandler(argv)
        self.stdin_buffer = stdin
        self.stdin_pos = 0

        if mode == 'streaming':
            self.io_handler = StreamingIOHandler()
        elif mode == 'tooluse':
            self.io_handler = ToolUseIOHandler()
        elif mode == 'native':
            self.io_handler = NativeIOHandler()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.mode = mode

    def setup(self, embedding: torch.Tensor):
        """Initialize embedding with argc and prepare for execution."""
        self.argv_handler.setup_argc(embedding)

    @property
    def finished(self) -> bool:
        return self.io_handler.finished

    @property
    def exit_code(self) -> int:
        return self.io_handler.exit_code

    @property
    def output_buffer(self) -> str:
        return self.io_handler.output_buffer

    def check_io(self, embedding: torch.Tensor) -> Optional[str]:
        """
        Check for all I/O needs: argv and stdio.

        Returns output string if any.
        """
        # First check argv needs
        if self.argv_handler.check_argv(embedding):
            return None  # Argv provided, no output

        # Then check stdio needs
        result = self.io_handler.check_io(embedding)

        # Auto-provide stdin if needed and available
        if self.io_handler.waiting_for_input and self.stdin_pos < len(self.stdin_buffer):
            char = ord(self.stdin_buffer[self.stdin_pos])
            self.stdin_pos += 1
            self.io_handler.provide_input(char, embedding)
            self.io_handler.waiting_for_input = False

        return result

    @classmethod
    def from_plaintext(cls, text: str, mode: str = 'streaming') -> 'CombinedIOHandler':
        """
        Parse handler from plaintext format.

        Format (argc is explicit):
            <ARGV>
            4
            calculator
            add
            5
            3
            </ARGV>
            <STDIN>
            user input here
            </STDIN>

        First line in ARGV is argc, followed by that many argv strings.
        If no ARGV tag, argc=0 and argv=[].
        """
        text = text.strip()

        # Parse ARGV section
        argv = []
        stdin = ""

        if '<ARGV>' in text:
            # Extract ARGV section
            start = text.index('<ARGV>') + 6
            end = text.index('</ARGV>')
            argv_text = text[start:end].strip()
            lines = [line.strip() for line in argv_text.split('\n') if line.strip()]

            if lines:
                # First line is argc
                argc = int(lines[0])
                argv = lines[1:argc+1]

        # Extract STDIN if present (regardless of ARGV)
        if '<STDIN>' in text:
            start = text.index('<STDIN>') + 7
            end = text.index('</STDIN>')
            stdin = text[start:end]

        # If no ARGV tag and no STDIN tag, try simple space-separated format
        if '<ARGV>' not in text and '<STDIN>' not in text and text:
            parts = text.split()
            if parts:
                try:
                    argc = int(parts[0])
                    argv = parts[1:argc+1]
                except ValueError:
                    # Not a valid format, leave argv empty
                    pass

        return cls(argv=argv, stdin=stdin, mode=mode)


def demo_streaming_io():
    """Demonstrate streaming I/O mode."""
    print("=" * 60)
    print("Streaming I/O Mode Demo")
    print("=" * 60)
    print()

    # Create embedding tensor
    embedding = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

    # Create handler
    handler = StreamingIOHandler()

    # Simulate PUTCHAR sequence for "Hello"
    print("Simulating PUTCHAR for 'Hello':")
    for c in "Hello":
        embedding[0, 0, E.IO_CHAR] = float(ord(c))
        embedding[0, 0, E.IO_OUTPUT_READY] = 1.0
        output = handler.check_io(embedding)
        print(f"  PUTCHAR('{c}') -> '{output}'")

    print(f"\nOutput buffer: '{handler.output_buffer}'")

    # Simulate GETCHAR (need input)
    print("\nSimulating GETCHAR (waiting for input):")
    embedding[0, 0, E.IO_NEED_INPUT] = 1.0
    output = handler.check_io(embedding)
    print(f"  GETCHAR -> '{output}'")
    print(f"  waiting_for_input: {handler.waiting_for_input}")

    # Provide input
    print("\nProviding input 'X':")
    handler.provide_input(ord('X'), embedding)
    print(f"  IO_INPUT_READY: {embedding[0, 0, E.IO_INPUT_READY].item()}")
    print(f"  IO_CHAR: {int(embedding[0, 0, E.IO_CHAR].item())} ('{chr(int(embedding[0, 0, E.IO_CHAR].item()))}')")

    # Simulate EXIT
    print("\nSimulating EXIT(42):")
    embedding[0, 0, E.IO_EXIT_CODE] = 42.0
    embedding[0, 0, E.IO_PROGRAM_END] = 1.0
    output = handler.check_io(embedding)
    print(f"  EXIT -> '{output}'")
    print(f"  finished: {handler.finished}")
    print(f"  exit_code: {handler.exit_code}")


def demo_tooluse_io():
    """Demonstrate tool-use I/O mode."""
    print()
    print("=" * 60)
    print("Tool-Use I/O Mode Demo")
    print("=" * 60)
    print()

    # Create embedding tensor
    embedding = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

    # Create handler
    handler = ToolUseIOHandler()

    # Simulate PUTCHAR
    print("Simulating PUTCHAR('H'):")
    embedding[0, 0, E.IO_CHAR] = float(ord('H'))
    embedding[0, 0, E.IO_OUTPUT_READY] = 1.0
    output = handler.check_io(embedding)
    print(f"  -> {output}")

    # Handle response
    response = ToolResponse(call_id=1, success=True, result="ok")
    handler.handle_tool_response(response, embedding)
    print(f"  Response handled")

    # Simulate GETCHAR
    print("\nSimulating GETCHAR:")
    embedding[0, 0, E.IO_NEED_INPUT] = 1.0
    output = handler.check_io(embedding)
    print(f"  -> {output}")

    # Handle response with input
    response = ToolResponse(call_id=2, success=True, result="A")
    handler.handle_tool_response(response, embedding)
    print(f"  Response handled, char={int(embedding[0, 0, E.IO_CHAR].item())}")

    # Simulate EXIT
    print("\nSimulating EXIT(0):")
    embedding[0, 0, E.IO_EXIT_CODE] = 0.0
    embedding[0, 0, E.IO_PROGRAM_END] = 1.0
    output = handler.check_io(embedding)
    print(f"  -> {output}")

    print(f"\nAll tool calls: {[tc.to_string() for tc in handler.all_tool_calls()]}")


def demo_argv_handler():
    """Demonstrate argv streaming."""
    print()
    print("=" * 60)
    print("Argv Streaming Demo")
    print("=" * 60)
    print()

    # Create embedding tensor
    embedding = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

    # Create argv handler
    handler = ArgvHandler(["myprogram", "hello", "world"])
    handler.setup_argc(embedding)

    print(f"argc = {int(embedding[0, 0, E.IO_ARGC].item())}")
    print()

    # Simulate reading all argv
    print("Reading argv strings character by character:")
    current_arg = []
    all_args = []

    while True:
        # Request next character
        embedding[0, 0, E.IO_NEED_ARGV] = 1.0
        handler.check_argv(embedding)

        if embedding[0, 0, E.IO_ALL_ARGV_READ].item() > 0.5:
            print("  All argv read!")
            break

        char_code = int(embedding[0, 0, E.IO_CHAR].item())

        if embedding[0, 0, E.IO_ARGV_END].item() > 0.5:
            arg_str = ''.join(current_arg)
            all_args.append(arg_str)
            print(f"  argv[{len(all_args)-1}] = \"{arg_str}\"")
            current_arg = []
        else:
            current_arg.append(chr(char_code))

    print(f"\nParsed argv: {all_args}")


def demo_combined_handler():
    """Demonstrate combined argv + stdio handler."""
    print()
    print("=" * 60)
    print("Combined Argv + Stdio Demo")
    print("=" * 60)
    print()

    # Create from plaintext
    plaintext = """<ARGV>
calculator
add
5
3
</ARGV>
<STDIN>
y
</STDIN>"""

    print("Input plaintext:")
    print(plaintext)
    print()

    handler = CombinedIOHandler.from_plaintext(plaintext)
    embedding = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    handler.setup(embedding)

    print(f"Parsed argc = {int(embedding[0, 0, E.IO_ARGC].item())}")
    print(f"Parsed argv = {handler.argv_handler.argv}")
    print(f"Parsed stdin = {repr(handler.stdin_buffer)}")
    print()

    # Read argv
    print("Reading argv:")
    args = []
    current = []
    while True:
        embedding[0, 0, E.IO_NEED_ARGV] = 1.0
        handler.argv_handler.check_argv(embedding)
        if embedding[0, 0, E.IO_ALL_ARGV_READ].item() > 0.5:
            break
        char = int(embedding[0, 0, E.IO_CHAR].item())
        if embedding[0, 0, E.IO_ARGV_END].item() > 0.5:
            args.append(''.join(current))
            current = []
        else:
            current.append(chr(char))
    print(f"  {args}")

    # Simulate stdin read
    print("\nReading stdin (GETCHAR):")
    embedding[0, 0, E.IO_NEED_INPUT] = 1.0
    handler.check_io(embedding)
    print(f"  char = '{chr(int(embedding[0, 0, E.IO_CHAR].item()))}'")


if __name__ == "__main__":
    demo_streaming_io()
    demo_tooluse_io()
    demo_argv_handler()
    demo_combined_handler()
