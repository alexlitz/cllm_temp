#!/usr/bin/env python3
"""
Tool-Use Handler for C Programs

Handles the tool-use protocol from C programs:
  Input:  TOOL_CALL:<type>:<id>:{params_json}
  Output: TOOL_RESPONSE:<id>:<result>

Can be used to:
1. Run interactive C programs with Python-based I/O handling
2. Connect C programs to external APIs or MCP servers
3. Sandbox file I/O operations
"""

import sys
import subprocess
import json
import re
from typing import Dict, Any, Callable, Optional


class CToolUseHandler:
    """Handler for tool-use protocol from C programs."""

    def __init__(
        self,
        output_callback: Optional[Callable[[str], None]] = None,
        input_callback: Optional[Callable[[], str]] = None,
        allow_file_io: bool = False,
        file_sandbox: Optional[str] = None
    ):
        self.output_callback = output_callback or (lambda s: print(s, end='', flush=True))
        self.input_callback = input_callback or input
        self.allow_file_io = allow_file_io
        self.file_sandbox = file_sandbox

        # Virtual file system
        self.files: Dict[int, Any] = {}
        self.next_fd = 3

        # Input buffer for character-by-character reading
        self.input_buffer: list[int] = []

        # Call history
        self.call_history = []

    def handle_call(self, call_type: str, call_id: int, params: Dict[str, Any]) -> Any:
        """Handle a tool call and return the result."""
        self.call_history.append((call_type, call_id, params))

        if call_type == 'print':
            text = params.get('text', '')
            self.output_callback(text)
            return 0

        elif call_type == 'putchar':
            char = params.get('char', 0)
            self.output_callback(chr(char & 0xFF))
            return char

        elif call_type == 'getchar':
            # If we have buffered input, return next character
            if self.input_buffer:
                return self.input_buffer.pop(0)

            # Get new input
            try:
                line = self.input_callback()
                if line:
                    # Buffer all characters
                    for c in line + '\n':
                        self.input_buffer.append(ord(c))
                    # Return first
                    return self.input_buffer.pop(0) if self.input_buffer else -1
                return -1  # EOF
            except EOFError:
                return -1

        elif call_type == 'file_open':
            path = params.get('path', '')
            mode = params.get('mode', 'r')
            if self.allow_file_io:
                try:
                    fd = self.next_fd
                    self.next_fd += 1
                    self.files[fd] = {'path': path, 'mode': mode, 'pos': 0, 'content': b''}
                    if 'r' in mode and self.file_sandbox:
                        import os
                        full_path = os.path.join(self.file_sandbox, path)
                        with open(full_path, 'rb') as f:
                            self.files[fd]['content'] = f.read()
                    return fd
                except Exception:
                    return -1
            return -1

        elif call_type == 'file_read':
            fd = params.get('fd', 0)
            size = params.get('size', 1)
            if fd in self.files:
                f = self.files[fd]
                data = f['content'][f['pos']:f['pos']+size]
                f['pos'] += len(data)
                return len(data)
            return -1

        elif call_type == 'file_close':
            fd = params.get('fd', 0)
            if fd in self.files:
                del self.files[fd]
                return 0
            return -1

        elif call_type == 'exit':
            return params.get('code', 0)

        return 0

    def parse_tool_call(self, line: str) -> Optional[tuple]:
        """Parse a TOOL_CALL line."""
        match = re.match(r'TOOL_CALL:(\w+):(\d+):\{(.*)\}', line)
        if match:
            call_type = match.group(1)
            call_id = int(match.group(2))
            try:
                params = json.loads('{' + match.group(3) + '}')
            except json.JSONDecodeError:
                params = {}
            return (call_type, call_id, params)
        return None

    def run_program(self, program_path: str, args: list = None):
        """Run a C program with tool-use protocol."""
        cmd = [program_path] + (args or [])

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break

                line = line.rstrip('\n')

                parsed = self.parse_tool_call(line)
                if parsed:
                    call_type, call_id, params = parsed
                    result = self.handle_call(call_type, call_id, params)

                    # Send response for getchar calls
                    if call_type == 'getchar':
                        process.stdin.write(f"TOOL_RESPONSE:{call_id}:{result}\n")
                        process.stdin.flush()

                    # Exit on exit call
                    if call_type == 'exit':
                        break
                else:
                    # Regular output
                    print(line)

        except KeyboardInterrupt:
            process.terminate()
        finally:
            process.wait()

        return process.returncode


def run_interactive(program_path: str, args: list = None):
    """Run a C program interactively with tool-use handling."""
    handler = CToolUseHandler(
        output_callback=lambda s: print(s, end='', flush=True),
        input_callback=input
    )
    return handler.run_program(program_path, args)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tooluse_c_handler.py <program> [args...]")
        print()
        print("Example:")
        print("  python tooluse_c_handler.py ./tooluse_onnx_c4 -i -t")
        sys.exit(1)

    program = sys.argv[1]
    args = sys.argv[2:]

    exit_code = run_interactive(program, args)
    sys.exit(exit_code)
