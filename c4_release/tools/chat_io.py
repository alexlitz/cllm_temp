#!/usr/bin/env python3
"""
Chat-Based I/O for VM Programs

Allows programs to run with I/O flowing through a chat interface.
The VM runs until it needs input, then returns control to the chat.

Usage (in LLM chat):
    session = ChatSession('eliza_simple.c')
    output = session.start()           # Returns greeting + prompt
    output = session.send("hello")     # Send user input, get response
    output = session.send("bye")       # Continue conversation
"""

import sys
import os
import pickle
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Tuple
from dataclasses import dataclass
from src.compiler import compile_c
from tooluse_io import ToolUseVM, ToolUseIOHandler, ToolCallType


@dataclass
class ChatSession:
    """
    A chat-based VM session that can be serialized between turns.
    """
    source_file: str
    _vm: Optional[ToolUseVM] = None
    _handler: Optional[ToolUseIOHandler] = None
    _output_buffer: str = ""
    _waiting_for_input: bool = False
    _finished: bool = False
    _exit_code: int = 0

    def __post_init__(self):
        if self._vm is None:
            self._compile_and_load()

    def _compile_and_load(self):
        """Compile source and initialize VM."""
        with open(self.source_file, 'r') as f:
            source = f.read()

        bytecode, data = compile_c(source)

        # Handler that buffers output
        self._output_buffer = ""

        def capture_output(s):
            self._output_buffer += s

        self._handler = ToolUseIOHandler(
            output_callback=capture_output,
            input_callback=None  # We'll handle input differently
        )

        self._vm = ToolUseVM(self._handler)
        self._vm.load(bytecode, data)

    def _run_until_input_or_done(self) -> Tuple[str, bool, bool]:
        """
        Run VM until it needs input or exits.

        Returns:
            (output_text, needs_input, is_finished)
        """
        self._output_buffer = ""
        max_steps = 1_000_000
        steps = 0

        while steps < max_steps and not self._vm.halted:
            call = self._vm.step()

            if call:
                if call.call_type == ToolCallType.PUTCHAR:
                    # Handle output immediately
                    char = call.params.get("char", 0)
                    self._output_buffer += chr(char & 0xFF)
                    self._vm.provide_response(
                        type('Response', (), {'call_id': call.call_id, 'success': True, 'result': char})()
                    )

                elif call.call_type == ToolCallType.USER_INPUT:
                    # Need input from user - pause here
                    self._waiting_for_input = True
                    return (self._output_buffer, True, False)

                elif call.call_type == ToolCallType.EXIT:
                    self._finished = True
                    self._exit_code = call.params.get("code", 0)
                    self._vm.provide_response(
                        type('Response', (), {'call_id': call.call_id, 'success': True, 'result': 0})()
                    )
                    return (self._output_buffer, False, True)

                else:
                    # Other calls - just acknowledge
                    self._vm.provide_response(
                        type('Response', (), {'call_id': call.call_id, 'success': True, 'result': 0})()
                    )

            steps += 1

        return (self._output_buffer, False, True)

    def start(self) -> str:
        """
        Start the program and run until it needs input.

        Returns:
            Initial output (greeting, prompt, etc.)
        """
        output, needs_input, finished = self._run_until_input_or_done()
        self._waiting_for_input = needs_input
        self._finished = finished
        return output

    def send(self, user_input: str) -> str:
        """
        Send user input and get the program's response.

        Args:
            user_input: What the user typed

        Returns:
            Program's response output
        """
        if self._finished:
            return "[Program has ended]"

        if not self._waiting_for_input:
            return "[Program not waiting for input]"

        # Feed the input character by character
        input_chars = list(user_input + '\n')
        char_idx = 0

        # Provide first character response
        if input_chars:
            first_char = ord(input_chars[0])
            char_idx = 1
            # Get the pending call and respond
            pending = self._vm.pending_call
            if pending:
                self._vm.provide_response(
                    type('Response', (), {'call_id': pending.call_id, 'success': True, 'result': first_char})()
                )

        self._waiting_for_input = False
        self._output_buffer = ""

        # Continue running, providing remaining input chars as needed
        max_steps = 1_000_000
        steps = 0

        while steps < max_steps and not self._vm.halted:
            call = self._vm.step()

            if call:
                if call.call_type == ToolCallType.PUTCHAR:
                    char = call.params.get("char", 0)
                    self._output_buffer += chr(char & 0xFF)
                    self._vm.provide_response(
                        type('Response', (), {'call_id': call.call_id, 'success': True, 'result': char})()
                    )

                elif call.call_type == ToolCallType.USER_INPUT:
                    # Need more input
                    if char_idx < len(input_chars):
                        # Still have buffered input
                        char = ord(input_chars[char_idx])
                        char_idx += 1
                        self._vm.provide_response(
                            type('Response', (), {'call_id': call.call_id, 'success': True, 'result': char})()
                        )
                    else:
                        # Need new input from user
                        self._waiting_for_input = True
                        return self._output_buffer

                elif call.call_type == ToolCallType.EXIT:
                    self._finished = True
                    self._exit_code = call.params.get("code", 0)
                    self._vm.provide_response(
                        type('Response', (), {'call_id': call.call_id, 'success': True, 'result': 0})()
                    )
                    return self._output_buffer

                else:
                    self._vm.provide_response(
                        type('Response', (), {'call_id': call.call_id, 'success': True, 'result': 0})()
                    )

            steps += 1

        return self._output_buffer

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def is_waiting(self) -> bool:
        return self._waiting_for_input

    def serialize(self) -> str:
        """Serialize session state to string (for persistence between chat turns)."""
        # Note: Full serialization would require pickling VM state
        # For now, return a simple state indicator
        return base64.b64encode(pickle.dumps({
            'source_file': self.source_file,
            'finished': self._finished,
            'waiting': self._waiting_for_input,
        })).decode()


# Convenience function for quick testing
def chat_with_eliza():
    """Interactive chat with Eliza in the terminal."""
    session = ChatSession(
        os.path.join(os.path.dirname(__file__), 'eliza_simple.c')
    )

    # Start and show greeting
    output = session.start()
    print(output, end='')

    # Chat loop
    while not session.is_finished:
        if session.is_waiting:
            try:
                user_input = input()
            except (EOFError, KeyboardInterrupt):
                break
            output = session.send(user_input)
            print(output, end='')

    print("\n[Session ended]")


if __name__ == "__main__":
    chat_with_eliza()
