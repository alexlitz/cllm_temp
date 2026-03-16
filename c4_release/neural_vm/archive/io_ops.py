"""
I/O Operations for Neural VM.

These FFNs implement I/O through embedding slots (mailbox pattern):
- GETCHAR: Sets IO_NEED_INPUT, reads from IO_CHAR when IO_INPUT_READY
- PUTCHAR: Writes to IO_CHAR, sets IO_OUTPUT_READY
- EXIT: Sets IO_PROGRAM_END, writes exit code to IO_EXIT_CODE

Two modes are supported:
1. Streaming Mode: Uses <NEED_INPUT/>, <PROGRAM_END/> markers
2. Tool-Use Mode: Uses TOOL_CALL:type:id:{params} protocol
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode, IOToolCallType
from .base_layers import PureFFN


class GetcharSetNeedInputFFN(PureFFN):
    """
    GETCHAR step 1: Set IO_NEED_INPUT flag.

    When GETCHAR opcode is active and IO_INPUT_READY is 0:
    - Set IO_NEED_INPUT = 1.0
    - Set IO_TOOL_CALL_TYPE = GETCHAR (for tool-use mode)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # When GETCHAR active AND IO_INPUT_READY is 0, set IO_NEED_INPUT = 1
            # Gate: silu(S*GETCHAR) * (1 - IO_INPUT_READY)
            self.W_up[0, E.OP_START + Opcode.GETCHAR] = S
            self.W_gate[0, E.IO_INPUT_READY] = -1.0
            self.b_gate[0] = 1.0
            self.W_down[E.IO_NEED_INPUT, 0] = 1.0 / S

            # Also set tool call type
            self.W_up[1, E.OP_START + Opcode.GETCHAR] = S
            self.W_gate[1, E.IO_INPUT_READY] = -1.0
            self.b_gate[1] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 1] = float(IOToolCallType.GETCHAR) / S


class GetcharReadInputFFN(PureFFN):
    """
    GETCHAR step 2: Read character from IO_CHAR when IO_INPUT_READY.

    When GETCHAR opcode is active and IO_INPUT_READY is 1:
    - Copy IO_CHAR nibbles to RESULT nibbles (at each position)
    - Clear IO_INPUT_READY
    - Clear IO_NEED_INPUT
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # For each position, copy IO_CHAR to RESULT when input ready
            # Gate: silu(S*GETCHAR) * IO_INPUT_READY

            # Copy IO_CHAR to RESULT (position-independent for simplicity)
            # The IO_CHAR slot contains the character value at position 0
            self.W_up[0, E.OP_START + Opcode.GETCHAR] = S
            self.W_gate[0, E.IO_INPUT_READY] = 1.0
            self.W_gate[0, E.IO_CHAR] = 1.0  # Multiply by char value
            self.W_down[E.RESULT, 0] = 1.0 / S

            # Clear IO_INPUT_READY
            self.W_up[1, E.OP_START + Opcode.GETCHAR] = S
            self.W_gate[1, E.IO_INPUT_READY] = -1.0  # Subtracts current value
            self.W_down[E.IO_INPUT_READY, 1] = 1.0 / S

            # Clear IO_NEED_INPUT
            self.W_up[2, E.OP_START + Opcode.GETCHAR] = S
            self.W_gate[2, E.IO_NEED_INPUT] = -1.0
            self.W_down[E.IO_NEED_INPUT, 2] = 1.0 / S

            # Clear tool call type
            self.W_up[3, E.OP_START + Opcode.GETCHAR] = S
            self.W_gate[3, E.IO_TOOL_CALL_TYPE] = -1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 3] = 1.0 / S


class PutcharWriteOutputFFN(PureFFN):
    """
    PUTCHAR: Write character from operand A to IO_CHAR and set IO_OUTPUT_READY.

    When PUTCHAR opcode is active:
    - Copy NIB_A (at position 0) to IO_CHAR
    - Set IO_OUTPUT_READY = 1.0
    - Set IO_TOOL_CALL_TYPE = PUTCHAR (for tool-use mode)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy NIB_A to IO_CHAR
            # Gate: silu(S*PUTCHAR) * NIB_A
            self.W_up[0, E.OP_START + Opcode.PUTCHAR] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.IO_CHAR, 0] = 1.0 / S

            # Set IO_OUTPUT_READY = 1.0
            self.W_up[1, E.OP_START + Opcode.PUTCHAR] = S
            self.b_gate[1] = 1.0
            self.W_down[E.IO_OUTPUT_READY, 1] = 1.0 / S

            # Set tool call type
            self.W_up[2, E.OP_START + Opcode.PUTCHAR] = S
            self.b_gate[2] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 2] = float(IOToolCallType.PUTCHAR) / S


class ExitSetEndFFN(PureFFN):
    """
    EXIT: Set IO_PROGRAM_END and write exit code.

    When EXIT opcode is active:
    - Copy NIB_A (exit code) to IO_EXIT_CODE
    - Set IO_PROGRAM_END = 1.0
    - Set IO_TOOL_CALL_TYPE = EXIT (for tool-use mode)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=6)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy NIB_A to IO_EXIT_CODE
            self.W_up[0, E.OP_START + Opcode.EXIT] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.IO_EXIT_CODE, 0] = 1.0 / S

            # Set IO_PROGRAM_END = 1.0
            self.W_up[1, E.OP_START + Opcode.EXIT] = S
            self.b_gate[1] = 1.0
            self.W_down[E.IO_PROGRAM_END, 1] = 1.0 / S

            # Set tool call type
            self.W_up[2, E.OP_START + Opcode.EXIT] = S
            self.b_gate[2] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 2] = float(IOToolCallType.EXIT) / S


class PrintfFFN(PureFFN):
    """
    PRTF (printf): Similar to PUTCHAR but for formatted output.

    For now, just marks the printf tool call type.
    Actual formatting is handled by the external handler.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set IO_OUTPUT_READY
            self.W_up[0, E.OP_START + Opcode.PRTF] = S
            self.b_gate[0] = 1.0
            self.W_down[E.IO_OUTPUT_READY, 0] = 1.0 / S

            # Set tool call type
            self.W_up[1, E.OP_START + Opcode.PRTF] = S
            self.b_gate[1] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 1] = float(IOToolCallType.PRINTF) / S


class FileOpenFFN(PureFFN):
    """
    OPEN: File open operation (tool-use only).

    Sets IO_TOOL_CALL_TYPE = OPEN for external handler.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set tool call type
            self.W_up[0, E.OP_START + Opcode.OPEN] = S
            self.b_gate[0] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 0] = float(IOToolCallType.OPEN) / S


class FileReadFFN(PureFFN):
    """
    READ: File read operation (tool-use only).

    Sets IO_TOOL_CALL_TYPE = READ for external handler.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set tool call type
            self.W_up[0, E.OP_START + Opcode.READ] = S
            self.b_gate[0] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 0] = float(IOToolCallType.READ) / S


class FileCloseFFN(PureFFN):
    """
    CLOS: File close operation (tool-use only).

    Sets IO_TOOL_CALL_TYPE = CLOSE for external handler.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set tool call type
            self.W_up[0, E.OP_START + Opcode.CLOS] = S
            self.b_gate[0] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 0] = float(IOToolCallType.CLOSE) / S


class MallocFFN(PureFFN):
    """
    MALC (malloc): Memory allocation (tool-use only).

    Sets IO_TOOL_CALL_TYPE = MALLOC for external handler.
    NIB_A contains requested size.
    Handler returns pointer in IO_TOOL_RESPONSE.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set tool call type = 8 (MALLOC)
            self.W_up[0, E.OP_START + Opcode.MALC] = S
            self.b_gate[0] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 8.0 / S  # MALLOC = 8

            # Copy size (NIB_A) to IO_CHAR for handler
            self.W_up[1, E.OP_START + Opcode.MALC] = S
            self.W_gate[1, E.NIB_A] = 1.0
            self.W_down[E.IO_CHAR, 1] = 1.0 / S

            self.W_up[2, E.OP_START + Opcode.MALC] = -S
            self.W_gate[2, E.NIB_A] = -1.0
            self.W_down[E.IO_CHAR, 2] = 1.0 / S


class FreeFFN(PureFFN):
    """
    FREE: Memory deallocation (tool-use only).

    Sets IO_TOOL_CALL_TYPE = FREE for external handler.
    NIB_A contains pointer to free.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set tool call type = 9 (FREE)
            self.W_up[0, E.OP_START + Opcode.FREE] = S
            self.b_gate[0] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 9.0 / S  # FREE = 9

            # Copy pointer (NIB_A) to IO_CHAR for handler
            self.W_up[1, E.OP_START + Opcode.FREE] = S
            self.W_gate[1, E.NIB_A] = 1.0
            self.W_down[E.IO_CHAR, 1] = 1.0 / S

            self.W_up[2, E.OP_START + Opcode.FREE] = -S
            self.W_gate[2, E.NIB_A] = -1.0
            self.W_down[E.IO_CHAR, 2] = 1.0 / S


class MsetFFN(PureFFN):
    """
    MSET (memset): Memory fill (tool-use for now, subroutine later).

    Sets IO_TOOL_CALL_TYPE = MSET for external handler.
    Args from stack: ptr, val, size
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set tool call type = 10 (MSET)
            self.W_up[0, E.OP_START + Opcode.MSET] = S
            self.b_gate[0] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 10.0 / S


class McmpFFN(PureFFN):
    """
    MCMP (memcmp): Memory comparison (tool-use for now, subroutine later).

    Sets IO_TOOL_CALL_TYPE = MCMP for external handler.
    Args from stack: ptr1, ptr2, size
    Returns comparison result in AX.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Set tool call type = 11 (MCMP)
            self.W_up[0, E.OP_START + Opcode.MCMP] = S
            self.b_gate[0] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 11.0 / S


class ClearIOSlotsFFN(PureFFN):
    """
    Clear I/O slots after handler has processed them.

    Called by external handler after consuming output or providing input.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear each I/O slot
            slots = [E.IO_OUTPUT_READY, E.IO_INPUT_READY, E.IO_NEED_INPUT,
                    E.IO_TOOL_CALL_TYPE, E.IO_TOOL_CALL_ID]

            for i, slot in enumerate(slots):
                if i < 8:
                    self.b_up[i] = S
                    self.W_gate[i, slot] = -1.0
                    self.W_down[slot, i] = 1.0 / S
