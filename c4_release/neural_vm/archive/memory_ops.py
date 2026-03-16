"""
Memory operations for Neural VM V7.

LOAD, STORE, PUSH, POP, NOP, HALT

Memory Architecture:
- ALU operates on 8 nibble positions (32-bit values)
- Memory addresses are 32-bit, stored in NIB_A across all positions
- Memory values are 32-bit, stored in NIB_B (for STORE) or RESULT (for LOAD)
- Actual memory access happens via KV cache attention (see kv_memory.py)

The FFNs here prepare data for memory operations:
- LOAD: Extract address from NIB_A, prepare for KV lookup
- STORE: Extract address from NIB_A, value from NIB_B, prepare for KV write
- PUSH: Decrement SP, store value at new SP
- POP: Load value from SP, increment SP
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN


class LoadFFN(PureFFN):
    """
    LOAD: Prepare address for memory read.

    Input: NIB_A[0:7] = 32-bit address nibbles
    Output: TEMP = address (for KV lookup), RESULT cleared

    The actual memory fetch happens via KV cache attention after this FFN.
    The result is written back to RESULT by the attention layer.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy address (NIB_A) to TEMP for KV lookup
            self.W_up[0, E.OP_START + Opcode.LOAD] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.LOAD] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Clear RESULT (will be filled by attention)
            self.W_up[2, E.OP_START + Opcode.LOAD] = S
            self.W_gate[2, E.RESULT] = -1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.LOAD] = -S
            self.W_gate[3, E.RESULT] = 1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class StoreFFN(PureFFN):
    """
    STORE: Prepare address and value for memory write.

    Input: NIB_A[0:7] = 32-bit address nibbles
           NIB_B[0:7] = 32-bit value nibbles
    Output: TEMP = address, RESULT = value

    The actual memory write happens via KV cache append after this FFN.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy address (NIB_A) to TEMP
            self.W_up[0, E.OP_START + Opcode.STORE] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.STORE] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # Copy value (NIB_B) to RESULT
            self.W_up[2, E.OP_START + Opcode.STORE] = S
            self.W_gate[2, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.STORE] = -S
            self.W_gate[3, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class PushFFN(PureFFN):
    """
    PUSH: Prepare value for stack push.

    Input: NIB_A[0:7] = 32-bit value to push
    Output: RESULT = value (for storage at decremented SP)

    Stack pointer management happens externally.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy value (NIB_A) to RESULT
            self.W_up[0, E.OP_START + Opcode.PUSH] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.PUSH] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class PopFFN(PureFFN):
    """
    POP: Prepare for stack pop.

    Input: Stack pointer (external)
    Output: RESULT cleared (will be filled by KV lookup at SP)

    Stack pointer increment happens externally.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Clear RESULT (will be filled by attention from stack)
            self.W_up[0, E.OP_START + Opcode.POP] = S
            self.W_gate[0, E.RESULT] = -1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.POP] = -S
            self.W_gate[1, E.RESULT] = 1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class NopFFN(PureFFN):
    """NOP: No operation - just passes through."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=1)

    def _bake_weights(self):
        pass


class HaltFFN(PureFFN):
    """
    HALT: Signal program end.
    Sets RESULT to a special marker value (e.g., all 1s).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.b_up[0] = S
            self.W_gate[0, E.OP_START + Opcode.HALT] = 15.0
            self.W_down[E.RESULT, 0] = 1.0 / S
