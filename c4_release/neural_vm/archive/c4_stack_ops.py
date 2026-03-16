"""
C4 Stack and Control Flow Operations for Neural VM V7.

These operations implement the C4 compiler's stack-based calling convention:
- LEA: Load effective address (BP + offset)
- IMM: Load immediate value
- JSR: Jump to subroutine (push return address)
- BZ/BNZ: Branch if zero/not zero
- ENT: Enter function (setup stack frame)
- ADJ: Adjust stack pointer
- LEV: Leave function (teardown stack frame)
- PSH: Push AX to stack
- LI/LC: Load int/char from memory
- SI/SC: Store int/char to memory

The C4 VM uses:
- AX: Accumulator register
- PC: Program counter
- SP: Stack pointer
- BP: Base pointer (frame pointer)
"""

import torch

from .embedding import E, Opcode
from .base_layers import PureFFN


class LeaFFN(PureFFN):
    """
    LEA: Load Effective Address.

    AX = BP + immediate

    In our encoding:
    - NIB_A contains BP (base pointer) nibbles
    - NIB_B contains the immediate offset nibbles
    - RESULT = NIB_A + NIB_B (address calculation)

    This is essentially an ADD operation.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # RESULT = NIB_A + NIB_B (same as ADD's raw sum)
            self.W_up[0, E.OP_START + Opcode.LEA] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.LEA] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class LeaAddImmFFN(PureFFN):
    """Add immediate (NIB_B) to RAW_SUM for LEA."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.LEA] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RAW_SUM, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.LEA] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RAW_SUM, 1] = 1.0 / S


class LeaInitResultFFN(PureFFN):
    """Copy RAW_SUM to RESULT for LEA (before carry)."""

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy lower 4 bits of RAW_SUM to RESULT
            # Use step functions for each value 0-15
            for v in range(16):
                # We'd need many rows for exact copy
                # Simplified: direct copy works for small values
                pass

            # Simple approach: RESULT = RAW_SUM mod 16
            # step(RAW_SUM - v + 1) - step(RAW_SUM - v) for each v
            # For now, use linear approximation
            self.W_up[0, E.OP_START + Opcode.LEA] = S
            self.W_gate[0, E.RAW_SUM] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.LEA] = -S
            self.W_gate[1, E.RAW_SUM] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class ImmFFN(PureFFN):
    """
    IMM: Load Immediate.

    AX = immediate value

    Simply copies NIB_B (immediate) to RESULT.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # RESULT = NIB_B (immediate value)
            self.W_up[0, E.OP_START + Opcode.IMM] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.IMM] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class JsrFFN(PureFFN):
    """
    JSR: Jump to Subroutine.

    Push current PC (return address), then PC = target.

    In our encoding:
    - NIB_A contains return address (PC + 8)
    - NIB_B contains target address
    - RESULT = target address (new PC)
    - TEMP = return address (to be pushed to stack)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # RESULT = NIB_B (target address)
            self.W_up[0, E.OP_START + Opcode.JSR] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.JSR] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # TEMP = NIB_A (return address to push)
            self.W_up[2, E.OP_START + Opcode.JSR] = S
            self.W_gate[2, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.JSR] = -S
            self.W_gate[3, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class BzFFN(PureFFN):
    """
    BZ: Branch if Zero.

    if AX == 0 then PC = target

    In our encoding:
    - NIB_A contains AX (condition)
    - NIB_B contains target address
    - RESULT = target if AX == 0, else 0 (no branch)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # RESULT = NIB_B if NIB_A == 0
            # silu(1 - NIB_A) * NIB_B - silu(-NIB_A) * NIB_B
            self.W_up[0, E.NIB_A] = -S
            self.b_up[0] = S * 1.0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = -S
            self.b_up[1] = 0.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 1] = -1.0 / S


class BnzFFN(PureFFN):
    """
    BNZ: Branch if Not Zero.

    if AX != 0 then PC = target

    In our encoding:
    - NIB_A contains AX (condition)
    - NIB_B contains target address
    - RESULT = target if AX != 0, else 0 (no branch)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # RESULT = NIB_B if NIB_A >= 1
            # silu(NIB_A) * NIB_B - silu(NIB_A - 1) * NIB_B
            self.W_up[0, E.NIB_A] = S
            self.b_up[0] = 0.0
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.NIB_A] = S
            self.b_up[1] = -S * 1.0
            self.W_gate[1, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 1] = -1.0 / S


class EntFFN(PureFFN):
    """
    ENT: Enter Function.

    push BP; BP = SP; SP -= locals

    In our encoding:
    - NIB_A contains current BP (to save)
    - NIB_B contains locals size (stack space to allocate)
    - RESULT = new SP (SP - locals)
    - TEMP = old BP (to be pushed)

    The actual stack manipulation is done by the VM executor.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # TEMP = NIB_A (old BP to push)
            self.W_up[0, E.OP_START + Opcode.ENT] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ENT] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # RESULT = NIB_B (locals size - VM will compute SP - this)
            self.W_up[2, E.OP_START + Opcode.ENT] = S
            self.W_gate[2, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.ENT] = -S
            self.W_gate[3, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class AdjFFN(PureFFN):
    """
    ADJ: Adjust Stack.

    SP += adjustment

    In our encoding:
    - NIB_B contains adjustment value
    - RESULT = adjustment (VM will add to SP)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.ADJ] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.ADJ] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class LevFFN(PureFFN):
    """
    LEV: Leave Function.

    SP = BP; BP = pop; PC = pop

    In our encoding:
    - NIB_A contains saved BP (from stack)
    - NIB_B contains return address (from stack)
    - RESULT = return address (new PC)
    - TEMP = old BP (to restore)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # RESULT = NIB_B (return address)
            self.W_up[0, E.OP_START + Opcode.LEV] = S
            self.W_gate[0, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.LEV] = -S
            self.W_gate[1, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S

            # TEMP = NIB_A (saved BP to restore)
            self.W_up[2, E.OP_START + Opcode.LEV] = S
            self.W_gate[2, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.LEV] = -S
            self.W_gate[3, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 3] = 1.0 / S


class PshFFN(PureFFN):
    """
    PSH: Push AX to Stack.

    SP -= 8; *SP = AX

    In our encoding:
    - NIB_A contains AX (value to push)
    - RESULT = AX (for storage by VM)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.PSH] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.RESULT, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.PSH] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.RESULT, 1] = 1.0 / S


class LiFFN(PureFFN):
    """
    LI: Load Int.

    AX = *AX (load 8 bytes from address in AX)

    The actual memory lookup is done by the VM using KV cache.
    This layer just passes through the address for lookup.

    - NIB_A contains address
    - RESULT = value from memory (set by KV cache lookup)
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Pass through NIB_A as address for memory lookup
            self.W_up[0, E.OP_START + Opcode.LI] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S  # TEMP holds address

            self.W_up[1, E.OP_START + Opcode.LI] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class LcFFN(PureFFN):
    """
    LC: Load Char.

    AX = *(char*)AX (load 1 byte from address in AX)

    Same as LI but only loads a single byte.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=2)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            self.W_up[0, E.OP_START + Opcode.LC] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.LC] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S


class SiFFN(PureFFN):
    """
    SI: Store Int.

    *pop = AX (store 8 bytes)

    - NIB_A contains address (popped from stack)
    - NIB_B contains value (AX)
    - Both passed through for KV cache store
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # TEMP = NIB_A (address)
            self.W_up[0, E.OP_START + Opcode.SI] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SI] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # RESULT = NIB_B (value to store)
            self.W_up[2, E.OP_START + Opcode.SI] = S
            self.W_gate[2, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.SI] = -S
            self.W_gate[3, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S


class ScFFN(PureFFN):
    """
    SC: Store Char.

    *(char*)pop = AX (store 1 byte)

    Same as SI but only stores low byte.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # TEMP = NIB_A (address)
            self.W_up[0, E.OP_START + Opcode.SC] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.TEMP, 0] = 1.0 / S

            self.W_up[1, E.OP_START + Opcode.SC] = -S
            self.W_gate[1, E.NIB_A] = -1.0
            self.W_down[E.TEMP, 1] = 1.0 / S

            # RESULT = NIB_B (value to store, only low byte used)
            self.W_up[2, E.OP_START + Opcode.SC] = S
            self.W_gate[2, E.NIB_B] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S

            self.W_up[3, E.OP_START + Opcode.SC] = -S
            self.W_gate[3, E.NIB_B] = -1.0
            self.W_down[E.RESULT, 3] = 1.0 / S
