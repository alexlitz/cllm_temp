"""
Efficient ALU modules with toggleable lookup/efficient modes.

All modules use fp32 MagicFloor trick for carry/floor extraction
instead of massive lookup tables.

Each module supports:
- mode='lookup': Original lookup table approach (pure FFN)
- mode='efficient': MagicFloor-based computation (smaller params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# fp32 MAGIC constant for floor extraction
MAGIC32 = 1.5 * float(2**23)  # 12582912.0


def magic_floor(x: torch.Tensor) -> torch.Tensor:
    """Compute floor(x) using fp32 MAGIC trick.

    Valid for values in [0, 2^24). For chunk operations this is sufficient.
    """
    x32 = x.float()
    x_shifted = x32 - 0.5 + 0.001  # eps=0.001 for fp32 stability
    result = (x_shifted + MAGIC32) - MAGIC32
    return result


class MulModule(nn.Module):
    """MUL module supporting lookup and efficient modes.

    Computes: result = (a * b) mod 256, where a and b are 8-bit values
    encoded as nibbles (ALU_LO/HI for a, AX_CARRY_LO/HI for b).

    Mode 'lookup': 8192 hidden units (4096 per layer equivalent)
    Mode 'efficient': Direct multiplication with MagicFloor for carries
    """

    def __init__(self, d_model=512, S=100.0, mode='efficient'):
        super().__init__()
        self.d_model = d_model
        self.S = S
        self.mode = mode

        if mode == 'lookup':
            self._init_lookup_mode(S)
        elif mode == 'efficient':
            self._init_efficient_mode(S)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_lookup_mode(self, S):
        """Full lookup table: 256 * 256 = 65536 combinations."""
        from neural_vm.vm_step import _SetDim as BD

        n_units = 256 * 256
        self.W_up = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(n_units))
        self.W_gate = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(n_units))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, n_units))

        unit = 0
        for a in range(256):
            for b in range(256):
                a_lo, a_hi = a % 16, a // 16
                b_lo, b_hi = b % 16, b // 16

                result = (a * b) % 256
                r_lo, r_hi = result % 16, result // 16

                # 5-way AND: MARK_AX + ALU_LO + ALU_HI + AX_CARRY_LO + AX_CARRY_HI
                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S
                self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
                self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
                self.b_up.data[unit] = -4.5 * S

                self.W_gate.data[unit, BD.OP_MUL] = 1.0

                self.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0
                self.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0

                unit += 1

    def _init_efficient_mode(self, S):
        """Efficient mode: small FFN + runtime MagicFloor computation."""
        # Minimal FFN - actual computation done in forward()
        self.W_up = nn.Parameter(torch.zeros(1, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(1))
        self.W_gate = nn.Parameter(torch.zeros(1, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(1))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from neural_vm.vm_step import _SetDim as BD

        if self.mode == 'lookup':
            # Standard SwiGLU FFN
            up = F.linear(x, self.W_up, self.b_up)
            gate = F.linear(x, self.W_gate, self.b_gate)
            hidden = F.silu(up) * gate
            return x + F.linear(hidden, self.W_down)
        else:
            # Efficient mode: extract operands, compute, write result
            B = x.shape[0]

            # Check if MUL operation is active
            op_mul = x[:, BD.OP_MUL]
            mark_ax = x[:, BD.MARK_AX]
            active = (op_mul > 0.5) & (mark_ax > 0.5)

            if not active.any():
                return x

            # Extract a from ALU nibbles
            a_lo = x[:, BD.ALU_LO:BD.ALU_LO+16].argmax(dim=-1).float()
            a_hi = x[:, BD.ALU_HI:BD.ALU_HI+16].argmax(dim=-1).float()
            a = a_lo + 16 * a_hi

            # Extract b from AX_CARRY nibbles
            b_lo = x[:, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].argmax(dim=-1).float()
            b_hi = x[:, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16].argmax(dim=-1).float()
            b = b_lo + 16 * b_hi

            # Compute product mod 256 using MagicFloor
            product = a * b
            result = product - magic_floor(product / 256) * 256

            # Extract result nibbles
            r_lo = result - magic_floor(result / 16) * 16
            r_hi = magic_floor(result / 16)

            # Write to output
            delta = torch.zeros_like(x)
            for i in range(B):
                if active[i]:
                    r_lo_i = int(r_lo[i].item()) % 16
                    r_hi_i = int(r_hi[i].item()) % 16
                    delta[i, BD.OUTPUT_LO + r_lo_i] = 1.0
                    delta[i, BD.OUTPUT_HI + r_hi_i] = 1.0

            return x + delta


class ShiftModule(nn.Module):
    """SHL/SHR module supporting lookup and efficient modes.

    SHL: result = (a << b) mod 256
    SHR: result = a >> b

    Mode 'lookup': 256 * 32 * 2 = 16384 hidden units
    Mode 'efficient': Direct shift with MagicFloor
    """

    def __init__(self, d_model=512, S=100.0, mode='efficient'):
        super().__init__()
        self.d_model = d_model
        self.S = S
        self.mode = mode

        if mode == 'lookup':
            self._init_lookup_mode(S)
        elif mode == 'efficient':
            self._init_efficient_mode(S)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_lookup_mode(self, S):
        """Lookup table for SHL and SHR."""
        from neural_vm.vm_step import _SetDim as BD

        # 256 values * 8 shift amounts (0-7 meaningful for 8-bit) * 2 ops
        n_units = 256 * 8 * 2
        self.W_up = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(n_units))
        self.W_gate = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(n_units))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, n_units))

        unit = 0

        # SHL units
        for a in range(256):
            a_lo, a_hi = a % 16, a // 16
            for shift in range(8):
                result = (a << shift) % 256
                r_lo, r_hi = result % 16, result // 16

                # 4-way AND: MARK_AX + ALU_LO + ALU_HI + shift_amount
                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S
                self.W_up.data[unit, BD.AX_CARRY_LO + shift] = S
                self.b_up.data[unit] = -3.5 * S

                self.W_gate.data[unit, BD.OP_SHL] = 1.0

                self.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0
                self.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0

                unit += 1

        # SHR units
        for a in range(256):
            a_lo, a_hi = a % 16, a // 16
            for shift in range(8):
                result = a >> shift
                r_lo, r_hi = result % 16, result // 16

                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S
                self.W_up.data[unit, BD.AX_CARRY_LO + shift] = S
                self.b_up.data[unit] = -3.5 * S

                self.W_gate.data[unit, BD.OP_SHR] = 1.0

                self.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0
                self.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0

                unit += 1

    def _init_efficient_mode(self, S):
        """Minimal FFN for efficient mode."""
        self.W_up = nn.Parameter(torch.zeros(1, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(1))
        self.W_gate = nn.Parameter(torch.zeros(1, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(1))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from neural_vm.vm_step import _SetDim as BD

        if self.mode == 'lookup':
            up = F.linear(x, self.W_up, self.b_up)
            gate = F.linear(x, self.W_gate, self.b_gate)
            hidden = F.silu(up) * gate
            return x + F.linear(hidden, self.W_down)
        else:
            B = x.shape[0]

            op_shl = x[:, BD.OP_SHL]
            op_shr = x[:, BD.OP_SHR]
            mark_ax = x[:, BD.MARK_AX]
            active_shl = (op_shl > 0.5) & (mark_ax > 0.5)
            active_shr = (op_shr > 0.5) & (mark_ax > 0.5)

            if not (active_shl.any() or active_shr.any()):
                return x

            # Extract a from ALU nibbles
            a_lo = x[:, BD.ALU_LO:BD.ALU_LO+16].argmax(dim=-1).float()
            a_hi = x[:, BD.ALU_HI:BD.ALU_HI+16].argmax(dim=-1).float()
            a = a_lo + 16 * a_hi

            # Extract shift amount from AX_CARRY_LO (0-7)
            shift = x[:, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].argmax(dim=-1).float()
            shift = shift.clamp(0, 7)  # Only 0-7 meaningful

            # Compute shifts
            # SHL: (a * 2^shift) mod 256
            # SHR: floor(a / 2^shift)
            pow2 = (2.0 ** shift)

            shl_result = a * pow2
            shl_result = shl_result - magic_floor(shl_result / 256) * 256

            shr_result = magic_floor(a / pow2)

            # Select result based on operation
            result = torch.where(active_shl.unsqueeze(-1).expand_as(shl_result.unsqueeze(-1)).squeeze(-1),
                                shl_result, shr_result)

            # Extract result nibbles
            r_lo = result - magic_floor(result / 16) * 16
            r_hi = magic_floor(result / 16)

            # Write to output
            delta = torch.zeros_like(x)
            active = active_shl | active_shr
            for i in range(B):
                if active[i]:
                    r_lo_i = int(r_lo[i].item()) % 16
                    r_hi_i = int(r_hi[i].item()) % 16
                    delta[i, BD.OUTPUT_LO + r_lo_i] = 1.0
                    delta[i, BD.OUTPUT_HI + r_hi_i] = 1.0

            return x + delta


class AddSubModule(nn.Module):
    """ADD/SUB module supporting lookup and efficient modes.

    ADD: result = (a + b) mod 256, carry = (a + b) >= 256
    SUB: result = (a - b) mod 256, borrow = a < b

    Mode 'lookup': 256 * 256 * 2 = 131072 hidden units
    Mode 'efficient': Direct arithmetic with MagicFloor
    """

    def __init__(self, d_model=512, S=100.0, mode='efficient'):
        super().__init__()
        self.d_model = d_model
        self.S = S
        self.mode = mode

        if mode == 'lookup':
            self._init_lookup_mode(S)
        elif mode == 'efficient':
            self._init_efficient_mode(S)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_lookup_mode(self, S):
        """Full lookup for ADD and SUB."""
        from neural_vm.vm_step import _SetDim as BD

        n_units = 256 * 256 * 2  # ADD + SUB
        self.W_up = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(n_units))
        self.W_gate = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(n_units))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, n_units))

        unit = 0

        # ADD units
        for a in range(256):
            a_lo, a_hi = a % 16, a // 16
            for b in range(256):
                b_lo, b_hi = b % 16, b // 16

                result = (a + b) % 256
                carry = 1 if (a + b) >= 256 else 0
                r_lo, r_hi = result % 16, result // 16

                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S
                self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
                self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
                self.b_up.data[unit] = -4.5 * S

                self.W_gate.data[unit, BD.OP_ADD] = 1.0

                self.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0
                self.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0
                if carry:
                    self.W_down.data[BD.CARRY, unit] = 1.0

                unit += 1

        # SUB units
        for a in range(256):
            a_lo, a_hi = a % 16, a // 16
            for b in range(256):
                b_lo, b_hi = b % 16, b // 16

                result = (a - b) % 256
                borrow = 1 if a < b else 0
                r_lo, r_hi = result % 16, result // 16

                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S
                self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
                self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
                self.b_up.data[unit] = -4.5 * S

                self.W_gate.data[unit, BD.OP_SUB] = 1.0

                self.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0
                self.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0
                if borrow:
                    self.W_down.data[BD.CARRY, unit] = 1.0

                unit += 1

    def _init_efficient_mode(self, S):
        """Minimal FFN for efficient mode."""
        self.W_up = nn.Parameter(torch.zeros(1, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(1))
        self.W_gate = nn.Parameter(torch.zeros(1, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(1))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from neural_vm.vm_step import _SetDim as BD

        if self.mode == 'lookup':
            up = F.linear(x, self.W_up, self.b_up)
            gate = F.linear(x, self.W_gate, self.b_gate)
            hidden = F.silu(up) * gate
            return x + F.linear(hidden, self.W_down)
        else:
            B = x.shape[0]

            # Debug: check tensor shape
            if x.shape[1] < 300:
                print(f"WARNING: x.shape={x.shape}, expected d_model=512+")
                print(f"BD.OP_ADD={BD.OP_ADD}")
                raise ValueError(f"Tensor x has wrong shape: {x.shape}, expected at least 512 dimensions")

            op_add = x[:, BD.OP_ADD]
            op_sub = x[:, BD.OP_SUB]
            mark_ax = x[:, BD.MARK_AX]
            active_add = (op_add > 0.5) & (mark_ax > 0.5)
            active_sub = (op_sub > 0.5) & (mark_ax > 0.5)

            if not (active_add.any() or active_sub.any()):
                return x

            # Extract operands
            a_lo = x[:, BD.ALU_LO:BD.ALU_LO+16].argmax(dim=-1).float()
            a_hi = x[:, BD.ALU_HI:BD.ALU_HI+16].argmax(dim=-1).float()
            a = a_lo + 16 * a_hi

            b_lo = x[:, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].argmax(dim=-1).float()
            b_hi = x[:, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16].argmax(dim=-1).float()
            b = b_lo + 16 * b_hi

            # ADD: result = (a + b) mod 256, carry = floor((a+b)/256)
            add_sum = a + b
            add_result = add_sum - magic_floor(add_sum / 256) * 256
            add_carry = magic_floor(add_sum / 256)

            # SUB: result = (a - b + 256) mod 256, borrow = (a < b)
            sub_diff = a - b + 256
            sub_result = sub_diff - magic_floor(sub_diff / 256) * 256
            sub_borrow = (a < b).float()

            # Select based on operation
            result = torch.zeros(B, device=x.device)
            carry_borrow = torch.zeros(B, device=x.device)

            result = torch.where(active_add, add_result, result)
            result = torch.where(active_sub, sub_result, result)
            carry_borrow = torch.where(active_add, add_carry, carry_borrow)
            carry_borrow = torch.where(active_sub, sub_borrow, carry_borrow)

            # Extract result nibbles
            r_lo = result - magic_floor(result / 16) * 16
            r_hi = magic_floor(result / 16)

            # Write to output
            delta = torch.zeros_like(x)
            active = active_add | active_sub
            for i in range(B):
                if active[i]:
                    r_lo_i = int(r_lo[i].item()) % 16
                    r_hi_i = int(r_hi[i].item()) % 16
                    delta[i, BD.OUTPUT_LO + r_lo_i] = 1.0
                    delta[i, BD.OUTPUT_HI + r_hi_i] = 1.0
                    if carry_borrow[i] > 0.5:
                        delta[i, BD.CARRY] = 1.0

            return x + delta


class BitwiseModule(nn.Module):
    """AND/OR/XOR module supporting lookup and efficient modes.

    Mode 'lookup': 256 * 256 * 3 = 196608 hidden units
    Mode 'efficient': Direct bitwise ops
    """

    def __init__(self, d_model=512, S=100.0, mode='efficient'):
        super().__init__()
        self.d_model = d_model
        self.S = S
        self.mode = mode

        if mode == 'lookup':
            self._init_lookup_mode(S)
        elif mode == 'efficient':
            self._init_efficient_mode(S)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _init_lookup_mode(self, S):
        """Full lookup for AND, OR, XOR."""
        from neural_vm.vm_step import _SetDim as BD

        n_units = 256 * 256 * 3  # AND + OR + XOR
        self.W_up = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(n_units))
        self.W_gate = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(n_units))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, n_units))

        unit = 0

        for op_name, op_func, op_dim in [
            ('AND', lambda a, b: a & b, BD.OP_AND),
            ('OR', lambda a, b: a | b, BD.OP_OR),
            ('XOR', lambda a, b: a ^ b, BD.OP_XOR),
        ]:
            for a in range(256):
                a_lo, a_hi = a % 16, a // 16
                for b in range(256):
                    b_lo, b_hi = b % 16, b // 16

                    result = op_func(a, b)
                    r_lo, r_hi = result % 16, result // 16

                    self.W_up.data[unit, BD.MARK_AX] = S
                    self.W_up.data[unit, BD.ALU_LO + a_lo] = S
                    self.W_up.data[unit, BD.ALU_HI + a_hi] = S
                    self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
                    self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
                    self.b_up.data[unit] = -4.5 * S

                    self.W_gate.data[unit, op_dim] = 1.0

                    self.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0
                    self.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0

                    unit += 1

    def _init_efficient_mode(self, S):
        """Minimal FFN for efficient mode."""
        self.W_up = nn.Parameter(torch.zeros(1, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(1))
        self.W_gate = nn.Parameter(torch.zeros(1, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(1))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from neural_vm.vm_step import _SetDim as BD

        if self.mode == 'lookup':
            up = F.linear(x, self.W_up, self.b_up)
            gate = F.linear(x, self.W_gate, self.b_gate)
            hidden = F.silu(up) * gate
            return x + F.linear(hidden, self.W_down)
        else:
            B = x.shape[0]

            op_and = x[:, BD.OP_AND]
            op_or = x[:, BD.OP_OR]
            op_xor = x[:, BD.OP_XOR]
            mark_ax = x[:, BD.MARK_AX]

            active_and = (op_and > 0.5) & (mark_ax > 0.5)
            active_or = (op_or > 0.5) & (mark_ax > 0.5)
            active_xor = (op_xor > 0.5) & (mark_ax > 0.5)

            if not (active_and.any() or active_or.any() or active_xor.any()):
                return x

            # Extract operands
            a_lo = x[:, BD.ALU_LO:BD.ALU_LO+16].argmax(dim=-1)
            a_hi = x[:, BD.ALU_HI:BD.ALU_HI+16].argmax(dim=-1)
            a = a_lo + 16 * a_hi

            b_lo = x[:, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].argmax(dim=-1)
            b_hi = x[:, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16].argmax(dim=-1)
            b = b_lo + 16 * b_hi

            # Compute bitwise ops (using integer tensors)
            and_result = a & b
            or_result = a | b
            xor_result = a ^ b

            # Select based on operation
            result = torch.zeros(B, dtype=torch.long, device=x.device)
            result = torch.where(active_and, and_result, result)
            result = torch.where(active_or, or_result, result)
            result = torch.where(active_xor, xor_result, result)

            # Extract result nibbles
            r_lo = result % 16
            r_hi = result // 16

            # Write to output
            delta = torch.zeros_like(x)
            active = active_and | active_or | active_xor
            for i in range(B):
                if active[i]:
                    delta[i, BD.OUTPUT_LO + r_lo[i]] = 1.0
                    delta[i, BD.OUTPUT_HI + r_hi[i]] = 1.0

            return x + delta


# Global mode setting
_ALU_MODE = 'efficient'

def set_alu_mode(mode: str):
    """Set the default ALU mode for all operations."""
    global _ALU_MODE
    if mode not in ('lookup', 'efficient'):
        raise ValueError(f"Unknown mode: {mode}")
    _ALU_MODE = mode

def get_alu_mode() -> str:
    """Get the current default ALU mode."""
    return _ALU_MODE


def create_alu_modules(mode: str = None) -> dict:
    """Create all ALU modules with the specified mode.

    Returns dict with keys: 'mul', 'shift', 'addsub', 'bitwise'
    """
    if mode is None:
        mode = _ALU_MODE

    return {
        'mul': MulModule(mode=mode),
        'shift': ShiftModule(mode=mode),
        'addsub': AddSubModule(mode=mode),
        'bitwise': BitwiseModule(mode=mode),
    }


def count_alu_params(mode: str = None) -> dict:
    """Count parameters for each ALU module in the specified mode."""
    modules = create_alu_modules(mode)

    counts = {}
    for name, module in modules.items():
        total = sum(p.numel() for p in module.parameters())
        nonzero = sum((p != 0).sum().item() for p in module.parameters())
        counts[name] = {'total': total, 'nonzero': nonzero}

    return counts
