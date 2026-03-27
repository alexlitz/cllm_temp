"""
Efficient ALU operations for 8-bit values (C4 VM's actual bit width).

The C4 VM operates on 8-bit values (ALU_LO + ALU_HI = 2 nibbles), not 32-bit.
Using 8-bit chunk configs dramatically reduces parameter counts.

Parameter comparison (8-bit vs 32-bit NIBBLE):
- MUL: 798 vs 10,846 (93% reduction)
- SHL: 385 vs 2,812 (86% reduction)
- SHR: 385 vs 2,812 (86% reduction)
- ADD: ~164 vs 328 (50% reduction)
- SUB: ~164 vs 328 (50% reduction)
- AND: 486 vs 486 (same)
- OR: 510 vs 510 (same)
- XOR: 510 vs 510 (same)

Total: ~3,802 params (vs 18,632 for 32-bit = 80% reduction)
"""

import torch
import torch.nn as nn
from .alu.chunk_config import ChunkConfig
from .alu.ops.add import build_add_layers
from .alu.ops.sub import build_sub_layers
from .alu.ops.mul import build_mul_layers
from .alu.ops.shift import build_shl_layers, build_shr_layers
from .alu.ops.bitwise import build_and_layers, build_or_layers, build_xor_layers
from .alu.ops.common import GenericE


# 8-bit configuration: 4-bit chunks, 2 positions
NIBBLE_8BIT = ChunkConfig(chunk_bits=4, total_bits=8, precision='fp32')


class Efficient8BitALU_AddSub(nn.Module):
    """
    Efficient ADD/SUB for 8-bit values using 2-position nibble configuration.

    Parameters: ~328 (vs 656 for 32-bit = 50% reduction)
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.config = NIBBLE_8BIT
        self.ge = GenericE(NIBBLE_8BIT)

        # Build efficient 8-bit layers
        self.add_layers = build_add_layers(NIBBLE_8BIT, opcode=25)
        self.sub_layers = build_sub_layers(NIBBLE_8BIT, opcode=26)

        for layer in self.add_layers + self.sub_layers:
            layer.eval()

    def forward(self, x_bd):
        B, seq_len, _ = x_bd.shape
        x_bd_out = x_bd.clone()

        for b in range(B):
            for pos in range(seq_len):
                if x_bd[b, pos, self.BD.MARK_AX] < 0.5:
                    continue

                # Convert to GenericE (only 2 positions for 8-bit)
                x_ge = self.bd_to_ge(x_bd[b, pos])

                # Run operation
                if x_bd[b, pos, self.BD.OP_ADD] > 0.5:
                    x_ge = x_ge.unsqueeze(0)
                    with torch.no_grad():
                        for layer in self.add_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)
                elif x_bd[b, pos, self.BD.OP_SUB] > 0.5:
                    x_ge = x_ge.unsqueeze(0)
                    with torch.no_grad():
                        for layer in self.sub_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)
                else:
                    continue

                self.ge_to_bd(x_ge, x_bd_out[b, pos])

        return x_bd_out

    def bd_to_ge(self, x_bd_single):
        """Convert BD to GenericE for 8-bit (2 positions)."""
        x_ge = torch.zeros(2, self.ge.DIM, dtype=x_bd_single.dtype, device=x_bd_single.device)

        # Position 0: Low nibble
        for k in range(16):
            if x_bd_single[self.BD.ALU_LO + k] > 0.5:
                x_ge[0, self.ge.NIB_A] = float(k)
                break

        # Position 1: High nibble
        for k in range(16):
            if x_bd_single[self.BD.ALU_HI + k] > 0.5:
                x_ge[1, self.ge.NIB_A] = float(k)
                break

        # Operand B
        for k in range(16):
            if x_bd_single[self.BD.AX_CARRY_LO + k] > 0.5:
                x_ge[0, self.ge.NIB_B] = float(k)
                break
        for k in range(16):
            if x_bd_single[self.BD.AX_CARRY_HI + k] > 0.5:
                x_ge[1, self.ge.NIB_B] = float(k)
                break

        # Copy opcodes
        for opcode_idx in [25, 26]:
            op_val = x_bd_single[self.BD.OPCODE_BASE + opcode_idx]
            for p in range(2):
                x_ge[p, self.ge.OP_START + opcode_idx] = op_val

        return x_ge

    def ge_to_bd(self, x_ge, x_bd_single):
        """Convert GenericE result back to BD."""
        result_lo = int(x_ge[0, self.ge.RESULT].round().item())
        result_hi = int(x_ge[1, self.ge.RESULT].round().item())
        result_lo = max(0, min(15, result_lo))
        result_hi = max(0, min(15, result_hi))
        x_bd_single[self.BD.OUTPUT_LO + result_lo] += 2.0
        x_bd_single[self.BD.OUTPUT_HI + result_hi] += 2.0

    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class Efficient8BitALU_Mul(nn.Module):
    """
    Efficient MUL for 8-bit values using 2-position nibble configuration.

    Parameters: 798 (vs 10,846 for 32-bit = 93% reduction!)
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.config = NIBBLE_8BIT
        self.ge = GenericE(NIBBLE_8BIT)

        self.mul_layers = build_mul_layers(NIBBLE_8BIT, opcode=27)
        for layer in self.mul_layers:
            layer.eval()

    def forward(self, x_bd):
        B, seq_len, _ = x_bd.shape
        x_bd_out = x_bd.clone()

        for b in range(B):
            for pos in range(seq_len):
                if x_bd[b, pos, self.BD.MARK_AX] < 0.5 or x_bd[b, pos, self.BD.OP_MUL] < 0.5:
                    continue

                x_ge = self.bd_to_ge(x_bd[b, pos])
                x_ge = x_ge.unsqueeze(0)
                with torch.no_grad():
                    for layer in self.mul_layers:
                        x_ge = layer(x_ge)
                x_ge = x_ge.squeeze(0)
                self.ge_to_bd(x_ge, x_bd_out[b, pos])

        return x_bd_out

    def bd_to_ge(self, x_bd_single):
        x_ge = torch.zeros(2, self.ge.DIM, dtype=x_bd_single.dtype, device=x_bd_single.device)
        for k in range(16):
            if x_bd_single[self.BD.ALU_LO + k] > 0.5:
                x_ge[0, self.ge.NIB_A] = float(k)
                break
        for k in range(16):
            if x_bd_single[self.BD.ALU_HI + k] > 0.5:
                x_ge[1, self.ge.NIB_A] = float(k)
                break
        for k in range(16):
            if x_bd_single[self.BD.AX_CARRY_LO + k] > 0.5:
                x_ge[0, self.ge.NIB_B] = float(k)
                break
        for k in range(16):
            if x_bd_single[self.BD.AX_CARRY_HI + k] > 0.5:
                x_ge[1, self.ge.NIB_B] = float(k)
                break
        op_val = x_bd_single[self.BD.OPCODE_BASE + 27]
        for p in range(2):
            x_ge[p, self.ge.OP_START + 27] = op_val
        return x_ge

    def ge_to_bd(self, x_ge, x_bd_single):
        result_lo = int(x_ge[0, self.ge.RESULT].round().item())
        result_hi = int(x_ge[1, self.ge.RESULT].round().item())
        result_lo = max(0, min(15, result_lo))
        result_hi = max(0, min(15, result_hi))
        x_bd_single[self.BD.OUTPUT_LO + result_lo] += 2.0
        x_bd_single[self.BD.OUTPUT_HI + result_hi] += 2.0

    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class Efficient8BitALU_Shift(nn.Module):
    """
    Efficient SHIFT for 8-bit values using 2-position nibble configuration.

    Parameters: 770 total (385 each for SHL/SHR) vs 5,624 for 32-bit = 86% reduction
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.config = NIBBLE_8BIT
        self.ge = GenericE(NIBBLE_8BIT)

        self.shl_layers = build_shl_layers(NIBBLE_8BIT, opcode=23)
        self.shr_layers = build_shr_layers(NIBBLE_8BIT, opcode=24)
        for layer in self.shl_layers + self.shr_layers:
            layer.eval()

    def forward(self, x_bd):
        B, seq_len, _ = x_bd.shape
        x_bd_out = x_bd.clone()

        for b in range(B):
            for pos in range(seq_len):
                if x_bd[b, pos, self.BD.MARK_AX] < 0.5:
                    continue

                x_ge = self.bd_to_ge(x_bd[b, pos])

                if x_bd[b, pos, self.BD.OP_SHL] > 0.5:
                    x_ge = x_ge.unsqueeze(0)
                    with torch.no_grad():
                        for layer in self.shl_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)
                elif x_bd[b, pos, self.BD.OP_SHR] > 0.5:
                    x_ge = x_ge.unsqueeze(0)
                    with torch.no_grad():
                        for layer in self.shr_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)
                else:
                    continue

                self.ge_to_bd(x_ge, x_bd_out[b, pos])

        return x_bd_out

    def bd_to_ge(self, x_bd_single):
        x_ge = torch.zeros(2, self.ge.DIM, dtype=x_bd_single.dtype, device=x_bd_single.device)
        for k in range(16):
            if x_bd_single[self.BD.ALU_LO + k] > 0.5:
                x_ge[0, self.ge.NIB_A] = float(k)
                break
        for k in range(16):
            if x_bd_single[self.BD.ALU_HI + k] > 0.5:
                x_ge[1, self.ge.NIB_A] = float(k)
                break
        shift_amount = 0
        for k in range(16):
            if x_bd_single[self.BD.AX_CARRY_LO + k] > 0.5:
                shift_amount = k
                break
        for i in range(2):
            x_ge[i, self.ge.NIB_B] = float(shift_amount % 16)
            shift_amount //= 16
        for opcode_idx in [23, 24]:
            op_val = x_bd_single[self.BD.OPCODE_BASE + opcode_idx]
            for p in range(2):
                x_ge[p, self.ge.OP_START + opcode_idx] = op_val
        return x_ge

    def ge_to_bd(self, x_ge, x_bd_single):
        result_lo = int(x_ge[0, self.ge.RESULT].round().item())
        result_hi = int(x_ge[1, self.ge.RESULT].round().item())
        result_lo = max(0, min(15, result_lo))
        result_hi = max(0, min(15, result_hi))
        x_bd_single[self.BD.OUTPUT_LO + result_lo] += 2.0
        x_bd_single[self.BD.OUTPUT_HI + result_hi] += 2.0

    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class Efficient8BitALU_Bitwise(nn.Module):
    """
    Efficient bitwise operations for 8-bit values using 2-position nibble configuration.

    Parameters: ~1,506 (same as 32-bit - bitwise doesn't scale with positions)
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.config = NIBBLE_8BIT
        self.ge = GenericE(NIBBLE_8BIT)

        self.and_layers = build_and_layers(NIBBLE_8BIT, opcode=30)
        self.or_layers = build_or_layers(NIBBLE_8BIT, opcode=28)
        self.xor_layers = build_xor_layers(NIBBLE_8BIT, opcode=29)
        for layer in self.and_layers + self.or_layers + self.xor_layers:
            layer.eval()

    def forward(self, x_bd):
        B, seq_len, _ = x_bd.shape
        x_bd_out = x_bd.clone()

        for b in range(B):
            for pos in range(seq_len):
                if x_bd[b, pos, self.BD.MARK_AX] < 0.5:
                    continue

                is_and = x_bd[b, pos, self.BD.OP_AND] > 0.5
                is_or = x_bd[b, pos, self.BD.OP_OR] > 0.5
                is_xor = x_bd[b, pos, self.BD.OP_XOR] > 0.5
                if not (is_and or is_or or is_xor):
                    continue

                x_ge = self.bd_to_ge(x_bd[b, pos])

                if is_and:
                    x_ge = x_ge.unsqueeze(0)
                    with torch.no_grad():
                        for layer in self.and_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)
                elif is_or:
                    x_ge = x_ge.unsqueeze(0)
                    with torch.no_grad():
                        for layer in self.or_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)
                elif is_xor:
                    x_ge = x_ge.unsqueeze(0)
                    with torch.no_grad():
                        for layer in self.xor_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)

                self.ge_to_bd(x_ge, x_bd_out[b, pos])

        return x_bd_out

    def bd_to_ge(self, x_bd_single):
        x_ge = torch.zeros(2, self.ge.DIM, dtype=x_bd_single.dtype, device=x_bd_single.device)
        for k in range(16):
            if x_bd_single[self.BD.ALU_LO + k] > 0.5:
                x_ge[0, self.ge.NIB_A] = float(k)
                break
        for k in range(16):
            if x_bd_single[self.BD.ALU_HI + k] > 0.5:
                x_ge[1, self.ge.NIB_A] = float(k)
                break
        for k in range(16):
            if x_bd_single[self.BD.AX_CARRY_LO + k] > 0.5:
                x_ge[0, self.ge.NIB_B] = float(k)
                break
        for k in range(16):
            if x_bd_single[self.BD.AX_CARRY_HI + k] > 0.5:
                x_ge[1, self.ge.NIB_B] = float(k)
                break
        opcode_map = [(self.BD.OP_OR, 28), (self.BD.OP_XOR, 29), (self.BD.OP_AND, 30)]
        for bd_idx, ge_opcode in opcode_map:
            op_val = x_bd_single[bd_idx]
            for p in range(2):
                x_ge[p, self.ge.OP_START + ge_opcode] = op_val
        return x_ge

    def ge_to_bd(self, x_ge, x_bd_single):
        result_lo = int(x_ge[0, self.ge.RESULT].round().item())
        result_hi = int(x_ge[1, self.ge.RESULT].round().item())
        result_lo = max(0, min(15, result_lo))
        result_hi = max(0, min(15, result_hi))
        x_bd_single[self.BD.OUTPUT_LO + result_lo] += 2.0
        x_bd_single[self.BD.OUTPUT_HI + result_hi] += 2.0

    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


def integrate_efficient_8bit_alu(model, S, BD):
    """
    Integrate all efficient 8-bit ALU operations.

    Expected parameter reduction:
    - Before: ~107,250 params (layers 8-13)
    - After: ~3,802 params
    - Savings: ~103,448 params (96% reduction!)
    """
    stats = {'before': {}, 'after': {}}

    for i in [8, 9, 10, 11, 12, 13]:
        stats['before'][f'L{i}'] = sum((p != 0).sum().item() for p in model.blocks[i].ffn.parameters())

    print("Integrating efficient 8-bit ALU operations...")
    print("  L8: Efficient8BitALU_AddSub (ADD/SUB)")
    model.blocks[8].ffn = Efficient8BitALU_AddSub(S, BD)

    print("  L10: Efficient8BitALU_Bitwise (AND/OR/XOR)")
    model.blocks[10].ffn = Efficient8BitALU_Bitwise(S, BD)

    print("  L11: Efficient8BitALU_Mul (MUL)")
    model.blocks[11].ffn = Efficient8BitALU_Mul(S, BD)

    print("  L13: Efficient8BitALU_Shift (SHL/SHR)")
    model.blocks[13].ffn = Efficient8BitALU_Shift(S, BD)

    for i in [8, 9, 10, 11, 12, 13]:
        ffn = model.blocks[i].ffn
        if hasattr(ffn, 'add_layers'):
            params = sum(sum((p != 0).sum().item() for p in layer.parameters())
                        for layer in ffn.add_layers + ffn.sub_layers)
        elif hasattr(ffn, 'and_layers'):
            params = sum(sum((p != 0).sum().item() for p in layer.parameters())
                        for layer in ffn.and_layers + ffn.or_layers + ffn.xor_layers)
        elif hasattr(ffn, 'mul_layers'):
            params = sum(sum((p != 0).sum().item() for p in layer.parameters())
                        for layer in ffn.mul_layers)
        elif hasattr(ffn, 'shl_layers'):
            params = sum(sum((p != 0).sum().item() for p in layer.parameters())
                        for layer in ffn.shl_layers + ffn.shr_layers)
        else:
            params = sum((p != 0).sum().item() for p in ffn.parameters())
        stats['after'][f'L{i}'] = params

    print("\nParameter counts:")
    print("  Layer  | Before    | After    | Savings  | Reduction | Notes")
    print("  " + "-"*70)
    total_before = 0
    total_after = 0
    for i in [8, 9, 10, 11, 12, 13]:
        before = stats['before'][f'L{i}']
        after = stats['after'][f'L{i}']
        savings = before - after
        pct = savings / before * 100 if before > 0 else 0
        note = ""
        if i == 9:
            note = "(kept for comparisons)"
        elif i == 12:
            note = "(freed, MUL in L11)"
        print(f"  L{i:2d}    | {before:8,} | {after:7,} | {savings:7,} | {pct:5.1f}% | {note}")
        total_before += before
        total_after += after

    print("  " + "-"*70)
    total_savings = total_before - total_after
    total_pct = total_savings / total_before * 100 if total_before > 0 else 0
    print(f"  Total  | {total_before:8,} | {total_after:7,} | {total_savings:7,} | {total_pct:5.1f}%")

    return stats


if __name__ == '__main__':
    print("="*70)
    print("Efficient 8-bit ALU Integration")
    print("="*70)
    print("\nKey Insight: C4 VM uses 8-bit values (2 nibbles), not 32-bit (8 nibbles)")
    print("Using 8-bit chunk configs provides 80-96% parameter reduction!\n")

    def count_params(layers):
        return sum(sum((p != 0).sum().item() for p in layer.parameters()) for layer in layers)

    print("Parameter counts for 8-bit operations:")
    print(f"  ADD: {count_params(build_add_layers(NIBBLE_8BIT, 25)):,}")
    print(f"  SUB: {count_params(build_sub_layers(NIBBLE_8BIT, 26)):,}")
    print(f"  MUL: {count_params(build_mul_layers(NIBBLE_8BIT, 27)):,}")
    print(f"  SHL: {count_params(build_shl_layers(NIBBLE_8BIT, 23)):,}")
    print(f"  SHR: {count_params(build_shr_layers(NIBBLE_8BIT, 24)):,}")
    print(f"  AND: {count_params(build_and_layers(NIBBLE_8BIT, 30)):,}")
    print(f"  OR: {count_params(build_or_layers(NIBBLE_8BIT, 28)):,}")
    print(f"  XOR: {count_params(build_xor_layers(NIBBLE_8BIT, 29)):,}")

    total = (count_params(build_add_layers(NIBBLE_8BIT, 25)) +
             count_params(build_sub_layers(NIBBLE_8BIT, 26)) +
             count_params(build_mul_layers(NIBBLE_8BIT, 27)) +
             count_params(build_shl_layers(NIBBLE_8BIT, 23)) +
             count_params(build_shr_layers(NIBBLE_8BIT, 24)) +
             count_params(build_and_layers(NIBBLE_8BIT, 30)) +
             count_params(build_or_layers(NIBBLE_8BIT, 28)) +
             count_params(build_xor_layers(NIBBLE_8BIT, 29)))

    print(f"\n  Total: {total:,} params")
    print(f"  vs 32-bit: 18,632 params")
    print(f"  Savings: {18632 - total:,} params ({(18632-total)/18632*100:.1f}% reduction)")
