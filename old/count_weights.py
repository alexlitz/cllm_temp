"""
Count non-zero weights and biases in the C4 transformer.
"""

import torch
from c4_printf_autoregressive import C4AutoregressivePrintf
from c4_autoregressive import C4AutoregressiveExecutor
from c4_moe_top1 import (
    MoERouter, BitwiseExpert, ShlExpert, ShrExpert,
    DivExpert, ModExpert, CmpExpert
)


def count_params():
    print("C4 TRANSFORMER PARAMETER COUNT")
    print("=" * 60)
    print()

    # Configuration
    memory_size = 256
    num_addr_bits = 9
    num_type_bits = 5
    num_opcodes = 39
    max_shift = 32
    max_quotient = 64
    num_bits_bitwise = 16
    scale = 10.0

    print("Configuration:")
    print(f"  Memory size: {memory_size}")
    print(f"  Address bits: {num_addr_bits}")
    print(f"  Type bits: {num_type_bits}")
    print(f"  Opcodes: {num_opcodes}")
    print()

    weights = {}

    # 1. ATTENTION - Binary encoded keys
    print("1. ATTENTION (Memory Access)")
    # Keys: (memory_size + 4 registers) × (addr_bits + type_bits)
    num_positions = memory_size + 4  # memory + PC, SP, BP, AX
    key_dim = num_addr_bits + num_type_bits
    attention_keys = num_positions * key_dim
    # But values are only ±scale, so unique values = 2
    print(f"   Key matrix: {num_positions} × {key_dim} = {attention_keys} entries")
    print(f"   Unique values: ±{scale} = 2 distinct")
    weights['attention_keys'] = attention_keys

    # Query construction: same encoding
    # No separate Q matrix - computed from address

    # 2. DECODE - Hardwired division
    print("\n2. DECODE (Instruction → Opcode, Imm)")
    print(f"   Constants: 1/256, 256 = 2 values")
    weights['decode'] = 2

    # 3. MOE ROUTER - Opcode matching
    print("\n3. MOE ROUTER")
    # eq_gate for each opcode: needs center value (0, 1, 2, ..., 38)
    # Plus scale factor
    print(f"   Opcode centers: {num_opcodes} values (0 to {num_opcodes-1})")
    print(f"   Scale factor: 1 value")
    weights['router'] = num_opcodes + 1

    # 4. EXPERTS
    print("\n4. EXPERTS")

    # Simple experts (no weights beyond identity)
    simple_experts = ['IMM', 'LEA', 'ADD', 'SUB']
    print(f"   Simple (identity): {simple_experts} = 0 weights")

    # MUL - uses swiglu_mul (just scale=1 implicit)
    print(f"   MUL: swiglu_mul = 0 explicit weights (identity)")

    # DIV - enumeration over quotients
    # For each q in 0..max_quotient: threshold needs scale
    # Hidden dim = max_quotient * 4 (2 thresholds × 2 SiLU each)
    div_hidden = max_quotient * 4
    print(f"   DIV: {max_quotient} quotients × 4 SiLU = {div_hidden} hidden units")
    print(f"         Scale factor: 1, quotient values: {max_quotient}")
    weights['div'] = max_quotient + 1

    # MOD - reuses DIV
    print(f"   MOD: reuses DIV + swiglu_mul")

    # SHL/SHR - powers of 2 + pulse gating
    shift_powers = max_shift  # 2^0, 2^1, ..., 2^31
    shift_hidden = max_shift * 4  # 4 SiLU per pulse gate
    print(f"   SHL: {max_shift} powers of 2, {shift_hidden} hidden units")
    print(f"   SHR: {max_shift} powers of 2, {shift_hidden} hidden units")
    weights['shift'] = 2 * (shift_powers + 1)  # powers + scale, ×2 for SHL/SHR

    # Bitwise - bit extraction + recombine
    bitwise_powers = num_bits_bitwise  # 2^0 to 2^15
    bitwise_hidden = num_bits_bitwise * 4  # extraction
    print(f"   AND/OR/XOR: {num_bits_bitwise} powers of 2 each × 3 = {3 * bitwise_powers}")
    weights['bitwise'] = 3 * bitwise_powers

    # CMP - comparison gates
    print(f"   CMP (6 types): eq_gate/gt_gate/ge_gate, scale factors only")
    weights['cmp'] = 6  # 6 comparison types, each just needs scale

    # 5. REGISTER UPDATE GATES
    print("\n5. REGISTER UPDATES")
    # Each register update checks ~5-10 opcodes via eq_gate
    # Opcode constants: PSH, ADJ, ENT, LEV, JMP, JSR, BZ, BNZ, etc.
    update_opcodes = 15  # approximate
    print(f"   Opcode gates: ~{update_opcodes} distinct opcodes checked")
    print(f"   Constants: 8 (stack word size)")
    weights['register_update'] = update_opcodes + 1

    # 6. PRINTF - digit extraction
    print("\n6. PRINTF (Autoregressive)")
    powers_of_10 = 6  # 1, 10, 100, 1000, 10000, 100000
    ascii_constants = 3  # 48 ('0'), 10 ('\n'), 45 ('-')
    print(f"   Powers of 10: {powers_of_10}")
    print(f"   ASCII constants: {ascii_constants}")
    weights['printf'] = powers_of_10 + ascii_constants

    # SUMMARY
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Total "weight entries" (matrix positions)
    total_entries = attention_keys + div_hidden + 2 * shift_hidden + 3 * bitwise_hidden
    print(f"\nTotal weight matrix entries: ~{total_entries:,}")

    # Unique constants
    unique_constants = {
        'Scale factors': 2,  # 10.0, 20.0
        'Powers of 2': max_shift,  # 32 distinct
        'Powers of 10': powers_of_10,
        'Opcode values': num_opcodes,
        'ASCII codes': ascii_constants,
        'Offsets': 3,  # 8, 256, 1/256
    }

    total_unique = sum(unique_constants.values())
    print(f"\nUnique non-zero constants:")
    for name, count in unique_constants.items():
        print(f"  {name}: {count}")
    print(f"  TOTAL: {total_unique}")

    print(f"\nKey insight: All {total_entries:,} weight entries are built from")
    print(f"just {total_unique} unique constants (powers of 2, opcodes, etc.)")
    print(f"\nNo learned parameters - everything is hardwired!")

    # Compare to standard transformer
    print("\n" + "=" * 60)
    print("COMPARISON TO STANDARD TRANSFORMER")
    print("=" * 60)
    d_model = 256
    d_ff = 1024
    n_layers = 6
    vocab = 50000

    standard_params = (
        n_layers * (4 * d_model * d_model +  # Q, K, V, O
                    2 * d_model * d_ff) +      # FFN
        vocab * d_model                         # embeddings
    )
    print(f"\nStandard transformer ({n_layers}L, d={d_model}, ff={d_ff}):")
    print(f"  ~{standard_params:,} learned parameters")
    print(f"\nC4 transformer:")
    print(f"  0 learned parameters")
    print(f"  {total_unique} hardwired constants")


if __name__ == "__main__":
    count_params()
