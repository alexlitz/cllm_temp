"""
Weight count analysis for nibble-based vs byte-based operations.

Compares the number of neural network weights required for each approach.
"""

from .embedding import E


def analyze_nibble_weights():
    """Count weights for nibble-based (4-bit) operations."""
    weights = {}

    # Each nibble position has:
    # - NIB_A, NIB_B inputs
    # - RAW_SUM, CARRY_IN, CARRY_OUT, RESULT, TEMP intermediates
    # - Opcode one-hot (72 ops)

    # ADD/SUB: 8 nibbles, 8 carry stages
    # Per-nibble FFN: ~32 weights (W_up, W_gate, W_down, biases)
    # Carry propagation attention: 8 positions * 16 weights
    add_ffn_weights = 8 * 32  # 8 stages, 32 weights each
    add_attn_weights = 8 * 8 * 16  # 8 stages, 8 positions, 16 weights
    weights['ADD (nibble)'] = {
        'ffn': add_ffn_weights,
        'attention': add_attn_weights,
        'total': add_ffn_weights + add_attn_weights,
        'carry_stages': 8,
    }

    # MUL: 36 partial products (i+j < 8)
    # Each product needs: extract nibbles, multiply, accumulate
    mul_products = sum(1 for i in range(8) for j in range(8) if i + j < 8)
    mul_extract_weights = mul_products * 32  # Extract and multiply
    mul_carry_weights = 8 * 8 * 16  # 8 carry stages
    weights['MUL (nibble)'] = {
        'products': mul_products,
        'extract_weights': mul_extract_weights,
        'carry_weights': mul_carry_weights,
        'total': mul_extract_weights + mul_carry_weights,
    }

    # Comparison: 8 nibble comparisons + reduction
    cmp_ffn_weights = 8 * 32  # 8 nibble comparisons
    cmp_attn_weights = 8 * 16  # Reduction attention
    weights['CMP (nibble)'] = {
        'ffn': cmp_ffn_weights,
        'attention': cmp_attn_weights,
        'total': cmp_ffn_weights + cmp_attn_weights,
        'stages': 8,
    }

    return weights


def analyze_byte_weights():
    """Count weights for byte-based (8-bit) operations."""
    weights = {}

    # ADD/SUB: 4 bytes, 4 carry stages
    # Need extra attention to gather nibbles into bytes
    add_gather_attn = 4 * 4 * 16  # 4 bytes, 4 nibbles each, 16 weights
    add_ffn_weights = 4 * 32  # 4 stages, 32 weights each
    add_carry_attn = 4 * 4 * 16  # 4 carry stages
    weights['ADD (byte)'] = {
        'gather_attention': add_gather_attn,
        'ffn': add_ffn_weights,
        'carry_attention': add_carry_attn,
        'total': add_gather_attn + add_ffn_weights + add_carry_attn,
        'carry_stages': 4,
    }

    # MUL: 10 partial products (i+j < 4)
    mul_products = sum(1 for i in range(4) for j in range(4) if i + j < 4)
    mul_gather_attn = 4 * 4 * 16  # Gather 4 bytes
    mul_product_weights = mul_products * 32  # Product FFNs
    mul_carry_weights = 4 * 4 * 16  # 4 carry stages
    weights['MUL (byte)'] = {
        'products': mul_products,
        'gather_attention': mul_gather_attn,
        'product_weights': mul_product_weights,
        'carry_weights': mul_carry_weights,
        'total': mul_gather_attn + mul_product_weights + mul_carry_weights,
    }

    # Comparison: 4 byte comparisons + reduction
    cmp_gather_attn = 4 * 4 * 16  # Gather 4 bytes
    cmp_ffn_weights = 4 * 32  # 4 byte comparisons
    cmp_attn_weights = 4 * 16  # Reduction attention
    weights['CMP (byte)'] = {
        'gather_attention': cmp_gather_attn,
        'ffn': cmp_ffn_weights,
        'attention': cmp_attn_weights,
        'total': cmp_gather_attn + cmp_ffn_weights + cmp_attn_weights,
        'stages': 4,
    }

    return weights


def analyze_word_weights():
    """Count weights for word-based (16-bit) operations."""
    weights = {}

    # Comparison: 2 word comparisons + reduction
    cmp_gather_attn = 2 * 8 * 16  # Gather 2 words (8 nibbles each)
    cmp_ffn_weights = 2 * 32  # 2 word comparisons
    cmp_attn_weights = 2 * 16  # Reduction attention
    weights['CMP (word)'] = {
        'gather_attention': cmp_gather_attn,
        'ffn': cmp_ffn_weights,
        'attention': cmp_attn_weights,
        'total': cmp_gather_attn + cmp_ffn_weights + cmp_attn_weights,
        'stages': 2,
    }

    return weights


def compare_all():
    """Compare all operation modes."""
    print("=" * 70)
    print("WEIGHT COUNT ANALYSIS: NIBBLE vs BYTE vs WORD")
    print("=" * 70)

    nibble = analyze_nibble_weights()
    byte_ops = analyze_byte_weights()
    word = analyze_word_weights()

    # ADD/SUB comparison
    print("\n1. ADD/SUB Operations:")
    print(f"   Nibble-based:")
    print(f"      FFN weights:      {nibble['ADD (nibble)']['ffn']:,}")
    print(f"      Attention weights: {nibble['ADD (nibble)']['attention']:,}")
    print(f"      Total:            {nibble['ADD (nibble)']['total']:,}")
    print(f"      Carry stages:     {nibble['ADD (nibble)']['carry_stages']}")

    print(f"   Byte-based:")
    print(f"      Gather attention:  {byte_ops['ADD (byte)']['gather_attention']:,}")
    print(f"      FFN weights:       {byte_ops['ADD (byte)']['ffn']:,}")
    print(f"      Carry attention:   {byte_ops['ADD (byte)']['carry_attention']:,}")
    print(f"      Total:             {byte_ops['ADD (byte)']['total']:,}")
    print(f"      Carry stages:      {byte_ops['ADD (byte)']['carry_stages']}")

    savings = nibble['ADD (nibble)']['total'] - byte_ops['ADD (byte)']['total']
    if savings > 0:
        print(f"   Savings: {savings:,} weights ({100*savings/nibble['ADD (nibble)']['total']:.1f}%)")
    else:
        print(f"   Overhead: {-savings:,} weights (for gather attention)")

    # MUL comparison
    print("\n2. MUL Operations:")
    print(f"   Nibble-based:")
    print(f"      Partial products:  {nibble['MUL (nibble)']['products']}")
    print(f"      Extract weights:   {nibble['MUL (nibble)']['extract_weights']:,}")
    print(f"      Carry weights:     {nibble['MUL (nibble)']['carry_weights']:,}")
    print(f"      Total:             {nibble['MUL (nibble)']['total']:,}")

    print(f"   Byte-based:")
    print(f"      Partial products:  {byte_ops['MUL (byte)']['products']}")
    print(f"      Gather attention:  {byte_ops['MUL (byte)']['gather_attention']:,}")
    print(f"      Product weights:   {byte_ops['MUL (byte)']['product_weights']:,}")
    print(f"      Carry weights:     {byte_ops['MUL (byte)']['carry_weights']:,}")
    print(f"      Total:             {byte_ops['MUL (byte)']['total']:,}")

    savings = nibble['MUL (nibble)']['total'] - byte_ops['MUL (byte)']['total']
    print(f"   Savings: {savings:,} weights ({100*savings/nibble['MUL (nibble)']['total']:.1f}%)")

    # Comparison operations
    print("\n3. Comparison Operations (EQ/NE/LT/GT/LE/GE):")
    print(f"   Nibble-based:")
    print(f"      FFN weights:       {nibble['CMP (nibble)']['ffn']:,}")
    print(f"      Attention weights: {nibble['CMP (nibble)']['attention']:,}")
    print(f"      Total:             {nibble['CMP (nibble)']['total']:,}")
    print(f"      Stages:            {nibble['CMP (nibble)']['stages']}")

    print(f"   Byte-based:")
    print(f"      Gather attention:  {byte_ops['CMP (byte)']['gather_attention']:,}")
    print(f"      FFN weights:       {byte_ops['CMP (byte)']['ffn']:,}")
    print(f"      Attention weights: {byte_ops['CMP (byte)']['attention']:,}")
    print(f"      Total:             {byte_ops['CMP (byte)']['total']:,}")
    print(f"      Stages:            {byte_ops['CMP (byte)']['stages']}")

    print(f"   Word-based (16-bit):")
    print(f"      Gather attention:  {word['CMP (word)']['gather_attention']:,}")
    print(f"      FFN weights:       {word['CMP (word)']['ffn']:,}")
    print(f"      Attention weights: {word['CMP (word)']['attention']:,}")
    print(f"      Total:             {word['CMP (word)']['total']:,}")
    print(f"      Stages:            {word['CMP (word)']['stages']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_nibble = (nibble['ADD (nibble)']['total'] +
                   nibble['MUL (nibble)']['total'] +
                   nibble['CMP (nibble)']['total'])
    total_byte = (byte_ops['ADD (byte)']['total'] +
                 byte_ops['MUL (byte)']['total'] +
                 byte_ops['CMP (byte)']['total'])
    total_word_cmp = (byte_ops['ADD (byte)']['total'] +
                     byte_ops['MUL (byte)']['total'] +
                     word['CMP (word)']['total'])

    print(f"\nTotal weights (ADD+MUL+CMP):")
    print(f"   Nibble-based:         {total_nibble:,}")
    print(f"   Byte-based:           {total_byte:,}")
    print(f"   Byte+Word (hybrid):   {total_word_cmp:,}")

    print(f"\nReductions:")
    print(f"   Byte vs Nibble:       {100*(total_nibble-total_byte)/total_nibble:.1f}% fewer weights")
    print(f"   Hybrid vs Nibble:     {100*(total_nibble-total_word_cmp)/total_nibble:.1f}% fewer weights")

    return {
        'nibble': nibble,
        'byte': byte_ops,
        'word': word,
        'total_nibble': total_nibble,
        'total_byte': total_byte,
        'total_hybrid': total_word_cmp,
    }


if __name__ == "__main__":
    compare_all()
