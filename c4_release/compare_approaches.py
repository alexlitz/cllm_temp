"""Compare lookup vs efficient approaches for ALU ops."""

print("=" * 60)
print("CURRENT LOOKUP APPROACH (per operation)")
print("=" * 60)

# Current L8+L9 for ADD
add_lo_nibble = 256      # 16x16 combinations
add_carry = 120          # pairs where a+b >= 16
add_hi_nibble = 256      # another 16x16
add_hi_carry = 120       # pairs with carry propagation
print(f"ADD: {add_lo_nibble + add_carry + add_hi_nibble + add_hi_carry} units")

# Current MUL (L11+L12)
mul_units = 256 * 256    # full 8-bit lookup
print(f"MUL: {mul_units:,} units (full 8-bit lookup)")

# Current SHIFT
shift_units = 256 * 8 * 2  # value × shift_amt × (SHL+SHR)
print(f"SHIFT: {shift_units:,} units")

# Current BITWISE
bitwise_units = 256 * 256 * 3  # AND + OR + XOR
print(f"BITWISE: {bitwise_units:,} units")

# DIV/MOD (DivModModule lookup mode)
divmod_units = 256 * 256 * 2  # DIV + MOD
print(f"DIV/MOD: {divmod_units:,} units")

total_lookup = (add_lo_nibble + add_carry + add_hi_nibble + add_hi_carry +
                add_lo_nibble + add_carry + add_hi_nibble + add_hi_carry +  # SUB similar
                mul_units + shift_units + bitwise_units + divmod_units)
print(f"\nTOTAL LOOKUP: {total_lookup:,} units")

print("\n" + "=" * 60)
print("EFFICIENT APPROACH (with one-hot decoding overhead)")
print("=" * 60)

# One-hot to scalar decoding: 16 units per nibble
decode_overhead = 16 * 4  # 4 nibbles (ALU_LO, ALU_HI, AX_LO, AX_HI)
print(f"Decoding overhead: {decode_overhead} units (shared)")

# ADD with efficient carry lookahead
add_raw_sum = 6 * 2      # 6 units per position × 2 positions (lo/hi)
add_carry_lookahead = 8  # cross-position carry
add_final = 8 * 2        # final mod extraction
add_efficient = add_raw_sum + add_carry_lookahead + add_final
print(f"ADD: {add_efficient} units (+ decode overhead)")

# MUL with schoolbook
mul_partial = 16         # 4 nibble products (16×16 partial sums: 4 units each)
mul_carry = 20           # carry extraction
mul_efficient = mul_partial + mul_carry
print(f"MUL: ~{mul_efficient} units (+ decode overhead)")

# SHIFT - more complex due to bit routing
shift_efficient = 56     # precompute + select
print(f"SHIFT: ~{shift_efficient} units (+ decode overhead)")

# BITWISE - can use bit extraction
bitwise_efficient = 8 * 3  # 8 bits × 3 ops
print(f"BITWISE: {bitwise_efficient} units (+ decode overhead)")

# DIV/MOD - uses MagicFloor
divmod_efficient = 64    # already implemented
print(f"DIV/MOD: {divmod_efficient} units (+ decode overhead)")

total_efficient = (decode_overhead + add_efficient * 2 +  # ADD + SUB
                   mul_efficient + shift_efficient + bitwise_efficient + divmod_efficient)
print(f"\nTOTAL EFFICIENT: ~{total_efficient} units")

print("\n" + "=" * 60)
print(f"SAVINGS: ~{total_lookup - total_efficient:,} units ({100*(total_lookup-total_efficient)/total_lookup:.1f}%)")
print("=" * 60)
