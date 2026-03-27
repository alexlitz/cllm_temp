"""Identify what Layer 6 FFN unit 403 does."""

# From _set_layer6_routing_ffn, track unit numbers:

unit = 0

# IMM: FETCH → OUTPUT (LO: 0-15, HI: 16-31)
print(f"Units {unit}-{unit+31}: IMM routing (FETCH → OUTPUT)")
unit += 32

# EXIT: AX_CARRY → OUTPUT (LO: 32-47, HI: 48-63)
print(f"Units {unit}-{unit+31}: EXIT routing (AX_CARRY → OUTPUT)")
unit += 32

# NOP: AX_CARRY → OUTPUT (LO: 64-79, HI: 80-95)
print(f"Units {unit}-{unit+31}: NOP routing (AX_CARRY → OUTPUT)")
unit += 32

# JMP: AX_CARRY → OUTPUT (LO: 96-111, HI: 112-127)
print(f"Units {unit}-{unit+31}: JMP routing (AX_CARRY → OUTPUT)")
unit += 32

# JMP PC override: cancel PC+5, write JMP target (3 sections × 16 units each = 96 units)
print(f"Units {unit}-{unit+31}: JMP PC override - cancel OUTPUT_LO")
unit += 16
print(f"Units {unit}-{unit+15}: JMP PC override - cancel OUTPUT_HI")
unit += 16
print(f"Units {unit}-{unit+15}: JMP PC override - add AX_CARRY_LO")
unit += 16
print(f"Units {unit}-{unit+15}: JMP PC override - add AX_CARRY_HI")
unit += 16
print(f"Units {unit}-{unit+15}: FIRST-STEP JMP - cancel OUTPUT_LO")
unit += 16
print(f"Units {unit}-{unit+15}: FIRST-STEP JMP - cancel OUTPUT_HI")
unit += 16
print(f"Units {unit}-{unit+15}: FIRST-STEP JMP - add AX_CARRY_LO")
unit += 16
print(f"Units {unit}-{unit+15}: FIRST-STEP JMP - add AX_CARRY_HI")
unit += 16

# HALT detection: 1 unit
print(f"Unit {unit}: HALT detection")
unit += 1

# SP/BP/STACK0 identity carry (3 markers × 32 units = 96 units)
print(f"Units {unit}-{unit+95}: SP/BP/STACK0 identity carry (EMBED → OUTPUT)")
print(f"  SP identity:      {unit}-{unit+31}")
print(f"  BP identity:      {unit+32}-{unit+63}")
print(f"  STACK0 identity:  {unit+64}-{unit+95}")

identity_start = unit
identity_stack0_start = unit + 64
identity_stack0_hi_start = unit + 64 + 16

print()
print(f"Unit 403 falls in range: {identity_stack0_hi_start}-{identity_stack0_hi_start+15}")
print(f"This is STACK0 identity carry (EMBED_HI → OUTPUT_HI)")
print(f"Specific nibble: OUTPUT_HI[{403 - identity_stack0_hi_start}]")
