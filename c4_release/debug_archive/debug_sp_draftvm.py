"""Check DraftVM SP initialization."""
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

bytecode = [Opcode.IMM | (42 << 8)]
draft_vm = DraftVM(bytecode)

print(f"Initial state:")
print(f"  SP = 0x{draft_vm.sp:08x} ({draft_vm.sp})")
print(f"  Expected: 0x00010000 (65536)")

# SP bytes in little-endian:
sp_bytes = [
    draft_vm.sp & 0xFF,
    (draft_vm.sp >> 8) & 0xFF,
    (draft_vm.sp >> 16) & 0xFF,
    (draft_vm.sp >> 24) & 0xFF,
]
print(f"\n  SP bytes (little-endian): {sp_bytes}")
print(f"  Should be: [0, 0, 1, 0]")

draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"\nDraft tokens for SP:")
print(f"  SP_b0: {draft_tokens[11]} (expected: {sp_bytes[0]})")
print(f"  SP_b1: {draft_tokens[12]} (expected: {sp_bytes[1]})")
print(f"  SP_b2: {draft_tokens[13]} (expected: {sp_bytes[2]})")
print(f"  SP_b3: {draft_tokens[14]} (expected: {sp_bytes[3]})")
