#!/usr/bin/env python3
"""Debug what draft_tokens returns for LEA 8."""
import sys
sys.path.insert(0, '.')

from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token

BYTECODE = [Opcode.LEA | (8 << 8)]

draft = DraftVM(BYTECODE)
print(f"Initial state: PC={draft.pc}, AX={draft.ax}, BP={draft.bp}")

draft.step()
print(f"After step: PC={draft.pc}, AX={draft.ax}, BP={draft.bp}")

tokens = draft.draft_tokens()
print(f"\nDraft tokens (first 15): {tokens[:15]}")
print(f"  Token 0: {tokens[0]} (PC marker = {Token.MARK_PC})")
print(f"  Tokens 1-5 (PC bytes): {tokens[1:6]} - PC = {draft.pc}")
print(f"  Token 6: {tokens[6]} (AX marker = {Token.MARK_AX})")
print(f"  Tokens 7-11 (AX bytes): {tokens[7:12]} - AX = {draft.ax}")

print(f"\nExpected PC byte 0 at position 1: {tokens[1]} (lo nibble: {tokens[1] & 0xF})")
print(f"Expected AX byte 0 at position 7: {tokens[7]} (lo nibble: {tokens[7] & 0xF})")
