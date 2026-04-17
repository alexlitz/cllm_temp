#!/usr/bin/env python3
"""Fix PC formula in speculative.py to use idx*8."""

with open('neural_vm/speculative.py', 'r') as f:
    content = f.read()

# Replace all PC formula occurrences
content = content.replace('self.pc = 2           # PC = idx * 5 + 2',
                         'self.pc = 0           # PC = idx * 8')
content = content.replace('self.pc = self.idx * 5 + 2  # default: advance to next instruction',
                         'self.pc = self.idx * 8  # default: advance to next instruction (8 bytes per instruction)')
content = content.replace('(imm - 2) // 5', 'imm // 8')
content = content.replace('(ret_addr - 2) // 5', 'ret_addr // 8')

with open('neural_vm/speculative.py', 'w') as f:
    f.write(content)

print("Fixed PC formula in speculative.py: idx*8 (addresses start at 0)")
