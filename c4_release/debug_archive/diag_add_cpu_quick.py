"""Quick CPU-only smoke check: run handler-mode ADD and report the result.

Forces ``torch.cuda.is_available`` to False so the model is built on CPU
(avoids GPU contention with other agents). Useful as a fast verification
that handler-mode ADD does or does not work.
"""
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
# Force CPU
torch.cuda.is_available = lambda: False

from c4_release.neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from c4_release.neural_vm.embedding import Opcode  # noqa: E402

print("Building handler runner on CPU...")
hm = AutoregressiveVMRunner(pure_neural=False, trust_neural_alu=True)
hm._func_call_handlers = {}
hm._syscall_handlers = {}
bc = [(10 << 8) | Opcode.IMM, Opcode.PSH, (32 << 8) | Opcode.IMM, Opcode.ADD, Opcode.EXIT]
print("Running handler 10+32...")
out, code = hm.run(bc, b"", max_steps=30)
print("handler-mode ADD result:", code, "(expect 42)")

print("\nBuilding pure_neural runner on CPU (cached model)...")
pn = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
pn._func_call_handlers = {}
pn._syscall_handlers = {}
print("Running pure_neural 10+32...")
out, code = pn.run(bc, b"", max_steps=30)
print("pure_neural ADD result:", code, "(expect 42)")
