"""Whole-model param-hash harness for the ISA-semantics DSL byte-identity gate.

Builds ``compile_full_vm_dynamic(disk_cache=False)`` on CPU (CUDA hidden) and
SHA256-hashes the full ``state_dict``. Used to prove the ISA-DSL generators
reproduce the hand-built ops byte-identically (flag-on AND flag-off).

Usage::

    CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py
    CUDA_VISIBLE_DEVICES="" C4_BP_SAVE_DUMP=0 python tools/_isa_golden_hash.py
"""

import hashlib
import os
import sys

# Worktree root (parent of the ``c4_release`` package dir) on sys.path so the
# ``c4_release.*`` absolute imports resolve regardless of cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


def model_state_hash() -> str:
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    model, _layout = compile_full_vm_dynamic(disk_cache=False)
    h = hashlib.sha256()
    sd = model.state_dict()
    for key in sorted(sd.keys()):
        t = sd[key]
        if not isinstance(t, torch.Tensor):
            continue
        h.update(key.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.detach().to(torch.float64).cpu().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    flag = os.environ.get("C4_BP_SAVE_DUMP", "1")
    digest = model_state_hash()
    print(f"C4_BP_SAVE_DUMP={flag} state_dict_sha256={digest}")
    sys.stdout.flush()
