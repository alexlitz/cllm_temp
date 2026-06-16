"""Whole-model state_dict SHA256 (CPU, disk_cache=False) for the #221 ISA-DSL
byte-identity A/B. Run with CUDA_VISIBLE_DEVICES="" and a FIXED PYTHONHASHSEED
(the build has a pre-existing seed-dependent FFN-packing order). Clear
~/.cache/c4_release/compiled_vm/ before each invocation: the cache key does NOT
hash env flags.
"""
import hashlib
import sys

import torch

from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    compile_full_vm_dynamic,
)


def model_state_hash() -> str:
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
    print(model_state_hash())
    sys.stdout.flush()
