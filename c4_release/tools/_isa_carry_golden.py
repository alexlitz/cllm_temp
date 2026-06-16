"""Capture whole-model state_dict SHA256 for ISA-carry migration A/B gating.

CPU only (CUDA_VISIBLE_DEVICES=""), disk_cache=False. Clears the in-proc
caches per flag combo so each hash reflects the env at compile time. Prints
one ``FLAG=val ... -> <sha256>`` line per combo on the argv list.

Usage:
    CUDA_VISIBLE_DEVICES="" python -m c4_release.tools._isa_carry_golden \
        AX_ON STACK0_ON
The combo tokens map to env presets (see ``_COMBOS``). With no argv it runs
every combo.

CRITICAL: the disk cache key does NOT hash env flags, so the caller MUST clear
~/.cache/c4_release/compiled_vm/ before an A/B comparison if disk_cache were on
— here disk_cache=False so only the in-proc memo matters, which we reset.
"""

from __future__ import annotations

import hashlib
import os
import sys


def _model_state_hash() -> str:
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    import torch

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


# Each combo sets the two migration flags; all others stay at their defaults.
_COMBOS = {
    "AX_ON": {"C4_AX_BYTE1_DUMP": "1"},
    "AX_OFF": {"C4_AX_BYTE1_DUMP": "0"},
    "STACK0_ON": {"C4_STACK0_B0_DUMP": "1"},
    "STACK0_OFF": {"C4_STACK0_B0_DUMP": "0"},
    # Full production default (both on) — the canonical build.
    "DEFAULT": {},
}


def _reset_inproc_caches() -> None:
    """Drop any module-level memo so a fresh compile honours the new env."""
    import importlib

    mod = importlib.import_module(
        "c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic"
    )
    for attr in dir(mod):
        if attr.startswith("_") and "CACHE" in attr.upper():
            obj = getattr(mod, attr)
            if isinstance(obj, dict):
                obj.clear()


def main(argv: list[str]) -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    combos = argv if argv else list(_COMBOS)
    for name in combos:
        if name not in _COMBOS:
            print(f"unknown combo {name!r}; known: {sorted(_COMBOS)}",
                  file=sys.stderr)
            return 2
        # Apply preset (clearing the two flags first so a prior combo's value
        # never leaks).
        for k in ("C4_AX_BYTE1_DUMP", "C4_STACK0_B0_DUMP"):
            os.environ.pop(k, None)
        for k, v in _COMBOS[name].items():
            os.environ[k] = v
        _reset_inproc_caches()
        digest = _model_state_hash()
        flag_repr = " ".join(
            f"{k}={os.environ.get(k, '<unset>')}"
            for k in ("C4_AX_BYTE1_DUMP", "C4_STACK0_B0_DUMP")
        )
        print(f"{name}: {flag_repr} -> {digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
