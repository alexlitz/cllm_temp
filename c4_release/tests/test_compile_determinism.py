"""Determinism tests for ``compile_full_vm``.

After the compiler migration every weight in the model is supposed to be baked
by an ``Operation`` instance registered in ``all_core_ops()`` (plus the
L11/L12 mul + L10 divmod composite ops registered explicitly in
``compile_full_vm``). The goal: ``compile_full_vm`` should be a pure function
of its arguments — calling it twice should yield identical ``state_dict()``s.

If this fails it means some bake_fn has hidden non-determinism (random init
not seeded, ``torch.empty`` not zeroed, dict-iteration-order dependence in the
op list, etc.).
"""

import pytest
import torch

from c4_release.neural_vm.unified_compiler.full_vm_compiler import compile_full_vm


def _compare_state_dicts(sd1, sd2):
    """Return (missing_keys, diff_keys) for two state dicts."""
    missing = set(sd1.keys()) ^ set(sd2.keys())
    diffs = []
    for k in sd1.keys() & sd2.keys():
        t1 = sd1[k]
        t2 = sd2[k]
        if t1.shape != t2.shape:
            diffs.append((k, "shape", tuple(t1.shape), tuple(t2.shape)))
            continue
        if t1.dtype != t2.dtype:
            diffs.append((k, "dtype", t1.dtype, t2.dtype))
            continue
        if not torch.equal(t1, t2):
            diffs.append((k, "values"))
    return missing, diffs


def test_compile_full_vm_is_deterministic():
    """Default-arg compile should be bit-identical across two calls."""
    m1, _ = compile_full_vm()
    m2, _ = compile_full_vm()
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    assert sd1.keys() == sd2.keys(), f"keys differ: {set(sd1) ^ set(sd2)}"
    diffs = []
    for k in sd1:
        if not torch.equal(sd1[k], sd2[k]):
            diffs.append(k)
    assert not diffs, f"{len(diffs)} tensors differ between two builds: {diffs[:10]}"


def test_compile_full_vm_is_deterministic_efficient():
    """``alu_mode='efficient'`` compile should also be bit-identical."""
    m1, _ = compile_full_vm(alu_mode="efficient")
    m2, _ = compile_full_vm(alu_mode="efficient")
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    assert sd1.keys() == sd2.keys(), f"keys differ: {set(sd1) ^ set(sd2)}"
    diffs = [k for k in sd1 if not torch.equal(sd1[k], sd2[k])]
    assert not diffs, f"{len(diffs)} tensors differ between two builds: {diffs[:10]}"


def test_compile_full_vm_is_deterministic_conversational_io():
    """Conversational-IO compile should be bit-identical."""
    m1, _ = compile_full_vm(enable_conversational_io=True)
    m2, _ = compile_full_vm(enable_conversational_io=True)
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    assert sd1.keys() == sd2.keys(), f"keys differ: {set(sd1) ^ set(sd2)}"
    diffs = [k for k in sd1 if not torch.equal(sd1[k], sd2[k])]
    assert not diffs, f"{len(diffs)} tensors differ between two builds: {diffs[:10]}"


def test_compile_full_vm_is_deterministic_tool_calling():
    """Tool-calling compile should be bit-identical."""
    m1, _ = compile_full_vm(enable_tool_calling=True)
    m2, _ = compile_full_vm(enable_tool_calling=True)
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()
    assert sd1.keys() == sd2.keys(), f"keys differ: {set(sd1) ^ set(sd2)}"
    diffs = [k for k in sd1 if not torch.equal(sd1[k], sd2[k])]
    assert not diffs, f"{len(diffs)} tensors differ between two builds: {diffs[:10]}"
