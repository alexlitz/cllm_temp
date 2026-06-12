"""Byte-identity gate for the L14 mem-default-suppress declarative migration.

Mirrors tools/verify_carry_migration.py (commit 51a6634c): bakes the legacy
imperative ``_set_layer14_mem_addr_src_default_suppress`` /
``_set_layer14_jsr_mem_default_suppress`` helpers into a fresh ``PureFFN`` and
compares the resulting W_up / b_up / W_gate / b_gate / W_down tensors
element-for-element against the declarative IR lowering. A non-zero diff means
the FFNRule list does NOT reproduce the imperative writes exactly -> the
migration is not byte-identical and must be aborted.

Run: python tools/verify_l14_mem_default_suppress_migration.py
"""
import os
import sys

# Resolve the local ``neural_vm`` package (the worktree's c4_release/) ahead of
# any sibling checkout on sys.path. Also add the worktree root so the
# ``import c4_release.neural_vm...`` form used inside ir.py resolves to THIS
# checkout (not the main repo) for ``compare_symbolic_to_lowered_ffn``.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKTREE_ROOT = os.path.dirname(_PKG_ROOT)
sys.path.insert(0, _WORKTREE_ROOT)
sys.path.insert(0, _PKG_ROOT)

import torch

from neural_vm.base_layers import PureFFN
from neural_vm.vm_step import _SetDim
from neural_vm.setup_helpers_l14 import (
    _set_layer14_mem_addr_src_default_suppress,
    _set_layer14_jsr_mem_default_suppress,
)
from neural_vm.unified_compiler.primitives import Primitives
from neural_vm.unified_compiler.ops.shared import _as_setdim_proxy
from neural_vm.unified_compiler.ops.l14_ops import (
    _layer14_mem_addr_src_default_suppress_ir,
    _layer14_jsr_mem_default_suppress_ir,
    _layer14_mem_addr_src_default_suppress_rules,
    _layer14_jsr_mem_default_suppress_rules,
)
from neural_vm.unified_compiler.ir import compare_symbolic_to_lowered_ffn

S = 100.0
DIM = _SetDim.TOTAL_DIMS if hasattr(_SetDim, "TOTAL_DIMS") else 872
# _SetDim exposes dims as class attributes; the residual width is the max
# index + a margin. Use the compiled total if available, else a safe upper.


def _dim_total():
    # Find the largest integer attribute on _SetDim as a conservative width.
    mx = 0
    for k in dir(_SetDim):
        v = getattr(_SetDim, k)
        if isinstance(v, int):
            mx = max(mx, v)
    return mx + 64


def bake_imperative(helper, n_units):
    width = _dim_total()
    ffn = PureFFN(width, n_units)
    with torch.no_grad():
        helper(ffn, S, _SetDim, start_unit=0)
    return ffn


def bake_declarative(rules_fn, n_units):
    width = _dim_total()
    ffn = PureFFN(width, n_units)
    rules = rules_fn(S)
    dim_map = Primitives.dim_positions_from_bd(
        _as_setdim_proxy(_SetDim_dim_positions()),
        Primitives.ffn_rule_dim_names(rules),
    )
    with torch.no_grad():
        Primitives.lower_ffn_rules(ffn, rules, dim_map, start_unit=0, S=S)
    return ffn


def _SetDim_dim_positions():
    # The proxy resolves BD.<NAME> via attribute access on this object; a
    # plain dict of {name: index} mirrors the compact dim_positions map.
    d = {}
    for k in dir(_SetDim):
        if k.startswith("_"):
            continue
        v = getattr(_SetDim, k)
        if isinstance(v, int):
            d[k] = v
    return d


def diff_ffn(a, b, label):
    ok = True
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        ta = getattr(a, name).detach()
        tb = getattr(b, name).detach()
        if ta.shape != tb.shape:
            print(f"  [{label}] {name}: SHAPE MISMATCH {ta.shape} vs {tb.shape}")
            ok = False
            continue
        if not torch.equal(ta, tb):
            md = (ta - tb).abs().max().item()
            nz = int((ta != tb).sum().item())
            print(f"  [{label}] {name}: DIFF max={md} ncells={nz}")
            ok = False
        else:
            print(f"  [{label}] {name}: identical")
    return ok


def main():
    all_ok = True
    cases = [
        ("mem_addr_src_default_suppress",
         _set_layer14_mem_addr_src_default_suppress,
         _layer14_mem_addr_src_default_suppress_rules,
         _layer14_mem_addr_src_default_suppress_ir, 8),
        ("jsr_mem_default_suppress",
         _set_layer14_jsr_mem_default_suppress,
         _layer14_jsr_mem_default_suppress_rules,
         _layer14_jsr_mem_default_suppress_ir, 4),
    ]
    dim_positions = _SetDim_dim_positions()
    for label, helper, rules_fn, ir_fn, n_units in cases:
        print(f"=== {label} (n_units={n_units}) ===")
        imp = bake_imperative(helper, n_units)
        dec = bake_declarative(rules_fn, n_units)
        ok = diff_ffn(imp, dec, label)
        # Also run the IR's own symbolic-vs-lowered contract check.
        rep = compare_symbolic_to_lowered_ffn(ir_fn(S), dim_positions, S=S)
        print(f"  compare_symbolic_to_lowered_ffn.ok = {rep.ok}")
        if not rep.ok:
            print(rep.format())
        all_ok = all_ok and ok and rep.ok
    print()
    print("RESULT:", "BYTE-IDENTICAL OK" if all_ok else "MISMATCH -- ABORT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
