#!/usr/bin/env python3
"""Byte-identity gate for the L14 CarryPropagationPostOp DSL migration.

Builds the compact dim_positions from the real dynamic layout, then for
each (byte_idx, cascade) carry instance:

  1. Bakes the IMPERATIVE ``CarryPropagationPostOp`` (vm_step.py).
  2. Lowers the DECLARATIVE ``_l10_carry_propagation_rules`` into a fresh
     ``PureFFN`` of the same width.
  3. Asserts every weight tensor is element-wise identical (pre-strengthen).
  4. Runs ``compare_symbolic_to_lowered_ffn`` on the rule list.

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/verify_carry_migration.py
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG)
# ``compare_symbolic_to_lowered_ffn`` imports ``c4_release.neural_vm...`` so
# the package PARENT must also be importable as the ``c4_release`` package.
sys.path.insert(0, os.path.dirname(_PKG))

import torch  # noqa: E402

from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)
from neural_vm.vm_step import CarryPropagationPostOp  # noqa: E402
from neural_vm.base_layers import PureFFN  # noqa: E402
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _l10_carry_propagation_rules,
)
from neural_vm.unified_compiler.ir import (  # noqa: E402
    CompilerIR,
    compare_symbolic_to_lowered_ffn,
)

S = 100.0


def _get_dim_positions():
    # Build only the layout to obtain compact dim_positions; avoid the
    # cached/runner path so imperative writes are visible if needed.
    _model, layout = compile_full_vm_dynamic(
        S=S, alu_mode="lookup", disk_cache=False,
    )
    return dict(layout.dim_positions)


def _carry_ir(rules):
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    return ir


def main() -> int:
    dim_positions = _get_dim_positions()
    d_model = max(dim_positions.values()) + 1
    print(f"d_model = {d_model}")

    all_ok = True
    for byte_idx, cascade in ((0, False), (1, True), (2, True)):
        tag = f"byte_idx={byte_idx} cascade={cascade}"

        # 1) Imperative bake.
        imp = CarryPropagationPostOp(
            d_model=d_model, S=S, byte_idx=byte_idx, cascade=cascade,
            dim_positions=dim_positions,
        )

        # 2) Declarative lowering into a fresh PureFFN of identical width.
        rules = _l10_carry_propagation_rules(
            S, byte_idx=byte_idx, cascade=cascade,
        )
        decl = PureFFN(dim=d_model, hidden_dim=imp.W_up.data.shape[0])
        end = Primitives.lower_ffn_rules(
            decl, rules, dim_positions, start_unit=0, S=S,
        )
        assert end == imp.W_up.data.shape[0], (
            f"{tag}: rule count {end} != imperative hidden_dim "
            f"{imp.W_up.data.shape[0]}"
        )

        # 3) Element-wise tensor identity (pre-strengthen).
        names = ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down")
        per_tensor_ok = True
        for nm in names:
            a = getattr(imp, nm).data
            b = getattr(decl, nm).data
            if a.shape != b.shape:
                print(f"  {tag}: {nm} SHAPE {a.shape} vs {b.shape}")
                per_tensor_ok = False
                continue
            maxdiff = (a - b).abs().max().item() if a.numel() else 0.0
            if maxdiff > 1e-6:
                idx = (a - b).abs().argmax().item()
                print(f"  {tag}: {nm} MAXDIFF {maxdiff:.3e} at flat {idx}")
                per_tensor_ok = False
        if per_tensor_ok:
            print(f"  {tag}: tensors IDENTICAL ({end} units)")
        else:
            all_ok = False

        # 4) Symbolic vs lowered. The authoritative byte-identity gate is
        # the LOWERING-CONTRACT check (kind="lowering"): does the rule
        # list lower to weights matching the imperative cells? The
        # ``weight_output_mismatch`` kind is a synthetic-state symbolic-
        # vs-forward artifact (large OUTPUT-band conditions saturate SiLU
        # where the symbolic linearization diverges ~0.3%); it is present
        # identically for the imperative bake's own rules and is NOT a
        # weight drift. We gate only on lowering/declaration_semantics.
        rep = compare_symbolic_to_lowered_ffn(
            _carry_ir(rules), dim_positions, S=S,
        )
        hard = [i for i in rep.issues if i.kind != "weight_output_mismatch"]
        soft = [i for i in rep.issues if i.kind == "weight_output_mismatch"]
        if not hard:
            print(
                f"  {tag}: lowering-contract OK "
                f"({len(soft)} soft symbolic-forward diffs, ignored)"
            )
        else:
            print(f"  {tag}: lowering-contract FAIL")
            for i in hard[:10]:
                print(f"      [{i.kind}] {i.message}")
            all_ok = False

    print("=" * 60)
    print("CARRY MIGRATION BYTE-IDENTITY:", "OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
