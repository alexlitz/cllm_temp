"""Runtime vanilla audit gate.

This test enforces the directive: runtime = just attention + FFN.

It runs `verify_runtime_is_vanilla(model)` on models built via
`compile_full_vm` in each supported mode (default `lookup`, `efficient`, and
`enable_conversational_io=True`) and FAILS if any block contains a module
class that is NOT in either:
  - the canonical vanilla set (`PureFFN` / `FlattenedPureFFN` / `nn.Linear`
    chains / `AutoregressiveAttention` / `PureAttention` / `Identity`), or
  - the explicit `KNOWN_NON_VANILLA_ALLOWLIST` baseline below.

The allowlist captures the modules that are currently still installed as
runtime blocks at the time this gate was added. Each entry is annotated with
a TODO so future contributors can prune the allowlist as the hybrid-removal
work lands. Adding a NEW non-vanilla module (one not in the allowlist) will
break this test — which is the point: new non-vanilla code must opt in
explicitly.

Run:
    pytest tests/test_runtime_vanilla.py -v --tb=long
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm  # noqa: E402
from neural_vm.unified_compiler.runtime_audit import (  # noqa: E402
    verify_runtime_is_vanilla,
)


# =============================================================================
# Allowlist of currently-known non-vanilla module class names.
#
# Each entry is a class name (string) that the audit currently flags as a
# violation but which we accept as a *baseline*. Once the corresponding
# hybrid-removal work lands (each TODO below), the entry should be removed
# from this set. Anything not in the canonical vanilla set AND not in this
# allowlist will fail the test.
# =============================================================================

KNOWN_NON_VANILLA_ALLOWLIST = frozenset({
    # Lookup-mode (default) leftovers.
    "AddSub5StageBlock",          # TODO: remove once hybrid removal completes (L8/L9 ADD/SUB FFN)
    "ALUAndOrXor",                # TODO: remove once hybrid removal completes (L10 bitwise FFN)
    "ALUMul",                     # TODO: remove once hybrid removal completes (L11/L12 MUL FFN)
    "ALUShift",                   # TODO: remove once hybrid removal completes (L13 SHL/SHR FFN)

    # Efficient-mode flattened ALU composites — still non-vanilla wrappers,
    # not vanilla `nn.Sequential` of Linear yet.
    "FlattenedALUMul",            # TODO: remove once hybrid removal completes (L11/L12 efficient MUL)
    "ALUShiftComposite",          # TODO: remove once hybrid removal completes (L13 efficient SHL/SHR)

    # Post-op family: BinaryOpByteZeroingPostOp, CarryPropagationPostOp,
    # BitwiseBytePropagationPostOp, ComparisonCombine were previously listed
    # here. As of 2026-05-11 (a-postops-to-pureffn) they are re-baked into
    # vanilla `PureFFN` instances by `_expand_wrapper_blocks` before being
    # installed as block.ffn, so the audit now sees them as canonical PureFFN.
    # FlattenedDivMod is the only remaining post-op wrapper.
    "FlattenedDivMod",            # TODO: remove once hybrid removal completes (L10 DIV/MOD composite)
})


def _filter_violations(report):
    """Return only violations whose flagged class is NOT in the allowlist.

    Parses the violation strings produced by `verify_runtime_is_vanilla` and
    extracts the offending class name from the `_describe_module` output
    (e.g. `"L11: non-vanilla FFN ALUMul (...)"` -> `"ALUMul"`).
    """
    disallowed = []
    for v in report.violations:
        # Skip post_ops / fp64 / shape violations — these are structural,
        # not a single class. Keep them as disallowed since the allowlist is
        # class-name based.
        if "non-vanilla FFN" not in v and "non-vanilla attention" not in v:
            disallowed.append(v)
            continue
        # Format: "L<n>: non-vanilla FFN <ClassName>(...) (expected ...)" or
        #         "L<n>: non-vanilla FFN <ClassName> (expected ...)"
        try:
            after = v.split("non-vanilla ", 1)[1]
            # after is e.g. "FFN ALUMul (expected ...)" or
            #              "FFN ALUMul(ffn=...) (expected ...)"
            after = after.split(" ", 1)[1]   # drop "FFN " / "attention "
            cls_part = after.split(" ", 1)[0]  # "ALUMul(...)" or "ALUMul"
            cls_name = cls_part.split("(", 1)[0]
        except IndexError:
            disallowed.append(v)
            continue
        if cls_name not in KNOWN_NON_VANILLA_ALLOWLIST:
            disallowed.append(v)
    return disallowed


def _assert_audit_clean_or_allowlisted(model, *, mode_label: str):
    """Run the audit and fail the test if any non-allowlisted violations are found.

    The allowlist captures the currently-known baseline of non-vanilla runtime
    modules. NEW non-vanilla modules (anything not in `KNOWN_NON_VANILLA_ALLOWLIST`
    nor in the canonical vanilla set) will cause this assertion to fail, which
    is the gate's purpose: prevent future code from sneaking new non-FFN /
    non-attention modules into `model.blocks` without explicit opt-in.
    """
    report = verify_runtime_is_vanilla(model)
    disallowed = _filter_violations(report)

    assert not disallowed, (
        f"[{mode_label}] Runtime audit found {len(disallowed)} NEW non-vanilla "
        f"violation(s) NOT in the allowlist. If you added a new module class to "
        f"`model.blocks`, add it to `KNOWN_NON_VANILLA_ALLOWLIST` in "
        f"`tests/test_runtime_vanilla.py` with a TODO comment, or refactor it to "
        f"be a vanilla FFN.\n\n"
        f"Audit summary: {report.summary}\n\n"
        f"Disallowed violations:\n" + "\n".join(f"  - {v}" for v in disallowed)
    )


# =============================================================================
# Test class — also import-side-effect-compatible with test_smoke fixtures.
# =============================================================================

class TestRuntimeVanilla:
    """Runtime vanilla audit gate.

    Each test builds a model via `compile_full_vm` with a different mode and
    asserts that the audit reports no violations OTHER than those captured by
    the allowlist baseline. This means:
      - existing non-vanilla modules (HybridALUBlock split residue, ALU
        flattened composites, post-op family) are tolerated for now;
      - any NEW non-vanilla module class will FAIL the test.
    """

    def test_compile_full_vm_default_mode_audit(self):
        """Default `compile_full_vm()` produces no non-allowlisted violations."""
        model, _layout = compile_full_vm()
        _assert_audit_clean_or_allowlisted(model, mode_label="default (lookup)")

    def test_compile_full_vm_efficient_mode_audit(self):
        """`compile_full_vm(alu_mode='efficient')` produces no non-allowlisted violations."""
        model, _layout = compile_full_vm(alu_mode="efficient")
        _assert_audit_clean_or_allowlisted(model, mode_label="efficient")

    def test_compile_full_vm_conversational_io_audit(self):
        """`compile_full_vm(enable_conversational_io=True)` produces no non-allowlisted violations."""
        model, _layout = compile_full_vm(enable_conversational_io=True)
        _assert_audit_clean_or_allowlisted(model, mode_label="conversational_io")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=long"])
