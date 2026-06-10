#!/usr/bin/env python3
"""CLI: run the STEP_END migration parity harness for a named rule.

Companion to ``c4_release/tests/test_step_end_migration_parity.py``. Lets
contributors invoke the parity check from the command line for any rule
registered in ``RULE_FACTORIES`` without spinning up pytest.

Usage:

    python tools/check_rule_position_parity.py --list
    python tools/check_rule_position_parity.py cmp_combine
    python tools/check_rule_position_parity.py cmp_combine \\
        --target-dim OUTPUT_LO+1 --step-index 3

The default ``--test-program`` for each rule mirrors the example
assertion in the test file so a bare invocation reproduces the same
verdict as the corresponding ``pytest -k`` selection.

Exits 0 on parity, 1 on drift, 2 on usage error. Runs in <30s wall (the
heaviest rule, ``alu_bitwise_or``, has 512 conditions; symbolic_ffn is
linear in the rule count).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neural_vm.unified_compiler.ir import FFNRule  # noqa: E402
from tests.test_step_end_migration_parity import (  # noqa: E402
    RULE_FACTORIES,
    assert_step_end_parity,
)


# Default ``(test_program, step_index, target_dim)`` per rule — matches the
# in-test example assertions so the CLI gives the same verdict as
# ``pytest -k <rule>``.
_DEFAULT_CASES: dict[str, dict] = {
    "cmp_combine": {
        "test_program": {
            "MARK_AX": 1.5,
            "OP_EQ": 1.5,
            "CMP+1": 1.0,
            "CMP+2": 1.0,
        },
        "step_index": 3,
        "target_dim": "OUTPUT_LO+0",
    },
    "cmp_default": {
        "test_program": {"MARK_AX": 1.0, "OP_NE": 1.0},
        "step_index": 3,
        "target_dim": "OUTPUT_LO+1",
    },
    "alu_bitwise_or": {
        "test_program": {
            "MARK_AX": 1.0,
            "ALU_LO+10": 1.0,
            "AX_CARRY_LO+5": 1.0,
            "OP_OR": 1.0,
        },
        "step_index": 4,
        "target_dim": "OUTPUT_LO+15",
    },
    "alu_bitwise_xor": {
        "test_program": {
            "MARK_AX": 1.0,
            "ALU_LO+10": 1.0,
            "AX_CARRY_LO+5": 1.0,
            "OP_XOR": 1.0,
        },
        "step_index": 4,
        "target_dim": "OUTPUT_LO+15",
    },
    "alu_bitwise_and": {
        "test_program": {
            "MARK_AX": 1.0,
            "ALU_LO+10": 1.0,
            "AX_CARRY_LO+5": 1.0,
            "OP_AND": 1.0,
        },
        "step_index": 4,
        "target_dim": "OUTPUT_LO+0",
    },
}


def _list_rules() -> int:
    print("Registered rule factories:")
    for name in sorted(RULE_FACTORIES):
        default = _DEFAULT_CASES.get(name, {})
        target = default.get("target_dim", "<no default>")
        step = default.get("step_index", "?")
        rules = RULE_FACTORIES[name]()
        size = sum(1 for _ in rules) if hasattr(rules, "__iter__") else "?"
        if isinstance(rules, (list, tuple)):
            size = len(rules)
        print(
            f"  {name:20s}  rules={size!s:>4s}  default_target={target!s}  step={step!s}"
        )
    return 0


def _check_one(
    rule_name: str,
    test_program: Mapping[str, float],
    step_index: int,
    target_dim: str,
    *,
    rtol: float,
    atol: float,
) -> int:
    factory = RULE_FACTORIES[rule_name]
    rules: Sequence[FFNRule] = factory()
    print(
        f"[parity] rule={rule_name!r}  rules={len(rules)}  target={target_dim!r}"
        f"  step={step_index}"
    )
    t0 = time.perf_counter()
    try:
        assert_step_end_parity(
            rule_factory=factory,
            test_program=test_program,
            step_index=step_index,
            target_dim=target_dim,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError as exc:
        elapsed = time.perf_counter() - t0
        print(f"[parity] DRIFT in {elapsed*1000:.1f} ms")
        print(str(exc))
        return 1
    elapsed = time.perf_counter() - t0
    print(f"[parity] OK    in {elapsed*1000:.1f} ms")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a compute rule produces byte-identical output at "
            "STEP_END as at MARK_AX (Wave B STEP_END migration verifier)."
        ),
    )
    parser.add_argument(
        "rule",
        nargs="?",
        help="rule name (see --list).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered rule names and exit.",
    )
    parser.add_argument(
        "--test-program",
        type=str,
        default=None,
        help=(
            "JSON dict of residual-state at MARK_AX position "
            '(default: rule-specific). Example: \'{"MARK_AX":1.5,"OP_EQ":1.5}\'.'
        ),
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=None,
        help="Step index metadata (default: rule-specific).",
    )
    parser.add_argument(
        "--target-dim",
        type=str,
        default=None,
        help="Residual dim to compare, e.g. OUTPUT_LO+1.",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every rule's default case sequentially.",
    )

    args = parser.parse_args(argv)

    if args.list:
        return _list_rules()

    if args.all:
        worst = 0
        t0 = time.perf_counter()
        for name in sorted(RULE_FACTORIES):
            default = _DEFAULT_CASES.get(name)
            if default is None:
                print(f"[parity] SKIP {name}: no default case registered")
                continue
            verdict = _check_one(
                name,
                default["test_program"],
                default["step_index"],
                default["target_dim"],
                rtol=args.rtol,
                atol=args.atol,
            )
            worst = max(worst, verdict)
        elapsed = time.perf_counter() - t0
        print(f"[parity] total wall: {elapsed:.2f} s")
        return worst

    if args.rule is None:
        parser.print_usage(file=sys.stderr)
        print(
            "error: positional 'rule' is required (or use --list / --all)",
            file=sys.stderr,
        )
        return 2

    if args.rule not in RULE_FACTORIES:
        print(
            f"error: unknown rule {args.rule!r}. "
            f"Known: {sorted(RULE_FACTORIES)}",
            file=sys.stderr,
        )
        return 2

    default = _DEFAULT_CASES.get(args.rule, {})
    if args.test_program is not None:
        test_program = json.loads(args.test_program)
    elif "test_program" in default:
        test_program = default["test_program"]
    else:
        print(
            f"error: no default test_program for {args.rule!r}; "
            "pass --test-program.",
            file=sys.stderr,
        )
        return 2

    step_index = (
        args.step_index
        if args.step_index is not None
        else default.get("step_index", 0)
    )
    target_dim = (
        args.target_dim
        if args.target_dim is not None
        else default.get("target_dim")
    )
    if target_dim is None:
        print(
            f"error: no default target_dim for {args.rule!r}; "
            "pass --target-dim.",
            file=sys.stderr,
        )
        return 2

    return _check_one(
        args.rule,
        test_program,
        step_index,
        target_dim,
        rtol=args.rtol,
        atol=args.atol,
    )


if __name__ == "__main__":
    sys.exit(main())
