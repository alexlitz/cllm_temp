#!/usr/bin/env python3
"""Runner-override ratchet (vanilla-thesis restoration guard).

Per ``docs/VANILLA_RESTORE_INVENTORY_2026_06_09.md``, the runners must
become pure forward-pass wrappers:

    model.forward(input_ids) -> argmax -> append -> loop until EXIT.

Nothing else. No domain logic. No ALU recovery. No shadow memory. No
``_compute_alu_legacy``. No opcode-specific Python branches that
compute values. After waves A-E delete the Python overrides, this lint
guards against re-introduction.

Scope
-----
Walks the runner files only:

  * ``c4_release/neural_vm/run_vm.py``
  * ``c4_release/neural_vm/batched_pure_neural.py``

plus the "already clean" runners listed in the inventory
(``fast_runner.py``, ``batch_runner.py``, ``batch_runner_v2.py``,
``transformer_first_runner.py``) which should remain at 0 forever.

Forbidden patterns
------------------
Three AST-level pattern classes:

1. **Forbidden calls** (method or bare-name): every ALU-recovery,
   IO-shim, shadow-memory, and override-injection helper from the
   inventory. Includes ``_compute_alu_legacy``,
   ``_inject_getchar``, ``_handle_skipped_io_op``,
   ``_neural_prtf_emit``, ``_neural_open_emit``,
   ``_neural_clos_emit``, ``_neural_read_emit``,
   ``set_mem_store_positions``, ``_override_register_in_last_step``,
   ``_override_ax_in_last_step``, ``_inject_synthetic_step``,
   ``_track_memory_write``, ``_mem_store_word``, ``_mem_load_word``,
   ``_read_string``, ``_format_printf``, ``_syscall_*``,
   ``_handle_pure_neural_stall``, ``_borrow_serial_state``,
   ``_decode_bail_exit_code``, ``_track_mem_access``,
   ``_inject_mem_section``, etc. (Full list: ``_FORBIDDEN_CALLS``.)

2. **Forbidden names** (any reference, including reads of
   ``self.<name>``): the per-op classification constants and the
   shadow-state fields. ``_BINARY_POP_OPS``,
   ``_NON_COLLAPSED_RECOVERY_OPS``, ``_NEURAL_32BIT_OPS``,
   ``_RUNNER_ALU_OPS``, ``last_pushed_value``, ``stack0_shadow``,
   ``mem_history``.

3. **Per-opcode branches**: every ``if exec_op == Opcode.X`` /
   ``elif exec_op == Opcode.X`` comparison. The pure forward path
   never branches on the executed opcode — that is by definition
   override territory.

Ratchet contract
----------------
Each runner file has a baselined ``(calls, names, branches)`` triple
captured from the inventory at 2026-06-09. After each Wave A-E
deletion lands, the agent updates ``_BASELINE`` so the lint enforces
"no new" overrides — counts only walk downward.

Exit codes
----------
0  no regression vs baseline
1  regression: a runner grew or a brand-new pattern appeared
2  invocation / IO error

Usage::

    python c4_release/tools/lint_runner_overrides.py           # full scan
    python c4_release/tools/lint_runner_overrides.py --json    # machine-readable
    python c4_release/tools/lint_runner_overrides.py --list    # every hit
    python c4_release/tools/lint_runner_overrides.py --path PATH
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pattern catalog
# ---------------------------------------------------------------------------


# Method or bare-name calls that compute values, mutate shadow state,
# inject synthetic tokens, or cross the VM/host boundary outside the
# pure forward pass. Source: VANILLA_RESTORE_INVENTORY_2026_06_09.md.
_FORBIDDEN_CALLS: frozenset = frozenset(
    {
        # Legacy ALU (#1, #10, #46, #47).
        "_compute_alu_legacy",
        # IO shims (#17, #20-25, #34-37).
        "_inject_getchar",
        "_handle_skipped_io_op",
        "_handle_pure_neural_stall",
        "_peek_argc_from_adj",
        "_extract_imm_chain_args",
        "_neural_prtf_emit",
        "_neural_open_emit",
        "_neural_clos_emit",
        "_neural_read_emit",
        "_syscall_clos",
        "_syscall_open",
        "_syscall_read",
        "_syscall_prtf",
        "_format_printf",
        "_read_string",
        "_read_stack_arg",
        # Shadow memory (#28-32, #50, #54).
        "_track_memory_write",
        "_extract_mem_write",
        "_mem_store_word",
        "_mem_load_word",
        "_track_mem_access",
        # Override injection (#26, #27, #45).
        "_override_register_in_last_step",
        "_override_ax_in_last_step",
        "_inject_synthetic_step",
        "_inject_mem_section",
        # Embedding-side MEM-store position injection (#53).
        "set_mem_store_positions",
        # Batched plumbing (#44, #51, #52).
        "_borrow_serial_state",
        "_unborrow_serial_state",
        "_decode_bail_exit_code",
    }
)


# Bare identifiers (Name nodes) and attribute references whose mere
# presence in the runner indicates override-time per-op classification
# or shadow-state tracking.
_FORBIDDEN_NAMES: frozenset = frozenset(
    {
        # Per-op classification constants (#40).
        "_BINARY_POP_OPS",
        "_NON_COLLAPSED_RECOVERY_OPS",
        "_NEURAL_32BIT_OPS",
        "_RUNNER_ALU_OPS",
        # Forward-state shadow (#49, #50, #53, #54).
        "last_pushed_value",
        "_last_pushed_value",
        "stack0_shadow",
        "mem_history",
        # Stdin shadow buffer that drives IO shims (#55).
        "_stdin_buffer",
    }
)


# Scan targets. The two "dirty" runners are the legacy ones with the
# inventoried overrides; the four "clean" runners are at baseline 0
# and must stay there.
_DIRTY_RUNNERS: Tuple[str, ...] = (
    "c4_release/neural_vm/run_vm.py",
    "c4_release/neural_vm/batched_pure_neural.py",
)


_CLEAN_RUNNERS: Tuple[str, ...] = (
    "c4_release/neural_vm/fast_runner.py",
    "c4_release/neural_vm/batch_runner.py",
    "c4_release/neural_vm/batch_runner_v2.py",
    "c4_release/neural_vm/transformer_first_runner.py",
)


# Per-file baseline triple ``(calls, names, branches)`` captured
# 2026-06-09 by walking the AST of each runner with the patterns
# above. The lint allows up to these counts; a single increment in any
# axis is a regression. After each Wave A-E removal lands, decrement
# the entry in the SAME commit.
#
# Source-of-truth: this counts AST nodes (call sites, name references,
# and ``exec_op == Opcode.X`` comparisons). It is NOT identical to the
# inventory's "59 entries" — the inventory groups by code block, while
# the AST counts every individual hit so a single block (e.g. the LEV
# override at run_vm.py:2273-2288) contributes multiple overrides as
# it calls ``_override_register_in_last_step`` several times.
_BASELINE: Dict[str, Tuple[int, int, int]] = {
    # Wave D (2026-06-10): all runners cleared to (0, 0, 0). Handler-mode
    # dispatch chain + IO shims + shadow memory + per-op classification
    # constants + PUTCHAR/EXIT per-op branches all retired. The runners
    # are now pure forward-pass wrappers — any non-zero hit on a future
    # PR is a brand-new override that the ratchet must reject.
    "c4_release/neural_vm/run_vm.py": (0, 0, 0),
    "c4_release/neural_vm/batched_pure_neural.py": (0, 0, 0),
    "c4_release/neural_vm/fast_runner.py": (0, 0, 0),
    "c4_release/neural_vm/batch_runner.py": (0, 0, 0),
    "c4_release/neural_vm/batch_runner_v2.py": (0, 0, 0),
    "c4_release/neural_vm/transformer_first_runner.py": (0, 0, 0),
}


# ---------------------------------------------------------------------------
# Hit dataclass-ish tuple
# ---------------------------------------------------------------------------


# (kind, lineno, label)
#   kind   : "call" | "name" | "branch"
#   lineno : 1-based source line
#   label  : the matched identifier (e.g. "_compute_alu_legacy",
#            "_BINARY_POP_OPS", "Opcode.PSH")
Hit = Tuple[str, int, str]


# ---------------------------------------------------------------------------
# AST scanning
# ---------------------------------------------------------------------------


def _is_forbidden_call(node: ast.AST) -> Optional[str]:
    """Return the matched identifier if ``node`` is a forbidden call."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        # ``self._compute_alu_legacy(...)``, ``s._track_memory_write(...)``,
        # ``self.model.embed.set_mem_store_positions(...)`` etc.
        if func.attr in _FORBIDDEN_CALLS:
            return func.attr
    elif isinstance(func, ast.Name):
        if func.id in _FORBIDDEN_CALLS:
            return func.id
    return None


def _is_forbidden_name(node: ast.AST) -> Optional[str]:
    """Return the matched identifier if ``node`` references a forbidden name.

    Matches:
      * Bare ``Name`` nodes (``_BINARY_POP_OPS`` used as a set literal arg).
      * ``Attribute`` accesses (``s.last_pushed_value``,
        ``self._stdin_buffer``).
    """
    if isinstance(node, ast.Name):
        if node.id in _FORBIDDEN_NAMES:
            return node.id
    elif isinstance(node, ast.Attribute):
        if node.attr in _FORBIDDEN_NAMES:
            return node.attr
    return None


def _is_per_opcode_branch(node: ast.AST) -> Optional[str]:
    """Detect ``exec_op == Opcode.X`` / ``skipped_op == Opcode.X`` comparisons.

    Returns the readable label (e.g. ``"Opcode.PSH"``) when the node is
    a per-opcode dispatch comparison. ``If`` branches that compute
    values based on the executed opcode are override territory by
    definition — the pure forward path runs the model and consumes its
    argmax without consulting the bytecode opcode.
    """
    if not isinstance(node, ast.Compare):
        return None
    if not node.ops or not isinstance(node.ops[0], ast.Eq):
        return None
    left = node.left
    # Allow the LHS to be either a bare ``exec_op`` / ``skipped_op``
    # name or an attribute access (``s.exec_op``).
    left_name: Optional[str] = None
    if isinstance(left, ast.Name):
        left_name = left.id
    elif isinstance(left, ast.Attribute):
        left_name = left.attr
    if left_name not in {"exec_op", "skipped_op"}:
        return None
    if not node.comparators:
        return None
    rhs = node.comparators[0]
    if (
        isinstance(rhs, ast.Attribute)
        and isinstance(rhs.value, ast.Name)
        and rhs.value.id == "Opcode"
    ):
        return f"Opcode.{rhs.attr}"
    return None


def _scan_source(source: str) -> List[Hit]:
    """Return every forbidden hit in ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    hits: List[Hit] = []
    for node in ast.walk(tree):
        label = _is_forbidden_call(node)
        if label is not None:
            hits.append(("call", node.lineno, label))
            continue
        label = _is_per_opcode_branch(node)
        if label is not None:
            hits.append(("branch", node.lineno, label))
            continue
        label = _is_forbidden_name(node)
        if label is not None:
            hits.append(("name", node.lineno, label))
    return hits


def _scan_file(path: Path) -> List[Hit]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return _scan_source(source)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _counts(hits: List[Hit]) -> Tuple[int, int, int]:
    """Return ``(n_calls, n_names, n_branches)`` for a hit list."""
    n_calls = sum(1 for k, _, _ in hits if k == "call")
    n_names = sum(1 for k, _, _ in hits if k == "name")
    n_branches = sum(1 for k, _, _ in hits if k == "branch")
    return n_calls, n_names, n_branches


def scan(repo_root: Path) -> Dict[str, List[Hit]]:
    """Scan every tracked runner; return ``{relpath: [Hit, ...]}``."""
    by_file: Dict[str, List[Hit]] = {}
    for rel in (*_DIRTY_RUNNERS, *_CLEAN_RUNNERS):
        path = repo_root / rel
        if not path.exists():
            alt = repo_root / rel.replace("c4_release/", "")
            if alt.exists():
                path = alt
            else:
                continue
        hits = _scan_file(path)
        by_file[rel] = hits
    return by_file


def diff_against_baseline(
    hits_by_file: Dict[str, List[Hit]],
    baseline: Dict[str, Tuple[int, int, int]] = _BASELINE,
) -> Tuple[List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]], List[str]]:
    """Compare current counts against baseline.

    Returns ``(regressions, new_files)`` where:
      * ``regressions`` is ``[(file, baseline_triple, current_triple)]``
        for files where ANY axis grew.
      * ``new_files`` is the list of files we scanned that are not in
        the baseline (shouldn't happen with the static lists but kept
        for defense in depth).
    """
    regressions: List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]] = []
    new_files: List[str] = []
    for rel, hits in sorted(hits_by_file.items()):
        current = _counts(hits)
        if rel not in baseline:
            if any(current):
                new_files.append(rel)
            continue
        b_calls, b_names, b_branches = baseline[rel]
        c_calls, c_names, c_branches = current
        if c_calls > b_calls or c_names > b_names or c_branches > b_branches:
            regressions.append((rel, baseline[rel], current))
    return regressions, new_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _find_repo_root(start: Path) -> Optional[Path]:
    root = start.resolve()
    for _ in range(8):
        if (root / "c4_release").is_dir():
            return root
        if (root / "neural_vm").is_dir() and root.name == "c4_release":
            return root.parent
        parent = root.parent
        if parent == root:
            return None
        root = parent
    return None


def _print_text_report(
    hits_by_file: Dict[str, List[Hit]],
    regressions: List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]],
    new_files: List[str],
    list_all: bool,
    elapsed_ms: float,
) -> None:
    total_calls = sum(_counts(hs)[0] for hs in hits_by_file.values())
    total_names = sum(_counts(hs)[1] for hs in hits_by_file.values())
    total_branches = sum(_counts(hs)[2] for hs in hits_by_file.values())
    total = total_calls + total_names + total_branches
    print(
        f"lint_runner_overrides: scanned {len(hits_by_file)} runner(s) in "
        f"{elapsed_ms:.1f}ms -- {total} FORBIDDEN pattern(s) "
        f"({total_calls} call, {total_names} name, {total_branches} branch)."
    )
    if list_all:
        for rel, hits in sorted(hits_by_file.items()):
            calls, names, branches = _counts(hits)
            base = _BASELINE.get(rel, (0, 0, 0))
            print(
                f"\n  {rel}: calls={calls}/{base[0]} "
                f"names={names}/{base[1]} branches={branches}/{base[2]}"
            )
            for kind, lineno, label in hits:
                print(f"    {rel}:{lineno}: [{kind}] {label}")
    if not regressions and not new_files:
        print("\nlint_runner_overrides: OK -- all runners within baseline.")
        return
    print("\nlint_runner_overrides: REGRESSION")
    if regressions:
        print("\n  Runners that grew beyond baseline:")
        for rel, (bc, bn, bb), (cc, cn, cb) in regressions:
            deltas = []
            if cc > bc:
                deltas.append(f"calls +{cc - bc}")
            if cn > bn:
                deltas.append(f"names +{cn - bn}")
            if cb > bb:
                deltas.append(f"branches +{cb - bb}")
            print(
                f"    {rel}: baseline=({bc},{bn},{bb}) current=({cc},{cn},{cb}) "
                f"[{', '.join(deltas)}]"
            )
            for kind, lineno, label in hits_by_file[rel]:
                print(f"      {rel}:{lineno}: [{kind}] {label}")
    if new_files:
        print("\n  Runners NOT in baseline with forbidden patterns:")
        for rel in new_files:
            for kind, lineno, label in hits_by_file[rel]:
                print(f"    {rel}:{lineno}: [{kind}] {label}")
    print(
        "\nSee c4_release/docs/VANILLA_RESTORE_INVENTORY_2026_06_09.md "
        "for the full catalog + removal plan. Each Wave A-E commit that "
        "deletes overrides should ALSO decrement the matching baseline "
        "entry in tools/lint_runner_overrides.py (counts walk downward "
        "only)."
    )


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument(
        "--list",
        action="store_true",
        help="print every hit (not just regressions)",
    )
    ap.add_argument(
        "--root",
        default=None,
        help="repo root (defaults to auto-detect from cwd)",
    )
    ap.add_argument(
        "--path",
        default=None,
        help=(
            "lint a single .py file instead of the runners; used by the "
            "unit test for planted violations"
        ),
    )
    args = ap.parse_args(argv)

    t0 = time.perf_counter()

    if args.path is not None:
        target = Path(args.path).resolve()
        if not target.exists():
            print(f"error: --path target {target} not found", file=sys.stderr)
            return 2
        hits = _scan_file(target)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if args.json:
            print(
                json.dumps(
                    {
                        "file": str(target),
                        "hits": [
                            {"kind": k, "line": ln, "label": lbl}
                            for k, ln, lbl in hits
                        ],
                        "elapsed_ms": elapsed_ms,
                    },
                    indent=2,
                )
            )
        else:
            if hits:
                print(
                    f"lint_runner_overrides: {len(hits)} FORBIDDEN pattern(s) "
                    f"in {target}:"
                )
                for kind, lineno, label in hits:
                    print(f"  {target}:{lineno}: [{kind}] {label}")
                print(
                    "\nFORBIDDEN under the vanilla-thesis rule: runners are "
                    "pure forward-pass wrappers. See "
                    "c4_release/docs/VANILLA_RESTORE_INVENTORY_2026_06_09.md."
                )
            else:
                print(
                    f"lint_runner_overrides: 0 FORBIDDEN patterns in {target}"
                )
        return 1 if hits else 0

    if args.root:
        repo_root = Path(args.root).resolve()
    else:
        repo_root = _find_repo_root(Path.cwd())
        if repo_root is None:
            print(
                f"error: could not find c4_release/ from {Path.cwd()}",
                file=sys.stderr,
            )
            return 2

    hits_by_file = scan(repo_root)
    regressions, new_files = diff_against_baseline(hits_by_file)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if args.json:
        print(
            json.dumps(
                {
                    "elapsed_ms": elapsed_ms,
                    "files": {
                        rel: {
                            "counts": list(_counts(hs)),
                            "baseline": list(_BASELINE.get(rel, (0, 0, 0))),
                            "hits": [
                                {"kind": k, "line": ln, "label": lbl}
                                for k, ln, lbl in hs
                            ],
                        }
                        for rel, hs in hits_by_file.items()
                    },
                    "regressions": [
                        {
                            "file": rel,
                            "baseline": list(b),
                            "current": list(c),
                        }
                        for rel, b, c in regressions
                    ],
                    "new_files": new_files,
                    "total_forbidden": sum(
                        sum(_counts(hs)) for hs in hits_by_file.values()
                    ),
                },
                indent=2,
            )
        )
        return 1 if (regressions or new_files) else 0

    _print_text_report(
        hits_by_file, regressions, new_files, list_all=args.list, elapsed_ms=elapsed_ms
    )
    return 1 if (regressions or new_files) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
