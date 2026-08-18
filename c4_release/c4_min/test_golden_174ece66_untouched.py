"""Shared golden gate (CPU-safe, NO model load).

Asserts — via a pure `git diff --name-only` against this consolidation
branch's BASE commit — that NONE of the golden model-build-path files was
modified by the consolidation work. This is the ONLY sanctioned golden
check on the Wave-1 branches: rebuilding the model to hash it full-densifies
to ~108 GB RSS and OOM-kills the session, so we NEVER call
`tools/_isa_golden_hash.py` here. A build-path diff is sufficient and safe:
if no golden-build-path file changed, the golden state_dict (174ece66) is
byte-for-byte intact by construction.

The base commit is read from the sibling `_golden_gate_base.txt` marker
(one line, the base SHA) committed on the branch, so this file is identical
across every consolidation branch.

Golden build-path regex (files that DEFINE the golden weights):
    src/compiler.py$ | neural_vm/ | /vm_step.py$ | layer[0-9]+_ops.py$
    | isa_semantics | full_vm.py$ | primitives.py$
"""
from __future__ import annotations

import os
import re
import subprocess

import pytest

# Files whose contents lower the golden model weights. Any change here would
# (or could) move the golden state_dict hash 174ece66.
GOLDEN_BUILD_PATH_RE = re.compile(
    r"src/compiler\.py$|neural_vm/|/vm_step\.py$|layer[0-9]+_ops\.py$"
    r"|isa_semantics|full_vm\.py$|primitives\.py$"
)


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = subprocess.check_output(
        ["git", "-C", here, "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
    return root


def _base_commit() -> str:
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_golden_gate_base.txt")
    with open(marker, "r") as fh:
        return fh.read().strip()


def test_golden_build_path_untouched():
    """No golden-build-path file differs between the branch base and HEAD."""
    root = _repo_root()
    base = _base_commit()
    changed = subprocess.check_output(
        ["git", "-C", root, "diff", "--name-only", base, "HEAD"],
        text=True,
    ).splitlines()
    offenders = [p for p in changed if GOLDEN_BUILD_PATH_RE.search(p)]
    assert not offenders, (
        "Golden model-build-path files were modified by this branch "
        f"(base {base}..HEAD) — golden 174ece66 is NOT provably intact:\n  "
        + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
