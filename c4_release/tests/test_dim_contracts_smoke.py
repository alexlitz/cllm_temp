"""Smoke test: every registered ``DimContract`` must verify against the
full live ``compile_full_vm_dynamic`` layout.

This is the fail-closed CI gate for producer-consumer dim drift. The
companion CLI ``tools/dim_contracts_audit.py`` runs the same
verification with richer flags; this pytest entry point is what slots
into the smoke run and breaks the build when an op's ``reads``/``writes``
declaration drifts away from a registered contract (the failure mode
the A3.5 / A3.6 attribution surfaced and the verifier was built to
catch).

The starter set in
``c4_release/neural_vm/unified_compiler/dim_contracts.py`` registers
three load-bearing contracts at module import time:

  1. ``stack0_byte_val_1_lo_pshk2mem`` — L10 PSH broadcast -> L14 MEM
  2. ``stack0_byte_val_1_hi_pshk2mem`` — same pair, HI nibble
  3. ``mem_val_b1_l2_to_l14`` — L2 byte flags -> L14 MEM_VAL_B1 read

When a new producer/consumer pair is hardened (see the
``register_dim_contract`` helper in ``dim_contracts.py``), this smoke
test automatically picks it up — no change required here.

Why ``compile_full_vm_dynamic`` and not ``_build_layout_only``
------------------------------------------------------------
The audit CLI uses ``_build_layout_only`` (cheap — skips weight bake).
This smoke runs the full ``compile_full_vm_dynamic`` path so the
contract assertion lands on the exact same layout the rest of the
smoke suite consumes. The bake is memoised by
``compile_full_vm_dynamic``'s in-process LRU so the extra cost is
near-zero when this test runs alongside the rest of the smoke suite.
"""

from __future__ import annotations

import pytest

from c4_release.neural_vm.unified_compiler.dim_contracts import (
    registered_dim_contracts,
    verify_all_registered_contracts,
)
from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    compile_full_vm_dynamic,
)


# Session-scoped layout: build the full VM once and share across every
# contract assertion below. Same model the rest of the smoke suite uses;
# the disk + in-process cache make the second caller free.
@pytest.fixture(scope="session")
def _full_vm_layout():
    """Build the full VM via ``compile_full_vm_dynamic`` and return the
    compiled ``ModelLayout``.

    Memoised at session scope so the bake cost is paid exactly once across
    the contract suite. ``compile_full_vm_dynamic`` itself has an
    in-process kwargs-keyed memo, so even cross-test calls collapse to a
    single bake when the kwargs match.
    """
    _model, layout = compile_full_vm_dynamic()
    return layout


def test_all_registered_dim_contracts_pass(_full_vm_layout):
    """Every registered ``DimContract`` must verify clean against the
    full ``compile_full_vm_dynamic`` layout.

    Reports per-contract failure with ``ContractValidation.format()`` so
    the smoke output pinpoints the offending producer / consumer / dim
    triple without requiring the maintainer to rerun the audit CLI.
    """
    layout = _full_vm_layout
    contracts = registered_dim_contracts()
    assert contracts, (
        "no DimContracts registered -- did dim_contracts.py "
        "import-time registration drop the starter set?"
    )

    report = verify_all_registered_contracts(layout)
    assert len(report.results) == len(contracts), (
        f"verify_all_registered_contracts returned "
        f"{len(report.results)} results for {len(contracts)} contracts"
    )

    failed = [r for r in report.results if not r.ok]
    if failed:
        # Render each failure via ``ContractValidation.format()`` so the
        # smoke log carries the producer/consumer layer and error list
        # inline -- maintainers don't have to rerun
        # ``python -m c4_release.tools.dim_contracts_audit`` to triage.
        details = "\n".join(r.format() for r in failed)
        pytest.fail(
            f"{len(failed)}/{len(report.results)} registered DimContract(s) "
            f"failed against the live layout:\n{details}"
        )

    # All contracts pass -- positive assertion so failure modes surface as
    # an assertion error if the failed-list path is somehow bypassed.
    assert report.ok, report.format()
    for r in report.results:
        assert r.ok, r.format()
