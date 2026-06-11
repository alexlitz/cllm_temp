"""Regression ratchet: dim-alias violations on the authored-ops set == 0.

The dim-alias verifier
(:mod:`neural_vm.unified_compiler.dim_alias_verifier`) flags FFNRules that
read an aliased residual slot at positions where the alias could carry a
*different* semantic value (the textbook ``OPCODE_BYTE_LO`` read at a
``mark == MEM`` row that actually carries ``ADDR_B0_LO``'s address byte).

During the 2026-06-10/11 verifier wave the violation count on the curated
authored-ops set (the ops that have been scoped / ``dominates_at``'d and
hard-blocker-disambiguated) was driven to **0** across commits e988348b,
24219415, a86bcf65, 1d77e89d, 24a7113a, 2e350d74, and 17bc42c2.

This module is the ratchet that locks that win: any future rule edit that
reintroduces an undisambiguated aliased read into one of the curated
authored ops fails this test. The curated set is
:func:`decl_verifier.collect_all_authored_ops` — the ops the verifier work
explicitly cleaned (L10 tail correction, L16 LEV routing, L15 nibble copy,
L6 ENT-after-JSR fixup). It deliberately excludes the wider raw-factory
corpus, which still carries structural sub-bank noise the verifier does not
yet suppress (e.g. ``OP_OR``/``OP_AND`` inside the ``OPCODE_FLAGS`` bank
with identical semantics). Expanding the curated set is the way to broaden
this guard as more ops are scoped.

If this test starts failing:

  1. Run ``verify_dim_aliases_for_ops`` over ``collect_all_authored_ops()``
     and read ``format_violations`` to see which rule regressed.
  2. The fix is almost always a hard blocker (``-1e6``) on the marker that
     excludes the conflicting alias's semantics (e.g. ``MARK_MEM`` /
     ``MARK_PC`` / ``IS_BYTE``), NOT a verifier-side suppressor.
"""

from __future__ import annotations

from neural_vm.dim_registry import build_default_registry
from neural_vm.unified_compiler.decl_verifier import (
    collect_all_authored_ops,
)
from neural_vm.unified_compiler.dim_alias_verifier import (
    format_violations,
    verify_dim_aliases_for_ops,
)


def test_authored_ops_have_zero_dim_alias_violations():
    """The curated authored-ops set must report ZERO dim-alias
    violations against the default static registry.

    This is the locked win: every rule in these ops reads aliased
    residual slots only at positions disambiguated (by hard blockers,
    scoped predicates, or opcode/phase disjointness) from every other
    alias sharing the byte range. A regression here means a new or
    edited rule reintroduced an ambiguous aliased read.
    """
    registry = build_default_registry()
    ops = collect_all_authored_ops()

    # Guard the guard: if the curated set ever comes back empty (e.g. an
    # import regression silently drops every factory), the zero-count
    # assertion would pass vacuously. Require the set to be non-empty so
    # the ratchet keeps real teeth.
    assert ops, (
        "collect_all_authored_ops() returned no ops — the dim-alias "
        "ratchet would pass vacuously. Check decl_verifier imports."
    )

    violations = verify_dim_aliases_for_ops(ops, registry)

    assert not violations, (
        f"dim-alias regression: expected 0 violations across "
        f"{len(ops)} authored ops, got {len(violations)}.\n"
        + format_violations(violations)
    )
