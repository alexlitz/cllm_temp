"""Regression ratchet: dim-alias violations on the authored-ops set AND
the FULL raw-factory corpus == 0.

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


def _collect_raw_factory_ops(alu_mode: str = "lookup") -> list:
    """Collect the FULL raw-factory op corpus.

    Mirrors the op-REGISTRATION logic in
    ``decl_verifier._build_layout_only`` (which itself mirrors
    ``compile_full_vm_dynamic``) but collects the instantiated ops into a
    list rather than compiling a layout. This is the wide corpus the
    2026-06-11 verifier wave drove to zero (commit chain on top of the
    curated-set win 17bc42c2): ``all_core_ops`` +
    ``alu_postop_attach`` + ``alu_divmod`` + ``residual_alibi`` +
    ``contract_validation``.

    Missing imports / instantiation errors are tolerated so a partial
    environment still yields a meaningful (non-empty) corpus.
    """
    from neural_vm.unified_compiler.migrated_ops import (
        all_core_ops,
        make_alu_divmod_composite_ops,
        make_contract_validation_op,
        make_l10_post_op_attach_op,
        make_layer8_op_imm_relay_op,
        make_layer10_divmod_op,
        make_layer10_residual_alibi_slopes_op,
        make_residual_alibi_slopes_op,
        all_alu_postop_attach_ops,
    )

    ops: list = []
    ops.extend(
        all_core_ops(
            alu_mode=alu_mode,
            enable_conversational_io=True,
            enable_tool_calling=True,
        )
    )
    ops.append(make_l10_post_op_attach_op(alu_mode=alu_mode))
    ops.extend(make_alu_divmod_composite_ops(alu_mode=alu_mode))
    ops.append(make_layer10_divmod_op())
    ops.append(make_residual_alibi_slopes_op())
    ops.append(make_layer10_residual_alibi_slopes_op(alu_mode=alu_mode))
    ops.append(make_layer8_op_imm_relay_op())
    ops.append(make_contract_validation_op())
    if alu_mode == "lookup":
        ops.extend(all_alu_postop_attach_ops())
    return ops


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


def test_raw_factory_corpus_has_zero_dim_alias_violations():
    """The FULL raw-factory corpus must report ZERO dim-alias violations.

    This is the broader ratchet locked by the 2026-06-11 verifier wave
    (Improvements G/K/L/M on top of the curated-set win 17bc42c2). The
    raw corpus carried ~1669 violations before the wave, dominated by
    structural sub-bank / same-signal noise the verifier did not yet
    suppress:

    * ``OP_OR``/``OP_AND`` <-> ``OPCODE_FLAGS`` (1148) — one-hot opcode-
      flag CELLS of the parent bank whose DSL-keyword-fallback semantics
      is identical to the bank (Improvement L: containment + equivalent
      semantics => same-signal sub-bank).
    * ``ADDR_B0_LO`` <-> ``OPCODE_BYTE_LO`` (324) — reverse read of the
      NARROWER alias at the position it owns (Improvement G).
    * ``ADDR_B1_LO`` <-> ``OPCODE_BYTE_HI`` (64) — deliberate alias
      traversal at NON-MEM byte rows (Improvement K).
    * ``FETCH_HI`` <-> ``IMM_STAGING`` (64) and ``IO_FORMAT_POS`` <->
      ``MEM_EXEC`` (1) — logically-equivalent same-signal aliases
      (Improvement M).

    Every one of these is a verifier limitation (a false positive the
    verifier should suppress), NOT a real aliasing bug: each suppressor
    is provably safe (same physical signal / narrower-role-active /
    deliberate off-zone traversal). The forward textbook bug
    (``OPCODE_BYTE_LO`` read at ``mark == MEM``) is still flagged —
    see ``test_role_containment_does_not_suppress_forward_textbook_bug``.

    A regression here means a new/edited rule reintroduced an ambiguous
    aliased read, OR a registry edit broke a same-signal alias invariant.
    """
    registry = build_default_registry()
    ops = _collect_raw_factory_ops()

    # Guard the guard: the raw corpus is large (~150 ops). If it ever
    # comes back trivially small, the zero-count assertion would be too
    # weak — require a substantial corpus so the ratchet keeps teeth.
    assert len(ops) >= 100, (
        f"raw-factory corpus collapsed to {len(ops)} ops (<100) — the "
        f"dim-alias ratchet would be too weak. Check migrated_ops imports."
    )

    violations = verify_dim_aliases_for_ops(ops, registry)

    assert not violations, (
        f"raw-corpus dim-alias regression: expected 0 violations across "
        f"{len(ops)} raw-factory ops, got {len(violations)}.\n"
        + format_violations(violations)
    )
