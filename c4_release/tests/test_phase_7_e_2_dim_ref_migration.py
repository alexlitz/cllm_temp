"""
Phase 7.E.2 — pilot FFNRule (category, role) dim-ref migration tests.

Each pilot op below is migrated from hand-rolled ``"NAME+offset"``
strings to :func:`neural_vm.dim_registry.dim_ref` calls where the
offset is **role-meaningful** (a byte index in a multi-byte cascade,
or an opcode flag in the opcode-flag family). Tests pin the migrated
rules' DimRef shape so the migration is byte-identical:

    * gate dims resolve to the same ``DimRef(name=..., offset=0)``;
    * write dims resolve to the same ``DimRef(name=..., offset=0)``;
    * ``dominates_at`` keys still match the legacy string key, so the
      per-output-dim dominance check fires on the same dim.

Structural ``+N`` refs that DON'T move to ``dim_ref`` (operand-A and
operand-B one-hot lookups, output-nibble result writes) are also
asserted unchanged so accidental over-migration is caught.
"""

from __future__ import annotations

import pytest

from neural_vm.dim_registry import dim_ref
from neural_vm.unified_compiler.ir import DimRef
from neural_vm.unified_compiler.ops.l8_ops import (
    _layer8_alu_add_carry_rules,
    _layer8_alu_lea_carry_rules,
    _layer8_alu_sub_borrow_rules,
)


# ---------------------------------------------------------------------------
# Shared invariants helpers
# ---------------------------------------------------------------------------
def _check_carry_byte0_writes(rules, *, expected_scale: float) -> None:
    """Every rule writes exactly the ``(carry, alu)`` byte-0 cell with
    the same ``expected_scale``. The DimRef must equal the legacy
    ``CARRY+0`` form so the lowered weight lands on the same residual
    dim as the pre-migration rule.

    Tests the byte-identity contract of the Phase 7.E.2 migration: the
    ``dim_ref("carry", "alu", 0)`` write resolves to the same
    ``DimRef`` as the ``"CARRY+0"`` write.
    """
    expected_dim = DimRef.parse("CARRY+0")
    for rule in rules:
        assert len(rule.writes) == 1, (
            f"{rule.name}: expected single CARRY write, got {rule.writes}"
        )
        (write,) = rule.writes
        assert write.dim == expected_dim, (
            f"{rule.name}: write dim {write.dim} != {expected_dim}"
        )
        assert write.weight == pytest.approx(expected_scale), (
            f"{rule.name}: write weight {write.weight} != {expected_scale}"
        )


def _check_gate(rules, *, expected_opcode: str) -> None:
    """Every rule's gate is the ``OP_<opcode>`` flag from the
    opcode_flag family. Asserted by DimRef equality so both the legacy
    ``"OP_ADD"`` shape and the migrated ``dim_ref("opcode_flag",
    "ADD")`` shape pass."""
    expected = DimRef.parse(f"OP_{expected_opcode}")
    for rule in rules:
        assert rule.gate == expected, (
            f"{rule.name}: gate {rule.gate} != {expected}"
        )


def _check_dominates_at_carry0(rules, *, expected_scope: str) -> None:
    """``dominates_at`` is a {dim_key: predicate} mapping. The migrated
    rule's CARRY-byte-0 entry must use the canonical ``"CARRY+0"`` key
    string so existing decl_verifier scope checks see the same entry."""
    for rule in rules:
        assert rule.dominates_at is not None, (
            f"{rule.name}: dominates_at missing"
        )
        assert "CARRY+0" in rule.dominates_at, (
            f"{rule.name}: dominates_at keys {list(rule.dominates_at)} "
            f"missing CARRY+0"
        )
        assert rule.dominates_at["CARRY+0"] == expected_scope, (
            f"{rule.name}: dominates_at[CARRY+0] = "
            f"{rule.dominates_at['CARRY+0']!r} != {expected_scope!r}"
        )


# ---------------------------------------------------------------------------
# Pilot op 1: layer8_alu_add_carry
# ---------------------------------------------------------------------------
class TestLayer8AluAddCarry:
    """The ADD carry-detection rules write CARRY+0 (byte 0 of the
    inter-byte ALU cascade) when ``a + b >= 16``. Phase 7.E.2 expresses
    the carry-byte-0 write as ``dim_ref("carry", "alu", 0)`` and the
    OP_ADD gate as ``dim_ref("opcode_flag", "ADD")``; structural
    ``ALU_LO+a`` / ``AX_CARRY_LO+b`` operand reads stay as ``+N``."""

    S = 100.0

    def test_rule_count_unchanged(self):
        """120 units (16x16 cross-product filtered by a+b>=16) — the
        migration must not add or drop a rule."""
        rules = _layer8_alu_add_carry_rules(self.S)
        assert len(rules) == 120

    def test_writes_resolve_to_carry_byte0(self):
        rules = _layer8_alu_add_carry_rules(self.S)
        _check_carry_byte0_writes(
            rules, expected_scale=2.0 / (self.S * 5.0)
        )

    def test_gate_is_op_add_flag(self):
        rules = _layer8_alu_add_carry_rules(self.S)
        _check_gate(rules, expected_opcode="ADD")

    def test_dominates_at_unchanged(self):
        rules = _layer8_alu_add_carry_rules(self.S)
        _check_dominates_at_carry0(
            rules, expected_scope="MARK_AX and OP_ADD"
        )

    def test_operand_a_b_reads_stay_structural(self):
        """``ALU_LO+a`` and ``AX_CARRY_LO+b`` are one-hot nibble lookup
        tables — those ``+N`` offsets are structural and must NOT migrate
        to ``dim_ref``. Pin the operand reads on the first rule (a=1,
        b=15: the first (a, b) pair with a + b >= 16)."""
        rules = _layer8_alu_add_carry_rules(self.S)
        first = rules[0]
        # conditions: MARK_AX, MARK_PC, ALU_LO+1, AX_CARRY_LO+15
        cond_dims = [c.dim for c in first.conditions]
        assert DimRef(name="ALU_LO", offset=1) in cond_dims
        assert DimRef(name="AX_CARRY_LO", offset=15) in cond_dims


# ---------------------------------------------------------------------------
# Pilot op 2: layer8_alu_lea_carry
# ---------------------------------------------------------------------------
class TestLayer8AluLeaCarry:
    """LEA carry: same shape as ADD carry but with the OP_LEA gate and
    the FETCH_LO operand bus instead of AX_CARRY_LO."""

    S = 100.0

    def test_rule_count_unchanged(self):
        rules = _layer8_alu_lea_carry_rules(self.S)
        assert len(rules) == 120

    def test_writes_resolve_to_carry_byte0(self):
        rules = _layer8_alu_lea_carry_rules(self.S)
        _check_carry_byte0_writes(
            rules, expected_scale=2.0 / (self.S * 5.0)
        )

    def test_gate_is_op_lea_flag(self):
        rules = _layer8_alu_lea_carry_rules(self.S)
        _check_gate(rules, expected_opcode="LEA")

    def test_dominates_at_unchanged(self):
        rules = _layer8_alu_lea_carry_rules(self.S)
        _check_dominates_at_carry0(
            rules, expected_scope="MARK_AX and OP_LEA"
        )


# ---------------------------------------------------------------------------
# Pilot op 3: layer8_alu_sub_borrow
# ---------------------------------------------------------------------------
class TestLayer8AluSubBorrow:
    """SUB borrow: fires when ALU_LO[a] < AX_CARRY_LO[b]. Same CARRY+0
    write, OP_SUB gate."""

    S = 100.0

    def test_rule_count_unchanged(self):
        """SUB borrow: only (a, b) with a < b emit, so 120 of 256
        units. The migration must not add or drop a rule."""
        rules = _layer8_alu_sub_borrow_rules(self.S)
        assert len(rules) == 120

    def test_writes_resolve_to_carry_byte0(self):
        rules = _layer8_alu_sub_borrow_rules(self.S)
        _check_carry_byte0_writes(
            rules, expected_scale=2.0 / (self.S * 5.0)
        )

    def test_gate_is_op_sub_flag(self):
        rules = _layer8_alu_sub_borrow_rules(self.S)
        _check_gate(rules, expected_opcode="SUB")

    def test_dominates_at_unchanged(self):
        rules = _layer8_alu_sub_borrow_rules(self.S)
        _check_dominates_at_carry0(
            rules, expected_scope="MARK_AX and OP_SUB"
        )


# ---------------------------------------------------------------------------
# Cross-pilot byte-identity: dim_ref produces the legacy string verbatim
# ---------------------------------------------------------------------------
def test_dim_ref_carry_alu_byte0_equals_carry_plus_0():
    """``dim_ref("carry", "alu", 0)`` must produce the exact string
    ``"CARRY+0"`` so rule definitions that key on the legacy form
    (``dominates_at``) keep matching. This is the byte-identity
    contract of the Phase 7.E.2 migration."""
    assert dim_ref("carry", "alu", 0) == "CARRY+0"


def test_dim_ref_opcode_flag_resolves_to_op_name():
    """``dim_ref("opcode_flag", "ADD")`` returns ``"OP_ADD+0"`` —
    semantically identical to the legacy bare ``"OP_ADD"`` because
    :meth:`DimRef.parse` treats both as the same ``DimRef(name="OP_ADD",
    offset=0)``."""
    assert dim_ref("opcode_flag", "ADD") == "OP_ADD+0"
    assert dim_ref("opcode_flag", "LEA") == "OP_LEA+0"
    assert dim_ref("opcode_flag", "SUB") == "OP_SUB+0"
    # Both forms round-trip through DimRef.parse identically.
    assert DimRef.parse("OP_ADD") == DimRef.parse("OP_ADD+0")
