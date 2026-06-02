"""
Tests for the Phase 7.E.1 semantic-category extension to
:mod:`neural_vm.dim_registry`.

The category index lets rules reference dims by ``(category, role)``
instead of the legacy ``"OUTPUT_LO+15"``-style strings. The new path is
purely additive: existing slot-name lookups and ``+N`` offsets must
continue to work unchanged.

Coverage:

* per-category resolve hits for each of the suggested initial families
  (marker, byte_index, register_lo/hi, ax_carry_lo/hi, alu_lo/hi,
  carry, cmp_flag, memory_lo/hi, addr_key_nibble, output_lo/hi,
  temp_scratch, opcode_flag);
* the ``resolve_dim`` helper itself (hit, miss, post-hoc registration);
* backward-compat assertions for the legacy slot-name / ``+N`` path.
"""

from __future__ import annotations

import warnings

import pytest

from neural_vm.dim_registry import (
    DimRegistry,
    DimSlot,
    build_default_registry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _silent_default_registry():
    """Build the default registry, suppressing the F-3 semantics
    deprecation warnings the build emits."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return build_default_registry()


# ---------------------------------------------------------------------------
# 1. resolve_dim covers each initial category (one per family)
# ---------------------------------------------------------------------------
def test_resolve_dim_marker_register_byte_index():
    """Three independent families resolve to the right slot ``start``.

    * ``marker`` — MARK_AX lives at dim 1.
    * ``byte_index`` — BYTE_INDEX_2 lives at dim 140.
    * ``register_lo`` — AX_FULL_LO lives at dim 471.
    """
    reg = _silent_default_registry()
    assert reg.resolve_dim("marker", "AX") == 1
    assert reg.resolve_dim("byte_index", "2") == 140
    assert reg.resolve_dim("register_lo", "AX") == 471


def test_resolve_dim_register_hi_and_ax_carry():
    """register_hi + ax_carry_lo/hi categories cover the AX value
    bus and the carry-forward staging."""
    reg = _silent_default_registry()
    assert reg.resolve_dim("register_hi", "AX") == 487
    assert reg.resolve_dim("ax_carry_lo", "AX") == 328
    assert reg.resolve_dim("ax_carry_hi", "AX") == 344


def test_resolve_dim_alu_carry_cmp():
    """alu_lo / alu_hi / carry / cmp_flag — single-role-per-family
    aliases for the ALU + comparison cascades."""
    reg = _silent_default_registry()
    assert reg.resolve_dim("alu_lo", "result") == 360
    assert reg.resolve_dim("alu_hi", "result") == 376
    assert reg.resolve_dim("carry", "alu") == 392
    assert reg.resolve_dim("carry", "adj") == 313
    assert reg.resolve_dim("cmp_flag", "cascade") == 396


def test_resolve_dim_memory_addr_key():
    """memory_lo / memory_hi + addr_key_nibble cover the address byte
    nibble families and the one-hot address key."""
    reg = _silent_default_registry()
    assert reg.resolve_dim("memory_lo", "addr_b0") == 12
    assert reg.resolve_dim("memory_lo", "addr_b1") == 28
    assert reg.resolve_dim("memory_hi", "addr_b0") == 206
    assert reg.resolve_dim("memory_lo", "val_b3") == 464
    assert reg.resolve_dim("addr_key_nibble", "key") == 206


def test_resolve_dim_output_temp_opcode():
    """output_lo / output_hi / temp_scratch / opcode_flag round out
    the initial set. ``opcode_flag`` carries one role per opcode."""
    reg = _silent_default_registry()
    assert reg.resolve_dim("output_lo", "nibble") == 174
    assert reg.resolve_dim("output_hi", "nibble") == 190
    assert reg.resolve_dim("output_lo", "byte") == 480
    assert reg.resolve_dim("temp_scratch", "general") == 480
    assert reg.resolve_dim("opcode_flag", "LEA") == 262
    assert reg.resolve_dim("opcode_flag", "JMP") == 264
    assert reg.resolve_dim("opcode_flag", "PUTCHAR") == 294


# ---------------------------------------------------------------------------
# 2. resolve_dim API behaviour: miss + post-hoc registration
# ---------------------------------------------------------------------------
def test_resolve_dim_missing_raises_keyerror():
    """Unregistered ``(category, role)`` pairs raise ``KeyError`` with a
    message naming both halves of the pair, so a stack trace makes the
    typo obvious."""
    reg = _silent_default_registry()
    with pytest.raises(KeyError) as excinfo:
        reg.resolve_dim("nonexistent_category", "missing_role")
    msg = str(excinfo.value)
    assert "nonexistent_category" in msg
    assert "missing_role" in msg


def test_register_category_post_hoc_and_idempotent():
    """``register_category`` attaches a (category, role) pair to an
    already-allocated slot and exposes it via ``resolve_dim``. Re-
    registering the same pair on the same slot is a no-op; conflicting
    re-registration raises ``ValueError``."""
    reg = DimRegistry(d_model=64)
    reg.alloc("FOO", 4, 2, "test slot", semantics="is_byte")

    # Initially the slot has no category.
    assert reg.slots["FOO"].category is None
    assert reg.slots["FOO"].role is None

    # Attach category.
    reg.register_category("FOO", "carry", "alu")
    assert reg.slots["FOO"].category == "carry"
    assert reg.slots["FOO"].role == "alu"
    assert reg.resolve_dim("carry", "alu") == 4

    # Re-registering with identical (category, role) is a no-op.
    reg.register_category("FOO", "carry", "alu")
    assert reg.resolve_dim("carry", "alu") == 4

    # Re-registering with a different pair on the same slot raises.
    with pytest.raises(ValueError):
        reg.register_category("FOO", "carry", "adj")

    # Re-using an existing (category, role) on a DIFFERENT slot raises.
    reg.alloc("BAR", 8, 2, "second slot", semantics="is_byte")
    with pytest.raises(ValueError):
        reg.register_category("BAR", "carry", "alu")


# ---------------------------------------------------------------------------
# 3. Backward compat — legacy slot-name + ``+N`` references still work
# ---------------------------------------------------------------------------
def test_backward_compat_slot_name_lookup_unchanged():
    """The new (category, role) path is additive: every legacy slot
    is still reachable by name with its original (start, size,
    semantics), and ``resolve_names`` still resolves bare and wildcard
    slot patterns."""
    reg = _silent_default_registry()

    # Bare names resolve.
    output_lo = reg.slots["OUTPUT_LO"]
    assert output_lo.start == 174
    assert output_lo.size == 16

    temp = reg.slots["TEMP"]
    assert temp.start == 480
    assert temp.size == 32

    # The ``+N`` legacy reference style is just a string the consumer
    # parses — the registry hands those through ``resolve_names``
    # unchanged. Verify the prefix-wildcard path still finds the family.
    resolved = reg.resolve_names(["MARK_*"])
    assert "MARK_PC" in resolved
    assert "MARK_AX" in resolved

    # Bare ``"OUTPUT_LO+15"`` is not a slot name; resolve_names returns
    # it untouched so the existing string-parse path keeps owning it.
    assert reg.resolve_names(["OUTPUT_LO+15"]) == ["OUTPUT_LO+15"]


def test_alloc_without_category_still_works():
    """A slot allocated with the legacy ``(name, start, size, desc,
    semantics)`` 5-tuple is still usable — category/role default to
    ``None`` and the slot does NOT appear in the category index, so
    backward-compat callers never have to learn the new API."""
    reg = DimRegistry(d_model=32)
    slot = reg.alloc("LEGACY", 0, 4, "legacy alloc", semantics="is_byte")
    assert slot.category is None
    assert slot.role is None

    # And resolve_dim does not silently invent a category for it.
    with pytest.raises(KeyError):
        reg.resolve_dim("legacy", "LEGACY")

    # The categories() helper omits the slot since it has no binding.
    assert reg.categories() == {}


def test_alloc_with_category_indexes_in_resolve():
    """Inline category+role at alloc time is equivalent to a
    follow-up ``register_category`` call."""
    reg = DimRegistry(d_model=32)
    reg.alloc(
        "INLINE",
        0,
        4,
        "alloc with category",
        semantics="is_byte",
        category="opcode_flag",
        role="LEA",
    )
    assert reg.resolve_dim("opcode_flag", "LEA") == 0
    assert reg.slots["INLINE"].category == "opcode_flag"
    assert reg.slots["INLINE"].role == "LEA"

    # Half-specified registration (category without role, or vice versa)
    # is an error.
    with pytest.raises(ValueError):
        reg.alloc(
            "HALF",
            8,
            4,
            "missing role",
            semantics="is_byte",
            category="opcode_flag",
        )
