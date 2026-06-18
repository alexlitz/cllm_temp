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
    dim_ref,
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
    assert reg.resolve_dim("cmp_flag", "cascade") == 396

    # Phase 7.E.0 Bug-A: the ``("carry", "adj")`` -> ADJ_CARRY binding was
    # DROPPED. ADJ_CARRY (static 313) is an ADJ-only band the dim-liveness
    # allocator merges away in the BUILT layout (no surviving alias of the
    # same concept), so resolving it would either KeyError at lowering or
    # silently return an unrelated slot. The binding is intentionally absent
    # now — an ADJ op that needs the band back must re-collect it into the
    # build and tag it via ``register_band_category`` (the Bug-B path).
    with pytest.raises(KeyError):
        reg.resolve_dim("carry", "adj")


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


# ---------------------------------------------------------------------------
# 4. Phase 7.E.2 — dim_ref helper produces byte-identical legacy strings
# ---------------------------------------------------------------------------
def test_dim_ref_carry_alu_byte_index_roles():
    """``dim_ref("carry", "alu", k)`` returns ``"CARRY+k"`` for the
    four byte-index roles of the inter-byte ALU carry cascade.

    This is the canonical Phase 7.E.2 migration shape: a rule that
    today writes ``("CARRY+0", scale)`` re-expresses the same write as
    ``(dim_ref("carry", "alu", 0), scale)``. The returned string must
    be byte-identical to the legacy form so the rule lowers to the
    exact same SwiGLU matrix entries.
    """
    reg = _silent_default_registry()
    for k in range(4):
        assert dim_ref("carry", "alu", k, registry=reg) == f"CARRY+{k}"


def test_dim_ref_output_lo_hi_nibble_roles():
    """``output_lo`` / ``output_hi`` nibble bands resolve to
    ``OUTPUT_LO+k`` / ``OUTPUT_HI+k`` (k = nibble value, structural
    one-hot index)."""
    reg = _silent_default_registry()
    for k in range(16):
        assert dim_ref("output_lo", "nibble", k, registry=reg) == f"OUTPUT_LO+{k}"
        assert dim_ref("output_hi", "nibble", k, registry=reg) == f"OUTPUT_HI+{k}"


def test_dim_ref_ax_carry_register_roles():
    """``ax_carry_lo``/``ax_carry_hi`` map to the AX carry-forward
    staging slot families. The ``offset`` is the nibble value (one-hot
    cell within the 16-wide band)."""
    reg = _silent_default_registry()
    assert dim_ref("ax_carry_lo", "AX", 0, registry=reg) == "AX_CARRY_LO+0"
    assert dim_ref("ax_carry_lo", "AX", 15, registry=reg) == "AX_CARRY_LO+15"
    assert dim_ref("ax_carry_hi", "AX", 7, registry=reg) == "AX_CARRY_HI+7"


def test_dim_ref_opcode_role():
    """``opcode_flag`` carries one role per opcode (LEA, IMM, ...). The
    ``offset`` argument is always 0 for these 1-wide gate slots."""
    reg = _silent_default_registry()
    assert dim_ref("opcode_flag", "ADD", registry=reg) == "OP_ADD+0"
    assert dim_ref("opcode_flag", "LEA", registry=reg) == "OP_LEA+0"
    assert dim_ref("opcode_flag", "PUTCHAR", registry=reg) == "OP_PUTCHAR+0"


def test_dim_ref_missing_pair_raises():
    """Unregistered ``(category, role)`` pairs surface the same
    ``KeyError`` that :meth:`resolve_dim` raises so typos are caught at
    rule-construction time."""
    reg = _silent_default_registry()
    with pytest.raises(KeyError) as excinfo:
        dim_ref("nonexistent_category", "missing_role", registry=reg)
    msg = str(excinfo.value)
    assert "nonexistent_category" in msg
    assert "missing_role" in msg


def test_dim_ref_uses_default_registry_when_not_provided():
    """Callers that omit ``registry=`` get the lazy default-built
    registry. The result must match the explicit-registry path so
    production rule definitions don't need to thread a registry
    through."""
    reg = _silent_default_registry()
    # Suppress the default-registry build's deprecation warnings if any
    # leak through the lazy cache.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        implicit = dim_ref("carry", "alu", 2)
    explicit = dim_ref("carry", "alu", 2, registry=reg)
    assert implicit == explicit == "CARRY+2"


def test_slot_for_category_returns_name():
    """``slot_for_category`` returns the bare slot NAME (no ``+N``
    suffix), which callers can use to construct refs with a custom
    offset format."""
    reg = _silent_default_registry()
    assert reg.slot_for_category("carry", "alu") == "CARRY"
    assert reg.slot_for_category("ax_carry_lo", "AX") == "AX_CARRY_LO"
    assert reg.slot_for_category("opcode_flag", "ADD") == "OP_ADD"


def test_slot_for_category_missing_raises():
    """``slot_for_category`` raises ``KeyError`` for unregistered
    pairs (same diagnostic as :meth:`resolve_dim`)."""
    reg = _silent_default_registry()
    with pytest.raises(KeyError) as excinfo:
        reg.slot_for_category("nope", "missing")
    msg = str(excinfo.value)
    assert "nope" in msg
    assert "missing" in msg


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
