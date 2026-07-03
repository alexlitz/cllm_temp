"""Tests for :mod:`neural_vm.unified_compiler.dim_resolver`.

Two layers:

1. **Contract (fast, no bake):** ``resolve`` / ``resolve_many`` / slice
   helpers over a synthetic ``dim_positions`` map; the unknown-name guard
   raises :class:`UnknownDimError` with a suggestion.
2. **Trap (bake-backed):** on the REAL built layout, the resolver returns
   the BUILT position while the static registry returns a DIFFERENT (wrong)
   position for the same names — the widen-repack trap the module prevents.
   Kept in one test so the slow bake runs once.
"""

from __future__ import annotations

import warnings

import pytest

from neural_vm.unified_compiler.dim_resolver import (
    DimResolver,
    UnknownDimError,
)


class _FakeLayout:
    def __init__(self, positions, sizes):
        self.dim_positions = positions
        self.dim_sizes = sizes


def _resolver():
    pos = {"OP_PSH": 197, "ALU_LO": 330, "MARK_AX": 1}
    sizes = {"OP_PSH": 1, "ALU_LO": 16, "MARK_AX": 1}
    return DimResolver.from_layout(_FakeLayout(pos, sizes))


def test_resolve_basic():
    r = _resolver()
    assert r.resolve("OP_PSH") == 197
    assert r.resolve("ALU_LO") == 330
    assert r.resolve("ALU_LO", offset=15) == 345
    assert r.has("MARK_AX") and not r.has("NOPE")


def test_resolve_many_ordered():
    r = _resolver()
    got = r.resolve_many(["ALU_LO", "OP_PSH"])
    assert got == {"ALU_LO": 330, "OP_PSH": 197}
    assert list(got) == ["ALU_LO", "OP_PSH"]  # insertion order preserved


def test_slice_and_range_use_built_size():
    r = _resolver()
    assert r.range("ALU_LO") == range(330, 346)
    assert r.dim_slice("ALU_LO") == slice(330, 346)
    assert r.size("ALU_LO") == 16


def test_unknown_name_raises_with_suggestion():
    r = _resolver()
    with pytest.raises(UnknownDimError) as ei:
        r.resolve("OP_PHS")  # typo of OP_PSH
    msg = str(ei.value)
    assert "OP_PSH" in msg  # close-match suggestion
    assert "BUILT layout" in msg


def test_offset_out_of_range_raises():
    r = _resolver()
    with pytest.raises(UnknownDimError):
        r.resolve("MARK_AX", offset=5)  # size 1, offset 5 invalid


def test_from_positions_without_sizes():
    r = DimResolver.from_positions({"X": 7})
    assert r.resolve("X") == 7
    assert r.size("X") == 1  # defaults to 1 when no size given


def test_from_layout_rejects_non_layout():
    with pytest.raises(TypeError):
        DimResolver.from_layout(object())


@pytest.mark.slow
def test_built_layout_vs_static_registry_trap():
    """On the REAL build, resolver == built position != static position."""
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    from neural_vm.dim_registry_dynamic import build_default_registry_dynamic

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model, layout = compile_full_vm_dynamic(disk_cache=True)

    resolver = DimResolver.from_layout(layout)
    static = {n: s.start for n, s in build_default_registry_dynamic().slots.items()}

    moved = 0
    for name in ("OP_PSH", "PSH_AT_SP", "OPCODE_BYTE_LO",
                 "OUTPUT_LO", "AX_CARRY_LO"):
        built = layout.dim_positions[name]
        # Contract: resolver returns the BUILT position.
        assert resolver.resolve(name) == built
        # Trap: static registry returns a DIFFERENT position for these.
        if static.get(name) != built:
            moved += 1
    # At least these headline dims move — the trap is real.
    assert moved >= 4, f"expected the widen-repack to move these dims (got {moved})"
