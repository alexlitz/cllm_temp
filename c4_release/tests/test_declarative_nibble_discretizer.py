"""Tests for first-class declarative exact nibble/byte emission helpers."""

import pytest

from neural_vm.unified_compiler.ir import FFNRule, compare_symbolic_to_lowered_ffn
from neural_vm.unified_compiler.primitives import Primitives


def _write_map(writes):
    return {dim: weight for dim, weight in writes}


def _assert_exact_nibble(
    writes,
    base: str,
    selected: int,
    *,
    strength: float,
    competitor_strength: float,
):
    mapped = _write_map(writes)
    assert len(mapped) == 16
    for k in range(16):
        expected = strength if k == selected else -competitor_strength
        assert mapped[f"{base}+{k}"] == expected


def test_nibble_value_writes_selects_one_channel_and_suppresses_rest():
    writes = Primitives.nibble_value_writes(
        "OUTPUT_LO",
        0xA,
        strength=7.0,
        competitor_strength=3.0,
    )

    _assert_exact_nibble(
        writes,
        "OUTPUT_LO",
        0xA,
        strength=7.0,
        competitor_strength=3.0,
    )


@pytest.mark.parametrize(
    ("value", "lo", "hi"),
    [
        (0x00, 0x0, 0x0),
        (0x0F, 0xF, 0x0),
        (0x10, 0x0, 0x1),
        (0xFF, 0xF, 0xF),
    ],
)
def test_byte_value_writes_selects_exact_low_and_high_nibbles(value, lo, hi):
    writes = Primitives.byte_value_writes(value, strength=11.0)

    assert len(writes) == 32
    assert writes[0][0] == "OUTPUT_LO+0"
    assert writes[1][0] == "OUTPUT_HI+0"
    mapped = _write_map(writes)
    for k in range(16):
        assert mapped[f"OUTPUT_LO+{k}"] == (11.0 if k == lo else -11.0)
        assert mapped[f"OUTPUT_HI+{k}"] == (11.0 if k == hi else -11.0)


def test_byte_value_writes_supports_custom_target_bands():
    writes = Primitives.byte_value_writes(
        0x21,
        lo_base="TMP_LO",
        hi_base="TMP_HI",
        strength=5.0,
    )
    mapped = _write_map(writes)

    assert mapped["TMP_LO+1"] == 5.0
    assert mapped["TMP_HI+2"] == 5.0
    assert mapped["TMP_LO+0"] == -5.0
    assert mapped["TMP_HI+0"] == -5.0


def test_byte_value_writes_symbolic_and_lowered_ffn_match():
    rule = FFNRule.constant_write(
        conditions=(("COND", 1.0),),
        threshold=0.5,
        writes=Primitives.byte_value_writes(0x10, strength=13.0),
        name="emit_exact_0x10",
    )

    report = compare_symbolic_to_lowered_ffn(
        rule,
        {"COND": 0, "OUTPUT_LO": 1, "OUTPUT_HI": 17},
        S=100.0,
        atol=1e-4,
        rtol=1e-4,
    )

    assert report.ok, report.format()
    assert report.symbolic_state["OUTPUT_LO+0"] == 13.0
    assert report.symbolic_state["OUTPUT_HI+1"] == 13.0
    assert report.symbolic_state["OUTPUT_LO+1"] == -13.0
    assert report.symbolic_state["OUTPUT_HI+0"] == -13.0


def test_nibble_and_byte_values_validate_ranges():
    with pytest.raises(ValueError, match="nibble value"):
        Primitives.nibble_value_writes("OUTPUT_LO", 0x10)
    with pytest.raises(ValueError, match="byte value"):
        Primitives.byte_value_writes(0x100)
    with pytest.raises(ValueError, match="strength"):
        Primitives.byte_value_writes(0x00, strength=0.0)
