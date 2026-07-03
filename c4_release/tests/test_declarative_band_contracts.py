"""Tests for declarative one-hot band margin contracts."""

import torch

from neural_vm.verification.band_contracts import (
    ACTIVE_MARGIN_LOW,
    AMBIGUOUS_WINNER,
    INACTIVE_TOO_HIGH,
    MISSING_VALUE,
    PROJECTED,
    SCALAR_VALUE_DRIFT,
    OneHotBandContract,
    ScalarValueContract,
    declared_output_byte_scalar_contracts,
    project_one_hot_band,
    verify_declared_output_nibble_bands,
    verify_band_contracts,
    verify_one_hot_band,
    verify_scalar_value,
    verify_scalar_value_contracts,
)


def _contract(
    *,
    expected_index: int = 5,
    min_active_margin: float = 0.75,
    max_inactive_value: float = 0.2,
) -> OneHotBandContract:
    return OneHotBandContract(
        name="output_lo_nibble",
        band_base="OUTPUT_LO",
        width=16,
        expected_index=expected_index,
        min_active_margin=min_active_margin,
        max_inactive_value=max_inactive_value,
        tolerance=1e-6,
    )


def _values(active_index: int, active_value: float = 1.0):
    values = [0.0] * 16
    values[active_index] = active_value
    return values


def test_one_hot_band_contract_passes_clean_tensor_band():
    contract = _contract()
    report = verify_one_hot_band(torch.tensor(_values(5)), contract)

    assert report.ok
    assert report.winner_index == 5
    assert report.winner_margin == 1.0


def test_band_contract_reports_active_margin_too_low():
    contract = _contract(min_active_margin=0.75)
    values = _values(5, active_value=0.4)

    report = verify_one_hot_band(values, contract)

    assert not report.ok
    assert [violation.kind for violation in report.violations] == [
        ACTIVE_MARGIN_LOW,
    ]
    assert report.violations[0].index == 5
    assert report.violations[0].observed == 0.4


def test_band_contract_reports_inactive_lane_too_high_from_symbolic_mapping():
    contract = _contract(min_active_margin=0.4, max_inactive_value=0.2)
    output = {
        f"OUTPUT_LO+{idx}": value
        for idx, value in enumerate(_values(5, active_value=1.0))
    }
    output["OUTPUT_LO+9"] = 0.5

    report = verify_one_hot_band(output, contract)

    assert not report.ok
    assert [violation.kind for violation in report.violations] == [
        INACTIVE_TOO_HIGH,
    ]
    assert report.violations[0].index == 9
    assert report.violations[0].observed == 0.5


def test_projection_refuses_ambiguous_winner():
    contract = _contract(min_active_margin=0.25, max_inactive_value=1.0)
    values = _values(5, active_value=0.9)
    values[6] = 0.8

    result = project_one_hot_band(values, contract)

    assert result.status == AMBIGUOUS_WINNER
    assert result.projected_values is None
    assert not result.ok
    assert result.verification.winner_index == 5


def test_projection_corrects_clear_winner_even_with_inactive_violation():
    contract = _contract(min_active_margin=0.5, max_inactive_value=0.2)
    values = _values(5, active_value=1.2)
    values[7] = 0.4

    result = project_one_hot_band(values, contract)

    assert result.status == PROJECTED
    assert result.ok
    assert result.verification.has_violations()
    assert result.verification.violations[0].kind == INACTIVE_TOO_HIGH
    assert result.projected_values == tuple(
        1.0 if idx == 5 else 0.0
        for idx in range(16)
    )
    assert result.as_symbolic_writes()["OUTPUT_LO+5"] == 1.0
    assert result.as_symbolic_writes()["OUTPUT_LO+7"] == 0.0


def test_band_contract_pass_aggregates_multiple_contracts():
    output = {
        "OUTPUT_LO": _values(5, active_value=1.0),
        "OUTPUT_HI": _values(2, active_value=1.0),
    }
    report = verify_band_contracts(
        output,
        [
            _contract(expected_index=5),
            OneHotBandContract(
                band_base="OUTPUT_HI",
                width=16,
                expected_index=2,
                min_active_margin=0.75,
                max_inactive_value=0.2,
            ),
        ],
    )

    assert report.ok
    assert len(report.reports) == 2


def test_declared_output_nibble_report_exposes_actionable_metadata():
    output = {
        "OUTPUT_LO": _values(0xA, active_value=0.42),
        "OUTPUT_HI": _values(0x3, active_value=1.0),
    }
    output["OUTPUT_LO"][0xB] = 0.38
    output["OUTPUT_HI"][0x8] = 0.7

    report = verify_declared_output_nibble_bands(
        output,
        expected_byte=0x3A,
        min_active_margin=0.25,
        max_inactive_value=0.2,
        include_projection=True,
    )

    assert not report.ok
    metadata = report.as_dict()
    assert metadata["expected_byte"] == 0x3A
    assert metadata["expected_low_nibble"] == 0xA
    assert metadata["expected_high_nibble"] == 0x3
    assert any(
        violation["kind"] == ACTIVE_MARGIN_LOW
        and violation["band_base"] == "OUTPUT_LO"
        and violation["index"] == 0xA
        for violation in metadata["violations"]
    )
    assert any(
        violation["kind"] == INACTIVE_TOO_HIGH
        and violation["band_base"] == "OUTPUT_HI"
        and violation["index"] == 0x8
        for violation in metadata["violations"]
    )
    assert len(metadata["projections"]) == 2
    assert "projection_diag=" in report.format_inline()


def test_declared_output_nibble_report_flags_missing_band_lanes():
    output = {
        "OUTPUT_LO": _values(0x5, active_value=1.0),
        "OUTPUT_HI": [1.0],
    }

    report = verify_declared_output_nibble_bands(
        output,
        expected_byte=0x05,
        min_active_margin=0.25,
        max_inactive_value=0.2,
    )

    assert not report.ok
    missing = [
        violation.as_dict()
        for violation in report.violations
        if violation.kind == "missing_value"
    ]
    assert missing
    assert all(violation["band_base"] == "OUTPUT_HI" for violation in missing)


def test_symbolic_expected_byte_vs_neural_like_tensor_reports_band_drift():
    neural_like_output = {
        "OUTPUT_LO": torch.tensor(_values(0xB, active_value=0.60)),
        "OUTPUT_HI": torch.tensor(_values(0x2, active_value=1.0)),
    }
    neural_like_output["OUTPUT_LO"][0xA] = 0.50
    neural_like_output["OUTPUT_HI"][0x7] = 0.65

    report = verify_declared_output_nibble_bands(
        neural_like_output,
        expected_byte=0x2B,
        min_active_margin=0.25,
        max_inactive_value=0.2,
        include_projection=True,
    )

    assert not report.ok
    by_kind = {
        violation.kind: violation
        for violation in report.violations
    }
    assert by_kind[ACTIVE_MARGIN_LOW].band_base == "OUTPUT_LO"
    assert by_kind[ACTIVE_MARGIN_LOW].details["best_inactive_index"] == 10.0
    assert by_kind[INACTIVE_TOO_HIGH].band_base == "OUTPUT_HI"
    assert by_kind[INACTIVE_TOO_HIGH].index == 0x7
    assert report.projections[0].status == AMBIGUOUS_WINNER
    assert report.projections[1].status == PROJECTED


def test_scalar_value_contract_passes_exact_canonical_dim():
    contract = ScalarValueContract(
        name="low_scalar",
        dim_base="OUTPUT_LO_SCALAR",
        expected_value=0xA,
        tolerance=0.0,
    )

    report = verify_scalar_value({"OUTPUT_LO_SCALAR+0": 10.0}, contract)

    assert report.ok
    assert report.value == 10.0
    assert report.observed_delta == 0.0


def test_scalar_value_contract_passes_within_tolerance():
    contract = ScalarValueContract(
        dim_base="OUTPUT_LO_SCALAR",
        expected_value=0xA,
        tolerance=0.05,
    )

    report = verify_scalar_value({"OUTPUT_LO_SCALAR+0": 10.04}, contract)

    assert report.ok
    assert report.value == 10.04


def test_scalar_value_contract_reports_drift_violation():
    contract = ScalarValueContract(
        dim_base="OUTPUT_LO_SCALAR",
        expected_value=0xA,
        tolerance=0.05,
    )

    report = verify_scalar_value({"OUTPUT_LO_SCALAR+0": 10.2}, contract)

    assert not report.ok
    assert [violation.kind for violation in report.violations] == [
        SCALAR_VALUE_DRIFT,
    ]
    assert report.violations[0].index == 0
    assert report.violations[0].observed == 10.2
    assert report.violations[0].limit == 10.0
    assert report.violations[0].details["absolute_delta"] > 0.19


def test_scalar_value_contract_reports_missing_value():
    contract = ScalarValueContract(
        dim_base="OUTPUT_LO_SCALAR",
        expected_value=0xA,
        tolerance=0.05,
    )

    report = verify_scalar_value({"OTHER+0": 10.0}, contract)

    assert not report.ok
    assert [violation.kind for violation in report.violations] == [
        MISSING_VALUE,
    ]
    assert report.violations[0].index == 0
    assert report.violations[0].observed is None


def test_scalar_value_contract_pass_aggregates_low_high_byte_reports():
    contracts = declared_output_byte_scalar_contracts(
        0x3A,
        low_dim="OUTPUT_LO_SCALAR",
        high_dim="OUTPUT_HI_SCALAR",
        tolerance=0.05,
    )
    output = {
        "OUTPUT_LO_SCALAR+0": 10.0,
        "OUTPUT_HI_SCALAR+0": 2.8,
    }

    report = verify_scalar_value_contracts(output, contracts)

    assert not report.ok
    assert len(report.reports) == 2
    assert report.reports[0].ok
    assert not report.reports[1].ok
    assert [violation.kind for violation in report.violations] == [
        SCALAR_VALUE_DRIFT,
    ]
    metadata = report.as_dict()
    assert metadata["contracts_checked"] == 2
    assert metadata["reports"][0]["contract"]["expected_value"] == 10.0
    assert metadata["reports"][1]["contract"]["expected_value"] == 3.0
    assert "Scalar value contract pass report" in report.format()
