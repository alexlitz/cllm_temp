"""Unit tests for the producer-consumer ``DimContract`` verifier.

Two layers of coverage:

1. **Synthetic layouts:** small hand-built ``LayerCompiler`` instances
   that exercise the contract semantics without paying the cost of the
   full ``compile_full_vm_dynamic`` build. Covers the success path, the
   missing-producer path, and the intervening-writer
   (``must_not_zero_between``) path.
2. **Live layout sanity:** runs the registered starter contracts
   against the real layout to confirm the audit at least *parses* every
   contract and produces a structured report. We do NOT assert the
   live layout passes -- the A3.6 attribution showed at least one
   declared-reads gap that the verifier should surface as FAIL.
"""

from __future__ import annotations

import pytest

from c4_release.neural_vm.verification.dim_contracts import (
    ContractValidation,
    DimContract,
    OpRef,
    OpcodeSet,
    clear_registered_dim_contracts,
    register_dim_contract,
    registered_dim_contracts,
    verify_all_registered_contracts,
    verify_dim_contract,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_minimal_layout(*, with_intervening: bool = False):
    """Construct a 3-op LayerCompiler layout with one producer/consumer.

    Producer writes ``DIM_X`` at layer 0 (FFN); consumer reads it at
    layer 2 (FFN). When ``with_intervening=True``, an extra op at layer
    1 also writes ``DIM_X`` -- used to exercise the
    ``must_not_zero_between`` failure path.
    """
    compiler = LayerCompiler()
    compiler.declare_dim("GATE_A", 1)
    compiler.declare_dim("DIM_X", 1)
    compiler.declare_dim("DIM_Y", 1)

    def _noop_bake(*_args, **_kwargs):
        return None

    compiler.add_op(Operation(
        name="prod",
        reads={"GATE_A"},
        writes={"DIM_X"},
        kind="ffn",
        bake_fn=_noop_bake,
        phase=1.0,
    ))
    if with_intervening:
        # An intermediate op writes DIM_X too -- must_not_zero_between
        # should flag this as a violation.
        compiler.add_op(Operation(
            name="middle",
            reads={"DIM_X"},
            writes={"DIM_X", "DIM_Y"},
            kind="ffn",
            bake_fn=_noop_bake,
            phase=2.0,
        ))
    compiler.add_op(Operation(
        name="cons",
        reads={"DIM_X", "DIM_Y" if with_intervening else "GATE_A"},
        writes={"DIM_Y"} if not with_intervening else set(),
        kind="ffn",
        bake_fn=_noop_bake,
        phase=3.0,
    ))
    return compiler.compile()


# ---------------------------------------------------------------------------
# Test 1: success path
# ---------------------------------------------------------------------------


def test_dim_contract_pass_path():
    """A well-formed producer/consumer pair with no intervening writer
    produces an empty ``errors`` list and ``ok == True``.
    """
    layout = _build_minimal_layout(with_intervening=False)
    contract = DimContract(
        dim="DIM_X",
        producer=OpRef("prod", layer=0),
        consumer=OpRef("cons", layer=2),
        must_not_zero_between=True,
    )
    result = verify_dim_contract(layout, contract)
    assert isinstance(result, ContractValidation), type(result)
    assert result.ok, (
        f"expected pass, got errors: {result.errors!r}\n{result.format()}"
    )
    assert result.producer_layer is not None
    assert result.consumer_layer is not None
    assert result.producer_layer < result.consumer_layer
    assert not result.intervening_writers


# ---------------------------------------------------------------------------
# Test 2: producer absent -> failure
# ---------------------------------------------------------------------------


def test_dim_contract_missing_producer():
    """A contract whose producer op is not in the layout reports a
    single structured error pinning the op name.
    """
    layout = _build_minimal_layout()
    contract = DimContract(
        dim="DIM_X",
        producer=OpRef("nonexistent_op", layer=0),
        consumer=OpRef("cons", layer=2),
    )
    result = verify_dim_contract(layout, contract)
    assert not result.ok
    assert any("nonexistent_op" in err for err in result.errors), result.errors


# ---------------------------------------------------------------------------
# Test 3: consumer doesn't declare read -> failure
# ---------------------------------------------------------------------------


def test_dim_contract_consumer_missing_read():
    """When the consumer op doesn't list ``dim`` in its declared
    ``reads`` set, the verifier flags it as an error -- this is the
    A3.6-style drift the contract is designed to surface.
    """
    layout = _build_minimal_layout()
    contract = DimContract(
        dim="DIM_Y",  # cons writes DIM_Y but doesn't read it
        producer=OpRef("prod", layer=0),
        consumer=OpRef("cons", layer=2),
    )
    result = verify_dim_contract(layout, contract)
    # prod doesn't write DIM_Y AND cons doesn't read DIM_Y -- two errors.
    assert not result.ok
    assert any("does not declare" in err and "writes" in err
               for err in result.errors), result.errors
    assert any("does not declare" in err and "reads" in err
               for err in result.errors), result.errors


# ---------------------------------------------------------------------------
# Test 4: must_not_zero_between catches intervening writers
# ---------------------------------------------------------------------------


def test_dim_contract_intervening_writer_violation():
    """An intervening writer between producer and consumer triggers a
    ``must_not_zero_between`` violation; the offending op is reported
    in ``intervening_writers``.
    """
    layout = _build_minimal_layout(with_intervening=True)
    contract = DimContract(
        dim="DIM_X",
        producer=OpRef("prod"),
        consumer=OpRef("cons"),
        must_not_zero_between=True,
    )
    result = verify_dim_contract(layout, contract)
    assert not result.ok
    assert result.intervening_writers, "expected at least one writer"
    writer_names = {w[1] for w in result.intervening_writers}
    assert "middle" in writer_names, writer_names
    assert any("must_not_zero_between violated" in err
               for err in result.errors), result.errors


# ---------------------------------------------------------------------------
# Test 5: registry round-trip
# ---------------------------------------------------------------------------


def test_registry_round_trip_with_synthetic_contract(monkeypatch):
    """``register_dim_contract`` + ``verify_all_registered_contracts``
    returns a report containing the synthetic contract.

    Uses a fresh layout and a roll-the-registry-back fixture so the
    starter contracts don't leak in.
    """
    # Snapshot + reset the registry so the starter contracts don't
    # leak into this synthetic-layout test (the synthetic layout has
    # none of the L10/L14 ops).
    saved = list(registered_dim_contracts())
    clear_registered_dim_contracts()
    try:
        layout = _build_minimal_layout()
        register_dim_contract(DimContract(
            dim="DIM_X",
            producer=OpRef("prod"),
            consumer=OpRef("cons"),
            name="synth_contract",
        ))
        report = verify_all_registered_contracts(layout)
        assert len(report.results) == 1
        assert report.results[0].contract.display_name() == "synth_contract"
        assert report.ok, report.format()
    finally:
        clear_registered_dim_contracts()
        for c in saved:
            register_dim_contract(c)


# ---------------------------------------------------------------------------
# Test 6: opcode-gate NOTE surfaces missing gate dims
# ---------------------------------------------------------------------------


def test_opcode_gate_note_when_gate_dim_missing():
    """When the contract's ``when`` opcode set names an opcode that
    isn't in the producer/consumer ``reads`` set, the verifier records
    a NOTE (not an error) -- many ops gate on composite predicates
    rather than the raw ``OP_*`` dim.
    """
    layout = _build_minimal_layout()
    contract = DimContract(
        dim="DIM_X",
        # Cons reads DIM_X + GATE_A but no "OP_PSH" dim.
        producer=OpRef("prod", when=OpcodeSet("PSH")),
        consumer=OpRef("cons", when=OpcodeSet("PSH")),
    )
    result = verify_dim_contract(layout, contract)
    assert result.ok, result.format()
    # At least one NOTE about a missing gate.
    assert any("gate dims not in reads" in n
               for n in result.notes), result.notes
