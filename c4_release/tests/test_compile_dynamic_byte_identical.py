"""Scheduler-invariant tests for the dynamic compile path (B11).

The historical slow byte-identity tests (which ran ``compile_full_vm_dynamic``
under the legacy static phase-pruning body and diffed it against
``compile_full_vm_dynamic``) were removed in Phase 8.G.3 alongside the
static body itself. The fast scheduler invariants below — schedule
permutation, topological validity, cycle-member presence under
unpruned-but-not-pruned graphs — continue to gate any change to the
declared deps that would break the dep-pruned order.
"""

import pytest

from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    compute_dynamic_schedule,
    _build_dep_graph,
    _build_phase_pruned_graph,
    _collect_ops_for_compile,
    _find_cycle_members,
)


# ---------------------------------------------------------------------------
# Fast (non-build) scheduler invariants
# ---------------------------------------------------------------------------


def test_dynamic_schedule_orders_every_op_once():
    """The hybrid schedule must be a permutation of the input op list."""
    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    scheduled, source = compute_dynamic_schedule(ops)
    assert len(scheduled) == len(ops)
    assert {o.name for o in scheduled} == {o.name for o in ops}
    # Every op must have been classified as either dep-derived or
    # phase-fallback. No op may be silently dropped.
    assert set(source.keys()) == {o.name for o in ops}
    for src in source.values():
        assert src in ("dep", "phase")


def test_dynamic_schedule_is_a_valid_topological_sort():
    """The hybrid schedule must be a valid topological sort of the
    phase-pruned dep DAG. For every edge u -> v in the pruned graph,
    u must appear before v in the schedule.

    This is the precondition for B11 byte-identity: if the dep order is
    a valid topo sort under the same pruning rule the static path uses,
    then the static ``LayerCompiler`` will produce the same layout
    regardless of whether the dep order or the input order is the
    starting Kahn's input (because both are valid topo sorts of the
    same DAG, and ``LayerCompiler._assign_layers`` is layer-pinning-
    dominant on today's op set).
    """
    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    )
    by_dep, _ = compute_dynamic_schedule(ops)
    dep_position = {op.name: i for i, op in enumerate(by_dep)}
    # Mirror the static LayerCompiler: only attn/ffn ops participate in
    # the topological constraint check. Block / model ops are pinned by
    # layer_idx or phase, not by deps.
    _in_e, out_e, _cycle = _build_phase_pruned_graph(
        ops, restrict_to_kinds={"attn", "ffn"},
    )
    violations: list = []
    for u in ops:
        if u.kind not in ("attn", "ffn"):
            continue
        for v_name in out_e.get(u.name, set()):
            if dep_position[u.name] >= dep_position[v_name]:
                violations.append(
                    (u.name, dep_position[u.name],
                     v_name, dep_position[v_name])
                )
    assert not violations, (
        f"Dynamic schedule violates phase-pruned topo order in "
        f"{len(violations)} places; first 5: {violations[:5]}"
    )


def test_dynamic_schedule_cycle_members_exist_and_phase_pruning_breaks_them():
    """Verify the Phase A finding: the dep-only DAG (no phase pruning)
    has cycle members, and applying the static phase-pruning rule
    breaks every cycle.

    On the Phase A reference date (2026-06-01) the unpruned graph had
    57 SCC members across the lookup-mode "all-flags-on" op set; the
    repo has since grown the op count by ~8 with additional claims,
    so the cycle count tracks accordingly. We lock the LOWER BOUND
    (>50 cycle members) here — anything significantly under that
    suggests a major SCC break worth celebrating, and anything zero
    means the unpruned graph is now acyclic (drop B11's hybrid
    fallback entirely in that case).
    """
    ops = _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=True,
        enable_tool_calling=True,
        enable_neural_io_think_protocol=True,
    )
    # Unpruned dep graph: declared deps only, no phase pruning.
    in_edges, out_edges = _build_dep_graph(ops)
    cycle = _find_cycle_members(ops, in_edges, out_edges)
    assert len(cycle) >= 50, (
        f"unpruned dep cycle members dropped well below the Phase A "
        f"baseline (~57). Got {len(cycle)} — if you've broken the "
        f"largest SCC, celebrate AND relax this lower bound."
    )
    assert len(cycle) <= 120, (
        f"unpruned dep cycle members shot above the Phase A baseline "
        f"(~57, today ~92 after additional Phase 7.A op registrations). "
        f"Got {len(cycle)} — a new op likely introduced "
        f"unannotated back-edges."
    )

    # Phase-pruned graph: must be acyclic (this is the invariant the
    # static path depends on, and the precondition for B11
    # byte-identity).
    pruned_in, pruned_out, _ = _build_phase_pruned_graph(
        ops, restrict_to_kinds={"attn", "ffn"},
    )
    attn_ffn = [op for op in ops if op.kind in ("attn", "ffn")]
    pruned_cycle = _find_cycle_members(attn_ffn, pruned_in, pruned_out)
    assert pruned_cycle == set(), (
        f"Phase pruning fails to break all cycles: "
        f"{len(pruned_cycle)} ops stuck in {sorted(pruned_cycle)[:5]}..."
    )


# ---------------------------------------------------------------------------
# Slow: end-to-end build smoke
# ---------------------------------------------------------------------------
#
# Phase 8.G.3 deleted the byte-identity slow tests that previously ran
# ``compile_full_vm_dynamic`` against the legacy ``compile_full_vm_dynamic``
# static body via ``compare_compile_paths``. The static body is gone, so
# the diff has no left operand to compare against. The dep-graph
# regression surface lives in the fast scheduler-invariant tests above.


@pytest.mark.slow
def test_compile_full_vm_dynamic_runs_with_all_flags():
    """The hybrid scheduler must not crash on any op, including the
    flag-gated convo-IO / tool-call / think-protocol bakes that gate on
    runtime metadata. Cycle members are routed through the phase-fallback
    branch; this test exercises that path under flags=on.
    """
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    model, layout = compile_full_vm_dynamic(
        enable_conversational_io=True,
        enable_tool_calling=True,
        enable_neural_io_think_protocol=True,
        disk_cache=False,
    )
    # Sanity: the model + layout are non-empty.
    assert layout.n_layers > 0
    assert layout.d_model > 0
    sd = model.state_dict()
    assert len(sd) > 0


# ---------------------------------------------------------------------------
# V2 toggle-IR wiring: ``arch=ModelArchitectureSpec(...)``
# ---------------------------------------------------------------------------


def test_arch_spec_kwarg_rejects_mixed_individual_kwargs():
    """Passing both ``arch=`` and an individual architectural kwarg must
    fail loudly. The two surfaces are mutually exclusive — silently
    privileging one would let a caller think a toggle took effect when it
    didn't.
    """
    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    from c4_release.neural_vm.unified_compiler.ir import (
        ModelArchitectureSpec,
        PositionalEncodingSpec,
    )

    spec = ModelArchitectureSpec(
        positional_encoding=PositionalEncodingSpec(kind="rope", rope_base=10000.0),
    )
    with pytest.raises(TypeError, match="mutually exclusive"):
        compile_full_vm_dynamic(
            arch=spec,
            positional_encoding="alibi",  # conflicts with spec
            disk_cache=False,
        )


@pytest.mark.slow
def test_arch_spec_kwarg_byte_identical_to_individual_kwargs():
    """Building a VM via ``arch=ModelArchitectureSpec.from_compile_kwargs(...)``
    must be byte-identical to building it via the legacy individual
    architectural kwargs. The spec is just a typed name for the same
    five values, so the compiled state-dict tensors must match
    exactly.
    """
    import torch

    from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    from c4_release.neural_vm.unified_compiler.ir import ModelArchitectureSpec

    common_kwargs = dict(
        alu_mode="lookup",
        disk_cache=False,
    )

    # Path A: legacy individual kwargs (the historical surface).
    model_kwargs, layout_kwargs = compile_full_vm_dynamic(
        positional_encoding="alibi",
        attention_normalization="softmax1",
        rope_base=10000.0,
        use_rms_norm=False,
        rms_norm_eps=1e-6,
        **common_kwargs,
    )

    # Path B: single ``arch=`` spec (the V2 surface). Build the spec via
    # the documented adapter so the two paths are explicitly equivalent.
    spec = ModelArchitectureSpec.from_compile_kwargs(
        positional_encoding="alibi",
        attention_normalization="softmax1",
        rope_base=10000.0,
        use_rms_norm=False,
        rms_norm_eps=1e-6,
    )
    model_arch, layout_arch = compile_full_vm_dynamic(arch=spec, **common_kwargs)

    # Layout-level shape parity: both paths must produce the same
    # topology. (Layout deltas would surface before tensor diffs.)
    assert layout_kwargs.n_layers == layout_arch.n_layers
    assert layout_kwargs.d_model == layout_arch.d_model

    # State-dict tensor byte-identity: every key must match and every
    # tensor must be bitwise-equal. Random init is keyed by
    # ``torch.manual_seed`` inside the compile path, so the two builds
    # see identical RNG state when given byte-equivalent kwargs.
    sd_kwargs = model_kwargs.state_dict()
    sd_arch = model_arch.state_dict()
    assert set(sd_kwargs.keys()) == set(sd_arch.keys()), (
        "arch= path produced a different state_dict key set than the "
        "kwarg path"
    )
    for key in sd_kwargs:
        t_kw = sd_kwargs[key]
        t_ar = sd_arch[key]
        assert t_kw.shape == t_ar.shape, (
            f"shape diff at {key}: kwargs={tuple(t_kw.shape)} vs "
            f"arch={tuple(t_ar.shape)}"
        )
        assert torch.equal(t_kw, t_ar), (
            f"tensor diff at {key} between kwargs path and arch= path "
            f"(max-abs diff "
            f"{(t_kw - t_ar).abs().max().item() if t_kw.is_floating_point() else 'non-float'})"
        )
