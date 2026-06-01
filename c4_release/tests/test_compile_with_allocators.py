"""B16 wiring tests: dim / FFN-unit / attention-head allocators on
``ModelLayout`` after :func:`compile_full_vm_dynamic`.

Three coverage layers:

1. **Reachability** — every allocator is present on the returned layout
   under canonical attribute names, regardless of whether the bake was
   served from cache or freshly compiled.
2. **Byte-identity** — the pin-every-existing-dim wiring must NOT
   change the layout returned by ``compile_full_vm_dynamic``; the B11
   ``compare_compile_paths`` invariant must continue to report
   ``n_diffs=0`` against the static ``compile_full_vm``.
3. **Per-op declaration semantics** — a synthetic new op asking for
   ``N`` units (no ``pin=``) lands in a free gap that doesn't overlap
   the existing FFN tail at that layer, so future per-op migrations
   can request allocations declaratively.

The full-bake byte-identity check (#2) runs the static + dynamic
compiles back-to-back; it's marked ``slow`` to mirror
``tests/test_compile_dynamic_byte_identical.py``. The reachability and
synthetic-op tests do not bake a real model — they call the layout
helpers directly so they can stay in the fast suite.
"""

from __future__ import annotations

import pytest

from c4_release.neural_vm.dim_allocator import Allocator
from c4_release.neural_vm.ffn_unit_allocator import FFNUnitAllocator
from c4_release.neural_vm.attention_head_allocator import (
    AttentionHeadAllocator,
)
from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    LAYOUT_ATTENTION_HEAD_ALLOCATORS_ATTR,
    LAYOUT_DIM_ALLOCATOR_ATTR,
    LAYOUT_FFN_UNIT_ALLOCATORS_ATTR,
    _attach_allocators_to_layout,
    _collect_ops_for_compile,
    compare_compile_paths,
    compile_full_vm_dynamic,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
)
from c4_release.neural_vm.unified_compiler.ops.shared import (
    declare_setdim_compat_dims,
)


# ---------------------------------------------------------------------------
# Shared layout fixture
# ---------------------------------------------------------------------------


def _build_layout_for_tests():
    """Compile a layout WITHOUT running the bake.

    Runs the same op-collection / compile dance ``_bake_from_scheduled_ops``
    does, stopping after ``compiler.compile()`` so each test gets a fresh
    layout cheaply (the full bake takes ~40-70s and is exercised by the
    slow byte-identity test below).
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in _collect_ops_for_compile(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        enable_neural_io_think_protocol=False,
    ):
        compiler.add_op(op)
    layout = compiler.compile()
    if layout.d_model % 8 != 0:
        pad = 8 - (layout.d_model % 8)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()
    _attach_allocators_to_layout(layout, ffn_hidden=4096, n_heads=8)
    return layout


# ---------------------------------------------------------------------------
# 1. Reachability
# ---------------------------------------------------------------------------


def test_all_three_allocators_reachable_on_layout():
    """After ``_attach_allocators_to_layout`` the layout must expose the
    three allocators at the canonical attribute names and they must be
    the correct types so per-op migrations can rely on the contracts.
    """
    layout = _build_layout_for_tests()

    dim_alloc = getattr(layout, LAYOUT_DIM_ALLOCATOR_ATTR)
    ffn_allocs = getattr(layout, LAYOUT_FFN_UNIT_ALLOCATORS_ATTR)
    head_allocs = getattr(layout, LAYOUT_ATTENTION_HEAD_ALLOCATORS_ATTR)

    assert isinstance(dim_alloc, Allocator)
    assert isinstance(ffn_allocs, dict)
    assert isinstance(head_allocs, dict)

    assert dim_alloc.d_model == layout.d_model

    # Per-layer dicts cover every layer in the compiled model so a
    # follow-up migration of an op at any layer can rely on the
    # allocator being present without conditional construction.
    expected_layers = set(range(layout.n_layers))
    assert set(ffn_allocs.keys()) == expected_layers
    assert set(head_allocs.keys()) == expected_layers
    for layer_idx, a in ffn_allocs.items():
        assert isinstance(a, FFNUnitAllocator)
    for layer_idx, a in head_allocs.items():
        assert isinstance(a, AttentionHeadAllocator)


def test_dim_allocator_pins_every_existing_dim():
    """Every dim in ``layout.dim_positions`` must show up in the dim
    allocator at the same ``(start, size)`` — this is the
    pin-every-existing-dim seed that makes the wiring byte-identical to
    the static path.
    """
    layout = _build_layout_for_tests()
    dim_alloc = getattr(layout, LAYOUT_DIM_ALLOCATOR_ATTR)
    by_name = {s.name: s for s in dim_alloc.slots()}

    # Same set of names — no dropped dim, no surprise additions.
    assert set(by_name.keys()) == set(layout.dim_positions.keys())

    for name, pos in layout.dim_positions.items():
        slot = by_name[name]
        assert slot.start == pos, (
            f"dim_allocator slot {name!r} start={slot.start} disagrees "
            f"with layout.dim_positions[{name!r}]={pos}"
        )
        assert slot.size == layout.dim_sizes[name]
        assert slot.pinned is True


def test_ffn_unit_allocator_pins_existing_tail_per_layer():
    """Each layer's ``ffn_widths`` value must be reflected as a single
    pinned ``existing_layer<N>`` range starting at unit 0, so the next
    unpinned ``alloc(N)`` lands at the tail and never collides with the
    legacy bake's hand-picked unit offsets.
    """
    layout = _build_layout_for_tests()
    ffn_allocs = getattr(layout, LAYOUT_FFN_UNIT_ALLOCATORS_ATTR)

    for layer_idx in range(layout.n_layers):
        a = ffn_allocs[layer_idx]
        existing = layout.ffn_widths.get(layer_idx, 0)
        ranges = a.ranges()
        if existing > 0:
            assert len(ranges) == 1, (
                f"layer {layer_idx} expected 1 seed range, got {ranges}"
            )
            r = ranges[0]
            assert r.op_name == f"existing_layer{layer_idx}"
            assert r.start == 0
            assert r.n_units == existing
            assert r.pinned is True
        else:
            assert ranges == [], (
                f"layer {layer_idx} has no ffn_widths entry; allocator "
                f"should be empty but got {ranges}"
            )


def test_attention_head_allocator_picks_up_alibi_slope_declarations():
    """Per-op ``alibi_slopes`` keys are the declarative source for
    "this op binds head_idx X in some layer". The seed wiring must
    pre-pin every such head so a follow-up migration that asks for a
    head in the same layer never silently steals one.

    On the current op set L14's ``layer14_mem_generation`` declares all
    8 heads. Assert that the layer-14 allocator has those 8 heads
    pinned — this is the canonical "fully-occupied layer" sanity check.
    """
    layout = _build_layout_for_tests()
    head_allocs = getattr(layout, LAYOUT_ATTENTION_HEAD_ALLOCATORS_ATTR)

    l14 = head_allocs[14]
    head_indices = {rec.head_idx for rec in l14.heads()}
    assert head_indices == set(range(8)), (
        f"layer 14 should pre-pin every head [0..7]; got {head_indices}"
    )
    assert l14.free_heads(14) == [], (
        "layer 14 has no free heads after alibi_slope seeding"
    )


# ---------------------------------------------------------------------------
# 2. Synthetic per-op declaration: new op asks for units, no pin
# ---------------------------------------------------------------------------


def test_synthetic_new_op_gets_free_unit_range_after_existing_tail():
    """A new op declaring ``(N units, no pin)`` against an annotated
    layer must land at ``ffn_widths[layer]`` — the next-free unit after
    the existing tail.

    This is the per-op migration contract: future ops drop their hand-
    picked ``start_unit, next_unit = ffn._lXX_unit_counter`` pattern in
    favour of ``allocator.alloc("op", n_units)`` and the wiring
    guarantees they don't clobber the legacy bakes.
    """
    layout = _build_layout_for_tests()
    ffn_allocs = getattr(layout, LAYOUT_FFN_UNIT_ALLOCATORS_ATTR)

    # Pick an annotated layer with a known tail. L6 has 2328 units of
    # function-call weights — the next-free unit must be 2328.
    layer_idx = 6
    existing_tail = layout.ffn_widths[layer_idx]
    start, end = ffn_allocs[layer_idx].alloc("synthetic_new_op", 16)
    assert start == existing_tail, (
        f"synthetic new op should land at the existing tail "
        f"({existing_tail}); got {start}"
    )
    assert end == existing_tail + 16


def test_synthetic_new_op_in_empty_layer_lands_at_zero():
    """A layer with no annotated FFN op (no ``ffn_widths`` entry) gets
    an empty allocator, so the first unpinned ``alloc`` starts at unit
    0 — matches the conventional "fresh FFN" semantics."""
    layout = _build_layout_for_tests()
    ffn_allocs = getattr(layout, LAYOUT_FFN_UNIT_ALLOCATORS_ATTR)

    empty_layer = None
    for layer_idx in range(layout.n_layers):
        if layout.ffn_widths.get(layer_idx, 0) == 0:
            empty_layer = layer_idx
            break
    if empty_layer is None:
        pytest.skip(
            "no FFN-empty layers in current layout; nothing to assert"
        )
    start, end = ffn_allocs[empty_layer].alloc("fresh_op", 32)
    assert (start, end) == (0, 32)


def test_synthetic_dim_alloc_fails_cleanly_when_pool_full():
    """The compact-IO layout packs ``d_model`` to exactly the byte
    coverage required, leaving zero free room. An unpinned ``alloc``
    against the seeded dim allocator must therefore fail cleanly with a
    "no free gap" error — confirming the wiring inherits the dim
    allocator's first-fit semantics rather than silently writing past
    d_model. The follow-up wave that grows d_model (or moves an op into
    a vacated gap) will exercise the success path instead.
    """
    layout = _build_layout_for_tests()
    dim_alloc = getattr(layout, LAYOUT_DIM_ALLOCATOR_ATTR)

    # Sanity: every byte in [0, d_model) is currently claimed by the
    # pin-every-existing-dim seed. If this ever stops being true the
    # branch below should be flipped to a positive "auto-place lands
    # in the empty gap" assertion.
    assert dim_alloc.free_pool() == [], (
        "compact-IO layout used to have zero free pool; if free room "
        "now exists the synthetic alloc should succeed instead — flip "
        f"the assertion. Got free_pool={dim_alloc.free_pool()}"
    )

    with pytest.raises(Exception) as excinfo:
        dim_alloc.alloc("__synthetic_new_dim__", 4)
    assert "no free gap" in str(excinfo.value), str(excinfo.value)


def test_synthetic_dim_alloc_lands_in_free_gap_on_widened_pool():
    """A standalone allocator seeded with a small subset of dims +
    extra headroom must place an auto-placed dim in the free gap. This
    confirms the unpinned ``alloc`` path works through the wiring
    contract — the compact-IO layout simply leaves no room for the
    full-layout variant to exercise success."""
    a = Allocator(d_model=64)
    a.alloc("EXISTING_A", 8, pin=0)
    a.alloc("EXISTING_B", 8, pin=16)
    # Auto-place a size-4 dim: first free gap is [8, 16).
    new = a.alloc("__synthetic_new_dim__", 4)
    assert new.start == 8
    assert new.pinned is False
    assert new.overlap is False


def test_synthetic_head_pin_picks_free_head_in_layer():
    """For a layer that DOESN'T have all 8 heads pre-pinned (e.g. an
    empty layer), an unpinned ``alloc`` returns the lowest free head
    (head 0 on an empty layer). Confirms the per-layer pool isolation
    so a head_idx claim in layer A never collides with the same index
    in layer B.
    """
    layout = _build_layout_for_tests()
    head_allocs = getattr(layout, LAYOUT_ATTENTION_HEAD_ALLOCATORS_ATTR)

    # Layer 14 is fully pinned; pick any layer that isn't.
    target = None
    for layer_idx in range(layout.n_layers):
        if len(head_allocs[layer_idx].heads()) == 0:
            target = layer_idx
            break
    if target is None:
        pytest.skip("no head-empty layer in current layout")
    chosen = head_allocs[target].alloc("__synthetic_head__", target)
    assert chosen == 0, (
        f"empty-layer allocator should pick head 0; got {chosen}"
    )


# ---------------------------------------------------------------------------
# 3. Byte-identity (slow): compare_compile_paths must still report 0
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_allocator_wiring_preserves_byte_identity():
    """The full byte-identity invariant from B11 must hold after
    attaching the three allocators. This is the canonical regression
    gate — any change to the wiring that perturbs the layout would
    show up as a non-zero ``n_diffs`` or a non-empty ``layout_diff``.
    """
    report = compare_compile_paths(
        alu_mode="lookup",
        disk_cache=False,
    )
    assert report["n_diffs"] == 0, (
        f"dynamic vs static state_dict differs in {report['n_diffs']} "
        f"tensors after allocator wiring; first 5: "
        f"{report['diff_keys'][:5]}"
    )
    assert report["layout_diff"] == [], (
        f"dynamic vs static ModelLayout differs after allocator wiring: "
        f"{report['layout_diff']}"
    )


@pytest.mark.slow
def test_allocators_present_after_full_compile():
    """End-to-end check: a real ``compile_full_vm_dynamic`` call returns
    a layout with all three allocators reachable — confirms the wiring
    survives the bake + cache paths exactly the same way the bare
    layout-only helper sees them.
    """
    _model, layout = compile_full_vm_dynamic(disk_cache=False)
    assert hasattr(layout, LAYOUT_DIM_ALLOCATOR_ATTR)
    assert hasattr(layout, LAYOUT_FFN_UNIT_ALLOCATORS_ATTR)
    assert hasattr(layout, LAYOUT_ATTENTION_HEAD_ALLOCATORS_ATTR)
    dim_alloc = getattr(layout, LAYOUT_DIM_ALLOCATOR_ATTR)
    assert dim_alloc.d_model == layout.d_model
    ffn_allocs = getattr(layout, LAYOUT_FFN_UNIT_ALLOCATORS_ATTR)
    head_allocs = getattr(layout, LAYOUT_ATTENTION_HEAD_ALLOCATORS_ATTR)
    assert set(ffn_allocs.keys()) == set(range(layout.n_layers))
    assert set(head_allocs.keys()) == set(range(layout.n_layers))
