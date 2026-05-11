"""Regression test: layer_idx mispinning guard for migrated layerN_* ops.

History — the "L1 threshold mispinning" bug:
    Migrated attn/ffn ops named ``layerN_<...>`` were originally placed by
    the LayerCompiler's dep-graph alone. Several ops (notably L1's threshold
    attention) had reads/writes whose producers landed at the wrong block,
    so the dep-graph happily put the L1 threshold attn at a block other
    than block 1 — silently corrupting the bake. D2 found it by bisect;
    D4 and D1 independently confirmed.

    The fix: every migrated ``layerN_<...>`` attn/ffn op now carries an
    explicit ``layer_idx=N`` which pins the placement; ``_assign_layers``
    in ``layer_compiler.py`` enforces the pin and raises if the dep-graph
    would otherwise contradict it.

What this test locks in:
    1. Every Operation in ``all_core_ops()`` whose name matches
       ``layer{N}_*`` AND has ``migrated=True`` AND is of bake-kind
       (``attn``/``ffn``/``block``) MUST set ``layer_idx == N``.
       (``kind="model"`` ops are intentionally exempt: they operate on the
       whole model and never use ``layer_idx``.)
    2. After ``compile_full_vm()`` produces a layout, every migrated
       ``layerN_*`` attn/ffn op must actually appear at
       ``layout.ops_per_layer[N]`` — i.e. the pin was honored.
    3. After ``compile_full_vm()``, every migrated ``layerN_*`` block op
       must report ``layer_idx == N`` (block ops are dispatched by
       ``compile_full_vm`` via ``layout.resolve_block_op_layer`` /
       ``layer_idx`` directly).

If a future agent flips a new ``layerN_*`` op to ``migrated=True`` and
forgets ``layer_idx=N``, this test fails fast — long before any neural-VM
smoke test would notice the silent bake corruption.
"""

import re

import pytest

from c4_release.neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
from c4_release.neural_vm.unified_compiler.full_vm_compiler import compile_full_vm


# Matches an op name of the form ``layer{N}_<anything>`` and captures N.
# Names use lowercase ``layer`` (per convention in
# c4_release/neural_vm/unified_compiler/ops/l*_ops.py); short-form ``l{N}_``
# names (e.g. ``l13_alu_shift_*``) are intentionally not matched — they do
# not carry the ``layerN`` semantic contract.
_LAYER_NAME_RE = re.compile(r"^layer(\d+)_")

# kinds whose placement is governed by ``layer_idx``. ``kind="model"`` ops
# bake into the whole model and never use ``layer_idx``, so they are exempt.
_LAYER_PINNED_KINDS = frozenset({"attn", "ffn", "block"})


def _layer_idx_from_name(name: str):
    m = _LAYER_NAME_RE.match(name)
    return int(m.group(1)) if m else None


def test_migrated_layerN_ops_have_matching_layer_idx():
    """Every migrated ``layerN_*`` bake-kind op must set ``layer_idx == N``.

    This is the direct lock-in for the L1 threshold mispinning bug.
    """
    ops = all_core_ops()

    failures = []
    checked = 0
    for op in ops:
        n = _layer_idx_from_name(op.name)
        if n is None:
            continue
        if not op.migrated:
            continue
        if op.kind not in _LAYER_PINNED_KINDS:
            # kind="model" layerN_* ops (e.g. layer6_attn_bake) operate on
            # the whole model; layer_idx is not meaningful for them.
            continue

        checked += 1
        if op.layer_idx is None:
            failures.append(
                f"{op.name!r} (kind={op.kind!r}, migrated=True) has no "
                f"layer_idx; expected layer_idx={n}. This is the exact "
                f"shape of the L1 threshold mispinning bug — set "
                f"layer_idx={n} on the Operation."
            )
        elif op.layer_idx != n:
            failures.append(
                f"{op.name!r} (kind={op.kind!r}, migrated=True) has "
                f"layer_idx={op.layer_idx}; expected layer_idx={n} from "
                f"its name. Fix the name or the layer_idx so they agree."
            )

    assert checked > 0, (
        "Sanity check: no migrated layerN_* bake-kind ops found in "
        "all_core_ops(). Did the op naming convention change?"
    )
    assert not failures, (
        "Migrated layerN_* ops with inconsistent layer_idx "
        f"({len(failures)} of {checked} checked):\n  "
        + "\n  ".join(failures)
    )


def test_layout_ops_per_layer_pins_migrated_attn_ffn_to_named_layer():
    """After compile, every migrated ``layerN_*`` attn/ffn op must land at
    ``ops_per_layer[N]``.

    This is the runtime confirmation of the placement contract: even if
    ``layer_idx`` were set correctly on the Operation, a regression in
    ``_assign_layers`` could still send it to the wrong block. Walking
    ``ops_per_layer`` after compile catches that.
    """
    _, layout = compile_full_vm()

    failures = []
    # Build {name: layer_index} for every op the compiler placed in
    # ops_per_layer (attn/ffn only — block ops live in layout.block_ops).
    placement = {}
    for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer):
        for op in ops_at_layer:
            placement[op.name] = (layer_idx, op)

    # For every migrated layerN_* attn/ffn op in all_core_ops(), assert it
    # landed at ops_per_layer[N].
    checked = 0
    for op in all_core_ops():
        n = _layer_idx_from_name(op.name)
        if n is None:
            continue
        if not op.migrated:
            continue
        if op.kind not in ("attn", "ffn"):
            continue
        if op.name not in placement:
            failures.append(
                f"{op.name!r} (migrated attn/ffn, expected at layer {n}) "
                f"was not placed in layout.ops_per_layer at all."
            )
            continue
        actual_layer, _placed_op = placement[op.name]
        checked += 1
        if actual_layer != n:
            failures.append(
                f"{op.name!r} (migrated, expected at layer {n}) was "
                f"placed at layer {actual_layer} by the compiler. "
                f"layer_idx pinning broke."
            )

    assert checked > 0, (
        "Sanity check: no migrated layerN_* attn/ffn ops placed in "
        "ops_per_layer. Did the compile path change?"
    )
    assert not failures, (
        "Migrated layerN_* attn/ffn ops mispinned in layout.ops_per_layer:"
        "\n  " + "\n  ".join(failures)
    )


def test_layout_block_ops_carry_matching_layer_idx():
    """Block ops carry their own ``layer_idx`` (not via ops_per_layer)."""
    _, layout = compile_full_vm()

    block_ops_by_name = {op.name: op for op in layout.block_ops}

    failures = []
    checked = 0
    for op in all_core_ops():
        n = _layer_idx_from_name(op.name)
        if n is None:
            continue
        if not op.migrated:
            continue
        if op.kind != "block":
            continue
        # Block ops with target_op_name use op-reference binding and may
        # legitimately have layer_idx=None or a value that's overridden by
        # resolve_block_op_layer. We still check the layer_idx field if it's
        # set, since the convention for layerN_* names is that layer_idx=N.
        placed = block_ops_by_name.get(op.name)
        assert placed is not None, (
            f"Block op {op.name!r} not found in layout.block_ops after "
            f"compile (something is very wrong with op registration)."
        )
        checked += 1
        if placed.target_op_name is not None:
            # Op-reference binding overrides layer_idx; skip the layer_idx
            # consistency check for these (the name might still match a
            # different layer than where the referenced op landed).
            continue
        if placed.layer_idx != n:
            failures.append(
                f"{op.name!r} (migrated block op) has layer_idx="
                f"{placed.layer_idx}; expected layer_idx={n} from its name."
            )

    assert checked > 0, (
        "Sanity check: no migrated layerN_* block ops found in "
        "layout.block_ops. Did the compile path change?"
    )
    assert not failures, (
        "Migrated layerN_* block ops with inconsistent layer_idx:\n  "
        + "\n  ".join(failures)
    )
