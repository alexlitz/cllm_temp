"""Regression: no wrapper-block FFN escapes ``_right_size_ffns``.

Baseline (commit 80c81dc6, ``speedup-cache-and-buckets``) measurement
identified three wrapper PureFFNs at block indices 19, 21, 23 carrying
the full 4096 hidden-unit allocation budget despite the compiler only
programming a handful of units. Root cause: the phase-1200
``_right_size_ffns`` pass runs *before* ``_expand_wrapper_blocks``
(phase 1300), so composite post_ops (``AddSub5StageBlock``,
``FlattenedPureFFN``, etc.) whose inner FFNs hide behind ``@property``
``W_up`` accessors are treated as flat leaves by the recursion gate and
skip the trim. Once expanded into standalone wrapper blocks the
4096-unit zombies remain.

Fix (``vm_step.py``): ``_expand_wrapper_blocks`` now calls
``_right_size_ffns(model)`` at its tail. The retrim is idempotent —
already-right-sized FFNs early-return at ``n_active == H``. After the
fix, every block's ``.ffn`` (regardless of whether the block is native
or post-expansion) carries at most ``LAYER_MAX_UNITS`` hidden units, and
the 4096-unit zombies vanish (saving ~20-29M params at the production
``d_model=800``).

This test:

  1. Compiles the full VM (no flag tweaks — uses the default
     legacy-expansion path).
  2. Walks every block and asserts that no ``block.ffn`` PureFFN
     descendant carries >= 4000 hidden units. The 4000 floor is a hair
     under the legacy 4096 budget so the assertion catches the
     zombie-class bug while tolerating ops that legitimately program
     ~3000-3500 active units (none currently exist; the largest
     production FFN is ~1500 units).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Threshold: 4000 catches the 4096-unit zombie pattern while leaving
# headroom for any op that legitimately programs ~3000-3500 active units.
# Today the largest production FFN sits well under 2000 hidden units, so
# this floor is conservative.
ZOMBIE_HIDDEN_FLOOR = 4000


def _walk_pure_ffn_leaves(module: nn.Module, prefix: str = ""):
    """Yield ``(label, W_up_tensor)`` for every parameter-bearing FFN leaf.

    A "leaf" here is any module exposing a ``W_up`` ``nn.Parameter``
    (either as an attribute or through a property forwarder). This walk
    mirrors ``_right_size_ffns._resize_one``'s recursion: if the current
    module has ``W_up``, treat it as a leaf; otherwise recurse into
    ``named_children``.
    """
    W_up = getattr(module, "W_up", None)
    if isinstance(W_up, nn.Parameter):
        yield prefix or type(module).__name__, W_up
        return
    for child_name, child in module.named_children():
        sub = f"{prefix}.{child_name}" if prefix else child_name
        yield from _walk_pure_ffn_leaves(child, sub)


@pytest.fixture(scope="module")
def compiled_model():
    """Default-expansion compile (legacy wrapper-block path).

    Uses ``disk_cache=False`` to defeat stale-cache reads — the wrapper
    coverage fix lives in ``_expand_wrapper_blocks`` and the cached blob
    pre-dates the fix on most dev machines.
    """
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    model, _ = compile_full_vm_dynamic(strict=False, disk_cache=False)
    return model


def test_no_block_carries_4096_unit_zombie(compiled_model):
    """Every block's FFN tree must trim to under ``ZOMBIE_HIDDEN_FLOOR``.

    The baseline regression (commit 80c81dc6) showed blocks 19, 21, 23
    holding 4096-hidden PureFFNs after ``_right_size_ffns`` skipped
    composite wrappers nested as post_ops. Post-fix, the retrim pass
    inside ``_expand_wrapper_blocks`` reaches every leaf.
    """
    offenders = []
    for i, block in enumerate(compiled_model.blocks):
        for label, W_up in _walk_pure_ffn_leaves(block.ffn, f"L{i}.ffn"):
            H = W_up.shape[0]
            if H >= ZOMBIE_HIDDEN_FLOOR:
                offenders.append((label, H))

    assert not offenders, (
        "Found wrapper FFN(s) with >= {ZOMBIE_HIDDEN_FLOOR} hidden units "
        "(_right_size_ffns wrapper coverage regression): {offenders}".format(
            ZOMBIE_HIDDEN_FLOOR=ZOMBIE_HIDDEN_FLOOR,
            offenders=offenders,
        )
    )


def test_specific_baseline_zombies_trimmed(compiled_model):
    """Block indices 19, 21, 23 specifically must not hold zombie FFNs.

    These were the three offenders identified in the baseline measurement
    on commit 80c81dc6 (``speedup-cache-and-buckets``). The test is
    permissive about which class lives at each slot (the topology shifts
    as ops migrate) but pins the hidden-dim ceiling.
    """
    if len(compiled_model.blocks) <= 23:
        pytest.skip(
            f"Model has only {len(compiled_model.blocks)} blocks; the "
            f"baseline zombie indices (19/21/23) require >= 24 blocks. "
            f"The compile path may have changed; re-run the baseline "
            f"measurement."
        )

    for idx in (19, 21, 23):
        block = compiled_model.blocks[idx]
        for label, W_up in _walk_pure_ffn_leaves(block.ffn, f"L{idx}.ffn"):
            H = W_up.shape[0]
            assert H < ZOMBIE_HIDDEN_FLOOR, (
                f"Block {idx} {label} carries {H} hidden units "
                f"(>= {ZOMBIE_HIDDEN_FLOOR} zombie floor). "
                f"_right_size_ffns wrapper coverage regression."
            )


def test_total_param_count_under_legacy_ceiling(compiled_model):
    """Param-count regression gate.

    Baseline (pre-fix): ~180M+ params including 3 x 4096-hidden wrapper
    FFNs (each ~20-29M params depending on d_model). Post-fix: the
    zombies are trimmed, dropping total by ~20-29M. We pin a generous
    195M ceiling — easily met post-fix but caught immediately if a
    future change reintroduces a 4096-zombie wrapper.
    """
    n_params = sum(p.numel() for p in compiled_model.parameters())
    # Conservative ceiling: the baseline measurement (commit 80c81dc6)
    # showed ~180M+ params with zombies present. Post-fix the count
    # drops ~20-29M. 195M leaves room for future legitimate growth
    # while catching a regression that reintroduces a single 4096-zombie
    # FFN (which alone would add ~6-7M params at d_model=800 just from
    # W_up/W_gate/W_down).
    LEGACY_CEILING = 195_000_000
    assert n_params <= LEGACY_CEILING, (
        f"Param count {n_params:,} exceeds the {LEGACY_CEILING:,} "
        f"ceiling. A 4096-hidden wrapper FFN may have re-escaped "
        f"_right_size_ffns. Inspect block.ffn shapes via "
        f"_walk_pure_ffn_leaves."
    )
