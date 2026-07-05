"""Test-only imperative-signature adapters for the L6 attention bakes.

The imperative ``_bake_layer6_attn_spec`` / ``_bake_layer6_relay_heads_spec``
writers were migrated to declarative
:func:`neural_vm.unified_compiler.ops.l6_ops._layer6_attn_head_specs` /
``_layer6_relay_head_specs`` (commits ``8458f14b`` / ``d2516b83`` and
follow-ups). The declarative specs are the golden production truth (gated by
``tools/_isa_golden_hash.py``); the old imperative helper names are gone.

Several parity tests in ``test_declarative_ffn_bakes_l6.py`` and
``test_declarative_attention_specs.py`` still call the imperative helper by its
old ``(attn, BD, HD)`` signature and assert the per-cell weight layout it wrote.
These thin adapters keep that call surface alive **for tests only** by lowering
the current declarative production spec through
``Primitives.generate_attention_heads`` (the exact code path the production
``make_layer6_attn_bake_op`` / ``make_layer6_relay_heads_bake_op`` use). The
weights they produce are therefore byte-identical to golden by construction.

Two production-fidelity notes preserved here (so the fixture matches the real
bake, not just the isolated spec):

* Head 5's first-step FETCH-relay K-scale 10x bump is folded into the declarative
  head-5 spec (slot-0 K[MARK_PC] = 500.0). The old imperative helper did NOT fold
  it; the production ``make_layer6_attn_bake_op`` applied it as a post-helper row
  multiply. Callers reproducing the legacy path (the byte-identity tests) must
  therefore NOT apply the ``*= 10.0`` a second time to the generated side.
* The head-5 OPCODE_BYTE_HI k=13/14/15 spillover (legacy flat indexing crossed
  into head-6 slots 0/1/2) is declared on head 6 inside
  ``_layer6_relay_head_specs``, so the attn adapter alone omits those three
  cells; baking the relay adapter restores them at the same (row, col).

NOTHING here is imported by the compiler; it lives entirely under ``tests/``.
"""

from neural_vm.unified_compiler.primitives import Primitives
from neural_vm.unified_compiler.ops.l6_ops import (
    _layer6_attn_head_specs,
    _layer6_relay_head_specs,
)


def bake_layer6_attn_spec(attn, BD, HD):
    """Imperative-signature adapter for L6 heads 0, 1, 2, 3, 5.

    Lowers ``_layer6_attn_head_specs(BD)`` (the declarative production spec)
    into ``attn`` via the same primitive the production bake uses. The head-5
    K-scale 10x bump is already folded into the spec, so unlike the removed
    imperative helper this adapter needs no post-multiply.
    """
    Primitives.generate_attention_heads(attn, _layer6_attn_head_specs(BD), HD)


def bake_layer6_relay_heads_spec(attn, BD, HD):
    """Imperative-signature adapter for L6 PSH relay heads 6, 7.

    Lowers ``_layer6_relay_head_specs(BD)`` (the declarative production spec)
    into ``attn``. Includes the head-5 OPCODE_BYTE_HI k=13/14/15 spillover cells
    (declared on head 6) and the post-LEV AX_CARRY refresh sub-pattern.
    """
    Primitives.generate_attention_heads(attn, _layer6_relay_head_specs(BD), HD)
