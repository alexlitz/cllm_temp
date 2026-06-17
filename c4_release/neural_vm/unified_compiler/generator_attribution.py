"""Generator-level attribution for diverging declarative rules.

When the faithful interpreter / oracle gate attributes a wrong byte to a single
owning ``FFNRule``, that rule may have been EMITTED by a higher-level ISA-DSL
generator (``cross_step_carry`` / ``consumer_lookahead_gate`` /
``full_width_byte_emission`` in :mod:`isa_semantics_dsl`). The generators name
every rule they emit with a recognizable ``{spec.name}_<role>`` suffix, so a
diverging rule can be lifted from "rule X" to "the GENERATOR that produced rule
X" — the honest, higher-altitude attribution the brief asks for.

This is a pure name-pattern classifier (no model state), so it is cheap and
safe to call on any attributed rule name. It deliberately does NOT guess for
roots that have no FFN writer (attention/relay/emission roots) or for the
imperative ALU composites (#230) — those return ``None`` and the caller reports
"attribution impossible (no declarative rule)" honestly.
"""

from __future__ import annotations

import re
from typing import Optional


# (generator_name, compiled suffix regex). The suffixes mirror the literal
# rule-name templates in ``isa_semantics_dsl.py`` (kept in sync by
# tests/test_isa_semantics_dsl.py owning the generators):
#
#   cross_step_carry           -> ``{name}_val{k}_lo_{nib}`` / ``{name}_val{k}_hi_{nib}``
#   full_width_byte_emission   -> ``{name}_head_token_{v}`` / ``{name}_fill_{v}``
#   consumer_lookahead_gate    -> ``{name}_consumer_{op}`` / ``{name}_dump_block_and``
#                                 / ``{name}_dump_block_passthrough``
_GENERATOR_SUFFIXES = (
    ("cross_step_carry", re.compile(r"_val\d+_(?:lo|hi)_\d+$")),
    ("full_width_byte_emission", re.compile(r"_head_token_\d+$")),
    ("full_width_byte_emission", re.compile(r"_fill_\d+$")),
    ("consumer_lookahead_gate", re.compile(r"_consumer_[a-z0-9_]+$")),
    ("consumer_lookahead_gate", re.compile(r"_dump_block_(?:and|passthrough)$")),
)


def generator_for_rule(rule_name: Optional[str]) -> Optional[str]:
    """Return the ISA-DSL generator that emitted ``rule_name``, or ``None``.

    ``None`` means the rule was NOT emitted by one of the three tracked
    generators (a hand-authored rule, a building-blocks-DSL rule, or there is no
    rule at all). The caller should then fall back to the rule name itself (or
    report attribution-impossible for attention/relay/ALU roots).
    """
    if not rule_name:
        return None
    for gen, rx in _GENERATOR_SUFFIXES:
        if rx.search(rule_name):
            return gen
    return None


def attribute_to_generator(
    rule_name: Optional[str], *, op_name: Optional[str] = None,
    is_alu_step: bool = False,
) -> str:
    """Human-readable generator-or-rule attribution string.

    * ``is_alu_step`` -> the diverging step is an imperative composite ALU block
      (#230); there is no declarative rule, so attribution is honestly coarse.
    * a rule name matching a tracked generator -> attribute to that GENERATOR.
    * a rule name not matching any generator -> attribute to the rule (the
      finest declarative grain available).
    * no rule name -> attribution impossible (an attention / relay / emission
      root with no FFN writer).
    """
    if is_alu_step:
        return ("imperative composite ALU block (#230) — no declarative rule "
                "to attribute (coarse block attribution only)")
    gen = generator_for_rule(rule_name)
    if gen is not None:
        return f"GENERATOR {gen} (emitted rule {rule_name!r})"
    if rule_name:
        return f"rule {rule_name!r} (hand-authored / building-blocks DSL)"
    return ("attribution impossible — no FFN writer (an attention / relay / "
            "emission root)")


__all__ = [
    "generator_for_rule",
    "attribute_to_generator",
]
