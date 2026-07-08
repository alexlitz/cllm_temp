"""
S-8: Calibration loader for backbone contribution bounds.

Reads .agent-logs/backbone-bounds/v1.json produced by
tools/observe_backbone_contributions.py and exposes:

    bounds = BackboneBounds.load(path)
    bounds.max_positive_contribution(output_dim_name, offset, position_class) -> float
    bounds.max_negative_contribution(output_dim_name, offset, position_class) -> float

Used by S-6 verify_rule_strength: when a rule claims dominance at
(output_dim, position_class), the backbone's max-positive bound at
that (dim, class) becomes an additive term the rule must beat.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BackboneBounds:
    version: int
    corpus_size: int
    model_commit: str
    # bounds[key_str][position_class] = {"max_positive_contribution": float, ...}
    bounds: dict

    @classmethod
    def load(cls, path: str | Path) -> "BackboneBounds":
        with open(path) as f:
            data = json.load(f)
        return cls(
            version=data["version"],
            corpus_size=data["corpus_size"],
            model_commit=data["model_commit"],
            bounds=data["bounds"],
        )

    @classmethod
    def empty(cls) -> "BackboneBounds":
        """Empty bounds - every lookup returns 0 (no backbone constraint).
        Useful for testing the strength verifier without a corpus."""
        return cls(version=0, corpus_size=0, model_commit="", bounds={})

    def _key(self, output_dim: str, offset: int) -> str:
        return f"{output_dim}+{offset}"

    def max_positive_contribution(self, output_dim: str, offset: int, position_class: str) -> float:
        """Return the empirical max positive logit observed at
        (output_dim+offset, position_class) during corpus observation.
        Returns 0.0 if not present (conservative - no constraint)."""
        key = self._key(output_dim, offset)
        dim_data = self.bounds.get(key)
        if dim_data is None:
            warnings.warn(
                f"BackboneBounds: no entry for {key!r}; falling back to 0.0",
                stacklevel=2,
            )
            return 0.0
        pos_data = dim_data.get(position_class)
        if pos_data is None:
            return 0.0
        return float(pos_data.get("max_positive_contribution", 0.0))

    def max_negative_contribution(self, output_dim: str, offset: int, position_class: str) -> float:
        """Same as above but for max-negative (most-negative) contribution."""
        key = self._key(output_dim, offset)
        dim_data = self.bounds.get(key)
        if dim_data is None:
            return 0.0
        pos_data = dim_data.get(position_class)
        if pos_data is None:
            return 0.0
        return float(pos_data.get("max_negative_contribution", 0.0))

    def has_entry(self, output_dim: str, offset: int, position_class: str) -> bool:
        """Return True iff there's a concrete entry for this key."""
        key = self._key(output_dim, offset)
        dim_data = self.bounds.get(key)
        if dim_data is None:
            return False
        return position_class in dim_data

    def as_strength_bound(self, output_dim: str, offset: int, scope_str: str) -> float:
        """Adapter for S-6 verify_rule_strength backbone_bounds parameter.

        Maps a `scope_str` predicate to the closest matching position_class
        (or a conservative max across classes if no match). For V1 we look
        for an exact match on scope_str; if absent, return the MAX positive
        contribution across all classes for this dim (conservative - assumes
        backbone could fire at any class)."""
        key = self._key(output_dim, offset)
        dim_data = self.bounds.get(key)
        if dim_data is None:
            return 0.0
        if scope_str in dim_data:
            return float(dim_data[scope_str].get("max_positive_contribution", 0.0))
        # Fall back: conservative max across all classes
        return max(
            (float(d.get("max_positive_contribution", 0.0)) for d in dim_data.values()),
            default=0.0,
        )


def load_default_bounds() -> BackboneBounds:
    """Load .agent-logs/backbone-bounds/v1.json from the repo root.
    Falls back to empty bounds with a warning if the file doesn't exist."""
    # __file__ -> .../c4_release/neural_vm/verification/backbone_bounds.py
    # parents: [0]=verification [1]=neural_vm [2]=c4_release
    c4_root = Path(__file__).resolve().parent.parent.parent
    path = c4_root / ".agent-logs" / "backbone-bounds" / "v1.json"
    if not path.exists():
        warnings.warn(
            f"BackboneBounds: {path} not found; using empty (zero-constraint) bounds. "
            f"Run `python -m tools.observe_backbone_contributions --output {path}` to generate.",
            stacklevel=2,
        )
        return BackboneBounds.empty()
    return BackboneBounds.load(path)
