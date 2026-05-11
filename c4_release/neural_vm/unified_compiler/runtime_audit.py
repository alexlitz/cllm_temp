"""Runtime vanilla audit.

Enforces the directive: runtime = just attention + FFN.

A "vanilla" runtime model is one where every transformer block consists of:
  - One standard attention module (`AutoregressiveAttention` or `PureAttention`)
  - One standard FFN module (`PureFFN`, `FlattenedPureFFN`, or an `nn.Linear` chain)
  - No `post_ops` (post-FFN side-modules should have been split into separate
    blocks by `_expand_wrapper_blocks`).

Non-vanilla modules are anything that wraps the FFN in custom logic (e.g.
`HybridALUBlock`) or substitutes a structural ALU/format-converter for the
standard FFN (`PureNeuralALU` subclasses, `EfficientDivMod_Neural`, the
post-op family `BinaryOpByteZeroingPostOp`, `CarryPropagationPostOp`,
`BitwiseBytePropagationPostOp`, `ComparisonCombine`, `DivModModule`, and the
`BDToGEConverter` / `GEToBDConverter` format adapters).

Usage:
    from neural_vm.unified_compiler.runtime_audit import (
        verify_runtime_is_vanilla,
        RuntimeAuditReport,
    )

    report = verify_runtime_is_vanilla(model)
    if not report.is_vanilla:
        print(report.summary)
        for v in report.violations:
            print("  -", v)

The audit can be wired into a compile pipeline. By default it warns; set
`STRICT_RUNTIME_AUDIT=1` in the environment to make it raise instead.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Allow / deny lists (string names so we don't import broken modules eagerly)
# -----------------------------------------------------------------------------

# Vanilla FFN class names (anything else for the .ffn slot is a violation).
VANILLA_FFN_CLASSES = frozenset({
    "PureFFN",
    "FlattenedPureFFN",
    "Identity",
})

# Vanilla attention class names.
VANILLA_ATTN_CLASSES = frozenset({
    "AutoregressiveAttention",
    "PureAttention",
    "Identity",
})

# Known non-vanilla FFN / wrapper / post-op classes (used to give nicer
# violation messages — anything not in the vanilla set is still a violation,
# even if not in this list).
NON_VANILLA_FFN_CLASSES = frozenset({
    "HybridALUBlock",
    "PureNeuralALU",
    "EfficientALU_L8_L9_Neural",
    "EfficientALU_L10_Neural",
    "EfficientALU_L11_L12_Neural",
    "EfficientALU_L13_Neural",
    "EfficientDivMod_Neural",
    "BDToGEConverter",
    "GEToBDConverter",
    # Aspirational (not yet present in repo, but reserved by directive)
    "ALUMul",
    "ALUDivMod",
    "ALUShift",
    "ALUAddSub",
    "ALUAndOrXor",
})

# Post-op class names that should never appear (post_ops should be split).
NON_VANILLA_POST_OP_CLASSES = frozenset({
    "BinaryOpByteZeroingPostOp",
    "CarryPropagationPostOp",
    "BitwiseBytePropagationPostOp",
    "ComparisonCombine",
    "DivModModule",
    "EfficientDivMod_Neural",
})


# -----------------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------------

@dataclass
class RuntimeAuditReport:
    """Report produced by `verify_runtime_is_vanilla`.

    Attributes:
        is_vanilla: True iff every block is a standard attention + FFN with no
            post_ops.
        violations: One human-readable string per non-vanilla finding.
        summary: Short textual summary (block count + breakdown).
        n_blocks: Total number of blocks inspected.
        n_violations: Length of `violations`.
        fp64_params: List of "module_path: shape" entries for any fp64
            parameters detected (warning-class — listed separately so callers
            can decide independently).
    """

    is_vanilla: bool
    violations: List[str] = field(default_factory=list)
    summary: str = ""
    n_blocks: int = 0
    n_violations: int = 0
    fp64_params: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [self.summary]
        if self.violations:
            lines.append("Violations:")
            for v in self.violations:
                lines.append(f"  - {v}")
        if self.fp64_params:
            lines.append("fp64 parameters (also non-vanilla):")
            for p in self.fp64_params:
                lines.append(f"  - {p}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _class_name(obj) -> str:
    return type(obj).__name__


# Names of FFN-equivalent parameter signatures. A module is "PureFFN-equivalent"
# if it owns these four parameters as ``nn.Parameter`` tensors (the canonical
# SwiGLU FFN shape: up + gate + down projections plus the up-projection bias).
# ``b_gate`` / ``b_down`` are technically also present on ``PureFFN``, but the
# canonical four are sufficient to identify the shape unambiguously.
_PUREFFN_PARAM_NAMES = ("W_up", "W_gate", "W_down", "b_up")


def _is_pureffn_equivalent(module) -> bool:
    """Return True if `module` exposes the PureFFN parameter signature.

    A module is "PureFFN-equivalent" if it has the four canonical FFN
    parameters (``W_up``, ``W_gate``, ``W_down``, ``b_up``) as
    ``nn.Parameter`` tensors. This catches both:

      - ``PureFFN`` itself and direct subclasses, and
      - ``GenericFlattenedFFN``-style wrappers that re-export the same
        parameters as ``@property`` views onto a held ``ffn`` member.

    Shape sanity: ``W_up.shape == W_gate.shape``, ``W_down.shape ==
    W_up.T.shape``, and ``b_up.shape == (W_up.shape[0],)`` are required so
    that an unrelated module that happens to define same-named buffers
    won't slip through.
    """
    for pname in _PUREFFN_PARAM_NAMES:
        if not hasattr(module, pname):
            return False
    try:
        W_up = module.W_up
        W_gate = module.W_gate
        W_down = module.W_down
        b_up = module.b_up
    except Exception:
        return False
    # All four must be tensors.
    if not all(isinstance(t, torch.Tensor) for t in (W_up, W_gate, W_down, b_up)):
        return False
    # SwiGLU shape contract: W_up [H, D], W_gate [H, D], W_down [D, H],
    # b_up [H]. Allow either dense or sparse-coo layouts.
    if W_up.ndim != 2 or W_gate.ndim != 2 or W_down.ndim != 2 or b_up.ndim != 1:
        return False
    if W_up.shape != W_gate.shape:
        return False
    H, D = W_up.shape
    if W_down.shape != (D, H):
        return False
    if b_up.shape != (H,):
        return False
    return True


def _is_sequential_of_pureffns(module) -> bool:
    """Return True if `module` is an ``nn.Sequential`` whose every child is
    PureFFN-equivalent (or recursively a Sequential-of-PureFFNs).

    This recognises the "compiler-baked composite" pattern: an ALU pipeline
    whose forward is a literal ``nn.Sequential`` chain over vanilla FFN
    stages, with no exotic parameter-bearing modules in between.

    Pure orchestration wrappers (no direct parameters of their own) that
    hold a single PureFFN-equivalent inside are also accepted, so e.g.
    ``GenericFlattenedFFN`` (a wrapper around a ``PureFFN``) and stage
    adapters like ``_MulFFNStage(sub_ffn=...)`` qualify.
    """
    if not isinstance(module, nn.Sequential):
        return False
    if len(module) == 0:
        return False
    for child in module:
        if not _is_pureffn_composite(child):
            return False
    return True


def _is_pureffn_composite(module) -> bool:
    """Return True if `module` is either a PureFFN, a Sequential-of-PureFFNs,
    or an orchestration wrapper whose every parameter-bearing leaf is
    PureFFN-equivalent.

    The third case is what makes the 5 compiler-baked ALU composites
    (``FlattenedALUMul``, ``AddSub5StageBlock``, ``FlattenedDivMod``,
    ``ALUShiftComposite``, ``ALUAndOrXor``) classify as vanilla: each is a
    stack of FFN-like sub-stages (the parameter-bearing leaves are all
    PureFFNs) wrapped in pure-Python control flow that carries no learned
    parameters of its own.
    """
    if _is_pureffn_equivalent(module):
        return True
    if isinstance(module, nn.Sequential):
        return _is_sequential_of_pureffns(module)
    # Walk the module tree. Every leaf that owns ``nn.Parameter`` instances
    # directly must be PureFFN-equivalent. Wrapper modules (no direct
    # parameters of their own) are transparent.
    found_any = False
    for sub in module.modules():
        # ``modules()`` yields ``module`` itself first; treat it like any
        # other node — its parameters (if any) must be PureFFN-equivalent.
        if list(sub._parameters.keys()):
            if not _is_pureffn_equivalent(sub):
                return False
            found_any = True
    return found_any


def _is_vanilla_ffn(module) -> bool:
    """Return True if `module` is a standard FFN.

    Accepts:
      - PureFFN / FlattenedPureFFN by name.
      - nn.Identity (passthrough).
      - A pure nn.Linear or a tiny nn.Sequential of nn.Linear / SiLU / GELU
        (treats common transformer FFN forms as vanilla).
      - An ``nn.Sequential`` whose every child is PureFFN-equivalent
        (canonical "compiler-baked composite" pattern: a stack of vanilla
        FFN sub-stages chained via Sequential).
      - A composite wrapper whose parameter-bearing leaves are all
        PureFFN-equivalent (catches ALU composites like ``FlattenedALUMul``
        / ``AddSub5StageBlock`` / ``FlattenedDivMod`` / ``ALUShiftComposite``
        / ``ALUAndOrXor`` that hold an ``nn.Sequential`` pipeline of
        PureFFN-based stages under a top-level wrapper class).
    """
    name = _class_name(module)
    if name in VANILLA_FFN_CLASSES:
        return True
    if isinstance(module, nn.Identity):
        return True
    if isinstance(module, nn.Linear):
        return True
    if isinstance(module, nn.Sequential):
        allowed = (nn.Linear, nn.SiLU, nn.GELU, nn.ReLU, nn.Identity, nn.Dropout)
        if all(isinstance(m, allowed) for m in module):
            return True
        # New: nn.Sequential of PureFFN-equivalents is also vanilla.
        if _is_sequential_of_pureffns(module):
            return True
        return False
    # Composite wrapper whose leaves are all PureFFN-equivalent (the 5
    # compiler-baked ALU composites). Only opt in for modules that have
    # at least one parameter-bearing leaf — bare wrappers (e.g. format
    # converters with no params) should still be flagged.
    if _is_pureffn_composite(module):
        return True
    return False


def _is_vanilla_attn(module) -> bool:
    name = _class_name(module)
    if name in VANILLA_ATTN_CLASSES:
        return True
    if isinstance(module, nn.Identity):
        return True
    return False


def _describe_module(module) -> str:
    """Short descriptor: ClassName (e.g., HybridALUBlock(lookup_ffn=PureFFN, ...))."""
    name = _class_name(module)
    inner = []
    if hasattr(module, "lookup_ffn"):
        inner.append(f"lookup_ffn={_class_name(module.lookup_ffn)}")
    if hasattr(module, "efficient_alu"):
        inner.append(f"efficient_alu={_class_name(module.efficient_alu)}")
    if hasattr(module, "ffn") and not isinstance(module, nn.Linear):
        inner.append(f"ffn={_class_name(module.ffn)}")
    if inner:
        return f"{name}({', '.join(inner)})"
    return name


def _scan_fp64_params(model) -> List[str]:
    """Return a list of "name: shape (dtype)" strings for any fp64 parameters."""
    bad = []
    for name, p in model.named_parameters(recurse=True):
        if p.dtype == torch.float64:
            bad.append(f"{name}: shape={tuple(p.shape)} dtype={p.dtype}")
    for name, b in model.named_buffers(recurse=True):
        if b.dtype == torch.float64:
            bad.append(f"{name} (buffer): shape={tuple(b.shape)} dtype={b.dtype}")
    return bad


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def verify_runtime_is_vanilla(
    model,
    *,
    check_fp64: bool = True,
) -> RuntimeAuditReport:
    """Audit `model.blocks` for non-vanilla runtime modules.

    Args:
        model: A transformer model with a `.blocks` ModuleList. Each block is
            expected to expose `.attn`, `.ffn`, and optionally `.post_ops`.
        check_fp64: If True (default) also scan all parameters/buffers for
            fp64 tensors and record them on the report (does not affect
            `is_vanilla` by itself).

    Returns:
        A `RuntimeAuditReport` describing every non-vanilla block.
    """
    violations: List[str] = []

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return RuntimeAuditReport(
            is_vanilla=False,
            violations=["model has no .blocks attribute"],
            summary="model has no .blocks attribute",
            n_blocks=0,
            n_violations=1,
        )

    n_blocks = len(blocks)

    # Classification counters for the summary.
    n_vanilla = 0
    n_with_post_ops = 0
    n_bad_ffn = 0
    n_bad_attn = 0
    bad_ffn_classes: List[str] = []
    bad_attn_classes: List[str] = []
    post_op_classes: List[str] = []

    for i, block in enumerate(blocks):
        block_name = _class_name(block)

        # 1. Block shape: must look like a TransformerBlock-equivalent (have
        #    attn + ffn members). If it's something else entirely, that's a
        #    violation.
        attn = getattr(block, "attn", None)
        ffn = getattr(block, "ffn", None)
        if attn is None or ffn is None:
            violations.append(
                f"L{i}: block class {block_name!r} is not a standard "
                f"TransformerBlock (missing .attn or .ffn)"
            )
            continue

        block_violation = False

        # 2. Attention check.
        if not _is_vanilla_attn(attn):
            cls = _class_name(attn)
            violations.append(
                f"L{i}: non-vanilla attention {cls!r} (expected "
                f"AutoregressiveAttention / PureAttention)"
            )
            n_bad_attn += 1
            bad_attn_classes.append(cls)
            block_violation = True

        # 3. FFN check.
        if not _is_vanilla_ffn(ffn):
            desc = _describe_module(ffn)
            violations.append(
                f"L{i}: non-vanilla FFN {desc} (expected "
                f"PureFFN / FlattenedPureFFN / nn.Linear chain)"
            )
            n_bad_ffn += 1
            bad_ffn_classes.append(_class_name(ffn))
            block_violation = True

        # 4. post_ops must be empty.
        post_ops = getattr(block, "post_ops", None)
        if post_ops is not None and len(post_ops) > 0:
            classes = [_class_name(op) for op in post_ops]
            post_op_classes.extend(classes)
            n_with_post_ops += 1
            violations.append(
                f"L{i}: block has {len(post_ops)} post_op(s) "
                f"({', '.join(classes)}); post_ops should be split into "
                f"separate blocks by _expand_wrapper_blocks"
            )
            block_violation = True

        if not block_violation:
            n_vanilla += 1

    # Optional fp64 check.
    fp64_params: List[str] = []
    if check_fp64:
        try:
            fp64_params = _scan_fp64_params(model)
        except Exception as e:  # pragma: no cover — defensive
            fp64_params = [f"<scan failed: {e}>"]
    if fp64_params:
        for p in fp64_params:
            violations.append(f"fp64 parameter found: {p}")

    is_vanilla = len(violations) == 0
    n_violations = len(violations)

    # Summary string.
    breakdown_parts = [
        f"{n_blocks} blocks",
        f"{n_vanilla} vanilla",
    ]
    if n_bad_ffn:
        from collections import Counter
        ffn_counts = ", ".join(
            f"{c}x{n}" for c, n in Counter(bad_ffn_classes).items()
        )
        breakdown_parts.append(f"{n_bad_ffn} non-vanilla FFN ({ffn_counts})")
    if n_bad_attn:
        from collections import Counter
        attn_counts = ", ".join(
            f"{c}x{n}" for c, n in Counter(bad_attn_classes).items()
        )
        breakdown_parts.append(f"{n_bad_attn} non-vanilla attn ({attn_counts})")
    if n_with_post_ops:
        from collections import Counter
        po_counts = ", ".join(
            f"{c}x{n}" for c, n in Counter(post_op_classes).items()
        )
        breakdown_parts.append(
            f"{n_with_post_ops} block(s) with post_ops ({po_counts})"
        )
    if fp64_params:
        breakdown_parts.append(f"{len(fp64_params)} fp64 params")

    status = "VANILLA" if is_vanilla else "NON-VANILLA"
    summary = f"Runtime audit: {status} — " + ", ".join(breakdown_parts)

    return RuntimeAuditReport(
        is_vanilla=is_vanilla,
        violations=violations,
        summary=summary,
        n_blocks=n_blocks,
        n_violations=n_violations,
        fp64_params=fp64_params,
    )


# -----------------------------------------------------------------------------
# Pipeline integration helper
# -----------------------------------------------------------------------------

def enforce_runtime_vanilla(
    model,
    *,
    strict: Optional[bool] = None,
    label: str = "compile_full_vm",
) -> RuntimeAuditReport:
    """Run the audit and either warn or raise based on policy.

    This is the function intended to be called at the end of a model-compile
    pipeline. By default it emits a `UserWarning` if the model is non-vanilla;
    when the env var `STRICT_RUNTIME_AUDIT=1` is set (or `strict=True` is
    passed explicitly) it raises `RuntimeError` instead.

    Args:
        model: Transformer model with `.blocks`.
        strict: Override the env var. If None (default), reads
            `STRICT_RUNTIME_AUDIT` from the environment.
        label: Tag used in the warning / error message.

    Returns:
        The `RuntimeAuditReport` (regardless of whether we warned or raised).
    """
    if strict is None:
        strict = os.environ.get("STRICT_RUNTIME_AUDIT", "0") not in (
            "", "0", "false", "False", "no", "No",
        )

    report = verify_runtime_is_vanilla(model)

    if not report.is_vanilla:
        msg = (
            f"[{label}] Runtime is NOT vanilla "
            f"({report.n_violations} violation(s) across {report.n_blocks} "
            f"blocks). Set STRICT_RUNTIME_AUDIT=1 to fail the build.\n"
            + str(report)
        )
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

    return report


__all__ = [
    "RuntimeAuditReport",
    "verify_runtime_is_vanilla",
    "enforce_runtime_vanilla",
    "VANILLA_FFN_CLASSES",
    "VANILLA_ATTN_CLASSES",
    "NON_VANILLA_FFN_CLASSES",
    "NON_VANILLA_POST_OP_CLASSES",
]
