#!/usr/bin/env python3
"""Auto-attribute a 1096 neural-vs-symbolic failure to a specific op.

Step 6 of ``docs/IR_INCREMENTAL_IMPROVEMENTS.md``: given a single failing
1096 test, this tool runs both the neural model and the symbolic
declarative interpreter, finds the first emitted token where they diverge,
walks the per-layer residual stream at the AX (or relevant slot) marker
position, and identifies the FIRST op whose declared ``produces`` claim
does not match the runtime residual. A one-page Markdown fix brief is
emitted under ``c4_release/.agent-logs/``.

Usage:

    # Numeric index into ``generate_test_programs()``:
    python tools/attribute_1096_failure.py --test_id 0

    # Substring match against the test description (e.g. "add_42"):
    python tools/attribute_1096_failure.py --test_id add_42

The neural VM is compiled ONCE per process via the
``BatchedPureNeuralRunner`` session fixture path (warm in-process and
on-disk caches per ``compile_full_vm_dynamic`` memoisation). Constraints
applied:

  * ``alu_mode='lookup'`` (the production default)
  * spec_k=0, kv_cache off (so per-step neural output is deterministic)
  * declarations_only bake on by default
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make ``import neural_vm`` work whether the tool is run from
# c4_release/tools or from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


def _select_test(test_id: str) -> Tuple[int, str, int, str]:
    """Look up one entry from ``generate_test_programs()`` by index or label.

    Accepts a bare integer (``"0"``, ``"42"``) or a substring that the test
    description contains (``"add_0"``, ``"add_42"``). Returns
    ``(idx, source, expected, description)``. Raises ``ValueError`` if no
    test matches.
    """
    from tests.test_suite_1000 import generate_test_programs

    tests = generate_test_programs()
    raw = test_id.strip()

    try:
        idx = int(raw)
    except ValueError:
        idx = None

    if idx is not None:
        if not (0 <= idx < len(tests)):
            raise ValueError(
                f"--test_id {raw!r} out of range (have {len(tests)} tests)"
            )
        source, expected, description = tests[idx]
        return idx, source, expected, description

    # Substring search across description.
    matches = [
        (i, s, e, d)
        for i, (s, e, d) in enumerate(tests)
        if raw in d
    ]
    if not matches:
        raise ValueError(
            f"--test_id {raw!r} matched no description; "
            f"first few are: {[t[2] for t in tests[:5]]!r}"
        )
    return matches[0]


@dataclass
class Attribution:
    """Final attribution payload, including any first-op-mismatch findings."""

    test_idx: int
    description: str
    suite_expected: int
    declarative_exit: Optional[int]
    declarative_steps: Optional[int]
    neural_exit: Optional[int]
    divergence_step: Optional[int] = None
    divergence_slot: Optional[str] = None
    expected_token: Optional[int] = None
    neural_token: Optional[int] = None
    suspect_dims: Tuple[str, ...] = ()
    first_mismatch_op: Optional[str] = None
    first_mismatch_layer: Optional[int] = None
    first_mismatch_file: Optional[str] = None
    first_mismatch_line: Optional[int] = None
    first_mismatch_reason: Optional[str] = None
    candidate_ops: List[Tuple[str, str]] = None  # (op_name, file:line)
    notes: List[str] = None

    def __post_init__(self):
        if self.candidate_ops is None:
            self.candidate_ops = []
        if self.notes is None:
            self.notes = []


# ---------------------------------------------------------------------------
# Symbolic + neural execution
# ---------------------------------------------------------------------------


def _run_symbolic(bytecode, data):
    """Run the SymbolicDeclarativeProgramRunner; return ``(state, expected_tokens, prefix_len)``."""
    from tests.test_1096_neural_declarative_diagnostic import (
        _build_symbolic_expected_execution,
    )

    expected = _build_symbolic_expected_execution(bytecode, data)
    return expected


def _capture_neural_context(runner, bytecode, data, expected_steps):
    """Capture the neural model's emitted token stream for one program."""
    from tests.test_1096_neural_declarative_diagnostic import (
        _capture_single_neural_context,
    )

    return _capture_single_neural_context(
        runner,
        bytecode,
        data,
        expected_steps=expected_steps,
        spec_k=0,
        max_context_window=512,
    )


# ---------------------------------------------------------------------------
# Slot/register → dim mapping
# ---------------------------------------------------------------------------


# Slot families → likely residual dims that drive the head logits for that
# slot. The mapping covers the per-step token slots emitted by the VM
# (see ``_STEP_SLOT_NAMES`` in tests/test_1096_neural_declarative_diagnostic.py).
# When the divergence lands on a byte slot, we look at OUTPUT_LO/OUTPUT_HI
# (the head nibble bands) plus any register-specific carry dims.
_SLOT_TO_SUSPECT_DIMS: Dict[str, Tuple[str, ...]] = {
    # Pure byte emissions: head reads OUTPUT_LO/OUTPUT_HI for the byte value,
    # but the upstream producer is usually a register-specific carry.
    "AX_byte0": ("OUTPUT_LO", "OUTPUT_HI", "AX_CARRY_LO", "AX_CARRY_HI"),
    "AX_byte1": ("OUTPUT_LO", "OUTPUT_HI", "AX_CARRY_LO", "AX_CARRY_HI"),
    "AX_byte2": ("OUTPUT_LO", "OUTPUT_HI", "AX_CARRY_LO", "AX_CARRY_HI"),
    "AX_byte3": ("OUTPUT_LO", "OUTPUT_HI", "AX_CARRY_LO", "AX_CARRY_HI"),
    "PC_byte0": ("OUTPUT_LO", "OUTPUT_HI", "PC_CARRY_LO", "PC_CARRY_HI"),
    "PC_byte1": ("OUTPUT_LO", "OUTPUT_HI", "PC_CARRY_LO", "PC_CARRY_HI"),
    "PC_byte2": ("OUTPUT_LO", "OUTPUT_HI", "PC_CARRY_LO", "PC_CARRY_HI"),
    "PC_byte3": ("OUTPUT_LO", "OUTPUT_HI", "PC_CARRY_LO", "PC_CARRY_HI"),
    "SP_byte0": ("OUTPUT_LO", "OUTPUT_HI", "SP_CARRY_LO", "SP_CARRY_HI"),
    "SP_byte1": ("OUTPUT_LO", "OUTPUT_HI", "SP_CARRY_LO", "SP_CARRY_HI"),
    "SP_byte2": ("OUTPUT_LO", "OUTPUT_HI", "SP_CARRY_LO", "SP_CARRY_HI"),
    "SP_byte3": ("OUTPUT_LO", "OUTPUT_HI", "SP_CARRY_LO", "SP_CARRY_HI"),
    "BP_byte0": ("OUTPUT_LO", "OUTPUT_HI", "BP_CARRY_LO", "BP_CARRY_HI"),
    "BP_byte1": ("OUTPUT_LO", "OUTPUT_HI", "BP_CARRY_LO", "BP_CARRY_HI"),
    "BP_byte2": ("OUTPUT_LO", "OUTPUT_HI", "BP_CARRY_LO", "BP_CARRY_HI"),
    "BP_byte3": ("OUTPUT_LO", "OUTPUT_HI", "BP_CARRY_LO", "BP_CARRY_HI"),
    "STACK0_byte0": ("OUTPUT_LO", "OUTPUT_HI"),
    "STACK0_byte1": ("OUTPUT_LO", "OUTPUT_HI"),
    "STACK0_byte2": ("OUTPUT_LO", "OUTPUT_HI"),
    "STACK0_byte3": ("OUTPUT_LO", "OUTPUT_HI"),
    "MEM_addr0": ("OUTPUT_LO", "OUTPUT_HI"),
    "MEM_addr1": ("OUTPUT_LO", "OUTPUT_HI"),
    "MEM_addr2": ("OUTPUT_LO", "OUTPUT_HI"),
    "MEM_addr3": ("OUTPUT_LO", "OUTPUT_HI"),
    "MEM_value0": ("OUTPUT_LO", "OUTPUT_HI"),
    "MEM_value1": ("OUTPUT_LO", "OUTPUT_HI"),
    "MEM_value2": ("OUTPUT_LO", "OUTPUT_HI"),
    "MEM_value3": ("OUTPUT_LO", "OUTPUT_HI"),
    # Marker tokens: head reads a NEXT_* flag.
    "REG_PC": ("NEXT_PC",),
    "REG_AX": ("NEXT_AX",),
    "REG_SP": ("NEXT_SP",),
    "REG_BP": ("NEXT_BP",),
    "STACK0": ("NEXT_STACK0",),
    "MEM": ("NEXT_MEM",),
    "STEP_END": ("NEXT_SE", "NEXT_HALT"),
}


def _suspect_dims_for_slot(slot: str) -> Tuple[str, ...]:
    return _SLOT_TO_SUSPECT_DIMS.get(slot, ("OUTPUT_LO", "OUTPUT_HI"))


# ---------------------------------------------------------------------------
# Op file:line resolution
# ---------------------------------------------------------------------------


def _op_source_location(op) -> Optional[Tuple[str, int]]:
    """Best-effort ``(file, line)`` for an Operation's bake site.

    Resolution order:
      1. ``declarative_bake_fn`` (canonical for declarations-only mode).
      2. ``bake_fn`` (imperative bake).
      3. ``compiler_ir_factory`` (late-bound IR producers).
    """
    for attr in ("declarative_bake_fn", "bake_fn", "compiler_ir_factory"):
        fn = getattr(op, attr, None)
        if fn is None:
            continue
        try:
            src = inspect.getsourcefile(fn)
            _, lineno = inspect.getsourcelines(fn)
        except (TypeError, OSError):
            continue
        if src is None:
            continue
        return (src, lineno)
    return None


# ---------------------------------------------------------------------------
# Layer residual probe at the AX (or relevant) marker
# ---------------------------------------------------------------------------


def _probe_layer_residuals(
    model,
    expected,
    divergence,
    max_context_window: int = 512,
):
    """Forward the teacher-forced prefix and snapshot residuals per block.

    Returns ``(per_block_residuals, logit_pos)`` where each entry is a
    detached tensor of shape ``[1, T, D]`` for blocks 0..N. Index 0 holds
    the embedding output (pre-block 0).
    """
    import torch
    from tests.test_1096_neural_declarative_diagnostic import (
        _window_expected_prefix_for_prediction,
    )

    prefix, logit_pos = _window_expected_prefix_for_prediction(
        expected,
        token_index=divergence.token_index,
        max_context_window=max_context_window,
    )
    if not prefix:
        return None, None

    device = next(model.parameters()).device
    token_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    model.embed.set_mem_history_end(0)

    residuals = []
    with torch.no_grad():
        x = model.embed(token_ids)
        residuals.append(x.detach().clone())
        for block in model.blocks:
            x = block(x)
            residuals.append(x.detach().clone())
    return residuals, logit_pos


def _residual_slice(residual, layout, dim_name: str, pos: int):
    """Return the residual values for ``dim_name`` at token position ``pos``."""
    import torch

    if dim_name not in layout.dim_positions:
        return None
    start = int(layout.dim_positions[dim_name])
    size = int(layout.dim_sizes.get(dim_name, 1))
    return residual[0, pos, start : start + size]


# ---------------------------------------------------------------------------
# Walk ops topologically, find first mismatch
# ---------------------------------------------------------------------------


def _ops_by_layer(layout) -> List[Tuple[int, Any]]:
    """Return ``[(layer_idx, op)]`` in topological execution order."""
    out: List[Tuple[int, Any]] = []
    for layer_idx, ops_at in enumerate(layout.ops_per_layer):
        for op in ops_at:
            out.append((layer_idx, op))
    # Block ops fire after attn/ffn at their target layer; resolve their
    # layer_idx via layout when possible.
    for op in layout.block_ops:
        layer_idx = getattr(op, "layer_idx", None)
        if layer_idx is None:
            try:
                layer_idx = layout.resolve_block_op_layer(op)
            except Exception:
                layer_idx = None
        if layer_idx is None:
            layer_idx = len(layout.ops_per_layer) - 1
        out.append((int(layer_idx), op))
    return out


def _first_mismatching_op(
    layout,
    residuals,
    pos: int,
    suspect_dims: Tuple[str, ...],
    *,
    epsilon: float = 1e-3,
):
    """Walk ops layer-by-layer; for each op whose ``produces`` overlaps one of
    the suspect dims, check that the residual at ``pos`` is non-zero for that
    dim after the op's layer ran. Return the FIRST op where the declaration
    fires but the residual is silent (the most actionable lead).
    """
    import torch

    if not residuals:
        return None

    suspect_set = set(suspect_dims)
    ops = _ops_by_layer(layout)

    seen: set = set()
    for layer_idx, op in ops:
        if op.name in seen:
            continue
        seen.add(op.name)
        produces = getattr(op, "produces", None) or {}
        # Restrict attention to the suspect dim family.
        relevant = [d for d in produces.keys() if d in suspect_set]
        if not relevant:
            continue
        if layer_idx + 1 >= len(residuals):
            continue
        post = residuals[layer_idx + 1]
        for dim_name in relevant:
            slice_vals = _residual_slice(post, layout, dim_name, pos)
            if slice_vals is None:
                continue
            magnitude = float(slice_vals.abs().max().item())
            if magnitude < epsilon:
                return {
                    "op": op,
                    "layer_idx": layer_idx,
                    "dim": dim_name,
                    "register": produces.get(dim_name),
                    "magnitude": magnitude,
                    "reason": (
                        f"declares produces[{dim_name!r}]="
                        f"{produces.get(dim_name)!r} but residual abs-max "
                        f"at AX-marker pos={pos} after layer {layer_idx} "
                        f"is {magnitude:.3e} (< {epsilon:.0e})"
                    ),
                }
    return None


def _candidate_ops_for_dims(
    layout, suspect_dims: Tuple[str, ...]
) -> List[Tuple[str, str]]:
    """All ops whose ``produces`` overlaps the suspect dim family.

    Returns ``[(op_name, "file:line")]`` so the brief can list the obvious
    edit candidates even if the first-mismatch heuristic returns None.
    """
    suspect_set = set(suspect_dims)
    out: List[Tuple[str, str]] = []
    seen: set = set()
    all_ops = []
    for ops_at in layout.ops_per_layer:
        all_ops.extend(ops_at)
    all_ops.extend(layout.block_ops)
    all_ops.extend(layout.model_ops)
    for op in all_ops:
        if op.name in seen:
            continue
        seen.add(op.name)
        produces = getattr(op, "produces", None) or {}
        if not any(d in suspect_set for d in produces):
            continue
        loc = _op_source_location(op)
        loc_str = f"{loc[0]}:{loc[1]}" if loc else "<unknown>"
        out.append((op.name, loc_str))
    return out


# ---------------------------------------------------------------------------
# Markdown emitter
# ---------------------------------------------------------------------------


def _format_token(token: Optional[int]) -> str:
    if token is None:
        return "<missing>"
    if 0 <= token < 256:
        return f"0x{token:02x}"
    try:
        from neural_vm.vm_step import Token

        for name, value in vars(Token).items():
            if name.isupper() and value == token:
                return name
    except Exception:
        pass
    return str(token)


def _format_brief(attr: Attribution) -> str:
    lines: List[str] = []
    lines.append(f"# 1096 failure attribution — test {attr.test_idx:04d}")
    lines.append("")
    lines.append(f"- **description**: `{attr.description}`")
    lines.append(f"- **suite_expected**: `{attr.suite_expected & 0xFFFFFFFF}`")
    lines.append(f"- **declarative_exit**: `{attr.declarative_exit}`")
    lines.append(f"- **declarative_steps**: `{attr.declarative_steps}`")
    lines.append(f"- **neural_exit**: `{attr.neural_exit}`")
    lines.append("")
    if attr.divergence_slot is not None:
        lines.append("## First token divergence")
        lines.append("")
        lines.append(f"- **step**: {attr.divergence_step}")
        lines.append(f"- **slot**: `{attr.divergence_slot}`")
        lines.append(f"- **expected_token**: {_format_token(attr.expected_token)}")
        lines.append(f"- **neural_token**: {_format_token(attr.neural_token)}")
        lines.append(
            "- **suspect_dims**: "
            + ", ".join(f"`{d}`" for d in attr.suspect_dims)
        )
        lines.append("")
    else:
        lines.append("## First token divergence")
        lines.append("")
        lines.append("_No token divergence observed; neural matches symbolic._")
        lines.append("")

    lines.append("## First op with mismatching produces")
    lines.append("")
    if attr.first_mismatch_op is not None:
        lines.append(f"- **op**: `{attr.first_mismatch_op}`")
        lines.append(f"- **layer_idx**: {attr.first_mismatch_layer}")
        if attr.first_mismatch_file is not None:
            lines.append(
                f"- **source**: `{attr.first_mismatch_file}:{attr.first_mismatch_line}`"
            )
        if attr.first_mismatch_reason is not None:
            lines.append(f"- **reason**: {attr.first_mismatch_reason}")
        lines.append("")
        lines.append("### Suggested fix shape")
        lines.append("")
        lines.append(
            "The op's declared ``produces`` includes a suspect dim but the "
            "residual at the AX marker after this op's layer is effectively "
            "zero. Likely causes (highest to lowest probability):"
        )
        lines.append("")
        lines.append(
            "1. The op's ``effective_predicate`` no longer fires on this "
            "step's opcode (an upstream rename or opcode-gate change "
            "silenced it). Inspect the rule's ``reads`` and the active "
            "opcode at the divergence step."
        )
        lines.append(
            "2. The op fires but writes to a different dim than the "
            "declaration claims — verify ``writes`` vs ``produces`` keys "
            "in the bake (or rule.writes for IR ops)."
        )
        lines.append(
            "3. A competing same-layer writer is cancelling this op's "
            "contribution. Cross-reference with ``tools/attribute_failures.py``'s "
            "writer index for the suspect dim."
        )
        lines.append("")
    else:
        lines.append("_No op declares ``produces`` over the suspect dims_ "
                     "(or the residual check could not be performed).")
        lines.append("")

    if attr.candidate_ops:
        lines.append("## All candidate ops for suspect dims")
        lines.append("")
        for name, loc in attr.candidate_ops:
            lines.append(f"- `{name}` — `{loc}`")
        lines.append("")

    if attr.notes:
        lines.append("## Notes")
        lines.append("")
        for n in attr.notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main attribution pipeline
# ---------------------------------------------------------------------------


def attribute_test(
    test_id: str,
    *,
    runner=None,
    alu_mode: str = "lookup",
    declarations_only: bool = True,
) -> Tuple[Attribution, str]:
    """Run the attribution pipeline for one test; return (attribution, markdown)."""
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program
    from tests.test_1096_neural_declarative_diagnostic import (
        _first_token_divergence,
    )

    idx, source, expected, description = _select_test(test_id)
    attr = Attribution(
        test_idx=idx,
        description=description,
        suite_expected=expected,
        declarative_exit=None,
        declarative_steps=None,
        neural_exit=None,
    )

    try:
        bytecode, data = compile_c(source)
    except Exception as exc:
        attr.notes.append(f"compile_c failed: {exc!r}")
        return attr, _format_brief(attr)

    decl = declarative_oracle_for_program(
        bytecode, data, suite_expected=expected, label=f"id={idx:04d}",
        max_steps=None,
    )
    attr.declarative_exit = decl.exit_code
    attr.declarative_steps = decl.steps
    if decl.error is not None or decl.steps is None:
        attr.notes.append(
            f"declarative oracle did not halt cleanly: {decl.error}"
        )
        return attr, _format_brief(attr)

    expected_exec = _run_symbolic(bytecode, data)

    # Stand up / reuse the neural runner. Single compile per process.
    if runner is None:
        os.environ.setdefault("C4_BATCH_USE_KV_CACHE", "0")
        os.environ.setdefault("C4_SPEC_K", "0")
        if declarations_only:
            os.environ.setdefault("C4_DECLARATIONS_ONLY_BAKE", "1")
        from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

        runner = BatchedPureNeuralRunner(max_seq_len=4096)

    # Force alu_mode='lookup' — the production default. The runner's
    # underlying model picks up alu_mode through compile_full_vm_dynamic
    # which already defaults to 'lookup'; we do not override here because
    # changing alu_mode mid-process would force a recompile and violate the
    # ONE compile constraint.
    try:
        neural_results = runner.run_batch(
            [bytecode],
            data_list=[data],
            max_steps=None,
            expected_steps_list=[decl.steps],
            max_context_window=512,
            spec_k=0,
        )
    except Exception as exc:
        attr.notes.append(f"neural run failed: {exc!r}")
        return attr, _format_brief(attr)

    neural_output, neural_exit = neural_results[0]
    attr.neural_exit = neural_exit

    if neural_exit == decl.exit_code:
        attr.notes.append(
            "neural exit matches declarative exit — final output OK; "
            "no first-token divergence to attribute."
        )
        return attr, _format_brief(attr)

    neural_context = _capture_neural_context(
        runner, bytecode, data, expected_steps=decl.steps
    )
    divergence = _first_token_divergence(expected_exec, neural_context)
    if divergence is None:
        attr.notes.append(
            "no first-token divergence detected even though neural exit "
            "differs from declarative — likely a halt-horizon or final-step "
            "mismatch outside the prefix window."
        )
        return attr, _format_brief(attr)

    attr.divergence_step = divergence.step
    attr.divergence_slot = divergence.slot
    attr.expected_token = divergence.expected_token
    attr.neural_token = divergence.neural_token
    suspect_dims = _suspect_dims_for_slot(divergence.slot)
    attr.suspect_dims = suspect_dims

    # Locate the layout for the op walk. AutoregressiveVMRunner discards
    # ``_layout`` after the bake, so re-call ``compile_full_vm_dynamic`` —
    # both the in-process memo and the disk cache short-circuit this to a
    # no-op when the kwargs match the warm cache key.
    layout = getattr(runner.model, "layout", None)
    if layout is None:
        layout = getattr(getattr(runner, "_serial", None), "layout", None)
    if layout is None:
        try:
            from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
                compile_full_vm_dynamic,
            )
            _model, layout = compile_full_vm_dynamic(
                alu_mode=alu_mode,
                strict=False,
            )
        except Exception as exc:
            attr.notes.append(f"could not reacquire layout: {exc!r}")
            attr.candidate_ops = []
            return attr, _format_brief(attr)

    # Snapshot per-block residuals at the divergence position.
    residuals, logit_pos = _probe_layer_residuals(
        runner.model, expected_exec, divergence,
    )
    if residuals is None or logit_pos is None:
        attr.notes.append("could not build residual snapshot at divergence")
        attr.candidate_ops = _candidate_ops_for_dims(layout, suspect_dims)
        return attr, _format_brief(attr)

    mismatch = _first_mismatching_op(layout, residuals, logit_pos, suspect_dims)
    if mismatch is not None:
        op = mismatch["op"]
        loc = _op_source_location(op)
        attr.first_mismatch_op = op.name
        attr.first_mismatch_layer = mismatch["layer_idx"]
        if loc is not None:
            attr.first_mismatch_file = loc[0]
            attr.first_mismatch_line = loc[1]
        attr.first_mismatch_reason = mismatch["reason"]

    attr.candidate_ops = _candidate_ops_for_dims(layout, suspect_dims)
    return attr, _format_brief(attr)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _default_out_path(idx_or_label: str) -> str:
    label = idx_or_label.strip().replace(" ", "_")
    base = os.path.join(_PKG, ".agent-logs")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"1096_fail_{label}.md")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-attribute a 1096 failure to a producing op.",
    )
    parser.add_argument(
        "--test_id",
        required=True,
        help=(
            "Numeric index into generate_test_programs(), or a substring "
            "of the test description (e.g. 'add_42')."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Markdown output path (default: .agent-logs/1096_fail_<id>.md).",
    )
    parser.add_argument(
        "--no-declarations-only",
        action="store_true",
        help="Disable C4_DECLARATIONS_ONLY_BAKE=1 (default: enabled).",
    )
    args = parser.parse_args(argv)

    out_path = args.out or _default_out_path(args.test_id)
    attr, markdown = attribute_test(
        args.test_id,
        declarations_only=not args.no_declarations_only,
    )
    with open(out_path, "w") as fh:
        fh.write(markdown)
        if not markdown.endswith("\n"):
            fh.write("\n")

    sys.stderr.write(
        f"[attribute_1096] wrote {out_path} "
        f"(test_idx={attr.test_idx:04d} "
        f"first_mismatch_op={attr.first_mismatch_op!r})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
