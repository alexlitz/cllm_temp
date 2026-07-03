"""S-7: observe per-output-dim, per-position-class signed contributions on
a corpus. Used by the strength verifier to know what the model's output
head produces at each position-class.

Usage:
    PYTHONPATH=.:c4_release python -m tools.observe_backbone_contributions \\
        --output .agent-logs/backbone-bounds/v1.json --limit 32

Observed contribution at each (output_dim, position_class) is the max
pre-softmax logit observed in the corpus. This is an over-approximation
of any individual non-rule writer's contribution -- sufficient for the
strength verifier's conservative bound (V1 simplification).

Position classes are coarse buckets (mark==PC/AX/SP/BP/MEM/STACK0/SE,
byte_index in 0/1/2/3, has_se, step_is_fresh) chosen to match F-1+F-2
DSL atoms. A token position can belong to multiple classes simultaneously
(e.g. a byte at byte_index==1 within a step that has already passed a
STEP_END belongs to both ``is_byte AND byte_index==1`` and ``has_se``).

Output dim naming: the model head produces vocab logits (shape
``[batch, seq, vocab_size]``). For V1, the ``output_dim`` is named by
token-name when one exists in ``Token``, else as ``BYTE_<hex>`` for the
0..255 byte range. The ``+offset`` suffix is always ``+0`` for vocab
logits (it remains in the schema for compatibility with future versions
that may expose residual-dim slices instead).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# Ensure ``c4_release`` is importable when invoked from the repo root via
# ``python -m tools.observe_backbone_contributions``.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_C4_RELEASE_DIR = os.path.dirname(_THIS_DIR)
if _C4_RELEASE_DIR not in sys.path:
    sys.path.insert(0, _C4_RELEASE_DIR)


POSITION_CLASSES = [
    "mark==PC",
    "mark==AX",
    "mark==SP",
    "mark==BP",
    "mark==MEM",
    "mark==STACK0",
    "mark==SE",
    "is_byte AND byte_index==0",
    "is_byte AND byte_index==1",
    "is_byte AND byte_index==2",
    "is_byte AND byte_index==3",
    "has_se",
    "step_is_fresh",
]


def _vocab_dim_name(token_id: int) -> str:
    """Return a stable human-readable name for a vocab logit index."""
    from neural_vm.vm_step import Token

    for name, value in vars(Token).items():
        if name.isupper() and isinstance(value, int) and value == token_id:
            return name
    if 0 <= token_id < 256:
        return f"BYTE_0x{token_id:02X}"
    return f"VOCAB_{token_id}"


def _classify_positions(context: Sequence[int]) -> List[List[str]]:
    """For each token position, return the set of position classes it
    matches.

    Classification is purely structural: the per-step layout is fixed
    (PC(5) + AX(5) + SP(5) + BP(5) + STACK0(5) + MEM(9) + SE(1) = 35
    tokens), so we scan the symbolic context once and assign each
    position to its set of buckets.

    A token belongs to ``step_is_fresh`` if it is the FIRST token of a
    new step (i.e. immediately after a STEP_END / right at the start of
    the program). It belongs to ``has_se`` once at least one STEP_END
    has been emitted earlier in the stream.
    """
    from neural_vm.vm_step import Token

    classes: List[List[str]] = []
    has_se_seen = False
    just_saw_step_end_or_data_end = False
    cur_marker: str = ""  # one of "PC", "AX", "SP", "BP", "STACK0", "MEM", ""
    byte_index_within_marker = -1  # 0-based index of byte position after marker

    for tok in context:
        bucket: List[str] = []
        marker_for_this_pos = ""

        if tok == Token.REG_PC:
            marker_for_this_pos = "PC"
            cur_marker = "PC"
            byte_index_within_marker = 0
        elif tok == Token.REG_AX:
            marker_for_this_pos = "AX"
            cur_marker = "AX"
            byte_index_within_marker = 0
        elif tok == Token.REG_SP:
            marker_for_this_pos = "SP"
            cur_marker = "SP"
            byte_index_within_marker = 0
        elif tok == Token.REG_BP:
            marker_for_this_pos = "BP"
            cur_marker = "BP"
            byte_index_within_marker = 0
        elif tok == Token.STACK0:
            marker_for_this_pos = "STACK0"
            cur_marker = "STACK0"
            byte_index_within_marker = 0
        elif tok == Token.MEM:
            marker_for_this_pos = "MEM"
            cur_marker = "MEM"
            byte_index_within_marker = 0
        elif tok in (Token.STEP_END, Token.HALT):
            marker_for_this_pos = "SE"
            cur_marker = ""
            byte_index_within_marker = -1
        elif 0 <= tok < 256:
            # byte token following the current marker
            if cur_marker:
                # MEM has 8 byte-positions (addr 0..3, value 0..3);
                # values 4..7 also map to byte_index 0..3 (val byte i).
                bi = byte_index_within_marker % 4
                bucket.append(f"is_byte AND byte_index=={bi}")
                byte_index_within_marker += 1

        if marker_for_this_pos:
            bucket.append(f"mark=={marker_for_this_pos}")

        if has_se_seen:
            bucket.append("has_se")

        if just_saw_step_end_or_data_end:
            bucket.append("step_is_fresh")
            just_saw_step_end_or_data_end = False

        if tok in (Token.STEP_END, Token.HALT, Token.DATA_END):
            has_se_seen = True
            just_saw_step_end_or_data_end = True

        classes.append(bucket)

    return classes


def _build_context_prefix(bytecode: Sequence[int], data: Sequence[int]) -> List[int]:
    """Build the standard CODE_START..DATA_END token prefix for a program."""
    from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE
    from neural_vm.vm_step import Token

    tokens: List[int] = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(IMMEDIATE_SIZE):
            tokens.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            tokens.append(0)
    tokens.extend([Token.CODE_END, Token.DATA_START])
    tokens.extend(int(b) & 0xFF for b in data)
    tokens.append(Token.DATA_END)
    return tokens


def _append_symbolic_step_tokens(context: List[int], state, step_trace) -> None:
    from neural_vm.vm_step import Token

    def append_u32(value: int) -> None:
        value &= 0xFFFFFFFF
        for i in range(4):
            context.append((value >> (i * 8)) & 0xFF)

    context.append(Token.REG_PC)
    append_u32(state.pc)
    context.append(Token.REG_AX)
    append_u32(state.ax)
    context.append(Token.REG_SP)
    append_u32(state.sp)
    context.append(Token.REG_BP)
    append_u32(state.bp)
    context.append(Token.STACK0)
    append_u32(state.mem_read(state.sp))
    context.append(Token.MEM)
    append_u32(step_trace.mem_addr)
    append_u32(step_trace.mem_value)
    context.append(Token.STEP_END if not state.halted else Token.HALT)


def _build_symbolic_context(
    bytecode: Sequence[int], data: Sequence[int]
) -> List[int]:
    """Run the declarative symbolic interpreter to produce the
    teacher-forced token sequence for one program."""
    from neural_vm.verification.symbolic_program import (
        SymbolicDeclarativeProgramRunner,
    )

    symbolic = SymbolicDeclarativeProgramRunner()
    state = symbolic.init_state(bytecode, data)
    context = _build_context_prefix(bytecode, data)
    while symbolic.step(state):
        _append_symbolic_step_tokens(context, state, state.trace[-1])
        if state.halted:
            break
    return context


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _select_programs(limit: int, offset: int):
    """Return ``[(source, expected, description), ...]`` from the 1096 suite."""
    from tests.test_suite_1000 import generate_test_programs

    programs = list(generate_test_programs())
    if offset:
        programs = programs[offset:]
    return programs[:limit]


def _format_bound_key(token_id: int) -> str:
    """The output_dim key shape ``<name>+<offset>``."""
    return f"{_vocab_dim_name(token_id)}+0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="Number of programs from the 1096 suite (default 32; override "
        "via env C4_1096_LIMIT for sweep-friendly throttling)",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Model max_seq_len (must accommodate the teacher-forced contexts)",
    )
    args = parser.parse_args()

    # Honour C4_1096_LIMIT so the parent harness can throttle without
    # rewriting the command line.
    env_limit = os.environ.get("C4_1096_LIMIT")
    if env_limit:
        try:
            args.limit = min(args.limit, int(env_limit))
        except ValueError:
            pass

    import torch

    from src.compiler import compile_c
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    print(
        f"[observe] building model (max_seq_len={args.max_seq_len})...",
        flush=True,
    )
    runner = BatchedPureNeuralRunner(max_seq_len=args.max_seq_len)
    model = runner.model
    device = next(model.parameters()).device

    programs = _select_programs(args.limit, args.offset)
    print(
        f"[observe] selected {len(programs)} programs (offset={args.offset})",
        flush=True,
    )

    bounds: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "max_positive_contribution": 0.0,
                "max_negative_contribution": 0.0,
                "samples": 0,
            }
        )
    )

    processed = 0
    skipped = 0

    for idx, (source, expected, description) in enumerate(programs):
        try:
            bytecode, data = compile_c(source)
        except Exception as exc:
            print(
                f"[observe] program {idx:04d} compile failed: {exc!r}",
                flush=True,
            )
            skipped += 1
            continue

        try:
            context = _build_symbolic_context(bytecode, data)
        except Exception as exc:
            print(
                f"[observe] program {idx:04d} symbolic failed: {exc!r}",
                flush=True,
            )
            skipped += 1
            continue

        if len(context) > args.max_seq_len:
            # Truncate-left to fit; this is acceptable for an
            # over-approximation: we still observe the tail-of-program
            # logits, which dominate steady-state contribution.
            context = context[-args.max_seq_len :]

        classes_per_pos = _classify_positions(context)

        token_ids = torch.tensor([context], dtype=torch.long, device=device)
        model.embed.set_mem_history_end(0)
        with torch.no_grad():
            logits = model(token_ids)  # [1, seq, vocab]
        logits = logits[0].detach().to("cpu")  # [seq, vocab]
        vocab_size = int(logits.shape[1])

        # For each (output_dim, position_class) update max-positive /
        # max-negative observed logit.
        for pos, buckets in enumerate(classes_per_pos):
            if not buckets:
                continue
            row = logits[pos]
            row_max = float(row.max().item())
            row_min = float(row.min().item())
            # The max/min over the full vocab are useful only if we
            # also want a global cap; the per-output-dim cap is more
            # informative. Iterate per output dim:
            row_list = row.tolist()
            for dim in range(vocab_size):
                v = row_list[dim]
                key = _format_bound_key(dim)
                for cls in buckets:
                    entry = bounds[key][cls]
                    if v > entry["max_positive_contribution"]:
                        entry["max_positive_contribution"] = v
                    if v < entry["max_negative_contribution"]:
                        entry["max_negative_contribution"] = v
                    entry["samples"] += 1
            del row_list  # release the python list
            del row_max, row_min

        processed += 1
        print(
            f"[observe] program {idx:04d} processed "
            f"(context_len={len(context)}, vocab={vocab_size})",
            flush=True,
        )

    git_sha = _git_sha()

    out: Dict[str, object] = {
        "version": 1,
        "corpus_size": processed,
        "model_commit": git_sha,
        "position_classes": POSITION_CLASSES,
        "bounds": {
            k: {c: dict(entry) for c, entry in cls_map.items()}
            for k, cls_map in bounds.items()
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    total_entries = sum(
        1
        for cls_map in bounds.values()
        for _ in cls_map.values()
    )
    print(
        f"[observe] wrote {len(bounds)} output_dims x "
        f"{len(POSITION_CLASSES)} position_classes "
        f"({total_entries} populated entries) to {output_path} "
        f"(processed={processed}, skipped={skipped})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
