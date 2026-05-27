#!/usr/bin/env python3
"""Focused neural-vs-declarative diagnostics for the 1096 suite.

This is intentionally opt-in because it builds/runs the neural VM.  Example:

    C4_1096_DIAG=1 C4_1096_OFFSET=0 C4_1096_LIMIT=8 \
    C4_BATCH_USE_KV_CACHE=0 C4_SPEC_K=0 \
    pytest -q c4_release/tests/test_1096_neural_declarative_diagnostic.py -s

The diagnostic uses declarative symbolic execution as the oracle for both the
expected exit value and the halt horizon.  Rows are printed only for programs
where neural execution diverges from declarative execution.

Set C4_1096_TRACE_FAILURES=1 to rerun failing rows and print the first emitted
token mismatch plus an output-head/residual diagnosis for that symbolic token.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import sys
import types
from typing import Any, Iterable, List, Optional, Sequence, TextIO

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs
from tests.declarative_oracle import declarative_oracle_for_program


def _parse_spec_k(raw: str) -> int:
    raw = (raw or "").strip().lower()
    if raw == "adaptive":
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def _parse_trace_limit(raw: str) -> int:
    raw = (raw or "").strip().lower()
    if raw in {"", "none", "all"}:
        return 1_000_000
    return max(0, int(raw))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _shorten(text: str, width: int = 72) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


_STEP_SLOT_NAMES = (
    "REG_PC",
    "PC_byte0",
    "PC_byte1",
    "PC_byte2",
    "PC_byte3",
    "REG_AX",
    "AX_byte0",
    "AX_byte1",
    "AX_byte2",
    "AX_byte3",
    "REG_SP",
    "SP_byte0",
    "SP_byte1",
    "SP_byte2",
    "SP_byte3",
    "REG_BP",
    "BP_byte0",
    "BP_byte1",
    "BP_byte2",
    "BP_byte3",
    "STACK0",
    "STACK0_byte0",
    "STACK0_byte1",
    "STACK0_byte2",
    "STACK0_byte3",
    "MEM",
    "MEM_addr0",
    "MEM_addr1",
    "MEM_addr2",
    "MEM_addr3",
    "MEM_value0",
    "MEM_value1",
    "MEM_value2",
    "MEM_value3",
    "STEP_END",
)


def _token_name(token: Optional[int]) -> str:
    if token is None:
        return "<missing>"
    if 0 <= token < 256:
        return f"0x{token:02x}"
    try:
        from neural_vm.vm_step import Token
    except Exception:
        return str(token)
    for name, value in vars(Token).items():
        if name.isupper() and value == token:
            return name
    return str(token)


@dataclass(frozen=True)
class SymbolicExpectedExecution:
    """Teacher-forced token stream implied by declarative symbolic execution."""

    context: List[int]
    prefix_len: int
    steps: int
    exit_code: Optional[int]
    halted: bool


@dataclass(frozen=True)
class TokenDivergence:
    """First emitted-token mismatch between neural and symbolic streams."""

    token_index: int
    generated_index: int
    step: int
    offset: int
    slot: str
    expected_token: Optional[int]
    neural_token: Optional[int]

    def format(self) -> str:
        return (
            f"first_token_divergence=step{self.step}:{self.slot} "
            f"abs={self.token_index} gen={self.generated_index} "
            f"expected={_token_name(self.expected_token)} "
            f"neural={_token_name(self.neural_token)}"
        )


@dataclass(frozen=True)
class ResidualSupportSnapshot:
    """How strongly one residual snapshot supports the expected token."""

    label: str
    block_index: Optional[int]
    original_layer_index: Optional[int]
    expected_token: int
    neural_token: Optional[int]
    argmax_token: int
    expected_logit: float
    argmax_logit: float
    neural_logit: Optional[float]
    expected_margin: float
    residual_note: str = ""
    band_contract_report: Optional[Any] = None

    @property
    def supports_expected(self) -> bool:
        return self.argmax_token == self.expected_token

    def format(self) -> str:
        block = "-" if self.block_index is None else str(self.block_index)
        layer = (
            "-"
            if self.original_layer_index is None
            else str(self.original_layer_index)
        )
        neural = (
            ""
            if self.neural_token is None or self.neural_logit is None
            else f" neural_logit={self.neural_logit:+.2f}"
        )
        note = f" {self.residual_note}" if self.residual_note else ""
        band_contracts = (
            f" {self.band_contract_report.format_inline()}"
            if self.band_contract_report is not None
            else ""
        )
        return (
            f"block={block} layer={layer} label={self.label!r} "
            f"expected={_token_name(self.expected_token)} "
            f"argmax={_token_name(self.argmax_token)} "
            f"expected_logit={self.expected_logit:+.2f} "
            f"argmax_logit={self.argmax_logit:+.2f} "
            f"margin={self.expected_margin:+.2f}"
            f"{neural}{note}{band_contracts}"
        )


@dataclass(frozen=True)
class ResidualDivergenceReport:
    """Layer-by-layer residual/head support diagnosis for one token mismatch."""

    kind: str
    snapshot: ResidualSupportSnapshot

    def format(self) -> str:
        return f"residual_diagnosis={self.kind} {self.snapshot.format()}"


@dataclass(frozen=True)
class NeuralDeclarativeDiagnosticRow:
    """One selected 1096 program compared across declarative and neural paths."""

    test_idx: int
    description: str
    suite_expected: int
    declarative_exit: Optional[int]
    declarative_steps: Optional[int]
    neural_exit: Optional[int]
    neural_output: str = ""
    error: Optional[str] = None
    first_token_divergence: Optional[TokenDivergence] = None
    residual_diagnosis: Optional[ResidualDivergenceReport] = None
    trace_error: Optional[str] = None
    comparison_mode: str = "final-output"

    @property
    def status(self) -> str:
        if self.declarative_exit != (self.suite_expected & 0xFFFFFFFF):
            if self.declarative_exit is not None:
                return "suite/declarative-mismatch"
        if self.error is not None:
            return "error"
        if self.declarative_exit != (self.suite_expected & 0xFFFFFFFF):
            return "suite/declarative-mismatch"
        if self.comparison_mode == "strict-first-safe-token":
            if self.neural_exit != self.declarative_exit:
                return "strict-first-safe-token-divergence"
            return "strict-ok"
        if self.neural_exit != self.declarative_exit:
            return "neural-divergence"
        return "ok"

    @property
    def suite_declarative_match(self) -> bool:
        return self.declarative_exit == (self.suite_expected & 0xFFFFFFFF)

    def format(self) -> str:
        output = (
            f" output={self.neural_output!r}"
            if self.neural_output
            else ""
        )
        error = f" error={self.error}" if self.error else ""
        trace_error = (
            f" trace_error={self.trace_error}"
            if self.trace_error is not None
            else ""
        )
        token_divergence = (
            f" {self.first_token_divergence.format()}"
            if self.first_token_divergence is not None
            else ""
        )
        residual = (
            f" {self.residual_diagnosis.format()}"
            if self.residual_diagnosis is not None
            else ""
        )
        return (
            f"[1096-diag] id={self.test_idx:04d} "
            f"mode={self.comparison_mode} "
            f"status={self.status} "
            f"suite_decl={'match' if self.suite_declarative_match else 'mismatch'} "
            f"desc={_shorten(self.description)!r} "
            f"expected={self.suite_expected & 0xFFFFFFFF} "
            f"decl={self.declarative_exit} "
            f"decl_steps={self.declarative_steps} "
            f"neural={self.neural_exit}"
            f"{output}{error}{trace_error}{token_divergence}{residual}"
        )


def _build_context_prefix(bytecode: Sequence[int], data: Sequence[int] | bytes) -> List[int]:
    from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE
    from neural_vm.vm_step import Token

    tokens = [Token.CODE_START]
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


def _build_symbolic_expected_execution(
    bytecode: Sequence[int],
    data: Sequence[int] | bytes,
) -> SymbolicExpectedExecution:
    from neural_vm.unified_compiler.symbolic_program import (
        SymbolicDeclarativeProgramRunner,
    )

    symbolic = SymbolicDeclarativeProgramRunner()
    state = symbolic.init_state(bytecode, data)
    context = _build_context_prefix(bytecode, data)
    prefix_len = len(context)
    while symbolic.step(state):
        _append_symbolic_step_tokens(context, state, state.trace[-1])
        if state.halted:
            break
    return SymbolicExpectedExecution(
        context=context,
        prefix_len=prefix_len,
        steps=state.steps,
        exit_code=state.ax if state.halted else None,
        halted=state.halted,
    )


def _slot_name(offset: int) -> str:
    if 0 <= offset < len(_STEP_SLOT_NAMES):
        return _STEP_SLOT_NAMES[offset]
    return f"offset{offset}"


def _first_token_divergence(
    expected: SymbolicExpectedExecution,
    neural_context: Sequence[int],
) -> Optional[TokenDivergence]:
    from neural_vm.vm_step import Token

    first_generated = expected.prefix_len
    limit = min(len(expected.context), len(neural_context))
    token_index = None
    for i in range(first_generated, limit):
        if expected.context[i] != neural_context[i]:
            token_index = i
            break
    if token_index is None:
        if len(neural_context) < len(expected.context):
            token_index = len(neural_context)
        elif len(neural_context) > len(expected.context):
            token_index = len(expected.context)
        else:
            return None

    generated_index = token_index - first_generated
    step = generated_index // Token.STEP_TOKENS
    offset = generated_index % Token.STEP_TOKENS
    expected_token = (
        expected.context[token_index]
        if token_index < len(expected.context)
        else None
    )
    neural_token = (
        int(neural_context[token_index])
        if token_index < len(neural_context)
        else None
    )
    return TokenDivergence(
        token_index=token_index,
        generated_index=generated_index,
        step=step,
        offset=offset,
        slot=_slot_name(offset),
        expected_token=expected_token,
        neural_token=neural_token,
    )


def _capture_single_neural_context(
    runner,
    bytecode: Sequence[int],
    data: Sequence[int] | bytes,
    *,
    expected_steps: Optional[int],
    spec_k: int,
    max_context_window: int,
) -> List[int]:
    from neural_vm import batched_pure_neural as bpn

    runner._reset_kv_cache()
    runner._reset_spec_stats()
    adaptive = spec_k < 0
    state = runner._build_element(
        list(bytecode),
        data,
        [],
        "",
        spec_k=spec_k if not adaptive else 0,
        adaptive_start_k=bpn._ADAPTIVE_START_K if adaptive else 0,
        expected_steps=expected_steps,
    )
    if adaptive:
        runner._run_speculative(
            [state],
            max_steps=None,
            max_context_window=max_context_window,
            spec_k=0,
            adaptive=True,
        )
    elif spec_k > 0:
        runner._run_speculative(
            [state],
            max_steps=None,
            max_context_window=max_context_window,
            spec_k=spec_k,
            adaptive=False,
        )
    else:
        runner._run_unspeculative(
            [state],
            max_steps=None,
            max_context_window=max_context_window,
        )
    return list(state.context)


def _dim(model, name: str) -> int:
    from neural_vm.vm_step import _SetDim

    positions = getattr(model, "dim_positions", None)
    if isinstance(positions, dict) and name in positions:
        return int(positions[name])
    return int(getattr(_SetDim, name))


_NEXT_FLAG_BY_TOKEN_NAME = {
    "REG_PC": "NEXT_PC",
    "REG_AX": "NEXT_AX",
    "REG_SP": "NEXT_SP",
    "REG_BP": "NEXT_BP",
    "STACK0": "NEXT_STACK0",
    "MEM": "NEXT_MEM",
    "STEP_END": "NEXT_SE",
    "HALT": "NEXT_HALT",
    "TOOL_CALL": "NEXT_TOOL_CALL",
    "THINKING_START": "NEXT_THINKING_START",
    "THINKING_END": "NEXT_THINKING_END",
}


def _next_flag_for_token(token: int) -> Optional[str]:
    from neural_vm.vm_step import Token

    for token_name, flag_name in _NEXT_FLAG_BY_TOKEN_NAME.items():
        if getattr(Token, token_name, None) == token:
            return flag_name
    return None


def _residual_note_for_token(vec, model, token: int) -> str:
    import torch

    if 0 <= token < 256:
        out_lo = _dim(model, "OUTPUT_LO")
        out_hi = _dim(model, "OUTPUT_HI")
        lo = token & 0xF
        hi = (token >> 4) & 0xF
        lo_band = vec[out_lo : out_lo + 16]
        hi_band = vec[out_hi : out_hi + 16]
        lo_arg = int(torch.argmax(lo_band).item())
        hi_arg = int(torch.argmax(hi_band).item())
        return (
            f"OUT_LO[{lo}]={float(lo_band[lo].item()):+.2f} "
            f"arg={lo_arg}/{float(lo_band[lo_arg].item()):+.2f} "
            f"OUT_HI[{hi}]={float(hi_band[hi].item()):+.2f} "
            f"arg={hi_arg}/{float(hi_band[hi_arg].item()):+.2f}"
        )

    flag_name = _next_flag_for_token(token)
    if flag_name is None:
        return ""
    try:
        flag_value = float(vec[_dim(model, flag_name)].item())
    except (AttributeError, KeyError):
        return f"{flag_name}=<missing-dim>"
    return f"{flag_name}={flag_value:+.2f}"


def _band_contract_report_for_token(
    vec,
    model,
    token: int,
    *,
    include_projection: bool = False,
    min_active_margin: float = 0.5,
    max_inactive_value: float = 0.2,
):
    if not 0 <= token < 256:
        return None

    from neural_vm.unified_compiler.band_contracts import (
        verify_declared_output_nibble_bands,
    )

    out_lo = _dim(model, "OUTPUT_LO")
    out_hi = _dim(model, "OUTPUT_HI")
    output_bands = {
        "OUTPUT_LO": vec[out_lo : out_lo + 16],
        "OUTPUT_HI": vec[out_hi : out_hi + 16],
    }
    return verify_declared_output_nibble_bands(
        output_bands,
        expected_byte=token,
        min_active_margin=min_active_margin,
        max_inactive_value=max_inactive_value,
        include_projection=include_projection,
    )


def _head_logits(model, x_at_pos):
    from neural_vm.vm_step import sparse_linear

    weight = model.head.weight
    if getattr(weight, "is_sparse", False):
        return sparse_linear(
            x_at_pos.unsqueeze(0),
            model.head.weight,
            model.head.bias,
        ).squeeze(0)
    return model.head(x_at_pos.unsqueeze(0)).squeeze(0)


def _residual_support_snapshot(
    *,
    model,
    x,
    pos: int,
    label: str,
    block_index: Optional[int],
    expected_token: int,
    neural_token: Optional[int],
    include_band_projection: bool = False,
    band_min_active_margin: float = 0.5,
    band_max_inactive_value: float = 0.2,
) -> ResidualSupportSnapshot:
    import torch

    vec = x[0, pos]
    logits = _head_logits(model, vec)
    argmax_token = int(torch.argmax(logits).item())
    expected_logit = float(logits[expected_token].item())
    argmax_logit = float(logits[argmax_token].item())
    masked = logits.clone()
    masked[expected_token] = float("-inf")
    best_other = float(masked.max().item())
    neural_logit = (
        float(logits[neural_token].item())
        if neural_token is not None and 0 <= neural_token < logits.numel()
        else None
    )
    return ResidualSupportSnapshot(
        label=label,
        block_index=block_index,
        original_layer_index=None,
        expected_token=expected_token,
        neural_token=neural_token,
        argmax_token=argmax_token,
        expected_logit=expected_logit,
        argmax_logit=argmax_logit,
        neural_logit=neural_logit,
        expected_margin=expected_logit - best_other,
        residual_note=_residual_note_for_token(vec, model, expected_token),
        band_contract_report=_band_contract_report_for_token(
            vec,
            model,
            expected_token,
            include_projection=include_band_projection,
            min_active_margin=band_min_active_margin,
            max_inactive_value=band_max_inactive_value,
        ),
    )


def _block_label(block, block_index: int) -> tuple[str, Optional[int]]:
    attn = getattr(block, "attn", None)
    original_layer = getattr(attn, "layer_idx", None)
    ffn = getattr(block, "ffn", None)
    w_up = getattr(ffn, "W_up", None)
    width = None if w_up is None else int(w_up.shape[0])
    width_part = "" if width is None else f" width={width}"
    layer_part = "-" if original_layer is None else str(original_layer)
    return f"block{block_index} layer={layer_part}{width_part}", original_layer


def _window_expected_prefix_for_prediction(
    expected: SymbolicExpectedExecution,
    *,
    token_index: int,
    max_context_window: int,
) -> tuple[List[int], int]:
    prefix = expected.context[:token_index]
    if len(prefix) <= expected.prefix_len + max_context_window:
        return prefix, len(prefix) - 1
    windowed = prefix[: expected.prefix_len] + prefix[-max_context_window:]
    return windowed, len(windowed) - 1


def _trace_residual_support_for_divergence(
    runner,
    expected: SymbolicExpectedExecution,
    divergence: TokenDivergence,
    *,
    max_context_window: int,
    include_band_projection: bool = False,
    band_min_active_margin: float = 0.5,
    band_max_inactive_value: float = 0.2,
) -> Optional[ResidualDivergenceReport]:
    if divergence.expected_token is None or divergence.token_index <= 0:
        return None

    import torch

    model = runner.model
    expected_token = int(divergence.expected_token)
    neural_token = divergence.neural_token
    prefix, logit_pos = _window_expected_prefix_for_prediction(
        expected,
        token_index=divergence.token_index,
        max_context_window=max_context_window,
    )
    if not prefix:
        return None

    device = next(model.parameters()).device
    token_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    model.embed.set_mem_history_end(0)

    snapshots: List[ResidualSupportSnapshot] = []
    with torch.no_grad():
        x = model.embed(token_ids)
        snapshots.append(
            _residual_support_snapshot(
                model=model,
                x=x,
                pos=logit_pos,
                label="embed",
                block_index=None,
                expected_token=expected_token,
                neural_token=neural_token,
                include_band_projection=include_band_projection,
                band_min_active_margin=band_min_active_margin,
                band_max_inactive_value=band_max_inactive_value,
            )
        )
        for block_index, block in enumerate(model.blocks):
            x = block(x)
            label, original_layer = _block_label(block, block_index)
            snapshot = _residual_support_snapshot(
                model=model,
                x=x,
                pos=logit_pos,
                label=label,
                block_index=block_index,
                expected_token=expected_token,
                neural_token=neural_token,
                include_band_projection=include_band_projection,
                band_min_active_margin=band_min_active_margin,
                band_max_inactive_value=band_max_inactive_value,
            )
            snapshots.append(
                replace(snapshot, original_layer_index=original_layer)
            )

    saw_support = False
    for snapshot in snapshots:
        if snapshot.supports_expected:
            saw_support = True
            continue
        if saw_support:
            return ResidualDivergenceReport(
                kind="first-loss-after-symbolic-support",
                snapshot=snapshot,
            )

    final = snapshots[-1]
    if final.supports_expected:
        return ResidualDivergenceReport(
            kind="no-residual-divergence-on-teacher-forced-prefix",
            snapshot=final,
        )
    return ResidualDivergenceReport(
        kind="expected-token-never-wins",
        snapshot=final,
    )


def _attach_failure_trace(
    row: NeuralDeclarativeDiagnosticRow,
    *,
    runner,
    bytecode: Sequence[int],
    data: Sequence[int] | bytes,
    expected_steps: Optional[int],
    spec_k: int,
    max_context_window: int,
    include_band_projection: bool = False,
    band_min_active_margin: float = 0.5,
    band_max_inactive_value: float = 0.2,
) -> NeuralDeclarativeDiagnosticRow:
    try:
        expected = _build_symbolic_expected_execution(bytecode, data)
        neural_context = _capture_single_neural_context(
            runner,
            bytecode,
            data,
            expected_steps=expected_steps,
            spec_k=spec_k,
            max_context_window=max_context_window,
        )
        first = _first_token_divergence(expected, neural_context)
        residual = (
            _trace_residual_support_for_divergence(
                runner,
                expected,
                first,
                max_context_window=max_context_window,
                include_band_projection=include_band_projection,
                band_min_active_margin=band_min_active_margin,
                band_max_inactive_value=band_max_inactive_value,
            )
            if first is not None
            else None
        )
        return replace(
            row,
            first_token_divergence=first,
            residual_diagnosis=residual,
        )
    except Exception as exc:
        return replace(row, trace_error=f"{exc!r}")


def _selected_1096_tests(
    *,
    offset: int,
    limit: Optional[int],
) -> List[tuple[int, str, int, str]]:
    tests = generate_test_programs()
    selected = list(enumerate(tests))
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return [
        (idx, source, expected, description)
        for idx, (source, expected, description) in selected
    ]


def run_1096_neural_declarative_diagnostic(
    *,
    offset: int = 0,
    limit: Optional[int] = 8,
    chunk_size: int = 8,
    spec_k: int = 0,
    max_context_window: int = 512,
    model_max_seq_len: int = 4096,
    trace_failures: bool = False,
    trace_failure_limit: int = 8,
    include_band_projection: bool = False,
    band_min_active_margin: float = 0.5,
    band_max_inactive_value: float = 0.2,
    sort_by_steps: bool = False,
    progress_stream: Optional[TextIO] = None,
    comparison_mode: str = "final-output",
) -> List[NeuralDeclarativeDiagnosticRow]:
    """Run a focused 1096 slice and return declarative/neural comparison rows."""

    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    selected = _selected_1096_tests(offset=offset, limit=limit)
    neural_runner = BatchedPureNeuralRunner(max_seq_len=model_max_seq_len)
    rows: List[NeuralDeclarativeDiagnosticRow] = []
    traced_failures = 0

    selected_groups = [selected]
    if not sort_by_steps:
        selected_groups = [
            selected[start : start + chunk_size]
            for start in range(0, len(selected), chunk_size)
        ]

    for selected_group in selected_groups:
        prepared_entries = []

        for idx, source, expected, description in selected_group:
            try:
                bytecode, data = compile_c(source)
                declarative = declarative_oracle_for_program(
                    bytecode,
                    data,
                    suite_expected=expected,
                    label=f"id={idx:04d}",
                    max_steps=None,
                )
            except Exception as exc:
                rows.append(
                    NeuralDeclarativeDiagnosticRow(
                        test_idx=idx,
                        description=description,
                        suite_expected=expected,
                        declarative_exit=None,
                        declarative_steps=None,
                        neural_exit=None,
                        error=f"compile/declarative error: {exc!r}",
                        comparison_mode=comparison_mode,
                    )
                )
                continue

            decl_exit = declarative.exit_code
            decl_steps = declarative.steps
            if declarative.error is not None or decl_steps is None:
                rows.append(
                    NeuralDeclarativeDiagnosticRow(
                        test_idx=idx,
                        description=description,
                        suite_expected=expected,
                        declarative_exit=decl_exit,
                        declarative_steps=decl_steps,
                        neural_exit=None,
                        error=(
                            declarative.error
                            or "declarative execution did not halt"
                        ),
                        comparison_mode=comparison_mode,
                    )
                )
                continue

            prepared_entries.append(
                (
                    idx,
                    expected,
                    description,
                    decl_exit,
                    decl_steps,
                    bytecode,
                    data,
                )
            )

        if sort_by_steps:
            prepared_entries.sort(key=lambda item: item[4], reverse=True)

        for start in range(0, len(prepared_entries), chunk_size):
            chunk = prepared_entries[start : start + chunk_size]
            bytecodes = []
            data_list = []
            compiled_slots = []
            expected_steps = []

            for entry in chunk:
                compiled_slots.append(entry)
                bytecodes.append(entry[5])
                data_list.append(entry[6])
                expected_steps.append(entry[4])

            if not bytecodes:
                continue

            chunk_ids = [entry[0] for entry in compiled_slots]
            if progress_stream is not None:
                print(
                    "[1096-progress] "
                    f"mode={comparison_mode} "
                    "phase=start "
                    f"rows={len(rows)}/{len(selected)} "
                    f"batch={len(compiled_slots)} "
                    f"ids={min(chunk_ids):04d}-{max(chunk_ids):04d} "
                    f"max_steps={max(expected_steps)}",
                    file=progress_stream,
                    flush=True,
                )

            try:
                neural_results = neural_runner.run_batch(
                    bytecodes,
                    data_list=data_list,
                    max_steps=None,
                    expected_steps_list=expected_steps,
                    max_context_window=max_context_window,
                    spec_k=spec_k,
                )
            except Exception as exc:
                for (
                    idx,
                    expected,
                    description,
                    decl_exit,
                    decl_steps,
                    _bytecode,
                    _data,
                ) in compiled_slots:
                    rows.append(
                        NeuralDeclarativeDiagnosticRow(
                            test_idx=idx,
                            description=description,
                            suite_expected=expected,
                            declarative_exit=decl_exit,
                            declarative_steps=decl_steps,
                            neural_exit=None,
                            error=f"neural batch error: {exc!r}",
                            comparison_mode=comparison_mode,
                        )
                    )
                continue

            for (
                idx,
                expected,
                description,
                decl_exit,
                decl_steps,
                bytecode,
                data,
            ), (neural_output, neural_exit) in zip(compiled_slots, neural_results):
                row = NeuralDeclarativeDiagnosticRow(
                    test_idx=idx,
                    description=description,
                    suite_expected=expected,
                    declarative_exit=decl_exit,
                    declarative_steps=decl_steps,
                    neural_exit=neural_exit,
                    neural_output=neural_output,
                    comparison_mode=comparison_mode,
                )
                if (
                    trace_failures
                    and traced_failures < trace_failure_limit
                    and row.status in {
                        "neural-divergence",
                        "strict-first-safe-token-divergence",
                    }
                ):
                    row = _attach_failure_trace(
                        row,
                        runner=neural_runner,
                        bytecode=bytecode,
                        data=data,
                        expected_steps=decl_steps,
                        spec_k=spec_k,
                        max_context_window=max_context_window,
                        include_band_projection=include_band_projection,
                        band_min_active_margin=band_min_active_margin,
                        band_max_inactive_value=band_max_inactive_value,
                    )
                    traced_failures += 1
                rows.append(row)

            if progress_stream is not None:
                print(
                    "[1096-progress] "
                    f"mode={comparison_mode} "
                    "phase=done "
                    f"rows={len(rows)}/{len(selected)} "
                    f"batch={len(compiled_slots)} "
                    f"ids={min(chunk_ids):04d}-{max(chunk_ids):04d} "
                    f"max_steps={max(expected_steps)}",
                    file=progress_stream,
                    flush=True,
                )

    return rows


def print_divergence_rows(
    rows: Iterable[NeuralDeclarativeDiagnosticRow],
    *,
    stream: TextIO = sys.stderr,
) -> int:
    """Print concise rows for non-matching entries and return the count."""

    count = 0
    for row in rows:
        if row.status == "ok":
            continue
        print(row.format(), file=stream, flush=True)
        count += 1
    return count


def print_diagnostic_summary(
    rows: Sequence[NeuralDeclarativeDiagnosticRow],
    *,
    mode: str,
    stream: TextIO = sys.stderr,
) -> dict[str, int]:
    """Print a machine-readable pass-rate summary for one diagnostic slice."""

    ok_statuses = {"ok"} if mode == "final-output" else {"strict-ok"}
    ok = sum(1 for row in rows if row.status in ok_statuses)
    errors = sum(1 for row in rows if row.status == "error")
    suite_mismatches = sum(
        1 for row in rows if row.status == "suite/declarative-mismatch"
    )
    divergences = len(rows) - ok - errors - suite_mismatches
    summary = {
        "selected": len(rows),
        "ok": ok,
        "divergences": divergences,
        "errors": errors,
        "suite_mismatches": suite_mismatches,
    }
    print(
        "[1096-summary] "
        f"mode={mode} "
        f"selected={summary['selected']} "
        f"ok={summary['ok']} "
        f"divergences={summary['divergences']} "
        f"errors={summary['errors']} "
        f"suite_mismatches={summary['suite_mismatches']}",
        file=stream,
        flush=True,
    )
    return summary


def test_diagnostic_row_format_is_concise():
    row = NeuralDeclarativeDiagnosticRow(
        test_idx=7,
        description="large ADD carry case",
        suite_expected=768,
        declarative_exit=768,
        declarative_steps=5,
        neural_exit=512,
    )

    text = row.format()

    assert "id=0007" in text
    assert "mode=final-output" in text
    assert "status=neural-divergence" in text
    assert "suite_decl=match" in text
    assert "decl=768" in text
    assert "decl_steps=5" in text
    assert "neural=512" in text


def test_strict_first_safe_token_mode_reports_separate_status():
    row = NeuralDeclarativeDiagnosticRow(
        test_idx=7,
        description="large ADD carry case",
        suite_expected=768,
        declarative_exit=768,
        declarative_steps=5,
        neural_exit=None,
        comparison_mode="strict-first-safe-token",
    )

    text = row.format()

    assert row.status == "strict-first-safe-token-divergence"
    assert "mode=strict-first-safe-token" in text
    assert "status=strict-first-safe-token-divergence" in text


def test_first_token_divergence_reports_symbolic_step_slot():
    from neural_vm.vm_step import Token

    expected = SymbolicExpectedExecution(
        context=[
            Token.CODE_START,
            Token.CODE_END,
            Token.REG_PC,
            0x02,
            0x00,
            0x00,
        ],
        prefix_len=2,
        steps=1,
        exit_code=0,
        halted=True,
    )
    neural = [
        Token.CODE_START,
        Token.CODE_END,
        Token.REG_PC,
        0x0A,
        0x00,
        0x00,
    ]

    divergence = _first_token_divergence(expected, neural)

    assert divergence is not None
    assert divergence.step == 0
    assert divergence.offset == 1
    assert divergence.slot == "PC_byte0"
    assert divergence.expected_token == 0x02
    assert divergence.neural_token == 0x0A


def test_row_format_includes_token_and_residual_diagnostics():
    token_divergence = TokenDivergence(
        token_index=42,
        generated_index=7,
        step=0,
        offset=7,
        slot="AX_byte1",
        expected_token=0x05,
        neural_token=0x03,
    )
    residual = ResidualDivergenceReport(
        kind="first-loss-after-symbolic-support",
        snapshot=ResidualSupportSnapshot(
            label="block24 layer=15 width=42",
            block_index=24,
            original_layer_index=15,
            expected_token=0x05,
            neural_token=0x03,
            argmax_token=0x03,
            expected_logit=4.0,
            argmax_logit=6.0,
            neural_logit=6.0,
            expected_margin=-2.0,
            residual_note="OUT_LO[5]=+0.91 arg=3/+1.09",
        ),
    )
    row = NeuralDeclarativeDiagnosticRow(
        test_idx=5,
        description="ADD residual overwrite case",
        suite_expected=1450,
        declarative_exit=1450,
        declarative_steps=5,
        neural_exit=938,
        first_token_divergence=token_divergence,
        residual_diagnosis=residual,
    )

    text = row.format()

    assert "first_token_divergence=step0:AX_byte1" in text
    assert "residual_diagnosis=first-loss-after-symbolic-support" in text
    assert "block=24 layer=15" in text


def test_diagnostic_summary_counts_final_and_strict_modes_separately(capsys):
    final_rows = [
        NeuralDeclarativeDiagnosticRow(0, "ok", 1, 1, 1, 1),
        NeuralDeclarativeDiagnosticRow(1, "bad", 1, 1, 1, 2),
    ]
    strict_rows = [
        NeuralDeclarativeDiagnosticRow(
            0,
            "strict ok",
            1,
            1,
            1,
            1,
            comparison_mode="strict-first-safe-token",
        ),
        NeuralDeclarativeDiagnosticRow(
            1,
            "strict bad",
            1,
            1,
            1,
            None,
            comparison_mode="strict-first-safe-token",
        ),
    ]

    final = print_diagnostic_summary(
        final_rows,
        mode="final-output",
        stream=sys.stderr,
    )
    strict = print_diagnostic_summary(
        strict_rows,
        mode="strict-first-safe-token",
        stream=sys.stderr,
    )

    captured = capsys.readouterr()
    assert final == {
        "selected": 2,
        "ok": 1,
        "divergences": 1,
        "errors": 0,
        "suite_mismatches": 0,
    }
    assert strict == final
    assert "[1096-summary] mode=final-output" in captured.err
    assert "[1096-summary] mode=strict-first-safe-token" in captured.err


def test_sorted_batching_uses_step_count_and_reports_progress(monkeypatch, capsys):
    selected = [
        (0, "return 10;", 10, "ten steps"),
        (1, "return 30;", 30, "thirty steps"),
        (2, "return 20;", 20, "twenty steps"),
    ]
    steps_by_expected = {10: 10, 20: 20, 30: 30}

    class FakeRunner:
        calls = []

        def __init__(self, *, max_seq_len):
            self.max_seq_len = max_seq_len

        def run_batch(
            self,
            bytecodes,
            *,
            data_list,
            max_steps,
            expected_steps_list,
            max_context_window,
            spec_k,
        ):
            FakeRunner.calls.append(list(expected_steps_list))
            return [
                ("", bytecode[0])
                for bytecode in bytecodes
            ]

    def fake_compile(source):
        expected = int(source.removeprefix("return ").removesuffix(";"))
        return [expected], []

    def fake_oracle(bytecode, data, *, suite_expected, label, max_steps):
        return types.SimpleNamespace(
            exit_code=suite_expected,
            steps=steps_by_expected[suite_expected],
            error=None,
        )

    fake_module = types.SimpleNamespace(BatchedPureNeuralRunner=FakeRunner)
    monkeypatch.setitem(sys.modules, "neural_vm.batched_pure_neural", fake_module)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_selected_1096_tests",
        lambda *, offset, limit: selected,
    )
    monkeypatch.setattr(sys.modules[__name__], "compile_c", fake_compile)
    monkeypatch.setattr(
        sys.modules[__name__],
        "declarative_oracle_for_program",
        fake_oracle,
    )

    rows = run_1096_neural_declarative_diagnostic(
        offset=0,
        limit=3,
        chunk_size=2,
        sort_by_steps=True,
        progress_stream=sys.stderr,
        comparison_mode="final-output",
    )

    captured = capsys.readouterr()
    assert [row.test_idx for row in rows] == [1, 2, 0]
    assert FakeRunner.calls == [[30, 20], [10]]
    assert (
        "mode=final-output phase=start rows=0/3 batch=2 "
        "ids=0001-0002 max_steps=30"
    ) in captured.err
    assert (
        "mode=final-output phase=done rows=2/3 batch=2 "
        "ids=0001-0002 max_steps=30"
    ) in captured.err


def test_band_contract_report_for_symbolic_expected_token_is_actionable():
    import torch

    class FakeModel:
        dim_positions = {"OUTPUT_LO": 0, "OUTPUT_HI": 16}

    vec = torch.zeros(32)
    vec[0x5] = 0.46
    vec[0x3] = 0.42
    vec[16 + 0x0] = 1.0
    vec[16 + 0x8] = 0.7

    report = _band_contract_report_for_token(
        vec,
        FakeModel(),
        0x05,
        include_projection=True,
        min_active_margin=0.25,
        max_inactive_value=0.2,
    )

    assert report is not None
    assert not report.ok
    metadata = report.as_dict()
    assert metadata["expected_byte"] == 0x05
    assert {
        violation["kind"]
        for violation in metadata["violations"]
    } == {"active_margin_low", "inactive_too_high"}
    assert any(
        violation["band_base"] == "OUTPUT_LO" and violation["index"] == 5
        for violation in metadata["violations"]
    )
    assert "projection_diag=" in report.format_inline()


def test_1096_neural_declarative_diagnostic_slice():
    if os.environ.get("C4_1096_DIAG") != "1":
        pytest.skip("set C4_1096_DIAG=1 to run the neural diagnostic")

    offset = int(os.environ.get("C4_1096_OFFSET", "0"))
    limit_raw = os.environ.get("C4_1096_LIMIT", "8")
    limit = None if limit_raw.lower() in {"", "none", "all"} else int(limit_raw)
    chunk_size = int(os.environ.get("C4_BATCH_CHUNK", "8"))
    spec_k = _parse_spec_k(os.environ.get("C4_SPEC_K", "0"))
    max_context_window = int(os.environ.get("C4_BATCH_CONTEXT_WINDOW", "512"))
    model_max_seq_len = int(os.environ.get("C4_BATCH_MODEL_MAX_SEQ_LEN", "4096"))
    trace_failures = os.environ.get(
        "C4_1096_TRACE_FAILURES", "1"
    ).strip().lower() not in {"", "0", "false", "no", "off"}
    trace_failure_limit = _parse_trace_limit(
        os.environ.get("C4_1096_TRACE_LIMIT", "8")
    )
    progress_stream = (
        sys.stderr
        if _env_flag("C4_1096_PROGRESS", False)
        else None
    )
    comparison_mode = (
        "strict-first-safe-token"
        if _env_flag("C4_SPEC_FAIL_ON_CORRECTION", False)
        else "final-output"
    )

    rows = run_1096_neural_declarative_diagnostic(
        offset=offset,
        limit=limit,
        chunk_size=chunk_size,
        spec_k=spec_k,
        max_context_window=max_context_window,
        model_max_seq_len=model_max_seq_len,
        trace_failures=trace_failures,
        trace_failure_limit=trace_failure_limit,
        include_band_projection=_env_flag("C4_1096_BAND_PROJECTION_DIAG", True),
        band_min_active_margin=_env_float("C4_1096_BAND_MIN_MARGIN", 0.5),
        band_max_inactive_value=_env_float("C4_1096_BAND_MAX_INACTIVE", 0.2),
        sort_by_steps=_env_flag("C4_1096_SORT_BY_STEPS", False),
        progress_stream=progress_stream,
        comparison_mode=comparison_mode,
    )
    divergences = print_divergence_rows(rows)
    print_diagnostic_summary(rows, mode=comparison_mode)

    if os.environ.get("C4_1096_DIAG_ASSERT", "1") != "0":
        assert divergences == 0, (
            f"{divergences}/{len(rows)} selected 1096 programs diverged"
        )
