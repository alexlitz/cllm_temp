"""Opt-in full-model residual trace for a multi-byte ADD result.

Run with:

    C4_FULL_ADD_TRACE=1 C4_DECLARATIONS_ONLY_BAKE=1 \
    pytest -q c4_release/tests/test_full_model_add_trace.py -s

The diagnostic teacher-forces the correct symbolic VM token stream for:

    IMM 654; PSH; IMM 114; ADD; EXIT

and inspects the residual at the ADD step's AX byte-0 token.  In the
autoregressive layout that position predicts AX byte 1.  The expected ADD
result is 768 (0x00000300), so byte 1 should become 0x03 after the L10
high-byte ADD base result and carry propagation have run.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Iterable, Mapping, Optional, Sequence, TextIO

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE
from neural_vm.embedding import Opcode
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
from neural_vm.verification.symbolic_program import (
    SymbolicDeclarativeProgramRunner,
)
from neural_vm.vm_step import Token, _SetDim


_TRACE_ENV = "C4_FULL_ADD_TRACE"


def _make_bytecode(ops: Sequence[int | tuple[int, int]]) -> list[int]:
    bytecode: list[int] = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append((imm << 8) | int(opcode))
        else:
            bytecode.append(int(op))
    return bytecode


def _build_context_prefix(bytecode: Sequence[int], data: bytes = b"") -> list[int]:
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
    tokens.extend(data)
    tokens.append(Token.DATA_END)
    return tokens


def _append_step(
    context: list[int],
    *,
    pc: int,
    ax: int,
    sp: int,
    bp: int,
    stack0: int,
    mem_addr: int,
    mem_value: int,
) -> int:
    step_start = len(context)
    context.append(Token.REG_PC)
    for i in range(4):
        context.append((pc >> (i * 8)) & 0xFF)
    context.append(Token.REG_AX)
    for i in range(4):
        context.append((ax >> (i * 8)) & 0xFF)
    context.append(Token.REG_SP)
    for i in range(4):
        context.append((sp >> (i * 8)) & 0xFF)
    context.append(Token.REG_BP)
    for i in range(4):
        context.append((bp >> (i * 8)) & 0xFF)
    context.append(Token.STACK0)
    for i in range(4):
        context.append((stack0 >> (i * 8)) & 0xFF)
    context.append(Token.MEM)
    for i in range(4):
        context.append((mem_addr >> (i * 8)) & 0xFF)
    for i in range(4):
        context.append((mem_value >> (i * 8)) & 0xFF)
    context.append(Token.STEP_END)
    return step_start


def _dim_positions(model) -> Mapping[str, int]:
    positions = getattr(model, "dim_positions", None)
    return positions if isinstance(positions, dict) else {}


def _dim(name: str, positions: Mapping[str, int]) -> int:
    return int(positions.get(name, getattr(_SetDim, name)))


def _argmax_band(vec: torch.Tensor, base: int, size: int = 16) -> tuple[int, float]:
    band = vec[base : base + size]
    idx = int(torch.argmax(band).item())
    return idx, float(band[idx].item())


def _byte_from_bands(
    vec: torch.Tensor,
    positions: Mapping[str, int],
    lo: str,
    hi: str,
) -> int:
    lo_idx, _ = _argmax_band(vec, _dim(lo, positions))
    hi_idx, _ = _argmax_band(vec, _dim(hi, positions))
    return lo_idx | (hi_idx << 4)


@dataclass(frozen=True)
class AddResidualSnapshot:
    label: str
    block_index: Optional[int]
    output_byte: int
    alu_byte: int
    output_lo: tuple[int, float]
    output_hi: tuple[int, float]
    alu_lo: tuple[int, float]
    alu_hi: tuple[int, float]
    temp8: float
    temp9: float
    carry1: float
    carry2: float
    carry3: float
    op_add: float
    op_sub: float
    mark_ax: float
    is_byte: float
    h1_ax: float
    byte_index_0: float
    byte_index_1: float
    byte_index_2: float
    delta_output_byte: Optional[int] = None
    delta_output_l1: float = 0.0
    delta_alu_l1: float = 0.0

    def format(self) -> str:
        delta = ""
        if self.delta_output_byte is not None:
            delta = (
                f" d_out={self.delta_output_byte:+d}"
                f" d|OUT|={self.delta_output_l1:.2f}"
                f" d|ALU|={self.delta_alu_l1:.2f}"
            )
        block = "-" if self.block_index is None else str(self.block_index)
        return (
            f"[add-trace] block={block:>2} {self.label:<34} "
            f"OUT=0x{self.output_byte:02x} "
            f"(lo={self.output_lo[0]:x}/{self.output_lo[1]:+.2f} "
            f"hi={self.output_hi[0]:x}/{self.output_hi[1]:+.2f}) "
            f"ALU=0x{self.alu_byte:02x} "
            f"(lo={self.alu_lo[0]:x}/{self.alu_lo[1]:+.2f} "
            f"hi={self.alu_hi[0]:x}/{self.alu_hi[1]:+.2f}) "
            f"T8={self.temp8:+.2f} T9={self.temp9:+.2f} "
            f"C1={self.carry1:+.2f} C2={self.carry2:+.2f} C3={self.carry3:+.2f} "
            f"ADD={self.op_add:+.2f} SUB={self.op_sub:+.2f} "
            f"AX={self.mark_ax:+.2f} BYTE={self.is_byte:+.2f} "
            f"H1AX={self.h1_ax:+.2f} BI0={self.byte_index_0:+.2f} "
            f"BI1={self.byte_index_1:+.2f} BI2={self.byte_index_2:+.2f}"
            f"{delta}"
        )


def _block_width(block) -> Optional[int]:
    ffn = getattr(block, "ffn", None)
    w_up = getattr(ffn, "W_up", None)
    if w_up is None:
        return None
    return int(w_up.shape[0])


def _original_layer_index(block) -> Optional[int]:
    attn = getattr(block, "attn", None)
    return getattr(attn, "layer_idx", None)


def _label_blocks(model) -> dict[int, str]:
    labels: dict[int, str] = {}
    original_to_final: dict[int, int] = {}
    for i, block in enumerate(model.blocks):
        layer_idx = _original_layer_index(block)
        if isinstance(layer_idx, int) and layer_idx not in original_to_final:
            original_to_final[layer_idx] = i
            labels[i] = f"L{layer_idx} main"

    l10 = original_to_final.get(10)
    l11 = original_to_final.get(11, len(model.blocks))
    if l10 is not None:
        post_names = [
            "L10 BinaryOpByteZeroing",
            "L10 AddSubBytePropagation",
            "L10 CarryPropagation b1",
            "L10 CarryPropagation b2",
            "L10 CarryPropagation b3",
            "L10 BitwiseBytePropagation",
            "L10 ComparisonCombine",
            "L10 DivMod composite",
        ]
        post_range = range(l10 + 1, min(l11, l10 + 1 + len(post_names)))
        for offset, block_idx in enumerate(post_range):
            labels[block_idx] = post_names[offset]

    interesting_widths = {
        4: "BinaryOpByteZeroing-like",
        18: "ComparisonCombine-like",
        512: "CarryPropagation-like",
        1536: "AddSub/BitwisePropagation-like",
    }
    for i, block in enumerate(model.blocks):
        if i in labels:
            continue
        layer_idx = _original_layer_index(block)
        width = _block_width(block)
        width_label = interesting_widths.get(width)
        if width_label is not None:
            labels[i] = f"{width_label} (layer_idx={layer_idx})"
        elif width is not None:
            labels[i] = f"block width={width} layer_idx={layer_idx}"
        else:
            labels[i] = f"{type(getattr(block, 'ffn', block)).__name__} layer_idx={layer_idx}"
    return labels


def _snapshot(
    label: str,
    x: torch.Tensor,
    *,
    pos: int,
    positions: Mapping[str, int],
    block_index: Optional[int] = None,
    previous: Optional[torch.Tensor] = None,
) -> AddResidualSnapshot:
    vec = x[0, pos]
    delta_output_byte = None
    delta_output_l1 = 0.0
    delta_alu_l1 = 0.0
    if previous is not None:
        prev = previous[0, pos]
        old_out = _byte_from_bands(prev, positions, "OUTPUT_LO", "OUTPUT_HI")
        new_out = _byte_from_bands(vec, positions, "OUTPUT_LO", "OUTPUT_HI")
        delta_output_byte = new_out - old_out
        out_lo = _dim("OUTPUT_LO", positions)
        out_hi = _dim("OUTPUT_HI", positions)
        alu_lo = _dim("ALU_LO", positions)
        alu_hi = _dim("ALU_HI", positions)
        delta_output_l1 = float(
            (vec[out_lo : out_lo + 16] - prev[out_lo : out_lo + 16]).abs().sum().item()
            + (vec[out_hi : out_hi + 16] - prev[out_hi : out_hi + 16]).abs().sum().item()
        )
        delta_alu_l1 = float(
            (vec[alu_lo : alu_lo + 16] - prev[alu_lo : alu_lo + 16]).abs().sum().item()
            + (vec[alu_hi : alu_hi + 16] - prev[alu_hi : alu_hi + 16]).abs().sum().item()
        )

    carry = _dim("CARRY", positions)
    temp = _dim("TEMP", positions)
    h1 = _dim("H1", positions)
    return AddResidualSnapshot(
        label=label,
        block_index=block_index,
        output_byte=_byte_from_bands(vec, positions, "OUTPUT_LO", "OUTPUT_HI"),
        alu_byte=_byte_from_bands(vec, positions, "ALU_LO", "ALU_HI"),
        output_lo=_argmax_band(vec, _dim("OUTPUT_LO", positions)),
        output_hi=_argmax_band(vec, _dim("OUTPUT_HI", positions)),
        alu_lo=_argmax_band(vec, _dim("ALU_LO", positions)),
        alu_hi=_argmax_band(vec, _dim("ALU_HI", positions)),
        temp8=float(vec[temp + 8].item()),
        temp9=float(vec[temp + 9].item()),
        carry1=float(vec[carry + 1].item()),
        carry2=float(vec[carry + 2].item()),
        carry3=float(vec[carry + 3].item()),
        op_add=float(vec[_dim("OP_ADD", positions)].item()),
        op_sub=float(vec[_dim("OP_SUB", positions)].item()),
        mark_ax=float(vec[_dim("MARK_AX", positions)].item()),
        is_byte=float(vec[_dim("IS_BYTE", positions)].item()),
        h1_ax=float(vec[h1 + 1].item()),
        byte_index_0=float(vec[_dim("BYTE_INDEX_0", positions)].item()),
        byte_index_1=float(vec[_dim("BYTE_INDEX_1", positions)].item()),
        byte_index_2=float(vec[_dim("BYTE_INDEX_2", positions)].item()),
        delta_output_byte=delta_output_byte,
        delta_output_l1=delta_output_l1,
        delta_alu_l1=delta_alu_l1,
    )


def build_teacher_forced_add_context() -> tuple[list[int], int, dict[str, int]]:
    bytecode = _make_bytecode([
        (Opcode.IMM, 654),
        Opcode.PSH,
        (Opcode.IMM, 114),
        Opcode.ADD,
        Opcode.EXIT,
    ])
    symbolic = SymbolicDeclarativeProgramRunner()
    state = symbolic.init_state(bytecode)
    context = _build_context_prefix(bytecode)
    metadata = {
        "bytecode_len": len(bytecode),
        "add_step": -1,
        "add_step_start": -1,
        "target_pos": -1,
        "expected_result": 654 + 114,
        "expected_byte1": ((654 + 114) >> 8) & 0xFF,
        "stack_byte1": (654 >> 8) & 0xFF,
        "ax_byte1": (114 >> 8) & 0xFF,
    }

    while symbolic.step(state):
        step = state.trace[-1]
        stack0 = state.mem_read(state.sp)
        step_start = _append_step(
            context,
            pc=state.pc,
            ax=state.ax,
            sp=state.sp,
            bp=state.bp,
            stack0=stack0,
            mem_addr=step.mem_addr,
            mem_value=step.mem_value,
        )
        if step.opcode == Opcode.ADD:
            metadata["add_step"] = step.step
            metadata["add_step_start"] = step_start
            # Step layout: REG_PC + 4 bytes + REG_AX + AX byte0.
            metadata["target_pos"] = step_start + 6
            break

    if metadata["target_pos"] < 0:
        raise AssertionError("ADD step was not reached in teacher-forced program")
    return context, metadata["target_pos"], metadata


def trace_full_model_add_residual(
    *,
    stream: TextIO = sys.stderr,
) -> list[AddResidualSnapshot]:
    context, target_pos, metadata = build_teacher_forced_add_context()
    model, _layout = compile_full_vm_dynamic(
        alu_mode="efficient",
        declarations_only=True,
        max_seq_len=max(4096, len(context) + 8),
        disk_cache=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    positions = _dim_positions(model)
    labels = _label_blocks(model)
    device = next(model.parameters()).device
    token_ids = torch.tensor([context], dtype=torch.long, device=device)
    print(
        "[add-trace] program=IMM654;PSH;IMM114;ADD;EXIT "
        f"context_len={len(context)} target_pos={target_pos} "
        f"target_token={context[target_pos]} "
        f"expected_result={metadata['expected_result']} "
        f"expected_byte1=0x{metadata['expected_byte1']:02x} "
        f"stack_byte1=0x{metadata['stack_byte1']:02x} "
        f"ax_byte1=0x{metadata['ax_byte1']:02x} "
        f"blocks={len(model.blocks)} d_model={model.d_model}",
        file=stream,
        flush=True,
    )

    snapshots: list[AddResidualSnapshot] = []
    with torch.no_grad():
        x = model.embed(token_ids)
        snapshots.append(
            _snapshot("after embed", x, pos=target_pos, positions=positions)
        )
        print(snapshots[-1].format(), file=stream, flush=True)

        for i, block in enumerate(model.blocks):
            before = x
            x = block(x)
            snap = _snapshot(
                labels.get(i, f"block {i}"),
                x,
                pos=target_pos,
                positions=positions,
                block_index=i,
                previous=before,
            )
            snapshots.append(snap)
            print(snap.format(), file=stream, flush=True)

    print(
        _diagnose_add_trace(snapshots, expected_byte1=metadata["expected_byte1"]),
        file=stream,
        flush=True,
    )
    return snapshots


def _find_snapshot(
    snapshots: Iterable[AddResidualSnapshot],
    label_part: str,
) -> Optional[AddResidualSnapshot]:
    for snapshot in snapshots:
        if label_part in snapshot.label:
            return snapshot
    return None


def _diagnose_add_trace(
    snapshots: Sequence[AddResidualSnapshot],
    *,
    expected_byte1: int,
) -> str:
    addsub = _find_snapshot(snapshots, "L10 AddSubBytePropagation")
    carry1 = _find_snapshot(snapshots, "L10 CarryPropagation b1")
    final = snapshots[-1]

    if addsub is None or carry1 is None:
        return (
            "[add-trace] diagnosis=missing-l10-labels "
            "Could not identify the structural L10 AddSub/carry blocks; inspect rows above."
        )

    if addsub.output_byte != 0x02:
        relay_note = ""
        if (
            addsub.alu_lo[0] != 0x2
            or addsub.alu_lo[1] < 1.5
            or addsub.alu_hi[0] != 0x0
        ):
            relay_note = (
                " upstream_stack_relay=weak-or-ambiguous"
                f" ALU_LO[2]={addsub.alu_lo[1]:+.2f}"
                f" ALU_HI[argmax]={addsub.alu_hi[0]:x}/{addsub.alu_hi[1]:+.2f}"
            )
        return (
            "[add-trace] diagnosis=first-divergence:L10-AddSubBytePropagation "
            f"expected base byte1 0x02 after AddSub, got 0x{addsub.output_byte:02x}; "
            f"OUT_LO[0]={addsub.output_lo[1]:+.2f} OUT_HI[0]={addsub.output_hi[1]:+.2f} "
            f"T8={addsub.temp8:+.2f} H1AX={addsub.h1_ax:+.2f} BI0={addsub.byte_index_0:+.2f}"
            f"{relay_note}"
        )
    if carry1.output_byte != expected_byte1:
        return (
            "[add-trace] diagnosis=first-divergence:L10-CarryPropagation-b1 "
            f"expected carried byte1 0x{expected_byte1:02x}, got 0x{carry1.output_byte:02x}; "
            f"C1={carry1.carry1:+.2f} C3={carry1.carry3:+.2f} T8={carry1.temp8:+.2f}"
        )
    if final.output_byte != expected_byte1:
        first_bad_after_carry = next(
            (
                snapshot
                for snapshot in snapshots
                if snapshot.block_index is not None
                and snapshot.block_index > (carry1.block_index or -1)
                and snapshot.output_byte != expected_byte1
            ),
            final,
        )
        return (
            "[add-trace] diagnosis=first-divergence:post-carry-tail "
            f"first_bad_block={first_bad_after_carry.block_index} "
            f"label={first_bad_after_carry.label!r} "
            f"expected byte1 0x{expected_byte1:02x}, got 0x{first_bad_after_carry.output_byte:02x}"
        )
    return (
        "[add-trace] diagnosis=no-divergence-at-byte1-site "
        f"L10 AddSub, first carry, and final residual all decode byte1=0x{expected_byte1:02x}"
    )


def test_full_model_add_trace_opt_in():
    if os.environ.get(_TRACE_ENV) != "1":
        pytest.skip(f"set {_TRACE_ENV}=1 to run the full-model ADD residual trace")

    snapshots = trace_full_model_add_residual()

    if os.environ.get("C4_FULL_ADD_TRACE_ASSERT") == "1":
        assert snapshots[-1].output_byte == 0x03
