"""Uniform per-step per-layer residual capture API.

Replaces ad-hoc probes (``probe_l14_li_consumer.py``,
``l10_tail_lea_residual_probe.py``, etc.) with a single general tool that
lets agents inspect ``model.blocks[N].forward`` input + output residuals
at any step ``K`` of a program, for an arbitrary list of named residual
dimensions.

CLI
---

    python capture_residual_trace.py \\
        --program "IMM 5; PSH; IMM 5; EQ; EXIT" \\
        --layers 0,5,9,10,11,12 \\
        --step 3 \\
        --dim CMP+0,CMP+1,CMP+2,ALU_LO+0

Optional ``--compare-against oracle`` flag attaches the
:class:`~c4_release.neural_vm.unified_compiler.dim_oracle.ReferenceOracle`
state at the same step and prints the per-dim drift alongside the
observed residual values.

Programmatic API
----------------

The :class:`ResidualTracer` class can be imported into a probe / fix
script::

    from c4_release.tools.capture_residual_trace import ResidualTracer

    tracer = ResidualTracer()           # builds the runner once
    capture = tracer.run("IMM 5; PSH; IMM 5; EQ; EXIT")
    # `capture` exposes: capture.after_layer(layer_idx) -> tensor [1, S, D]
    #                    capture.before_layer(layer_idx) -> tensor [1, S, D]
    #                    capture.dim(name) -> int
    #                    capture.step_rows(step_idx)  -> (lo, hi)
    #
    # Pretty-print a slice:
    print(tracer.format_trace(capture,
                              layers=[0, 5, 9, 10, 11, 12],
                              step=3,
                              dims=["CMP+0", "CMP+1", "ALU_LO+0"]))

Each step in the autoregressive VM emits ``Token.STEP_TOKENS`` (=35)
residual rows (PC + AX + SP + BP + STACK0 + MEM + SE). The tracer
exposes both the *step-row* view (one residual value per row, useful
when the dim is row-local like ``OUTPUT_LO`` at the AX marker) and the
*aggregate* view (max over the step's rows, useful when the dim is
broadcast across rows like ``CMP+0``).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PROJ_ROOT = os.path.dirname(REPO_ROOT)
for _p in (PROJ_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from c4_release.neural_vm.embedding import Opcode  # noqa: E402
from c4_release.neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from c4_release.neural_vm.vm_step import Token  # noqa: E402

try:  # Oracle is optional; the import is heavy.
    from c4_release.neural_vm.unified_compiler.dim_oracle import (  # noqa: E402
        ReferenceOracle,
        project_state_to_residual,
    )
    _HAS_ORACLE = True
except Exception:  # pragma: no cover - oracle is best-effort
    ReferenceOracle = None  # type: ignore[assignment]
    project_state_to_residual = None  # type: ignore[assignment]
    _HAS_ORACLE = False


STEP_TOKENS = Token.STEP_TOKENS


# ---------------------------------------------------------------------------
# Program parsing
# ---------------------------------------------------------------------------


def parse_program(text: str) -> List[int]:
    """Parse a ``"IMM 5; PSH; IMM 5; EQ; EXIT"``-style string into bytecode.

    Each instruction is ``MNEMONIC [IMM]`` where ``MNEMONIC`` is any
    attribute of :class:`Opcode`. The immediate is decimal or
    ``0x``-prefixed hex. Instructions are separated by ``;`` or newlines.
    """

    out: List[int] = []
    chunks = [chunk.strip() for chunk in text.replace("\n", ";").split(";")]
    for chunk in chunks:
        if not chunk:
            continue
        parts = chunk.split()
        mnemonic = parts[0].upper()
        if not hasattr(Opcode, mnemonic):
            raise ValueError(f"Unknown opcode {mnemonic!r} in program")
        opcode = getattr(Opcode, mnemonic)
        if len(parts) > 1:
            imm_text = parts[1]
            imm = int(imm_text, 0)
            out.append(opcode | (imm << 8))
        else:
            out.append(opcode)
    return out


def parse_dim_list(text: str) -> List[str]:
    """Split a ``"CMP+0,CMP+1,ALU_LO+0"`` argument into a clean list."""

    return [d.strip() for d in text.split(",") if d.strip()]


def parse_layer_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_dim(dim_positions: Mapping[str, int], dim_spec: str) -> int:
    """Resolve ``"NAME"`` or ``"NAME+offset"`` to a residual position."""

    if "+" in dim_spec:
        base, off = dim_spec.split("+", 1)
        base = base.strip()
        off = int(off.strip(), 0)
    else:
        base, off = dim_spec.strip(), 0
    if base not in dim_positions:
        raise KeyError(f"Unknown dim base {base!r} in dim_positions")
    return dim_positions[base] + off


# ---------------------------------------------------------------------------
# Capture object
# ---------------------------------------------------------------------------


@dataclass
class ResidualCapture:
    """One forward pass worth of per-layer residuals.

    ``after_layer[N]`` is the tensor returned by ``model.blocks[N].forward``
    (shape ``[1, seq_len, d_model]``). ``before_layer[N]`` is the tensor
    fed into the same block (= ``after_layer[N-1]`` for N >= 1, and the
    embedding output for N == 0).
    """

    program: List[int]
    dim_positions: Mapping[str, int]
    seq_len: int
    after_layer: Dict[int, torch.Tensor] = field(default_factory=dict)
    before_layer: Dict[int, torch.Tensor] = field(default_factory=dict)
    result: object = None

    def num_steps(self) -> int:
        return self.seq_len // STEP_TOKENS

    def step_rows(self, step_idx: int) -> Tuple[int, int]:
        """Inclusive-exclusive ``(lo, hi)`` row range for a step.

        Returns ``(seq_len, seq_len)`` (empty range) when the step is
        beyond the captured sequence — caller should treat that as "step
        not present".
        """

        lo = step_idx * STEP_TOKENS
        if lo >= self.seq_len:
            return self.seq_len, self.seq_len
        hi = min(lo + STEP_TOKENS, self.seq_len)
        return lo, hi

    def dim(self, dim_spec: str) -> int:
        return resolve_dim(self.dim_positions, dim_spec)

    def value_at(
        self,
        layer_idx: int,
        dim_spec: str,
        step_idx: int,
        side: str = "out",
        reducer: str = "max",
    ) -> float:
        """Pull one dim's value from the requested layer/step.

        Parameters
        ----------
        side : {"in", "out"}
            "in" reads the block's input residual (= prev block's output);
            "out" reads the block's output residual.
        reducer : {"max", "mean", "row<row_idx>"}
            How to collapse the 35 rows of a step into one number. The
            ``row<N>`` form picks a specific in-step row (0..34).
        """

        bank = self.before_layer if side == "in" else self.after_layer
        if layer_idx not in bank:
            raise KeyError(f"Layer {layer_idx} not captured (side={side})")
        arr = bank[layer_idx]
        d = self.dim(dim_spec)
        lo, hi = self.step_rows(step_idx)
        if hi <= lo or arr.shape[1] <= lo:
            return float("nan")
        hi = min(hi, arr.shape[1])
        if reducer.startswith("row"):
            offset = int(reducer[3:])
            pos = lo + offset
            if pos >= hi:
                return float("nan")
            return float(arr[0, pos, d].item())
        slice_ = arr[0, lo:hi, d]
        if reducer == "max":
            return float(slice_.max().item())
        if reducer == "mean":
            return float(slice_.mean().item())
        raise ValueError(f"Unknown reducer {reducer!r}")


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class ResidualTracer:
    """High-level API for capturing per-layer residuals over a program.

    Construct once, reuse across multiple programs. The runner / model
    build is the slow part (~10s); a single :class:`ResidualTracer`
    amortises that cost.

    Example
    -------

    >>> tracer = ResidualTracer()
    >>> capture = tracer.run("IMM 5; PSH; IMM 5; EQ; EXIT")
    >>> capture.value_at(layer_idx=10, dim_spec="CMP+1", step_idx=3)
    1.0
    """

    def __init__(
        self,
        runner: Optional[AutoregressiveVMRunner] = None,
        trust_neural_alu: bool = True,
        pure_neural: bool = True,
    ):
        if runner is None:
            with contextlib.redirect_stdout(io.StringIO()):
                runner = AutoregressiveVMRunner(
                    trust_neural_alu=trust_neural_alu,
                    pure_neural=pure_neural,
                )
        self.runner = runner
        self.model = runner.model
        self.dim_positions = self.model.embed._dim_positions
        self.num_blocks = len(self.model.blocks)

    # -- core: run + capture ------------------------------------------------

    def run(
        self,
        program: str | Sequence[int],
        layers: Optional[Sequence[int]] = None,
        max_steps: int = 64,
    ) -> ResidualCapture:
        """Execute ``program`` and return a :class:`ResidualCapture`.

        If ``layers`` is None, every block is hooked.
        """

        bc = (
            parse_program(program) if isinstance(program, str) else list(program)
        )
        layer_set = (
            set(range(self.num_blocks)) if layers is None else set(layers)
        )

        # Reset per-run state — match probe_l14 conventions.
        runner = self.runner
        runner._func_call_handlers = {}
        runner._syscall_handlers = {}
        runner._memory = {}
        runner._mem_history = {}
        runner._mem_access_order = []

        captures: List[Dict[str, torch.Tensor]] = []

        def embed_hook(module, inputs, output):
            captures.append({"_token_ids": inputs[0].detach().clone()})

        def pre_hook(layer_idx: int):
            def fn(module, inputs):
                if captures:
                    captures[-1][f"in_L{layer_idx}"] = inputs[0].detach().clone()
            return fn

        def post_hook(layer_idx: int):
            def fn(module, inputs, output):
                if captures:
                    captures[-1][f"out_L{layer_idx}"] = output.detach().clone()
            return fn

        handles = [self.model.embed.register_forward_hook(embed_hook)]
        for li in sorted(layer_set):
            if li < 0 or li >= self.num_blocks:
                continue
            handles.append(
                self.model.blocks[li].register_forward_pre_hook(pre_hook(li))
            )
            handles.append(
                self.model.blocks[li].register_forward_hook(post_hook(li))
            )

        result = None
        try:
            try:
                result = runner.run(bc, b"", max_steps=max_steps)
            except Exception as exc:
                # Don't abort the capture — the residuals up to the failure
                # are usually what the caller wants.
                result = f"<runner raised: {exc!r}>"
        finally:
            for h in handles:
                h.remove()

        # Pick the capture with the longest block-output sequence. KV-cache
        # incremental forwards only deliver 1 token per call; we want the
        # full-sequence forward (typically the first call, before the
        # KV-cache builds up).
        def _full_seq(cap: Dict[str, torch.Tensor]) -> int:
            for k, v in cap.items():
                if k.startswith("out_L"):
                    return v.shape[1]
            ids = cap.get("_token_ids")
            return ids.shape[1] if ids is not None else 0

        best = None
        best_len = 0
        for cap in captures:
            n = _full_seq(cap)
            if n > best_len:
                best = cap
                best_len = n
        if best is None:
            raise RuntimeError("No forward calls were captured")

        seq_len = best_len
        cap_obj = ResidualCapture(
            program=bc,
            dim_positions=self.dim_positions,
            seq_len=seq_len,
            result=result,
        )
        for li in sorted(layer_set):
            in_key, out_key = f"in_L{li}", f"out_L{li}"
            if in_key in best:
                cap_obj.before_layer[li] = best[in_key]
            if out_key in best:
                cap_obj.after_layer[li] = best[out_key]
        return cap_obj

    # -- formatting ---------------------------------------------------------

    def format_trace(
        self,
        capture: ResidualCapture,
        layers: Sequence[int],
        step: int,
        dims: Sequence[str],
        reducer: str = "max",
        compare_against: Optional[str] = None,
    ) -> str:
        """Pretty-print the per-layer residual values for one step.

        Only the dims in ``dims`` are printed (the full residual is too
        noisy). The first layer where any output dim diverges from the
        input is annotated with ``FIRST DIVERGENCE``.
        """

        prog = capture.program
        opcode_label = _opcode_label_for_step(prog, step)
        lo, hi = capture.step_rows(step)
        lines = [
            f"step={step} ({opcode_label})  rows=[{lo}..{hi})  "
            f"seq_len={capture.seq_len}  reducer={reducer}"
        ]

        oracle_values: Dict[str, float] = {}
        if compare_against == "oracle":
            oracle_values = self._oracle_values_for_step(prog, step, dims)
            if oracle_values:
                parts = [f"{d}={oracle_values[d]:.2f}" for d in dims if d in oracle_values]
                if parts:
                    lines.append("oracle:        " + "  ".join(parts))

        prev_out_vals: Optional[List[float]] = None
        flagged = False
        for li in layers:
            in_vals = []
            out_vals = []
            for d in dims:
                try:
                    in_vals.append(capture.value_at(li, d, step, "in", reducer))
                except KeyError:
                    in_vals.append(float("nan"))
                try:
                    out_vals.append(capture.value_at(li, d, step, "out", reducer))
                except KeyError:
                    out_vals.append(float("nan"))

            in_str = "  ".join(f"{d}={v:.2f}" for d, v in zip(dims, in_vals))
            out_str = "  ".join(f"{d}={v:.2f}" for d, v in zip(dims, out_vals))
            lines.append(f"layer {li:2d} in:  {in_str}")

            tag = ""
            if not flagged:
                # First layer whose output diverges from its input by >0.05
                # on any tracked dim (skipping NaN cells).
                def _diff(a: float, b: float) -> float:
                    if math.isnan(a) or math.isnan(b):
                        return 0.0
                    return abs(a - b)
                if any(_diff(a, b) > 0.05 for a, b in zip(in_vals, out_vals)):
                    tag = "  # FIRST DIVERGENCE"
                    flagged = True
            lines.append(f"layer {li:2d} out: {out_str}{tag}")
            prev_out_vals = out_vals

        if oracle_values:
            # Per-layer drift vs oracle for the final (deepest) requested layer.
            li_last = layers[-1]
            drift = []
            for d in dims:
                try:
                    obs = capture.value_at(li_last, d, step, "out", reducer)
                except KeyError:
                    continue
                if d in oracle_values:
                    drift.append(f"{d}: obs={obs:.2f} expect={oracle_values[d]:.2f}")
            if drift:
                lines.append(
                    f"drift @L{li_last} vs oracle:  " + "  |  ".join(drift)
                )
        return "\n".join(lines)

    def _oracle_values_for_step(
        self,
        program: Sequence[int],
        step: int,
        dims: Sequence[str],
    ) -> Dict[str, float]:
        if not _HAS_ORACLE:
            return {}
        try:
            oracle = ReferenceOracle(list(program))
            state = oracle.state_at_step(step)
            projected = project_state_to_residual(state, per_token=False)
        except Exception:
            return {}
        # The oracle's projection is keyed by ``"NAME+offset"`` so we can
        # look up requested dims directly. We collapse all positions to a
        # single value (max across rows).
        out: Dict[str, float] = {}
        for d in dims:
            best = 0.0
            found = False
            for (pos, key), val in projected.items():
                if key == d:
                    found = True
                    if val > best:
                        best = val
            if found:
                out[d] = best
        return out


# ---------------------------------------------------------------------------
# Step labeling
# ---------------------------------------------------------------------------


_OPCODE_NAME_BY_VALUE: Dict[int, str] = {}
for _name in dir(Opcode):
    if _name.startswith("_"):
        continue
    val = getattr(Opcode, _name)
    if isinstance(val, int) and val not in _OPCODE_NAME_BY_VALUE:
        _OPCODE_NAME_BY_VALUE[val] = _name


def _opcode_label_for_step(program: Sequence[int], step_idx: int) -> str:
    if step_idx < 0 or step_idx >= len(program):
        return "post-EXIT"
    word = program[step_idx]
    op = word & 0xFF
    imm = word >> 8
    mnemonic = _OPCODE_NAME_BY_VALUE.get(op, f"OP_{op}")
    if imm:
        return f"{mnemonic} {imm} dispatch"
    return f"{mnemonic} dispatch"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-step per-layer residual capture for the neural VM",
    )
    ap.add_argument(
        "--program",
        required=True,
        help='Program text, e.g. "IMM 5; PSH; IMM 5; EQ; EXIT"',
    )
    ap.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices, e.g. 0,5,9,10,11,12",
    )
    ap.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step index (0-based) of interest",
    )
    ap.add_argument(
        "--dim",
        required=True,
        help='Comma-separated dim list, e.g. "CMP+0,CMP+1,ALU_LO+0"',
    )
    ap.add_argument(
        "--reducer",
        default="max",
        help="How to collapse step's 35 rows into one number "
        '("max", "mean", or "row<N>" for a specific in-step row).',
    )
    ap.add_argument(
        "--compare-against",
        choices=["oracle"],
        default=None,
        help="Optional reference state; 'oracle' uses ReferenceOracle.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="VM step cap for the run (default 64).",
    )
    args = ap.parse_args(argv)

    program = parse_program(args.program)
    layers = parse_layer_list(args.layers)
    dims = parse_dim_list(args.dim)

    tracer = ResidualTracer()
    capture = tracer.run(program, layers=layers, max_steps=args.max_steps)
    print(
        tracer.format_trace(
            capture,
            layers=layers,
            step=args.step,
            dims=dims,
            reducer=args.reducer,
            compare_against=args.compare_against,
        )
    )
    print(f"\nrunner.run -> {capture.result!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
