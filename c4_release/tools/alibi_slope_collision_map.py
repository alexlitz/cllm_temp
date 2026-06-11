#!/usr/bin/env python3
"""ALiBi-slope collision map — diagnostic for the operand-relay bug class.

WHY THIS EXISTS
---------------
The attention DSL covers Q/K/V/O via ``DeclarativeAttentionHeadSpec``
(plus ``AP`` / ``AO``), and the spec carries an ``alibi_slope`` field.
But ``alibi_slope=None`` means "the op writes ``attn.alibi_slopes[...]``
itself imperatively" — and *nothing* checks whether two different ops
write the SAME (block, head) slope. When they do, the later write
silently clobbers the earlier one.

That is exactly the bug behind the operand-relay transmission failure:
``make_layer10_residual_alibi_slopes_op`` (alu_ops.py, phase ~999.1)
imperatively overwrites ``alibi_slopes[3]`` / ``[4]`` on block 10's
shared physical attention AFTER an earlier op already set them, muting
the relay head. See
``project_operand_gather_hybrid_encoding_is_cmp_alu_root.md`` Wall 2.

WHAT THIS TOOL DOES
-------------------
It instruments the *real* model build (the same path the smoke gate
uses — ``GroundTruthProbe.build`` => ``set_vm_weights`` via the
declarative compiler) so that EVERY write into any block's
``alibi_slopes`` buffer is recorded with:

  * the (physical block, head_idx) it targets,
  * the value written,
  * the op / source file:line / function that wrote it (from the call
    stack at write time).

It then prints (and, with ``--md PATH``, writes) a per-head table and
flags every (block, head) where >= 2 *distinct* writer sources set the
slope — a COLLISION, the relay-bug class.

The model is never mutated by this tool beyond the normal build; the
instrumentation only *observes* writes (it forwards each write through
the original implementation unchanged).

USAGE
-----
    python c4_release/tools/alibi_slope_collision_map.py           # text
    python c4_release/tools/alibi_slope_collision_map.py --json
    python c4_release/tools/alibi_slope_collision_map.py \
        --md c4_release/docs/ALIBI_SLOPE_COLLISION_MAP_2026_06_11.md

Exit codes:
  0  built + recorded successfully (collisions are reported, not fatal)
  2  build / IO error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)  # .../c4_release (the package root)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Pin the smoke-gate execution path defensively (mirrors probe_groundtruth).
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"


@dataclass
class SlopeWrite:
    """One observed write into a block's ``alibi_slopes`` buffer."""

    block: int
    head: Optional[int]  # None = whole-buffer write (fill_ / slice)
    value: Optional[float]
    source: str  # "file:line (func)" attributed to the writing op
    op_hint: str  # best-effort op / function name
    op_name: str = "<unknown>"  # Operation.name (from the live dispatcher frame)


@dataclass
class HeadRecord:
    block: int
    head: int
    final_value: Optional[float] = None
    writes: List[SlopeWrite] = field(default_factory=list)

    @property
    def distinct_sources(self) -> List[str]:
        seen: List[str] = []
        for w in self.writes:
            if w.source not in seen:
                seen.append(w.source)
        return seen

    @property
    def distinct_ops(self) -> List[str]:
        seen: List[str] = []
        for w in self.writes:
            if w.op_name not in seen:
                seen.append(w.op_name)
        return seen

    @property
    def is_collision(self) -> bool:
        return len(self.distinct_sources) >= 2

    @property
    def is_cross_op(self) -> bool:
        """True when >= 2 DISTINCT ops write this slope — the relay-bug class.

        A single op that does ``fill_`` then per-head overrides counts as
        ONE op (benign, intentional intra-op default+override). Only when
        two different ``Operation``s touch the same physical (block, head)
        can a later op silently clobber an earlier op's slope.
        """
        real_ops = [o for o in self.distinct_ops if o != "<unknown>"]
        return len(real_ops) >= 2

    @property
    def is_value_changing(self) -> bool:
        """True when distinct writers disagree on the value (a real clobber).

        Redundant collisions (every writer sets the same value) are
        harmless; value-changing ones mean the final slope depends on
        write ORDER — the dangerous case.
        """
        vals = {round(w.value, 9) for w in self.writes if w.value is not None}
        return len(vals) >= 2


# Files we treat as "framework plumbing", not the originating op, when
# attributing a write to a source. The collision map wants the OP that
# decided the slope, not the generic lowering helper it flowed through.
_PLUMBING_SUFFIXES = (
    "unified_compiler/primitives.py",
    "unified_compiler/ir.py",
    "unified_compiler/layer_compiler.py",
    "unified_compiler/building_blocks_dsl.py",
    "unified_compiler/wide_alu_dsl.py",
    "attention_head_allocator.py",
    "tools/alibi_slope_collision_map.py",
    "vm_step.py",  # buffer init lives here; not an op decision
)


def _relpath(fn: str) -> str:
    """Best-effort path relative to the repo root that contains c4_release/."""
    try:
        return str(
            Path(fn).resolve().relative_to(Path(_PKG_ROOT).resolve().parent)
        )
    except Exception:
        idx = fn.find("c4_release/")
        return fn[idx:] if idx >= 0 else fn


def _attribute(stack: traceback.StackSummary) -> Tuple[str, str]:
    """Pick the originating op frame from a write's call stack.

    Returns ``(source, op_hint)`` where ``source`` is
    ``"<relpath>:<lineno> (<func>)"`` for the deepest non-plumbing frame
    and ``op_hint`` is that frame's function name. Falls back to the
    deepest non-tool frame when every frame is plumbing.
    """
    for frame in reversed(stack):
        fn = frame.filename
        if any(fn.endswith(suf) for suf in _PLUMBING_SUFFIXES):
            continue
        return f"{_relpath(fn)}:{frame.lineno} ({frame.name})", frame.name
    for frame in reversed(stack):
        if frame.filename.endswith("tools/alibi_slope_collision_map.py"):
            continue
        return f"{_relpath(frame.filename)}:{frame.lineno} ({frame.name})", frame.name
    return "<unknown>", "<unknown>"


class _Recorder:
    """Instruments ``alibi_slopes`` buffer writes during model build."""

    def __init__(self) -> None:
        # id(tensor) -> block_idx (resolved post-build; -1 until then).
        self.tag: Dict[int, int] = {}
        self.writes: List[SlopeWrite] = []
        self._orig_register_buffer = None
        self._orig_setitem = None
        self._orig_fill = None

    def install(self) -> None:
        import torch
        from torch import nn

        rec = self
        orig_register_buffer = nn.Module.register_buffer

        def patched_register_buffer(self, name, tensor, persistent=True):  # noqa: ANN001
            res = orig_register_buffer(self, name, tensor, persistent=persistent)
            if name == "alibi_slopes" and tensor is not None:
                buf = getattr(self, name)
                rec.tag[id(buf)] = -1
            return res

        nn.Module.register_buffer = patched_register_buffer
        self._orig_register_buffer = orig_register_buffer

        orig_setitem = torch.Tensor.__setitem__

        def patched_setitem(self, idx, value):  # noqa: ANN001
            if id(self) in rec.tag:
                rec._record(self, idx, value)
            return orig_setitem(self, idx, value)

        torch.Tensor.__setitem__ = patched_setitem
        self._orig_setitem = orig_setitem

        orig_fill = torch.Tensor.fill_

        def patched_fill(self, value):  # noqa: ANN001
            if id(self) in rec.tag:
                rec._record(self, slice(None), value)
            return orig_fill(self, value)

        torch.Tensor.fill_ = patched_fill
        self._orig_fill = orig_fill

    def uninstall(self) -> None:
        import torch
        from torch import nn

        if self._orig_register_buffer is not None:
            nn.Module.register_buffer = self._orig_register_buffer
        if self._orig_setitem is not None:
            torch.Tensor.__setitem__ = self._orig_setitem
        if self._orig_fill is not None:
            torch.Tensor.fill_ = self._orig_fill

    def _record(self, tensor, idx, value) -> None:
        head = self._idx_to_head(idx)
        val = self._coerce_value(value)
        source, op_hint = _attribute(traceback.extract_stack())
        op_name = self._op_name_from_live_frames()
        # Store the tensor id; block resolved after build.
        self.writes.append(
            SlopeWrite(
                block=id(tensor),  # temporary: holds tensor id until resolved
                head=head,
                value=val,
                source=source,
                op_hint=op_hint,
                op_name=op_name,
            )
        )

    @staticmethod
    def _op_name_from_live_frames() -> str:
        """Walk live frames for the dispatcher's ``op.name`` (the Operation).

        The bake/lowering dispatcher (``dispatch_operation_bake`` /
        ``_dispatch_operation_ir`` / the per-op run loop) carries the
        active ``Operation`` in a local named ``op``. Reading it lets the
        collision map distinguish a single op's intentional fill+override
        from a genuine cross-op clobber.
        """
        frame = sys._getframe(0)
        op_name = "<unknown>"
        # Walk outward; the OUTERMOST frame holding an ``op`` with a
        # ``.name`` is the dispatcher driving this bake.
        while frame is not None:
            loc = frame.f_locals
            cand = loc.get("op", None)
            if cand is not None:
                name = getattr(cand, "name", None)
                if isinstance(name, str) and name:
                    op_name = name  # keep walking; prefer outermost dispatcher
            frame = frame.f_back
        return op_name

    @staticmethod
    def _idx_to_head(idx) -> Optional[int]:
        if isinstance(idx, bool):
            return None
        if isinstance(idx, int):
            return int(idx)
        try:
            import torch

            if isinstance(idx, torch.Tensor) and idx.numel() == 1:
                return int(idx.item())
        except Exception:
            pass
        return None  # slice / fancy index = whole-buffer-ish

    @staticmethod
    def _coerce_value(value) -> Optional[float]:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return float(value.item()) if value.numel() == 1 else None
            return float(value)
        except Exception:
            return None


def _build_cold_model():
    """Cold-compile the smoke-gate model so every slope write is observable.

    The cached build path (``GroundTruthProbe.build`` /
    ``AutoregressiveVMRunner``) short-circuits the bake to a ``torch.load``
    of a pickled model when the disk cache is warm — bypassing the
    imperative ``alibi_slopes[...] = v`` writes entirely. To attribute
    slope writes we MUST run the bake, so we call
    ``compile_full_vm_dynamic`` directly with ``disk_cache=False`` and the
    exact smoke-gate kwargs.

    Smoke-gate kwargs come from ``AutoregressiveVMRunner(pure_neural=True,
    trust_neural_alu=True, spec_k=0)`` => ``alu_mode='efficient'`` (because
    ``trust_neural_alu=True``), default heads/ffn/seq, all IO/MoE flags
    off. This is a read-only build — the model is inspected for final
    slope values and discarded.
    """
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    from neural_vm.vm_step import DEFAULT_N_HEADS, DEFAULT_FFN_HIDDEN

    model, _layout = compile_full_vm_dynamic(
        enable_conversational_io=False,
        enable_neural_io_think_protocol=False,
        alu_mode="efficient",  # trust_neural_alu=True => efficient
        n_heads=DEFAULT_N_HEADS,
        ffn_hidden=DEFAULT_FFN_HIDDEN,
        max_seq_len=4096,
        enable_moe_routing=False,
        strict=False,
        disk_cache=False,  # force the bake to run so writes are observed
    )
    return model


def build_records(recorder: _Recorder, model) -> Dict[Tuple[int, int], HeadRecord]:
    """Fold recorded writes into per-(block, head) records + final values."""
    id_to_block: Dict[int, int] = {}
    block_nheads: Dict[int, int] = {}
    for bi, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        buf = getattr(attn, "alibi_slopes", None) if attn is not None else None
        if buf is not None:
            id_to_block[id(buf)] = bi
            block_nheads[bi] = int(buf.shape[0])

    records: Dict[Tuple[int, int], HeadRecord] = {}
    for w in recorder.writes:
        block = id_to_block.get(w.block, -1)
        if block < 0:
            continue  # write to a buffer not present on the final model
        if w.head is None:
            for h in range(block_nheads.get(block, 0)):
                key = (block, h)
                rec = records.setdefault(key, HeadRecord(block=block, head=h))
                rec.writes.append(
                    SlopeWrite(block, h, w.value, w.source, w.op_hint, w.op_name)
                )
        else:
            key = (block, w.head)
            rec = records.setdefault(key, HeadRecord(block=block, head=w.head))
            rec.writes.append(
                SlopeWrite(block, w.head, w.value, w.source, w.op_hint, w.op_name)
            )

    for (block, head), rec in records.items():
        attn = getattr(model.blocks[block], "attn", None)
        buf = getattr(attn, "alibi_slopes", None) if attn is not None else None
        if buf is not None and 0 <= head < buf.shape[0]:
            rec.final_value = float(buf[head].item())
    return records


def _format_writes(rec: HeadRecord) -> str:
    parts = []
    for w in rec.writes:
        v = "?" if w.value is None else f"{w.value:g}"
        parts.append(f"{v}@{w.op_name}:{w.source}")
    return " ; ".join(parts)


def _collision_class(rec: HeadRecord) -> str:
    """Human-readable severity class for a collision record."""
    if rec.is_cross_op and rec.is_value_changing:
        return "CROSS-OP CLOBBER"
    if rec.is_cross_op:
        return "cross-op (redundant)"
    if rec.is_value_changing:
        return "same-op fill+override"
    return "same-op redundant"


def render_markdown(
    records: Dict[Tuple[int, int], HeadRecord],
    model_summary: str,
) -> str:
    collisions = sorted(
        (r for r in records.values() if r.is_collision),
        key=lambda r: (r.block, r.head),
    )
    lines: List[str] = []
    lines.append("# ALiBi Slope Collision Map — 2026-06-11")
    lines.append("")
    lines.append(
        "Generated by `c4_release/tools/alibi_slope_collision_map.py`. "
        "Records every write into each physical block's `alibi_slopes` "
        "buffer during the smoke-gate model build "
        "(`GroundTruthProbe.build` => `set_vm_weights`), attributing each "
        "write to the originating op via the call stack. A **collision** "
        "is any `(block, head)` written by >= 2 distinct sources — the "
        "operand-relay-bug class (a later op silently clobbers an earlier "
        "op's slope on a shared physical attention block)."
    )
    lines.append("")
    lines.append(model_summary)
    cross_op = [r for r in collisions if r.is_cross_op]
    clobbers = [r for r in collisions if r.is_cross_op and r.is_value_changing]
    lines.append("")
    lines.append(
        f"- Total `(block, head)` slope slots written: **{len(records)}**"
    )
    lines.append(
        f"- Collisions (>= 2 distinct writers): **{len(collisions)}**"
    )
    lines.append(
        f"- Cross-op collisions (>= 2 distinct OPS write the slot): "
        f"**{len(cross_op)}**"
    )
    lines.append(
        f"- **CROSS-OP CLOBBERS (>= 2 ops AND value-changing — the "
        f"relay-bug class): {len(clobbers)}**"
    )
    lines.append("")
    lines.append(
        "Severity classes: a single op that does `alibi_slopes.fill_(d)` "
        "then per-head overrides counts as ONE op (an intentional "
        "default+override, benign). A **cross-op clobber** is when two "
        "*different* `Operation`s write the same physical `(block, head)` "
        "AND disagree on the value — the later op silently changes the "
        "earlier op's slope (the L10/operand-relay class). Cross-op "
        "*redundant* collisions (two ops, same value) are latent risks: "
        "they work today only because the values happen to agree."
    )
    lines.append("")

    lines.append("## Cross-op CLOBBERS (the relay-bug class)")
    lines.append("")
    if not clobbers:
        lines.append("_No value-changing cross-op clobbers detected._")
    else:
        lines.append(
            "| Block | Head | Final | Ops | Writers (value @ op : source) |"
        )
        lines.append(
            "|------:|-----:|------:|-----|-------------------------------|"
        )
        for r in clobbers:
            fv = "?" if r.final_value is None else f"{r.final_value:g}"
            writers = "<br>".join(
                f"`{('?' if w.value is None else format(w.value,'g'))}` "
                f"@ `{w.op_name}` : {w.source}"
                for w in r.writes
            )
            ops = ", ".join(f"`{o}`" for o in r.distinct_ops if o != "<unknown>")
            lines.append(
                f"| {r.block} | {r.head} | {fv} | {ops} | {writers} |"
            )
    lines.append("")

    lines.append("## All flagged collisions (>= 2 distinct writers)")
    lines.append("")
    if not collisions:
        lines.append("_No collisions detected._")
    else:
        lines.append(
            "| Block | Head | Final | Class | Writers (value @ op : source) |"
        )
        lines.append(
            "|------:|-----:|------:|-------|-------------------------------|"
        )
        for r in collisions:
            fv = "?" if r.final_value is None else f"{r.final_value:g}"
            writers = "<br>".join(
                f"`{('?' if w.value is None else format(w.value,'g'))}` "
                f"@ `{w.op_name}` : {w.source}"
                for w in r.writes
            )
            lines.append(
                f"| {r.block} | {r.head} | {fv} | "
                f"{_collision_class(r)} | {writers} |"
            )
    lines.append("")

    # The logical L10 ALiBi op (alu_ops.make_layer10_residual_alibi_slopes_op)
    # writes physical block 11's attn (logical L10 -> physical block 11 after
    # _expand_wrapper_blocks; L8 +1 shifts every later block). Highlight the
    # slots it clobbers regardless of which physical block they land on.
    l10 = [
        r
        for r in collisions
        if any(
            "make_layer10_residual_alibi_slopes" in w.op_name
            or "layer10_residual_alibi_slopes" in w.op_name
            or "alu_ops.py:18" in w.source
            for w in r.writes
        )
    ]
    if l10:
        lines.append(
            "### The L10 operand-relay clobber "
            "(`make_layer10_residual_alibi_slopes_op`)"
        )
        lines.append("")
        lines.append(
            "Logical L10 maps to a physical block after "
            "`_expand_wrapper_blocks`. The op `fill`s nothing — it writes "
            "individual head slopes that an earlier op (the L9/relay heads) "
            "already set, silently muting the operand relay (Wall 2 of "
            "`project_operand_gather_hybrid_encoding_is_cmp_alu_root.md`)."
        )
        lines.append("")
        for r in sorted(l10, key=lambda x: (x.block, x.head)):
            fv = "?" if r.final_value is None else f"{r.final_value:g}"
            lines.append(
                f"- **block {r.block} head {r.head}** final=`{fv}` "
                f"({_collision_class(r)}): " + _format_writes(r)
            )
        lines.append("")

    lines.append("## Full per-head slope-writer table")
    lines.append("")
    lines.append("| Block | Head | Final | Class | Writers (value @ op : source) |")
    lines.append("|------:|-----:|------:|-------|-------------------------------|")
    for (block, head), r in sorted(records.items()):
        fv = "?" if r.final_value is None else f"{r.final_value:g}"
        cls = _collision_class(r) if r.is_collision else "single-writer"
        srcs = "<br>".join(
            f"`{('?' if w.value is None else format(w.value,'g'))}` "
            f"@ `{w.op_name}` : {w.source}"
            for w in r.writes
        )
        lines.append(f"| {block} | {head} | {fv} | {cls} | {srcs} |")
    lines.append("")
    lines.append(
        "_Class legend: `CROSS-OP CLOBBER` = >= 2 ops, value-changing "
        "(the relay-bug class); `cross-op (redundant)` = >= 2 ops, same "
        "value (latent risk); `same-op fill+override` = one op's "
        "intentional default+override; `single-writer` = exactly one write._"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--md", default=None, help="write the markdown report to PATH")
    args = ap.parse_args(argv)

    recorder = _Recorder()
    recorder.install()
    try:
        model = _build_cold_model()
    except Exception as exc:  # pragma: no cover - build failures
        recorder.uninstall()
        print(f"alibi_slope_collision_map: build failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    finally:
        recorder.uninstall()

    records = build_records(recorder, model)
    n_blocks = len(model.blocks)
    n_alibi_blocks = sum(
        1
        for b in model.blocks
        if getattr(getattr(b, "attn", None), "alibi_slopes", None) is not None
    )
    model_summary = (
        f"Model: {n_blocks} physical blocks, {n_alibi_blocks} with an "
        f"`alibi_slopes` buffer (ALiBi / hybrid<3 attention)."
    )
    collisions = sorted(
        (r for r in records.values() if r.is_collision),
        key=lambda r: (r.block, r.head),
    )
    cross_op = [r for r in collisions if r.is_cross_op]
    clobbers = [r for r in collisions if r.is_cross_op and r.is_value_changing]

    if args.json:
        payload = {
            "n_blocks": n_blocks,
            "n_alibi_blocks": n_alibi_blocks,
            "n_slots_written": len(records),
            "n_collisions": len(collisions),
            "n_cross_op_collisions": len(cross_op),
            "n_cross_op_clobbers": len(clobbers),
            "collisions": [
                {
                    "block": r.block,
                    "head": r.head,
                    "final_value": r.final_value,
                    "collision_class": _collision_class(r),
                    "is_cross_op": r.is_cross_op,
                    "is_value_changing": r.is_value_changing,
                    "ops": r.distinct_ops,
                    "writers": [
                        {
                            "value": w.value,
                            "source": w.source,
                            "op": w.op_name,
                            "func": w.op_hint,
                        }
                        for w in r.writes
                    ],
                }
                for r in collisions
            ],
            "all": [
                {
                    "block": r.block,
                    "head": r.head,
                    "final_value": r.final_value,
                    "n_writers": len(r.distinct_sources),
                    "n_ops": len([o for o in r.distinct_ops if o != "<unknown>"]),
                    "writers": [
                        {"value": w.value, "source": w.source, "op": w.op_name}
                        for w in r.writes
                    ],
                }
                for (_, _), r in sorted(records.items())
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(model_summary)
        print(
            f"slope slots written: {len(records)}  |  "
            f"collisions: {len(collisions)}  |  "
            f"cross-op: {len(cross_op)}  |  "
            f"CROSS-OP CLOBBERS: {len(clobbers)}"
        )
        print()
        if clobbers:
            print("CROSS-OP CLOBBERS (the relay-bug class):")
            for r in clobbers:
                fv = "?" if r.final_value is None else f"{r.final_value:g}"
                print(
                    f"  block {r.block} head {r.head}  final={fv}: "
                    f"{_format_writes(r)}"
                )
            print()
        if collisions:
            print("ALL COLLISIONS:")
            for r in collisions:
                fv = "?" if r.final_value is None else f"{r.final_value:g}"
                print(
                    f"  block {r.block} head {r.head}  final={fv}  "
                    f"[{_collision_class(r)}]: {_format_writes(r)}"
                )
        else:
            print("No collisions detected.")

    if args.md is not None:
        md = render_markdown(records, model_summary)
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nWrote markdown report to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
