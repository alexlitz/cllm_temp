#!/usr/bin/env python3
"""Unified probe library for the c4_release neural VM.

ONE place for the patterns every ``tools/probe_*.py`` re-implements by hand:

  1. **Bytecode assembly** — ``bc([...])`` folds ``(Opcode, imm)`` tuples into
     the ``op | (imm << 8)`` word form (was ``make_bc`` / ``_mk`` in 100+
     probes).
  2. **Ground-truth probe build** — ``build_probe()`` returns the exact
     spec_k=0 batched smoke-path ``GroundTruthProbe`` (no hooks, no weight
     overrides). Thin re-export of ``probe_groundtruth.build_groundtruth_probe``
     so callers ``import probe_lib`` once.
  3. **Built-layout dim resolution** — ``dim_positions(model)`` /
     ``resolve(model, "OUTPUT_LO")`` read ``model.dim_positions`` (the BUILT
     layout, post widen-repack) NOT the static registry. The widen repack
     MOVES almost every dim (OP_PSH 275→197, OPCODE_BYTE_LO 12→474); probing at
     static-registry positions reads the WRONG cell. See memory note
     ``feedback_probe_dims_use_built_layout_not_static_registry.md``.
  4. **Reverse dim→name** — ``DimNamer(model).name_for(di)`` maps a raw dim
     index back to ``"<band>+<offset>"`` from the built layout (was a
     hand-rolled ``name_for`` in ~11 probes, several of which used the STALE
     ``build_default_registry_dynamic()`` static slots).
  5. **Residual-row dump at a block** — ``residual_row(...)`` /
     ``band(...)`` read a post-block hidden-state row via the model's opt-in
     ``stop_after_block`` kwarg (no hooks). ``hot(...)`` filters a band to its
     non-trivial cells.
  6. **LM-head logit attribution** — ``logit_attrib(...)`` computes the
     per-dim contribution of the (no-final-norm) head to
     ``logit[want] - logit[got]``, the pattern in ~31 probes
     (``probe_ax_logit_attrib``, ``probe_ax_byte23_func``, ...).
  7. **Block↔layer map** — ``block_layer_map(...)`` /
     ``print_block_layer_map(...)`` re-export the 37-physical-block ↔
     27-logical-layer mapping.
  8. **Per-step register markers** — ``register_marker_rows(...)`` locates the
     REG_PC/REG_AX/... marker positions in a trace, and ``decode_register(...)``
     reads the 4 value bytes after a marker.

TOOLING ONLY. No hooks, no weight overrides, spec_k=0 throughout. The model is
never mutated → byte-identical to golden (``tools/_isa_golden_hash.py``).

Quick start::

    from tools import probe_lib as P
    probe = P.build_probe()
    dp    = P.dim_positions(probe.model)                 # {name: dim} BUILT
    olo   = P.resolve(probe.model, "OUTPUT_LO")          # single dim
    row   = P.band(probe, bc, block_idx=27, position=-1,
                   base="OUTPUT_LO", width=16)           # [16] cells
    attr  = P.logit_attrib(probe, bc, position=-1, want=0x02, got=0x00)

Run directly to self-validate (reproduces probe_groundtruth's block↔layer map,
an OUTPUT_LO resolution, and the 4-program byte-identity check)::

    python tools/probe_lib.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# spec_k=0 is the smoke gate's ground-truth path. Pin both knobs before any
# neural_vm import so transitively-read defaults never land on spec_k=8.
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)  # .../c4_release
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch  # noqa: E402

from tools.probe_groundtruth import (  # noqa: E402
    GroundTruthProbe,
    build_groundtruth_probe,
)
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402


# ======================================================================
# 1. Bytecode assembly  (replaces make_bc / _mk / bc in 100+ probes)
# ======================================================================
def bc(ops: Sequence) -> List[int]:
    """Fold a program into VM word form.

    Each item is either a bare opcode int, or a ``(opcode, imm)`` tuple which
    becomes ``opcode | (imm << 8)``. Mirrors the ``make_bc`` / ``_mk`` helper
    copy-pasted across the probe corpus.

        bc([(Opcode.IMM, 42), Opcode.PSH, (Opcode.IMM, 42), Opcode.EQ,
            Opcode.EXIT])
    """
    out: List[int] = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            out.append(int(opcode) | (int(imm) << 8))
        else:
            out.append(int(op))
    return out


# Re-export so ``P.Opcode`` / ``P.Token`` work without a second import.
__all_reexport = (Opcode, Token)


# ======================================================================
# 2. Ground-truth probe build
# ======================================================================
def build_probe() -> GroundTruthProbe:
    """Return the spec_k=0 batched smoke-path probe (no hooks, no overrides).

    Thin wrapper over ``probe_groundtruth.build_groundtruth_probe`` so a caller
    only ``import probe_lib``. The build bakes the model once (slow); reuse one
    probe across many programs.
    """
    return build_groundtruth_probe()


# ======================================================================
# 3+4. Built-layout dim resolution + reverse dim->name
# ======================================================================
def dim_positions(model) -> Dict[str, int]:
    """Return the BUILT layout ``{name: dim}`` map (post widen-repack).

    ALWAYS use this, NOT ``build_default_registry_dynamic()`` /
    ``dim_registry`` static slots — the auto-widen repacks nearly every dim, so
    static positions read the wrong cell. ``model.dim_positions`` is the map the
    model was actually compiled with.
    """
    dp = getattr(model, "dim_positions", None)
    if dp is None:
        raise AttributeError(
            "model has no .dim_positions — pass the BUILT model "
            "(probe.model), not a bare registry.")
    return dict(dp)


def resolve(model, name: str) -> int:
    """Resolve one named band's START dim from the BUILT layout.

    ``resolve(model, "OUTPUT_LO")`` -> the residual dim index. Raises if the
    name is absent (a dead/renamed dim), which surfaces stale probe refs.
    """
    dp = dim_positions(model)
    if name not in dp:
        raise KeyError(
            f"dim {name!r} not in built layout (dead/renamed?). "
            f"Nearby: {sorted(k for k in dp if name.split('_')[0] in k)[:6]}")
    return int(dp[name])


class DimNamer:
    """Reverse map: raw dim index -> ``"<band>+<offset>"`` from BUILT layout.

    Replaces the hand-rolled ``name_for`` in ~11 probes (some of which used the
    STALE ``build_default_registry_dynamic()`` static slots and mislabeled
    every dim after the widen repack). Build once per model, call ``name_for``
    per dim.
    """

    def __init__(self, model):
        self._dp = dim_positions(model)
        # Sort by start so the nearest-<=-start lookup is a bisect.
        self._items = sorted(self._dp.items(), key=lambda kv: kv[1])
        self._starts = [v for _, v in self._items]
        self._names = [k for k, _ in self._items]

    def name_for(self, dim: int) -> str:
        """Nearest named band start <= ``dim``, plus the offset."""
        import bisect
        i = bisect.bisect_right(self._starts, dim) - 1
        if i < 0:
            return f"dim{dim}"
        return f"{self._names[i]}+{dim - self._starts[i]}"


# ======================================================================
# 5. Residual-row dump at a block  (no hooks; stop_after_block)
# ======================================================================
@torch.no_grad()
def residual_row(
    probe: GroundTruthProbe,
    program_bytes: Sequence[int],
    block_idx: int,
    position: int,
    *,
    max_steps: Optional[int] = None,
) -> torch.Tensor:
    """Return the FULL post-block residual row ``[D]`` at ``position``.

    Replays the spec_k=0 emission loop to rebuild the final context, then runs
    one forward truncated to ``stop_after_block=block_idx`` and reads the
    model's own post-block hidden state. NO hooks. ``position<0`` indexes from
    the end. This is the raw form ``residual_at`` wraps — use it when you want
    the whole row (e.g. for logit attribution) rather than named dims.
    """
    ctx = probe._final_context(program_bytes, max_steps=max_steps)
    if position < 0:
        position = len(ctx) + position
    if not (0 <= position < len(ctx)):
        raise IndexError(f"position {position} out of range [0,{len(ctx)})")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)  # [1,S,D]
    row = x[0, position]
    return row.to_dense() if row.is_sparse else row


@torch.no_grad()
def band(
    probe: GroundTruthProbe,
    program_bytes: Sequence[int],
    block_idx: int,
    position: int,
    base: str,
    width: int,
    *,
    max_steps: Optional[int] = None,
) -> List[float]:
    """Return the ``[width]`` cells of a named band at a post-block row.

    ``base`` is resolved from the BUILT layout (``resolve``). Equivalent to the
    ``onehot(...)`` helper copy-pasted across the operand/OUTPUT probes.
    """
    start = resolve(probe.model, base)
    row = residual_row(probe, program_bytes, block_idx, position,
                       max_steps=max_steps)
    D = row.shape[-1]
    return [float(row[start + i]) if 0 <= start + i < D else float("nan")
            for i in range(width)]


def hot(cells: Optional[Sequence[float]], thr: float = 0.3
        ) -> List[Tuple[int, float]]:
    """Filter a band to ``[(idx, round(val,2)), ...]`` for ``|val| > thr``."""
    if cells is None:
        return []
    return [(i, round(float(v), 2)) for i, v in enumerate(cells)
            if abs(float(v)) > thr]


# ======================================================================
# 6. LM-head logit attribution  (no final norm -> fully attributable)
# ======================================================================
@dataclass
class LogitAttrib:
    """Result of ``logit_attrib``: the two logits and per-dim contributions."""
    logit_want: float
    logit_got: float
    diff: float                              # logit_want - logit_got
    contrib: torch.Tensor                    # [D] per-dim contribution to diff
    residual: torch.Tensor                   # [D] the row that was read

    def top(self, k: int = 18) -> List[int]:
        """Return the ``k`` dim indices with the largest |contribution|."""
        order = torch.argsort(self.contrib.abs(), descending=True)
        return order[:k].tolist()


@torch.no_grad()
def logit_attrib(
    probe: GroundTruthProbe,
    program_bytes: Sequence[int],
    position: int,
    want: int,
    got: int,
    *,
    block_idx: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> LogitAttrib:
    """Attribute ``logit[want] - logit[got]`` to residual dims at ``position``.

    The model has NO final norm, so ``logit[t] = head.weight[t] . resid +
    head.bias[t]`` is exactly attributable. Contribution of dim ``d`` to
    ``(logit_want - logit_got)`` is ``(W[want,d] - W[got,d]) * resid[d]``. This
    is the pattern in ~31 probes (``probe_ax_logit_attrib`` et al.).

    ``block_idx`` defaults to the last block (the residual the LM head reads).
    Pair with ``DimNamer.name_for`` to label the top dims.
    """
    model = probe.model
    if block_idx is None:
        block_idx = len(model.blocks) - 1
    res = residual_row(probe, program_bytes, block_idx, position,
                       max_steps=max_steps)
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = model.head.bias
    res_f = res.to(W.device).float()
    dw = (W[want] - W[got]) * res_f  # [D]
    if dw.is_sparse:
        dw = dw.to_dense()
    logit_want = float((W[want] * res_f).sum() + b[want])
    logit_got = float((W[got] * res_f).sum() + b[got])
    return LogitAttrib(
        logit_want=logit_want,
        logit_got=logit_got,
        diff=logit_want - logit_got,
        contrib=dw,
        residual=res_f,
    )


# ======================================================================
# 7. Block <-> layer map
# ======================================================================
def block_layer_map(probe: GroundTruthProbe) -> List[dict]:
    """Re-export the 37-physical-block ↔ 27-logical-layer mapping."""
    return probe.block_layer_map()


def print_block_layer_map(probe: GroundTruthProbe, file=sys.stdout) -> None:
    """Print the physical-block ↔ logical-layer table."""
    probe.print_block_layer_map(file=file)


# ======================================================================
# 8. Per-step register markers
# ======================================================================
_MARKER_TOKENS = {
    "REG_PC": int(Token.REG_PC),
    "REG_AX": int(Token.REG_AX),
    "REG_SP": int(Token.REG_SP),
    "REG_BP": int(Token.REG_BP),
}


def register_marker_rows(trace: Dict[int, dict], reg: str = "REG_AX"
                         ) -> List[int]:
    """Return the sorted context positions of ``reg`` markers in a probe trace.

    ``trace`` is the dict returned by ``probe.probe(...)``. ``reg`` is one of
    REG_PC/REG_AX/REG_SP/REG_BP. Replaces the ``markers = [p for p in ... if
    trace[p]["token"] == RAX]`` idiom in dozens of probes.
    """
    tok = _MARKER_TOKENS.get(reg)
    if tok is None:
        raise KeyError(f"unknown register marker {reg!r}; "
                       f"one of {sorted(_MARKER_TOKENS)}")
    return [p for p in sorted(trace) if trace[p]["token"] == tok]


def decode_register(trace: Dict[int, dict], marker_pos: int) -> List[Optional[int]]:
    """Return the 4 little-endian value-byte tokens after a register marker."""
    return [trace.get(marker_pos + 1 + b, {}).get("token") for b in range(4)]


# ======================================================================
# Self-validation
# ======================================================================
def _validate(verbose: bool = True) -> bool:
    """Reproduce probe_groundtruth's block↔layer map + an OUTPUT_LO resolve,
    and confirm the 4-program byte-identity ground-truth match."""
    from tools.probe_groundtruth import validate as gt_validate

    probe = build_probe()

    # (a) block<->layer map: 37 physical blocks, expansions tagged.
    rows = block_layer_map(probe)
    n_logical = len({r["logical"] for r in rows})
    if verbose:
        print(f"[map] {len(rows)} physical blocks <- {n_logical} logical "
              f"layers")
    assert len(rows) >= 27, f"expected >=27 blocks, got {len(rows)}"

    # (b) BUILT-layout dim resolution + reverse namer round-trip.
    olo = resolve(probe.model, "OUTPUT_LO")
    namer = DimNamer(probe.model)
    back = namer.name_for(olo)
    if verbose:
        print(f"[dim] OUTPUT_LO -> dim {olo}; name_for({olo}) = {back!r}")
    assert back.startswith("OUTPUT_LO"), (
        f"reverse map broke: name_for(OUTPUT_LO@{olo}) = {back!r}")

    # (c) bytecode assembly matches the hand-rolled form.
    prog = bc([(Opcode.IMM, 42), Opcode.PSH, (Opcode.IMM, 42), Opcode.EQ,
               Opcode.EXIT])
    assert prog[0] == (Opcode.IMM | (42 << 8)), "bc() word form wrong"

    # (d) a residual band read + logit attribution run end-to-end.
    trace = probe.probe(prog, max_steps=20)
    ax_rows = register_marker_rows(trace, "REG_AX")
    assert ax_rows, "no REG_AX markers in EQ program trace"
    last_block = len(probe.model.blocks) - 1
    cells = band(probe, prog, last_block, ax_rows[-1], "OUTPUT_LO", 16,
                 max_steps=20)
    if verbose:
        print(f"[band] EQ OUTPUT_LO@AXrow hot cells: {hot(cells)}")
    attr = logit_attrib(probe, prog, ax_rows[-1] + 1, want=1, got=0,
                        max_steps=20)
    if verbose:
        top = attr.top(3)
        print(f"[attrib] diff(1-0)={attr.diff:.2f} top dims "
              f"{[(d, namer.name_for(d)) for d in top]}")

    # (e) byte-identity ground-truth match (delegates to probe_groundtruth).
    gt_validate(probe, verbose=False)
    if verbose:
        print("[gt] 4-program byte-identity ground-truth match confirmed.")
        print("\nprobe_lib self-validation PASSED.")
    return True


if __name__ == "__main__":
    print("Building spec_k=0 probe (bakes model once; slow)...\n",
          file=sys.stderr)
    _validate(verbose=True)
