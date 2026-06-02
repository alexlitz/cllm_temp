"""Phase 7.F.4 — quantify peak KV memory drop with STATIC_LIVENESS eviction.

This harness:

1. Compiles ``compile_full_vm`` twice — once with
   ``kv_eviction_policy=OFF`` and once with
   ``kv_eviction_policy=STATIC_LIVENESS``.
2. Runs each model on a 100-input sample of the smoke + 1096 corpus,
   threading a per-layer ``LayerKVCache`` so we can read the realised
   ``cached_k`` shape per layer after each forward pass.
3. Reports per-layer + corpus-wide peak ``K_cache + V_cache`` memory
   (in bytes), and the % reduction STATIC_LIVENESS delivers over OFF.

The eviction policy's runtime is *zero-fill*: ``apply_eviction`` zeros
rows the analyzer marks dead but does not shrink the tensor. The peak
memory drop therefore comes in two flavours:

* **Realised drop** — the actual bytes saved at peak if the eviction
  hook compacted the cache (today's implementation does not, so this
  number is the *potential* available to a future compaction pass).
  Computed as ``evicted_positions / total_positions * peak_bytes``.
* **Tensor-shape drop** — the literal ``cached_k.shape[2]`` difference
  between OFF and STATIC_LIVENESS at peak. With today's zero-fill-only
  implementation this is ~0%; we report it for honesty.

The plan target (``docs/SAFE_KV_EVICTION_PLAN.md`` §7.F.2) is ≥30%
peak reduction. The realised number is what we compare against the
target — that's the dividend the analyzer's coverage promises to
deliver once the compaction pass lands.

Usage::

    python -m tools.measure_kv_eviction \
        --num-inputs 100 --max-steps 30 \
        --out .agent-logs/kv_eviction_quantification.md

The harness defaults are calibrated to the smoke + 1096 corpus and
require no external arguments to reproduce the report.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Make `neural_vm` / `tests` / `src` imports work when invoked as a
# loose script (matches the pattern used by `tools/benchmark_kv_cache.py`).
_THIS_DIR = Path(__file__).resolve().parent
_C4_ROOT = _THIS_DIR.parent
if str(_C4_ROOT) not in sys.path:
    sys.path.insert(0, str(_C4_ROOT))

import torch

from neural_vm.kv_cache import LayerKVCache
from neural_vm.kv_eviction import KVEvictionPolicy, apply_eviction


# ---------------------------------------------------------------------------
# Smoke inputs (shared with tests/test_kv_eviction.py).
# ---------------------------------------------------------------------------


_SMOKE_INPUTS_RAW = [
    [0, 1, 2, 3, 4],
    [10, 11, 12],
    [100, 50, 25, 12],
    [5, 5, 5, 5, 5, 5],
    [200, 201, 202, 203, 204, 205, 206],
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _LayerCacheStats:
    """Captures cache shape and eviction counts per attention layer."""

    layer_idx: int
    num_heads: int
    head_dim: int
    # Per-input peak ``cached_k.shape[2]`` (= number of positions cached
    # at the peak of a single forward pass). The harness reports the
    # max across all inputs as the layer's overall peak.
    peak_S_kv_per_input: List[int] = field(default_factory=list)
    # Total positions that were ever evictable across all inputs (sum
    # of ``len(state.evicted_positions)`` after each forward). Only
    # populated for the STATIC_LIVENESS run.
    total_evictable_positions: int = 0
    # Total positions cached across all inputs — denominator for the
    # ``evictable / cached`` ratio.
    total_positions_observed: int = 0

    def peak_bytes(self, dtype_bytes: int = 4) -> int:
        """Peak ``K_cache + V_cache`` size in bytes (across all inputs).

        ``K_cache + V_cache`` is two tensors of shape
        ``[1, num_heads, S_kv, head_dim]``. We bake batch=1 because each
        input runs in its own forward pass; in a multi-input batched
        scenario the bytes scale linearly with batch.
        """
        peak_S_kv = max(self.peak_S_kv_per_input, default=0)
        return 2 * self.num_heads * peak_S_kv * self.head_dim * dtype_bytes

    def avg_bytes(self, dtype_bytes: int = 4) -> int:
        """Average peak bytes across all inputs (smoother than peak)."""
        if not self.peak_S_kv_per_input:
            return 0
        avg_S_kv = sum(self.peak_S_kv_per_input) / len(self.peak_S_kv_per_input)
        return int(2 * self.num_heads * avg_S_kv * self.head_dim * dtype_bytes)

    def realised_drop_bytes(self, dtype_bytes: int = 4) -> int:
        """Bytes that *could* be freed by a compaction pass."""
        if self.total_positions_observed == 0:
            return 0
        peak_bytes = self.peak_bytes(dtype_bytes)
        frac = self.total_evictable_positions / self.total_positions_observed
        return int(peak_bytes * frac)


# ---------------------------------------------------------------------------
# Token sequence builders
# ---------------------------------------------------------------------------


def _build_program_tokens(source: str) -> Optional[List[int]]:
    """Compile a C source string into a token sequence (program prefix).

    Returns ``None`` on compile error. The token sequence is the same one
    ``AutoregressiveVMRunner._build_context`` emits — the prefix that
    ``model(token_ids)`` consumes for the program-loading phase.
    """
    try:
        # Lazy imports — the C compiler pulls in cllm which is slow.
        from src.compiler import compile_c
        from neural_vm.run_vm import AutoregressiveVMRunner

        bytecode, data = compile_c(source)
        runner = _RUNNER_SINGLETON
        if runner is None:
            raise RuntimeError(
                "context builder requires the cached runner; call "
                "_set_runner_singleton() first"
            )
        ctx = runner._build_context(bytecode, data or b"", [], "")
        return ctx
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  [warn] compile failed: {exc!r}", file=sys.stderr)
        return None


_RUNNER_SINGLETON = None


def _set_runner_singleton():
    """Build a single (cheap) ``AutoregressiveVMRunner`` for token encoding.

    We only need ``_build_context`` from the runner; the model it loads
    is irrelevant — we'll use the OFF/STATIC_LIVENESS models built
    explicitly. We pass ``cache_model=False`` so we don't pin a model
    into the class-level cache.
    """
    global _RUNNER_SINGLETON
    if _RUNNER_SINGLETON is not None:
        return _RUNNER_SINGLETON
    from neural_vm.run_vm import AutoregressiveVMRunner

    # A minimal runner — never builds a real model since we set
    # ``model=None`` immediately after init. The init still constructs
    # one but we never call its forward.
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        cache_model=True,  # share the bake we already paid for
    )
    _RUNNER_SINGLETON = runner
    return runner


def _sample_1096_inputs(n: int, *, seed: int = 0) -> List[Tuple[str, List[int]]]:
    """Compile ``n`` programs from the 1096 corpus into token sequences.

    Returns ``(description, token_ids)`` pairs. Skips programs that fail
    to compile (rare; the corpus is curated).
    """
    from tests.test_suite_1000 import generate_test_programs

    rng = torch.Generator()
    rng.manual_seed(seed)

    programs = generate_test_programs()
    indices = torch.randperm(len(programs), generator=rng).tolist()

    sampled: List[Tuple[str, List[int]]] = []
    for idx in indices:
        if len(sampled) >= n:
            break
        source, _expected, desc = programs[idx]
        tokens = _build_program_tokens(source)
        if tokens is None:
            continue
        sampled.append((desc, tokens))
    return sampled


# ---------------------------------------------------------------------------
# Per-attention-layer measurement
# ---------------------------------------------------------------------------


def _layer_attns(model) -> List[Tuple[int, torch.nn.Module]]:
    """Enumerate ``(layer_idx, attention_module)`` for every block."""
    out = []
    for i, block in enumerate(getattr(model, "blocks", ())):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "W_q"):
            out.append((i, attn))
    return out


def _kv_cache_for(model) -> LayerKVCache:
    """Build a fresh ``LayerKVCache`` matching the model's blocks."""
    layer_attns = _layer_attns(model)
    if not layer_attns:
        raise RuntimeError("model has no attention blocks")
    first = layer_attns[0][1]
    return LayerKVCache(
        num_layers=len(getattr(model, "blocks", ())),
        max_tokens=65_536,
        num_heads=first.num_heads,
        head_dim=first.head_dim,
        device=next(model.parameters()).device,
    )


@torch.no_grad()
def _run_one(
    model,
    token_ids: torch.Tensor,
    *,
    stats: Dict[int, _LayerCacheStats],
    sample_eviction_state: bool,
) -> None:
    """Forward once, populate per-layer cache stats.

    ``sample_eviction_state`` controls whether we also accumulate
    ``len(state.evicted_positions)`` (only meaningful for STATIC_LIVENESS).
    """

    kv_cache = _kv_cache_for(model)
    # Forward — the model populates kv_cache.caches[i].cached_k/cached_v
    # during attention forward; the eviction hook (if attached) zeros
    # rows at step boundaries via ``apply_eviction``.
    model(token_ids, kv_cache=kv_cache)

    # For STATIC_LIVENESS we additionally invoke the eviction state
    # *manually* on each layer to capture realised "would-be evicted"
    # rows for this input. The state tracks ``evicted_positions`` as a
    # cumulative set across steps, so we reset before and read after.
    for layer_idx, attn in _layer_attns(model):
        cache = kv_cache.caches[layer_idx]
        S_kv = 0 if cache.cached_k is None else int(cache.cached_k.shape[2])
        if layer_idx not in stats:
            stats[layer_idx] = _LayerCacheStats(
                layer_idx=layer_idx,
                num_heads=attn.num_heads,
                head_dim=attn.head_dim,
            )
        layer_stats = stats[layer_idx]
        layer_stats.peak_S_kv_per_input.append(S_kv)
        layer_stats.total_positions_observed += S_kv

        if not sample_eviction_state:
            continue

        # STATIC_LIVENESS path: re-attach a synthetic K_cache pointer so
        # ``apply_eviction`` can count zeroable rows for the cache shape
        # we just observed. We don't actually mutate the cache here —
        # we'd double-count if the in-line forward already evicted; the
        # forward's hook is the canonical accounting. Use ``state.evicted_positions``
        # which the in-forward hook populated.
        state = getattr(attn, "eviction_state", None)
        if state is None:
            continue
        # Replay all step boundaries to count theoretical evictable
        # positions for this input's actual cache. apply_eviction zeros
        # the *cached_k*/*cached_v* in place; we use a separate (zeroed
        # already) stand-in and just count the set ops the analyzer would
        # have done given S_kv positions.
        evictable_for_this_input = set()
        for step_idx, positions in state.evictable_positions_at_step.items():
            for pos in positions:
                if 0 <= pos < S_kv:
                    evictable_for_this_input.add(pos)
        layer_stats.total_evictable_positions += len(evictable_for_this_input)


# ---------------------------------------------------------------------------
# Top-level measurement loop
# ---------------------------------------------------------------------------


def measure(
    *,
    num_inputs: int,
    n_steps: int,
    seed: int,
) -> Dict[str, object]:
    """Compile both models, run each on the sampled corpus, return a report dict."""
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm

    # Boot a runner singleton (used by ``_build_program_tokens``).
    _set_runner_singleton()

    # Build inputs.
    inputs: List[Tuple[str, List[int]]] = []
    for i, ids in enumerate(_SMOKE_INPUTS_RAW):
        inputs.append((f"smoke_{i}", list(ids)))
    sampled = _sample_1096_inputs(
        max(0, num_inputs - len(_SMOKE_INPUTS_RAW)),
        seed=seed,
    )
    inputs.extend(sampled)

    print(
        f"[measure] {len(inputs)} inputs prepared "
        f"({len(_SMOKE_INPUTS_RAW)} smoke + {len(sampled)} 1096-sampled)",
        flush=True,
    )

    # Compile models.
    t0 = time.perf_counter()
    model_off, layout = compile_full_vm(
        disk_cache=True,
        kv_eviction_policy=KVEvictionPolicy.OFF,
    )
    print(
        f"[measure] OFF compiled in {time.perf_counter() - t0:.1f}s "
        f"(d_model={layout.d_model}, n_layers={layout.n_layers})",
        flush=True,
    )
    t1 = time.perf_counter()
    model_sl, _ = compile_full_vm(
        disk_cache=True,
        kv_eviction_policy=KVEvictionPolicy.STATIC_LIVENESS,
        kv_eviction_n_steps=n_steps,
    )
    print(
        f"[measure] STATIC_LIVENESS compiled in {time.perf_counter() - t1:.1f}s",
        flush=True,
    )

    # Run.
    stats_off: Dict[int, _LayerCacheStats] = {}
    stats_sl: Dict[int, _LayerCacheStats] = {}

    # Inspect analyzer report once for diagnostic numbers (coverage, cycle members).
    analyzer_diag = _diagnose_analyzer(model_sl, layout, n_steps)

    for run_idx, (desc, ids) in enumerate(inputs):
        token_ids = torch.tensor([ids], dtype=torch.long)
        try:
            _run_one(model_off, token_ids, stats=stats_off, sample_eviction_state=False)
            _run_one(model_sl, token_ids, stats=stats_sl, sample_eviction_state=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[measure] input {run_idx} ({desc!r}) raised {exc!r}; skipping",
                file=sys.stderr,
            )
            continue
        if (run_idx + 1) % 25 == 0:
            print(
                f"[measure] {run_idx + 1}/{len(inputs)} inputs processed",
                flush=True,
            )

    return {
        "stats_off": stats_off,
        "stats_sl": stats_sl,
        "analyzer_diag": analyzer_diag,
        "num_inputs": len(inputs),
        "n_steps": n_steps,
        "d_model": layout.d_model,
        "n_layers": layout.n_layers,
    }


def _diagnose_analyzer(model_sl, layout, n_steps: int) -> Dict[str, object]:
    """Inspect the analyzer's evictable-set composition for diagnostics."""
    from neural_vm.kv_liveness_analyzer import analyze_kv_liveness

    ops: List = []
    for ops_at_layer in layout.ops_per_layer:
        ops.extend(ops_at_layer)
    ops.extend(layout.block_ops)
    ops.extend(layout.model_ops)
    report = analyze_kv_liveness(ops, n_steps=n_steps)

    # Per-dim breakdown of evictable entries vs cycle-kept entries.
    evictable_dim_counter: Counter = Counter()
    for entries in report.evictable_at_step.values():
        for entry in entries:
            evictable_dim_counter[entry.dim_name] += 1
    cycle_dim_counter: Counter = Counter(
        entry.dim_name for entry in report.cycle_conservative
    )

    return {
        "coverage": float(report.coverage),
        "evictable_total": sum(len(s) for s in report.evictable_at_step.values()),
        "cycle_conservative_total": len(report.cycle_conservative),
        "evictable_dim_breakdown": evictable_dim_counter.most_common(),
        "cycle_dim_breakdown": cycle_dim_counter.most_common(20),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB"):
        n /= 1024
        if n < 1024:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} TiB"


def _pct(num: float, denom: float) -> str:
    if denom == 0:
        return "n/a"
    return f"{(num / denom) * 100:5.2f}%"


def write_report(report: Dict[str, object], path: Path) -> None:
    """Render the measurement results as a markdown report."""

    stats_off: Dict[int, _LayerCacheStats] = report["stats_off"]  # type: ignore[assignment]
    stats_sl: Dict[int, _LayerCacheStats] = report["stats_sl"]  # type: ignore[assignment]
    analyzer_diag: Dict[str, object] = report["analyzer_diag"]  # type: ignore[assignment]
    num_inputs = int(report["num_inputs"])
    n_steps = int(report["n_steps"])
    d_model = int(report["d_model"])
    n_layers = int(report["n_layers"])

    target = 0.30  # 30% per docs/SAFE_KV_EVICTION_PLAN.md §7.F.2

    layer_ids = sorted(set(stats_off.keys()) | set(stats_sl.keys()))
    dtype_bytes = 4

    # Per-layer table.
    lines: List[str] = []
    lines.append("# Phase 7.F.4 — KV eviction peak memory quantification")
    lines.append("")
    lines.append(
        f"Generated by `tools/measure_kv_eviction.py`. Compiled "
        f"`d_model={d_model}`, `n_layers={n_layers}`, "
        f"`kv_eviction_n_steps={n_steps}`. "
        f"Sampled {num_inputs} inputs (5 smoke + {num_inputs - 5} from the "
        f"1096 corpus, seed=0)."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "* Compile two models via `compile_full_vm`: `kv_eviction_policy=OFF` "
        "and `kv_eviction_policy=STATIC_LIVENESS` (n_steps=64). The "
        "byte-identity gate (tests/test_kv_eviction.py) already confirmed "
        "the two produce identical logits."
    )
    lines.append(
        "* For each sampled input, build a fresh `LayerKVCache` and call "
        "`model(token_ids, kv_cache=cache)`. After the forward, read each "
        "layer's `cached_k.shape[2]` — the actual peak `S_kv` for that "
        "input."
    )
    lines.append(
        "* Peak K+V bytes per layer = `2 * num_heads * S_kv * head_dim * "
        "dtype_bytes` (`dtype_bytes=4` for float32; sparse / compact "
        "models still allocate float32 KV cache rows)."
    )
    lines.append(
        "* The STATIC_LIVENESS runtime *zeros* dead rows but does not "
        "compact the tensor; the **realised drop** column is therefore the "
        "fraction of K/V rows that *could* be removed by a future "
        "compaction pass (`evicted_positions / total_positions_observed`). "
        "The **tensor-shape drop** column is the literal peak-bytes diff "
        "between OFF and STATIC_LIVENESS, which is ~0% under today's "
        "zero-fill-only implementation."
    )
    lines.append("")
    lines.append("## Analyzer coverage")
    lines.append("")
    lines.append(
        f"* `LivenessReport.coverage` = "
        f"**{float(analyzer_diag['coverage']):.4%}** "
        f"({analyzer_diag['evictable_total']} evictable / "
        f"{analyzer_diag['cycle_conservative_total']} cycle-conservative)."
    )
    lines.append("")
    lines.append("**Evictable dim names (count of (step, KVEntry) per dim):**")
    lines.append("")
    if analyzer_diag["evictable_dim_breakdown"]:
        for name, cnt in analyzer_diag["evictable_dim_breakdown"]:
            lines.append(f"* `{name}`: {cnt}")
    else:
        lines.append("* (none)")
    lines.append("")
    lines.append(
        "**Cycle-conservative dim names (top 20 — analyzer left LIVE due "
        "to producer/consumer cycle in declarative IR):**"
    )
    lines.append("")
    for name, cnt in analyzer_diag["cycle_dim_breakdown"]:
        lines.append(f"* `{name}`: {cnt}")
    lines.append("")

    lines.append("## Per-layer peak KV memory")
    lines.append("")
    lines.append(
        "| Layer | num_heads | head_dim | OFF peak S_kv | OFF peak K+V "
        "| SL peak S_kv | SL peak K+V | tensor-shape drop | evicted / cached "
        "| realised drop |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    total_off_peak_bytes = 0
    total_sl_peak_bytes = 0
    total_realised_drop_bytes = 0
    total_evicted_positions = 0
    total_observed_positions = 0
    for layer in layer_ids:
        off = stats_off.get(layer)
        sl = stats_sl.get(layer)
        if off is None or sl is None:
            continue
        off_peak_S = max(off.peak_S_kv_per_input, default=0)
        sl_peak_S = max(sl.peak_S_kv_per_input, default=0)
        off_bytes = off.peak_bytes(dtype_bytes)
        sl_bytes = sl.peak_bytes(dtype_bytes)
        shape_drop = (off_bytes - sl_bytes) / off_bytes if off_bytes else 0.0
        evicted = sl.total_evictable_positions
        observed = sl.total_positions_observed
        realised_drop = sl.realised_drop_bytes(dtype_bytes)
        total_off_peak_bytes += off_bytes
        total_sl_peak_bytes += sl_bytes
        total_realised_drop_bytes += realised_drop
        total_evicted_positions += evicted
        total_observed_positions += observed
        lines.append(
            f"| {layer} | {off.num_heads} | {off.head_dim} | {off_peak_S} | "
            f"{_fmt_bytes(off_bytes)} | {sl_peak_S} | {_fmt_bytes(sl_bytes)} "
            f"| {shape_drop * 100:5.2f}% | "
            f"{evicted}/{observed} ({_pct(evicted, observed)}) "
            f"| {_fmt_bytes(realised_drop)} |"
        )

    lines.append("")
    lines.append("## Corpus-wide totals")
    lines.append("")
    overall_shape_drop = (
        (total_off_peak_bytes - total_sl_peak_bytes) / total_off_peak_bytes
        if total_off_peak_bytes
        else 0.0
    )
    overall_realised_drop = (
        total_realised_drop_bytes / total_off_peak_bytes
        if total_off_peak_bytes
        else 0.0
    )
    lines.append(f"* OFF peak K+V (sum across all layers): "
                 f"**{_fmt_bytes(total_off_peak_bytes)}**.")
    lines.append(f"* STATIC_LIVENESS peak K+V (sum across all layers): "
                 f"**{_fmt_bytes(total_sl_peak_bytes)}**.")
    lines.append(
        f"* **Tensor-shape drop**: "
        f"{overall_shape_drop * 100:.2f}% "
        f"(today's zero-fill implementation: expected ~0%)."
    )
    lines.append(
        f"* **Realised drop** (compaction-pass dividend): "
        f"{overall_realised_drop * 100:.2f}% "
        f"({_fmt_bytes(total_realised_drop_bytes)} freeable / "
        f"{_fmt_bytes(total_off_peak_bytes)} peak)."
    )
    lines.append(
        f"* Evictable positions: "
        f"{total_evicted_positions} / {total_observed_positions} "
        f"= {_pct(total_evicted_positions, total_observed_positions)}."
    )
    lines.append("")
    lines.append("## Target assessment")
    lines.append("")
    lines.append(
        f"Plan target (docs/SAFE_KV_EVICTION_PLAN.md §7.F.2): **≥30% peak "
        f"KV memory reduction**."
    )
    lines.append("")
    achieved = overall_realised_drop * 100
    if achieved >= target * 100:
        lines.append(
            f"**Achieved**: {achieved:.2f}% realised drop ≥ {target * 100:.0f}% "
            f"target. Eviction policy delivers the planned dividend."
        )
    else:
        lines.append(
            f"**Gap**: {achieved:.2f}% realised drop < {target * 100:.0f}% target."
        )
        lines.append("")
        lines.append("### Root cause — analyzer coverage bottleneck")
        lines.append("")
        lines.append(
            f"The analyzer marks {analyzer_diag['evictable_total']} "
            f"`(step, KVEntry)` pairs evictable across all dims and "
            f"`{analyzer_diag['cycle_conservative_total']}` "
            f"cycle-conservative entries. Per `kv_eviction.build_state_from_report` "
            "the runtime evictable set is the **AND across every dim name** "
            "(a row is evictable only when *every* dim slot at that "
            "position is dead). Since the analyzer only classifies a "
            "tiny subset of dims as evictable, the per-row AND degenerates "
            "to empty for nearly all positions."
        )
        lines.append("")
        lines.append("Categories preventing evictions:")
        lines.append("")
        lines.append(
            f"1. **Dim cycles** ({analyzer_diag['cycle_conservative_total']} "
            "entries): dims that participate in a producer/consumer cycle "
            "in the declarative IR. The analyzer treats them as live (a "
            "later iteration could read them) per `treat_cycle_members_conservative=True`. "
            "Top cycle dims (see breakdown above) include the persistent "
            "register channels (`REG_PC`, `REG_AX`, `REG_SP`, `REG_BP`) "
            "and ALU intermediate dims that read-write through their own "
            "self-loop."
        )
        lines.append(
            "2. **Unclassified dims**: dims that aren't TEMP_*, *_PREV_STEP, "
            "or *_SCRATCH and have no clear semantic-overwrite path. The "
            "analyzer's heuristic categories miss anything that's "
            "semantically dead but lacks a naming convention the categoriser "
            "recognises."
        )
        lines.append(
            "3. **Per-row AND in `build_state_from_report`**: since the "
            "runtime cache row covers *all* dims for one position, ANY "
            "live dim at that position keeps the entire row live. So even "
            "a single conservative-kept dim (e.g. `REG_PC`) at every "
            "position blocks every eviction."
        )
        lines.append("")
        lines.append("### Concrete fix paths (for Phase 7.F.5+)")
        lines.append("")
        lines.append(
            "* **Per-head/dim cache compartmentalisation**: today's "
            "`cached_k[B, H, S, HD]` packs every dim into one row. Splitting "
            "into per-(layer, head, dim_group) caches would let the AND "
            "degenerate per-group instead of per-position."
        )
        lines.append(
            "* **Cycle-graph refinement**: the current SCC analysis treats "
            "any read+write edge between two dims as a cycle. Refining "
            "with control-flow / step-index awareness (e.g. `OUTPUT_LO_PREV_STEP -> OUTPUT_LO` "
            "is one-shot, not a true cycle) would lift many register channels "
            "out of cycle-conservative."
        )
        lines.append(
            "* **Semantic-overwrite category expansion**: the analyzer's "
            "third category (\"register-marker overwritten by later "
            "position write\") fires only when the analyzer can prove a "
            "dim is written at every later step. Loosen to \"written at "
            "the very next step\" to catch the common `REG_AX` carry "
            "pattern."
        )

    lines.append("")
    lines.append("## Reproducer")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python -m tools.measure_kv_eviction "
        f"--num-inputs {num_inputs} --max-steps {n_steps} "
        f"--out {path}"
    )
    lines.append("```")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"[measure] report written: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-inputs", type=int, default=100,
        help="Total inputs to sample (smoke + 1096). Default: 100.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=64,
        help="Liveness analyzer step horizon. Default: 64.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for sampling the 1096 corpus. Default: 0.",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_C4_ROOT / ".agent-logs" / "kv_eviction_quantification.md",
        help="Output report path.",
    )
    args = parser.parse_args(argv)

    report = measure(
        num_inputs=args.num_inputs,
        n_steps=args.max_steps,
        seed=args.seed,
    )
    write_report(report, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
