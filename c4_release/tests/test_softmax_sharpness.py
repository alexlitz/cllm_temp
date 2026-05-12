"""Softmax-sharpness audit harness for every attention head.

Per Phase 1 Agent A of ``docs/ARCH_LEAKAGE_FIX_PLAN.md`` — today's diagnostics
show that several attention heads are not steep enough at the softmax stage:
when the target K position is supposed to "win" the head's attention, its
softmax mass leaks into other K positions. That leakage is what makes L4 PC+2
writes contaminate L7/L8 ALU_HI slots (and the wider class of bugs the plan
documents).

This file implements the AUDIT (no fixes — that lands in Phase 2 Agent D as
``sharpen-leaky-attention-heads``). For every ``AutoregressiveAttention`` in
the compiled VM:

  * inspect each head's baked ``W_q`` / ``W_k`` to find which residual-stream
    input dims it "reads" (the K-gate / Q-gate dim sets);
  * construct a synthetic context (length ``S_AUDIT``) where exactly one K
    position carries those K-gate features ("target K") and the Q position
    carries the Q-gate features;
  * run the attention module standalone, capture the softmax probabilities,
    and measure the mass the head puts on the intended target K position;
  * record per-(layer, head): ``mass_on_target``, ``mass_on_runner_up``,
    ``gap_in_logits``, current ``alibi_slope``, current K-scale (``self.scale``
    times the K-gate magnitude implicit in W_k).

Heads whose K-gate or Q-gate is essentially zero (no positive-weighted input
dims above a small threshold) are classified as **dormant** — they don't
participate in routing, so leakage is irrelevant. They are reported but not
counted as failures.

The remaining heads must reach >= ``MASS_TARGET_THRESHOLD`` (99% by default)
on the target position. Heads below that are flagged with a concrete fix
suggestion (bump Q-scale or ALiBi slope) and the report is written to
``c4_release/docs/SOFTMAX_SHARPNESS_AUDIT.md``.

The harness itself is exercised by ``test_audit_runs_and_writes_report`` —
which is a *non-strict* test: it asserts the audit can run end-to-end and
produces a well-formed report. It does NOT assert any specific head passes
the 99% threshold (that would tautologically lock today's bake; Phase 2 D is
the one that fixes the failing heads).

A separate, *opt-in* gate (``test_strict_all_heads_sharp``) is skipped by
default and only enabled once Phase 2 D has shipped its sharpening fixes;
flipping the env var ``C4_STRICT_SOFTMAX_SHARPNESS=1`` runs it.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm  # noqa: E402
from neural_vm.vm_step import AutoregressiveAttention  # noqa: E402


# -----------------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------------
MASS_TARGET_THRESHOLD = 0.99  # head must put >= 99% of softmax mass on target

# Synthetic context length. Must be large enough to give the softmax a real
# "field of distractors" (so leakage shows up) but small enough that ALiBi
# decay does the work other-positions deserve. 16 mirrors a typical VM-step
# residual slice (5 + 5 + 5 + ...).
S_AUDIT = 16

# Target K position: where we put the head's K-gate features. Must be < Q_POS
# so causal masking allows the Q to see it.
TARGET_K_POS = 4

# Query position: where we run the audit Q.
Q_POS = 8

# Magnitude of the synthetic gate-dim activations (each "on" gate dim is set
# to this scalar). Choosing 1.0 keeps the synthetic context comparable to the
# residual scale of one-hot marker dims (which are 1.0 from the embedding).
GATE_VALUE = 1.0

# Minimum |W| in a column of W_q / W_k (restricted to a head's row-slice) for
# that input dim to count as part of the head's gate set. Tunes how
# aggressively we filter out vestigial weights left by `compact()`.
GATE_WEIGHT_EPS = 1e-3

# How many top-magnitude input dims to use as the head's K-gate / Q-gate set.
# Picking the top dims (rather than all above eps) yields a clean synthetic
# context even when W_k has hundreds of nonzero entries from broad bakes.
TOP_GATE_DIMS = 8


# -----------------------------------------------------------------------------
# Per-head probing
# -----------------------------------------------------------------------------
def _head_gate_dims(W: torch.Tensor, head_idx: int, head_dim: int,
                    top_k: int, eps: float):
    """Return the indices of input dims that this head "gates on" via W.

    W is the full attention weight matrix (shape [D_out, D_in], where
    D_out == num_heads * head_dim for the non-compact path). The head's
    output rows are W[head_idx * HD : (head_idx + 1) * HD, :]. The set of
    input dims it reads is determined by the column-wise L2 norm of that
    row slice — large norm means this input dim strongly contributes to
    the head's projection.

    Returns the indices of the top ``top_k`` columns whose column-norm
    exceeds ``eps`` (so dormant heads return an empty list).
    """
    s, e = head_idx * head_dim, (head_idx + 1) * head_dim
    head_rows = W[s:e]  # [HD, D_in]
    col_norm = head_rows.abs().sum(dim=0)  # [D_in]
    above_eps = (col_norm > eps).nonzero(as_tuple=True)[0]
    if len(above_eps) == 0:
        return torch.empty(0, dtype=torch.long, device=W.device)
    # Pick the top-k by column norm (sorted descending) — these are the
    # input dims that dominate the head's projection.
    vals = col_norm[above_eps]
    order = vals.argsort(descending=True)
    keep = above_eps[order][:top_k]
    return keep


def _build_synthetic_input(d_model: int, target_k_pos: int, q_pos: int,
                           k_gate_dims: torch.Tensor, q_gate_dims: torch.Tensor,
                           seq_len: int, device, dtype):
    """Build x of shape [1, seq_len, d_model] with K-gates lit at target only.

    Strategy:
      * x is initialized to all zeros (the residual-stream prior in our VM is
        that unset dims are 0; embedding lights only specific positions).
      * At ``target_k_pos`` we set ``x[0, target_k_pos, k_gate_dims] = 1.0``.
        This is the "ideal target" — the position the head should attend to.
      * At ``q_pos`` we set ``x[0, q_pos, q_gate_dims] = 1.0``. This makes the
        Q vector mirror what the head expects to see when it's "firing".
      * No other position has either gate set, so all distractor K positions
        produce a zero K vector (modulo any tiny W_k weights below eps).
    """
    x = torch.zeros(1, seq_len, d_model, device=device, dtype=dtype)
    if len(k_gate_dims) > 0:
        x[0, target_k_pos, k_gate_dims] = GATE_VALUE
    if len(q_gate_dims) > 0:
        x[0, q_pos, q_gate_dims] = GATE_VALUE
    return x


def _isolate_head_attn_probs(attn: AutoregressiveAttention,
                             x: torch.Tensor,
                             head_idx: int):
    """Run ``attn`` on x and return the softmax probabilities at the Q row
    for the requested head.

    We re-implement the head's score computation here (against the attn
    module's actual W_q / W_k / scale / ALiBi / causal mask) rather than
    hooking into ``forward()`` — that way we don't get tangled with KV cache
    branches, post-RoPE rotation, or the residual add. The math mirrors
    ``AutoregressiveAttention.forward`` lines that compute ``scores`` and the
    final softmax1 attn weights.

    Returns (attn_probs[seq_len], scores[seq_len]) — probs for the requested
    head at row Q_POS, post-causal-mask, post-softmax1. ``scores`` are the
    pre-softmax logits at that row.
    """
    if getattr(attn, "_is_compact", False):
        raise RuntimeError(
            "compact attention not supported by audit (this codebase only "
            "compacts on opt-in; bake-time models stay non-compact)."
        )
    B, S, D = x.shape
    H = attn.num_heads
    HD = attn.head_dim

    Q = torch.nn.functional.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
    K = torch.nn.functional.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)

    # Optional RoPE (mirrors forward()).
    if attn._rope_cos is not None:
        from neural_vm.base_layers import rotate_half
        cos = attn._rope_cos[:S].unsqueeze(0).unsqueeze(0)
        sin = attn._rope_sin[:S].unsqueeze(0).unsqueeze(0)
        Q = (Q * cos) + (rotate_half(Q) * sin)
        K = (K * cos) + (rotate_half(K) * sin)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale  # [1, H, S, S]

    if attn.alibi_slopes is not None:
        pos = torch.arange(S, device=x.device, dtype=torch.float32)
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()  # [S, S]
        alibi = -attn.alibi_slopes.view(1, H, 1, 1) * dist
        scores = scores + alibi

    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=x.device), diagonal=1
    )
    scores = scores + causal_mask

    # softmax1 with anchor=0 (mirrors ``forward``).
    anchor = torch.zeros((), device=x.device, dtype=scores.dtype)
    max_val = torch.max(scores.amax(dim=-1, keepdim=True), anchor)
    exp_scores = torch.exp(scores - max_val)
    exp_anchor = torch.exp(anchor - max_val)
    probs = exp_scores / (exp_anchor + exp_scores.sum(dim=-1, keepdim=True))

    # Row at Q_POS, head head_idx.
    row_probs = probs[0, head_idx, Q_POS]  # [S]
    row_scores = scores[0, head_idx, Q_POS]  # [S]
    return row_probs, row_scores


def _classify_head(attn: AutoregressiveAttention, head_idx: int):
    """Run the synthetic audit on one (layer, head) and return a result dict.

    Result keys:
      head_idx         : int
      k_gate_dims      : list[int] -- top-magnitude W_k input dims for this head
      q_gate_dims      : list[int] -- top-magnitude W_q input dims for this head
      dormant          : bool      -- both gate sets empty
      mass_on_target   : float     -- softmax prob at TARGET_K_POS
      mass_on_runner_up: float     -- max softmax prob over j != TARGET_K_POS (causal-visible)
      gap_in_logits    : float     -- score@target - score@runner-up
      alibi_slope      : float|None
      head_scale       : float     -- attn.scale (constant per layer; logged
                                      because it interacts with W_k magnitude)
    """
    HD = attn.head_dim
    k_gate = _head_gate_dims(attn.W_k.data, head_idx, HD,
                             top_k=TOP_GATE_DIMS, eps=GATE_WEIGHT_EPS)
    q_gate = _head_gate_dims(attn.W_q.data, head_idx, HD,
                             top_k=TOP_GATE_DIMS, eps=GATE_WEIGHT_EPS)
    dormant = (len(k_gate) == 0 and len(q_gate) == 0)

    slope = None
    if attn.alibi_slopes is not None:
        slope = float(attn.alibi_slopes[head_idx].item())

    result = {
        "head_idx": head_idx,
        "k_gate_dims": k_gate.tolist(),
        "q_gate_dims": q_gate.tolist(),
        "dormant": dormant,
        "qk_coupled": False,
        "mass_on_target": float("nan"),
        "mass_on_runner_up": float("nan"),
        "gap_in_logits": float("nan"),
        "score_on_target": float("nan"),
        "score_on_runner_up": float("nan"),
        "runner_up_pos": -1,
        "alibi_slope": slope,
        "head_scale": float(attn.scale),
    }
    if dormant:
        return result

    device = attn.W_q.device
    dtype = attn.W_q.dtype
    # Q-gate dims that are ALSO in the K-gate set inadvertently light the K
    # projection at Q_POS too, causing spurious self-attention. Exclude them
    # from the Q-side synthetic activation so we measure the head's K-routing
    # in isolation. (The head's "real" inputs typically have positions with
    # disjoint gate features; this approximation matches that.)
    k_gate_set = set(k_gate.tolist())
    q_gate_filtered = torch.tensor(
        [d for d in q_gate.tolist() if d not in k_gate_set],
        dtype=torch.long, device=device,
    )
    # If filtering removed every Q-gate dim, the head's Q and K gates fully
    # overlap. The synthetic probe is fundamentally ambiguous for such heads:
    # any pattern that activates K (the routing signal we want at the target)
    # also activates Q (which would then accidentally activate K at the Q
    # position too). Mark and skip the score measurement for these.
    if len(q_gate_filtered) == 0 and len(q_gate) > 0:
        result["qk_coupled"] = True
        return result
    x = _build_synthetic_input(
        d_model=attn.dim,
        target_k_pos=TARGET_K_POS,
        q_pos=Q_POS,
        k_gate_dims=k_gate.to(device),
        q_gate_dims=q_gate_filtered,
        seq_len=S_AUDIT,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        probs, scores = _isolate_head_attn_probs(attn, x, head_idx)

    # Mass on target.
    mass_on_target = float(probs[TARGET_K_POS].item())

    # Mass on runner-up: max over j in [0, Q_POS] excluding TARGET_K_POS.
    # (Causal visible positions are 0..Q_POS inclusive; positions > Q_POS are
    # masked to -inf and contribute 0 to the softmax denominator after the
    # anchor.)
    causal_slice = probs[:Q_POS + 1].clone()
    causal_slice[TARGET_K_POS] = -1.0  # mask the target so argmax skips it
    runner_up_pos = int(causal_slice.argmax().item())
    mass_on_runner_up = float(probs[runner_up_pos].item())

    # Logit gap between target and runner-up.
    score_target = float(scores[TARGET_K_POS].item())
    score_runner = float(scores[runner_up_pos].item())
    gap_in_logits = score_target - score_runner

    result["mass_on_target"] = mass_on_target
    result["mass_on_runner_up"] = mass_on_runner_up
    result["gap_in_logits"] = gap_in_logits
    result["runner_up_pos"] = runner_up_pos
    result["score_on_target"] = score_target
    result["score_on_runner_up"] = score_runner
    return result


# -----------------------------------------------------------------------------
# Whole-model audit
# -----------------------------------------------------------------------------
def _enumerate_attn_layers(model):
    """Yield (block_idx, attn_module) for every AutoregressiveAttention in
    ``model.blocks``.

    Phase-0 block expansion can produce passthrough blocks with zero-init
    attention; those still type as AutoregressiveAttention. We include them
    so the audit can flag them as dormant (informational) rather than
    silently skip them.
    """
    for i, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        if isinstance(attn, AutoregressiveAttention):
            yield i, attn


def audit_softmax_sharpness(model):
    """Audit every (layer, head) and return a list of result dicts.

    Caller is responsible for compiling the model. The audit only reads
    weights — it never mutates the model.
    """
    results = []
    for layer_idx, attn in _enumerate_attn_layers(model):
        for h in range(attn.num_heads):
            r = _classify_head(attn, h)
            r["layer_idx"] = layer_idx
            results.append(r)
    return results


def _suggest_fix(r):
    """Return a short concrete fix recommendation string for a flagged head.

    Heuristic: estimate the logit-score needed for >= 99% mass and translate
    that into either a Q/K-scale bump or an ALiBi-slope bump.

    Math: with softmax1 anchor=0 and N causal-visible distractor positions,
        mass_on_target = e^(s_t - m) / (1 + sum_j e^(s_j - m))
    For mass >= 0.99 with target dominating, s_t must be ~ ln(99) above
    both the anchor (0) AND any other distractor. So the gap-to-runner-up
    metric is informative but insufficient if the target's *absolute*
    score is too small (the anchor=0 then wins via softmax1).

    Recommendations:
      * if target absolute score s_t < ~5, primary issue is K-scale (or W_k
        magnitude) — boost so s_t reaches ~10 nats above 0;
      * else if gap_to_runner_up < 5, primary issue is positional
        separation — boost ALiBi slope (if ALiBi) or downstream baseline.
    """
    if r["dormant"]:
        return "dormant head (no gates); no fix needed unless this head is supposed to be active"

    target_gap_nats = 4.6  # ln(99): mass ~99% on target vs one rival
    safety_nats = 6.0  # we want headroom against the softmax1 anchor too
    target_abs = r.get("score_on_target", None)  # may be missing in older runs
    cur_gap = r["gap_in_logits"]
    cur_slope = r["alibi_slope"]
    qk_dist = abs(Q_POS - TARGET_K_POS)

    # If we don't have absolute score, fall back to gap-only logic.
    s_t = target_abs if target_abs is not None else cur_gap

    parts = []

    # 1) Target absolute-score check: s_t must exceed the softmax1 anchor (0)
    # by at least ~ln(99) AFTER subtracting the max-shift, OR equivalently
    # the unshifted s_t must clear 0 by safety_nats.
    if s_t < safety_nats:
        # Need to bump magnitude of Q*K product. Estimate factor:
        # we want s_t' = max(safety_nats, current * factor). For positive
        # s_t, factor = safety_nats / max(s_t, 0.5); for non-positive, use
        # a fixed 10x as a first attempt.
        if s_t > 0.5:
            scale_factor = max(2.0, safety_nats / s_t)
        else:
            scale_factor = 10.0
        parts.append(f"bump K-scale ~{scale_factor:.1f}x (raise s_target)")

    # 2) Gap-to-runner-up check: must exceed ln(99) (~4.6) for the runner-up
    # to be drowned. Independent of the absolute-score test above.
    if cur_gap < target_gap_nats:
        if cur_gap > 0.5:
            gap_factor = max(2.0, target_gap_nats / cur_gap)
        else:
            gap_factor = 10.0
        if cur_slope is not None and cur_slope < 1.0 and qk_dist > 0:
            # ALiBi can deliver the gap without touching Q/K scale.
            suggested_slope = max(cur_slope * gap_factor, 1.0)
            parts.append(
                f"raise ALiBi slope {cur_slope:.4g} -> ~{suggested_slope:.2g} "
                f"(close gap@runner-up dist {qk_dist})"
            )
        else:
            parts.append(f"close gap by ~{gap_factor:.1f}x via Q*K bump")

    if not parts:
        # Both criteria are satisfied — but mass is still < 99%? Likely the
        # gap is OK and absolute score is OK but multiple distractors share
        # the runner-up's score. Recommend further sharpening.
        return "marginal: bump K-scale ~2x or ALiBi slope ~2x"

    return "; ".join(parts)


def render_report(results) -> str:
    """Render the audit results as Markdown."""
    lines = []
    lines.append("# Softmax-sharpness audit")
    lines.append("")
    lines.append("Generated by `tests/test_softmax_sharpness.py` per Phase 1 Agent A")
    lines.append("of `docs/ARCH_LEAKAGE_FIX_PLAN.md`. Audits every attention head")
    lines.append("in the model returned by `compile_full_vm()` and measures how")
    lines.append("sharp the head's softmax is when a synthetic context lights")
    lines.append("its W_k gates at exactly one target K position.")
    lines.append("")
    lines.append(f"- Threshold: mass on target >= **{MASS_TARGET_THRESHOLD * 100:.1f}%**")
    lines.append(f"- Synthetic context length: {S_AUDIT}")
    lines.append(f"- Target K position: {TARGET_K_POS}; Q position: {Q_POS}")
    lines.append(f"- Top gate dims per head (by |W| column-norm): {TOP_GATE_DIMS}")
    lines.append("")
    lines.append("## Methodology and caveats")
    lines.append("")
    lines.append(
        "The probe uses `|W_q|` / `|W_k|` column-magnitude to identify each "
        "head's 'gate' input dims, then constructs a synthetic context where "
        "the target K position has those K-gate dims set to 1.0 and the Q "
        "position has the head's Q-gate dims (minus any K-gate overlap) set "
        "to 1.0. The score reported is the actual softmax mass the head "
        "places on the target position under this synthetic input."
    )
    lines.append("")
    lines.append(
        "**Caveats** — interpret failing heads carefully:"
    )
    lines.append("")
    lines.append(
        "- Some heads use **negative** W_q weights at certain input dims to "
        "BLOCK firing on specific patterns (e.g. `MARK_PC_BLOCK = -S*50` "
        "is a common 'don't fire at PC-marker positions' pattern). Lighting "
        "such a dim at the Q position drives the score negative — the head "
        "fails the synthetic probe but is correct in real contexts where "
        "that dim is dark. A very negative `s_target` is a tell-tale sign."
    )
    lines.append("")
    lines.append(
        "- Heads keyed on dims OUTSIDE the top-N column-norm set may have "
        "their actual routing signal underrepresented in the synthetic "
        "context. Increase `TOP_GATE_DIMS` and re-run if a specific head "
        "looks suspicious."
    )
    lines.append("")
    lines.append(
        "- For the audit's intended purpose (catching softmax-leakage in "
        "the L4-L8 head zone identified in `ARCH_LEAKAGE_FIX_PLAN.md`), the "
        "useful signal is heads with **positive `s_target`** but low "
        "**mass@target** — those are the heads that "
        "see their target K but fail to drown the distractors. They are the "
        "primary Phase 2 D candidates."
    )
    lines.append("")

    def _is_active(r):
        return (not r["dormant"]) and (not r.get("qk_coupled", False))

    passing = [r for r in results if _is_active(r) and r["mass_on_target"] >= MASS_TARGET_THRESHOLD]
    failing = [r for r in results if _is_active(r) and r["mass_on_target"] < MASS_TARGET_THRESHOLD]
    dormant = [r for r in results if r["dormant"]]
    coupled = [r for r in results if r.get("qk_coupled", False)]

    # Split failing into primary leakage vs probe-artifact for the summary.
    primary_failing_count = sum(1 for r in failing if r["score_on_target"] >= 0.0)
    artifact_failing_count = len(failing) - primary_failing_count

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Active heads passing (mass >= {MASS_TARGET_THRESHOLD * 100:.1f}%): **{len(passing)}**")
    lines.append(f"- Active heads failing: **{len(failing)}** (primary leakage: **{primary_failing_count}**, probe artifacts: **{artifact_failing_count}**)")
    lines.append(f"- QK-coupled heads (probe ambiguous, informational): **{len(coupled)}**")
    lines.append(f"- Dormant heads (no gates, informational): **{len(dormant)}**")
    lines.append(f"- Total heads audited: {len(results)}")
    lines.append("")

    # Split failing heads into two buckets by `s_target`:
    #   * primary: positive s_target — head saw its target but couldn't drown
    #     distractors; these are the Phase 2 D candidates.
    #   * artifact: heavily negative s_target — likely a probe artifact (the
    #     synthetic input lights a dim the head uses negatively as a gate).
    def _is_primary(r):
        return r["score_on_target"] >= 0.0
    primary_failing = [r for r in failing if _is_primary(r)]
    artifact_failing = [r for r in failing if not _is_primary(r)]

    if primary_failing:
        lines.append("## Failing heads — primary leakage candidates (Phase 2 Agent D)")
        lines.append("")
        lines.append(
            "These heads register a positive `s_target` (the synthetic K "
            "pattern *was* the strongest match for the head) but fail to "
            "concentrate softmax mass on it. They are the **leak-prone heads** "
            "Phase 2 D should sharpen."
        )
        lines.append("")
        lines.append("| Layer | Head | mass@target | mass@runner-up | runner-up pos | s_target | gap (logits) | ALiBi slope | scale | fix recommendation |")
        lines.append("|------:|-----:|-----------:|--------------:|--------------:|---------:|-------------:|-----------:|------:|:--------------------|")
        for r in sorted(primary_failing, key=lambda r: (r["layer_idx"], r["head_idx"])):
            slope_str = f"{r['alibi_slope']:.4g}" if r["alibi_slope"] is not None else "RoPE"
            lines.append(
                f"| {r['layer_idx']} | {r['head_idx']} "
                f"| {r['mass_on_target']:.4f} | {r['mass_on_runner_up']:.4f} "
                f"| {r['runner_up_pos']} | {r['score_on_target']:.3f} "
                f"| {r['gap_in_logits']:.3f} "
                f"| {slope_str} | {r['head_scale']:.4f} "
                f"| {_suggest_fix(r)} |"
            )
        lines.append("")

    if artifact_failing:
        lines.append("## Failing heads — probable probe artifacts (review only)")
        lines.append("")
        lines.append(
            "These heads have **negative** `s_target` against the synthetic "
            "K pattern, which usually indicates a negative-weighted (anti-gate) "
            "input dim the synthetic context lights. Review each case manually; "
            "in most cases the head is correct as baked, but a real-context "
            "probe would be more informative."
        )
        lines.append("")
        lines.append("| Layer | Head | mass@target | s_target | gap | ALiBi slope | likely diagnosis |")
        lines.append("|------:|-----:|-----------:|---------:|----:|-----------:|:------------------|")
        for r in sorted(artifact_failing, key=lambda r: (r["layer_idx"], r["head_idx"])):
            slope_str = f"{r['alibi_slope']:.4g}" if r["alibi_slope"] is not None else "RoPE"
            if r["score_on_target"] < -50.0:
                diagnosis = "synthetic input lights an anti-gate dim (W has strong negative weight); audit can't measure real-context sharpness"
            else:
                diagnosis = "mild negative score; may indicate Q-K mismatch (head's intended use needs a different residual setup than the audit constructs)"
            lines.append(
                f"| {r['layer_idx']} | {r['head_idx']} "
                f"| {r['mass_on_target']:.4f} | {r['score_on_target']:.3f} "
                f"| {r['gap_in_logits']:.3f} | {slope_str} | {diagnosis} |"
            )
        lines.append("")

    if passing:
        lines.append("## Passing heads")
        lines.append("")
        lines.append("| Layer | Head | mass@target | gap (logits) | ALiBi slope |")
        lines.append("|------:|-----:|-----------:|-------------:|-----------:|")
        for r in sorted(passing, key=lambda r: (r["layer_idx"], r["head_idx"])):
            slope_str = f"{r['alibi_slope']:.4g}" if r["alibi_slope"] is not None else "RoPE"
            lines.append(
                f"| {r['layer_idx']} | {r['head_idx']} "
                f"| {r['mass_on_target']:.4f} | {r['gap_in_logits']:.3f} "
                f"| {slope_str} |"
            )
        lines.append("")

    if coupled:
        lines.append("## QK-coupled heads (probe ambiguous)")
        lines.append("")
        lines.append(
            "These heads' top-k Q-gate input dims are a subset of their K-gate "
            "input dims (every dim that the head reads for Q also routes "
            "into K). The synthetic probe can't distinguish 'attend to "
            "TARGET_K_POS' from 'attend to self' for such heads — any pattern "
            "that lights K at the target also lights K at the Q position. "
            "They need *real-context* sharpness evaluation rather than the "
            "synthetic probe used here. They are listed informationally; "
            "their leakage status must be checked via a downstream test."
        )
        lines.append("")
        lines.append("| Layer | Head | shared gate dims | ALiBi slope |")
        lines.append("|------:|-----:|-----------------:|-----------:|")
        for r in sorted(coupled, key=lambda r: (r["layer_idx"], r["head_idx"])):
            slope_str = f"{r['alibi_slope']:.4g}" if r["alibi_slope"] is not None else "RoPE"
            lines.append(
                f"| {r['layer_idx']} | {r['head_idx']} "
                f"| {sorted(set(r['q_gate_dims']) & set(r['k_gate_dims']))[:6]} "
                f"| {slope_str} |"
            )
        lines.append("")

    if dormant:
        lines.append("## Dormant heads (no positive W_q/W_k gates above eps)")
        lines.append("")
        lines.append(
            "These heads have no input dims whose column-norm in either W_q "
            "or W_k exceeds the gate-eps threshold. They contribute residual "
            "passthrough only — if any of these are **supposed** to be "
            "active, the bake for that head has silently failed."
        )
        lines.append("")
        lines.append("| Layer | Head | ALiBi slope |")
        lines.append("|------:|-----:|-----------:|")
        for r in sorted(dormant, key=lambda r: (r["layer_idx"], r["head_idx"])):
            slope_str = f"{r['alibi_slope']:.4g}" if r["alibi_slope"] is not None else "RoPE"
            lines.append(f"| {r['layer_idx']} | {r['head_idx']} | {slope_str} |")
        lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def compiled_model():
    """Compile once per pytest session to amortize the ~minute-long bake."""
    model, _ = compile_full_vm()
    model.eval()
    return model


def test_audit_runs_and_writes_report(compiled_model):
    """Non-strict: run the audit end-to-end and write the report.

    Asserts:
      * the audit returns one result per (layer, head),
      * every result row is well-formed (required keys present),
      * the rendered report is written to docs/SOFTMAX_SHARPNESS_AUDIT.md.

    Does NOT assert mass thresholds — that's the job of
    ``test_strict_all_heads_sharp`` once Phase 2 Agent D has shipped.
    """
    results = audit_softmax_sharpness(compiled_model)

    # Sanity: at least one head per attn layer.
    n_attn_layers = sum(
        1 for b in compiled_model.blocks
        if isinstance(b.attn, AutoregressiveAttention)
    )
    assert n_attn_layers > 0, "model has no AutoregressiveAttention layers"
    assert len(results) > 0, "audit produced no results"

    required_keys = {
        "layer_idx", "head_idx", "dormant",
        "mass_on_target", "mass_on_runner_up", "gap_in_logits",
        "alibi_slope", "head_scale", "k_gate_dims", "q_gate_dims",
    }
    for r in results:
        missing = required_keys - r.keys()
        assert not missing, f"result row missing keys: {missing} ({r})"

    # At least one active (non-dormant, non-coupled) head — otherwise the
    # test is vacuous and the model is broken in a way the audit wouldn't
    # catch.
    active = [r for r in results if not r["dormant"] and not r.get("qk_coupled", False)]
    assert len(active) > 0, "no active heads found by gate-norm probe"

    # Write the report next to the plan doc so it's easy to read.
    report = render_report(results)
    doc_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "docs", "SOFTMAX_SHARPNESS_AUDIT.md"
    ))
    with open(doc_path, "w") as f:
        f.write(report)

    # Echo the pass/fail summary to pytest's captured stdout so CI logs
    # surface it without a second pytest invocation.
    passing = [r for r in active if r["mass_on_target"] >= MASS_TARGET_THRESHOLD]
    failing = [r for r in active if r["mass_on_target"] < MASS_TARGET_THRESHOLD]
    dormant = [r for r in results if r["dormant"]]
    coupled = [r for r in results if r.get("qk_coupled", False)]
    print(
        f"\n[softmax-sharpness audit] "
        f"passing={len(passing)} failing={len(failing)} "
        f"coupled={len(coupled)} dormant={len(dormant)} total={len(results)}"
    )
    print(f"[softmax-sharpness audit] report written to: {doc_path}")


@pytest.mark.skipif(
    os.environ.get("C4_STRICT_SOFTMAX_SHARPNESS") != "1",
    reason=(
        "Phase 1 audit is observational only — set C4_STRICT_SOFTMAX_SHARPNESS=1 "
        "to enforce the >=99% threshold (Phase 2 Agent D gate)."
    ),
)
def test_strict_all_heads_sharp(compiled_model):
    """Strict gate (skipped until Phase 2 D lands).

    Once Phase 2 D ("sharpen-leaky-attention-heads") has bumped Q-scale /
    ALiBi slope per the audit report, flip this gate on by setting
    ``C4_STRICT_SOFTMAX_SHARPNESS=1`` in CI. The gate then prevents future
    bakes from regressing back into softmax-leakage territory.
    """
    results = audit_softmax_sharpness(compiled_model)
    failing = [
        r for r in results
        if (not r["dormant"]) and (not r.get("qk_coupled", False))
        and r["mass_on_target"] < MASS_TARGET_THRESHOLD
    ]
    msg_lines = [
        f"L{r['layer_idx']} h{r['head_idx']}: "
        f"mass@target={r['mass_on_target']:.4f}, "
        f"gap={r['gap_in_logits']:.3f}, "
        f"slope={r['alibi_slope']}, "
        f"suggest: {_suggest_fix(r)}"
        for r in failing
    ]
    assert not failing, (
        f"{len(failing)} active heads below {MASS_TARGET_THRESHOLD * 100:.1f}% "
        f"target-mass threshold:\n  " + "\n  ".join(msg_lines)
    )


if __name__ == "__main__":
    # Allow ad-hoc invocation: `python tests/test_softmax_sharpness.py`
    # produces the report and prints the pass/fail summary.
    model, _ = compile_full_vm()
    model.eval()
    results = audit_softmax_sharpness(model)
    report = render_report(results)
    out_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "docs", "SOFTMAX_SHARPNESS_AUDIT.md"
    ))
    with open(out_path, "w") as f:
        f.write(report)
    def _act(r):
        return (not r["dormant"]) and (not r.get("qk_coupled", False))
    passing = [r for r in results if _act(r) and r["mass_on_target"] >= MASS_TARGET_THRESHOLD]
    failing = [r for r in results if _act(r) and r["mass_on_target"] < MASS_TARGET_THRESHOLD]
    dormant = [r for r in results if r["dormant"]]
    coupled = [r for r in results if r.get("qk_coupled", False)]
    print(
        f"[softmax-sharpness audit] "
        f"passing={len(passing)} failing={len(failing)} "
        f"coupled={len(coupled)} dormant={len(dormant)} total={len(results)}"
    )
    print(f"[softmax-sharpness audit] report at: {out_path}")
