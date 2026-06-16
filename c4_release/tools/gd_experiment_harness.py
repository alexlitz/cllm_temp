#!/usr/bin/env python3
"""Gradient-descent stability / effectiveness experiment harness.

RESEARCH harness (NOT a production fix). It trains a *COPY* of the hand-built
declarative C4 neural VM under gradient descent and measures two things:

  (A) STABILITY  — do currently-CORRECT programs stay correct under GD, or does
                   GD clobber the carefully-constructed sparse one-hot structure?
  (B) EFFECTIVENESS — does GD learn currently-FAILING programs, and WHERE does
                   it plateau (does it stall at the same architectural caps the
                   declarative DSL addresses, e.g. the byte-1 16-cell emission
                   cap that blocks edge_literal)?

It NEVER touches the production model, the ops/ files, or any committed weights.
``compile_full_vm_dynamic(disk_cache=False)`` builds a fresh in-memory module;
all training mutates that copy's parameters only.

Design
======
The codebase is a COMPILER, not a trainer — no loss/optimizer exists. This
harness supplies:

1. **Teacher-forced data** from the DraftVM oracle. For each program we build:
     * the code/data PREFIX (``runner._serial._build_context`` — the same
       bytecode/data tape the inference runner feeds the model), then
     * the flattened 35-token-per-step ORACLE TAPE: for every VM step we append
       ``DraftVM.draft_tokens()`` (the byte-identity-correct next 35 tokens the
       model *should* emit). This is exactly the reference the production
       fail-fast/strict-trace path compares the model against, so a model that
       emits the oracle tape token-for-token PASSES full_trace by construction.

2. **Next-token cross-entropy loss** on the deterministic VM-step tokens only.
   The model is causal: logits at position ``p`` predict the token at ``p+1``.
   We compute CE over positions whose TARGET lands inside the oracle tape
   (prefix targets are masked out — the model is not asked to predict its own
   bytecode), and we additionally mask the per-step ``_UNSAFE_OFFSETS``
   (offsets 26..33 = the MEM addr/val metadata bytes), because the embedding's
   MEM-metadata injection makes those positions disagree between the
   full-section-visible (oracle tape) view and the model's would-be
   one-token-at-a-time emission — they are NOT a faithful supervision target
   (this mirrors ``batched_pure_neural._UNSAFE_OFFSETS`` and the strict_trace
   criterion's safe-offset set).

   SANITY: at the hand-built weights this loss is ~0 on PASSING programs
   (the model already emits the oracle tape). The harness asserts this.

3. **Evaluation** = the canonical full_trace pass/fail, via the SAME
   ``run_batch_fail_fast(criterion="full_trace")`` the canonical runner uses,
   but pointed at the *trained copy* (we wrap the trained model in a
   ``BatchedPureNeuralRunner`` that reuses the already-built model object). A
   program PASSES iff every completed VM step's decoded (PC, AX) matches the
   declarative oracle through HALT.

This module is import-only machinery; the driver scripts
(``gd_experiment_run.py``) call it.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Cut allocator fragmentation OOM — must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402

from neural_vm.vm_step import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402

# The metadata offsets inside each 35-token VM step whose model emission is
# NOT a faithful flat-forward supervision target. Mirrors
# ``batched_pure_neural._UNSAFE_OFFSETS`` (MEM addr bytes 26..29 + val bytes
# 30..33).
_UNSAFE_OFFSETS = frozenset(range(26, 34))
STEP_TOKENS = Token.STEP_TOKENS  # 35

# Named supervision-offset sets (which of the 35 per-step tokens to put a CE
# target on). The default is PC_AX — the bytes the canonical full_trace
# criterion actually checks. This matters because flat-forward teacher-forced
# token-identity is STRICTER than full_trace: on a PASSING program the model
# legitimately mismatches the oracle tape at
#   * offset 0 (the REG_PC marker), which production RE-ANCHORS via the Python
#     STEP_END dispatch (the model never predicts it from a flat forward), and
#   * the STACK0 / SP / BP / STEP_END bytes, which carry cross-step ZERO /
#     stale propagation that full_trace tolerates (it checks only PC + AX).
# So PC_AX is the only offset set on which the hand-built weights give ~0 loss
# on passing programs (the harness sanity check). SAFE_ALL (all non-_UNSAFE,
# non-marker offsets) is provided for completeness / a stricter view, and
# AX_ONLY isolates the result register (where the byte-1 emission cap lives).
#
# PC = step offsets 1..4 (4 LE bytes after the REG_PC marker at 0);
# AX = step offsets 6..9 (after REG_AX marker at 5). Markers themselves
# (0, 5, 10, 15, 20, 25) are EXCLUDED from PC_AX/SAFE_ALL because production
# re-anchors them.
_PC_BYTES = frozenset(range(1, 5))      # PC[0..3]
_AX_BYTES = frozenset(range(6, 10))     # AX[0..3]
_MARKERS = frozenset({0, 5, 10, 15, 20, 25})
# SAFE_ALL = every offset that is neither a marker nor an unsafe MEM byte.
_SAFE_ALL = frozenset(
    o for o in range(STEP_TOKENS) if o not in _UNSAFE_OFFSETS and o not in _MARKERS
)

SUPERVISION_SETS: Dict[str, frozenset] = {
    "pc_ax": _PC_BYTES | _AX_BYTES,
    "ax_only": _AX_BYTES,
    "safe_all": _SAFE_ALL,
}


# ---------------------------------------------------------------------------
# Program preparation: compile + oracle + teacher-forced tape
# ---------------------------------------------------------------------------


@dataclass
class PreparedProgram:
    """One program ready for teacher-forced training + full_trace eval."""

    idx: int
    description: str
    cluster: str
    suite_expected: int
    bytecode: List[int]
    data: bytes
    decl_exit: int
    decl_steps: int
    # The model input tape: prefix context tokens + flattened oracle tokens.
    input_tape: List[int]
    prefix_len: int
    # Boolean supervision mask, len == len(input_tape). mask[p] is True when the
    # TARGET of position p (i.e. input_tape[p+1]) is a clean oracle-step token.
    target_mask: List[bool]
    # baseline (hand-built) full_trace pass verdict, filled by eval.
    baseline_pass: Optional[bool] = None


def _cluster_of(description: str) -> str:
    import re

    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


def build_oracle_tape(
    runner,
    bytecode: List[int],
    data: bytes,
    *,
    max_oracle_steps: Optional[int] = None,
    supervised_offsets: frozenset = SUPERVISION_SETS["pc_ax"],
) -> Tuple[List[int], int, List[bool], int, int]:
    """Build the teacher-forced tape + supervision mask for one program.

    Returns ``(input_tape, prefix_len, target_mask, decl_exit, decl_steps)``.

    The tape is ``prefix + step0_tokens + step1_tokens + ... + lastStep_tokens``
    where each ``stepK_tokens`` is ``DraftVM.draft_tokens()`` AFTER stepping the
    VM K+1 times (the 35-token reference for step K, ending in STEP_END or, on
    the halting step, HALT). ``target_mask[p]`` is True iff ``input_tape[p+1]``
    is an oracle-region token whose within-step offset is in
    ``supervised_offsets`` (default: the PC + AX bytes the full_trace criterion
    checks).
    """
    # Prefix: the exact bytecode/data context the inference runner feeds.
    prefix = runner._serial._build_context(bytecode, data or b"", [], "")
    prefix = list(prefix)
    prefix_len = len(prefix)

    # Oracle: step the DraftVM, capturing the 35-token draft after each step.
    vm = DraftVM(list(bytecode))
    if isinstance(data, (bytes, bytearray, list)):
        for k, b in enumerate(data):
            vm.memory[0x10000 + k] = int(b)
    cap = max_oracle_steps if max_oracle_steps is not None else 4096
    step_tokens: List[List[int]] = []
    decl_steps = 0
    for _ in range(cap):
        if vm.halted:
            break
        if not vm.step():
            break
        step_tokens.append([int(t) for t in vm.draft_tokens()])
        decl_steps += 1
        if vm.halted:
            break
    decl_exit = int(vm.ax) & 0xFFFFFFFF

    oracle_flat: List[int] = []
    for toks in step_tokens:
        oracle_flat.extend(toks)

    input_tape = prefix + oracle_flat
    n = len(input_tape)

    # target_mask[p] supervises the prediction of input_tape[p+1]. We supervise
    # a position iff its target index lies in the oracle region AND the target's
    # within-step offset is in the requested supervised set.
    target_mask = [False] * n
    oracle_start = prefix_len  # first oracle token index in input_tape
    for p in range(n - 1):
        t = p + 1  # target index
        if t < oracle_start:
            continue  # target is still inside the prefix -> not supervised
        within = t - oracle_start  # offset of target within the oracle tape
        offset = within % STEP_TOKENS
        if offset not in supervised_offsets:
            continue
        target_mask[p] = True

    return input_tape, prefix_len, target_mask, decl_exit, decl_steps


def prepare_programs(
    runner,
    ids: List[int],
    *,
    max_oracle_steps: Optional[int] = None,
    supervision: str = "pc_ax",
) -> List[PreparedProgram]:
    """Compile + oracle + teacher-force every selected id.

    Programs that fail to compile or whose declarative oracle does not halt /
    exceeds ``max_oracle_steps`` are dropped (with a stderr note) — the sample
    is curated for tractable (<= a few dozen step) programs.
    """
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs

    sup_offsets = SUPERVISION_SETS[supervision]
    all_tests = generate_test_programs()
    out: List[PreparedProgram] = []
    for idx in ids:
        if not (0 <= idx < len(all_tests)):
            continue
        source, expected, description = all_tests[idx]
        try:
            bytecode, data = compile_c(source)
        except Exception as exc:  # noqa: BLE001
            print(f"[prep] id={idx} compile error: {exc!r}", file=sys.stderr)
            continue
        try:
            tape, plen, mask, dexit, dsteps = build_oracle_tape(
                runner,
                bytecode,
                data,
                max_oracle_steps=max_oracle_steps,
                supervised_offsets=sup_offsets,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[prep] id={idx} oracle error: {exc!r}", file=sys.stderr)
            continue
        if dsteps == 0:
            print(f"[prep] id={idx} oracle produced 0 steps; skip", file=sys.stderr)
            continue
        if max_oracle_steps is not None and dsteps > max_oracle_steps:
            print(
                f"[prep] id={idx} decl_steps={dsteps} > cap; skip",
                file=sys.stderr,
            )
            continue
        out.append(
            PreparedProgram(
                idx=idx,
                description=description,
                cluster=_cluster_of(description),
                suite_expected=int(expected),
                bytecode=list(bytecode),
                data=data,
                decl_exit=dexit,
                decl_steps=dsteps,
                input_tape=tape,
                prefix_len=plen,
                target_mask=mask,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Teacher-forced loss
# ---------------------------------------------------------------------------


def _pad_batch(
    tapes: List[List[int]],
    masks: List[List[bool]],
    device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a minibatch of (tape, mask) into tensors.

    Returns ``(input_ids, target_ids, loss_mask, _lengths)``:
      * ``input_ids``  [B, L]   padded with Token.HALT (matches runner padding)
      * ``target_ids`` [B, L]   input_ids shifted left by one (target at p is
                                 input at p+1); last column is padding.
      * ``loss_mask``  [B, L]   bool; True where position p supervises a clean
                                 oracle target (== the program's target_mask,
                                 padded False).
    """
    B = len(tapes)
    lens = [len(t) for t in tapes]
    L = max(lens)
    input_ids = torch.full((B, L), Token.HALT, dtype=torch.long)
    target_ids = torch.full((B, L), Token.HALT, dtype=torch.long)
    loss_mask = torch.zeros((B, L), dtype=torch.bool)
    for b, (tape, mask, n) in enumerate(zip(tapes, masks, lens)):
        t = torch.tensor(tape, dtype=torch.long)
        input_ids[b, :n] = t
        # target at position p is input at p+1
        if n > 1:
            target_ids[b, : n - 1] = t[1:]
        m = torch.tensor(mask, dtype=torch.bool)
        loss_mask[b, :n] = m
    lengths = torch.tensor(lens, dtype=torch.long)
    return (
        input_ids.to(device),
        target_ids.to(device),
        loss_mask.to(device),
        lengths.to(device),
    )


# Default logit temperature for the CE loss. The hand-built weights emit
# extreme one-hot logits (|logit| up to ~1e26 from SwiGLU scale S=100 chained
# across 50 blocks). Raw softmax/CE over those is numerically degenerate
# (gradients are 0 on saturated-correct positions and ~inf on the few wrong
# ones), which makes the loss unusable for GD. Dividing logits by a temperature
# T preserves the ARGMAX (so the model's decode behaviour / full_trace verdict
# is unchanged) while bringing the CE into a well-conditioned range where
# gradients are finite and informative. T is applied identically at every
# position, so it is just a global rescale of the supervision signal — it does
# NOT change which token is the model's prediction. We default to 1e4 (logits
# land roughly in [-1e22, 1e22] still large but finite-grad after the log-sum-
# exp clamp; in practice the dominant competing logits are O(1e2..1e8), so T=1e4
# resolves the bulk of positions to a clean soft target). The loss is reported
# at this T; ``token_acc`` is T-invariant (argmax).
_DEFAULT_LOGIT_TEMPERATURE = 1.0e4

# Clamp band for the temperature-scaled logits before CE (keeps log-sum-exp in
# fp32 range; exp(60) ~ 1.1e26 << fp32 max ~3.4e38). Argmax-preserving.
_LOGIT_CLAMP = 60.0


def _teacher_forced_loss_single(
    model,
    progs: List[PreparedProgram],
    device,
    *,
    temperature: float = _DEFAULT_LOGIT_TEMPERATURE,
) -> Tuple[torch.Tensor, int, float]:
    """One-forward CE core. Returns ``(sum_ce, n_sup, sum_correct)``.

    ``sum_ce`` is the SUM (not mean) of the per-supervised-position CE and is
    differentiable; ``n_sup`` is the number of supervised positions; and
    ``sum_correct`` is the number of argmax-correct positions (a float, for
    accumulating the T-invariant token-accuracy across chunks).
    """
    tapes = [p.input_tape for p in progs]
    masks = [p.target_mask for p in progs]
    input_ids, target_ids, loss_mask, _ = _pad_batch(tapes, masks, device)
    logits = model.forward(input_ids)  # [B, L, V]
    B, L, V = logits.shape
    flat_logits = logits.reshape(B * L, V)
    flat_targets = target_ids.reshape(B * L)
    flat_mask = loss_mask.reshape(B * L)
    sel = flat_mask.nonzero(as_tuple=False).flatten()
    if sel.numel() == 0:
        return logits.sum() * 0.0, 0, 0.0
    sel_logits = flat_logits.index_select(0, sel)
    sel_targets = flat_targets.index_select(0, sel)
    scaled = sel_logits / float(temperature)
    # Numerical safety: even after /T the hand-built one-hots can reach ~1e22,
    # whose exp() overflows to inf -> NaN CE/gradient. Clamp the scaled logits
    # to a finite band. This does NOT change the argmax (decode behaviour) and
    # preserves the gradient SIGN toward the correct token; it only caps the
    # magnitude so the log-sum-exp stays finite. ``_LOGIT_CLAMP`` is chosen so
    # exp() is representable in fp32 (exp(80) ~ 5.5e34, well under fp32 max).
    scaled = scaled.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
    ce_sum = torch.nn.functional.cross_entropy(
        scaled, sel_targets, reduction="sum"
    )
    with torch.no_grad():
        pred = sel_logits.argmax(dim=-1)  # T-invariant
        n_correct = float((pred == sel_targets).sum().item())
    return ce_sum, int(sel.numel()), n_correct


def teacher_forced_loss(
    model,
    progs: List[PreparedProgram],
    device,
    *,
    reduce: str = "mean",
    temperature: float = _DEFAULT_LOGIT_TEMPERATURE,
    chunk: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Temperature-scaled next-token CE on the supervised oracle tokens.

    Returns ``(loss, stats)`` where ``stats`` has ``ce`` (mean T-scaled CE on
    supervised positions) and ``token_acc`` (fraction whose argmax == target —
    the T-invariant teacher-forced token-identity proxy for full_trace).

    The logits are divided by ``temperature`` before CE (see
    ``_DEFAULT_LOGIT_TEMPERATURE``): this leaves the argmax (decode behaviour)
    untouched but conditions the loss for gradient descent.

    ``chunk`` groups the programs so the forward never materialises the whole
    sample's [B, L, V] logits at once (the full-sample REPORTING forward of
    ~170 long tapes OOMs a 24 GB GPU). When ``chunk`` is given, the per-chunk
    CE-sums are accumulated (sum-then-divide gives the identical mean CE as one
    forward — CE is position-additive — so the reported number is unchanged;
    only the peak memory drops). Training minibatches (small ``B``) pass
    ``chunk=None`` for a single forward. NOTE: with grad enabled, a chunked call
    builds a graph spanning all chunks (memory grows with chunk count); use
    chunking under ``no_grad`` for reporting, or small ``B`` for the train step.
    """
    if chunk is None or chunk >= len(progs):
        ce_sum, n_sup, n_correct = _teacher_forced_loss_single(
            model, progs, device, temperature=temperature
        )
        if n_sup == 0:
            return ce_sum, {"ce": 0.0, "token_acc": 1.0, "n_sup": 0}
        ce_mean = ce_sum / float(n_sup)
        stats = {
            "ce": float(ce_mean.item()),
            "token_acc": n_correct / float(n_sup),
            "n_sup": n_sup,
        }
        return (ce_sum if reduce == "sum" else ce_mean), stats

    total_ce_sum = None
    total_nsup = 0
    total_correct = 0.0
    for start in range(0, len(progs), chunk):
        sub = progs[start : start + chunk]
        ce_sum, n_sup, n_correct = _teacher_forced_loss_single(
            model, sub, device, temperature=temperature
        )
        total_ce_sum = ce_sum if total_ce_sum is None else total_ce_sum + ce_sum
        total_nsup += n_sup
        total_correct += n_correct
    if total_nsup == 0:
        return total_ce_sum, {"ce": 0.0, "token_acc": 1.0, "n_sup": 0}
    ce_mean = total_ce_sum / float(total_nsup)
    stats = {
        "ce": float(ce_mean.item()),
        "token_acc": total_correct / float(total_nsup),
        "n_sup": total_nsup,
    }
    return (total_ce_sum if reduce == "sum" else ce_mean), stats


# ---------------------------------------------------------------------------
# full_trace evaluation against the trained copy
# ---------------------------------------------------------------------------


def make_runner_for_model(model):
    """Wrap an already-built (and possibly trained) model in a runner.

    Reuses the public ``BatchedPureNeuralRunner`` by handing it a thin
    ``AutoregressiveVMRunner``-like shell whose ``.model`` is our trained copy.
    We construct a real ``AutoregressiveVMRunner`` once (cold) and then swap its
    ``.model`` to our trained module so the fail-fast machinery (DraftVM oracle,
    per-step (PC,AX) compare) runs against the trained weights.
    """
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    # csr_inference=False: training mutates dense parameters; the CSR shim would
    # read stale sparse copies. Disable it for the experiment.
    os.environ.setdefault("C4_CSR_INFERENCE", "0")
    runner = BatchedPureNeuralRunner(csr_inference=False)
    runner.model = model
    runner._serial.model = model
    runner._device = next(model.parameters()).device
    return runner


@torch.no_grad()
def eval_full_trace(
    runner,
    progs: List[PreparedProgram],
    *,
    spec_k: int = 1,
    max_context_window: int = 512,
    chunk: int = 16,
) -> Dict[int, bool]:
    """Return {idx: passed} via the canonical full_trace criterion.

    Uses ``run_batch_fail_fast(criterion="full_trace")`` (the canonical runner's
    pass criterion) against ``runner.model`` (the trained copy). Chunked to keep
    memory bounded on a shared GPU.
    """
    model = runner.model
    was_training = model.training
    model.eval()
    verdicts: Dict[int, bool] = {}
    try:
        for start in range(0, len(progs), chunk):
            sub = progs[start : start + chunk]
            results = runner.run_batch_fail_fast(
                [p.bytecode for p in sub],
                data_list=[p.data for p in sub],
                expected_steps_list=[p.decl_steps for p in sub],
                max_steps=None,
                max_context_window=max_context_window,
                spec_k=spec_k,
                criterion="full_trace",
            )
            for p, r in zip(sub, results):
                verdicts[p.idx] = (r.get("status") == "pass")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
    return verdicts


# ---------------------------------------------------------------------------
# Convenience: split a prepared sample into passing / failing subsets at the
# hand-built baseline (filled by an eval at step 0).
# ---------------------------------------------------------------------------


@dataclass
class SampleSplit:
    all_progs: List[PreparedProgram]
    passing_ids: List[int] = field(default_factory=list)
    failing_ids: List[int] = field(default_factory=list)

    def passing(self) -> List[PreparedProgram]:
        s = set(self.passing_ids)
        return [p for p in self.all_progs if p.idx in s]

    def failing(self) -> List[PreparedProgram]:
        s = set(self.failing_ids)
        return [p for p in self.all_progs if p.idx in s]


def split_by_baseline(
    progs: List[PreparedProgram], baseline_verdicts: Dict[int, bool]
) -> SampleSplit:
    split = SampleSplit(all_progs=progs)
    for p in progs:
        v = bool(baseline_verdicts.get(p.idx, False))
        p.baseline_pass = v
        if v:
            split.passing_ids.append(p.idx)
        else:
            split.failing_ids.append(p.idx)
    return split


# Forward-residual clamp band for fp32 training. The hand-built model emits a
# ~5e25 residual at the OUTPUT/AX-byte1 emission blocks (39+); fp32 max is
# ~3.4e38 so the clean forward is finite, but it sits so close to the overflow
# wall that ANY weight perturbation (even lr=1e-6, grad clipped to norm 1.0)
# tips a downstream matmul accumulation to inf -> NaN that poisons the whole
# residual. ``install_overflow_guards`` registers forward hooks that clamp each
# block's output residual to +/-_RESIDUAL_CLAMP. This is a TRAINING-HARNESS
# wrapper (hooks, not a weight/op edit): it preserves the argmax (clamp band is
# 100,000x above the 5e25 signal) and lets fp32 batched GD proceed without the
# NaN cascade, so we can actually MEASURE effectiveness/stability instead of
# every run dying at step 1. Without it, "GD instantly NaNs" is the only
# finding; with it, we can separate true degradation from fp32 overflow.
_RESIDUAL_CLAMP = 1.0e30


def install_overflow_guards(model, clamp: float = _RESIDUAL_CLAMP):
    """Register forward hooks that NaN-scrub + clamp EVERY module's output.

    The hand-built model's fp32 forward is finite at the exact hand-built
    weights, but a tiny weight perturbation makes an INTERNAL matmul/softmax of
    the final emission blocks overflow to inf and then nan (inf-inf in a
    residual add / inf in a softmax) — clamping only the per-block OUTPUT is too
    late because the nan already formed inside. So we hook every module (leaf +
    container) and, on its output tensor(s), replace non-finite values with the
    clamp band and clamp the magnitude. This keeps the forward finite under
    perturbation while preserving the argmax (the clamp band is 100,000x above
    the ~5e25 emission signal, and nan_to_num only touches positions that would
    otherwise be inf/nan — i.e. positions GD has not yet made meaningful).

    Returns hook handles. This is a harness wrapper (hooks only); it never edits
    a weight or an op.
    """
    handles = []

    def _scrub(t):
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            t = torch.nan_to_num(t, nan=0.0, posinf=clamp, neginf=-clamp)
            return t.clamp(-clamp, clamp)
        return t

    def _hook(_module, _inp, out):
        if isinstance(out, torch.Tensor):
            return _scrub(out)
        if isinstance(out, tuple):
            return tuple(_scrub(o) for o in out)
        return out

    for module in model.modules():
        if module is model:
            continue
        handles.append(module.register_forward_hook(_hook))
    return handles


# Per-block residual rescale threshold for STABILIZED training. The hand-built
# weights keep the residual ~O(10) through the stack via EXACT term
# cancellation; one GD step breaks the cancellation and the SwiGLU scale (S=100)
# amplifies the residual MULTIPLICATIVELY every block (13 -> 1e11 by block 16 ->
# 1e107 by block 24 -> inf), so neither fp32 NOR fp64 can take a single step.
# To MEASURE effectiveness at all, stabilized mode rescales each block's output
# residual back to a bounded per-row norm when it exceeds the threshold (a soft,
# direction-preserving cap, like a hard RMSNorm ceiling). This DOES change the
# forward, so it is an explicit opt-in ("stabilized" training) and we verify it
# preserves the BASELINE full_trace verdicts (the rescale only fires when the
# residual blows past the threshold, which at the hand-built weights it never
# does — so step-0 behaviour is identical; it only engages once GD perturbs the
# weights and the blowup starts).
#
# The threshold must sit ABOVE the clean operating point's residual norm (the
# emission blocks legitimately reach ~5e25) but BELOW the perturbation blowup
# (1e107+). 1e28 leaves the hand-built forward untouched (norms <= ~5e25 < 1e28)
# while capping any GD-induced blowup at 1e28 so the stack stays finite.
_RESIDUAL_RESCALE_MAXNORM = 1.0e28


def install_residual_normalizers(
    model, max_rownorm: float = _RESIDUAL_RESCALE_MAXNORM
):
    """Hook each block to rescale its output residual to a bounded row-norm.

    For each block output ``x`` of shape ``[B, S, D]``, compute the per-position
    L2 norm; where it exceeds ``max_rownorm`` divide that row by
    ``rownorm / max_rownorm`` (so its norm becomes exactly ``max_rownorm``,
    direction preserved). Rows under the threshold are untouched. This caps the
    multiplicative blowup so GD can take steps, while leaving the hand-built
    operating point (norms << threshold) byte-identical. Returns hook handles.
    """
    handles = []

    def _hook(_module, _inp, out):
        if not isinstance(out, torch.Tensor) or out.dim() < 1:
            return out
        out = torch.nan_to_num(out, nan=0.0, posinf=max_rownorm, neginf=-max_rownorm)
        norm = out.norm(dim=-1, keepdim=True)  # [..., 1]
        scale = torch.clamp(norm / max_rownorm, min=1.0)
        return out / scale

    for block in model.blocks:
        handles.append(block.register_forward_hook(_hook))
    return handles


def build_model(device: Optional[str] = None):
    """Build a FRESH (no disk cache) copy of the hand-built model on device."""
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    t0 = time.monotonic()
    model, layout = compile_full_vm_dynamic(disk_cache=False, strict=False)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    print(
        f"[harness] built fresh model in {time.monotonic() - t0:.1f}s "
        f"d_model={getattr(model, 'd_model', '?')} blocks={len(model.blocks)} "
        f"params={sum(p.numel() for p in model.parameters())} device={dev}",
        file=sys.stderr,
        flush=True,
    )
    return model, layout
