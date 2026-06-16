#!/usr/bin/env python3
"""GD stability / effectiveness training driver (one run = one config).

Builds a FRESH copy of the hand-built C4 neural VM, baselines the sample's
full_trace pass/fail split, then trains the copy under AdamW for N steps at a
fixed LR, periodically re-measuring:

  * CE loss + teacher-forced token accuracy (the optimization surrogate),
  * PASSING-subset full_trace pass-count (STABILITY: do correct programs
    survive GD?),
  * FAILING-subset full_trace pass-count (EFFECTIVENESS: does GD learn gaps?).

Everything is written to a JSON trajectory. NO production weights/ops are
touched — the trained module is a throw-away in-memory copy.

Example
-------
    CUDA_VISIBLE_DEVICES=0 python tools/gd_experiment_run.py \
        --ids 0-9,250-259,1031-1045 \
        --lr 1e-5 --steps 300 --eval-every 25 --batch 6 \
        --trainable all --supervision pc_ax \
        --output /tmp/gd_lr1e-5.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Training mutates dense params; the CSR shim would read stale sparse copies.
os.environ.setdefault("C4_CSR_INFERENCE", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402

import tools.gd_experiment_harness as H  # noqa: E402


def _parse_ids(spec: str) -> List[int]:
    out: List[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(piece))
    return out


def _select_trainable_params(model, mode: str, last_k: int):
    """Return the list of params to optimize for a given ablation mode.

    * ``all``     — every parameter (full fine-tune).
    * ``head_lastk`` — only the LM head + the last ``last_k`` transformer
      blocks' parameters (a cheaper, lower-capacity ablation: can GD fix the
      OUTPUT side without disturbing the whole stack?).
    """
    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        return [p for p in model.parameters() if p.requires_grad]

    # Freeze all, then unfreeze head + last K blocks.
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    head = getattr(model, "head", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    nblocks = len(model.blocks)
    for bi in range(max(0, nblocks - last_k), nblocks):
        for p in model.blocks[bi].parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def _pass_count(verdicts: Dict[int, bool], ids: List[int]) -> int:
    s = set(ids)
    return sum(1 for k, v in verdicts.items() if k in s and v)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GD stability/effectiveness run.")
    ap.add_argument("--ids", type=str, required=True,
                    help="Comma/range ids, e.g. '0-9,250-259,1031-1045'.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4,
                    help="Programs per GD minibatch (kept small for a shared GPU).")
    ap.add_argument("--loss-chunk", type=int, default=4,
                    help="Programs per forward when reporting the FULL-sample "
                         "CE/acc (no-grad). Keeps peak memory bounded; the "
                         "reported mean CE is chunk-invariant.")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optimizer", type=str, default="adamw",
                    choices=["adamw", "sgd"],
                    help="adamw (RMS-normalized step ~lr) or sgd (step ~ lr*grad)")
    ap.add_argument("--anchor-lambda", type=float, default=0.0,
                    help="L2 trust-region anchor to the INITIAL compiled weights: "
                         "loss += lambda * ||W - W0||^2. Makes the exact-VM point "
                         "an attractor instead of a non-attracting saddle. 0 = off.")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="Max global grad-norm (the hand-built logits emit "
                         "huge grads; clipping is essential).")
    ap.add_argument("--temperature", type=float, default=H._DEFAULT_LOGIT_TEMPERATURE)
    ap.add_argument("--trainable", type=str, default="all",
                    choices=["all", "head_lastk"])
    ap.add_argument("--last-k", type=int, default=4,
                    help="Blocks to unfreeze in head_lastk mode.")
    ap.add_argument("--stabilize", action="store_true", default=True,
                    help="Install residual-norm caps + gradient nan-scrub so "
                         "GD can take steps at all (the hand-built point's "
                         "exact-cancellation forward NaNs on the first raw "
                         "step). On by default; verified to preserve baseline "
                         "verdicts.")
    ap.add_argument("--no-stabilize", dest="stabilize", action="store_false",
                    help="Disable stabilization (training will NaN on step 1 — "
                         "used to DEMONSTRATE the raw fragility).")
    ap.add_argument("--supervision", type=str, default="pc_ax",
                    choices=list(H.SUPERVISION_SETS.keys()))
    ap.add_argument("--max-oracle-steps", type=int, default=60)
    ap.add_argument("--eval-spec-k", type=int, default=1)
    ap.add_argument("--eval-chunk", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    label = args.label or f"lr{args.lr}_steps{args.steps}_{args.trainable}_{args.supervision}"
    t0 = time.monotonic()

    model, _layout = H.build_model()
    runner = H.make_runner_for_model(model)
    if args.stabilize:
        H.install_residual_normalizers(model)
        print(f"[run:{label}] residual normalizers installed (stabilized mode)",
              file=sys.stderr, flush=True)

    ids = _parse_ids(args.ids)
    progs = H.prepare_programs(
        runner, ids,
        max_oracle_steps=args.max_oracle_steps,
        supervision=args.supervision,
    )
    dev = next(model.parameters()).device
    print(f"[run:{label}] prepared {len(progs)}/{len(ids)} programs", file=sys.stderr, flush=True)

    # Baseline (step 0) full_trace split.
    base_verdicts = H.eval_full_trace(
        runner, progs, spec_k=args.eval_spec_k, chunk=args.eval_chunk
    )
    split = H.split_by_baseline(progs, base_verdicts)
    passing_ids = list(split.passing_ids)
    failing_ids = list(split.failing_ids)
    n_pass0 = len(passing_ids)
    n_fail0 = len(failing_ids)
    print(
        f"[run:{label}] BASELINE full_trace: pass={n_pass0} fail={n_fail0} "
        f"(passing_ids={passing_ids[:20]}{'...' if len(passing_ids) > 20 else ''})",
        file=sys.stderr, flush=True,
    )

    # Cluster breakdown of the baseline split (for the report).
    from collections import Counter
    base_pass_clusters = Counter(p.cluster for p in split.passing())
    base_fail_clusters = Counter(p.cluster for p in split.failing())

    trainable = _select_trainable_params(model, args.trainable, args.last_k)
    n_train = sum(p.numel() for p in trainable)
    print(f"[run:{label}] trainable params: {n_train} ({args.trainable})", file=sys.stderr, flush=True)
    if args.optimizer == "sgd":
        # Plain SGD: the per-step weight delta scales with the ACTUAL gradient
        # magnitude (no RMS normalization). Isolates whether the first-step
        # cliff is AdamW's gradient-normalization (step ~ lr regardless of |grad|)
        # vs. the architecture's S=100 amplification of any perturbation.
        opt = torch.optim.SGD(trainable, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # Trust-region anchor: snapshot the compiled weights so the loss can pull
    # GD back toward the exact-VM point. Only clone when active (doubles the
    # trainable-param memory footprint).
    anchor_ref = None
    if args.anchor_lambda and args.anchor_lambda > 0:
        anchor_ref = [p.detach().clone() for p in trainable]
        print(f"[run:{label}] anchor L2 to compiled weights, lambda={args.anchor_lambda}",
              file=sys.stderr, flush=True)

    traj: List[dict] = []

    def record(step: int, ce: float, acc: float, n_sup: int):
        model.eval()
        verdicts = H.eval_full_trace(
            runner, progs, spec_k=args.eval_spec_k, chunk=args.eval_chunk
        )
        model.train()
        pass_total = sum(1 for v in verdicts.values() if v)
        pass_passing = _pass_count(verdicts, passing_ids)
        pass_failing = _pass_count(verdicts, failing_ids)
        row = {
            "step": step,
            "ce": ce,
            "token_acc": acc,
            "n_sup": n_sup,
            "full_trace_pass_total": pass_total,
            "passing_subset_pass": pass_passing,
            "passing_subset_n": n_pass0,
            "failing_subset_pass": pass_failing,
            "failing_subset_n": n_fail0,
            "wall_s": round(time.monotonic() - t0, 1),
        }
        traj.append(row)
        print(
            f"[run:{label}] step={step:4d} CE={ce:.4f} tok_acc={acc:.4f} | "
            f"full_trace total={pass_total}/{len(progs)} "
            f"PASSING-survive={pass_passing}/{n_pass0} "
            f"FAILING-learned={pass_failing}/{n_fail0} "
            f"wall={row['wall_s']}s",
            file=sys.stderr, flush=True,
        )
        return verdicts

    # Step 0 record (== baseline, sanity). Reporting-only forward, no grad,
    # chunked so the full-sample [B, L, V] logits never materialise at once.
    model.eval()
    with torch.no_grad():
        _, st0 = H.teacher_forced_loss(
            model, progs, dev, temperature=args.temperature, chunk=args.loss_chunk
        )
    record(0, st0["ce"], st0["token_acc"], st0["n_sup"])

    # Training loop.
    order = list(range(len(progs)))
    bi = 0
    for step in range(1, args.steps + 1):
        # Cycle through shuffled minibatches.
        if bi == 0:
            random.shuffle(order)
        batch_idx = order[bi : bi + args.batch]
        bi += args.batch
        if bi >= len(order):
            bi = 0
        batch = [progs[i] for i in batch_idx]

        model.train()
        opt.zero_grad(set_to_none=True)
        loss, st = H.teacher_forced_loss(model, batch, dev, temperature=args.temperature)
        if anchor_ref is not None:
            anchor_pen = sum(((p - a) ** 2).sum() for p, a in zip(trainable, anchor_ref))
            loss = loss + args.anchor_lambda * anchor_pen
        loss.backward()
        if args.stabilize:
            # Scrub nan/inf grads -> 0 (the backward through the 50-block
            # SwiGLU-scale-100 chain explodes once GD perturbs the cancellation
            # structure; an un-scrubbed nan grad would propagate to every weight
            # and clipping cannot rescue a nan norm).
            for p in trainable:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            # Re-measure the FULL sample's CE/acc (not just the batch) for the
            # trajectory point, then full_trace eval.
            model.eval()
            with torch.no_grad():
                _, st_full = H.teacher_forced_loss(
                    model, progs, dev, temperature=args.temperature,
                    chunk=args.loss_chunk,
                )
            record(step, st_full["ce"], st_full["token_acc"], st_full["n_sup"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {
        "label": label,
        "config": {
            "ids": args.ids,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "anchor_lambda": args.anchor_lambda,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "batch": args.batch,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "temperature": args.temperature,
            "trainable": args.trainable,
            "last_k": args.last_k,
            "supervision": args.supervision,
            "max_oracle_steps": args.max_oracle_steps,
            "eval_spec_k": args.eval_spec_k,
            "seed": args.seed,
        },
        "baseline": {
            "n_prepared": len(progs),
            "n_pass": n_pass0,
            "n_fail": n_fail0,
            "passing_ids": passing_ids,
            "failing_ids": failing_ids,
            "pass_clusters": dict(base_pass_clusters),
            "fail_clusters": dict(base_fail_clusters),
        },
        "trajectory": traj,
        "total_wall_s": round(time.monotonic() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[run:{label}] wrote {args.output} (wall={out['total_wall_s']}s)", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
