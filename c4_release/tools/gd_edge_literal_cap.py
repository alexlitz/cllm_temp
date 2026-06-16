#!/usr/bin/env python3
"""Experiment 2: can GD learn the AX byte-1 >= 16 emission cap?

The edge_literal cluster is a clean controlled probe of the "byte-1 16-cell
emission cap". The hand-built model PASSES edge_literal programs whose oracle AX
byte-1 is < 16 and FAILS those whose byte-1 is >= 16 (the emitter only has 16
one-hot cells for byte-1). This script trains a FRESH copy *only* on the failing
edge_literal programs (byte-1 >= 16), with AX-byte supervision, and tracks both:

  * full_trace pass-count on the failing set (does GD close the gap?), and
  * the actual decoded AX byte-1 the trained model emits per program (does GD
    push the emitter past 15, or does it plateau at <=15 — the architectural
    cap?).

It also tracks the PASSING (byte-1 < 16) edge_literal programs as a stability
control. No production weights are touched (in-memory copy only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("C4_CSR_INFERENCE", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402

import tools.gd_experiment_harness as H  # noqa: E402


def _oracle_ax_byte1(bytecode, data) -> int:
    from neural_vm.speculative import DraftVM

    vm = DraftVM(list(bytecode))
    for k, b in enumerate(data or b""):
        vm.memory[0x10000 + k] = int(b)
    for _ in range(400):
        if vm.halted:
            break
        if not vm.step():
            break
        if vm.halted:
            break
    return (int(vm.ax) >> 8) & 0xFF


@torch.no_grad()
def _decoded_ax_byte1(runner, progs) -> Dict[int, int]:
    """Decode the model's emitted final-AX byte-1 per program via full_trace run.

    We use the runner's per-step decode: run each program one at a time with the
    fail-fast machinery disabled-on-mismatch by reading the LAST decoded AX. We
    approximate via run_batch_fail_fast which returns decoded (pc, ax) trace; we
    read the final decoded ax byte-1. If the run errors we record -1.
    """
    out: Dict[int, int] = {}
    model = runner.model
    was = model.training
    model.eval()
    try:
        for p in progs:
            try:
                res = runner.run_batch_fail_fast(
                    [p.bytecode],
                    data_list=[p.data],
                    expected_steps_list=[p.decl_steps],
                    max_steps=None,
                    max_context_window=512,
                    spec_k=1,
                    criterion="full_trace",
                )[0]
                # On a PASS the model reached the oracle's final AX; on a FAIL,
                # got_ax is the model's decoded AX at the divergence step (the
                # byte the emitter actually produced). decoded_exit is the
                # neural exit code at the stop point (== final AX on a pass).
                ax = res.get("got_ax", None)
                if ax is None or ax < 0:
                    ax = res.get("decoded_exit", -1)
                out[p.idx] = ((int(ax) >> 8) & 0xFF) if ax is not None and ax >= 0 else -1
            except Exception as exc:  # noqa: BLE001
                out[p.idx] = -1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if was:
            model.train()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=H._DEFAULT_LOGIT_TEMPERATURE)
    ap.add_argument("--supervision", type=str, default="ax_only")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    t0 = time.monotonic()
    model, _ = H.build_model()
    runner = H.make_runner_for_model(model)
    H.install_residual_normalizers(model)

    edge_ids = list(range(1031, 1046))
    progs = H.prepare_programs(
        runner, edge_ids, max_oracle_steps=60, supervision=args.supervision
    )
    dev = next(model.parameters()).device

    # Classify by oracle byte-1 (>=16 => architecturally capped/failing target).
    b1 = {p.idx: _oracle_ax_byte1(p.bytecode, p.data) for p in progs}
    capped_ids = [p.idx for p in progs if b1[p.idx] >= 16]
    uncapped_ids = [p.idx for p in progs if b1[p.idx] < 16]
    train_progs = [p for p in progs if p.idx in set(capped_ids)]

    print(
        f"[edge] capped(byte1>=16) train ids={capped_ids} "
        f"uncapped(byte1<16) ids={uncapped_ids}",
        file=sys.stderr,
        flush=True,
    )

    base_verdicts = H.eval_full_trace(runner, progs, spec_k=1, chunk=12)
    base_decoded = _decoded_ax_byte1(runner, progs)

    trainable = [p for p in model.parameters()]
    for p in trainable:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    traj: List[dict] = []

    def record(step: int, ce: float, acc: float):
        verdicts = H.eval_full_trace(runner, progs, spec_k=1, chunk=12)
        decoded = _decoded_ax_byte1(runner, progs)
        capped_pass = sum(1 for i in capped_ids if verdicts.get(i))
        uncapped_pass = sum(1 for i in uncapped_ids if verdicts.get(i))
        # Decoded byte-1 on the capped (training) targets: did GD push >15?
        capped_decoded_b1 = {i: decoded.get(i, -1) for i in capped_ids}
        max_emit = max([v for v in capped_decoded_b1.values() if v >= 0] + [-1])
        row = {
            "step": step,
            "ce": ce,
            "token_acc": acc,
            "capped_pass": capped_pass,
            "capped_n": len(capped_ids),
            "uncapped_pass": uncapped_pass,
            "uncapped_n": len(uncapped_ids),
            "capped_decoded_b1": capped_decoded_b1,
            "capped_oracle_b1": {i: b1[i] for i in capped_ids},
            "max_emitted_b1_on_capped": max_emit,
            "wall_s": round(time.monotonic() - t0, 1),
        }
        traj.append(row)
        print(
            f"[edge] step={step:4d} CE={ce:.4f} acc={acc:.4f} | "
            f"capped_pass={capped_pass}/{len(capped_ids)} "
            f"uncapped_pass={uncapped_pass}/{len(uncapped_ids)} "
            f"max_emit_b1={max_emit} (cap=15) wall={row['wall_s']}s",
            file=sys.stderr,
            flush=True,
        )

    model.train()
    _, st0 = H.teacher_forced_loss(model, train_progs, dev, temperature=args.temperature)
    record(0, st0["ce"], st0["token_acc"])

    import random

    order = list(range(len(train_progs)))
    bi = 0
    for step in range(1, args.steps + 1):
        if bi == 0:
            random.shuffle(order)
        idx = order[bi : bi + args.batch]
        bi += args.batch
        if bi >= len(order):
            bi = 0
        batch = [train_progs[i] for i in idx]
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = H.teacher_forced_loss(model, batch, dev, temperature=args.temperature)
        loss.backward()
        for p in trainable:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                _, stf = H.teacher_forced_loss(
                    model, train_progs, dev, temperature=args.temperature
                )
            record(step, stf["ce"], stf["token_acc"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = {
        "label": "edge_literal_cap",
        "config": vars(args),
        "capped_ids": capped_ids,
        "uncapped_ids": uncapped_ids,
        "oracle_b1": b1,
        "baseline_verdicts": {str(k): v for k, v in base_verdicts.items()},
        "baseline_decoded_b1": {str(k): v for k, v in base_decoded.items()},
        "trajectory": traj,
        "total_wall_s": round(time.monotonic() - t0, 1),
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[edge] wrote {args.output}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
