#!/usr/bin/env python3
"""Long-horizon per-step (PC, AX) stability probe (teacher-forced free-run).

STRATEGIC QUESTION
------------------
Can the pure-neural VM stay byte-stable over hundreds-to-thousands of VM
steps, or is there an architectural position ceiling (ALiBi saturation /
KV eviction / attention-entropy collapse) that makes the deep clusters
(loop_*, rec_*, gcd_*) unreachable regardless of the prologue fix?

The normal fail-fast runner cannot answer this: every deep program diverges
in the PROLOGUE (step 0-1, the AX/0xFF/framing wall), so it stops at step 1
and tells us nothing about long-horizon behaviour.

METHOD (teacher-force past the prologue, then free-run)
-------------------------------------------------------
For a program with a long oracle trace (loop_sum_* ~ 300-450 steps):

  1. Build the element context (bytecode prologue) exactly as production.
  2. Compute the full per-step 35-token DraftVM oracle (declarative
     byte-identity) for every step.
  3. TEACHER-FORCE steps 0..K-1: append the oracle's 35-token blocks
     directly to the context (so the model's context is GUARANTEED correct
     through step K-1, bypassing the prologue bug). Run the pure-neural
     dispatch after each forced STEP_END so the runner's register tracking
     stays consistent.
  4. FREE-RUN from step K: at each step boundary, drive the model one token
     per forward (spec_k=0 semantics), append the model's argmax, and after
     each completed step decode (PC, AX) from that step's own 35-token slice
     and compare to the oracle. Record the FIRST step where the free-run
     (PC, AX) diverges, together with the ABSOLUTE POSITION (len(context))
     at that point.

Because the context is correct through step K-1, any divergence after K is
PURE per-step / position behaviour, not prologue contamination. By choosing
K large (deep into a 400-step loop) we observe the model at high absolute
position (thousands of tokens) on a CORRECT context.

This is a MEASUREMENT HARNESS ONLY. It does not touch model weights. It uses
the same compiled model, the same _build_context, _forward_argmax_batch,
_step_one, and _oracle_pc_ax_steps primitives as the production fail-fast
path. spec_k is effectively 0 (one token per forward, model is sole authority
in the free-run window).

USAGE
-----
  CUDA_VISIBLE_DEVICES=1 python tools/probe_longhorizon_stability.py \
      --ids loop_sum_0,loop_sum_4 --tf-steps 1 --max-free-steps 460

  --tf-steps K        teacher-force the FIRST K steps (correct context), then
                      free-run. K=1 forces only the prologue's first step.
  --max-free-steps N  cap the free-run at N steps after K (default: run to
                      oracle end).
  --model-max-seq-len M   model window (default 4096).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

# Match the canonical runner's allocator hint (cut fragmentation OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.vm_step import Token  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402


STEP = Token.STEP_TOKENS


def _decode_step_register(step_tokens: List[int], marker: int) -> Optional[int]:
    """Decode a 32-bit register from one 35-token VM-step slice (own slice)."""
    for i, tk in enumerate(step_tokens):
        if tk == marker and i + 4 < len(step_tokens):
            val = 0
            for j in range(4):
                val |= (int(step_tokens[i + 1 + j]) & 0xFF) << (j * 8)
            return val
    return None


def _select_programs(ids_spec: Optional[str]) -> List[Tuple[int, str, int, str]]:
    tests = generate_test_programs()
    out: List[Tuple[int, str, int, str]] = []
    if not ids_spec:
        # Default: a few loop_sum programs of varying depth.
        wanted = {"loop_sum_0", "loop_sum_2", "loop_sum_4"}
    else:
        wanted = set(s.strip() for s in ids_spec.split(",") if s.strip())
    for i, (src, exp, desc) in enumerate(tests):
        name = desc.split(":")[0].strip()
        if name in wanted or desc.split(":")[0] in wanted:
            out.append((i, src, exp, desc))
    return out


def probe_one(
    runner: BatchedPureNeuralRunner,
    bytecode: List[int],
    data: bytes,
    *,
    tf_steps: int,
    max_free_steps: Optional[int],
    model_max_seq: int,
) -> dict:
    """Teacher-force tf_steps, then free-run, return stability stats."""
    # --- oracle: full per-step (pc,ax) + 35-token reference -----------------
    s = runner._build_element(bytecode, data, [], "", spec_k=1)
    oracle_steps, oracle_tokens = runner._oracle_pc_ax_steps(
        bytecode, data, "", expected_steps=None, with_tokens=True
    )
    total_oracle_steps = len(oracle_steps)
    s.expected_steps = total_oracle_steps

    prefix_len = s.prefix_len
    tf_steps = max(0, min(tf_steps, total_oracle_steps))

    # --- teacher-force steps 0..tf_steps-1 ----------------------------------
    # Append each oracle step's 35 tokens verbatim. Run _step_one so the
    # STEP_END dispatch updates the runner's register tracking exactly as
    # production (the model is bypassed for these forced steps).
    for st_idx in range(tf_steps):
        for tok in oracle_tokens[st_idx]:
            runner._step_one(s, int(tok), 0)
            if s.halted:
                break
        if s.halted:
            break

    forced_ok = (s.token_pos // STEP) == tf_steps and not s.halted

    # --- free-run from step tf_steps ----------------------------------------
    first_div_step: Optional[int] = None
    first_div_pos: Optional[int] = None
    first_div_detail: Optional[dict] = None
    last_ok_step = tf_steps - 1
    last_ok_pos = len(s.context)
    free_steps_run = 0

    step_cap = total_oracle_steps
    if max_free_steps is not None:
        step_cap = min(step_cap, tf_steps + max_free_steps)

    cur_step = tf_steps
    while cur_step < step_cap and not s.halted:
        # Emit one full 35-token step, one token per forward.
        step_start_pos = len(s.context)
        for _ in range(STEP):
            ctx = list(s.context)
            # Window to the model's max_seq_len from the TAIL (production
            # serial runner does ctx[-max_seq_len:]); the batched fresh
            # forward path pads/truncates similarly. We mirror the serial
            # truncation so absolute distance == relative position in-window.
            if len(ctx) > model_max_seq:
                ctx = ctx[-model_max_seq:]
            first_logit_pos = len(ctx) - 1
            preds, _start, _real = runner._forward_argmax_batch(
                [ctx],
                [0],
                first_logit_pos=first_logit_pos,
                allow_kv=False,
                protected_prefix_lens=[min(prefix_len, len(ctx))],
                gather_positions=[[len(ctx) - 1]],
            )
            next_tok = int(preds[0][0])
            runner._step_one(s, next_tok, 0)
            if s.halted:
                break
        free_steps_run += 1

        completed = s.token_pos // STEP
        if completed <= cur_step:
            # The step did not complete (model halted mid-step). The pure-neural
            # dispatch performs a neural-authoritative early EXIT: when the
            # model's emitted PC points at an EXIT instruction it stops WITHOUT
            # emitting the remaining oracle steps and sets exit_code from AX.
            # That is the CORRECT terminal behaviour, not a divergence, iff the
            # exit code equals the oracle's final AX.
            oracle_final_ax = int(oracle_steps[-1][1]) & 0xFFFFFFFF
            de = None if s.exit_code is None else int(s.exit_code) & 0xFFFFFFFF
            if s.halted and de is not None and de == oracle_final_ax:
                # Clean early exit with correct result -> still stable.
                last_ok_step = cur_step - 1
                break
            first_div_step = cur_step
            first_div_pos = len(s.context)
            first_div_detail = {
                "reason": "halted_mid_step",
                "completed": completed,
                "exit_code": de,
                "oracle_final_ax": oracle_final_ax,
            }
            break

        # Decode this step's (pc, ax) from its own slice.
        slice_start = prefix_len + cur_step * STEP
        step_tokens = s.context[slice_start : slice_start + STEP]
        g_pc = _decode_step_register(step_tokens, Token.REG_PC)
        g_ax = _decode_step_register(step_tokens, Token.REG_AX)
        e_pc, e_ax = oracle_steps[cur_step]

        gp = None if g_pc is None else g_pc & 0xFFFFFFFF
        ga = None if g_ax is None else g_ax & 0xFFFFFFFF
        if gp != (e_pc & 0xFFFFFFFF) or ga != (e_ax & 0xFFFFFFFF):
            first_div_step = cur_step
            first_div_pos = step_start_pos  # absolute pos where this step began
            first_div_detail = {
                "expected_pc": int(e_pc) & 0xFFFFFFFF,
                "expected_ax": int(e_ax) & 0xFFFFFFFF,
                "got_pc": gp,
                "got_ax": ga,
            }
            break

        last_ok_step = cur_step
        last_ok_pos = slice_start + STEP
        cur_step += 1

    stable = first_div_step is None
    return {
        "total_oracle_steps": total_oracle_steps,
        "tf_steps": tf_steps,
        "forced_ok": forced_ok,
        "free_steps_run": free_steps_run,
        "stable": stable,
        "first_div_step": first_div_step,
        "first_div_abs_pos": first_div_pos,
        "first_div_detail": first_div_detail,
        "last_ok_step": last_ok_step,
        "last_ok_abs_pos": last_ok_pos,
        "halted": s.halted,
        "exit_code": s.exit_code,
        "prefix_len": prefix_len,
    }


def single_step_sweep(
    runner: BatchedPureNeuralRunner,
    bytecode: List[int],
    data: bytes,
    *,
    sweep_steps: List[int],
    model_max_seq: int,
) -> List[dict]:
    """Purest position-isolation: for each target step P, teacher-force the
    CORRECT context through step P-1, then free-run EXACTLY ONE step and record
    whether that single step's (PC, AX) matched the oracle.

    This decouples "correct context at position P" from "free-run accumulation".
    If the single-step match rate is flat across P (low or high), the behaviour
    is PER-STEP-OP, not position-dependent. If it DEGRADES as P (and absolute
    token pos) grows, there is a position ceiling.
    """
    oracle_steps, oracle_tokens = runner._oracle_pc_ax_steps(
        bytecode, data, "", expected_steps=None, with_tokens=True
    )
    total = len(oracle_steps)
    results: List[dict] = []
    for P in sweep_steps:
        if P < 1 or P >= total:
            continue
        # Fresh element each time so forced context is pristine.
        s = runner._build_element(bytecode, data, [], "", spec_k=1)
        s.expected_steps = total
        prefix_len = s.prefix_len
        # Force steps 0..P-1 (correct context).
        for st_idx in range(P):
            for tok in oracle_tokens[st_idx]:
                runner._step_one(s, int(tok), 0)
                if s.halted:
                    break
            if s.halted:
                break
        if s.halted or (s.token_pos // STEP) != P:
            results.append({"target_step": P, "ok": None, "note": "force_failed"})
            continue
        abs_pos_before = len(s.context)
        # Free-run exactly one step.
        for _ in range(STEP):
            ctx = list(s.context)
            if len(ctx) > model_max_seq:
                ctx = ctx[-model_max_seq:]
            preds, _a, _b = runner._forward_argmax_batch(
                [ctx], [0],
                first_logit_pos=len(ctx) - 1,
                allow_kv=False,
                protected_prefix_lens=[min(prefix_len, len(ctx))],
                gather_positions=[[len(ctx) - 1]],
            )
            runner._step_one(s, int(preds[0][0]), 0)
            if s.halted:
                break
        completed = s.token_pos // STEP
        e_pc, e_ax = oracle_steps[P]
        if completed <= P:
            # Halted mid-step: ok iff clean early-exit with right code.
            de = None if s.exit_code is None else int(s.exit_code) & 0xFFFFFFFF
            okv = bool(s.halted and de == (int(oracle_steps[-1][1]) & 0xFFFFFFFF))
            results.append({
                "target_step": P, "ok": okv, "abs_pos": abs_pos_before,
                "win_pos": min(abs_pos_before, model_max_seq),
                "note": "early_exit" if okv else "halted_mid_step",
            })
            continue
        sl = prefix_len + P * STEP
        st_tok = s.context[sl:sl + STEP]
        g_pc = _decode_step_register(st_tok, Token.REG_PC)
        g_ax = _decode_step_register(st_tok, Token.REG_AX)
        gp = None if g_pc is None else g_pc & 0xFFFFFFFF
        ga = None if g_ax is None else g_ax & 0xFFFFFFFF
        pc_ok = gp == (int(e_pc) & 0xFFFFFFFF)
        ax_ok = ga == (int(e_ax) & 0xFFFFFFFF)
        results.append({
            "target_step": P,
            "ok": bool(pc_ok and ax_ok),
            "pc_ok": pc_ok, "ax_ok": ax_ok,
            "abs_pos": abs_pos_before,
            "win_pos": min(abs_pos_before, model_max_seq),
            "exp_pc": int(e_pc) & 0xFFFFFFFF, "got_pc": gp,
            "exp_ax": int(e_ax) & 0xFFFFFFFF, "got_ax": ga,
        })
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", type=str, default=None,
                    help="comma-separated program names, e.g. loop_sum_0,loop_sum_4")
    ap.add_argument("--tf-steps", type=int, default=1,
                    help="teacher-force the first K steps (correct context)")
    ap.add_argument("--max-free-steps", type=int, default=None,
                    help="cap free-run at N steps after K (default: oracle end)")
    ap.add_argument("--model-max-seq-len", type=int,
                    default=int(os.environ.get("C4_BATCH_MODEL_MAX_SEQ_LEN", "4096")))
    ap.add_argument("--single-step-sweep", type=str, default=None,
                    help="comma-separated target steps P; force correct context "
                         "through P-1 then free-run ONE step, per P. Purest "
                         "position-isolation. e.g. 1,5,20,50,100,200,280")
    args = ap.parse_args(argv)

    progs = _select_programs(args.ids)
    if not progs:
        print("no programs matched --ids", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    runner = BatchedPureNeuralRunner(max_seq_len=args.model_max_seq_len)
    print(f"[probe] model built in {time.monotonic()-t0:.1f}s device={runner._device} "
          f"model_max_seq={args.model_max_seq_len}", file=sys.stderr, flush=True)

    print(f"\n{'='*78}")
    print(f"LONG-HORIZON STABILITY PROBE  (teacher-force K then free-run, spec_k=0)")
    print(f"tf_steps={args.tf_steps}  model_max_seq={args.model_max_seq_len}")
    print(f"{'='*78}")
    if args.single_step_sweep:
        sweep = [int(x) for x in args.single_step_sweep.split(",") if x.strip()]
        for idx, src, exp, desc in progs:
            bc, data = compile_c(src)
            st0 = time.monotonic()
            rows = single_step_sweep(
                runner, bc, data, sweep_steps=sweep,
                model_max_seq=args.model_max_seq_len,
            )
            print(f"\n--- id={idx} {desc}  (single-step sweep, {time.monotonic()-st0:.1f}s) ---")
            print(f"  {'step':>5} {'abs_pos':>8} {'win_pos':>8}  {'ok':>5} {'pc_ok':>6} {'ax_ok':>6}"
                  f"  exp(pc,ax) -> got(pc,ax)")
            for r in rows:
                if r.get("ok") is None:
                    print(f"  {r['target_step']:>5} {'-':>8} {'-':>8}  {'SKIP':>5}  ({r.get('note')})")
                    continue
                print(f"  {r['target_step']:>5} {r['abs_pos']:>8} {r['win_pos']:>8}  "
                      f"{('OK' if r['ok'] else 'X'):>5} "
                      f"{str(r.get('pc_ok')):>6} {str(r.get('ax_ok')):>6}  "
                      f"({r.get('exp_pc')},{r.get('exp_ax')}) -> ({r.get('got_pc')},{r.get('got_ax')})"
                      f"{'  '+r['note'] if r.get('note') else ''}")
            sys.stdout.flush()
        return 0

    for idx, src, exp, desc in progs:
        bc, data = compile_c(src)
        st0 = time.monotonic()
        r = probe_one(
            runner, bc, data,
            tf_steps=args.tf_steps,
            max_free_steps=args.max_free_steps,
            model_max_seq=args.model_max_seq_len,
        )
        dt = time.monotonic() - st0
        verdict = "STABLE" if r["stable"] else "DIVERGED"
        print(f"\n--- id={idx} {desc}  ({dt:.1f}s) ---")
        print(f"  oracle_steps={r['total_oracle_steps']}  tf_steps={r['tf_steps']}  "
              f"forced_ok={r['forced_ok']}  free_steps_run={r['free_steps_run']}")
        print(f"  VERDICT: {verdict}")
        if r["stable"]:
            print(f"  stable through step {r['last_ok_step']} "
                  f"(abs token pos {r['last_ok_abs_pos']}); halted={r['halted']} "
                  f"exit={r['exit_code']} expected={exp}")
        else:
            print(f"  first divergence at FREE step {r['first_div_step']} "
                  f"(abs token pos {r['first_div_abs_pos']})")
            print(f"  last OK step {r['last_ok_step']} (abs pos {r['last_ok_abs_pos']})")
            print(f"  detail: {r['first_div_detail']}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
