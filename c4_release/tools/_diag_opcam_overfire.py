#!/usr/bin/env python3
"""Diagnose the C4_OPCAM_FRAME amplifier's over-firing (task #421).

TEACHER-FORCED (DraftVM oracle tape) single-forward-per-program diagnostic:
for a BROAD stratified sample, build the oracle token tape (prompt + per-step
DraftVM ``draft_tokens()``), run ONE forward, and at EVERY LEA AX-marker row
evaluate the OPCAM amplifier's firing gate against the pre-amplifier (block-40
output == block-41 tail-bank INPUT) residual:

  * pre-amp OUTPUT byte-0  (what the ALU/keystone rules delivered)
  * amplifier gate SCORE + slammed byte ``(h<<4)|k``
  * whether the amplifier CHANGES the byte (``amp != pre``)
  * FETCH imm nibbles + OP_ENT residue (the discriminators the KEYSTONE rules
    use but the amplifier does NOT)

Teacher-forcing HIDES cross-step framing verdicts, but it EXACTLY reveals
which LEA rows the amplifier fires+changes on — the over-fire question. ONE
forward per program (no AR loop) so it survives CPU contention.

Run:
  CUDA_VISIBLE_DEVICES="" C4_CAMPAIGN=1 C4_OPCAM_FRAME=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_opcam_on \
    python -u tools/_diag_opcam_overfire.py --ids <list>
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_CAMPAIGN", "1")
os.environ.setdefault("C4_OPCAM_FRAME", "1")
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402
from neural_vm.verification.faithful_autoregressive import (  # noqa: E402
    build_cpu_model, FaithfulAutoregressiveRunner,
)
from tests.test_suite_1000 import generate_test_programs  # noqa: E402

STEP = int(Token.STEP_TOKENS)


def nib_win(vec, base):
    seg = vec[base:base + 16]
    i = int(torch.argmax(seg))
    return i, float(seg[i])


def oracle_tape(bc_words, data, max_steps):
    vm = DraftVM(list(bc_words))
    vm.load_data(data or b"")
    toks = []
    for _ in range(max_steps):
        if not vm.step():
            break
        toks.extend(vm.draft_tokens())
        if vm.halted:
            break
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--max-steps", type=int, default=20)
    args = ap.parse_args()
    want_ids = [int(x) for x in args.ids.split(",") if x.strip()]

    model, layout = build_cpu_model(disk_cache=True)
    dp = dict(layout.dim_positions)
    n_blocks = len(model.blocks)
    P = generate_test_programs()

    d = {k: dp[k] for k in (
        "OP_LEA", "MARK_AX", "HAS_SE", "OUTPUT_LO", "OUTPUT_HI_THIS_STEP",
        "FETCH_LO", "FETCH_HI", "IS_BYTE", "MARK_PC", "MARK_SP", "MARK_BP",
        "MARK_STACK0", "MARK_MEM", "OP_IMM", "OP_DIV", "OP_MOD", "OP_ENT",
    )}
    pre_blk = 40
    tail_blk = n_blocks - 1

    runner = FaithfulAutoregressiveRunner(model=model, layout=layout)
    inner = runner._inner

    def amp_score_and_byte(r_pre):
        best = None
        for h in (14, 13):
            for k in (0, 8):
                s = 0.0
                s += 1.0 * float(r_pre[d["MARK_AX"]])
                s += 1.0 * float(r_pre[d["HAS_SE"]])
                s += 1.0 * float(r_pre[d["OP_LEA"]])
                s += 8.0 * float(r_pre[d["OUTPUT_HI_THIS_STEP"] + h])
                s += 8.0 * float(r_pre[d["OUTPUT_LO"] + k])
                s += -1000.0 * float(r_pre[d["OUTPUT_HI_THIS_STEP"] + 0])
                s += -1_000_000.0 * float(r_pre[d["OP_IMM"]])
                s += -1e9 * float(r_pre[d["OP_DIV"]])
                s += -1e9 * float(r_pre[d["OP_MOD"]])
                s += -10.0 * float(r_pre[d["IS_BYTE"]])
                for m in ("MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
                          "MARK_MEM"):
                    s += -10000.0 * float(r_pre[d[m]])
                fires = s >= 20.0
                cand = (fires, s, h, k)
                if best is None or (fires and not best[0]) or \
                        (fires == best[0] and s > best[1]):
                    best = cand
        return best

    for pid in want_ids:
        src, exp, desc = P[pid]
        bc_words, data = compile_c(src)
        prompt = inner._serial._build_context(list(bc_words), b"", [], "")
        pl = len(prompt)
        tape = oracle_tape(bc_words, data, args.max_steps)
        full_ctx = list(prompt) + tape
        padded = torch.tensor([full_ctx], dtype=torch.long)

        with torch.no_grad():
            pre = model.forward(padded, stop_after_block=pre_blk)[0]
            if pre.is_sparse:
                pre = pre.to_dense()
            tail = model.forward(padded, stop_after_block=tail_blk)[0]
            if tail.is_sparse:
                tail = tail.to_dense()

        nsteps = (len(full_ctx) - pl) // STEP
        rows = []
        for step in range(nsteps):
            lo = pl + step * STEP
            hi = lo + STEP
            if hi > len(full_ctx):
                break
            ax_off = int(torch.argmax(tail[lo:hi, d["MARK_AX"]]))
            ax_pos = lo + ax_off
            if float(pre[ax_pos, d["OP_LEA"]]) > 0.5:
                rows.append((step, ax_pos))

        cluster = desc.split()[0] if desc else "?"
        print(f"\n== id{pid} [{cluster}] exp={exp!r} LEA-steps="
              f"{[s for s, _ in rows]} src={src.strip()[:56]!r}", flush=True)
        for step, ax_pos in rows:
            r_pre = pre[ax_pos]
            r_tail = tail[ax_pos]
            pl_lo, _ = nib_win(r_pre, d["OUTPUT_LO"])
            ph_hi, _ = nib_win(r_pre, d["OUTPUT_HI_THIS_STEP"])
            pre_byte = (ph_hi << 4) | pl_lo
            tl_lo, _ = nib_win(r_tail, d["OUTPUT_LO"])
            th_hi, _ = nib_win(r_tail, d["OUTPUT_HI_THIS_STEP"])
            tail_byte = (th_hi << 4) | tl_lo
            fires, score, bh, bk = amp_score_and_byte(r_pre)
            amp_byte = (bh << 4) | bk
            fl, _ = nib_win(r_pre, d["FETCH_LO"])
            fh, _ = nib_win(r_pre, d["FETCH_HI"])
            fetch_byte = (fh << 4) | fl
            ent = float(r_pre[d["OP_ENT"]])
            tag = ""
            if fires:
                tag = "FIRES"
                if amp_byte != pre_byte:
                    tag += "&CHANGES(%02x->%02x)" % (pre_byte, amp_byte)
            print(f"   step{step:2d} pos{ax_pos}: pre=0x{pre_byte:02x} "
                  f"tail=0x{tail_byte:02x} amp=0x{amp_byte:02x} "
                  f"score={score:8.1f} FETCH=0x{fetch_byte:02x} "
                  f"ENT={ent:+5.2f} {tag}", flush=True)


if __name__ == "__main__":
    main()
