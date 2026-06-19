#!/usr/bin/env python3
"""Inc-3 ROOT A: decompose the carry head's per-row Q.K score + ALiBi bias to
see WHY the 30-tok stride flips the winner from step1 AX_marker (golden) to
step4 AX[0] (campaign).

=== ROOT A K-RE-KEY IS ZERO-SUM (GPU-grounded, 2026-06-19) — DO NOT RETRY ===
The carry head ``layer13_ax_byte1_dump_carry.head_7`` (blk18, head7) selects the
prior-step row whose cross-step ``H1.*.-1`` it V-copies into ``H1_PREV_STEP``.
GOLDEN hard-attends row=step1 AX_marker (raw Q.K +1384, unique peak); CAMPAIGN
hard-attends row=step4 AX[0] (the recency-favoured row, raw +970) -> wrong
prev-step H1 -> byte-1 dump dark -> got_ax = expected & 0xFF (var_simple x>=256).

THE DECISIVE SLOT: head-dim slot 2 (the spec's ``-k_axc_w * AX_CARRY``, which the
``position_source='mixed'`` resolver lands on the REGISTRY AX_CARRY position =
the BUILT-layout ``ALU_LO/HI.*.-1`` cross-step band, dims 328-359). golden's
establishing-step AX-marker has a uniquely DEEP cross-step band (~ -3295 ->
slot2 K ~ +659); CAMPAIGN markers SATURATE UNIFORM (~ -1336 -> slot2 K ~ +267),
so step1's marker no longer stands out and the recency-biased step4 AX[0] wins.
The per-step DIFFERENTIATION the head relies on is DESTROYED by the 5-token-
shorter stride. There is NO stride-stable residual signal that re-identifies
golden's correct target row (scanned MARK_AX/OP_*/AX_FULL/ADDR/H1 -- all
uniform or 0 across markers), and the byte VALUE 0x03 is genuinely ABSENT from
the step5 predictor residual (scanned every 16-wide band for argmax==3, none).

A flag-gated slot-2 ``k_axc_w`` scale (``C4_INC3_ROOTA_KAXC``) was BUILT (byte-
identical OFF 4958b35b) + GPU-gated. It MOVES the row selection (2.5x and 8x ->
step5 AX_marker; negative ALiBi slope doesn't reach step1) but the emitted byte
is UNCHANGED on the authoritative gate: GPU run_1096_canonical full_trace
ids 250-274 = var_simple 5/25 WITH the re-key == 5/25 baseline (all 20 multi-byte
x>=256 still got_ax = expected & 0xFF); controls 800-849 = 5/50 == 5/50 baseline
(no regression). REVERTED -- the re-key is a verdict no-op. cpu_full_trace
DISAGREES with GPU here (it FAILs id 262 step7 ax=65536 where GPU PASSes) so it
is NOT a trustworthy pre-check for var_simple; use the GPU gate only.

THE REAL LEVER (next session): the byte-1=0x03 is not recoverable at the carry
head -- it must be re-DELIVERED into the step5 LI-reload AX[1] row from a stride-
stable source (mem[SP-local] byte-1, i.e. the Inc-1/Inc-2 mem-CAM extended to the
LI->AX path -- the original Inc-3 'mem-load high-byte delivery' note), OR the
cross-step ALU band's per-step depth must be restored upstream. NOT a single-head
K re-key.

Run TWICE (clear ~/.cache/c4_release/compiled_vm/ between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_rootA_carryscore.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_rootA_carryscore.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
import math
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token, _step_offset_field  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 990; return x; }"
BLK = 18
HEAD = 7


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    off5 = pl + 5 * STEP + 6
    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], device=p._device)
    x = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])
    attn = p.model.blocks[BLK].attn
    nh = attn.num_heads
    HD = attn.W_q.shape[0] // nh
    Wq = td(attn.W_q)[HEAD * HD:(HEAD + 1) * HD]
    Wk = td(attn.W_k)[HEAD * HD:(HEAD + 1) * HD]
    d_model = Wq.shape[1]
    xr = x[:, :d_model]
    q = xr[off5] @ Wq.T
    K = xr @ Wk.T
    scores = K @ q / math.sqrt(HD)
    slope = float(td(attn.alibi_slopes)[HEAD]) if getattr(attn, "alibi_slopes", None) is not None else 0.0
    T = xr.shape[0]
    pos = torch.arange(T, dtype=torch.float32)
    bias = -slope * (off5 - pos).abs()
    print(f"=== {cfg} STEP={STEP} off5={off5} slope={slope} ===")
    print("  per AX-marker/AX[0] row: raw Q.K score, dist, alibi bias, final")
    for st in range(0, 6):
        base = pl + st * STEP
        for o in (5, 6, 7):
            r = base + o
            if r > off5:
                break
            dist = off5 - r
            fin = float(scores[r] + bias[r])
            print(f"    s{st}.{o:02d} row={r:4d} {_step_offset_field(o):11s} "
                  f"raw={float(scores[r]):+8.1f} dist={dist:4d} "
                  f"alibi={float(bias[r]):+7.1f} final={fin:+8.1f}")


if __name__ == "__main__":
    main()
