#!/usr/bin/env python3
"""Per-block logit-evolution probe at the STACK0-byte3 desync row.

For a func/rec program, replays the spec_k=0 emission, finds the JSR/ENT
desync step (first step whose emitted token count != 35), locates the row that
PREDICTS the token where the STACK0 4-byte value is truncated (MEM marker wins
over the 4th STACK0 value byte), and sweeps stop_after_block to show at which
physical block the MEM (261) logit overtakes the value-byte logit at that row.

Usage: python tools/_probe_desync_logit.py <id> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

SE = int(Token.STEP_END)
STACK0 = int(Token.STACK0)
MEM = int(Token.MEM)
STEP = int(Token.STEP_TOKENS)


def inv_tok():
    d = {}
    for n in dir(Token):
        if n.startswith("_"):
            continue
        try:
            d[int(getattr(Token, n))] = n
        except Exception:
            pass
    return d


@torch.no_grad()
def main():
    pid = int(sys.argv[1])
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    m = probe.model
    inv = inv_tok()
    prompt = probe._build_context(bc)
    pl = len(prompt)
    ctx = probe._final_context(bc, max_steps=ms)
    tail = ctx[pl:]
    se_abs = [pl + i for i, t in enumerate(tail) if t == SE]

    prev = pl - 1
    desync_step = None
    bounds = None
    for si, se in enumerate(se_abs):
        n = se - prev
        if n != STEP and desync_step is None:
            desync_step = si
            bounds = (prev + 1, se)
        prev = se
    if desync_step is None:
        print(f"id{pid}: no desync in first {ms} steps")
        return
    s_start, s_end = bounds
    seg = ctx[s_start:s_end + 1]
    print(f"id{pid} {desc} desync_step={desync_step} len={len(seg)} (expect {STEP})")

    try:
        st_off = seg.index(STACK0)
    except ValueError:
        print("no STACK0 in desync step")
        return
    mem_off = None
    for k in range(st_off + 1, len(seg)):
        if seg[k] == MEM:
            mem_off = k
            break
    if mem_off is None:
        print("no MEM after STACK0")
        return
    nvals = mem_off - st_off - 1
    pred_row = s_start + mem_off - 1
    print(f"STACK0 at seg_off={st_off} emitted {nvals} value bytes "
          f"(clean=4); MEM at seg_off={mem_off}; predicting row abs={pred_row}")
    print(f"  context[pred_row]={ctx[pred_row]}({inv.get(ctx[pred_row],'')}) "
          f"-> emitted next={ctx[pred_row+1]}({inv.get(ctx[pred_row+1],'')})")

    padded = torch.tensor([ctx[:pred_row + 1]], dtype=torch.long, device=probe._device)
    nblk = len(m.blocks)

    def logits_after(blk):
        resid = m.forward(padded, stop_after_block=blk)
        return m.head(resid[0, -1:])[0]  # keep 2-D for CSR head

    full = m.forward(padded)[0, -1]
    topk = torch.topk(full, 6)
    print("\nFULL logits top-6 at pred_row:")
    for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"   tok={i}({inv.get(i,'')}) logit={v:.2f}")

    print(f"\nPer-block: MEM(261) logit vs best VALUE-byte(0..255) logit at pred_row")
    print(f"  {'blk':>3} {'logical':>7} {'exp':>4}  {'MEM':>9}  {'bestVAL':>9} {'valTok':>6}  {'winner'}")
    bmap = probe.block_layer_map()
    for blk in range(nblk):
        lg = logits_after(blk)
        mem_l = float(lg[MEM])
        val_part = lg[:256]
        vbest = int(val_part.argmax())
        vbest_l = float(val_part[vbest])
        winner = "MEM" if mem_l > vbest_l else "VAL"
        bm = bmap[blk]
        print(f"  {blk:>3} {bm['logical']:>7} {str(bm['is_post_op_expansion'])[0]:>4}  "
              f"{mem_l:9.2f}  {vbest_l:9.2f} {vbest:6d}  {winner}")


if __name__ == "__main__":
    main()
