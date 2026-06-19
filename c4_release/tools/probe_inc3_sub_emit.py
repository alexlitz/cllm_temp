#!/usr/bin/env python3
"""Inc-3 CLAW-BACK — per-position EMITTED-token dump of the SUB result step.

Feeds the production autoregressive tape (`_final_context`) and re-forwards it,
printing the LM-head argmax (the emitted token) at every position of the result
step, in BOTH configs, so we pin EXACTLY which emitted position diverges
(golden delivers the SUB high byte; campaign drops it to 0x00). Then for the
diverging position it block-traces OUTPUT_LO / OUTPUT_HI_THIS_STEP / ALU bands.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_sub_emit.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_sub_emit.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { return 827 - 26; }")
RESULT_STEP = int(os.environ.get("PROBE_STEP", "3"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def topslots(row, base, n=16, k=3):
    vals = [(i, float(row[base + i])) for i in range(n)]
    vals = [(i, v) for i, v in vals if abs(v) > 0.3]
    vals.sort(key=lambda x: -abs(x[1]))
    return " ".join(f"[{i}]={v:.1f}" for i, v in vals[:k]) or "(empty)"


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30)" if nostk else "GOLDEN(35)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=RESULT_STEP + 2)
    padded = torch.tensor([ctx], device=p._device)
    # full-forward logits to read the emitted token at each position
    logits = td(p._forward_logits(ctx))
    base = pl + RESULT_STEP * STEP
    print(f"=== {cfg} STEP={STEP} step={RESULT_STEP} pl={pl} base={base} src={SRC!r} ===")
    print("  pos  in  ctx_tok  emit_argmax")
    for j in range(STEP):
        pos = base + j
        if pos >= len(ctx):
            break
        emit = int(logits[pos].argmax().item()) if pos < logits.shape[0] else -1
        ctok = ctx[pos] if pos < len(ctx) else -1
        # the token that lands at pos+1 is predicted by row pos
        nxt = ctx[pos + 1] if pos + 1 < len(ctx) else -1
        print(f"  {pos:4d} {j:3d}  {ctok:5d}    emit@row->{emit:4d}  (next_ctx={nxt})")

    # block-trace OUTPUT bands at the AX byte rows (the 4 value rows after marker)
    o_lo, o_hi = dp["OUTPUT_LO"], dp["OUTPUT_HI_THIS_STEP"]
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]
    blk_map = p.block_layer_map()
    nblk = len(p.model.blocks)
    # find AX marker offset within the step
    axmark = None
    for j in range(STEP):
        if ctx[base + j] == int(Token.REG_AX):
            axmark = j
            break
    print(f"  [AX marker at in-step offset {axmark}]")
    if axmark is not None:
        for vb in (0, 1):  # byte 0 and byte 1 value rows
            # value byte vb sits at position base+axmark+1+vb; predicted by the
            # row one earlier
            predrow = base + axmark + vb  # row whose argmax -> value byte vb
            print(f"--- byte{vb} predictor row in={axmark+vb} abs={predrow} ---")
            for blk in (11, 14, 15, 16, 29, 30, 33, 34, nblk - 1):
                x = td(p.model.forward(padded, stop_after_block=blk)[0])
                row = x[predrow]
                lg = blk_map[blk]
                lg = lg.get("logical") if isinstance(lg, dict) else lg
                print(f"  blk{blk:2d}(L{lg}): O_LO {topslots(row,o_lo)} | "
                      f"O_HI {topslots(row,o_hi)} | ALU_LO {topslots(row,alu_lo)} | "
                      f"ALU_HI {topslots(row,alu_hi)}")


if __name__ == "__main__":
    main()
