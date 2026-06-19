#!/usr/bin/env python3
"""Inc-3 if_gt: decompose the byte-token logit at the GT-step STEP_END row (the
row that predicts step4[0]) into per-residual-dim contributions
(head.weight[byte, d] * resid[d]). Identifies WHICH residual dim drives the
spurious byte2 win (so we patch the RIGHT op, not OUTPUT_LO if that's not it).

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_ifgt_headcol.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")


def main():
    STEP = int(Token.STEP_TOKENS)
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    # Use the ACTUAL probe model's dim_positions (a separate compile_full_vm_dynamic
    # build can have a DIFFERENT layout -> mis-indexed dim reads).
    dp = dict(p.model.dim_positions)
    inv = {}
    for nm, idx in dp.items():
        inv.setdefault(int(idx), nm)
    prompt_len = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=8)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    nblk = len(p.model.blocks)
    with torch.no_grad():
        resid = p.model.forward(padded, stop_after_block=nblk - 1)[0]  # [T, d]
    head_w = p.model.head.weight  # [vocab, d]
    # Default: GT-step STEP_END row predicts step4[0]. Override with PROBE_ROW.
    row = int(os.environ["PROBE_ROW"]) if os.environ.get("PROBE_ROW") else \
        prompt_len + 4 * STEP - 1
    r = resid[row]
    print(f"=== STEP={STEP} src={SRC!r} row={row} predicts pos {row+1} ===")
    emitted = int(ctx[row + 1]) if row + 1 < len(ctx) else -1
    print(f"  emitted token at pos {row+1} = {emitted}")
    extra = os.environ.get("PROBE_BYTES", "")
    want = {emitted, 0, 2}
    for b in extra.split(","):
        if b.strip().isdigit():
            want.add(int(b))
    for byteval in sorted(want):
        w = head_w[byteval]                      # [d]
        contrib = (w * r)                        # per-dim contribution
        finite = torch.nan_to_num(contrib, nan=0.0, posinf=1e30, neginf=-1e30)
        order = finite.abs().topk(min(10, finite.numel())).indices.tolist()
        print(f"\n  byte {byteval}: total_logit≈{finite.sum().item():.3e}")
        for d in order:
            print(f"     dim {d:4d} ({inv.get(d,'?'):18s}) "
                  f"resid={r[d].item():+.3e} hw={w[d].item():+.3f} "
                  f"contrib={finite[d].item():+.3e}")
    # also REG_PC marker contributions for comparison
    w = head_w[257]
    contrib = torch.nan_to_num(w * r, nan=0.0, posinf=1e30, neginf=-1e30)
    print(f"\n  REG_PC(257): total_logit≈{contrib.sum().item():.3e}")
    for d in contrib.abs().topk(6).indices.tolist():
        print(f"     dim {d:4d} ({inv.get(d,'?'):18s}) "
              f"resid={r[d].item():+.3e} hw={w[d].item():+.3f} contrib={contrib[d].item():+.3e}")


if __name__ == "__main__":
    main()
