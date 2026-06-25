#!/usr/bin/env python3
"""Find the discriminating INPUT dim to physical block 16 at the GT-result SE row.

Block 16 is a logical-L14 post_op EXPANSION passthrough (zero-init attn =
residual identity). So block 16's only active component is its FFN. The doc
proved every cmp FFN GATE (SE_OP_GT, SE_CMP_GROUP, CMP cascade, SE_ALU/ALU) is
byte-identical between the FAILING ifVAR and the PASSING literal ifGT at this
row, yet block-16 OUTPUT_LO cell0/cell1 flips. => some OTHER residual dim going
INTO block 16 (set by an EARLIER block's attention over the loaded-var memory
rows, present only in the ifVAR KV) differs and drives the block-16 FFN.

This probe dumps the FULL post-block-15 residual at the result SE row for BOTH
programs and prints every dim whose value differs by > THRESH, labelled by the
nearest named dim. That pinpoints the discriminator the fix must neutralise.

  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_varupd2 python tools/probe_ifvar_blk16_indiff.py
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

PROGS = {
    "ifVAR_66gt24_T": ("int main() { int x; x = 66; if (x > 24) return 1; return 0; }", 10),
    "ifGT_66gt24_T":  ("int main() { if (66 > 24) return 1; return 0; }", 3),
}
THRESH = 0.5


def se_row_for(p, model, bc, res_step):
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    se_mark = dp.get("MARK_SE_ONLY")
    ctx = p._final_context(bc, max_steps=25)
    prefix_len = len(p._build_context(bc))
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb = model.embed(padded)[0]
    step_start = prefix_len + res_step * STEP
    se_row = None
    for off in range(STEP):
        r = step_start + off
        if r < len(ctx) and se_mark is not None and emb[r, se_mark].abs().item() > 0.5:
            se_row = r
            break
    return padded, se_row, ctx


def nearest_dim_name(dp_items, idx):
    # dp_items: sorted list of (pos, name). Return "name+offset" for the
    # largest pos <= idx.
    best = None
    for pos, name in dp_items:
        if pos <= idx:
            best = (pos, name)
        else:
            break
    if best is None:
        return f"<{idx}>"
    return f"{best[1]}+{idx - best[0]}"


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dp_items = sorted((int(v), k) for k, v in dp.items())

    resids = {}
    for pname, (src, res_step) in PROGS.items():
        bc, data = compile_c(src)
        padded, se_row, ctx = se_row_for(p, model, bc, res_step)
        if se_row is None:
            print(f"{pname}: no SE row", flush=True)
            return
        with torch.no_grad():
            r15 = model.forward(padded, stop_after_block=15)[0][se_row]
            r16 = model.forward(padded, stop_after_block=16)[0][se_row]
        resids[pname] = (r15.cpu(), r16.cpu())
        olo = dp["OUTPUT_LO"]
        print(f"{pname}: SE_row={se_row} blk15 OUTPUT_LO[0..2]="
              f"{r15[olo].item():+.3f}/{r15[olo+1].item():+.3f}/{r15[olo+2].item():+.3f}"
              f"  blk16={r16[olo].item():+.3f}/{r16[olo+1].item():+.3f}/{r16[olo+2].item():+.3f}",
              flush=True)

    v15, v16 = resids["ifVAR_66gt24_T"]
    g15, g16 = resids["ifGT_66gt24_T"]
    n = min(len(v15), len(g15))
    print("\n=== INPUT-to-block16 dims that DIFFER (|ifVAR - ifGT| > %.2f) ===" % THRESH, flush=True)
    diffs = []
    for i in range(n):
        d = float(v15[i].item() - g15[i].item())
        if abs(d) > THRESH:
            diffs.append((abs(d), i, float(v15[i].item()), float(g15[i].item()), d))
    diffs.sort(reverse=True)
    for ad, i, vv, gv, d in diffs[:60]:
        print(f"  dim {i:4d} [{nearest_dim_name(dp_items, i):28s}] ifVAR={vv:+12.4f} ifGT={gv:+12.4f} d={d:+12.4f}", flush=True)
    print(f"\n  total differing dims: {len(diffs)}", flush=True)


if __name__ == "__main__":
    main()
