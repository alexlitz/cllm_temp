#!/usr/bin/env python3
"""Nail down the K-side discriminator for the SI-store-addr CAM (id275).

The LEVER CAM keys Q on the LI-query AX_CARRY (target addr 0xE8) and K on the
candidate ADDR_B0 (store addr). But nibble-matching ADDR_B0=0xE8 selects BOTH
the a-store AX-marker (row ~281, AX_CARRY=0x17=value) AND an earlier
address-computation AX-marker (row ~251, AX_CARRY=0xE8=address). We must PREFER
the store marker. This probe dumps the full nonzero-dim signature of every
MARK_AX row in a's frame (251/281/311) and b's frame (371/401/431) at the L15
block input, to find the dim that is HOT on the store marker but not on the
address-computation marker (candidate: MEM_STORE / MEM_STORE_AT_VAL / OP_SI
residue / MARK_MEM).

Run: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
     python tools/_probe_si_marker_disc.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
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
from neural_vm.speculative import DraftVM  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int a; int b; a = 23; b = 47; return a * b; }"


def oracle_windows(bc):
    vm = DraftVM(list(bc))
    steps, toks = [], []
    for _ in range(40):
        if vm.halted:
            break
        if not vm.step():
            break
        steps.append((int(vm.pc) & 0xFFFFFFFF, int(vm.ax) & 0xFFFF))
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break
    return steps, toks


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = model.embed._dim_positions
    rev = {v: k for k, v in dp.items()}

    op_li = dp["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li].item())) > 1000.0:
            L15 = bi

    bc, _ = compile_c(SRC)
    prompt = p._build_context(bc)
    steps, toks = oracle_windows(bc)
    ctx = list(prompt)
    for w in toks:
        ctx.extend(w)
    padded = torch.tensor([ctx], device=p._device)

    cap = {}
    h = model.blocks[L15].register_forward_pre_hook(
        lambda m, i: cap.__setitem__("pre", i[0].detach().clone()))
    with torch.no_grad():
        model(padded)
    h.remove()
    pre = cap["pre"][0]
    seq = pre.shape[0]
    mark_ax = dp["MARK_AX"]

    # candidate discriminator dims
    DISC = ["MEM_STORE", "MEM_STORE_AT_VAL", "OP_SI", "MARK_MEM", "MARK_STACK0",
            "MEM_VAL_B0", "MEM_VAL_B1", "MEM_ADDR_SRC", "ADDR_B0_VALID",
            "MEM_STORE_MARK", "HAS_SE", "OP_LI", "OP_LI_RELAY", "OP_ENT",
            "OP_MUL", "OP_PSH", "CLEAN_EMBED_LO"]
    DISC = [d for d in DISC if d in dp]

    def nib(pos, base):
        band = pre[pos, dp[base]:dp[base] + 16]
        mx = float(band.max().item())
        return (int(band.argmax().item()), round(mx, 2)) if mx > 0.3 else (None, round(mx, 2))

    def hexval(pos, lo, hi):
        l, _ = nib(pos, lo); hh, _ = nib(pos, hi)
        if l is None and hh is None:
            return None
        return ((hh or 0) << 4) | (l or 0)

    ax_rows = [pos for pos in range(seq) if pre[pos, mark_ax].item() > 0.5]
    print(f"L15={L15} seq={seq}")
    print(f"MARK_AX rows: {ax_rows}")
    print("\n=== per MARK_AX row: ADDR_B0, AX_CARRY, discriminator dims ===")
    hdr = f"{'row':>4} {'ADDR_B0':>7} {'AXCARRY':>7}  " + " ".join(
        f"{d[:11]:>11}" for d in DISC)
    print(hdr)
    for pos in ax_rows:
        ab = hexval(pos, "ADDR_B0_LO", "ADDR_B0_HI")
        ac = hexval(pos, "AX_CARRY_LO", "AX_CARRY_HI")
        vals = []
        for d in DISC:
            if d == "CLEAN_EMBED_LO":
                v = nib(pos, "CLEAN_EMBED_LO")[0]
                vals.append(f"{('%x'%v) if v is not None else '-':>11}")
            else:
                vals.append(f"{pre[pos, dp[d]].item():11.2f}")
        s_ab = "-" if ab is None else hex(ab)
        s_ac = "-" if ac is None else hex(ac)
        print(f"{pos:>4} {s_ab:>7} {s_ac:>7}  " + " ".join(vals))


if __name__ == "__main__":
    main()
