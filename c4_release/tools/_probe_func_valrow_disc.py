#!/usr/bin/env python3
"""Find a CAM discriminator: genuine store VALUE rows vs spurious ADDR_B0_LO=8
byte rows. Dumps MEM_VAL_B0..B3, MEM_STORE*, NEXT_MEM, ADDR_B0_HI for the
a-value row (271, correct but loses), b-value row (331, correct + wins), and the
spurious winners (292-295) at func_add step9/step12, so we can pick a positive K
that lifts genuine value rows over the spurious cluster WITHOUT hurting var.
"""
from __future__ import annotations
import os, sys, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(REPO)
for p in (PROJ, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("C4_SMOKE_SPEC_K", "0"); os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa
from c4_release.neural_vm.vm_step import Token  # noqa
from tools.interp_oracle_gate import (build_production_model, build_code_prompt,
                                       oracle_tape_and_steps)  # noqa
STEP = int(Token.STEP_TOKENS); PROGS = generate_test_programs()

DISC = ["MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3", "MEM_STORE",
        "NEXT_MEM", "ADDR_B0_HI", "ADDR_B0_LO", "MARK_MEM", "CLEAN_EMBED_LO"]


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dimpos = layout.dim_positions
    op_li = dimpos["OP_LI_RELAY"]
    L15 = next(bi for bi, blk in enumerate(model.blocks)
               if abs(float((blk.attn.W_q.to_dense() if blk.attn.W_q.is_sparse
                             else blk.attn.W_q)[0, op_li].item())) > 1000.0)
    pid = 575
    src, exp, _ = PROGS[pid]
    bc = compile_c(src)[0]
    prompt = build_code_prompt(bc, b"")
    ot = oracle_tape_and_steps(bc, b"", max_steps=40)
    tok = torch.tensor([list(prompt) + list(ot.draft_tokens)], dtype=torch.long)
    cap = {}
    h = model.blocks[L15].register_forward_pre_hook(
        lambda m, i: cap.__setitem__("pre", i[0].detach().clone()))
    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()):
            model.forward(tok)
    h.remove()
    pre = cap["pre"][0]

    def nib(pos, base):
        band = pre[pos, dimpos[base]:dimpos[base]+16]
        return int(band.argmax().item()) if float(band.max()) > 0.3 else -1

    rows = {"a-VAL(271)": 271, "b-VAL(331)": 331,
            "spur292": 292, "spur293": 293, "spur294": 294, "spur295": 295,
            "a-store(266)": 266, "b-store(326)": 326}
    print(f"{'row':14}", *(f"{d.replace('MEM_VAL_','V').replace('_EMBED',''):>8}" for d in DISC[:8]))
    for tag, pos in rows.items():
        vals = []
        for d in DISC[:8]:
            if d in ("ADDR_B0_HI", "ADDR_B0_LO"):
                vals.append(f"{nib(pos,d):>8}")
            else:
                vals.append(f"{float(pre[pos, dimpos[d]].item()):8.2f}")
        print(f"{tag:14}", *vals)
    # also CLEAN_LO nibble (the value byte0 the V/O copies)
    print("\nCLEAN_LO nib:", {t: nib(p, "CLEAN_EMBED_LO") for t, p in rows.items()})


if __name__ == "__main__":
    main()
