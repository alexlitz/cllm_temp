#!/usr/bin/env python3
"""SE-row discriminator probe for the C4_CMP_COMBINE_MARGIN clamp.

The clamp fires at ``MARK_SE_ONLY AND SE_OP_<cmp>``. Its over-fire risk is a
row where ``SE_OP_<cmp>`` is hot but the row is NOT a genuine boolean-result
decode (a value-carrying row whose OUTPUT_HI legitimately holds a value's high
nibble). This probe dumps EVERY row that has ``SE_OP_<cmp>`` hot and reports:
  - the fresh L9 CMP cascade state (CMP+0..3 at that row),
  - OUTPUT_LO / OUTPUT_HI argmax (the emitted byte),
so we can see whether "fresh CMP present" cleanly discriminates the genuine
boolean-result decode from a value-carrying leak.

Run (single forward, ~1 min after bake):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 C4_CMP_COMBINE_MARGIN=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_cmpon PROBE_THREADS=8 \
    python tools/_probe_cmp_se_row_disc.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_cmpon")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "6")))

from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


# max(a,b)-style: compare then RETURN a value with a nonzero high nibble.
# GT(0x63, 0x12): a>b -> result 1 ; then IMM 0x63 (=99) is a value w/ hi nibble 6.
CASES = {
    # a compare (GT 0x23 vs 0x12 -> 1) followed by pushing a value 0x63 whose
    # high nibble (6) must NOT be clamped if SE_OP_GT leaks onto that row.
    "gt_then_val63": _mk([
        (Opcode.IMM, 0x23), Opcode.PSH, (Opcode.IMM, 0x12), Opcode.GT,
        (Opcode.IMM, 0x63), Opcode.EXIT,
    ]),
    # pure compare
    "gt_35_18": _mk([
        (Opcode.IMM, 0x23), Opcode.PSH, (Opcode.IMM, 0x12), Opcode.GT,
        Opcode.EXIT,
    ]),
}


def _add_funcmax():
    """Compile func_max (id 650) from the real corpus so we see the SE rows in
    a genuine function frame (compare inside the callee, then RETURN a value
    with a nonzero high nibble)."""
    try:
        from tests.test_suite_1000 import generate_test_programs
        from tools.run_1096_fast import _compile_and_oracle
        tests = generate_test_programs()
        sel = []
        for idx in (650, 425):
            src, exp, desc = tests[idx]
            sel.append((idx, src, exp, desc))
        prepared, _errs = _compile_and_oracle(sel)
        for entry in prepared:
            _idx, _exp, _desc, _de, _ds, bc, _data = entry
            CASES[f"corpus_{_idx}"] = list(bc)
    except Exception as e:  # pragma: no cover - probe convenience
        print(f"[probe] corpus load skipped: {e}", flush=True)


_add_funcmax()


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    se_base = dp["MARK_SE_ONLY"]
    ohi_key = "OUTPUT_HI_THIS_STEP" if "OUTPUT_HI_THIS_STEP" in dp else "OUTPUT_HI"
    cmp_names = ["EQ", "NE", "LT", "GT", "LE", "GE"]
    se_op = {}
    for nm in cmp_names:
        key = f"SE_OP_{nm}"
        if key in dp:
            se_op[nm] = dp[key]
    # Last block index (post-all-FFN residual).
    nblk = len(model.blocks) if hasattr(model, "blocks") else 60
    for cname, bc in CASES.items():
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        # Read residual at the block hosting the cmp clamp (L10-main + post-ops).
        # Use the final residual (all blocks) so SE_OP + CMP + OUTPUT all settled.
        with torch.no_grad():
            resid = model.forward(toks, stop_after_block=nblk - 1)[0]
        print(f"=== {cname}  S={S} ===", flush=True)
        for r in range(S):
            row = resid[r]
            se_hot = float(row[se_base].item())
            if se_hot < 0.5:
                continue
            # which SE_OP flags are hot here
            hot_ops = [nm for nm, base in se_op.items()
                       if float(row[base].item()) > 0.3]
            se_grp = float(row[dp["SE_CMP_GROUP"]].item()) if "SE_CMP_GROUP" in dp else -9.0
            cmp = [float(row[dp["CMP"] + i].item()) for i in range(4)]
            lo = torch.tensor([row[dp["OUTPUT_LO"] + i] for i in range(16)])
            hi = torch.tensor([row[dp[ohi_key] + i] for i in range(16)])
            lo_arg, hi_arg = int(lo.argmax()), int(hi.argmax())
            byte = lo_arg | (hi_arg << 4)
            cmp_any = any(abs(c) > 0.3 for c in cmp)
            print(f"  row{r:3d} SE_OP={hot_ops or '-'} SE_GRP={se_grp:.2f} "
                  f"CMP=[{cmp[0]:.2f},{cmp[1]:.2f},{cmp[2]:.2f},{cmp[3]:.2f}] "
                  f"freshCMP={'Y' if cmp_any else 'N'} "
                  f"byte=0x{byte:02x} HI_arg={hi_arg} HImax={float(hi.max()):.1f}",
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
