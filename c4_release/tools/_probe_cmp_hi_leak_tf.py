#!/usr/bin/env python3
"""cmp_combine OUTPUT_HI (hi<<4) leak — teacher-forced single-forward probe.

Contention-robust: ONE build + a handful of truncated forwards over a
teacher-forced canonical compare tape (no AR replay). Observes, at the
compare step's AX marker row, whether OUTPUT_HI_THIS_STEP carries a
leaked operand HIGH nibble that would turn the emitted result byte into
(hi<<4) | result. Mechanistic observation only (not a verdict).

We build the compare tape by hand at STEP_TOKENS granularity so the
compare AX row is deterministic. The tape mirrors the spec_k=0 canonical
state layout for IMM a; PSH; IMM b; <CMP>; EXIT.

Run:
  CUDA_VISIBLE_DEVICES="" C4_VM_CACHE_DIR=/tmp/c4cache_cmpmargin \
    python tools/_probe_cmp_hi_leak_tf.py 24 25 26 27 28
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_cmpmargin")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "4")))

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


def cmp2(opc, a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), opc, Opcode.EXIT])


# Both-nibbles-nonzero operands (drift trigger) + clean-nibble contrast.
CASES = {
    "gt_35_18": (cmp2(Opcode.GT, 35, 18), 1),   # 0x23 vs 0x12  A>B GT=1
    "gt_18_35": (cmp2(Opcode.GT, 18, 35), 0),   # A<B GT=0
    "lt_18_35": (cmp2(Opcode.LT, 18, 35), 1),
    "eq_35_35": (cmp2(Opcode.EQ, 35, 35), 1),   # eq both-nib-nonzero
    "eq_18_35": (cmp2(Opcode.EQ, 18, 35), 0),
    "gt_5_3":   (cmp2(Opcode.GT, 5, 3), 1),     # clean low-only contrast
    "gt_3_5":   (cmp2(Opcode.GT, 3, 5), 0),
}


def fmt(row, base, width=16, thr=0.3):
    vals = [(i, float(row[base + i].item())) for i in range(width)]
    hot = [(i, v) for i, v in vals if abs(v) > thr]
    hot.sort(key=lambda t: -t[1])
    return "[" + ", ".join(f"{v:.2f}@{i}" for i, v in hot) + "]"


def main(blocks):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ohi_key = "OUTPUT_HI_THIS_STEP" if "OUTPUT_HI_THIS_STEP" in dp else "OUTPUT_HI"
    se_key = "MARK_SE_ONLY" if "MARK_SE_ONLY" in dp else None
    for pname, (bc, expect) in CASES.items():
        # AR replay to get the true canonical tape for this program (once).
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_base = dp["MARK_AX"]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        se_rows = ([r for r in range(S) if se_key and
                    emb[r, dp[se_key]].abs().item() > 0.5] if se_key else [])
        flag = "PASS" if got == expect else "**FAIL**"
        print(f"=== {pname} expect={expect} got={got} {flag} "
              f"ax_rows={ax_rows} se_rows={se_rows} ===", flush=True)
        for b in blocks:
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=b)[0]
            # dump both AX and SE rows (the decode may land at SE under
            # the campaign step_end relay).
            check = sorted(set(ax_rows + se_rows))
            for r in check:
                row = resid[r]
                cmp = [float(row[dp["CMP"] + i].item()) for i in range(4)]
                lo_vec = torch.tensor([row[dp["OUTPUT_LO"] + i] for i in range(16)])
                hi_vec = torch.tensor([row[dp[ohi_key] + i] for i in range(16)])
                byte = int(lo_vec.argmax()) | (int(hi_vec.argmax()) << 4)
                lo = fmt(row, dp["OUTPUT_LO"])
                hi = fmt(row, dp[ohi_key])
                kind = "SE" if r in se_rows else "AX"
                cmphot = "*" if any(abs(c) > 0.3 for c in cmp) else " "
                print(f"  blk{b:2d} {kind}{r:3d} byte=0x{byte:02x}{cmphot} "
                      f"CMP=[{cmp[0]:.2f},{cmp[1]:.2f},{cmp[2]:.2f},{cmp[3]:.2f}] "
                      f"LO={lo} HI={hi}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [26, 27, 28]
    main(blks)
