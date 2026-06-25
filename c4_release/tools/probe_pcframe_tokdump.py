#!/usr/bin/env python3
"""PC-framing token dump (campaign, spec_k=0): raw AR token stream per step.

For var_update id325 (SI store) + if_var id425 (BZ branch), runs the production
spec_k=0 AR decode and DUMPS the raw emitted tokens around the divergence step,
annotating each token with its marker name. Shows WHERE the 30-tok frame breaks
(an extra/missing token -> the per-step PC marker shifts off offset 0 -> the
fixed-30-slice decoder reads pc=None).

  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    python tools/probe_pcframe_tokdump.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

# Default cases: var_update (id325-like SI store) + if_var (id425-like BZ branch).
PROGS = {
    "varupd_x50_p7": ("int main() { int x; x = 50; x = x + 7; return x; }",),
    "varupd_x36_p16": ("int main() { int x; x = 36; x = x + 16; return x; }",),
    "ifv_66_24_T": ("int main() { int x; x = 66; if (x > 24) return 1; return 0; }",),
    "ifv_85_48_T": ("int main() { int x; x = 85; if (x > 48) return 1; return 0; }",),
    "ifv_23_62_F": ("int main() { int x; x = 23; if (x > 62) return 1; return 0; }",),
}

# Marker token ids -> names, for annotation.
def marker_names():
    names = {}
    for attr in dir(Token):
        if attr.startswith("REG_") or attr in (
            "MARK_AX", "STEP_END", "HALT", "MEM_STORE", "ADDR_KEY",
        ):
            try:
                v = int(getattr(Token, attr))
                names.setdefault(v, attr)
            except Exception:
                pass
    # Common markers used in the per-step layout.
    for nm in ("REG_PC", "REG_AX", "REG_SP", "REG_BP", "STACK0",
               "MEM_MARKER", "STEP_END", "HALT"):
        if hasattr(Token, nm):
            names.setdefault(int(getattr(Token, nm)), nm)
    return names


NAMES = marker_names()


def tname(t):
    return NAMES.get(int(t), "")


def main():
    p = build_groundtruth_probe()
    runner = p.runner
    STEP = int(Token.STEP_TOKENS)
    pc_marker = int(Token.REG_PC)
    ax_marker = int(Token.REG_AX)

    for nm, (src,) in PROGS.items():
        bc, data = compile_c(src)
        opc_ax, _ = runner._oracle_pc_ax_steps(
            bc, data or b"", "", expected_steps=None, with_tokens=True)
        ctx = p._final_context(bc, max_steps=30)
        prefix_len = len(p._build_context(bc))
        nemit = (len(ctx) - prefix_len) // STEP
        # find the FIRST step where REG_PC is not at the segment start (the
        # framing break) by scanning fixed-30 segments.
        first_break = None
        for s in range(nemit):
            seg = ctx[prefix_len + s * STEP: prefix_len + (s + 1) * STEP]
            if not seg or seg[0] != pc_marker:
                first_break = s
                break
        print(f"=== {nm} prefix_len={prefix_len} nemit={nemit} "
              f"oracle_steps={len(opc_ax)} first_frame_break_step={first_break} ===",
              flush=True)
        # Dump raw tokens from prefix_len onward, marking every REG_PC and
        # counting tokens between consecutive REG_PC markers (true step length).
        body = ctx[prefix_len:]
        pc_positions = [i for i, t in enumerate(body) if t == pc_marker]
        print(f"   REG_PC marker abs-offsets (from prefix): {pc_positions[:20]}",
              flush=True)
        # inter-marker gaps (true emitted step lengths)
        gaps = [pc_positions[i + 1] - pc_positions[i]
                for i in range(len(pc_positions) - 1)]
        print(f"   inter-PC gaps (true step token counts): {gaps[:20]}", flush=True)

        # Per-step (PC,AX) decode-vs-oracle to find the FIRST value divergence.
        def dec_reg(seg, mk):
            for i, t in enumerate(seg):
                if t == mk and i + 4 < len(seg):
                    v = 0
                    for j in range(4):
                        v |= (int(seg[i + 1 + j]) & 0xFF) << (j * 8)
                    return v
            return None
        first_div = None
        for s in range(min(nemit, len(opc_ax))):
            seg = ctx[prefix_len + s * STEP: prefix_len + (s + 1) * STEP]
            gp = dec_reg(seg, pc_marker)
            ga = dec_reg(seg, ax_marker)
            ep, ea = opc_ax[s]
            ok = (gp == ep and ga == ea)
            print(f"   step{s:2d} exp(pc={ep},ax={ea}) got(pc={gp},ax={ga})"
                  f"{'' if ok else '  <-- DIVERGE'}", flush=True)
            if not ok and first_div is None:
                first_div = s
        # Detailed dump around the first value divergence (fallback to break/last).
        focus_step = (first_div if first_div is not None else
                      first_break if first_break is not None else
                      (len(gaps) - 1 if gaps else 0))
        lo = max(0, prefix_len + (focus_step - 1) * STEP)
        hi = min(len(ctx), prefix_len + (focus_step + 3) * STEP)
        print(f"   --- raw token dump (focus step {focus_step}, "
              f"abs {lo}..{hi}) ---", flush=True)
        for i in range(lo, hi):
            t = int(ctx[i])
            rel = i - prefix_len
            step = rel // STEP
            off = rel % STEP
            mk = tname(t)
            tag = ""
            if t == pc_marker:
                tag = "  <== REG_PC"
            elif t == ax_marker:
                tag = "  <- REG_AX"
            elif mk == "STEP_END":
                tag = "  <- STEP_END"
            print(f"      abs{i:4d} step{step:2d} off{off:2d}  tok={t:4d} "
                  f"{mk:10s}{tag}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
