"""Probe AX byte-1 cross-step persistence for test_si_li_16bit_value.

Bytecode:
  IMM 0x200; PSH; IMM 0x1234; SI; IMM 0x200; LI; EXIT

Expected: LI returns 0x1234 (4660). Actual: 52 (0x34 - byte 0 only).
Hypothesis: AX byte 1 (0x12) is dropped at some VM step boundary.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

PROG = [
    (Opcode.IMM, 0x200),
    Opcode.PSH,
    (Opcode.IMM, 0x1234),
    Opcode.SI,
    (Opcode.IMM, 0x200),
    Opcode.LI,
    Opcode.EXIT,
]


def make_bc(prog):
    out = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            out.append(op | (imm << 8))
        else:
            out.append(item)
    return out


def nibble_to_byte(lo_band, hi_band):
    lo_idx = int(torch.argmax(lo_band).item())
    hi_idx = int(torch.argmax(hi_band).item())
    return lo_idx | (hi_idx << 4), float(lo_band[lo_idx].item()), float(hi_band[hi_idx].item())


def find_marker_rows(x, dim_positions, marker_name, thresh=0.5):
    col = x[0, :, dim_positions[marker_name]]
    return torch.nonzero(col > thresh, as_tuple=False).flatten().tolist()


def main():
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True,
                                        use_kv_cache=False)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []

    model = runner.model
    dim_positions = model.embed._dim_positions
    n_blocks = len(model.blocks)
    print(f"num blocks: {n_blocks}")

    # Keep all forward passes; the last one is the longest sequence we use.
    all_captures = []  # list of dict keyed by ("pre"/"post", block_idx)
    current = {}

    def make_pre_hook(idx):
        def hook(module, inputs):
            nonlocal current
            # Block 0 pre marks the start of a new forward
            if idx == 0 and current:
                all_captures.append(current)
                current = {}
            current[("pre", idx)] = inputs[0].detach().clone()
        return hook

    def make_post_hook(idx):
        def hook(module, inputs, output):
            current[("post", idx)] = output.detach().clone()
        return hook

    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_pre_hook(make_pre_hook(i)))
        handles.append(blk.register_forward_hook(make_post_hook(i)))

    bc = make_bc(PROG)
    try:
        try:
            result = runner.run(bc, b"", max_steps=30)
        except Exception as e:
            print(f"runner raised: {e}")
            result = None
    finally:
        for h in handles:
            h.remove()
        if current:
            all_captures.append(current)

    print(f"Result: {result} (expected 4660)")
    print(f"Captured {len(all_captures)} forward passes")
    for i, c in enumerate(all_captures):
        if ("pre", 0) in c:
            print(f"  pass {i}: seq_len={c[('pre', 0)].shape[1]}")

    # Pick the longest forward = the last full one before EXIT.
    captures = max(all_captures, key=lambda c: c.get(("pre", 0), torch.zeros(1, 1, 1)).shape[1])
    x0 = captures[("pre", 0)]
    seq_len = x0.shape[1]
    print(f"Chosen forward: seq_len = {seq_len}")

    ax_rows = find_marker_rows(x0, dim_positions, "MARK_AX")
    mem_rows = find_marker_rows(x0, dim_positions, "MARK_MEM")
    sp_rows = find_marker_rows(x0, dim_positions, "MARK_SP")
    print(f"MARK_AX rows: {ax_rows}")
    print(f"MARK_MEM rows: {mem_rows}")
    print(f"MARK_SP  rows: {sp_rows}")

    si_ax = ax_rows[-2] if len(ax_rows) >= 2 else ax_rows[-1]
    li_ax = ax_rows[-1]
    print(f"SI AX row: {si_ax}, LI AX row: {li_ax}")

    LO = dim_positions["OUTPUT_LO"]
    HI = dim_positions["OUTPUT_HI"]
    AX_FULL_LO = dim_positions["AX_FULL_LO"]
    AX_FULL_HI = dim_positions["AX_FULL_HI"]
    AX_CARRY_LO = dim_positions["AX_CARRY_LO"]
    AX_CARRY_HI = dim_positions["AX_CARRY_HI"]
    EMB_LO = dim_positions["EMBED_LO"]
    EMB_HI = dim_positions["EMBED_HI"]
    CL_LO = dim_positions["CLEAN_EMBED_LO"]
    CL_HI = dim_positions["CLEAN_EMBED_HI"]
    MEM_VAL_B0 = dim_positions["MEM_VAL_B0"]
    MEM_VAL_B1 = dim_positions["MEM_VAL_B1"]

    def dump_row(label, x, row):
        out_lo = x[0, row, LO:LO+16]; out_hi = x[0, row, HI:HI+16]
        b0, b0lo_v, b0hi_v = nibble_to_byte(out_lo, out_hi)
        af_lo = x[0, row, AX_FULL_LO:AX_FULL_LO+16]; af_hi = x[0, row, AX_FULL_HI:AX_FULL_HI+16]
        af_b, af_lo_v, af_hi_v = nibble_to_byte(af_lo, af_hi)
        ac_lo = x[0, row, AX_CARRY_LO:AX_CARRY_LO+16]; ac_hi = x[0, row, AX_CARRY_HI:AX_CARRY_HI+16]
        ac_b, ac_lo_v, ac_hi_v = nibble_to_byte(ac_lo, ac_hi)
        em_lo = x[0, row, EMB_LO:EMB_LO+16]; em_hi = x[0, row, EMB_HI:EMB_HI+16]
        eb, _, _ = nibble_to_byte(em_lo, em_hi)
        cl_lo = x[0, row, CL_LO:CL_LO+16]; cl_hi = x[0, row, CL_HI:CL_HI+16]
        cb, _, _ = nibble_to_byte(cl_lo, cl_hi)
        print(f"  {label:14s}: OUT=0x{b0:02x}({b0lo_v:.1f}/{b0hi_v:.1f}) AX_FULL=0x{af_b:02x}({af_lo_v:.1f}/{af_hi_v:.1f}) AX_CARRY=0x{ac_b:02x}({ac_lo_v:.1f}/{ac_hi_v:.1f}) EMB=0x{eb:02x} CL=0x{cb:02x}")

    # AX value occupies 4 rows: marker+0(=marker), +1=byte0, +2=byte1, +3=byte2, +4=byte3
    # (positions per token_layout.py: POS_AX_MARKER=5, POS_AX_BYTE0=6, BYTE1=7, BYTE2=8, BYTE3=9)
    # marker row is offset 0; byte0 = +1, byte1 = +2, byte2 = +3, byte3 = +4
    def trace_step(label, ax_marker_row):
        for byte_idx, dy in enumerate([1, 2, 3, 4]):
            row = ax_marker_row + dy
            if row >= seq_len:
                continue
            print(f"  --- {label} AX byte {byte_idx} (row {row}) ---")
            # Just dump key blocks: 0, 3, 5, 6, 10, 14, 28, 34, 36
            for i in [0, 3, 5, 6, 10, 14, 15, 20, 28, 34, 35, 36]:
                if ("post", i) in captures and i < n_blocks:
                    dump_row(f"L{i:02d}", captures[("post", i)], row)

    # Walk through ALL AX rows: 0=IMM 0x200, 1=PSH, 2=IMM 0x1234, 3=SI, 4=IMM 0x200, 5=LI
    step_labels = ["IMM_0x200", "PSH", "IMM_0x1234", "SI", "IMM_0x200_b", "LI"]
    for idx, ax_pos in enumerate(ax_rows):
        if idx < len(step_labels):
            label = step_labels[idx]
        else:
            label = f"step{idx}"
        print(f"\n=== {label} step (AX @ {ax_pos}) byte rows ===")
        trace_step(label, ax_pos)

    if mem_rows:
        last_mem = mem_rows[-1]
        print(f"\n=== Last MEM frame @ {last_mem}: byte rows ===")
        for d in range(0, 10):
            p = last_mem + d
            if p >= seq_len:
                continue
            mb0 = float(x0[0, p, MEM_VAL_B0].item())
            mb1 = float(x0[0, p, MEM_VAL_B1].item())
            # Use the final (post last block) state
            final = captures[("post", n_blocks - 1)]
            cl_lo_band = final[0, p, CL_LO:CL_LO+16]
            cl_hi_band = final[0, p, CL_HI:CL_HI+16]
            b, _, _ = nibble_to_byte(cl_lo_band, cl_hi_band)
            out_lo_band = final[0, p, LO:LO+16]
            out_hi_band = final[0, p, HI:HI+16]
            ob, olo, ohi = nibble_to_byte(out_lo_band, out_hi_band)
            print(f"  p={p} d={d}: MEM_VAL_B0={mb0:.2f} MEM_VAL_B1={mb1:.2f} final CLEAN=0x{b:02x} final OUT=0x{ob:02x}({olo:.1f}/{ohi:.1f})")


if __name__ == "__main__":
    main()
