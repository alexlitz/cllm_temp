"""Wave 1 A3.5 L14 mem_generation consumer probe (2026-06-09).

The L10 broadcast head (commit b02be67c) now correctly populates
``STACK0_BYTE_VAL_h_LO/HI`` at the STACK0 byte-h row of the PSH-step
frame (verified by ``probe_a3_broadcast_diagnostic.py``: STACK0@80,
byte 1 row has V1_LO=2/3.00 for PSH of IMM 0x200, matching byte 1 =
0x02). But the L14 mem_generation consumer (heads 1/2/3) still emits
wrong OUTPUT at the SI MEM addr-byte rows.

This probe dumps:

  1. STACK0_BYTE_VAL_h_LO/HI values at every STACK0+BYTE_INDEX_h row
     in the full-context capture, after physical block 25 (L13, the
     block right before L14 mem_generation).
  2. MEM_ADDR_SRC values at the SI step MARK_MEM and addr-byte rows.
  3. CLEAN_EMBED at the same rows.
  4. OUTPUT_LO/HI written by L14 (after block 27) at the SI addr/val
     rows, for comparison.

Goal: confirm that the broadcast is alive at the input of L14, then
diagnose whether the L14 head's K-side fails to attend to the right
STACK0 byte-h row (vs spreading attention across all BYTE_INDEX_h rows
in context).
"""

import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(REPO_ROOT))

import torch

from c4_release.neural_vm.run_vm import AutoregressiveVMRunner
from c4_release.neural_vm.embedding import Opcode

PROG = [
    (Opcode.IMM, 0x200),
    Opcode.PSH,
    (Opcode.IMM, 42),
    Opcode.SI,
    (Opcode.IMM, 0x200),
    Opcode.LI,
    Opcode.EXIT,
]


def make_bc(prog):
    out = []
    for item in prog:
        if isinstance(item, tuple):
            opcode, imm = item
            out.append(opcode | (imm << 8))
        else:
            out.append(item)
    return out


def _band(after, pos, dim_start, width=16):
    band = after[0, pos, dim_start:dim_start + width]
    idx = int(torch.argmax(band).item())
    v = float(band[idx].item())
    return idx, v


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    _probe(runner)


def _probe(runner):
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []

    model = runner.model
    dp = model.embed._dim_positions

    captures = []

    def embed_hook(module, inputs, output):
        captures.append({"token_ids": inputs[0].detach().clone()})

    def block_hook(name):
        def fn(module, inputs, output):
            if captures:
                captures[-1][name] = output.detach().clone()
        return fn

    handles = [model.embed.register_forward_hook(embed_hook)]
    for li in range(min(32, len(model.blocks))):
        handles.append(
            model.blocks[li].register_forward_hook(block_hook(f"after_L{li}"))
        )

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

    print(f"Result: {result}")

    # Last capture = full context
    best = None
    for c in captures:
        if "after_L25" not in c:
            continue
        if best is None or c["after_L25"].shape[1] > best["after_L25"].shape[1]:
            best = c

    if best is None:
        print("No usable capture.")
        return

    after_L13 = best["after_L25"]  # input to L14 mem_generation
    after_L14 = best["after_L27"]
    seq = after_L13.shape[1]
    print(f"Full-context capture: seq_len={seq}")

    mark_stack0 = dp["MARK_STACK0"]
    mark_mem = dp["MARK_MEM"]
    op_si = dp["OP_SI"]
    op_psh = dp["OP_PSH"]
    mem_addr_src = dp["MEM_ADDR_SRC"]
    mem_store = dp["MEM_STORE"]
    bi = [dp[f"BYTE_INDEX_{j}"] for j in range(4)]
    sb_v1_lo = dp["STACK0_BYTE_VAL_1_LO"]
    sb_v1_hi = dp["STACK0_BYTE_VAL_1_HI"]
    sb_v2_lo = dp["STACK0_BYTE_VAL_2_LO"]
    sb_v2_hi = dp["STACK0_BYTE_VAL_2_HI"]
    sb_v3_lo = dp["STACK0_BYTE_VAL_3_LO"]
    sb_v3_hi = dp["STACK0_BYTE_VAL_3_HI"]
    clean_lo = dp["CLEAN_EMBED_LO"]
    clean_hi = dp["CLEAN_EMBED_HI"]
    out_lo = dp["OUTPUT_LO"]
    out_hi = dp["OUTPUT_HI"]

    stack0_positions = torch.nonzero(
        after_L13[0, :, mark_stack0] > 0.5, as_tuple=False
    ).flatten().tolist()
    mem_positions = torch.nonzero(
        after_L13[0, :, mark_mem] > 0.5, as_tuple=False
    ).flatten().tolist()
    si_rows = torch.nonzero(
        after_L13[0, :, op_si] > 0.5, as_tuple=False
    ).flatten().tolist()
    psh_rows = torch.nonzero(
        after_L13[0, :, op_psh] > 0.5, as_tuple=False
    ).flatten().tolist()

    print(f"\nMARK_STACK0 positions: {stack0_positions}")
    print(f"MARK_MEM positions:    {mem_positions}")
    print(f"OP_PSH active rows:    {psh_rows}")
    print(f"OP_SI  active rows:    {si_rows}")

    # === Section 1: STACK0_BYTE_VAL_h at each STACK0+BYTE_INDEX_h row.
    print("\n== Section 1: STACK0_BYTE_VAL_h (after L13, input to L14) ==")
    print("  Expected: only the STACK0 frame corresponding to the PSH step")
    print("  should have non-zero broadcast values.")
    for s_pos in stack0_positions:
        for h in (1, 2, 3):
            tgt = None
            for d in range(0, 10):
                p = s_pos + d
                if p >= seq:
                    continue
                if float(after_L13[0, p, bi[h]].item()) > 0.5:
                    tgt = p
                    break
            if tgt is None:
                continue
            lo_dim = [None, sb_v1_lo, sb_v2_lo, sb_v3_lo][h]
            hi_dim = [None, sb_v1_hi, sb_v2_hi, sb_v3_hi][h]
            lo = _band(after_L13, tgt, lo_dim)
            hi = _band(after_L13, tgt, hi_dim)
            cl_lo = _band(after_L13, tgt, clean_lo)
            cl_hi = _band(after_L13, tgt, clean_hi)
            print(
                f"  S0@{s_pos:3d} BI{h} @ p={tgt:3d}  "
                f"VAL{h}_LO={lo[0]:2d}/{lo[1]:5.2f}  VAL{h}_HI={hi[0]:2d}/{hi[1]:5.2f}  "
                f"CL_LO={cl_lo[0]:2d}/{cl_lo[1]:4.2f} CL_HI={cl_hi[0]:2d}/{cl_hi[1]:4.2f}"
            )

    # === Section 2: For each MEM row, dump MEM_ADDR_SRC at addr-byte rows.
    print("\n== Section 2: MEM_ADDR_SRC at MEM addr-byte rows (after L13) ==")
    for m_pos in mem_positions:
        print(f"\n  MARK_MEM @ {m_pos}")
        for d in range(0, 8):
            p = m_pos + d
            if p >= seq:
                continue
            mas = float(after_L13[0, p, mem_addr_src].item())
            ms = float(after_L13[0, p, mem_store].item())
            mark_m = float(after_L13[0, p, mark_mem].item())
            bi_now = [float(after_L13[0, p, bi[j]].item()) for j in range(4)]
            print(
                f"    d={d} p={p:3d}  MEM_ADDR_SRC={mas:5.2f}  MEM_STORE={ms:5.2f}  "
                f"MARK_MEM={mark_m:4.1f}  BI=[{bi_now[0]:3.0f},{bi_now[1]:3.0f},{bi_now[2]:3.0f},{bi_now[3]:3.0f}]"
            )

    # === Section 3: enumerate all rows where BYTE_INDEX_1 fires AND look at
    # STACK0_BYTE_VAL_1_LO/HI. The L14 head 1 K-side selector for slot 0 is
    # AP(0, BYTE_INDEX_1, L). Without further K-side gating that restricts
    # the row to "the STACK0 byte 1 row of the PSH step," softmax will
    # spread attention across every BYTE_INDEX_1 row. Show which ones carry
    # the broadcast (and which don't).
    print("\n== Section 3: All BYTE_INDEX_1 rows (K candidates for L14 head 1) ==")
    bi1_rows = torch.nonzero(
        after_L13[0, :, bi[1]] > 0.5, as_tuple=False
    ).flatten().tolist()
    print(f"  BYTE_INDEX_1 rows: {bi1_rows}")
    for p in bi1_rows:
        # MEM_ADDR_SRC at this K row — slot 2 gates on K MARKER dims.
        cl_lo = _band(after_L13, p, clean_lo)
        cl_hi = _band(after_L13, p, clean_hi)
        v_lo = _band(after_L13, p, sb_v1_lo)
        v_hi = _band(after_L13, p, sb_v1_hi)
        # Markers at this row:
        m_ax = float(after_L13[0, p, dp["MARK_AX"]].item())
        m_sp = float(after_L13[0, p, dp["MARK_SP"]].item())
        m_bp = float(after_L13[0, p, dp["MARK_BP"]].item())
        m_pc = float(after_L13[0, p, dp["MARK_PC"]].item())
        m_s0 = float(after_L13[0, p, mark_stack0].item())
        m_m = float(after_L13[0, p, mark_mem].item())
        print(
            f"  p={p:3d}  "
            f"AX={m_ax:.0f} SP={m_sp:.0f} BP={m_bp:.0f} PC={m_pc:.0f} S0={m_s0:.0f} MEM={m_m:.0f}  "
            f"CL_LO={cl_lo[0]:2d}/{cl_lo[1]:4.2f} CL_HI={cl_hi[0]:2d}/{cl_hi[1]:4.2f}  "
            f"VAL1_LO={v_lo[0]:2d}/{v_lo[1]:5.2f} VAL1_HI={v_hi[0]:2d}/{v_hi[1]:5.2f}"
        )

    # === Section 4: After L14, show OUTPUT_LO/HI at each MEM addr-byte row
    # (compares against the input STACK0_BYTE_VAL_h broadcast).
    print("\n== Section 4: OUTPUT_LO/HI at MEM rows (after L14) ==")
    for m_pos in mem_positions:
        print(f"\n  MARK_MEM @ {m_pos}")
        for d in range(0, 8):
            p = m_pos + d
            if p >= seq:
                continue
            ol = _band(after_L14, p, out_lo)
            oh = _band(after_L14, p, out_hi)
            bi_now = [float(after_L14[0, p, bi[j]].item()) for j in range(4)]
            print(
                f"    d={d} p={p:3d}  BI=[{bi_now[0]:3.0f},{bi_now[1]:3.0f},{bi_now[2]:3.0f},{bi_now[3]:3.0f}]  "
                f"OUT_LO={ol[0]:2d}/{ol[1]:5.2f}  OUT_HI={oh[0]:2d}/{oh[1]:5.2f}"
            )


if __name__ == "__main__":
    main()
