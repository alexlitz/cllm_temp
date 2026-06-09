"""L14/L15/L17 LI consumer probe (2026-06-09).

After commit f6ebf6e7 (A3.10 OP_IMM discriminator), the L10 broadcast
correctly delivers byte-1 = 2 (0x02) at all STACK0 BI1 rows. SI step
also lands the right MEM_value bytes (verified by probe_a3_5_l14_consumer
output above: MEM@118 has MEM_STORE active, byte-1 = 0x2A would imply
val=42).

Remaining bug: test_si_li_roundtrip returns 512 (=0x200) instead of 42.
The 0x200 is the ADDRESS pushed at the PSH step (PSH of IMM 0x200).

This probe focuses on the LI step (OP_LI row):
  1. Identify the LI step's MARK_AX row (= where AX byte 0 lands).
  2. Dump REG_AX_LO/HI band at MARK_AX after L13, L14, L15, L17.
  3. Dump OUTPUT_LO/HI at MARK_AX.
  4. Identify which layer flips AX from "load value 42" to "address 0x200".
  5. For L15 mem_lookup head (loads MEM_value into AX), dump attention
     weights from MARK_AX query against all K candidate rows.
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
    for li in range(min(40, len(model.blocks))):
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

    best = None
    for c in captures:
        if "after_L25" not in c:
            continue
        if best is None or c["after_L25"].shape[1] > best["after_L25"].shape[1]:
            best = c

    if best is None:
        print("No usable capture.")
        return

    # Capture layers — L25..L35
    keys_to_try = [f"after_L{i}" for i in range(0, 40)]
    available = {k: best[k] for k in keys_to_try if k in best}
    print("Captured layers:", sorted(available.keys()))

    after_L25 = best["after_L25"]  # input to L14 mem_generation
    seq = after_L25.shape[1]
    print(f"seq_len={seq}")

    mark_ax = dp["MARK_AX"]
    mark_mem = dp["MARK_MEM"]
    mark_sp = dp["MARK_SP"]
    mark_stack0 = dp["MARK_STACK0"]
    op_li = dp["OP_LI"]
    op_si = dp["OP_SI"]
    op_psh = dp["OP_PSH"]
    op_imm = dp["OP_IMM"]
    op_li_relay = dp["OP_LI_RELAY"]
    bi = [dp[f"BYTE_INDEX_{j}"] for j in range(4)]
    out_lo = dp["OUTPUT_LO"]
    out_hi = dp["OUTPUT_HI"]
    clean_lo = dp["CLEAN_EMBED_LO"]
    clean_hi = dp["CLEAN_EMBED_HI"]
    reg_ax_lo = dp.get("REG_AX_LO")
    reg_ax_hi = dp.get("REG_AX_HI")
    addr_b0_lo = dp.get("ADDR_B0_LO")
    addr_b0_hi = dp.get("ADDR_B0_HI")
    mem_val_b1 = dp.get("MEM_VAL_B1")
    mem_val_b2 = dp.get("MEM_VAL_B2")
    mem_val_b3 = dp.get("MEM_VAL_B3")
    mem_store = dp.get("MEM_STORE")
    has_se = dp.get("HAS_SE")

    op_li_rows = torch.nonzero(
        after_L25[0, :, op_li] > 0.5, as_tuple=False
    ).flatten().tolist()
    ax_rows = torch.nonzero(
        after_L25[0, :, mark_ax] > 0.5, as_tuple=False
    ).flatten().tolist()
    mem_rows = torch.nonzero(
        after_L25[0, :, mark_mem] > 0.5, as_tuple=False
    ).flatten().tolist()

    print(f"\nOP_LI rows:    {op_li_rows}")
    print(f"MARK_AX rows:  {ax_rows}")
    print(f"MARK_MEM rows: {mem_rows}")

    # Find the LI step's AX rows. LI step is the LAST AX block, after the
    # SI step. Use heuristic: the AX block whose rows are > max OP_LI row.
    if op_li_rows:
        li_last = max(op_li_rows)
        li_ax_rows = [r for r in ax_rows if r >= li_last]
    else:
        li_ax_rows = []
    print(f"LI step MARK_AX rows: {li_ax_rows[:8]}")

    # Also find SI step AX rows for comparison.
    op_si_rows = torch.nonzero(
        after_L25[0, :, op_si] > 0.5, as_tuple=False
    ).flatten().tolist()
    if op_si_rows:
        si_last = max(op_si_rows)
        upper = op_li_rows[0] if op_li_rows else 10**9
        si_ax_rows = [r for r in ax_rows if si_last < r < upper]
        if not si_ax_rows:
            si_ax_rows = [r for r in ax_rows if r >= si_last - 5 and r < upper]
    else:
        si_ax_rows = []

    # Dump REG_AX_LO/HI byte-1 across layers.
    print("\n== REG_AX byte-1 evolution across layers (at LI step AX rows) ==")
    if li_ax_rows:
        # LI's AX block: byte 1 row = first AX row + 1 (since BI=0 at row+0, BI=1 at row+1...)
        ax0 = li_ax_rows[0]
        for d in range(0, 5):
            p = ax0 + d
            if p >= seq:
                continue
            print(f"\n  p={p:3d}  (LI step, AX block offset d={d})")
            for ln in sorted(available.keys(), key=lambda s: int(s.split("L")[1])):
                arr = available[ln]
                if arr.shape[1] <= p:
                    continue
                row = f"    {ln}: "
                if reg_ax_lo is not None:
                    rlo = _band(arr, p, reg_ax_lo)
                    rhi = _band(arr, p, reg_ax_hi)
                    row += f"AX_LO={rlo[0]:2d}/{rlo[1]:5.2f}  AX_HI={rhi[0]:2d}/{rhi[1]:5.2f}  "
                olo = _band(arr, p, out_lo)
                ohi = _band(arr, p, out_hi)
                row += f"OUT_LO={olo[0]:2d}/{olo[1]:5.2f}  OUT_HI={ohi[0]:2d}/{ohi[1]:5.2f}"
                bi_now = [float(arr[0, p, bi[j]].item()) for j in range(4)]
                row += f"  BI=[{bi_now[0]:3.0f},{bi_now[1]:3.0f},{bi_now[2]:3.0f},{bi_now[3]:3.0f}]"
                print(row)

    # Same evolution for SI step AX rows (sanity).
    print("\n== REG_AX byte-1 evolution across layers (at SI step AX rows) ==")
    if si_ax_rows:
        ax0 = si_ax_rows[0]
        for d in range(0, 5):
            p = ax0 + d
            if p >= seq:
                continue
            print(f"\n  p={p:3d}  (SI step, AX block offset d={d})")
            for ln in sorted(available.keys(), key=lambda s: int(s.split("L")[1])):
                arr = available[ln]
                if arr.shape[1] <= p:
                    continue
                row = f"    {ln}: "
                if reg_ax_lo is not None:
                    rlo = _band(arr, p, reg_ax_lo)
                    rhi = _band(arr, p, reg_ax_hi)
                    row += f"AX_LO={rlo[0]:2d}/{rlo[1]:5.2f}  AX_HI={rhi[0]:2d}/{rhi[1]:5.2f}  "
                olo = _band(arr, p, out_lo)
                ohi = _band(arr, p, out_hi)
                row += f"OUT_LO={olo[0]:2d}/{olo[1]:5.2f}  OUT_HI={ohi[0]:2d}/{ohi[1]:5.2f}"
                print(row)

    # Check after L25 (pre-L14) — what's the AX row's K-side signature?
    # Specifically: what's at ADDR_B0_LO/HI on the LI AX rows?
    print("\n== LI step AX row Q-side signatures (markers/addr) ==")
    if li_ax_rows:
        for p in li_ax_rows[:5]:
            mk = []
            for nm, d in (("AX", mark_ax), ("MEM", mark_mem), ("SP", mark_sp), ("STACK0", mark_stack0)):
                mk.append(f"{nm}={float(after_L25[0, p, d].item()):.1f}")
            relay = float(after_L25[0, p, op_li_relay].item())
            li_o = float(after_L25[0, p, op_li].item())
            adlo = _band(after_L25, p, addr_b0_lo) if addr_b0_lo is not None else (-1, 0)
            adhi = _band(after_L25, p, addr_b0_hi) if addr_b0_hi is not None else (-1, 0)
            mvb1 = float(after_L25[0, p, mem_val_b1].item()) if mem_val_b1 is not None else 0
            print(
                f"  p={p:3d}  {' '.join(mk)}  OP_LI={li_o:.2f}  LI_RELAY={relay:.2f}  "
                f"ADDR_B0_LO={adlo[0]:2d}/{adlo[1]:.2f}  ADDR_B0_HI={adhi[0]:2d}/{adhi[1]:.2f}  MEM_VAL_B1={mvb1:.2f}"
            )

    # Now: enumerate MEM rows (K candidates for L15 lookup). What does
    # the LI AX row "see" if it picks each MEM row?
    print("\n== MEM row K-side signatures (candidate sources for L15 LI lookup) ==")
    after_L26 = available.get("after_L26")
    if after_L26 is None:
        after_L26 = after_L25
    after_L27 = available.get("after_L27")
    if after_L27 is None:
        after_L27 = after_L26
    for m_pos in mem_rows:
        for d in range(0, 8):
            p = m_pos + d
            if p >= seq:
                continue
            ms = float(after_L27[0, p, mem_store].item()) if mem_store is not None else 0
            mvb1 = float(after_L27[0, p, mem_val_b1].item()) if mem_val_b1 is not None else 0
            mvb2 = float(after_L27[0, p, mem_val_b2].item()) if mem_val_b2 is not None else 0
            adlo = _band(after_L27, p, addr_b0_lo) if addr_b0_lo is not None else (-1, 0)
            adhi = _band(after_L27, p, addr_b0_hi) if addr_b0_hi is not None else (-1, 0)
            olo = _band(after_L27, p, out_lo)
            ohi = _band(after_L27, p, out_hi)
            print(
                f"  m@{m_pos} d={d} p={p:3d}  MEM_STORE={ms:.2f}  "
                f"ADDR_B0_LO={adlo[0]:2d}/{adlo[1]:.2f} ADDR_B0_HI={adhi[0]:2d}/{adhi[1]:.2f}  "
                f"MVB1={mvb1:.2f} MVB2={mvb2:.2f}  OUT_LO={olo[0]:2d}/{olo[1]:.2f} OUT_HI={ohi[0]:2d}/{ohi[1]:.2f}"
            )

    # Token sequence near end
    print("\n== Last 20 token positions and markers ==")
    for p in range(max(0, seq-25), seq):
        m_ax = float(after_L25[0, p, mark_ax].item())
        m_mem = float(after_L25[0, p, mark_mem].item())
        m_sp = float(after_L25[0, p, mark_sp].item())
        m_s0 = float(after_L25[0, p, mark_stack0].item())
        op_li_v = float(after_L25[0, p, op_li].item())
        op_si_v = float(after_L25[0, p, op_si].item())
        op_imm_v = float(after_L25[0, p, op_imm].item())
        bi_now = [float(after_L25[0, p, bi[j]].item()) for j in range(4)]
        markers = []
        if m_ax > 0.5: markers.append("AX")
        if m_mem > 0.5: markers.append("MEM")
        if m_sp > 0.5: markers.append("SP")
        if m_s0 > 0.5: markers.append("S0")
        if op_li_v > 0.5: markers.append("OP_LI")
        if op_si_v > 0.5: markers.append("OP_SI")
        if op_imm_v > 0.5: markers.append("OP_IMM")
        print(f"  p={p:3d}  {','.join(markers) or '-'}  BI={bi_now}")

    # Detailed look at p=193 OUTPUT_LO[0..15] across all layers
    print("\n== p=193 (MEM@188 val byte 1) OUTPUT_LO[0..15] across layers ==")
    for ln in sorted(available.keys(), key=lambda s: int(s.split("L")[1])):
        arr = available[ln]
        if arr.shape[1] <= 193:
            continue
        vals = arr[0, 193, out_lo:out_lo+16].tolist()
        s = " ".join(f"{v:5.2f}" for v in vals)
        print(f"  {ln}: [{s}]")

    # Also: STACK0_BYTE_VAL_1_LO at p=193 across layers — sanity check
    print("\n== p=193 STACK0_BYTE_VAL_1_LO[0..15] across layers ==")
    sb1_lo_dim = dp.get("STACK0_BYTE_VAL_1_LO")
    if sb1_lo_dim is not None:
        for ln in sorted(available.keys(), key=lambda s: int(s.split("L")[1])):
            arr = available[ln]
            if arr.shape[1] <= 193:
                continue
            vals = arr[0, 193, sb1_lo_dim:sb1_lo_dim+16].tolist()
            s = " ".join(f"{v:5.2f}" for v in vals)
            print(f"  {ln}: [{s}]")

    # Layer-by-layer evolution at MEM@188 byte rows (the SI step MEM frame).
    print("\n== MEM@188 OUTPUT evolution by layer (SI MEM frame) ==")
    for d in range(4, 8):
        p = 188 + d
        if p >= seq:
            continue
        print(f"\n  p={p} (MEM@188 d={d}, val byte {d-4})")
        for ln in sorted(available.keys(), key=lambda s: int(s.split("L")[1])):
            arr = available[ln]
            if arr.shape[1] <= p:
                continue
            olo = _band(arr, p, out_lo)
            ohi = _band(arr, p, out_hi)
            print(f"    {ln}: OUT_LO={olo[0]:2d}/{olo[1]:6.2f}  OUT_HI={ohi[0]:2d}/{ohi[1]:6.2f}")

    # Final logits at the actual exit step. Check what EXIT step's PC and
    # AX look like.
    print("\n== Final-step AX bytes (from last available layer) ==")
    last_ln = sorted(available.keys(), key=lambda s: int(s.split("L")[1]))[-1]
    last = available[last_ln]
    # Use last AX block which represents the EXIT step's AX
    if ax_rows:
        last_ax = max(ax_rows)
        # Find continuous AX block starting at last_ax going backward
        block_start = last_ax
        while block_start - 1 in ax_rows or (block_start - 1 > 0 and float(last[0, block_start - 1, mark_ax].item()) > 0.5):
            block_start -= 1
            if block_start <= 0:
                break
        print(f"Last AX block start={block_start}, dumping {last_ln}:")
        for d in range(0, 5):
            p = block_start + d
            if p >= seq:
                continue
            rlo = _band(last, p, reg_ax_lo) if reg_ax_lo is not None else (-1, 0)
            rhi = _band(last, p, reg_ax_hi) if reg_ax_hi is not None else (-1, 0)
            olo = _band(last, p, out_lo)
            ohi = _band(last, p, out_hi)
            bi_now = [float(last[0, p, bi[j]].item()) for j in range(4)]
            print(
                f"  p={p:3d} BI=[{bi_now[0]:3.0f},{bi_now[1]:3.0f},{bi_now[2]:3.0f},{bi_now[3]:3.0f}]  "
                f"AX_LO={rlo[0]:2d}/{rlo[1]:5.2f} AX_HI={rhi[0]:2d}/{rhi[1]:5.2f}  "
                f"OUT_LO={olo[0]:2d}/{olo[1]:5.2f} OUT_HI={ohi[0]:2d}/{ohi[1]:5.2f}"
            )


if __name__ == "__main__":
    main()
