"""L8 sp_gather STACK0_BYTE1/2/3 audit (2026-06-07).

Probe whether `layer8_sp_gather_bake` broadcasts STACK0 byte 1/2/3
values to the rows where L14 `mem_generation` expects to read them.

Per `docs/MEMORY_L14_FIX_ATTEMPT_2026_06_07.md`:
  - L14 heads 1/2/3 SI source path attends via BP-relative thresholds
    (L1H4+BP_I etc.), which select the STACK0 byte 1/2/3 rows.
  - At those rows, L14 reads CLEAN_EMBED_LO/HI to copy out the byte
    value.
  - If L8 sp_gather doesn't put the right byte values in
    CLEAN_EMBED_LO/HI at those rows (or alternatively into
    STACK0_BYTE1/2/3 dims that L14 could read instead), then L14
    emits garbage for addr bytes 1-3.

This script runs `test_si_li_roundtrip` programmatically and dumps
the post-L8 residuals at the STACK0-byte rows.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import torch

# Ensure the c4_release package is importable from the worktree root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)  # .../c4_release
sys.path.insert(0, REPO_ROOT)

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

# --- bytecode for test_si_li_roundtrip ---
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
    """Mirror the make_bytecode fixture: list of ints op|(imm<<8)."""
    out = []
    for item in prog:
        if isinstance(item, tuple):
            opcode, imm = item
            out.append(opcode | (imm << 8))
        else:
            out.append(item)
    return out


def main():
    # Build with print-suppress
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    if False:
        pass  # placeholder so the runner stays at base indent
    _do_probe(runner)


def _do_probe(runner):
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []

    model = runner.model
    dim_positions = model.embed._dim_positions

    # Dims of interest.
    name_to_dim = {}
    interesting = [
        "MARK_STACK0", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_AX",
        "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
        "OUTPUT_LO", "OUTPUT_HI",
        "ADDR_B0_LO", "ADDR_B0_HI",
        "ADDR_B1_LO", "ADDR_B1_HI",
        "ADDR_B2_LO", "ADDR_B2_HI",
        "MEM_STORE", "MEM_ADDR_SRC",
        "EMBED_LO", "EMBED_HI",
    ]
    for n in interesting:
        if n in dim_positions:
            name_to_dim[n] = dim_positions[n]

    print("Dims found:")
    for n, p in sorted(name_to_dim.items(), key=lambda x: x[1]):
        print(f"  {n:18s} -> {p}")
    print()

    captures = []

    def embed_hook(module, inputs, output):
        captures.append({"token_ids": inputs[0].detach().clone()})

    def block_hook(name):
        def fn(module, inputs, output):
            if captures:
                captures[-1][name] = output.detach().clone()
        return fn

    handles = []
    handles.append(model.embed.register_forward_hook(embed_hook))
    for li in (2, 5, 7, 8, 9, 10, 11, 12, 13, 14):
        if li < len(model.blocks):
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

    print(f"\nResult: {result}")
    print(f"Number of forward captures: {len(captures)}")

    # Find a "full-context" capture (last forward, biggest token sequence).
    best = None
    for c in captures:
        if "after_L8" not in c:
            continue
        if best is None or c["after_L8"].shape[1] > best["after_L8"].shape[1]:
            best = c
    if best is None:
        print("\nNo full-context capture with after_L8 available.")
        return

    after_L8 = best["after_L8"]
    after_L2 = best.get("after_L2")
    after_L13 = best.get("after_L13")
    after_L14 = best.get("after_L14")
    tok = best["token_ids"]
    seq_len = after_L8.shape[1]
    print(f"\nUsing full-context capture: seq_len={seq_len}, tok shape={tok.shape}")

    # Locate marker positions by scanning MARK_* dim.
    mark_stack0_dim = name_to_dim["MARK_STACK0"]
    mark_mem_dim = name_to_dim["MARK_MEM"]
    mark_bp_dim = name_to_dim["MARK_BP"]
    mark_sp_dim = name_to_dim["MARK_SP"]

    def find_markers(after, dim, thresh=0.5):
        col = after[0, :, dim]
        return torch.nonzero(col > thresh, as_tuple=False).flatten().tolist()

    stack0_positions = find_markers(after_L8, mark_stack0_dim)
    mem_positions = find_markers(after_L8, mark_mem_dim)
    bp_positions = find_markers(after_L8, mark_bp_dim)
    sp_positions = find_markers(after_L8, mark_sp_dim)
    print(f"\nMARK_STACK0 positions: {stack0_positions}")
    print(f"MARK_MEM positions:    {mem_positions}")
    print(f"MARK_BP positions:     {bp_positions}")
    print(f"MARK_SP positions:     {sp_positions}")

    # For each STACK0 marker, dump bytes d=0..9 (the 10 rows of that
    # STACK0 frame: marker + 9 byte rows).
    def dump_band(after, pos, dim_lo, dim_hi):
        lo = after[0, pos, dim_lo : dim_lo + 16]
        hi = after[0, pos, dim_hi : dim_hi + 16]
        lo_idx = int(torch.argmax(lo).item())
        hi_idx = int(torch.argmax(hi).item())
        return lo_idx, float(lo[lo_idx].item()), hi_idx, float(hi[hi_idx].item())

    print("\n--- After L8: STACK0 frame residual probe ---")
    for s_pos in stack0_positions:
        print(f"\nSTACK0 marker @ position {s_pos}:")
        print(
            f"  {'d':>3}  pos  STACK0_B0  STACK0_B1  STACK0_B2  STACK0_B3  "
            f"BYTE_IDX_0  BYTE_IDX_1  BYTE_IDX_2  BYTE_IDX_3  "
            f"CLEAN_LO  CLEAN_HI  OUTPUT_LO OUTPUT_HI"
        )
        for d in range(0, 10):
            p = s_pos + d
            if p >= seq_len:
                continue
            sb0 = after_L8[0, p, name_to_dim["STACK0_BYTE0"]].item()
            sb1 = after_L8[0, p, name_to_dim["STACK0_BYTE1"]].item()
            sb2 = after_L8[0, p, name_to_dim["STACK0_BYTE2"]].item()
            sb3 = after_L8[0, p, name_to_dim["STACK0_BYTE3"]].item()
            bi0 = after_L8[0, p, name_to_dim["BYTE_INDEX_0"]].item()
            bi1 = after_L8[0, p, name_to_dim["BYTE_INDEX_1"]].item()
            bi2 = after_L8[0, p, name_to_dim["BYTE_INDEX_2"]].item()
            bi3 = after_L8[0, p, name_to_dim["BYTE_INDEX_3"]].item()
            cl_lo_idx, cl_lo_v, cl_hi_idx, cl_hi_v = dump_band(
                after_L8, p, name_to_dim["CLEAN_EMBED_LO"], name_to_dim["CLEAN_EMBED_HI"]
            )
            op_lo_idx, op_lo_v, op_hi_idx, op_hi_v = dump_band(
                after_L8, p, name_to_dim["OUTPUT_LO"], name_to_dim["OUTPUT_HI"]
            )
            print(
                f"  {d:>3}  {p:3d}  "
                f"{sb0:9.3f}  {sb1:9.3f}  {sb2:9.3f}  {sb3:9.3f}  "
                f"{bi0:10.3f}  {bi1:10.3f}  {bi2:10.3f}  {bi3:10.3f}  "
                f"{cl_lo_idx:2d}/{cl_lo_v:5.2f}  {cl_hi_idx:2d}/{cl_hi_v:5.2f}  "
                f"{op_lo_idx:2d}/{op_lo_v:5.2f}  {op_hi_idx:2d}/{op_hi_v:5.2f}"
            )

    # Also dump the same probe with after_L2 (to see L2 FFN's writes) and
    # after_L13 (to see if anything moves them later).
    if after_L2 is not None:
        print("\n--- After L2 (for comparison) ---")
        for s_pos in stack0_positions[:1]:
            print(f"\nSTACK0 marker @ {s_pos}:")
            for d in range(0, 10):
                p = s_pos + d
                if p >= seq_len:
                    continue
                sb1 = after_L2[0, p, name_to_dim["STACK0_BYTE1"]].item()
                sb2 = after_L2[0, p, name_to_dim["STACK0_BYTE2"]].item()
                sb3 = after_L2[0, p, name_to_dim["STACK0_BYTE3"]].item()
                bi1 = after_L2[0, p, name_to_dim["BYTE_INDEX_1"]].item()
                print(
                    f"  d={d:2d} pos={p}  STACK0_B1={sb1:.3f}  "
                    f"STACK0_B2={sb2:.3f}  STACK0_B3={sb3:.3f}  "
                    f"BYTE_IDX_1={bi1:.3f}"
                )

    # Decode what byte the CLEAN_EMBED says is at the STACK0 byte 1/2/3 rows.
    # In the test program, after PSH at step 2, the SP value (token 261)
    # is pushed onto STACK0 holding the actual pushed value's bytes. For
    # IMM 0x200 + PSH, STACK0 holds bytes [0x00, 0x02, 0x00, 0x00].
    # Token IDs for byte N are 0..255 (raw bytes).
    print("\n--- INTERPRETATION ---")
    print("Expected: at STACK0 marker + d=6,7,8,9 (byte 0,1,2,3 rows),")
    print("  CLEAN_EMBED should encode bytes 0x00, 0x02, 0x00, 0x00 of pushed value 0x200")
    print("  STACK0 marker @ position p means byte 0 is at p+6 (per L2 FFN BYTE_INDEX_0 row).")
    print("  argmax(CLEAN_EMBED_LO) at byte row J == low nibble of byte J")
    print("  argmax(CLEAN_EMBED_HI) at byte row J == high nibble of byte J")

    # Compare ADDR_B0/B1/B2 at STACK0 marker rows across L7/L8/L13/L14.
    def addr_at(after, pos):
        out = {}
        for k_name in ("ADDR_B0_LO", "ADDR_B0_HI",
                       "ADDR_B1_LO", "ADDR_B1_HI",
                       "ADDR_B2_LO", "ADDR_B2_HI"):
            band = after[0, pos, name_to_dim[k_name] : name_to_dim[k_name] + 16]
            idx = int(torch.argmax(band).item())
            v = float(band[idx].item())
            out[k_name] = (idx, v)
        return out

    print("\n--- Cross-layer ADDR_B at MARK_STACK0 rows ---")
    after_L7 = best.get("after_L7")
    for s_pos in stack0_positions:
        print(f"\n  STACK0 @ {s_pos}:")
        for lname, layer_cap in (("L7", after_L7), ("L8", after_L8),
                                  ("L13", after_L13), ("L14", after_L14)):
            if layer_cap is None:
                continue
            v = addr_at(layer_cap, s_pos)
            print(
                f"    {lname}: B0=[{v['ADDR_B0_LO'][0]:2d}/{v['ADDR_B0_LO'][1]:5.2f}, "
                f"{v['ADDR_B0_HI'][0]:2d}/{v['ADDR_B0_HI'][1]:5.2f}]  "
                f"B1=[{v['ADDR_B1_LO'][0]:2d}/{v['ADDR_B1_LO'][1]:5.2f}, "
                f"{v['ADDR_B1_HI'][0]:2d}/{v['ADDR_B1_HI'][1]:5.2f}]  "
                f"B2=[{v['ADDR_B2_LO'][0]:2d}/{v['ADDR_B2_LO'][1]:5.2f}, "
                f"{v['ADDR_B2_HI'][0]:2d}/{v['ADDR_B2_HI'][1]:5.2f}]"
            )

    # Print ADDR_B0/B1/B2 at STACK0 marker positions to see L8 sp_gather
    # output (it should fire at MARK_STACK0).
    print("\n--- After L8: ADDR_B0/B1/B2 at STACK0 marker rows ---")
    for s_pos in stack0_positions:
        b0_lo_idx, b0_lo_v, b0_hi_idx, b0_hi_v = dump_band(
            after_L8, s_pos, name_to_dim["ADDR_B0_LO"], name_to_dim["ADDR_B0_HI"]
        )
        b1_lo_idx, b1_lo_v, b1_hi_idx, b1_hi_v = dump_band(
            after_L8, s_pos, name_to_dim["ADDR_B1_LO"], name_to_dim["ADDR_B1_HI"]
        )
        b2_lo_idx, b2_lo_v, b2_hi_idx, b2_hi_v = dump_band(
            after_L8, s_pos, name_to_dim["ADDR_B2_LO"], name_to_dim["ADDR_B2_HI"]
        )
        print(
            f"  STACK0@{s_pos}  "
            f"B0=[{b0_lo_idx:2d}/{b0_lo_v:5.2f},{b0_hi_idx:2d}/{b0_hi_v:5.2f}] "
            f"B1=[{b1_lo_idx:2d}/{b1_lo_v:5.2f},{b1_hi_idx:2d}/{b1_hi_v:5.2f}] "
            f"B2=[{b2_lo_idx:2d}/{b2_lo_v:5.2f},{b2_hi_idx:2d}/{b2_hi_v:5.2f}]"
        )

    # Also dump ADDR_B at MARK_AX positions (L7 memory_heads target).
    print("\n--- After L8: ADDR_B at MARK_AX positions (L7) ---")
    ax_positions = find_markers(after_L8, name_to_dim["MARK_AX"])
    for ax_pos in ax_positions[-3:]:
        b0_lo_idx, b0_lo_v, b0_hi_idx, b0_hi_v = dump_band(
            after_L8, ax_pos, name_to_dim["ADDR_B0_LO"], name_to_dim["ADDR_B0_HI"]
        )
        b1_lo_idx, b1_lo_v, b1_hi_idx, b1_hi_v = dump_band(
            after_L8, ax_pos, name_to_dim["ADDR_B1_LO"], name_to_dim["ADDR_B1_HI"]
        )
        b2_lo_idx, b2_lo_v, b2_hi_idx, b2_hi_v = dump_band(
            after_L8, ax_pos, name_to_dim["ADDR_B2_LO"], name_to_dim["ADDR_B2_HI"]
        )
        print(
            f"  AX@{ax_pos}  "
            f"B0=[{b0_lo_idx:2d}/{b0_lo_v:5.2f},{b0_hi_idx:2d}/{b0_hi_v:5.2f}] "
            f"B1=[{b1_lo_idx:2d}/{b1_lo_v:5.2f},{b1_hi_idx:2d}/{b1_hi_v:5.2f}] "
            f"B2=[{b2_lo_idx:2d}/{b2_lo_v:5.2f},{b2_hi_idx:2d}/{b2_hi_v:5.2f}]"
        )

    # Critical: dump CLEAN_EMBED at the byte rows AND at the SP positions
    # (where L8 sp_gather K-side fires).
    print("\n--- After L8: CLEAN_EMBED at SP byte rows ---")
    for sp_pos in sp_positions:
        print(f"\n  MARK_SP @ {sp_pos}: SP bytes at d=1..4")
        for d in range(0, 5):
            p = sp_pos + d
            if p >= seq_len:
                continue
            cl_lo_idx, cl_lo_v, cl_hi_idx, cl_hi_v = dump_band(
                after_L8, p, name_to_dim["CLEAN_EMBED_LO"], name_to_dim["CLEAN_EMBED_HI"]
            )
            bi0 = after_L8[0, p, name_to_dim["BYTE_INDEX_0"]].item()
            bi1 = after_L8[0, p, name_to_dim["BYTE_INDEX_1"]].item()
            bi2 = after_L8[0, p, name_to_dim["BYTE_INDEX_2"]].item()
            print(
                f"    d={d:2d} pos={p}  CLEAN_LO={cl_lo_idx:2d}/{cl_lo_v:5.2f}  "
                f"CLEAN_HI={cl_hi_idx:2d}/{cl_hi_v:5.2f}  "
                f"BI0={bi0:.2f} BI1={bi1:.2f} BI2={bi2:.2f}"
            )

    # Print the L8 sp_gather output (ADDR_B0/B1/B2) at the MEM addr-byte
    # positions to confirm sp_gather actually broadcasts SP-derived bytes
    # at all (only relevant to ADDR_B*).
    print("\n--- After L8: ADDR_B0/B1/B2 at MEM addr-byte positions ---")
    for m_pos in mem_positions:
        print(f"\nMEM marker @ {m_pos}:")
        for d in range(0, 8):
            p = m_pos + d
            if p >= seq_len:
                continue
            b0_lo_idx, b0_lo_v, b0_hi_idx, b0_hi_v = dump_band(
                after_L8, p, name_to_dim["ADDR_B0_LO"], name_to_dim["ADDR_B0_HI"]
            )
            b1_lo_idx, b1_lo_v, b1_hi_idx, b1_hi_v = dump_band(
                after_L8, p, name_to_dim["ADDR_B1_LO"], name_to_dim["ADDR_B1_HI"]
            )
            b2_lo_idx, b2_lo_v, b2_hi_idx, b2_hi_v = dump_band(
                after_L8, p, name_to_dim["ADDR_B2_LO"], name_to_dim["ADDR_B2_HI"]
            )
            print(
                f"  d={d:2d} pos={p}  "
                f"B0=[{b0_lo_idx:2d}/{b0_lo_v:5.2f},{b0_hi_idx:2d}/{b0_hi_v:5.2f}] "
                f"B1=[{b1_lo_idx:2d}/{b1_lo_v:5.2f},{b1_hi_idx:2d}/{b1_hi_v:5.2f}] "
                f"B2=[{b2_lo_idx:2d}/{b2_lo_v:5.2f},{b2_hi_idx:2d}/{b2_hi_v:5.2f}]"
            )


if __name__ == "__main__":
    main()
