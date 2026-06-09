"""Wave 1 A3 broadcast head diagnostic probe (2026-06-09).

Companion to ``docs/A3_BROADCAST_DIAGNOSTIC_2026_06_07.md``.

After commit fd109b86 added the L10 ``layer10_psh_ax_broadcast_bake``
heads (slots 8/9/10) and commit 6c5be295 migrated L14 mem_generation
heads 1/2/3 to read ``STACK0_BYTE_VAL_{h}_{LO,HI}``, memory smoke is
still 0/5. The A3.2 caveat in the L14 read commit says:

    the deeper SI failure mode remains... likely follow-ups:
    verify L10 broadcast head actually populates STACK0_BYTE_VAL_h.

This probe runs ``test_si_li_roundtrip`` (the same program the L8 audit
used) and dumps STACK0_BYTE_VAL_{h}_LO/HI residuals at MARK_STACK0
byte rows on the after-L10 capture (where the broadcast heads write)
and the after-L13 / after-L14 captures (the read side). The argmax
nibble pair at the BYTE_INDEX_h row reveals whether the broadcast head
actually populated the new dim family with AX = 0x200's bytes
(0x02, 0x00, 0x00 at h=1/2/3) on PSH.

Two outcomes:

    (A) broadcast values present at MARK_STACK0 + BYTE_INDEX_h row after
        L10. -> bug is L14 K-side attention; head's Q gate selects the
        wrong row OR Q/K rotation drops the broadcast.

    (B) broadcast values zero/empty after L10. -> bug is L10 head spec;
        Q gate may not fire on the PSH step or K complement at slot 33
        is mis-tuned (softmax1-cancelling).
"""

import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)  # .../c4_release
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


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    _do_probe(runner)


def _dump_band(after, pos, dim_start):
    band = after[0, pos, dim_start:dim_start + 16]
    idx = int(torch.argmax(band).item())
    v = float(band[idx].item())
    return idx, v


def _do_probe(runner):
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []

    model = runner.model
    dp = model.embed._dim_positions

    needed = [
        "MARK_STACK0", "MARK_AX", "MARK_SP",
        "OP_PSH", "OP_SI",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
        "STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI",
        "STACK0_BYTE_VAL_2_LO", "STACK0_BYTE_VAL_2_HI",
        "STACK0_BYTE_VAL_3_LO", "STACK0_BYTE_VAL_3_HI",
        "OUTPUT_LO", "OUTPUT_HI",
    ]
    name_to_dim = {n: dp[n] for n in needed if n in dp}
    missing = [n for n in needed if n not in dp]
    if missing:
        print(f"MISSING dims: {missing}")
    else:
        print("All probe dims present.")
    print()

    captures = []

    def embed_hook(module, inputs, output):
        captures.append({"token_ids": inputs[0].detach().clone()})

    def block_hook(name):
        def fn(module, inputs, output):
            if captures:
                captures[-1][name] = output.detach().clone()
        return fn

    # After `_expand_wrapper_blocks` (phase 1300) the logical L10/L14
    # bake targets land at physical block indices L12/L27 respectively
    # in the production layout. Hook a wide range so the probe is robust
    # to layout shifts; the cross-layer dump below will tell us where the
    # broadcast actually lands.
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
    print(f"Forward captures: {len(captures)}")

    # The PSH step is the second autoregressive forward (after IMM 0x200).
    # We want the FIRST capture in which MARK_STACK0 appears AND OP_PSH
    # is set on the current step row. Easier: scan captures, look for
    # OP_PSH dim being set at the last row; that's the step where the
    # transformer is emitting the PSH STACK0 frame.
    op_psh_dim = name_to_dim.get("OP_PSH")
    op_si_dim = name_to_dim.get("OP_SI")
    mark_stack0_dim = name_to_dim["MARK_STACK0"]

    PRODUCER_KEY = "after_L12"

    def find_step_captures(predicate):
        out = []
        for i, c in enumerate(captures):
            if PRODUCER_KEY not in c:
                continue
            after = c[PRODUCER_KEY]
            if predicate(after):
                out.append((i, c))
        return out

    def has_op(after, dim):
        col = after[0, :, dim]
        return bool((col > 0.5).any().item())

    print(f"\n== Capture inventory ({PRODUCER_KEY} forward passes) ==")
    for i, c in enumerate(captures):
        if PRODUCER_KEY not in c:
            continue
        after = c[PRODUCER_KEY]
        seq = after.shape[1]
        op_psh_active = has_op(after, op_psh_dim) if op_psh_dim is not None else False
        op_si_active = has_op(after, op_si_dim) if op_si_dim is not None else False
        ms0 = (after[0, :, mark_stack0_dim] > 0.5).sum().item()
        print(
            f"  capture[{i}]: seq_len={seq}  OP_PSH_present={op_psh_active}  "
            f"OP_SI_present={op_si_active}  MARK_STACK0_count={int(ms0)}"
        )

    # Pick the LAST capture: full context has every step's row.
    best = None
    for c in captures:
        if PRODUCER_KEY not in c:
            continue
        if best is None or c[PRODUCER_KEY].shape[1] > best[PRODUCER_KEY].shape[1]:
            best = c
    if best is None:
        print("No usable capture.")
        return

    # The L10 broadcast bake (logical L10) lands at physical block 12
    # after _expand_wrapper_blocks (phase 1300). The L14 mem_generation
    # consumer lands at physical block 27. Use the producer block here.
    after_L10 = best["after_L12"] if "after_L12" in best else best.get("after_L10")
    after_L11 = best.get("after_L13")
    after_L13 = best.get("after_L25")
    after_L14 = best.get("after_L27")
    after_L9 = best.get("after_L11")
    seq = after_L10.shape[1]
    print(
        "\n  (NOTE: 'after_L10/L11/L13/L14' below map to physical blocks "
        "12/13/25/27 in the production layout; logical L10 broadcast lands "
        "at physical block 12 after expand_wrapper_blocks phase 1300.)"
    )
    print(f"\nFull-context capture: seq_len={seq}")

    # Locate STACK0 markers
    stack0_positions = torch.nonzero(
        after_L10[0, :, mark_stack0_dim] > 0.5, as_tuple=False
    ).flatten().tolist()
    print(f"MARK_STACK0 positions: {stack0_positions}")

    # Locate OP_PSH rows (which steps had PSH active)
    if op_psh_dim is not None:
        psh_rows = torch.nonzero(
            after_L10[0, :, op_psh_dim] > 0.5, as_tuple=False
        ).flatten().tolist()
        print(f"OP_PSH active rows: {psh_rows}")

    # === Core probe: per STACK0 frame, dump STACK0_BYTE_VAL_h at the
    #     row where BYTE_INDEX_h fires (d=7,8,9 for h=1,2,3). For each
    #     STACK0 marker we also report whether OP_PSH was active on
    #     that row (so we know the broadcast Q gate should have fired).
    print("\n== After L10: STACK0_BYTE_VAL_{h}_LO/HI residual at STACK0+d rows ==")
    print("  (Expected for PSH of IMM 0x200: byte 1 = 0x02 -> LO=2, HI=0;")
    print("   byte 2 = 0x00 -> LO=0, HI=0; byte 3 = 0x00 -> LO=0, HI=0)")
    for s_pos in stack0_positions:
        op_psh_on_this_row = (
            float(after_L10[0, s_pos, op_psh_dim].item())
            if op_psh_dim is not None else 0.0
        )
        print(f"\n  STACK0 marker @ pos={s_pos}  OP_PSH_at_marker={op_psh_on_this_row:.2f}")
        for d in range(0, 10):
            p = s_pos + d
            if p >= seq:
                continue
            # BYTE_INDEX_h indicator at the row
            bi = [
                float(after_L10[0, p, name_to_dim[f"BYTE_INDEX_{j}"]].item())
                for j in range(4)
            ]
            op_psh_v = (
                float(after_L10[0, p, op_psh_dim].item())
                if op_psh_dim is not None else 0.0
            )
            # New dim family
            v1_lo = _dump_band(after_L10, p, name_to_dim["STACK0_BYTE_VAL_1_LO"])
            v1_hi = _dump_band(after_L10, p, name_to_dim["STACK0_BYTE_VAL_1_HI"])
            v2_lo = _dump_band(after_L10, p, name_to_dim["STACK0_BYTE_VAL_2_LO"])
            v2_hi = _dump_band(after_L10, p, name_to_dim["STACK0_BYTE_VAL_2_HI"])
            v3_lo = _dump_band(after_L10, p, name_to_dim["STACK0_BYTE_VAL_3_LO"])
            v3_hi = _dump_band(after_L10, p, name_to_dim["STACK0_BYTE_VAL_3_HI"])
            cl_lo = _dump_band(after_L10, p, name_to_dim["CLEAN_EMBED_LO"])
            cl_hi = _dump_band(after_L10, p, name_to_dim["CLEAN_EMBED_HI"])
            print(
                f"    d={d:2d} p={p:3d}  "
                f"BI=[{bi[0]:.1f},{bi[1]:.1f},{bi[2]:.1f},{bi[3]:.1f}]  "
                f"OP_PSH={op_psh_v:.1f}  "
                f"V1_LO={v1_lo[0]:2d}/{v1_lo[1]:5.2f} V1_HI={v1_hi[0]:2d}/{v1_hi[1]:5.2f}  "
                f"V2_LO={v2_lo[0]:2d}/{v2_lo[1]:5.2f} V2_HI={v2_hi[0]:2d}/{v2_hi[1]:5.2f}  "
                f"V3_LO={v3_lo[0]:2d}/{v3_lo[1]:5.2f} V3_HI={v3_hi[0]:2d}/{v3_hi[1]:5.2f}  "
                f"CL=[{cl_lo[0]:2d}/{cl_lo[1]:4.1f},{cl_hi[0]:2d}/{cl_hi[1]:4.1f}]"
            )

    # === AX row residual: did the K side of the broadcast head fire at
    #     the AX byte source row?
    ax_dim = name_to_dim["MARK_AX"]
    ax_positions = torch.nonzero(
        after_L10[0, :, ax_dim] > 0.5, as_tuple=False
    ).flatten().tolist()
    print(f"\n== After L10: AX row CLEAN_EMBED (source for broadcast) ==")
    for ax_pos in ax_positions:
        op_psh_v = (
            float(after_L10[0, ax_pos, op_psh_dim].item())
            if op_psh_dim is not None else 0.0
        )
        print(f"\n  MARK_AX @ {ax_pos}  OP_PSH={op_psh_v:.1f}")
        for d in range(0, 6):
            p = ax_pos + d
            if p >= seq:
                continue
            bi = [
                float(after_L10[0, p, name_to_dim[f"BYTE_INDEX_{j}"]].item())
                for j in range(4)
            ]
            cl_lo = _dump_band(after_L10, p, name_to_dim["CLEAN_EMBED_LO"])
            cl_hi = _dump_band(after_L10, p, name_to_dim["CLEAN_EMBED_HI"])
            print(
                f"    d={d:2d} p={p:3d}  "
                f"BI=[{bi[0]:.1f},{bi[1]:.1f},{bi[2]:.1f},{bi[3]:.1f}]  "
                f"CL_LO={cl_lo[0]:2d}/{cl_lo[1]:5.2f} CL_HI={cl_hi[0]:2d}/{cl_hi[1]:5.2f}"
            )

    # === Compare across layers to see if the broadcast lands at L10 and
    #     survives to L13/L14 (where mem_generation reads).
    print("\n== Cross-layer STACK0_BYTE_VAL_1_LO/HI at STACK0+BYTE_INDEX_1 row ==")
    for s_pos in stack0_positions:
        # find the row at this STACK0 frame where BYTE_INDEX_1 fires
        bi1_dim = name_to_dim["BYTE_INDEX_1"]
        target = None
        for d in range(0, 10):
            p = s_pos + d
            if p >= seq:
                continue
            if float(after_L10[0, p, bi1_dim].item()) > 0.5:
                target = p
                break
        if target is None:
            continue
        print(f"\n  STACK0@{s_pos}, BYTE_INDEX_1 row @ p={target}:")
        for lname, cap in (
            ("L9 ", after_L9), ("L10", after_L10), ("L11", after_L11),
            ("L13", after_L13), ("L14", after_L14),
        ):
            if cap is None:
                continue
            v_lo = _dump_band(cap, target, name_to_dim["STACK0_BYTE_VAL_1_LO"])
            v_hi = _dump_band(cap, target, name_to_dim["STACK0_BYTE_VAL_1_HI"])
            cl_lo = _dump_band(cap, target, name_to_dim["CLEAN_EMBED_LO"])
            cl_hi = _dump_band(cap, target, name_to_dim["CLEAN_EMBED_HI"])
            print(
                f"    {lname}: V1_LO={v_lo[0]:2d}/{v_lo[1]:6.2f}  "
                f"V1_HI={v_hi[0]:2d}/{v_hi[1]:6.2f}  "
                f"CL_LO={cl_lo[0]:2d}/{cl_lo[1]:5.2f}  "
                f"CL_HI={cl_hi[0]:2d}/{cl_hi[1]:5.2f}"
            )

    # === L14 OUTPUT_LO/HI at MARK_MEM rows (where mem_generation writes)
    mark_mem_dim = dp.get("MARK_MEM")
    if mark_mem_dim is not None and after_L14 is not None:
        mem_positions = torch.nonzero(
            after_L14[0, :, mark_mem_dim] > 0.5, as_tuple=False
        ).flatten().tolist()
        print(f"\n== After L14: OUTPUT_LO/HI at MEM addr-byte rows ==")
        for m_pos in mem_positions:
            print(f"\n  MARK_MEM @ {m_pos}")
            for d in range(0, 6):
                p = m_pos + d
                if p >= seq:
                    continue
                bi = [
                    float(after_L14[0, p, name_to_dim[f"BYTE_INDEX_{j}"]].item())
                    for j in range(4)
                ]
                op_lo = _dump_band(after_L14, p, name_to_dim["OUTPUT_LO"])
                op_hi = _dump_band(after_L14, p, name_to_dim["OUTPUT_HI"])
                print(
                    f"    d={d:2d} p={p:3d}  "
                    f"BI=[{bi[0]:.1f},{bi[1]:.1f},{bi[2]:.1f},{bi[3]:.1f}]  "
                    f"OUT_LO={op_lo[0]:2d}/{op_lo[1]:5.2f} OUT_HI={op_hi[0]:2d}/{op_hi[1]:5.2f}"
                )


if __name__ == "__main__":
    main()
