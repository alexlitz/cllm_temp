"""Wave 1 A3.6: trace STACK0_BYTE_VAL_1_LO/HI bands through every
physical block at all 5 S0 BI_1 rows.

A3.5 identified that the broadcast lands at S0@80 BI_1 (p=82) with
VAL1_LO=2/3.00 (slot 2, since byte 1 of 0x200 = 0x02), but subsequent
STACK0 frames (S0@114, 148, 183, 218) read 0 after L12.

Dim 602 (LO base) and 618 (HI base) are 16-slot one-hot bands.
Aliased with FORMAT_PTR_LO/HI.

This probe dumps the L1 norm (sum |x|) and argmax/val of each 16-slot
band at each block boundary for each S0 BI_1 row. Identifies blocks
where the band changes (ATTN delta vs FFN delta).
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


def _band_stats(x, p, dim_start, width=16):
    band = x[0, p, dim_start:dim_start + width]
    a = int(torch.argmax(band).item())
    mx = float(band[a].item())
    s = float(band.abs().sum().item())
    return a, mx, s


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
    LO = dp["STACK0_BYTE_VAL_1_LO"]
    HI = dp["STACK0_BYTE_VAL_1_HI"]

    captures = []

    def embed_hook(module, inputs, output):
        captures.append({"embed": output.detach().clone()})

    handles = [model.embed.register_forward_hook(embed_hook)]

    for li in range(min(32, len(model.blocks))):
        block = model.blocks[li]

        def make_pre_hook(idx):
            def fn(module, inputs):
                if captures:
                    captures[-1][f"pre_L{idx}"] = inputs[0].detach().clone()
            return fn

        def make_attn_hook(idx):
            def fn(module, inputs, output):
                if captures:
                    captures[-1][f"attn_out_L{idx}"] = output.detach().clone()
            return fn

        def make_block_hook(idx):
            def fn(module, inputs, output):
                if captures:
                    captures[-1][f"block_out_L{idx}"] = output.detach().clone()
            return fn

        handles.append(block.register_forward_pre_hook(make_pre_hook(li)))
        handles.append(block.attn.register_forward_hook(make_attn_hook(li)))
        handles.append(block.register_forward_hook(make_block_hook(li)))

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
        last = c.get("block_out_L31")
        if last is None:
            continue
        if best is None or last.shape[1] > best["block_out_L31"].shape[1]:
            best = c

    if best is None:
        print("No usable capture.")
        return

    seq = best["block_out_L31"].shape[1]
    print(f"Full-context capture: seq_len={seq}")

    mark_stack0 = dp["MARK_STACK0"]
    bi1 = dp["BYTE_INDEX_1"]
    last_block = best["block_out_L31"]
    stack0_positions = torch.nonzero(
        last_block[0, :, mark_stack0] > 0.5, as_tuple=False
    ).flatten().tolist()

    s0_bi1_targets = []
    for s_pos in stack0_positions:
        for d in range(0, 10):
            p = s_pos + d
            if p >= seq:
                continue
            if float(last_block[0, p, bi1].item()) > 0.5:
                s0_bi1_targets.append((s_pos, p))
                break

    print(f"S0 BI_1 targets (s_pos, p): {s0_bi1_targets}")

    # Targets to compare: working frame S0@80 (broadcast OK) and broken
    # frames S0@114, 148, 183, 218 (broadcast missing).
    for (s0_pos, p) in s0_bi1_targets:
        print(f"\n{'=' * 96}")
        print(f"S0@{s0_pos} BI_1 row @ p={p}")
        print(f"{'=' * 96}")
        # Header: per block, LO band (argmax,val,abssum) and HI band, +
        # attn delta to abssum, ffn delta to abssum.
        print(
            f"  {'blk':<5} | {'LO[a/v/s]':<22} {'HI[a/v/s]':<22} "
            f"| {'aLO_d':>8}{'aHI_d':>8}{'fLO_d':>8}{'fHI_d':>8}"
        )
        prev_lo_sum = 0.0
        prev_hi_sum = 0.0
        for li in range(32):
            pre = best.get(f"pre_L{li}")
            attn_out = best.get(f"attn_out_L{li}")
            blk_out = best.get(f"block_out_L{li}")
            if pre is None or blk_out is None:
                continue
            pre_lo = _band_stats(pre, p, LO)
            pre_hi = _band_stats(pre, p, HI)
            ao_lo = _band_stats(attn_out, p, LO)
            ao_hi = _band_stats(attn_out, p, HI)
            bo_lo = _band_stats(blk_out, p, LO)
            bo_hi = _band_stats(blk_out, p, HI)

            # ATTN contribution to band L1: attn_out band's L1 (signed).
            # FFN contribution: block_out - pre - attn_out band's L1.
            attn_lo_d = ao_lo[2]
            attn_hi_d = ao_hi[2]
            ffn_lo_d = bo_lo[2] - pre_lo[2] - ao_lo[2]
            ffn_hi_d = bo_hi[2] - pre_hi[2] - ao_hi[2]
            marker = ""
            if abs(attn_lo_d) + abs(attn_hi_d) > 0.05 or abs(ffn_lo_d) + abs(ffn_hi_d) > 0.05:
                marker = "  <<<"
            lo_str = f"{bo_lo[0]:2d}/{bo_lo[1]:5.2f}/{bo_lo[2]:6.2f}"
            hi_str = f"{bo_hi[0]:2d}/{bo_hi[1]:5.2f}/{bo_hi[2]:6.2f}"
            print(
                f"  L{li:<4} | {lo_str:<22} {hi_str:<22} "
                f"| {attn_lo_d:>8.2f}{attn_hi_d:>8.2f}{ffn_lo_d:>8.2f}{ffn_hi_d:>8.2f}{marker}"
            )


if __name__ == "__main__":
    main()
