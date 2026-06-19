#!/usr/bin/env python3
"""Inc-3 ROOT A: decompose the ACTUAL ax_byte1_dump_carry head (the cross-step
AX byte-1 deliverer) at the step5 AX[0]-predictor row for var_simple x=990, in
EITHER config. Finds the head by op name (layer13_ax_byte1_dump_carry.head_7) so
it is robust to the physical-index reshuffle between widths.

ROOT A CHARACTERIZATION (2026-06-19, this session — ROOT A is NOT a clean
single-head re-key; it is downstream of a pervasive 30-tok MEM_STORE broadcast):
The byte-1 (0x03) for `return x` is delivered in GOLDEN at block-16/L11 head-1 =
``layer10_byte_passthrough_bake.head_1`` (the L10 AX byte_passthrough head,
_layer10_ax_byte_passthrough_head_spec, l10_ops.py:2085) — NOT this dump-carry
head (which attends step1, irrelevant). That AX byte_passthrough head carries the
prior AX[1] (golden +2.0 to OUTPUT_HI[0], attends step4 AX[1] via the AX-carry
slots 0-5, source H1+AX).

In CAMPAIGN the head's MEM_STORE store-select slots (44-47, K = MEM_STORE*500)
PIN it on the wrong row: in the 30-token frame MEM_STORE is BROADCAST onto ~every
register-byte + PC-marker + MEM-store row (the SI/SC store-bit relay over-fires at
the shorter stride), so the 500-weight store-select overwhelms the 100-weight
AX-carry on every candidate row and the head attends a MEM_STORE row (PC marker /
step3 MEM marker) -> byte-1 -> 0. A K-side MARK_PC/AX/SP/BP veto on slots 44-47
only moves the leak (PC marker -> the MARK_MEM store row, which legitimately
carries MEM_STORE) -> still byte-1=0 (tested + reverted this session). The prior
AX[1] source row ITSELF carries the MEM_STORE leak, so the head cannot cleanly
select ANY row. => ROOT A's real lever is the 30-tok MEM_STORE BROADCAST (route
the head's store-select onto the Inc-2 MEM_STORE_AT_VAL discriminator, OR fix the
store-bit relay's over-broadcast in the 30-token frame, OR build a NEW flag-gated
AX-byte1 carry head immune to MEM_STORE), NOT an in-place veto. The MEM_STORE
broadcast also likely gates the step-9/10/11 var_three/var_update/nested clusters
the ROOT-B advance exposed. Multi-session; do NOT force a single-head patch.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_axb1_carryhead.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_axb1_carryhead.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token, _step_offset_field  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 990; return x; }"


def todense(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    rev = {}
    for n, i in dp.items():
        rev.setdefault(i, n)

    def dname(i):
        if i in rev:
            return rev[i]
        for off in range(1, 17):
            if (i - off) in rev:
                return f"{rev[i-off]}+{off}"
        return f"dim{i}"

    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    off6 = pl + 5 * STEP + 6   # step5 AX[0] predictor (predicts byte-1 token)
    blk_map = p.block_layer_map()
    # The carry head is hosted on the block whose attn owns the
    # ax_byte1_dump_carry allocator. It lands physically near L11/L13; scan all
    # blocks for the head whose op is the dump carry by checking the allocator
    # attribute set during bake.
    target_blk = None
    target_head = None
    for b, m in enumerate(blk_map):
        attn = p.model.blocks[b].attn
        alloc = getattr(attn, "_l13_ax_byte1_dump_carry_head_allocator", None)
        if alloc is not None:
            target_blk = b
            target_head = alloc.heads()[-1].head_idx
            break
    print(f"=== {cfg} STEP={STEP} carry head blk={target_blk} head={target_head} "
          f"off6={off6} ===")
    if target_blk is None:
        print("  carry head allocator not found on any block")
        return
    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], device=p._device)
    x_in = todense(p.model.forward(padded, stop_after_block=target_blk - 1)[0])
    attn = p.model.blocks[target_blk].attn
    H = attn.num_heads
    HD = attn.W_q.shape[0] // H
    Wq = todense(attn.W_q.data)
    Wk = todense(attn.W_k.data)
    S = x_in.shape[0]
    Q = (x_in @ Wq.T).view(S, H, HD)
    K = (x_in @ Wk.T).view(S, H, HD)
    h = target_head
    sc = (K[:, h, :] @ Q[off6, h, :]) * attn.scale
    if attn.alibi_slopes is not None:
        sl = float(attn.alibi_slopes[h])
        pos = torch.arange(S).float()
        sc = sc + sl * (pos - off6).clamp(max=0)
    sc[off6 + 1:] = -1e30
    win = int(sc.argmax())
    print(f"  Q@off6 -> attends pos{win} (step{(win-pl)//STEP} "
          f"{_step_offset_field((win-pl)%STEP)}) score={float(sc[win]):+.1f}")
    # K-score breakdown at the winning row
    qh = Q[off6, h, :]
    wk_head = Wk[h * HD:(h + 1) * HD, :]
    kproj = (qh @ wk_head) * attn.scale
    contribs = x_in[win] * kproj
    top = contribs.abs().topk(10)
    print("   K-score dims at winning row:")
    for idx in top.indices.tolist():
        c = float(contribs[idx])
        if abs(c) < 0.5:
            continue
        print(f"     {dname(idx):24s} x={float(x_in[win,idx]):+.2f} -> {c:+.1f}")
    # Where IS the prev-step (step4) AX[1] row, and its H1 one-hot?
    s4ax1 = pl + 4 * STEP + 7
    print(f"  step4 AX[1] row pos={s4ax1}: "
          f"ADDR_B0_LO+5={float(x_in[s4ax1, dp['ADDR_B0_LO']+5]):+.2f} "
          f"H1band={[round(float(x_in[s4ax1, dp['H1']+k]),2) for k in range(4)]}")


if __name__ == "__main__":
    main()
