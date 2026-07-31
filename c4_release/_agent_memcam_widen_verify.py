#!/usr/bin/env python3
"""WALL #1 verify: WIDE §Memory CAM (qwen_full_vm._bake_memory_cam).

Three checks, all cheap (bakes only the ONE mem-cam attention block on a bare
Qwen attn module — no full model, no GPU needed):

  1. DEFAULT (8-bit) weights are byte-IDENTICAL to the historical hardcoded-8
     formula (the corpus/golden-neutral guarantee).
  2. WIDE (18-bit) resolves HIGH 32-bit addresses (0x10000 / 0x10400 / stack
     addrs near 0xffff / addrs that ALIAS under 8-bit) with the CORRECT store
     value and NO aliasing — run through the REAL Qwen attention forward on a
     minimal window, compared to an isa-style reference (right value from the
     right address).
  3. WIDE default(8) still aliases those SAME high addresses (proving the bug
     the widening fixes is real, and that 8-bit is the byte-identical default).

Run: PYTHONPATH=<c4_release> python3 _agent_memcam_widen_verify.py
"""
from __future__ import annotations

import argparse

import torch

from c4_min import qwen_full_vm as Q
from c4_min.qwen_full_vm import (
    QwenFullLayout, QWEN2_5_ARCH, SUBSET_MEM, NIB_PER_REG, NORM_K,
    _address_bits, _bake_memory_cam,
)
from c4_min import blogspec_vocab as V
from c4_min.blogspec_memory import ADDR_BITS


def _bake_mem_cam_OLD8(attn, QL, arch, comp, K):
    """Byte-for-byte the HISTORICAL _bake_memory_cam (n_bits hardcoded min(8,...))."""
    import math
    from c4_min.qwen_full_vm import NIB_PER_REG as NPR
    L = QL.L
    hd = arch.head_dim
    q_w = attn.q_proj.weight; k_w = attn.k_proj.weight
    v_w = attn.v_proj.weight; o_w = attn.o_proj.weight
    half = hd // 2
    n_bits = min(8, ADDR_BITS, half - 3)
    G = 16.0
    for b in range(n_bits):
        lane = half - 1 - b
        q_w[lane, L.QRY_BIN + b] = 2.0 * G
        q_w[lane, L.IS_LOAD] = -G
        k_w[lane, L.ADDR_BIN + b] = 2.0 * G
        k_w[lane, L.IS_STORE] = -G
    bias_lane = half - 1 - n_bits
    B = math.sqrt((n_bits - 0.5)) * G
    q_w[bias_lane, L.IS_LOAD] = -B
    k_w[bias_lane, L.IS_STORE] = B
    gate_lane = half - 1 - n_bits - 1
    P = 60.0
    q_w[gate_lane, L.ONE] = P
    q_w[gate_lane, L.IS_LOAD] = -P
    k_w[gate_lane, L.IS_STORE] = -P
    if getattr(L, "IS_CODE", None) is not None:
        excl_lane = half - 1 - n_bits - 2
        Pc = 60.0
        q_w[excl_lane, L.ONE] = Pc
        k_w[excl_lane, L.IS_CODE] = -Pc
    rec_lane = 3
    MEM_RECENCY = 8.0
    q_w[rec_lane, L.IS_LOAD] = MEM_RECENCY
    k_w[rec_lane, L.IS_STORE] = MEM_RECENCY
    for j in range(NPR):
        v_w[j, L.VAL_NIB + j] = 1.0
        o_w[L.AX + j, j] = 1.0


def _make_attn(hidden):
    """A bare Qwen2 self-attn module sized to `hidden` (1 KV group for the CAM head)."""
    from transformers.models.qwen2 import Qwen2Model
    from c4_min.qwen_full_vm import _qwen_config
    arch = QWEN2_5_ARCH
    cfg = _qwen_config(hidden, hidden * 2, 1, V.VOCAB, arch)
    m = Qwen2Model(cfg).eval()
    attn = m.layers[0].self_attn
    with torch.no_grad():
        attn.q_proj.weight.zero_(); attn.k_proj.weight.zero_()
        attn.v_proj.weight.zero_(); attn.o_proj.weight.zero_()
        if attn.q_proj.bias is not None:
            for lin in (attn.q_proj, attn.k_proj, attn.v_proj):
                if lin.bias is not None: lin.bias.zero_()
    return m, attn


def _layout():
    QL = QwenFullLayout(code_size=4, subset=SUBSET_MEM, efficient_alu=False,
                        code_from_memory=False)
    return QL


def check_default_byte_identical():
    QL = _layout()
    L = QL.L
    hidden = QWEN2_5_ARCH.hidden_for(QL.D_used + 1)
    m1, a1 = _make_attn(hidden)
    m2, a2 = _make_attn(hidden)
    with torch.no_grad():
        _bake_memory_cam(a1, QL, QWEN2_5_ARCH, None, NORM_K, mem_addr_bits=8)
        _bake_mem_cam_OLD8(a2, QL, QWEN2_5_ARCH, None, NORM_K)
    ok = True
    for wn in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w1 = getattr(a1, wn).weight; w2 = getattr(a2, wn).weight
        d = (w1 - w2).abs().max().item()
        ok = ok and (d == 0.0)
        print(f"  {wn}: max|new(8) - old8| = {d}")
    print(f"  DEFAULT(8) byte-identical to historical formula: {'YES' if ok else 'NO'}")
    return ok


def run_mem_cam(mem_addr_bits, stores, load_addr, hidden, QL):
    """Bake the CAM at `mem_addr_bits` and run ONE Qwen attention forward on a
    minimal window: BOS(sink) + one store token per (addr,val) + a load query row.
    Returns the decoded AX value (argmax of the loaded VAL_NIB nibbles)."""
    L = QL.L
    m, attn = _make_attn(hidden)
    with torch.no_grad():
        _bake_memory_cam(attn, QL, QWEN2_5_ARCH, None, NORM_K,
                         mem_addr_bits=mem_addr_bits)

    n = len(stores)
    S = 1 + n + 1                       # BOS + stores + query row
    x = torch.zeros(1, S, hidden)
    x[0, :, L.ONE] = 1.0                # ONE lane on every row (BOS sink included)
    for si, (addr, val) in enumerate(stores):
        p = 1 + si
        x[0, p, L.IS_STORE] = 1.0
        for b, bit in enumerate(_address_bits(addr, ADDR_BITS)):
            x[0, p, L.ADDR_BIN + b] = bit
        for j, nv in enumerate(V.nibbles_of_value(val, NIB_PER_REG)):
            x[0, p, L.VAL_NIB + j] = float(nv)
    # load query on the last row
    x[0, -1, L.IS_LOAD] = 1.0
    for b, bit in enumerate(_address_bits(load_addr, ADDR_BITS)):
        x[0, -1, L.QRY_BIN + b] = float(bit)

    with torch.no_grad():
        out = attn(hidden_states=x,
                   position_embeddings=_rope(m, x),
                   attention_mask=None)[0]
    axrow = out[0, -1]
    # decode the AX nibble band the CAM's O wrote (VAL_NIB -> AX)
    nibs = [int(round(float(axrow[L.AX + j].item()))) for j in range(NIB_PER_REG)]
    val = 0
    for j, nv in enumerate(nibs):
        val |= (nv & 0xF) << (4 * j)
    return val & 0xFFFFFFFF


def _rope(model, x):
    S = x.shape[1]
    pos = torch.arange(S).unsqueeze(0)
    rot = model.rotary_emb
    return rot(x, pos)


def check_high_address(mem_addr_bits):
    """Store several HIGH 32-bit addresses (doom-like) whose low bytes ALIAS, then
    load each and confirm the CORRECT value comes back."""
    QL = _layout()
    hidden = QWEN2_5_ARCH.hidden_for(QL.D_used + 1)
    # addresses that all share the SAME low byte (0x00 / 0x08) -> alias under 8-bit,
    # distinct under >=18-bit: doom data seg + heap + stack pattern.
    stores = [
        (0x10000, 0x11),   # DATA_BASE
        (0x10400, 0x22),   # +0x400 (sintab region) -> low byte 0x00 SAME as 0x10000
        (0x20000, 0x33),   # heap high -> low byte 0x00 SAME
        (0x0FF00, 0x44),   # stack region near 0xffff -> low byte 0x00 SAME
        (0x10008, 0x55),   # low byte 0x08
        (0x20008, 0x66),   # low byte 0x08 SAME as 0x10008 -> aliases under 8-bit
    ]
    results = []
    for addr, want in stores:
        got = run_mem_cam(mem_addr_bits, stores, addr, hidden, QL)
        ok = (got == want)
        results.append((addr, want, got, ok))
    return results


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    torch.manual_seed(0)

    print("=== CHECK 1: DEFAULT(8) byte-identical to historical formula ===")
    ident = check_default_byte_identical()

    print("\n=== CHECK 2: WIDE(18) resolves high 32-bit addresses (no aliasing) ===")
    r18 = check_high_address(18)
    n_ok18 = sum(1 for *_, ok in r18 if ok)
    for addr, want, got, ok in r18:
        print(f"  load {hex(addr):>9}: want 0x{want:02x} got 0x{got:02x}  "
              f"{'OK' if ok else 'ALIASED/WRONG'}")
    print(f"  WIDE(18): {n_ok18}/{len(r18)} high addresses resolve correctly")

    print("\n=== CHECK 3: DEFAULT(8) ALIASES those same high addresses (bug is real) ===")
    r8 = check_high_address(8)
    n_ok8 = sum(1 for *_, ok in r8 if ok)
    for addr, want, got, ok in r8:
        print(f"  load {hex(addr):>9}: want 0x{want:02x} got 0x{got:02x}  "
              f"{'OK' if ok else 'ALIASED/WRONG'}")
    print(f"  DEFAULT(8): {n_ok8}/{len(r8)} resolve (expect << {len(r8)} due to aliasing)")

    print("\n=== SUMMARY ===")
    print(f"  default byte-identical : {'PASS' if ident else 'FAIL'}")
    print(f"  wide(18) resolves high : {'PASS' if n_ok18 == len(r18) else 'FAIL'} "
          f"({n_ok18}/{len(r18)})")
    print(f"  8-bit aliases (bug real): {'PASS' if n_ok8 < len(r8) else 'FAIL'} "
          f"({n_ok8}/{len(r8)} — fewer is the bug)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
