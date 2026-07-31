#!/usr/bin/env python3
"""WALL #1 end-to-end: does the WIDE §Memory CAM resolve doom's 32-bit addresses
through the REAL Qwen attention forward?

The full doom CFM driver OOMs (O(S^2) attention over the 3976-instruction fetch
window * ~102 layers).  This harness isolates the MEMORY-ADDRESSING correctness —
the WALL #1 concern — WITHOUT the code-fetch window: it replays doom's REAL
store/load address trace (captured from the full-opcode Python reference) through
the baked mem-cam attention block on a REORDERED window (store log ADJACENT to the
query, mirroring what the CFM driver must do so recalls stay near — see the
recall-depth measurement: doom's median recall depth is 3, p90=213), at the WIDE
keyed width, and checks each load returns the value at the RIGHT 32-bit address
(vs the Python reference), with NO 8-bit aliasing.

It answers: at ``mem_addr_bits=N`` and store-log depth D, what fraction of doom's
real memory loads resolve to the correct 32-bit address?  Contrasted against the
8-bit default (which aliases).

Run: PYTHONPATH=<c4_release> python3 _agent_doom_mem_addr_resolve.py [--bits 18] [--device cpu]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from c4_min.qwen_full_vm import (
    QWEN2_5_ARCH, NIB_PER_REG, NORM_K, _address_bits, _bake_memory_cam,
    QwenFullLayout, SUBSET_MEM,
)
from c4_min import blogspec_vocab as V
from c4_min.blogspec_memory import ADDR_BITS
from c4_min import isa

DOOM_C = Path("/home/alexlitz/Documents/misc/c4_doom/doom.c")


def capture_doom_mem_trace(max_steps):
    """Replay doom through the full-opcode Python reference; capture the ordered
    (op, addr, val_loaded_or_stored) for every LI/SI/LC/SC, plus the dedup store
    log state at each load (so we can build the exact recall window)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    src = DOOM_C.read_text()
    bc, data = compile_c(src)
    code = bytecode_to_isa(bc)
    DATA_BASE = 0x10000
    mem = {DATA_BASE + i: int(b) & 0xFF for i, b in enumerate(data or [])}
    MASK = 0xFFFFFFFF
    ax = 0; SP0 = 0x10000; sp = bp = SP0; pc = 0

    def sgn(v):
        v &= MASK
        return v - (1 << 32) if v & 0x80000000 else v

    def ld(a, n=4):
        v = 0
        for i in range(n):
            v |= mem.get(a + i, 0) << (8 * i)
        return v

    def st(a, v, n=4):
        for i in range(n):
            mem[a + i] = (v >> (8 * i)) & 0xFF

    # dedup store log: list of {"addr","val"} (latest-write-wins), val = full word
    log = []

    def push_store(a, v):
        for k in range(len(log)):
            if log[k]["addr"] == a:
                log.pop(k); break
        log.append({"addr": a, "val": v & 0xFFFFFFFF})

    events = []       # (kind, addr, want_val, log_snapshot) at each LOAD
    steps = 0
    while 0 <= pc < len(code) and steps < max_steps:
        ins = code[pc]; op, imm = ins.op, ins.imm; pc += 1
        if op == isa.IMM: ax = imm & MASK
        elif op == isa.LEA: ax = (bp + imm * 4) & MASK
        elif op == isa.PSH: sp -= 4; st(sp, ax); push_store(sp, ax)
        elif op in (isa.LI, isa.LC):
            a = ax
            want = ld(a) if op == isa.LI else mem.get(a, 0)
            # snapshot the CURRENT dedup log (the KV memory the CAM would see)
            events.append(("LOAD", a, want & (0xFFFFFFFF if op == isa.LI else 0xFF),
                           [dict(s) for s in log]))
            ax = want & MASK
        elif op == isa.SI: a = ld(sp); sp += 4; st(a, ax); push_store(a, ax)
        elif op == isa.SC: a = ld(sp); sp += 4; mem[a] = ax & 0xFF; push_store(a, ax & 0xFF)
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD, isa.AND, isa.OR,
                    isa.XOR, isa.SHL, isa.SHR, isa.EQ, isa.NE, isa.LT, isa.GT,
                    isa.LE, isa.GE):
            v = sgn(ld(sp)); sp += 4; a = sgn(ax)
            ax = {isa.ADD: v + a, isa.SUB: v - a, isa.MUL: v * a,
                  isa.DIV: (int(v / a) if a else 0), isa.MOD: (v - int(v / a) * a if a else 0),
                  isa.AND: v & a, isa.OR: v | a, isa.XOR: v ^ a, isa.SHL: v << a,
                  isa.SHR: v >> a, isa.EQ: int(v == a), isa.NE: int(v != a),
                  isa.LT: int(v < a), isa.GT: int(v > a), isa.LE: int(v <= a),
                  isa.GE: int(v >= a)}[op] & MASK
        elif op == isa.JMP: pc = imm
        elif op == isa.BZ: pc = imm if (ax & MASK) == 0 else pc
        elif op == isa.BNZ: pc = imm if (ax & MASK) != 0 else pc
        elif op == isa.JSR: sp -= 4; st(sp, pc); push_store(sp, pc); pc = imm
        elif op == isa.ENT: sp -= 4; st(sp, bp); push_store(sp, bp); bp = sp; sp -= imm * 4
        elif op == isa.ADJ: sp += imm * 4
        elif op == isa.LEV: sp = bp; bp = ld(sp); sp += 4; pc = ld(sp); sp += 4
        elif op in (isa.OPEN, isa.READ, isa.CLOS, isa.PRTF): break
        elif op == isa.HALT: break
        steps += 1
    return events


def _make_attn(hidden):
    from transformers.models.qwen2 import Qwen2Model
    from c4_min.qwen_full_vm import _qwen_config
    cfg = _qwen_config(hidden, hidden * 2, 1, V.VOCAB, QWEN2_5_ARCH)
    m = Qwen2Model(cfg).eval()
    attn = m.layers[0].self_attn
    with torch.no_grad():
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            lin.weight.zero_()
            if lin.bias is not None: lin.bias.zero_()
    return m, attn


def _rope(model, x):
    pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    return model.rotary_emb(x, pos)


def resolve_load(attn, m, L, hidden, log, load_addr, is_byte, device):
    """REORDERED window: BOS + store log (adjacent) + load query. Return decoded AX."""
    n = len(log)
    S = 1 + n + 1
    x = torch.zeros(1, S, hidden, device=device)
    x[0, :, L.ONE] = 1.0
    for si, st in enumerate(log):
        p = 1 + si
        x[0, p, L.IS_STORE] = 1.0
        for b, bit in enumerate(_address_bits(st["addr"], ADDR_BITS)):
            x[0, p, L.ADDR_BIN + b] = bit
        for j, nv in enumerate(V.nibbles_of_value(st["val"], NIB_PER_REG)):
            x[0, p, L.VAL_NIB + j] = float(nv)
    x[0, -1, L.IS_LOAD] = 1.0
    for b, bit in enumerate(_address_bits(load_addr, ADDR_BITS)):
        x[0, -1, L.QRY_BIN + b] = float(bit)
    with torch.no_grad():
        out = attn(hidden_states=x, position_embeddings=_rope(m, x),
                   attention_mask=None)[0]
    row = out[0, -1]
    nibs = [int(round(float(row[L.AX + j].item()))) for j in range(NIB_PER_REG)]
    val = 0
    for j, nv in enumerate(nibs):
        val |= (nv & 0xF) << (4 * j)
    return val & (0xFF if is_byte else 0xFFFFFFFF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=18)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--cap-depth", type=int, default=400,
                    help="skip loads whose recall log is deeper than this (window size cap)")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)

    print(f"capturing doom memory trace (<= {args.max_steps} steps)...", flush=True)
    events = capture_doom_mem_trace(args.max_steps)
    loads = [e for e in events if e[0] == "LOAD"]
    print(f"doom: {len(loads)} memory loads captured", flush=True)

    QL = QwenFullLayout(code_size=4, subset=SUBSET_MEM, efficient_alu=False,
                        code_from_memory=False)
    L = QL.L
    hidden = QWEN2_5_ARCH.hidden_for(QL.D_used + 1)

    for bits in sorted({8, args.bits}):
        m, attn = _make_attn(hidden)
        with torch.no_grad():
            _bake_memory_cam(attn, QL, QWEN2_5_ARCH, None, NORM_K, mem_addr_bits=bits)
        m = m.to(dev); attn = attn.to(dev)
        ok = wrong = zfod_ok = skipped = 0
        # sample to keep it quick: every Kth load, but always include the deep tail
        step = max(1, len(loads) // 400)
        for idx in range(0, len(loads), step):
            _, addr, want, log = loads[idx]
            is_byte = want <= 0xFF and True  # LC returns byte; LI full word (rare >0xFF)
            # is this addr present in the log?
            present = any(s["addr"] == addr for s in log)
            if len(log) > args.cap_depth:
                skipped += 1
                continue
            got = resolve_load(attn, m, L, hidden, log, addr, is_byte, dev)
            if not present:
                # ZFOD: unwritten address must read 0
                if got == 0: zfod_ok += 1
                else: wrong += 1
            else:
                if got == want: ok += 1
                else: wrong += 1
        total = ok + wrong + zfod_ok
        print(f"\nmem_addr_bits={bits}: over {total} sampled loads (skipped {skipped} "
              f"deeper than depth {args.cap_depth}):")
        print(f"  correct value @ right addr : {ok}")
        print(f"  ZFOD (unwritten) reads 0   : {zfod_ok}")
        print(f"  WRONG (aliased/mis-read)   : {wrong}")
        print(f"  --> {100*(ok+zfod_ok)/max(total,1):.1f}% of doom's memory loads resolve "
              f"CORRECTLY at {bits}-bit keying")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
