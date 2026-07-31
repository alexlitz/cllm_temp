#!/usr/bin/env python3
"""Probe: does the CFM lean forward decode FULL 32-bit values + wide-CAM 32-bit
memory addresses byte-exact?  This is the make-or-break for a 32-bit doom driver:
the stock lean drivers fold to 8 bits, but the underlying model computes 32-bit in
the nibble band.  We build the SUBSET_MULDIV recurrent model with mem_addr_bits=18
and decode AX / LI at 32 bits directly (mask=0xFFFFFFFF), bypassing the 8-bit driver.
"""
from __future__ import annotations
import argparse, warnings, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bits", type=int, default=18)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    from c4_min.nibble_pure_forward_complete import _decode_reg_from_nibbles
    from c4_min.qwen_full_vm import _snap
    from _agent_lean_recurrent import build_lean_recurrent

    dev = torch.device(args.device)
    print(f"[build] SUBSET_MULDIV recurrent CFM, mem_addr_bits={args.bits} ...", flush=True)
    vm = Q.build(code_size=48, subset=Q.SUBSET_MULDIV, recurrent_divmod=True,
                 code_from_memory=True, mem_addr_bits=args.bits)
    lean = build_lean_recurrent(vm, device=str(dev))
    L = lean.QL.L
    print(f"[built] {lean.n_layers}L hidden={lean.hidden_size} cfm={lean.code_from_memory}", flush=True)
    if dev.type=="cuda": print(f"[vram] {torch.cuda.memory_allocated(dev)/1e9:.2f} GB", flush=True)

    # ---- (A) 32-bit ALU: a fixed-point-style term.  1000*1000 = 1_000_000 (needs 20 bits) ----
    from c4_min.nibble_pure_forward import SP_INIT
    def decode32_last_op(prog, op_to_read):
        code = isa.assemble(prog)
        reg_state={"PC":0,"AX":0,"SP":SP_INIT,"BP":SP_INIT,"STACK0":0}
        store_log=[]; cur_pc=0
        from c4_min.qwen_lean_stack import LeanDataStack
        ds=LeanDataStack()
        got=None
        for _ in range(64):
            op = code[cur_pc].op if 0<=cur_pc<len(code) else None
            reg_state["STACK0"]=ds.top()
            x,pos = LF._build_stream_and_overlay(lean, code, reg_state, store_log, None)
            with torch.no_grad(): hidden,_=lean.forward(x,past=None,q_positions=pos)
            state=hidden[0,-1]
            pc=_snap(state[L.PC_VAL])
            if op in lean._nib_ax_ops():
                ax32=_decode_reg_from_nibbles(state,L,L.AX)&0xFFFFFFFF
            else:
                ax32=_decode_reg_from_nibbles(state,L,L.AX)&0xFFFFFFFF
            prev_ax = reg_state["AX"]
            if op==op_to_read:
                got=ax32
            # advance stack (need prev ax 8bit? use 32) -- we track ds with 32-bit prev
            ds.apply(op, prev_ax=prev_ax, model_ax=ax32)
            reg_state={"PC":pc,"AX":ax32,"SP":_snap(state[L.SP_VAL]),
                       "BP":_snap(state[L.BP_VAL]),"STACK0":ds.top()}
            cur_pc=pc
            if float(state[L.HALTED])>0.5 or not(0<=cur_pc<len(code)): break
        return got

    tests = [
        ("1000*1000=1000000", [("IMM",1000),("PSH",0),("IMM",1000),("MUL",0),("HALT",0)], isa.MUL, 1_000_000),
        ("70000/256=273",     [("IMM",70000),("PSH",0),("IMM",256),("DIV",0),("HALT",0)], isa.DIV, 273),
        ("65535*3=196605",    [("IMM",65535),("PSH",0),("IMM",3),("MUL",0),("HALT",0)], isa.MUL, 196605),
    ]
    print("\n=== (A) 32-bit ALU decode (mask=0xFFFFFFFF) ===")
    okA=True
    for name,prog,op,want in tests:
        got=decode32_last_op(prog,op)
        ok = got==want; okA=okA and ok
        print(f"  {'OK ' if ok else 'FAIL'} {name:20s} want={want} got={got}")

    # ---- (B) 32-bit MEMORY address: store to a real doom-style pointer (0x10000+), load it back ----
    print("\n=== (B) 32-bit memory address (wide CAM, wall #1) ===")
    from c4_min.qwen_lean_stack import LeanDataStack
    # store val 200 at addr 0x10040 (66112), then load it. tests 32-bit addr disambiguation.
    def store_then_load(addr, val):
        # IMM val; PSH; IMM addr; SI ; IMM addr; LI ; HALT
        prog=[("IMM",val),("PSH",0),("IMM",addr),("SI",0),("IMM",addr),("LI",0),("HALT",0)]
        code=isa.assemble(prog)
        reg_state={"PC":0,"AX":0,"SP":SP_INIT,"BP":SP_INIT,"STACK0":0}
        store_log=[]; cur_pc=0; ds=LeanDataStack(); got=None
        for _ in range(32):
            op=code[cur_pc].op if 0<=cur_pc<len(code) else None
            reg_state["STACK0"]=ds.top()
            load_addr=None
            if op in (isa.LI,isa.LC): load_addr = reg_state["AX"] & ((1<<args.bits)-1)
            x,pos=LF._build_stream_and_overlay(lean,code,reg_state,store_log,load_addr)
            with torch.no_grad(): hidden,_=lean.forward(x,past=None,q_positions=pos)
            state=hidden[0,-1]
            pc=_snap(state[L.PC_VAL])
            ax32=_decode_reg_from_nibbles(state,L,L.AX)&0xFFFFFFFF
            prev_ax=reg_state["AX"]
            if op in (isa.SI,isa.SC):
                store_addr = ds.values[-1] if ds.values else 0   # FULL 32-bit addr (no &0xFF)
                store_val = ax32
                store_log=[s for s in store_log if s["addr"]!=store_addr]
                store_log.append({"addr":store_addr,"val":store_val})
            if op==isa.LI: got=ax32
            ds.apply(op, prev_ax=prev_ax, model_ax=ax32)
            reg_state={"PC":pc,"AX":ax32,"SP":_snap(state[L.SP_VAL]),"BP":_snap(state[L.BP_VAL]),"STACK0":ds.top()}
            cur_pc=pc
            if float(state[L.HALTED])>0.5 or not(0<=cur_pc<len(code)): break
        return got
    okB=True
    for addr,val in [(0x10040,200),(0x10044,77),(0x200000,123)]:
        got=store_then_load(addr,val)
        ok=(got==val); okB=okB and ok
        print(f"  {'OK ' if ok else 'FAIL'} store {val} @ 0x{addr:x} -> load got={got}")

    print(f"\n[SUMMARY] 32bit_ALU={'PASS' if okA else 'FAIL'}  32bit_MEM={'PASS' if okB else 'FAIL'}")
    return 0 if (okA and okB) else 1

if __name__=="__main__":
    raise SystemExit(main())
