#!/usr/bin/env python3
import warnings, torch, argparse
warnings.filterwarnings("ignore")
from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF
from c4_min.nibble_pure_forward_complete import _decode_reg_from_nibbles
from c4_min.qwen_full_vm import _snap
from c4_min.nibble_pure_forward import SP_INIT
from _agent_lean_recurrent import build_lean_recurrent
ap=argparse.ArgumentParser(); ap.add_argument("--bits",type=int,default=18); a=ap.parse_args()
dev=torch.device("cuda:0")
vm=Q.build(code_size=48, subset=Q.SUBSET_MULDIV, recurrent_divmod=True, code_from_memory=True, mem_addr_bits=a.bits)
lean=build_lean_recurrent(vm, device=str(dev)); L=lean.QL.L
print("built hidden", lean.hidden_size)

# First: does HF run_program (the reference model path) decode 1000*1000 at 32-bit?
prog=[("IMM",1000),("PSH",0),("IMM",1000),("MUL",0),("HALT",0)]
code=isa.assemble(prog)
hf=Q.run_program(vm, code, max_steps=32, mask=0xFFFFFFFF)
print("HF run_program ax_trace (mask 32):", hf["ax_trace"], "exact-vs-nothing")

# Now single-step the lean forward and print the raw AX nibble decode at each step.
reg_state={"PC":0,"AX":0,"SP":SP_INIT,"BP":SP_INIT,"STACK0":0}
store_log=[]; cur_pc=0
for st in range(6):
    op=code[cur_pc].op if 0<=cur_pc<len(code) else None
    x,pos=LF._build_stream_and_overlay(lean,code,reg_state,store_log,None)
    with torch.no_grad(): hidden,_=lean.forward(x,past=None,q_positions=pos)
    state=hidden[0,-1]
    pc=_snap(state[L.PC_VAL]); sp=_snap(state[L.SP_VAL])
    ax8=_snap(state[L.AX_VAL])&0xFF
    ax32=_decode_reg_from_nibbles(state,L,L.AX)&0xFFFFFFFF
    stk32=_decode_reg_from_nibbles(state,L,L.STK)&0xFFFFFFFF if hasattr(L,"STK") else None
    print(f"step {st} op={isa.NAMES.get(op,op):5s} pc->{pc} ax8={ax8} ax32={ax32} sp={sp}")
    # advance minimal: pass ax32 forward as AX
    reg_state={"PC":pc,"AX":ax32,"SP":sp,"BP":_snap(state[L.BP_VAL]),"STACK0":reg_state["AX"] if op==isa.PSH else reg_state["STACK0"]}
    cur_pc=pc
    if float(state[L.HALTED])>0.5: break
