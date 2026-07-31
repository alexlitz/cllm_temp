"""A clean 32-bit c4 interpreter (SP-addressed, 4-byte slots) to get doom's
GROUND-TRUTH step count + PC trace to first printf, matching native ./c4 semantics.
Then diff against draft_pf_program to localize any divergence."""
import sys
sys.path.insert(0,"/home/alexlitz/Documents/misc/c4_doom")
from pathlib import Path
from c4_min import isa
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, data_segment

src = Path("/home/alexlitz/Documents/misc/c4_doom/doom.c").read_text()
bc, data = compile_c(src)
code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
data_seg = data_segment(data)
M = 0xFFFFFFFF
SP_INIT = 0xFC  # match the draft's SP_INIT snapshot
def s32(v):
    v &= M
    return v - (1<<32) if v & 0x80000000 else v

mem = dict(data_seg)   # data segment seeded
sp = bp = SP_INIT
ax = pc = 0
steps = 0
first_prtf = None
PRTF=isa.PRTF; READ=isa.READ; OPEN=isa.OPEN; CLOS=isa.CLOS
while 0 <= pc < len(code) and steps < 500000:
    steps += 1
    ins = code[pc]; op, imm = ins.op, ins.imm; pc += 1
    n = isa.NAMES.get(op, op)
    if   n=="IMM": ax = imm & M
    elif n=="LEA": ax = (bp + 4*s32(imm)) & M
    elif n=="PSH": sp -= 4; mem[sp]=ax & M
    elif n=="LI":  ax = mem.get(ax & M,0) & M
    elif n=="LC":  ax = mem.get(ax & M,0) & 0xFF
    elif n=="SI":  a=mem.get(sp,0); sp+=4; mem[a & M]=ax & M
    elif n=="SC":  a=mem.get(sp,0); sp+=4; mem[a & M]=ax & 0xFF
    elif n=="ADD": v=mem.get(sp,0); sp+=4; ax=(v+ax)&M
    elif n=="SUB": v=mem.get(sp,0); sp+=4; ax=(v-ax)&M
    elif n=="MUL": v=mem.get(sp,0); sp+=4; ax=(v*ax)&M
    elif n=="DIV": v=mem.get(sp,0); sp+=4; ax=(int(s32(v)/s32(ax)) if ax else 0)&M
    elif n=="MOD": v=mem.get(sp,0); sp+=4; ax=(s32(v)-int(s32(v)/s32(ax))*s32(ax) if ax else 0)&M
    elif n=="OR":  v=mem.get(sp,0); sp+=4; ax=(v|ax)&M
    elif n=="XOR": v=mem.get(sp,0); sp+=4; ax=(v^ax)&M
    elif n=="AND": v=mem.get(sp,0); sp+=4; ax=(v&ax)&M
    elif n=="SHL": v=mem.get(sp,0); sp+=4; ax=(v<<(ax&31))&M
    elif n=="SHR": v=mem.get(sp,0); sp+=4; ax=(s32(v)>>(ax&31))&M
    elif n=="EQ":  v=mem.get(sp,0); sp+=4; ax=1 if v==ax else 0
    elif n=="NE":  v=mem.get(sp,0); sp+=4; ax=1 if v!=ax else 0
    elif n=="LT":  v=mem.get(sp,0); sp+=4; ax=1 if s32(v)<s32(ax) else 0
    elif n=="GT":  v=mem.get(sp,0); sp+=4; ax=1 if s32(v)>s32(ax) else 0
    elif n=="LE":  v=mem.get(sp,0); sp+=4; ax=1 if s32(v)<=s32(ax) else 0
    elif n=="GE":  v=mem.get(sp,0); sp+=4; ax=1 if s32(v)>=s32(ax) else 0
    elif n=="JMP": pc = imm
    elif n=="BZ":  pc = imm if (ax&M)==0 else pc
    elif n=="BNZ": pc = imm if (ax&M)!=0 else pc
    elif n=="JSR": sp-=4; mem[sp]=pc & M; pc = imm
    elif n=="ENT": mem[sp-4]=bp & M; sp-=4; bp=sp; sp-=4*imm
    elif n=="ADJ": sp += 4*imm
    elif n=="LEV": sp=bp; bp=mem.get(sp,0); pc=mem.get(sp+4,0); sp+=8
    elif n=="PRTF":
        if first_prtf is None:
            first_prtf = steps
            print(f"FIRST PRTF at step {steps}, pc(prev)={pc-1}, ax={ax}")
            break
    elif n in ("NOP",): pass
    elif n in ("HALT","EXIT"): print(f"HALT at step {steps}"); break
    else: print(f"unhandled {n} at step {steps} pc {pc-1}"); break
print(f"total steps run: {steps}; first_prtf={first_prtf}")
