import sys, os
sys.path.insert(0,'/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/task853-compiler-cfm/c4_release')
sys.path.insert(0,'/home/alexlitz/Documents/misc/c4_doom')
from src.compiler import compile_c
from c4_min import isa
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, data_segment
src=open(sys.argv[1]).read(); inp=open(sys.argv[2],'rb').read()
bc,data=compile_c(src)
code=tag_compiler_syscalls(bytecode_to_isa(bc),isa); data_seg=data_segment(data)
M=0xFFFFFFFF; SP_INIT=0xFC
def s32(v):
    v&=M; return v-(1<<32) if v&0x80000000 else v
mem=dict(data_seg); sp=bp=SP_INIT; ax=pc=0; steps=0
brk=[0x40000]; N=isa.NAMES; files={0:[inp,0]}; out=bytearray()
min_sp=SP_INIT; max_depth=0; maxjsr=0
while 0<=pc<len(code) and steps<200000:
    steps+=1; ins=code[pc]; op=ins.op; imm=ins.imm; ppc=pc; pc+=1; n=N.get(op,op)
    if   n=="IMM": ax=imm&M
    elif n=="LEA": ax=(bp+4*s32(imm))&M
    elif n=="PSH": sp-=4; mem[sp]=ax&M
    elif n=="LI": ax=mem.get(ax&M,0)&M
    elif n=="LC": ax=mem.get(ax&M,0)&0xFF
    elif n=="SI": a=mem.get(sp,0); sp+=4; mem[a&M]=ax&M
    elif n=="SC": a=mem.get(sp,0); sp+=4; mem[a&M]=ax&0xFF
    elif n=="ADD": v=mem.get(sp,0); sp+=4; ax=(v+ax)&M
    elif n=="SUB": v=mem.get(sp,0); sp+=4; ax=(v-ax)&M
    elif n=="MUL": v=mem.get(sp,0); sp+=4; ax=(v*ax)&M
    elif n=="DIV": v=mem.get(sp,0); sp+=4; ax=(int(s32(v)/s32(ax)) if ax else 0)&M
    elif n=="MOD": v=mem.get(sp,0); sp+=4; ax=(s32(v)-int(s32(v)/s32(ax))*s32(ax) if ax else 0)&M
    elif n in("OR","XOR","AND"): v=mem.get(sp,0); sp+=4; ax=({'OR':v|ax,'XOR':v^ax,'AND':v&ax}[n])&M
    elif n in("SHL","SHR"):
        v=mem.get(sp,0); sp+=4
        ax=((v<<(ax&31))&M) if n=="SHL" else ((s32(v)>>(ax&31))&M)
    elif n in("EQ","NE","LT","GT","LE","GE"):
        v=mem.get(sp,0); sp+=4; sa,sv=s32(ax),s32(v)
        r={'EQ':sv==sa,'NE':sv!=sa,'LT':sv<sa,'GT':sv>sa,'LE':sv<=sa,'GE':sv>=sa}[n]; ax=1 if r else 0
    elif n=="JMP": pc=imm
    elif n=="BZ": pc=imm if ax==0 else pc
    elif n=="BNZ": pc=imm if ax!=0 else pc
    elif n=="JSR": sp-=4; mem[sp]=ppc+1; pc=imm
    elif n=="ENT": mem[sp-4]=bp&M; sp-=4; bp=sp; sp-=4*imm
    elif n=="ADJ": sp+=4*imm
    elif n=="LEV": sp=bp; bp=mem.get(sp,0); pc=mem.get(sp+4,0); sp+=8
    elif n=="MALC": p=brk[0]; brk[0]+=((ax+7)&~7); ax=p&M
    elif n=="OPEN": ax=(-1)&M
    elif n=="READ":
        argc=imm
        def a(i): return mem.get(sp+4*(argc-1-i),0)&M
        fd,buf,cnt=a(0),a(1),a(2); db,cur=files.get(fd,[b'',0]); ch=db[cur:cur+cnt]
        for j,bb in enumerate(ch): mem[(buf+j)&M]=bb
        files[fd][1]=cur+len(ch); ax=len(ch)&M
    elif n=="CLOS": ax=0
    elif n=="PRTF":
        argc=imm
        def a(i): return mem.get(sp+4*(argc-1-i),0)&M
        fp=a(0); s=b''; p=fp
        while (mem.get(p,0)&0xFF)!=0: s+=bytes([mem.get(p,0)&0xFF]); p+=1
        fmt=s.decode('latin1'); args=[a(i) for i in range(1,argc)]; oi=0; ai=0; res=b''
        while oi<len(fmt):
            c=fmt[oi]
            if c=='%' and oi+1<len(fmt):
                sp2=fmt[oi+1]; oi+=2
                if sp2=='d': res+=str(s32(args[ai])).encode(); ai+=1
                elif sp2=='s':
                    pp=args[ai]; ai+=1
                    while (mem.get(pp,0)&0xFF)!=0: res+=bytes([mem.get(pp,0)&0xFF]); pp+=1
                elif sp2=='c': res+=bytes([args[ai]&0xFF]); ai+=1
                else: res+=('%'+sp2).encode()
            else: res+=c.encode('latin1'); oi+=1
        out.extend(res)
    elif n=="EXIT": break
    elif n=="HALT": break
    else: print("unhandled",n); break
    if sp<min_sp: min_sp=sp
    if SP_INIT-sp>max_depth: max_depth=SP_INIT-sp
print("steps",steps,"stack_depth_bytes",max_depth,"min_sp",min_sp,"(SP_INIT=%d)"%SP_INIT)
print("stack goes to", s32(min_sp) if min_sp>0x80000000 else min_sp)
print("OUTPUT:",repr(bytes(out)))
