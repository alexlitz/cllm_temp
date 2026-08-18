/* _fast_native_c4.c — a C port of id_port/c90_e2e/native_c4.py semantics EXACTLY.
 * 8-byte stack cell, byte-offset LEA/ENT/ADJ, 32-bit truncating ALU, flat memory.
 * Built only for the local speed-ceiling measurement (path 3). Verified byte-exact
 * against native_c4.run by the Python driver before timing.
 *
 * Opcodes match c4_min.isa. Memory is a flat int32 byte array over a 24 MiB window
 * (stack 1MiB, data at 0x10000, heap at 0x400000) which covers the benchmark set.
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* opcodes (c4_min.isa) */
enum { OP_LEA=0, OP_IMM=1, OP_JMP=2, OP_JSR=3, OP_BZ=4, OP_BNZ=5, OP_ENT=6,
       OP_ADJ=7, OP_LEV=8, OP_LI=9, OP_LC=10, OP_SI=11, OP_SC=12, OP_PSH=13,
       OP_OR=14, OP_XOR=15, OP_AND=16, OP_EQ=17, OP_NE=18, OP_LT=19, OP_GT=20,
       OP_LE=21, OP_GE=22, OP_SHL=23, OP_SHR=24, OP_ADD=25, OP_SUB=26, OP_MUL=27,
       OP_DIV=28, OP_MOD=29, OP_PRTF=33, OP_MALC=34, OP_FREE=35, OP_MSET=36,
       OP_MCMP=37, OP_HALT=38, OP_NOP=39 };

#define CELL 8LL
#define SP_INIT (1LL<<20)
#define HEAP_BASE (1LL<<22)
#define MEMWIN (24*1024*1024)   /* 24 MiB byte window */

static uint8_t *MEM = NULL;

static inline uint32_t loadw(int64_t a){
    return (uint32_t)MEM[a] | ((uint32_t)MEM[a+1]<<8) |
           ((uint32_t)MEM[a+2]<<16) | ((uint32_t)MEM[a+3]<<24);
}
static inline void storew(int64_t a, uint32_t v){
    MEM[a]=v&0xFF; MEM[a+1]=(v>>8)&0xFF; MEM[a+2]=(v>>16)&0xFF; MEM[a+3]=(v>>24)&0xFF;
}

/* returns (ax<<32)|steps-flag not needed; we write outputs via pointers */
int64_t vm_run(const int32_t *ops, const int64_t *imms, int64_t code_len,
               const int64_t *init_addr, const int64_t *init_val, int64_t init_n,
               int64_t max_steps, int64_t *out_steps){
    if(!MEM) MEM = (uint8_t*)malloc(MEMWIN);
    memset(MEM, 0, MEMWIN);
    for(int64_t i=0;i<init_n;i++) MEM[init_addr[i]] = (uint8_t)(init_val[i]&0xFF);

    int64_t heap = HEAP_BASE;
    uint32_t ax = 0;
    int64_t pc = 0, sp = SP_INIT, bp = SP_INIT, steps = 0;
    while(pc>=0 && pc<code_len && steps<max_steps){
        steps++;
        int32_t op = ops[pc];
        int64_t imm = imms[pc];
        int64_t i = pc;
        pc++;
        switch(op){
        case OP_IMM: ax = (uint32_t)imm; break;
        case OP_LEA: ax = (uint32_t)(bp + imm); break;
        case OP_PSH: sp -= CELL; storew(sp, ax); break;
        case OP_ADD: { uint32_t s=loadw(sp); sp+=CELL; ax = s+ax; } break;
        case OP_SUB: { uint32_t s=loadw(sp); sp+=CELL; ax = s-ax; } break;
        case OP_MUL: { uint32_t s=loadw(sp); sp+=CELL; ax = s*ax; } break;
        case OP_DIV: { uint32_t s=loadw(sp); sp+=CELL;
                       int32_t sv=(int32_t)s, sa=(int32_t)ax;
                       ax = sa? (uint32_t)(sv/sa):0; } break;
        case OP_MOD: { uint32_t s=loadw(sp); sp+=CELL;
                       int32_t sv=(int32_t)s, sa=(int32_t)ax;
                       ax = sa? (uint32_t)(sv - (sv/sa)*sa):0; } break;
        case OP_OR:  { uint32_t s=loadw(sp); sp+=CELL; ax = s|ax; } break;
        case OP_XOR: { uint32_t s=loadw(sp); sp+=CELL; ax = s^ax; } break;
        case OP_AND: { uint32_t s=loadw(sp); sp+=CELL; ax = s&ax; } break;
        case OP_SHL: { uint32_t s=loadw(sp); sp+=CELL; ax = s<<(ax&31); } break;
        case OP_SHR: { uint32_t s=loadw(sp); sp+=CELL; int32_t sv=(int32_t)s;
                       ax = (uint32_t)(sv>>(ax&31)); } break;
        case OP_EQ:  { uint32_t v=loadw(sp); sp+=CELL; ax = (v==ax); } break;
        case OP_NE:  { uint32_t v=loadw(sp); sp+=CELL; ax = (v!=ax); } break;
        case OP_LT:  { uint32_t v=loadw(sp); sp+=CELL; ax = ((int32_t)v<(int32_t)ax); } break;
        case OP_GT:  { uint32_t v=loadw(sp); sp+=CELL; ax = ((int32_t)v>(int32_t)ax); } break;
        case OP_LE:  { uint32_t v=loadw(sp); sp+=CELL; ax = ((int32_t)v<=(int32_t)ax); } break;
        case OP_GE:  { uint32_t v=loadw(sp); sp+=CELL; ax = ((int32_t)v>=(int32_t)ax); } break;
        case OP_LI:  ax = loadw(ax); break;
        case OP_LC:  { uint8_t b=MEM[ax]; ax = (b&0x80)? (uint32_t)(b-0x100):b; } break;
        case OP_SI:  { int64_t a=loadw(sp); sp+=CELL; storew(a, ax); } break;
        case OP_SC:  { int64_t a=loadw(sp); sp+=CELL; MEM[a]=ax&0xFF; } break;
        case OP_JMP: pc = imm; break;
        case OP_BZ:  if(ax==0) pc=imm; break;
        case OP_BNZ: if(ax!=0) pc=imm; break;
        case OP_JSR: sp-=CELL; storew(sp, (uint32_t)(i+1)); pc=imm; break;
        case OP_ENT: sp-=CELL; storew(sp, (uint32_t)bp); bp=sp; sp-=imm; break;
        case OP_ADJ: sp += imm; break;
        case OP_LEV: sp=bp; bp=loadw(sp); pc=loadw(sp+CELL); sp+=2*CELL; break;
        case OP_MALC:{ int64_t n=loadw(sp); sp+=CELL; n=(n+7)&~7LL; ax=heap; heap+=n; } break;
        case OP_FREE: sp+=CELL; break;
        case OP_MSET:{ int64_t n=ax; uint32_t c=loadw(sp); sp+=CELL; int64_t p=loadw(sp); sp+=CELL;
                       for(int64_t k=0;k<n;k++) MEM[p+k]=c&0xFF; ax=p; } break;
        case OP_MCMP:{ int64_t n=ax; int64_t pb=loadw(sp); sp+=CELL; int64_t pa=loadw(sp); sp+=CELL;
                       ax=0; for(int64_t k=0;k<n;k++){ uint32_t d=(uint32_t)(MEM[pa+k]-MEM[pb+k]);
                       if(d){ ax=d; break; } } } break;
        case OP_PRTF: break;
        case OP_NOP:  break;
        case OP_HALT: goto done;
        default: *out_steps=steps; return -1; /* unimpl */
        }
    }
done:
    *out_steps = steps;
    return (int64_t)ax;
}
