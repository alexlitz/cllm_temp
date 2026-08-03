"""Full-ISA byte-exact verification of the FULL efficient-ALU recurrent model
via the DENSE lean forward (CPU driver), vs isa.interpret."""
import os, time
os.environ.setdefault('C4_VM_CACHE_DIR','/tmp/c4cache_agentfull')
import torch
torch.manual_seed(0)
from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF

t0=time.time()
print('building FULL efficient_alu + recurrent_divmod ...', flush=True)
vm = Q.build(code_size=24, subset=Q.SUBSET_FULL, efficient_alu=True, recurrent_divmod=True)
print(f'built {time.time()-t0:.1f}s stored_layers={vm.n_layers} applied={vm.n_applied} H={vm.hidden_size} I={vm.intermediate_size}', flush=True)
lean = LF.LeanQwenVM.from_full_vm(vm, device='cpu')
print(f'lean layers (applied)={len(lean.layers)}', flush=True)

A=isa.assemble
# One representative program per opcode / opcode-family. imm operands chosen to
# exercise real values. Every program ends in HALT so the driver stops.
progs = {
 'IMM'      : A([('IMM',42),('HALT',0)]),
 'PSH+ADD'  : A([('IMM',7),('PSH',0),('IMM',35),('ADD',0),('HALT',0)]),
 'SUB'      : A([('IMM',50),('PSH',0),('IMM',8),('SUB',0),('HALT',0)]),
 'MUL'      : A([('IMM',6),('PSH',0),('IMM',7),('MUL',0),('HALT',0)]),
 'DIV'      : A([('IMM',100),('PSH',0),('IMM',7),('DIV',0),('HALT',0)]),
 'MOD'      : A([('IMM',100),('PSH',0),('IMM',7),('MOD',0),('HALT',0)]),
 'OR'       : A([('IMM',0xF0),('PSH',0),('IMM',0x0F),('OR',0),('HALT',0)]),
 'XOR'      : A([('IMM',0xFF),('PSH',0),('IMM',0x0F),('XOR',0),('HALT',0)]),
 'AND'      : A([('IMM',0xF3),('PSH',0),('IMM',0x0F),('AND',0),('HALT',0)]),
 'SHL'      : A([('IMM',3),('PSH',0),('IMM',2),('SHL',0),('HALT',0)]),
 'SHR'      : A([('IMM',200),('PSH',0),('IMM',2),('SHR',0),('HALT',0)]),
 'EQ_t'     : A([('IMM',5),('PSH',0),('IMM',5),('EQ',0),('HALT',0)]),
 'EQ_f'     : A([('IMM',5),('PSH',0),('IMM',6),('EQ',0),('HALT',0)]),
 'NE'       : A([('IMM',5),('PSH',0),('IMM',6),('NE',0),('HALT',0)]),
 'LT'       : A([('IMM',3),('PSH',0),('IMM',9),('LT',0),('HALT',0)]),
 'GT'       : A([('IMM',9),('PSH',0),('IMM',3),('GT',0),('HALT',0)]),
 'LE'       : A([('IMM',5),('PSH',0),('IMM',5),('LE',0),('HALT',0)]),
 'GE'       : A([('IMM',5),('PSH',0),('IMM',5),('GE',0),('HALT',0)]),
 'LI+SI'    : A([('IMM',77),('PSH',0),('IMM',12),('SI',0),   # mem[12]=77
                 ('IMM',12),('LI',0),('HALT',0)]),           # ax=mem[12]
 'JMP'      : A([('IMM',1),('JMP',3),('IMM',99),('IMM',55),('HALT',0)]),
 'BZ_taken' : A([('IMM',0),('BZ',3),('IMM',99),('IMM',55),('HALT',0)]),
 'BNZ_taken': A([('IMM',1),('BNZ',3),('IMM',99),('IMM',55),('HALT',0)]),
 'countdown': A([('IMM',5),('PSH',0),('IMM',1),('SUB',0),('BNZ',1),('HALT',0)]),
}
# functions (JSR/ENT/LEV) — use the function-aware oracle path in the driver.
func_prog = A([('JSR',3),('HALT',0),('NOP',0),   # 0 JSR fn; 1 HALT
               ('ENT',0),('IMM',88),('LEV',0)])  # 3 fn: ent; ax=88; lev
progs['JSR/ENT/LEV'] = func_prog

npass=0; nfail=0; fails=[]
for name, code in progs.items():
    t=time.time()
    r = LF.run_program_lean(lean, code, max_steps=64)
    ok = r['exact']
    tag='PASS' if ok else 'FAIL'
    if ok: npass+=1
    else: nfail+=1; fails.append(name)
    print(f'  {tag} {name:12s} steps={r["steps"]:3d} model={r["ax_trace"]} ref={r["ref_trace"]}  ({time.time()-t:.1f}s)', flush=True)

print(f'\nRESULT: {npass}/{npass+nfail} programs byte-exact via DENSE lean forward.', flush=True)
if fails: print('FAILS:', fails, flush=True)
