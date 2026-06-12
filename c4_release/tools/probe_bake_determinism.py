#!/usr/bin/env python3
"""Is the bake deterministic? Build the production model TWICE in ONE process
(identical args) and compare eq_true + a weight checksum. If they differ, the
bake itself is nondeterministic. spec_k=0, disk_cache=False, efficient.
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.dirname(_ROOT))
import torch
from neural_vm.embedding import Opcode
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
from neural_vm.vm_step import DEFAULT_N_HEADS, DEFAULT_FFN_HIDDEN

def _mk(ops):
    bc=[]
    for op in ops:
        if isinstance(op,tuple): opc,imm=op; bc.append(opc|(imm<<8))
        else: bc.append(op)
    return bc
EQ_TRUE=_mk([(Opcode.IMM,5),Opcode.PSH,(Opcode.IMM,5),Opcode.EQ,Opcode.EXIT])

def build_and_run(tag):
    model,_=compile_full_vm_dynamic(strict=False,disk_cache=False,alu_mode="efficient",
        n_heads=DEFAULT_N_HEADS,ffn_hidden=DEFAULT_FFN_HIDDEN,max_seq_len=4096)
    if torch.cuda.is_available(): model=model.cuda()
    model.eval()
    # weight checksum of block 11 ffn (the eq engine host) + total param sum
    chk=0.0
    for p in model.parameters():
        chk += float(p.detach().double().abs().sum().item())
    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    mr=AutoregressiveVMRunner(pure_neural=True,trust_neural_alu=True,spec_k=0,cache_model=False)
    mr.model=model; mr._func_call_handlers={}; mr._syscall_handlers={}
    runner=BatchedPureNeuralRunner(model_runner=mr)
    r=runner.run_batch([EQ_TRUE],max_steps=20,spec_k=0,bucket_by_predicted_length=False)
    print(f"[{tag}] eq_true={r[0][1]} param_abs_sum={chk:.6f} nblocks={len(model.blocks)}",flush=True)
    return r[0][1], chk

def main():
    print("BUILD A...",flush=True); a,ca=build_and_run("A")
    print("BUILD B...",flush=True); b,cb=build_and_run("B")
    print(f"\neq_true: A={a} B={b}  {'SAME' if a==b else 'DIFFER'}")
    print(f"param_abs_sum: A={ca:.6f} B={cb:.6f}  {'IDENTICAL WEIGHTS' if ca==cb else 'DIFFERENT WEIGHTS'}")

if __name__=="__main__":
    main()
