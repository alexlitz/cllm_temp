#!/usr/bin/env python3
"""Byte-identity hash via a FRESH compile_full_vm_dynamic(disk_cache=False).

Builds the production-config model with NO disk cache and NO in-process model
cache, then hashes all params+buffers. Run flag-off here and at a HEAD ref to
prove the flag-off build is byte-identical. Mirrors run_vm's production kwargs.

Usage: CUDA_VISIBLE_DEVICES=0 [C4_DIV_MULTIBYTE=1] python tools/probe_byteident_nocache.py
"""
import os, sys, hashlib
os.environ.setdefault("C4_TEST_SPEC_K","0"); os.environ.setdefault("C4_SMOKE_SPEC_K","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

def main():
    # Production geometry (mirror run_vm AutoregressiveVMRunner defaults for
    # the groundtruth/smoke path: pure_neural + trust_neural_alu => efficient).
    model, layout = compile_full_vm_dynamic(
        alu_mode="efficient",
        disk_cache=False,
    )
    h=hashlib.sha256()
    n=0
    def upd(name, t):
        nonlocal n
        td=t.detach()
        if td.is_sparse or td.layout != torch.strided:
            td=td.to_dense()
        h.update(name.encode())
        h.update(td.to(torch.float32).cpu().contiguous().numpy().tobytes())
        n+=td.numel()
    for name,p in sorted(model.named_parameters(), key=lambda kv: kv[0]): upd(name,p)
    for name,b in sorted(model.named_buffers(), key=lambda kv: kv[0]): upd(name,b)
    print(f"C4_DIV_MULTIBYTE={os.environ.get('C4_DIV_MULTIBYTE','0')}")
    print(f"n_blocks={len(model.blocks)} total_elems={n}")
    print(f"PARAM_HASH={h.hexdigest()}")

if __name__=="__main__":
    main()
