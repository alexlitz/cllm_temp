#!/usr/bin/env python3
"""Compute a deterministic hash of the built model's parameters (flag-off
byte-identity check). Prints a sha256 over all named parameters in sorted
order. Run with the flag off (default) to compare against a HEAD build.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/probe_model_param_hash.py
"""
import os, sys, hashlib
os.environ.setdefault("C4_TEST_SPEC_K","0"); os.environ.setdefault("C4_SMOKE_SPEC_K","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from tools.probe_groundtruth import build_groundtruth_probe

def main():
    probe=build_groundtruth_probe(); model=probe.model
    def _hash_tensor(name, t):
        h.update(name.encode())
        td = t.detach()
        if td.is_sparse or td.layout != torch.strided:
            td = td.to_dense()
        arr = td.to(torch.float32).cpu().contiguous().numpy().tobytes()
        h.update(arr)
        return td.numel()

    h=hashlib.sha256()
    nparams=0
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        nparams += _hash_tensor(name, p)
    # also buffers (W_proj etc. live in buffers)
    for name, b in sorted(model.named_buffers(), key=lambda kv: kv[0]):
        nparams += _hash_tensor(name, b)
    print(f"C4_DIV_MULTIBYTE={os.environ.get('C4_DIV_MULTIBYTE','0')}")
    print(f"n_blocks={len(model.blocks)} d_model={model.dim_positions and len(model.dim_positions)>0 and 'ok'}")
    print(f"total_elems={nparams}")
    print(f"PARAM_HASH={h.hexdigest()}")

if __name__=="__main__":
    main()
