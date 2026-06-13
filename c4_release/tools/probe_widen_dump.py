"""Dump per-block residual + logits for the lea program to a .pt file.

Run twice (baseline and widened, each a fresh process so the env-gated
widen + caches are clean), then diff the two dumps.

Usage:
  CUDA_VISIBLE_DEVICES=1 python tools/probe_widen_dump.py /tmp/base.pt
  CUDA_VISIBLE_DEVICES=1 C4_FORCE_WIDEN_DMODEL=920 python tools/probe_widen_dump.py /tmp/w920.pt
  python tools/probe_widen_dump.py --diff /tmp/base.pt /tmp/w920.pt
"""
from __future__ import annotations

import os
import sys

import torch

from neural_vm.embedding import Opcode


def _make_bytecode(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


def _build_tokens():
    """Encode the lea program as tokens WITHOUT building a runner (which
    would compile a model and pollute the in-process op cache)."""
    from neural_vm.vm_step import Token
    from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE
    bc = _make_bytecode([Opcode.ENT, (Opcode.IMM, 0), (Opcode.LEA, 2), Opcode.EXIT])
    tokens = [Token.CODE_START]
    for instr in bc:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(IMMEDIATE_SIZE):
            tokens.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            tokens.append(0)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens


def dump(path):
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

    tok_list = _build_tokens()
    tokens = torch.tensor(tok_list[:70], device="cuda").unsqueeze(0)

    extra = None
    raw = os.environ.get("C4_EXTRA_RESIDUAL_DIMS", "").strip()
    if raw:
        extra = {}
        for e in raw.split(","):
            n, _, s = e.partition(":")
            extra[n.strip()] = int(s.strip())
    m, l = compile_full_vm_dynamic(
        disk_cache=False, strict=False,
        alu_mode="efficient", max_seq_len=4096,
        extra_residual_dims=extra,
    )
    m = m.cuda().eval()
    D = l.d_model
    hd = m.blocks[0].attn.head_dim
    nh = m.blocks[0].attn.num_heads
    nblk = len(m.blocks)

    caps = []
    with torch.no_grad():
        x = m.embed(tokens)
        for blk in m.blocks:
            x = blk(x)
            caps.append(x[:, :, :872].clone().cpu())
        logits = m.head(x).cpu()
    torch.save(
        {"d_model": D, "head_dim": hd, "num_heads": nh, "nblk": nblk,
         "caps": caps, "logits": logits},
        path,
    )
    print(f"dumped {path}: d_model={D} head_dim={hd} num_heads={nh} nblk={nblk}")


def diff(a, b):
    da = torch.load(a, weights_only=False)
    db = torch.load(b, weights_only=False)
    print(f"A: d_model={da['d_model']} hd={da['head_dim']} nblk={da['nblk']}")
    print(f"B: d_model={db['d_model']} hd={db['head_dim']} nblk={db['nblk']}")
    n = min(len(da["caps"]), len(db["caps"]))
    first = None
    for i in range(n):
        d = (da["caps"][i] - db["caps"][i]).abs().max().item()
        if d > 1e-4 and first is None:
            first = i
        if d > 1e-4 or i < 3 or i >= n - 3:
            print(f"  block {i:2d}: max|diff|={d:.6g}")
    ld = (da["logits"] - db["logits"]).abs().max().item()
    print(f"logits max|diff|={ld:.6g}")
    aa = da["logits"][0, -1].argmax().item()
    bb = db["logits"][0, -1].argmax().item()
    print(f"last argmax A={aa} B={bb} match={aa==bb}")
    print(f"FIRST DIVERGENT BLOCK: {first}")


if __name__ == "__main__":
    if sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        dump(sys.argv[1])
