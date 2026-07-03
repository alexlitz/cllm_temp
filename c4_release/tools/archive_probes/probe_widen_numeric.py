"""Probe: are baseline (872) and widened (981/9) models numerically
identical on existing dims for a fixed input?

Compares per-layer residual streams and final logits token-by-token for
the lea_basic program. If they diverge, prints the first layer + dim
where they differ.

Run: CUDA_VISIBLE_DEVICES=1 python tools/probe_widen_numeric.py
"""
from __future__ import annotations

import os

import torch

from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
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


def build(target=None, nheads=None):
    if target:
        os.environ["C4_FORCE_WIDEN_DMODEL"] = str(target)
        if nheads:
            os.environ["C4_FORCE_WIDEN_NHEADS"] = str(nheads)
    else:
        os.environ.pop("C4_FORCE_WIDEN_DMODEL", None)
        os.environ.pop("C4_FORCE_WIDEN_NHEADS", None)
    m, l = compile_full_vm_dynamic(disk_cache=False, strict=False)
    m = m.cuda().eval()
    return m, l


def main():
    # Build the embedding token ids for the lea program via a runner.
    from neural_vm.run_vm import AutoregressiveVMRunner
    serial = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    bc = _make_bytecode([
        Opcode.ENT, (Opcode.IMM, 0), (Opcode.LEA, 2), Opcode.EXIT,
    ])
    ctx = serial._build_context(bc, b"", [], "")
    tok_list = ctx if isinstance(ctx, list) else ctx.token_ids
    tokens = torch.tensor(tok_list[:70], device="cuda").unsqueeze(0)
    print("input tokens:", tokens.shape)

    import sys
    tgt = int(sys.argv[1]) if len(sys.argv) > 1 else 981
    nh = int(sys.argv[2]) if len(sys.argv) > 2 else None
    m0, l0 = build(None)
    m1, l1 = build(tgt, nh)
    D0 = l0.d_model
    print(f"baseline d_model={l0.d_model} hd={m0.blocks[0].attn.head_dim} "
          f"nblk={len(m0.blocks)}")
    print(f"widened  d_model={l1.d_model} hd={m1.blocks[0].attn.head_dim} "
          f"nblk={len(m1.blocks)} nheads={m1.blocks[0].attn.num_heads}")
    if len(m0.blocks) != len(m1.blocks):
        print("WARNING: block counts differ — structural change, not a pure widen")

    with torch.no_grad():
        # Hook each block to capture residual after attn and after block.
        def run_capture(model):
            caps = []
            x = model.embed(tokens)
            for bi, blk in enumerate(model.blocks):
                x = blk(x)
                caps.append(x[:, :, :D0].clone())
            logits = model.head(x)
            return caps, logits

        caps0, logits0 = run_capture(m0)
        caps1, logits1 = run_capture(m1)

    print(f"\nComparing residuals on existing {D0} dims, per block:")
    first_div = None
    for bi, (c0, c1) in enumerate(zip(caps0, caps1)):
        diff = (c0 - c1).abs()
        md = diff.max().item()
        if md > 1e-4 and first_div is None:
            first_div = bi
        argd = diff.argmax().item()
        flat = diff.flatten()
        pos = argd
        dim = pos % D0
        tok = (pos // D0) % c0.shape[1]
        print(f"  block {bi:2d}: max|diff|={md:.6g}  (at token {tok}, dim {dim})")

    ld = (logits0 - logits1).abs().max().item()
    print(f"\nfinal logits max|diff| = {ld:.6g}")
    a0 = logits0[0, -1].argmax().item()
    a1 = logits1[0, -1].argmax().item()
    print(f"last-token argmax: baseline={a0}  widened={a1}  match={a0==a1}")
    if first_div is not None:
        print(f"\nFIRST DIVERGENT BLOCK: {first_div}")
    else:
        print("\nNo residual divergence > 1e-4 on existing dims.")


if __name__ == "__main__":
    main()
