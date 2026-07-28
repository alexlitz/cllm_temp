#!/usr/bin/env python3
"""_agent_build_fullisa_sparse.py — build the FULL-ISA compact block-stack model,
export + lower to the COO ``.nblbin``, and expose helpers to compile BOTH the
dense and the SPARSE whole-forward C runtimes.

This is the driver for the "one self-contained native binary, sparse whole
forward, byte-exact echo/quine/mandelbrot, MEASURE the real speed" task.  It uses
the CANONICAL full-ISA compact model (``compact_alloc.build_compact_pure_forward_model``,
every opcode present, packed-residual) — NOT the 207-node toy.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
C4MIN = os.path.dirname(HERE)
DENSE_RT = os.path.join(C4MIN, "onnx_runtime_nibble.c")
SPARSE_RT = os.path.join(C4MIN, "onnx_runtime_nibble_sparse.c")
SELFCONTAINED_RT = os.path.join(C4MIN, "onnx_runtime_nibble_selfcontained.c")


def compile_selfcontained(exe, binp):
    """gcc -O2 -static, EMBEDDING the .nblbin at ``binp`` via .incbin — one
    self-contained static binary that needs NO external model file."""
    r = subprocess.run(
        ["gcc", "-O2", "-static", f"-DMODEL_BLOB_PATH={binp}",
         "-o", exe, SELFCONTAINED_RT, "-lm"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcc (selfcontained) failed:\n{r.stderr}")
    return exe


def build_model(code_size=48):
    from c4_min import compact_alloc as CA
    model, L, stats = CA.build_compact_pure_forward_model(code_size=code_size)
    model.eval()
    return model, L, stats


def export_and_lower(model, L, out_dir):
    from c4_min import export_onnx as E
    from c4_min.onnx_to_c4bin import lower_onnx_to_bin
    os.makedirs(out_dir, exist_ok=True)
    dense = os.path.join(out_dir, "blockstack.onnx")
    sp = os.path.join(out_dir, "blockstack_sparse.onnx")
    binp = os.path.join(out_dir, "blockstack.nblbin")
    E.export_blockstack_onnx(model, dense, dim=L.D)
    E.to_sparse_onnx(dense, sp)
    info = lower_onnx_to_bin(sp, binp)
    return binp, info


def compile_rt(csrc, exe, static=True):
    err = ""
    trials = ([["-O2", "-static", "-static-libgcc"], ["-O2", "-static"], ["-O2"]]
              if static else [["-O2"]])
    for flags in trials:
        r = subprocess.run(["gcc"] + flags + ["-o", exe, csrc, "-lm"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return exe, flags
        err = r.stderr
    raise RuntimeError(f"gcc failed for {csrc}:\n{err}")


def dump_frames_and_embed(model, L, out_dir, srcs, nframes=3):
    """Collect real VM residual frames (pre-embed + overlay) + torch block-stack
    reference outputs, and the embed table, so downstream verify/measure runs never
    rebuild the model.  Saves an .npz."""
    import numpy as np
    import torch
    from c4_min import nibble_pure_forward_complete as PFC
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min import blogspec_vocab as V
    from src.compiler import compile_c
    embed = model.embed.detach().numpy().astype(np.float32)
    frames = []
    ref_outs = []
    for src in srcs:
        code = bytecode_to_isa(compile_c(src)[0])
        stream = [V.BOS] + PFC._build_frame(0, 0, PFC.SP_INIT, PFC.SP_INIT, 0)
        for _ in range(nframes):
            overlay = PFC.make_overlay_complete(code, L, store_log={})
            x = torch.from_numpy(embed[np.asarray(stream)]).unsqueeze(0).clone()
            overlay(x)
            frames.append(x.numpy().astype(np.float32))
            with torch.no_grad():
                xt = x.clone()
                for blk in model.blocks:
                    xt = blk(xt)
            ref_outs.append(xt.numpy().astype(np.float32))
            st = xt[0, -1]
            pc = PFC._snap_lane(st[L.PC_VAL]); sp = PFC._snap_lane(st[L.SP_VAL])
            bp = PFC._snap_lane(st[L.BP_VAL]); stk = PFC._snap_lane(st[L.STK_VAL])
            ax = PFC._decode_reg_from_nibbles(st, L, L.AX)
            stream += PFC._build_frame(pc, ax, sp, bp, stk)
            if float(st[L.HALTED]) > 0.5 or pc < 0 or pc >= len(code):
                break
    d = {"embed": embed, "D": np.int64(L.D)}
    for i, (f, r) in enumerate(zip(frames, ref_outs)):
        d[f"frame_{i}"] = f
        d[f"ref_{i}"] = r
    d["n"] = np.int64(len(frames))
    np.savez(os.path.join(out_dir, "frames.npz"), **d)
    return len(frames)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse"
    t0 = time.time()
    print("building compact full-ISA model...", flush=True)
    model, L, stats = build_model()
    print(f"  built in {time.time()-t0:.1f}s  dim(D)={L.D}  n_blocks={len(model.blocks)}")
    t1 = time.time()
    print("exporting + lowering to .nblbin ...", flush=True)
    binp, info = export_and_lower(model, L, out_dir)
    print(f"  lowered in {time.time()-t1:.1f}s")
    for k, v in info.items():
        print(f"    {k}: {v}")
    print(f"  nblbin size: {os.path.getsize(binp):,} bytes", flush=True)
    print("dumping real VM frames + torch reference outputs ...", flush=True)
    srcs = ["int main(){return 42;}",
            "int main(){return 6+7;}",
            "int main(){return 100-58;}",
            "int main(){return 7*6;}",
            "int main(){return 84/2;}"]
    nf = dump_frames_and_embed(model, L, out_dir, srcs, nframes=3)
    print(f"  dumped {nf} frames + embed to frames.npz "
          f"({os.path.getsize(os.path.join(out_dir, 'frames.npz')):,} bytes)")
    print("compiling runtimes (dense, sparse, self-contained)...", flush=True)
    dense_exe = os.path.join(out_dir, "rt_dense")
    sparse_exe = os.path.join(out_dir, "rt_sparse")
    prog_exe = os.path.join(out_dir, "prog")
    compile_rt(DENSE_RT, dense_exe)
    compile_rt(SPARSE_RT, sparse_exe)
    compile_selfcontained(prog_exe, binp)
    print(f"  rt_dense : {os.path.getsize(dense_exe):,} bytes")
    print(f"  rt_sparse: {os.path.getsize(sparse_exe):,} bytes")
    print(f"  prog     : {os.path.getsize(prog_exe):,} bytes (self-contained, "
          f"embeds the {os.path.getsize(binp):,}-byte model)")
