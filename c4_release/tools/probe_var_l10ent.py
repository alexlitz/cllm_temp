#!/usr/bin/env python3
"""Pin the step-1 ENT-AX byte0=0xf0 leak to L10 (phys block 11) FFN unit 2304.

spec_k=0, hook-free. Reads dims via ``probe.model.dim_positions`` (NOT the
stale dim_registry_dynamic). For var_simple_12 (id 262):

1. Replays the spec_k=0 emission loop, prints the step-1 REG_AX bytes.
2. Decodes physical-block-11 FFN unit 2304's gate/up reads + down writes so we
   can confirm it is the OP_ENT-gated OUTPUT_HI+15 writer.
3. Reads the residual at the step-1 AX byte0 prediction row after block 11 to
   show OUTPUT_HI+15 (dim OUTPUT_HI+15 = 100) winning vs OUTPUT_LO band.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

probe = build_groundtruth_probe()
m = probe.model
dp = m.dim_positions
inv = {}
for k, v in dp.items():
    if int(v) not in inv and ".*." not in str(k):
        inv[int(v)] = k

def lbl(d):
    for bd in range(d, max(-1, d - 40), -1):
        if bd in inv:
            return f"{inv[bd]}+{d-bd}" if d != bd else inv[bd]
    return str(d)

tests = generate_test_programs()
src, exp, _ = tests[262]
bc, _ = compile_c(src)
print(f"id 262 src={src!r} exp={exp}", flush=True)
print(f"bytecode={[hex(b) for b in bc]}", flush=True)

# --- 1. emitted step-1 REG_AX bytes ---
ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))
ax_positions = [i for i in range(pl, len(ctx)) if ctx[i] == int(Token.REG_AX)]
print(f"\nREG_AX positions (post-prompt): {ax_positions}", flush=True)
if len(ax_positions) >= 2:
    ax1 = ax_positions[1]
    axbytes = [ctx[ax1 + 1 + j] & 0xFF for j in range(4)]
    print(f"step-1 REG_AX bytes = {[hex(b) for b in axbytes]} "
          f"(byte0={hex(axbytes[0])}, expect 0x00)", flush=True)

# --- 2. decode unit 2304 ---
U = 2304
PHYS = 11
ffn = m.blocks[PHYS].ffn
Wg = ffn.W_gate.to_dense().float() if hasattr(ffn.W_gate, "to_dense") else ffn.W_gate.float()
Wu = ffn.W_up.to_dense().float() if hasattr(ffn.W_up, "to_dense") else ffn.W_up.float()
Wd = ffn.W_down.to_dense().float() if hasattr(ffn.W_down, "to_dense") else ffn.W_down.float()
print(f"\nphys block {PHYS} ffn={type(ffn).__name__} "
      f"Wg{tuple(Wg.shape)} Wu{tuple(Wu.shape)} Wd{tuple(Wd.shape)}", flush=True)
gate = Wg[U]; up = Wu[U]; down = Wd[:, U]

def show(vec, title, k=12):
    print(f"=== unit {U} {title} (top |w|) ===", flush=True)
    order = torch.argsort(vec.abs(), descending=True)[:k]
    for i in order.tolist():
        if abs(float(vec[i])) < 1e-6:
            continue
        print(f"  dim {i:4d} {lbl(i):30s} w={float(vec[i]):+.4f}", flush=True)

show(gate, "GATE reads")
show(up, "UP reads")
show(down, "DOWN writes")

# --- 3. residual at step-1 AX byte0 row after block 11 ---
if len(ax_positions) >= 2:
    read_pos = ax_positions[1]
    dev = next(m.parameters()).device
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    OUT_LO = dp["OUTPUT_LO"]; OUT_HI = dp["OUTPUT_HI"]
    watch = {f"OUTPUT_LO+{k}": OUT_LO + k for k in range(16)}
    watch.update({f"OUTPUT_HI+{k}": OUT_HI + k for k in range(16)})
    with torch.no_grad():
        r11 = m.forward(padded, stop_after_block=PHYS)[0].float()
    print(f"\n=== residual after block {PHYS} at AX-byte0 row pos {read_pos} "
          f"(tok {ctx[read_pos]}=REG_AX) ===", flush=True)
    row = r11[read_pos]
    for name, d in watch.items():
        v = float(row[d])
        if abs(v) > 0.05:
            print(f"  {name:16s} (dim {d}) = {v:+.3f}", flush=True)
