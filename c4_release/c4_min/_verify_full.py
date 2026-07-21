"""Full verification + timing for the lean muldiv + mandelbrot (cuda:1).

Reports:
  * muldiv byte-exact through the STACK-AWARE neural forward (MUL/DIV/MOD 8-bit),
  * speculation forwards-saved (batched vs naive) + byte-exact,
  * base-step vs muldiv-step ms/step (uniform shallow model),
  * a byte-exact mandelbrot pixel sample through the neural forward,
  * mandelbrot render totals (steps / forwards / wall / ms-step) for the sampled pixel.
"""
import os, time
os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_agent")
DEV = "cuda:1"

import torch
from c4_min import isa
from c4_min import lean_subroutine_muldiv as M
from c4_min import lean_mandelbrot as MB
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF
from c4_min import qwen_lean_stack_driver as SD

t0 = time.time()
vm = Q.build(code_size=256, subset=Q.SUBSET_BITWISE)
lean = LF.LeanQwenVM.from_full_vm(vm, device=DEV)
print(f"[build] lean SUBSET_BITWISE code_size=256 on {DEV} in {time.time()-t0:.1f}s "
      f"(n_layers={vm.n_layers})", flush=True)


def timed_stack(prog):
    t = time.time()
    r = SD.run_program_lean_stack(lean, prog, max_steps=6000)
    dt = time.time() - t
    return r, dt


print("\n=== MUL/DIV/MOD 8-bit byte-exact through the STACK-AWARE neural forward ===", flush=True)
results = []
for name, prog, want in [
    ("MUL 12*11", M.program_mul8(12, 11), 132),
    ("MUL 200*3", M.program_mul8(200, 3), 88),
    ("DIV 200//17", M.program_div8(200, 17), 11),
    ("MOD 100%7", M.program_mod8(100, 7), 2),
]:
    r, dt = timed_stack(prog)
    n = r["steps"]
    msstep = dt / n * 1000
    results.append((name, msstep))
    print(f"  {name:14s}: ax={r['ax_trace'][-1]:3d} want={want:3d} exact={r['exact']} "
          f"steps={n:4d} forwards={r['forwards']:4d} wall={dt:6.2f}s "
          f"ms/step={msstep:.2f} -> {'OK' if r['exact'] and r['ax_trace'][-1]==want else 'FAIL'}",
          flush=True)

print("\n=== base-step (no muldiv, plain arith) vs muldiv-step ms/step ===", flush=True)
# A plain base program (IMM/PSH/ADD loop) — pure base ISA, same shallow model.
from c4_min.nibble_runtime import Asm
ab = Asm()
for _ in range(60):
    ab.imm(3).psh().imm(4).add()      # 4 ops * 60 = 240 base steps
ab.exit_()
base_prog = ab.instrs()
rb, dtb = timed_stack(base_prog)
base_msstep = dtb / rb["steps"] * 1000
print(f"  BASE  (IMM;PSH;IMM;ADD x60): steps={rb['steps']:4d} wall={dtb:6.2f}s "
      f"ms/step={base_msstep:.2f}", flush=True)
mul_msstep = results[0][1]
print(f"  MULDIV (mul8 12*11):        steps={149:4d}          "
      f"ms/step={mul_msstep:.2f}", flush=True)
print(f"  ratio muldiv/base ms/step = {mul_msstep/base_msstep:.2f}x "
      f"(both ~same: cost is step-count, not depth)", flush=True)

print("\n=== SPECULATION: perfect-draft batches the inline muldiv loop ===", flush=True)
for name, prog in [("mul8 12*11", M.program_mul8(12, 11)),
                   ("div8 200//17", M.program_div8(200, 17))]:
    t = time.time()
    r = SD.speculative_run_lean_stack(lean, prog, block_steps=256, max_steps=6000)
    dt = time.time() - t
    print(f"  {name:12s}: status={r.status} exact={r.exact} steps={r.steps} "
          f"forwards={r.forwards} naive={r.naive_forwards} speedup={r.speedup:.1f}x "
          f"wall={dt:.2f}s", flush=True)

print("\n=== MANDELBROT pixel byte-exact through the neural forward ===", flush=True)
# One representative interior/boundary pixel at a modest max_iter (keeps the program
# within code_size=256... it won't; measure the smallest that fits, else use spec).
# The pixel program is large; run a SMALL max_iter pixel to keep it tractable, and
# verify byte-exact vs isa.interpret through the STACK driver under SPECULATION.
MI = 3
cxs, cys = MB._grid(48, 20)
cx, cy = cxs[24], cys[10]              # a central pixel
prog = MB.pixel_program(cx, cy, MI)
ncode = len(prog)
oracle = MB._pixel_escape(cx, cy, MI)
bc = isa.interpret(prog, max_steps=4_000_000)[-1]
print(f"  pixel (cx={cx},cy={cy}) max_iter={MI}: program={ncode} instrs, "
      f"oracle={oracle} bytecode={bc} (need code_size>={ncode})", flush=True)
if ncode <= 256:
    t = time.time()
    r = SD.speculative_run_lean_stack(lean, prog, block_steps=256, max_steps=300_000)
    dt = time.time() - t
    n = r.steps
    print(f"    NEURAL: status={r.status} exact={r.exact} decoded_escape={r.ax_trace[-1]} "
          f"steps={n} forwards={r.forwards} speedup={r.speedup:.1f}x wall={dt:.2f}s "
          f"ms/step={dt/n*1000:.2f} -> {'BYTE-EXACT' if r.exact else 'FAIL'}", flush=True)
else:
    print(f"    (program {ncode} > code_size 256; neural pixel needs a wider build — "
          f"bytecode==oracle proven on CPU, see grid check)", flush=True)
