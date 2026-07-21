"""Ad-hoc verification driver (NOT a committed test) for the WIP lean muldiv.

Builds the SUBSET_BITWISE lean model ONCE on cuda:1 and checks:
  1. stock run_program_lean (depth-1 STACK0) on the muldiv (expected: may diverge
     because the subroutine reaches stack depth 2-3),
  2. the STACK-AWARE run_program_lean_stack (depth-N),
  3. speculation (speculative_run_lean_stack) — forwards saved, byte-exact.
"""
import os, sys, time

os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_agent")
DEV = "cuda:1"

import torch
from c4_min import isa
from c4_min import lean_subroutine_muldiv as M
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF
from c4_min import qwen_lean_stack_driver as SD

t0 = time.time()
vm = Q.build(code_size=64, subset=Q.SUBSET_BITWISE)
lean = LF.LeanQwenVM.from_full_vm(vm, device=DEV)
print(f"[build] lean SUBSET_BITWISE on {DEV} in {time.time()-t0:.1f}s "
      f"(n_layers={vm.n_layers})", flush=True)

CASES = [
    ("MUL8", M.program_mul8(12, 11), (12 * 11) & 0xFF),
    ("MUL8", M.program_mul8(200, 3), (200 * 3) & 0xFF),
    ("DIV8", M.program_div8(200, 17), (200 // 17) & 0xFF),
    ("MOD8", M.program_mod8(100, 7), (100 % 7) & 0xFF),
]


def stock(prog):
    r = LF.run_program_lean(lean, prog, max_steps=5000)
    return r["ax_trace"][-1], r["exact"], r["steps"]


def stackdrv(prog):
    r = SD.run_program_lean_stack(lean, prog, max_steps=5000)
    return r["ax_trace"][-1], r["exact"], r["steps"], r["forwards"]


print("\n=== STOCK run_program_lean (depth-1 STACK0) ===", flush=True)
for name, prog, want in CASES:
    try:
        got, exact, n = stock(prog)
        print(f"  {name}: last_ax={got} want={want} exact_vs_isa={exact} steps={n} "
              f"-> {'OK' if got == want else 'WRONG'}", flush=True)
    except Exception as e:
        print(f"  {name}: EXC {type(e).__name__}: {e}", flush=True)

print("\n=== STACK-AWARE run_program_lean_stack (depth-N) ===", flush=True)
for name, prog, want in CASES:
    got, exact, n, fwd = stackdrv(prog)
    print(f"  {name}: last_ax={got} want={want} exact_vs_isa={exact} steps={n} "
          f"forwards={fwd} -> {'OK' if got == want and exact else 'WRONG'}", flush=True)

print("\n=== SPECULATION speculative_run_lean_stack ===", flush=True)
for name, prog, want in CASES:
    t = time.time()
    r = SD.speculative_run_lean_stack(lean, prog, block_steps=256, max_steps=5000)
    dt = time.time() - t
    print(f"  {name}: status={r.status} exact={r.exact} steps={r.steps} "
          f"forwards={r.forwards} naive={r.naive_forwards} "
          f"speedup={r.speedup:.1f}x last_ax={r.ax_trace[-1]} want={want} "
          f"wall={dt*1000:.0f}ms", flush=True)
