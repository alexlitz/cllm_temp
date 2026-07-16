import sys, time
sys.path.insert(0, '.')
import c4_min.nibble_pure_forward as _PF, c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0; _PFC.SP_INIT = 0xF0
from c4_min.nibble_pure_forward_complete import build_pure_forward_complete_model
from c4_min.sparse_forward import SparseTransformer
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min.run_1096_pure_forward import bytecode_to_isa
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs

out = open('/tmp/trace_probe.txt', 'w')
def w(s):
    out.write(s + "\n"); out.flush()

import torch
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model, L = build_pure_forward_complete_model(code_size=64, include_bitwise=False, include_divmod=False)
sp = SparseTransformer(model, compute_mode='dense_kernel').to(dev)
w(f"device={dev}")
tests = generate_test_programs()
idx = 450  # loop_sum_0 sum(1..18) exp 171
src, exp, desc = tests[idx]
code = bytecode_to_isa(compile_c(src)[0])
t0 = time.time()
# run token-by-token cached driver to COMPLETION (ground truth for the model)
tr = run_pure_forward_cached(sp, L, code, max_steps=600, mask=0xFFFFFFFF, evict=True,
                             prune_interval=120)
w(f"idx {idx} {desc}: exp={exp} DRIVER final={tr[-1] if tr else None} steps={len(tr)} "
  f"halted={(len(tr) < 600)} wall={time.time()-t0:.0f}s")
out.close()
print("DONE")
