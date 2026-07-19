"""One-shot probe: verify rec_fib id=734 (8369 steps) on the LEAN sparse model at a
given block_steps; report forwards / matched / final AX / peak VRAM.  Run per-bs in
a fresh process so a hard CUDA OOM can't poison the others."""
import sys, time
import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xFC; _PFC.SP_INIT = 0xFC
from c4_min.compact_alloc import load_sparse_transformer
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.run_corpus_stacked import bytecode_to_isa
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

bs = int(sys.argv[1])
which = int(sys.argv[2]) if len(sys.argv) > 2 else 734
model_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/fs_lean_sparse.pt"
dev = "cuda:0"
lean, L = load_sparse_transformer(model_path, compute_mode="dense_kernel")
lean = lean.to(dev)
tests = generate_test_programs()
src, exp, desc = tests[which]
code = bytecode_to_isa(compile_c(src)[0])
draft = draft_pf_program(code, max_steps=300000)
torch.cuda.reset_peak_memory_stats(dev)
t = time.monotonic()
try:
    vr = verify_blocks(lean, L, code, draft, block_steps=bs, device=dev,
                       evict=True, prune_interval=120)
    dt = time.monotonic() - t
    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    print(f"id={which} steps={draft.step_count} bs={bs}: forwards={vr.forwards} "
          f"matched={vr.all_matched} finalAX={vr.decoded_final_ax} exp={exp} "
          f"peakVRAM={peak:.2f}GB {dt:.1f}s")
except torch.cuda.OutOfMemoryError:
    print(f"id={which} steps={draft.step_count} bs={bs}: OOM")
