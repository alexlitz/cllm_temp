"""Build the FULL doom model with cfm ON — confirm dim is FIXED (small-model dim,
NOT ~25,731) and it fits RAM+VRAM.  Report dim + VRAM + build time.
Additive; golden 069cc32f unchanged."""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time, resource
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ['C4_PF_CFM']='1'
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from src.compiler import compile_c
from c4_min import isa
from c4_min.run_1096_pure_forward import bytecode_to_isa
from c4_min.compact_alloc import build_compact_sparse_streaming

src = Path('/home/alexlitz/Documents/misc/c4_doom/doom.c').read_text()
bc, data = compile_c(src)
code = bytecode_to_isa(bc)
cs = max(len(code)+2, 64)
print(f'[cfm-build] doom instrs={len(code)} code_size={cs}', flush=True)

t0=time.time()
sparse, L, stats = build_compact_sparse_streaming(code_size=cs, recurrent_divmod=True, compute_mode='dense_kernel')
build_wall=time.time()-t0
rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
dim = sparse.embed.shape[1]
print(f'[cfm-build] BUILT dim={dim} n_blocks={len(sparse.blocks)} '
      f'build_wall={build_wall:.0f}s peak_RSS={rss_gb:.1f}GB', flush=True)

# move to VRAM, measure
dev='cuda:0' if torch.cuda.is_available() else 'cpu'
vram_gb=0.0
if dev!='cpu':
    torch.cuda.reset_peak_memory_stats()
    sparse=sparse.to(dev)
    vram_gb=torch.cuda.max_memory_allocated()/1e9
    print(f'[cfm-build] on {dev}: model VRAM={vram_gb:.2f}GB', flush=True)
print(f'[cfm-build] SUMMARY: dim={dim} (FIXED, baked-would-be ~25,731) '
      f'RSS={rss_gb:.1f}GB VRAM~{vram_gb:.1f}GB build={build_wall:.0f}s '
      f'vs baked >110GB RAM (never finished)', flush=True)
