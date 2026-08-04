"""Profile the DOOM step (K=512, full lean stack + cam_vec) to find doom's wall."""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('C4_EXACT_EVICT', '1')
os.environ.setdefault('C4_MEM_EFF', '2000000')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
for f, d in [('C4_DIRECT_CAM_BATCHED', '1'), ('C4_DIRECT_LOCAL_CAM', '1'),
             ('C4_BANDED_LOCAL_ATTN', '1'), ('C4_DIRECT_CAM_LIVE_LOCAL', '1'),
             ('C4_FROZEN_ROW_SKIP', '1'), ('C4_BATCHED_BLOCK_SKIP', '1'),
             ('C4_DEAD_BLOCK_FUSION', '1'), ('C4_OVERLAY_BATCHED', '1'),
             ('C4_BATCHED_DECODE', '1'), ('C4_DIRECT_CAM_VEC', '1')]:
    os.environ.setdefault(f, d)
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment
DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def main():
    dev = 'cuda:0'; torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '2000')); K = int(os.environ.get('K', '512'))
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code)+2, 64), recurrent_divmod=True, addr32=True, compute_mode='dense_kernel')
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    install_local_attention(sparse, window=96, drop_local_kv=True, content_bound_global=False, verbose=False)
    ns = draft.step_count
    print(f'[dprof] steps={ns} K={K} instrs={len(code)}', flush=True)
    # warmup
    verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True, mask=0xFFFFFFFF, fast=True)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True, mask=0xFFFFFFFF, fast=True)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    print(f'[dprof] matched={vr.all_matched} forwards={vr.forwards}', flush=True)
    print('[dprof] TOP by CUDA time:', flush=True)
    rows = sorted(ka, key=lambda e: getattr(e, 'self_device_time_total', 0), reverse=True)
    tot_cuda = sum(getattr(e, 'self_device_time_total', 0) for e in ka)
    for e in rows[:14]:
        c = getattr(e, 'self_device_time_total', 0)
        print(f'    {e.key[:40]:40s} cnt={e.count:6d} cuda={c/1e3:8.2f}ms ({100*c/max(tot_cuda,1):.0f}%)', flush=True)
    print(f'[dprof] total self CUDA = {tot_cuda/1e3:.1f}ms over {vr.forwards} forwards', flush=True)
    print('[dprof] TOP by CPU time:', flush=True)
    rows2 = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)
    for e in rows2[:10]:
        print(f'    {e.key[:40]:40s} cnt={e.count:6d} cpu={e.self_cpu_time_total/1e3:8.2f}ms', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
