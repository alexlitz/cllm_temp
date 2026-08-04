"""_agent_doom_camvec.py — the REAL doom.c stream on the FULL lean lever stack,
measuring the host-sync-elimination (C4_DIRECT_CAM_VEC) + megakernel (C4_GRAPH_MEGAKERNEL)
deltas, byte-exact vs the perfect draft, at K=512.

Env: N_STEPS_CAP (default 3000), K (default 512), VEC (0/1, default 1),
     GRAPH (0/1, default 0).  Reports ms/step + steps/sec + frame@6.89M/s.
LEAN STREAMING (C4_PF_CFM=1). CUDA_VISIBLE_DEVICES set by the caller.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('C4_EXACT_EVICT', '1')
os.environ.setdefault('C4_MEM_EFF', '2000000')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# the host-sync-elimination + composed lean levers:
for f, d in [('C4_DIRECT_CAM_BATCHED', '1'), ('C4_DIRECT_LOCAL_CAM', '1'),
             ('C4_BANDED_LOCAL_ATTN', '1'), ('C4_DIRECT_CAM_LIVE_LOCAL', '1'),
             ('C4_FROZEN_ROW_SKIP', '1'), ('C4_BATCHED_BLOCK_SKIP', '1'),
             ('C4_DEAD_BLOCK_FUSION', '1'), ('C4_OVERLAY_BATCHED', '1'),
             ('C4_BATCHED_DECODE', '1')]:
    os.environ.setdefault(f, d)
os.environ['C4_DIRECT_CAM_VEC'] = os.environ.get('VEC', '1')
if os.environ.get('GRAPH', '0') == '1':
    os.environ['C4_GRAPH_MEGAKERNEL'] = '1'
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def _mem_gb():
    with open('/proc/meminfo') as f:
        for l in f:
            if l.startswith('MemAvailable:'):
                return float(l.split()[1]) / 1e6
    return 1e9


def build_doom(dev, cap, window):
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                              stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code) + 2, 64),
                                             recurrent_divmod=True, addr32=True,
                                             compute_mode='dense_kernel')
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    return sparse, L, code, draft, fio


def main():
    if _mem_gb() < 25.0:
        raise SystemExit(f'[MEM-GUARD] {_mem_gb():.1f}GB<25GB STOP')
    dev = 'cuda:0'; torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '3000'))
    K = int(os.environ.get('K', '512'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    reps = int(os.environ.get('REPS', '2'))
    vec = os.environ['C4_DIRECT_CAM_VEC']; graph = os.environ.get('C4_GRAPH_MEGAKERNEL', '0')
    print(f'[doom] cap={cap} K={K} window={window} VEC={vec} GRAPH={graph}', flush=True)
    sparse, L, code, draft, fio = build_doom(dev, cap, window)
    print(f'[doom] instrs={len(code)} dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} '
          f'draft_steps={draft.step_count} code_off={draft.code_off}', flush=True)

    def run():
        stats = {}
        vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                           mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                           min_block_steps=4, prime_chunk=2048)
        return vr, stats

    # warmup
    vr, st = run()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        vr, st = run()
    torch.cuda.synchronize()
    wall = (time.time() - t0) / reps
    ms = wall / max(draft.step_count, 1) * 1e3
    sps = draft.step_count / wall
    print(f'[doom] matched={vr.all_matched} accepted={vr.accepted_steps}/{vr.total_steps} '
          f'forwards={vr.forwards} eff_K={st.get("effective_block_steps")} '
          f'peak_vram={st.get("peak_vram_gb",0):.1f}GB final_ax={vr.decoded_final_ax}', flush=True)
    if vr.first_mismatch:
        print(f'[doom] first_mismatch: {vr.first_mismatch}', flush=True)
    print(f'[doom] ms/step={ms:.3f}  steps/sec={sps:.0f}  frame@6.89M={6.89e6/sps:.0f}s '
          f'({6.89e6/sps/3600:.1f}h)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
