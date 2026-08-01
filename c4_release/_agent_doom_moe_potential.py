"""Measure the block-MoE POTENTIAL: FFN+attn over the live-union (57 blocks) vs the
full 242, at K=25 span.  This is the ms/step floor if doom adopted self-emu's
live-union block-MoE.  (Correctness of wiring it into the persistent-KV verify path
is assessed separately; this is the achievable-compute measurement.)
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('C4_KV_STACK', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def med_time(fn, iters=12, warm=4):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort(); return ts[len(ts)//2]


def main():
    dev = 'cuda:0'; torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '4000'))
    K = int(os.environ.get('K', '25'))
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code)+2, 64), recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    D = sparse.embed.shape[1]; nblk = len(sparse.blocks); sparse = sparse.to(dev)
    sparse.materialize_dense(device=dev)
    from c4_min.step_block_skip import build_live_index
    live_index = build_live_index(sparse, L)
    name_to_op = {v: k for k, v in isa.NAMES.items()}

    # pick a representative span's live-union
    import statistics
    n_steps = draft.step_count
    s0 = n_steps - K - 1
    full = set(range(nblk))
    u = set()
    for s in range(s0, s0 + K):
        opcode = name_to_op.get(draft.frames[s]['op'])
        live = live_index.get(opcode)
        if live is None: u = full; break
        u |= set(live)
    union = sorted(u)
    S = K * 30
    x = torch.randn(1, S, D, device=dev)
    print(f'[moe-pot] span K={K} S={S}: live-union={len(union)}/{nblk} blocks', flush=True)

    def ffn_all():
        with torch.no_grad():
            h = x
            for b in range(nblk):
                h = sparse.blocks[b].ffn.forward(h)
    def ffn_union():
        with torch.no_grad():
            h = x
            for b in union:
                h = sparse.blocks[b].ffn.forward(h)
    ms_all = med_time(ffn_all)
    ms_union = med_time(ffn_union)
    print(f'[moe-pot] FFN over ALL {nblk}   blocks: {ms_all:6.2f} ms ({ms_all/K:.4f} ms/step)', flush=True)
    print(f'[moe-pot] FFN over UNION {len(union)} blocks: {ms_union:6.2f} ms ({ms_union/K:.4f} ms/step)  '
          f'-> {ms_all/ms_union:.2f}x fewer', flush=True)

    # graphed union FFN (the megakernel of only the live blocks)
    static_in = torch.zeros(1, S, D, device=dev)
    def run_union(inp):
        h = inp
        for b in union:
            h = sparse.blocks[b].ffn.forward(h)
        return h
    st = torch.cuda.Stream(device=dev); st.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(st):
        for _ in range(3):
            with torch.no_grad(): run_union(static_in)
    torch.cuda.current_stream(dev).wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad(): so = run_union(static_in)
    def ffn_union_graphed():
        static_in.copy_(x); g.replay()
    ms_ug = med_time(ffn_union_graphed)
    print(f'[moe-pot] FFN union GRAPHED     : {ms_ug:6.2f} ms ({ms_ug/K:.4f} ms/step)  '
          f'-> {ms_all/ms_ug:.2f}x vs all-eager', flush=True)
    print(f'\n[moe-pot] block-MoE + graph FFN floor ~= {ms_ug/K:.4f} ms/step '
          f'(attention + overlay + decode + eviction sit on top).', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
