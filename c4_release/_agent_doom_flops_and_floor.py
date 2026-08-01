"""PART 2 analysis — (a) direct-CAM analytic O(S)->O(1) reduction for the doom
global CAM, (b) the FFN floor (best-case graphed FFN-only ms/step, the compute
floor doom cannot go below without shrinking the model), (c) the global-cache
growth that caps K.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('C4_EXACT_EVICT', '1')
os.environ.setdefault('C4_MEM_EFF', '2000000')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from c4_min.direct_cam_read import attention_flop_report
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
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)

    # (a) direct-CAM analytic reduction over the value CAM (mem/pop/lev) reads.
    rep = attention_flop_report(draft)
    print(f'[floor] === DIRECT-CAM analytic reduction (value CAM reads, cap={cap} steps) ===', flush=True)
    print(f'[floor]   n_reads={rep["n_reads"]}  mean_cache_depth_K={rep["mean_K"]:.0f}  '
          f'total_softmax_row_scores={rep["total_softmax_row_scores"]:,}  '
          f'total_direct_gathers={rep["total_direct_row_gathers"]:,}  '
          f'reduction={rep["total_softmax_row_scores"]/max(rep["total_direct_row_gathers"],1):.0f}x', flush=True)
    print(f'[floor]   (this is the value-band read; the code-fetch + PC-transition global '
          f'head is NOT covered by resolve_load_rows.)', flush=True)

    # store-log growth (the global cache = heap store rows the CAM scores O(S)).
    n_stores = len(draft.store_log)
    print(f'[floor]   store_log entries (global-cache store rows at cap={cap}): {n_stores:,}  '
          f'code_frames={draft.code_off}  total_tokens={len(draft.tokens):,}', flush=True)

    # (b) the FFN floor: build doom, time the FFN-chain ONLY (graphed) at small K.
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code)+2, 64), recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    D = sparse.embed.shape[1]; nblk = len(sparse.blocks); sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    sparse.materialize_dense(device=dev)

    for K in (25, 64, 128):
        S = K * 30  # ~30 tokens/frame
        x = torch.randn(1, S, D, device=dev)
        # FFN-only chain over the whole span (all 242 blocks)
        def ffn_chain():
            with torch.no_grad():
                h = x
                for b in range(nblk):
                    h = sparse.blocks[b].ffn.forward(h)
        ms_ffn = med_time(ffn_chain)
        # graphed FFN chain (capture the whole 242-block FFN as one graph over the span)
        static_in = torch.zeros(1, S, D, device=dev)
        def run_all(inp):
            h = inp
            for b in range(nblk):
                h = sparse.blocks[b].ffn.forward(h)
            return h
        st = torch.cuda.Stream(device=dev); st.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(st):
            for _ in range(3):
                with torch.no_grad(): run_all(static_in)
        torch.cuda.current_stream(dev).wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad(): static_out = run_all(static_in)
        def ffn_graphed():
            static_in.copy_(x); g.replay()
        ms_graph = med_time(ffn_graphed)
        print(f'[floor]   K={K:3d} S={S}: FFN-chain eager={ms_ffn:6.2f}ms ({ms_ffn/K:.3f} ms/step)  '
              f'graphed={ms_graph:6.2f}ms ({ms_graph/K:.4f} ms/step)  graph_speedup={ms_ffn/ms_graph:.2f}x', flush=True)
        del g, static_in, static_out
        torch.cuda.empty_cache()

    print(f'\n[floor] The graphed FFN-only ms/step is the COMPUTE FLOOR for doom '
          f'(242 dense SwiGLU blocks, dim={D}) — attention + eviction + overlay + '
          f'decode sit ON TOP of this.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
