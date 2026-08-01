"""PART 1c — split the 108ms attention into GLOBAL O(S) code/mem CAM vs LOCAL heads.
Time each live block's attention (a) as-is, (b) with the global head forced LOCAL
(mask=all-False -> no full-history score), and diff.  This isolates the intrinsic
O(3976 code + heap) softmax cost.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
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
from c4_min.pf_speculative import draft_pf_program, build_code_vec, apply_overlay_window_fast
from c4_min.nibble_pure_forward_cached import BlockKVCacheBatched
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def med_time(fn, iters=15, warm=4):
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
    max_steps = int(os.environ.get('N_STEPS_CAP', '4000'))
    PROF_K = int(os.environ.get('PROF_K', '25'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code)+2, 64), recurrent_divmod=True, addr32=True, compute_mode='dense_kernel')
    D = sparse.embed.shape[1]; nblk = len(sparse.blocks); sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    H = sparse.blocks[0].attn.n_heads
    caches = [BlockKVCacheBatched(H, sparse.blocks[b].attn.head_dim, sparse.blocks[b].attn.alibi_slopes) for b in range(nblk)]
    for b in range(nblk):
        at = sparse.blocks[b].attn
        if getattr(at, '_drop_local_kv', False) and getattr(at, '_local_window', None):
            gmask = at._global_head_mask
            g_idx = [int(h) for h in range(H) if bool(gmask[h])]
            caches[b].set_head_groups(g_idx, int(at._local_window), content_bound=False, content_cR=None)
    store_log = draft.store_log
    code_vec = build_code_vec(code, L, D, torch.device(dev), dtype=sparse.embed.dtype)
    n_steps = draft.step_count
    PROF_STEP = int(os.environ.get('PROF_STEP', str(max(0, n_steps - PROF_K - 1))))
    lead_end = draft.win_starts[PROF_STEP]
    pos = 0
    while pos < lead_end:
        ce = min(pos + prime_chunk, lead_end); span_toks = draft.tokens[pos:ce]; S = len(span_toks)
        win_toks = torch.tensor([span_toks], device=dev); q_positions = torch.arange(pos, pos + S, device=dev)
        with torch.no_grad():
            x = sparse.embed[win_toks].clone()
            apply_overlay_window_fast(x, pos, L, store_log, code_vec, query_rows=[], code=code, code_off=draft.code_off)
            past = [caches[b].as_past_kv() for b in range(nblk)]
            _, new_kv = sparse.forward_hidden_cached(x, past_key_values=past, q_positions=q_positions, use_cache=True)
            for b in range(nblk):
                if new_kv[b] is None: continue
                K_all, Va, pa = new_kv[b]; caches[b].commit(K_all[:, :, -S:, :], Va[:, :, -S:, :], pa[-S:])
        pos = ce
    torch.cuda.synchronize()
    g_rows = max(c.size() for c in caches)

    step, end = PROF_STEP, min(PROF_STEP + PROF_K, n_steps)
    span_start = draft.win_starts[step]
    span_end = (draft.win_starts[end] + 1) if end < n_steps else len(draft.tokens)
    span_toks = draft.tokens[span_start:span_end]; S = len(span_toks)
    win_toks = torch.tensor([span_toks], device=dev)
    q_positions = torch.arange(span_start, span_start + S, device=dev)
    q_local = [draft.win_starts[s] - span_start for s in range(step, end)]
    Kq = len(q_local)
    x0 = sparse.embed[win_toks].clone()
    apply_overlay_window_fast(x0, span_start, L, store_log, code_vec, query_rows=q_local, code=code, code_off=draft.code_off)

    live_blocks = [b for b in range(nblk) if not getattr(sparse.blocks[b].attn, '_dead_block_fused', False)]

    # As-is attention (global + local) — the real config.
    def attn_real():
        with torch.no_grad():
            for b in live_blocks:
                sparse.blocks[b].attn.forward(x0, past_kv=caches[b].as_past_kv(), q_positions=q_positions, use_cache=True)
    ms_real = med_time(attn_real)

    # LOCAL-ONLY: force every global head to local (so it reads only the split's
    # LOCAL cache window, NOT the 152k global cache) -> removes the O(S) CAM score.
    # We do this by swapping in an all-False global mask + a split cache that has
    # NO global rows.  Rebuild the split with all-local so the global cache is empty.
    orig_masks = {b: sparse.blocks[b].attn._global_head_mask.clone() for b in live_blocks}
    local_caches = [BlockKVCacheBatched(H, sparse.blocks[b].attn.head_dim, sparse.blocks[b].attn.alibi_slopes) for b in range(nblk)]
    for b in range(nblk):
        at = sparse.blocks[b].attn
        # all-local mask (no global head)
        at._global_head_mask = torch.zeros_like(sparse.blocks[b].attn._global_head_mask)
        if getattr(at, '_drop_local_kv', False) and getattr(at, '_local_window', None):
            local_caches[b].set_head_groups([], int(at._local_window), content_bound=False, content_cR=None)
    # re-prime the local caches (cheap: only the window matters, but the split needs
    # the last-W local rows; prime the same lead so local windows are populated).
    pos = 0
    while pos < lead_end:
        ce = min(pos + prime_chunk, lead_end); st = draft.tokens[pos:ce]; SS = len(st)
        wt = torch.tensor([st], device=dev); qp = torch.arange(pos, pos + SS, device=dev)
        with torch.no_grad():
            x = sparse.embed[wt].clone()
            apply_overlay_window_fast(x, pos, L, store_log, code_vec, query_rows=[], code=code, code_off=draft.code_off)
            past = [local_caches[b].as_past_kv() for b in range(nblk)]
            _, new_kv = sparse.forward_hidden_cached(x, past_key_values=past, q_positions=qp, use_cache=True)
            for b in range(nblk):
                if new_kv[b] is None: continue
                K_all, Va, pa = new_kv[b]; local_caches[b].commit(K_all[:, :, -SS:, :], Va[:, :, -SS:, :], pa[-SS:])
        pos = ce
    torch.cuda.synchronize()

    def attn_local_only():
        with torch.no_grad():
            for b in live_blocks:
                sparse.blocks[b].attn.forward(x0, past_kv=local_caches[b].as_past_kv(), q_positions=q_positions, use_cache=True)
    ms_local = med_time(attn_local_only)

    print(f'\n[prof1c] === ATTENTION SPLIT (K={Kq}, S={S}, global_cache={g_rows}) ===', flush=True)
    print(f'  attention REAL (global O(S) CAM + local)  : {ms_real:7.2f} ms  ({ms_real/Kq:.3f} ms/step)', flush=True)
    print(f'  attention LOCAL-ONLY (window={window} only): {ms_local:7.2f} ms  ({ms_local/Kq:.3f} ms/step)', flush=True)
    print(f'  => GLOBAL O(S) code/mem CAM cost          : {ms_real-ms_local:7.2f} ms  '
          f'({100*(ms_real-ms_local)/max(ms_real,1e-9):.0f}% of attention, {(ms_real-ms_local)/Kq:.3f} ms/step)', flush=True)
    print(f'  global cache rows = {g_rows} (of which {draft.code_off} = code frames)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
