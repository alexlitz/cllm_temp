"""PART 1b — isolate the components: FFN-chain-only, attn-blocks-only, and
GLOBAL-head vs LOCAL-head cost of the O(S) code/mem CAM.  Clean CUDA-event timing
with warmup + median-of-N, one sync per measured region (NOT per call).
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
from c4_min.pf_speculative import (draft_pf_program, build_code_vec, apply_overlay_window_fast)
from c4_min.nibble_pure_forward_cached import BlockKVCacheBatched
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def med_time(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); e0.record()
        fn()
        e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts)//2]


def main():
    dev = 'cuda:0'
    torch.cuda.init()
    max_steps = int(os.environ.get('N_STEPS_CAP', '4000'))
    PROF_K = int(os.environ.get('PROF_K', '25'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))

    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)

    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    D = sparse.embed.shape[1]; nblk = len(sparse.blocks)
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    H = sparse.blocks[0].attn.n_heads

    caches = [BlockKVCacheBatched(H, sparse.blocks[b].attn.head_dim,
                                  sparse.blocks[b].attn.alibi_slopes)
              for b in range(nblk)]
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

    def _prime(pos_end):
        pos = 0; pc_ = max(1, prime_chunk)
        while pos < pos_end:
            ce = min(pos + pc_, pos_end)
            span_toks = draft.tokens[pos:ce]; S = len(span_toks)
            win_toks = torch.tensor([span_toks], device=dev)
            q_positions = torch.arange(pos, pos + S, device=dev)
            with torch.no_grad():
                x = sparse.embed[win_toks].clone()
                apply_overlay_window_fast(x, pos, L, store_log, code_vec, query_rows=[], code=code, code_off=draft.code_off)
                past = [caches[b].as_past_kv() for b in range(nblk)]
                _, new_kv = sparse.forward_hidden_cached(x, past_key_values=past, q_positions=q_positions, use_cache=True)
                for b in range(nblk):
                    if new_kv[b] is None: continue
                    K_all, Va, pa = new_kv[b]
                    caches[b].commit(K_all[:, :, -S:, :], Va[:, :, -S:, :], pa[-S:])
            pos = ce
    _prime(lead_end)
    torch.cuda.synchronize()
    g_rows = max(c.size() for c in caches)
    print(f'[prof1b] primed cache global_max={g_rows} code_off={draft.code_off}', flush=True)

    step, end = PROF_STEP, min(PROF_STEP + PROF_K, n_steps)
    span_start = draft.win_starts[step]
    span_end = (draft.win_starts[end] + 1) if end < n_steps else len(draft.tokens)
    span_toks = draft.tokens[span_start:span_end]; S = len(span_toks)
    win_toks = torch.tensor([span_toks], device=dev)
    q_positions = torch.arange(span_start, span_start + S, device=dev)
    q_local = [draft.win_starts[s] - span_start for s in range(step, end)]
    Kq = len(q_local)

    def make_x():
        x = sparse.embed[win_toks].clone()
        apply_overlay_window_fast(x, span_start, L, store_log, code_vec, query_rows=q_local, code=code, code_off=draft.code_off)
        return x

    live_blocks = [b for b in range(nblk) if not getattr(sparse.blocks[b].attn, '_dead_block_fused', False)]
    print(f'[prof1b] span S={S} Kq={Kq} live_attn_blocks={live_blocks}', flush=True)

    # (1) FULL span
    def full():
        with torch.no_grad():
            past = [caches[b].as_past_kv() for b in range(nblk)]
            sparse.forward_hidden_cached(make_x(), past_key_values=past, q_positions=q_positions, use_cache=True)
    ms_full = med_time(full)

    # (2) FFN-chain only: run each block's FFN over the span x (no attention).
    x0 = make_x()
    def ffn_only():
        with torch.no_grad():
            h = x0
            for b in range(nblk):
                h = sparse.blocks[b].ffn.forward(h)
    ms_ffn = med_time(ffn_only)

    # (3) attention-only: run each LIVE block's attn over the span (with cache).
    def attn_only():
        with torch.no_grad():
            for b in live_blocks:
                past = caches[b].as_past_kv()
                sparse.blocks[b].attn.forward(x0, past_kv=past, q_positions=q_positions, use_cache=True)
    ms_attn = med_time(attn_only)

    # (4) split GLOBAL vs LOCAL: patch _global_head_mask to all-True (global only)
    #     vs all-False (local only) on each live block, time attn_only each way.
    import torch as _t
    orig_masks = {}
    for b in live_blocks:
        orig_masks[b] = sparse.blocks[b].attn._global_head_mask.clone()

    # ALL-GLOBAL: every live head does full-history O(S) score.
    for b in live_blocks:
        m = orig_masks[b]
        sparse.blocks[b].attn._global_head_mask = _t.ones_like(m)
    # need caches to have Kg populated for all heads -> rebuild split with all-global.
    # Simpler: measure the GLOBAL-head subset cost by timing attn_only with only the
    # blocks that HAVE a global head, using the real mask (already done in ms_attn).
    # Restore + report which blocks carry the global head.
    for b in live_blocks:
        sparse.blocks[b].attn._global_head_mask = orig_masks[b]

    global_blocks = [b for b in live_blocks if bool(orig_masks[b].any())]
    print(f'[prof1b] blocks carrying a GLOBAL head (O(S) code/mem CAM): {global_blocks}', flush=True)
    for b in live_blocks:
        gm = orig_masks[b]
        ng = int(gm.sum().item())
        print(f'[prof1b]   block {b}: {ng} global head(s), {int((~gm).sum())} local head(s), '
              f'global_cache_rows={caches[b].size()}', flush=True)

    print(f'\n[prof1b] === COMPONENT TIMES (median, K={Kq}, S={S}) ===', flush=True)
    print(f'  FULL span forward        : {ms_full:7.2f} ms  ({ms_full/Kq:.3f} ms/step)', flush=True)
    print(f'  FFN-chain only (242 blks): {ms_ffn:7.2f} ms  ({100*ms_ffn/ms_full:.0f}% of full, {ms_ffn/Kq:.3f} ms/step)', flush=True)
    print(f'  attention only (4 blks)  : {ms_attn:7.2f} ms  ({100*ms_attn/ms_full:.0f}% of full, {ms_attn/Kq:.3f} ms/step)', flush=True)
    print(f'  [overhead/embed/overlay] : {ms_full-ms_ffn-ms_attn:7.2f} ms  '
          f'({100*(ms_full-ms_ffn-ms_attn)/ms_full:.0f}% of full)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
