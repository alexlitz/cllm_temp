"""PROFILE the doom forward with the FULL composed config (direct-CAM + sparse-FFN +
local-attn) and decompose attn-vs-FFN with/without the BANDED local kernel.

This is the honest attention-%-collapse measurement: the GLOBAL/CAM heads are already
O(1) direct-CAM gathers (direct_cam_batched), so the ONLY remaining O(S^2) attention is
the LOCAL heads' masked-full score — exactly what C4_BANDED_LOCAL_ATTN replaces with an
O(S*W) band.  Times one representative big-K span, splitting the wall into
(local-attn, global-direct-CAM, FFN) via CUDA events, at PROF_K (span size) and at
several S to show FLAT-in-S.

Env: N_STEPS_CAP (draft cap), PROF_K (span K), C4_BANDED_LOCAL_ATTN (0/1),
C4_LOCAL_WINDOW (W).  Mirrors _agent_doom_speed_floor.py's install order exactly.
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
from c4_min.pf_speculative import (draft_pf_program, _draft_cmp32, build_code_vec,
                                   apply_overlay_window_fast)
from c4_min.nibble_pure_forward_cached import BlockKVCacheBatched
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def _ev():
    return torch.cuda.Event(enable_timing=True)


def main():
    dev = 'cuda:0'
    torch.cuda.init()
    banded = os.environ.get('C4_BANDED_LOCAL_ATTN', '0') == '1'
    cap = int(os.environ.get('N_STEPS_CAP', '4000'))
    PROF_K = int(os.environ.get('PROF_K', '200'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))
    print(f'[bprof] BANDED={banded} cap={cap} PROF_K={PROF_K} window={window}', flush=True)

    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    print(f'[bprof] draft steps={draft.step_count} code_off={draft.code_off} '
          f'n_tokens={len(draft.tokens)}', flush=True)

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
    from c4_min.block_sparse_ffn import install_block_sparse_ffn
    install_block_sparse_ffn(sparse, mode='coo', verbose=False)
    from c4_min import direct_cam_batched as DCB
    DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)

    H = sparse.blocks[0].attn.n_heads
    caches = [BlockKVCacheBatched(H, sparse.blocks[b].attn.head_dim,
                                  sparse.blocks[b].attn.alibi_slopes) for b in range(nblk)]
    for b in range(nblk):
        at = sparse.blocks[b].attn
        if getattr(at, '_drop_local_kv', False) and getattr(at, '_local_window', None):
            gmask = at._global_head_mask
            g_idx = [int(h) for h in range(H) if bool(gmask[h])]
            caches[b].set_head_groups(g_idx, int(at._local_window), content_bound=False, content_cR=None)
    store_log = draft.store_log
    code_vec = build_code_vec(code, L, D, torch.device(dev), dtype=sparse.embed.dtype)
    n_steps = draft.step_count

    # live blocks = the direct-CAM / local blocks whose attn actually runs.
    live_blocks = [b for b in range(nblk) if not getattr(sparse.blocks[b].attn, '_dead_block_fused', False)]
    print(f'[bprof] live_attn_blocks={live_blocks}', flush=True)

    def prime(pos_end):
        pos = 0; pc_ = max(1, prime_chunk)
        while pos < pos_end:
            ce = min(pos + pc_, pos_end); S = ce - pos
            win_toks = torch.tensor([draft.tokens[pos:ce]], device=dev)
            q_positions = torch.arange(pos, pos + S, device=dev)
            with torch.no_grad():
                x = sparse.embed[win_toks].clone()
                apply_overlay_window_fast(x, pos, L, store_log, code_vec, query_rows=[],
                                          code=code, code_off=draft.code_off)
                past = [caches[b].as_past_kv() for b in range(nblk)]
                _, new_kv = sparse.forward_hidden_cached(x, past_key_values=past,
                                                         q_positions=q_positions, use_cache=True)
                for b in range(nblk):
                    if new_kv[b] is None: continue
                    K_all, Va, pa = new_kv[b]
                    caches[b].commit(K_all[:, :, -S:, :], Va[:, :, -S:, :], pa[-S:])
            pos = ce

    def run_span(step, K):
        end = min(step + K, n_steps)
        span_start = draft.win_starts[step]
        span_end = (draft.win_starts[end] + 1) if end < n_steps else len(draft.tokens)
        S = span_end - span_start
        win_toks = torch.tensor([draft.tokens[span_start:span_end]], device=dev)
        q_positions = torch.arange(span_start, span_start + S, device=dev)
        q_local = [draft.win_starts[s] - span_start for s in range(step, end)]
        x0 = sparse.embed[win_toks].clone()
        apply_overlay_window_fast(x0, span_start, L, store_log, code_vec, query_rows=q_local,
                                  code=code, code_off=draft.code_off)
        Kq = len(q_local)

        # attn-vs-ffn event split
        class Timer:
            def __init__(s): s.ev = []
            def wrap(s, fn):
                def w(*a, **k):
                    e0, e1 = _ev(), _ev(); e0.record(); r = fn(*a, **k); e1.record()
                    s.ev.append((e0, e1)); return r
                return w
            def ms(s): return sum(a.elapsed_time(b) for a, b in s.ev)
        at_t, ffn_t = Timer(), Timer()
        for b in live_blocks:
            sparse.blocks[b].attn._po = sparse.blocks[b].attn.forward
            sparse.blocks[b].attn.forward = at_t.wrap(sparse.blocks[b].attn.forward)
        for b in range(nblk):
            fobj = sparse.blocks[b].ffn
            if hasattr(fobj, 'forward'):
                fobj._po = fobj.forward; fobj.forward = ffn_t.wrap(fobj.forward)
        # warm
        with torch.no_grad():
            past = [caches[b].as_past_kv() for b in range(nblk)]
            sparse.forward_hidden_cached(x0.clone(), past_key_values=past, q_positions=q_positions, use_cache=True)
        torch.cuda.synchronize()
        at_t.ev.clear(); ffn_t.ev.clear()
        torch.cuda.reset_peak_memory_stats(dev)
        t = time.perf_counter()
        with torch.no_grad():
            past = [caches[b].as_past_kv() for b in range(nblk)]
            sparse.forward_hidden_cached(x0.clone(), past_key_values=past, q_positions=q_positions, use_cache=True)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t) * 1e3
        vram = torch.cuda.max_memory_allocated(dev) / 1024**3
        at_ms, ffn_ms = at_t.ms(), ffn_t.ms()
        for b in live_blocks: sparse.blocks[b].attn.forward = sparse.blocks[b].attn._po
        for b in range(nblk):
            fobj = sparse.blocks[b].ffn
            if hasattr(fobj, '_po'): fobj.forward = fobj._po
        return S, Kq, wall_ms, at_ms, ffn_ms, vram

    # prime to just before the profiling window (use the last cap steps)
    PROF_STEP = max(0, n_steps - PROF_K - 1)
    prime(draft.win_starts[PROF_STEP])
    torch.cuda.synchronize()
    gmax = max(c.size() for c in caches)
    print(f'[bprof] primed. global_cache_max={gmax}', flush=True)

    S, Kq, wall, at_ms, ffn_ms, vram = run_span(PROF_STEP, PROF_K)
    tot = at_ms + ffn_ms
    print(f'\n[bprof] === SPAN S={S} Kq={Kq}  (BANDED={banded}) ===', flush=True)
    print(f'  wall={wall:.2f}ms ({wall/Kq:.3f} ms/step)  peak_vram={vram:.2f}GB', flush=True)
    print(f'  attn={at_ms:.2f}ms ({100*at_ms/tot:.1f}%)  ffn={ffn_ms:.2f}ms ({100*ffn_ms/tot:.1f}%)', flush=True)

    # FLAT-IN-S: re-run the same PROF_STEP span with different K to vary S.
    print(f'\n[bprof] FLAT-IN-S (attn ms vs span S):', flush=True)
    for K in (50, 100, 200, 400):
        st = max(0, n_steps - K - 1)
        # (cache already primed to PROF_STEP; for a fair per-S attn we just re-time the
        #  attn sublayer at this span — the global cache is the same)
        try:
            S, Kq, wall, at_ms, ffn_ms, vram = run_span(st, K)
            print(f'  K={K:4d} S={S:6d}  attn={at_ms:7.2f}ms  ffn={ffn_ms:7.2f}ms  '
                  f'wall={wall:7.2f}ms  attn%={100*at_ms/(at_ms+ffn_ms):5.1f}  vram={vram:5.2f}GB', flush=True)
        except RuntimeError as e:
            oom = 'OOM' if 'out of memory' in str(e).lower() else str(e)[:30]
            print(f'  K={K:4d}  {oom}', flush=True); torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
