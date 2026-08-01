"""PART 1 — PROFILE the doom cfm neural forward: decompose one representative
verify_blocks span's wall-time into (global code-CAM attn, global mem/stack CAM,
local heads, FFN, weight reads, launch overhead).

Answers: is the 18.1 ms/step dominated by (a) O(3976) code-frame attention
[intrinsic] or (b) launch/compute a megakernel+K would amortize [config]?

Method: build the doom cfm model exactly as _agent_doom_cfm_continuous.py does,
prime a representative leading context, then time ONE big-K span forward
(model.forward_hidden_cached over the drafted stream) with per-sublayer CUDA-event
instrumentation + kernel-launch counting.  Additive/gated; no model edits.

Env: N_STEPS_CAP (default 4000, cap the draft to keep build+prime fast),
PROF_K (default 400, the span K), PROF_STEP (representative start step, default
uses the last cap step so cache is realistic).
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
from c4_min import blogspec_vocab as V
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
    max_steps = int(os.environ.get('N_STEPS_CAP', '4000'))
    PROF_K = int(os.environ.get('PROF_K', '25'))   # match the real run's eff_K=25
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))

    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    print(f'[prof] instrs={len(code)} cap_steps={max_steps} K={PROF_K} window={window}', flush=True)

    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    print(f'[prof] draft steps={draft.step_count} code_off={draft.code_off} '
          f'n_tokens={len(draft.tokens)} wall={time.time()-t0:.1f}s', flush=True)

    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    D = sparse.embed.shape[1]
    nblk = len(sparse.blocks)
    print(f'[prof] built dim={D} blocks={nblk} build={time.time()-t0:.0f}s', flush=True)
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    dbf = install_dead_block_fusion(sparse, verbose=True)
    print(f'[prof] dead-block-fusion: {dbf}', flush=True)

    H = sparse.blocks[0].attn.n_heads
    n_dead = 0
    n_global_heads_total = 0
    n_local_heads_total = 0
    for b in range(nblk):
        at = sparse.blocks[b].attn
        if getattr(at, '_dead_block_fused', False):
            n_dead += 1
            continue
        gmask = getattr(at, '_global_head_mask', None)
        if gmask is not None:
            ng = int(gmask.sum().item())
            n_global_heads_total += ng
            n_local_heads_total += (H - ng)
    n_live = nblk - n_dead
    print(f'[prof] H={H}/block  live_attn_blocks={n_live}  dead_attn_blocks={n_dead}  '
          f'global_head_slots={n_global_heads_total}  local_head_slots={n_local_heads_total}',
          flush=True)

    from c4_min.moe_top1 import Top1RoutedFFN
    n_routed = sum(1 for b in sparse.blocks if isinstance(getattr(b, 'ffn', None), Top1RoutedFFN)
                   or getattr(b, '_routed', False))
    print(f'[prof] FFN: routed_blocks={n_routed}/{nblk} (rest dense-sparse SwiGLU)', flush=True)

    caches = [BlockKVCacheBatched(H, sparse.blocks[b].attn.head_dim,
                                  sparse.blocks[b].attn.alibi_slopes)
              for b in range(nblk)]
    for b in range(nblk):
        at = sparse.blocks[b].attn
        if getattr(at, '_drop_local_kv', False) and getattr(at, '_local_window', None):
            gmask = at._global_head_mask
            g_idx = [int(h) for h in range(H) if bool(gmask[h])]
            caches[b].set_head_groups(g_idx, int(at._local_window),
                                      content_bound=False, content_cR=None)
    store_log = draft.store_log
    code_vec = build_code_vec(code, L, D, torch.device(dev), dtype=sparse.embed.dtype)

    n_steps = draft.step_count
    PROF_STEP = int(os.environ.get('PROF_STEP', str(max(0, n_steps - PROF_K - 1))))
    lead_end = draft.win_starts[PROF_STEP]
    print(f'[prof] PROF_STEP={PROF_STEP}  priming leading context to pos {lead_end} '
          f'in {prime_chunk}-chunks ...', flush=True)

    def _prime(pos_end):
        pos = 0
        pc_ = max(1, prime_chunk)
        while pos < pos_end:
            ce = min(pos + pc_, pos_end)
            span_toks = draft.tokens[pos:ce]
            S = len(span_toks)
            win_toks = torch.tensor([span_toks], device=dev)
            q_positions = torch.arange(pos, pos + S, device=dev)
            with torch.no_grad():
                x = sparse.embed[win_toks].clone()
                apply_overlay_window_fast(x, pos, L, store_log, code_vec,
                                          query_rows=[], code=code, code_off=draft.code_off)
                past = [caches[b].as_past_kv() for b in range(nblk)]
                _, new_kv = sparse.forward_hidden_cached(
                    x, past_key_values=past, q_positions=q_positions, use_cache=True)
                for b in range(nblk):
                    if new_kv[b] is None:
                        continue
                    K_all, Va, pa = new_kv[b]
                    caches[b].commit(K_all[:, :, -S:, :], Va[:, :, -S:, :], pa[-S:])
            pos = ce
    t0 = time.time()
    _prime(lead_end)
    torch.cuda.synchronize()
    cache0 = caches[0].size()
    print(f'[prof] primed. cache[0].size={cache0}  global_cache_max='
          f'{max(c.size() for c in caches)}  prime_wall={time.time()-t0:.1f}s', flush=True)

    step, end = PROF_STEP, min(PROF_STEP + PROF_K, n_steps)
    span_start = draft.win_starts[step]
    span_end = (draft.win_starts[end] + 1) if end < n_steps else len(draft.tokens)
    span_toks = draft.tokens[span_start:span_end]
    S = len(span_toks)
    win_toks = torch.tensor([span_toks], device=dev)
    q_positions = torch.arange(span_start, span_start + S, device=dev)
    q_local = [draft.win_starts[s] - span_start for s in range(step, end)]
    Kq = len(q_local)
    print(f'[prof] span: S={S} rows  Kq={Kq} query rows  span_start={span_start}  '
          f'cache_rows={cache0}', flush=True)

    def _make_x():
        x = sparse.embed[win_toks].clone()
        apply_overlay_window_fast(x, span_start, L, store_log, code_vec,
                                  query_rows=q_local, code=code, code_off=draft.code_off)
        return x

    import c4_min.local_attention as LA

    def time_full(x, iters=8):
        with torch.no_grad():
            past = [caches[b].as_past_kv() for b in range(nblk)]
            sparse.forward_hidden_cached(x.clone(), past_key_values=past,
                                         q_positions=q_positions, use_cache=True)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            xx = x.clone()
            past = [caches[b].as_past_kv() for b in range(nblk)]
            torch.cuda.synchronize()
            ev0, ev1 = _ev(), _ev()
            ev0.record()
            with torch.no_grad():
                sparse.forward_hidden_cached(xx, past_key_values=past,
                                             q_positions=q_positions, use_cache=True)
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))
        ts.sort()
        return ts[len(ts) // 2]

    x = _make_x()
    ms_full = time_full(x)
    print(f'\n[prof] === FULL SPAN forward: {ms_full:.2f} ms  '
          f'({ms_full/Kq:.3f} ms/step over Kq={Kq}) ===', flush=True)

    # ---- attn-vs-ffn split: record events per call, but SYNC ONCE at the end
    # (no per-call sync tax).  Sum of (start->end) event intervals per sublayer.
    class Timer:
        def __init__(self):
            self.evpairs = []; self.n = 0
        def wrap(self, fn):
            def wrapped(*a, **kw):
                e0, e1 = _ev(), _ev()
                e0.record()
                r = fn(*a, **kw)
                e1.record()
                self.evpairs.append((e0, e1)); self.n += 1
                return r
            return wrapped
        def total_ms(self):
            return sum(a.elapsed_time(b) for a, b in self.evpairs)

    at_timer = Timer(); ffn_timer = Timer()
    live_blocks = [b for b in range(nblk) if not getattr(sparse.blocks[b].attn, '_dead_block_fused', False)]
    for b in live_blocks:
        blk = sparse.blocks[b]
        blk.attn._prof_orig = blk.attn.forward
        blk.attn.forward = at_timer.wrap(blk.attn.forward)
    for b in range(nblk):
        fobj = sparse.blocks[b].ffn
        if hasattr(fobj, 'forward'):
            fobj._prof_orig = fobj.forward
            fobj.forward = ffn_timer.wrap(fobj.forward)

    with torch.no_grad():
        past = [caches[b].as_past_kv() for b in range(nblk)]
        sparse.forward_hidden_cached(x.clone(), past_key_values=past,
                                     q_positions=q_positions, use_cache=True)
    torch.cuda.synchronize()
    at_ms = at_timer.total_ms(); ffn_ms = ffn_timer.total_ms()
    print(f'[prof] SUBLAYER split (events, single end-sync; GPU-time per sublayer):', flush=True)
    print(f'         attn total = {at_ms:.2f} ms  ({at_timer.n} live-block calls, '
          f'{at_ms/max(Kq,1):.3f} ms/step)', flush=True)
    print(f'         ffn  total = {ffn_ms:.2f} ms  ({ffn_timer.n} block calls, '
          f'{ffn_ms/max(Kq,1):.3f} ms/step)', flush=True)
    print(f'         attn %% = {100*at_ms/(at_ms+ffn_ms):.1f}%%   ffn %% = {100*ffn_ms/(at_ms+ffn_ms):.1f}%%',
          flush=True)

    for b in live_blocks:
        sparse.blocks[b].attn.forward = sparse.blocks[b].attn._prof_orig
    for b in range(nblk):
        fobj = sparse.blocks[b].ffn
        if hasattr(fobj, '_prof_orig'):
            fobj.forward = fobj._prof_orig

    g_rows = max(c.size() for c in caches)
    print(f'[prof] attn cache geometry: GLOBAL heads see {g_rows} rows (full history incl '
          f'{draft.code_off} code frames), LOCAL heads see <= window={window} rows.', flush=True)

    # ---- torch.profiler kernel breakdown
    print(f'\n[prof] === torch.profiler kernel breakdown (1 span) ===', flush=True)
    try:
        from torch.profiler import profile, ProfilerActivity
        x2 = _make_x()
        past = [caches[b].as_past_kv() for b in range(nblk)]
        with torch.no_grad():
            sparse.forward_hidden_cached(x2.clone(), past_key_values=past,
                                         q_positions=q_positions, use_cache=True)
        torch.cuda.synchronize()
        past = [caches[b].as_past_kv() for b in range(nblk)]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                sparse.forward_hidden_cached(x2.clone(), past_key_values=past,
                                             q_positions=q_positions, use_cache=True)
            torch.cuda.synchronize()
        ka = prof.key_averages()
        n_cuda_launches = sum(int(e.count) for e in ka if e.self_device_time_total > 0)
        total_cuda_us = sum(e.self_device_time_total for e in ka)
        total_cpu_us = sum(e.self_cpu_time_total for e in ka)
        print(f'[prof] total CUDA kernel launches (this span) ~= {n_cuda_launches}', flush=True)
        print(f'[prof] sum self_device_time = {total_cuda_us/1000:.2f} ms  '
              f'sum self_cpu_time = {total_cpu_us/1000:.2f} ms', flush=True)
        print(prof.key_averages().table(sort_by='self_device_time_total', row_limit=25), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'[prof] torch.profiler failed: {e}', flush=True)

    print(f'\n[prof] SUMMARY: full_span={ms_full:.2f}ms Kq={Kq} -> {ms_full/Kq:.3f} ms/step; '
          f'blocks={nblk} live_attn={n_live} dead_attn={n_dead}; '
          f'global_cache_rows={g_rows} (code_off={draft.code_off})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
