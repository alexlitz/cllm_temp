"""FFN MEGAKERNEL wired into the batched verify_blocks DOOM path.

GOAL (measured, not asserted): after direct-CAM + banded attention the doom
per-step cost is dominated by the launch-bound 242-block FFN chain (~1628 CUDA
launches/step).  Block-MoE can't help (DIV density -> 217/242 live).  The lever is
LAUNCH AMORTIZATION: fuse the per-block FFN chain into ONE CUDA-graph replay so
the per-block launch overhead collapses.

This agent composes the full doom fast stack
  C4_PF_CFM + C4_DIRECT_CAM_BATCHED + C4_BANDED_LOCAL_ATTN + local-attn(window)
  + dead-block-fusion
and then, gated by C4_FFN_MEGAKERNEL={triton|graphed|triton_dense|off}, installs an
FFN megakernel into the block-stack forward the verifier calls
(model.forward_hidden_cached) and MEASURES:
  * doom ms/step eager vs megakernel (full run to first printf = steady-state)
  * FFN %-of-forward (instrumented block-level timers)
  * CUDA launches/forward (torch.cuda profiler, eager vs megakernel)
  * byte-exactness: accepted==total, first-7 stdout == printf 'q' | ./c4 doom.c
  * eff_K, peak VRAM

Variants:
  triton        big dead-FFN segment (blocks 12..242) run as the per-block Triton
                gather-gate-scatter (fused_ffn_megakernel.GraphedTritonBlockGSChain),
                the whole chain CUDA-graphed; short segs dense-graphed; live eager.
  graphed       GraphedFusedForward: dead segments CUDA-graphed as DENSE GEMV chains.
  triton_dense  same wiring as triton but forced dense (GraphedFusedFFNChain).
  off           eager (baseline).

MEASURED (RTX A5000, full run to first printf @ step 29754, K=200, 30200/30200
byte-exact, stdout first7 = 1b5b324a1b5b48 = 7/7 vs `printf 'q'|./c4 doom.c` ALL):
  eager           3.77 ms/step  (baseline)   peak 8.5 GB
  graphed (dense) 3.66 ms/step  (1.03x)      peak 12.2 GB  <- fades at doom's large S
  TRITON  (COO)   2.69 ms/step  (1.40x)      peak 12.0 GB  <- WINNER
  Small-S (K=8) launch-bound win is larger: eager 21.97 -> triton 6.40 (3.43x).
  Launch collapse (S=31 span): eager 17564 launches/forward -> triton 2126.
  TRITON byte-exactness is EMPIRICAL (real-doom residual keeps the COO atomic-add
  residue below the nibble decode margin; random-input L-inf is NOT tiny ~1e2) —
  authority = 30200/30200 accept + 7/7 stdout.  The dense `graphed` variant is
  by-CONSTRUCTION byte-exact (L-inf ~2e-2) but only 1.03x.

Env:
  C4_FFN_MEGAKERNEL   triton | graphed | triton_dense | off (default off)
  N_STEPS_CAP         model-verify step cap (default 4000; 30200 = full run)
  K                   block_steps (default 200)
  C4_LOCAL_WINDOW     local window (default 96)
  DOOM_PRIME_CHUNK    leading-context prime chunk (default 2048)
  C4_BLOCK_SPARSE_FFN 0/1  (does NOT compose with triton — both are COO; the graphed
                      megakernel is strictly faster.  triton raises a clear error.)
  PROFILE             0/1  run the FFN%-of-forward + launches/forward profiler
  golden 069cc32f UNCHANGED (all gated; additive).
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
os.environ.setdefault('C4_DIRECT_CAM_BATCHED', '1')
os.environ.setdefault('C4_BANDED_LOCAL_ATTN', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
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
REF = '/tmp/doom_ref.bin'


def build_doom(dev, cap_steps, window, block_sparse):
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code) + 2, 64), recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    # optional COO block-sparse FFN (compose with the megakernel graph)
    if block_sparse:
        from c4_min.block_sparse_ffn import install_block_sparse_ffn
        install_block_sparse_ffn(sparse, mode='coo', verbose=True)
    # direct-CAM batched (installed last, wins on the CAM blocks)
    from c4_min import direct_cam_batched as DCB
    DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
    return sparse, L, code, draft, fio, draft.code_off


def install_megakernel(sparse, dev, variant):
    """Install the chosen FFN megakernel into model.forward_hidden_cached.

    variant='graphed' -> GraphedFusedForward (CUDA-graph the dead segments =
        the per-block FFN chain).  variant='coo' -> already-installed COO FFN
        (no graph; just the block-sparse forward), returns None.  variant='off'
        -> eager.  Returns the installed wrapper or None."""
    if variant == 'graphed':
        from c4_min.graphed_fused_forward import install_graphed_fused_forward
        return install_graphed_fused_forward(sparse, dev, verbose=True)
    if variant == 'triton':
        from c4_min.triton_ffn_megaseg import install_triton_ffn_megaseg
        return install_triton_ffn_megaseg(sparse, dev, min_seg_len=16,
                                          use_triton=True, verbose=True)
    if variant == 'triton_dense':
        from c4_min.triton_ffn_megaseg import install_triton_ffn_megaseg
        return install_triton_ffn_megaseg(sparse, dev, min_seg_len=16,
                                          use_triton=False, verbose=True)
    return None


def profile_span_ffn(sparse, L, code, draft, dev, code_off, step0=None):
    """Instrument ONE representative span: time attention vs FFN per block, count
    CUDA kernel launches.  Returns a dict."""
    from c4_min.pf_speculative import (apply_overlay_window_fast, build_code_vec)
    n_steps = draft.step_count
    s0 = step0 if step0 is not None else min(1500, n_steps - 2)
    span_start = draft.win_starts[s0]
    span_end = draft.win_starts[min(s0 + 199, n_steps - 1)] + 1
    span_toks = draft.tokens[span_start:span_end]
    S = len(span_toks)
    win_toks = torch.tensor([span_toks], device=dev)
    q_positions = torch.arange(span_start, span_start + S, device=dev)
    code_vec = build_code_vec(code, L, sparse.embed.shape[1], torch.device(dev),
                              dtype=sparse.embed.dtype)
    with torch.no_grad():
        x = sparse.embed[win_toks].clone()
        q_local = [draft.win_starts[s] - span_start for s in range(s0, min(s0 + 200, n_steps))]
        try:
            apply_overlay_window_fast(x, span_start, L, {}, code_vec,
                                      query_rows=[q for q in q_local if 0 <= q < S],
                                      code=code, code_off=code_off)
        except Exception:
            pass
    n = len(sparse.blocks)
    attn_ms = ffn_ms = 0.0
    # warm
    with torch.no_grad():
        h = x
        for bi in range(n):
            blk = sparse.blocks[bi]
            a, kv = blk.attn.forward(h, past_kv=None, q_positions=q_positions, use_cache=True)
            h = blk.ffn.forward(a) if not blk._routed else blk.ffn(a)
    torch.cuda.synchronize()
    with torch.no_grad():
        h = x
        for bi in range(n):
            blk = sparse.blocks[bi]
            torch.cuda.synchronize(); t0 = time.time()
            a, kv = blk.attn.forward(h, past_kv=None, q_positions=q_positions, use_cache=True)
            torch.cuda.synchronize(); t1 = time.time()
            out = blk.ffn.forward(a) if not blk._routed else blk.ffn(a)
            torch.cuda.synchronize(); t2 = time.time()
            attn_ms += (t1 - t0) * 1e3
            ffn_ms += (t2 - t1) * 1e3
            h = out
    launches = None
    try:
        from torch.profiler import profile, ProfilerActivity
        past = [None] * n
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                sparse.forward_hidden_cached(x, past_key_values=past,
                                             q_positions=q_positions, use_cache=True)
        launches = sum(1 for e in prof.events()
                       if getattr(e, 'device_type', None) == torch.autograd.DeviceType.CUDA)
    except Exception as e:
        launches = f'err:{e}'
    return {'S': S, 'attn_ms': attn_ms, 'ffn_ms': ffn_ms,
            'ffn_frac': ffn_ms / max(attn_ms + ffn_ms, 1e-9), 'launches': launches}


def run_verify(sparse, L, code, draft, dev, K, prime_chunk):
    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=prime_chunk)
    wall = time.time() - t0
    ms = wall / max(draft.step_count, 1) * 1e3
    return vr, stats, wall, ms


def main():
    dev = 'cuda:0'; torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '4000'))
    K = int(os.environ.get('K', '200'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))
    variant = os.environ.get('C4_FFN_MEGAKERNEL', 'off')
    block_sparse = os.environ.get('C4_BLOCK_SPARSE_FFN', '0') == '1'
    do_profile = os.environ.get('PROFILE', '1') == '1'
    ref = Path(REF).read_bytes()

    print(f'[ffn-mega] variant={variant} block_sparse={block_sparse} cap={cap} K={K} '
          f'window={window} prime_chunk={prime_chunk} ref7={ref[:7].hex()}', flush=True)

    sparse, L, code, draft, fio, code_off = build_doom(dev, cap, window, block_sparse)
    n_blocks = len(sparse.blocks)
    print(f'[ffn-mega] instrs={len(code)} dim={sparse.embed.shape[1]} blocks={n_blocks} '
          f'draft_steps={draft.step_count} code_off={code_off} '
          f'first_prtf={draft.prtf_steps[:1]}', flush=True)

    if do_profile:
        p = profile_span_ffn(sparse, L, code, draft, dev, code_off)
        print(f'[ffn-mega] PROFILE span S={p["S"]}: attn={p["attn_ms"]:.1f}ms '
              f'ffn={p["ffn_ms"]:.1f}ms ffn_frac={p["ffn_frac"]*100:.1f}% '
              f'launches/forward={p["launches"]}', flush=True)

    mk = install_megakernel(sparse, dev, variant)
    print(f'[ffn-mega] megakernel={"ON:"+variant if (mk is not None or variant=="coo") else "OFF (eager)"}', flush=True)

    # POST-INSTALL launch count over ONE representative forward (the launches/step
    # collapse metric).  Build the same representative span and profile the
    # (now-megakernel) forward_hidden_cached.
    if do_profile and mk is not None:
        try:
            from c4_min.pf_speculative import apply_overlay_window_fast, build_code_vec
            from torch.profiler import profile, ProfilerActivity
            n_steps = draft.step_count
            s0 = min(1500, n_steps - 2)
            span_start = draft.win_starts[s0]
            span_end = draft.win_starts[min(s0 + 199, n_steps - 1)] + 1
            span_toks = draft.tokens[span_start:span_end]; S = len(span_toks)
            win_toks = torch.tensor([span_toks], device=dev)
            q_positions = torch.arange(span_start, span_start + S, device=dev)
            code_vec = build_code_vec(code, L, sparse.embed.shape[1], torch.device(dev),
                                      dtype=sparse.embed.dtype)
            with torch.no_grad():
                x = sparse.embed[win_toks].clone()
                q_local = [draft.win_starts[s] - span_start
                           for s in range(s0, min(s0 + 200, n_steps))]
                apply_overlay_window_fast(x, span_start, L, {}, code_vec,
                                          query_rows=[q for q in q_local if 0 <= q < S],
                                          code=code, code_off=code_off)
                past = [None] * n_blocks
                # warm (capture graphs) then profile
                sparse.forward_hidden_cached(x, past_key_values=past,
                                             q_positions=q_positions, use_cache=True)
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    sparse.forward_hidden_cached(x, past_key_values=past,
                                                 q_positions=q_positions, use_cache=True)
            # count device kernel launches robustly across torch versions.
            lc = 0
            try:
                ka = prof.key_averages()
                lc = sum(int(getattr(e, 'count', 0)) for e in ka
                         if getattr(e, 'device_type', None) == torch.autograd.DeviceType.CUDA
                         and getattr(e, 'cuda_time_total', getattr(e, 'device_time_total', 0)))
            except Exception:
                lc = 0
            if lc == 0:
                try:
                    evs = prof.events()
                    lc = sum(1 for e in (evs or [])
                             if getattr(e, 'device_type', None) == torch.autograd.DeviceType.CUDA)
                except Exception:
                    lc = -1
            n_steps_in_span = len(q_local)
            print(f'[ffn-mega] POST-INSTALL launches/forward={lc} (S={S}, '
                  f'~{lc/max(n_steps_in_span,1):.1f} launches/step) with megakernel={variant}',
                  flush=True)
        except Exception as e:
            print(f'[ffn-mega] post-install launch profile err: {e}', flush=True)

    vr, stats, wall, ms = run_verify(sparse, L, code, draft, dev, K, prime_chunk)
    n_graphs = len(mk._graphs) if mk is not None else 0
    print(f'[ffn-mega] verify matched={vr.all_matched} accepted={vr.accepted_steps}/{vr.total_steps} '
          f'forwards={vr.forwards} wall={wall:.1f}s ms/step={ms:.2f} '
          f'eff_K={stats.get("effective_block_steps")} peak_vram={stats.get("peak_vram_gb",0):.1f}GB '
          f'n_graph_shapes={n_graphs}', flush=True)
    if vr.first_mismatch:
        print(f'[ffn-mega] FIRST MISMATCH: {vr.first_mismatch}', flush=True)

    draft_out = bytes(fio.runner.stdout)
    first7_ok = draft_out[:7] == ref[:7]
    reached = (draft.prtf_steps and cap > draft.prtf_steps[0])
    print(f'[ffn-mega] stdout first7={draft_out[:7].hex()} first7_ok={first7_ok} '
          f'(reached_printf={reached}) accepted_all={vr.all_matched}', flush=True)
    ok = vr.all_matched and first7_ok
    print(f'[ffn-mega] BYTE-EXACT (accepted_all & first7): {ok}', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
