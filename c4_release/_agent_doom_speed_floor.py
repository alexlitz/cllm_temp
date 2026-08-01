"""MEASURE the byte-exact speed FLOOR of the doom CFM run by composing the sparse
levers (direct-CAM + batched block-MoE + block-sparse-FFN / megakernel) and
reporting real ms/step + byte-exactness + block-count/step for EACH composition.

Additive measurement harness (no build-path weight change).  Env:
  K, N_STEPS_CAP, C4_MAX_STEPS as _agent_doom_directcam.py.
  LEVERS (compose by setting the flags): C4_DIRECT_CAM_BATCHED, C4_BATCHED_BLOCK_SKIP,
  C4_BLOCK_SPARSE_FFN (mode via C4_BSF_MODE=coo|dense_active).
Reports: byte-exact (accepted==total, first7==ref7), ms/step, eff_K, peak VRAM,
  mean blocks/span (242 -> ?), block reduction, forwards.
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
from c4_min.pf_speculative import draft_pf_program, verify_blocks, _draft_cmp32
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'
REF = '/tmp/doom_ref.bin'


def main():
    assert _draft_cmp32()
    max_steps = int(os.environ.get('C4_MAX_STEPS', '30200'))
    K = int(os.environ.get('K', '200'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    ncap = os.environ.get('N_STEPS_CAP')
    ncap = int(ncap) if ncap else None

    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    ref = Path(REF).read_bytes()

    lev = []
    if os.environ.get('C4_DIRECT_CAM_BATCHED', '0') == '1':
        lev.append('direct-CAM')
    if os.environ.get('C4_BATCHED_BLOCK_SKIP', '0') == '1':
        lev.append('block-skip')
    if os.environ.get('C4_BLOCK_SPARSE_FFN', '0') == '1':
        lev.append('sparse-FFN(%s)' % os.environ.get('C4_BSF_MODE', 'coo'))
    print(f'[floor] LEVERS={lev} K={K} ncap={ncap} window={window}', flush=True)

    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                              stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    draft_out = bytes(fio.runner.stdout)
    print(f'[floor] draft steps={draft.step_count} first_prtf={draft.prtf_steps[:1]} '
          f'stdout={draft_out[:7].hex()} wall={time.time()-t0:.2f}s', flush=True)
    if not draft.prtf_steps:
        print('[floor] draft did not reach printf'); return 2
    fp = draft.prtf_steps[0]

    if ncap is not None:
        n = min(ncap, draft.step_count)
        draft.frames = draft.frames[:n]
        draft.win_starts = draft.win_starts[:n]
        draft.step_count = n
        last_pos = draft.win_starts[-1]
        draft.tokens = draft.tokens[:last_pos + 1]
        print(f'[floor] CAPPED verify to first {n} steps', flush=True)

    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    n_blocks = len(sparse.blocks)
    print(f'[floor] built dim={sparse.embed.shape[1]} blocks={n_blocks} '
          f'build={time.time()-t0:.0f}s', flush=True)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev != 'cpu':
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)

    # block-sparse-FFN (swap the dense SwiGLU for a COO gather-scale-scatter). Install
    # BEFORE direct-CAM (it only touches FFN, not attn) so both compose.
    if os.environ.get('C4_BLOCK_SPARSE_FFN', '0') == '1':
        from c4_min.block_sparse_ffn import install_block_sparse_ffn
        mode = os.environ.get('C4_BSF_MODE', 'coo')
        st = install_block_sparse_ffn(sparse, mode=mode, verbose=True)
        print(f'[floor] block-sparse-FFN installed: {st}', flush=True)

    from c4_min import direct_cam_batched as DCB
    _dcb = DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=True)

    stats = {}
    do_evict = os.environ.get('DOOM_EVICT', '1') == '1'
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))
    ev_int = os.environ.get('DOOM_EVICT_INTERVAL')
    ev_int = int(ev_int) if ev_int else None
    # warm + timed
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=do_evict,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=prime_chunk,
                       evict_interval_steps=ev_int)
    run_wall = time.time() - t0
    ms = run_wall / max(draft.step_count, 1) * 1e3
    byte_exact = (vr.all_matched and draft_out[:7] == ref[:7])
    print(f'[floor] RESULT: byte_exact={byte_exact} accepted={vr.accepted_steps}/{vr.total_steps} '
          f'ms/step={ms:.2f} eff_K={stats.get("effective_block_steps")} '
          f'forwards={vr.forwards} peak_vram={stats.get("peak_vram_gb", 0):.1f}GB', flush=True)
    if 'bbs_mean_blocks_per_span' in stats:
        print(f'[floor] BLOCK-SKIP: mean_blocks/span={stats["bbs_mean_blocks_per_span"]:.1f}/{n_blocks} '
              f'reduction={stats["bbs_block_reduction"]:.2f}x spans={stats["bbs_spans"]}', flush=True)
    if vr.first_mismatch:
        print(f'[floor] FIRST MISMATCH: {vr.first_mismatch}', flush=True)
    print(f'[floor] SELF-EMU COMPARISON: {ms:.2f} ms/step vs 0.044 ms/step self-emu '
          f'= {ms/0.044:.0f}x slower', flush=True)
    return 0 if byte_exact else 1


if __name__ == '__main__':
    raise SystemExit(main())
