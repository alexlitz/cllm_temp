"""Run doom_divlite.c through the c4_min transformer with the FULL fast stack and
MEASURE the block-MoE lever.

Adapted from _agent_doom_directcam.py.  Adds the batched block-skip (block-MoE)
+ block-sparse FFN toggles and reports the MEAN BLOCK-MoE UNION SIZE per span
(bbs_mean_blocks_per_span) -- the key number: does cutting doom's DIV density
make block-MoE actually PRUNE (union << 242)?

Full fast stack:
  C4_PF_CFM=1                 code-from-memory pure-forward
  C4_DIRECT_CAM_BATCHED=1     O(1) direct-CAM gather (removes the global softmax)
  C4_BANDED_LOCAL_ATTN=1      O(S*W) banded local attention
  C4_BATCHED_BLOCK_SKIP=1     per-span block-MoE union (THE lever under test)
  C4_BLOCK_SPARSE_FFN=1       COO block-sparse FFN inside each live block

Env: K (block_steps, default 200), N_STEPS_CAP (cap the model verify at N steps
for a fast datapoint; default None = to first printf), DOOM_SRC (default
doom_divlite.c; set to doom.c for the baseline).  golden 069cc32f UNCHANGED.
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
# full fast stack (all gated; golden unchanged)
os.environ.setdefault('C4_DIRECT_CAM_BATCHED', '1')
os.environ.setdefault('C4_BANDED_LOCAL_ATTN', '1')
os.environ.setdefault('C4_BATCHED_BLOCK_SKIP', '1')
os.environ.setdefault('C4_BLOCK_SPARSE_FFN', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks, _draft_cmp32
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                        data_segment)

DOOM = os.environ.get('DOOM_SRC', '/home/alexlitz/Documents/misc/c4_doom/doom_divlite.c')


def main():
    assert _draft_cmp32()
    K = int(os.environ.get('K', '200'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    max_steps = int(os.environ.get('C4_MAX_STEPS', '200000'))
    ncap = os.environ.get('N_STEPS_CAP')
    ncap = int(ncap) if ncap else None

    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    # c4 reference for byte-exactness
    import subprocess
    ref = subprocess.run(['/home/alexlitz/Documents/misc/c4_doom/c4', DOOM],
                         input=b'q', capture_output=True).stdout
    print(f'[divlite-tf] src={DOOM.split("/")[-1]} instrs={len(code)} K={K} window={window} '
          f'ncap={ncap} ref7={ref[:7].hex()}', flush=True)

    # DRAFT
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    draft_out = bytes(fio.runner.stdout)
    print(f'[divlite-tf] draft steps={draft.step_count} halted={draft.halted} '
          f'n_prtf={len(draft.prtf_steps)} first_prtf_step={draft.prtf_steps[:1]} '
          f'stdout={draft_out[:16].hex()} wall={time.time()-t0:.1f}s', flush=True)
    if not draft.prtf_steps:
        print('[divlite-tf] DRAFT did not reach first printf'); return 2
    fp = draft.prtf_steps[0]

    if ncap is not None:
        n = min(ncap, draft.step_count)
        draft.frames = draft.frames[:n]
        draft.win_starts = draft.win_starts[:n]
        draft.step_count = n
        last_pos = draft.win_starts[-1]
        draft.tokens = draft.tokens[:last_pos + 1]
        print(f'[divlite-tf] CAPPED model verify to first {n} steps (of {fp} to first printf)', flush=True)

    # BUILD
    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    nblk = len(sparse.blocks)
    print(f'[divlite-tf] built dim={sparse.embed.shape[1]} blocks={nblk} '
          f'build={time.time()-t0:.0f}s', flush=True)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev != 'cpu':
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    from c4_min import direct_cam_batched as DCB
    _dcb = DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
    # block-sparse FFN inside each live block (COO)
    from c4_min.block_sparse_ffn import block_sparse_ffn_enabled, install_block_sparse_ffn
    if block_sparse_ffn_enabled():
        install_block_sparse_ffn(sparse, mode='coo', verbose=False)
    from c4_min.batched_block_skip import batched_block_skip_enabled
    print(f'[divlite-tf] fast stack: local-attn(w={window}) dead-block-fusion '
          f'direct-CAM={"ON" if _dcb else "OFF"} block-sparse-FFN={block_sparse_ffn_enabled()} '
          f'batched-block-skip={batched_block_skip_enabled()}', flush=True)

    # VERIFY
    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=int(os.environ.get('DOOM_PRIME_CHUNK', '2048')))
    run_wall = time.time() - t0
    ms = run_wall / max(draft.step_count, 1) * 1e3
    print(f'\n[divlite-tf] === RESULT ===', flush=True)
    print(f'[divlite-tf] verify matched={vr.all_matched} forwards={vr.forwards} '
          f'accepted={vr.accepted_steps}/{vr.total_steps}', flush=True)
    print(f'[divlite-tf] ms/step={ms:.3f}  wall={run_wall:.1f}s  eff_K={stats.get("effective_block_steps")} '
          f'peak_vram={stats.get("peak_vram_gb",0):.1f}GB', flush=True)
    # block-MoE union
    mean_union = stats.get('bbs_mean_blocks_per_span')
    if mean_union is not None:
        print(f'[divlite-tf] block-MoE UNION: mean {mean_union:.1f}/{nblk} blocks/span '
              f'({100*mean_union/nblk:.0f}%)  spans={stats.get("bbs_spans")} '
              f'block-reduction={stats.get("bbs_block_reduction",1):.2f}x  '
              f'-> block-MoE {"FIRES (prunes)" if mean_union < nblk*0.9 else "INERT (union~full)"}', flush=True)
    else:
        print(f'[divlite-tf] block-MoE: batched-block-skip NOT active (no bbs stats)', flush=True)
    if vr.first_mismatch:
        print(f'[divlite-tf] FIRST MISMATCH: {vr.first_mismatch}', flush=True)

    # byte-exactness
    n = min(len(draft_out), len(ref)); m = 0
    for i in range(n):
        if draft_out[i] == ref[i]: m += 1
        else: break
    ok = (draft_out[:7] == ref[:7]) and vr.all_matched
    print(f'[divlite-tf] neural stdout first7={draft_out[:7].hex()} byte-exact_prefix={m} '
          f'transformer==./c4: {ok}', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
