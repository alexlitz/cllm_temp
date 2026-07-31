"""WALL #6 — run the doom cfm MODEL verify up to a step cap and report first mismatch.

Unlike _agent_doom_cfm_continuous.py this does NOT require the draft to reach the
first printf; it caps the draft + model verify at C4_MAX_STEPS and reports the exact
first divergence (step, got vs want).  Lets us A/B eviction policies + IMM_NIBS
around the reported wall at step ~15,573 WITHOUT a 30k-step full run.

Env:
  C4_MAX_STEPS   (default 15700) — draft + model steps
  K              (block_steps, default 400)
  C4_LOCAL_WINDOW(default 96)
  C4_EXACT_EVICT / C4_STACK_POP_FREE / C4_EVICT_SCHEDULE — eviction policy
  C4_IMM_NIBS    — IMM width (5 default, 8 for full 32-bit)
  DOOM_EVICT_INTERVAL — eviction cadence in steps (default = K)
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
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


def main():
    max_steps = int(os.environ.get('C4_MAX_STEPS', '15700'))
    K = int(os.environ.get('K', '400'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    ev_int = os.environ.get('DOOM_EVICT_INTERVAL')
    ev_int = int(ev_int) if ev_int else None
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))

    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    print(f'[run] imm_nibs={os.environ.get("C4_IMM_NIBS","5")} '
          f'exact_evict={os.environ.get("C4_EXACT_EVICT","0")} '
          f'pop_free={os.environ.get("C4_STACK_POP_FREE","0")} '
          f'evict_sched={os.environ.get("C4_EVICT_SCHEDULE","0")} '
          f'instrs={len(code)} K={K} window={window} max_steps={max_steps} ev_int={ev_int}', flush=True)

    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    print(f'[run] draft steps={draft.step_count} halted={draft.halted} wall={time.time()-t0:.2f}s', flush=True)

    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True, addr32=True, compute_mode='dense_kernel')
    print(f'[run] built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} build={time.time()-t0:.0f}s', flush=True)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev != 'cpu':
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    cbg = os.environ.get('DOOM_CBG', '0') == '1'
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=cbg, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)

    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True, min_block_steps=4,
                       prime_chunk=prime_chunk, evict_interval_steps=ev_int)
    run_wall = time.time() - t0
    ms = run_wall / max(draft.step_count, 1) * 1e3
    print(f'[run] matched={vr.all_matched} accepted={vr.accepted_steps}/{vr.total_steps} '
          f'wall={run_wall:.1f}s ms/step={ms:.1f} '
          f'sched_stores={stats.get("sched_stores")} sched_superseded={stats.get("sched_superseded")} '
          f'sched_dead_unread={stats.get("sched_dead_unread")} raf={stats.get("sched_read_after_free")} '
          f'exact={stats.get("exact_evict")} evict_rounds={stats.get("evict_rounds")}', flush=True)
    if vr.first_mismatch:
        print(f'[run] FIRST MISMATCH: {vr.first_mismatch}', flush=True)
        if 'diag_stack0_model' in stats:
            print(f'[run] DIAG op={stats.get("diag_op")} model_STACK0={stats.get("diag_stack0_model")} '
                  f'draft_stk={stats.get("diag_stk_draft")}', flush=True)
    else:
        print(f'[run] NO MISMATCH through {vr.accepted_steps} steps (cap {max_steps})', flush=True)
    return 0 if vr.all_matched else 1


if __name__ == '__main__':
    raise SystemExit(main())
