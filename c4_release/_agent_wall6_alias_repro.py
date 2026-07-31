"""WALL #6 — MINIMAL reproduction of the address-CAM recency wall.

A tight loop that repeatedly stores DIFFERENT values to the SAME frame-local slot
and reads them back — the doom hot-loop pattern (addr 200 written 1,777 times).
Small code_size (fast build), but many STEPS (deep same-address reuse), so it
reproduces the within-eviction-window CAM aliasing at manageable build cost.

Runs the program through the FAST verify path under several eviction policies +
IMM_NIBS settings and reports the first divergence step for each.  Byte-exact if
the model's per-step decoded AX == the draft's for all steps.
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

# A loop that writes a local `a` many times (same frame slot => same address) and reads
# it back into a multiply — the doom `x = a * b` hot-loop shape.  N iterations => N
# stores to the same slot.  `a` alternates so a stale read is detectable.
REPRO_C = r'''
int main() {
  int i, a, b, s;
  s = 0;
  i = 0;
  while (i < %(N)d) {
    a = 0;          /* zero-value store to the hot slot `a` */
    b = i + 7;
    a = a + b;      /* read back the 0, then overwrite -> zero-value latest at times */
    s = s + a * b;
    a = 0;          /* another zero to the same slot right before a read */
    s = s + a;      /* the pop must read the LATEST (0), not a stale non-zero */
    a = -1;
    b = a * i;      /* -1 * i : the doom `expr * -1` shape at 5-nibble truncation */
    s = s + b;
    i = i + 1;
  }
  return s;
}
'''


def run_once(N, imm_nibs, exact_evict, evict_sched, pop_free, K, evict_int, window=96):
    # env is process-global; caller runs each config in a fresh subprocess for IMM_NIBS.
    src = REPRO_C % {'N': N}
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"", neural=True)))
    draft = draft_pf_program(code, max_steps=N * 40 + 100, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True, addr32=True, compute_mode='dense_kernel')
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev != 'cpu':
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    stats = {}
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True, min_block_steps=4,
                       prime_chunk=2048, evict_interval_steps=evict_int,
                       evict_schedule=evict_sched, exact_evict=exact_evict)
    return draft, vr, stats


def main():
    N = int(os.environ.get('REPRO_N', '400'))
    K = int(os.environ.get('K', '400'))
    evict_int = os.environ.get('DOOM_EVICT_INTERVAL')
    evict_int = int(evict_int) if evict_int else None
    imm_nibs = os.environ.get('C4_IMM_NIBS', '5')
    ee = os.environ.get('C4_EXACT_EVICT', '0') not in ('0', '', 'false')
    es = os.environ.get('C4_EVICT_SCHEDULE', '0') not in ('0', '', 'false')
    pf = os.environ.get('C4_STACK_POP_FREE', '0') not in ('0', '', 'false')
    print(f'[repro] N={N} imm_nibs={imm_nibs} K={K} evict_int={evict_int} '
          f'exact={ee} sched={es} popfree={pf}', flush=True)
    t0 = time.time()
    draft, vr, stats = run_once(N, imm_nibs, ee, es, pf, K, evict_int)
    print(f'[repro] draft steps={draft.step_count} '
          f'top_addr_stores={_top_addr(draft)} wall={time.time()-t0:.1f}s', flush=True)
    print(f'[repro] matched={vr.all_matched} accepted={vr.accepted_steps}/{vr.total_steps} '
          f'evict_rounds={stats.get("evict_rounds")}', flush=True)
    if vr.first_mismatch:
        print(f'[repro] FIRST MISMATCH: {vr.first_mismatch}', flush=True)
    else:
        print(f'[repro] BYTE-EXACT through all {vr.accepted_steps} steps', flush=True)
    return 0 if vr.all_matched else 1


def _top_addr(draft):
    from collections import Counter
    c = Counter(a for (a, v) in (draft.store_log or {}).values())
    return c.most_common(3)


if __name__ == '__main__':
    raise SystemExit(main())
