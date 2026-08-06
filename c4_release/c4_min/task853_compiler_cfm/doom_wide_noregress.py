"""#854 doom no-regression under wide-LEA: draft a bounded window of the REAL id-Doom
bytecode with the wide-LEA config ON and VERIFY the model accepts every step byte-exact
(verify_blocks matched=True). Proves the wide-LEA decode correctly implements the wide
doom draft — which was shown (t854_doom_native) to equal the NATIVE c4 32-bit oracle's
LEA addresses (BP+4*imm, no fold), so wide-LEA is the byte-exact-CORRECT superset of the
8-bit fold for doom's own BP>255 frame LEAs.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time, resource
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '4')
# doom config + the FULL fine-grained wide set + wide-LEA
os.environ['C4_PF_CFM'] = '1'
os.environ['C4_DRAFT_CMP32'] = '1'
os.environ['C4_CMP32'] = '1'
os.environ['C4_CMP32_ORDER'] = '1'
os.environ['C4_SHIFT32'] = '1'
os.environ['C4_PC_WIDE'] = '1'
os.environ['C4_CODE_ADDR_BITS'] = '20'
os.environ['C4_IMM_NIBS'] = '6'
os.environ['C4_GLOBAL_ADDR32'] = '1'
os.environ['C4_SP_WIDE'] = '1'
os.environ['C4_DIVMOD_SIGNED'] = '1'
os.environ['C4_LEA_WIDE'] = '1'
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('C4_EXACT_EVICT', '1')
os.environ.setdefault('C4_MEM_EFF', '2000000')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

WT = '/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/task854-wide-lea/c4_release'
sys.path.insert(0, WT)
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import numpy as np
import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = PFC.SP_INIT = _PFCa.SP_INIT = 0x10000   # doom compact stack base
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from run_c4_min import data_segment, tag_compiler_syscalls, install_compiler_abi_file_dispatcher

DRAFT_STEPS = int(os.environ.get('DOOM_STEPS', '4000'))


def mem_avail_gb():
    for ln in open('/proc/meminfo'):
        if ln.startswith('MemAvailable'):
            return int(ln.split()[1]) / 1024 / 1024
    return 999


def main():
    assert PFC._lea_wide_enabled(), "wide-LEA gate OFF"
    print(f'[doom-wide] SP_INIT={PFC.SP_INIT:#x} lea_wide={PFC._lea_wide_enabled()} '
          f'DRAFT_STEPS={DRAFT_STEPS}', flush=True)
    if mem_avail_gb() < 25:
        print(f'[doom-wide] MemAvailable {mem_avail_gb():.1f}GB < 25GB — abort'); return 3

    snap = np.load(os.path.join(WT, '_doom_bytecode_snapshot.npz'))
    ops, imms, data = list(snap['ops']), list(snap['imms']), snap['data']
    n = len(ops)
    install_compiler_abi_file_dispatcher()
    code = tag_compiler_syscalls(
        [isa.Instr(int(ops[i]), int(imms[i]) & 0xFFFFFFFF) for i in range(n)], isa)
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b'q', neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=DRAFT_STEPS, mask=0xFFFFFFFF,
                             data_seg=data_segment([int(b) for b in data]), fio=fio)
    n_lea = sum(1 for i in range(draft.step_count) if draft.frames[i].get('op') == 'LEA')
    # divmod would need the recurrent path; skip windows with generic DIV/MOD past prime
    dm = [i for i in range(draft.step_count) if draft.frames[i].get('op') in ('DIV', 'MOD')]
    print(f'[doom-wide] drafted {draft.step_count} doom steps ({n_lea} LEA, {len(dm)} DIV/MOD) '
          f'wall={time.time()-t0:.1f}s', flush=True)

    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(n + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True, addr32=True,
                                             compute_mode='dense_kernel')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f'[doom-wide] built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} '
          f'build={time.time()-t0:.0f}s peakRSS={rss:.1f}GB', flush=True)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev != 'cpu':
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=int(os.environ.get('C4_LOCAL_WINDOW', '96')),
                            drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)

    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=int(os.environ.get('K', '100')),
                       device=dev, evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       oom_backoff=True, min_block_steps=4,
                       prime_chunk=int(os.environ.get('DOOM_PRIME_CHUNK', '2048')))
    result = (f'[doom-wide] verify_blocks matched={vr.all_matched} forwards={vr.forwards} '
              f'accepted={vr.accepted_steps}/{vr.total_steps} wall={time.time()-t0:.1f}s '
              f'peak_vram={stats.get("peak_vram_gb",0):.1f}GB\n')
    if vr.first_mismatch:
        result += f'[doom-wide] FIRST MISMATCH: {vr.first_mismatch}\n'
    result += f'[doom-wide] DOOM byte-exact on the WIDE-LEA transformer: {vr.all_matched}\n'
    print(result, flush=True)
    rf = os.environ.get('DOOM_RESULT_FILE')
    if rf:
        with open(rf, 'w') as f:
            f.write(result)
    return 0 if vr.all_matched else 1


if __name__ == '__main__':
    raise SystemExit(main())
