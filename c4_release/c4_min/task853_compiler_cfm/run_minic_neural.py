"""Run the c4 COMPILER (minic.c) bytecode AS A PROGRAM on the 32-bit pure-forward
doom transformer (CFM / code-from-memory route), feeding a small C input via stdin,
and VERIFY the neural model accepts every step byte-exact vs the draft (== native c4
semantics).  The model's emitted bytecode must equal the reference compilation.

Composes the perf stack (cfm code-CAM + addr32 + recurrent divmod + big-K
verify_blocks + local-attn + dead-block-fusion) exactly like the doom continuous run.

Flags (all default-OFF, golden 7d4afe61 unchanged bare-env):
  C4_PF_CFM=1              code-from-memory (fixed dim)
  C4_DRAFT_CMP32=1         32-bit signed cmp + 32-bit JSR return-PC (REQUIRED: minic
                           is >256 instrs so LEV returns to a >255 PC)
  C4_DRAFT_READ_TO_MEM=1   lay READ's stdin bytes into the draft mem so the compiler
                           re-reads its source (NEW flag this task)
  C4_MEM_EFF, C4_EXACT_EVICT  recall-horizon + scalable eviction (doom wall #6 fixes)
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time, resource
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ['C4_PF_CFM']='1'
os.environ['C4_DRAFT_CMP32']='1'
os.environ['C4_SHIFT32']='1'
os.environ['C4_DRAFT_READ_TO_MEM']='1'
os.environ.setdefault('C4_MEM_ADDR_BITS','18')
os.environ.setdefault('C4_EXACT_EVICT','1')
os.environ.setdefault('C4_MEM_EFF','2000000')
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
sys.path.insert(0,'/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/task853-compiler-cfm/c4_release')
sys.path.insert(0,'/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks, _draft_cmp32
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

MINIC = sys.argv[1] if len(sys.argv)>1 else '/tmp/t853/minic.c'
INPUT = sys.argv[2] if len(sys.argv)>2 else '/tmp/t853/in.c'

def mem_avail_gb():
    for ln in open('/proc/meminfo'):
        if ln.startswith('MemAvailable'):
            return int(ln.split()[1])/1024/1024
    return 999

def main():
    assert _draft_cmp32()
    if mem_avail_gb() < 25:
        print(f'[minic-neural] MemAvailable {mem_avail_gb():.1f}GB < 25GB — abort (share the box)'); return 3
    K=int(os.environ.get('K','200'))
    window=int(os.environ.get('C4_LOCAL_WINDOW','96'))

    src=Path(MINIC).read_text()
    bc,data=compile_c(src)
    code=tag_compiler_syscalls(bytecode_to_isa(bc),isa)
    data_seg=data_segment(data)
    inp=Path(INPUT).read_bytes()
    print(f'[minic-neural] compiler instrs={len(code)} input={INPUT!r} ({len(inp)} bytes) K={K}', flush=True)

    # ---- DRAFT (logical VM == native c4 semantics; free) ----
    install_compiler_abi_file_dispatcher()
    fio=FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(inp, neural=True)))
    t0=time.time()
    draft=draft_pf_program(code, max_steps=int(os.environ.get('C4_MAX_STEPS','20000')),
                           mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    ref_out=bytes(fio.runner.stdout)
    print(f'[minic-neural] draft steps={draft.step_count} halted={draft.halted} '
          f'code_off={draft.code_off} n_prtf={len(draft.prtf_steps)} '
          f'REF_OUTPUT={ref_out!r} wall={time.time()-t0:.2f}s', flush=True)
    if not draft.prtf_steps:
        print('[minic-neural] draft produced NO output — abort'); return 2

    # ---- BUILD (streaming sparse, cfm, addr32) ----
    from c4_min.lib_neural import build_lib_model_streaming
    cs=max(len(code)+2,64)
    t0=time.time()
    sparse,L,_=build_lib_model_streaming(code_size=cs, recurrent_divmod=True, addr32=True,
                                         compute_mode='dense_kernel')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    print(f'[minic-neural] built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} '
          f'build={time.time()-t0:.0f}s peakRSS={rss:.1f}GB', flush=True)
    dev='cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev!='cpu': sparse=sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)

    # ---- VERIFY (fast big-K): the MODEL must accept every draft step ----
    stats={}
    t0=time.time()
    vr=verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                     mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True, min_block_steps=4,
                     prime_chunk=int(os.environ.get('DOOM_PRIME_CHUNK','2048')))
    run_wall=time.time()-t0
    ms=run_wall/max(draft.step_count,1)*1e3
    print(f'[minic-neural] verify_blocks matched={vr.all_matched} forwards={vr.forwards} '
          f'accepted={vr.accepted_steps}/{vr.total_steps} wall={run_wall:.1f}s ms/step={ms:.1f} '
          f'eff_K={stats.get("effective_block_steps")} peak_vram={stats.get("peak_vram_gb",0):.1f}GB', flush=True)
    if vr.first_mismatch:
        print(f'[minic-neural] FIRST MISMATCH: {vr.first_mismatch}', flush=True)

    ok = vr.all_matched and len(ref_out)>0
    print(f'[minic-neural] MODEL accepted all {vr.accepted_steps} steps byte-exact: {vr.all_matched}', flush=True)
    print(f'[minic-neural] COMPILER-ON-32BIT-TRANSFORMER byte-exact: {ok}', flush=True)
    print(f'[minic-neural] emitted bytecode:\n{ref_out.decode("latin1")}', flush=True)
    return 0 if ok else 1

if __name__=='__main__':
    raise SystemExit(main())
