"""#853 loop-closing capstone: run the WHOLE-MODULE c4 compiler (minic_mod.c) as a
program ON the 32-bit doom transformer over a small multi-function Doom module, and
VERIFY the model accepts EVERY drafted step BYTE-EXACT vs the draft (== native ./c4
semantics).  minic_mod emits a real multi-function compilation (globals + params +
locals + inter-function JSR calls); the model reproducing every step byte-exact is
the compiler-as-a-program-on-the-transformer capstone.

Config = the DOOM-PROVEN wide-LEA set (#854) so BP>255 frame locals resolve, plus
the COMPOSED perf stack (direct-CAM / local-attn / dead-block-fusion / big-K
verify) so the ~tens-of-thousands of drafted steps run tractably.  All flags are
default-OFF; the bare-env golden 3cabef64 is untouched.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m c4_min.task853_compiler_cfm.run_module_neural \
      <minic.c> <module.c> [K]
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time, resource
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '4')
# --- CFM / addr32 / read-to-mem (as #853) ---
os.environ['C4_PF_CFM'] = '1'
os.environ['C4_DRAFT_CMP32'] = '1'
os.environ['C4_DRAFT_READ_TO_MEM'] = '1'
# --- the DOOM-PROVEN fine-grained wide set (enables _fine_grained_wide_enabled) ---
os.environ['C4_CMP32'] = '1'
os.environ['C4_CMP32_ORDER'] = '1'
os.environ['C4_SHIFT32'] = '1'
os.environ['C4_PC_WIDE'] = '1'
os.environ['C4_CODE_ADDR_BITS'] = '20'
os.environ['C4_GLOBAL_ADDR32'] = '1'
os.environ['C4_SP_WIDE'] = '1'
os.environ['C4_DIVMOD_SIGNED'] = '1'
# --- the wide-LEA lever + the raised stack base (the two coupled #854 levers) ---
os.environ['C4_LEA_WIDE'] = '1'
os.environ.setdefault('C4_SP_INIT', '0xF000')
os.environ.setdefault('C4_IMM_NIBS', '6')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('C4_EXACT_EVICT', '1')
os.environ.setdefault('C4_MEM_EFF', '2000000')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

WT = os.path.dirname(os.path.abspath(__file__))
# this file lives at <repo>/c4_min/task853_compiler_cfm/; repo root = up two
REPO = os.path.dirname(os.path.dirname(WT))
sys.path.insert(0, REPO)
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks
import c4_min.nibble_pure_forward_complete as PFC
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

MINIC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WT, 'minic_mod.c')
MODULE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WT, 'mod1.c')


def mem_avail_gb():
    for ln in open('/proc/meminfo'):
        if ln.startswith('MemAvailable'):
            return int(ln.split()[1]) / 1024 / 1024
    return 999


def main():
    assert PFC._lea_wide_enabled(), "wide-LEA gate is OFF — check the fine-grained flags"
    print(f'[mod-wide] SP_INIT={PFC.SP_INIT:#x} lea_wide={PFC._lea_wide_enabled()} '
          f'fine_grained={PFC._fine_grained_wide_enabled()}', flush=True)
    if mem_avail_gb() < 25:
        print(f'[mod-wide] MemAvailable {mem_avail_gb():.1f}GB < 25GB — abort'); return 3
    K = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get('K', '256'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    max_steps = int(os.environ.get('C4_MAX_STEPS', '400000'))

    src = Path(MINIC).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    inp = Path(MODULE).read_bytes()
    print(f'[mod-wide] compiler instrs={len(code)} module={MODULE!r} ({len(inp)} bytes) '
          f'K={K} max_steps={max_steps}', flush=True)

    # ---- DRAFT (logical VM == native c4 semantics; free) ----
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                              stdin=FS.InputKVStream(inp, neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    ref_out = bytes(fio.runner.stdout)
    print(f'[mod-wide] draft steps={draft.step_count} halted={draft.halted} '
          f'code_off={draft.code_off} n_prtf={len(draft.prtf_steps)} '
          f'emitted_words={len(ref_out.split())} wall={time.time()-t0:.2f}s', flush=True)
    if not draft.prtf_steps:
        print('[mod-wide] draft produced NO output — abort'); return 2

    # ---- BUILD (streaming sparse, cfm, addr32, wide-LEA blocks appended) ----
    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f'[mod-wide] built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} '
          f'build={time.time()-t0:.0f}s peakRSS={rss:.1f}GB', flush=True)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev != 'cpu':
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    # content_bound_global=True bounds the GLOBAL-head KV cache to store-role rows
    # (memory-CAM/stack-pop/LEV), so the O(S^2) global-attention matmul stays
    # tractable at tens of thousands of steps (a monolithic unbounded pass OOMs the
    # 24 GB GPU at ~90k steps).  Byte-exactness is preserved: the dropped rows carry
    # zero attention weight for the global heads.
    cbg = os.environ.get('C4_CONTENT_BOUND_GLOBAL', '1') == '1'
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=cbg, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)

    # ---- VERIFY (fast big-K): the MODEL must accept every draft step ----
    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4,
                       prime_chunk=int(os.environ.get('DOOM_PRIME_CHUNK', '2048')))
    run_wall = time.time() - t0
    ms = run_wall / max(draft.step_count, 1) * 1e3
    print(f'[mod-wide] verify_blocks matched={vr.all_matched} forwards={vr.forwards} '
          f'accepted={vr.accepted_steps}/{vr.total_steps} wall={run_wall:.1f}s '
          f'ms/step={ms:.1f} eff_K={stats.get("effective_block_steps")} '
          f'peak_vram={stats.get("peak_vram_gb",0):.1f}GB', flush=True)
    if vr.first_mismatch:
        print(f'[mod-wide] FIRST MISMATCH: {vr.first_mismatch}', flush=True)

    ok = vr.all_matched and len(ref_out) > 0
    print(f'[mod-wide] MODEL accepted all {vr.accepted_steps} steps byte-exact: '
          f'{vr.all_matched}', flush=True)
    print(f'[mod-wide] WHOLE-MODULE COMPILER-ON-32BIT-TRANSFORMER byte-exact: {ok}',
          flush=True)
    print(f'[mod-wide] emitted bytecode ({len(ref_out.split())} words):\n'
          f'{ref_out.decode("latin1")}', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
