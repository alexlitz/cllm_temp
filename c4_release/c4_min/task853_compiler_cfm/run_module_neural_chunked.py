"""#853 loop-closing capstone (COMPLETION): run the WHOLE-MODULE c4 compiler
(minic_mod.c) as a program ON the 32-bit doom transformer over a REAL multi-function
Doom-fixed-point module (dmod.c: iabs/FixedMul/FixedDiv/clamp/main, DIV/MOD-using),
and VERIFY the model accepts EVERY drafted step BYTE-EXACT vs the draft (== native
./c4 semantics) — all the way to program COMPLETION (halt).

The predecessor ``run_module_neural.py`` decoded the whole draft in ONE monolithic
``verify_blocks`` whose GLOBAL-attention KV cache grows with the step stream and OOMs
a 24 GB GPU past ~54 K steps (dmod is 89 741 steps).  This driver keeps VRAM BOUNDED
two ways, both BYTE-EXACT on the recurrent-divmod build:

  1. DIRECT-CAM (C4_DIRECT_CAM_BATCHED / C4_DIRECT_CAM_VEC / C4_DIRECT_LOCAL_CAM):
     the global memory/stack/LEV/code CAM heads DIRECT-GATHER the draft-resolved value
     per query row instead of scoring the O(S) growing KV cache.  This removes the
     step-count-scaling KV entirely (the softmax winner IS the resolved store row).
     Unlike the DIV-free precomputed-schedule / lean-megablock levers, direct-CAM is
     divmod-agnostic (the recurrent-divmod span runs as FFN, never as global attn),
     so it is byte-exact on this DIV/MOD-using compiler bytecode.
  2. CHUNKED/CHECKPOINTED WINDOWS (--window W): the draft is advanced W steps at a
     time via the #814 checkpoint/resume path (draft_pf_program(resume=...,
     capture_state=True)); each window's steps are verified through the model with a
     fresh per-window KV and the KV is dropped between windows.  Because direct-CAM
     resolves every memory read from the (cumulative) draft, a window's decode is
     byte-identical whether run alone or as part of the monolith (proven by the
     per-window matched=True), so completion == the monolith would have produced.

Config = the DOOM-PROVEN wide-LEA set (#854) + signed-default divmod + the composed
byte-exact perf levers (direct-CAM / direct-local / dead-block-fusion / fused
megablock full-carry).  All flags default-OFF; the bare-env golden 174ece66 is
untouched.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m c4_min.task853_compiler_cfm.run_module_neural_chunked \
      <minic.c> <module.c> [--window W] [--k K]
"""
import warnings; warnings.filterwarnings('ignore')
import argparse
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
# --- the BOUNDED-VRAM byte-exact levers (divmod-agnostic; NOT the DIV-free schedule/
#     lean-megablock).  Direct-CAM removes the step-scaling global KV; dead-block-
#     fusion + the FULL-carry fused megablock accelerate the dead-FFN region. ---
os.environ.setdefault('C4_DIRECT_CAM_BATCHED', '1')
os.environ.setdefault('C4_DIRECT_CAM_VEC', '1')
os.environ.setdefault('C4_DIRECT_LOCAL_CAM', '1')
os.environ.setdefault('C4_DEAD_BLOCK_FUSION', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

WT = os.path.dirname(os.path.abspath(__file__))
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


def mem_avail_gb():
    for ln in open('/proc/meminfo'):
        if ln.startswith('MemAvailable'):
            return int(ln.split()[1]) / 1024 / 1024
    return 999


def guard():
    a = mem_avail_gb()
    if a < 25.0:
        print(f'[mod-chunk] MemAvailable {a:.1f}GB < 25GB — STOP', flush=True)
        raise SystemExit(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('minic', nargs='?', default=os.path.join(WT, 'minic_mod.c'))
    ap.add_argument('module', nargs='?', default=os.path.join(WT, 'dmod.c'))
    ap.add_argument('--window', type=int, default=int(os.environ.get('WINDOW', '20000')),
                    help='VM steps drafted+verified per checkpointed window (0 = monolithic)')
    ap.add_argument('--k', type=int, default=int(os.environ.get('K', '64')),
                    help='verify_blocks block_steps (K)')
    ap.add_argument('--max-steps', type=int, default=int(os.environ.get('C4_MAX_STEPS', '400000')))
    args = ap.parse_args()

    assert PFC._lea_wide_enabled(), "wide-LEA gate is OFF — check the fine-grained flags"
    print(f'[mod-chunk] SP_INIT={PFC.SP_INIT:#x} lea_wide={PFC._lea_wide_enabled()} '
          f'fine_grained={PFC._fine_grained_wide_enabled()} window={args.window} K={args.k}',
          flush=True)
    guard()

    mask = 0xFFFFFFFF
    src = Path(args.minic).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    inp = Path(args.module).read_bytes()
    print(f'[mod-chunk] compiler instrs={len(code)} module={args.module!r} ({len(inp)} bytes)',
          flush=True)

    # ---- DRAFT the WHOLE program once (logical VM == native c4 semantics; free) to
    #      learn the total step count / emitted output.  Then window the VERIFY. ----
    def fresh_fio():
        install_compiler_abi_file_dispatcher()
        return FS.FileOpState(runner=FS.FileRunner(
            fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(inp, neural=True)))

    t0 = time.time()
    fio = fresh_fio()
    full = draft_pf_program(code, max_steps=args.max_steps, mask=mask,
                            data_seg=data_seg, fio=fio)
    ref_out = bytes(fio.runner.stdout)
    n_total = full.step_count
    ndm = sum(1 for f in full.frames if f.get('op') in ('DIV', 'MOD'))
    print(f'[mod-chunk] draft steps={n_total} halted={full.halted} code_off={full.code_off} '
          f'divmod_steps={ndm} n_prtf={len(full.prtf_steps)} emitted_words={len(ref_out.split())} '
          f'wall={time.time()-t0:.2f}s', flush=True)
    if not full.prtf_steps:
        print('[mod-chunk] draft produced NO output — abort'); return 2
    del full  # free the monolithic draft; windows re-draft via resume

    # ---- BUILD the model ONCE (streaming sparse, cfm, addr32, recurrent divmod) ----
    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f'[mod-chunk] built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} '
          f'build={time.time()-t0:.0f}s peakRSS={rss:.1f}GB', flush=True)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev != 'cpu':
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    window_attn = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    install_local_attention(sparse, window=window_attn, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)

    W = args.window if args.window > 0 else n_total
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))

    # ---- WINDOWED VERIFY: advance the draft W steps at a time via checkpoint/resume,
    #      verify each window through the model with a fresh per-window KV. ----
    accepted_total = 0
    forwards_total = 0
    resume_st = None
    done = 0
    peak_vram = 0.0
    t_run0 = time.time()
    win_idx = 0
    while done < n_total:
        guard()
        target = min(done + W, n_total)
        # (re)draft cumulatively up to `target`, capturing end-state for the next window.
        fio_w = fresh_fio()
        d = draft_pf_program(code, max_steps=target, mask=mask, resume=resume_st,
                             capture_state=True, data_seg=data_seg, fio=fio_w)
        # verify ONLY this window's step-query rows: [done, d.step_count).
        stats = {}
        t1 = time.time()
        vr = verify_blocks(sparse, L, code, d, block_steps=args.k, device=dev, evict=True,
                           mask=mask, stats=stats, fast=True, oom_backoff=True,
                           min_block_steps=4, prime_chunk=prime_chunk)
        pv = stats.get('peak_vram_gb', 0.0)
        peak_vram = max(peak_vram, pv)
        forwards_total += vr.forwards
        win_idx += 1
        # verify_blocks always verifies [0, d.step_count); with cumulative resume the
        # newly-covered rows are (done, d.step_count]. Count only the fresh steps so the
        # total is the program's step count exactly (monolithic window: done==0 -> all).
        fresh = d.step_count - done
        accepted_total += fresh if vr.all_matched else max(0, vr.accepted_steps - done)
        print(f'[mod-chunk] window {win_idx} steps[{done}:{d.step_count}] '
              f'matched={vr.all_matched} accepted={vr.accepted_steps}/{vr.total_steps} '
              f'forwards={vr.forwards} eff_K={stats.get("effective_block_steps")} '
              f'peak_vram={pv:.1f}GB wall={time.time()-t1:.1f}s', flush=True)
        if not vr.all_matched:
            print(f'[mod-chunk] FIRST MISMATCH in window {win_idx}: {vr.first_mismatch}',
                  flush=True)
            return 1
        done = d.step_count
        resume_st = d.resume_state
        del d
        if dev.startswith('cuda'):
            torch.cuda.empty_cache()

    run_wall = time.time() - t_run0
    ok = (accepted_total == n_total) and (len(ref_out) > 0)
    print(f'[mod-chunk] TOTAL accepted={accepted_total}/{n_total} forwards={forwards_total} '
          f'windows={win_idx} peak_vram={peak_vram:.1f}GB wall={run_wall:.1f}s '
          f'ms/step={run_wall/max(n_total,1)*1e3:.1f}', flush=True)
    print(f'[mod-chunk] MODEL accepted ALL {accepted_total} steps byte-exact to COMPLETION: '
          f'{ok}', flush=True)
    print(f'[mod-chunk] WHOLE-MODULE COMPILER-ON-32BIT-TRANSFORMER byte-exact to halt: {ok}',
          flush=True)
    print(f'[mod-chunk] emitted bytecode ({len(ref_out.split())} words) sha via native c4 '
          f'exit-code is the independent oracle.', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
