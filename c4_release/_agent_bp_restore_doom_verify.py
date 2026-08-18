"""Verify the C4_BP_RESTORE_HIBYTE fix does NOT regress DOOM.

Runs the REAL id linuxdoom stream through the byte-exact doom verify_blocks path with
C4_BP_RESTORE_HIBYTE OFF then ON, asserting all_matched (byte-exact) in BOTH states.
Also characterises doom's LEV traffic in the tested span: how many LEVs, and the max
saved-BP / return-PC value (to confirm they fit in the 5-nibble fp32-safe recompose
range that the flag-ON path reads — i.e. the flag-ON path is byte-exact for doom's LEVs).
"""
from __future__ import annotations
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
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                        data_segment)

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def build_doom(dev, cap_steps, window):
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                              stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap_steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code) + 2, 64),
                                             recurrent_divmod=True, addr32=True,
                                             compute_mode='dense_kernel')
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    from c4_min import direct_cam_batched as DCB
    DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
    return sparse, L, code, draft, fio


def lev_stats(draft):
    """Count LEV steps and the max saved-BP / return-PC value in the drafted span."""
    n_lev = 0
    max_bp = max_pc = 0
    for fr in draft.frames:
        if fr.get('op') == 'LEV':
            n_lev += 1
            max_bp = max(max_bp, fr['bp'] & 0xFFFFFFFF)
            max_pc = max(max_pc, fr['pc'] & 0xFFFFFFFF)
    return n_lev, max_bp, max_pc


def run(flag_on, cap, K, window, dev):
    if flag_on:
        os.environ['C4_BP_RESTORE_HIBYTE'] = '1'
    else:
        os.environ.pop('C4_BP_RESTORE_HIBYTE', None)
    sparse, L, code, draft, fio = build_doom(dev, cap, window)
    n_lev, max_bp, max_pc = lev_stats(draft)
    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=2048)
    wall = time.time() - t0
    del sparse; torch.cuda.empty_cache()
    return dict(matched=vr.all_matched, accepted=vr.accepted_steps, total=vr.total_steps,
                fm=vr.first_mismatch, n_lev=n_lev, max_bp=max_bp, max_pc=max_pc, wall=wall)


def main():
    torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '30000'))  # reach the first LEVs
    K = int(os.environ.get('K', '200'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    dev = 'cuda:0'
    print(f'[bp-doom] cap={cap} K={K} window={window}', flush=True)
    res = {}
    for on in (False, True):
        r = run(on, cap, K, window, dev)
        res[on] = r
        tag = 'ON ' if on else 'OFF'
        print(f'[bp-doom] BP_RESTORE_HIBYTE={tag}: matched={r["matched"]} '
              f'accepted={r["accepted"]}/{r["total"]} n_LEV={r["n_lev"]} '
              f'max_saved_bp={r["max_bp"]} max_pc={r["max_pc"]} wall={r["wall"]:.1f}s', flush=True)
        if not r['matched'] and r['fm']:
            print(f'[bp-doom]   first_mismatch={r["fm"]}', flush=True)
    off, on = res[False], res[True]
    # 5-nibble recompose range = bits 0..19 = 0xFFFFF (1048575).  Confirm doom's LEV
    # saved-BP + return-PC all fit (so the flag-ON path is byte-exact for doom).
    fits5 = max(off['max_bp'], off['max_pc']) <= 0xFFFFF
    ok = off['matched'] and on['matched'] and off['accepted'] == on['accepted']
    print(f'\n[bp-doom] DOOM byte-exact OFF={off["matched"]} ON={on["matched"]} '
          f'accepted_OFF==ON={off["accepted"] == on["accepted"]}', flush=True)
    print(f'[bp-doom] doom LEV values fit 5-nibble range (<=0xFFFFF): {fits5} '
          f'(max_bp={off["max_bp"]:#x} max_pc={off["max_pc"]:#x})', flush=True)
    print(f'[bp-doom] >>> DOOM NOT REGRESSED (OFF & ON byte-exact): {ok} <<<', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
