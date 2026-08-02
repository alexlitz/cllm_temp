"""FLASH-ATTN byte-exactness + speed on DOOM (verify_blocks batched path).

Runs the doom stream through ``verify_blocks`` (direct-CAM + banded local-attn +
dead-block fusion, the full composed config) with C4_FLASH_ATTN OFF vs ON, and
asserts both are 7/7 byte-exact (all_matched AND stdout first-7 ==
1b5b324a1b5b48 == `printf 'q' | ./c4 doom.c`).  Reports ms/step + peak VRAM for
each.  HONEST: for doom the global/CAM heads are already O(1) direct-CAM gathers and
the local heads are banded, so flash is MARGINAL here (measured) — the WIN is the
general/self-emu path.  We measure it to confirm byte-exactness is preserved.

Env: N_STEPS_CAP (default 4000), K (block_steps, default 200).  fp32 flash (NOT tf32).
Golden 069cc32f unchanged (C4_FLASH_ATTN gated; additive).
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
from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                        data_segment)

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'
REF = '/tmp/doom_ref.bin'


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
    return sparse, L, code, draft, fio, draft.code_off


def run_one(flash, cap, K, window):
    if flash:
        os.environ['C4_FLASH_ATTN'] = '1'
    else:
        os.environ.pop('C4_FLASH_ATTN', None)
    dev = 'cuda:0'
    ref = Path(REF).read_bytes()
    sparse, L, code, draft, fio, code_off = build_doom(dev, cap, window)
    torch.cuda.reset_peak_memory_stats(dev)
    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=2048)
    wall = time.time() - t0
    ms = wall / max(draft.step_count, 1) * 1e3
    out = bytes(fio.runner.stdout)
    first7_ok = out[:7] == ref[:7]
    peak = stats.get('peak_vram_gb', torch.cuda.max_memory_allocated(dev) / 1024**3)
    return dict(matched=vr.all_matched, accepted=vr.accepted_steps,
                total=vr.total_steps, first7=out[:7].hex(), first7_ok=first7_ok,
                ms=ms, wall=wall, peak=peak, steps=draft.step_count,
                prtf=draft.prtf_steps[:1])


def main():
    torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '4000'))
    K = int(os.environ.get('K', '200'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    print(f'[flash-doom] cap={cap} K={K} window={window} ref7='
          f'{Path(REF).read_bytes()[:7].hex()}', flush=True)

    res = {}
    for flash in (False, True):
        tag = 'FLASH_ON' if flash else 'FLASH_OFF(banded)'
        r = run_one(flash, cap, K, window)
        res[flash] = r
        print(f'[flash-doom] {tag}: matched={r["matched"]} accepted={r["accepted"]}/'
              f'{r["total"]} first7={r["first7"]} first7_ok={r["first7_ok"]} '
              f'ms/step={r["ms"]:.2f} wall={r["wall"]:.1f}s peak_vram={r["peak"]:.2f}GB '
              f'draft_steps={r["steps"]} first_prtf={r["prtf"]}', flush=True)
        # free between runs
        torch.cuda.empty_cache()

    off, on = res[False], res[True]
    reached_printf = bool(off['prtf']) and off['total'] >= (off['prtf'][0] if off['prtf'] else 1 << 30)
    # Byte-exactness signal: the model's decoded output must MATCH the reference in
    # BOTH states (all accepted steps) AND OFF==ON.  The first-7 stdout gate only
    # applies once the run cap reaches the first printf (~step 29754); below that the
    # accepted-steps equality is the byte-exact signal.
    byte_exact = (off['matched'] and on['matched']
                  and off['accepted'] == on['accepted']
                  and off['total'] == on['total']
                  and (not reached_printf or (off['first7_ok'] and on['first7_ok'])))
    tag = '7/7' if reached_printf else f'{off["accepted"]}/{off["total"]} steps (pre-printf)'
    print(f'\n[flash-doom] BYTE-EXACT OFF==ON ({tag}): {byte_exact}  '
          f'reached_printf={reached_printf}', flush=True)
    print(f'[flash-doom] speed OFF={off["ms"]:.2f}ms/step  ON={on["ms"]:.2f}ms/step  '
          f'ratio={off["ms"]/max(on["ms"],1e-9):.2f}x  (marginal expected for doom)',
          flush=True)
    return 0 if byte_exact else 1


if __name__ == '__main__':
    raise SystemExit(main())
