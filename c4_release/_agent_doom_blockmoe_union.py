"""Measure the block-MoE lever for doom: over a real K-step doom span, what is the
live-union of blocks?  If it's ~all 242, block-MoE can't help (heterogeneous ops).
If it's small, doom could run only the live blocks per span (like self-emu).
Also report the per-op live-block counts and the op-frequency in the doom stream.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('C4_KV_STACK', '1')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from collections import Counter
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def main():
    dev = 'cuda:0'
    cap = int(os.environ.get('N_STEPS_CAP', '4000'))
    K = int(os.environ.get('K', '25'))
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)

    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code)+2, 64), recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    nblk = len(sparse.blocks)
    from c4_min.step_block_skip import build_live_index
    live_index = build_live_index(sparse, L)
    full = set(range(nblk))

    # op frequency in the doom stream
    op_counts = Counter(fr['op'] for fr in draft.frames)
    print(f'[moe] doom stream op frequency (cap={cap} steps): '
          f'{dict(op_counts.most_common(20))}', flush=True)

    # per-op live-block count
    name_to_op = {v: k for k, v in isa.NAMES.items()}
    print(f'[moe] per-op live-block count (of {nblk} blocks):', flush=True)
    op_live = {}
    for opname, cnt in op_counts.most_common():
        opcode = name_to_op.get(opname)
        live = live_index.get(opcode, live_index.get(None))
        op_live[opname] = len(live) if live is not None else nblk
        print(f'[moe]   {opname:5s} x{cnt:6d}: {op_live[opname]:3d} live blocks', flush=True)

    # live-UNION over consecutive K-step doom spans (the real verify_blocks span)
    unions = []
    n_steps = draft.step_count
    for s0 in range(0, n_steps - K, max(1, (n_steps - K) // 50)):
        u = set()
        for s in range(s0, min(s0 + K, n_steps)):
            opname = draft.frames[s]['op']
            opcode = name_to_op.get(opname)
            live = live_index.get(opcode)
            if live is None:
                u = full; break
            u |= set(live)
        unions.append(len(u))
    import statistics
    print(f'\n[moe] === LIVE-UNION over K={K}-step doom spans (50 samples) ===', flush=True)
    print(f'[moe]   union blocks: min={min(unions)} median={int(statistics.median(unions))} '
          f'max={max(unions)}  (of {nblk} total)', flush=True)
    print(f'[moe]   -> block-MoE would run median {int(statistics.median(unions))}/{nblk} blocks '
          f'= {100*statistics.median(unions)/nblk:.0f}% of the FFN.', flush=True)
    # also K=1 (single-step decode-live set), the theoretical best
    singles = []
    for s in range(0, n_steps, max(1, n_steps // 200)):
        opname = draft.frames[s]['op']; opcode = name_to_op.get(opname)
        live = live_index.get(opcode)
        singles.append(nblk if live is None else len(live))
    print(f'[moe]   per-STEP (K=1) live blocks: median={int(statistics.median(singles))} '
          f'max={max(singles)}  -> the block-MoE floor if K=1.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
