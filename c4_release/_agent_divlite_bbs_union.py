"""Block-MoE union via the REAL BatchedBlockSkipPlan (what the transformer prunes).

Builds the model once, then for doom.c and doom_divlite.c measures the per-span
block-MoE union (BatchedBlockSkipPlan.span_live_mask over the span's opcodes) in
TWO phases: the DIV-heavy INIT (first spans) and the STEADY RENDER (a full frame
region after init).  The headline: in the render phase, does DIV-light shrink
the union below the DIV megablock so block-MoE prunes far more than original?

golden 069cc32f UNCHANGED (measurement only).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, statistics
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('C4_BATCHED_BLOCK_SKIP', '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
from collections import Counter
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                        data_segment)


def draft_of(path, max_steps):
    src = open(path).read()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    ds = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b'q', neural=True)))
    return draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF, data_seg=ds, fio=fio)


def span_unions(plan, opstr_to_int, dr, lo, hi, K, nb):
    """union block count per K-step span over [lo,hi).  span_live_mask takes op
    STRINGS (frame op names), NOT ints."""
    out = []
    step = max(1, (hi - lo - K) // 40)
    for s0 in range(lo, hi - K, step):
        ops = set(dr.frames[s]['op'] for s in range(s0, min(s0 + K, hi)))
        mask = plan.span_live_mask(list(ops))
        out.append(int(mask.sum()))
    return out


def main():
    K = int(os.environ.get('K', '200'))
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.batched_block_skip import BatchedBlockSkipPlan
    sparse, L, _ = build_lib_model_streaming(code_size=4400, recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    nb = len(sparse.blocks)
    plan = BatchedBlockSkipPlan(sparse, L)
    # opstr -> int (isa)
    opstr_to_int = {v: k for k, v in isa.NAMES.items()}
    print(f'[bbs] model blocks={nb} DIV live-distinct={len(plan.op_live_distinct.get(isa.DIV,set()))} '
          f'divmod-span={len(plan.div_distinct)}', flush=True)

    for path, label, ms in [
        ('/home/alexlitz/Documents/misc/c4_doom/doom.c', 'ORIG', 60000),
        ('/home/alexlitz/Documents/misc/c4_doom/doom_divlite.c', 'DIVLITE', 120000)]:
        dr = draft_of(path, ms)
        fp = dr.prtf_steps[0] if dr.prtf_steps else dr.step_count
        # INIT phase: first 6000 steps.  RENDER phase: last 6000 steps before printf.
        init_u = span_unions(plan, opstr_to_int, dr, 0, min(6000, fp), K, nb)
        r_lo = max(0, fp - 6000)
        rend_u = span_unions(plan, opstr_to_int, dr, r_lo, fp, K, nb)
        div_init = sum(1 for s in range(0, min(6000, fp)) if dr.frames[s]['op'] in ('DIV', 'MOD'))
        div_rend = sum(1 for s in range(r_lo, fp) if dr.frames[s]['op'] in ('DIV', 'MOD'))
        print(f'\n[bbs] === {label}  (first printf @ {fp}) ===', flush=True)
        print(f'[bbs]   INIT   [0..{min(6000,fp)}]:  DIV/MOD={div_init} (1/{min(6000,fp)/max(1,div_init):.0f})  '
              f'union median={int(statistics.median(init_u))}/{nb} '
              f'({100*statistics.median(init_u)/nb:.0f}%) max={max(init_u)}', flush=True)
        print(f'[bbs]   RENDER [{r_lo}..{fp}]:  DIV/MOD={div_rend} (1/{6000/max(1,div_rend):.0f})  '
              f'union median={int(statistics.median(rend_u))}/{nb} '
              f'({100*statistics.median(rend_u)/nb:.0f}%) max={max(rend_u)} min={min(rend_u)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
