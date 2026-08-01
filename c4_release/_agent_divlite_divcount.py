"""DIV/MOD reduction measurement: doom.c vs doom_divlite.c.

Compiles both to c4_min ISA, counts STATIC DIV/MOD, then drafts each to the
FIRST printf and counts DYNAMIC DIV/MOD + the DIV STEP FREQUENCY, split into
the ONE-TIME cold init (init_sin / init_recip table builds) vs the per-frame
RENDER/game-logic divides (the ones block-MoE actually sees over a real game).
No model build -- pure draft (free).  golden 069cc32f unchanged.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
from collections import Counter
from src.compiler import compile_c
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                        data_segment)


def analyze(path, label, max_steps):
    src = open(path).read()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    sc = Counter(isa.NAMES.get(i.op, str(i.op)) for i in code)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b'q', neural=True)))
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    fp = draft.prtf_steps[0] if draft.prtf_steps else None
    div = sum(1 for fr in draft.frames if fr['op'] == 'DIV')
    mod = sum(1 for fr in draft.frames if fr['op'] == 'MOD')
    shr = sum(1 for fr in draft.frames if fr['op'] == 'SHR')
    shl = sum(1 for fr in draft.frames if fr['op'] == 'SHL')
    divpc = Counter()
    for fr in draft.frames:
        if fr['op'] == 'DIV':
            divpc[fr['pc']] += 1
    # Cold one-time table builds show up as a single PC with a huge count
    # (init_sin ~65/entry-PC, init_recip == DIST_MAX iterations at one PC).
    cold = sum(v for pc, v in divpc.items() if v > 400)
    hot = div - cold
    print(f'\n=== {label} ({path.split("/")[-1]}) ===', flush=True)
    print(f'  static:  instrs={len(code)}  DIV={sc.get("DIV",0)}  MOD={sc.get("MOD",0)}'
          f'  MUL={sc.get("MUL",0)}  SHL={sc.get("SHL",0)}  SHR={sc.get("SHR",0)}', flush=True)
    print(f'  dynamic (to first printf, step={fp}, ran {draft.step_count} steps):', flush=True)
    print(f'    DIV={div}  MOD={mod}  SHR={shr}  SHL={shl}', flush=True)
    print(f'    DIV split: COLD one-time table builds={cold}  HOT render/game={hot}', flush=True)
    if div:
        print(f'    DIV freq (whole run incl cold init): 1 per {draft.step_count/div:.1f} steps', flush=True)
    if hot:
        print(f'    DIV freq (HOT render/game only): 1 per {draft.step_count/hot:.1f} steps', flush=True)
    else:
        print(f'    DIV freq (HOT render/game only): ZERO -- no per-frame DIV opcodes', flush=True)
    top_hot = [(pc, v) for pc, v in divpc.most_common() if v <= 400][:12]
    print(f'    HOT DIV PCs: {top_hot}', flush=True)
    return dict(instrs=len(code), static_div=sc.get('DIV', 0), dyn_div=div,
                cold=cold, hot=hot, steps=draft.step_count, fp=fp)


def main():
    o = analyze('/home/alexlitz/Documents/misc/c4_doom/doom.c', 'ORIGINAL', 60000)
    d = analyze('/home/alexlitz/Documents/misc/c4_doom/doom_divlite.c', 'DIV-LIGHT', 200000)
    print('\n=== SUMMARY ===', flush=True)
    print(f'  static DIV:  {o["static_div"]} -> {d["static_div"]}  '
          f'({100*(o["static_div"]-d["static_div"])/max(1,o["static_div"]):.0f}% fewer)', flush=True)
    print(f'  HOT (per-frame) DIV to first printf: {o["hot"]} -> {d["hot"]}  '
          f'({100*(o["hot"]-d["hot"])/max(1,o["hot"]):.0f}% fewer)', flush=True)
    o_freq = o['steps'] / max(1, o['hot'])
    d_freq = (d['steps'] / d['hot']) if d['hot'] else float('inf')
    print(f'  HOT DIV freq: 1/{o_freq:.0f} steps -> '
          f'{"1/%.0f steps" % d_freq if d["hot"] else "ZERO (no per-frame DIV)"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
