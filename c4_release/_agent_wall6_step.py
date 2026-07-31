"""WALL #6 — dissect the EXACT divergence step (~15,573) of the doom cfm draft.

The model diverges producing ax=69868 where the draft has 361758375.  This probe
dumps the frames + read/store logs around step 15,573 to localize the failing op
and the exact CAM read (address, resolved store, recency depth).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
IMM_NIBS = os.environ.get('C4_IMM_NIBS', '5')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from c4_min.nibble_evict_schedule import resolve_load_rows, store_row_position
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'
LO, HI = int(os.environ.get('LO', '15560')), int(os.environ.get('HI', '15590'))


def main():
    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=int(os.environ.get('C4_MAX_STEPS', '15700')),
                             mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    print(f'[step] IMM_NIBS={IMM_NIBS} draft steps={draft.step_count}', flush=True)
    reads = resolve_load_rows(draft)
    store_log = draft.store_log or {}
    read_log = draft.read_log or {}

    # frame_idx of frame index s: frames list is 0-based over steps; the draft frame
    # for step s is frames[s], and its frame_idx (for store/read logs) is n_seed + s
    # ... but store_log/read_log key on the internal frame_idx.  We map by scanning.
    print(f'\n[step] frames[{LO}:{HI}] (step: op pc ax sp bp | store | reads):', flush=True)
    for s in range(LO, min(HI, len(draft.frames))):
        f = draft.frames[s]
        # find this frame's store/read via resolve (read_frame is the internal frame_idx)
        # The frames list index != frame_idx (file ops add extra frames). Reconstruct:
        line = (f"  step {s}: op={f['op']:<4} pc={f['pc']} ax={f['ax']&0xFFFFFFFF} "
                f"sp={f['sp']} bp={f['bp']} stk={f.get('stk',0)&0xFFFFFFFF}")
        if f.get('is_store'):
            line += f" | STORE addr={f['s_addr']} val={f['s_val']&0xFFFFFFFF}"
        print(line, flush=True)

    # The read_log is keyed on internal frame_idx; dump reads whose read_frame lands in
    # a window (translate: internal frame_idx ~ n_seed + step for non-file runs, but
    # file ops shift it; use resolve output directly filtering by read value ranges).
    print(f'\n[step] resolved CAM reads with read_frame in a window near divergence:', flush=True)
    # find the internal frame_idx range by matching ax==361758375 producing frames.
    # Just dump all reads and filter to those near the largest read_frame.
    all_rf = sorted(reads.keys())
    max_rf = all_rf[-1] if all_rf else 0
    for rf in all_rf:
        if rf < max_rf - 60:
            continue
        for r in reads[rf]:
            sf = r.store_frame
            sval = store_log.get(sf, (None, None))[1] if sf is not None else None
            # recency depth: #stores to same addr strictly between store_frame and read_frame
            depth = sum(1 for k, (a, v) in store_log.items()
                        if a == r.addr and (sf if sf is not None else -1) < k < rf)
            print(f"  read@fi={rf} head={r.head} addr={r.addr} -> store_fi={sf} "
                  f"val={r.value} depth_between={depth}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
