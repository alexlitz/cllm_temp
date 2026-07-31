"""WALL #6 PROBE — analyze the address-CAM recency wall in the doom cfm draft.

Runs ONLY the draft (no model forward), then dissects the store/read logs around
the reported divergence (~step 15,573) to answer:

  * What address does the failing pop query, and how many stores hit it?
  * Does the LATEST-write-wins resolver (resolve_load_rows) give the CORRECT store?
  * Does exact-evict / supersession leave only ONE resident store to that address
    at the query frame (which would make the CAM alias-free)?
  * Does C4_IMM_NIBS=8 move the divergence (IMM-truncation compounding)?

No model, no GPU — pure draft + schedule analysis, seconds.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
# IMM_NIBS is read at import time of nibble_pure_forward_complete — set via env BEFORE import.
IMM_NIBS = os.environ.get('C4_IMM_NIBS', '5')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from c4_min.nibble_evict_schedule import (build_eviction_schedule, resolve_load_rows,
                                          store_row_position, positions_to_drop_through)
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def main():
    max_steps = int(os.environ.get('C4_MAX_STEPS', '16200'))
    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    print(f'[probe] IMM_NIBS={IMM_NIBS} instrs={len(code)} max_steps={max_steps}', flush=True)

    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    print(f'[probe] draft steps={draft.step_count} halted={draft.halted} '
          f'n_stores={len(draft.store_log or {})} n_reads={len(draft.read_log or {})} '
          f'wall={time.time()-t0:.1f}s', flush=True)

    store_log = draft.store_log or {}
    read_log = draft.read_log or {}

    # --- histogram: how many stores per address, and the top reused addresses ---
    from collections import Counter
    addr_store_count = Counter(a for (a, v) in store_log.values())
    top = addr_store_count.most_common(8)
    print(f'[probe] top-reused store addresses (addr: #stores):')
    for a, c in top:
        print(f'         addr={a} ({hex(a)}): {c} stores', flush=True)

    # --- resolve every read latest-write-wins ---
    reads = resolve_load_rows(draft)

    # --- find the FIRST read whose resolved store is NOT the last-committed store to
    #     that address (would only happen if the resolver itself is wrong — sanity) ---
    # and characterise reads to the top-reused address.
    target = top[0][0] if top else None
    print(f'\n[probe] === reads querying the most-reused addr={target} ({hex(target)}) ===', flush=True)
    reads_to_target = []
    for rf in sorted(reads.keys()):
        for r in reads[rf]:
            if r.addr == target:
                reads_to_target.append(r)
    print(f'[probe] #reads to addr {target}: {len(reads_to_target)}', flush=True)
    if reads_to_target:
        # sample a few, showing store_frame recency depth (how many stores newer? = 0 always for LWW)
        for r in reads_to_target[:3] + reads_to_target[-3:]:
            print(f'         read@frame {r.read_frame} head={r.head} -> store_frame={r.store_frame} '
                  f'val={r.value} pos={r.store_position}', flush=True)

    # --- SUPERSESSION / EXACT-EVICT: does the schedule leave only ONE resident store
    #     to the hottest address at any point? ---
    for mode, kwargs in [("supersession_only", dict(supersession_only=True)),
                         ("exact_evict", dict(supersession_only=True, exact_evict=True, pop_free=False))]:
        sched = build_eviction_schedule(draft, slope_min=1.0, **kwargs)
        # At the frame just before the LAST read to target, how many stores to target
        # are still RESIDENT (evict_frame None or > that frame)?
        if reads_to_target:
            probe_frame = reads_to_target[-1].read_frame
            resident = [e for e in sched.stores if e.addr == target
                        and (e.evict_frame is None or e.evict_frame > probe_frame)
                        and e.frame_idx < probe_frame]
            print(f'[probe] mode={mode}: at frame {probe_frame} (last read to addr {target}), '
                  f'resident stores to that addr = {len(resident)} '
                  f'(n_live_total={sched.n_live} n_superseded={sched.n_superseded} '
                  f'n_dead_unread={sched.n_dead_unread} raf={sched.n_read_after_free})', flush=True)
            if len(resident) <= 3:
                for e in resident:
                    print(f'           resident: frame={e.frame_idx} val={e.val} '
                          f'evict_frame={e.evict_frame} reason={e.evict_reason}', flush=True)

    # --- MUL operand check: the wall says AX 69868 != 361758375 at the divergence.
    #     find a step producing ax 361758375 or its wrong value 69868 to localize ---
    print(f'\n[probe] === scanning frames for the reported MUL divergence values ===', flush=True)
    for tgt in (361758375, 69868):
        hits = [i for i, f in enumerate(draft.frames) if (f.get('ax', -1) & 0xFFFFFFFF) == tgt]
        print(f'         ax=={tgt}: frames {hits[:5]}{"..." if len(hits)>5 else ""} ({len(hits)} total)', flush=True)

    # --- IMM -1 check: how does doom's IMM -1 materialize with this IMM_NIBS? ---
    imm_nibs = int(IMM_NIBS)
    imm_mask = (1 << (4 * imm_nibs)) - 1
    print(f'\n[probe] IMM_NIBS={imm_nibs} => IMM -1 materializes as {(-1) & imm_mask} '
          f'({hex((-1) & imm_mask)}); 0xFFFFFFFF would be {0xFFFFFFFF}', flush=True)
    # count IMM ops with negative / >20-bit literals in doom
    neg_imm = [(i, ins.imm) for i, ins in enumerate(code)
               if ins.op == isa.IMM and (ins.imm & 0xFFFFFFFF) > 0xFFFFF]
    print(f'[probe] doom IMM ops with literal > 0xFFFFF (truncated at 5 nibbles): {len(neg_imm)}', flush=True)
    for i, v in neg_imm[:6]:
        print(f'         code[{i}]: IMM {v} (0x{v & 0xFFFFFFFF:08x}) -> 5nib={v & 0xFFFFF} 8nib={v & 0xFFFFFFFF}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
