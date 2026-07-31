"""Verify store_row_position(fi) vs the ACTUAL store-row token position in CFM mode.

If the schedule's store_row_position omits code_off, the schedule drops the WRONG
cache rows in CFM mode -> the schedule-eviction divergence.  This confirms the bug
by comparing store_row_position(fi) to the true position derived from the draft's
token layout ([BOS] + code_off code tokens + n_seed*FRAME_LEN + frames).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import c4_min.blogspec_vocab as V
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program
from c4_min.nibble_evict_schedule import store_row_position
from c4_min.nibble_pure_forward import _MEM_MARKER_LOCAL
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

SRC = 'int main(){int i,a,s; s=0; i=0; while(i<20){a=i*3; s=s+a; a=0; s=s+a; i=i+1;} return s;}'


def main():
    bc, data = compile_c(SRC)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"", neural=True)))
    draft = draft_pf_program(code, max_steps=400, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    code_off = draft.code_off
    print(f'code_off={code_off} n_stores={len(draft.store_log or {})} FRAME_LEN={V.FRAME_LEN} '
          f'MEM_MARKER_LOCAL={_MEM_MARKER_LOCAL} n_tokens={len(draft.tokens)}', flush=True)

    # ACTUAL position of a store row: the token stream is
    #   [BOS] + code_off code-tokens + seed_frames + init_frame + per-step frames.
    # A frame at frame_idx fi occupies [1 + code_off + fi*FRAME_LEN, ...) and its MEM
    # marker sits at + _MEM_MARKER_LOCAL.  Compare that to store_row_position(fi).
    store_log = draft.store_log or {}
    mism = 0
    for fi in sorted(store_log.keys())[:8]:
        sched_pos = store_row_position(fi, code_off)   # FIXED: pass code_off
        true_pos = 1 + code_off + V.FRAME_LEN * fi + _MEM_MARKER_LOCAL
        tok = draft.tokens[true_pos] if true_pos < len(draft.tokens) else None
        tok_sched = draft.tokens[sched_pos] if sched_pos < len(draft.tokens) else None
        ok = (sched_pos == true_pos)
        mism += (0 if ok else 1)
        print(f'  fi={fi}: store_row_position={sched_pos} true_pos(+code_off)={true_pos} '
              f'{"MATCH" if ok else "MISMATCH (off by %d)" % (true_pos - sched_pos)} '
              f'tok@sched={tok_sched} tok@true={tok}', flush=True)
    print(f'\nRESULT: {"BUG CONFIRMED — store_row_position omits code_off" if mism else "positions match"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
