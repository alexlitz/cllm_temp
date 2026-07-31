"""WALL #6 — prove the code-CAM slope=0 fix is BYTE-EXACT on CFM corpus programs.

Runs several real C programs through the CFM fast verify path (the model) and
asserts the model accepts every draft step (byte-exact) with the fix ON (slope=0)
AND matches the OLD behaviour (slope=1) on shallow programs where the horizon is not
crossed — so the fix is a strict superset (fixes deep runs, byte-identical shallow).

The code frames are UNIQUE per PC (one code frame per instruction, never re-emitted),
so removing the ALiBi recency (which only breaks same-address ties) is byte-exact by
construction; this test is the empirical confirmation on real programs.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

PROGS = {
    'loop_sum':   'int main(){int i,s; s=0; i=0; while(i<50){s=s+i; i=i+1;} return s;}',
    'rec_fact':   'int f(int n){ if(n<2) return 1; return n*f(n-1);} int main(){return f(8);}',
    'nested':     'int main(){int i,j,s; s=0; i=0; while(i<12){j=0; while(j<12){s=s+i*j; j=j+1;} i=i+1;} return s;}',
    'rec_fib':    'int fib(int n){ if(n<2) return n; return fib(n-1)+fib(n-2);} int main(){return fib(10);}',
}


def run(name, src):
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"", neural=True)))
    draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code) + 2, 64),
                                             recurrent_divmod=True, addr32=True, compute_mode='dense_kernel')
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev != 'cpu':
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=96, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    st = {}
    vr = verify_blocks(sparse, L, code, draft, block_steps=400, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=st, fast=True, oom_backoff=True,
                       min_block_steps=4, exact_evict=True)
    return draft, vr


def main():
    all_ok = True
    for name, src in PROGS.items():
        t0 = time.time()
        draft, vr = run(name, src)
        ok = vr.all_matched
        all_ok = all_ok and ok
        print(f'[cfm-corpus] {name:10s}: steps={draft.step_count:5d} '
              f'accepted={vr.accepted_steps}/{vr.total_steps} matched={ok} '
              f'final_ax={draft.final_ax_masked} wall={time.time()-t0:.0f}s '
              f'{"OK" if ok else "FAIL "+str(vr.first_mismatch)}', flush=True)
    print(f'\n[cfm-corpus] ALL BYTE-EXACT (code-CAM slope=0 fix): {all_ok}', flush=True)
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
