"""CFM CONTINUOUS DOOM MODEL RUN to the first byte-exact output.

The FULL doom model (3976 instrs) built with CODE-FROM-MEMORY on the pure-forward
path (C4_PF_CFM=1) -> dim is FIXED (small-model dim, NOT ~25,731), so it builds +
fits VRAM.  Runs doom continuously from step 0 through the MODEL forward
(verify_blocks over the draft) to the FIRST printf (~step 29,754) and verifies the
MODEL emits 1b5b324a1b5b48 (== ./c4 first 7 bytes).

Composes: cfm code-CAM + addr32 mem-prep (wall #1) + recurrent divmod (wall #2) +
depth-N lean stack (wall #3) + C4_DRAFT_CMP32 (>255 counters) + big-K verify_blocks
+ local-attn (bounded KV) + dead-block-fusion + compiler-ABI FileRunner I/O.

Env: K (block_steps, default 400), C4_MAX_STEPS (default 30200), C4_LOCAL_WINDOW (96),
DOOM_CBG (content-bound-global 0/1, default 0 -> code frames persist), N_STEPS_CAP
(optional: stop the MODEL after this many steps for a partial-run datapoint),
DOOM_PRIME_CHUNK (leading-context prime chunk, default 2048), DOOM_EVICT_INTERVAL,
DOOM_EVICT (0/1).  Use C4_EXACT_EVICT=1 for the conservative eviction schedule.

RESULT (2026-07-31, C4_PF_CFM):
  * BUILD: doom (3976 instrs) builds at a FIXED dim=1392 (== the 64-instr model),
    peak RSS 1.7 GB, 8 s, model VRAM ~0.01 GB — vs the baked >110 GB (never finished).
    The residual dim is now INDEPENDENT of code_size (the scale wall is broken).
  * CONTINUOUS MODEL RUN: the MODEL forward (verify_blocks over the draft) accepts
    the first 15,574 steps BYTE-EXACT vs the draft (== ./c4 mechanism), ~3.5-7 ms/step,
    then DIVERGES at step 15,574 — an eviction-INDEPENDENT (identical step across the
    cosine / plain-schedule / exact-evict policies, ±content-bound) MEMORY/STACK-CAM
    recency miss: doom writes stack slot addr 200 **1,777 times**, and the pop at step
    15,573 (which needs the store val 1048575 = 0xFFFFF = doom's IMM -1 at 5 nibbles)
    reads a STALE store among the 1,777 -> wrong MUL operand -> AX 69868 != 361758375
    -> SP desync.  This is ORTHOGONAL to code-from-memory: the code CAM fetches PC
    correctly at EVERY step (pc + bp always MATCH at the mismatch; only sp/ax diverge).
    It is the documented ADDRESS-CAM fidelity window at deep address reuse (memory note
    project_self_emulation_cost_reality), a SEPARATE wall from wall #5 (build scale).
  So: wall #5 (build scale) is SOLVED (fixed dim, builds + fits); the continuous MODEL
  reaches 15,574 / 29,754 = 52.3% of the way to the first printf byte-exact before the
  memory-CAM recency wall.  The DRAFT (free) reaches the first printf 7/7 byte-exact."""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ['C4_PF_CFM']='1'
os.environ.setdefault('C4_DRAFT_CMP32','1')
os.environ.setdefault('C4_MEM_ADDR_BITS','18')
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
sys.path.insert(0,'/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks, _draft_cmp32
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM='/home/alexlitz/Documents/misc/c4_doom/doom.c'
REF='/tmp/doom_ref.bin'

def main():
    assert _draft_cmp32()
    max_steps=int(os.environ.get('C4_MAX_STEPS','30200'))
    K=int(os.environ.get('K','400'))
    window=int(os.environ.get('C4_LOCAL_WINDOW','96'))
    cbg=os.environ.get('DOOM_CBG','0')=='1'
    ncap=os.environ.get('N_STEPS_CAP')
    ncap=int(ncap) if ncap else None

    src=Path(DOOM).read_text()
    bc,data=compile_c(src)
    code=tag_compiler_syscalls(bytecode_to_isa(bc),isa)
    data_seg=data_segment(data)
    ref=Path(REF).read_bytes()
    print(f'[doom-cfm] instrs={len(code)} K={K} max_steps={max_steps} window={window} '
          f'cbg={cbg} ncap={ncap} ref7={ref[:7].hex()}', flush=True)

    # DRAFT (free)
    install_compiler_abi_file_dispatcher()
    fio=FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0=time.time()
    draft=draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    draft_out=bytes(fio.runner.stdout)
    print(f'[doom-cfm] draft steps={draft.step_count} halted={draft.halted} '
          f'n_prtf={len(draft.prtf_steps)} first_prtf_step={draft.prtf_steps[:1]} '
          f'code_off={draft.code_off} stdout={draft_out[:16].hex()} wall={time.time()-t0:.2f}s', flush=True)
    if not draft.prtf_steps:
        print('[doom-cfm] DRAFT did not reach first printf'); return 2

    # optionally CAP the model run to the first N steps (partial datapoint)
    fp = draft.prtf_steps[0]
    if ncap is not None:
        # trim the draft frames/win_starts/tokens to ncap steps for a partial model run
        import copy
        n = min(ncap, draft.step_count)
        draft.frames = draft.frames[:n]
        draft.win_starts = draft.win_starts[:n]
        draft.step_count = n
        # keep tokens up to the last needed query row + its frame
        last_pos = draft.win_starts[-1]
        draft.tokens = draft.tokens[:last_pos+1]
        print(f'[doom-cfm] CAPPED model verify to first {n} steps (of {fp} to first printf)', flush=True)

    # BUILD (streaming sparse, cfm) via lib_neural addr32
    from c4_min.lib_neural import build_lib_model_streaming
    cs=max(len(code)+2,64)
    t0=time.time()
    sparse,L,_=build_lib_model_streaming(code_size=cs, recurrent_divmod=True, addr32=True, compute_mode='dense_kernel')
    print(f'[doom-cfm] built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} build={time.time()-t0:.0f}s', flush=True)
    dev='cuda:0' if torch.cuda.is_available() else 'cpu'
    if dev!='cpu': sparse=sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=cbg, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    print(f'[doom-cfm] installed local-attn(window={window},cbg={cbg}) + dead-block-fusion', flush=True)

    # VERIFY (fast big-K) : the MODEL must accept every draft step
    stats={}
    do_evict = os.environ.get('DOOM_EVICT','1')=='1'
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK','2048'))
    ev_int = os.environ.get('DOOM_EVICT_INTERVAL')
    ev_int = int(ev_int) if ev_int else None
    t0=time.time()
    vr=verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=do_evict,
                     mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True, min_block_steps=4,
                     prime_chunk=prime_chunk, evict_interval_steps=ev_int)
    run_wall=time.time()-t0
    ms=run_wall/max(draft.step_count,1)*1e3
    print(f'[doom-cfm] verify_blocks matched={vr.all_matched} forwards={vr.forwards} '
          f'accepted={vr.accepted_steps}/{vr.total_steps} wall={run_wall:.1f}s ms/step={ms:.1f} '
          f'eff_K={stats.get("effective_block_steps")} peak_vram={stats.get("peak_vram_gb",0):.1f}GB', flush=True)
    if vr.first_mismatch:
        print(f'[doom-cfm] FIRST MISMATCH: {vr.first_mismatch}', flush=True)

    # VERDICT
    n=min(len(draft_out),len(ref)); m=0
    for i in range(n):
        if draft_out[i]==ref[i]: m+=1
        else: break
    print(f'[doom-cfm] neural stdout first7={draft_out[:7].hex()} byte-exact_prefix={m} '
          f'({"ESC[2J ESC[H OK" if draft_out[:7]==ref[:7] else "MISMATCH"})', flush=True)
    # the MODEL emits the printf bytes only if it accepted through the printf step
    model_reached_printf = (vr.all_matched and ncap is None) or (ncap is not None and ncap > fp and vr.all_matched)
    ok = (draft_out[:7]==ref[:7]) and vr.all_matched and (ncap is None or ncap>fp)
    print(f'[doom-cfm] MODEL accepted all {vr.accepted_steps} steps: {vr.all_matched}', flush=True)
    print(f'[doom-cfm] CONTINUOUS BYTE-EXACT DOOM MODEL RUN to first output: {ok}', flush=True)
    return 0 if ok else 1

if __name__=='__main__':
    raise SystemExit(main())
