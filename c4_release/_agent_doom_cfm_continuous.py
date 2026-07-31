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

RESULT (2026-07-31, C4_PF_CFM — WALL #6 SOLVED, run with
        ``C4_EXACT_EVICT=1 C4_MEM_EFF=2000000``):
  * BUILD: doom (3976 instrs) builds at a FIXED dim=1416, ~9 s (scale wall #5 SOLVED).
  * CONTINUOUS MODEL RUN: the MODEL forward (verify_blocks over the draft) now accepts
    ALL 30,200 steps BYTE-EXACT (accepted=30200/30200, matched=True), reaches the first
    printf at step 29,754, and the MODEL EMITS 1b5b324a1b5b48 = 7/7 byte-exact vs
    ``printf 'q' | ./c4 doom.c``.  ~18 ms/step, peak VRAM 23.2 GB (eff_K backed off to
    25 under VRAM pressure).  THE WALL AT 15,574 IS BROKEN — doom runs continuously
    byte-exact through ONE neural model to its first output.

  WALL #6 root cause (three composed fixes, all additive, golden 069cc32f UNCHANGED):
    (A) ``store_row_position`` OMITTED ``code_off`` in CFM mode -> the schedule/exact-evict
        eviction dropped the WRONG cache rows (off by code_off=3976) -> exact-evict
        diverged at step 431.  FIXED (nibble_evict_schedule.py): add code_off; now
        exact-evict is byte-exact + SCALABLE (the O(S^2) cosine cdist OOMs at doom scale).
    (B) The CODE-fetch CAM had ALiBi recency slope=1.0 -> a recall HORIZON of EFF/slope =
        500,000 tokens.  The code frames sit at the START of the stream, so once the query
        row crossed ~500,000 tokens (doom step ~15,574 at token 500,746) the code fetch
        FADED to the softmax1 sink -> the op didn't decode -> IS_POP unset -> the stack-pop
        CAM applied its -PEN_GATE and returned ZFOD -> the SI/pop desynced (the "stale MUL
        operand"/AX-69868 symptom).  The code frames are UNIQUE per PC (no same-address
        ties for recency to break), so slope=0 is byte-exact.  FIXED
        (nibble_pure_forward_complete._bake_code_cam_head): code-CAM slope 1.0 -> 0.0
        (kill-switch C4_CODE_CAM_SLOPE).  This alone advanced doom 15,574 -> 16,224.
    (C) The MEM/STACK CAM's EFF=500,000 gave a 500,000-token recall horizon; doom's
        ~893,000-token stream READS its data-segment map bytes (stored at frame 0) via LC
        ~516,000+ tokens later -> PAST the horizon -> ZFOD 0 (step 16,224).  FIXED
        (blogspec_memory.EFF via C4_MEM_EFF): raise EFF so the horizon covers the FULL
        doom stream (2,000,000 tokens here).  Recency-SAFE: the latest-write-wins margin
        exp(slope·Δ) is INDEPENDENT of EFF; only the horizon scales.  DEFAULT 500000 ->
        byte-identical to golden (corpus deepest gap 250,839 < 500,000).
    NOTES vs the original wall report: IMM_NIBS=8 does NOT move the divergence (rules out
    IMM truncation — confirmed); the wall is eviction-INDEPENDENT (it reproduces with
    eviction OFF); it is NOT a same-address stale-store SELECTION (the exact-address store
    scored -PEN only because IS_POP was unset by the upstream code-fetch fade) — it is the
    documented recall-HORIZON limit, now lifted for the whole doom stream."""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ['C4_PF_CFM']='1'
os.environ.setdefault('C4_DRAFT_CMP32','1')
os.environ.setdefault('C4_MEM_ADDR_BITS','18')
# WALL #6 fixes (composed so this reproduces the full byte-exact run out of the box):
#   * C4_EXACT_EVICT=1  — byte-exact, SCALABLE eviction (the O(S^2) cosine cdist OOMs
#     at doom scale); needs the store_row_position code_off fix (nibble_evict_schedule).
#   * C4_MEM_EFF=2000000 — lift the §Memory recall HORIZON (EFF/slope) past doom's
#     ~893,000-token stream so the data-segment LC reads don't fade to ZFOD (step 16,224).
#   (The code-CAM slope=0 fix is default-ON in _bake_code_cam_head — kill-switch
#    C4_CODE_CAM_SLOPE.)  All three leave golden 069cc32f UNCHANGED (gated / CFM-only).
os.environ.setdefault('C4_EXACT_EVICT','1')
os.environ.setdefault('C4_MEM_EFF','2000000')
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
