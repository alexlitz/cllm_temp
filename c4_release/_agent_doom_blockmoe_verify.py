"""PART 2 LEVER (block-MoE, the biggest FFN lever) — a byte-exact composed verify:
run ONLY the span's live-union blocks (like self-emu) PLUS always the KV-carrying
live-attention blocks (so their persistent cache stays complete), skipping the
FFN-only dead-attention blocks not needed by any step in the span.

This is self-emu's live-union block-MoE adapted to doom's persistent-KV verify path.
Measured against the EAGER full-242-block verify_blocks for byte-exactness (accepted
steps + decoded final AX must match) and ms/step.

Byte-safety: the ~238 dead-attention blocks have NO KV (dead_block_forward returns
x, no cache), so skipping their FFN only drops their residual contribution — valid
iff the live-index is a decode-superset (the corpus gate's contract).  The 4
KV-carrying live-attention blocks (0,2,7,11) are ALWAYS run so no cache row is
missed (skipping them WOULD desync a later span's global CAM).

Env: N_STEPS_CAP (default 2000), K (default 25), GRAPH_MOE (0/1) also graph the union.
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
os.environ.setdefault('C4_KV_STACK', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')
import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment

DOOM = '/home/alexlitz/Documents/misc/c4_doom/doom.c'


def build_doom(dev, cap, window):
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code)+2, 64), recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    return sparse, L, code, draft


def install_blockmoe(sparse, L, draft):
    """Install a forward_hidden_cached that skips blocks not in draft's per-span
    live-union.  The caller sets sparse._moe_live (a set of block idxs to RUN) before
    each forward; blocks not in it AND not KV-carrying are passed through (identity)."""
    from c4_min.step_block_skip import build_live_index
    live_index = build_live_index(sparse, L)
    kv_blocks = set(b for b in range(len(sparse.blocks))
                    if not getattr(sparse.blocks[b].attn, '_dead_block_fused', False))
    orig_fhc = sparse.forward_hidden_cached
    name_to_op = {v: k for k, v in isa.NAMES.items()}

    def fhc_moe(x, past_key_values=None, q_positions=None, use_cache=False):
        run_set = getattr(sparse, '_moe_live', None)
        if run_set is None:
            return orig_fhc(x, past_key_values=past_key_values,
                            q_positions=q_positions, use_cache=use_cache)
        n = len(sparse.blocks)
        if past_key_values is None:
            past_key_values = [None] * n
        new_caches = [None] * n
        h = x
        for b in range(n):
            if b in run_set or b in kv_blocks:
                h, kv = sparse.blocks[b](h, past_kv=past_key_values[b],
                                         q_positions=q_positions, use_cache=True)
                new_caches[b] = kv
            # else: identity passthrough (FFN-only dead block, no KV) -> h unchanged,
            #       cache stays None (dead blocks never write KV anyway).
        return h, new_caches

    sparse.forward_hidden_cached = fhc_moe
    return live_index, name_to_op, kv_blocks


def compute_span_live(draft, live_index, name_to_op, nblk, step, end):
    full = set(range(nblk))
    u = set()
    for s in range(step, end):
        opcode = name_to_op.get(draft.frames[s]['op'])
        live = live_index.get(opcode)
        if live is None:
            return full
        u |= set(live)
    return u


def run_verify_moe(sparse, L, code, draft, dev, K, live_index, name_to_op, kv_blocks, prime_chunk=2048):
    """Reproduce verify_blocks but set sparse._moe_live per span from the draft ops.
    We wrap verify_blocks by pre-setting a callback: verify_blocks builds spans of K
    consecutive steps [step, step+K); we replicate that stepping to set _moe_live.

    Simplest robust approach: monkeypatch draft.win_starts consumption is internal, so
    instead we set a per-forward hook: intercept sparse.forward_hidden_cached to derive
    the live-union from q_positions -> which steps are in the span.  q_positions maps to
    absolute token positions; the query rows are draft.win_starts[step..end].  We map
    the span's q_positions range back to steps via win_starts."""
    nblk = len(sparse.blocks)
    import bisect
    ws = draft.win_starts
    n_steps = draft.step_count
    orig_fhc = sparse.forward_hidden_cached  # the moe fhc

    def fhc_auto(x, past_key_values=None, q_positions=None, use_cache=False):
        # derive the step range covered by this forward from q_positions.
        if q_positions is not None and q_positions.numel() > 0:
            p0 = int(q_positions[0].item()); p1 = int(q_positions[-1].item())
            # steps whose query row (win_starts[s]) falls in [p0, p1]
            s_lo = bisect.bisect_left(ws, p0)
            s_hi = bisect.bisect_right(ws, p1)
            if s_hi > s_lo:
                sparse._moe_live = compute_span_live(draft, live_index, name_to_op, nblk, s_lo, min(s_hi, n_steps))
            else:
                sparse._moe_live = None   # priming chunk (no query rows) -> run all
        else:
            sparse._moe_live = None
        return orig_fhc(x, past_key_values=past_key_values, q_positions=q_positions, use_cache=use_cache)

    sparse.forward_hidden_cached = fhc_auto
    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=prime_chunk)
    wall = time.time() - t0
    return vr, stats, wall


def run_verify_plain(sparse, L, code, draft, dev, K, prime_chunk=2048):
    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=prime_chunk)
    wall = time.time() - t0
    return vr, stats, wall


def main():
    dev = 'cuda:0'; torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '2000'))
    K = int(os.environ.get('K', '25'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))

    # EAGER full-242 baseline
    print(f'[bmoe] === EAGER full-242 baseline (cap={cap} K={K}) ===', flush=True)
    s1, L1, code1, draft1 = build_doom(dev, cap, window)
    print(f'[bmoe] dim={s1.embed.shape[1]} blocks={len(s1.blocks)} steps={draft1.step_count}', flush=True)
    vr_b, st_b, wall_b = run_verify_plain(s1, L1, code1, draft1, dev, K)
    ms_b = wall_b / max(draft1.step_count, 1) * 1e3
    print(f'[bmoe] BASE: matched={vr_b.all_matched} accepted={vr_b.accepted_steps}/{vr_b.total_steps} '
          f'forwards={vr_b.forwards} ms/step={ms_b:.2f} eff_K={st_b.get("effective_block_steps")} '
          f'final_ax={vr_b.decoded_final_ax}', flush=True)
    del s1; torch.cuda.empty_cache()

    # BLOCK-MoE
    print(f'\n[bmoe] === BLOCK-MoE (live-union + KV-block always-on) ===', flush=True)
    s2, L2, code2, draft2 = build_doom(dev, cap, window)
    li, n2o, kvb = install_blockmoe(s2, L2, draft2)
    print(f'[bmoe] KV-carrying blocks (always run): {sorted(kvb)}', flush=True)
    vr_m, st_m, wall_m = run_verify_moe(s2, L2, code2, draft2, dev, K, li, n2o, kvb)
    ms_m = wall_m / max(draft2.step_count, 1) * 1e3
    print(f'[bmoe] MoE : matched={vr_m.all_matched} accepted={vr_m.accepted_steps}/{vr_m.total_steps} '
          f'forwards={vr_m.forwards} ms/step={ms_m:.2f} eff_K={st_m.get("effective_block_steps")} '
          f'final_ax={vr_m.decoded_final_ax}', flush=True)
    if vr_m.first_mismatch:
        print(f'[bmoe] MoE first_mismatch: {vr_m.first_mismatch}', flush=True)

    same = (vr_b.all_matched == vr_m.all_matched and vr_b.accepted_steps == vr_m.accepted_steps
            and vr_b.decoded_final_ax == vr_m.decoded_final_ax)
    print(f'\n[bmoe] BYTE-EXACT base==MoE: matched_eq={vr_b.all_matched==vr_m.all_matched} '
          f'accepted_eq={vr_b.accepted_steps==vr_m.accepted_steps} '
          f'final_ax base={vr_b.decoded_final_ax} moe={vr_m.decoded_final_ax} -> {same}', flush=True)
    print(f'[bmoe] SPEEDUP: base {ms_b:.2f} ms/step -> MoE {ms_m:.2f} ms/step  ({ms_b/max(ms_m,1e-9):.2f}x)', flush=True)
    return 0 if same else 1


if __name__ == '__main__':
    raise SystemExit(main())
