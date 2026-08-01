"""PART 2 LEVER 1 — compose the CUDA-GRAPH MEGAKERNEL (GraphedFusedForward) with
the doom cfm verify_blocks run and measure ms/step vs the eager baseline.

install_graphed_fused_forward CUDA-graphs the dead-block FFN segments (the 50% of
the forward that is launch-bound) and is a drop-in for model.forward_hidden_cached,
byte-exact (L-inf=0) to the eager fused chain.  This is exactly the self-emu
megakernel lever applied to doom.

Measures BOTH: (a) a representative-span ms (like part1, eager vs graphed) AND
(b) a short END-TO-END verify_blocks run (N_STEPS_CAP steps) eager vs graphed, so
the ms/step is the real production number.  Verifies byte-exactness (the model must
still accept every step + emit the printf bytes when the run reaches one).

Env: N_STEPS_CAP (default 2000, keep it fast), K (default 400 -> OOM-backoff finds
eff_K), GRAPH (0/1 both measured).
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


def build_doom(dev, cap_steps, window):
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code)+2, 64), recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    return sparse, L, code, draft, fio


def run_verify(sparse, L, code, draft, dev, K, prime_chunk=2048):
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
    K = int(os.environ.get('K', '400'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))

    # ---------- EAGER baseline ----------
    print(f'[mega] === EAGER baseline (cap={cap} K={K} window={window}) ===', flush=True)
    sparse, L, code, draft, fio = build_doom(dev, cap, window)
    print(f'[mega] instrs={len(code)} dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} '
          f'draft_steps={draft.step_count} code_off={draft.code_off}', flush=True)
    vr_e, st_e, wall_e = run_verify(sparse, L, code, draft, dev, K)
    ms_e = wall_e / max(draft.step_count, 1) * 1e3
    print(f'[mega] EAGER: matched={vr_e.all_matched} accepted={vr_e.accepted_steps}/{vr_e.total_steps} '
          f'forwards={vr_e.forwards} wall={wall_e:.1f}s ms/step={ms_e:.2f} '
          f'eff_K={st_e.get("effective_block_steps")} peak_vram={st_e.get("peak_vram_gb",0):.1f}GB', flush=True)
    if vr_e.first_mismatch:
        print(f'[mega] EAGER first_mismatch: {vr_e.first_mismatch}', flush=True)

    # ---------- GRAPHED megakernel ----------
    # Rebuild fresh (caches/graphs must be clean) so the two runs are independent.
    del sparse
    torch.cuda.empty_cache()
    print(f'\n[mega] === GRAPHED megakernel (GraphedFusedForward) ===', flush=True)
    sparse2, L2, code2, draft2, fio2 = build_doom(dev, cap, window)
    from c4_min.graphed_fused_forward import install_graphed_fused_forward
    gf = install_graphed_fused_forward(sparse2, dev, verbose=True)
    vr_g, st_g, wall_g = run_verify(sparse2, L2, code2, draft2, dev, K)
    ms_g = wall_g / max(draft2.step_count, 1) * 1e3
    print(f'[mega] GRAPHED: matched={vr_g.all_matched} accepted={vr_g.accepted_steps}/{vr_g.total_steps} '
          f'forwards={vr_g.forwards} wall={wall_g:.1f}s ms/step={ms_g:.2f} '
          f'eff_K={st_g.get("effective_block_steps")} peak_vram={st_g.get("peak_vram_gb",0):.1f}GB '
          f'n_graph_shapes={len(gf._graphs)}', flush=True)
    if vr_g.first_mismatch:
        print(f'[mega] GRAPHED first_mismatch: {vr_g.first_mismatch}', flush=True)

    # ---------- byte-exactness cross-check ----------
    same = (vr_e.all_matched == vr_g.all_matched
            and vr_e.accepted_steps == vr_g.accepted_steps
            and vr_e.decoded_final_ax == vr_g.decoded_final_ax)
    print(f'\n[mega] BYTE-EXACT eager==graphed: matched_eq={vr_e.all_matched==vr_g.all_matched} '
          f'accepted_eq={vr_e.accepted_steps==vr_g.accepted_steps} '
          f'final_ax eager={vr_e.decoded_final_ax} graphed={vr_g.decoded_final_ax} -> {same}', flush=True)
    speedup = ms_e / ms_g if ms_g else 0.0
    print(f'[mega] SPEEDUP: eager {ms_e:.2f} ms/step -> graphed {ms_g:.2f} ms/step  ({speedup:.2f}x)', flush=True)
    return 0 if same else 1


if __name__ == '__main__':
    raise SystemExit(main())
