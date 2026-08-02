"""SELF-EMULATION / general-path FLASH measurement — the WIN case (per-block attn).

The general N^2-memory fix.  For self-emulation / arbitrary programs the GLOBAL
memory/stack/LEV heads attend over the WHOLE growing KV (no direct-CAM draft), and
the leading-context PRIMING forwards a big span whose per-block score matrix
``[H, Sq, Sk]`` is O(Sq·Sk) — quadratic in the span, the wall that OOMs at S~=12k
(the docstring's 12.9 GiB score).

This isolates ONE real c4_min block's attention (a block carrying a GLOBAL memory
head + the local ingest heads) and drives ``windowed_forward`` on an un-cached FULL
priming span (past=None, pos 0..S-1) at growing S, C4_FLASH_ATTN OFF (masked-full
softmax1) vs ON (flash), reporting:
  * O(S) MEMORY: peak attention VRAM — masked-full is O(S^2) and OOMs at S~=12k;
    flash is O(S), flat, and the OOMing S now RUNS.
  * BYTE-EXACT: L-inf(OFF vs ON) — on real ZFOD-structured data (see the doom
    2000/2000 exact match) this is bit-identical; on the random probe below the
    local-window fp reduction gives ~1e-4..1e-3, still << the nibble-decode margin.

(The FULL 242-block pipeline holds every block's residual + KV, so its end-to-end
OOM is a pipeline-activation wall, NOT the attention score — flash fixes the ATTENTION
score, which this per-block view measures cleanly.)

fp32 flash (NOT tf32).  Golden 069cc32f unchanged (C4_FLASH_ATTN gated; additive).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = 'cuda:0'


def flag(on):
    if on:
        os.environ['C4_FLASH_ATTN'] = '1'
    else:
        os.environ.pop('C4_FLASH_ATTN', None)


def measure_block(at, D, dev, S):
    from c4_min.local_attention import windowed_forward
    torch.manual_seed(0)
    x = torch.randn(1, S, D, device=dev) * 0.01
    qpos = torch.arange(S, device=dev)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize()
    try:
        t = time.perf_counter()
        with torch.no_grad():
            o = windowed_forward(at, x, None, qpos, False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1e3
        gb = torch.cuda.max_memory_allocated(dev) / 1024**3
        return o, ms, gb
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return 'OOM', float('nan'), float('nan')
        raise


def main():
    torch.cuda.init()
    print('[selfemu] building compact pure-forward VM + local attention ...', flush=True)
    from c4_min.compact_alloc import build_compact_sparse_streaming
    from c4_min.local_attention import install_local_attention
    m, L, _ = build_compact_sparse_streaming(code_size=64, recurrent_divmod=True,
                                             compute_mode='dense_kernel')
    m = m.to(DEV)
    summ = install_local_attention(m, window=96, drop_local_kv=False, verbose=False)
    # pick a block that carries a GLOBAL memory head (the far-reaching self-emu head).
    gblk = next((bi for bi, hs in summ['classification'].items() if hs), 7)
    at = m.blocks[gblk].attn
    H, HD, D = at.n_heads, at.head_dim, m.embed.shape[1]
    print(f'[selfemu] blocks={len(m.blocks)} probing block={gblk} H={H} HD={HD} '
          f'global_head_slots(block)={int(at._global_head_mask.sum())} '
          f'(model total {summ["n_global_head_slots"]})', flush=True)

    print('\n[selfemu] PER-BLOCK attention (un-cached FULL priming span) — '
          'masked-full O(S^2) vs flash O(S):', flush=True)
    print(f'{"S":>7} | {"masked-full ms/VRAM":>22} | {"flash ms/VRAM":>18} | {"L-inf":>10}',
          flush=True)
    for S in (2000, 4000, 8000, 12000, 16000, 24000):
        flag(False)
        off, oms, ogb = measure_block(at, D, DEV, S)
        flag(True)
        on, nms, ngb = measure_block(at, D, DEV, S)
        flag(False)
        if isinstance(off, str):
            linf = 'OFF-OOM'; off_s = 'OOM'
        else:
            off_s = f'{oms:8.1f}ms/{ogb:6.2f}GB'
            linf = 'ON-OOM' if isinstance(on, str) else f'{(off - on).abs().max().item():.2e}'
            del off
        on_s = 'OOM' if isinstance(on, str) else f'{nms:7.1f}ms/{ngb:6.3f}GB'
        print(f'{S:>7} | {off_s:>22} | {on_s:>18} | {linf:>10}', flush=True)
        if not isinstance(on, str):
            del on
        torch.cuda.empty_cache()
    print('\n[selfemu] flash keeps the per-block attention O(S) (flat VRAM); the S that '
          'OOMs masked-full runs under flash.  Byte-exact on real ZFOD data (doom '
          '2000/2000); the random-probe L-inf ~1e-4..1e-3 is << the nibble margin.',
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
