"""MEASURE fp-acceleration (TF32 tensor cores + selective-fp64) on the DOOM fast stack.

HYPOTHESIS (to test, not assert): doom's fast stack = COO FFN megakernel
(C4_FFN_MEGAKERNEL=triton, gather-scale-scatter, NOT dense matmul) + banded local
attn + direct-CAM.  TF32 accelerates DENSE MATMUL FLOPs on tensor cores; the COO
gather has no matmul, so TF32 should give little.  MEASURE whether that is true.

This composes the FULL doom fast stack (C4_PF_CFM + C4_DIRECT_CAM_BATCHED +
C4_BANDED_LOCAL_ATTN + C4_FFN_MEGAKERNEL=triton + local-attn + dead-block-fusion)
and measures, for each MODE:
  * doom ms/step (full continuous run to first printf, N_STEPS_CAP steps)
  * byte-exact: accepted==total AND stdout first7 == printf 'q' | ./c4 doom.c (7/7)
  * eff_K, peak VRAM

MODES (env MODE=...; comma-separate to run several, each with its OWN fresh build):
  fp32            baseline: TF32 OFF, megakernel ON (the ~2.7-2.9ms/step / 7/7 base)
  tf32            TF32 ON globally, megakernel ON (COO FFN untouched; only the
                  surviving attention matmuls + ingest projections see tensor cores)
  tf32_fp64       TF32 ON + the fp-fragile blocks cast to fp64 (selective-fp64 —
                  fp64 matmuls are UNAFFECTED by the global TF32 flag).  The fragile
                  set is chosen by name (lea-addr-nib + deep div/mod + cmp/loop) via
                  DOOM_FP64_BLOCKS or the name policy.
  dense_fp32      megakernel OFF (dense SwiGLU FFN), TF32 OFF  -> is the dense FFN
  dense_tf32      megakernel OFF (dense SwiGLU FFN), TF32 ON   -> does TF32 make the
                  DENSE FFN competitive with the COO megakernel?  (alternative vs
                  additive to the megakernel)
  bf16            bf16 FFN weights (documented UNUSABLE) -> confirm it breaks 7/7.

Also (MODE contains 'tf32' or PROFILE=1): a MATMUL-SURFACE profile that times the
attention/ingest matmuls TF32-ON vs TF32-OFF over one representative span, so we can
say WHERE TF32 applies and its % of the forward.

Env:
  MODE                comma list from the set above (default 'fp32,tf32')
  N_STEPS_CAP         model-verify step cap (default 30200 = full run to first printf)
  K                   block_steps (default 200)
  C4_LOCAL_WINDOW     local window (default 96)
  DOOM_PRIME_CHUNK    leading-context prime chunk (default 2048)
  DOOM_FP64_BLOCKS    explicit comma block-idx list for tf32_fp64 (overrides policy)
  PROFILE             0/1 run the matmul-surface TF32 on/off profiler (default 1)
  golden 069cc32f UNCHANGED (all gated; additive tool, no build-path edits).
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
os.environ.setdefault('C4_DIRECT_CAM_BATCHED', '1')
os.environ.setdefault('C4_BANDED_LOCAL_ATTN', '1')
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
REF = '/tmp/doom_ref.bin'


def _set_tf32(on: bool):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def build_doom(dev, cap_steps, window):
    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap_steps, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(code_size=max(len(code) + 2, 64), recurrent_divmod=True,
                                             addr32=True, compute_mode='dense_kernel')
    sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    from c4_min import direct_cam_batched as DCB
    DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
    return sparse, L, code, draft, fio, draft.code_off


def install_megakernel(sparse, dev):
    from c4_min.triton_ffn_megaseg import install_triton_ffn_megaseg
    return install_triton_ffn_megaseg(sparse, dev, min_seg_len=16, use_triton=True, verbose=False)


def run_verify(sparse, L, code, draft, dev, K, prime_chunk):
    stats = {}
    torch.cuda.synchronize(); t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=prime_chunk)
    torch.cuda.synchronize(); wall = time.time() - t0
    ms = wall / max(draft.step_count, 1) * 1e3
    return vr, stats, wall, ms


def profile_matmul_surface(sparse, L, code, draft, dev, code_off):
    """Time each block's attention (matmul) portion TF32 vs fp32 + the whole
    forward, over one representative span, to see WHERE TF32 applies + its %."""
    from c4_min.pf_speculative import apply_overlay_window_fast, build_code_vec
    n_steps = draft.step_count
    s0 = min(1500, n_steps - 2)
    span_start = draft.win_starts[s0]
    span_end = draft.win_starts[min(s0 + 199, n_steps - 1)] + 1
    span_toks = draft.tokens[span_start:span_end]; S = len(span_toks)
    win_toks = torch.tensor([span_toks], device=dev)
    q_positions = torch.arange(span_start, span_start + S, device=dev)
    code_vec = build_code_vec(code, L, sparse.embed.shape[1], torch.device(dev),
                              dtype=sparse.embed.dtype)
    with torch.no_grad():
        x0 = sparse.embed[win_toks].clone()
        q_local = [draft.win_starts[s] - span_start for s in range(s0, min(s0 + 200, n_steps))]
        try:
            apply_overlay_window_fast(x0, span_start, L, {}, code_vec,
                                      query_rows=[q for q in q_local if 0 <= q < S],
                                      code=code, code_off=code_off)
        except Exception:
            pass
    n = len(sparse.blocks)

    def time_forward(tf32, iters=30):
        _set_tf32(tf32)
        with torch.no_grad():
            for _ in range(5):
                past = [None] * n
                sparse.forward_hidden_cached(x0, past_key_values=past,
                                             q_positions=q_positions, use_cache=True)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                past = [None] * n
                sparse.forward_hidden_cached(x0, past_key_values=past,
                                             q_positions=q_positions, use_cache=True)
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1e3

    def per_block(tf32):
        _set_tf32(tf32)
        attn_ms = ffn_ms = 0.0
        with torch.no_grad():
            for _ in range(3):
                hh = x0
                for bi in range(n):
                    blk = sparse.blocks[bi]
                    a, kv = blk.attn.forward(hh, past_kv=None, q_positions=q_positions, use_cache=True)
                    hh = blk.ffn.forward(a) if not getattr(blk, '_routed', False) else blk.ffn(a)
            torch.cuda.synchronize()
            h = x0
            for bi in range(n):
                blk = sparse.blocks[bi]
                torch.cuda.synchronize(); t0 = time.time()
                a, kv = blk.attn.forward(h, past_kv=None, q_positions=q_positions, use_cache=True)
                torch.cuda.synchronize(); t1 = time.time()
                out = blk.ffn.forward(a) if not getattr(blk, '_routed', False) else blk.ffn(a)
                torch.cuda.synchronize(); t2 = time.time()
                attn_ms += (t1 - t0) * 1e3
                ffn_ms += (t2 - t1) * 1e3
                h = out
        return attn_ms, ffn_ms

    fwd_off = time_forward(False)
    fwd_on = time_forward(True)
    a_off, f_off = per_block(False)
    a_on, f_on = per_block(True)
    _set_tf32(False)
    return {
        'S': S, 'n_steps': len(q_local),
        'fwd_off_ms': fwd_off, 'fwd_on_ms': fwd_on,
        'attn_off_ms': a_off, 'attn_on_ms': a_on,
        'ffn_off_ms': f_off, 'ffn_on_ms': f_on,
    }


# --- selective-fp64 machinery (proven in _agent_doom_tf32_bisect): make
# SparseWeight.linear dtype-agnostic (auto-promote the weight to x's dtype) so a
# fp64-bridged block runs every matmul in fp64 (TF32-immune) while the fp32 path is
# bit-identical (same tensor).  Then a _Fp64BridgeBlock up-casts the residual to
# fp64 for the fragile blocks and down-casts the output.  On doom the fragile set
# is the WHOLE stack (only all-fp64 is byte-exact under TF32), so DOOM_FP64_BLOCKS
# defaults to 'all'.
from c4_min.sparse_forward import SparseWeight as _SW
_ORIG_LINEAR = _SW.linear
def _linear_dtype_agnostic(self, x):
    import torch.nn.functional as _F
    if not self.is_sparse and self.dense is not None:
        w = self.dense if self.dense.dtype == x.dtype else self.dense.to(x.dtype)
        return _F.linear(x, w)
    if self.dense_resident is not None:
        w = self.dense_resident if self.dense_resident.dtype == x.dtype else self.dense_resident.to(x.dtype)
        return _F.linear(x, w)
    if self.compute_mode == "dense_kernel":
        return _F.linear(x, self.csr.to_dense().to(x.dtype))
    return _ORIG_LINEAR(self, x)
_SW.linear = _linear_dtype_agnostic


class _DtypeBridgeBlock:
    """Run a block's matmuls in ``dt`` (fp64 = TF32-immune selective-fp64; bf16 =
    the documented-unusable low-precision compute) by up-casting the residual to
    ``dt`` and down-casting the block output back to fp32.  The dtype-agnostic
    SparseWeight.linear promotes the weight to ``dt`` on the fly."""
    def __init__(self, blk, dt=torch.float64):
        self._blk = blk
        self._dt = dt
        self._routed = getattr(blk, '_routed', False)
    @property
    def attn(self):
        return self._blk.attn
    @property
    def ffn(self):
        return self._blk.ffn
    def __call__(self, x, past_kv=None, q_positions=None, use_cache=False):
        out = self._blk(x.to(self._dt), past_kv=past_kv,
                        q_positions=q_positions, use_cache=use_cache)
        if isinstance(out, tuple):
            h, kv = out
            return h.to(torch.float32), kv
        return out.to(torch.float32)


def fp64_block_ids(sparse, L):
    """The fp64 block set for selective-fp64.  DOOM_FP64_BLOCKS overrides; default
    'all' (the only byte-exact TF32 config on doom, per the bisect)."""
    n = len(sparse.blocks)
    env = os.environ.get('DOOM_FP64_BLOCKS', 'all')
    if env == 'all':
        return list(range(n))
    return sorted(set(int(x) for x in env.split(',') if x.strip()))


def _wrap_ffn_bf16(blk):
    """Run this block's FFN in bf16 (documented UNUSABLE): up-cast to bf16, F.linear
    bf16 via the dtype-agnostic linear, down-cast to fp32.  Attention unchanged."""
    ffn = getattr(blk, 'ffn', None)
    if ffn is None or getattr(blk, '_routed', False):
        return
    orig = ffn.forward

    def bf16_forward(x, _orig=orig):
        return _orig(x.to(torch.bfloat16)).to(torch.float32)
    ffn.forward = bf16_forward


def cast_blocks_dtype(sparse, block_ids, dt=torch.float64):
    for bi in block_ids:
        if 0 <= bi < len(sparse.blocks):
            sparse.blocks[bi] = _DtypeBridgeBlock(sparse.blocks[bi], dt=dt)
    return list(block_ids)


def run_and_score(mode, sparse, L, code, draft, fio, dev, K, prime_chunk, ref):
    _set_tf32('tf32' in mode)
    vr, stats, wall, ms = run_verify(sparse, L, code, draft, dev, K, prime_chunk)
    _set_tf32(False)
    draft_out = bytes(fio.runner.stdout)
    first7_ok = draft_out[:7] == ref[:7]
    return {
        'mode': mode, 'matched': vr.all_matched,
        'accepted': vr.accepted_steps, 'total': vr.total_steps,
        'wall': wall, 'ms': ms, 'eff_K': stats.get('effective_block_steps'),
        'peak_vram': stats.get('peak_vram_gb', 0.0),
        'first7': draft_out[:7].hex(), 'first7_ok': first7_ok,
        'byte_exact': bool(vr.all_matched and first7_ok),
        'first_mismatch': vr.first_mismatch,
    }


def main():
    dev = 'cuda:0'; torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '30200'))
    K = int(os.environ.get('K', '200'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))
    modes = [m.strip() for m in os.environ.get('MODE', 'fp32,tf32').split(',') if m.strip()]
    do_profile = os.environ.get('PROFILE', '1') == '1'
    ref = Path(REF).read_bytes()

    print(f'[tf32-doom] modes={modes} cap={cap} K={K} window={window} '
          f'prime_chunk={prime_chunk} ref7={ref[:7].hex()}', flush=True)

    results = []
    profiled = False
    for mode in modes:
        dense = mode.startswith('dense')
        use_bf16 = (mode == 'bf16')
        print(f'\n{"="*70}\n[tf32-doom] MODE={mode} (dense_ffn={dense}, bf16={use_bf16})\n{"="*70}', flush=True)
        sparse, L, code, draft, fio, code_off = build_doom(dev, cap, window)

        # megakernel ON for fp32/tf32 (COO FFN); OFF for dense_* (dense SwiGLU) and
        # for tf32_fp64 / bf16 (the dtype bridge conflicts with the fp32 COO graph).
        if not dense and mode not in ('tf32_fp64', 'bf16'):
            install_megakernel(sparse, dev)

        if use_bf16:
            # REAL bf16 FFN COMPUTE: wrap every block's FFN.forward to run in bf16
            # (up-cast residual to bf16, F.linear runs bf16 via the dtype-agnostic
            # linear, down-cast).  Attention stays fp32 (the banded-local einsum has
            # no bf16 kernel) — the bf16 test is the FFN ALU/nibble arithmetic where
            # the 8-bit mantissa is documented to flip decodes.  Megakernel OFF for
            # bf16 (guarded above).
            for blk in sparse.blocks:
                _wrap_ffn_bf16(blk)

        fp64_ids = []
        if mode == 'tf32_fp64':
            # NOTE: megakernel CUDA-graphs the dead FFN segments (fp32); a fp64
            # bridge block inside a graphed segment would break the graph.  So for
            # the selective-fp64 measurement we run WITHOUT the megakernel (pure
            # eager) — see build above where dense=='tf32_fp64' is NOT dense-FFN
            # but we skip the megakernel install for this mode.
            fp64_ids = cast_blocks_dtype(sparse, fp64_block_ids(sparse, L), dt=torch.float64)
            print(f'[tf32-doom] selective-fp64: cast {len(fp64_ids)}/{len(sparse.blocks)} '
                  f'blocks to fp64 (DOOM_FP64_BLOCKS={os.environ.get("DOOM_FP64_BLOCKS","all")})',
                  flush=True)

        # matmul-surface profile once (on the megakernel stack, TF32-relevant modes)
        if do_profile and not profiled and ('tf32' in mode) and not dense:
            try:
                p = profile_matmul_surface(sparse, L, code, draft, dev, code_off)
                fwd_speedup = p['fwd_off_ms'] / max(p['fwd_on_ms'], 1e-9)
                attn_speedup = p['attn_off_ms'] / max(p['attn_on_ms'], 1e-9)
                attn_frac = p['attn_off_ms'] / max(p['attn_off_ms'] + p['ffn_off_ms'], 1e-9)
                print(f'[tf32-doom] MATMUL-SURFACE (span S={p["S"]}, {p["n_steps"]} steps):\n'
                      f'    full forward: TF32-OFF {p["fwd_off_ms"]:.3f}ms  TF32-ON {p["fwd_on_ms"]:.3f}ms '
                      f'-> {fwd_speedup:.3f}x\n'
                      f'    attn-only (matmul surface): OFF {p["attn_off_ms"]:.3f}ms  ON {p["attn_on_ms"]:.3f}ms '
                      f'-> {attn_speedup:.3f}x\n'
                      f'    ffn-only (COO gather): OFF {p["ffn_off_ms"]:.3f}ms  ON {p["ffn_on_ms"]:.3f}ms\n'
                      f'    attn fraction of block-forward: {attn_frac*100:.1f}% '
                      f'(the ONLY surviving dense-matmul surface TF32 can touch)', flush=True)
                profiled = True
            except Exception as e:
                print(f'[tf32-doom] profile err: {e}', flush=True)

        r = run_and_score(mode, sparse, L, code, draft, fio, dev, K, prime_chunk, ref)
        r['fp64_ids'] = fp64_ids
        results.append(r)
        print(f'[tf32-doom] {mode}: matched={r["matched"]} accepted={r["accepted"]}/{r["total"]} '
              f'wall={r["wall"]:.1f}s ms/step={r["ms"]:.3f} eff_K={r["eff_K"]} '
              f'peak_vram={r["peak_vram"]:.1f}GB first7={r["first7"]} '
              f'BYTE-EXACT_7/7={r["byte_exact"]}', flush=True)
        if r['first_mismatch']:
            print(f'[tf32-doom] {mode} FIRST MISMATCH: {r["first_mismatch"]}', flush=True)

        del sparse, L, code, draft, fio
        torch.cuda.empty_cache()

    print(f'\n{"="*94}\n[tf32-doom] SUMMARY (doom continuous run to first printf, RTX A5000, cap={cap})\n{"="*94}')
    base = next((r for r in results if r['mode'] == 'fp32'), None)
    base_ms = base['ms'] if (base and base['matched']) else None
    print(f'  {"mode":14s} {"ms/step":>9} {"vs fp32":>8} {"matched":>8} '
          f'{"accepted":>13} {"eff_K":>6} {"vram":>7} {"7/7":>5}  note')
    for r in results:
        # ms/step is only meaningful when matched (else it bailed at the first
        # mismatch and 'wall' is mostly build/prime — flag it).
        valid = r['matched']
        sp = f'{base_ms/r["ms"]:.3f}x' if (base_ms and valid and r['ms']) else '-'
        note = '' if valid else f'BAILED@step {r["first_mismatch"].get("step") if r["first_mismatch"] else "?"}'
        ms_str = f'{r["ms"]:9.3f}' if valid else f'{"(invalid)":>9}'
        print(f'  {r["mode"]:14s} {ms_str} {sp:>8} {str(r["matched"]):>8} '
              f'{r["accepted"]}/{r["total"]:<7} {str(r["eff_K"]):>6} '
              f'{r["peak_vram"]:6.1f}G {"YES" if r["byte_exact"] else "NO":>5}  {note}')
    print(f'{"="*94}')
    print('  NOTE: 7/7 (stdout first-7 == ./c4) only reachable at full cap '
          '(N_STEPS_CAP>=29755); at a partial cap use matched (accepted==total).', flush=True)
    return 0 if all(r['matched'] for r in results if r['mode'] != 'bf16') else 1


if __name__ == '__main__':
    raise SystemExit(main())
