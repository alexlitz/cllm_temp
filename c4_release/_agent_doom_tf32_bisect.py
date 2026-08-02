"""Bisect WHICH doom block(s) TF32 must keep in fp64 to stay byte-exact.

Builds the doom fast stack ONCE (no megakernel — pure eager so ANY block can be
fp64), sets global TF32 ON, then for a candidate fp64 block set casts those
blocks' SparseWeight tensors to fp64 (dtype-bridged forward) and runs the verify
at a small cap.  Reports the first-mismatch step per candidate set so we can find
the minimal fragile set.  Selective-fp64 fp64 matmuls are UNAFFECTED by TF32.

Env: CAND (comma fp64 block idx list, or 'none'/'all'/'policy'), N_STEPS_CAP (600),
K (200).  golden 069cc32f UNCHANGED (additive tool)."""
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

# --- make SparseWeight.linear dtype-agnostic: cast the weight to x's dtype on the
# fly when they differ, so a fp64 (bridged, TF32-immune) input is matmul'd in fp64
# and a fp32 input in fp32 (TF32-affected).  Bit-identical for the fp32 path (same
# tensor); the fp64 path is the selective-fp64 lever.  Tool-only monkeypatch.
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
        w = self.csr.to_dense().to(x.dtype)
        return _F.linear(x, w)
    return _ORIG_LINEAR(self, x)
_SW.linear = _linear_dtype_agnostic


def _set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def _cast_weight_dtype(sw, dt):
    for nm in ('dense', 'dense_resident'):
        t = getattr(sw, nm, None)
        if t is not None:
            setattr(sw, nm, t.to(dt))
    # materialize csr->dense_resident in dt so linear uses it (avoids fp32 csr path)
    if getattr(sw, 'is_sparse', False) and getattr(sw, 'dense_resident', None) is None:
        try:
            sw.dense_resident = sw.csr.to_dense().to(dt)
        except Exception:
            pass


def _cast_all_float(obj, dt):
    """Cast every float32 tensor attribute of obj (SparseWeights + bias tensors)
    to dtype dt, in place."""
    for nm in list(vars(obj).keys()) if hasattr(obj, '__dict__') else []:
        v = getattr(obj, nm, None)
        if hasattr(v, 'linear'):                     # a SparseWeight
            _cast_weight_dtype(v, dt)
        elif torch.is_tensor(v) and v.dtype == torch.float32:
            setattr(obj, nm, v.to(dt))


def _block_weights(blk):
    ws = []
    for sub in ('attn', 'ffn'):
        o = getattr(blk, sub, None)
        if o is None:
            continue
        for nm in dir(o):
            if nm.startswith('W_') or nm in ('W_up', 'W_gate', 'W_down'):
                w = getattr(o, nm, None)
                if hasattr(w, 'linear'):
                    ws.append(w)
    return ws


class _Fp64BridgeBlock:
    """Proxy around a SparseBlock whose weights are fp64: up-casts the incoming
    fp32 residual to fp64, runs the (fp64, TF32-immune) block, down-casts the
    output to fp32.  Defined as a class so ``blk(...)`` dispatches to __call__
    (an instance __call__ assignment would NOT be picked up by call syntax)."""
    def __init__(self, blk):
        self._blk = blk
        self._routed = getattr(blk, '_routed', False)
    @property
    def attn(self):
        return self._blk.attn
    @property
    def ffn(self):
        return self._blk.ffn
    def __call__(self, x, past_kv=None, q_positions=None, use_cache=False):
        x64 = x.to(torch.float64)
        out = self._blk(x64, past_kv=past_kv, q_positions=q_positions, use_cache=use_cache)
        if isinstance(out, tuple):
            h, kv = out
            return h.to(torch.float32), kv
        return out.to(torch.float32)


def cast_block_fp64(sparse, bi):
    """Replace block ``bi`` with a fp64 bridge proxy.  With the dtype-agnostic
    SparseWeight.linear monkeypatch, the bridge up-casting x to fp64 is enough:
    every matmul in the block runs fp64 (TF32-immune); biases auto-promote."""
    sparse.blocks[bi] = _Fp64BridgeBlock(sparse.blocks[bi])


def main():
    dev = 'cuda:0'; torch.cuda.init()
    cap = int(os.environ.get('N_STEPS_CAP', '600'))
    K = int(os.environ.get('K', '200'))
    window = int(os.environ.get('C4_LOCAL_WINDOW', '96'))
    prime_chunk = int(os.environ.get('DOOM_PRIME_CHUNK', '2048'))
    cand = os.environ.get('CAND', 'none')

    src = Path(DOOM).read_text(); bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa); data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
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
    names = list(getattr(L, '_block_names', []) or [])
    n = len(sparse.blocks)

    def is_fragile(nm):
        nml = str(nm).lower()
        return ('lea-addr-nib' in nml or 'alu-div' in nml or 'alu-mod' in nml
                or nml.startswith('div') or nml.startswith('mod')
                or 'cmp' in nml or 'loop' in nml)

    if cand == 'none':
        ids = []
    elif cand == 'all':
        ids = list(range(n))
    elif cand == 'policy':
        ids = [i for i in range(n) if i < len(names) and is_fragile(names[i])]
    elif cand == 'code':
        ids = [i for i in range(min(12, n))]  # code/mem/stack path
    elif cand == 'code_cmp_lea':
        ids = ([i for i in range(min(16, n))]
               + [i for i in range(n) if i < len(names)
                  and ('lea' in str(names[i]).lower() or 'cmp' in str(names[i]).lower())])
    else:
        ids = [int(x) for x in cand.split(',') if x.strip()]
    ids = sorted(set(ids))

    _set_tf32(True)
    for i in ids:
        cast_block_fp64(sparse, i)
    shown = [names[i] if i < len(names) else i for i in ids]
    print(f'[bisect] TF32 ON, fp64={len(ids)} blocks {shown[:20]}{"..." if len(shown)>20 else ""}', flush=True)

    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=True,
                       mask=0xFFFFFFFF, stats=stats, fast=True, oom_backoff=True,
                       min_block_steps=4, prime_chunk=prime_chunk)
    _set_tf32(False)
    ms = (time.time() - t0) / max(draft.step_count, 1) * 1e3
    print(f'[bisect] matched={vr.all_matched} accepted={vr.accepted_steps}/{vr.total_steps} '
          f'ms/step={ms:.2f}', flush=True)
    if vr.first_mismatch:
        fm = vr.first_mismatch
        print(f'[bisect] FIRST MISMATCH step={fm.get("step")} pos={fm.get("query_pos")} '
              f'got={fm.get("got")} want={fm.get("want")}', flush=True)
    return 0 if vr.all_matched else 1


if __name__ == '__main__':
    raise SystemExit(main())
