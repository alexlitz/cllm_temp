"""FPS / throughput impact of C4_DIRECT_CAM_VERIFY_ADDR.

The address check lives in the direct-CAM ``verify_blocks`` path (``direct_forward`` +
the per-span sink poll) -- the actual autoregressive speculative VERIFIER.  It is O(1)
per read (a per-bit sign decode of the model's OWN computed query, no O(n_store)
softmax).  This measures the verify_blocks steps/sec on the direct-CAM composed path
with the flag ON vs OFF, on a deep loop with heavy memory traffic (the workload the
address check exercises), so the delta is the check's real cost.

Run:  python -m c4_min._agent_verify_addr_fps
"""
import warnings; warnings.filterwarnings('ignore')
import os, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
import torch
from c4_min import isa
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.local_attention import install_local_attention, uninstall_local_attention
from c4_min.live_head_attention import install_dead_block_fusion, uninstall_dead_block_fusion
from c4_min import direct_cam_batched as DCB


def _deep_loop(N):
    return [
        isa.Instr(isa.ENT, 2),
        isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH), isa.Instr(isa.IMM, N), isa.Instr(isa.SI),
        isa.Instr(isa.LEA, 1), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 0), isa.Instr(isa.SI),
        isa.Instr(isa.LEA, 0), isa.Instr(isa.LI), isa.Instr(isa.BZ, 30),
        isa.Instr(isa.LEA, 1), isa.Instr(isa.PSH), isa.Instr(isa.LEA, 1), isa.Instr(isa.LI), isa.Instr(isa.PSH),
        isa.Instr(isa.LEA, 0), isa.Instr(isa.LI), isa.Instr(isa.ADD), isa.Instr(isa.SI),
        isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH), isa.Instr(isa.LEA, 0), isa.Instr(isa.LI), isa.Instr(isa.PSH),
        isa.Instr(isa.IMM, 1), isa.Instr(isa.SUB), isa.Instr(isa.SI), isa.Instr(isa.JMP, 9),
        isa.Instr(isa.LEA, 1), isa.Instr(isa.LI), isa.Instr(isa.HALT),
    ]


def _timeit(sparse, L, code, draft, dev, verify_addr, K=256, reps=3):
    times = []
    acc = None
    for _ in range(reps):
        uninstall_dead_block_fusion(sparse); uninstall_local_attention(sparse)
        install_local_attention(sparse, window=96, drop_local_kv=True, verbose=False)
        install_dead_block_fusion(sparse, verbose=False)
        os.environ['C4_DIRECT_CAM_BATCHED'] = '1'
        os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '1' if verify_addr else '0'
        DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
        if dev != 'cpu':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev,
                           evict=False, mask=0xFFFFFFFF, fast=True)
        if dev != 'cpu':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        acc = vr.accepted_steps
    os.environ['C4_DIRECT_CAM_BATCHED'] = '0'
    os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '0'
    best = min(times)
    return best, acc, draft.step_count


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    N = 300
    code = _deep_loop(N)
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=64, recurrent_divmod=True, compute_mode='dense_kernel')
    if dev != 'cpu':
        sparse = sparse.to(dev)
    draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
    print(f"deep loop steps={draft.step_count} dev={dev}", flush=True)

    off_t, off_acc, n = _timeit(sparse, L, code, draft, dev, verify_addr=False)
    on_t, on_acc, _ = _timeit(sparse, L, code, draft, dev, verify_addr=True)

    off_sps = n / off_t
    on_sps = n / on_t
    print(f"\n  verify-addr OFF: {off_t*1e3:8.1f} ms total  "
          f"{off_t/n*1e3:7.4f} ms/step  {off_sps:9.0f} steps/s  acc={off_acc}/{n}", flush=True)
    print(f"  verify-addr ON : {on_t*1e3:8.1f} ms total  "
          f"{on_t/n*1e3:7.4f} ms/step  {on_sps:9.0f} steps/s  acc={on_acc}/{n}", flush=True)
    delta = (on_t - off_t) / off_t * 100.0
    print(f"  => overhead: {delta:+.2f}%  (O(1)/read, expected within noise)", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
