"""END-TO-END byte-exactness for C4_FLASH_ATTN wired into the real attention forwards.

Compares C4_FLASH_ATTN OFF (masked-full softmax1) vs ON (flash) on a REAL SparseAttn
block on GPU, in BOTH the paths the wiring touches:
  (A) SparseAttn.forward, un-cached FULL (past_kv None, q_positions None) -> SDPA path
  (B) SparseAttn.forward, cached (q_positions given) -> Triton path
  (C) local_attention.windowed_forward GLOBAL heads (window=None) cached -> Triton
  (D) local_attention.windowed_forward LOCAL heads (window=W) -> Triton windowed
The L-inf must be < 1e-4 (fp32 flash noise, below the nibble-decode margin).

Golden 069cc32f unchanged (runtime flag only; touches no stored weight).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_min import sparse_forward as _SF
from c4_min import local_attention as LA
from c4_min.blogspec_model import Attn

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def mk_attn(dim=64, nh=4, seed=0):
    torch.manual_seed(seed)
    a = Attn(dim, nh, max_seq_len=4096, positional="alibi")
    with torch.no_grad():
        for w in (a.W_q, a.W_k, a.W_v, a.W_o):
            w.data.copy_(torch.randn(dim, dim) * 0.1)
    sa = _SF.SparseAttn(a, density_thresh=1.0, min_numel=10 ** 9, log={},
                        compute_mode="dense_kernel")
    return sa.to(DEV), nh, dim


def _flag(on):
    if on:
        os.environ["C4_FLASH_ATTN"] = "1"
    else:
        os.environ.pop("C4_FLASH_ATTN", None)


def run(name, fn):
    _flag(False); off = fn()
    _flag(True); on = fn()
    _flag(False)
    d = (off - on).abs().max().item()
    ok = d < 1e-4
    print(f"  [{name}] L-inf(OFF vs ON)={d:.3e}  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    print(f"device={DEV}", flush=True)
    if DEV == 'cpu':
        print("(cpu — this e2e needs GPU for the flash path)"); return 0
    allok = True

    # (A) un-cached FULL forward.
    sa, nh, dim = mk_attn(seed=0)
    x = (torch.randn(1, 300, dim, device=DEV) * 0.1)
    allok &= run("A: SparseAttn.forward un-cached FULL S=300",
                 lambda: sa.forward(x, use_cache=False))

    # (B) cached forward (q_positions given) — the KV-cache regime.
    sa, nh, dim = mk_attn(seed=1)
    xc = torch.randn(1, 200, dim, device=DEV) * 0.1
    _, kv = LA._global_forward(sa, xc, None, torch.arange(200, device=DEV), True)
    xq = torch.randn(1, 30, dim, device=DEV) * 0.1
    qpos = torch.arange(200, 230, device=DEV)
    allok &= run("B: SparseAttn.forward cached Sq=30 past=200",
                 lambda: sa.forward(xq, past_kv=kv, q_positions=qpos, use_cache=True)[0])

    # (C) windowed_forward, all heads GLOBAL (window=None) cached.
    sa, nh, dim = mk_attn(seed=2)
    _, kv = LA._global_forward(sa, torch.randn(1, 150, dim, device=DEV) * 0.1,
                               None, torch.arange(150, device=DEV), True)
    xq = torch.randn(1, 20, dim, device=DEV) * 0.1
    qpos = torch.arange(150, 170, device=DEV)
    def runC():
        sa._local_window = 8
        sa._global_head_mask = torch.ones(nh, dtype=torch.bool, device=DEV)
        return LA.windowed_forward(sa, xq, kv, qpos, True)[0]
    allok &= run("C: windowed_forward all-GLOBAL cached Sq=20 past=150", runC)

    # (D) windowed_forward, all heads LOCAL (window covers span -> byte-exact).
    sa, nh, dim = mk_attn(seed=3)
    _, kv = LA._global_forward(sa, torch.randn(1, 100, dim, device=DEV) * 0.1,
                               None, torch.arange(100, device=DEV), True)
    xq = torch.randn(1, 10, dim, device=DEV) * 0.1
    qpos = torch.arange(100, 110, device=DEV)
    def runD():
        sa._local_window = 1000
        sa._global_head_mask = torch.zeros(nh, dtype=torch.bool, device=DEV)
        return LA.windowed_forward(sa, xq, kv, qpos, True)[0]
    allok &= run("D: windowed_forward all-LOCAL W=1000 cached Sq=10 past=100", runD)

    # (D2) windowed_forward LOCAL with a SMALL window (genuinely windowed).
    sa, nh, dim = mk_attn(seed=4)
    _, kv = LA._global_forward(sa, torch.randn(1, 100, dim, device=DEV) * 0.1,
                               None, torch.arange(100, device=DEV), True)
    xq = torch.randn(1, 10, dim, device=DEV) * 0.1
    qpos = torch.arange(100, 110, device=DEV)
    def runD2():
        sa._local_window = 16
        sa._global_head_mask = torch.zeros(nh, dtype=torch.bool, device=DEV)
        return LA.windowed_forward(sa, xq, kv, qpos, True)[0]
    allok &= run("D2: windowed_forward all-LOCAL W=16 (genuinely windowed)", runD2)

    print(f"E2E BYTE-EXACT ALL: {'PASS' if allok else 'FAIL'}", flush=True)
    return 0 if allok else 1


if __name__ == '__main__':
    raise SystemExit(main())
