"""BYTE-EXACT gate for C4_DIRECT_CAM_VERIFY_ADDR on CORRECT execution.

Confirms the O(1) address check does NOT false-positive on a CORRECT draft: with the
flag ON, the direct-CAM path must still ACCEPT every step and decode the SAME final AX
as the softmax path (L-inf=0), across the DIV-free battery + a deep loop.

Mirrors _agent_directcam_verify.py's battery (add/mul/mem/loop/func/bigimm/jmp/bnz),
plus a 5455-step-scale deep loop.  Div is included (the recurrent-divmod build handles
it) but the point is the address heads (mem/pop/lev/code) all resolve CORRECTLY.

Run:  python -m c4_min._agent_verify_addr_byteexact
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


def _programs():
    P = {}
    P['add'] = ([isa.Instr(isa.IMM, 300), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 44),
                 isa.Instr(isa.ADD), isa.Instr(isa.HALT)], 344)
    P['mul'] = ([isa.Instr(isa.IMM, 12), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 11),
                 isa.Instr(isa.MUL), isa.Instr(isa.HALT)], 132)
    P['div'] = ([isa.Instr(isa.IMM, 100), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 7),
                 isa.Instr(isa.DIV), isa.Instr(isa.HALT)], 14)
    P['mem'] = ([isa.Instr(isa.ENT, 1), isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH),
                 isa.Instr(isa.IMM, 777), isa.Instr(isa.SI),
                 isa.Instr(isa.LEA, 0), isa.Instr(isa.LI), isa.Instr(isa.HALT)], 777)
    func = [isa.Instr(isa.JSR, 2), isa.Instr(isa.HALT), isa.Instr(isa.ENT, 0),
            isa.Instr(isa.IMM, 99), isa.Instr(isa.LEV)]
    P['func'] = (func, 99)
    P['bigimm'] = ([isa.Instr(isa.IMM, 70000), isa.Instr(isa.PSH),
                    isa.Instr(isa.IMM, 5000), isa.Instr(isa.ADD),
                    isa.Instr(isa.HALT)], 75000)
    P['jmp'] = ([isa.Instr(isa.JMP, 3), isa.Instr(isa.IMM, 99), isa.Instr(isa.HALT),
                 isa.Instr(isa.IMM, 7), isa.Instr(isa.HALT)], 7)
    P['bnz'] = ([isa.Instr(isa.IMM, 3), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 1),
                 isa.Instr(isa.SUB), isa.Instr(isa.BNZ, 1), isa.Instr(isa.HALT)], 0)
    # DEEP loop: sum 1..N via a countdown (many pop/mem reads + backward branch).
    # N chosen so step_count is in the ~5455 band the brief asks for.
    N = 300
    loop = [
        isa.Instr(isa.ENT, 2),
        isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH), isa.Instr(isa.IMM, N), isa.Instr(isa.SI),
        isa.Instr(isa.LEA, 1), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 0), isa.Instr(isa.SI),
        isa.Instr(isa.LEA, 0), isa.Instr(isa.LI),          # 9-10 AX=i
        isa.Instr(isa.BZ, 30),                             # 11 if i==0 -> end
        isa.Instr(isa.LEA, 1), isa.Instr(isa.PSH),         # 12-13 &s
        isa.Instr(isa.LEA, 1), isa.Instr(isa.LI), isa.Instr(isa.PSH),  # 14-16 push s
        isa.Instr(isa.LEA, 0), isa.Instr(isa.LI),          # 17-18 AX=i
        isa.Instr(isa.ADD),                                # 19 s+i
        isa.Instr(isa.SI),                                 # 20 s=
        isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH),         # 21-22 &i
        isa.Instr(isa.LEA, 0), isa.Instr(isa.LI), isa.Instr(isa.PSH),  # 23-25 push i
        isa.Instr(isa.IMM, 1), isa.Instr(isa.SUB),         # 26-27 i-1
        isa.Instr(isa.SI),                                 # 28 i=
        isa.Instr(isa.JMP, 9),                             # 29 loop
        isa.Instr(isa.LEA, 1), isa.Instr(isa.LI),          # 30-31 AX=s
        isa.Instr(isa.HALT),                               # 32
    ]
    P['loop'] = (loop, N * (N + 1) // 2)
    return P


def run_one(name, code, expected, sparse, L, dev, K=64):
    draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
    res = {}
    for mode in ('softmax', 'direct+verify'):
        uninstall_dead_block_fusion(sparse); uninstall_local_attention(sparse)
        install_local_attention(sparse, window=96, drop_local_kv=True, verbose=False)
        install_dead_block_fusion(sparse, verbose=False)
        if mode == 'softmax':
            os.environ['C4_DIRECT_CAM_BATCHED'] = '0'
            os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '0'
        else:
            os.environ['C4_DIRECT_CAM_BATCHED'] = '1'
            os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '1'
            DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
        stats = {}
        vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev,
                           evict=False, mask=0xFFFFFFFF, stats=stats, fast=True)
        res[mode] = (vr.all_matched, vr.accepted_steps, vr.total_steps,
                     vr.decoded_final_ax, vr.first_mismatch)
    os.environ['C4_DIRECT_CAM_BATCHED'] = '0'
    os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '0'
    sm, dv = res['softmax'], res['direct+verify']
    ok = (sm[0] and dv[0] and sm[3] == dv[3] and dv[3] == expected and dv[1] == sm[1])
    print(f"[{name:7s}] steps={draft.step_count:5d} exp={expected:6d} | "
          f"softmax acc={sm[1]}/{sm[2]} ax={sm[3]} | "
          f"direct+verify acc={dv[1]}/{dv[2]} ax={dv[3]} | "
          f"L-inf=0 {'OK' if ok else 'FAIL'}", flush=True)
    if not ok and dv[4]:
        print("   direct+verify mismatch:", dv[4], flush=True)
    return ok


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    progs = _programs()
    maxcode = max(len(c) for c, _ in progs.values())
    t0 = time.time()
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=max(maxcode + 2, 64), recurrent_divmod=True, compute_mode='dense_kernel')
    print(f"built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} "
          f"in {time.time()-t0:.1f}s dev={dev}", flush=True)
    if dev != 'cpu':
        sparse = sparse.to(dev)
    all_ok = True
    for name, (code, exp) in progs.items():
        all_ok &= run_one(name, code, exp, sparse, L, dev)
    print("ALL BYTE-EXACT (verify-addr ON)" if all_ok else "SOME FAILED", flush=True)
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
