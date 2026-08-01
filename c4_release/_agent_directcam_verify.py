"""DIRECT-CAM BATCHED byte-exactness on small cfm programs (fast, no doom).

Builds the cfm model once, drafts each small program, and runs verify_blocks TWICE:
  * softmax (C4_DIRECT_CAM_BATCHED unset): the vanilla global-CAM path.
  * direct  (C4_DIRECT_CAM_BATCHED=1):     the direct-index gather path.
Asserts BOTH accept every step AND decode the SAME final AX -- the direct path is
byte-exact to the softmax path.  Exercises loop / mul / div / mem / func (LEV).

Additive; golden 069cc32f unchanged (both paths gated by the env, model weights
identical).  Usage: python _agent_directcam_verify.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
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
    # add: IMM 300; PSH; IMM 44; ADD; HALT  -> 344   (mem pop read)
    P['add'] = ([isa.Instr(isa.IMM, 300), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 44),
                 isa.Instr(isa.ADD), isa.Instr(isa.HALT)], 344)
    # mul: IMM 12; PSH; IMM 11; MUL; HALT -> 132
    P['mul'] = ([isa.Instr(isa.IMM, 12), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 11),
                 isa.Instr(isa.MUL), isa.Instr(isa.HALT)], 132)
    # div: IMM 100; PSH; IMM 7; DIV; HALT -> 14
    P['div'] = ([isa.Instr(isa.IMM, 100), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 7),
                 isa.Instr(isa.DIV), isa.Instr(isa.HALT)], 14)
    # mem: IMM 500; PSH; ... store to a stack slot then LI back.
    #   ENT 1 (frame w/ 1 local); IMM 777; PSH; LEA 0; SI (store 777 @ local0);
    #   LEA 0; LI (load it back); LEV/HALT  -> AX = 777
    P['mem'] = ([isa.Instr(isa.ENT, 1),
                 isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH),   # push &local0
                 isa.Instr(isa.IMM, 777),
                 isa.Instr(isa.SI),                            # *(&local0) = 777
                 isa.Instr(isa.LEA, 0), isa.Instr(isa.LI),     # AX = *(&local0)
                 isa.Instr(isa.HALT)], 777)
    # loop: sum 0..9 via a countdown  (BNZ + repeated pop/add)  -> 45
    #   uses a local counter; exercises many pop reads + a backward branch.
    #   i=10 (local0), s=0 (local1); while(i){ s+=i; i--; }
    loop = [
        isa.Instr(isa.ENT, 2),
        isa.Instr(isa.IMM, 10), isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH),
        isa.Instr(isa.SI),                       # local0 = 10 ; (LEA0;PSH;IMM;SI order below)
    ]
    # rebuild loop cleanly with explicit store idiom: *addr = val is PSH addr; IMM val; SI
    loop = [
        isa.Instr(isa.ENT, 2),                   # 0: frame, 2 locals
        isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 10), isa.Instr(isa.SI),   # 1-4: local0=10
        isa.Instr(isa.LEA, 1), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 0), isa.Instr(isa.SI),    # 5-8: local1=0
        # loop: (pc 9) if local0==0 goto end
        isa.Instr(isa.LEA, 0), isa.Instr(isa.LI),                # 9-10: AX=local0
        isa.Instr(isa.BZ, 24),                                   # 11: if 0 -> end(24)
        # s += i :  local1 = local1 + local0
        isa.Instr(isa.LEA, 1), isa.Instr(isa.PSH),               # 12-13: &local1
        isa.Instr(isa.LEA, 1), isa.Instr(isa.LI), isa.Instr(isa.PSH),  # 14-16: push local1
        isa.Instr(isa.LEA, 0), isa.Instr(isa.LI),                # 17-18: AX=local0
        isa.Instr(isa.ADD),                                      # 19: AX=local1+local0
        isa.Instr(isa.SI),                                       # 20: local1 = AX
        # i -= 1 : local0 = local0 - 1
        isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH),               # 21-22: &local0
        isa.Instr(isa.LEA, 0), isa.Instr(isa.LI), isa.Instr(isa.PSH),  # 23-25: push local0
        isa.Instr(isa.IMM, 1), isa.Instr(isa.SUB),               # 26-27: AX=local0-1
        isa.Instr(isa.SI),                                       # 28: local0=AX
        isa.Instr(isa.JMP, 9),                                   # 29: loop
        # end (pc 30): AX = local1
        isa.Instr(isa.LEA, 1), isa.Instr(isa.LI),                # 30-31: AX=local1
        isa.Instr(isa.HALT),                                     # 32
    ]
    # fix the BZ / JMP targets to the actual pcs above
    # (recompute: end block starts at index 30)
    for k, ins in enumerate(loop):
        if ins.op == isa.BZ:
            loop[k] = isa.Instr(isa.BZ, 30)
        if ins.op == isa.JMP:
            loop[k] = isa.Instr(isa.JMP, 9)
    P['loop'] = (loop, 55)     # sum 10+9+...+1 = 55
    # func: call a function that returns a constant -> exercises JSR + ENT + LEV
    #   main: JSR f; HALT ;  f: ENT 0; IMM 99; LEV
    func = [
        isa.Instr(isa.JSR, 2),      # 0: call f (@pc 2)
        isa.Instr(isa.HALT),        # 1: AX = 99 after return
        isa.Instr(isa.ENT, 0),      # 2: f: frame
        isa.Instr(isa.IMM, 99),     # 3: AX = 99
        isa.Instr(isa.LEV),         # 4: return (PC <- MEM[BP+4], BP <- MEM[BP])
    ]
    P['func'] = (func, 99)
    # bigimm: 70000 + 5000 = 75000  (stresses the 20-bit IMM via the code CAM)
    P['bigimm'] = ([isa.Instr(isa.IMM, 70000), isa.Instr(isa.PSH),
                    isa.Instr(isa.IMM, 5000), isa.Instr(isa.ADD),
                    isa.Instr(isa.HALT)], 75000)
    # jmp: JMP over a dead IMM
    P['jmp'] = ([isa.Instr(isa.JMP, 3), isa.Instr(isa.IMM, 99), isa.Instr(isa.HALT),
                 isa.Instr(isa.IMM, 7), isa.Instr(isa.HALT)], 7)
    # bnz-loop: 3 -1 -1 -1 -> 0  (backward BNZ)
    P['bnz'] = ([isa.Instr(isa.IMM, 3), isa.Instr(isa.PSH), isa.Instr(isa.IMM, 1),
                 isa.Instr(isa.SUB), isa.Instr(isa.BNZ, 1), isa.Instr(isa.HALT)], 0)
    return P


def run_one(name, code, expected, sparse, L, dev, K=64):
    draft = draft_pf_program(code, max_steps=4000, mask=0xFFFFFFFF)
    results = {}
    for mode in ('softmax', 'direct'):
        # reset the block forwards to the clean local-attn + dead-fusion state.
        uninstall_dead_block_fusion(sparse)
        uninstall_local_attention(sparse)
        install_local_attention(sparse, window=96, drop_local_kv=True, verbose=False)
        install_dead_block_fusion(sparse, verbose=False)
        tbl = None
        if mode == 'direct':
            os.environ['C4_DIRECT_CAM_BATCHED'] = '1'
            tbl = DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
        else:
            os.environ['C4_DIRECT_CAM_BATCHED'] = '0'
        stats = {}
        vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev,
                           evict=False, mask=0xFFFFFFFF, stats=stats, fast=True)
        results[mode] = (vr.all_matched, vr.accepted_steps, vr.total_steps,
                         vr.decoded_final_ax, vr.first_mismatch)
    os.environ['C4_DIRECT_CAM_BATCHED'] = '0'
    sm, dm = results['softmax'], results['direct']
    ok = (sm[0] and dm[0] and sm[3] == dm[3] and dm[3] == expected
          and dm[1] == sm[1])
    print(f'[{name:5s}] steps={draft.step_count:5d} exp={expected:6d} | '
          f'softmax matched={sm[0]} acc={sm[1]}/{sm[2]} ax={sm[3]} | '
          f'direct matched={dm[0]} acc={dm[1]}/{dm[2]} ax={dm[3]} | '
          f'{"OK" if ok else "FAIL"}', flush=True)
    if not ok:
        if sm[4]: print('   softmax mismatch:', sm[4], flush=True)
        if dm[4]: print('   direct mismatch:', dm[4], flush=True)
    return ok


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    progs = _programs()
    maxcode = max(len(c) for c, _ in progs.values())
    t0 = time.time()
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=max(maxcode + 2, 64), recurrent_divmod=True, compute_mode='dense_kernel')
    print(f'built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} '
          f'names?{hasattr(L, "_block_names")} in {time.time()-t0:.1f}s', flush=True)
    if dev != 'cpu':
        sparse = sparse.to(dev)
    all_ok = True
    for name, (code, exp) in progs.items():
        all_ok &= run_one(name, code, exp, sparse, L, dev)
    print('ALL BYTE-EXACT' if all_ok else 'SOME FAILED', flush=True)
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
