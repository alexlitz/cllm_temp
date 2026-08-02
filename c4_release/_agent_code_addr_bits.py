"""CODE-SIZE LIMIT lift: measure the two code-fetch paths past the 4096 cap.

The c4_min transformer fetches the opcode at PC from the code §Memory two ways:
  (1) the SOFTMAX code CAM (``nibble_pure_forward_complete._bake_code_cam_head`` /
      ``qwen_full_vm._bake_code_cam``, address-keyed softmax1 over the code frames),
      capped historically at CODE_ADDR_BITS=12 = 4096 instructions;
  (2) the DIRECT-CAM code fetch (``direct_cam_batched``, C4_DIRECT_CAM_BATCHED):
      the perfect draft resolves the code frame at PC O(1) and the fetch head's
      output is reconstructed from ``code[pc]`` — no softmax over the frames.

This harness (SUBPROCESS-DRIVEN so ``CODE_ADDR_BITS`` — captured at import — can be
set per run) builds LARGE synthetic programs whose control flow lands on PCs far
past 4096 (5000 / 50000 / 99999) with only a HANDFUL of executed steps (chained
JMPs over NOP filler), then runs them through the transformer in both modes and
verifies BYTE-EXACT vs ``isa.interpret``.

Usage (driver):   python _agent_code_addr_bits.py drive
Usage (worker):   C4_CODE_ADDR_BITS=<n> python _agent_code_addr_bits.py \
                      run --mode <softmax|direct> --size <N> --hi <pc0,pc1,...>

Additive; both fetch paths are gated (C4_PF_CFM / C4_DIRECT_CAM_BATCHED), the
CODE_ADDR_BITS widen defaults to 12 -> golden 069cc32f UNCHANGED.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time, json, subprocess

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')


# ---------------------------------------------------------------------------
# LARGE synthetic program: chain JMPs over NOP filler so control lands ON the
# high PCs (5000/50000/99999) but only ~2 STEPS execute per landing.  The last
# landing sets a distinctive < 256 marker into AX and HALTs, so the final AX is
# byte-exact-comparable at mask 0xFF (isa.interpret) AND 0xFFFFFFFF (draft).
# ---------------------------------------------------------------------------
def build_program(size, hi_pcs):
    from c4_min import isa
    landings = sorted(hi_pcs)
    size = max(size, landings[-1] + 2)           # room for the final IMM + HALT
    # program is `size` instructions: all NOP, overwritten by the control chain.
    code = [isa.Instr(isa.NOP, 0) for _ in range(size)]
    # pc 0 : JMP landings[0]
    code[0] = isa.Instr(isa.JMP, landings[0])
    marker = 0
    for k, pc in enumerate(landings):
        marker = (marker + 37 + k) & 0xFF        # distinctive per-landing marker
        code[pc] = isa.Instr(isa.IMM, marker)    # AX = marker (proves fetch@pc)
        if k + 1 < len(landings):
            code[pc + 1] = isa.Instr(isa.JMP, landings[k + 1])
        else:
            code[pc + 1] = isa.Instr(isa.HALT, 0)
    return code, marker


def reference_ax(code):
    from c4_min import isa
    emitted = isa.interpret(code, mem_size=256, max_steps=len(code) + 100)
    return emitted[-1] & 0xFF


def run_worker(mode, size, hi_pcs):
    import torch
    from c4_min import isa
    from c4_min.pf_speculative import draft_pf_program, verify_blocks
    from c4_min.compact_alloc import build_compact_sparse_streaming
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    from c4_min import direct_cam_batched as DCB
    from c4_min.nibble_pure_forward_complete import CODE_ADDR_BITS

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    code, marker = build_program(size, hi_pcs)
    size = len(code)
    ref = reference_ax(code)

    t0 = time.time()
    # code_size must cover the whole program (the cfm layout is code_size-INDEPENDENT
    # for its residual, but the builder still validates code fits).
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=size + 4, recurrent_divmod=True, compute_mode='dense_kernel')
    build_s = time.time() - t0
    if dev != 'cpu':
        sparse = sparse.to(dev)

    draft = draft_pf_program(code, max_steps=size + 100, mask=0xFFFFFFFF)

    install_local_attention(sparse, window=96, drop_local_kv=True, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    tbl = None
    if mode == 'direct':
        os.environ['C4_DIRECT_CAM_BATCHED'] = '1'
        tbl = DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
    else:
        os.environ['C4_DIRECT_CAM_BATCHED'] = '0'

    stats = {}
    t1 = time.time()
    if dev != 'cpu':
        torch.cuda.reset_peak_memory_stats(dev)
    vr = verify_blocks(sparse, L, code, draft, block_steps=64, device=dev,
                       evict=False, mask=0xFFFFFFFF, stats=stats, fast=True)
    run_s = time.time() - t1
    peak_gb = (torch.cuda.max_memory_allocated(dev) / 1e9) if dev != 'cpu' else 0.0

    out = {
        'mode': mode, 'size': size, 'hi_pcs': hi_pcs, 'code_addr_bits': CODE_ADDR_BITS,
        'steps': draft.step_count, 'ref_ax': ref, 'decoded_ax': vr.decoded_final_ax,
        'matched': bool(vr.all_matched), 'accepted': vr.accepted_steps,
        'total': vr.total_steps, 'byte_exact': bool(vr.all_matched
                                                      and vr.decoded_final_ax == ref),
        'first_mismatch': vr.first_mismatch, 'build_s': round(build_s, 1),
        'run_s': round(run_s, 1), 'peak_gb': round(peak_gb, 2),
        'primed_leading_rows': stats.get('primed_leading_rows', 0),
    }
    print('RESULT ' + json.dumps(out), flush=True)
    return 0 if out['byte_exact'] else 1


# ---------------------------------------------------------------------------
# DRIVER: fan out subprocesses (fresh import per CODE_ADDR_BITS / mode).
# ---------------------------------------------------------------------------
def _spawn(bits, mode, size, hi):
    env = dict(os.environ)
    env['C4_CODE_ADDR_BITS'] = str(bits)
    cmd = [sys.executable, os.path.abspath(__file__), 'run',
           '--mode', mode, '--size', str(size), '--hi', ','.join(map(str, hi))]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    res = None
    for ln in p.stdout.splitlines():
        if ln.startswith('RESULT '):
            res = json.loads(ln[len('RESULT '):])
    if res is None:
        print(f'  [bits={bits} {mode} size={size}] NO RESULT\n{p.stdout[-2000:]}\n'
              f'{p.stderr[-2000:]}', flush=True)
    return res


def drive():
    print('== PATH 2: DIRECT-CAM code-size independence (the Doom path) ==', flush=True)
    # bits stays at the default 12; direct-CAM does NOT use the softmax address at all,
    # so it should fetch PC=5000/50000/99999 byte-exact regardless of CODE_ADDR_BITS.
    for size, hi in [(6000, [5000]), (60000, [5000, 50000]),
                     (100000, [5000, 50000, 99999])]:
        r = _spawn(12, 'direct', size, hi)
        if r:
            print(f'  [direct bits=12 size={size:6d} hi={hi}] steps={r["steps"]} '
                  f'ref={r["ref_ax"]} got={r["decoded_ax"]} '
                  f'primed={r["primed_leading_rows"]} peak={r["peak_gb"]}GB '
                  f'run={r["run_s"]}s -> {"BYTE-EXACT" if r["byte_exact"] else "FAIL"}',
                  flush=True)

    print('\n== PATH 1: SOFTMAX code CAM fidelity vs CODE_ADDR_BITS ==', flush=True)
    # A >4096 program (size 6000, land at 5000) run through the SOFTMAX CAM at
    # increasing CODE_ADDR_BITS: 12 (too narrow -> aliases PC 5000&0xFFF=904, MIS-fetch),
    # then 13..20 (wide enough).  This finds the softmax path's usable ceiling.
    for bits in [12, 13, 14, 16, 18, 20]:
        r = _spawn(bits, 'softmax', 6000, [5000])
        if r:
            print(f'  [softmax bits={bits:2d} size=6000 hi=[5000]] '
                  f'ref={r["ref_ax"]} got={r["decoded_ax"]} '
                  f'matched={r["matched"]} run={r["run_s"]}s -> '
                  f'{"BYTE-EXACT" if r["byte_exact"] else "MIS-FETCH"}', flush=True)

    print('\n== PATH 1 deep: softmax CAM at high PC (50000/99999) at bits=17/20 ==',
          flush=True)
    for size, hi, bits in [(60000, [50000], 17), (100000, [99999], 17),
                           (100000, [99999], 20)]:
        r = _spawn(bits, 'softmax', size, hi)
        if r:
            print(f'  [softmax bits={bits:2d} size={size:6d} hi={hi}] '
                  f'ref={r["ref_ax"]} got={r["decoded_ax"]} matched={r["matched"]} '
                  f'peak={r["peak_gb"]}GB run={r["run_s"]}s -> '
                  f'{"BYTE-EXACT" if r["byte_exact"] else "MIS-FETCH"}', flush=True)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'run':
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('run')
        ap.add_argument('--mode', required=True)
        ap.add_argument('--size', type=int, required=True)
        ap.add_argument('--hi', required=True)
        a = ap.parse_args()
        hi = [int(x) for x in a.hi.split(',')]
        return run_worker(a.mode, a.size, hi)
    drive()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
