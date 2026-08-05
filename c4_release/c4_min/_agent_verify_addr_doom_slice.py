"""REAL-DOOM SLICE byte-exact gate for C4_DIRECT_CAM_VERIFY_ADDR.

Drafts a bounded slice of the REAL id-doom program (compiled from the actual
doom.c, the same loader the composed benches use) and runs the direct-CAM
verify_blocks path with the address check ON vs OFF.  Confirms:
  * with the flag ON the slice still decodes byte-exact (acc == softmax acc, same
    final AX) -- the check does NOT false-positive on the real doom memory traffic;
  * the flag is byte-neutral (ON and OFF accept identically).

Memory-safe: lean sparse-resident streaming build (build_lib_model_streaming), a
BOUNDED --steps slice, no dense densify.

Run:  python -m c4_min._agent_verify_addr_doom_slice --steps 4000 --device cuda:0
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time, argparse
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom")
import torch

DOOM = "/home/alexlitz/Documents/misc/c4_doom/doom.c"


def _build(dev, code):
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.local_attention import install_local_attention
    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    sparse = sparse.to(dev)
    install_local_attention(sparse, window=96, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    return sparse, L


def _reset(sparse):
    from c4_min.local_attention import install_local_attention, uninstall_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion, uninstall_dead_block_fusion
    uninstall_dead_block_fusion(sparse); uninstall_local_attention(sparse)
    install_local_attention(sparse, window=96, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    dev = args.device if torch.cuda.is_available() else "cpu"

    from pathlib import Path
    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program, verify_blocks
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from run_c4_min import (tag_compiler_syscalls,
                            install_compiler_abi_file_dispatcher, data_segment)
    from c4_min import direct_cam_batched as DCB

    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=args.steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    ns = draft.step_count
    print(f"[doom-slice] instrs={len(code)} drafted steps={ns} K={args.K} dev={dev}",
          flush=True)

    sparse, L = _build(dev, code)
    res = {}
    for mode in ("softmax", "direct", "direct+verify"):
        _reset(sparse)
        if mode == "softmax":
            os.environ['C4_DIRECT_CAM_BATCHED'] = '0'
            os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '0'
        else:
            os.environ['C4_DIRECT_CAM_BATCHED'] = '1'
            os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '1' if mode == "direct+verify" else '0'
            DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
        t0 = time.perf_counter()
        vr = verify_blocks(sparse, L, code, draft, block_steps=args.K, device=dev,
                           evict=False, mask=0xFFFFFFFF, fast=True)
        dt = time.perf_counter() - t0
        res[mode] = (vr.all_matched, vr.accepted_steps, vr.decoded_final_ax, dt,
                     vr.first_mismatch)
        print(f"  [{mode:14s}] matched={vr.all_matched} acc={vr.accepted_steps}/{ns} "
              f"ax={vr.decoded_final_ax} wall={dt*1e3:.0f}ms", flush=True)
    os.environ['C4_DIRECT_CAM_BATCHED'] = '0'
    os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '0'

    sm, d, dv = res["softmax"], res["direct"], res["direct+verify"]
    byte_exact = (sm[1] == dv[1] and sm[2] == dv[2] and d[1] == dv[1] and d[2] == dv[2])
    print(f"\n  byte-exact (verify ON == softmax == direct): {byte_exact}", flush=True)
    if not byte_exact and dv[4]:
        print("  verify mismatch:", dv[4], flush=True)
    print("  RESULT:", "PASS" if byte_exact else "FAIL", flush=True)
    return 0 if byte_exact else 1


if __name__ == '__main__':
    raise SystemExit(main())
