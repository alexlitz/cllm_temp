#!/usr/bin/env python3
"""CAPSTONE PHASE B — EXECUTE-PIPELINE SMOKE on the 552K Doom snapshot.

Runs a SHORT prefix of the SNAPSHOT bytecode (_doom_bytecode_snapshot.npz, id_port
read-only-at-t0) through the FULL built CFM model and confirms the pipeline
executes self-consistently (fetch->decode->ALU->mem->branch) WITHOUT hitting an
unhandled op / OOM, and that fetch+decode are byte-exact vs the snapshot.

FULL byte-exact-vs-32-bit-VM is Phase C (needs the other agent's 32-bit VM).
Here: (1) the DRAFT executes the snapshot bytecode's transition (the reference
logical VM) for N steps -> proves no unhandled op; (2) verify_blocks confirms the
MODEL forward reproduces the draft register state at every step -> proves the
neural fetch/decode/ALU/mem/branch match the draft (which fetches from the
snapshot), i.e. fetch+decode byte-exact vs snapshot through the run.
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_PF_CFM"] = "1"
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("C4_EXACT_EVICT", "1")
os.environ.setdefault("C4_MEM_EFF", "2000000")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    N_CAP = int(os.environ.get("N_STEPS_CAP", "2000"))
    K = int(os.environ.get("K", "400"))
    window = int(os.environ.get("C4_LOCAL_WINDOW", "96"))

    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program, verify_blocks, _draft_cmp32
    from c4_min.compact_alloc import build_compact_sparse_streaming
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                            data_segment)

    assert _draft_cmp32()
    snap = np.load(os.path.join(_HERE, "_doom_bytecode_snapshot.npz"))
    ops, imms, data = snap["ops"], snap["imms"], snap["data"]
    n_instr = len(ops)
    # rebuild isa code from the snapshot (NOT recompiling id_port).
    code = [isa.Instr(int(ops[i]), int(imms[i])) for i in range(n_instr)]
    code = tag_compiler_syscalls(code, isa)
    data_seg = data_segment([int(b) for b in data])
    print(f"[exec-smoke] snapshot instrs={n_instr} data={len(data)} "
          f"N_CAP={N_CAP} K={K} window={window}", flush=True)

    # DRAFT (free reference execution of the snapshot bytecode).
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=N_CAP, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    print(f"[exec-smoke] DRAFT ran {draft.step_count} steps halted={draft.halted} "
          f"code_off={draft.code_off} wall={time.time()-t0:.1f}s  "
          f"(no unhandled op through {draft.step_count} steps)", flush=True)
    n_drafted = draft.step_count

    # BUILD the full CFM model (code_size covers the whole 552K program).
    cs = max(n_instr + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=cs, recurrent_divmod=True, compute_mode="dense_kernel")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        sparse = sparse.to(dev)
    print(f"[exec-smoke] built dim={sparse.embed.shape[1]} "
          f"blocks={len(sparse.blocks)} build={time.time()-t0:.0f}s dev={dev}",
          flush=True)
    install_local_attention(sparse, window=window, drop_local_kv=True, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    # DIRECT-CAM (C4_DIRECT_CAM_BATCHED): the 552K CODE-frame stream would make the
    # global code CAM score O(552K) per fetch query (the OOM wall).  The direct-CAM
    # gathers the draft-resolved code[pc] O(1), removing the O(S) code score + its
    # score-matrix VRAM.  Byte-identical to the softmax path (see module docstring).
    from c4_min import direct_cam_batched as DCB
    _tbl = None
    if os.environ.get("C4_DIRECT_CAM_BATCHED", "0") not in ("0", "", "false"):
        _tbl = DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=True)

    # VERIFY the MODEL reproduces the draft over the prefix (fetch/decode/ALU/mem/
    # branch self-consistency; a mismatch = model argmax != draft = a real fail).
    stats = {}
    if dev != "cpu":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    try:
        vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev,
                           evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                           oom_backoff=True, min_block_steps=4, prime_chunk=2048)
        oom = False
    except torch.cuda.OutOfMemoryError as e:
        print(f"[exec-smoke] OOM during verify: {str(e)[:160]}", flush=True)
        vr, oom = None, True
    wall = time.time() - t0
    peak = stats.get("peak_vram_gb", 0.0)

    if vr is not None:
        ms = wall / max(vr.total_steps, 1) * 1e3
        print(f"[exec-smoke] verify_blocks matched={vr.all_matched} "
              f"accepted={vr.accepted_steps}/{vr.total_steps} forwards={vr.forwards} "
              f"wall={wall:.1f}s ms/step={ms:.1f} eff_K={stats.get('effective_block_steps')} "
              f"peak_vram={peak:.1f}GB", flush=True)
        if vr.first_mismatch:
            print(f"[exec-smoke] FIRST MISMATCH: {vr.first_mismatch}", flush=True)
        clean = vr.accepted_steps
        result = {
            "n_instr": n_instr, "drafted_steps": n_drafted,
            "model_accepted_steps": vr.accepted_steps,
            "model_total_steps": vr.total_steps,
            "all_matched": bool(vr.all_matched),
            "first_mismatch": vr.first_mismatch,
            "ms_per_step": round(ms, 2),
            "peak_vram_gb": round(peak, 2),
            "eff_K": stats.get("effective_block_steps"),
            "oom": False,
            "unhandled_op": False,
        }
        print("RESULT " + json.dumps(result))
        return 0 if vr.all_matched else 1
    else:
        print("RESULT " + json.dumps({
            "n_instr": n_instr, "drafted_steps": n_drafted,
            "oom": True, "peak_vram_gb": round(peak, 2)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
