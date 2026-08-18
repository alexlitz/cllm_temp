#!/usr/bin/env python3
"""_agent_doom_wide_verifyblocks.py — the AIRTIGHT byte-exact arm of the #854 wide
default-ON doom-render gate: run REAL doom through ``verify_blocks`` (fast=True) — the
SAME per-block argmax path the 30K/30K capstone proof used — under the BARE-ENV wide
default (this branch), and report matched / first_mismatch.

This is the branch-local twin of ``task853_compiler_cfm/doom_wide_noregress.py`` (which
hardcodes a foreign worktree on sys.path); this one uses THIS worktree so it verifies
the ACTUAL default-ON build (golden eda39045).  It leaves C4_GLOBAL_ADDR32 / C4_SP_WIDE
to the branch default (bare env => 1), only setting the doom-runner config the doom
runners already setdefault (CMP32/SHIFT32/PC_WIDE/...).  LEAN, ONE process, stops <25GB.
"""
from __future__ import annotations
import os, sys, time, resource

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_PF_CFM"] = "1"
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_CMP32", "1")
os.environ.setdefault("C4_CMP32_ORDER", "1")
os.environ.setdefault("C4_SHIFT32", "1")
os.environ.setdefault("C4_PC_WIDE", "1")
os.environ.setdefault("C4_CODE_ADDR_BITS", "20")
os.environ.setdefault("C4_IMM_NIBS", "6")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("C4_EXACT_EVICT", "1")
os.environ.setdefault("C4_MEM_EFF", "2000000")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOOM_SIB = "/home/alexlitz/Documents/misc/c4_doom"
if os.path.isdir(_DOOM_SIB) and _DOOM_SIB not in sys.path:
    sys.path.insert(0, _DOOM_SIB)

import numpy as np
import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = PFC.SP_INIT = _PFCa.SP_INIT = 0x10000
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from run_c4_min import (data_segment, tag_compiler_syscalls,
                        install_compiler_abi_file_dispatcher)

IMM_OP, DIV_OP, MOD_OP, SHR_OP, AND_OP = 1, 28, 29, 24, 16


def _apply_pow2(ops, imms):
    n = len(ops); npow = 0
    for i in range(n - 1):
        o0, m0, o1 = int(ops[i]), int(imms[i]), int(ops[i + 1])
        if o0 == IMM_OP and m0 > 0 and (m0 & (m0 - 1)) == 0:
            if o1 == DIV_OP:
                imms[i] = m0.bit_length() - 1; ops[i + 1] = SHR_OP; npow += 1
            elif o1 == MOD_OP:
                imms[i] = m0 - 1; ops[i + 1] = AND_OP; npow += 1
    return npow


def _mem_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 1024 / 1024
    return 999


def main():
    steps = int(os.environ.get("DOOM_STEPS", "4000"))
    assert PFC._lea_wide_enabled(), "wide-LEA gate OFF (bare env should be ON on this branch)"
    print(f"[doom-wide-vb] SP_INIT={PFC.SP_INIT:#x} lea_wide={PFC._lea_wide_enabled()} "
          f"fine_grained_wide={PFC._fine_grained_wide_enabled()} DOOM_STEPS={steps}", flush=True)
    if _mem_gb() < 25:
        print(f"[doom-wide-vb] MemAvailable {_mem_gb():.1f}GB < 25GB — abort"); return 3

    snap = np.load(os.path.join(_HERE, "..", "_doom_bytecode_snapshot.npz"))
    ops, imms, data = list(snap["ops"]), list(snap["imms"]), snap["data"]
    _apply_pow2(ops, imms)
    n = len(ops)
    install_compiler_abi_file_dispatcher()
    code = tag_compiler_syscalls(
        [isa.Instr(int(ops[i]), int(imms[i]) & 0xFFFFFFFF) for i in range(n)], isa)
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=steps, mask=0xFFFFFFFF,
                             data_seg=data_segment([int(b) for b in data]), fio=fio)
    n_lea = sum(1 for i in range(draft.step_count) if draft.frames[i].get("op") == "LEA")
    dm = [i for i in range(draft.step_count) if draft.frames[i].get("op") in ("DIV", "MOD")]
    print(f"[doom-wide-vb] drafted {draft.step_count} steps ({n_lea} LEA, {len(dm)} DIV/MOD) "
          f"wall={time.time()-t0:.1f}s", flush=True)

    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(n + 2, 64)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True, addr32=True,
                                             compute_mode="dense_kernel")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[doom-wide-vb] built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} "
          f"build={time.time()-t0:.0f}s peakRSS={rss:.1f}GB", flush=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=int(os.environ.get("C4_LOCAL_WINDOW", "96")),
                            drop_local_kv=True, content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)

    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=int(os.environ.get("K", "100")),
                       device=dev, evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       oom_backoff=True, min_block_steps=4,
                       prime_chunk=int(os.environ.get("DOOM_PRIME_CHUNK", "2048")))
    print(f"[doom-wide-vb] verify_blocks matched={vr.all_matched} forwards={vr.forwards} "
          f"accepted={vr.accepted_steps}/{vr.total_steps} wall={time.time()-t0:.1f}s "
          f"peak_vram={stats.get('peak_vram_gb',0):.1f}GB", flush=True)
    if vr.first_mismatch:
        print(f"[doom-wide-vb] FIRST MISMATCH: {vr.first_mismatch}", flush=True)
    print(f"[doom-wide-vb] DOOM byte-exact on the WIDE default-ON build: {vr.all_matched}",
          flush=True)
    return 0 if vr.all_matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
