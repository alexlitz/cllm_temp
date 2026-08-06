#!/usr/bin/env python3
"""run_doom_checkpoint.py — CHECKPOINT/RESUME runner for the doom-on-transformer step-loop.

Task #814/#866.  A multi-million-step doom render is executed as a sequence of
bounded WINDOWS of VM steps.  Each window's register decode is produced THROUGH THE
MODEL (the composed precomputed-schedule replay — the same byte-exact path
``_agent_doom_real_continuous`` measures).  Between windows the runner CHECKPOINTS the
logical-VM state (``PFResumeState``: registers + memory image + KV store/read logs +
the emitted token prefix) to a ``.npz``.  A wall-clock kill loses at most the current
window; a fresh process ``--resume`` reloads the last checkpoint and continues the
run BYTE-EXACT — the resumed continuation's decoded (PC,SP,BP,AX) lanes are identical
to an uninterrupted run over the same steps (L-inf=0).

WHY BYTE-EXACT.  The draft (``draft_pf_program``) is a deterministic interpreter, so
its future token stream / frames / store_log are a pure function of its state.  The
model verify over a window is a per-query-row independent decode against the FROZEN
token prefix (see ``_agent_doom_real_continuous._slice_draft``): given the same prefix
(a deterministic function of the checkpoint) the model decodes the same registers.  So
resume == uninterrupted at every row.

Modes
-----
  --verify        : run the whole window uninterrupted AND via a mid-window
                    checkpoint->reload->resume, decode both through the model, assert
                    the (PC,SP,BP,AX) lanes are L-inf=0.  Single process (the airtight
                    self-check).
  --run           : advance the run one launch at a time — draft to
                    min(done+window, total), decode through the model, checkpoint, and
                    (if --kill-at is passed and reached) EXIT to simulate a kill.  Re-run
                    with the same --checkpoint to resume.  This is the operational form
                    the 100-frame / full-module runs use.

Memory discipline: LEAN streaming build (C4_PF_CFM), ONE model process, polls
MemAvailable and STOPS if < 25 GB.

Run
---
  # airtight proof (small window, tractable program or a doom slice):
  CUDA_VISIBLE_DEVICES=0 python -m c4_min.run_doom_checkpoint --verify \
      --program nested --steps 4000 --window 1200 --checkpoint /tmp/ck.npz --device cuda:0

  # doom slice proof:
  CUDA_VISIBLE_DEVICES=0 python -m c4_min.run_doom_checkpoint --verify \
      --program doom --steps 60000 --window 20000 --checkpoint /tmp/ck.npz --device cuda:0

  # operational: kill mid-run then resume
  ... --run --steps 400000 --window 100000 --kill-at 150000 --checkpoint /tmp/ck.npz
  ... --run --steps 400000 --window 100000 --checkpoint /tmp/ck.npz   # resumes
"""
from __future__ import annotations
import argparse
import hashlib
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
# LEAN doom config (matches _agent_doom_real_continuous): code-from-memory + 32-bit
# cmp/shift + wide PC/IMM + exact eviction.  These select the DRAFT transition width;
# a resume asserts the checkpoint used the same set.
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_CMP32", "1")
os.environ.setdefault("C4_CMP32_ORDER", "1")
os.environ.setdefault("C4_SHIFT32", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("C4_EXACT_EVICT", "1")
os.environ.setdefault("C4_MEM_EFF", "1")
os.environ.setdefault("C4_IMM_NIBS", "6")
os.environ.setdefault("C4_PC_WIDE", "1")
os.environ.setdefault("C4_CODE_ADDR_BITS", "20")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOOM_SIB = "/home/alexlitz/Documents/misc/c4_doom"
if os.path.isdir(_DOOM_SIB) and _DOOM_SIB not in sys.path:
    sys.path.insert(0, _DOOM_SIB)


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP (share the box)")


def _pin_sp_init(sp_init: int) -> None:
    """RE-ASSERT the drafting stack base right before a draft.  Several modules
    (``tight_attn_compose`` at import, ``build_lib_model_streaming`` internally) hard-set
    ``SP_INIT=0xFC`` as a side effect, which would silently clobber the doom base
    (0x10000).  Pinning it immediately before each ``draft_pf_program`` makes the runner
    immune to import order — the checkpoint's ``sp_init`` config-guard then reflects the
    true value and a resume asserts it matches."""
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = sp_init


# composed levers (the airtight byte-exact doom stack, all default-OFF flags flipped
# on HERE for the run; the bare-env golden is untouched).
_COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
             "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
             "C4_DIRECT_CAM_VEC"]


def _levers_on(chunk):
    for f in _COMPOSED:
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ["C4_MEGABLOCK_BLOCK_K"] = "512"


# ---------------------------------------------------------------------------
# program builders: a tractable synthetic loop, or the REAL doom bytecode.
# ---------------------------------------------------------------------------
def _pow2_reduce(ops, imms):
    IMM_OP, DIV_OP, MOD_OP, SHR_OP, AND_OP = 1, 28, 29, 24, 16
    n = len(ops); npow = 0
    for i in range(n - 1):
        o0, m0, o1 = int(ops[i]), int(imms[i]), int(ops[i + 1])
        if o0 == IMM_OP and m0 > 0 and (m0 & (m0 - 1)) == 0:
            if o1 == DIV_OP:
                imms[i] = m0.bit_length() - 1; ops[i + 1] = SHR_OP; npow += 1
            elif o1 == MOD_OP:
                imms[i] = m0 - 1; ops[i + 1] = AND_OP; npow += 1
    return npow


def build_program(program):
    """Return (code_isa, draft_kwargs) for the chosen program.  draft_kwargs carries
    data_seg / fio for the doom driver.  code_size is the model code band width."""
    from c4_min import isa
    if program == "doom":
        import numpy as np
        from c4_min import nibble_filesys as FS
        from run_c4_min import (tag_compiler_syscalls, data_segment,
                                install_compiler_abi_file_dispatcher)
        snap = np.load(os.path.join(_HERE, "..", "_doom_bytecode_snapshot.npz"))
        ops, imms, data = list(snap["ops"]), list(snap["imms"]), snap["data"]
        npow = _pow2_reduce(ops, imms)
        print(f"[doom] pow2 strength-reduction: {npow} DIV/MOD -> SHR/AND", flush=True)
        install_compiler_abi_file_dispatcher()
        n = len(ops)
        code = tag_compiler_syscalls(
            [isa.Instr(int(ops[i]), int(imms[i]) & 0xFFFFFFFF) for i in range(n)], isa)
        fio = FS.FileOpState(runner=FS.FileRunner(
            fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
        return code, {"data_seg": data_segment([int(b) for b in data]), "fio": fio}
    elif program == "nested":
        from c4_min.bench_fast_path import build_nested
        return build_nested(120, 255)[0], {}
    else:
        raise ValueError(program)


def _code_hash(code):
    h = hashlib.sha256()
    for ins in code:
        h.update(f"{ins.op},{ins.imm};".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# model decode of a draft window -> the (PC,SP,BP,AX) lanes.
# ---------------------------------------------------------------------------
def _dispatch(sched, sg, dev, n, torch):
    got_pc = torch.empty(n, dtype=torch.long, device=dev)
    got_sp = torch.empty(n, dtype=torch.long, device=dev)
    got_bp = torch.empty(n, dtype=torch.long, device=dev)
    got_ax = torch.empty(n, dtype=torch.long, device=dev)
    onchip_ = sched.onchip
    h0s = sched.h0_folded if onchip_ else sched.h0_table
    for lo in range(0, n, sg.chunk):
        hi = min(lo + sg.chunk, n)
        h0 = h0s[lo:hi].unsqueeze(0)
        if onchip_:
            delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, None, None, delta=delta, resident=False)
        else:
            ing = sched.ing_table[lo:hi].permute(1, 0, 2).unsqueeze(0)
            cam = {b: t[lo:hi].permute(1, 0, 2).unsqueeze(0)
                   for b, t in sched.cam_tables.items()}
            pc_c, sp_c, bp_c, ax_c = sg.replay(h0, ing, cam, resident=False)
        got_pc[lo:hi].copy_(pc_c); got_sp[lo:hi].copy_(sp_c)
        got_bp[lo:hi].copy_(bp_c); got_ax[lo:hi].copy_(ax_c)
    return got_pc, got_sp, got_bp, got_ax


def decode_window(model, L, code, draft, device, chunk, mask, torch, PS,
                  install_composed, uninstall_composed):
    """Decode a whole draft window through the composed schedule; return the
    device-side (PC,SP,BP,AX) lane tensors on CPU + the model's byte-exact-vs-draft
    flag (the verify's own accept check)."""
    _levers_on(chunk)
    dev = torch.device(device)
    install_composed(model, verbose=False)
    try:
        sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=mask)
        n = draft.step_count
        pc, sp, bp, ax = _dispatch(sched, sg, dev, n, torch)
        ax = ax & mask
        # the schedule carries the draft's per-step targets; accept = decode==target.
        want_pc, want_ax = sched.want_pc, sched.want_ax & mask
        want_sp, want_bp = sched.want_sp, sched.want_bp
        is_file, is_halt = sched.is_file, sched.is_halt
        bad_normal = ((pc != want_pc) | (ax != want_ax) | (sp != want_sp)
                      | (bp != want_bp))
        bad = torch.where(is_halt, ax != want_ax, bad_normal) & (~is_file)
        n_bad = int(bad.sum())
        torch.cuda.synchronize(dev)
        out = (pc.cpu(), sp.cpu(), bp.cpu(), ax.cpu(), n_bad)
        del sched, sg
        return out
    finally:
        uninstall_composed(model)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
def cmd_verify(args):
    """Airtight single-process proof: uninterrupted-decode == checkpoint/resume-decode,
    L-inf=0 on every register lane, over the [0, steps) window, with a mid-window
    checkpoint written to disk, reloaded, and resumed."""
    _guard()
    import torch
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed, uninstall_composed
    from c4_min import precomputed_schedule as PS
    from c4_min.pf_speculative import draft_pf_program
    from c4_min.ckpt_resume import save_checkpoint, load_checkpoint

    # SP_INIT MUST be set AFTER importing tight_attn_compose — that module hard-sets
    # SP_INIT=0xFC at import time, which would otherwise clobber the doom stack base.
    sp_init = 0x10000 if args.program == "doom" else 0xFC
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = sp_init

    mask = 0xFFFFFFFF
    code, dkw = build_program(args.program)
    chash = _code_hash(code)
    print(f"[verify] program={args.program} code_len={len(code)} hash={chash} "
          f"steps={args.steps} window(split)={args.window} sp_init={sp_init:#x}", flush=True)

    t0 = time.time()
    _pin_sp_init(sp_init)
    full = draft_pf_program(code, max_steps=args.steps, mask=mask, **dkw)
    n = full.step_count
    print(f"[verify] drafted {n} steps halted={full.halted} in {time.time()-t0:.1f}s "
          f"sp_init={sp_init:#x}", flush=True)
    split = min(args.window, n - 1)

    # checkpoint at `split`, persist, reload, resume to n.
    _pin_sp_init(sp_init)
    part = draft_pf_program(code, max_steps=split, mask=mask, capture_state=True, **dkw)
    sz = save_checkpoint(args.checkpoint, part.resume_state,
                         meta={"program": args.program, "code_hash": chash})
    del part
    print(f"[verify] checkpoint @ step {split}: {sz} bytes ({sz/1e6:.2f} MB) "
          f"-> {args.checkpoint}", flush=True)
    st, meta = load_checkpoint(args.checkpoint)
    assert meta.get("code_hash") == chash, "checkpoint code hash mismatch"
    t_r0 = time.time()
    _pin_sp_init(sp_init)
    resumed = draft_pf_program(code, max_steps=args.steps, mask=mask, resume=st, **dkw)
    t_resume_draft = time.time() - t_r0
    assert resumed.step_count == n, (resumed.step_count, n)

    # draft-level identity first (cheap, catches any resume bug before the model).
    assert resumed.tokens == full.tokens, "resumed tokens != full tokens"
    assert resumed.store_log == full.store_log, "resumed store_log != full"
    print(f"[verify] draft-level tail byte-exact (tokens+store_log); resume-draft "
          f"{t_resume_draft:.1f}s", flush=True)
    _guard()

    # build the model ONCE, decode both drafts.
    tb = time.time()
    model, L, _ = build_lib_model_streaming(code_size=max(full.code_off + 2, 256),
                                            recurrent_divmod=True, addr32=True,
                                            compute_mode="dense_kernel")
    model = model.to(args.device)
    print(f"[verify] model built {len(model.blocks)} blocks in {time.time()-tb:.1f}s "
          f"memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()

    u = decode_window(model, L, code, full, args.device, args.chunk, mask, torch, PS,
                      install_composed, uninstall_composed)
    r = decode_window(model, L, code, resumed, args.device, args.chunk, mask, torch, PS,
                      install_composed, uninstall_composed)
    (upc, usp, ubp, uax, un_bad) = u
    (rpc, rsp, rbp, rax, rn_bad) = r

    linf = 0
    for lane, a, b in (("PC", upc, rpc), ("SP", usp, rsp), ("BP", ubp, rbp),
                       ("AX", uax, rax)):
        d = int((a - b).abs().max()) if a.numel() else 0
        nbad = int((a != b).sum())
        print(f"  lane {lane}: n={a.numel()} mismatches={nbad} L-inf={d}", flush=True)
        linf = max(linf, d)
    print(f"[verify] model accept (uninterrupted) bad={un_bad}  (resumed) bad={rn_bad}",
          flush=True)
    ok = (linf == 0) and (un_bad == 0) and (rn_bad == 0)
    print(f"\n=== CHECKPOINT/RESUME BYTE-EXACT: {'YES (L-inf=0)' if ok else 'NO'} ===",
          flush=True)
    print(f"    checkpoint size {sz/1e6:.2f} MB @ step {split} | resume-draft overhead "
          f"{t_resume_draft:.1f}s | decoded {n} steps through the model", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def cmd_run(args):
    """Operational: advance the run from wherever the checkpoint left off, decode
    through the model, checkpoint, and (if --kill-at reached) exit to simulate a kill.
    Re-invoking with the same --checkpoint resumes."""
    _guard()
    import torch
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed, uninstall_composed
    from c4_min import precomputed_schedule as PS
    from c4_min.pf_speculative import draft_pf_program
    from c4_min.ckpt_resume import save_checkpoint, load_checkpoint

    # SP_INIT MUST be set AFTER importing tight_attn_compose — that module hard-sets
    # SP_INIT=0xFC at import time, which would otherwise clobber the doom stack base.
    sp_init = 0x10000 if args.program == "doom" else 0xFC
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = sp_init

    mask = 0xFFFFFFFF
    code, dkw = build_program(args.program)
    chash = _code_hash(code)

    resume_st = None
    done = 0
    if os.path.exists(args.checkpoint):
        resume_st, meta = load_checkpoint(args.checkpoint)
        assert meta.get("code_hash") == chash, "checkpoint is for a different program"
        done = resume_st.steps
        print(f"[run] RESUMING from checkpoint @ step {done} (program={args.program})",
              flush=True)
    else:
        print(f"[run] fresh start (program={args.program}, target {args.steps} steps)",
              flush=True)

    if done >= args.steps:
        print(f"[run] already at/past target ({done} >= {args.steps}); nothing to do.",
              flush=True)
        return 0

    tb = time.time()
    # code band width must fit the whole program (constant across windows).
    _pin_sp_init(sp_init)
    probe = draft_pf_program(code, max_steps=1, mask=mask, **dkw)
    model, L, _ = build_lib_model_streaming(code_size=max(probe.code_off + 2, 256),
                                            recurrent_divmod=True, addr32=True,
                                            compute_mode="dense_kernel")
    model = model.to(args.device)
    del probe
    print(f"[run] model built {len(model.blocks)} blocks in {time.time()-tb:.1f}s "
          f"memAvail={_mem_avail_gb():.1f}GB", flush=True)

    while done < args.steps:
        _guard()
        target = min(done + args.window, args.steps)
        t0 = time.time()
        # draft this window (resume from the prior checkpoint state), capturing the
        # new end-state for the next checkpoint.  Pin SP_INIT immediately before the
        # draft (build_lib_model_streaming / install_composed clobber it to 0xFC).
        _pin_sp_init(sp_init)
        d = draft_pf_program(code, max_steps=target, mask=mask, resume=resume_st,
                             capture_state=True, **dkw)
        t_draft = time.time() - t0
        if d.halted:
            print(f"[run] program HALTED at step {d.step_count}", flush=True)
        # decode the whole (cumulative) draft window through the model = verify.
        t1 = time.time()
        pc, sp, bp, ax, n_bad = decode_window(
            model, L, code, d, args.device, args.chunk, mask, torch, PS,
            install_composed, uninstall_composed)
        t_dec = time.time() - t1
        assert n_bad == 0, f"model rejected the draft ({n_bad} bad rows) — NOT byte-exact"
        done = d.step_count
        sz = save_checkpoint(args.checkpoint, d.resume_state,
                             meta={"program": args.program, "code_hash": chash})
        resume_st = d.resume_state
        print(f"[run] window -> step {done}/{args.steps} | draft {t_draft:.1f}s "
              f"decode {t_dec:.1f}s | accept bad={n_bad} | ckpt {sz/1e6:.2f}MB",
              flush=True)
        del d
        if (args.kill_at is not None and done >= args.kill_at and done < args.steps):
            print(f"[run] --kill-at {args.kill_at} reached (@ {done}); SIMULATING KILL "
                  f"(exit). Re-run to resume from {done}.", flush=True)
            os._exit(42)   # hard exit — no cleanup, mimics a real kill
    print(f"[run] COMPLETE: {done} steps (checkpoint {args.checkpoint})", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="airtight single-process byte-exact proof")
    ap.add_argument("--run", action="store_true",
                    help="operational windowed run with disk checkpoints")
    ap.add_argument("--program", default="nested", choices=["doom", "nested"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=4000, help="total VM steps to run")
    ap.add_argument("--window", type=int, default=1200,
                    help="steps per window/checkpoint (in --verify, the split step)")
    ap.add_argument("--chunk", type=int, default=131072, help="schedule dispatch chunk")
    ap.add_argument("--checkpoint", default="/tmp/doom_ckpt.npz")
    ap.add_argument("--kill-at", type=int, default=None,
                    help="[--run] exit(42) after crossing this step, to simulate a kill")
    args = ap.parse_args(argv)
    if not (args.verify or args.run):
        ap.error("choose --verify or --run")
    if args.verify:
        return cmd_verify(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
