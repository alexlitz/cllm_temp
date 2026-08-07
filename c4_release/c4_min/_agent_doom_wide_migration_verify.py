#!/usr/bin/env python3
"""_agent_doom_wide_migration_verify.py — GOLDEN-MIGRATION GATE for the wide-address
default-ON flip (#854).

The concern: flipping C4_CMP32 / C4_SHIFT32 / C4_PC_WIDE / C4_GLOBAL_ADDR32 / C4_SP_WIDE
DEFAULT-ON changes doom's PRODUCTION build.  The doom runners (run_doom_checkpoint,
_agent_doom_real_continuous) ALREADY setdefault C4_CMP32=C4_SHIFT32=C4_PC_WIDE=1 (+
C4_DRAFT_CMP32=1), so for THOSE three flags the flip is a no-op for doom.  The migration's
TRUE delta for doom is the two flags the doom runners do NOT set:

    C4_GLOBAL_ADDR32  (0 -> 1)   — MODEL-side load-query width (8b -> 32b); DRAFT-independent
    C4_SP_WIDE        (0 -> 1)   — MODEL-side SP/BP signed-wide decode; DRAFT-independent

Neither is read by the DRAFT (pf_speculative), only by the MODEL decode.  So the doom
DRAFT (the ground-truth register/store trace) is IDENTICAL across the two settings; only
the MODEL's decode of that fixed trace can change.  This makes it a clean superset test:

  * mode "draft"  : draft REAL doom N steps (production config), dump the draft's
                    per-step targets (want_pc/sp/bp/ax) + a store_log hash to --out.
                    Sets NEITHER migration flag -> proves the draft is independent
                    (run it twice with the two flag settings; the dumps must be identical).
  * mode "decode" : build the model with the CURRENT env (caller sets the two flags),
                    decode the fixed doom draft, dump the model's per-step decoded
                    (pc/sp/bp/ax) lanes + the bad-vs-draft count to --out.
  * mode "compare": load two decode dumps (wide vs narrow), report per-lane L-inf, the
                    per-setting bad-vs-draft, and whether the doom OUTPUT is byte-identical.

Byte-exact-superset PASS = the doom draft is identical across the two flag settings AND
the wide model's bad-vs-draft <= the narrow model's (the wide decode is a superset: it
byte-exactly reproduces every row the narrow one did, and possibly more).

LEAN (C4_PF_CFM=1), ONE model process, stops < 25 GB host RAM.  Same doom-proven config
as _agent_doom_real_continuous.
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "4")
# doom-proven config (identical to _agent_doom_real_continuous) — these are set in EVERY
# mode so the DRAFT + the schedule width are constant; the migration flags GLOBAL_ADDR32
# and SP_WIDE are left to the caller (bare env => the branch default).
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

_DOOM_SIB = "/home/alexlitz/Documents/misc/c4_doom"
if os.path.isdir(_DOOM_SIB) and _DOOM_SIB not in sys.path:
    sys.path.insert(0, _DOOM_SIB)
_HERE = os.path.dirname(os.path.abspath(__file__))


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP")


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


def _load_snapshot():
    import numpy as np
    snap = np.load(os.path.join(_HERE, "..", "_doom_bytecode_snapshot.npz"))
    return list(snap["ops"]), list(snap["imms"]), snap["data"]


def _build_doom(steps):
    """Draft the REAL doom bytecode `steps` steps (pow2 image, production config)."""
    import numpy as np
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0x10000
    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program
    from run_c4_min import (data_segment, tag_compiler_syscalls,
                            install_compiler_abi_file_dispatcher)
    ops, imms, data = _load_snapshot()
    npow = _apply_pow2(ops, imms)
    print(f"[doom] pow2 strength-reduction: {npow} DIV/MOD -> SHR/AND", flush=True)
    install_compiler_abi_file_dispatcher()
    n = len(ops)
    code = tag_compiler_syscalls(
        [isa.Instr(int(ops[i]), int(imms[i]) & 0xFFFFFFFF) for i in range(n)], isa)
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    d = draft_pf_program(code, max_steps=steps, mask=0xFFFFFFFF,
                         data_seg=data_segment([int(b) for b in data]), fio=fio)
    print(f"[doom] drafted {d.step_count} steps halted={d.halted} in {time.time()-t0:.1f}s",
          flush=True)
    dm = [i for i in range(d.step_count) if d.frames[i].get("op") in ("DIV", "MOD")]
    assert not dm, f"draft not DIV-free: {dm[:5]}"
    return code, d


def _draft_targets(d):
    """Deterministic per-step targets + a store_log fingerprint (draft ground truth)."""
    import numpy as np
    n = d.step_count
    pc = np.array([d.frames[i]["pc"] for i in range(n)], dtype=np.int64)
    ax = np.array([d.frames[i]["ax"] & 0xFFFFFFFF for i in range(n)], dtype=np.int64)
    sp = np.array([d.frames[i]["sp"] & 0xFFFFFFFF for i in range(n)], dtype=np.int64)
    bp = np.array([d.frames[i]["bp"] & 0xFFFFFFFF for i in range(n)], dtype=np.int64)
    h = hashlib.sha256()
    for k in sorted(d.store_log.keys()):
        h.update(f"{k}:{d.store_log[k]};".encode())
    slog = h.hexdigest()
    th = hashlib.sha256((",".join(str(t) for t in d.tokens)).encode()).hexdigest()
    return {"pc": pc, "ax": ax, "sp": sp, "bp": bp, "store_log_sha": slog,
            "tokens_sha": th, "n": n}


def cmd_draft(args):
    _guard()
    print(f"[draft] GLOBAL_ADDR32={os.environ.get('C4_GLOBAL_ADDR32','<unset>')} "
          f"SP_WIDE={os.environ.get('C4_SP_WIDE','<unset>')} (should NOT affect the draft)",
          flush=True)
    _code, d = _build_doom(args.steps)
    tg = _draft_targets(d)
    import numpy as np
    np.savez(args.out, pc=tg["pc"], ax=tg["ax"], sp=tg["sp"], bp=tg["bp"],
             store_log_sha=tg["store_log_sha"], tokens_sha=tg["tokens_sha"], n=tg["n"])
    print(f"[draft] n={tg['n']} store_log_sha={tg['store_log_sha'][:16]} "
          f"tokens_sha={tg['tokens_sha'][:16]} -> {args.out}", flush=True)
    return 0


def cmd_decode(args):
    _guard()
    import torch
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed, uninstall_composed
    from c4_min import precomputed_schedule as PS

    ga = os.environ.get("C4_GLOBAL_ADDR32", "<unset>")
    sw = os.environ.get("C4_SP_WIDE", "<unset>")
    print(f"[decode] MIGRATION FLAGS: C4_GLOBAL_ADDR32={ga} C4_SP_WIDE={sw}", flush=True)

    code, d = _build_doom(args.steps)
    mask = 0xFFFFFFFF
    dev = torch.device(args.device)
    _guard()
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(code_size=max(d.code_off + 2, 256),
                                            recurrent_divmod=True, addr32=True,
                                            compute_mode="dense_kernel")
    model = model.to(dev)
    print(f"[decode] model built {len(model.blocks)} blocks dim={model.embed.shape[1]} "
          f"in {time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()

    _levers_on(args.chunk)
    install_composed(model, verbose=False)
    try:
        sched, sg = PS.build_schedule(model, L, code, d, dev, mask=mask)
        n = d.step_count
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
        got_ax = got_ax & mask
        want_pc = sched.want_pc; want_ax = sched.want_ax & mask
        want_sp = sched.want_sp; want_bp = sched.want_bp
        is_file = sched.is_file; is_halt = sched.is_halt
        bad_normal = ((got_pc != want_pc) | (got_ax != want_ax) | (got_sp != want_sp)
                      | (got_bp != want_bp))
        bad = torch.where(is_halt, got_ax != want_ax, bad_normal) & (~is_file)
        n_bad = int(bad.sum())
        bad_idx = torch.nonzero(bad).flatten()[:20].cpu().numpy().tolist()
        torch.cuda.synchronize(dev)
        import numpy as np
        np.savez(args.out,
                 pc=got_pc.cpu().numpy(), sp=got_sp.cpu().numpy(),
                 bp=got_bp.cpu().numpy(), ax=got_ax.cpu().numpy(),
                 want_pc=want_pc.cpu().numpy(), want_ax=want_ax.cpu().numpy(),
                 want_sp=want_sp.cpu().numpy(), want_bp=want_bp.cpu().numpy(),
                 is_file=is_file.cpu().numpy(), is_halt=is_halt.cpu().numpy(),
                 n_bad=n_bad, n=n, ga=int(ga == "1" or ga == "<unset>"),
                 sw=int(sw == "1" or sw == "<unset>"))
        print(f"[decode] n={n} bad-vs-draft={n_bad}  first_bad_idx={bad_idx}", flush=True)
        del sched, sg
    finally:
        uninstall_composed(model)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return 0


def cmd_compare(args):
    import numpy as np
    A = np.load(args.a); B = np.load(args.b)   # A = wide (default), B = narrow (rollback)
    print(f"[compare] A(wide) bad-vs-draft={int(A['n_bad'])}  "
          f"B(narrow) bad-vs-draft={int(B['n_bad'])}  n={int(A['n'])}", flush=True)
    ok_lanes = True
    for lane in ("pc", "sp", "bp", "ax"):
        a, b = A[lane], B[lane]
        linf = int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) if a.size else 0
        nbad = int((a != b).sum())
        print(f"  lane {lane.upper()}: n={a.size} mismatches(wide vs narrow)={nbad} L-inf={linf}",
              flush=True)
        ok_lanes = ok_lanes and (nbad == 0)
    # superset test: every row the NARROW model got right (matched draft), the WIDE model
    # must also get right.
    wa = A; na = B
    want_ok = lambda D: ~(np.where(D["is_halt"].astype(bool),
                                   D["ax"] != D["want_ax"],
                                   (D["pc"] != D["want_pc"]) | (D["ax"] != D["want_ax"]) |
                                   (D["sp"] != D["want_sp"]) | (D["bp"] != D["want_bp"]))
                          ) & (~na["is_file"].astype(bool))
    narrow_ok = want_ok(na)
    wide_ok = want_ok(wa)
    narrow_but_not_wide = int((narrow_ok & (~wide_ok)).sum())
    wide_but_not_narrow = int((wide_ok & (~narrow_ok)).sum())
    print(f"\n  [SUPERSET] rows narrow-matched-draft but wide-did-NOT = {narrow_but_not_wide}",
          flush=True)
    print(f"  [SUPERSET] rows wide-matched-draft but narrow-did-NOT = {wide_but_not_narrow}",
          flush=True)
    print(f"\n  identical output (wide == narrow, all lanes): "
          f"{'YES' if ok_lanes else 'NO'}", flush=True)
    print(f"  wide is a byte-exact SUPERSET of narrow (narrow_but_not_wide==0): "
          f"{'YES' if narrow_but_not_wide == 0 else 'NO'}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=["draft", "decode", "compare"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--out", default=None)
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    args = ap.parse_args(argv)
    if args.mode == "draft":
        return cmd_draft(args)
    if args.mode == "decode":
        return cmd_decode(args)
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
