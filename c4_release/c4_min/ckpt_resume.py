#!/usr/bin/env python3
"""ckpt_resume.py — persistent CHECKPOINT / RESUME for the logical-VM draft.

Task #814/#866.  A multi-million-step doom-on-transformer run must survive a
wall-clock kill: checkpoint the VM state periodically, and on restart RESUME from
the last checkpoint and continue BYTE-EXACT (no divergence across the boundary).

The verify pipeline (draft -> precomputed schedule -> replay decode) is driven off
an IMMUTABLE ``PFDraft`` produced by the deterministic ``draft_pf_program`` logical
VM.  Because that interpreter is deterministic, its FUTURE token stream / frames /
store_log are a pure function of its current register+memory state.  So the unit we
checkpoint is that interpreter state (``PFResumeState`` in ``pf_speculative``): save
it, and a fresh process can re-seed the interpreter and draft the identical tail.

This module is pure state I/O + the window/checkpoint bookkeeping.  It builds NO
model and imports no torch, so it is memory-safe (the model-driving runner is
``run_doom_checkpoint.py``).  The checkpoint is a single ``.npz``:

  * scalar registers/counters + the config guard   -> a JSON blob (one npz field);
  * ``mem`` (VM memory image)                       -> two int64 arrays (keys, vals);
  * ``store_log`` (KV write log)                    -> three int64 arrays (frame, addr, val);
  * ``load_log`` / ``read_log`` (CAM read logs)     -> flat int64 arrays;
  * ``tokens`` / ``frames`` / ``win_starts`` / ``out`` / ``prtf_steps`` (the emitted
    prefix the schedule/verify present as context) -> int arrays + a packed frames block.

Frames are packed as a ``[n, 9]`` int64 block (pc, ax, sp, bp, stk, is_store, s_addr,
s_val, is_halt) + a parallel op-name list; this is the exact per-step frame dict the
draft produced, reconstructed on load.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from c4_min.pf_speculative import PFResumeState

_FRAME_INT_KEYS = ("pc", "ax", "sp", "bp", "stk", "is_store", "s_addr", "s_val",
                   "is_halt")
# file ops add two extra per-frame ints (is_file, n_byte_stores); keep them if present.
_FRAME_EXTRA_KEYS = ("is_file", "n_byte_stores")


def _pack_frames(frames):
    n = len(frames)
    base = np.zeros((n, len(_FRAME_INT_KEYS)), dtype=np.int64)
    extra = np.zeros((n, len(_FRAME_EXTRA_KEYS)), dtype=np.int64)
    ops = []
    for i, f in enumerate(frames):
        for j, k in enumerate(_FRAME_INT_KEYS):
            base[i, j] = int(f.get(k, 0))
        for j, k in enumerate(_FRAME_EXTRA_KEYS):
            extra[i, j] = int(f.get(k, 0))
        ops.append(str(f.get("op", "")))
    return base, extra, ops


def _unpack_frames(base, extra, ops, has_extra):
    frames = []
    for i in range(base.shape[0]):
        f = {k: int(base[i, j]) for j, k in enumerate(_FRAME_INT_KEYS)}
        f["is_store"] = bool(f["is_store"])
        f["is_halt"] = bool(f["is_halt"])
        f["op"] = ops[i]
        if has_extra:
            f["is_file"] = bool(int(extra[i, 0]))
            f["n_byte_stores"] = int(extra[i, 1])
        frames.append(f)
    return frames


def _pack_read_log(read_log):
    """read_log: frame_idx -> [(head_str, addr), ...].  Flatten to (frames, heads,
    addrs) with a per-frame count so it round-trips exactly.  head_str in {mem,pop,lev}
    -> a small code."""
    _HCODE = {"mem": 0, "pop": 1, "lev": 2}
    frames, counts, heads, addrs = [], [], [], []
    for fi in sorted(read_log):
        entries = read_log[fi]
        frames.append(fi)
        counts.append(len(entries))
        for (hd, a) in entries:
            heads.append(_HCODE[hd])
            addrs.append(int(a))
    return (np.asarray(frames, dtype=np.int64), np.asarray(counts, dtype=np.int64),
            np.asarray(heads, dtype=np.int64), np.asarray(addrs, dtype=np.int64))


def _unpack_read_log(frames, counts, heads, addrs):
    _HNAME = {0: "mem", 1: "pop", 2: "lev"}
    read_log = {}
    off = 0
    for i in range(frames.shape[0]):
        fi = int(frames[i]); c = int(counts[i])
        read_log[fi] = [(_HNAME[int(heads[off + j])], int(addrs[off + j]))
                        for j in range(c)]
        off += c
    return read_log


def save_checkpoint(path: str, st: PFResumeState, *, meta: Optional[dict] = None) -> int:
    """Atomically persist a ``PFResumeState`` to ``path`` (.npz).  Returns bytes written.

    Atomic: write ``path.tmp`` then ``os.replace`` so a kill mid-write never corrupts a
    prior good checkpoint.  ``meta`` (e.g. program name, code hash) is stored alongside
    the scalar blob for a resume-time sanity check."""
    scalars = dict(
        pc=st.pc, ax=st.ax, sp=st.sp, bp=st.bp, stk=st.stk,
        cur_pc=st.cur_pc, cur_sp=st.cur_sp, cur_bp=st.cur_bp,
        steps=st.steps, frame_idx=st.frame_idx, stream_len=st.stream_len,
        n_seed=st.n_seed, code_off=st.code_off, halted=st.halted,
        cmp32=st.cmp32, shift32=st.shift32, imm_nibs=st.imm_nibs, sp_init=st.sp_init,
        meta=(meta or {}),
    )
    mem_k = np.fromiter(st.mem.keys(), dtype=np.int64, count=len(st.mem))
    mem_v = np.fromiter((st.mem[k] for k in st.mem), dtype=np.int64, count=len(st.mem))
    sl = st.store_log
    sl_f = np.fromiter(sl.keys(), dtype=np.int64, count=len(sl))
    sl_a = np.fromiter((sl[k][0] for k in sl), dtype=np.int64, count=len(sl))
    sl_v = np.fromiter((sl[k][1] for k in sl), dtype=np.int64, count=len(sl))
    ll = st.load_log
    ll_f = np.fromiter(ll.keys(), dtype=np.int64, count=len(ll))
    ll_a = np.fromiter((ll[k] for k in ll), dtype=np.int64, count=len(ll))
    rl_f, rl_c, rl_h, rl_a = _pack_read_log(st.read_log)
    fr_base, fr_extra, fr_ops = _pack_frames(st.frames)
    has_extra = any(("is_file" in f) for f in st.frames)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            scalars=np.frombuffer(json.dumps(scalars).encode("utf-8"), dtype=np.uint8),
            mem_k=mem_k, mem_v=mem_v,
            sl_f=sl_f, sl_a=sl_a, sl_v=sl_v,
            ll_f=ll_f, ll_a=ll_a,
            rl_f=rl_f, rl_c=rl_c, rl_h=rl_h, rl_a=rl_a,
            tokens=np.asarray(st.tokens, dtype=np.int64),
            win_starts=np.asarray(st.win_starts, dtype=np.int64),
            out=np.asarray(st.out, dtype=np.int64),
            prtf_steps=np.asarray(st.prtf_steps, dtype=np.int64),
            fr_base=fr_base, fr_extra=fr_extra,
            fr_ops=np.frombuffer("\n".join(fr_ops).encode("utf-8"), dtype=np.uint8),
            has_extra=np.asarray([1 if has_extra else 0], dtype=np.int64),
        )
    os.replace(tmp, path)
    return os.path.getsize(path)


def load_checkpoint(path: str) -> "tuple[PFResumeState, dict]":
    """Load a ``PFResumeState`` (+ its meta dict) from ``save_checkpoint`` output."""
    z = np.load(path, allow_pickle=False)
    scalars = json.loads(bytes(z["scalars"].tobytes()).decode("utf-8"))
    meta = scalars.pop("meta", {})
    mem = {int(k): int(v) for k, v in zip(z["mem_k"], z["mem_v"])}
    store_log = {int(f): (int(a), int(v))
                 for f, a, v in zip(z["sl_f"], z["sl_a"], z["sl_v"])}
    load_log = {int(f): int(a) for f, a in zip(z["ll_f"], z["ll_a"])}
    read_log = _unpack_read_log(z["rl_f"], z["rl_c"], z["rl_h"], z["rl_a"])
    ops_blob = bytes(z["fr_ops"].tobytes()).decode("utf-8")
    fr_ops = ops_blob.split("\n") if ops_blob else []
    has_extra = bool(int(z["has_extra"][0]))
    frames = _unpack_frames(z["fr_base"], z["fr_extra"], fr_ops, has_extra)
    st = PFResumeState(
        pc=scalars["pc"], ax=scalars["ax"], sp=scalars["sp"], bp=scalars["bp"],
        stk=scalars["stk"], cur_pc=scalars["cur_pc"], cur_sp=scalars["cur_sp"],
        cur_bp=scalars["cur_bp"], steps=scalars["steps"],
        frame_idx=scalars["frame_idx"], stream_len=scalars["stream_len"],
        n_seed=scalars["n_seed"], code_off=scalars["code_off"],
        halted=bool(scalars["halted"]),
        mem=mem, store_log=store_log, load_log=load_log, read_log=read_log,
        tokens=[int(t) for t in z["tokens"]], frames=frames,
        win_starts=[int(w) for w in z["win_starts"]],
        out=[int(o) for o in z["out"]], prtf_steps=[int(p) for p in z["prtf_steps"]],
        cmp32=bool(scalars["cmp32"]), shift32=bool(scalars["shift32"]),
        imm_nibs=int(scalars["imm_nibs"]), sp_init=int(scalars["sp_init"]),
    )
    return st, meta
