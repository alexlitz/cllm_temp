#!/usr/bin/env python3
"""_agent_run_program_c.py — run a WHOLE c4 program end-to-end with the block-stack
evaluated by the SPARSE (or dense) native C runtime instead of torch.

This is a byte-for-byte port of nibble_pure_forward_complete.run_pure_forward_complete
with ONLY the ``for blk in model.blocks: x = blk(x)`` torch block-stack swapped for
a persistent C-runtime serve process (rt.forward_residual).  The embed lookup,
program overlay, frame build, IO dispatch (PRTF visible output + OPEN/READ/CLOS file
ops), and store bookkeeping stay in Python EXACTLY as the torch driver — the C
runtime executes the same vanilla transformer forward per step.

Used to run echo / quine / mandelbrot byte-exact and measure per-program wall.
"""
from __future__ import annotations

import os
import struct
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min import nibble_pure_forward_complete as PFC
from c4_min import nibble_filesys as _FS


class CServe:
    """Persistent C-runtime serve process bound to one .nblbin block-stack graph."""

    def __init__(self, exe: str, binp: str, dim: int):
        self.exe = exe; self.binp = binp; self.dim = dim
        self.steps = 0
        self.wall = 0.0
        self._proc = subprocess.Popen(
            [exe, binp, "-", "--residual-in", "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def forward_residual(self, x: np.ndarray) -> np.ndarray:
        B, S, D = x.shape
        payload = struct.pack("<iii", B, S, D) + np.ascontiguousarray(x, "<f4").tobytes()
        need = B * S * D * 4
        t0 = time.time()
        self._proc.stdin.write(payload); self._proc.stdin.flush()
        buf = b""
        while len(buf) < need:
            chunk = self._proc.stdout.read(need - len(buf))
            if not chunk:
                raise RuntimeError("C runtime serve closed early")
            buf += chunk
        self.wall += time.time() - t0
        self.steps += 1
        return np.frombuffer(buf, dtype="<f4").reshape(B, S, D)

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close(); self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None


def run_program_c(rt: CServe, L, code: List[isa.Instr], embed: np.ndarray,
                  max_steps: int = 20000, mask: int = 0xFF,
                  fio=None, data_seg=None, out: Optional[List[int]] = None,
                  seed_mem=None) -> List[int]:
    """Byte-for-byte port of PFC.run_pure_forward_complete with the block stack run
    by the C runtime.  Returns the per-step AX trace; if ``out`` is given, PRTF bytes
    are appended to it (the visible output)."""
    init_frame = PFC._build_frame(0, 0, PFC.SP_INIT, PFC.SP_INIT, 0)
    seed_frames, store_log = PFC._seed_frames(seed_mem or {})
    n_seed = len(store_log)
    stream: List[int] = [V.BOS] + seed_frames + init_frame
    trace: List[int] = []
    cur_pc = 0
    cur_sp = cur_bp = PFC.SP_INIT
    cur_ax = 0
    frame_idx = n_seed
    for _ in range(max_steps):
        overlay = PFC.make_overlay_complete(code, L, store_log=store_log)
        x = torch.from_numpy(embed[np.asarray(stream)]).unsqueeze(0).clone()
        overlay(x)
        _t = time.time()
        hidden = rt.forward_residual(x.numpy())
        if os.environ.get("PROG_VERBOSE"):
            import sys as _sys
            print(f"  step {rt.steps:3d}  S={x.shape[1]:4d}  "
                  f"{(time.time()-_t)*1000:8.1f} ms", flush=True, file=_sys.stderr)
        state = torch.from_numpy(hidden[0, -1])
        pc = PFC._snap_lane(state[L.PC_VAL])
        sp = PFC._snap_lane(state[L.SP_VAL])
        bp = PFC._snap_lane(state[L.BP_VAL])
        stk = PFC._snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        imm = code[cur_pc].imm if 0 <= cur_pc < len(code) else 0
        ax = PFC._decode_reg_from_nibbles(state, L, L.AX)
        # FILE OP (OPEN/READ/CLOS/PRTF via TOOL_CALL) — driver performs the real I/O.
        if fio is not None and op in _FS.FILE_OPCODES:
            new_ax, new_sp, byte_stores = _FS.dispatch_file_op_driver(
                op, cur_ax & 0xFFFFFFFF, imm, cur_sp, store_log, fio,
                data_seg=data_seg, slot=4)
            pc = cur_pc + 1; sp = new_sp; bp = cur_bp; ax = new_ax & 0xFFFFFFFF
            frame = PFC._build_frame(pc, ax, sp, bp, stk)
            trace.append(ax & mask); frame_idx += 1; stream += frame
            for (baddr, bval) in byte_stores:
                bframe = PFC._build_frame(pc, ax, sp, bp, stk,
                                          mem_addr=baddr, mem_val=bval & 0xFF)
                frame_idx += 1
                store_log[frame_idx] = (baddr, bval & 0xFF)
                stream += bframe
            cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
            if pc < 0 or pc >= len(code):
                break
            continue
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = PFC._mem_top(store_log, cur_sp); s_val = ax & mask
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = PFC._build_frame(pc, ax, sp, bp, stk,
                                 mem_addr=(s_addr if is_store else 0),
                                 mem_val=(s_val if is_store else 0))
        trace.append(ax & mask)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        if op == isa.PRTF:
            emit_b = ax & 0xFF
            if out is not None:
                out.append(emit_b)
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    return trace
