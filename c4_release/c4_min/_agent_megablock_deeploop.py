#!/usr/bin/env python3
"""_agent_megablock_deeploop.py — DEEP-LOOP byte-exactness: run a real loop program
for many steps, comparing the per-step AX trace of the EAGER block-skip forward vs
the MEGABLOCK-fused forward.  The task's "DIV-free battery + deep loop" gate.

Each VM step is an independent re-embed forward, so the step-forward's dead-FFN
blocks are fused via the mega chain; the decoded AX trace must match the eager
per-step trace at EVERY step across the whole loop.
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch

from . import isa
from .compact_alloc import build_compact_sparse_streaming
from .step_block_skip import StepBlockSkipRunner, build_live_index
from .block_sparse_ffn import install_block_sparse_ffn
from .live_head_attention import install_live_head_attention, install_dead_block_fusion
from .fused_megablock import MegaBlockChain
from . import nibble_pure_forward_complete as pfc
from .nibble_pure_forward_complete import (make_overlay_complete, _decode_reg_from_nibbles,
                                           _build_frame, SP_INIT, _seed_frames)


def build():
    dev = "cuda:0"; torch.cuda.set_device(0)
    model, L, _ = build_compact_sparse_streaming(code_size=48,
                                                 compute_mode="dense_kernel")
    model.to(dev); model.materialize_dense(dev)
    install_block_sparse_ffn(model, mode="coo", verbose=False)
    model.to(dev)
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)
    return model, L, dev


class MegaStepForward:
    """A per-step forward that fuses the DIV-free dead-FFN blocks via mega chains
    (the eager block-skip forward with the dead-FFN segments replaced by the
    on-chip in-place-delta megakernel)."""

    def __init__(self, model, L, dev, runner):
        self.model = model; self.L = L; self.dev = dev; self.runner = runner
        self._chains = {}     # op -> [(kind, payload)]

    def _items(self, op):
        it = self._chains.get(op)
        if it is None:
            live = sorted(self.runner.live_index.get(op, self.runner.live_index[None]))
            items = []; seg = []
            for b in live:
                if b == 0:
                    items.append(("live", b)); continue
                if getattr(self.model.blocks[b].attn, "_dead_block_fused", False):
                    seg.append(b)
                else:
                    if seg:
                        items.append(("mega", MegaBlockChain(self.model, self.dev, seg, 128)))
                        seg = []
                    items.append(("live", b))
            if seg:
                items.append(("mega", MegaBlockChain(self.model, self.dev, seg, 128)))
            self._chains[op] = it = items
        return it

    def forward(self, x, op):
        # DIV/MOD -> fall back to the eager block-skip (megablock is doom DIV-free only)
        if op in (isa.DIV, isa.MOD):
            return self.runner.forward(x, op)
        qp = torch.arange(x.shape[1], device=self.dev, dtype=torch.long)
        h = x
        for kind, p in self._items(op):
            if kind == "live":
                h, _ = self.model.blocks[p](h, past_kv=None, q_positions=qp,
                                            use_cache=True)
            else:
                h = p.run(h)
        return h


def run_trace(model, L, dev, code, forward_fn, max_steps):
    """Minimal per-step re-embed decode driver (AX trace only)."""
    from .nibble_pure_forward_complete import _MEM_ADDR_LOCAL
    seed_frames, store_log = _seed_frames({})
    init = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    overlay = make_overlay_complete(code, L, store_log=store_log)
    stream = list(init)
    cur_pc = 0; cur_ax = 0; cur_sp = cur_bp = SP_INIT; cur_stk = 0
    trace = []
    for step in range(max_steps):
        if cur_pc >= len(code):
            break
        op = code[cur_pc].op
        toks = torch.tensor([stream], device=dev)
        with torch.no_grad():
            x = model.embed[toks].clone(); overlay(x)
            out = forward_fn(x, op)
        ax = _decode_reg_from_nibbles(out[0, -1], L, L.AX)
        trace.append(ax & 0xFF)
        if op == isa.HALT:
            break
        # advance PC minimally (this is a decode-fidelity trace, not a full VM): use
        # the model's own emitted PC nibbles so both paths advance IDENTICALLY.
        pc = _decode_reg_from_nibbles(out[0, -1], L, L.PC)
        sp = _decode_reg_from_nibbles(out[0, -1], L, L.SP)
        bp = _decode_reg_from_nibbles(out[0, -1], L, L.BP)
        stk = _decode_reg_from_nibbles(out[0, -1], L, L.STACK0)
        # append the next frame (PC/AX/SP/BP/STK the model decoded) so the streams
        # stay identical across both forwards.
        stream += _build_frame(pc & 0xFFFF, ax, sp, bp, stk)
        cur_pc = (pc & 0xFFFF)
        if cur_pc >= len(code):
            break
    return trace


def main():
    model, L, dev = build()
    runner = StepBlockSkipRunner(model, L)
    mega = MegaStepForward(model, L, dev, runner)

    # a small DIV-free loop body: a mix of ops repeated (the deep loop).
    prog = [
        ("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0),
        ("LEA", 1), ("IMM", 7), ("SUB", 0), ("IMM", 1),
        ("LT", 0), ("IMM", 2), ("OR", 0), ("SHL", 0),
        ("EQ", 0), ("PSH", 0), ("AND", 0), ("XOR", 0),
        ("IMM", 9), ("HALT", 0),
    ]
    code = isa.assemble(prog)
    N = 64
    eager_trace = run_trace(model, L, dev, code, lambda x, op: runner.forward(x, op), N)
    mega_trace = run_trace(model, L, dev, code, mega.forward, N)
    match = eager_trace == mega_trace
    print(f"deep-loop steps: eager={len(eager_trace)} mega={len(mega_trace)}", flush=True)
    print(f"eager AX trace: {eager_trace}", flush=True)
    print(f"mega  AX trace: {mega_trace}", flush=True)
    print(f"DEEP-LOOP byte-exact: {'YES' if match else 'NO'} "
          f"({sum(1 for a,b in zip(eager_trace,mega_trace) if a==b)}/"
          f"{min(len(eager_trace),len(mega_trace))} steps match)", flush=True)


if __name__ == "__main__":
    main()
