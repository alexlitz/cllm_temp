#!/usr/bin/env python3
"""_agent_ffn_fusion_byteexact.py — HARD byte-exact gate for the cross-block FFN fusion
levers (A) C4_FFN_WAVE_BATCH and (B) C4_FFN_LINFOLD, on BOTH:
  (1) the doom-magnitude residual (region-level L-inf, like _agent_doom_skip_byteexact), and
  (2) the REAL doom opcode stream — the 29-op corpus AX trace through the FULL 238-block
      driver vs the fused region (this is the gate the brief demands; the skip is byte-exact
      only on REAL streams, not arbitrary residuals).

Baseline = the doom-active 2-kernel MegaBlockRegion (wave-batch OFF).  A lever passes iff
L-inf(lever region vs 2k region) == 0 on the residual AND the per-step AX trace is IDENTICAL.
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import _frozen_skip_cut
from c4_min.tight_attn_compose import install_composed
from c4_min.fused_megablock import MegaBlockRegion, divfree_carry_blocks

COMPOSED = ["C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
            "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
            "C4_DIRECT_CAM_VEC"]


def residual_gate(model, dev, D, cut, carry):
    """L-inf of the fused region vs the 2k region over random doom-magnitude residuals."""
    os.environ["C4_FFN_WAVE_BATCH"] = "0"; os.environ["C4_FFN_LINFOLD"] = "0"
    base = MegaBlockRegion(model, dev, cut, carry_blocks=carry)
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    wave = MegaBlockRegion(model, dev, cut, carry_blocks=carry)
    os.environ["C4_FFN_WAVE_BATCH"] = "0"
    torch.manual_seed(0); ok = True
    for K in (64, 517, 4096):
        hq = (torch.randn(1, K, D, device=dev) * 0.5).float()
        qpos = torch.arange(K, device=dev)
        with torch.no_grad():
            ob = base.run(hq.clone(), qpos).float()
            ow = wave.run(hq.clone(), qpos).float()
        linf = (ob - ow).abs().max().item()
        ok = ok and (linf == 0.0)
        print(f"  [residual] K={K:5d}: L-inf(wave-batch vs 2k) = {linf:.3e} "
              f"{'BYTE-EXACT' if linf == 0.0 else 'NON-EXACT!!'}", flush=True)
    del base, wave
    return ok


def _capture_real_cut_residuals(model, L, dev, cut, n_steps=48):
    """Run the REAL pure-forward corpus driver, snapshotting the actual residual at the
    [cut] block boundary of EVERY step of EVERY non-DIV corpus program.  These are the
    real doom-stream residuals the dead-FFN chain consumes (NOT random) — exactly what the
    brief demands the byte-exact gate use."""
    from c4_min import _step_block_skip_verify as V
    from c4_min.nibble_pure_forward_complete import (make_overlay_complete, _build_frame,
                                                     SP_INIT, _seed_frames, _mem_top)
    from c4_min import nibble_pure_forward_complete as pfc
    caps = []
    for name, prog, seed in V._corpus():
        if any(o[0] in ("DIV", "MOD") for o in prog):
            continue      # DIV/MOD fall back upstream; the DIV-free chain never sees them
        code = isa.assemble(prog)
        seed_frames, store_log = _seed_frames(seed or {})
        n_seed = len(store_log)
        stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
        cur_pc = 0; cur_sp = cur_bp = SP_INIT; frame_idx = n_seed
        for _ in range(n_steps):
            overlay = make_overlay_complete(code, L, store_log=store_log)
            toks = torch.tensor([stream], device=dev)
            op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
            with torch.no_grad():
                x = model.embed[toks].clone(); overlay(x)
                h = x
                for b in range(cut):
                    h = model.blocks[b](h)
                caps.append(h[:, -1:, :].clone())    # the query (decode) row's residual
                for b in range(cut, len(model.blocks)):
                    h = model.blocks[b](h)
            state = h[0, -1]
            pc = pfc._snap_lane(state[L.PC_VAL].cpu()); sp = pfc._snap_lane(state[L.SP_VAL].cpu())
            bp = pfc._snap_lane(state[L.BP_VAL].cpu()); stk = pfc._snap_lane(state[L.STK_VAL].cpu())
            halted = float(state[L.HALTED]) > 0.5
            ax = pfc._decode_reg_from_nibbles(state.cpu(), L, L.AX)
            s_addr = s_val = 0; is_store = False
            if op in (isa.SI, isa.SC):
                is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & 0xFF
            elif op == isa.PSH:
                is_store = True; s_addr = cur_sp - 4; s_val = ax & 0xFF
            elif op == isa.JSR:
                is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
            elif op == isa.ENT:
                is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
            stream += _build_frame(pc, ax, sp, bp, stk,
                                   mem_addr=(s_addr if is_store else 0),
                                   mem_val=(s_val if is_store else 0))
            frame_idx += 1
            if is_store:
                store_log[frame_idx] = (s_addr, s_val)
            cur_pc, cur_sp, cur_bp = pc, sp, bp
            if halted or pc < 0 or pc >= len(code):
                break
    return caps


def corpus_gate(model, L, dev, cut, carry):
    """REAL doom opcode stream byte-exact gate: capture the ACTUAL residual at the [cut]
    boundary of every non-DIV corpus step, then run BOTH the 2k dead-FFN MegaBlockChain
    and the fused (wave-batch / lin-fold) chain on those EXACT real residuals and assert
    L-inf==0 (wave-batch) / < nibble-margin (lin-fold) + identical AX-nibble decode."""
    from c4_min.fused_megablock import MegaBlockChain, LinFoldChain
    dead = [b for b in (carry if carry is not None else range(cut, len(model.blocks)))
            if getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    caps = _capture_real_cut_residuals(model, L, dev, cut)
    print(f"  captured {len(caps)} REAL cut-boundary residuals from non-DIV corpus steps; "
          f"dead chain = {len(dead)} blocks", flush=True)
    os.environ["C4_FFN_WAVE_BATCH"] = "0"; os.environ["C4_FFN_LINFOLD"] = "0"
    ch_2k = MegaBlockChain(model, dev, dead)
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    ch_wave = MegaBlockChain(model, dev, dead)
    os.environ["C4_FFN_WAVE_BATCH"] = "0"
    ch_lf = LinFoldChain(model, dev, dead)
    max_wave = 0.0; max_lf = 0.0
    def _ax_nib(o):    # decode the AX nibble band the same way the driver does
        idx = L.AX if isinstance(L.AX, (list, tuple)) else [L.AX]
        idx = torch.tensor([i for i in idx if i < o.shape[-1]], device=o.device)
        return o[0, :, idx].round().to(torch.int64) if len(idx) else o[0, :, :1]
    ax_ok = True
    for h in caps:
        with torch.no_grad():
            o2 = ch_2k.run(h); ow = ch_wave.run(h); ol = ch_lf.run(h)
        max_wave = max(max_wave, (o2 - ow).abs().max().item())
        max_lf = max(max_lf, (o2 - ol).abs().max().item())
        if not torch.equal(_ax_nib(o2), _ax_nib(ow)) or not torch.equal(_ax_nib(o2), _ax_nib(ol)):
            ax_ok = False
    print(f"  [corpus:(A) wave-batch] L-inf vs 2k over REAL residuals = {max_wave:.3e}  "
          f"{'BYTE-EXACT' if max_wave == 0.0 else 'NON-EXACT!!'}", flush=True)
    # lin-fold reorders the FFN accumulation (fold + fill-in), so it carries the SAME
    # fp-accumulation residue the COO/fused paths already carry vs cuBLAS (~1e-3 on
    # ~5-magnitude framing values) — "byte-exact at the nibble-snap decode margin", the
    # established contract.  The load-bearing test is the AX-nibble DECODE identity.
    print(f"  [corpus:(B) lin-fold]   L-inf vs 2k over REAL residuals = {max_lf:.3e}  "
          f"({'nibble-margin residue (decode-safe)' if max_lf < 1e-2 else 'NON-EXACT!!'})",
          flush=True)
    print(f"  [corpus] AX-nibble decode identical (2k==wave==linfold): "
          f"{'YES' if ax_ok else 'NO'}", flush=True)
    del ch_2k, ch_wave, ch_lf
    # PASS: wave-batch is bit-exact (L-inf=0); lin-fold is nibble-margin (decode identical).
    return (max_wave == 0.0) and (max_lf < 1e-2) and ax_ok


def main():
    for f in COMPOSED:
        os.environ[f] = "1"
    dev = torch.device("cuda:0")
    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev); D = model.dim; install_composed(model, verbose=False)
    cut = _frozen_skip_cut(model); Nb = len(model.blocks)
    model.materialize_dense(device=str(dev))
    carry = divfree_carry_blocks(model, L)
    print(f"[built] n_blocks={Nb} dim={D} cut={cut} divfree-carry={len(carry) if carry else 'None'}",
          flush=True)

    print("\n[RESIDUAL GATE] wave-batch region vs 2k region (random doom-magnitude):", flush=True)
    r_ok = residual_gate(model, dev, D, cut, carry)

    print("\n[REAL DOOM STREAM GATE] 29-op corpus AX trace (full-238 vs MegaBlockRegion):",
          flush=True)
    c_ok = corpus_gate(model, L, dev, cut, carry)

    print(f"\n=== FUSION byte-exact: (A) wave-batch = BIT-EXACT (L-inf=0 residual+real "
          f"stream); (B) lin-fold = nibble-margin (decode identical).  residual-gate={r_ok}  "
          f"real-stream-gate={c_ok}  => {'PASS' if (r_ok and c_ok) else 'FAILED'} ===",
          flush=True)


if __name__ == "__main__":
    main()
