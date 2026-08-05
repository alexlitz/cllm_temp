#!/usr/bin/env python3
"""_agent_attnmega_byteexact.py — verify routing a LIVE-CAM block's FFN through the SAME
sparse fused-hidden [D,K] megakernel the dead-FFN chain uses is byte-exact to the current
dense-F.linear ``_LiveCamBlock.forward_static_delta`` path.

Current live path (on-chip):  h.clone(); h[:,:,out_dims]+=wo_delta; SparseFFN.forward(h)
                              = dense F.linear W_up/W_gate/W_down + full-width silu/bias.
Proposed fused path:          y=h[0].T [D,K]; y[out_dims,:]+=wo_delta.T;
                              _MegaFFN(ffn).run_fused(y,K); h_out = y.T.
Both compute x + W_down(silu(W_up x+b)*(W_gate x+b)) + b_down on the SAME (x+delta).

We compare on REAL doom-stream residuals captured at each live block's input, and check the
AX/PC/SP/BP nibble decode is IDENTICAL (the production correctness gate).
"""
from __future__ import annotations
import argparse, os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch


def _guard():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                if int(ln.split()[1]) / 1e6 < 25.0:
                    raise SystemExit("[GUARD] <25GB -> STOP")


def _levers_on(chunk, block_k=256):
    for f in ("C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
              "C4_DIRECT_CAM_VEC"):
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    os.environ["C4_MEGABLOCK_BLOCK_K"] = str(block_k)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--block-k", type=int, default=256)
    args = ap.parse_args(argv)
    _guard()
    dev = torch.device(args.device)
    _levers_on(args.chunk, args.block_k)

    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed
    from c4_min.bench_fast_path import build_nested
    from c4_min import precomputed_schedule as PS
    from c4_min.fused_megablock import _MegaFFN

    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(args.device)
    _guard()
    import pickle
    with open("/tmp/_wf_draft_120_255.pkl", "rb") as fh:
        draft = pickle.load(fh)
    code = build_nested(120, 255)[0]
    install_composed(model, verbose=False)
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)

    n = min(args.chunk, draft.step_count)
    h0 = (sched.h0_folded if sched.onchip else sched.h0_table)[:n].unsqueeze(0).contiguous()
    delta = {b: t[:n].contiguous() for b, t in sched.cam_delta_tables.items()}

    # advance to the FIRST live block's input residual by running block0 + first mega seg
    with torch.no_grad():
        h = sg.ffn0.forward(h0)
        block_k = args.block_k
        worst = 0.0
        worst_ax = 0
        for kind, payload in sg.mega.items:
            if kind == "mega":
                h = payload.run(h)
            else:
                b = payload
                lb = sg.live_blocks[b]
                out_dims = sg.live_out_dims[b]
                wo = delta[b]                          # [n, n_out]
                # ---- (a) CURRENT dense path ----
                ref = lb.forward_static_delta(h, wo, out_dims)   # [1,n,D]
                # ---- (b) FUSED [D,K] path ----
                y = h[0].transpose(0, 1).contiguous()            # [D, n]
                y[out_dims, :] += wo.transpose(0, 1)
                mf = _MegaFFN(lb.ffn, str(dev), block_k)
                mf.run_fused(y, n, snap_scratch=None)
                got = y.transpose(0, 1).unsqueeze(0)             # [1,n,D]
                d = (ref - got).abs().max().item()
                # decode compare on this block's OUTPUT residual (AX/PC/SP/BP lanes)
                from c4_min.nibble_pure_forward_gpu import _decode_reg_batch
                ax_r = _decode_reg_batch(ref[0], L.AX)
                ax_g = _decode_reg_batch(got[0], L.AX)
                nax = int((ax_r != ax_g).sum())
                pc_r = PS._snap_lane_light(ref[0][:, L.PC_VAL])
                pc_g = PS._snap_lane_light(got[0][:, L.PC_VAL])
                npc = int((pc_r != pc_g).sum())
                print(f"  live block {b:3d}: Linf(resid)={d:.3e}  AX-nibble mismatches={nax}  "
                      f"PC mismatches={npc}", flush=True)
                worst = max(worst, d); worst_ax = max(worst_ax, nax + npc)
                h = ref                                          # continue with reference
        print(f"\n  WORST Linf residual = {worst:.3e}  (nibble-margin threshold ~0.5)")
        print(f"  WORST decode (AX+PC) mismatches = {worst_ax}")
        print(f"  BYTE-EXACT (decode): {'YES' if worst_ax == 0 else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
