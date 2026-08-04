#!/usr/bin/env python3
"""Quick check: after install_dead_block_fusion, how many dead segments does
GraphedFusedForward see, and do they capture into a CUDA graph at K-row shape?"""
import os, sys
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch


def main():
    dev = os.environ.get("PROBE_DEV", "cuda:0")
    from .lib_neural import build_lib_model_streaming
    from .local_attention import install_local_attention
    from .live_head_attention import (install_live_head_attention,
                                       install_dead_block_fusion,
                                       live_head_attention_stats)
    model, L, _ = build_lib_model_streaming(code_size=192, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(dev)
    install_local_attention(model, window=64, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    st = live_head_attention_stats(model)
    print(f"[seg] n_blocks={st['n_blocks']} live_attn_blocks={st['live_attention_blocks']} "
          f"dead_attn_blocks={st['dead_attention_blocks']}", flush=True)
    print(f"[seg] live head blocks: {sorted(st['per_block_live_heads'].keys())}", flush=True)
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)
    from .graphed_fused_forward import _dead_segments
    live, segs = _dead_segments(model)
    seg_lens = [e - s for s, e in segs]
    print(f"[seg] AFTER dead-fusion: live blocks {live}", flush=True)
    print(f"[seg] dead segments: {segs}", flush=True)
    print(f"[seg] segment lens: {seg_lens}  (total dead {sum(seg_lens)})", flush=True)

    # try to capture ONE dead segment at a K-row shape (frozen-skip => K query rows)
    if segs:
        from .graphed_fused_forward import GraphedFusedForward
        gf = GraphedFusedForward(model, dev)
        S = 256  # K query rows past the frozen-skip cut
        qpos = torch.arange(S, device=dev)
        try:
            seg = segs[0]
            g, gin, gout = gf._capture_segment(seg, S, qpos)
            print(f"[seg] CAPTURE OK for seg {seg} at S={S}: out shape {tuple(gout.shape)}", flush=True)
        except Exception as e:
            print(f"[seg] CAPTURE FAILED for seg {segs[0]} at S={S}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
