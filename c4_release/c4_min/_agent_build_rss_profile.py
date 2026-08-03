"""RSS profiler for the doom-scale streaming sparse build.

Instruments c4_min.compact_alloc.build_compact_sparse_streaming at DOOM scale
(code_size≈3978, C4_PF_CFM=1) by wrapping every internal build stage with a
/proc/self/status VmRSS sample so we can see WHICH construction step balloons
the RSS toward the reported ~108 GB.

Run:
  PYTHONPATH=<c4_release> C4_PF_CFM=1 CUDA_VISIBLE_DEVICES=0,1 \
    python -m c4_min._agent_build_rss_profile
"""
from __future__ import annotations
import os, sys, time, gc

os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import warnings; warnings.filterwarnings("ignore")

import torch


def _rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    return -1.0


_PEAK = {"gb": 0.0, "where": ""}
_SAMPLES = []


def sample(tag: str):
    gb = _rss_gb()
    _SAMPLES.append((tag, gb, time.time()))
    if gb > _PEAK["gb"]:
        _PEAK["gb"] = gb
        _PEAK["where"] = tag
    print(f"[rss] {gb:8.3f} GB   {tag}", flush=True)
    return gb


def instrument():
    """Wrap the build sub-stages with RSS sampling."""
    import c4_min.compact_alloc as CA

    # Wrap the heavy internal helpers.
    def wrap(modname_attr, label):
        obj, name = modname_attr
        orig = getattr(obj, name)

        def wrapped(*a, **k):
            sample(f">> ENTER {label}")
            t0 = time.time()
            r = orig(*a, **k)
            sample(f"<< EXIT  {label} ({time.time()-t0:.1f}s)")
            return r
        setattr(obj, name, wrapped)
        return orig

    wrap((CA, "_build_stream_intermediate"), "_build_stream_intermediate")
    wrap((CA, "compute_liveness"), "compute_liveness")
    wrap((CA, "empirical_value_liveness"), "empirical_value_liveness")
    wrap((CA, "_default_probe_programs"), "_default_probe_programs")
    wrap((CA, "color_dims"), "color_dims")
    wrap((CA, "remap_layout"), "remap_layout")
    wrap((CA, "_rebuild_layout"), "_rebuild_layout")


def main():
    from c4_min.lib_neural import build_lib_model_streaming
    # replicate the doom code_size: doom.c compiles to ~3976 instrs → cs=len+2.
    cs = int(os.environ.get("CODE_SIZE", "3978"))
    print(f"[profile] code_size={cs} C4_PF_CFM={os.environ.get('C4_PF_CFM')}", flush=True)
    instrument()
    sample("START (before build)")
    t0 = time.time()
    sparse, L, stats = build_lib_model_streaming(
        code_size=cs, recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    wall = time.time() - t0
    sample("DONE (after build)")
    print(f"[profile] built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} "
          f"build_wall={wall:.1f}s", flush=True)
    print(f"[profile] PEAK RSS = {_PEAK['gb']:.3f} GB at: {_PEAK['where']}", flush=True)
    # nnz footprint of the finished sparse model
    try:
        tot = sparse.stats().total_nnz
        print(f"[profile] finished model total_nnz={tot}", flush=True)
    except Exception as e:
        print(f"[profile] stats err {e}")


if __name__ == "__main__":
    main()
