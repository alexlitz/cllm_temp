"""Profile the pure_neural test path.

Measures:
  - End-to-end run time for test_imm_then_exit
  - Number of model.forward() calls (= tokens generated)
  - Per-forward latency
  - Per-layer (block) time breakdown via PyTorch hooks
  - cProfile of Python-level hot spots
"""
import os
import sys
import time
import json
import cProfile
import pstats
import io
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode


def build_runner():
    runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    return runner


def make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def run_test(runner):
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    bc = make_bc([(Opcode.IMM, 5), Opcode.EXIT])
    _, result = runner.run(bc, b"", max_steps=30)
    return result


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    section("BUILD MODEL")
    t0 = time.time()
    runner = build_runner()
    build_time = time.time() - t0
    print(f"Build time: {build_time:.3f}s")
    print(f"Params: {sum(p.numel() for p in runner.model.parameters()):,}")
    print(f"d_model={runner.model.d_model}, n_blocks={len(runner.model.blocks)}, n_heads={runner.model.blocks[0].attn.num_heads}")

    # Warm up
    section("WARMUP (CUDA kernels, autotune)")
    t0 = time.time()
    result = run_test(runner)
    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_time = time.time() - t0
    print(f"Warmup run: {warmup_time:.3f}s, result={result}")

    # Instrument: count forward calls and per-call wall time
    section("END-TO-END + FORWARD COUNT")
    forward_call_times = []
    orig_forward = runner.model.forward

    def timed_forward(token_ids, kv_cache=None):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_forward(token_ids, kv_cache=kv_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_call_times.append((time.perf_counter() - t, token_ids.shape[1]))
        return out

    runner.model.forward = timed_forward
    t0 = time.time()
    result = run_test(runner)
    if device.type == "cuda":
        torch.cuda.synchronize()
    e2e_time = time.time() - t0
    runner.model.forward = orig_forward
    n_calls = len(forward_call_times)
    total_fwd = sum(d for d, _ in forward_call_times)
    print(f"End-to-end: {e2e_time:.3f}s, result={result}")
    print(f"Forward calls: {n_calls}")
    print(f"Total forward time: {total_fwd:.3f}s ({100*total_fwd/e2e_time:.1f}% of e2e)")
    print(f"Avg per-forward: {1000*total_fwd/n_calls:.3f} ms")
    print(f"Avg context len at end: {forward_call_times[-1][1]}")
    print(f"First call context: {forward_call_times[0][1]} tokens")
    print(f"Last call context:  {forward_call_times[-1][1]} tokens")

    # Histogram of per-forward times bucketed by context length
    buckets = defaultdict(list)
    for dur, ctx in forward_call_times:
        bucket = (ctx // 10) * 10
        buckets[bucket].append(dur)
    print("\nPer-forward latency by context length bucket:")
    print(f"  {'ctx_len':>8}  {'n':>5}  {'avg_ms':>8}  {'tot_ms':>8}")
    for bucket in sorted(buckets):
        durs = buckets[bucket]
        avg_ms = 1000 * sum(durs) / len(durs)
        tot_ms = 1000 * sum(durs)
        print(f"  {bucket:>8}  {len(durs):>5}  {avg_ms:>8.3f}  {tot_ms:>8.3f}")

    # Per-block forward times via hooks
    section("PER-BLOCK BREAKDOWN (forward hooks)")
    block_times = defaultdict(float)
    block_counts = defaultdict(int)
    embed_times = []
    head_times = []

    handles = []

    def make_pre_hook(name):
        def pre_hook(module, inputs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            module._t_start = time.perf_counter()
        return pre_hook

    def make_post_hook(name):
        def post_hook(module, inputs, output):
            if device.type == "cuda":
                torch.cuda.synchronize()
            dur = time.perf_counter() - module._t_start
            block_times[name] += dur
            block_counts[name] += 1
        return post_hook

    # Hook the embed, each block, and the head
    handles.append(runner.model.embed.register_forward_pre_hook(make_pre_hook("embed")))
    handles.append(runner.model.embed.register_forward_hook(make_post_hook("embed")))
    for i, block in enumerate(runner.model.blocks):
        name = f"block_{i:02d}"
        handles.append(block.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(block.register_forward_hook(make_post_hook(name)))
        # Also break down into attn / ffn
        a_name = f"  {name}.attn"
        handles.append(block.attn.register_forward_pre_hook(make_pre_hook(a_name)))
        handles.append(block.attn.register_forward_hook(make_post_hook(a_name)))
        f_name = f"  {name}.ffn"
        handles.append(block.ffn.register_forward_pre_hook(make_pre_hook(f_name)))
        handles.append(block.ffn.register_forward_hook(make_post_hook(f_name)))
    handles.append(runner.model.head.register_forward_pre_hook(make_pre_hook("head")))
    handles.append(runner.model.head.register_forward_hook(make_post_hook("head")))

    t0 = time.time()
    result = run_test(runner)
    if device.type == "cuda":
        torch.cuda.synchronize()
    hook_e2e = time.time() - t0
    print(f"With hooks: {hook_e2e:.3f}s, result={result}")

    for h in handles:
        h.remove()

    # Sort and print
    items = sorted(block_times.items(), key=lambda x: -x[1])
    total_hooked = sum(t for n, t in items if not n.startswith("  "))
    print(f"\nTotal hooked top-level (embed + blocks + head): {total_hooked:.3f}s")
    print(f"\n{'name':>22}  {'n_calls':>8}  {'total_ms':>10}  {'avg_ms':>8}  {'%fwd':>6}")
    for name, total in items:
        n = block_counts[name]
        avg_ms = 1000 * total / n
        tot_ms = 1000 * total
        pct = 100 * total / total_hooked if not name.startswith("  ") else 100 * total / total_hooked
        print(f"  {name:>20}  {n:>8}  {tot_ms:>10.2f}  {avg_ms:>8.3f}  {pct:>5.1f}%")

    # cProfile to identify Python-level hot spots
    section("cProfile (Python-level)")
    pr = cProfile.Profile()
    pr.enable()
    result = run_test(runner)
    pr.disable()
    if device.type == "cuda":
        torch.cuda.synchronize()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    print(s.getvalue())

    # tottime (self-time) version
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(35)
    print("\n--- by tottime ---\n")
    print(s2.getvalue())

    # PyTorch profiler for op-level breakdown
    section("torch.profiler (top operators)")
    from torch.profiler import profile, ProfilerActivity, record_function

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof:
        with record_function("e2e"):
            result = run_test(runner)
        if device.type == "cuda":
            torch.cuda.synchronize()

    if device.type == "cuda":
        sort_by = "self_cuda_time_total"
    else:
        sort_by = "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=30))

    # Save data
    out = {
        "device": str(device),
        "n_params": sum(p.numel() for p in runner.model.parameters()),
        "n_blocks": len(runner.model.blocks),
        "build_time_s": build_time,
        "warmup_e2e_s": warmup_time,
        "instrumented_e2e_s": e2e_time,
        "n_forward_calls": n_calls,
        "total_forward_s": total_fwd,
        "avg_per_forward_ms": 1000 * total_fwd / n_calls,
        "first_ctx_len": forward_call_times[0][1],
        "last_ctx_len": forward_call_times[-1][1],
        "block_times_ms": {n: 1000 * t for n, t in block_times.items()},
        "block_counts": dict(block_counts),
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "docs", "_profile_data.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON to {out_path}")


if __name__ == "__main__":
    main()
