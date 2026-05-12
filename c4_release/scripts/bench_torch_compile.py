"""Benchmark torch.compile with various recipes against the c4 VM model.

Times warmup + steady-state forward latency at varying seq_len, simulating the
shape pattern that the AutoregressiveVMRunner produces during decoding (initial
context ~50 tokens, growing one token per forward call).

Recipes:
    A. dynamic=True + mark_dynamic
    B. static padding to power-of-2 buckets
    C. raw CUDA graphs (capture at fixed shape, replay)
    none: baseline uncompiled forward

Usage:
    python -m c4_release.scripts.bench_torch_compile <recipe> [--max-seq <N>] [--steps <N>]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

# Path setup: support running from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token


SHAPE_BUCKETS = (64, 128, 256, 512, 1024)


def _empty_cuda_cache():
    """Free up CUDA cache to reduce footprint."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _build_runner():
    """Build a pure_neural runner (no compile yet)."""
    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        compile_mode="none",
    )
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    return runner


def _build_real_context(runner):
    """Build the real initial context used by the runner for an IMM-EXIT
    program. Returns a list of int tokens.
    """
    bytecode = []
    for op in [(Opcode.IMM, 42), Opcode.EXIT]:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    ctx = runner._build_context(bytecode, b"", [], "")
    return list(ctx)


def _to_device_tensor(ids, device):
    return torch.tensor([ids], dtype=torch.long, device=device)


def _pad_to_bucket(token_ids, pad_value=0):
    S = token_ids.shape[1]
    bucket = next((b for b in SHAPE_BUCKETS if b >= S), SHAPE_BUCKETS[-1])
    pad = bucket - S
    if pad > 0:
        token_ids = F.pad(token_ids, (0, pad), value=pad_value)
    return token_ids, bucket


def bench_no_compile(model, init_ctx, device, n_steps=10):
    seq_lens = []
    walls = []
    cur = list(init_ctx)
    for step in range(n_steps):
        tok = _to_device_tensor(cur, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            logits = model(tok)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        seq_lens.append(tok.shape[1])
        walls.append(dt)
        next_tok = int(logits[0, -1, :].argmax(-1).item())
        cur.append(next_tok)
    return seq_lens, walls


def bench_recipe_a(model, init_ctx, device, n_steps=10, compile_mode="reduce-overhead"):
    """Recipe A: dynamic=True + mark_dynamic."""
    import torch._dynamo as _dynamo
    _dynamo.config.recompile_limit = max(getattr(_dynamo.config, "recompile_limit", 8), 128)
    _dynamo.config.force_parameter_static_shapes = False
    compiled = torch.compile(model, mode=compile_mode, dynamic=True)

    seq_lens = []
    walls = []
    cur = list(init_ctx)
    for step in range(n_steps):
        tok = _to_device_tensor(cur, device)
        if step == 0:
            # Mark the seq dim as dynamic with explicit bounds. Per
            # torch.compile docs: only effective on FIRST forward to set
            # symbolic shape guards.
            try:
                _dynamo.mark_dynamic(tok, 1, min=1, max=2048)
            except Exception as e:
                print(f"  (mark_dynamic failed: {e!r})")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        if compile_mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            logits = compiled(tok)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        seq_lens.append(tok.shape[1])
        walls.append(dt)
        next_tok = int(logits[0, -1, :].argmax(-1).item())
        cur.append(next_tok)
    return seq_lens, walls


def bench_recipe_b(model, init_ctx, device, n_steps=10, compile_mode="reduce-overhead"):
    """Recipe B: pad inputs to power-of-2 buckets, compile per bucket."""
    import torch._dynamo as _dynamo
    _dynamo.config.recompile_limit = max(getattr(_dynamo.config, "recompile_limit", 8), 128)
    _dynamo.config.force_parameter_static_shapes = False
    # One compiled object handles all buckets; per-bucket specialization will
    # happen automatically with static shapes (dynamic=False).
    compiled = torch.compile(model, mode=compile_mode, dynamic=False)

    seq_lens = []
    walls = []
    cur = list(init_ctx)
    for step in range(n_steps):
        tok = _to_device_tensor(cur, device)
        padded, bucket = _pad_to_bucket(tok)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        if compile_mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            logits = compiled(padded)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        # The real last position is tok.shape[1] - 1, not bucket - 1.
        real_last_pos = tok.shape[1] - 1
        seq_lens.append(f"{tok.shape[1]}->b{bucket}")
        walls.append(dt)
        next_tok = int(logits[0, real_last_pos, :].argmax(-1).item())
        cur.append(next_tok)
    return seq_lens, walls


def bench_recipe_c(model, init_ctx, device, n_steps=10):
    """Recipe C: raw CUDA graph capture-and-replay at a fixed shape (bucket).

    Allocates one static input buffer per bucket (sized to bucket length).
    Captures one CUDA graph per bucket on first encounter.
    Replays the graph on subsequent calls by copying the new token ids into
    the static buffer and running ``graph.replay()``.

    The replay path avoids ANY Python / Dynamo / Inductor overhead — just
    one ``cudaGraphLaunch``. Per-shape capture is a one-time cost.
    """
    if device.type != "cuda":
        raise RuntimeError("Recipe C requires CUDA")

    # Disable the embedding prefix cache: it does a host-device sync
    # (`.to(cached.device)`) on a Python state-mutating path that the
    # CUDA graph capture forbids. We:
    #   1) clear any pre-populated cache state, and
    #   2) monkey-patch ``_populate_prefix_cache`` to a no-op so the cache
    #      stays empty across warmup forwards.
    if hasattr(model, "embed"):
        model.embed._prefix_cache_token_ids = None
        model.embed._prefix_cache_delta = None
        model.embed._prefix_cache_len = 0
        model.embed._populate_prefix_cache = lambda *a, **kw: None

    static_inputs = {}   # bucket -> static input tensor (long, [1, bucket])
    static_outputs = {}  # bucket -> static output tensor (logits)
    graphs = {}          # bucket -> torch.cuda.CUDAGraph

    seq_lens = []
    walls = []
    cur = list(init_ctx)
    for step in range(n_steps):
        tok = _to_device_tensor(cur, device)
        S = tok.shape[1]
        bucket = next((b for b in SHAPE_BUCKETS if b >= S), SHAPE_BUCKETS[-1])

        if bucket not in graphs:
            # Capture path: warmup the model at the bucket shape with a
            # padded input (using same pad value 0). Several warmup iters
            # are needed before capture so all kernels are compiled/cached.
            static_in = torch.zeros((1, bucket), dtype=torch.long, device=device)
            # Initial fill from the real input
            static_in[:, :S].copy_(tok)
            torch.cuda.synchronize()
            t_cap = time.time()
            # Warmup: run forward a few times so all lazy CUDA inits complete
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.no_grad():
                    for _ in range(3):
                        out_warm = model(static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            # Capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    static_out = model(static_in)
            torch.cuda.synchronize()
            capture_wall = time.time() - t_cap
            print(f"  (bucket {bucket} captured in {capture_wall:.2f}s)")
            static_inputs[bucket] = static_in
            static_outputs[bucket] = static_out
            graphs[bucket] = g

        # Replay: copy real input into static buffer, replay graph, read out
        static_in = static_inputs[bucket]
        static_in.zero_()
        static_in[:, :S].copy_(tok)

        torch.cuda.synchronize()
        t0 = time.time()
        graphs[bucket].replay()
        torch.cuda.synchronize()
        dt = time.time() - t0

        logits = static_outputs[bucket]
        real_last_pos = S - 1
        next_tok = int(logits[0, real_last_pos, :].argmax(-1).item())

        seq_lens.append(f"{S}->b{bucket}")
        walls.append(dt)
        cur.append(next_tok)

    return seq_lens, walls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", choices=["none", "A", "B", "C"])
    parser.add_argument("--steps", type=int, default=10,
                        help="number of forward calls to time")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["reduce-overhead", "max-autotune", "default"])
    parser.add_argument("--budget", type=float, default=300.0,
                        help="seconds budget for total benchmark wall (warmup+steady)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Recipe: {args.recipe}")
    print(f"Compile mode: {args.compile_mode}")
    print(f"Steps: {args.steps}")
    print(f"Budget: {args.budget}s")

    _empty_cuda_cache()
    t0 = time.time()
    runner = _build_runner()
    init_ctx = _build_real_context(runner)
    print(f"Build runner: {time.time() - t0:.2f}s, init context len: {len(init_ctx)}")

    model = runner.model.eval()
    n_blocks = len(model.blocks)
    print(f"Model: d_model={model.d_model}, n_blocks={n_blocks}")

    t_start = time.time()

    def hit_budget():
        return time.time() - t_start > args.budget

    if args.recipe == "none":
        seq_lens, walls = bench_no_compile(
            model, init_ctx, device, n_steps=args.steps
        )
    elif args.recipe == "A":
        seq_lens, walls = bench_recipe_a(
            model, init_ctx, device, n_steps=args.steps,
            compile_mode=args.compile_mode,
        )
    elif args.recipe == "B":
        seq_lens, walls = bench_recipe_b(
            model, init_ctx, device, n_steps=args.steps,
            compile_mode=args.compile_mode,
        )
    elif args.recipe == "C":
        seq_lens, walls = bench_recipe_c(
            model, init_ctx, device, n_steps=args.steps,
        )
    else:
        print(f"Recipe {args.recipe} not yet implemented")
        sys.exit(2)

    total_wall = time.time() - t_start
    print()
    print(f"Total bench wall: {total_wall:.2f}s")
    print()
    print(f"{'step':>4} {'seq_len':>12} {'wall_s':>10}")
    for i, (sl, w) in enumerate(zip(seq_lens, walls)):
        print(f"{i:>4} {str(sl):>12} {w:>10.4f}")

    # Warmup = first call. Steady-state = mean of last min(3, N-1) calls
    if len(walls) > 1:
        warmup_wall = walls[0]
        tail_n = min(3, len(walls) - 1)
        steady = sum(walls[-tail_n:]) / tail_n
        print()
        print(f"Warmup (step 0):    {warmup_wall:.4f}s")
        print(f"Steady-state mean ({tail_n} tail): {steady:.4f}s")
        if walls[1:]:
            print(f"After-warmup walls min={min(walls[1:]):.4f}s max={max(walls[1:]):.4f}s")


if __name__ == "__main__":
    main()
