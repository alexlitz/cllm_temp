"""CUDA-graphs prototype benchmark.

Exploratory benchmark comparing per-token forward-pass latency under three modes:

    1. baseline  - vanilla eager ``model.forward(token_ids)`` (no compile)
    2. compile   - ``torch.compile(mode="reduce-overhead")`` (CUDA-graph capture
                   per submodule, plus dispatcher overhead per call)
    3. cuda_graph - manual single-step CUDA-graph capture-and-replay around a
                    fixed-shape forward, with pre-allocated input/output buffers

The goal is to see whether manual CUDA-graph capture (option 3) beats
``torch.compile(reduce-overhead)`` (option 2) for the per-step decode case.
If it doesn't, the multi-step capture work isn't worth pursuing in isolation
of speculative decoding.

Run:
    timeout 600 python -m c4_release.neural_vm.cuda_graph_bench
"""

from __future__ import annotations

import sys
import time
from typing import Callable, List

import torch


def _build_model():
    from c4_release.neural_vm.unified_compiler.full_vm_compiler import compile_full_vm

    model, _ = compile_full_vm()
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def _build_token_input(seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Construct a fixed-shape token-id tensor for benchmarking.

    Values don't matter for timing — we just need a stable [1, seq_len] int64
    tensor on the right device.
    """
    torch.manual_seed(0)
    return torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long, device=device)


def _time_fn(fn: Callable[[], None], iters: int, warmup: int) -> List[float]:
    """Run ``fn`` ``warmup + iters`` times, return per-call ms for the last
    ``iters`` runs.

    Uses ``torch.cuda.synchronize()`` per call to get a host-observable timing,
    so the numbers include any CPU↔GPU sync overhead caller code would incur.
    This is the relevant metric: the VM decode loop has Python in the loop and
    must read the argmax back, so we can't measure raw GPU time only.
    """
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _time_fn_events(fn: Callable[[], None], iters: int, warmup: int) -> List[float]:
    """Time ``fn`` using CUDA events — measures GPU-side time only.

    Useful when the host-side ``time.perf_counter()`` is noisy due to
    contention from other GPU users.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        end_ev.synchronize()
        times.append(start_ev.elapsed_time(end_ev))
    return times


def bench_baseline(model, token_ids):
    """Vanilla eager forward."""

    def step():
        with torch.no_grad():
            logits = model.forward(token_ids)
            # Match what AutoregressiveVMRunner._generate_next_cached does: take
            # the last-position argmax. We don't transfer to CPU here because
            # `_time_fn` already synchronizes; the .item() in the real loop is
            # measured separately by the per-step microbenchmark.
            _ = logits[0, -1, :].argmax(-1)

    return step


def bench_compile(model, token_ids):
    """``torch.compile(reduce-overhead)`` over the full forward.

    Uses the same Dynamo config tweaks as ``AutoregressiveVMRunner.__init__``
    (``recompile_limit=128``, ``force_parameter_static_shapes=False``) so the
    benchmark mirrors production.
    """
    import torch._dynamo as _dynamo

    _dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, 128)
    _dynamo.config.force_parameter_static_shapes = False
    compiled = torch.compile(model, mode="reduce-overhead")

    def step():
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            logits = compiled(token_ids)
            _ = logits[0, -1, :].argmax(-1)

    return step


def bench_cuda_graph_single(model, token_ids):
    """Manual single-step CUDA-graph capture and replay.

    Captures one forward pass into a CUDA graph, then replays it on subsequent
    calls. Input is a pre-allocated buffer that the caller writes new token IDs
    into in-place (here: just left identical because timing is what matters).

    Caveats baked in here:
      * Fixed input shape. A different seq_len would need a new capture.
      * The embedding's prefix-cache fast path is incompatible with stream
        capture (it does CPU↔GPU sync via ``.to(cpu)`` + ``torch.equal``).
        We monkey-patch it to always-miss during the capture window so the
        pure-tensor un-cached path is the one captured. After capture the
        patch is left in place because replay re-runs the captured tensor
        kernels (not Python).
      * Captured graph closes over any Python-side state in the model; we
        run with no kv_cache to avoid aliasing.
      * First-call cost (capture) is excluded from the timing band but
        reported once.
    """
    # Bypass several Python-side fast paths that contain CPU/GPU sync points
    # (``.item()``, ``.to(cpu)``, ``torch.equal``, ``if t.any():``). These
    # raise ``cudaErrorStreamCaptureUnsupported`` mid-capture.
    #
    # 1) Embedding prefix-cache and divmod-skip: gated by
    #    ``torch.compiler.is_compiling()``; we hijack that signal so the
    #    same compile-friendly fallback is taken under capture.
    # 2) Long-division ALU's ``if b_is_zero.any():`` branch: not gated.
    #    We replace it with the divide-by-zero-tolerant tail (which the
    #    real test never exercises) so capture doesn't sync.
    embed = model.embed
    embed.reset_prefix_cache()
    import torch.compiler as _tc
    original_is_compiling = _tc.is_compiling
    _tc.is_compiling = lambda: True

    # Replace LongDivisionModule's ``if b_is_zero.any():`` branch with a
    # mask-always-apply form (algebraically equivalent when ``b!=0`` since
    # ``mask`` is zero everywhere). Tests never pass ``b=0`` so this is a
    # no-op behaviorally, but it removes the .any()-driven CPU sync that
    # blocks capture.
    try:
        from .alu.ops.divmod_longdiv import LongDivisionModule as _LDM

        if not getattr(_LDM, "_cuda_graph_patched", False):
            _orig_ld_forward = _LDM.forward

            def _patched_ld_forward(self, x):
                ge = self.ge
                N = ge.NUM_POSITIONS
                B = x.shape[0]
                device = x.device
                orig_dtype = x.dtype
                opcode_w = x[:, 0, ge.OP_START + self.opcode]
                a = x[:, :N, ge.NIB_A].float()
                b = x[:, :N, ge.NIB_B].float()
                a = torch.round(a.clamp(0, 15))
                b = torch.round(b.clamp(0, 15))
                b9 = torch.zeros(B, 9, dtype=torch.float32, device=device)
                b9[:, :8] = b
                b_is_zero = (b9.sum(dim=-1) < 0.5).to(torch.float32)
                partial = torch.zeros(B, 9, dtype=torch.float32, device=device)
                q_out = torch.zeros(B, N, dtype=torch.float32, device=device)
                for i in range(N - 1, -1, -1):
                    partial = self._shift_left_one_nibble(partial, a[:, i])
                    q_count = torch.zeros(B, dtype=torch.float32, device=device)
                    for k in range(1, 16):
                        k_t = torch.full((B,), float(k), dtype=torch.float32, device=device)
                        qb = self._trial_multiply(k_t, b9)
                        le = self._compare_le(qb, partial)
                        q_count = q_count + le
                    q_i = q_count
                    q_out[:, i] = q_i
                    qb_final = self._trial_multiply(q_i, b9)
                    partial = self._subtract(partial, qb_final)
                rem = partial[:, :N]
                # b_is_zero applied unconditionally (zero everywhere in tests).
                mask = b_is_zero[:, None]
                q_out = q_out * (1 - mask) + 15.0 * mask
                rem = rem * (1 - mask) + a * mask
                delta = torch.zeros_like(x)
                old_q = x[:, :N, ge.SLOT_QUOTIENT]
                old_r = x[:, :N, ge.SLOT_REMAINDER]
                opc = opcode_w[:, None].to(orig_dtype)
                delta[:, :N, ge.SLOT_QUOTIENT] = (-old_q + q_out.to(orig_dtype)) * opc
                delta[:, :N, ge.SLOT_REMAINDER] = (-old_r + rem.to(orig_dtype)) * opc
                return x + delta

            _LDM.forward = _patched_ld_forward
            _LDM._cuda_graph_patched = True
            _LDM._cuda_graph_orig_forward = _orig_ld_forward
    except ImportError:
        pass

    # Pre-allocate input/output buffers. The captured graph reads from
    # static_in and writes into static_logits / static_argmax.
    static_in = token_ids.clone()

    # Warmup stream: required before capture so kernels are JIT'd.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                logits = model.forward(static_in)
                _ = logits[0, -1, :].argmax(-1)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()

    capture_t0 = time.perf_counter()
    with torch.cuda.graph(g):
        with torch.no_grad():
            out = model.forward(static_in)
            # Use the trailing argmax so the captured graph reads the position
            # the caller cares about; keep the result tensor alive by writing
            # to a per-graph static buffer (we'd hand it back to the caller in
            # a real integration).
            argmax = out[0, -1, :].argmax(-1)
            # `argmax` aliases the captured tensor; we keep it referenced so
            # ``g`` retains the allocation across replays.
    capture_ms = (time.perf_counter() - capture_t0) * 1000.0

    # Restore ``is_compiling``. Captured kernels are already baked into the
    # graph; future eager forwards on this model should see the normal path.
    _tc.is_compiling = original_is_compiling

    def step():
        # In a real loop, the caller would write the new token IDs into
        # static_in BEFORE calling replay; for the timing benchmark we keep
        # the buffer identical so the replay path is exactly what'd be hot.
        g.replay()

    step._capture_ms = capture_ms
    return step


def main():
    """Benchmark driver.

    Modes are gated via the ``CUDA_GRAPH_BENCH_MODES`` env var so the script
    can run in a bounded time budget — ``torch.compile(reduce-overhead)`` on
    this ~30-block model takes minutes to capture all the distinct FFN
    shapes (see ``AutoregressiveVMRunner.__init__`` notes on recompile_limit).

    Default modes: ``baseline,cuda_graph`` — both finish in <60s after the
    ~100s model build. Add ``compile`` to also run torch.compile — expect
    multi-minute capture for the full set of distinct block shapes.
    """
    import os

    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        sys.exit(0)

    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    print("Building model...", flush=True)
    t0 = time.perf_counter()
    model = _build_model()
    print(f"  built in {time.perf_counter() - t0:.2f}s", flush=True)

    # Use a representative seq_len for test_imm_exit's pure_neural path:
    # context length grows from prefix (~70 tokens) over a few steps. Cap at
    # 128 which is well within the model's max_seq_len (1024) but big enough
    # to be a realistic forward-pass size. Override via env to study scaling.
    seq_lens = [int(s) for s in os.environ.get("CUDA_GRAPH_BENCH_SEQLENS", "128").split(",")]
    vocab_size = model.vocab_size

    warmup = 5
    iters = 30

    modes = os.environ.get("CUDA_GRAPH_BENCH_MODES", "baseline,cuda_graph").split(",")
    print(f"\nBenchmark config: seq_lens={seq_lens}, iters={iters}, warmup={warmup}, modes={modes}")
    print()

    for seq_len in seq_lens:
        token_ids = _build_token_input(seq_len, vocab_size, device)
        print(f"=== seq_len={seq_len} ===")

        if "baseline" in modes:
            print("Mode: baseline (eager forward)", flush=True)
            step = bench_baseline(model, token_ids)
            times = _time_fn(step, iters=iters, warmup=warmup)
            ev_times = _time_fn_events(step, iters=iters, warmup=2)
            print(f"  host: mean={sum(times)/len(times):.3f}ms  min={min(times):.3f}ms  max={max(times):.3f}ms",
                  flush=True)
            print(f"  gpu : mean={sum(ev_times)/len(ev_times):.3f}ms  min={min(ev_times):.3f}ms  max={max(ev_times):.3f}ms",
                  flush=True)

        if "cuda_graph" in modes:
            print("Mode: manual single-step CUDA graph", flush=True)
            step = bench_cuda_graph_single(model, token_ids)
            print(f"  one-time capture cost: {step._capture_ms:.2f}ms", flush=True)
            times = _time_fn(step, iters=iters, warmup=warmup)
            ev_times = _time_fn_events(step, iters=iters, warmup=2)
            print(f"  host: mean={sum(times)/len(times):.3f}ms  min={min(times):.3f}ms  max={max(times):.3f}ms",
                  flush=True)
            print(f"  gpu : mean={sum(ev_times)/len(ev_times):.3f}ms  min={min(ev_times):.3f}ms  max={max(ev_times):.3f}ms",
                  flush=True)

        if "compile" in modes:
            print("Mode: compile(reduce-overhead)", flush=True)
            step = bench_compile(model, token_ids)
            times = _time_fn(step, iters=iters, warmup=warmup)
            ev_times = _time_fn_events(step, iters=iters, warmup=2)
            print(f"  host: mean={sum(times)/len(times):.3f}ms  min={min(times):.3f}ms  max={max(times):.3f}ms",
                  flush=True)
            print(f"  gpu : mean={sum(ev_times)/len(ev_times):.3f}ms  min={min(ev_times):.3f}ms  max={max(ev_times):.3f}ms",
                  flush=True)
        print()

    print("\nDone.")


if __name__ == "__main__":
    main()
