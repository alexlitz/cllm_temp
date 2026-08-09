# CLEVER MIN-FLOP OPTIMIZED — the fused-persistent min-param step, measured

**Question.** Prior work measured only the RAW throughput of the min-param
(min-FLOP) clever c4 VM step (`0.259 µs/lane-step` at K=8192, `0.11 µs` at K=262144;
unoptimized eager PyTorch). This doc applies the FULL execution optimizations
— **fused-persistent** (whole 15-layer chain as one CUDA-graph launch, 64-dim
residual kept on-chip layer-to-layer), **one-token** (T=1 query-token path), and
the **dim-64 framing floor** — and re-measures whether the optimized per-step
**BEATS the wide VM's composed `0.2793 µs/step`** and **clears 35-fps gameplay
WITHOUT the render structural folds**.

- **Build script:** [`examples/clever_minflop_optimized.py`](../examples/clever_minflop_optimized.py)
- **Model:** the min-param clever VM (`CompactStepModel`, `direct_attn=True`) —
  `d_model=64`, **15 layers** (radix-4096 ADD-step depth), radix 4096, compact
  **direct-floor** scoring (FFN band 32), fp32 limb-MUL, O(1) direct-CAM.
  473 nonzero looped / 2,559 unrolled params; byte-exact op-cells (L-inf=0).
- **Measured on:** idle NVIDIA RTX A5000 (both cards free), fp32, warmup +
  `torch.cuda.synchronize`. 2-GPU is a REAL two-device concurrent run.
- **Golden `174ece66` untouched** — example + doc only, no build-path file.

---

## 1. Optimized per-step (MEASURED, fp32, 1 GPU)

`CompactStepModel` forward is captured as ONE CUDA graph (the fused-persistent form:
no per-layer kernel launch, no inter-layer HBM/L2 round-trip of the (K,1,64)
residual). The captured output is **byte-identical to eager (L-inf = 0)** at every K
— fusing the launch changes nothing about the arithmetic.

| K (batch) | eager µs/step | eager ns/lane | **FUSED µs/step** | **FUSED ns/lane** | fused speedup | L-inf | vs wide 0.2793 µs |
|----------:|--------------:|--------------:|------------------:|------------------:|--------------:|------:|:------------------|
| 1         | 1772.7        | 1 772 675     | 188.3             | 188 276           | **9.42×**     | 0     | slower (latency)  |
| 512       | 1883.9        | 3 679.6       | 355.3             | 693.9             | **5.30×**     | 0     | slower            |
| **8192**  | 1853.7        | 226.3         | **850.8**         | **103.9**         | **2.18×**     | 0     | **BEATS (2.69×)** |
| 65536     | 7659.3        | 116.9         | 7502.7            | 114.5             | 1.02×         | 0     | **BEATS**         |
| 262144    | 28870.0       | 110.1         | 28805.8           | 109.9             | 1.00×         | 0     | **BEATS**         |

**Headline:** the optimized min-flop step is **`0.1039 µs/lane-step`** at the
spec-verify batch (K≈8192), which **BEATS the wide VM's composed `0.2793 µs/step`
by 2.69×**. It beats the wide VM at every K ≥ 8192.

---

## 2. dim-64 framing floor (MEASURED, per-lane) vs wide VM's 0.161 µs

The wide VM's binding per-token framing floor is `0.161 µs` (block-0 ingest attention
`0.082` + CAM scatter `0.053` + decode `0.026`). At dim 64 the same three terms are:

| term            | wide (dim 1440) | **min-flop (dim 64)** |
|-----------------|----------------:|----------------------:|
| ingest attention| 0.082 µs        | **0.00186 µs (1.86 ns)** |
| CAM read        | 0.053 µs        | **0.00460 µs (4.60 ns)** |
| decode          | 0.026 µs        | **0.00015 µs (0.15 ns)** |
| **framing floor**| **0.161 µs**   | **0.00660 µs (6.60 ns)** |

The dim-64 framing floor is **24.4× cheaper** than the wide VM's — as expected from
the ~64/1440 ≈ 22× width ratio (the CAM read is the largest term because it is a
gather + nibble unpack, not a width-scaled GEMM). **The min-flop framing floor
(6.6 ns) is ~16× below the min-flop step's own 103.9 ns — framing is NOT the binding
term here; the 15-layer FFN datapath is.**

---

## 3. Occupancy-bound vs work-bound

The `fused speedup vs eager` column is the tell:

- **K = 1 → 512 → 8192:** graph fusion gives **9.4× → 5.3× → 2.18×**. The 15 tiny
  dim-64 layers are **launch/occupancy-bound**: the per-layer GEMMs are far too small
  to fill the A5000's SMs / tensor cores, so per-step time is dominated by
  kernel-launch + scheduling overhead, which the CUDA graph erases.
- **K ≥ 65536:** speedup collapses to **1.02× → 1.00×**. The GEMMs finally saturate
  the SMs — the step is now **work-bound**, and there is no launch overhead left to
  fuse away.

**So at the spec-verify batch (K≈8192) the single-stream min-flop step IS
occupancy-bound** (dim 64 under-fills the tensor cores). The fused-persistent CUDA
graph is exactly the optimization that recovers this: it takes the eager `226 ns/lane`
down to `104 ns/lane` (2.18×) by removing the launch overhead the occupancy-bound
regime pays. It does NOT change the fact that the tiny GEMMs under-fill the cores —
that only resolves by going work-bound at K ≳ 65536 (or by running two streams / two
GPUs to fill the idle SMs).

---

## 4. Gameplay fps projection (PROJECTED from the measured per-step)

fps = optimized lane-steps/s ÷ frame-step-count, on each of the four Doom frame folds.
1-GPU uses the throughput-optimal 1-GPU fused config (K=8192, `9.63M lane-steps/s`);
2-GPU is the REAL measured two-device throughput (`17.99M lane-steps/s`, eager
concurrent at the work-bound K=262144).

| frame fold        | steps/frame | **1-GPU fps** | **2-GPU fps** | clears 35? |
|-------------------|------------:|--------------:|--------------:|:-----------|
| **render_reduced**| 358 058     | **26.89**     | **50.23**     | 1-GPU NO / 2-GPU **YES** |
| current_fold      | 1 151 277   | 8.36          | 15.62         | NO / NO    |
| folded            | 111 102     | 86.66         | 161.90        | **YES** / **YES** |
| raw               | 8 068 960   | 1.19          | 2.23          | NO / NO    |

---

## 5. Verdict (blunt)

**(a) How fast vs the wide VM.** The optimized min-flop step is **`0.1039 µs/lane-step`
(spec batch K≈8192), which BEATS the wide VM's composed `0.2793 µs/step` by 2.69×.**
It wins at every K ≥ 8192. The dim-64 framing floor (`6.6 ns`) is 24.4× below the wide
VM's `0.161 µs`, and is not even the binding term for the min-flop step (the 15-layer
FFN datapath is).

**(b) Does it clear 35 fps on the render-reduced frame (358,058) WITHOUT the structural
folds?** **On 1 GPU, NO — 26.89 fps (misses 35 by ~1.30×).** **On 2 GPUs, YES —
50.23 fps (measured concurrent).** So a single optimized min-flop stream does *not*
clear 35 on the render-reduced frame without any structural fold; it takes the second
GPU (or a structural fold — the `folded` 111k frame clears 35 on 1 GPU at 86.66 fps).

**(c) Occupancy-bound single-stream, or does the optimization fill it?** **Single-stream
at the spec batch it is occupancy-bound** — dim 64 under-fills the tensor cores, which
is exactly why the CUDA-graph fusion buys 2.18× at K=8192 (it erases the launch
overhead the occupancy-bound regime pays) but only 1.0× at K≥65536 (work-bound). The
fused-persistent optimization *recovers the launch overhead* but does **not** fill the
cores; the cores fill only by going to a large work-bound batch or by adding a second
stream/GPU. This is why the 2-GPU run scales cleanly (~1.9×): the second card absorbs
the SMs the single dim-64 stream leaves idle.

**(d) Honest caveat.** These are the OPTIMIZED per-step / framing / fps numbers for the
min-flop clever *step model* (`CompactStepModel` + direct-CAM), byte-exact op-cells.
**It does NOT run the Doom program yet** — driving actual Doom needs the neural runtime
(the fetch-decode-execute sequencer over the real trace, the KV heap, the render loop).
The per-step is the correct unit for the wide-VM comparison and the fps projection, but
"clears 35 fps" here is a *throughput projection at the byte-exact per-step*, not a
measured end-to-end Doom frame.

---

## Reproduce

```bash
# byte-exact op-cells (CPU, fast)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=<c4_release> \
  python examples/clever_minflop_optimized.py --verify

# optimized per-step sweep + framing floor + 2-GPU + fps projection (idle GPU)
PYTHONPATH=<c4_release> \
  python examples/clever_minflop_optimized.py --bench --two-gpu --json out.json
```

Golden `174ece66` (c4_min `_fingerprint_build`) is unchanged — this is an
example + doc, no build-path file.
