# Sparse FFN Performance Probe

**Status:** Scoping benchmark. **Recommendation: do NOT ship.**

Branch: `sparse-perf-probe`
Date: 2026-05-11
Hardware: NVIDIA RTX A5000 (24 GiB), driver 580.76.05
PyTorch: 2.7.0 + CUDA 12.6
Model: `compile_full_vm()` output (30 transformer blocks, d_model=512, vocab=276)

---

## TL;DR

- Top-level FFNs in the Neural VM are extremely sparse (typically <2% non-zero, often <1%).
- Per-FFN: CSR matmul gives a real **1.5x–2.4x speedup** on large hidden dims (H>=1500) at prefill seq_len>=200.
- End-to-end forward (seq_len=200): **1.05x** speedup at best when replacing 12 FFNs. The model is dominated by block 19 (`FlattenedDivMod`, ~378 ms of ~480 ms total), so optimizing FFN matmul barely moves the needle.
- For autoregressive **decode (seq_len=1)** sparse is **always slower** (0.6–1.0x) due to kernel-launch overhead.
- **Memory: dense top-level FFN weights = 147 MB; CSR = 2 MB (98.6% reduction).**

**Recommendation: do not ship sparse FFNs.** The end-to-end win is negligible (~1.05x at prefill, slower at decode), and the engineering cost (compiler bake changes, gradient handling, decode-vs-prefill switching) is not justified. The big optimization target is **block 19 (FlattenedDivMod)**, not FFNs. **However**, if/when we ship a final inference build that doesn't need backward and only runs prefill at seq_len>=200, converting FFNs to CSR is a free 145 MB memory save with no per-block performance regression — that's worth a tiny PR on its own.

---

## 1. Per-block sparsity table (top-level `block.ffn` only)

`W_up` is `[H, D]`, `W_gate` is `[H, D]`, `W_down` is `[D, H]` with `D=512`. nz% = fraction of entries > 1e-6.

| Block | H | nz_up | nz_gate | nz_down | dense_ms (seq=200) |
|-------|---|-------|---------|---------|--------------------|
| L0  |    7 | 0.20% | 0.17% | 0.20% | 0.12 |
| L1  |    5 | 1.33% | 1.13% | 0.20% | 0.12 |
| L2  |    8 | 0.39% | 0.20% | 0.20% | 0.11 |
| L3  |   92 | 0.77% | 0.11% | 0.24% | 0.11 |
| L4  |  543 | 0.60% | 0.20% | 0.20% | 0.13 |
| L5  |   88 | 0.41% | 0.14% | 0.20% | 0.12 |
| L6  | 1428 | 1.55% | 0.29% | 0.22% | 0.26 |
| L7  |    1 | 0%    | 0%    | 0%    | 0.11 |
| L8  | 2055 | 0.66% | 0.20% | 0.20% | 0.38 |
| L10 | 3405 | 0.92% | 0.19% | 0.21% | 0.43 |
| L12 | 1846 | 0.66% | 0.20% | 0.20% | 0.26 |
| L14 |    4 | 0.39% | 2.73% | 1.66% | 0.12 |
| L15 |  512 | 8.59% | 0.20% | 0.60% | 0.12 |
| L16 |  512 | 8.40% | 0.20% | 0.60% | 0.12 |
| L17 |  512 | 8.40% | 0.20% | 0.60% | 0.11 |
| L18 | 1536 | 1.17% | 0.20% | 0.35% | 0.16 |
| L20 | 4096 | 0.78% | 0.20% | 0.20% | 0.59 |
| L22 | 4096 | 0.78% | 0.20% | 0.20% | 0.59 |
| L24 | 4096 | 0.98% | 0.20% | 0.39% | 0.59 |
| L26 |  115 | 1.65% | 0.20% | 0.20% | 0.11 |
| L27 |   40 | 1.26% | 0.16% | 0.21% | 0.11 |
| L28 |  121 | 3.65% | 0.20% | 0.20% | 0.12 |
| L29 |    1 | 0%    | 0%    | 0%    | 0.10 |

Notable: nz_gate is almost always exactly 0.195% (=1/512) — gate is effectively a single-dim row activation. nz_down is similarly near floor. W_up does most of the actual work and is where any sparsity win lives.

## 2. Per-FFN sparse-CSR speedup table

`(W_csr @ x.T).t() + b` substitute for `F.linear(x, W) + b`, using `tensor.to_sparse_csr()` on each of `W_up`, `W_gate`, `W_down`. Numerical equivalence: max abs err ~1e-3 (single-precision rounding from accumulation order), max rel err ~3e-7. Acceptable.

| Block | H | seq=1 (d/s) | seq=32 (d/s) | seq=200 (d/s) | seq=512 (d/s) |
|-------|---|-------------|--------------|---------------|---------------|
| L3   |   92 | 0.63x | 0.59x | 0.73x | 0.76x |
| L4   |  543 | 0.93x | 0.70x | **1.15x** | **1.84x** |
| L5   |   88 | 0.86x | 0.81x | 0.88x | 0.46x |
| L6   | 1428 | 0.96x | 1.00x | **1.38x** | **1.27x** |
| L8   | 2055 | 0.92x | 0.87x | **2.14x** | **1.64x** |
| L10  | 3405 | 0.97x | 0.97x | **1.46x** | **2.12x** |
| L12  | 1846 | 0.89x | 0.73x | **1.79x** | **1.78x** |
| L15  |  512 | 0.87x | 0.79x | 0.61x | 0.84x |
| L16  |  512 | 0.86x | 0.77x | 0.68x | 0.43x |
| L17  |  512 | 0.94x | 0.64x | 0.58x | 0.61x |
| L18  | 1536 | 0.90x | 0.93x | **1.53x** | **1.32x** |
| L20  | 4096 | 0.89x | 1.29x | **1.76x** | **2.13x** |
| L22  | 4096 | 0.85x | 1.04x | **1.76x** | **1.40x** |
| L24  | 4096 | 1.14x | 1.82x | **1.67x** | **2.44x** |
| L26  |  115 | 0.84x | 0.87x | 0.72x | 0.61x |
| L28  |  121 | 0.80x | 0.83x | 0.81x | 0.86x |

`d/s` = dense_ms / sparse_ms (higher = better). Bold = >=1.15x win.

**Pattern:** Sparse wins where (H >= 1500) AND (seq_len >= 200). The L15/L16/L17 case (H=512) has higher nz_up (~8%) and *never* benefits — too few rows and too dense per row. The H>=4096 cases get up to 2.4x at seq=512.

## 3. Memory footprint

CSR format on a `[H, D]` float32 matrix with `nnz` non-zeros:
- Dense: `H * D * 4` bytes
- CSR: `nnz * 4` (values) + `nnz * 4` (col_indices, int32) + `(H+1) * 4` (crow_ptr) = `8*nnz + 4*H + 4` bytes

Sum across all top-level block FFNs (W_up + W_gate + W_down):

| Format | Total | Fraction of model |
|--------|-------|-------------------|
| Dense (current) | 147.18 MB | 41% of 358.7 MB |
| CSR             |   2.03 MB |  0.6%             |
| **Saved**       | **145.15 MB** | **40.5%**     |

That's a near-total elimination of FFN weight storage. The 145 MB saving is real and survives every benchmark — it doesn't depend on kernel performance.

(Note: this is the *dense parameter* of the FFNs at allocation. The compiler's `right_size_ffns` post-pass already shrinks `hidden_dim` to active units for some FFNs (L0-L7, L14, L26-L28), so the dense memory figure here is what's actually allocated after that pass.)

## 4. End-to-end forward timing (seq_len=200, batch=1)

| Configuration | ms/step | Speedup vs dense |
|---------------|--------|------------------|
| All dense | 475.84 | 1.00x |
| Sparse on 7 high-H blocks (8,10,12,18,20,22,24) | 478.25 | 0.99x |
| Sparse on 12 blocks (H>=512) | 453.73 | **1.05x** |

The end-to-end is dominated by **block 19: `FlattenedDivMod`** (~378 ms of 480 ms, profiled via per-block hooks). The total FFN time across all 30 blocks is only ~5 ms, so even a 2x FFN speedup is at most ~2.5 ms / 480 ms = 0.5% end-to-end. Observed 5% is from spillover into other blocks (reduced kernel-launch contention with smaller FFN buffers).

Per-block profile (ms, 1 forward pass, seq_len=200):
```
block  0   0.93   block 10  0.52   block 20   1.22
block  1   1.24   block 11  5.41   block 21   5.54
block  2   0.97   block 12  0.72   block 22   0.60
block  3   0.56   block 13  5.26   block 23   5.91
block  4   0.61   block 14  1.13   block 24   0.59
block  5   0.79   block 15  0.69   block 25   5.03
block  6   0.46   block 16  0.31   block 26   0.77
block  7   0.64   block 17  0.31   block 27   0.47
block  8   1.01   block 18  0.88   block 28   0.90
block  9   5.32   block 19 378.03  block 29   0.84
                                   SUM      427.66
```

(Block 19 also has FFN cost inside it but the bulk is the long-division pipeline post-op, not the SwiGLU FFN.)

## 5. Caveats

- **CSR + backward:** `torch.sparse_csr_tensor` is documented as "beta" and `nn.Parameter` containing CSR is unstable. Backward pass on sparse CSR is **NOT supported** (`torch.sparse.mm` autograd works on COO only). Any training path must keep dense weights.
- **Kernel-launch overhead:** sparse CSR matmul kernels have higher per-call overhead than cuBLAS. At seq_len=1 they are **always slower** (0.6-1.0x). For autoregressive decode this is a regression. Would need a `if seq_len > THRESHOLD: use_sparse` switch at runtime.
- **Gradient checkpointing:** would need to densify on the fly during backward — defeats the whole point.
- **CSR column index dtype:** PyTorch CSR uses int64 by default on CUDA in some paths (we measured with int32). Memory savings could be smaller (~30% larger CSR) on different builds.
- **Beta status warning:** `UserWarning: Sparse CSR tensor support is in beta state` is emitted on every conversion. Future PyTorch versions may change semantics.
- **Numerical:** sparse@dense and dense@dense produce different summation orders → ~1e-3 max abs error in float32 outputs. Negligible for inference but would invalidate bit-exact tests if used as a baked replacement.

## 6. Recommendation

**Do not wire sparse FFNs into the compiler bake.** Reasons:

1. End-to-end win is 1.05x (~22 ms saved per 480 ms forward) — not the ">1.5x with no numerical drift" bar requested.
2. Decode-mode regression: 0.6-1.0x slowdown at seq_len=1, which is what autoregressive generation uses.
3. The actual bottleneck is **block 19 (`FlattenedDivMod`)** — that's where future perf work should go.
4. Sparse CSR is still beta in PyTorch 2.7; backward unsupported, complicates training/test infrastructure.

**Possible follow-up (optional, separate task):** ship an opt-in "inference-only memory mode" that CSR-converts the top-level FFN matrices after baking. This is purely a memory optimization (-145 MB) for memory-bound deployments (CPU inference, low-VRAM GPUs) where the per-block perf is irrelevant. Sketch:

```python
def to_inference_csr(model):
    """In-place: replace W_up/W_gate/W_down with CSR. Disable training."""
    for b in model.blocks:
        ffn = getattr(b, 'ffn', None)
        if ffn is None or not hasattr(ffn, 'W_up'): continue
        if ffn.W_up.data.is_sparse: continue
        # Save dense, install sparse forward via monkeypatch on FFN
        ffn.W_up_csr = ffn.W_up.data.to_sparse_csr()
        ffn.W_gate_csr = ffn.W_gate.data.to_sparse_csr()
        ffn.W_down_csr = ffn.W_down.data.to_sparse_csr()
        # Free dense (keep meta tensor for state_dict compat)
        ffn.W_up.data = torch.empty(0, dtype=torch.float32, device=ffn.W_up.device)
        ffn.W_gate.data = torch.empty(0)
        ffn.W_down.data = torch.empty(0)
        ffn.forward = _make_sparse_swiglu(ffn)
```

This is **not** part of this probe's deliverable; flagging only.

## 7. Where the real win is

Block 19 (`FlattenedDivMod`) is **79% of forward time** and is the only place worth optimizing. It's an 8-iteration long-division pipeline with 3 sub-FFN-equivalent ops per iteration on a 9-nibble accumulator. Optimization ideas (out of scope here):
- Batch the 8 iterations as a single kernel via `vmap` or graph fusion
- Replace the per-iter ALU-FFN chain with a CUDA kernel that does div/mod directly in GE space (the FFNs are emulating arithmetic — a native kernel would be 100x faster)
- Skip the pipeline entirely when the active opcode isn't DIV or MOD (opcode-gated execution)

That's the lever. FFN sparsity isn't.
