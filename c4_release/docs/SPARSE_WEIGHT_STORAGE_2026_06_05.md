# Sparse Weight Storage — Design + Measurements (2026-06-05)

## TL;DR

The compiled `AutoregressiveVM` model written by `compile_full_vm()` is
**99.78% sparse**. A bit-mask + packed-value sidecar can shrink the on-disk
cache from **~556 MB to ~19 MB (~30x)**, and a COO (index+value) encoding
can shrink it to **~3.6 MB (~155x)** at the cost of a slightly slower load
path. This document records the measurements and a serialization design;
no implementation is attempted here.

## 1. Measurements (warm cache load)

Measured against the warm cache entry
`~/.cache/c4_release/compiled_vm/2e28b8af4d88680712d512379ee50d7eca9b1422a5a088c285a548a23a1a0216.pt`
(kwargs: `alu_mode=lookup, pin_io_only=True, max_seq_len=4096, ffn_hidden=4096`).
The kwargs differ slightly from the current `compile_full_vm` defaults
(`max_seq_len=8192`), but only the per-layer Parameter shapes differ; the
sparsity structure is identical because all bake_fns write sparse rules.

### Global totals (parameters only)

| metric                   | value          |
|--------------------------|----------------|
| `nn.Parameter` tensors   | **543**        |
| Total dense params       | **145,747,010**|
| Total **nonzero** params | **325,687**    |
| Sparsity                 | **99.7765%**   |
| Dense bytes (fp32)       | **555.98 MB**  |
| Unique nonzero values    | 370            |
| ±1 weights               | 39,308 (12.1%) |
| Param dtypes             | all `float32`  |
| Buffers                  | 68 tensors / 4.78 M elems / 18.25 MB (8,903 nz) |

The current on-disk cache (`.pt`, full pickle) is 575–602 MB; the
`.safetensors` sidecars left behind by other tooling are ~584 MB.

### Per-block parameter breakdown

| block | total       | nonzero | sparsity   | dense MB |
|-------|-------------|---------|------------|----------|
|  0    |  2,135,966  |     162 | 99.9924%   |   8.15   |
|  1    |  2,131,594  |     146 | 99.9932%   |   8.13   |
|  2    |  2,142,524  |      72 | 99.9966%   |   8.17   |
|  3    |  2,321,776  |   1,120 | 99.9518%   |   8.86   |
|  4    |  3,307,662  |   3,547 | 99.8928%   |  12.62   |
|  5    |  2,313,032  |   1,344 | 99.9419%   |   8.82   |
|  6    |  5,242,272  |  22,838 | 99.5643%   |  20.00   |
|  7    |  2,122,850  |     421 | 99.9802%   |   8.10   |
|  8    |  6,612,894  |  14,374 | 99.7826%   |  25.23   |
|  9    |  2,631,090  |     784 | 99.9702%   |  10.04   |
| 10    |  9,563,994  |  26,657 | 99.7213%   |  36.48   |
| 11    |  2,631,090  |     784 | 99.9702%   |  10.04   |
| 12    |  6,156,020  |  12,163 | 99.8024%   |  23.48   |
| 13    |  2,344,704  |   1,634 | 99.9303%   |   8.94   |
| 14    |  2,129,408  |     102 | 99.9952%   |   8.12   |
| 15    |  3,239,896  |  25,122 | 99.2246%   |  12.36   |
| 16    |  3,239,896  |  24,610 | 99.2404%   |  12.36   |
| 17    |  3,239,896  |  24,608 | 99.2405%   |  12.36   |
| 18    |  5,478,360  |  15,004 | 99.7261%   |  20.90   |
| 19    |  2,647,840  |     512 | 99.9807%   |  10.10   |
| 20    | 11,074,520  |  28,672 | 99.7411%   |  42.25   |
| 21    | 10,735,248  |  10,974 | 99.8978%   |  40.95   |
| 22    | 11,074,520  |  28,672 | 99.7411%   |  42.25   |
| 23    | 10,735,248  |  10,974 | 99.8978%   |  40.95   |
| 24    | 11,074,520  |  37,083 | 99.6652%   |  42.25   |
| 25    |  6,237,968  |  10,936 | 99.8247%   |  23.80   |
| 26    |  5,003,998  |  10,648 | 99.7872%   |  19.09   |
| 27    |  3,268,072  |   3,800 | 99.8837%   |  12.47   |
| 28    |  2,385,170  |   2,720 | 99.8860%   |   9.10   |
| 29    |  2,122,850  |       0 | 100.0000%  |   8.10   |

Outside blocks: `embed.embed.weight` (1,589 nz / 200,928), `head.weight`
(3,339 nz / 200,928), `head.bias` (276 nz / 276 — fully dense).

### Sub-module breakdown (attn vs ffn)

Attention bias/weight is dominated by giant fully-zero ALiBi/QKV tensors:
every block has ~2.12 M attn params with sparsity 99.94%–100%. The
non-zero attn mass is concentrated in **block 27** (3,419 nz across a
3.18 M-param attn block) and the L8–L26 routing residuals.

FFN weights are similarly dominated by per-block hidden×d_model blocks
that are mostly zero rows. The **L20–L24 ALU lookup tables** are the
biggest source of nonzeros: block 24's FFN alone holds 36,864 nz, and
blocks 20/22/24 each carry the 28,672-nz schoolbook-MUL lookup.

### Tensor shape distribution

| rank | count | total nz |
|------|-------|----------|
| 1D   |  211  |  33,738  |
| 2D   |  332  | 291,949  |
| 3D+  |    0  |       0  |

All sparse-storable tensors are 1D or 2D, which keeps the encoder simple.

## 2. Format candidates (estimated)

Bytes per nonzero scaled per shape rank; per-tensor overhead estimated at
64 B (name + shape + dtype + format tag). 543 tensors → 34.8 KB headers.

| format                     | est. size | vs dense |
|----------------------------|-----------|----------|
| Dense (current, fp32)      | 555.98 MB |    1.0x  |
| Bit-mask + packed fp32     | **18.65 MB** |   30x  |
| Bit-mask + packed fp16     | 18.03 MB  |   31x   |
| CSR (rowptr+col+val, fp32) |  3.31 MB  |  168x   |
| **COO (int32 idx, fp32 val)** | **3.63 MB** | **153x** |
| COO + fp16 values          |  3.01 MB  |  185x   |
| COO + 16-bit code-book (370 unique vals) | ~2.6 MB | 215x |

fp16 is only safe if the unique-value set (370 distinct nonzeros, range
[-179200, 10000]) fits losslessly — it doesn't (values like -179100 lose
1 ULP at fp16). **Byte-identity therefore requires fp32 values** or an
explicit code-book table (370 unique floats stored once as fp32, indices
as int16 per nz).

## 3. Recommended format: COO + fp32 (lossless)

A flat sidecar file `weights_sparse.npz` alongside the existing `.pt`
cache, written **only for the parameter weights**, with the existing
`.pt` carrying the pickled `nn.Module` skeleton (state_dict elided).

### File layout

Tightly packed binary, single file (gzip-then-`.safetensors` is fine but
not required). Conceptually:

    header:
        magic            b"C4SPARSEV1"
        n_tensors        uint32
        kwargs_snapshot  json blob (matches existing _cache_key)
        source_hash      sha256 (matches _hash_source_bytes)
    per tensor:
        name             utf-8 length-prefixed
        dtype            uint8 (0=f32, 1=f16, 2=i32, 3=i64)
        rank             uint8
        shape            rank * uint32
        nnz              uint32
        idx              nnz * rank * uint32   # flat indices into shape
        val              nnz * sizeof(dtype) bytes

For 2D tensors `idx` is `(row, col)` int32 (the largest dim in the model
is 4096 → fits in uint16, but int32 keeps the format uniform and still
costs only 8 B / nz). For 1D tensors `idx` is a single uint32.

### Load path

1. `torch.load(...pt)` returns the module skeleton (no parameter storage
   — see "write path" below). All `nn.Parameter` tensors are
   zero-initialized to their declared shape from
   `layout.dim_positions` / `layout.dim_sizes`.
2. `weights_sparse.npz` is mmap-opened, traversed once, and each tensor's
   `(idx, val)` is scattered into the corresponding `nn.Parameter` via
   `param.view(-1).index_copy_(0, flat_idx, val)`.
3. Buffers (18 MB, mostly dense) stay in the existing `.pt` payload —
   they're small enough not to justify sparse encoding.

### Write path (when saving cache)

Replace `_try_save_cached` with:

1. Walk `model.named_parameters()`, build `(name, idx, val)` triples for
   every Parameter where nnz/total ≤ 0.5. Otherwise store the dense
   tensor inline in the `.pt`.
2. Replace each sparse parameter's storage in the pickle with a
   zero-stride zero-data placeholder (or set the module's `state_dict`
   to drop those keys before `torch.save`). Re-attach storage on load.
3. Atomic temp-file + `os.replace`, mirroring the current code path.

### Byte-identity invariants

- `_cache_key()` / `_hash_source_bytes()` already gate validity. The
  sparse format gets bumped behind `_CACHE_FORMAT_VERSION = 2` so a
  pre-existing v1 dense cache invalidates gracefully.
- Values are stored at parameter dtype (fp32). The model's
  `state_dict()` after sparse-load is `torch.equal`-identical to the
  fully-baked one — this is the test gate (already covered by
  `tests/test_compile_determinism`; extend it with a save→load roundtrip
  assertion).
- Flat-index ordering is deterministic (row-major from `torch.nonzero`),
  so the file is byte-stable across runs once the bake is.

## 4. Why this beats torch.sparse_csr_tensor

PyTorch's built-in sparse tensors can't be parameters of a vanilla
`nn.Module` without changing every consumer (`F.linear` on a CSR
parameter needs a dispatcher path). Keeping dense `nn.Parameter`s and
loading sparsely on disk is a pure I/O change — runtime is byte-identical
because the parameters in RAM are still dense `fp32`.

## 5. Expected savings

- Cache file: **555.98 MB → ~3.6 MB (≥150x reduction)**, well past the
  ≥10x target.
- Cache load time: dominated by 325K small writes (~5 ms scatter) vs.
  ~1 s pickle decode for the dense `.pt` — likely faster.
- RAM footprint at runtime: **unchanged** (still 556 MB dense fp32).

If RAM is a future target, the same COO file could be replayed against
`torch.sparse_csr_tensor` parameters once the consumers (`PureFFN`,
`AutoregressiveVM`) are taught to dispatch on sparse weights. That's a
separate project — this document only covers on-disk size.
