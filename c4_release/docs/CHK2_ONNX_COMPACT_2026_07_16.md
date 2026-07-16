# CHK-2 — the compact whole-VM C4 model as a VANILLA ONNX graph, run through onnxruntime

**Date:** 2026-07-16 · **Branch:** `chk1-onnx-export` · **Base:** `chk1-corpus-fast @ c0614d90`
**Checklist:** #3 (run the corpus through onnxruntime byte-identically) + #14 (vanilla ONNX, standard transformer ops only).

> "someone should be able to look at the forward or the onnx and say *yup that is a
> standard transformer nothing much strange going on there*." — BLOG_SPEC §3

This exports the **compact** pure-forward C4 VM (`compact_alloc`, dim-shared +
per-block-hidden, dim 2576→1702, byte-identical L∞=0) — the whole VM *step*
(softmax1 + ALiBi attention + SwiGLU FFN over the 42-block stack) — to a **single
vanilla ONNX graph**, runs it through `onnxruntime`, and drives the KV-cached
per-step corpus loop with the ONNX graph as the forward.  It re-applies the
`nibble-onnx-sparse` mechanism to the compact whole-VM model instead of the
foundation model.

New code (non-invasive — wraps the existing `forward_hidden_cached`, no shared
model change): `c4_min/export_onnx_compact.py`, `c4_min/run_onnx_corpus.py`,
`c4_min/test_export_onnx_compact.py`.

Repro:
```
python -m c4_min.export_onnx_compact                       # export + vanilla audit
python -m c4_min.run_onnx_corpus --battery                 # per-op byte-identity
python -m c4_min.run_onnx_corpus --exclude-deep            # corpus through ORT
python -m pytest c4_min/test_export_onnx_compact.py        # 10 tests
```

---

## 0. The export target — the KV-cached windowed step

The graph is exactly the per-step forward the driver calls
(`Transformer.forward_hidden_cached`), with the per-block `(K,V,pos)` cache
flattened into fixed-arity tensor inputs/outputs — the standard KV-cache decode
interface:

```
(x_window[1,W,D], q_pos[W], {pastK_b, pastV_b, pastPos_b}_b=0..41)
    -> (hidden[1,W,D], {newK_b, newV_b}_b)
```

Each block carries its **own** cached-position vector — after the softmax1+ALiBi
eviction the block caches keep different survivor sets, so per-block positions are
load-bearing (a single shared vector fails at `blocks.1/attn/Sub`).  The register
decode + eviction stay in the caller.  `OnnxCachedModel` wraps the ORT session
behind the same `.embed`/`.blocks`/`.forward_hidden_cached` API, so
`run_pure_forward_cached` drives the ONNX model unchanged.

Compact LEAN model: **dim 1702, 42 blocks, 23 heads, head_dim 74, vocab 265**;
640.6M dense params, 180.5k non-zero → **99.972 % sparse**.

## 1. Export size

| file | size |
|---|---|
| `compact_vm.onnx` (dense) | 681.6 MB |
| `compact_vm_sparse.onnx` (COO `sparse_initializer`) | **1.6 MB** (128 tensors sparsified) |

The dense tensors are stored as ONNX `sparse_initializer` COO; ORT reconstructs
the dense weight for the `MatMul`, so the forward is byte-identical — the file
just drops the 99.97 % zeros (incl. the 21544-hidden `bw-select` lookup FFN).

## 2. Vanilla audit — `VANILLA = YES`

Full op inventory (per-block counts are exactly 42×, proving no per-block anomaly):

```
MatMul   x378   (42 blocks × 9 = QKVO + QKᵀ + attn·V + FFN up/gate/down)
Exp x84  ReduceMax x42  ReduceSum x42  Div x42   -> softmax1 (ZFOD, §491)
Sigmoid x42  Mul x168                            -> SwiGLU gate (§467-478)
Sub x126  Abs x42  Neg x42                        -> ALiBi bias −slope·|i−j|
Greater x42  Where x42                            -> causal mask (abs positions)
Concat x294                                       -> KV-cache append (NOT a Loop)
Gather x126  Reshape/Transpose/Unsqueeze/Cast/Shape/Constant/Add/Clip/Identity
```

No `Loop`/`Scan`/`If`, no `*Normalization`/`RNN`/`LSTM`/`GRU`/fused-`Attention`,
no custom op-domains, no external-memory ops.  A textbook decode-only transformer
with a KV cache.

## 3. Byte-identity (torch vs onnxruntime)

**Per-op battery** (KV-cached driver, torch forward vs ONNX forward):

| prog | trace== | prog | trace== |
|---|---|---|---|
| add | ✅ | eq | ✅ |
| sub | ✅ | si_li (mem store→load) | ✅ |
| mul | ✅ | func_jsr_lev (JSR/ENT/LEV) | ✅ |
| add32 (60000+60000) | ✅ | loop_countdown | ✅ |
| mul32 (1000×1000) | ✅ | lt | ✅ |

**10/10 emitted traces byte-identical.**  Worst per-step hidden **relative** diff
≈ fp accumulation-order residue (~1e-6); the large absolute diffs are only on the
huge-magnitude arithmetic bands.  **Register-decode argmax flips: 0.**

**Cross-check sample** (torch vs ONNX, full byte trace + verdict, 10 clusters —
add/sub/mul/div/mod/if_gt/var_simple/var_mul/var_three/var_update): **30/30
agree, 0 disagree, every trace_ident=True.**

## 4. Corpus through onnxruntime

Full non-deep corpus (1072 programs ≤ 400 ref-steps), driven by the KV-cached
per-step loop with the **ONNX graph as the forward**, 2 CPU shards:

```
scored 1072 | ONNX PASS 725  FAIL 156  TIMEOUT 191   ->  725/1072 = 67.63 %
wall ~86 min/shard (CPU, onnxruntime) | 43,609 ONNX VM steps
```

**725 pass — byte-for-byte the same as the torch pure-forward non-deep result.**
The non-passing clusters are exactly the expected ones and are identical to the
torch run (confirmed by the cross-check, which included div/mod):

* **div / mod / expr_mod / expr_mul_div / edge_*div** — the LEAN model has no
  DIV/MOD blocks (build divmod for these; torch LEAN fails them identically).
* **191 TIMEOUTs** (91 in `loop_*`/`rec_*`/`gcd`) — the deep-loop tail hitting the
  `--step-cap`, the same throughput caveat as the torch run (not blocked on).

## Verdict

The compact whole-VM C4 model is a **standard transformer**: it exports to a
vanilla ONNX graph (no control-flow / custom / external-memory ops), runs in
onnxruntime **byte-identically** to the torch pure-forward model (10/10 battery +
30/30 cross-check, 0 argmax flips), and drives the corpus to the **same 725-pass**
non-deep result through onnxruntime.
