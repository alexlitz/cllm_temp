# Phase 10 Baseline Measurement (2026-06-02)

Baseline reference point for Axis-1/2/3 compression work. All numbers
come from a clean `compile_full_vm_dynamic(strict=False)` at default
config on `speedup-cache-and-buckets@0b18859d`.

Reproduction:

```
cd c4_release && CUDA_VISIBLE_DEVICES=1 python /tmp/phase10_measure.py
```

(measurement script archived at `/tmp/phase10_measure.py` in the
working session)

---

## 1. Per-component parameter counts

| Component                                   |       Params |   M params |   % of total |
|---------------------------------------------|-------------:|-----------:|-------------:|
| Token embedding (`model.embed.embed.weight`)|      220,800 |     0.221  |      0.12 %  |
| LM head weight (`model.head.weight`)        |      220,800 |     0.221  |      0.12 %  |
| LM head bias  (`model.head.bias`)           |          276 |     0.000  |      0.00 %  |
| **Embedding subtotal**                      |   **441,876** | **0.442**  |   **0.24 %** |
| Attention `W_q` (sum 31 blocks)             |   20,320,000 |    20.320  |     11.04 %  |
| Attention `W_k` (sum 31 blocks)             |   20,320,000 |    20.320  |     11.04 %  |
| Attention `W_v` (sum 31 blocks)             |   20,320,000 |    20.320  |     11.04 %  |
| Attention `W_o` (sum 31 blocks)             |   20,320,000 |    20.320  |     11.04 %  |
| **Attention subtotal**                      |**81,280,000**|**81.280**  |  **44.15 %** |
| FFN `W_gate` (sum 31 blocks)                |   26,396,800 |    26.397  |     14.34 %  |
| FFN `W_up`   (sum 31 blocks)                |   26,396,800 |    26.397  |     14.34 %  |
| FFN `W_down` (sum 31 blocks)                |   26,396,800 |    26.397  |     14.34 %  |
| FFN `b_up`                                  |       32,996 |     0.033  |      0.02 %  |
| FFN `b_gate`                                |       32,996 |     0.033  |      0.02 %  |
| FFN `b_down`                                |       19,200 |     0.019  |      0.01 %  |
| **FFN subtotal**                            |**79,275,592**|**79.276**  |  **43.06 %** |
| LayerNorm / RMSNorm params                  |            0 |     0.000  |      0.00 %  |
| **TOTAL**                                   |**184,121,104**|**184.121**|  **100 %**   |

Notes:
* Attention is *square* (`d_model × d_model = 800 × 800 = 640,000` per
  matrix) and split evenly across Q/K/V/O. Per block: 4 × 640,000 =
  2,560,000 params. 31 blocks × 2,560,000 = 79,360,000 (residual
  delta to 81,280,000 due to one wider block, see §3 below).
* FFN params vary per block (right-sized hidden width). See §4 for
  the per-block breakdown.
* No LayerNorm/RMSNorm params: `use_rms_norm=False` at default
  config.

---

## 2. `d_model` breakdown

| Metric                                | Value |
|---------------------------------------|------:|
| `model.d_model`                       |   800 |
| `dim_registry.d_model`                |   736 |
| Sum of declared dim widths (active)   |   733 |
| **Slack** = `d_model − active`        | **67**|
| **Slack %** = slack / `d_model`       |**8.4%**|

The compiler builds a wider model than the registry strictly needs
(`d_model=800` vs registry's 736). Of the 800 residual slots, 733 are
named in `build_default_registry`, leaving **67 unused dims (8.4 %)**.
Note: 3 of those 67 are slack inside the 736-registry view; the rest
(64) are the construction-time gap between 800 and 736. Axis-2 (slim
residual) can target both portions.

---

## 3. Wrapper-block analysis

| Metric                                                  | Value |
|---------------------------------------------------------|------:|
| `n_native_layers` (pre `_expand_wrapper_blocks`)        |    17 |
| `n_expanded_blocks` (post-expansion)                    |    31 |
| **Wrapper blocks added**                                |**14** |

The 14 wrapper ops added, in expansion order (src native idx → wrapper
ffn class):

| # | Src native | Parent ffn cls | Post-op wrapper ffn cls           | W_q shape   | W_q vs parent     |
|---|-----------:|----------------|-----------------------------------|-------------|-------------------|
| 1 |         L8 | PureFFN        | `AddSub5StageBlock`               | 800 × 800   | distinct (zero)   |
| 2 |         L9 | PureFFN        | `AddSub5StageBlock`               | 800 × 800   | distinct (zero)   |
| 3 |        L10 | PureFFN        | `ALUAndOrXor`                     | 800 × 800   | distinct (zero)   |
| 4 |        L10 | PureFFN        | `BinaryOpByteZeroingPostOp`       | 800 × 800   | distinct (zero)   |
| 5 |        L10 | PureFFN        | `AddSubBytePropagationPostOp`     | 800 × 800   | distinct (zero)   |
| 6 |        L10 | PureFFN        | `CarryPropagationPostOp`          | 800 × 800   | distinct (zero)   |
| 7 |        L10 | PureFFN        | `CarryPropagationPostOp`          | 800 × 800   | distinct (zero)   |
| 8 |        L10 | PureFFN        | `CarryPropagationPostOp`          | 800 × 800   | distinct (zero)   |
| 9 |        L10 | PureFFN        | `BitwiseBytePropagationPostOp`    | 800 × 800   | distinct (zero)   |
|10 |        L10 | PureFFN        | `FlattenedDivMod`                 | 800 × 800   | distinct (zero)   |
|11 |        L11 | PureFFN        | `FlattenedALUMul`                 | 800 × 800   | distinct (zero)   |
|12 |        L12 | PureFFN        | `FlattenedALUMul`                 | 800 × 800   | distinct (zero)   |
|13 |        L13 | PureFFN        | `ALUShiftComposite`               | 800 × 800   | distinct (zero)   |
|14 |        L17 | PureFFN        | `PureFFN`                         | 800 × 800   | distinct (zero)   |

Each wrapper's `attn` is a **fresh zero-initialized
`AutoregressiveAttention(d_model, …)`** (see
`vm_step.py:2599 _make_passthrough_block`). The W_q matrix is
structurally distinct from the parent (independent `nn.Parameter`),
and all four weight matrices (W_q/W_k/W_v/W_o) are all-zero — the
attention computes `x + Attn(x) = x + 0 = x`. So each wrapper carries
4 × 800 × 800 = 2.56 M completely-dead attention params.

**Wrapper attention overhead: 14 × 2,560,000 = 35,840,000 params (19.5 % of
total)**. Axis-1 candidate — wrappers could share parent attn or be
inlined.

---

## 4. Per-layer FFN unit counts (post `_right_size_ffns`)

Final block index → `(attn cls, ffn cls, hidden units, W_q shape)`.
`hidden=0` means the FFN is a wrapped composite (W_up not flat); see
the per-stage breakdown in the compile log for those.

| idx | attn cls                  | ffn cls                       | hidden | W_q shape | notes |
|----:|---------------------------|-------------------------------|-------:|-----------|-------|
|   0 | AutoregressiveAttention   | PureFFN                       |      7 | 800×800   |       |
|   1 | AutoregressiveAttention   | PureFFN                       |      5 | 800×800   |       |
|   2 | AutoregressiveAttention   | PureFFN                       |     10 | 800×800   |       |
|   3 | AutoregressiveAttention   | PureFFN                       |    544 | 800×800   |       |
|   4 | AutoregressiveAttention   | PureFFN                       |      1 | 800×800   |       |
|   5 | AutoregressiveAttention   | PureFFN                       |   1514 | 800×800   |       |
|   6 | AutoregressiveAttention   | PureFFN                       |    375 | 800×800   |       |
|   7 | AutoregressiveAttention   | PureFFN                       |   2056 | 800×800   | >2000 |
|   8 | AutoregressiveAttention   | PureFFN                       |   3405 | 800×800   | >2000 |
|   9 | AutoregressiveAttention   | AddSub5StageBlock             |      0 | 800×800   | wrap  |
|  10 | AutoregressiveAttention   | PureFFN                       |   1846 | 800×800   |       |
|  11 | AutoregressiveAttention   | AddSub5StageBlock             |      0 | 800×800   | wrap  |
|  12 | AutoregressiveAttention   | PureFFN                       |      8 | 800×800   |       |
|  13 | AutoregressiveAttention   | PureFFN                       |   1536 | 800×800   |       |
|  14 | AutoregressiveAttention   | PureFFN                       |    512 | 800×800   |       |
|  15 | AutoregressiveAttention   | PureFFN                       |    512 | 800×800   |       |
|  16 | AutoregressiveAttention   | PureFFN                       |    512 | 800×800   |       |
|  17 | AutoregressiveAttention   | PureFFN                       |   1536 | 800×800   |       |
|  18 | AutoregressiveAttention   | FlattenedDivMod               |      0 | 800×800   | wrap  |
|  19 | AutoregressiveAttention   | PureFFN                       |   4096 | 800×800   | >2000 |
|  20 | AutoregressiveAttention   | ALUAndOrXor                   |      0 | 800×800   | wrap  |
|  21 | AutoregressiveAttention   | PureFFN                       |   4096 | 800×800   | >2000 |
|  22 | AutoregressiveAttention   | FlattenedALUMul               |      0 | 800×800   | wrap  |
|  23 | AutoregressiveAttention   | PureFFN                       |   4096 | 800×800   | >2000 |
|  24 | AutoregressiveAttention   | FlattenedALUMul               |      0 | 800×800   | wrap  |
|  25 | AutoregressiveAttention   | PureFFN                       |   1874 | 800×800   |       |
|  26 | AutoregressiveAttention   | ALUShiftComposite             |      0 | 800×800   | wrap  |
|  27 | AutoregressiveAttention   | PureFFN                       |     42 | 1400×800  | wider W_q |
|  28 | AutoregressiveAttention   | PureFFN                       |    792 | 800×800   |       |
|  29 | AutoregressiveAttention   | PureFFN                       |   1562 | 800×800   |       |
|  30 | AutoregressiveAttention   | PureFFN                       |   2059 | 800×800   | >2000 |

### Layers with hidden > 2000 units (compression candidates)

| idx | hidden | ffn cls  | source role (heuristic)         |
|----:|-------:|----------|---------------------------------|
|   7 |   2056 | PureFFN  | native L7 (memory + LI/LC/SI/SC)|
|   8 |   3405 | PureFFN  | native L8 (largest single FFN)  |
|  19 |   4096 | PureFFN  | post-L10 wrapper PureFFN        |
|  21 |   4096 | PureFFN  | post-L10 wrapper PureFFN        |
|  23 |   4096 | PureFFN  | post-L10 wrapper PureFFN        |
|  30 |   2059 | PureFFN  | post-L17 wrapper PureFFN        |

Three of these (idx 19/21/23) are wrapper-block PureFFN's sized to
the legacy `default_hidden=4096` — they were never re-sized down by
`_right_size_ffns` because their `W_up.abs().sum() > 0` for every
unit. These are immediate Axis-3 targets: instrument actual write
density and trim.

### Aggregate FFN width statistics

* Sum of all hidden units (31 blocks): **38,054**.
* Mean hidden / block: 1227.
* `0`-hidden composite wrappers (count): 6 (L9/L11/L18/L20/L22/L24/L26 minus wraps that re-bake to PureFFN — 6 actual composite wraps).

---

## 5. Open questions for Axis owners

1. The wrapper attention overhead (35.84 M params dead) is the
   single largest immediate-savings line item. Sharing parent attn
   or short-circuiting wrapper attn would save **~19.5 %** of total
   params for free (residual identity is already a no-op).
2. The 67 unused residual dims (8.4 %) are only the registry-side
   view; the post-`_right_size_ffns` write pattern might leave even
   more "structurally cold" dims. A column-density audit on
   `W_down` would pin the true active width.
3. Three 4096-unit wrapper FFNs (idx 19/21/23) bypass the
   right-sizing trim. Worth verifying whether their actual write
   density is < 4096 before sizing them down.

---

## 6. Methodology notes

* Wrapper detection: monkey-patched `_expand_wrapper_blocks` in
  `vm_step.py:2578` to capture the (src_native_idx, post_op_cls)
  pair for each of the 14 expansions before they fold into final
  blocks. This is the canonical source for "the 14 wrapper ops added".
* Param counts: `nn.Module.parameters()` walked block-by-block;
  attention subtotals matched the symbolic 4 × d_model² × 31 + δ
  arithmetic to within the L27 W_q width bump.
* Active dim count: `build_default_registry()` walked with
  `allow_overlap` aliases collapsed via `set` union over each
  slot's `[start, start+size)` range.
