# CHK-5 — the quine on the SMALL model (byte-exact self-output + bundle)

**Branch:** `chk1-quine-small` (off `chk1-corpus-fast` @ `c0614d90`).
**Substrate:** `c4_min/` pure-forward whole-VM transformer, compacted + dim-shared
(`compact_alloc`) and stored sparse (`sparse_forward`), driven by the KV-cached
pure-forward driver (`nibble_pure_forward_cached.run_pure_forward_cached`).
**Authority:** `docs/BLOG_SPEC.md` §"Making a Quine" (line 922, the classic string
quine) + §"Printing and Reading Input" (line 851, the think-tag I/O contract).
**Output opcode:** **PRTF** (printf, op 33) — the spec's I/O opcode — NOT PUTCHAR.

---

## TL;DR

A **C4-C quine** (a program that prints its own source) compiled to bytecode and
run **through the small pure-forward transformer** (the KV-cached driver): its
**visible output is byte-for-byte its own source**, and the whole thing is packaged
as a **single 3.06 MB bundle** (runtime-ref + small weights + bytecode) that, when
executed, reproduces the source.

```
$ python -m c4_min.quine_prtf                      # reference self-check (no model)
instructions: 27   source bytes: 54
QUINE OK (reference visible output == own source): True

# neural, small model, KV-cached driver:
quine: 27 instrs, 54 source bytes
cached run 492s steps=1199 max_cache=1680 max_seq=37591 evicted=1573977
printed 54 bytes
MATCH source (BYTE-EXACT QUINE): True
QUINE bytes: [1, 240, 13, 0, 1, 0, 11, 0, 1, 240, 9, 0, 13, 0, 1, 54, 26, 0, ...]  == its own source
```

---

## The quine (`c4_min/quine_prtf.py`)

The classic string-quine: **data that encodes the code + code that prints the data.**

* **Data** — the program's own WIDTH=2 byte serialization `S = [op0, imm0, ...]`
  lives in memory as a table `Q` (byte cells `Q[i] = mem[i]`). On the pure-forward
  VM (memory is built from stores only) this is the **data segment `seed_mem`** the
  driver lays into the KV memory as leading MEM-STORE frames before step 0 — the
  analogue of the string literal a textbook C quine embeds.
* **Code** — a loop walks `Q` with `LI` and emits each byte via `PRTF` for `N`
  bytes; the counter `i` lives at `mem[0xF0]`. Because `Q` **is** `S`, the emitted
  visible stream equals the source. The two self-reference literals (`N` and the
  `BZ` exit target) don't change the instruction count, so the fixed point is one
  pass. 27 instructions / 54 source bytes.

**PRTF as visible output** (`isa.py`, `blogspec_vocab.py`, the two drivers): a PRTF
step emits `THINK_END, <byte>, THINK_START` around one visible byte, decoded from
the model's **own AX byte-0 nibbles by the LM byte-head** (a genuine argmax, not a
python copy). `blogspec_vocab.visible_output` recovers the between-THINK bytes.
PRTF's register transition is a pure `PC += 1` (I/O only, AX/SP/BP unchanged) —
one FFN dispatch rule + one opcode-decode entry.

## The small model + memory horizon

`build_compact_pure_forward_model(code_size=64, include_divmod=False)` → compact
dim-shared model (**dim=1702, 42 blocks, 180 941 nnz**), wrapped
`SparseTransformer(compute_mode="dense_kernel")` (L∞=0 to the dense forward). Sparse
storage **≈ 9 MB** (dense-equiv 2.56 GB; naive ≈ 30 GB).

The quine's data segment `Q` is loaded at arbitrary future steps over a ~37 k-token
run — beyond the default §Memory ALiBi reachability horizon (`EFF/MEM_ALIBI_SLOPE ≈
666` tokens). The build widens it to **`MEM_ALIBI_SLOPE = 0.05`** (`EFF/slope ≈ 80 k`
tokens) so a far-back data store still reads weight ~1 while same-address counter
writes (spaced ~1 frame apart) still get decisive latest-write-wins recency. This
slope is a **bundle-local model parameter**; the CHK-1 corpus model uses the default
6.0. The KV-cached driver also **pins the data-segment store rows** against eviction
(`protect_positions`) so `Q` survives while every register frame still evicts — the
cache stays bounded (max 1680).

## The bundle (`c4_min/quine_bundle.py`)

`build_bundle(path)` writes a single `.pt` (**3.06 MB**) carrying:
  a) **runtime/driver** — referenced from the bundle's `run_bundle` entrypoint (the
     `c4_min` KV-cached VM + `SparseTransformer`);
  b) **small model weights** — the compact model, COO-sparse (180 941 nnz);
  c) **quine bytecode** — the `(op, imm)` code table + the `seed_mem` data segment +
     the saved **compacted layout** (compaction re-indexes dims, so the layout is
     stored, not rebuilt) + config (slope, sp_init).

`run_bundle(path)` loads it, runs the bytecode through the model, and returns the
visible output == the bundled source. Bundle-load fidelity is verified L∞=0 vs a
fresh compact build, and the loaded model's visible output matches the source.

## Validated vs dependency-blocked

* **VALIDATED (this deliverable):**
  * quine self-output is **byte-exact its own source** on the SMALL pure-forward
    transformer via the KV-cached driver (`MATCH: True`, 54/54 bytes);
  * the reference self-check (no model) is byte-exact (`quine_prtf.run_reference`);
  * the **bundle assembles** (3.06 MB) and its reconstructed model is L∞=0 to a
    fresh build and reproduces the source;
  * no regression: `c4_min/test_pure_forward.py` 11/11 pass; the new PRTF op is a
    silent `PC += 1` no-op on the non-I/O paths.
* **DEPENDS ON separate CHK-1 pieces (NOT validated here):**
  * the checklist's "quine must pass the 1000+ tests" is the **CHK-1 corpus result
    on the bundled model** — that runs on `run_1096_sparse_gpu_cached.py` with the
    DEFAULT `MEM_ALIBI_SLOPE=6.0` and is scored separately;
  * ONNX-runtime packaging of the bundle (a C runtime executing the ONNX-exported
    small model) is a separate track — here the "runtime" is the Python KV-cached
    driver the bundle ships with.

## Files

* `c4_min/quine_prtf.py` — the PRTF string quine + reference self-check.
* `c4_min/quine_bundle.py` — build/run the single-artifact bundle.
* `c4_min/test_quine_prtf.py` — 6 fast tests + 1 opt-in neural test
  (`C4_RUN_NEURAL_QUINE=1`).
* `c4_min/isa.py`, `c4_min/blogspec_vocab.py` — PRTF opcode + think-tag I/O tokens
  + `visible_output`.
* `c4_min/nibble_pure_forward_complete.py`, `c4_min/nibble_pure_forward_cached.py` —
  PRTF visible-byte emission, `seed_mem` data segment, data-segment eviction pin.
