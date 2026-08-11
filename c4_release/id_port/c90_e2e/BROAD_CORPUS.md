# Broad C90 transpiler-conformance corpus

Turns the 117 hand-written cases (`transpiler_corpus.py`) into a **broad,
standard C90 conformance suite** by folding in the **c-testsuite** single-exec
suite (https://github.com/c-testsuite/c-testsuite, `tests/single-exec/*.c` — the
collaborative C-compiler conformance database, 220 cases).  This is the
foundation for case-by-case transpiler debugging (task #880).

Each case runs through the SAME SOURCE-path port pipeline the Doom port uses:

```
transpile.py  ->  compile_c  ->  c4vm.C4VM     (the PORT path, judged)
   vs
gcc -m32 -std=c90                              (the x86 ORACLE: exit%256 + stdout)
```

CPU-only, no model load; golden `174ece66` unchanged (nothing here is on the
model build path).

## Run it

```
export C4_DOOM_IDPORT=<c4_doom>/id_port          # transpile.py + c4vm.py + struct_engine
export C4_RELEASE_ROOT=<c4_release>              # src/compiler.py
export C4_CTESTSUITE_DIR=<c-testsuite>/tests/single-exec
cd $C4_DOOM_IDPORT
python3 <c4_release>/id_port/c90_e2e/run_broad_conformance.py \
        --json BROAD_CONFORMANCE_MATRIX.json
```

Get the c-testsuite (network):
`git clone --depth 1 https://github.com/c-testsuite/c-testsuite`.
If unreachable, the runner still works on the hand corpus alone
(`--only hand`); the c-testsuite loader (`ctestsuite_loader.py`) returns [] when
the checkout is absent.

## Corpus size & coverage

| source                     | cases | note |
|----------------------------|------:|------|
| hand (`transpiler_corpus`) |  117  | curated per-construct + B1..B9 reproducers |
| c-testsuite single-exec    |  220  | standard collaborative C90 suite |
| **total attempted**        |  337  | |

The c-testsuite loader auto-classifies each case by DOMINANT construct
(multidim / structarr / union / enum / fnptr / varargs / bitwise / loops /
switch / goto / recursion / strings / storage / compound / ptrarith / misc) and
flags OUT-OF-SUBSET cases (function pointers → B3, user `va_arg` → B9, float/
double, long-long) so a fail there is a documented boundary, not a bug.

## Conformance matrix (measured)

`gcc -m32` is the oracle; cases gcc itself rejects (c99/c11-only, needs-64-bit)
are `gcc_skip` and excluded from the in-subset denominator.

```
corpus total (attempted):        337
  gcc-rejected (excluded/skip):   23   (no valid oracle)
  out-of-subset (by-design):      28   (B3 fnptr / B9 varargs / float / longlong)
  IN-SUBSET (judged):            286
  ------------------------------------
  IN-SUBSET PASS (pre-fix):      207/286 = 72.4%
```

The 117 hand cases are at 95.7% (they were authored to the transpiler's known
surface); the broad c-testsuite portion drops the aggregate to ~72% because it
exercises constructs the Doom-scoped transpiler never had to lower.  **That drop
IS the finding** — the broad corpus surfaces the transpiler's real conformance
gaps beyond the hand-tuned set.

See `BROAD_CORPUS_BUG_CLASSES.md` for the NEW bug classes (beyond B1..B9) the
broad corpus exposes, minimal reproducers, and the prioritized fix list.
