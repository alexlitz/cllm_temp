#!/bin/bash
# _agent_allc_build.sh HEADER OUT [nblbin]
# Compile the ALL-C VM binary embedding HEADER (allc_gen.h) + the block-stack model.
set -e
HDR="$1"; OUT="$2"; BINP="${3:-/tmp/fullisa_sparse/blockstack.nblbin}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/onnx_runtime_nibble_allc.c"
TMPD="$(mktemp -d)"
cp "$HDR" "$TMPD/allc_gen.h"
cp "$SRC" "$TMPD/onnx_runtime_nibble_allc.c"
# Optimisation flags (byte-exactness-preserving):
#   -O3 -funroll-loops : full opt + loop unrolling (free speed; auto-vectorises the
#                        elementwise silu/residual/scatter loops, which do NOT reorder).
#   -ffp-contract=off  : NO fused-multiply-add contraction -> the fp accumulation order
#                        is exactly the source order (load-bearing for byte-exactness at
#                        the ~1e29 residual magnitudes; a contracted FMA flips low bits).
#   NEVER -ffast-math  : it reassociates fp -> breaks the decode argmax (non-byte-exact).
# -march=native tunes to THIS host (the C itself is portable: only libc memset + gcc
# auto-vec, no non-portable intrinsics).  For a portable binary that runs anywhere,
# drop -march=native (or use -mtune=generic) — byte-identical output, slightly slower.
MARCH="${ALLC_MARCH:--march=native}"
gcc -O3 $MARCH -ffp-contract=off -funroll-loops -DUSE_PTHREADS -pthread \
    -static -I"$TMPD" -DMODEL_BLOB_PATH="$BINP" \
    -o "$OUT" "$TMPD/onnx_runtime_nibble_allc.c" -lm
echo "built $OUT ($(stat -c%s "$OUT") bytes)"
rm -rf "$TMPD"
