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
gcc -O3 -march=native -ffp-contract=off -funroll-loops -DUSE_PTHREADS -pthread \
    -static -I"$TMPD" -DMODEL_BLOB_PATH="$BINP" \
    -o "$OUT" "$TMPD/onnx_runtime_nibble_allc.c" -lm
echo "built $OUT ($(stat -c%s "$OUT") bytes)"
rm -rf "$TMPD"
