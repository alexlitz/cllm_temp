#!/bin/bash
# Build CLLM (C4 LLM) utility programs
#
# Pipeline:
#   1. Exports transformer VM weights -> .c4onnx model (if needed)
#   2. Python bundler: compiles C source -> bytecode, bundles with model -> single C file
#   3. Compiles bundle with gcc -> native executable
#
# The full C4 Transformer VM is embedded in each executable.
# Every operation (add, multiply, compare, shift, etc.) flows through
# the model's neural weight matrices via matmul + softmax + activation.
#
# Usage:
#   ./build_cllm_utils.sh              # build all
#   ./build_cllm_utils.sh clean        # remove build artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"
GEN_DIR="${BUILD_DIR}/bundled"
CLLM_DIR="${SCRIPT_DIR}/cllm"
BUNDLER="${SCRIPT_DIR}/bundler/neural_bundler.py"
MODEL="${SCRIPT_DIR}/models/transformer_vm.c4onnx"
EXPORTER="${SCRIPT_DIR}/tools/export_vm_weights.py"

# Handle clean
if [ "${1:-}" = "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR"
    echo "Done."
    exit 0
fi

# Create build directories
mkdir -p "$BIN_DIR" "$GEN_DIR"

# Export model weights if needed
if [ ! -f "$MODEL" ]; then
    echo "Exporting transformer VM weights..."
    python3 "$EXPORTER" -o "$MODEL"
    echo ""
fi

# Discover utilities from cllm/ directory
UTILS=""
for f in "$CLLM_DIR"/*_cllm.c; do
    [ -f "$f" ] && UTILS="$UTILS $(basename "$f" .c)"
done

echo "=== CLLM Utilities Build ==="
echo "Model:   $MODEL ($(du -h "$MODEL" | awk '{print $1}'))"
echo "Bin dir: $BIN_DIR"
echo "Gen dir: $GEN_DIR"
echo ""

SUCCESS=0
FAIL=0

for util in $UTILS; do
    src="${CLLM_DIR}/${util}.c"
    name="${util%_cllm}"
    output="${BIN_DIR}/${name}-cllm"
    bundled="${GEN_DIR}/${util}_bundled.c"

    printf "  %-12s " "${name}-cllm"
    if python3 "$BUNDLER" "$MODEL" "$src" > "$bundled" 2>/dev/null; then
        if gcc -O2 -o "$output" "$bundled" -lm -w 2>/dev/null; then
            size=$(ls -lh "$output" | awk '{print $5}')
            bc_count=$(grep "program_code_len" "$bundled" | grep -o '[0-9]*' || echo "?")
            echo "OK  ${size}  (${bc_count} bytecode instructions)"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "FAIL (gcc error)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (compile/bundle error)"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Build Summary ==="
echo "  OK:     $SUCCESS"
echo "  Failed: $FAIL"
echo ""
echo "Executables:"
ls -lh "$BIN_DIR"/*-cllm 2>/dev/null | awk '{printf "  %-20s %s\n", $NF, $5}' || echo "  (none)"
echo ""
echo "Bundled C sources:"
ls -lh "$GEN_DIR"/*.c 2>/dev/null | awk '{printf "  %-40s %s\n", $NF, $5}' || echo "  (none)"
