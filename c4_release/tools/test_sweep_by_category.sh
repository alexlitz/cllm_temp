#!/usr/bin/env bash
# Category-based test sweep across major test domains beyond smoke + 1096.
#
# Usage: bash tools/test_sweep_by_category.sh [out_dir]
#
# Each category is time-boxed (per-test --timeout). Results are written to
# <out_dir>/<category>.log so the matrix doc can be regenerated.

set -u

OUT="${1:-/tmp/cat_sweep}"
mkdir -p "$OUT"

run_cat () {
    local name="$1"
    local timeout="$2"
    shift 2
    local files=("$@")
    echo "=== $name ==="
    pytest "${files[@]}" --tb=line -q --timeout="$timeout" \
        > "$OUT/${name}.log" 2>&1
    local rc=$?
    echo "rc=$rc  -> $OUT/${name}.log"
}

run_cat allocator_contracts 60 \
    tests/test_attention_head_allocator.py \
    tests/test_ffn_unit_allocator.py \
    tests/test_dim_allocator.py

run_cat dsl_primitives 300 \
    tests/test_building_blocks_dsl.py \
    tests/test_compiler_ir.py \
    tests/test_ir_types.py \
    tests/test_compare_symbolic_to_lowered_attn.py

run_cat compile_contracts 600 \
    tests/test_compile_dynamic_byte_identical.py \
    tests/test_compile_dynamic_strict_mode.py \
    tests/test_compile_cross_step_safety.py \
    tests/test_compile_determinism.py

run_cat kv_cache_autoreg 600 \
    tests/test_autoregressive_kv_cache.py \
    tests/test_autoregressive_kv_cache_byte_identical.py \
    tests/test_batched_kv_eviction_validation.py

run_cat conversational_io 600 \
    tests/test_conversational_io.py \
    tests/test_conversational_io_comprehensive.py

run_cat architecture_toggles 60 \
    tests/test_architecture_toggles.py

run_cat slot_registry 60 \
    tests/test_slot_registry.py

run_cat misc_opcode_specific 300 \
    tests/test_bz_bnz_neural.py \
    tests/test_control_flow_neural.py \
    tests/test_addr_key_neural_decode.py \
    tests/test_alibi_mem_attn.py

echo
echo "Done. Logs in $OUT"
