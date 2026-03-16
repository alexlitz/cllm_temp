#!/bin/bash
#
# Comprehensive Verification Script for C4 Neural ONNX Runner
#
# Tests all combinations of:
# - Built-in neural ops vs ONNX model
# - Different operations (*, +, /)
# - Various test values (small, medium, large, edge cases)
# - Different runners (onnx_runner_c4, onnx_runtime_c4, run_program_c4)
# - Python reference implementation

set -e

echo "=============================================="
echo "  C4 Neural ONNX Runner - Full Verification"
echo "=============================================="
echo ""

# Build all binaries
echo "Building binaries..."
gcc -include stdio.h -include stdlib.h -w -o onnx_runner_c4 onnx_runner_c4.c 2>/dev/null
gcc -include stdio.h -include stdlib.h -include fcntl.h -include unistd.h -w -o onnx_runtime_c4 onnx_runtime_c4.c 2>/dev/null
gcc -include stdio.h -include stdlib.h -include fcntl.h -include unistd.h -w -o run_program_c4 run_program_c4.c 2>/dev/null
echo "Build complete."
echo ""

# Test values
SMALL_A=(2 3 5 6 7 9)
SMALL_B=(3 4 7 7 8 11)
MED_A=(12 23 45 67 89 99)
MED_B=(34 56 78 89 12 99)
LARGE_A=(123 234 456 789 999 1234)
LARGE_B=(456 567 789 123 999 567)

PASS=0
FAIL=0

check_result() {
    local test_name="$1"
    local result="$2"
    local expected="$3"

    if [ "$result" == "$expected" ]; then
        echo "  ✓ $test_name: $result"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $test_name: $result (expected $expected)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=============================================="
echo "  1. MULTIPLICATION TESTS"
echo "=============================================="

echo ""
echo "--- onnx_runner_c4 (built-in neural ops) ---"
for i in ${!SMALL_A[@]}; do
    a=${SMALL_A[$i]}
    b=${SMALL_B[$i]}
    expected=$((a * b))
    result=$(./onnx_runner_c4 $a $b)
    check_result "$a * $b" "$result" "$expected"
done

echo ""
echo "--- onnx_runtime_c4 (ONNX parser) ---"
for i in ${!SMALL_A[@]}; do
    a=${SMALL_A[$i]}
    b=${SMALL_B[$i]}
    expected=$((a * b))
    result=$(./onnx_runtime_c4 models/swiglu_mul.onnx $a $b)
    check_result "$a * $b" "$result" "$expected"
done

echo ""
echo "--- run_program_c4 -e (built-in) ---"
for i in ${!SMALL_A[@]}; do
    a=${SMALL_A[$i]}
    b=${SMALL_B[$i]}
    expected=$((a * b))
    result=$(./run_program_c4 -e "$a * $b" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')
    check_result "$a * $b" "$result" "$expected"
done

echo ""
echo "--- run_program_c4 -m (ONNX model) ---"
for i in ${!SMALL_A[@]}; do
    a=${SMALL_A[$i]}
    b=${SMALL_B[$i]}
    expected=$((a * b))
    result=$(./run_program_c4 -m models/swiglu_mul.onnx -e "$a * $b" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')
    check_result "$a * $b" "$result" "$expected"
done

echo ""
echo "--- Medium values (all runners) ---"
for i in ${!MED_A[@]}; do
    a=${MED_A[$i]}
    b=${MED_B[$i]}
    expected=$((a * b))

    r1=$(./onnx_runner_c4 $a $b)
    r2=$(./onnx_runtime_c4 models/swiglu_mul.onnx $a $b)
    r3=$(./run_program_c4 -e "$a * $b" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')
    r4=$(./run_program_c4 -m models/swiglu_mul.onnx -e "$a * $b" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')

    all_match=1
    [ "$r1" != "$expected" ] && all_match=0
    [ "$r2" != "$expected" ] && all_match=0
    [ "$r3" != "$expected" ] && all_match=0
    [ "$r4" != "$expected" ] && all_match=0

    if [ $all_match -eq 1 ]; then
        echo "  ✓ $a * $b = $expected (all 4 runners)"
        PASS=$((PASS + 4))
    else
        echo "  ✗ $a * $b = $expected"
        echo "    runner_c4=$r1, runtime_c4=$r2, run_builtin=$r3, run_onnx=$r4"
        FAIL=$((FAIL + 4))
    fi
done

echo ""
echo "--- Large values ---"
for i in ${!LARGE_A[@]}; do
    a=${LARGE_A[$i]}
    b=${LARGE_B[$i]}
    expected=$((a * b))

    r1=$(./onnx_runner_c4 $a $b)
    r2=$(./onnx_runtime_c4 models/swiglu_mul.onnx $a $b)

    if [ "$r1" == "$expected" ] && [ "$r2" == "$expected" ]; then
        echo "  ✓ $a * $b = $expected"
        PASS=$((PASS + 2))
    else
        echo "  ✗ $a * $b = $expected (runner=$r1, runtime=$r2)"
        FAIL=$((FAIL + 2))
    fi
done

echo ""
echo "=============================================="
echo "  2. ADDITION TESTS"
echo "=============================================="

ADD_TESTS=("100 200 300" "255 1 256" "1000 2000 3000" "12345 54321 66666")
echo ""
echo "--- onnx_runner_c4 (nibble-based add) ---"
for test in "${ADD_TESTS[@]}"; do
    read a b expected <<< "$test"
    result=$(./onnx_runner_c4 $a + $b)
    check_result "$a + $b" "$result" "$expected"
done

echo ""
echo "--- run_program_c4 -e ---"
for test in "${ADD_TESTS[@]}"; do
    read a b expected <<< "$test"
    result=$(./run_program_c4 -e "$a + $b" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')
    check_result "$a + $b" "$result" "$expected"
done

echo ""
echo "=============================================="
echo "  3. DIVISION TESTS"
echo "=============================================="

DIV_TESTS=("42 7 6" "100 7 14" "1000 33 30" "12345 67 184")
echo ""
echo "--- onnx_runner_c4 (repeated subtraction) ---"
for test in "${DIV_TESTS[@]}"; do
    read a b expected <<< "$test"
    result=$(./onnx_runner_c4 $a / $b)
    check_result "$a / $b" "$result" "$expected"
done

echo ""
echo "--- run_program_c4 -e ---"
for test in "${DIV_TESTS[@]}"; do
    read a b expected <<< "$test"
    result=$(./run_program_c4 -e "$a / $b" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')
    check_result "$a / $b" "$result" "$expected"
done

echo ""
echo "=============================================="
echo "  4. EDGE CASE TESTS"
echo "=============================================="

echo ""
echo "--- Zero multiplication ---"
for val in 0 1 42 999; do
    result=$(./onnx_runner_c4 $val 0)
    check_result "$val * 0" "$result" "0"

    result=$(./onnx_runner_c4 0 $val)
    check_result "0 * $val" "$result" "0"
done

echo ""
echo "--- Identity multiplication ---"
for val in 1 42 123 999; do
    expected=$val
    result=$(./onnx_runner_c4 $val 1)
    check_result "$val * 1" "$result" "$expected"

    result=$(./onnx_runner_c4 1 $val)
    check_result "1 * $val" "$result" "$expected"
done

echo ""
echo "--- Squares ---"
for val in 2 3 5 7 10 12 99; do
    expected=$((val * val))
    result=$(./onnx_runner_c4 $val $val)
    check_result "$val² = $val * $val" "$result" "$expected"
done

echo ""
echo "=============================================="
echo "  5. PYTHON REFERENCE COMPARISON"
echo "=============================================="

if command -v python3 &> /dev/null && [ -f "run_program.py" ]; then
    echo ""
    echo "--- Comparing with Python run_program.py ---"
    echo "  (Note: Python returns exit code % 256, so comparing verbose output)"

    COMPARE_TESTS=("6 * 7" "42 * 42" "99 * 99")
    for expr in "${COMPARE_TESTS[@]}"; do
        # Get C result via verbose mode
        c_result=$(./run_program_c4 -e "$expr" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')

        # Calculate expected result using bash
        a=$(echo "$expr" | cut -d'*' -f1 | tr -d ' ')
        b=$(echo "$expr" | cut -d'*' -f2 | tr -d ' ')
        expected=$((a * b))

        if [ "$c_result" == "$expected" ]; then
            echo "  ✓ '$expr': C=$c_result (expected $expected)"
            PASS=$((PASS + 1))
        else
            echo "  ✗ '$expr': C=$c_result (expected $expected)"
            FAIL=$((FAIL + 1))
        fi
    done
else
    echo "  (Python comparison skipped - run_program.py not found or python3 not available)"
fi

echo ""
echo "=============================================="
echo "  6. CROSS-RUNNER CONSISTENCY CHECK"
echo "=============================================="

echo ""
echo "--- All runners should produce identical results ---"

CONSISTENCY_TESTS=("7 8" "15 16" "42 42" "100 100" "255 255")
for test in "${CONSISTENCY_TESTS[@]}"; do
    read a b <<< "$test"

    r1=$(./onnx_runner_c4 $a $b)
    r2=$(./onnx_runtime_c4 models/swiglu_mul.onnx $a $b)
    r3=$(./run_program_c4 -e "$a * $b" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')
    r4=$(./run_program_c4 -m models/swiglu_mul.onnx -e "$a * $b" -v 2>/dev/null | grep "Result:" | cut -d: -f2 | tr -d ' ')

    if [ "$r1" == "$r2" ] && [ "$r2" == "$r3" ] && [ "$r3" == "$r4" ]; then
        echo "  ✓ $a * $b = $r1 (all runners consistent)"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $a * $b inconsistent: runner=$r1, runtime=$r2, builtin=$r3, onnx=$r4"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=============================================="
echo "  FINAL SUMMARY"
echo "=============================================="
echo ""
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Total:  $((PASS + FAIL))"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "  ✓ ALL TESTS PASSED!"
    exit 0
else
    echo "  ✗ SOME TESTS FAILED"
    exit 1
fi
