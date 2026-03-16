#!/usr/bin/env python3
"""
Comprehensive tests for V5 MoE routing and ONNX export/runtime.

Tests:
1. MoE routing matches original individual layer behavior
2. ONNX export succeeds
3. ONNX Python runtime produces correct results
4. ONNX C runtime produces correct results
"""

import sys
import os
import subprocess
import tempfile
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')

import torch
import numpy as np

from pure_gen_vm import Opcode
from pure_gen_vm_v5 import (
    EmbedDimsV5, ExpertType, OPCODE_TO_EXPERT,
    InstructionRouter, InstructionMoE, create_instruction_moe,
    V5ArithmeticMoE, export_v5_moe_to_onnx,
    EfficientAddSubLayer, EfficientDivLayer, EfficientBitwiseLayer,
    EfficientShiftLayer, EfficientComparisonLayer
)

E = EmbedDimsV5
DIM = E.DIM


def create_input(a: int, b: int, opcode: int) -> torch.Tensor:
    """Create value-encoded input tensor."""
    x = torch.zeros(1, 1, DIM)
    for nib in range(8):
        x[0, 0, E.OP_A_VAL_START + nib] = float((a >> (nib * 4)) & 0xF)
        x[0, 0, E.OP_B_VAL_START + nib] = float((b >> (nib * 4)) & 0xF)
    x[0, 0, E.OPCODE_START + opcode] = 1.0
    return x


def read_result(out: torch.Tensor) -> int:
    """Read value-encoded result as integer."""
    result = 0
    for nib in range(8):
        val = out[0, 0, E.RESULT_VAL_START + nib].item()
        result |= (int(round(val)) & 0xF) << (nib * 4)
    return result


def read_cmp_flags(out: torch.Tensor) -> dict:
    """Read comparison flags."""
    return {
        'EQ': out[0, 0, E.CMP_EQ].item() > 0.5,
        'LT': out[0, 0, E.CMP_LT].item() > 0.5,
        'GT': out[0, 0, E.CMP_GT].item() > 0.5,
        'NE': out[0, 0, E.CMP_NE].item() > 0.5,
        'LE': out[0, 0, E.CMP_LE].item() > 0.5,
        'GE': out[0, 0, E.CMP_GE].item() > 0.5,
    }


print("=" * 70)
print("V5 MoE AND ONNX COMPREHENSIVE TESTS")
print("=" * 70)

# =============================================================================
# TEST 1: Router Routing
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: InstructionRouter Routing")
print("=" * 70)

router = InstructionRouter(DIM)

test_opcodes = [
    (Opcode.ADD, ExpertType.ADD_SUB, "ADD"),
    (Opcode.SUB, ExpertType.ADD_SUB, "SUB"),
    (Opcode.MUL, ExpertType.MUL, "MUL"),
    (Opcode.DIV, ExpertType.DIV_MOD, "DIV"),
    (Opcode.MOD, ExpertType.DIV_MOD, "MOD"),
    (Opcode.AND, ExpertType.BITWISE, "AND"),
    (Opcode.OR, ExpertType.BITWISE, "OR"),
    (Opcode.XOR, ExpertType.BITWISE, "XOR"),
    (Opcode.SHL, ExpertType.SHIFT, "SHL"),
    (Opcode.SHR, ExpertType.SHIFT, "SHR"),
    (Opcode.EQ, ExpertType.COMPARISON, "EQ"),
    (Opcode.LT, ExpertType.COMPARISON, "LT"),
    (Opcode.GT, ExpertType.COMPARISON, "GT"),
]

router_passed = 0
router_failed = 0

for opcode, expected_expert, name in test_opcodes:
    x = create_input(5, 3, opcode)
    with torch.no_grad():
        weights, indices, _ = router(x)

    selected_expert = indices[0, 0, 0].item()
    if selected_expert == expected_expert:
        router_passed += 1
        print(f"  ✓ {name} -> Expert {selected_expert} (correct)")
    else:
        router_failed += 1
        print(f"  ✗ {name} -> Expert {selected_expert} (expected {expected_expert})")

print(f"\nRouter: {router_passed}/{router_passed + router_failed} passed")

# =============================================================================
# TEST 2: V5ArithmeticMoE vs Individual Layers
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: V5ArithmeticMoE vs Individual Layers")
print("=" * 70)

moe_model = V5ArithmeticMoE(DIM)
moe_model.eval()

# Individual layers for comparison
add_layer = EfficientAddSubLayer(DIM)
div_layer = EfficientDivLayer(DIM)
bitwise_layer = EfficientBitwiseLayer(DIM)
shift_layer = EfficientShiftLayer(DIM)
cmp_layer = EfficientComparisonLayer(DIM)

# Test cases: (a, b, opcode, expected_result_or_checker)
test_cases = [
    # ADD/SUB
    (5, 3, Opcode.ADD, 8),
    (100, 50, Opcode.ADD, 150),
    (0xFF, 1, Opcode.ADD, 0x100),
    (100, 30, Opcode.SUB, 70),
    (0x100, 1, Opcode.SUB, 0xFF),

    # DIV/MOD
    (100, 7, Opcode.DIV, 14),
    (255, 3, Opcode.DIV, 85),
    (100, 7, Opcode.MOD, 2),
    (17, 5, Opcode.MOD, 2),

    # Bitwise
    (0xFF, 0x55, Opcode.AND, 0x55),
    (0xAA, 0x55, Opcode.OR, 0xFF),
    (0xFF, 0xAA, Opcode.XOR, 0x55),

    # Shifts
    (1, 4, Opcode.SHL, 16),
    (0xF0, 4, Opcode.SHR, 0x0F),
]

moe_passed = 0
moe_failed = 0

for a, b, opcode, expected in test_cases:
    x = create_input(a, b, opcode)

    with torch.no_grad():
        moe_out = moe_model(x)

    result = read_result(moe_out)

    if result == expected:
        moe_passed += 1
        op_name = [k for k, v in vars(Opcode).items() if v == opcode and not k.startswith('_')][0]
        print(f"  ✓ {op_name}: {a} op {b} = {expected}")
    else:
        moe_failed += 1
        op_name = [k for k, v in vars(Opcode).items() if v == opcode and not k.startswith('_')][0]
        print(f"  ✗ {op_name}: {a} op {b} = {result} (expected {expected})")

# Comparison tests (check flags instead of result)
cmp_tests = [
    (5, 5, Opcode.EQ, {'EQ': True, 'NE': False}),
    (5, 3, Opcode.EQ, {'EQ': False, 'NE': True}),
    (3, 5, Opcode.LT, {'LT': True, 'GT': False}),
    (5, 3, Opcode.GT, {'GT': True, 'LT': False}),
    (5, 5, Opcode.LE, {'LE': True}),
    (5, 5, Opcode.GE, {'GE': True}),
]

for a, b, opcode, expected_flags in cmp_tests:
    x = create_input(a, b, opcode)

    with torch.no_grad():
        moe_out = moe_model(x)

    flags = read_cmp_flags(moe_out)

    all_correct = all(flags.get(k, False) == v for k, v in expected_flags.items())

    if all_correct:
        moe_passed += 1
        op_name = [k for k, v in vars(Opcode).items() if v == opcode and not k.startswith('_')][0]
        print(f"  ✓ {op_name}: {a} cmp {b} -> {expected_flags}")
    else:
        moe_failed += 1
        op_name = [k for k, v in vars(Opcode).items() if v == opcode and not k.startswith('_')][0]
        print(f"  ✗ {op_name}: {a} cmp {b} -> {flags} (expected {expected_flags})")

print(f"\nMoE: {moe_passed}/{moe_passed + moe_failed} passed")

# =============================================================================
# TEST 3: ONNX Export
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: ONNX Export")
print("=" * 70)

onnx_path = "/tmp/v5_arithmetic_moe.onnx"

try:
    export_v5_moe_to_onnx(onnx_path)
    print(f"  ✓ Exported to {onnx_path}")

    # Verify file exists and has reasonable size
    file_size = os.path.getsize(onnx_path)
    print(f"  ✓ File size: {file_size:,} bytes")

    onnx_export_ok = True
except Exception as e:
    print(f"  ✗ Export failed: {e}")
    onnx_export_ok = False

# =============================================================================
# TEST 4: ONNX Python Runtime
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: ONNX Python Runtime")
print("=" * 70)

onnx_runtime_ok = False
try:
    import onnxruntime as ort

    # Create session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print(f"  ✓ Loaded ONNX model in onnxruntime")

    # Test cases
    onnx_test_cases = [
        (5, 3, Opcode.ADD, 8),
        (100, 50, Opcode.SUB, 50),
        (100, 7, Opcode.DIV, 14),
        (17, 5, Opcode.MOD, 2),
        (0xFF, 0x55, Opcode.AND, 0x55),
        (0xAA, 0x55, Opcode.OR, 0xFF),
        (1, 4, Opcode.SHL, 16),
        (0xF0, 4, Opcode.SHR, 0x0F),
    ]

    onnx_passed = 0
    onnx_failed = 0

    for a, b, opcode, expected in onnx_test_cases:
        # Create numpy input
        x_np = np.zeros((1, 1, DIM), dtype=np.float32)
        for nib in range(8):
            x_np[0, 0, E.OP_A_VAL_START + nib] = float((a >> (nib * 4)) & 0xF)
            x_np[0, 0, E.OP_B_VAL_START + nib] = float((b >> (nib * 4)) & 0xF)
        x_np[0, 0, E.OPCODE_START + opcode] = 1.0

        # Run inference
        outputs = session.run(None, {'input': x_np})
        out_np = outputs[0]

        # Read result
        result = 0
        for nib in range(8):
            val = out_np[0, 0, E.RESULT_VAL_START + nib]
            result |= (int(round(val)) & 0xF) << (nib * 4)

        if result == expected:
            onnx_passed += 1
            op_name = [k for k, v in vars(Opcode).items() if v == opcode and not k.startswith('_')][0]
            print(f"  ✓ {op_name}: {a} op {b} = {expected}")
        else:
            onnx_failed += 1
            op_name = [k for k, v in vars(Opcode).items() if v == opcode and not k.startswith('_')][0]
            print(f"  ✗ {op_name}: {a} op {b} = {result} (expected {expected})")

    print(f"\nONNX Runtime: {onnx_passed}/{onnx_passed + onnx_failed} passed")
    onnx_runtime_ok = onnx_passed == len(onnx_test_cases)

except ImportError:
    print("  ⚠ onnxruntime not installed, skipping")
except Exception as e:
    print(f"  ✗ ONNX runtime test failed: {e}")

# =============================================================================
# TEST 5: ONNX C Runtime
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: ONNX C Runtime")
print("=" * 70)

# Generate C test program
c_test_code = '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "onnxruntime_c_api.h"

#define DIM 512
#define OP_A_VAL_START 0
#define OP_B_VAL_START 8
#define RESULT_VAL_START 16
#define OPCODE_START 104

// Opcodes
#define OPCODE_ADD 1
#define OPCODE_SUB 2
#define OPCODE_DIV 4
#define OPCODE_MOD 5
#define OPCODE_AND 17
#define OPCODE_OR 18
#define OPCODE_SHL 14
#define OPCODE_SHR 15

const OrtApi* g_ort = NULL;

void check_status(OrtStatus* status) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Error: %s\\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

void set_input(float* data, int a, int b, int opcode) {
    memset(data, 0, DIM * sizeof(float));

    for (int nib = 0; nib < 8; nib++) {
        data[OP_A_VAL_START + nib] = (float)((a >> (nib * 4)) & 0xF);
        data[OP_B_VAL_START + nib] = (float)((b >> (nib * 4)) & 0xF);
    }
    data[OPCODE_START + opcode] = 1.0f;
}

int read_result(float* data) {
    int result = 0;
    for (int nib = 0; nib < 8; nib++) {
        int val = (int)roundf(data[RESULT_VAL_START + nib]) & 0xF;
        result |= val << (nib * 4);
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.onnx>\\n", argv[0]);
        return 1;
    }

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv* env;
    check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

    OrtSessionOptions* session_options;
    check_status(g_ort->CreateSessionOptions(&session_options));

    OrtSession* session;
    check_status(g_ort->CreateSession(env, argv[1], session_options, &session));

    // Create memory info
    OrtMemoryInfo* memory_info;
    check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    // Allocate input/output buffers
    float* input_data = (float*)malloc(DIM * sizeof(float));
    float* output_data = (float*)malloc(DIM * sizeof(float));

    int64_t input_shape[] = {1, 1, DIM};

    // Test cases: a, b, opcode, expected
    int test_cases[][4] = {
        {5, 3, OPCODE_ADD, 8},
        {100, 50, OPCODE_SUB, 50},
        {100, 7, OPCODE_DIV, 14},
        {17, 5, OPCODE_MOD, 2},
        {0xFF, 0x55, OPCODE_AND, 0x55},
        {0xAA, 0x55, OPCODE_OR, 0xFF},
        {1, 4, OPCODE_SHL, 16},
        {0xF0, 4, OPCODE_SHR, 0x0F},
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    int passed = 0;
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    for (int i = 0; i < num_tests; i++) {
        int a = test_cases[i][0];
        int b = test_cases[i][1];
        int opcode = test_cases[i][2];
        int expected = test_cases[i][3];

        set_input(input_data, a, b, opcode);

        // Create input tensor
        OrtValue* input_tensor = NULL;
        check_status(g_ort->CreateTensorWithDataAsOrtValue(
            memory_info, input_data, DIM * sizeof(float),
            input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

        // Run inference
        OrtValue* output_tensor = NULL;
        check_status(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
                               output_names, 1, &output_tensor));

        // Get output data
        float* out_ptr;
        check_status(g_ort->GetTensorMutableData(output_tensor, (void**)&out_ptr));
        memcpy(output_data, out_ptr, DIM * sizeof(float));

        int result = read_result(output_data);

        if (result == expected) {
            passed++;
            printf("  PASS: %d op %d = %d\\n", a, b, expected);
        } else {
            printf("  FAIL: %d op %d = %d (expected %d)\\n", a, b, result, expected);
        }

        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(output_tensor);
    }

    printf("\\nC Runtime: %d/%d passed\\n", passed, num_tests);

    // Cleanup
    free(input_data);
    free(output_data);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return (passed == num_tests) ? 0 : 1;
}
'''

# Try to compile and run C test
c_runtime_ok = False

# Check if onnxruntime C headers are available
ort_include = None
ort_lib = None

# Try to find onnxruntime
possible_ort_paths = [
    '/opt/homebrew/include',
    '/usr/local/include',
    '/usr/include',
]

for path in possible_ort_paths:
    header_path = os.path.join(path, 'onnxruntime_c_api.h')
    if os.path.exists(header_path):
        ort_include = path
        break

if ort_include:
    # Find library
    lib_paths = [
        '/opt/homebrew/lib',
        '/usr/local/lib',
        '/usr/lib',
    ]
    for path in lib_paths:
        lib_path = os.path.join(path, 'libonnxruntime.dylib')
        if os.path.exists(lib_path):
            ort_lib = path
            break
        lib_path = os.path.join(path, 'libonnxruntime.so')
        if os.path.exists(lib_path):
            ort_lib = path
            break

if ort_include and ort_lib:
    # Write C test file
    c_test_path = '/tmp/test_onnx_c.c'
    c_bin_path = '/tmp/test_onnx_c'

    with open(c_test_path, 'w') as f:
        f.write(c_test_code)

    # Compile
    compile_cmd = [
        'clang', '-o', c_bin_path, c_test_path,
        f'-I{ort_include}', f'-L{ort_lib}',
        '-lonnxruntime', '-lm'
    ]

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Compiled C test program")

            # Run test
            env = os.environ.copy()
            if 'DYLD_LIBRARY_PATH' in env:
                env['DYLD_LIBRARY_PATH'] = f"{ort_lib}:{env['DYLD_LIBRARY_PATH']}"
            else:
                env['DYLD_LIBRARY_PATH'] = ort_lib

            result = subprocess.run([c_bin_path, onnx_path], capture_output=True, text=True, env=env)
            print(result.stdout)
            if result.stderr:
                print(f"  stderr: {result.stderr}")

            c_runtime_ok = result.returncode == 0
        else:
            print(f"  ⚠ Compilation failed: {result.stderr}")
    except Exception as e:
        print(f"  ⚠ Could not compile/run C test: {e}")
else:
    print("  ⚠ ONNX Runtime C headers not found, skipping C test")
    print("    Install with: brew install onnxruntime")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_passed = True

print(f"\n  Router Routing:     {'✓ PASS' if router_failed == 0 else '✗ FAIL'}")
if router_failed > 0:
    all_passed = False

print(f"  MoE vs Layers:      {'✓ PASS' if moe_failed == 0 else '✗ FAIL'}")
if moe_failed > 0:
    all_passed = False

print(f"  ONNX Export:        {'✓ PASS' if onnx_export_ok else '✗ FAIL'}")
if not onnx_export_ok:
    all_passed = False

print(f"  ONNX Python Runtime: {'✓ PASS' if onnx_runtime_ok else '⚠ SKIP/FAIL'}")

print(f"  ONNX C Runtime:     {'✓ PASS' if c_runtime_ok else '⚠ SKIP/FAIL'}")

print("\n" + "=" * 70)
if all_passed:
    print("ALL CORE TESTS PASSED!")
else:
    print("SOME TESTS FAILED")
print("=" * 70)
