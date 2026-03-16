"""
ONNX Export for Neural VM.

Exports the full VM step (or PureALU) to ONNX format for deployment.
The exported model runs pure neural forward passes with baked weights.
"""

import torch
import torch.nn as nn
import os
import sys

from .embedding import E, Opcode
from .pure_alu import PureALU
from .vm_step import build_vm_step, set_position_encoding, set_opcode, set_nib_a, set_immediate


def export_to_onnx(output_path: str = "neural_alu.onnx",
                   opset_version: int = 18,
                   verbose: bool = True,
                   full_vm_step: bool = False) -> str:
    """
    Export PureALU or full VM step to ONNX format.

    Args:
        output_path: Path for the output ONNX file
        opset_version: ONNX opset version
        verbose: Print export info
        full_vm_step: If True, export full VM step (register read/write + ALU + PC update).
                      If False, export just the PureALU.

    Returns:
        Path to exported ONNX file
    """
    if full_vm_step:
        if verbose:
            print("Building full VM step model...")
        model = build_vm_step(num_carry_iters=7)
        default_output = "neural_vm_step.onnx"
    else:
        if verbose:
            print("Building PureALU model...")
        model = PureALU()
        default_output = "neural_alu.onnx"

    if output_path == "neural_alu.onnx" and full_vm_step:
        output_path = default_output

    model.eval()

    # Create dummy input [batch, 8 nibbles, embedding dim]
    dummy_input = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

    # Set up a simple ADD operation for tracing
    for i in range(E.NUM_POSITIONS):
        dummy_input[0, i, E.NIB_A] = float(i)
        dummy_input[0, i, E.NIB_B] = float(i)
        dummy_input[0, i, E.OP_START + Opcode.ADD] = 1.0
        dummy_input[0, i, E.POS] = float(i)

    if verbose:
        print(f"Input shape: {dummy_input.shape}")
        print(f"Exporting to ONNX opset {opset_version}...")

    # Export to ONNX (fixed batch size for simplicity)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['embedding'],
        output_names=['output']
    )

    if verbose:
        print(f"Exported to: {output_path}")
        file_size = os.path.getsize(output_path)
        print(f"File size: {file_size / 1024:.1f} KB")

    return output_path


def verify_onnx_model(onnx_path: str, verbose: bool = True) -> bool:
    """
    Verify ONNX model loads and runs correctly.

    Args:
        onnx_path: Path to ONNX file
        verbose: Print verification info

    Returns:
        True if verification passes
    """
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("ONNX/ONNXRuntime not installed. Install with: pip install onnx onnxruntime")
        return False

    if verbose:
        print(f"\nVerifying {onnx_path}...")

    # Check model validity
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    if verbose:
        print("  ONNX model check passed")

    # Test with ONNX Runtime
    session = ort.InferenceSession(onnx_path)

    # Create test input: 5 + 3 = 8
    test_input = np.zeros((1, E.NUM_POSITIONS, E.DIM), dtype=np.float32)
    a, b = 5, 3
    for i in range(E.NUM_POSITIONS):
        test_input[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
        test_input[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
        test_input[0, i, E.OP_START + Opcode.ADD] = 1.0
        test_input[0, i, E.POS] = float(i)

    # Run inference
    outputs = session.run(None, {'embedding': test_input})
    result_embedding = outputs[0]

    # Extract result
    result = 0
    for i in range(E.NUM_POSITIONS):
        nib = int(round(result_embedding[0, i, E.RESULT]))
        nib = max(0, min(15, nib))
        result |= (nib << (i * 4))

    expected = a + b
    if result == expected:
        if verbose:
            print(f"  ONNX inference test: {a} + {b} = {result} (correct)")
        return True
    else:
        if verbose:
            print(f"  ONNX inference test: {a} + {b} = {result} (expected {expected})")
        return False


def compare_pytorch_onnx(onnx_path: str, num_tests: int = 10) -> None:
    """
    Compare PyTorch and ONNX outputs for consistency.
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("ONNXRuntime not installed")
        return

    print(f"\nComparing PyTorch vs ONNX ({num_tests} tests)...")

    # Load both models
    pytorch_model = PureALU()
    pytorch_model.eval()

    onnx_session = ort.InferenceSession(onnx_path)

    test_cases = [
        (Opcode.ADD, 5, 3),
        (Opcode.ADD, 255, 1),
        (Opcode.SUB, 10, 3),
        (Opcode.MUL, 6, 7),
        (Opcode.AND, 0xFF, 0xAA),
        (Opcode.OR, 0x55, 0xAA),
        (Opcode.XOR, 0xFF, 0xF0),
        (Opcode.EQ, 5, 5),
        (Opcode.EQ, 5, 6),
        (Opcode.LT, 3, 5),
    ][:num_tests]

    all_match = True
    for opcode, a, b in test_cases:
        # Create input
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
            x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
            x[0, i, E.OP_START + opcode] = 1.0
            x[0, i, E.POS] = float(i)

        # PyTorch inference
        with torch.no_grad():
            pytorch_out = pytorch_model(x)

        # ONNX inference
        onnx_out = onnx_session.run(None, {'embedding': x.numpy()})[0]

        # Extract results
        def extract_result(embedding):
            result = 0
            for i in range(E.NUM_POSITIONS):
                nib = int(round(embedding[0, i, E.RESULT]))
                nib = max(0, min(15, nib))
                result |= (nib << (i * 4))
            return result

        pytorch_result = extract_result(pytorch_out.numpy())
        onnx_result = extract_result(onnx_out)

        op_name = [k for k, v in vars(Opcode).items() if v == opcode][0]
        match = pytorch_result == onnx_result
        if not match:
            all_match = False

        status = "OK" if match else "MISMATCH"
        print(f"  {status}: {op_name}({a}, {b}) PyTorch={pytorch_result} ONNX={onnx_result}")

    if all_match:
        print("\nAll tests passed - PyTorch and ONNX outputs match!")
    else:
        print("\nSome tests failed - outputs differ!")


def main():
    """Export and verify ONNX model."""
    import argparse

    parser = argparse.ArgumentParser(description='Export Neural VM to ONNX')
    parser.add_argument('-o', '--output', default='neural_alu.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=18,
                        help='ONNX opset version')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported model')
    parser.add_argument('--compare', action='store_true',
                        help='Compare PyTorch vs ONNX outputs')

    args = parser.parse_args()

    # Export
    onnx_path = export_to_onnx(args.output, args.opset)

    # Verify
    if args.verify:
        success = verify_onnx_model(onnx_path)
        if not success:
            sys.exit(1)

    # Compare
    if args.compare:
        compare_pytorch_onnx(onnx_path)


if __name__ == "__main__":
    main()
