"""
ONNX Export for C4 Transformer VM

Exports the core arithmetic modules to ONNX format for
deployment in non-Python environments.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from pathlib import Path


class SwiGLUMulONNX(nn.Module):
    """ONNX-compatible SwiGLU multiply."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute a * b using SwiGLU.

        a * b = silu(a) * b + silu(-a) * (-b)
        """
        return torch.nn.functional.silu(a) * b + torch.nn.functional.silu(-a) * (-b)


class ByteToNibbleONNX(nn.Module):
    """ONNX-compatible byte to nibble conversion."""

    def __init__(self):
        super().__init__()

        # W2: maps byte to two one-hot nibbles
        W2 = torch.zeros(256, 32)
        for b in range(256):
            high = (b >> 4) & 0xF
            low = b & 0xF
            W2[b, high] = 1.0
            W2[b, 16 + low] = 1.0

        self.register_buffer('W2', W2)

    def forward(self, byte_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert byte index to nibble one-hots.

        Args:
            byte_idx: Scalar tensor with byte value (0-255)

        Returns:
            (high_nibble, low_nibble): Each [16] one-hot
        """
        # Create one-hot
        byte_onehot = torch.zeros(256)
        byte_onehot[byte_idx] = 1.0

        # Map to nibbles
        nibbles = byte_onehot @ self.W2

        return nibbles[:16], nibbles[16:]


class NibbleTableONNX(nn.Module):
    """ONNX-compatible nibble operation table."""

    def __init__(self, table: torch.Tensor):
        """
        Args:
            table: [16, 16] table where table[a, b] = op(a, b)
        """
        super().__init__()

        # Flatten table for linear lookup
        W = torch.zeros(256, 16)
        for a in range(16):
            for b in range(16):
                k = a * 16 + b
                result = int(table[a, b].item()) & 0xF
                W[k, result] = 1.0

        self.register_buffer('W', W)

    def forward(self, a_idx: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute nibble operation.

        Args:
            a_idx, b_idx: Scalar tensors with nibble values (0-15)

        Returns:
            Result nibble as one-hot [16]
        """
        k = a_idx * 16 + b_idx

        # One-hot query
        query = torch.zeros(256)
        query[k] = 1.0

        return query @ self.W


class DivisionONNX(nn.Module):
    """ONNX-compatible integer division."""

    def __init__(self, table_bits: int = 8):
        super().__init__()

        table_size = 2 ** table_bits

        # Reciprocal table
        W_table = torch.zeros(table_size)
        for i in range(table_size):
            x = 0.5 + i / (2 * table_size)
            W_table[i] = 1.0 / x

        self.register_buffer('W_table', W_table)
        self.table_size = table_size
        self.mul = SwiGLUMulONNX()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute a // b.

        Args:
            a, b: Scalar integer tensors

        Returns:
            Integer division result
        """
        # Handle edge cases
        if b == 0:
            return torch.tensor(0)

        sign = torch.sign(a) * torch.sign(b)
        a = torch.abs(a).float()
        b = torch.abs(b).float()

        if a < b:
            return torch.tensor(0)
        if b == 1:
            return (a * sign).int()

        # Normalize b to [0.5, 1.0)
        b_float = b
        exp = 0
        while b_float >= 1.0:
            b_float = b_float * 0.5
            exp = exp + 1
        while b_float < 0.5:
            b_float = b_float * 2.0
            exp = exp - 1

        # Table lookup
        idx = ((b_float - 0.5) * 2 * self.table_size).int()
        idx = torch.clamp(idx, 0, self.table_size - 1)

        y = self.W_table[idx]

        # Newton iterations
        for _ in range(2):
            by = self.mul(b_float, y)
            two_minus_by = 2.0 - by
            y = self.mul(y, two_minus_by)

        # Scale back
        for _ in range(exp):
            y = y * 0.5

        # Multiply a * (1/b)
        result = self.mul(a, y)
        result_int = result.int()

        return result_int * sign


def export_swiglu_to_onnx(
    save_path: str,
    opset_version: int = 14,
) -> str:
    """
    Export SwiGLU multiplication to ONNX.

    Args:
        save_path: Path to save ONNX file
        opset_version: ONNX opset version

    Returns:
        Path to saved ONNX file
    """
    model = SwiGLUMulONNX()

    # Sample inputs
    a = torch.tensor(6.0)
    b = torch.tensor(7.0)

    # Export
    torch.onnx.export(
        model,
        (a, b),
        save_path,
        input_names=['a', 'b'],
        output_names=['result'],
        opset_version=opset_version,
        dynamic_axes={
            'a': {0: 'batch'},
            'b': {0: 'batch'},
            'result': {0: 'batch'},
        }
    )

    return save_path


def export_division_to_onnx(
    save_path: str,
    opset_version: int = 14,
) -> str:
    """
    Export division to ONNX.

    Note: Due to control flow, this exports a simplified version.
    """
    # For ONNX, we export a traced version without control flow
    class SimplifiedDivision(nn.Module):
        def __init__(self):
            super().__init__()
            self.mul = SwiGLUMulONNX()

            # Reciprocal table
            table_size = 256
            W_table = torch.zeros(table_size)
            for i in range(table_size):
                x = 0.5 + i / (2 * table_size)
                W_table[i] = 1.0 / x
            self.register_buffer('W_table', W_table)

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # Simplified division using table lookup
            # This is a demonstration; full division needs scripting
            idx = (b * 0.5).int().clamp(0, 255)
            reciprocal = self.W_table[idx]
            return self.mul(a, reciprocal).round()

    model = SimplifiedDivision()

    a = torch.tensor(100.0)
    b = torch.tensor(7.0)

    torch.onnx.export(
        model,
        (a, b),
        save_path,
        input_names=['a', 'b'],
        output_names=['result'],
        opset_version=opset_version,
    )

    return save_path


def export_full_vm_components(
    save_directory: str,
    opset_version: int = 14,
) -> List[str]:
    """
    Export all VM components to ONNX files.

    Args:
        save_directory: Directory to save ONNX files
        opset_version: ONNX opset version

    Returns:
        List of saved file paths
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Export SwiGLU
    swiglu_path = str(save_dir / "swiglu_mul.onnx")
    export_swiglu_to_onnx(swiglu_path, opset_version)
    saved_files.append(swiglu_path)

    # Export byte-to-nibble
    b2n = ByteToNibbleONNX()
    b2n_path = str(save_dir / "byte_to_nibble.onnx")
    try:
        torch.onnx.export(
            b2n,
            torch.tensor(0xAB),
            b2n_path,
            input_names=['byte'],
            output_names=['high_nibble', 'low_nibble'],
            opset_version=opset_version,
        )
        saved_files.append(b2n_path)
    except Exception as e:
        print(f"Warning: Could not export byte_to_nibble: {e}")

    # Export nibble AND table
    and_table = torch.zeros(16, 16)
    for a in range(16):
        for b in range(16):
            and_table[a, b] = a & b

    nib_and = NibbleTableONNX(and_table)
    and_path = str(save_dir / "nibble_and.onnx")
    try:
        torch.onnx.export(
            nib_and,
            (torch.tensor(0xF), torch.tensor(0xA)),
            and_path,
            input_names=['a', 'b'],
            output_names=['result'],
            opset_version=opset_version,
        )
        saved_files.append(and_path)
    except Exception as e:
        print(f"Warning: Could not export nibble_and: {e}")

    return saved_files


def verify_onnx_export(onnx_path: str) -> bool:
    """
    Verify an ONNX export is valid.

    Args:
        onnx_path: Path to ONNX file

    Returns:
        True if valid
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return True
    except ImportError:
        print("Install onnx package for verification: pip install onnx")
        return False
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False


def run_onnx_inference(onnx_path: str, inputs: dict) -> dict:
    """
    Run inference on an ONNX model.

    Args:
        onnx_path: Path to ONNX file
        inputs: Dict of input name -> numpy array

    Returns:
        Dict of output name -> numpy array
    """
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)

        # Convert inputs to numpy if needed
        np_inputs = {}
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                np_inputs[name] = value.numpy()
            else:
                np_inputs[name] = np.array(value, dtype=np.float32)

        outputs = session.run(None, np_inputs)

        output_names = [o.name for o in session.get_outputs()]
        return dict(zip(output_names, outputs))

    except ImportError:
        raise ImportError("Install onnxruntime: pip install onnxruntime")


__all__ = [
    'SwiGLUMulONNX',
    'ByteToNibbleONNX',
    'NibbleTableONNX',
    'DivisionONNX',
    'export_swiglu_to_onnx',
    'export_division_to_onnx',
    'export_full_vm_components',
    'verify_onnx_export',
    'run_onnx_inference',
]
