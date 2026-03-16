"""
Sparse ONNX Export for Neural ALU using packed COO format.

The PureALU model is 99.9% sparse (only ~5K of 4.4M parameters are non-zero).
Standard ONNX stores all parameters densely, resulting in ~9 MB files.

This module provides:
1. export_sparse_coo() - Export weights in packed COO sparse format
2. load_sparse_coo() - Load and reconstruct sparse PyTorch model
3. verify_sparse_coo() - Verify sparse model produces same results

File format (packed COO):
- weights_coo.npz with 4 arrays:
  - names: tensor names as bytes
  - shapes: (num_tensors, 2) shapes
  - indices: concatenated flat indices (row * cols + col)
  - values: concatenated values
  - offsets: (num_tensors,) start offset for each tensor
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Tuple, Dict


def export_sparse_coo(
    output_path: str = "neural_alu_coo.npz",
    verbose: bool = True
) -> Tuple[str, Dict]:
    """
    Export PureALU weights in packed COO sparse format.

    Args:
        output_path: Path to save compressed sparse weights
        verbose: Print progress and statistics

    Returns:
        Tuple of (output_path, stats)
    """
    from .pure_alu import build_pure_alu

    if verbose:
        print("Building PureALU model...")
    model = build_pure_alu()
    model.eval()

    if verbose:
        print("Converting to packed COO sparse format...")

    # Collect all sparse data
    names = []
    shapes = []
    all_indices = []
    all_values = []
    offsets = [0]

    total_elements = 0
    nonzero_elements = 0

    for name, param in model.named_parameters():
        total_elements += param.numel()

        # Convert to sparse COO
        sparse = param.data.to_sparse()
        if sparse._nnz() > 0:
            idx = sparse._indices().numpy()  # (ndim, nnz)
            vals = sparse._values().numpy().astype(np.float32)
            shape = param.shape

            # Flatten indices: row * cols + col for 2D
            if len(shape) == 2:
                flat_idx = idx[0] * shape[1] + idx[1]
            else:
                flat_idx = idx[0]

            names.append(name)
            shapes.append(list(shape) + [0] * (2 - len(shape)))  # Pad to 2D
            all_indices.append(flat_idx.astype(np.int32))
            all_values.append(vals)
            offsets.append(offsets[-1] + len(vals))
            nonzero_elements += len(vals)

    # Pack into arrays
    names_bytes = '\n'.join(names).encode('utf-8')
    shapes_arr = np.array(shapes, dtype=np.int32)
    indices_arr = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    values_arr = np.concatenate(all_values) if all_values else np.array([], dtype=np.float32)
    offsets_arr = np.array(offsets[:-1], dtype=np.int32)  # Start offsets only

    np.savez_compressed(
        output_path,
        names=np.frombuffer(names_bytes, dtype=np.uint8),
        shapes=shapes_arr,
        indices=indices_arr,
        values=values_arr,
        offsets=offsets_arr
    )

    file_size = os.path.getsize(output_path)

    stats = {
        'num_tensors': len(names),
        'total_elements': total_elements,
        'nonzero_elements': nonzero_elements,
        'sparsity': 1 - nonzero_elements / total_elements,
        'file_size_kb': file_size / 1024,
    }

    if verbose:
        print(f"\nPacked COO Sparse Export Statistics:")
        print(f"  Number of tensors: {stats['num_tensors']}")
        print(f"  Total parameters: {stats['total_elements']:,}")
        print(f"  Non-zero parameters: {stats['nonzero_elements']:,}")
        print(f"  Sparsity: {stats['sparsity']*100:.1f}%")
        print(f"  File size: {stats['file_size_kb']:.2f} KB")
        print(f"  Exported to: {output_path}")

    return output_path, stats


def load_sparse_coo(weights_path: str) -> Dict[str, torch.Tensor]:
    """
    Load packed COO sparse weights and convert to sparse PyTorch tensors.

    Args:
        weights_path: Path to weights_coo.npz

    Returns:
        Dict mapping parameter names to sparse PyTorch tensors
    """
    data = np.load(weights_path)

    names = data['names'].tobytes().decode('utf-8').split('\n')
    shapes = data['shapes']
    indices = data['indices']
    values = data['values']
    offsets = data['offsets']

    sparse_tensors = {}
    for i, name in enumerate(names):
        shape = tuple(s for s in shapes[i] if s > 0)
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < len(offsets) else len(values)

        idx = indices[start:end].astype(np.int64)
        vals = values[start:end]

        # Unflatten indices
        if len(shape) == 2:
            rows = idx // shape[1]
            cols = idx % shape[1]
            idx_2d = np.stack([rows, cols])
        else:
            idx_2d = idx.reshape(1, -1)

        sparse = torch.sparse_coo_tensor(
            torch.from_numpy(idx_2d),
            torch.from_numpy(vals),
            shape
        )
        sparse_tensors[name] = sparse

    return sparse_tensors


def load_model_with_sparse_weights(weights_path: str):
    """
    Load PureALU model and populate with sparse weights.

    Args:
        weights_path: Path to weights_coo.npz

    Returns:
        PureALU model with loaded weights
    """
    from .pure_alu import build_pure_alu

    # Build model
    model = build_pure_alu()

    # Load sparse weights
    sparse_tensors = load_sparse_coo(weights_path)

    # Convert to dense and load into model
    state_dict = {}
    for name, sparse in sparse_tensors.items():
        state_dict[name] = sparse.to_dense()

    # Load weights
    model.load_state_dict(state_dict, strict=False)

    return model


def verify_sparse_coo(weights_path: str, verbose: bool = True) -> bool:
    """
    Verify sparse COO export produces same results as original model.

    Args:
        weights_path: Path to weights_coo.npz
        verbose: Print test results

    Returns:
        True if all tests pass
    """
    from .embedding import E, Opcode

    # Load model with sparse weights
    model = load_model_with_sparse_weights(weights_path)
    model.eval()

    # Test cases
    test_cases = [
        (Opcode.ADD, 12345, 67890, 80235, 'ADD'),
        (Opcode.SUB, 1000, 1, 999, 'SUB'),
        (Opcode.MUL, 100, 100, 10000, 'MUL'),
        (Opcode.DIV, 42, 6, 7, 'DIV'),
        (Opcode.MOD, 1000, 33, 10, 'MOD'),
        (Opcode.EQ, 42, 42, 1, 'EQ'),
        (Opcode.LT, 10, 20, 1, 'LT'),
    ]

    all_pass = True
    for opcode, a, b, expected, name in test_cases:
        # Create input
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.NIB_A] = float((a >> (4*i)) & 0xF)
            x[0, i, E.NIB_B] = float((b >> (4*i)) & 0xF)
            x[0, i, E.OP_START + opcode] = 1.0
            x[0, i, E.POS] = float(i)

        # Run model
        with torch.no_grad():
            y = model(x)

        # Extract result
        result = 0
        for i in range(E.NUM_POSITIONS):
            nib = int(round(y[0, i, E.RESULT].item()))
            nib = max(0, min(15, nib))
            result |= (nib << (4*i))

        ok = result == expected
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"  {name}({a}, {b}) = {result} (expected {expected}) [{status}]")

        if not ok:
            all_pass = False

    return all_pass


if __name__ == "__main__":
    print("=" * 60)
    print("COO Sparse Export for Neural ALU")
    print("=" * 60)

    output_path, stats = export_sparse_coo(
        output_path="neural_alu_coo.npz",
        verbose=True
    )

    print("\nVerifying sparse export...")
    success = verify_sparse_coo(output_path, verbose=True)
    print(f"\nVerification: {'PASS' if success else 'FAIL'}")
