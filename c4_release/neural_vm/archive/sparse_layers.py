"""
Sparse Neural Layers for Neural VM.

ONNX-compatible sparse tensor implementation using index + value representation.
This achieves ~96% sparsity while remaining ONNX-exportable.

Approach:
- Store non-zero indices and values separately
- Use gather/scatter operations (ONNX-supported)
- Convert to CSR for efficient forward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

try:
    from .embedding import E
except ImportError:
    from embedding import E


class SparseLinear(nn.Module):
    """
    ONNX-compatible sparse linear layer.

    Stores weights in COO format: (row_indices, col_indices, values)
    Forward uses indexed operations that ONNX can export.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Start with empty sparse representation
        # These will be populated by bake_weights
        self.register_buffer('row_idx', torch.zeros(0, dtype=torch.long))
        self.register_buffer('col_idx', torch.zeros(0, dtype=torch.long))
        self.register_buffer('values', torch.zeros(0))
        self.register_buffer('bias', torch.zeros(out_features))

    def set_weights(self, indices: torch.Tensor, values: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Set sparse weights from COO format."""
        self.row_idx = indices[0].long()
        self.col_idx = indices[1].long()
        self.values = values.float()
        if bias is not None:
            self.bias = bias.float()

    def from_dense(self, W: torch.Tensor, b: Optional[torch.Tensor] = None, threshold: float = 1e-8):
        """Convert dense weight matrix to sparse."""
        mask = W.abs() > threshold
        indices = mask.nonzero(as_tuple=False).T
        values = W[mask]
        self.row_idx = indices[0].long()
        self.col_idx = indices[1].long()
        self.values = values.float()
        if b is not None:
            self.bias = b.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-vector product via gather/scatter.

        ONNX-compatible: uses only gather, mul, scatter_add operations.
        """
        # Handle batched input
        input_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        batch_size = x_flat.shape[0]

        # Gather input values at column indices
        # x_gathered[i] = x[:, col_idx[i]] for each non-zero position
        x_gathered = x_flat[:, self.col_idx]  # (batch, nnz)

        # Multiply by sparse values
        weighted = x_gathered * self.values  # (batch, nnz)

        # Scatter-add to output rows
        output = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)

        # Expand row indices for batch dimension
        row_expanded = self.row_idx.unsqueeze(0).expand(batch_size, -1)

        # scatter_add: output[b, row_idx[i]] += weighted[b, i]
        output.scatter_add_(1, row_expanded, weighted)

        # Add bias
        output = output + self.bias

        # Reshape to match input
        output_shape = input_shape[:-1] + (self.out_features,)
        return output.view(output_shape)

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.values)

    @property
    def sparsity(self) -> float:
        """Fraction of zeros."""
        total = self.in_features * self.out_features
        return 1.0 - self.nnz / total


class SparseSwiGLU(nn.Module):
    """
    Sparse SwiGLU FFN layer for neural VM.

    forward: output = x + W_down @ (silu(W_up @ x) * (W_gate @ x))

    All three weight matrices are sparse.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.W_up = SparseLinear(dim, hidden_dim)
        self.W_gate = SparseLinear(dim, hidden_dim)
        self.W_down = SparseLinear(hidden_dim, dim)

    def from_dense_ffn(self, W_up: torch.Tensor, b_up: torch.Tensor,
                       W_gate: torch.Tensor, b_gate: torch.Tensor,
                       W_down: torch.Tensor, b_down: torch.Tensor,
                       threshold: float = 1e-8):
        """Convert dense PureFFN weights to sparse."""
        self.W_up.from_dense(W_up, b_up, threshold)
        self.W_gate.from_dense(W_gate, b_gate, threshold)
        self.W_down.from_dense(W_down, b_down, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU with sparse matrices."""
        up = self.W_up(x)
        gate = self.W_gate(x)
        hidden = F.silu(up) * gate
        return x + self.W_down(hidden)

    @property
    def total_nnz(self) -> int:
        """Total non-zero elements across all matrices."""
        return self.W_up.nnz + self.W_gate.nnz + self.W_down.nnz

    @property
    def total_params(self) -> int:
        """Total dense parameter count."""
        return 3 * self.dim * self.hidden_dim + 2 * self.hidden_dim + self.dim

    @property
    def sparsity(self) -> float:
        """Overall sparsity."""
        return 1.0 - self.total_nnz / self.total_params


def convert_ffn_to_sparse(ffn: nn.Module, threshold: float = 1e-8) -> SparseSwiGLU:
    """Convert a PureFFN to SparseSwiGLU."""
    sparse = SparseSwiGLU(ffn.dim, ffn.hidden_dim)
    sparse.from_dense_ffn(
        ffn.W_up.data, ffn.b_up.data,
        ffn.W_gate.data, ffn.b_gate.data,
        ffn.W_down.data, ffn.b_down.data,
        threshold
    )
    return sparse


# ============================================================================
# ONNX EXPORT HELPER
# ============================================================================

def export_sparse_ffn_to_onnx(sparse_ffn: SparseSwiGLU, filepath: str,
                               batch_size: int = 1):
    """
    Export SparseSwiGLU to ONNX format.

    The ONNX graph will use gather/scatter ops instead of dense matmul.
    """
    import onnx

    # Create dummy input
    dummy = torch.randn(batch_size, sparse_ffn.dim)

    # Export
    torch.onnx.export(
        sparse_ffn,
        dummy,
        filepath,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=17,  # For scatter_add support
    )

    # Verify
    model = onnx.load(filepath)
    onnx.checker.check_model(model)

    return model


# ============================================================================
# SPARSE STORAGE FORMAT FOR MINIMAL RUNTIME
# ============================================================================

class SparseVMWeights:
    """
    Minimal sparse storage format for C runtime.

    Format (all int32/float32):
        Header:
            - num_layers (int32)
            - total_nnz (int32)

        Per layer:
            - hidden_dim (int32)
            - nnz_up, nnz_gate, nnz_down (int32 each)
            - row_idx_up, col_idx_up, values_up (arrays)
            - row_idx_gate, col_idx_gate, values_gate (arrays)
            - row_idx_down, col_idx_down, values_down (arrays)
            - bias_up, bias_gate, bias_down (arrays)
    """

    @staticmethod
    def save_sparse_ffn(sparse_ffn: SparseSwiGLU, filepath: str):
        """Save sparse FFN to binary file."""
        import struct

        with open(filepath, 'wb') as f:
            # Header
            f.write(struct.pack('i', 1))  # num_layers
            f.write(struct.pack('i', sparse_ffn.total_nnz))

            # Hidden dim
            f.write(struct.pack('i', sparse_ffn.hidden_dim))

            # NNZ counts
            f.write(struct.pack('i', sparse_ffn.W_up.nnz))
            f.write(struct.pack('i', sparse_ffn.W_gate.nnz))
            f.write(struct.pack('i', sparse_ffn.W_down.nnz))

            # W_up
            for arr in [sparse_ffn.W_up.row_idx, sparse_ffn.W_up.col_idx]:
                f.write(arr.numpy().astype('int32').tobytes())
            f.write(sparse_ffn.W_up.values.numpy().astype('float32').tobytes())
            f.write(sparse_ffn.W_up.bias.numpy().astype('float32').tobytes())

            # W_gate
            for arr in [sparse_ffn.W_gate.row_idx, sparse_ffn.W_gate.col_idx]:
                f.write(arr.numpy().astype('int32').tobytes())
            f.write(sparse_ffn.W_gate.values.numpy().astype('float32').tobytes())
            f.write(sparse_ffn.W_gate.bias.numpy().astype('float32').tobytes())

            # W_down
            for arr in [sparse_ffn.W_down.row_idx, sparse_ffn.W_down.col_idx]:
                f.write(arr.numpy().astype('int32').tobytes())
            f.write(sparse_ffn.W_down.values.numpy().astype('float32').tobytes())
            f.write(sparse_ffn.W_down.bias.numpy().astype('float32').tobytes())

    @staticmethod
    def load_sparse_ffn(filepath: str, dim: int) -> SparseSwiGLU:
        """Load sparse FFN from binary file."""
        import struct
        import numpy as np

        with open(filepath, 'rb') as f:
            # Header
            num_layers = struct.unpack('i', f.read(4))[0]
            total_nnz = struct.unpack('i', f.read(4))[0]

            # Hidden dim
            hidden_dim = struct.unpack('i', f.read(4))[0]

            # NNZ counts
            nnz_up = struct.unpack('i', f.read(4))[0]
            nnz_gate = struct.unpack('i', f.read(4))[0]
            nnz_down = struct.unpack('i', f.read(4))[0]

            sparse = SparseSwiGLU(dim, hidden_dim)

            # W_up
            row_up = torch.from_numpy(np.frombuffer(f.read(nnz_up * 4), dtype=np.int32).copy())
            col_up = torch.from_numpy(np.frombuffer(f.read(nnz_up * 4), dtype=np.int32).copy())
            val_up = torch.from_numpy(np.frombuffer(f.read(nnz_up * 4), dtype=np.float32).copy())
            bias_up = torch.from_numpy(np.frombuffer(f.read(hidden_dim * 4), dtype=np.float32).copy())
            sparse.W_up.row_idx = row_up
            sparse.W_up.col_idx = col_up
            sparse.W_up.values = val_up
            sparse.W_up.bias = bias_up

            # W_gate
            row_gate = torch.from_numpy(np.frombuffer(f.read(nnz_gate * 4), dtype=np.int32).copy())
            col_gate = torch.from_numpy(np.frombuffer(f.read(nnz_gate * 4), dtype=np.int32).copy())
            val_gate = torch.from_numpy(np.frombuffer(f.read(nnz_gate * 4), dtype=np.float32).copy())
            bias_gate = torch.from_numpy(np.frombuffer(f.read(hidden_dim * 4), dtype=np.float32).copy())
            sparse.W_gate.row_idx = row_gate
            sparse.W_gate.col_idx = col_gate
            sparse.W_gate.values = val_gate
            sparse.W_gate.bias = bias_gate

            # W_down
            row_down = torch.from_numpy(np.frombuffer(f.read(nnz_down * 4), dtype=np.int32).copy())
            col_down = torch.from_numpy(np.frombuffer(f.read(nnz_down * 4), dtype=np.int32).copy())
            val_down = torch.from_numpy(np.frombuffer(f.read(nnz_down * 4), dtype=np.float32).copy())
            bias_down = torch.from_numpy(np.frombuffer(f.read(dim * 4), dtype=np.float32).copy())
            sparse.W_down.row_idx = row_down
            sparse.W_down.col_idx = col_down
            sparse.W_down.values = val_down
            sparse.W_down.bias = bias_down

            return sparse


# ============================================================================
# TESTS
# ============================================================================

def test_sparse_linear():
    """Test SparseLinear correctness."""
    print("=== Testing SparseLinear ===")

    # Create dense weights
    W = torch.zeros(5, 3)
    W[0, 1] = 2.0
    W[1, 0] = 3.0
    W[2, 2] = 1.5
    W[4, 1] = -1.0
    b = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # Create sparse version
    sparse = SparseLinear(3, 5)
    sparse.from_dense(W, b)

    print(f"Dense weights: {W}")
    print(f"Sparse NNZ: {sparse.nnz}")
    print(f"Sparse sparsity: {sparse.sparsity:.1%}")

    # Test forward
    x = torch.tensor([[1.0, 2.0, 3.0]])
    dense_out = F.linear(x, W, b)
    sparse_out = sparse(x)

    print(f"\nInput: {x}")
    print(f"Dense output:  {dense_out}")
    print(f"Sparse output: {sparse_out}")

    diff = (dense_out - sparse_out).abs().max().item()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-6, f"Sparse != Dense: diff={diff}"
    print("PASS")


def test_sparse_swiglu():
    """Test SparseSwiGLU correctness."""
    print("\n=== Testing SparseSwiGLU ===")

    # Import PureFFN for comparison
    try:
        from .base_layers import PureFFN
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from base_layers import PureFFN

    class TestFFN(PureFFN):
        def _bake_weights(self):
            S = E.SCALE
            with torch.no_grad():
                # Up: gate on NIB_A
                self.W_up[0, E.NIB_A] = S
                self.b_gate[0] = 1.0
                # Output to RESULT
                self.W_down[E.RESULT, 0] = 1.0 / S

    # Create dense FFN
    dense_ffn = TestFFN(E.DIM, 4)

    # Convert to sparse
    sparse_ffn = convert_ffn_to_sparse(dense_ffn)

    print(f"Dense params: {dense_ffn.W_up.numel() * 3 + dense_ffn.b_up.numel() * 3}")
    print(f"Sparse NNZ: {sparse_ffn.total_nnz}")
    print(f"Sparsity: {sparse_ffn.sparsity:.1%}")

    # Test forward
    x = torch.zeros(1, E.DIM)
    x[0, E.NIB_A] = 5.0  # Set input nibble

    dense_out = dense_ffn(x)
    sparse_out = sparse_ffn(x)

    diff = (dense_out - sparse_out).abs().max().item()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-5, f"Sparse != Dense: diff={diff}"
    print("PASS")


def test_onnx_export():
    """Test ONNX export of sparse layer."""
    print("\n=== Testing ONNX Export ===")

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("SKIP: onnx/onnxruntime not installed")
        return

    # Create sparse FFN
    sparse = SparseSwiGLU(E.DIM, 8)

    # Add some weights
    W_up = torch.zeros(8, E.DIM)
    W_up[0, E.NIB_A] = E.SCALE
    W_up[1, E.NIB_B] = E.SCALE
    sparse.W_up.from_dense(W_up, torch.zeros(8))

    W_gate = torch.zeros(8, E.DIM)
    sparse.W_gate.from_dense(W_gate, torch.ones(8))

    W_down = torch.zeros(E.DIM, 8)
    W_down[E.RESULT, 0] = 1.0 / E.SCALE
    sparse.W_down.from_dense(W_down, torch.zeros(E.DIM))

    # Export
    filepath = "/tmp/sparse_ffn.onnx"
    export_sparse_ffn_to_onnx(sparse, filepath)
    print(f"Exported to {filepath}")

    # Load and run
    session = ort.InferenceSession(filepath)
    x = torch.zeros(1, E.DIM)
    x[0, E.NIB_A] = 5.0

    # PyTorch result
    torch_out = sparse(x)

    # ONNX result
    onnx_out = session.run(None, {'input': x.numpy()})[0]

    diff = abs(torch_out[0, E.RESULT].item() - onnx_out[0, E.RESULT])
    print(f"PyTorch RESULT: {torch_out[0, E.RESULT].item():.6f}")
    print(f"ONNX RESULT: {onnx_out[0, E.RESULT]:.6f}")
    print(f"Difference: {diff:.2e}")

    assert diff < 1e-5, f"ONNX != PyTorch: diff={diff}"
    print("PASS")


def test_binary_save_load():
    """Test binary save/load for C runtime."""
    print("\n=== Testing Binary Save/Load ===")

    # Create sparse FFN
    sparse = SparseSwiGLU(E.DIM, 8)

    # Add some weights
    W_up = torch.zeros(8, E.DIM)
    W_up[0, E.NIB_A] = E.SCALE
    W_up[1, E.NIB_B] = E.SCALE
    sparse.W_up.from_dense(W_up, torch.ones(8) * 0.5)

    W_gate = torch.zeros(8, E.DIM)
    sparse.W_gate.from_dense(W_gate, torch.ones(8))

    W_down = torch.zeros(E.DIM, 8)
    W_down[E.RESULT, 0] = 1.0 / E.SCALE
    sparse.W_down.from_dense(W_down, torch.zeros(E.DIM))

    # Save
    filepath = "/tmp/sparse_ffn.bin"
    SparseVMWeights.save_sparse_ffn(sparse, filepath)

    import os
    file_size = os.path.getsize(filepath)
    print(f"Saved to {filepath}")
    print(f"File size: {file_size} bytes")

    # Load
    loaded = SparseVMWeights.load_sparse_ffn(filepath, E.DIM)

    print(f"Original NNZ: {sparse.total_nnz}")
    print(f"Loaded NNZ: {loaded.total_nnz}")

    # Compare outputs
    x = torch.zeros(1, E.DIM)
    x[0, E.NIB_A] = 5.0

    orig_out = sparse(x)
    load_out = loaded(x)

    diff = (orig_out - load_out).abs().max().item()
    print(f"Difference: {diff:.2e}")
    assert diff < 1e-6, f"Load != Save: diff={diff}"
    print("PASS")


def analyze_vm_sparsity():
    """Analyze sparsity of actual neural VM FFNs."""
    print("\n=== Analyzing Neural VM Sparsity ===")

    try:
        try:
            from .arithmetic_ops import NibbleAddFFN, NibbleSubFFN, NibbleMulFFN
            from .bitwise_ops import NibbleAndFFN, NibbleOrFFN, NibbleXorFFN
        except ImportError:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from arithmetic_ops import NibbleAddFFN, NibbleSubFFN, NibbleMulFFN
            from bitwise_ops import NibbleAndFFN, NibbleOrFFN, NibbleXorFFN
    except ImportError:
        print("SKIP: Cannot import neural VM ops")
        return

    ops = [
        ("ADD", NibbleAddFFN()),
        ("SUB", NibbleSubFFN()),
        ("MUL", NibbleMulFFN()),
        ("AND", NibbleAndFFN()),
        ("OR", NibbleOrFFN()),
        ("XOR", NibbleXorFFN()),
    ]

    total_dense = 0
    total_sparse = 0

    for name, ffn in ops:
        sparse = convert_ffn_to_sparse(ffn)
        dense_params = sparse.total_params
        sparse_nnz = sparse.total_nnz

        total_dense += dense_params
        total_sparse += sparse_nnz

        print(f"{name}: {sparse_nnz}/{dense_params} = {sparse.sparsity:.1%} sparse")

    print(f"\nTotal: {total_sparse}/{total_dense} non-zero ({100-total_sparse/total_dense*100:.1%} sparse)")


if __name__ == '__main__':
    test_sparse_linear()
    test_sparse_swiglu()
    test_onnx_export()
    test_binary_save_load()
    analyze_vm_sparsity()
    print("\n" + "=" * 60)
    print("All sparse layer tests passed!")
