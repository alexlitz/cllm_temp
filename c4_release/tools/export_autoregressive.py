#!/usr/bin/env python3
"""
Export baked AutoregressiveVM to .arvm binary format.

Binary format:
  Header (28 bytes):
    magic      4B = 0x4D565241 ("ARVM")
    version    4B = 2
    vocab_size 4B
    d_model    4B
    n_layers   4B
    n_heads    4B
    ffn_hidden 4B

  Per tensor:
    storage_type  4B (0=dense, 1=sparse_COO)
    If dense:  count(4B) + count x float32
    If sparse: nnz(4B) + nnz x (index uint32 + value float32)

  Tensor order:
    embed.weight          [vocab_size x d_model]
    For each layer (0..n_layers-1):
      alibi_slopes        [n_heads]
      W_q                 [d_model x d_model]
      W_k                 [d_model x d_model]
      W_v                 [d_model x d_model]
      W_o                 [d_model x d_model]
      W_up                [hidden x d_model]
      b_up                [hidden]
      W_gate              [hidden x d_model]
      b_gate              [hidden]
      W_down              [d_model x hidden]
      b_down              [d_model]
    head.weight           [vocab_size x d_model]
    head.bias             [vocab_size]

Usage:
    from tools.export_autoregressive import export_autoregressive, load_arvm
    export_autoregressive(model, 'model.arvm')
    weights = load_arvm('model.arvm')
"""

import struct
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ARVM_MAGIC = 0x4D565241  # "ARVM" in little-endian
ARVM_VERSION = 2

STORAGE_DENSE = 0
STORAGE_SPARSE_COO = 1


def _write_u32(f, val):
    """Write 32-bit unsigned little-endian integer."""
    f.write(struct.pack('<I', val & 0xFFFFFFFF))


def _write_f32(f, val):
    """Write 32-bit float."""
    f.write(struct.pack('<f', val))


def write_tensor(f, tensor, sparse=True):
    """Write a single tensor to file.

    Args:
        f: file handle (binary write mode)
        tensor: numpy array (float32)
        sparse: if True, use sparse COO when >50% zeros
    """
    flat = tensor.flatten().astype(np.float32)
    count = len(flat)

    if sparse:
        nonzero_mask = flat != 0.0
        nnz = int(nonzero_mask.sum())
        # Use sparse if more than 50% zeros
        if nnz * 2 < count:
            _write_u32(f, STORAGE_SPARSE_COO)
            _write_u32(f, nnz)
            indices = np.where(nonzero_mask)[0].astype(np.uint32)
            values = flat[nonzero_mask]
            for idx, val in zip(indices, values):
                _write_u32(f, int(idx))
                _write_f32(f, float(val))
            return

    # Dense storage
    _write_u32(f, STORAGE_DENSE)
    _write_u32(f, count)
    f.write(flat.tobytes())


def export_autoregressive(model, path, sparse=True):
    """Export baked AutoregressiveVM to .arvm binary format.

    Args:
        model: AutoregressiveVM instance (with baked weights)
        path: output file path
        sparse: use sparse COO for tensors with >50% zeros
    """
    import torch

    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.num_heads
    ffn_hidden = model.blocks[0].ffn.W_up.shape[0]

    with open(path, 'wb') as f:
        # Header (28 bytes)
        _write_u32(f, ARVM_MAGIC)
        _write_u32(f, ARVM_VERSION)
        _write_u32(f, model.vocab_size)
        _write_u32(f, model.d_model)
        _write_u32(f, n_layers)
        _write_u32(f, n_heads)
        _write_u32(f, ffn_hidden)

        # embed.weight [vocab_size x d_model]
        write_tensor(f, model.embed.weight.detach().cpu().numpy(), sparse)

        # Per-layer weights
        for i in range(n_layers):
            block = model.blocks[i]
            attn = block.attn
            ffn = block.ffn

            # alibi_slopes [n_heads]
            write_tensor(f, attn.alibi_slopes.detach().cpu().numpy(), sparse=False)

            # Attention weights
            write_tensor(f, attn.W_q.detach().cpu().numpy(), sparse)
            write_tensor(f, attn.W_k.detach().cpu().numpy(), sparse)
            write_tensor(f, attn.W_v.detach().cpu().numpy(), sparse)
            write_tensor(f, attn.W_o.detach().cpu().numpy(), sparse)

            # FFN weights (SwiGLU)
            write_tensor(f, ffn.W_up.detach().cpu().numpy(), sparse)
            write_tensor(f, ffn.b_up.detach().cpu().numpy(), sparse)
            write_tensor(f, ffn.W_gate.detach().cpu().numpy(), sparse)
            write_tensor(f, ffn.b_gate.detach().cpu().numpy(), sparse)
            write_tensor(f, ffn.W_down.detach().cpu().numpy(), sparse)
            write_tensor(f, ffn.b_down.detach().cpu().numpy(), sparse)

        # head.weight [vocab_size x d_model] and head.bias [vocab_size]
        write_tensor(f, model.head.weight.detach().cpu().numpy(), sparse)
        write_tensor(f, model.head.bias.detach().cpu().numpy(), sparse)

    file_size = os.path.getsize(path)
    print(f"Exported {path}: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  vocab_size={model.vocab_size}, d_model={model.d_model}, "
          f"n_layers={n_layers}, n_heads={n_heads}, ffn_hidden={ffn_hidden}")


def _read_u32(f):
    """Read 32-bit unsigned little-endian integer."""
    data = f.read(4)
    if len(data) < 4:
        raise ValueError("Unexpected end of file")
    return struct.unpack('<I', data)[0]


def _read_f32(f):
    """Read 32-bit float."""
    data = f.read(4)
    if len(data) < 4:
        raise ValueError("Unexpected end of file")
    return struct.unpack('<f', data)[0]


def _read_tensor(f):
    """Read a single tensor from file. Returns numpy float32 array (flat)."""
    storage_type = _read_u32(f)

    if storage_type == STORAGE_DENSE:
        count = _read_u32(f)
        data = f.read(count * 4)
        return np.frombuffer(data, dtype=np.float32).copy()

    elif storage_type == STORAGE_SPARSE_COO:
        nnz = _read_u32(f)
        # We don't know the full size yet - caller will reshape
        # Read index-value pairs
        indices = np.zeros(nnz, dtype=np.uint32)
        values = np.zeros(nnz, dtype=np.float32)
        for i in range(nnz):
            indices[i] = _read_u32(f)
            values[i] = _read_f32(f)
        return (indices, values, nnz)

    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


def _expand_sparse(data, total_size):
    """Expand sparse COO tuple to dense array."""
    if isinstance(data, tuple):
        indices, values, nnz = data
        result = np.zeros(total_size, dtype=np.float32)
        for i in range(nnz):
            result[indices[i]] = values[i]
        return result
    return data


def load_arvm(path):
    """Load .arvm file and return dict of numpy arrays.

    Returns:
        dict with keys:
            'vocab_size', 'd_model', 'n_layers', 'n_heads',
            'embed_weight',
            'layers': list of dicts with attention + FFN weights,
            'head_weight', 'head_bias'
    """
    with open(path, 'rb') as f:
        magic = _read_u32(f)
        if magic != ARVM_MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{ARVM_MAGIC:08X})")
        version = _read_u32(f)
        if version not in (1, 2):
            raise ValueError(f"Unsupported version: {version}")

        vocab_size = _read_u32(f)
        d_model = _read_u32(f)
        n_layers = _read_u32(f)
        n_heads = _read_u32(f)
        if version >= 2:
            hidden = _read_u32(f)
        else:
            hidden = d_model * 4

        result = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'ffn_hidden': hidden,
        }

        # embed.weight
        data = _read_tensor(f)
        result['embed_weight'] = _expand_sparse(data, vocab_size * d_model).reshape(vocab_size, d_model)

        # Per-layer weights
        layers = []
        for i in range(n_layers):
            layer = {}

            data = _read_tensor(f)
            layer['alibi_slopes'] = _expand_sparse(data, n_heads)

            data = _read_tensor(f)
            layer['W_q'] = _expand_sparse(data, d_model * d_model).reshape(d_model, d_model)
            data = _read_tensor(f)
            layer['W_k'] = _expand_sparse(data, d_model * d_model).reshape(d_model, d_model)
            data = _read_tensor(f)
            layer['W_v'] = _expand_sparse(data, d_model * d_model).reshape(d_model, d_model)
            data = _read_tensor(f)
            layer['W_o'] = _expand_sparse(data, d_model * d_model).reshape(d_model, d_model)

            data = _read_tensor(f)
            layer['W_up'] = _expand_sparse(data, hidden * d_model).reshape(hidden, d_model)
            data = _read_tensor(f)
            layer['b_up'] = _expand_sparse(data, hidden)

            data = _read_tensor(f)
            layer['W_gate'] = _expand_sparse(data, hidden * d_model).reshape(hidden, d_model)
            data = _read_tensor(f)
            layer['b_gate'] = _expand_sparse(data, hidden)

            data = _read_tensor(f)
            layer['W_down'] = _expand_sparse(data, d_model * hidden).reshape(d_model, hidden)
            data = _read_tensor(f)
            layer['b_down'] = _expand_sparse(data, d_model)

            layers.append(layer)

        result['layers'] = layers

        # head.weight and head.bias
        data = _read_tensor(f)
        result['head_weight'] = _expand_sparse(data, vocab_size * d_model).reshape(vocab_size, d_model)
        data = _read_tensor(f)
        result['head_bias'] = _expand_sparse(data, vocab_size)

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export AutoregressiveVM to .arvm format')
    parser.add_argument('-o', '--output', default='model.arvm', help='Output file path')
    parser.add_argument('--dense', action='store_true', help='Force dense storage (no sparse COO)')
    args = parser.parse_args()

    import torch
    from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()
    export_autoregressive(model, args.output, sparse=not args.dense)
