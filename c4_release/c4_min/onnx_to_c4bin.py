"""Lower an ONNX graph to the flat binary the nibble C ONNX runtime consumes.

The nibble VM's ONNX export (``c4_min/export_onnx.py``) is a *vanilla* traced
decode-only transformer (softmax1 + ALiBi + SwiGLU, 207 nodes, 24 op types).
The in-repo C ONNX runtime (``vm/onnx_runtime_c4.c``) reads a *custom* compact
binary, not the ONNX protobuf, and only knew ~17 ops.  This module produces the
compact binary for the **nibble** graph so the (retargeted) C runtime
``c4_min/onnx_runtime_nibble.c`` can load + run it.

Design (kept deliberately simple so the C loader is C4-subset-friendly):

* **Tensors** are a flat table. Every graph *initializer* AND every ``Constant``
  node's value tensor is a table entry (we constant-fold ``Constant`` at export
  time, exactly as ONNX Runtime would — this is the *only* op we fold, and it is
  pure data, no compute).  Each tensor: dtype (0=float32, 1=int64), rank, dims,
  and the raw values (float32 as IEEE bits, int64 as two 32-bit halves).  This
  is the value table; runtime-produced tensors get empty slots appended by name.
* **Nodes** are the graph nodes *in topological order* (ONNX guarantees this),
  minus the folded ``Constant`` nodes.  Each node: op enum, input tensor indices,
  output tensor indices, and a small per-op attribute blob (perm, axis, to,
  upper, keepdims, axes, allowzero).
* The C runtime allocates every non-initializer tensor lazily (its shape is
  computed by the op that writes it — the shape-plumbing ops Shape/Gather/
  Unsqueeze/Concat/ConstantOfShape/Range produce the concrete int shapes the
  compute ops consume, so nothing is baked; feed any [B,S] token frame).

Binary layout (all ints little-endian int32 unless noted):

  magic         "NBL1"  (0x314C424E)
  n_tensors
  n_nodes
  input_tid              index of the 'tokens' input tensor
  output_tid            index of the 'logits' output tensor
  --- n_tensors times ---
    name_len, name[bytes]
    is_init               1 if a value is stored here, 0 if runtime-produced
    dtype                 0=float32, 1=int64   (runtime tensors: writer sets it)
    rank, dims[rank]      (runtime tensors: 0, filled at run time)
    n_elems               number of stored elements (0 if not is_init)
    data[n_elems]         float32-bits (dtype0) or int64-as-2x-int32 (dtype1)
  --- n_nodes times ---
    op                    op enum (see OP_* below)
    n_in, in_tid[n_in]
    n_out, out_tid[n_out]
    attr_count, (attr_key, attr_nvals, attr_vals[nvals])*   (attr_key enum)
"""
from __future__ import annotations

import struct
from typing import Dict, List

import numpy as np
import onnx
from onnx import numpy_helper


# ---- op enum (must match onnx_runtime_nibble.c init_ops) --------------------
OP = {
    "MatMul": 0, "Add": 1, "Sub": 2, "Mul": 3, "Div": 4,
    "Gather": 5, "Reshape": 6, "Transpose": 7, "Concat": 8, "Unsqueeze": 9,
    "Shape": 10, "Cast": 11, "Exp": 12, "Neg": 13, "Abs": 14, "Range": 15,
    "ReduceMax": 16, "ReduceSum": 17, "Clip": 18, "Trilu": 19,
    "ConstantOfShape": 20, "Sigmoid": 21, "Identity": 22,
}

# ---- attribute-key enum (must match C) -------------------------------------
# Integer-valued attributes are keyed here.  ConstantOfShape's float fill value
# is carried in a dedicated int32 field (its IEEE-754 bit pattern) via key
# COS_FILL_BITS so the C loader can reconstruct the exact float (always -inf for
# the causal mask, but kept general).
ATTR = {
    "perm": 0, "axis": 1, "to": 2, "upper": 3, "keepdims": 4,
    "axes": 5, "allowzero": 6, "cos_fill_bits": 7,
}

MAGIC = 0x314C424E  # "NBL1"

DT_FLOAT = 0
DT_INT64 = 1


def _dtype_code(onnx_dtype: int) -> int:
    if onnx_dtype == onnx.TensorProto.FLOAT:
        return DT_FLOAT
    if onnx_dtype in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
        return DT_INT64
    raise ValueError(f"unsupported tensor dtype {onnx_dtype}")


class _Writer:
    def __init__(self):
        self.buf = bytearray()

    def i32(self, v: int):
        self.buf += struct.pack("<i", int(v))

    def i64(self, v: int):
        # stored as two int32 (lo, hi) — the C side reads two int32 into a long
        v = int(v)
        self.buf += struct.pack("<q", v)

    def f32(self, v: float):
        self.buf += struct.pack("<f", float(v))

    def f32_bulk(self, arr: np.ndarray):
        self.buf += np.ascontiguousarray(arr, dtype="<f4").tobytes()

    def i64_bulk(self, arr: np.ndarray):
        self.buf += np.ascontiguousarray(arr, dtype="<i8").tobytes()

    def raw(self, b: bytes):
        self.buf += b


def lower_onnx_to_bin(onnx_path: str, out_path: str,
                      sparse_min_numel: int = 4096) -> Dict:
    model = onnx.load(onnx_path)
    g = model.graph

    # ---- 1. build the tensor table --------------------------------------
    # tensor id (tid) is an index into this list. name -> tid.
    names: List[str] = []
    tid: Dict[str, int] = {}
    init_dtype: Dict[int, int] = {}
    init_dims: Dict[int, List[int]] = {}
    init_vals: Dict[int, np.ndarray] = {}   # tid -> flat array (float32 or int64)
    # COO-sparse initializers: large 99%-zero weights stored as (flat_idx, value)
    # pairs so the .nblbin is ~MB not ~GB. The C loader reconstructs the dense
    # tensor on load (dense-in-RAM, sparse-on-disk), byte-identical to dense.
    init_coo: Dict[int, tuple] = {}         # tid -> (flat_idx int64[], vals f32[])

    def intern(name: str) -> int:
        if name == "":
            return -1
        if name not in tid:
            tid[name] = len(names)
            names.append(name)
        return tid[name]

    def _store_init(i: int, arr: np.ndarray, dt: int) -> None:
        """Store initializer ``i`` dense, or COO-sparse iff it is big + sparse."""
        init_dtype[i] = dt
        init_dims[i] = list(arr.shape)
        flat = arr.astype(np.float32 if dt == DT_FLOAT else np.int64).ravel()
        numel = flat.size
        if dt == DT_FLOAT and numel >= sparse_min_numel:
            nz = np.nonzero(flat)[0]
            if nz.size * 3 < numel:              # genuinely sparse -> COO
                init_coo[i] = (nz.astype(np.int64), flat[nz].astype(np.float32))
                return
        init_vals[i] = flat

    # graph initializers (dense) + sparse_initializers (COO in the ONNX itself)
    for t in g.initializer:
        i = intern(t.name)
        _store_init(i, numpy_helper.to_array(t), _dtype_code(t.data_type))
    for st in g.sparse_initializer:
        # An ONNX ``SparseTensorProto`` holds a 1-D ``values`` tensor + an
        # ``indices`` tensor ([nnz, rank] COO coordinates or a flat [nnz] index)
        # plus the dense ``dims``.  ``numpy_helper.to_array`` only accepts a dense
        # ``TensorProto``, so we reconstruct the dense array by hand and let
        # ``_store_init`` re-COO it into the compact-binary flat-index format.
        name = st.values.name
        i = intern(name)
        dims = list(st.dims)
        vals = numpy_helper.to_array(st.values).reshape(-1)
        idx = numpy_helper.to_array(st.indices)
        dense = np.zeros(dims, dtype=vals.dtype)
        if idx.ndim == 2:                          # [nnz, rank] coordinates
            dense[tuple(idx.T)] = vals
        else:                                      # [nnz] linearized indices
            dense.reshape(-1)[idx.astype(np.int64)] = vals
        _store_init(i, dense, _dtype_code(st.values.data_type))

    # constant-fold Constant nodes into the tensor table.
    # NB: index by position, not id(node) — protobuf repeated-field iteration
    # yields fresh Python wrappers each pass, so id() is not stable across loops.
    folded = set()
    for ni, n in enumerate(g.node):
        if n.op_type == "Constant":
            out = n.output[0]
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    i = intern(out)
                    _store_init(i, arr, _dtype_code(a.t.data_type))
                    folded.add(ni)

    # intern every remaining node input/output (runtime tensors)
    for ni, n in enumerate(g.node):
        if ni in folded:
            continue
        for x in list(n.input) + list(n.output):
            intern(x)

    input_tid = intern(g.input[0].name)
    output_tid = intern(g.output[0].name)

    # ---- 2. serialize ----------------------------------------------------
    w = _Writer()
    w.i32(MAGIC)
    w.i32(len(names))
    real_nodes = [n for ni, n in enumerate(g.node) if ni not in folded]
    w.i32(len(real_nodes))
    w.i32(input_tid)
    w.i32(output_tid)

    # is_init flag: 0 = runtime tensor, 1 = dense initializer, 2 = COO-sparse
    # initializer (nnz index/value pairs; the C loader reconstructs dense).
    for i, name in enumerate(names):
        nb = name.encode("utf-8")
        w.i32(len(nb))
        w.raw(nb)
        if i in init_coo:
            idx, vals = init_coo[i]
            dims = init_dims[i]
            numel = int(np.prod(dims)) if dims else 1
            w.i32(2)              # is_init = 2 (sparse)
            w.i32(init_dtype[i])  # always DT_FLOAT for COO path
            w.i32(len(dims))
            for d in dims:
                w.i32(d)
            w.i32(numel)          # DENSE element count (loader allocs this)
            w.i32(len(vals))      # nnz
            w.i64_bulk(idx)       # nnz int64 flat indices
            w.f32_bulk(vals)      # nnz float32 values
        elif i in init_vals:
            dt = init_dtype[i]
            dims = init_dims[i]
            vals = init_vals[i]
            w.i32(1)              # is_init = 1 (dense)
            w.i32(dt)
            w.i32(len(dims))
            for d in dims:
                w.i32(d)
            w.i32(len(vals))
            if dt == DT_FLOAT:
                w.f32_bulk(vals)
            else:
                w.i64_bulk(vals)
        else:
            w.i32(0)     # runtime tensor
            w.i32(0)     # dtype placeholder (writer fills at runtime)
            w.i32(0)     # rank 0
            w.i32(0)     # n_elems 0

    for n in real_nodes:
        w.i32(OP[n.op_type])
        ins = [intern(x) for x in n.input]
        w.i32(len(ins))
        for x in ins:
            w.i32(x)
        outs = [intern(x) for x in n.output]
        w.i32(len(outs))
        for x in outs:
            w.i32(x)
        # attributes
        attrs = []
        for a in n.attribute:
            if a.name == "value" and n.op_type == "ConstantOfShape":
                fill = float(numpy_helper.to_array(a.t).ravel()[0])
                bits = struct.unpack("<i", struct.pack("<f", fill))[0]
                attrs.append((ATTR["cos_fill_bits"], [bits]))
                continue
            if a.name not in ATTR:
                continue
            if a.type == a.INT:
                attrs.append((ATTR[a.name], [a.i]))
            elif a.type == a.INTS:
                attrs.append((ATTR[a.name], list(a.ints)))
        w.i32(len(attrs))
        for key, vals in attrs:
            w.i32(key)
            w.i32(len(vals))
            for v in vals:
                w.i32(v)

    with open(out_path, "wb") as f:
        f.write(w.buf)

    return {
        "n_tensors": len(names),
        "n_nodes": len(real_nodes),
        "n_folded_constants": len(folded),
        "n_dense_inits": len(init_vals),
        "n_sparse_inits": len(init_coo),
        "sparse_nnz_stored": int(sum(v.size for _, v in init_coo.values())),
        "input_tid": input_tid,
        "output_tid": output_tid,
        "bytes": len(w.buf),
        "op_types": sorted({n.op_type for n in real_nodes}),
    }


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nibble_onnx/nibble_vm.onnx"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nibble_onnx/nibble_vm.nblbin"
    info = lower_onnx_to_bin(src, dst)
    print("lowered", src, "->", dst)
    for k, v in info.items():
        print(f"  {k}: {v}")
