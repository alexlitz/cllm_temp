"""Reference interpreter for the flat .nblbin graph (see onnx_to_c4bin.py).

This is a *pure-numpy* executor of the exact same node list + tensor table the C
runtime ``onnx_runtime_nibble.c`` runs.  It exists to (a) validate the binary
serialization, and (b) be the byte-for-byte semantic spec the C port must match:
every op here is written the way the C code implements it (float32, same reduce
order), so ``python nbl_bin_interp == C runtime`` and both == torch export.
"""
from __future__ import annotations

import struct
from typing import Dict, List

import numpy as np

OP_NAMES = [
    "MatMul", "Add", "Sub", "Mul", "Div", "Gather", "Reshape", "Transpose",
    "Concat", "Unsqueeze", "Shape", "Cast", "Exp", "Neg", "Abs", "Range",
    "ReduceMax", "ReduceSum", "Clip", "Trilu", "ConstantOfShape", "Sigmoid",
    "Identity",
]
ATTR_NAMES = ["perm", "axis", "to", "upper", "keepdims", "axes", "allowzero",
              "cos_fill_bits"]
MAGIC = 0x314C424E


class _Reader:
    def __init__(self, buf: bytes):
        self.b = buf
        self.p = 0

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.b, self.p)[0]
        self.p += 4
        return v

    def i64(self) -> int:
        v = struct.unpack_from("<q", self.b, self.p)[0]
        self.p += 8
        return v

    def f32(self) -> float:
        v = struct.unpack_from("<f", self.b, self.p)[0]
        self.p += 4
        return v

    def raw(self, n: int) -> bytes:
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def f32_bulk(self, n: int) -> np.ndarray:
        v = np.frombuffer(self.b, dtype="<f4", count=n, offset=self.p)
        self.p += 4 * n
        return v.astype(np.float32)

    def i64_bulk(self, n: int) -> np.ndarray:
        v = np.frombuffer(self.b, dtype="<i8", count=n, offset=self.p)
        self.p += 8 * n
        return v.astype(np.int64)


class Graph:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            r = _Reader(f.read())
        assert r.i32() == MAGIC, "bad magic"
        n_tensors = r.i32()
        n_nodes = r.i32()
        self.input_tid = r.i32()
        self.output_tid = r.i32()

        self.names: List[str] = []
        self.is_init: List[int] = []
        self.init_dtype: List[int] = []
        self.init_dims: List[List[int]] = []
        self.init_vals: List[np.ndarray] = []
        for _ in range(n_tensors):
            nl = r.i32()
            self.names.append(r.raw(nl).decode("utf-8"))
            isi = r.i32()
            self.is_init.append(isi)
            if isi == 1:
                dt = r.i32()
                rank = r.i32()
                dims = [r.i32() for _ in range(rank)]
                ne = r.i32()
                if dt == 0:
                    arr = r.f32_bulk(ne)
                else:
                    arr = r.i64_bulk(ne)
                self.init_dtype.append(dt)
                self.init_dims.append(dims)
                self.init_vals.append(arr.reshape(dims) if dims else arr.reshape(()))
            elif isi == 2:
                # COO-sparse float initializer: dense numel, nnz, idx[], vals[]
                dt = r.i32()
                rank = r.i32()
                dims = [r.i32() for _ in range(rank)]
                ne = r.i32()          # dense numel
                nnz = r.i32()
                idx = r.i64_bulk(nnz)
                vals = r.f32_bulk(nnz)
                arr = np.zeros(ne, dtype=np.float32)
                arr[idx] = vals
                self.init_dtype.append(dt)
                self.init_dims.append(dims)
                self.init_vals.append(arr.reshape(dims) if dims else arr.reshape(()))
            else:
                r.i32(); r.i32(); r.i32()
                self.init_dtype.append(-1)
                self.init_dims.append([])
                self.init_vals.append(None)

        self.nodes = []
        for _ in range(n_nodes):
            op = r.i32()
            nin = r.i32()
            ins = [r.i32() for _ in range(nin)]
            nout = r.i32()
            outs = [r.i32() for _ in range(nout)]
            nattr = r.i32()
            attrs: Dict[str, List[int]] = {}
            for _ in range(nattr):
                key = r.i32()
                nv = r.i32()
                vals = [r.i32() for _ in range(nv)]
                attrs[ATTR_NAMES[key]] = vals
            self.nodes.append((op, ins, outs, attrs))

    def run(self, tokens: np.ndarray) -> np.ndarray:
        # value store keyed by tid
        vals: Dict[int, np.ndarray] = {}
        for i, isi in enumerate(self.is_init):
            if isi:
                vals[i] = self.init_vals[i]
        vals[self.input_tid] = tokens.astype(np.int64)

        for op, ins, outs, attrs in self.nodes:
            name = OP_NAMES[op]
            a = [vals[i] if i >= 0 else None for i in ins]
            if name == "MatMul":
                r = np.matmul(a[0], a[1])
            elif name == "Add":
                r = a[0] + a[1]
            elif name == "Sub":
                r = a[0] - a[1]
            elif name == "Mul":
                r = a[0] * a[1]
            elif name == "Div":
                r = a[0] / a[1]
            elif name == "Gather":
                axis = attrs.get("axis", [0])[0]
                r = np.take(a[0], a[1].astype(np.int64), axis=axis)
            elif name == "Reshape":
                shape = [int(x) for x in a[1].tolist()]
                r = a[0].reshape(shape)
            elif name == "Transpose":
                perm = attrs["perm"]
                r = np.transpose(a[0], perm)
            elif name == "Concat":
                axis = attrs.get("axis", [0])[0]
                r = np.concatenate([np.atleast_1d(x) for x in a], axis=axis)
            elif name == "Unsqueeze":
                axes = [int(x) for x in a[1].tolist()]
                r = a[0]
                for ax in sorted(axes):
                    r = np.expand_dims(r, ax)
            elif name == "Shape":
                r = np.array(a[0].shape, dtype=np.int64)
            elif name == "Cast":
                to = attrs["to"][0]
                r = a[0].astype(np.float32 if to == 1 else np.int64)
            elif name == "Exp":
                r = np.exp(a[0])
            elif name == "Neg":
                r = -a[0]
            elif name == "Abs":
                r = np.abs(a[0])
            elif name == "Range":
                start = int(a[0]); limit = int(a[1]); delta = int(a[2])
                r = np.arange(start, limit, delta, dtype=np.int64)
            elif name == "ReduceMax":
                axes = attrs.get("axes", [-1])
                kd = attrs.get("keepdims", [1])[0]
                r = np.max(a[0], axis=tuple(axes), keepdims=bool(kd))
            elif name == "ReduceSum":
                axes = [int(x) for x in a[1].tolist()] if len(a) > 1 else [-1]
                kd = attrs.get("keepdims", [1])[0]
                r = np.sum(a[0], axis=tuple(axes), keepdims=bool(kd))
            elif name == "Clip":
                lo = a[1] if len(a) > 1 and a[1] is not None else None
                hi = a[2] if len(a) > 2 and a[2] is not None else None
                r = a[0]
                if lo is not None:
                    r = np.maximum(r, lo)
                if hi is not None:
                    r = np.minimum(r, hi)
            elif name == "Trilu":
                upper = attrs.get("upper", [1])[0]
                k = int(a[1]) if len(a) > 1 and a[1] is not None else 0
                if upper:
                    r = np.triu(a[0], k)
                else:
                    r = np.tril(a[0], k)
            elif name == "ConstantOfShape":
                shape = [int(x) for x in a[0].tolist()]
                bits = attrs.get("cos_fill_bits", [0])[0]
                fill = struct.unpack("<f", struct.pack("<i", bits))[0]
                r = np.full(shape, fill, dtype=np.float32)
            elif name == "Sigmoid":
                r = 1.0 / (1.0 + np.exp(-a[0]))
            elif name == "Identity":
                r = a[0]
            else:
                raise ValueError(f"unknown op {name}")
            vals[outs[0]] = r
        return vals[self.output_tid]
