#!/usr/bin/env python3
"""
Export C4TransformerVM lookup tables as standard ONNX files with SparseTensorProto.

Produces 7 individual .onnx files (one per primitive), each a self-contained
ONNX model with sparse weight initializers and standard ops (MatMul, Softmax,
Concat, Slice, Mul). These are real .onnx files viewable in Netron and
runnable with onnxruntime.

Usage:
    python export_vm_onnx.py -o models/sparse_onnx/
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx
from onnx import TensorProto, helper, numpy_helper


def make_weight_initializer(name, array):
    """Create a TensorProto initializer from a numpy weight matrix.

    Stores as dense float32 (standard ONNX convention).
    The underlying data is sparse but ONNX SparseTensorProto lacks a name
    field, so we use dense TensorProto for full onnxruntime/Netron compat.
    """
    return numpy_helper.from_array(array.astype(np.float32), name=name)


def make_dense_initializer(name, array):
    """Create a dense TensorProto initializer."""
    return numpy_helper.from_array(array.astype(np.float32), name=name)


def make_scalar_initializer(name, value):
    """Create a scalar float constant initializer."""
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name=name)


def sparsity(arr):
    """Return fraction of zeros."""
    flat = arr.flatten()
    return 1.0 - np.count_nonzero(flat) / len(flat)


def build_b2n_model(W1, W2):
    """b2n: byte[256] -> (hi[16], lo[16])

    Pure matmul + slice, no softmax.
    """
    # Inputs
    byte_in = helper.make_tensor_value_info("byte", TensorProto.FLOAT, [256])

    # Outputs
    hi_out = helper.make_tensor_value_info("hi", TensorProto.FLOAT, [16])
    lo_out = helper.make_tensor_value_info("lo", TensorProto.FLOAT, [16])

    # Weights
    w1_init = make_weight_initializer("W1", W1)  # [256, 256]
    w2_init = make_weight_initializer("W2", W2)  # [256, 32]

    # Nodes
    # hidden[256] = byte @ W1
    matmul1 = helper.make_node("MatMul", ["byte", "W1"], ["hidden"])
    # nibbles[32] = hidden @ W2
    matmul2 = helper.make_node("MatMul", ["hidden", "W2"], ["nibbles"])
    # hi = nibbles[0:16], lo = nibbles[16:32]
    slice_hi = helper.make_node("Slice", ["nibbles", "starts_0", "ends_16", "axes_0"], ["hi"])
    slice_lo = helper.make_node("Slice", ["nibbles", "starts_16", "ends_32", "axes_0"], ["lo"])

    # Slice constants (must be int64 for ONNX Slice op)
    starts_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts_0")
    ends_16 = numpy_helper.from_array(np.array([16], dtype=np.int64), name="ends_16")
    starts_16 = numpy_helper.from_array(np.array([16], dtype=np.int64), name="starts_16")
    ends_32 = numpy_helper.from_array(np.array([32], dtype=np.int64), name="ends_32")
    axes_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes_0")

    graph = helper.make_graph(
        [matmul1, matmul2, slice_hi, slice_lo],
        "b2n",
        [byte_in],
        [hi_out, lo_out],
        initializer=[w1_init, w2_init, starts_0, ends_16, starts_16, ends_32, axes_0],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def build_standard_2in_1out(name, W1, W2, in1_name, in1_size, in2_name, in2_size, out_name, out_size):
    """Standard pattern: Concat -> MatMul -> Mul(100) -> Softmax -> MatMul

    Two inputs, one output.
    """
    inp1 = helper.make_tensor_value_info(in1_name, TensorProto.FLOAT, [in1_size])
    inp2 = helper.make_tensor_value_info(in2_name, TensorProto.FLOAT, [in2_size])
    out = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [out_size])

    w1_init = make_weight_initializer("W1", W1)
    w2_init = make_weight_initializer("W2", W2)
    temp_init = make_scalar_initializer("const_100", 100.0)

    concat = helper.make_node("Concat", [in1_name, in2_name], ["concat_out"], axis=0)
    matmul1 = helper.make_node("MatMul", ["concat_out", "W1"], ["hidden"])
    mul = helper.make_node("Mul", ["hidden", "const_100"], ["scaled"])
    softmax = helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1)
    matmul2 = helper.make_node("MatMul", ["probs", "W2"], [out_name])

    graph = helper.make_graph(
        [concat, matmul1, mul, softmax, matmul2],
        name,
        [inp1, inp2],
        [out],
        initializer=[w1_init, w2_init, temp_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def build_n2b_model(W1, W2):
    """n2b: (hi[16], lo[16]) -> byte[256]"""
    return build_standard_2in_1out("n2b", W1, W2, "hi", 16, "lo", 16, "byte", 256)


def build_nib_and_model(W1, W2):
    """nib_and: (a[16], b[16]) -> r[16]"""
    return build_standard_2in_1out("nib_and", W1, W2, "a", 16, "b", 16, "r", 16)


def build_nib_or_model(W1, W2):
    """nib_or: (a[16], b[16]) -> r[16]"""
    return build_standard_2in_1out("nib_or", W1, W2, "a", 16, "b", 16, "r", 16)


def build_nib_xor_model(W1, W2):
    """nib_xor: (a[16], b[16]) -> r[16]"""
    return build_standard_2in_1out("nib_xor", W1, W2, "a", 16, "b", 16, "r", 16)


def build_nib_add_model(W1, W2_sum, W2_cout):
    """nib_add: (a[16], b[16], cin[2]) -> (sum[16], cout[2])"""
    inp_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [16])
    inp_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [16])
    inp_cin = helper.make_tensor_value_info("cin", TensorProto.FLOAT, [2])
    out_sum = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [16])
    out_cout = helper.make_tensor_value_info("cout", TensorProto.FLOAT, [2])

    w1_init = make_weight_initializer("W1", W1)
    w2s_init = make_weight_initializer("W2_sum", W2_sum)
    w2c_init = make_weight_initializer("W2_cout", W2_cout)
    temp_init = make_scalar_initializer("const_100", 100.0)

    concat = helper.make_node("Concat", ["a", "b", "cin"], ["concat_out"], axis=0)
    matmul1 = helper.make_node("MatMul", ["concat_out", "W1"], ["hidden"])
    mul = helper.make_node("Mul", ["hidden", "const_100"], ["scaled"])
    softmax = helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1)
    matmul_sum = helper.make_node("MatMul", ["probs", "W2_sum"], ["sum"])
    matmul_cout = helper.make_node("MatMul", ["probs", "W2_cout"], ["cout"])

    graph = helper.make_graph(
        [concat, matmul1, mul, softmax, matmul_sum, matmul_cout],
        "nib_add",
        [inp_a, inp_b, inp_cin],
        [out_sum, out_cout],
        initializer=[w1_init, w2s_init, w2c_init, temp_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def build_nib_mul_model(W1, W2_lo, W2_hi):
    """nib_mul: (a[16], b[16]) -> (lo[16], hi[16])"""
    inp_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [16])
    inp_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [16])
    out_lo = helper.make_tensor_value_info("lo", TensorProto.FLOAT, [16])
    out_hi = helper.make_tensor_value_info("hi", TensorProto.FLOAT, [16])

    w1_init = make_weight_initializer("W1", W1)
    w2lo_init = make_weight_initializer("W2_lo", W2_lo)
    w2hi_init = make_weight_initializer("W2_hi", W2_hi)
    temp_init = make_scalar_initializer("const_100", 100.0)

    concat = helper.make_node("Concat", ["a", "b"], ["concat_out"], axis=0)
    matmul1 = helper.make_node("MatMul", ["concat_out", "W1"], ["hidden"])
    mul = helper.make_node("Mul", ["hidden", "const_100"], ["scaled"])
    softmax = helper.make_node("Softmax", ["scaled"], ["probs"], axis=-1)
    matmul_lo = helper.make_node("MatMul", ["probs", "W2_lo"], ["lo"])
    matmul_hi = helper.make_node("MatMul", ["probs", "W2_hi"], ["hi"])

    graph = helper.make_graph(
        [concat, matmul1, mul, softmax, matmul_lo, matmul_hi],
        "nib_mul",
        [inp_a, inp_b],
        [out_lo, out_hi],
        initializer=[w1_init, w2lo_init, w2hi_init, temp_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def extract_weights():
    """Extract weight matrices from C4TransformerVM."""
    from src.transformer_vm import C4TransformerVM
    vm = C4TransformerVM()
    alu = vm.alu

    weights = {}
    # b2n
    weights["b2n_W1"] = alu.b2n.W1.numpy()
    weights["b2n_W2"] = alu.b2n.W2.numpy()
    # n2b
    weights["n2b_W1"] = alu.n2b.W1.numpy()
    weights["n2b_W2"] = alu.n2b.W2.numpy()
    # nib_add
    weights["nib_add_W1"] = alu.nib_add.W1.numpy()
    weights["nib_add_W2_sum"] = alu.nib_add.W2_sum.numpy()
    weights["nib_add_W2_cout"] = alu.nib_add.W2_cout.numpy()
    # nib_mul
    weights["nib_mul_W1"] = alu.nib_mul.W1.numpy()
    weights["nib_mul_W2_lo"] = alu.nib_mul.W2_lo.numpy()
    weights["nib_mul_W2_hi"] = alu.nib_mul.W2_hi.numpy()
    # nib_and
    weights["nib_and_W1"] = alu.nib_and.W1.numpy()
    weights["nib_and_W2"] = alu.nib_and.W2.numpy()
    # nib_or
    weights["nib_or_W1"] = alu.nib_or.W1.numpy()
    weights["nib_or_W2"] = alu.nib_or.W2.numpy()
    # nib_xor
    weights["nib_xor_W1"] = alu.nib_xor.W1.numpy()
    weights["nib_xor_W2"] = alu.nib_xor.W2.numpy()

    return weights


def export_all(output_dir, verbose=True):
    """Export all 7 ONNX models."""
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("Extracting weights from C4TransformerVM...")
    weights = extract_weights()

    models = {}

    # b2n
    m = build_b2n_model(weights["b2n_W1"], weights["b2n_W2"])
    models["b2n"] = m

    # n2b
    m = build_n2b_model(weights["n2b_W1"], weights["n2b_W2"])
    models["n2b"] = m

    # nib_add
    m = build_nib_add_model(weights["nib_add_W1"], weights["nib_add_W2_sum"], weights["nib_add_W2_cout"])
    models["nib_add"] = m

    # nib_mul
    m = build_nib_mul_model(weights["nib_mul_W1"], weights["nib_mul_W2_lo"], weights["nib_mul_W2_hi"])
    models["nib_mul"] = m

    # nib_and
    m = build_nib_and_model(weights["nib_and_W1"], weights["nib_and_W2"])
    models["nib_and"] = m

    # nib_or
    m = build_nib_or_model(weights["nib_or_W1"], weights["nib_or_W2"])
    models["nib_or"] = m

    # nib_xor
    m = build_nib_xor_model(weights["nib_xor_W1"], weights["nib_xor_W2"])
    models["nib_xor"] = m

    for name, model in models.items():
        path = os.path.join(output_dir, f"{name}.onnx")
        onnx.checker.check_model(model)
        onnx.save(model, path)
        fsize = os.path.getsize(path)

        if verbose:
            n_init = len(model.graph.initializer)
            print(f"  {name}.onnx: {fsize:,} bytes "
                  f"({n_init} initializers, {len(model.graph.node)} nodes)")

    if verbose:
        print(f"\nExported {len(models)} models to {output_dir}/")

    return models


def main():
    parser = argparse.ArgumentParser(description="Export C4TransformerVM to standard ONNX")
    parser.add_argument("-o", "--output", default="models/sparse_onnx",
                        help="Output directory for .onnx files")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    export_all(args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
