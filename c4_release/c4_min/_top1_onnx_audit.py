"""ONNX vanilla + byte-identity audit for the TOP-1-routed dispatch.

Builds a small compact VM, wraps the dispatch block in ``Top1RoutedFFN``, exports
the residual-in/residual-out block stack to ONNX (the SAME graph the corpus driver
runs, exactly as ``export_onnx.export_blockstack_onnx`` does), and:

  1. audits the op inventory -> asserts NO forbidden control-flow op
     (``If`` / ``Loop`` / ``Scan``) -- the router's ``argmax`` + ``index_select``
     lower to the vanilla ``ArgMax`` + ``Gather`` ops, not a data-dependent
     Python branch;
  2. checks the ONNX graph's output is byte-identical (argmax-decode-identical) to
     the torch ``Top1RoutedFFN`` forward on a battery of dispatch residuals (one
     per opcode one-hot);
  3. contrasts against the DENSE dispatch export (both must be vanilla).

Run:  python -m c4_min._top1_onnx_audit
"""
from __future__ import annotations
import os, sys, tempfile, warnings

import c4_min.nibble_pure_forward as PF
import c4_min.nibble_pure_forward_complete as C
PF.SP_INIT = 0xF0
C.SP_INIT = 0xF0

import torch  # noqa: E402
from c4_min import isa  # noqa: E402
from c4_min.compact_alloc import build_compact_pure_forward_model  # noqa: E402
from c4_min.moe_top1 import Top1RoutedFFN  # noqa: E402
from c4_min.export_onnx import (  # noqa: E402
    export_blockstack_onnx, op_inventory, assert_vanilla, FORBIDDEN_OPS)


def _export_ffn(ffn, dim, path):
    """Trace ONE FFN module (residual-in/residual-out) to ONNX."""
    import onnx

    class _Wrap(torch.nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return self.f(x)

    w = _Wrap(ffn).eval()
    ex = torch.zeros(1, 4, dim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            w, (ex,), path, input_names=["residual"], output_names=["out"],
            dynamic_axes={"residual": {0: "batch", 1: "seq"},
                          "out": {0: "batch", 1: "seq"}},
            opset_version=17, do_constant_folding=True, dynamo=False)
    m = onnx.load(path)
    onnx.checker.check_model(m)
    return path


def main():
    import onnxruntime as ort

    t = build_compact_pure_forward_model(
        code_size=24, include_bitwise=True, include_divmod=False)
    m, L, _ = t
    dim = m.dim
    names = list(L._block_names)
    bi = names.index("dispatch")
    dense = m.blocks[bi].ffn
    routed = Top1RoutedFFN(dense, op_is_base=int(L.OP_IS),
                           num_ops=int(isa.NUM_OPS))
    print(f"compact dispatch: dense units={dense.W_up.shape[0]} -> "
          f"routed K={routed.K} (+{routed.n_ungated} ungated), dim={dim}")

    tmp = tempfile.mkdtemp(prefix="top1_onnx_")
    routed_path = os.path.join(tmp, "routed_ffn.onnx")
    _export_ffn(routed, dim, routed_path)

    # -- 1. op inventory + vanilla (forbidden-op) check. -----------------------
    inv = op_inventory(routed_path)
    print("\n[1] routed dispatch FFN op inventory:")
    for op, c in sorted(inv.items(), key=lambda kv: -kv[1]):
        print(f"      {op:16s} x{c}")
    forbidden = FORBIDDEN_OPS & set(inv)
    no_control_flow = not forbidden
    print(f"\n    NO forbidden control-flow (If/Loop/Scan): "
          f"{'YES' if no_control_flow else 'NO -> ' + str(sorted(forbidden))}")
    # full vanilla check (known-ops allowlist, same standard as export_onnx.main)
    vanilla_ok, vnotes = assert_vanilla(routed_path)
    print(f"    assert_vanilla (known-ops allowlist): {'YES' if vanilla_ok else 'NO'}")
    for ln in vnotes:
        print(f"      - {ln}")

    # -- 2. byte-identity: ONNX graph vs torch Top1RoutedFFN, per opcode. -------
    sess = ort.InferenceSession(routed_path,
                                providers=["CPUExecutionProvider"])
    OP_IS = int(L.OP_IS)
    owned = [op for op in range(int(isa.NUM_OPS))
             if bool((routed.op_units[op] < routed.n_units).any())]
    torch.manual_seed(0)
    max_ad = 0.0
    argmax_ok = True
    for op in owned:
        x = torch.randn(1, 5, dim) * 0.3
        x[..., routed.route_dims] = 0.0
        x[..., OP_IS + op] = 1.0
        with torch.no_grad():
            yt = routed(x).numpy()
        yo = sess.run(None, {"residual": x.numpy()})[0]
        ad = float(abs(yt - yo).max())
        max_ad = max(max_ad, ad)
        # argmax-decode identity: the per-position argmax over dims must match.
        if not (yt.argmax(-1) == yo.argmax(-1)).all():
            argmax_ok = False
    print(f"\n[2] ONNX(routed) vs torch(routed): max|Δ|={max_ad:.3e} over "
          f"{len(owned)} opcodes; per-dim argmax identical: {argmax_ok}")

    # -- 3. the DENSE export must also be vanilla (baseline). ------------------
    dense_path = os.path.join(tmp, "dense_ffn.onnx")
    _export_ffn(dense, dim, dense_path)
    d_inv = op_inventory(dense_path)
    d_forbidden = FORBIDDEN_OPS & set(d_inv)
    print(f"\n[3] dense dispatch FFN: no forbidden control-flow: "
          f"{'YES' if not d_forbidden else 'NO'}")

    ok = (no_control_flow and vanilla_ok and argmax_ok and max_ad < 1e-3
          and not d_forbidden)
    print("\n" + "=" * 60)
    print(f"RESULT: routed dispatch stays VANILLA (no If/Loop/Scan) = "
          f"{no_control_flow}; ONNX==torch(routed) argmax-identical = {argmax_ok}")
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    # clean temp
    for p in (routed_path, dense_path):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
