"""ONNX export probe for the production Neural VM model.

Attempts to call ``torch.onnx.export`` on the model produced by the production
constructor (``compile_full_vm``). The goal is **not** to produce a usable
ONNX file but to enumerate the concrete blockers (Python control flow,
``.item()`` syncs, setattr-on-module state, dynamic-shape arithmetic that
TorchScript can't trace) so they can be triaged.

Today (2026-05-11), the following ONNX-relevant invariants hold for
``AutoregressiveVM.forward``:

* ``SoftMoEFFN`` routes via ``_soft_forward`` when
  ``torch.onnx.is_in_onnx_export()`` is True — no ``.item()`` on routing.
* ``FlattenedALUMul.forward`` and ``FlattenedDivMod.forward`` both gate
  on ``x[..., BD.OP_*].max().item() < 0.1`` (perf early-out). These
  guard out the multi-stage ALU pipeline, but the ``.item()`` is a tracing
  blocker.
* ``NeuralVMEmbedding._inject_mem_store`` and
  ``NeuralVMEmbedding._inject_mem_metadata`` walk the token stream in
  Python and call ``token_ids[b, i].item()`` per position.
* ``NeuralVMEmbedding._mem_history_end`` is Python ``int`` state read in
  ``_inject_mem_store``. It is **not** mutated inside forward, so the
  exporter treats it as a constant equal to whatever value is set at
  export time (default 0).

Usage::

    python -m c4_release.scripts.export_onnx                   # default opset 17, no dynamo
    python -m c4_release.scripts.export_onnx --opset 18
    python -m c4_release.scripts.export_onnx --output /tmp/foo.onnx

The script logs every export warning/error and exits non-zero on failure
(but always emits a summary).
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout

# Match the path convention used by sibling scripts (c4_release/scripts/*).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def build_production_model():
    """Build the production model via ``compile_full_vm`` (no runner)."""
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm

    model, _layout = compile_full_vm()
    model.eval()
    return model


def make_sample_tokens(model, *, seq_len: int = 200):
    """Construct a representative ``[1, seq_len]`` token tensor.

    Tracing only cares about shapes/dtypes; the content is a deterministic
    in-range mix so any token-content-dependent path gets a sample input.
    """
    vocab = min(model.vocab_size, 256)
    ids = torch.arange(seq_len, dtype=torch.long) % vocab
    return ids.unsqueeze(0)  # [1, seq_len]


def attempt_export(
    *,
    output_path: str,
    opset_version: int,
    seq_len: int,
    verbose: bool,
) -> int:
    print("[export_onnx] Building production model via compile_full_vm()...")
    model = build_production_model()
    print(f"[export_onnx] Model built. d_model={getattr(model, 'd_model', '?')}, "
          f"n_layers={len(model.blocks)}, vocab={model.vocab_size}, "
          f"max_seq_len={model.max_seq_len}")

    token_ids = make_sample_tokens(model, seq_len=seq_len)
    print(f"[export_onnx] Sample input shape: {tuple(token_ids.shape)} "
          f"(dtype={token_ids.dtype})")

    print("[export_onnx] Sanity: running eager forward...")
    with torch.no_grad():
        eager_out = model(token_ids)
    print(f"[export_onnx] Eager forward OK. Output shape: {tuple(eager_out.shape)}")

    captured_warnings: list[tuple[str, str, str, int]] = []

    def _warn_hook(message, category, filename, lineno, file=None, line=None):
        captured_warnings.append((category.__name__, str(message), filename, lineno))

    print(f"[export_onnx] Calling torch.onnx.export "
          f"(opset={opset_version}, dynamo=False)...")
    err_buf = io.StringIO()
    out_buf = io.StringIO()
    export_exc: Exception | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _warn_hook
        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                torch.onnx.export(
                    model,
                    (token_ids,),
                    output_path,
                    opset_version=opset_version,
                    input_names=["token_ids"],
                    output_names=["logits"],
                    dynamic_axes={
                        "token_ids": {1: "seq"},
                        "logits": {1: "seq"},
                    },
                    dynamo=False,
                )
        except Exception as e:  # noqa: BLE001
            export_exc = e

    print("\n[export_onnx] === captured stdout ===")
    print(out_buf.getvalue() or "(empty)")
    print("\n[export_onnx] === captured stderr ===")
    print(err_buf.getvalue() or "(empty)")

    print(f"\n[export_onnx] === captured warnings ({len(captured_warnings)}) ===")
    shown = len(captured_warnings) if verbose else min(30, len(captured_warnings))
    for i, (cat, msg, fn, ln) in enumerate(captured_warnings[:shown]):
        print(f"  [{i}] {cat} ({fn}:{ln}): {msg}")
    if shown < len(captured_warnings):
        print(f"  ... +{len(captured_warnings) - shown} more (use --verbose to list all)")

    if export_exc is not None:
        print("\n[export_onnx] === EXPORT FAILED ===")
        print(f"  Exception type: {type(export_exc).__name__}")
        print(f"  Exception message: {export_exc}")
        print("\n[export_onnx] Traceback:")
        traceback.print_exception(type(export_exc), export_exc, export_exc.__traceback__)
        return 1

    print("\n[export_onnx] === EXPORT SUCCEEDED ===")
    print(f"  Output written to: {output_path}")
    try:
        size_bytes = os.path.getsize(output_path)
        print(f"  File size: {size_bytes:,} bytes")
    except OSError:
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/c4_neural_vm.onnx",
                        help="ONNX output path (default: /tmp/c4_neural_vm.onnx)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--seq-len", type=int, default=200,
                        help="Sample seq length for export tracing (default: 200)")
    parser.add_argument("--verbose", action="store_true",
                        help="List all captured warnings (default: first 30)")
    args = parser.parse_args()
    return attempt_export(
        output_path=args.output,
        opset_version=args.opset,
        seq_len=args.seq_len,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
