import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath('/home/alexlitz/Documents/misc/c4_release/c4_release')))
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

import torch
from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm

print("[dynamo] Building production model via compile_full_vm()...")
model, _layout = compile_full_vm()
model.eval()
print(f"[dynamo] Model built. d_model={getattr(model, 'd_model', '?')}, "
      f"n_layers={len(model.blocks)}, vocab={model.vocab_size}, "
      f"max_seq_len={model.max_seq_len}")

token_ids = torch.arange(32, dtype=torch.long).unsqueeze(0) % min(model.vocab_size, 256)
print(f"[dynamo] Sample input shape: {tuple(token_ids.shape)}")

print("[dynamo] Sanity: eager forward...")
with torch.no_grad():
    eager_out = model(token_ids)
print(f"[dynamo] Eager forward OK. Output shape: {tuple(eager_out.shape)}")

print("[dynamo] Calling torch.onnx.export (dynamo=True)...")
try:
    onnx_program = torch.onnx.export(
        model,
        (token_ids,),
        "/home/alexlitz/Documents/misc/c4_release/.agent-logs/onnx-audit/c4_neural_vm_dynamo.onnx",
        opset_version=18,
        input_names=["token_ids"],
        output_names=["logits"],
        dynamic_shapes={"token_ids": {1: torch.export.Dim.DYNAMIC}},
        dynamo=True,
    )
    print("[dynamo] === EXPORT SUCCEEDED ===")
except Exception as e:
    print(f"[dynamo] === EXPORT FAILED ===")
    print(f"  Exception type: {type(e).__name__}")
    print(f"  Exception message: {e}")
    traceback.print_exc()
    sys.exit(1)
