"""Export the COMPACT pure-forward C4 VM to a VANILLA ONNX graph + drive it.

CHK-2 (checklist #3 / #14): the whole VM *step* — softmax1 + ALiBi attention and
SwiGLU FFN over the 42-block stack — exported to a single **standard-transformer**
ONNX graph (no ``Loop``/``Scan``/``If``, no custom / external-memory ops), then
run through ``onnxruntime`` byte-identically to the torch pure-forward model and
used as the forward inside the KV-cached per-step driver over the corpus.

Why this is now feasible: ``compact_alloc`` collapsed the naive ~30 GB dense
model (dim 2576, FFN padded to the global-max 21544) to the small dense form
(dim ~1702, per-block hidden), and ``sparse_forward`` proved the weights are
~99.97 % zero.  The compact dense dims are what a real transformer exports; the
99.97 %-zero weights are stored as ONNX ``sparse_initializer`` COO tensors so the
file is small — ORT reconstructs the dense weight for the ``MatMul`` so the
forward is byte-identical (this is exactly the ``nibble-onnx-sparse`` mechanism,
re-applied to the compact whole-VM model instead of the foundation model).

The exported graph is the **KV-cached windowed block-stack forward** the driver
actually calls each step:

    (x_window[1,W,D], q_pos[W], past_pos[Sc], {pastK_b, pastV_b}_b)
        -> (hidden[1,W,D], {newK_b, newV_b}_b)

i.e. exactly ``Transformer.forward_hidden_cached`` with the per-block ``(K,V,pos)``
cache flattened into fixed-arity tensor inputs/outputs — the standard
autoregressive KV-cache decode interface.  The register decode + the
softmax1+ALiBi eviction policy stay in the caller (``run_pure_forward_cached``),
exactly as for the torch model.  ``OnnxCachedModel`` wraps the ORT session behind
the same ``.embed`` / ``.blocks`` / ``.forward_hidden_cached`` API the driver
uses, so the corpus runner drives the ONNX model with zero driver changes.

Run ``python -m c4_min.export_onnx_compact`` for the export + vanilla audit +
byte-identity battery report.
"""
from __future__ import annotations

import io
import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ONNX Runtime 1.23 supports IR version <= 11; torch may emit a higher one, so we
# pin it.  opset 17 covers every op the vanilla transformer needs.
ONNX_OPSET = 17
ONNX_IR_VERSION = 10


# The full standard-transformer op vocabulary the exported KV-cached graph is
# allowed to use.  Everything here is a bog-standard ONNX operator that appears in
# any HuggingFace / LLM KV-cache decode export — no custom domains, no control-flow
# subgraphs, no external-memory ops.
VANILLA_OPS = {
    # projections + attention/FFN matmuls (+ SwiGLU einsum in the routed MoE)
    "MatMul", "Gemm", "Einsum",
    # softmax1 primitives (exp(x-m) / (exp(-m)+sum)) + the SwiGLU gate
    "Exp", "ReduceMax", "ReduceSum", "Div", "Sigmoid", "Mul",
    # residual / bias / ALiBi / scores plumbing + causal-mask compare
    "Add", "Sub", "Neg", "Abs", "Clip", "Where", "Greater", "GreaterOrEqual",
    "Less", "Equal", "Not",
    # top-1 MoE router: ArgMax the opcode one-hot -> Gather the active expert
    # (structural-sparsity dispatch; a standard hard-routed MoE, NOT control flow)
    "ArgMax",
    # token-embedding lookup (Gather) + reshapes/transposes for head split + the
    # KV-cache concat
    "Gather", "GatherND", "GatherElements", "Reshape", "Transpose", "Concat",
    "Unsqueeze", "Squeeze", "Expand", "Slice", "Identity", "ScatterElements",
    # shape / positional math (all data-independent)
    "Trilu", "Range", "ConstantOfShape", "Constant", "Shape", "Cast", "Size",
}

# Ops that would betray a NON-vanilla graph: in-graph python loop / control-flow,
# custom kernels, encoder/decoder cross-attention plumbing, normalization we don't
# use, or any recurrent cell.  (The KV cache is plain Concat, NOT a Loop/Scan.)
FORBIDDEN_OPS = {
    "Loop", "Scan", "If", "SequenceAt", "SequenceInsert", "SequenceConstruct",
    "LayerNormalization", "GroupNormalization", "InstanceNormalization",
    "BatchNormalization", "RNN", "LSTM", "GRU", "Attention",
}


# ---------------------------------------------------------------------------
# 1. The fixed-arity KV-cached step module (the export target).
# ---------------------------------------------------------------------------
class CachedStepModule(nn.Module):
    """Fixed-arity wrapper around ``Transformer.forward_hidden_cached``.

    The per-block ``(K, V, pos)`` cache list the driver passes is flattened into
    fixed positional tensor args so the TorchScript tracer emits a plain
    (loop-free) graph: ``x, q_pos, pastK_0, pastV_0, pastPos_0, ...`` ->
    ``hidden, newK_0, newV_0, ...``.

    Each block carries its OWN cached-position vector ``pastPos_b`` — after the
    softmax1+ALiBi eviction each block's cache keeps a DIFFERENT survivor set
    (the driver evicts per block), so the cache length (and positions) vary
    per block.  A single shared position vector would be wrong post-eviction.
    The new K/V returned are the FULL (cache+window) K/V per block — the driver
    slices the fresh window tail off them.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_blocks = len(model.blocks)

    def forward(self, x, q_pos, *kv):  # noqa: D401
        n = self.n_blocks
        # kv is 3*n: pastK_b, pastV_b, pastPos_b interleaved.
        past_kv = [(kv[3 * b], kv[3 * b + 1], kv[3 * b + 2]) for b in range(n)]
        hidden, new_caches = self.model.forward_hidden_cached(
            x, past_key_values=past_kv, q_positions=q_pos, use_cache=True)
        outs: List[torch.Tensor] = [hidden]
        for (K, V, _pos) in new_caches:
            outs.append(K)
            outs.append(V)
        return tuple(outs)


def _io_names(n_blocks: int) -> Tuple[List[str], List[str]]:
    in_names = ["x", "q_pos"]
    for b in range(n_blocks):
        in_names += [f"pastK{b}", f"pastV{b}", f"pastPos{b}"]
    out_names = ["hidden"]
    for b in range(n_blocks):
        out_names += [f"newK{b}", f"newV{b}"]
    return in_names, out_names


def _dynamic_axes(n_blocks: int) -> Dict[str, Dict[int, str]]:
    dyn: Dict[str, Dict[int, str]] = {
        "x": {1: "w"}, "q_pos": {0: "w"}, "hidden": {1: "w"},
    }
    for b in range(n_blocks):
        dyn[f"pastK{b}"] = {2: f"sc{b}"}
        dyn[f"pastV{b}"] = {2: f"sc{b}"}
        dyn[f"pastPos{b}"] = {0: f"sc{b}"}
        dyn[f"newK{b}"] = {2: f"sall{b}"}
        dyn[f"newV{b}"] = {2: f"sall{b}"}
    return dyn


def _example_inputs(model, W: int = 5, Sc: int = 3):
    """A minimal but representative (window + non-empty cache) trace input.

    Uses a DISTINCT per-block cache length (Sc, Sc+1, Sc+2, ...) so the tracer
    can never collapse the per-block position axes into one shared symbol
    (post-eviction the block caches genuinely differ in length)."""
    D = model.dim
    H = model.blocks[0].attn.n_heads
    HD = model.blocks[0].attn.head_dim
    n = len(model.blocks)
    torch.manual_seed(0)
    x = torch.randn(1, W, D) * 0.01
    q_pos = torch.arange(100, 100 + W, dtype=torch.long)
    kv: List[torch.Tensor] = []
    for b in range(n):
        scb = Sc + (b % 3)                      # vary the cache length per block
        kv.append(torch.randn(1, H, scb, HD) * 0.01)
        kv.append(torch.randn(1, H, scb, HD) * 0.01)
        kv.append(torch.arange(scb, dtype=torch.long))
    return (x, q_pos, *kv)


# ---------------------------------------------------------------------------
# 2. Export (dense) + sparsify (COO initializers).
# ---------------------------------------------------------------------------
def export_cached_onnx(model, path: str) -> str:
    """Trace the KV-cached windowed block-stack forward to a dense ONNX file."""
    import onnx

    model = model.eval()
    step = CachedStepModule(model).eval()
    n = len(model.blocks)
    in_names, out_names = _io_names(n)
    args = _example_inputs(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            step, args, path,
            input_names=in_names, output_names=out_names,
            dynamic_axes=_dynamic_axes(n),
            opset_version=ONNX_OPSET, do_constant_folding=True, dynamo=False)
    m = onnx.load(path)
    m.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return path


def _to_sparse_initializer(dense_init):
    """Dense float ONNX initializer TensorProto -> COO ``SparseTensorProto``."""
    from onnx import helper, numpy_helper, TensorProto

    arr = numpy_helper.to_array(dense_init)
    nz = np.argwhere(arr != 0.0)                       # [nnz, rank]
    vals = arr[arr != 0.0].astype(arr.dtype, copy=False)
    vals_t = numpy_helper.from_array(vals, name=dense_init.name)
    idx_flat = nz.reshape(-1).astype(np.int64).tolist()
    idx_t = helper.make_tensor(dense_init.name + "_idx", TensorProto.INT64,
                               [int(nz.shape[0]), int(arr.ndim)], idx_flat)
    return helper.make_sparse_tensor(vals_t, idx_t, list(arr.shape)), int(nz.size)


def to_sparse_onnx(dense_path: str, sparse_path: str, min_numel: int = 256
                   ) -> Dict:
    """Rewrite ``dense_path`` so every large + genuinely-sparse weight initializer
    is stored as an ONNX ``sparse_initializer`` (COO).  No graph/op change — ORT
    reconstructs the dense tensor for the ``MatMul`` so the forward is
    byte-identical; the file just shrinks (99.97 % of entries are zeros).
    """
    import onnx

    m = onnx.load(dense_path)
    g = m.graph
    kept_dense, moved = [], 0
    dense_entries = sparse_stored = 0
    for init in list(g.initializer):
        arr_numel = int(np.prod(init.dims)) if init.dims else 1
        nnz = int((onnx.numpy_helper.to_array(init) != 0).sum())
        if arr_numel >= min_numel and nnz * 3 < arr_numel:
            sp, nnz_stored = _to_sparse_initializer(init)
            g.sparse_initializer.append(sp)
            dense_entries += arr_numel
            sparse_stored += nnz_stored
            moved += 1
        else:
            kept_dense.append(init)
    del g.initializer[:]
    g.initializer.extend(kept_dense)
    m.ir_version = ONNX_IR_VERSION
    onnx.checker.check_model(m)
    onnx.save(m, sparse_path)
    return {
        "tensors_sparsified": moved,
        "dense_entries_replaced": dense_entries,
        "sparse_entries_stored": sparse_stored,
        "dense_file_bytes": os.path.getsize(dense_path),
        "sparse_file_bytes": os.path.getsize(sparse_path),
    }


# ---------------------------------------------------------------------------
# 3. Vanilla check.
# ---------------------------------------------------------------------------
def _iter_nodes(graph):
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.g.ByteSize():
                yield from _iter_nodes(attr.g)
            for sg in attr.graphs:
                yield from _iter_nodes(sg)


def op_inventory(path_or_model) -> Counter:
    import onnx
    m = onnx.load(path_or_model) if isinstance(path_or_model, str) else path_or_model
    return Counter(n.op_type for n in _iter_nodes(m.graph))


def assert_vanilla(path_or_model) -> Tuple[bool, List[str]]:
    """Return ``(is_vanilla, notes)``: no forbidden op (no in-graph loop / norm /
    RNN / fused-Attention), every op in the standard-transformer set, no custom
    op-domain."""
    import onnx
    m = onnx.load(path_or_model) if isinstance(path_or_model, str) else path_or_model
    inv = op_inventory(m)
    notes: List[str] = []
    ok = True
    forbidden = FORBIDDEN_OPS & set(inv)
    if forbidden:
        ok = False
        notes.append(f"FORBIDDEN ops present (non-vanilla): {sorted(forbidden)}")
    unknown = set(inv) - VANILLA_OPS - FORBIDDEN_OPS
    if unknown:
        ok = False
        notes.append(f"UNKNOWN ops (review): {sorted(unknown)}")
    custom = sorted({n.domain for n in _iter_nodes(m.graph)
                     if n.domain not in ("", "ai.onnx", "ai.onnx.ml")})
    if custom:
        ok = False
        notes.append(f"CUSTOM op domains present: {custom}")
    if ok:
        notes.append(
            "standard decode-only transformer w/ KV cache: attention "
            "(MatMul Q/K/V/O + softmax1 primitives + ALiBi + causal compare + "
            "Concat KV cache) + SwiGLU (Sigmoid*Mul) FFN + token-embedding Gather "
            "in the caller; NO in-graph loop, NO custom / external-memory ops.")
    return ok, notes


# ---------------------------------------------------------------------------
# 4. The ORT-backed driver model — a drop-in for the KV-cached driver.
# ---------------------------------------------------------------------------
def make_session(path: str, intra_threads: int = 4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = intra_threads
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


class OnnxCachedModel:
    """Drives ``run_pure_forward_cached`` with the ONNX graph as the forward.

    Exposes the SAME API the driver uses:
      * ``.embed`` / ``.dim`` / ``.vocab`` (the embedding gather + decode read
        the residual bands directly — kept as the torch tensors),
      * ``.blocks`` (for the driver's per-block ``BlockKVCacheBatched`` metadata:
        ``.attn.n_heads`` / ``.head_dim`` / ``.alibi_slopes`` and the eviction
        replay — these are read-only metadata, no compute),
      * ``.forward_hidden_cached(x, past_key_values, q_positions, use_cache)`` —
        routed through the ONNX Runtime session (this is the ONLY compute path;
        the torch blocks are never forwarded).

    The block-stack forward is thus executed entirely by ``onnxruntime``; the
    caller's register decode + softmax1/ALiBi eviction are unchanged.
    """

    def __init__(self, torch_model, onnx_path: str, intra_threads: int = 4):
        self.dim = torch_model.dim
        self.vocab = torch_model.vocab
        self.max_seq_len = torch_model.max_seq_len
        self.embed = torch_model.embed.detach()
        self.blocks = torch_model.blocks       # metadata only (never forwarded)
        self.n_blocks = len(torch_model.blocks)
        self._sess = make_session(onnx_path, intra_threads)
        self._in_names, self._out_names = _io_names(self.n_blocks)

    def to(self, device):
        # ONNX session runs on CPU; the driver's decode reads .embed / metadata,
        # which we keep on CPU (the small residual windows move to CPU for the
        # ORT call anyway).  Keep embed on the requested device for the gather.
        self.embed = self.embed.to(device)
        return self

    def forward_hidden_cached(self, x, past_key_values=None, q_positions=None,
                              use_cache: bool = True):
        n = self.n_blocks
        if past_key_values is None:
            past_key_values = [None] * n
        H = self.blocks[0].attn.n_heads
        HD = self.blocks[0].attn.head_dim
        dev = x.device
        q_pos = q_positions.to(dtype=torch.long, device=dev)
        q_pos_np = q_pos.detach().cpu().numpy().astype(np.int64)

        feed = {
            "x": x.detach().cpu().numpy().astype(np.float32),
            "q_pos": q_pos_np,
        }
        # Each block carries its OWN cache (K/V/pos) — after eviction the block
        # caches differ in length AND survivor positions, so per-block positions
        # are load-bearing (a single shared vector broke `blocks.1/attn/Sub`).
        past_pos_b: List[torch.Tensor] = []
        for b in range(n):
            pkv = past_key_values[b]
            if pkv is None:
                Kc = np.zeros((1, H, 0, HD), dtype=np.float32)
                Vc = np.zeros((1, H, 0, HD), dtype=np.float32)
                pos_b = torch.zeros(0, dtype=torch.long, device=dev)
            else:
                Kc = pkv[0].detach().cpu().numpy().astype(np.float32)
                Vc = pkv[1].detach().cpu().numpy().astype(np.float32)
                pos_b = pkv[2].to(dtype=torch.long, device=dev)
            feed[f"pastK{b}"] = Kc
            feed[f"pastV{b}"] = Vc
            feed[f"pastPos{b}"] = pos_b.detach().cpu().numpy().astype(np.int64)
            past_pos_b.append(pos_b)

        outs = self._sess.run(self._out_names, feed)
        hidden = torch.from_numpy(outs[0]).to(dev)
        new_caches = []
        for b in range(n):
            K = torch.from_numpy(outs[1 + 2 * b]).to(dev)
            V = torch.from_numpy(outs[1 + 2 * b + 1]).to(dev)
            k_pos_all = torch.cat([past_pos_b[b], q_pos], dim=0)
            new_caches.append((K, V, k_pos_all))
        return hidden, new_caches


# ---------------------------------------------------------------------------
# CLI / report.
# ---------------------------------------------------------------------------
def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024.0


def build_compact(code_size=64):
    """Build the SINGLE full-op-set compact model for ONNX export.

    NOTE: ONNX export needs the DENSE compact model, and the full op set (incl.
    the ~300 DIV/MOD blocks) makes this dense-compact build ~48 GB RSS.  Run on a
    high-memory host; there is no reduced-op export variant.
    """
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xF0
    _PFC.SP_INIT = 0xF0
    from c4_min.compact_alloc import build_compact_pure_forward_model
    return build_compact_pure_forward_model(code_size=code_size)


def main(out_dir: str = "/tmp/c4_compact_onnx") -> int:
    os.makedirs(out_dir, exist_ok=True)
    dense_path = os.path.join(out_dir, "compact_vm.onnx")
    sparse_path = os.path.join(out_dir, "compact_vm_sparse.onnx")

    print("=" * 74)
    print("COMPACT C4 VM  ->  vanilla ONNX (KV-cached decode) + sparse tensors")
    print("=" * 74)

    compact, L, stats = build_compact()
    compact.eval()
    n_params = sum(p.numel() for p in compact.parameters())
    n_nz = sum(int((p != 0).sum()) for p in compact.parameters())
    print(f"\nmodel: dim={compact.dim} vocab={compact.vocab} "
          f"n_blocks={len(compact.blocks)} "
          f"n_heads={compact.blocks[0].attn.n_heads} "
          f"head_dim={compact.blocks[0].attn.head_dim}")
    print(f"params: {n_params:,} total, {n_nz:,} non-zero "
          f"-> {100 * (1 - n_nz / n_params):.3f}% sparse "
          f"({_fmt_bytes(n_params * 4)} dense fp32)")

    export_cached_onnx(compact, dense_path)
    print(f"\n[1] exported dense ONNX -> {dense_path}  "
          f"({_fmt_bytes(os.path.getsize(dense_path))})")

    ss = to_sparse_onnx(dense_path, sparse_path)
    print(f"[1b] sparse-initializer ONNX -> {sparse_path}  "
          f"({_fmt_bytes(ss['sparse_file_bytes'])}, "
          f"{ss['tensors_sparsified']} tensors sparsified)")

    inv = op_inventory(sparse_path)
    print("\n[2] op inventory (standard-transformer verification):")
    for op, c in sorted(inv.items(), key=lambda kv: -kv[1]):
        print(f"      {op:18s} x{c}")
    ok, notes = assert_vanilla(sparse_path)
    print(f"\n    VANILLA: {'YES' if ok else 'NO'}")
    for nline in notes:
        print(f"      - {nline}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
