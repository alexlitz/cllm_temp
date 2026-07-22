"""CHK-4 — single-file bundle: runtime + small model weights + program bytecode.

BLOG_SPEC §"Bundling Programs" (line 910): "a bundler which takes the [runtime],
the [model] weights and the target c program, compiles them into a single
binary."

This is the *pure-forward / KV-cached* instance of that bundler.  It fuses three
artifacts into ONE self-contained file (`.c4bundle`) that, when executed, runs
the embedded bytecode through the embedded model and prints the decoded result —
needing NO external files at run time:

  1. the RUNTIME     — the KV-cached pure-forward driver
                       (`run_pure_forward_cached`), the transformer forward
                       (softmax1 + ALiBi + SwiGLU), and the argmax byte decode.
                       This is the INTERIM runtime; the ONNX C runtime (CHK-2 /
                       CHK-3) is a drop-in replacement at `run_bundle` (see the
                       ONNX INTEGRATION POINT note there).
  2. the WEIGHTS     — the SMALL sparse model.  The dense bitwise config is
                       ~30 GB (7.5 B params, 99.998 % zero) and CANNOT be
                       bundled; the sparse COO re-encode (nnz-only, index_i32 +
                       value_f32 per nonzero) is ~32 MB, which fits in one file.
                       THIS is what makes bundling practical now.  The model
                       itself is built + reconstructed off the MEMORY-SAFE
                       streaming-sparse form (`build_compact_sparse_streaming`,
                       peak ~one block ≈ 12 GB), NEVER the padded dense
                       (`build_pure_forward_complete_model` → ~117 GB dense
                       hazard).  The bundled COO weights are byte-identical
                       (L-inf=0, `dense_kernel` mode) and the round-trip decodes
                       byte-exact.
  3. the BYTECODE    — the target C program's compiled ISA opcode/imm stream
                       (BLOG_SPEC §912: "we also perform the bytecode compilation
                       of the target c program during this step").

The bundle is a flat binary container (see `BundleHeader`), not a `.py` — so the
SAME container is producible by the C4-C bundler (`c4_bundler.c4`), byte-for-byte.

Usage
-----
    # assemble a bundle for one C source (compiles the C -> bytecode itself):
    python -m c4_min.bundle_small assemble \
        --source 'int main(){ return 500 + 700; }' \
        --expected 1200 --out /tmp/add.c4bundle
    # run a bundle end-to-end (loads model+bytecode, prints the decoded result):
    python -m c4_min.bundle_small run /tmp/add.c4bundle
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


# ---------------------------------------------------------------------------
# Container format.  A .c4bundle is:
#     MAGIC(8) VERSION(u32)
#     header_json_len(u32)  header_json(bytes)      # config + section table
#     [ RUNTIME  section ]  # utf-8 source of the embedded runner entry point
#     [ WEIGHTS  section ]  # sparse model, COO (nnz-only) — the 32 MB blob
#     [ BYTECODE section ]  # program ISA stream (op i32, imm i32) * n
# The header_json carries model config (dim/blocks/heads/vocab/code_size/...),
# the program metadata (expected result / description / step_cap) and the
# byte offset+length of each of the three sections.
# ---------------------------------------------------------------------------
MAGIC = b"C4BUNDLE"
VERSION = 1

# Weights blob magic + a per-tensor record:
#   tensor: name_len(u32) name(bytes) kind(u32) shape_rank(u32) shape(u32*rank)
#           count(u32)   then for kind 0: count * index_i32, count * value_f32
#                             for kind 1/2: count * value_f32 (dense, row-major)
# kind: 0 = 2-D weight (sparse COO over flattened [out*in]), 1 = 1-D vector
#       (dense f32, biases/slopes), 2 = 2-D dense (embed / lm_head, kept dense
#       because they are gathers / small).  A 2-D dense stores row-major f32.
WMAGIC = b"C4WTS\x00\x00\x00"


@dataclass
class BundleHeader:
    version: int
    config: Dict[str, object]        # model build args (dim/blocks/heads/...)
    program: Dict[str, object]       # expected / description / step_cap / n_instrs
    sections: Dict[str, Tuple[int, int]]   # name -> (offset, length)
    weights_stats: Dict[str, object]


# ---------------------------------------------------------------------------
# Streaming-sparse weight access.  The bundle's model is the MEMORY-SAFE
# streaming-sparse form (``build_compact_sparse_streaming``): the per-block
# Q/K/V/O and W_up/W_gate/W_down are ``sparse_forward.SparseWeight`` objects
# (CSR when sparse, dense when small/dense), not raw ``nn.Parameter`` tensors.
# We serialise the DISTINCT physical blocks (``_phys_blocks``); the recurrent
# apply order (if any) is rebuilt from the layout at reconstruct time, so a
# shared physical block is stored ONCE.
# ---------------------------------------------------------------------------
def _phys_blocks(model):
    """The DISTINCT stored blocks (dedups a recurrent apply order by identity)."""
    return list(getattr(model, "_phys_blocks", None) or model.blocks)


def _sw_to_dense(sw):
    """Materialise a ``SparseWeight`` (or a plain tensor) to a dense ``[out,in]``."""
    import torch
    if hasattr(sw, "is_sparse") and hasattr(sw, "out_dim"):   # SparseWeight
        if sw.dense_resident is not None:
            return sw.dense_resident.detach()
        if not sw.is_sparse:
            return sw.dense.detach()
        return sw.csr.to_dense().detach()
    return sw.detach()                                         # already a tensor


def _sw_scatter(sw, dense):
    """Scatter a freshly-deserialised dense ``[out,in]`` back into ``sw`` in place.

    Preserves the SparseWeight's storage KIND (CSR-vs-dense) so the reconstruct
    is bit-identical to the original streaming build: a weight that was stored
    sparse is re-CSR'd, a dense-kept one stays dense.  ``dense_resident`` (if the
    original had been materialised) is refreshed too so ``.linear`` stays exact.
    """
    dense = dense.contiguous()
    if sw.is_sparse:
        sw.csr = dense.to_sparse_csr()
    else:
        sw.dense = dense
    if sw.dense_resident is not None:
        sw.dense_resident = dense.clone()


def _tensor_manifest(model):
    """Ordered ``(name, kind, tensor)`` list — the SAME order for write and read.

    Reads the streaming-sparse model: ``SparseWeight`` bodies are densified to
    ``[out,in]`` (COO-serialised, kind 0); biases / ALiBi slopes stay dense
    (kind 1); embed / lm_head stay dense (kind 2).
    """
    tensors = [
        ("embed", 2, model.embed.detach()),
        ("lm_head", 2, model.lm_head.detach()),
        ("lm_bias", 1, model.lm_bias.detach()),
    ]
    for bi, blk in enumerate(_phys_blocks(model)):
        a = blk.attn
        tensors += [
            (f"b{bi}.attn.W_q", 0, _sw_to_dense(a.W_q)),
            (f"b{bi}.attn.W_k", 0, _sw_to_dense(a.W_k)),
            (f"b{bi}.attn.W_v", 0, _sw_to_dense(a.W_v)),
            (f"b{bi}.attn.W_o", 0, _sw_to_dense(a.W_o)),
            (f"b{bi}.attn.alibi_slopes", 1, a.alibi_slopes.detach()),
        ]
        f = blk.ffn
        tensors += [
            (f"b{bi}.ffn.W_up", 0, _sw_to_dense(f.W_up)),
            (f"b{bi}.ffn.b_up", 1, f.b_up.detach()),
            (f"b{bi}.ffn.W_gate", 0, _sw_to_dense(f.W_gate)),
            (f"b{bi}.ffn.b_gate", 1, f.b_gate.detach()),
            (f"b{bi}.ffn.W_down", 0, _sw_to_dense(f.W_down)),
            (f"b{bi}.ffn.b_down", 1, f.b_down.detach()),
        ]
    return tensors


def _serialize_weights(model, L) -> Tuple[bytes, Dict[str, object]]:
    """Serialise the model's parameters to the compact COO weights blob.

    Every 2-D linear weight (Q/K/V/O, W_up/W_gate/W_down) is stored as COO over
    the flattened ``[out*in]`` index space: only the nonzeros survive
    (``index_i32, value_f32`` per nonzero), so a 99.998 %-zero weight costs
    ``~8*nnz`` bytes instead of ``4*out*in``.  Biases / ALiBi slopes are tiny
    and kept dense.  The embedding + LM head are gathers / small and kept dense.
    Returns ``(blob, stats)``.
    """
    import torch

    buf = io.BytesIO()
    buf.write(WMAGIC)
    tensors = _tensor_manifest(model)
    buf.write(struct.pack("<I", len(tensors)))
    total_nnz = 0
    dense_equiv = 0
    for name, kind, t in tensors:
        t = t.contiguous().cpu().to(torch.float32)
        nb = name.encode("utf-8")
        buf.write(struct.pack("<I", len(nb)))
        buf.write(nb)
        buf.write(struct.pack("<I", kind))
        buf.write(struct.pack("<I", t.dim()))
        for d in t.shape:
            buf.write(struct.pack("<I", int(d)))
        dense_equiv += t.numel() * 4
        if kind == 0:
            flat = t.reshape(-1)
            nz = torch.nonzero(flat, as_tuple=False).reshape(-1)
            vals = flat[nz]
            nnz = int(nz.numel())
            total_nnz += nnz
            buf.write(struct.pack("<I", nnz))
            buf.write(nz.to(torch.int32).numpy().tobytes())   # all indices
            buf.write(vals.numpy().tobytes())                 # then all values
        else:
            n = t.numel()
            buf.write(struct.pack("<I", n))
            buf.write(t.reshape(-1).numpy().tobytes())
    blob = buf.getvalue()
    stats = {
        "n_tensors": len(tensors),
        "total_nnz": total_nnz,
        "weights_bytes": len(blob),
        "dense_equiv_bytes": dense_equiv,
        "compression": round(dense_equiv / max(1, len(blob)), 1),
    }
    return blob, stats


def _deserialize_weights(blob: bytes, model, L) -> None:
    """Scatter the COO weights blob back into a freshly-built streaming model.

    Mirrors ``_serialize_weights`` exactly and reconstructs into the
    MEMORY-SAFE streaming-sparse model (built zeroed by ``_build_model``).  For
    kind-0 (2-D linear) records the COO nonzeros are scattered into a dense
    ``[out,in]`` and pushed back through ``_sw_scatter`` (which preserves the
    original CSR-vs-dense storage kind), so the reconstruct is bit-identical to
    the original streaming build — same float bits, same sparsity structure.
    Kind-1/2 (biases / slopes / embed / lm_head) copy the dense payload in place.
    """
    import numpy as np
    import torch

    assert blob[:len(WMAGIC)] == WMAGIC, "bad weights magic"
    off = len(WMAGIC)

    def rd(fmt: str):
        nonlocal off
        sz = struct.calcsize(fmt)
        v = struct.unpack_from(fmt, blob, off)
        off += sz
        return v

    (n_tensors,) = rd("<I")
    # ``sw`` maps a kind-0 name -> its SparseWeight (scatter in place, preserving
    # storage kind); ``dense`` maps a kind-1/2 name -> a plain tensor to copy into.
    sw: Dict[str, object] = {}
    dense: Dict[str, torch.Tensor] = {}
    dense["embed"] = model.embed.data
    dense["lm_head"] = model.lm_head.data
    dense["lm_bias"] = model.lm_bias.data
    for bi, blk in enumerate(_phys_blocks(model)):
        a, f = blk.attn, blk.ffn
        sw[f"b{bi}.attn.W_q"] = a.W_q
        sw[f"b{bi}.attn.W_k"] = a.W_k
        sw[f"b{bi}.attn.W_v"] = a.W_v
        sw[f"b{bi}.attn.W_o"] = a.W_o
        dense[f"b{bi}.attn.alibi_slopes"] = a.alibi_slopes
        sw[f"b{bi}.ffn.W_up"] = f.W_up
        dense[f"b{bi}.ffn.b_up"] = f.b_up
        sw[f"b{bi}.ffn.W_gate"] = f.W_gate
        dense[f"b{bi}.ffn.b_gate"] = f.b_gate
        sw[f"b{bi}.ffn.W_down"] = f.W_down
        dense[f"b{bi}.ffn.b_down"] = f.b_down

    for _ in range(n_tensors):
        (nl,) = rd("<I")
        name = blob[off:off + nl].decode("utf-8"); off += nl
        (kind,) = rd("<I")
        (rank,) = rd("<I")
        shape = [rd("<I")[0] for _ in range(rank)]
        (count,) = rd("<I")
        if kind == 0:
            idx = np.frombuffer(blob, dtype="<i4", count=count, offset=off)
            off += 4 * count
            val = np.frombuffer(blob, dtype="<f4", count=count, offset=off)
            off += 4 * count
            w = sw[name]
            flat = torch.zeros(int(w.out_dim) * int(w.in_dim), dtype=torch.float32)
            flat[torch.from_numpy(idx.astype("int64"))] = torch.from_numpy(val.copy())
            _sw_scatter(w, flat.reshape(int(w.out_dim), int(w.in_dim)))
        else:
            payload = np.frombuffer(blob, dtype="<f4", count=count, offset=off)
            off += 4 * count
            d = dense[name]
            d.reshape(-1)[:] = torch.from_numpy(payload.copy())


# ---------------------------------------------------------------------------
# Embedded runtime source — the self-describing entry point.
# ---------------------------------------------------------------------------
_RUNTIME_SRC = """\
# Embedded bundle runtime (interim: torch/KV-cached pure-forward driver).
# The bundle is self-describing: this section documents HOW the container is run.
# The actual execution is bundle_small.run_bundle, which:
#   1. reads the header (model config + program metadata + section table),
#   2. rebuilds the zeroed model at that config,
#   3. scatters the embedded COO weights into it (byte-identical reconstruction),
#   4. drives the embedded bytecode via run_pure_forward_cached (KV-cached),
#   5. prints the argmax-decoded 32-bit result.
# ONNX INTEGRATION POINT: replace step (4)'s driver with the C ONNX runtime
# (CHK-2 export -> .c4onnx/.nblbin; CHK-3 C-in-C4 onnxruntime) once committed;
# the header + weights COO + bytecode sections are runtime-agnostic.
"""


# ---------------------------------------------------------------------------
# Assemble.
# ---------------------------------------------------------------------------
def _compile_source_to_bytecode(source: str) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Compile a C4 source string to the ISA ``(op, imm)`` stream + data segment.

    Uses the same front end + translation as ``run_1096_pure_forward``:
    ``src.compiler.compile_c`` -> ``bytecode_to_isa``.  Returns
    ``(code, data_seg)`` where ``data_seg`` is the compiler's data-segment byte
    list (string literals etc., addressed from 0) — an I/O program (``printf``)
    needs it; an arith program's is all-zero.
    """
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bytecode, data = compile_c(source)
    code = bytecode_to_isa(bytecode)
    return ([(int(i.op), int(i.imm) & 0xFFFFFFFF) for i in code],
            [int(b) & 0xFF for b in (data or [])])


def _build_model(config: Dict[str, object]):
    """Build the bundle's model via the MEMORY-SAFE streaming-sparse form.

    Routes OFF the padded dense ``build_pure_forward_complete_model`` (which
    materialises the whole ~117 GB dense attention + globally-max-padded FFN and
    was measured OOM-killed at 117 GB) and onto
    ``compact_alloc.build_compact_sparse_streaming`` — same SINGLE full-op-set
    interpreter, built ONE block at a time so peak live memory is a single
    block's dense weights (~12 GB), never the dense whole.  In ``dense_kernel``
    compute mode the streaming model is byte-identical (L-inf=0) to the dense
    build, so the bundled COO weights decode byte-exact.

    ``recurrent_divmod=False`` keeps the DIV/MOD span unrolled — byte-identical
    to the dense bundle's own ``recurrent_divmod=False`` build (the bundle just
    serialises the resulting per-block weights, which are the same either way for
    the non-recurrent path).
    """
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xF0
    _PFC.SP_INIT = 0xF0
    # Consolidation note (#661 over #656): #656 GUARDED the bundle behind the
    # dense-build opt-in (``C4_ALLOW_DENSE_BUILD``) because the OLD serialiser read
    # each block's dense ``ffn.W_up``/``W_down`` off the ~117 GB dense complete
    # model.  #661 is the MORE-COMPLETE fix: it routes the serialiser onto the
    # MEMORY-SAFE ``build_compact_sparse_streaming`` (one block at a time, peak
    # ~12 GB, well under the conftest 60 GB / per-build 20 GB ceilings), which in
    # ``dense_kernel`` mode is byte-identical (L-inf=0) to the dense build — so the
    # bundled COO weights still decode byte-exact WITHOUT the dense hazard.  The
    # #656 opt-in gate is therefore no longer needed here (the streaming route
    # removes the hazard entirely rather than merely refusing it).
    from c4_min.compact_alloc import build_compact_sparse_streaming
    # SINGLE full-op-set interpreter; ``dense_kernel`` = bit-identical decode.
    model, L, _stats = build_compact_sparse_streaming(
        code_size=int(config["code_size"]),
        compute_mode="dense_kernel",
        recurrent_divmod=False)
    return model, L


def _assemble_parts(source: str, *, expected, description, code_size,
                    step_cap, reuse_model, expected_stdout=None, mask=None):
    """Compile + serialise the three bundle sections and the finalised header.

    Returns ``(pre, hjson, runtime_bytes, weights_blob, bytecode_bytes, config,
    wstats, program)`` where ``pre + hjson + runtime + weights + bytecode`` IS the
    exact ``.c4bundle`` byte stream.  Factored out so the container-framing step
    (the part a section-fusing bundler — e.g. the C4-C ``c4_bundler.c4`` — does)
    is separable from the torch-only weight serialisation.  The offset fixed-point
    loop below is fully deterministic, so the SAME (pre, hjson, sections) is what
    the C4-C bundler concatenates to produce a byte-identical file.

    ``expected_stdout`` (bytes/str) marks an I/O program (``printf``): the
    data-segment string literals + a byte-mask are carried in the header so
    ``run_bundle`` can capture stdout and verify it byte-exact.  ``mask``
    overrides the register decode width (I/O programs ride the 8-bit AX byte,
    ``mask=0xFF``; arith programs use the full ``0xFFFFFFFF``).
    """
    config = {
        "code_size": code_size,
    }
    code, data_seg = _compile_source_to_bytecode(source)
    # default mask: 0xFF for an I/O program (pointers ride the low AX byte),
    # 0xFFFFFFFF for an arith program (full 32-bit result).
    if mask is None:
        mask = 0xFF if expected_stdout is not None else 0xFFFFFFFF
    if isinstance(expected_stdout, str):
        expected_stdout = expected_stdout.encode("latin-1")
    model, L = reuse_model if reuse_model is not None else _build_model(config)
    config.update({
        "dim": int(L.D), "n_blocks": len(model.blocks),
        "n_heads": int(model.blocks[0].attn.n_heads), "vocab": int(model.vocab),
    })
    weights_blob, wstats = _serialize_weights(model, L)

    runtime_bytes = _RUNTIME_SRC.encode("utf-8")
    code_buf = io.BytesIO()
    code_buf.write(struct.pack("<I", len(code)))
    for op, imm in code:
        code_buf.write(struct.pack("<iI", int(op), int(imm) & 0xFFFFFFFF))
    bytecode_bytes = code_buf.getvalue()

    program = {
        "expected": (None if expected is None else int(expected) & 0xFFFFFFFF),
        "description": description, "step_cap": step_cap, "n_instrs": len(code),
        "source": source,
        # I/O program support: the data segment (string literals, addressed from
        # 0), the decode mask, and the expected stdout (latin-1) if this is a
        # printf-style program.  Empty/None for a pure arith program.
        "data_seg": data_seg,
        "mask": int(mask) & 0xFFFFFFFF,
        "expected_stdout": (None if expected_stdout is None
                            else expected_stdout.decode("latin-1")),
    }
    header = {
        "version": VERSION, "config": config, "program": program,
        "weights_stats": wstats, "sections": {},
    }

    def _emit(sections_hdr):
        header["sections"] = sections_hdr
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pre = MAGIC + struct.pack("<II", VERSION, len(hjson))
        return pre, hjson

    # Two-pass: the header length shifts the section offsets, so iterate to a
    # fixed point (json int-width for the offsets can grow by a digit).
    sec = {"runtime": (0, len(runtime_bytes)),
           "weights": (0, len(weights_blob)),
           "bytecode": (0, len(bytecode_bytes))}
    for _ in range(6):
        pre, hjson = _emit(sec)
        base = len(pre) + len(hjson)
        r_off = base
        w_off = r_off + len(runtime_bytes)
        b_off = w_off + len(weights_blob)
        new_sec = {"runtime": (r_off, len(runtime_bytes)),
                   "weights": (w_off, len(weights_blob)),
                   "bytecode": (b_off, len(bytecode_bytes))}
        if new_sec == sec:
            break
        sec = new_sec
    pre, hjson = _emit(sec)
    return pre, hjson, runtime_bytes, weights_blob, bytecode_bytes, \
        config, wstats, program


def prepare_bundle(source: str, out_dir: str, *, expected: Optional[int] = None,
                   description: str = "", code_size: int = 64,
                   step_cap: int = 10000, reuse_model=None,
                   expected_stdout=None, mask=None) -> Dict[str, object]:
    """Emit the raw bundle parts a section-fusing bundler concatenates.

    Writes, into ``out_dir``:  ``header.bin`` (MAGIC+version+hlen+finalised header
    JSON), ``runtime.bin``, ``weights.bin``, ``bytecode.bin``, and a plain-text
    ``manifest.txt`` (one ``name length`` line per part, fuse order).  Then
    ``cat header.bin runtime.bin weights.bin bytecode.bin`` == the ``.c4bundle``
    that ``assemble_bundle`` would write — this is exactly what the C4-C bundler
    (``bundler/c4_bundler.c4``) does with ``open``/``read``/``putchar``.  The
    torch-only weight serialisation stays here; the C4-C side is pure byte I/O.
    """
    os.makedirs(out_dir, exist_ok=True)
    pre, hjson, runtime_bytes, weights_blob, bytecode_bytes, config, wstats, program = \
        _assemble_parts(source, expected=expected, description=description,
                        code_size=code_size, step_cap=step_cap,
                        reuse_model=reuse_model, expected_stdout=expected_stdout,
                        mask=mask)
    header_bin = pre + hjson
    parts = [("header.bin", header_bin), ("runtime.bin", runtime_bytes),
             ("weights.bin", weights_blob), ("bytecode.bin", bytecode_bytes)]
    for name, blob in parts:
        with open(os.path.join(out_dir, name), "wb") as fh:
            fh.write(blob)
    with open(os.path.join(out_dir, "manifest.txt"), "w") as fh:
        for name, blob in parts:
            fh.write(f"{name} {len(blob)}\n")
    total = sum(len(b) for _, b in parts)
    return {
        "out_dir": out_dir, "parts": {n: len(b) for n, b in parts},
        "bundle_bytes": total, "bundle_mb": round(total / 1e6, 2),
        "config": config, "program_expected": program["expected"],
    }


def assemble_bundle(source: str, out_path: str, *, expected: Optional[int] = None,
                    description: str = "", code_size: int = 64,
                    step_cap: int = 10000,
                    reuse_model=None, expected_stdout=None,
                    mask=None) -> Dict[str, object]:
    """Build a .c4bundle from a C4 source string.

    Compiles ``source`` -> bytecode, builds the SMALL model at ``config``,
    serialises its weights sparse, and writes the flat container.  ``reuse_model``
    (a pre-built ``(model, L)``) skips the (slow) rebuild when assembling several
    bundles at the same config.  ``expected_stdout`` marks a printf-style I/O
    program (the driver captures stdout and verifies it byte-exact); ``mask``
    overrides the decode width (defaults 0xFF for I/O, 0xFFFFFFFF for arith).
    """
    pre, hjson, runtime_bytes, weights_blob, bytecode_bytes, config, wstats, program = \
        _assemble_parts(source, expected=expected, description=description,
                        code_size=code_size, step_cap=step_cap,
                        reuse_model=reuse_model, expected_stdout=expected_stdout,
                        mask=mask)

    with open(out_path, "wb") as fh:
        fh.write(pre)
        fh.write(hjson)
        fh.write(runtime_bytes)
        fh.write(weights_blob)
        fh.write(bytecode_bytes)

    total = os.path.getsize(out_path)
    return {
        "out": out_path, "bundle_bytes": total,
        "bundle_mb": round(total / 1e6, 2),
        "weights_bytes": wstats["weights_bytes"],
        "weights_mb": round(wstats["weights_bytes"] / 1e6, 2),
        "dense_equiv_gb": round(wstats["dense_equiv_bytes"] / 1e9, 2),
        "compression": wstats["compression"],
        "runtime_bytes": len(runtime_bytes),
        "bytecode_bytes": len(bytecode_bytes), "n_instrs": program["n_instrs"],
        "config": config, "program_expected": program["expected"],
    }


# ---------------------------------------------------------------------------
# Read + run.
# ---------------------------------------------------------------------------
def read_header(path: str) -> Tuple[BundleHeader, bytes]:
    """Read a bundle's header (returns the whole file bytes for section slicing)."""
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:len(MAGIC)] == MAGIC, "not a .c4bundle (bad magic)"
    off = len(MAGIC)
    version, hlen = struct.unpack_from("<II", data, off)
    off += 8
    header = json.loads(data[off:off + hlen].decode("utf-8"))
    hdr = BundleHeader(version=version, config=header["config"],
                       program=header["program"],
                       sections={k: tuple(v) for k, v in header["sections"].items()},
                       weights_stats=header["weights_stats"])
    return hdr, data


def _section(data: bytes, hdr: BundleHeader, name: str) -> bytes:
    off, ln = hdr.sections[name]
    return data[off:off + ln]


def run_bundle(path: str, *, max_steps: Optional[int] = None,
               verbose: bool = True) -> Dict[str, object]:
    """Load a bundle and run it end-to-end; return the decoded result + verdict.

    Rebuilds the zeroed model at the header config, scatters the embedded COO
    weights, decodes the embedded bytecode, and drives it through
    ``run_pure_forward_cached`` (the interim KV-cached torch runtime).

    ONNX INTEGRATION POINT: this driver call is the single seam the ONNX C
    runtime replaces.  The header (config + program metadata), the weights COO
    blob and the bytecode section are runtime-agnostic — CHK-3's C-in-C4
    onnxruntime consumes the SAME three sections (it needs the weights as a
    .nblbin, which is the same COO re-encode with ONNX tensor names).
    """
    from c4_min import isa
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    hdr, data = read_header(path)
    t0 = time.monotonic()
    model, L = _build_model(hdr.config)          # zeroed structure at config
    _deserialize_weights(_section(data, hdr, "weights"), model, L)
    t_load = time.monotonic() - t0

    bc = _section(data, hdr, "bytecode")
    (n,) = struct.unpack_from("<I", bc, 0)
    code: List[isa.Instr] = []
    o = 4
    for _ in range(n):
        op, imm = struct.unpack_from("<iI", bc, o)
        o += 8
        code.append(isa.Instr(op, imm))

    cap = max_steps if max_steps is not None else int(hdr.program.get("step_cap", 10000))
    # I/O program support: wire the data segment (string literals) + an fio
    # stdout/stdin sink so a printf-style bundle can be captured byte-exact.  An
    # arith program carries an empty data_seg + no expected_stdout and behaves
    # exactly as before (full 32-bit register decode).
    mask = int(hdr.program.get("mask") or 0xFFFFFFFF)
    exp_stdout_s = hdr.program.get("expected_stdout")
    exp_stdout = (None if exp_stdout_s is None
                  else exp_stdout_s.encode("latin-1"))
    data_list = hdr.program.get("data_seg") or []
    data_seg = {i: (int(b) & 0xFF) for i, b in enumerate(data_list)}
    is_io = exp_stdout is not None

    fio = None
    if is_io:
        from c4_min import nibble_filesys as FS
        fio = FS.FileOpState(runner=FS.FileRunner(
            fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"")))

    stats: dict = {}
    t1 = time.monotonic()
    trace = run_pure_forward_cached(model, L, code, max_steps=cap,
                                    mask=mask, evict=True,
                                    prune_interval=120, stats=stats,
                                    fio=fio, data_seg=(data_seg or None))
    t_run = time.monotonic() - t1

    got_stdout = bytes(fio.runner.stdout) if is_io else None
    got = (int(trace[-1]) & mask) if trace else None
    exp = hdr.program.get("expected")
    steps = len(trace)
    if is_io:
        # I/O program: the verdict is byte-exact STDOUT, not the register value.
        if got_stdout == exp_stdout:
            status = "PASS"
        else:
            status = "FAIL"
    elif got is None:
        status = "ERROR"
    elif steps >= cap:
        status = "TIMEOUT"
    elif exp is None:
        status = "RAN"
    elif got == exp:
        status = "PASS"
    else:
        status = "FAIL"

    result = {
        "status": status, "got": got, "expected": exp, "steps": steps,
        "load_seconds": round(t_load, 2), "run_seconds": round(t_run, 2),
        "description": hdr.program.get("description", ""),
        "config": hdr.config, "kv_stats": stats,
    }
    if is_io:
        result["stdout"] = (None if got_stdout is None
                            else got_stdout.decode("latin-1"))
        result["expected_stdout"] = exp_stdout_s
    if verbose:
        tail = (f"stdout={got_stdout!r} expected_stdout={exp_stdout!r}"
                if is_io else f"got={got} expected={exp}")
        print(f"[bundle] {os.path.basename(path)}  status={status}  "
              f"{tail}  steps={steps}  "
              f"(load {t_load:.1f}s + run {t_run:.1f}s)", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("assemble", help="build a .c4bundle from a C4 source")
    pa.add_argument("--source", required=True)
    pa.add_argument("--expected", type=int, default=None)
    pa.add_argument("--expected-stdout", default=None,
                    help="mark a printf-style I/O program; verify stdout "
                         "byte-exact (accepts \\n escapes)")
    pa.add_argument("--description", default="")
    pa.add_argument("--out", required=True)
    pa.add_argument("--code-size", type=int, default=64)
    pa.add_argument("--step-cap", type=int, default=10000)

    pp = sub.add_parser("prepare", help="emit the raw bundle parts for a "
                        "section-fusing bundler (e.g. the C4-C c4_bundler.c4)")
    pp.add_argument("--source", required=True)
    pp.add_argument("--expected", type=int, default=None)
    pp.add_argument("--description", default="")
    pp.add_argument("--out-dir", required=True)
    pp.add_argument("--code-size", type=int, default=64)
    pp.add_argument("--step-cap", type=int, default=10000)

    pr = sub.add_parser("run", help="run a .c4bundle end-to-end")
    pr.add_argument("bundle")
    pr.add_argument("--max-steps", type=int, default=None)

    pi = sub.add_parser("info", help="print a bundle's header/sections")
    pi.add_argument("bundle")

    args = ap.parse_args(argv)

    if args.cmd == "assemble":
        exp_stdout = (args.expected_stdout.encode("latin-1")
                      .decode("unicode_escape").encode("latin-1")
                      if args.expected_stdout is not None else None)
        info = assemble_bundle(
            args.source, args.out, expected=args.expected,
            expected_stdout=exp_stdout,
            description=args.description, code_size=args.code_size,
            step_cap=args.step_cap)
        print(json.dumps(info, indent=2))
        return 0
    if args.cmd == "prepare":
        info = prepare_bundle(
            args.source, args.out_dir, expected=args.expected,
            description=args.description, code_size=args.code_size,
            step_cap=args.step_cap)
        print(json.dumps(info, indent=2))
        return 0
    if args.cmd == "run":
        res = run_bundle(args.bundle, max_steps=args.max_steps)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] in ("PASS", "RAN") else 1
    if args.cmd == "info":
        hdr, data = read_header(args.bundle)
        print(json.dumps({
            "version": hdr.version, "config": hdr.config,
            "program": {k: v for k, v in hdr.program.items() if k != "source"},
            "sections": hdr.sections, "weights_stats": hdr.weights_stats,
            "file_bytes": len(data),
        }, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
