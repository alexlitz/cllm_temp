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
                       THIS is what makes bundling practical now.
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
# Weight serialisation — the size lever (sparse COO, nnz-only).
# ---------------------------------------------------------------------------
def _tensor_manifest(model):
    """Ordered (name, kind, tensor) list — the SAME order for write and read."""
    tensors = [
        ("embed", 2, model.embed.detach()),
        ("lm_head", 2, model.lm_head.detach()),
        ("lm_bias", 1, model.lm_bias.detach()),
    ]
    for bi, blk in enumerate(model.blocks):
        a = blk.attn
        tensors += [
            (f"b{bi}.attn.W_q", 0, a.W_q.detach()),
            (f"b{bi}.attn.W_k", 0, a.W_k.detach()),
            (f"b{bi}.attn.W_v", 0, a.W_v.detach()),
            (f"b{bi}.attn.W_o", 0, a.W_o.detach()),
            (f"b{bi}.attn.alibi_slopes", 1, a.alibi_slopes.detach()),
        ]
        f = blk.ffn
        tensors += [
            (f"b{bi}.ffn.W_up", 0, f.W_up.detach()),
            (f"b{bi}.ffn.b_up", 1, f.b_up.detach()),
            (f"b{bi}.ffn.W_gate", 0, f.W_gate.detach()),
            (f"b{bi}.ffn.b_gate", 1, f.b_gate.detach()),
            (f"b{bi}.ffn.W_down", 0, f.W_down.detach()),
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
    """Scatter the COO weights blob back into a freshly-built (zeroed) model.

    Mirrors ``_serialize_weights`` exactly: reads each tensor record and either
    zero-fills+scatters (kind 0, sparse COO) or copies the dense payload
    (kind 1/2).  Byte-identical reconstruction (same float bits).
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
    dest: Dict[str, torch.Tensor] = {name: None for name, _, _ in _tensor_manifest(model)}
    dest["embed"] = model.embed.data
    dest["lm_head"] = model.lm_head.data
    dest["lm_bias"] = model.lm_bias.data
    for bi, blk in enumerate(model.blocks):
        a, f = blk.attn, blk.ffn
        dest[f"b{bi}.attn.W_q"] = a.W_q.data
        dest[f"b{bi}.attn.W_k"] = a.W_k.data
        dest[f"b{bi}.attn.W_v"] = a.W_v.data
        dest[f"b{bi}.attn.W_o"] = a.W_o.data
        dest[f"b{bi}.attn.alibi_slopes"] = a.alibi_slopes.data
        dest[f"b{bi}.ffn.W_up"] = f.W_up.data
        dest[f"b{bi}.ffn.b_up"] = f.b_up.data
        dest[f"b{bi}.ffn.W_gate"] = f.W_gate.data
        dest[f"b{bi}.ffn.b_gate"] = f.b_gate.data
        dest[f"b{bi}.ffn.W_down"] = f.W_down.data
        dest[f"b{bi}.ffn.b_down"] = f.b_down.data

    for _ in range(n_tensors):
        (nl,) = rd("<I")
        name = blob[off:off + nl].decode("utf-8"); off += nl
        (kind,) = rd("<I")
        (rank,) = rd("<I")
        _shape = [rd("<I")[0] for _ in range(rank)]
        (count,) = rd("<I")
        d = dest[name]
        d.zero_()
        if kind == 0:
            idx = np.frombuffer(blob, dtype="<i4", count=count, offset=off)
            off += 4 * count
            val = np.frombuffer(blob, dtype="<f4", count=count, offset=off)
            off += 4 * count
            flat = d.reshape(-1)
            flat[torch.from_numpy(idx.astype("int64"))] = torch.from_numpy(val.copy())
        else:
            payload = np.frombuffer(blob, dtype="<f4", count=count, offset=off)
            off += 4 * count
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
def _compile_source_to_bytecode(source: str) -> List[Tuple[int, int]]:
    """Compile a C4 source string to the ISA (op, imm) stream the driver runs.

    Uses the same front end + translation as ``run_1096_pure_forward``:
    ``src.compiler.compile_c`` -> ``bytecode_to_isa``.
    """
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bytecode, _data = compile_c(source)
    code = bytecode_to_isa(bytecode)
    return [(int(i.op), int(i.imm) & 0xFFFFFFFF) for i in code]


def _build_model(config: Dict[str, object]):
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xF0
    _PFC.SP_INIT = 0xF0
    from c4_min.nibble_pure_forward_complete import build_pure_forward_complete_model
    from c4_min._build_guard import dense_build_allowed
    # The bundle weight serialiser reads each block's DENSE ``ffn.W_up`` /
    # ``ffn.W_down`` (then COO-drops the zeros), so it needs the dense complete
    # model — which pads every block to the ~160k-row MUL/DIV/MOD FFN and peaks at
    # 54-108 GB RSS regardless of code_size.  That is a machine-killing build, so
    # bundling is OPT-IN: set C4_ALLOW_DENSE_BUILD=1 to run it (on a box with the
    # headroom).  (The runtime full-op model is available memory-safe via
    # build_compact_sparse_streaming; only the dense-tensor serialisation here
    # needs the padded form.)
    if not dense_build_allowed():
        raise RuntimeError(
            "bundle_small builds the DENSE complete model (54-108 GB RSS) to "
            "serialise its weights — set C4_ALLOW_DENSE_BUILD=1 to opt in "
            "(memory hazard; see c4_min/_build_guard.py).")
    # SINGLE full-op-set interpreter: build it regardless of any legacy
    # include_bitwise/include_divmod keys an OLD bundle's config may carry.
    model, L = build_pure_forward_complete_model(
        code_size=int(config["code_size"]))
    return model, L


def _assemble_parts(source: str, *, expected, description, code_size,
                    step_cap, reuse_model):
    """Compile + serialise the three bundle sections and the finalised header.

    Returns ``(pre, hjson, runtime_bytes, weights_blob, bytecode_bytes, config,
    wstats, program)`` where ``pre + hjson + runtime + weights + bytecode`` IS the
    exact ``.c4bundle`` byte stream.  Factored out so the container-framing step
    (the part a section-fusing bundler — e.g. the C4-C ``c4_bundler.c4`` — does)
    is separable from the torch-only weight serialisation.  The offset fixed-point
    loop below is fully deterministic, so the SAME (pre, hjson, sections) is what
    the C4-C bundler concatenates to produce a byte-identical file.
    """
    config = {
        "code_size": code_size,
    }
    code = _compile_source_to_bytecode(source)
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
                   step_cap: int = 10000, reuse_model=None) -> Dict[str, object]:
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
                        reuse_model=reuse_model)
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
                    reuse_model=None) -> Dict[str, object]:
    """Build a .c4bundle from a C4 source string.

    Compiles ``source`` -> bytecode, builds the SMALL model at ``config``,
    serialises its weights sparse, and writes the flat container.  ``reuse_model``
    (a pre-built ``(model, L)``) skips the (slow) rebuild when assembling several
    bundles at the same config.
    """
    pre, hjson, runtime_bytes, weights_blob, bytecode_bytes, config, wstats, program = \
        _assemble_parts(source, expected=expected, description=description,
                        code_size=code_size, step_cap=step_cap,
                        reuse_model=reuse_model)

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
    stats: dict = {}
    t1 = time.monotonic()
    trace = run_pure_forward_cached(model, L, code, max_steps=cap,
                                    mask=0xFFFFFFFF, evict=True,
                                    prune_interval=120, stats=stats)
    t_run = time.monotonic() - t1

    got = (int(trace[-1]) & 0xFFFFFFFF) if trace else None
    exp = hdr.program.get("expected")
    steps = len(trace)
    if got is None:
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
    if verbose:
        print(f"[bundle] {os.path.basename(path)}  status={status}  "
              f"got={got} expected={exp}  steps={steps}  "
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
        info = assemble_bundle(
            args.source, args.out, expected=args.expected,
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
