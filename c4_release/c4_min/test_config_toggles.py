"""CONFIG-TOGGLE MATRIX gate for the c4_min build/run path.

There is now a SINGLE full-op-set interpreter (every opcode incl. DIV/MOD +
bitwise) — the former include_divmod / include_bitwise op-subset splits are gone.
This file codifies the remaining REAL config axes of the CANONICAL builder
(``compact_alloc``) + the canonical KV-cached driver (``run_pure_forward_cached``):

  compute_mode    {sparse_mm, dense_kernel}   (argmax-identical)
  eviction        {off, on (base policy)}
  build path      {stream-build, load-sparse}  (byte-identical)
  device          {cpu, cuda:0}

Each combination is checked BOTH for numeric consistency (decode-band L-inf) AND
argmax correctness of the full op battery (ALU / memory / flow / func / bitwise /
div-mod).  The FULL model is built ONCE (module fixture) via the STREAMING sparse
builder so peak build RSS is ~one dense block (~9.5 GB), NEVER the ~130 GB full-op
dense build.

  * default (bare ``pytest``): CPU, streaming build (~30 s), full battery.
  * ``C4_TEST_CUDA=1``: also run the sanity batteries on ``cuda:0`` and assert
    the GPU result matches CPU.

Run:
  OMP_NUM_THREADS=4 python -m pytest c4_min/test_config_toggles.py -v
  C4_TEST_CUDA=1 OMP_NUM_THREADS=4 python -m pytest c4_min/test_config_toggles.py -v
"""
from __future__ import annotations

import os

import pytest
import torch

# CRITICAL: force CUDA init HERE, before the heavy c4_min imports below.
# Under pytest's import machinery those imports otherwise poison torch's lazy
# CUDA init and ``torch.cuda.is_available()`` then returns False for the rest
# of the session (so the GPU rows would silently SKIP even with
# C4_TEST_CUDA=1). A single ``.to('cuda:0')`` up front pins init True and
# survives the imports. Mirrors ``_matrix_toggles.main``'s force-init.
_CUDA_OK = False
if os.environ.get("C4_TEST_CUDA") == "1":
    try:
        _ = torch.zeros(1).to("cuda:0")
        _CUDA_OK = True
    except Exception:  # noqa: BLE001 — no GPU / driver busy: GPU rows stay skipped
        _CUDA_OK = False

# Mirror the canonical runner's SP_INIT pin so the reference interpreter and the
# cached-window driver agree on the frame arithmetic.
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min.compact_alloc import (
    build_compact_sparse_streaming,
    save_sparse_transformer,
    load_sparse_transformer,
)
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min.nibble_pure_forward_complete import (
    ref_interpret, make_overlay_complete, _build_frame, SP_INIT)
from c4_min.run_1096_pure_forward import bytecode_to_isa
from c4_min import blogspec_vocab as V


# ---- sanity programs (byte-exact expected) -------------------------------
ALU = [("add", "int main(){ return 500 + 700; }", 1200),
       ("sub", "int main(){ return 900 - 99; }", 801),
       ("mul", "int main(){ return 100 * 10; }", 1000)]
MEM = [("var", "int main(){ int x; x = 1000; return x; }", 1000)]
FLOW = [("if_gt", "int main(){ if (5 > 3) return 7; return 0; }", 7),
        ("if_eq", "int main(){ if (4 == 4) return 1; return 0; }", 1)]
FUNC = [("func_id", "int identity(int x){ return x; } "
                    "int main(){ return identity(1000); }", 1000)]
BITWISE = [("or", "int main(){ return 12 | 3; }", 15),
           ("and", "int main(){ return 12 & 10; }", 8),
           ("xor", "int main(){ return 12 ^ 10; }", 6),
           ("shl", "int main(){ return 3 << 2; }", 12),
           ("shr", "int main(){ return 48 >> 2; }", 12)]
DIVMOD = [("div", "int main(){ return 720 / 6; }", 120),
          ("mod", "int main(){ return 84 % 5; }", 4),
          ("divzero", "int main(){ return 5 / 0; }", 0)]

BASE = ALU + MEM + FLOW + FUNC
FULL_BATTERY = BASE + BITWISE + DIVMOD
# Use the force-init result captured at the top of the module — NOT a fresh
# ``torch.cuda.is_available()`` here, which the c4_min imports above have
# already poisoned to False under pytest.
_RUN_CUDA = os.environ.get("C4_TEST_CUDA") == "1" and _CUDA_OK


# ---- helpers -------------------------------------------------------------
_CODE = {}


def _compile(src):
    if src not in _CODE:
        from src.compiler import compile_c
        _CODE[src] = bytecode_to_isa(compile_c(src)[0])
    return _CODE[src]


def _final(model, L, src, evict, device):
    code = _compile(src)
    cap = len(ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)) + 6
    tr = run_pure_forward_cached(model, L, code, max_steps=cap, mask=0xFFFFFFFF,
                                 evict=evict, prune_interval=120)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    return tr[-1] & 0xFFFFFFFF if tr else None


def _battery(model, L, cases, evict=True, device="cpu"):
    model = model.to(device)
    try:
        for nm, src, exp in cases:
            got = _final(model, L, src, evict, device)
            assert got == exp, f"{nm}: got {got}, want {exp} " \
                               f"(evict={evict}, device={device})"
    finally:
        model.to("cpu")


def _decode_row(x, L):
    row = x[0, -1]
    vals = [row[getattr(L, nm)].reshape(-1) for nm in
            ("PC_VAL", "AX_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED")]
    vals += [row[L.AX + k].reshape(-1) for k in range(8)]
    return torch.cat(vals)


def _block_stack(model, L, code):
    stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    for _ in range(4):
        stream += _build_frame(1, 7, SP_INIT - 4, SP_INIT, 3)
    toks = torch.tensor([stream])
    ov = make_overlay_complete(code, L)
    with torch.no_grad():
        x = model.embed[toks].clone()
        ov(x)
        for b in model.blocks:
            x = b(x)
    return _decode_row(x, L).float()


def _linf(mA, LA, mB, LB, srcs):
    worst = 0.0
    for s in srcs:
        c = _compile(s)
        worst = max(worst, (_block_stack(mA, LA, c)
                            - _block_stack(mB, LB, c)).abs().max().item())
    return worst


_BID_SRCS = ["int main(){ return 500 + 700; }",
             "int main(){ int x; x = 1000; return x; }",
             "int main(){ return 12 | 3; }"]


# =========================================================================
# SINGLE FULL-OP-SET MODEL: build-path byte-identity + compute_mode +
# eviction + full op battery (incl. bitwise AND div/mod) + (optional) device.
# Built ONCE via the STREAMING sparse builder (peak ~9.5 GB, never dense).
# =========================================================================
@pytest.fixture(scope="module")
def full_model(tmp_path_factory):
    """(stream, loaded, stream_mm) + layouts for the single full-op-set model."""
    path = str(tmp_path_factory.mktemp("t") / "full.pt")
    stream, Ls, st = build_compact_sparse_streaming(
        code_size=64, compute_mode="dense_kernel")
    save_sparse_transformer(stream, Ls, st, path)
    loaded, Ll = load_sparse_transformer(path, compute_mode="dense_kernel")
    stream_mm, Lmm, _ = build_compact_sparse_streaming(
        code_size=64, compute_mode="sparse_mm")
    return dict(stream=(stream, Ls), load=(loaded, Ll), stream_mm=(stream_mm, Lmm))


def test_build_paths_byte_identical(full_model):
    """stream-build == load-sparse (L-inf = 0), across bitwise + div/mod srcs."""
    s, Ls = full_model["stream"]
    ld, Ll = full_model["load"]
    srcs = _BID_SRCS + ["int main(){ return 720 / 6; }",
                        "int main(){ return 84 % 5; }"]
    assert _linf(s, Ls, ld, Ll, srcs) < 1e-9     # stream vs load-sparse


def test_compute_mode_argmax_identical(full_model):
    """dense_kernel vs sparse_mm: fp-accum residue only (<1e-3), argmax-safe."""
    s, Ls = full_model["stream"]
    mm, Lmm = full_model["stream_mm"]
    d = _linf(s, Ls, mm, Lmm, _BID_SRCS)
    assert d < 1e-3, d           # tiny residue, never changes an argmax decode


@pytest.mark.parametrize("path", ["stream", "load", "stream_mm"])
def test_full_battery_all_paths(full_model, path):
    """Every build path + compute_mode runs the FULL op battery (incl. div/mod)."""
    m, L = full_model[path]
    _battery(m, L, FULL_BATTERY, evict=True, device="cpu")


def test_eviction_on_off_agree(full_model):
    """Eviction ON (base prune policy) and OFF give identical correct outputs."""
    m, L = full_model["stream"]
    _battery(m, L, FULL_BATTERY, evict=True, device="cpu")
    _battery(m, L, FULL_BATTERY, evict=False, device="cpu")


@pytest.mark.skipif(not _RUN_CUDA,
                    reason="set C4_TEST_CUDA=1 (needs cuda:0) to run the GPU rows")
def test_cuda_matches_cpu(full_model):
    """cuda:0 gives the SAME argmax-correct outputs as cpu (dense_kernel)."""
    m, L = full_model["stream"]
    _battery(m, L, FULL_BATTERY, evict=True, device="cuda:0")
    # sparse_mm on GPU too (argmax-identical).
    mm, Lmm = full_model["stream_mm"]
    _battery(mm, Lmm, FULL_BATTERY, evict=True, device="cuda:0")
