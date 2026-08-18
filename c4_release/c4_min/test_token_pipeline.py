"""Consolidated test for the #923 literal-0.5B WHOLE-ISA token pipeline.

Three off-build-path modules prove the whole ISA fits a LITERAL stock-Qwen2.5
0.5B geometry (hidden <= 896, <= 24 layers PER emission TOKEN) byte-exact, by
SLICING each VM step's compute across the step's ~30 register-emission tokens and
carrying the boundary values across via the real softmax1+ALiBi KV-memory CAM:

  * ``_token_pipeline_divmod``   — the DIV/MOD half (#922): full-32-bit-exact
    long-division sliced across tokens (6 tokens/DIV, per-token d_model <= 896).
  * ``_token_pipeline_base``     — the BASE VM half (#923): fetch/decode/PC/AX/
    SP/BP/STACK0/ALU/cmp/bitwise/dispatch/branch/fold FFN blocks remapped onto a
    COMPACT residual and sliced across tokens (byte-identical SwiGLU by column/row
    selection of the weights).
  * ``_token_pipeline_whole_isa``— the two assembled into ONE token-pipelined
    whole-ISA model, run on REAL programs byte-exact end-to-end vs ``isa.interpret``.

LANES
-----
FAST lane (default, run every commit; CPU-safe, RSS-bounded, no golden build):
  * ``test_divmod_pipeline_byte_exact`` — the #922 divmod token pipeline on a small
    operand subset, byte-exact vs ``nibble_muldivmod.div32/mod32`` (== the DIV/MOD
    of ``isa.interpret``, per the module docstring). ~600 MB, ~15 s.
  * ``test_base_pipeline_slice_byte_identical`` — the base VM FFN blocks run
    token-SLICED reproduce the single-forward compute BYTE-FOR-BYTE (``torch.equal``,
    the load-bearing "slicing is byte-identical" invariant). NOTE: constructing the
    whole-ISA ``QwenFullLayout`` + base ``_block_specs`` costs ~1.3-2.0 GB RSS (the
    whole-ISA 2926-dim layout is the floor, independent of the operand subset), so
    this lane's peak RSS is ~2 GB — under the 4 GB hard abort but above the 1 GB
    soft target; the divmod lane alone is < 1 GB.

SLOW lane (``@pytest.mark.slow``, NOT run in the default lane; > 190 s):
  * ``test_whole_isa_end_to_end`` — the assembled whole-ISA token pipeline on real
    programs + full-32-bit DIV/MOD byte-exact end-to-end vs ``isa.interpret``. This
    builds the radix-16-lean sparse forward (attention CAMs); still RSS-safe
    (< ~2 GB, watchdog aborts > 4 GB) but slow, so it is excluded from the fast lane.

MEASURED, off-build-path. Golden 174ece66 is UNTOUCHED (these are NEW files; the
neural-model build path is not modified — see ``test_golden_174ece66_untouched``).
"""
from __future__ import annotations

import resource

import pytest


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


# ===========================================================================
# FAST lane — divmod token pipeline byte-exact vs the interpreter reference.
# ===========================================================================
# (a, b) operands spanning the DIV/MOD boundary regimes the module targets:
#   b=1, b=2^k, b~256, b=2^16, q>2^16, q~max, b>a (q=0), a==b (q=1), b=0 guard.
_DIVMOD_CASES = [
    (0xFFFFFFFF, 3),          # q ~ 1.4e9
    (0xDEADBEEF, 0x1234),     # arbitrary large
    (700003, 7),              # q > 2^16
    (0xFFFFFFFF, 0xFFFF),     # q = 65537
    (2 ** 31, 7),
    (0xFFFFFFFF, 1),          # b = 1
    (0xFFFFFFFF, 2),          # b = 2^1
    (0xFFFFFFFF, 256),        # b = 2^8
    (0xFFFFFFFF, 0x10000),    # b = 2^16
    (0xFFFFFFFF, 0xFFFFFFFF),  # a == b -> q=1
    (0x1234, 0xFFFFFFFF),     # b > a -> q=0
    (12345, 0),               # b = 0 guard (q=r=0)
]


def test_divmod_pipeline_byte_exact():
    """#922 divmod token pipeline byte-exact vs ``nibble_muldivmod`` (the DIV/MOD of
    ``isa.interpret``) over a small full-32-bit operand subset.  CPU-safe (< 1 GB)."""
    from c4_min import _token_pipeline_divmod as TPD
    from c4_min import nibble_muldivmod as NM

    L = TPD.TPLayout(896)
    blocks = TPD.build_parallel_divmod_blocks(L, refine=True)

    ok = 0
    for a, b in _DIVMOD_CASES:
        q, r, ntok = TPD.run_packed_divmod(a, b, L, blocks, cap=23)
        want_q = (NM.div32(a, b) if b else 0) & 0xFFFFFFFF
        want_r = (NM.mod32(a, b) if b else 0) & 0xFFFFFFFF
        assert q == want_q and r == want_r, (
            f"divmod {a}/{b}: pipeline (q={q}, r={r}) != reference "
            f"(q={want_q}, r={want_r})")
        assert ntok <= 24, f"divmod {a}/{b} used {ntok} tokens (> 24-per-token cap)"
        ok += 1
    assert ok == len(_DIVMOD_CASES)
    assert _rss_mb() < 3800, f"RSS {_rss_mb()} MB exceeded safe bound"


# ===========================================================================
# FAST lane — base VM FFN blocks: token-slicing is BYTE-IDENTICAL to the single
# forward (the load-bearing invariant; the compact remap + cross-token CAM carry
# reproduce the whole-ISA base FFN math exactly).
# ===========================================================================
def test_base_pipeline_slice_byte_identical():
    """The base VM FFN blocks run token-SLICED reproduce the single-forward compute
    byte-for-byte (``torch.equal``).  NOTE peak RSS ~2 GB (whole-ISA layout floor)."""
    import torch

    from c4_min import _token_pipeline_base as TPB

    base_specs, L, _QL = TPB.build_base_blocks()
    cmap, compact = TPB.build_compact_base(base_specs, L)

    # per-token geometry is CONSTRUCTED <= the literal-0.5B budget
    assert cmap.D <= 896, f"base per-token d_model {cmap.D} > 896"

    torch.manual_seed(0)
    n = 3
    ok = 0
    for _ in range(n):
        x0 = torch.randn(cmap.D, dtype=torch.float32) * 0.1
        x0[cmap.ONE] = 1.0
        ref = TPB.run_base_single_token(x0, compact)
        sliced, ntok = TPB.run_base_token_pipeline(x0, cmap, compact, cap=23)
        assert torch.equal(ref, sliced), (
            "token-sliced base FFN diverged from the single forward "
            f"(max abs {float((ref - sliced).abs().max())})")
        assert ntok >= 2, "expected the base FFN to slice into >= 2 tokens"
        ok += 1
    assert ok == n
    assert _rss_mb() < 3800, f"RSS {_rss_mb()} MB exceeded safe bound"


# ===========================================================================
# SLOW lane — the assembled whole-ISA token pipeline on real programs, byte-exact
# end-to-end vs ``isa.interpret`` (incl full-32-bit DIV/MOD).  > 190 s; NOT in the
# default lane.  RSS-safe (< ~2 GB) but slow — builds the radix-16-lean sparse
# forward's attention CAMs.
# ===========================================================================
@pytest.mark.slow
def test_whole_isa_end_to_end():
    """Full end-to-end whole-ISA token pipeline byte-exact vs ``isa.interpret``."""
    from c4_min import _token_pipeline_whole_isa as W

    ok = W.main()
    assert ok, "whole-ISA token pipeline NOT byte-exact end-to-end"
    assert _rss_mb() < 3800, f"RSS {_rss_mb()} MB exceeded safe bound"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "not slow"]))
