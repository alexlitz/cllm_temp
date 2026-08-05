"""Log-sink divide toggle tests (C4_LOGSINK_DIV / C4_LOGSINK_DIV_FP32).

CHEAP tests only (reference byte-exactness + block-list wiring); no dense Qwen2Model
build (that full-densifies to ~108GB — see the model-load OOM memory note).  The
end-to-end model bake is validated via the golden fingerprint gate (flag-OFF byte-
identical `069cc32f`) and the reference byte-exactness (the blocks faithfully bake the
verified reference algorithm).
"""
import os
import random

import pytest

from . import isa  # noqa: F401 (import-safety)
from .nibble_muldivmod import divmod32


# ---------------------------------------------------------------------------
# The 32-bit divide battery: boundary, carry, div-by-1, div-by-self, large,
# small-b (large-quotient), signed bit-patterns, random.
# ---------------------------------------------------------------------------
def _battery(n_random: int = 20000):
    rng = random.Random(0xC4)
    edges = [0, 1, 2, 3, 7, 15, 16, 17, 255, 256, 257, 4095, 4096, 65535, 65536,
             65537, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF,
             0x0F0F0F0F, 0xF0F0F0F0, 0x11111111, 1000000, 999999999,
             0xCAFEBABE, 0xDEADBEEF, 251, 127, 8191]
    t = [(a, b) for a in edges for b in edges]
    for a in edges:
        t.append((a, 1))
        if a:
            t.append((a, a))
    for b in (1, 2, 3, 7, 13, 251, 65521):
        for a in (0xFFFFFFFF, 0xFFFFFFFE, 0x80000000, 0x7FFFFFFF):
            t.append((a, b))
    for _ in range(n_random):
        t.append((rng.randint(0, 0xFFFFFFFF), rng.randint(0, 0xFFFFFFFF)))
    for _ in range(n_random):
        t.append((rng.randint(0, 0xFFFFFFFF), rng.randint(1, 4096)))
    return t


def test_fp64_logsink_reference_byteexact():
    """The fp64 log-sink reference (the algorithm C4_LOGSINK_DIV bakes) is byte-exact
    vs divmod32 on the full 32-bit battery (quotient AND remainder)."""
    from . import nibble_logsink_div as LS
    bad = 0
    for a, b in _battery():
        if (LS.divmod_logsink(a, b)) != divmod32(a, b):
            bad += 1
    assert bad == 0, f"{bad} fp64 log-sink mismatches"


def test_fp32_refine_byteexact_no_fp64():
    """The fp32 approximate-then-refine divide is byte-exact vs divmod32 AND uses NO
    fp64 (the reference asserts float32 on every product)."""
    from . import nibble_logsink_fp32 as LS32
    bad = 0
    for a, b in _battery():
        if (LS32.divmod_logsink_fp32(a, b)) != divmod32(a, b):
            bad += 1
    assert bad == 0, f"{bad} fp32-refine mismatches"


def test_fp32_precision_analysis():
    """The self-checking precision analysis: ~20-bit fp32 reciprocal, 1 refine level
    + 1 correction, byte-exact, fp64-free."""
    from . import nibble_logsink_fp32 as LS32
    a = LS32.analyze()
    assert a["byteexact"] is True
    assert a["fp64_on_path"] is False
    assert a["refine_levels_needed"] == 1
    assert a["reciprocal_bits_correct"] >= 19.0
    assert a["int_correction_steps_max"] <= 1


def _specs(logsink: bool):
    from . import qwen_full_vm as Q
    key = "C4_LOGSINK_DIV"
    saved = os.environ.get(key)
    os.environ[key] = "1" if logsink else "0"
    try:
        QL = Q.QwenFullLayout(24, Q.SUBSET_MULDIV, efficient_alu=True,
                              code_from_memory=True, shift_via_mul=False)
        specs = Q._block_specs(QL.L, 24, Q.SUBSET_MULDIV, efficient_alu=True,
                               code_from_memory=True,
                               shift_via_mul=QL.shift_via_mul,
                               div_logsink=QL.div_logsink)
        return QL, [n for n, _ in specs]
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved


def test_flag_off_no_logsink_blocks():
    """Flag OFF: no log-sink blocks, radix-16 divide active (golden path)."""
    QL, names = _specs(logsink=False)
    assert QL.div_logsink is False
    assert QL.div_radix16 is True
    assert not any(n.startswith("ls") for n in names)
    assert getattr(QL.L, "LOGSINK", None) is None


def test_flag_on_wires_logsink_blocks():
    """Flag ON: log-sink blocks inserted, CAM-bake target present, DIV_RES shared with
    the ALU32 result band the ax-mux reads."""
    QL, names = _specs(logsink=True)
    assert QL.div_logsink is True
    assert QL.div_radix16 is False
    ls = [n for n in names if n.startswith("ls")]
    assert len(ls) > 50, f"only {len(ls)} log-sink blocks"
    assert "ls-recip-attn" in names            # the sink-CAM bake target
    assert QL.L.LOGSINK is not None
    assert QL.L.LOGSINK.DIV_RES == QL.L.ALU32.DIV_RES
    assert QL.L.LOGSINK.MOD_RES == QL.L.ALU32.MOD_RES
    assert len(getattr(QL.L, "_logsink_kdiv", set()) or []) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
