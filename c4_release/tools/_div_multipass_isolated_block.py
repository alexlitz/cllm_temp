"""ISOLATED-BLOCK NaN + correctness probe for the C4_DIV_MULTIPASS fix.

Builds the flag-ON model ONCE (CPU), extracts the live ``MultiPassDivBlock``
(physical block 30), and feeds it a SYNTHETIC block-input residual that
reproduces the REAL teacher-forced input footprint the a91d2b36 root-cause
pinned: clean marker/opcode one-hots at value ~1.0 but the DIVIDEND (ALU_LO/HI)
and DIVISOR (AX_CARRY_LO/HI) operand nibble bands scaled to max_abs ~72 (the
active lane amplified by the attention relays, and the inactive lanes driven
NEGATIVE by the L6 FFN) — exactly the input on which the un-clamped cascade
overflows to inf -> NaN by pass 21.

Two things are asserted, for every (a,b) in a 30-pair div/mod sweep:

  1. NO NaN in the block OUTPUT on the dirty (max_abs ~72) residual.
  2. The routed OUTPUT_LO/HI decode to the correct quotient (OP_DIV) /
     remainder (OP_MOD) nibble — i.e. the operand clamp not only stops the
     NaN but the cascade computes the RIGHT answer once seeded clean.

Also runs a CLEAN (residual==1.0) control to prove the fix is a NO-OP there
(the clamp of a clean one-hot is idempotent).

CPU-only. Isolated block: the rest of the model is not forwarded, so this is
cheap after the one build and never touches the GPU.
"""
import os
import sys

os.environ.setdefault("C4_CAMPAIGN", "1")
os.environ["C4_DIV_MULTIPASS"] = "1"

_THIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _THIS in sys.path:
    sys.path.remove(_THIS)
sys.path.insert(0, _THIS)

import torch  # noqa: E402


def _find_div_block(model):
    for i, b in enumerate(model.blocks):
        ffn = getattr(b, "ffn", None)
        if getattr(ffn, "_is_multipass_div_block", False):
            return i, ffn
        for po in (getattr(b, "post_ops", None) or []):
            if getattr(po, "_is_multipass_div_block", False):
                return i, po
    return None, None


# The 30-pair div/mod sweep (mix of single/multi-byte, /1, edge, b==0 -> q=0,r=a).
PAIRS = [
    (0, 1), (1, 1), (5, 1), (10, 3), (17, 5), (42, 6), (84, 2), (100, 7),
    (106, 4), (127, 3), (128, 4), (129, 8), (200, 13), (240, 16), (255, 1),
    (255, 15), (255, 16), (255, 255), (36, 6), (99, 9), (81, 9), (63, 7),
    (144, 12), (169, 13), (196, 14), (225, 15), (250, 5), (7, 2), (13, 4),
    # b==0 convention (q=0, r=a):
    (55, 0),
]


def _build_dirty_input(block_module, dp, d_model, opcode, dirty):
    """One [1, N, d_model] residual: N rows, each a div/mod-AX operand frame.

    ``opcode`` in {"OP_DIV","OP_MOD"}. If ``dirty`` the operand one-hots are
    scaled to ~72 and inactive lanes set to a negative floor (the real
    footprint); else they are clean unit one-hots (the control).
    """
    N = len(PAIRS)
    X = torch.zeros(1, N, d_model)
    mark = dp["MARK_AX"]
    op = dp[opcode]
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]
    ax_lo, ax_hi = dp["AX_CARRY_LO"], dp["AX_CARRY_HI"]

    # Amplitude of the active lane and the negative floor of inactive lanes on
    # the real residual (root cause: max_abs ~72, inactive lanes ~ -5).
    ACT = 72.0 if dirty else 1.0
    NEG = -5.0 if dirty else 0.0

    def set_nibble(row, base, value):
        # inactive lanes -> negative floor, then the active lane -> ACT.
        X[0, row, base:base + 16] = NEG
        X[0, row, base + (value & 0xF)] = ACT

    for row, (a, b) in enumerate(PAIRS):
        X[0, row, mark] = 1.0
        X[0, row, op] = 1.0
        set_nibble(row, alu_lo, a & 0xF)
        set_nibble(row, alu_hi, (a >> 4) & 0xF)
        set_nibble(row, ax_lo, b & 0xF)
        set_nibble(row, ax_hi, (b >> 4) & 0xF)
    return X


def _decode_output(y, dp):
    """Decode OUTPUT_LO/HI byte per row (argmax over the two nibble bands)."""
    olo, ohi = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
    lo = y[0, :, olo:olo + 16].argmax(dim=-1)
    hi = y[0, :, ohi:ohi + 16].argmax(dim=-1)
    return (hi.long() << 4) | lo.long()


def _run(block_module, dp, d_model, opcode, dirty):
    X = _build_dirty_input(block_module, dp, d_model, opcode, dirty)
    in_max = float(X[:, :, :].abs().max())
    with torch.no_grad():
        Y = block_module(X)
    n_nan = int(torch.isnan(Y).sum())
    out_max = float("nan") if n_nan else float(Y.abs().max())
    dec = _decode_output(Y, dp)
    if opcode == "OP_DIV":
        exp = torch.tensor([0 if b == 0 else a // b for a, b in PAIRS])
    else:
        exp = torch.tensor([a if b == 0 else a % b for a, b in PAIRS])
    bad = (dec != exp)
    nbad = int(bad.sum())
    tag = "DIRTY(max_abs~72)" if dirty else "CLEAN(1.0)"
    print(f"  [{opcode} {tag}] in_max={in_max:.1f} out_nan={n_nan} "
          f"out_max={out_max:.3g} result_fails={nbad}/{len(PAIRS)}")
    if nbad:
        idxs = bad.nonzero().flatten()[:8]
        for i in idxs:
            a, b = PAIRS[int(i)]
            print(f"      FAIL a={a} b={b}: got {int(dec[i])} want {int(exp[i])}")
    return n_nan, nbad


def main():
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    print("[isolated-block] building flag-ON model (C4_DIV_MULTIPASS=1)...")
    model, layout = compile_full_vm_dynamic(disk_cache=False)
    model = model.eval()
    dp = layout.dim_positions
    d_model = int(model.blocks[0].ffn.W_up.shape[1]) if hasattr(
        model.blocks[0].ffn, "W_up"
    ) else int(model.blocks[0].attn.dim)

    idx, block_module = _find_div_block(model)
    if block_module is None:
        print("[isolated-block] NO MultiPassDivBlock found — flag not live?")
        return 3
    has_clamp = getattr(block_module, "alu_lo", None) is not None
    print(f"[isolated-block] div block idx={idx} d_model={d_model} "
          f"operand_clamp_wired={has_clamp}")

    total_nan = 0
    total_bad = 0
    print("[isolated-block] --- DIRTY residual (the real max_abs~72 footprint) ---")
    for opc in ("OP_DIV", "OP_MOD"):
        n_nan, nbad = _run(block_module, dp, d_model, opc, dirty=True)
        total_nan += n_nan
        total_bad += nbad
    print("[isolated-block] --- CLEAN residual (control; fix is a no-op) ---")
    for opc in ("OP_DIV", "OP_MOD"):
        n_nan, nbad = _run(block_module, dp, d_model, opc, dirty=False)
        total_nan += n_nan
        total_bad += nbad

    ok = (total_nan == 0 and total_bad == 0)
    print(f"\n[isolated-block] TOTAL nan={total_nan} result_fails={total_bad} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if ok:
        print("[isolated-block] *** The operand clamp keeps the cascade NaN-free "
              "on the real max_abs~72 residual AND divmod computes correct. ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
