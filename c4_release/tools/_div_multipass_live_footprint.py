"""LIVE per-band footprint diff: MultiPassDivBlock vs FlattenedDivMod composite
on a REAL single-byte div program's emission row (not a synthetic frame).

ROOT CAUSE FOUND (2026-07-04, this probe): the flag-ON frame collapse is NOT a
footprint desync — it is a **NaN poisoning**. On the REAL decode residual the
MultiPassDivBlock (C4_DIV_MULTIPASS=1, physical block 30) receives a CLEAN input
(max_abs ~72, no NaN) but emits an ALL-NaN output (2997/2997 dims NaN) that
propagates to every downstream block and the LM head, so every div-step row
argmaxes to token 0 -> the step emits the wrong tokens -> next step's fixed
slice reads pc=None (step-2 collapse). Verified: div_6 (106/4=26) decodes AX
token 26 flag-OFF but token 0 flag-ON; block-29 residual clean, block-30 output
all-NaN.

Why the synthetic probes missed it: the isolated single-block diff
(``_div_multipass_footprint_diff.py``) + live-compute probe feed the cascade
CLEAN unit-magnitude one-hots (residual 1.0), where the 43-pass amplitude-
normalized cascade is stable. The REAL residual carries max_abs ~72 in the bands
the cascade reads; the normalized cascade (read weight 1.0, threshold k-0.5,
up=S*0.5, write 1/(S*0.5)) is NOT robust to that magnitude — the per-pass
amplitude AMPLIFIES instead of staying pinned, and the running residual EXPLODES
across the chained passes: max_abs reaches ~1.5e37 by pass 21 and OVERFLOWS to
inf -> NaN (localized pass-by-pass: block-29 out max_abs~72, clean through pass
20, ~1.5e37 into pass 21, all-NaN out). The AX_FULL footprint fix
(commits 14deb277 / 8fde9e5c) is therefore a NO-OP for the live verdict: the
emission-row SHARED bands even match byte-for-byte (0 delta) — the block just
NaNs its whole output. The real fix must make the cascade numerically robust to
the large real residual (clamp/renormalize the operand read + intermediate
passes, or hard-gate the block so a non-clean input never seeds the cascade).

This probe: teacher-forces BOTH models on the IDENTICAL flag-OFF decode context,
reads the residual after the div block (block 30) at the div AX row, reports (a)
shared-band cell deltas and (b) the per-row decoded-token diff across the div
step window — the (b) section is where the all-0 NaN collapse shows.
"""
import os
import sys

os.environ.setdefault("C4_CAMPAIGN", "1")

_THIS_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _THIS_PKG_ROOT in sys.path:
    sys.path.remove(_THIS_PKG_ROOT)
sys.path.insert(0, _THIS_PKG_ROOT)

import torch  # noqa: E402

# Fail loud if an editable c4_neural_vm install shadowed this worktree (the
# main-checkout copy lacks this branch's MultiPassDivBlock operand-clamp fix).
import neural_vm.efficient_alu_neural as _ean  # noqa: E402
import inspect as _inspect  # noqa: E402
assert "alu_lo" in _inspect.signature(_ean.MultiPassDivBlock.__init__).parameters, (
    f"WRONG neural_vm loaded (no alu_lo operand-clamp): {_ean.__file__}. "
    "Editable install shadowed this worktree — fix sys.path.")


def _find_div_block(model):
    for i, b in enumerate(model.blocks):
        ffn = getattr(b, "ffn", None)
        if getattr(ffn, "_is_multipass_div_block", False):
            return i, "multipass_ffn"
        if type(ffn).__name__ == "FlattenedDivMod":
            return i, "composite_ffn"
        for po in (getattr(b, "post_ops", None) or []):
            if getattr(po, "_is_multipass_div_block", False):
                return i, "multipass_post_op"
            if type(po).__name__ == "FlattenedDivMod":
                return i, "composite_post_op"
    return None, None


def _build_model(flag_on, disk_cache=False):
    if flag_on:
        os.environ["C4_DIV_MULTIPASS"] = "1"
    else:
        os.environ.pop("C4_DIV_MULTIPASS", None)
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    model, layout = compile_full_vm_dynamic(disk_cache=disk_cache)
    return model.eval(), layout


def main():
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    progs = generate_test_programs()
    src = None
    for s, ev, desc in progs:
        if desc.startswith("div_6:"):
            src = s
            break
    assert src is not None, "div_6 not found"
    bytecode, data = compile_c(src)
    print(f"[live-footprint] program: div_6 (106/4) bytecode_len={len(bytecode)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[live-footprint] device={dev}")

    # OFF model + probe for the real decode context.
    from tools.probe_groundtruth import GroundTruthProbe
    print("[live-footprint] building flag-OFF probe (composite)...")
    os.environ.pop("C4_DIV_MULTIPASS", None)
    probe = GroundTruthProbe.build()
    model_off = probe.model.to(dev).eval()

    # dim_positions from a fresh compile (cached ok — flag-OFF layout).
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    _, layout_off = compile_full_vm_dynamic(disk_cache=True)
    dp = layout_off.dim_positions

    # real emission context via the probe's own loop.
    ctx = probe._final_context(bytecode)
    print(f"[live-footprint] OFF decode context len={len(ctx)}")
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    mark_ax, op_div = dp["MARK_AX"], dp["OP_DIV"]
    # OP_DIV is decoded onto the AX row only from block ~8 (opcode-flag relay),
    # so scan at block 8 (block 6 has MARK_AX but not yet OP_DIV).
    with torch.no_grad():
        early = model_off.forward(padded, stop_after_block=8)[0]
    rows = ((early[:, op_div] > 0.5) & (early[:, mark_ax] > 0.5)).nonzero().flatten().tolist()
    print(f"[live-footprint] div AX rows (OP_DIV & MARK_AX @ blk8): {rows}")
    if not rows:
        print("[live-footprint] no div AX row; abort.")
        return 3
    ax_row = rows[-1]

    div_off, where_off = _find_div_block(model_off)
    print(f"[live-footprint] OFF div block idx={div_off} ({where_off})")
    with torch.no_grad():
        resid_off = model_off.forward(padded, stop_after_block=div_off)[0, ax_row].cpu()
    print(f"[live-footprint] OFF div-block-out row{ax_row}: max_abs="
          f"{float(resid_off.abs().max()):.2f} "
          f"nan={bool(torch.isnan(resid_off).any())}")

    # ON model on the SAME context.
    print("[live-footprint] building flag-ON model (multipass)...")
    model_on, layout_on = _build_model(True, disk_cache=False)
    model_on = model_on.to(dev).eval()
    div_on, where_on = _find_div_block(model_on)
    print(f"[live-footprint] ON div block idx={div_on} ({where_on})")
    # Guard: ON build must carry the operand-clamp fix.
    import inspect as _inspect
    import neural_vm.efficient_alu_neural as _ean
    assert "alu_lo" in _inspect.signature(_ean.MultiPassDivBlock.__init__).parameters, (
        f"WRONG neural_vm (no alu_lo operand-clamp): {_ean.__file__}")
    with torch.no_grad():
        resid_on = model_on.forward(padded, stop_after_block=div_on)[0, ax_row].cpu()
    on_nan = bool(torch.isnan(resid_on).any())
    n_nan = int(torch.isnan(resid_on).sum())
    print(f"[live-footprint] ON div-block-out row{ax_row}: max_abs="
          f"{(float('nan') if on_nan else float(resid_on.abs().max())):.2f} "
          f"nan={on_nan} ({n_nan}/{resid_on.shape[-1]} dims)")
    if on_nan:
        print("[live-footprint] *** ROOT CAUSE: MultiPassDivBlock emits NaN on "
              "the REAL residual -> LM head decodes token 0 everywhere -> frame "
              "collapse. The block is numerically unstable on large-magnitude "
              "input (synthetic unit-magnitude probes never triggered it). ***")

    # NAME-based band diff: OFF and ON layouts may place shared bands at
    # DIFFERENT dim indices (the ON flag-gated bands widen + can repack the
    # model). Compare each shared named band's 16-cell window by NAME, using
    # each layout's own dim_positions.
    dp_on = layout_on.dim_positions
    scratch = ("DIV_MULTIPASS_WS", "DIV_MP_", "MUL_MULTIPASS_WS")
    shared_names = [nm for nm in dp
                    if isinstance(dp.get(nm), int) and nm in dp_on
                    and not nm.startswith(scratch)]
    print(f"\n[live-footprint] div AX row={ax_row}  OFF blk={div_off}  ON blk={div_on}")
    print(f"[live-footprint] OFF d_model={resid_off.shape[-1]} "
          f"ON d_model={resid_on.shape[-1]}")
    deltas = []
    Doff, Don = resid_off.shape[-1], resid_on.shape[-1]
    for nm in shared_names:
        bo, bn = int(dp[nm]), int(dp_on[nm])
        for k in range(16):
            do, dn = bo + k, bn + k
            if do >= Doff or dn >= Don:
                continue
            vo, vn = resid_off[do].item(), resid_on[dn].item()
            if abs(vn - vo) > 1e-2:
                deltas.append((nm, k, vo, vn, abs(vn - vo)))
    deltas.sort(key=lambda t: -t[4])
    print(f"[live-footprint] shared-band cells differing (>1e-2): {len(deltas)}")
    for nm, k, vo, vn, dd in deltas[:80]:
        print(f"  {nm+'+'+str(k):32s} off={vo:+.4f} on={vn:+.4f}  |d|={dd:.3f}")

    # ---- DECODED-TOKEN diff for the div step window ----
    # Both models are teacher-forced on the SAME context. The step emitter
    # counts tokens by decoding each row's LM-head argmax. If the div step
    # emits a different token COUNT, the frame desyncs. Compare per-row argmax
    # logits (full forward) for the div step's 30-token window.
    from neural_vm.batched_pure_neural import Token
    STEP = int(Token.STEP_TOKENS)
    win_start = ax_row  # AX marker row begins the emitted-byte window
    win_end = min(len(ctx), ax_row + STEP + 2)
    with torch.no_grad():
        log_off = model_off.forward(padded)[0]       # [S, V]
        log_on = model_on.forward(padded.to(next(model_on.parameters()).device))[0]
    tok_off = log_off.argmax(-1).cpu().tolist()
    tok_on = log_on.argmax(-1).cpu().tolist()
    print(f"\n[live-footprint] div-step token argmax diff rows [{win_start}..{win_end}):")
    ndiff = 0
    for r in range(win_start, win_end):
        a = tok_off[r] if r < len(tok_off) else None
        b = tok_on[r] if r < len(tok_on) else None
        mark = "  <-- DIFF" if a != b else ""
        if a != b:
            ndiff += 1
        if a != b or r < win_start + 6:
            print(f"  row {r:3d}: off_tok={a:4} on_tok={b:4}{mark}")
    print(f"[live-footprint] div-step rows with differing decoded token: {ndiff}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
