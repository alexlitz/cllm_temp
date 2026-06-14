#!/usr/bin/env python3
"""Localize the post-ENT 0xFF sentinel over-emitter to a physical block.

Replays func_identity_0 (id 550) spec_k=0 to the FULL final context, finds the
spurious-0xFF emission position (the first of the 2 leading 255 tokens of a
post-ENT step), then for each physical block runs the model truncated at that
block (stop_after_block) and projects the residual through the LM head to see
the logit for token 255 at that position. The block where logit[255] jumps to
the ~4.4e10 sentinel magnitude is the injector.

Also dumps, for the injecting block, the residual dims with the largest
contribution to the token-255 logit (head.weight[255, :] * residual), so we can
identify the sentinel dim and its gate.

Usage: python tools/_probe_sentinel_block.py [id]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa
from neural_vm.embedding import E  # noqa

TOK_255 = 255


def _dense(t):
    try:
        if t.layout != torch.strided:
            return t.to_dense()
    except Exception:
        try:
            return t.to_dense()
        except Exception:
            pass
    return t


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model
    device = probe._device
    SE = int(Token.STEP_END)
    STEP = int(Token.STEP_TOKENS)

    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    print(f"id{pid} {desc} exp={exp} ctxlen={len(ctx)} prompt_len={pl} blocks={len(model.blocks)}")

    # Locate spurious-255 positions: a 255 token that is the FIRST token of a step
    # (immediately follows a STEP_END) OR the 2nd token (also 255) right after.
    # Simplest: find indices i>=pl where ctx[i]==255 AND ctx[i-1]==SE (start of step).
    spurious = []
    for i in range(pl, len(ctx)):
        if ctx[i] == TOK_255 and ctx[i - 1] == SE:
            spurious.append(i)
    print(f"spurious-255 step-leading positions: {spurious}")
    if not spurious:
        print("no spurious leading 255 found; aborting")
        return

    # For each, the EMITTING position is i-1's logits row -> argmax==255.
    # The model predicts ctx[i] from logits[i-1]. So we want logits row (i-1).
    for sp in spurious:
        emit_row = sp - 1  # logits[emit_row] argmax should be 255
        print(f"\n=== spurious 255 at ctx pos {sp} (predicted from logits row {emit_row}) ===")
        padded = torch.tensor([ctx], dtype=torch.long, device=device)
        full_logits = model.forward(padded)[0]  # [S,V]
        row = full_logits[emit_row]
        topk = torch.topk(row, k=6)
        pairs = [(int(t), float(v)) for v, t in zip(topk.values.tolist(), topk.indices.tolist())]
        print(f"  final logits top6: {pairs}")
        print(f"  logit[255]={float(row[TOK_255].item()):.4e}")

        # Per-block: project residual through head, watch logit[255].
        Wh = model.head.weight  # [V, D]
        Wh = _dense(Wh)
        bh = model.head.bias if model.head.bias is not None else None
        if bh is not None: bh = _dense(bh)
        w255 = Wh[TOK_255]  # [D]
        b255 = float(bh[TOK_255].item()) if bh is not None else 0.0
        print(f"  per-block logit[255] (residual @ head.weight[255]):")
        prev = 0.0
        for b in range(len(model.blocks)):
            resid = model.forward(padded, stop_after_block=b)[0, emit_row]  # [D]
            resid = _dense(resid)
            l255 = float((resid * w255).sum().item()) + b255
            delta = l255 - prev
            flag = "  <== JUMP" if abs(delta) > 1e8 else ""
            print(f"    block {b:2d}: logit255={l255:.4e}  delta={delta:+.4e}{flag}")
            prev = l255

        # Identify the injecting block automatically (largest positive jump).
        prevv = 0.0; jump_block = None; best = 0.0
        deltas = []
        for b in range(len(model.blocks)):
            resid = model.forward(padded, stop_after_block=b)[0, emit_row]
            resid = _dense(resid)
            l255 = float((resid * w255).sum().item()) + b255
            deltas.append((b, l255 - prevv)); prevv = l255
        for b, d in deltas:
            if d > best:
                best = d; jump_block = b
        print(f"  injecting block = {jump_block} (delta +{best:.4e})")

        name_by_dim = {int(v): k for k, v in vars(E).items() if isinstance(v, int)}
        # Diff residual BEFORE vs AFTER the injecting block to find which dims it set.
        if jump_block is not None and jump_block > 0:
            r_before = model.forward(padded, stop_after_block=jump_block - 1)[0, emit_row]
            r_after = model.forward(padded, stop_after_block=jump_block)[0, emit_row]
            r_before = _dense(r_before); r_after = _dense(r_after)
            diff = r_after - r_before
            contrib_diff = diff * w255
            top = torch.topk(contrib_diff.abs(), k=15)
            print(f"  dims block {jump_block} CHANGED, by |Δcontribution| to logit[255]:")
            for v, d in zip(top.values.tolist(), top.indices.tolist()):
                d = int(d); nm = name_by_dim.get(d, "")
                print(f"    dim {d:4d} {nm:28s} before={float(r_before[d].item()):+.4e} "
                      f"after={float(r_after[d].item()):+.4e} w255={float(w255[d].item()):+.4e} "
                      f"Δcontrib={float(contrib_diff[d].item()):+.4e}")


if __name__ == "__main__":
    main()
