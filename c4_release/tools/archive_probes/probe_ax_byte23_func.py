#!/usr/bin/env python3
"""Probe the AX bytes-1/2/3 register-dump leak on func_identity_0 step-1 (ENT).

Builds with all the prologue building-block flags ON, decodes func_identity_0
(id 550, identity(70)), locates the per-step AX register markers, and does
exact LM-head logit attribution at the byte-1, byte-2, byte-3 dump predictor
rows of step-1 (the callee ENT step) where AX should be 0 but leaks
0x02/0x09/0x0A. Compares to a WORKING add step where AX high bytes dump 0.

The model has no final norm, so logit[t] = head.weight[t] . resid + bias[t] is
fully attributable. We attribute (logit[got] - logit[want=0]) per residual dim.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_ax_byte23_func.py
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
# All building-block flags ON (defaults already ON except SE_SUPPRESS).
os.environ.setdefault("C4_BP_SAVE_DUMP", "1")
os.environ.setdefault("C4_ENT_SP_BYTE1_FF_H1_HARDEN", "1")
os.environ.setdefault("C4_PSH_ARG_VAL_AX", "1")
os.environ.setdefault("C4_POST_ENT_SE_SUPPRESS", "1")
os.environ.setdefault("C4_AX_BYTE1_DUMP", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tools.probe_groundtruth import GroundTruthProbe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402

# Resolve dims from the BUILT layout (never static registry).
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)


def _built_layout():
    _model, layout = compile_full_vm_dynamic(disk_cache=True)
    return dict(layout.dim_positions)


_DP = _built_layout()
# name_for: map a dim index back to the best (smallest) enclosing named band.
_SLOT_SIZES = {}


def _build_slot_index():
    # dim_positions maps name->start. We don't have sizes here, so infer ranges
    # from the sorted starts: each named start owns up to the next start.
    items = sorted(_DP.items(), key=lambda kv: kv[1])
    starts = [v for _, v in items]
    names = [k for k, _ in items]
    return items, starts, names


_ITEMS, _STARTS, _NAMES = _build_slot_index()


def name_for(pos):
    """Best-effort: nearest named band start <= pos, plus offset."""
    best = None
    for nm, st in _DP.items():
        if st <= pos and (best is None or st > best[1]):
            best = (nm, st)
    if best is None:
        return f"dim{pos}"
    return f"{best[0]}+{pos - best[1]}"


SRC_FUNC = ("int identity(int x) { return x; }\n"
            "            int main() { return identity(70); }")
SRC_ADD = "int main() { return 654 + 114; }"   # working: AX=768, byte1=0x02


def decode_steps(probe, bytecode, max_steps):
    """Return list of (ax_marker_pos, [b0,b1,b2,b3]) per AX-marker step."""
    trace = probe.probe(bytecode, max_steps=max_steps)
    RAX = int(Token.REG_AX)
    positions = sorted(trace.keys())
    markers = [p for p in positions if trace[p]["token"] == RAX]
    out = []
    for m in markers:
        bs = [trace.get(m + 1 + b, {}).get("token") for b in range(4)]
        out.append((m, bs))
    return out


@torch.no_grad()
def attribute_row(probe, bytecode, pos, max_steps, want, got, topn=20):
    """Exact LM-head logit attribution at predictor row ``pos``.

    Returns (logit_got, logit_want, sorted list of (dim, res, dW, contrib))
    where contrib drives (logit[got] - logit[want]).
    """
    model = probe.model
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = model.head.bias
    last_block = len(model.blocks) - 1
    ctx = probe._final_context(bytecode, max_steps=max_steps)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = model.forward(padded, stop_after_block=last_block)  # [1,S,D]
    res = x[0, pos]
    if res.is_sparse:
        res = res.to_dense()
    res = res.to(W.device).float().contiguous()
    dw = ((W[got] - W[want]) * res).to_dense() if (W[got] - W[want]).is_sparse \
        else (W[got] - W[want]) * res
    if dw.is_sparse:
        dw = dw.to_dense()
    dw = dw.float().cpu()
    logit_got = float((W[got] * res).sum() + b[got])
    logit_want = float((W[want] * res).sum() + b[want])
    order = torch.argsort(dw.abs(), descending=True)
    res_cpu = res.cpu()
    Wd = (W[got] - W[want])
    Wd = Wd.to_dense() if Wd.is_sparse else Wd
    Wd = Wd.cpu()
    rows = []
    for di in order[:topn].tolist():
        rows.append((di, float(res_cpu[di]),
                     float(Wd[di]), float(dw[di])))
    return logit_got, logit_want, rows


def main():
    probe = GroundTruthProbe.build()
    print("STEP_TOKENS =", int(Token.STEP_TOKENS))

    bc_func, _ = compile_c(SRC_FUNC)
    bc_add, _ = compile_c(SRC_ADD)

    print("\n=== func_identity_0 (identity(70)) per-step AX dump ===")
    fsteps = decode_steps(probe, bc_func, max_steps=8)
    for s, (m, bs) in enumerate(fsteps):
        dec = (bs[0] | (bs[1] << 8) | (bs[2] << 16) | (bs[3] << 24)
               if all(x is not None for x in bs) else None)
        decs = f"0x{dec:08x}" if dec is not None else "??"
        print(f"  step {s}: marker {m}  AX bytes {bs}  AX={decs}")

    print("\n=== add 654+114 (working) per-step AX dump ===")
    asteps = decode_steps(probe, bc_add, max_steps=4)
    for s, (m, bs) in enumerate(asteps):
        dec = (bs[0] | (bs[1] << 8) | (bs[2] << 16) | (bs[3] << 24)
               if all(x is not None for x in bs) else None)
        decs = f"0x{dec:08x}" if dec is not None else "??"
        print(f"  step {s}: marker {m}  AX bytes {bs}  AX={decs}")

    # ---- Attribution: func step-1 (ENT), bytes 1/2/3 (want=0, got=leak) ----
    if len(fsteps) < 2:
        print("\n[!] func has <2 AX steps; cannot probe step-1")
        return
    m1, bs1 = fsteps[1]
    print(f"\n########## func step-1 (ENT) marker {m1}  AX bytes {bs1} ##########")
    for byte_idx in (1, 2, 3):
        got = bs1[byte_idx]
        if got is None:
            continue
        pos = m1 + 1 + byte_idx  # predictor row for byte ``byte_idx``
        # predictor row = the row whose NEXT-token logits emit byte_idx; that's
        # the row at marker + byte_idx (its argmax = byte token byte_idx). The
        # byte token sits at marker+1+byte_idx, predicted from marker+byte_idx.
        pred = m1 + byte_idx
        lg, lw, rows = attribute_row(probe, bc_func, pred, 8,
                                     want=0x00, got=got, topn=22)
        print(f"\n--- byte-{byte_idx}: got=0x{got:02x} want=0x00  "
              f"predictor row {pred} ---")
        print(f"    logit[0x{got:02x}]={lg:.3f}  logit[0x00]={lw:.3f}  "
              f"diff(got-00)={lg - lw:.3f}")
        print(f"    top dims driving (logit_got - logit_00):")
        for di, r, dW, c in rows:
            print(f"      dim {di:4d} {name_for(di):30s} "
                  f"res={r:9.3f} dW={dW:7.3f} contrib={c:9.3f}")

    # ---- Working comparison: add step-0 (IMM, AX=768) bytes 1/2/3 ----
    if asteps:
        m0, bs0 = asteps[0]
        print(f"\n########## add step-0 marker {m0}  AX bytes {bs0} (AX=768) ##########")
        # byte1 should be 0x02, bytes 2/3 should be 0x00.
        for byte_idx in (2, 3):
            got = bs0[byte_idx]
            if got is None:
                continue
            pred = m0 + byte_idx
            lg, lw, rows = attribute_row(probe, bc_add, pred, 4,
                                         want=0x00, got=max(got, 1), topn=12)
            print(f"\n--- (working) byte-{byte_idx}: got=0x{got:02x} "
                  f"predictor row {pred} ---")
            print(f"    logit[0x00]={lw:.3f}  (this byte correctly emits 0)")
            for di, r, dW, c in rows[:10]:
                print(f"      dim {di:4d} {name_for(di):30s} "
                      f"res={r:9.3f} dW={dW:7.3f} contrib={c:9.3f}")


if __name__ == "__main__":
    main()
