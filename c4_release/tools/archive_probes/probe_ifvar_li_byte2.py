#!/usr/bin/env python3
"""Probe the if_var id425 LI-step AX byte-2 register-dump leak (0x01).

id425 = ``int main(){int x; x=96; if(x>59) return 1; return 0;}``. The first
divergence is the ``LI x`` load (pc9): the dump emits AX=0x010060 (byte-2=0x01)
where it should be 0x000060. This probe locates the per-step AX markers,
confirms the LI step, and does exact LM-head logit attribution at the byte-2
(and byte-1/3) dump predictor rows -- attributing ``(logit[got]-logit[0])`` per
residual dim, so the leaking band is nameable. Mirrors probe_ax_byte23_func.py.

Run: CUDA_VISIBLE_DEVICES=1 python tools/probe_ifvar_li_byte2.py
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tools.probe_groundtruth import GroundTruthProbe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)


def _built_layout():
    _model, layout = compile_full_vm_dynamic(disk_cache=True)
    return dict(layout.dim_positions)


_DP = _built_layout()


def name_for(pos):
    best = None
    for nm, st in _DP.items():
        if st <= pos and (best is None or st > best[1]):
            best = (nm, st)
    if best is None:
        return f"dim{pos}"
    return f"{best[0]}+{pos - best[1]}"


SRC = "int main() { int x; x = 96; if (x > 59) return 1; return 0; }"


def decode_steps(probe, bytecode, max_steps):
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
def attribute_row(probe, bytecode, pred, max_steps, want, got, topn=24):
    model = probe.model
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = model.head.bias
    last_block = len(model.blocks) - 1
    ctx = probe._final_context(bytecode, max_steps=max_steps)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = model.forward(padded, stop_after_block=last_block)
    res = x[0, pred]
    if res.is_sparse:
        res = res.to_dense()
    res = res.to(W.device).float().contiguous()
    Wd = (W[got] - W[want])
    Wd = Wd.to_dense() if Wd.is_sparse else Wd
    dw = (Wd * res)
    if dw.is_sparse:
        dw = dw.to_dense()
    dw = dw.float().cpu()
    logit_got = float((W[got] * res).sum() + b[got])
    logit_want = float((W[want] * res).sum() + b[want])
    order = torch.argsort(dw.abs(), descending=True)
    res_cpu = res.cpu()
    Wd = Wd.cpu()
    rows = []
    for di in order[:topn].tolist():
        rows.append((di, float(res_cpu[di]), float(Wd[di]), float(dw[di])))
    return logit_got, logit_want, rows


def main():
    probe = GroundTruthProbe.build()
    print("STEP_TOKENS =", int(Token.STEP_TOKENS))
    bc, _ = compile_c(SRC)
    steps = decode_steps(probe, bc, max_steps=16)
    print("\n=== id425 per-step AX dump ===")
    li_step = None
    for s, (m, bs) in enumerate(steps):
        dec = (bs[0] | (bs[1] << 8) | (bs[2] << 16) | (bs[3] << 24)
               if all(x is not None for x in bs) else None)
        decs = f"0x{dec:08x}" if dec is not None else "??"
        flag = ""
        if bs[2] not in (None, 0):
            flag = "  <<< BYTE-2 LEAK"
            li_step = (s, m, bs)
        print(f"  step {s}: marker {m}  AX bytes {bs}  AX={decs}{flag}")

    if li_step is None:
        print("\n[!] no byte-2 leak step found")
        return
    # Per-byte-row gate-dim scan for the LI step (so the gate doesn't touch
    # byte-0/byte-1, which carry the real loaded value).
    @torch.no_grad()
    def per_byte_scan(marker, names):
        model = probe.model
        last_block = len(model.blocks) - 1
        ctx = probe._final_context(bc, max_steps=16)
        padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
        x = model.forward(padded, stop_after_block=last_block)
        print(f"\n=== LI step per-byte-row scan (marker {marker}) ===")
        hdr = "  row " + "".join(f"{n:>16s}" for n in names)
        print(hdr)
        for bi in range(4):
            row = marker + bi  # predictor row for byte bi
            d = x[0, row].to_dense() if x.is_sparse else x[0, row]
            vals = []
            for n in names:
                base = n.split("+", 1)[0]
                off = int(n.split("+", 1)[1]) if "+" in n else 0
                vals.append(float(d[_DP[base] + off]) if base in _DP else float("nan"))
            print(f"  b{bi}  " + "".join(f"{v:16.3f}" for v in vals))
    per_byte_scan(li_step[1], ["OP_LI", "ADDR_KEY+16", "BYTE_INDEX_1",
                               "BYTE_INDEX_2", "OUTPUT_LO+0", "OUTPUT_LO+1"])

    s, m, bs = li_step
    # find a genuine LEA step (AX high bytes 0, byte0/1 = address ffe8) just
    # before the LI step, for a clean reference byte-2 row.
    lea_marker = None
    for ss, (mm, bb) in enumerate(steps):
        if ss < s and bb[1] == 0xff and bb[2] in (0, None):
            lea_marker = mm
    # a clean IMM step (AX small, byte2=0) for a second clean reference.
    imm_marker = None
    for ss, (mm, bb) in enumerate(steps):
        if ss < s and bb[0] not in (None, 0) and bb[1] == 0 and bb[2] in (0, None):
            imm_marker = mm
    if lea_marker is not None:
        dump_gate_dims(probe, bc, m + 2, lea_marker + 2, 16,
                       pred_imm=(imm_marker + 2 if imm_marker else None))
    print(f"\n########## LI step {s} marker {m}  AX bytes {bs} ##########")
    for byte_idx in (1, 2, 3):
        got = bs[byte_idx]
        if got in (None, 0) and byte_idx != 2:
            # still probe byte2 even if 0; skip clean 0 bytes 1/3
            continue
        pred = m + byte_idx
        gtok = got if got else 1
        lg, lw, rows = attribute_row(probe, bc, pred, 16, want=0x00, got=gtok)
        print(f"\n--- byte-{byte_idx}: got=0x{(got or 0):02x} want=0x00  "
              f"predictor row {pred} ---")
        print(f"    logit[0x{gtok:02x}]={lg:.3f}  logit[0x00]={lw:.3f}  "
              f"diff(got-00)={lg - lw:.3f}")
        for di, r, dW, c in rows:
            print(f"      dim {di:4d} {name_for(di):32s} "
                  f"res={r:9.3f} dW={dW:7.3f} contrib={c:9.3f}")


@torch.no_grad()
def dump_gate_dims(probe, bytecode, pred_li, pred_lea, max_steps, pred_imm=None):
    """Dump the lea_first_step_ax_byte2 gate dims at the LI-step byte-2 row
    vs a genuine LEA-step byte-2 row (final residual; the gate dims persist)."""
    model = probe.model
    last_block = len(model.blocks) - 1
    ctx = probe._final_context(bytecode, max_steps=max_steps)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = model.forward(padded, stop_after_block=last_block)
    gate_names = ["CMP+7", "H1+1", "IS_BYTE", "BYTE_INDEX_1", "HAS_SE",
                  "OP_LI", "OP_LI_RELAY", "OP_LEA", "OP_GT", "OP_ENT",
                  "OP_IMM", "OP_SI", "MARK_AX", "AX_CARRY_OVERFLOW",
                  "MEM_STORE", "STACK0_BYTE1",
                  "OUTPUT_LO+0", "OUTPUT_LO+1"]
    def _pos(nm):
        base = nm.split("+", 1)[0]
        off = int(nm.split("+", 1)[1]) if "+" in nm else 0
        if base not in _DP:
            return None
        return _DP[base] + off
    print("\n=== gate dims: LI(leak) vs LEA(clean) vs IMM(clean) byte2 rows ===")
    print(f"  {'dim':>22s}  {'LI-leak':>9s}  {'LEA':>9s}  {'IMM':>9s}")
    def _r(row, p):
        if row is None:
            return float("nan")
        return float(x[0, row].to_dense()[p]) if x.is_sparse else float(x[0, row, p])
    for nm in gate_names:
        p = _pos(nm)
        if p is None:
            print(f"  {nm:>22s}  (no dim)")
            continue
        print(f"  {nm:>22s}  {_r(pred_li, p):9.3f}  {_r(pred_lea, p):9.3f}  "
              f"{_r(pred_imm, p):9.3f}")

    # Find ALL dims where LI-leak row differs from BOTH clean rows by > 0.15,
    # to surface a strong LI-uniqueness discriminator for the gate.
    print("\n=== dims unique to LI-leak row (|LI - LEA|>0.15 AND |LI - IMM|>0.15) ===")
    li_d = x[0, pred_li].to_dense() if x.is_sparse else x[0, pred_li]
    le_d = x[0, pred_lea].to_dense() if x.is_sparse else x[0, pred_lea]
    im_d = (x[0, pred_imm].to_dense() if x.is_sparse else x[0, pred_imm]) \
        if pred_imm is not None else le_d
    diffs = []
    for di in range(li_d.shape[0]):
        a = float(li_d[di]); b = float(le_d[di]); c = float(im_d[di])
        if abs(a - b) > 0.15 and abs(a - c) > 0.15:
            diffs.append((abs(a - b) + abs(a - c), di, a, b, c))
    diffs.sort(reverse=True)
    for _, di, a, b, c in diffs[:30]:
        print(f"  dim {di:4d} {name_for(di):30s} LI={a:8.3f} LEA={b:8.3f} IMM={c:8.3f}")


if __name__ == "__main__":
    main()
