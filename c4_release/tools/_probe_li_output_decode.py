#!/usr/bin/env python3
"""Probe: byte-0 OUTPUT decode at LI rows across all blocks.

For a given id + step, teacher-forces the oracle tape and reports, at the
AX byte-0 emit row, the OUTPUT_LO / OUTPUT_HI nibble bands at the INPUT of
every block (pre-hook residual) so we can see WHICH block writes the wrong
byte-0. The L15 head-0 delivers the CLEAN_EMBED value into OUTPUT_LO; a
downstream tail block (L25 / block 35+) may overwrite it with 0x00.

Run:
  CUDA_VISIBLE_DEVICES="" C4_CAMPAIGN=1 VT_PROBE_ID=675 VT_STEP=12 python tools/_probe_li_output_decode.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
os.environ.setdefault("C4_CAMPAIGN", "1")
os.environ.setdefault("C4_JSR_BP_BYTE3_CLEAR", "1")
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402

_ID = int(os.environ.get("VT_PROBE_ID", "675"))
_STEP = int(os.environ.get("VT_STEP", "12"))
_TESTS = generate_test_programs()
SRC = _TESTS[_ID][0]
DESC = _TESTS[_ID][2]


def oracle_windows(bc):
    vm = DraftVM(list(bc))
    steps, toks = [], []
    for _ in range(60):
        if vm.halted:
            break
        if not vm.step():
            break
        steps.append((int(vm.pc) & 0xFFFFFFFF, int(vm.ax) & 0xFFFF))
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break
    return steps, toks


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = model.embed._dim_positions
    STEP = int(Token.STEP_TOKENS)

    bc, _ = compile_c(SRC)
    prompt = p._build_context(bc)
    pl = len(prompt)
    steps, toks = oracle_windows(bc)
    ctx = list(prompt)
    for w in toks:
        ctx.extend(w)
    padded = torch.tensor([ctx], device=p._device)

    caps = {}
    hooks = []
    for bi, blk in enumerate(model.blocks):
        def mk(bi):
            def _h(m, i):
                caps[bi] = i[0].detach().clone()
            return _h
        hooks.append(blk.register_forward_pre_hook(mk(bi)))
    # capture final output too
    final = {}
    def _fh(m, i, o):
        final["res"] = (o[0] if isinstance(o, tuple) else o).detach().clone()
    fh = model.blocks[-1].register_forward_hook(_fh)

    with torch.no_grad():
        logits = model(padded)
        if logits.is_sparse:
            logits = logits.to_dense()
    for h in hooks:
        h.remove()
    fh.remove()
    preds = torch.argmax(logits[0], dim=-1)

    pc, ax = steps[_STEP]
    base = pl + _STEP * STEP
    b0row = base + 5
    emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
    got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
    print(f"id{_ID} [{DESC}] step{_STEP} pc={pc} ax={ax:#06x} "
          f"got=0x{got:04x} {'OK' if got == ax else 'XX'} b0row={b0row}")

    def band(t, name):
        row = t[0, b0row] if t.dim() == 3 else t[b0row]
        b = row[dp[name]:dp[name] + 16]
        mx = float(b.max().item())
        idx = int(b.argmax().item())
        return (idx if mx > 0.3 else None, round(mx, 2))

    def fullband(t, name):
        row = t[0, b0row] if t.dim() == 3 else t[b0row]
        b = row[dp[name]:dp[name] + 16]
        return [round(float(x), 1) for x in b]

    print(f"\n{'block':>6} {'OUTPUT_LO':>18} {'OUTPUT_HI':>18}")
    for bi in sorted(caps):
        t = caps[bi][0]
        lo = band(t, "OUTPUT_LO")
        hi = band(t, "OUTPUT_HI")
        marker = ""
        if bi in (34, 35, 36, 37):
            marker = f"\n         LO16={fullband(t, 'OUTPUT_LO')}\n         HI16={fullband(t, 'OUTPUT_HI')}"
        print(f"{bi:>6} lo={str(lo):>14} hi={str(hi):>14}{marker}")
    if "res" in final:
        t = final["res"][0]
        lo = band(t, "OUTPUT_LO")
        hi = band(t, "OUTPUT_HI")
        print(f"{'FINAL':>6} lo={str(lo):>14} hi={str(hi):>14}")

    # LM-head logit for byte-0 token
    b0logits = logits[0, b0row]
    tk = torch.topk(b0logits, k=5)
    print("\nbyte-0 LM-head top-5 tokens:")
    for r in range(5):
        print(f"   tok={int(tk.indices[r])} logit={float(tk.values[r]):.1f}")


if __name__ == "__main__":
    main()
