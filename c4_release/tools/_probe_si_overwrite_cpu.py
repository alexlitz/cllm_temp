#!/usr/bin/env python3
"""CPU teacher-forced probe for the SI-store-addr CAM on the ABS-address path.

The verdict-mover gate for the L15 head-16 firing veto (flag C4_SI_STORE_ADDR).
It teacher-forces the ``test_si_li_overwrite`` program (store 10 @ 0x200; store
55 @ 0x200; LI 0x200 -> want 55) and the ``var_mul`` program (LI a -> want 23)
and reports, at each LI byte-0 lookup row:

  - the head-16 winning K row + its ADDR_B0/AX_CARRY (the store the CAM lands on),
  - the head-16 delivered value vs the head-0 delivered value,
  - the teacher-forced decoded AX (the argmax of the byte-0..3 emit rows).

THE BUG (agent a8653eca): on the ABS-address path 0x200, the LI-query AX_CARRY
byte-0 is 0x00 (both null nibbles hot), so head-16's per-nibble match keys the
NULL nibbles (slots 16/58) and null-matches a spurious ADDR_B0=0x00 store marker
(r~217) that delivers value 0, overriding head-0's correct 55.

THE FIX (this veto): head-16 must NOT fire when the LI-query AX_CARRY is
all-zero (null-null = no relative target) so it never overrides head-0 on the
abs-address path, while STILL firing for the var_mul relative-address case.

GATE (head-16 ON):
  - overwrite: ostep LI decode == 55 (NOT 0), and the head-16 winner is either
    the sink/self (veto engaged) OR at least NOT the spurious null store.
  - var_mul:   ostep LI decode == 23 (head-16 still delivers the relative case).

Run (CPU, contention-robust):
  CUDA_VISIBLE_DEVICES="" C4_SI_STORE_ADDR=1 C4_NO_STACK0_EMIT=1 \
      C4_OPERAND_FROM_MEMSP=1 python tools/_probe_si_overwrite_cpu.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU (contention-robust)
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
os.environ.setdefault("C4_SI_STORE_ADDR", "1")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _make_bytecode(ops):
    """Packed bytecode list from (Opcode, imm) tuples / bare Opcodes (smoke)."""
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


# The abs-address overwrite program is raw bytecode (the C compiler has no
# pointer-cast syntax) — IDENTICAL to tests/test_smoke.py test_si_li_overwrite.
_OVERWRITE_BC = _make_bytecode([
    (Opcode.IMM, 0x200), Opcode.PSH,
    (Opcode.IMM, 10), Opcode.SI,
    (Opcode.IMM, 0x200), Opcode.PSH,
    (Opcode.IMM, 55), Opcode.SI,
    (Opcode.IMM, 0x200), Opcode.LI,
    Opcode.EXIT,
])

# The two programs the veto must reconcile. Each is ("src"|None, bytecode|None,
# want): the var_mul relative case compiles from C; the overwrite abs case is
# raw bytecode.
PROGRAMS = {
    # store 10 @ 0x200; store 55 @ 0x200 (overwrite); LI 0x200 -> want 55.
    "overwrite": (None, _OVERWRITE_BC, 55),
    # LI a -> want 23 (the relative-address case head-16 was built for).
    "var_mul": ("int main(){int a;int b;a=23;b=47;return a*b;}", None, 23),
}
LI_STEPS = list(range(4, 24))
HEAD16 = 16
HEAD0 = 0


def oracle_windows(bc):
    vm = DraftVM(list(bc))
    steps, toks = [], []
    for _ in range(40):
        if vm.halted:
            break
        if not vm.step():
            break
        steps.append((int(vm.pc) & 0xFFFFFFFF, int(vm.ax) & 0xFFFF))
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break
    return steps, toks


def _dense(t):
    if t.is_sparse_csr or t.is_sparse:
        return t.to_dense()
    return t


def run_one(p, model, dp, L15, nH, src, bc_raw, want):
    STEP = int(Token.STEP_TOKENS)
    if bc_raw is not None:
        bc = list(bc_raw)
    else:
        bc, _ = compile_c(src)
    prompt = p._build_context(bc)
    pl = len(prompt)
    steps, toks = oracle_windows(bc)
    ctx = list(prompt)
    for w in toks:
        ctx.extend(w)
    padded = torch.tensor([ctx], device=p._device)

    cap = {}
    h = model.blocks[L15].register_forward_pre_hook(
        lambda m, i: cap.__setitem__("pre", i[0].detach().clone()))
    with torch.no_grad():
        logits = model(padded)
        if logits.is_sparse:
            logits = logits.to_dense()
    h.remove()
    preds = torch.argmax(logits[0], dim=-1)

    pre = cap["pre"][0]
    seq = pre.shape[0]
    attn = model.blocks[L15].attn
    hd = attn.W_q.shape[0] // nH
    Wq = _dense(attn.W_q)
    Wk = _dense(attn.W_k)
    Q = (pre @ Wq.T).view(seq, nH, hd)
    K = (pre @ Wk.T).view(seq, nH, hd)

    def nib(pos, base):
        band = pre[pos, dp[base]:dp[base] + 16]
        mx = float(band.max().item())
        return (int(band.argmax().item()), round(mx, 2)) if mx > 0.3 else (None, mx)

    def hexval(pos, lo, hi):
        l, _ = nib(pos, lo)
        hh, _ = nib(pos, hi)
        if l is None and hh is None:
            return None
        return ((hh or 0) << 4) | (l or 0)

    ok_any = False
    for si in LI_STEPS:
        if si >= len(steps):
            continue
        base = pl + si * STEP
        b0row = base + 5
        if b0row >= seq:
            continue
        opli = pre[b0row, dp['OP_LI']].item()
        if opli < 0.5:
            continue  # only real LI emit rows
        _pc, ax = steps[si]
        emit = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
        got = sum(b << (8 * j) for j, b in enumerate(emit)) & 0xFFFF
        qaxc = hexval(b0row, "AX_CARRY_LO", "AX_CARRY_HI")
        ok = (got == want)
        ok_any = ok_any or ok
        print(f"  ostep{si} want=0x{want:04x} got=0x{got:04x} "
              f"{'PASS' if ok else 'FAIL'} OP_LI={opli:.2f} "
              f"q_AXCARRY={None if qaxc is None else hex(qaxc)}")
        # head-16 winner
        for HEAD, tag in ((HEAD16, "h16"), (HEAD0, "h0 ")):
            qv = Q[b0row, HEAD]
            raw = (K[:, HEAD, :] @ qv) / (hd ** 0.5)
            scores = raw.clone()
            scores[b0row + 1:] = float("-inf")
            probs = torch.softmax(scores, dim=0)
            topk = torch.topk(probs, k=min(3, seq))
            parts = []
            for r in range(topk.indices.shape[0]):
                kp = int(topk.indices[r].item())
                pr = float(topk.values[r].item())
                ab = hexval(kp, "ADDR_B0_LO", "ADDR_B0_HI")
                ac = hexval(kp, "AX_CARRY_LO", "AX_CARRY_HI")
                self_tag = "*SELF" if kp == b0row else ""
                parts.append(f"@{kp}{self_tag} p={pr:.2f} raw={float(raw[kp]):.0f} "
                             f"AB={None if ab is None else hex(ab)} "
                             f"AC={None if ac is None else hex(ac)}")
            # sink raw score (softmax1 = extra virtual 0-logit key). Report the
            # winner's raw vs 0 so the veto can be sized to push it below 0.
            print(f"      {tag}: " + " | ".join(parts) + f" | SINK=0")
    return ok_any


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = model.embed._dim_positions
    op_li = dp["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = _dense(blk.attn.W_q)
        if abs(float(wq[0, op_li].item())) > 1000.0:
            L15 = bi
    nH = model.blocks[L15].attn.num_heads
    print(f"L15 block={L15} num_heads={nH} (head-16 {'PRESENT' if nH > 16 else 'ABSENT'})\n")

    results = {}
    for name, (src, bc_raw, want) in PROGRAMS.items():
        label = src if src is not None else "<raw bytecode>"
        print(f"=== {name}: {label} (want 0x{want:04x}) ===")
        results[name] = run_one(p, model, dp, L15, nH, src, bc_raw, want)
        print()

    print("=== SUMMARY ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
