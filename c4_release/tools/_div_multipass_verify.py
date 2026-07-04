"""Verify multi_pass_div_rules compute-correctness via the NEURAL PureFFN
forward (the same path the live install runs) — NOT run_symbolic (which is a
linear step executor that does NOT apply the silu amplitude fixed point the
normalized cascade convention depends on).

Lowers each pass to a PureFFN, threads the residual through the stack (torch
batch), decodes the quotient/remainder nibble lanes by argmax, compares to
Python divmod over a sweep of (a,b).
"""
import sys

import torch

from neural_vm.base_layers import PureFFN
from neural_vm.unified_compiler.wide_alu_dsl import multi_pass_div_rules

S = 100.0

# ad-hoc dim layout (contiguous bands)
POS = {}
DIM = 0
# workspace must be >= 108 lanes * 16 = 1728 wide (measured); give headroom.
for _nm, _w in (
    ("MARK", 1), ("OP_DIV", 1),
    ("A", 32), ("B", 32),
    ("QLO", 16), ("QHI", 16), ("RLO", 16), ("RHI", 16),
    ("WS", 108 * 16),
):
    POS[_nm] = DIM
    DIM += _w


def build_mp():
    return multi_pass_div_rules(
        dividend_a_base="A", divisor_b_base="B",
        quotient_lane_bases=("QLO", "QHI"),
        remainder_lane_bases=("RLO", "RHI"),
        workspace_base="WS", opcode_gate="OP_DIV", marker_gate="MARK",
        S=S, width_bytes=1,
    )


def lower(mp):
    flat = mp.as_flat_ir()
    ffns = []
    for i, p in enumerate(mp.passes):
        f = PureFFN(dim=DIM, hidden_dim=max(1, p.hidden_units))
        flat.lower_ffn(f, POS, layer_idx=i, S=S)
        ffns.append(f)
    return ffns


def batch_inputs(pairs):
    N = len(pairs)
    X = torch.zeros(N, 1, DIM)
    for idx, (a, b) in enumerate(pairs):
        X[idx, 0, POS["MARK"]] = 1.0
        X[idx, 0, POS["OP_DIV"]] = 1.0
        X[idx, 0, POS["A"] + (a & 0xF)] = 1.0
        X[idx, 0, POS["A"] + 16 + ((a >> 4) & 0xF)] = 1.0
        X[idx, 0, POS["B"] + (b & 0xF)] = 1.0
        X[idx, 0, POS["B"] + 16 + ((b >> 4) & 0xF)] = 1.0
    return X


def main():
    full = "--full" in sys.argv
    mp = build_mp()
    print(f"passes={mp.num_passes} units={mp.hidden_units} "
          f"ws_lanes={getattr(mp, '_div_ws_lanes', '?')} dim={DIM}")
    ffns = lower(mp)

    if full:
        pairs = [(a, b) for a in range(256) for b in range(256)]
    else:
        import random
        random.seed(1)
        pairs = [(0, 0), (0, 1), (5, 0), (255, 0), (255, 1), (255, 255),
                 (42, 6), (100, 7), (84, 2), (255, 15), (17, 5), (200, 13),
                 (1, 1), (128, 128), (129, 128), (127, 3), (240, 16)]
        pairs += [(random.randint(0, 255), random.randint(0, 255))
                  for _ in range(500)]

    with torch.no_grad():
        Y = batch_inputs(pairs)
        for f in ffns:
            Y = f(Y)

    def dec(base):
        return Y[:, 0, POS[base]:POS[base] + 16].argmax(dim=-1)

    qlo, qhi = dec("QLO"), dec("QHI")
    rlo, rhi = dec("RLO"), dec("RHI")
    q = (qhi.long() << 4) | qlo.long()
    r = (rhi.long() << 4) | rlo.long()

    exp_q = torch.tensor([0 if b == 0 else a // b for a, b in pairs])
    exp_r = torch.tensor([a if b == 0 else a % b for a, b in pairs])

    q_bad = (q != exp_q)
    r_bad = (r != exp_r)
    bad = q_bad | r_bad
    n = int(bad.sum())
    print(f"pairs={len(pairs)} fails={n}  (q_fails={int(q_bad.sum())} "
          f"r_fails={int(r_bad.sum())})")
    if n:
        idxs = bad.nonzero().flatten()[:25]
        for i in idxs:
            a, b = pairs[int(i)]
            print(f"  FAIL a={a} b={b}: got q={int(q[i])} r={int(r[i])} "
                  f"want q={int(exp_q[i])} r={int(exp_r[i])}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
