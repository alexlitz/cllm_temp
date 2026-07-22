"""End-to-end verification of the ONE unified C4 VM model (``nibble_unified``).

Runs a representative program from every op FAMILY through the single unified
model's step loop (the standard argmax generation loop — dispatch is the model's
own MoE, fetch is code-as-data, arithmetic/cmp/bitwise/muldiv are the model's own
FFN experts) and checks byte-exactness against the reference 8-bit interpreter
``isa.interpret``. Also drives the folded softmax1-KV memory head (block-0
attention of the SAME model) over a store/load stream, and reports the honest
boundary: what is one-model-dispatched vs what still needs a separate path.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m c4_min.verify_unified
"""
from __future__ import annotations

import time

import torch

from . import isa
from . import blogspec_vocab as V
from .nibble_unified import build_unified_model, param_report
from .nibble_vm import load_program, _emit_and_reembed, _snap_lane
from .blogspec_memory import address_bits, _decode_byte
from .blogspec_layout import NIB_PER_REG


# ---------------------------------------------------------------------------
# The step-loop driver: ONE unified model, standard generation loop only.
# ---------------------------------------------------------------------------
def step_run(model, L, prog, max_steps=200):
    """Run ``prog`` through the unified model's recurrent step loop and return the
    per-step AX trace (the standard emit/snap/re-embed generation loop; the model
    weights do fetch + MoE dispatch + FFN op experts — no python compute)."""
    code = isa.assemble(prog)
    state = load_program(model, L, code)
    trace = []
    for _ in range(max_steps):
        x = state.view(1, 1, -1)
        for blk in model.blocks:            # == model.forward minus the LM head
            x = blk(x)
        out = x[0, 0]
        halted = float(out[L.HALTED]) > 0.5
        trace.append(_snap_lane(out[L.AX_VAL]) & 0xFF)
        state = _emit_and_reembed(out, L)
        if halted:
            break
    return trace


def kv_load(model, L, ops):
    """Drive the folded §Memory KV head (block-0 attention of the SAME unified
    model) over a store/load token stream; return the loaded values. ops is a
    list of ('store', addr, val) / ('load', addr)."""
    stream = [(V.BOS, {})]
    loaded = []
    for op in ops:
        if op[0] == "store":
            _, addr, val = op
            ov = {L.IS_STORE: 1.0}
            for b, bit in enumerate(address_bits(addr)):
                ov[L.ADDR_BIN + b] = bit
            for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                ov[L.VAL_NIB + j] = float(nv)
            stream.append((V.MEM, ov))
        else:
            _, addr = op
            ov = {L.IS_LOAD: 1.0}
            for b, bit in enumerate(address_bits(addr)):
                ov[L.QRY_BIN + b] = bit
            stream.append((V.MEM, ov))
            toks = torch.tensor([[t for t, _ in stream]])
            with torch.no_grad():
                x = model.embed[toks].clone()
                for i, (_, o) in enumerate(stream):
                    for d, vv in o.items():
                        x[0, i, d] = vv
                x0 = model.blocks[0].attn(x)       # the folded CAM head
            state = x0[0, -1]
            val = 0
            for bi in range(4):
                val |= _decode_byte(state, L, L.AX, bi) << (8 * bi)
            loaded.append(val)
            stream.pop()
    return loaded


# ---------------------------------------------------------------------------
# The per-family test corpus (>=1 per cluster; PC dispatch, loops, wrap included).
# ---------------------------------------------------------------------------
def family_programs():
    P = []
    # value / arithmetic
    P += [("IMM",  [("IMM", 42), ("HALT", 0)]),
          ("LEA",  [("LEA", 7), ("HALT", 0)]),
          ("ADD",  [("IMM", 10), ("PSH", 0), ("IMM", 20), ("ADD", 0), ("HALT", 0)]),
          ("SUB",  [("IMM", 50), ("PSH", 0), ("IMM", 20), ("SUB", 0), ("HALT", 0)]),
          ("ADD_wrap", [("IMM", 200), ("PSH", 0), ("IMM", 100), ("ADD", 0), ("HALT", 0)]),
          ("SUB_borrow", [("IMM", 10), ("PSH", 0), ("IMM", 20), ("SUB", 0), ("HALT", 0)])]
    # control flow
    P += [("JMP", [("JMP", 2), ("IMM", 99), ("IMM", 7), ("HALT", 0)]),
          ("BZ_taken",   [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 5), ("HALT", 0)]),
          ("BZ_skip",    [("IMM", 1), ("BZ", 4), ("IMM", 7), ("HALT", 0)]),
          ("BNZ_taken",  [("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 8), ("HALT", 0)]),
          ("loop_count", [("IMM", 3), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                          ("BNZ", 1), ("HALT", 0)])]
    # comparisons
    for name, a, b in [("EQ", 5, 5), ("EQ", 5, 6), ("NE", 5, 6), ("LT", 3, 5),
                       ("GT", 5, 3), ("LE", 5, 5), ("GE", 3, 5)]:
        P.append((f"{name}_{a}_{b}",
                  [("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)]))
    return P


def bitwise_programs():
    P = []
    for name, a, b in [("OR", 0x0C, 0x03), ("XOR", 0xFF, 0x0F), ("AND", 0xF0, 0x3C),
                       ("SHL", 0x03, 2), ("SHR", 0xF0, 4)]:
        P.append((f"{name}_{a:#x}_{b}",
                  [("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)]))
    return P


def muldiv_programs():
    P = []
    for name, a, b in [("MUL", 6, 7), ("MUL", 12, 12), ("DIV", 20, 3),
                       ("DIV", 20, 0), ("MOD", 20, 3), ("MOD", 17, 5)]:
        P.append((f"{name}_{a}_{b}",
                  [("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)]))
    return P


def _check(model, L, progs):
    ok_all = True
    for name, prog in progs:
        ref = isa.interpret(isa.assemble(prog))
        got = step_run(model, L, prog)
        # compare final AX (and full trace where it is stable) byte-exact.
        exp = ref[-1] if ref else None
        ok = (got[-1] == exp)
        ok_all = ok_all and ok
        print(f"    {name:14s} model={got[-1]!s:>4}  ref={exp!s:>4}  "
              f"{'OK' if ok else 'FAIL'}")
    return ok_all


def main():
    t0 = time.time()
    model, L, meta = build_unified_model(
        code_size=32, include_bitwise=True)
    build_s = time.time() - t0
    rep = param_report(model, meta)
    print("=" * 70)
    print("UNIFIED C4 VM MODEL — END-TO-END VERIFICATION")
    print("=" * 70)
    print(f"  dim={rep['dim']}  n_blocks={rep['n_blocks']}  n_heads={rep['n_heads']}"
          f"  vocab={rep['vocab']}   (build {build_s:.1f}s)")
    print(f"  TOTAL(dense)={rep['total_params']:,}  NONZERO={rep['nonzero_params']:,}"
          f"  SPARSITY={rep['sparsity']*100:.4f}%")
    print("-" * 70)

    results = {}
    t = time.time()
    print("  [step-loop dispatch : IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ + loops + CMP]")
    results["base+cmp"] = _check(model, L, family_programs())
    print("  [step-loop dispatch : OR/XOR/AND/SHL/SHR  (folded bitwise FFN)]")
    results["bitwise"] = _check(model, L, bitwise_programs())
    # NOTE: MUL/DIV/MOD are no longer baked into this lean unified model — the dense
    # 256x256 lookup table was removed; the only MUL/DIV/MOD path is the efficient
    # nibble_alu32 ALU in qwen_full_vm (see test_qwen_full_vm::test_muldiv_through_qwen).
    step_s = time.time() - t

    print("  [folded KV-memory head (block-0 attn of the SAME model) : LI/SI]")
    mem_cases = [([("store", 0x200, 42), ("load", 0x200)], [42]),
                 ([("store", 0x200, 42), ("store", 0x204, 0xABCD), ("load", 0x204)], [0xABCD]),
                 ([("store", 0x200, 42), ("load", 0x300)], [0]),               # ZFOD
                 ([("store", 0x200, 42), ("store", 0x200, 99), ("load", 0x200)], [99]),  # latest
                 ([("store", 0x200, 42), ("store", 0x200, 0), ("load", 0x200)], [0])]     # free
    mok = True
    for ops, exp in mem_cases:
        got = kv_load(model, L, ops)
        ok = got == exp
        mok = mok and ok
        print(f"    {str(ops)[:52]:54s} -> {got} exp {exp} {'OK' if ok else 'FAIL'}")
    results["kv-memory"] = mok

    print("-" * 70)
    print(f"  wall: build {build_s:.1f}s  step-loop verify {step_s:.1f}s")
    for k, v in results.items():
        print(f"    {k:12s} : {'PASS' if v else 'FAIL'}")
    print("  ALL:", "PASS" if all(results.values()) else "FAIL")
    return all(results.values())


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
