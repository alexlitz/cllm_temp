"""#916 TRUE-FLOOR BAKE + RUN (radix-16 LEAN divmod, the DEFAULT production path).

The c6ddae1b end-to-end bake assembled the LOG-SINK divmod (127 inline blocks ->
whole ISA 153 layers, fp64, byte-exact only for dividend <= ~2^16).  This module
assembles the SAME whole full-ISA but with the DEFAULT ``div_radix16_lean``
digit-recurrence divmod (``div_logsink=False``) — 81 inline divmod blocks -> whole
ISA 107 layers, fp32 — and runs the SAME battery PLUS the FULL-32-BIT divmod cases
the log-sink build DIVERGED on (max/3, 0xDEADBEEF/0x1234, max/65535, q>2^16).

Reuses the sparse-resident lean forward from ``_e2e_inline_bake_916`` (RSS-safe,
never densifies; watchdog aborts > 4 GB).  Golden 174ece66 UNTOUCHED (off the build
path).

Run:  PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._e2e_floor_r16lean_916
"""
from __future__ import annotations

import os
from typing import List, Tuple

import torch

from . import _e2e_inline_bake_916 as E


def build_sparse_r16lean(dtype=torch.float32) -> Tuple["E.SparseLeanVM", dict]:
    """Assemble the whole full-ISA with the radix-16 LEAN divmod (div_logsink=False,
    the default production DIV/MOD path).  Sparse-resident, never densified."""
    os.environ["C4_LOGSINK_DIV"] = "0"
    from . import qwen_full_vm as Q

    subset = Q.SUBSET_FULL
    arch = Q.QWEN2_5_ARCH
    K = Q.NORM_K
    code_size = 24
    efficient_alu = True
    recurrent_divmod = False
    code_from_memory = True
    shift_via_mul = True

    QL = Q.QwenFullLayout(code_size, subset, efficient_alu=efficient_alu,
                          recurrent_divmod=recurrent_divmod,
                          code_from_memory=code_from_memory,
                          shift_via_mul=shift_via_mul, div_logsink=False)
    assert not QL.div_logsink, "expected radix-16 lean (div_logsink resolved False)"
    L = QL.L
    block_specs = Q._block_specs(L, code_size, subset, efficient_alu=efficient_alu,
                                 recurrent_divmod=recurrent_divmod,
                                 code_from_memory=code_from_memory,
                                 shift_via_mul=QL.shift_via_mul,
                                 div_logsink=QL.div_logsink)
    block_names = [nm for nm, _ in block_specs]

    dim_needed = QL.D_used + 1
    hidden = arch.hidden_for(dim_needed)
    inter = max(int(s["W_up"].shape[0]) for _, s in block_specs)
    inter = max(inter, arch.num_attention_heads * arch.head_dim, 8)
    n_layers = len(block_specs)
    comp = QL.D_used

    gamma = Q.rmsnorm_identity_gamma(hidden, K).to(dtype)
    embed = Q._build_embedding(L, hidden, comp, K).to(dtype)

    attn_bakers = {0: "register"}
    if code_from_memory:
        attn_bakers[block_names.index("code-cam")] = "code"
    if subset.memory:
        attn_bakers[block_names.index("mem-cam")] = "memory"
    # NO recip-sink attn (that is a log-sink-only head).

    layers: List["E._SparseLayer"] = []
    for i, (name, spec) in enumerate(block_specs):
        h_units = spec["W_up"].shape[0]
        Dg = spec["W_up"].shape[1]
        bd = spec.get("b_down")
        if bd is not None and float(bd.abs().max()) != 0.0:
            raise AssertionError(f"nonzero b_down in {name}")
        gate_w = torch.zeros(inter, hidden, dtype=dtype)
        up_w = torch.zeros(inter, hidden, dtype=dtype)
        down_w = torch.zeros(hidden, inter, dtype=dtype)
        gate_w[:h_units, :Dg] = spec["W_up"].to(dtype)
        up_w[:h_units, :Dg] = spec["W_gate"].to(dtype)
        down_w[:Dg, :h_units] = spec["W_down"].to(dtype)
        gate_w[:h_units, L.ONE] += spec["b_up"].to(dtype)
        up_w[:h_units, L.ONE] += spec["b_gate"].to(dtype)
        lay = E._SparseLayer(ln1=gamma.clone(), ln2=gamma.clone(),
                             gate_csr=E._to_csr(gate_w), up_csr=E._to_csr(up_w),
                             down_csr=E._to_csr(down_w))
        del gate_w, up_w, down_w

        if i in attn_bakers:
            kind = attn_bakers[i]
            stub = E._StubAttn(hidden, arch, dtype)
            if kind == "register":
                Q._bake_register_cam(stub, QL, arch, comp, K)
            elif kind == "code":
                Q._bake_code_cam(stub, QL, arch, comp, K)
            elif kind == "memory":
                Q._bake_memory_cam(stub, QL, arch, comp, K, mem_addr_bits=None)
            lay.has_attn = True
            lay.q_w = stub.q_proj.weight; lay.k_w = stub.k_proj.weight
            lay.v_w = stub.v_proj.weight; lay.o_w = stub.o_proj.weight
            lay.q_b = stub.q_proj.bias; lay.k_b = stub.k_proj.bias
            lay.v_b = stub.v_proj.bias
        layers.append(lay)

    vm = E.SparseLeanVM(layers, gamma.clone(), embed, hidden, arch, dtype, QL, subset,
                        code_from_memory, efficient_alu, QL.shift_via_mul,
                        Q.ROPE_THETA, 1e-6, div_logsink=False)
    n_divmod = sum(1 for n in block_names if n.startswith("lean-"))
    info = {"n_layers": n_layers, "hidden": hidden, "intermediate": inter,
            "D_used": QL.D_used, "block_names": block_names,
            "n_divmod_blocks": n_divmod,
            "dtype": str(dtype), "attn_layers": sorted(attn_bakers.keys()),
            "peak_rss_mb": E._rss_mb()}
    return vm, info


# ---------------------------------------------------------------------------
# Full-32-bit divmod battery through the ASSEMBLED forward (the cases the log-sink
# build DIVERGED on).  One-instruction DIV/MOD, operands seeded 32-bit direct.
# ---------------------------------------------------------------------------
def _run_one_op_32bit(vm, op_name, a, b):
    from . import qwen_full_vm as Q
    from . import isa
    from .nibble_pure_forward_complete import _decode_reg_from_nibbles
    QL, L = vm.QL, vm.QL.L
    code = isa.assemble([(op_name, 0)])
    reg_state = {"PC": 0, "AX": b & 0xFFFFFFFF, "SP": Q.SP_INIT,
                 "BP": Q.SP_INIT, "STACK0": a & 0xFFFFFFFF}
    x = Q._build_stream_and_overlay(vm, code, reg_state, [], None)
    state = vm.forward(x, q_positions=None)[0, -1]
    return _decode_reg_from_nibbles(state, L, L.AX) & 0xFFFFFFFF


def full32_cases():
    """The exact FULL-32-BIT divmod cases the task names + the log-sink build's
    documented divergences.  All must be byte-exact for a true full-32-bit inline
    divmod."""
    M = 0xFFFFFFFF
    named = [
        ("DIV", M, 3, "max/3 (q~1.4e9)"),
        ("MOD", M, 3, "max %% 3"),
        ("DIV", 0xDEADBEEF, 0x1234, "0xDEADBEEF/0x1234"),
        ("MOD", 0xDEADBEEF, 0x1234, "0xDEADBEEF %% 0x1234"),
        ("DIV", M, 0xFFFF, "max/65535 (q=65537)"),
        ("MOD", M, 0xFFFF, "max %% 65535"),
        ("DIV", 700003, 7, "700003/7 (q>2^16)"),
        ("DIV", M, 1, "max/1 (b=1)"),
        ("DIV", M, 2, "max/2 (b=2^1)"),
        ("DIV", M, 256, "max/256 (b~256)"),
        ("DIV", M, 255, "max/255"),
        ("DIV", 0xCAFEBABE, 0x100, "0xCAFEBABE/256"),
        ("DIV", M, M, "max/max (q=1)"),
        ("DIV", 2**31, 3, "2^31/3"),
        ("MOD", 2**31, 7, "2^31 %% 7"),
        ("DIV", M, 5, "max/5 (q>2^16)"),
        ("DIV", 4000000000, 3, "4e9/3"),
        ("DIV", M, 0x10000, "max/65536 (b=2^16)"),
        ("MOD", M, 0x10000, "max %% 65536"),
    ]
    return named


def run_full32(vm):
    from . import nibble_muldivmod as NM
    print("\n" + "-" * 92)
    print("FULL-32-BIT inline divmod (the cases the log-sink 153-layer build DIVERGED on)")
    print("operands seeded 32-bit direct, one-instruction DIV/MOD, vs nibble_muldivmod32")
    print("-" * 92)
    ok = tot = 0
    for op_name, a, b, desc in full32_cases():
        got = _run_one_op_32bit(vm, op_name, a, b)
        want = NM.div32(a, b) if op_name == "DIV" else NM.mod32(a, b)
        tot += 1
        good = (got == want)
        ok += int(good)
        tag = "PASS" if good else "FAIL"
        extra = "" if good else f"  got={got} want={want}"
        print(f"  {desc:<34} [{tag}]{extra}")
    print(f"  named full-32-bit divmod byte-exact: {ok}/{tot}")

    import random
    rng = random.Random(20260815)
    r_ok = r_tot = 0
    fails = []
    for _ in range(400):
        a = rng.randint(0, 2**32 - 1)
        b = rng.randint(1, 2**32 - 1)
        dg = _run_one_op_32bit(vm, "DIV", a, b)
        mg = _run_one_op_32bit(vm, "MOD", a, b)
        r_tot += 2
        dok = dg == NM.div32(a, b)
        mok = mg == NM.mod32(a, b)
        r_ok += int(dok) + int(mok)
        if (not dok or not mok) and len(fails) < 8:
            fails.append((a, b, dg, mg, NM.div32(a, b), NM.mod32(a, b)))
    print(f"  RANDOM full-32-bit (400 pairs, a,b in [0,2^32)): {r_ok}/{r_tot} byte-exact")
    if fails:
        print("  first random fails (a,b,dgot,mgot,dwant,mwant):")
        for f in fails:
            print("   ", f)
    return ok, tot, r_ok, r_tot


def main():
    from . import isa
    from . import _e2e_battery_916 as B
    E._rss_watchdog(4000)
    print("=" * 92)
    print("#916 TRUE FLOOR: assembled sparse whole full-ISA inline+FF with the DEFAULT")
    print("radix-16 LEAN divmod (fp32), run programs + FULL-32-BIT divmod byte-exact")
    print("=" * 92)
    vm, info = build_sparse_r16lean()
    print(f"\nASSEMBLED MODEL: {info['n_layers']} physical layers "
          f"(of which {info['n_divmod_blocks']} inline divmod), "
          f"hidden {info['hidden']}, inter {info['intermediate']}, "
          f"dtype {info['dtype']}")
    print(f"  attention CAM layers: {info['attn_layers']}  (register / code-fetch / mem)")
    print(f"  D_used {info['D_used']}  build peak RSS {info['peak_rss_mb']} MB")
    print(f"  NO subroutine, NO loop (divmod UNROLLED, identity apply order)")

    all_prog_ok = all_prog_tot = 0
    all_step_ok = all_step_tot = 0
    divmod_ok = divmod_tot = 0
    for title, progs, is_divmod in (
            ("DIVMOD boundary (b=1, b=2^k, b~256, large q, b=0 guard)",
             B.divmod_programs(), True),
            ("Multi-op branching / memory / cmp programs",
             B.multiop_programs(), False)):
        print("\n" + "-" * 92)
        print(title)
        print("-" * 92)
        for name, prog in progs.items():
            code = isa.assemble(prog)
            r = E.run_program(vm, code, max_steps=120)
            ok = r["exact"]
            all_prog_tot += 1
            all_prog_ok += int(ok)
            all_step_tot += r["n_cmp"]
            all_step_ok += (r["n_cmp"] if ok else
                            sum(1 for i in range(r["n_cmp"])
                                if r["ax_trace"][i] == r["ref_trace"][i]))
            if is_divmod:
                divmod_tot += 1
                divmod_ok += int(ok)
            tag = "PASS" if ok else "FAIL"
            extra = "" if ok else f"  ax={r['ax_trace']} ref={r['ref_trace']}"
            print(f"  {name:<22} steps={r['n_cmp']:>3}  [{tag}]{extra}")

    f_ok, f_tot, r_ok, r_tot = run_full32(vm)

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  8-bit programs byte-exact : {all_prog_ok}/{all_prog_tot}")
    print(f"  8-bit steps byte-exact    : {all_step_ok}/{all_step_tot}")
    print(f"  8-bit divmod cases        : {divmod_ok}/{divmod_tot}")
    print(f"  FULL-32-BIT named divmod  : {f_ok}/{f_tot}")
    print(f"  FULL-32-BIT random divmod : {r_ok}/{r_tot}")
    print(f"  peak RSS                  : {E._rss_mb()} MB")
    print(f"  assembled layers          : {info['n_layers']} (hidden {info['hidden']}, "
          f"inter {info['intermediate']}, {info['n_divmod_blocks']} inline divmod)")
    return (all_prog_ok == all_prog_tot and f_ok == f_tot and r_ok == r_tot)


if __name__ == "__main__":
    ok = main()
    print(f"\nALL BYTE-EXACT (incl full-32-bit): {ok}")
