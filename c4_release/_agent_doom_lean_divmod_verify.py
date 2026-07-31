#!/usr/bin/env python3
"""VERIFY the fixed CFM recurrent-divmod LEAN forward is BYTE-EXACT for MUL/DIV/MOD.

Three checks, all on the SAME fitting model (SUBSET_MULDIV, recurrent_divmod=True,
code_from_memory=True, hidden=2944, ~11.9 GB fits 24 GB):

  1. 8-bit trace battery -- every op's run_program_lean (naive) + speculative_run_lean
     (big-K) AX trace == isa.interpret == HF run_program.  Covers MUL/DIV/MOD incl
     div-by-1, div-by-0, mod-by-0, 255/1, 255/255, 0/5, the digit-format C idiom, and
     the "signed" (unsigned-folded) edge cases the edge_corpus uses.

  2. 32-bit result equality -- for the int-promotion cases (64*4=256, 255*255=65025,
     0-1==0xFFFFFFFF/...) the efficient ALU computes the FULL 32-bit product; the lean
     nibble decode with mask=0xFFFFFFFF must equal HF run_program(mask=0xFFFFFFFF).

  3. ms/step -- naive per-step vs big-K speculative on a deterministic loop, plus HF
     run_program per-step, so we report the fast-path speed the fix preserves.
"""
from __future__ import annotations

import argparse
import time
import warnings

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block-k", type=int, default=32)
    ap.add_argument("--spin-steps", type=int, default=800)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    from _agent_lean_recurrent import build_lean_recurrent

    dev = torch.device(args.device)
    subset = Q.SUBSET_MULDIV

    print("[build] CFM recurrent-divmod muldiv (cfm=True, hidden target 2944) ...", flush=True)
    t0 = time.perf_counter()
    vm = Q.build(code_size=24, subset=subset, recurrent_divmod=True, code_from_memory=True)
    L = vm.QL.L
    vm.embed = vm.embed.to(dev)
    vm.qmodel = vm.qmodel.to(dev)
    print(f"[build] stored={vm.n_layers}L applied={vm.n_applied} hidden={vm.hidden_size} "
          f"efficient_alu={vm.efficient_alu} shift_via_mul={vm.shift_via_mul} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)
    if dev.type == "cuda":
        print(f"[vram] HF model: {torch.cuda.memory_allocated(dev)/1e9:.2f} GB", flush=True)

    lean = build_lean_recurrent(vm, device=str(dev))
    print(f"[lean] applied={lean.n_layers} efficient_alu={lean.efficient_alu} "
          f"shift_via_mul={lean.shift_via_mul}", flush=True)

    # ---- (1) 8-bit trace battery -----------------------------------------
    def bin_prog(a, b, opname):
        return [("IMM", a), ("PSH", 0), ("IMM", b), (opname, 0), ("HALT", 0)]

    battery = [
        ("mul_12x7",     bin_prog(12, 7, "MUL")),
        ("mul_0x40_4",   bin_prog(0x40, 4, "MUL")),     # 64*4=256 -> 8bit 0
        ("mul_ff_ff",    bin_prog(0xFF, 0xFF, "MUL")),  # 65025 -> 8bit 1
        ("mul_by_0",     bin_prog(13, 0, "MUL")),
        ("mul_1x1",      bin_prog(1, 1, "MUL")),
        ("div_100_7",    bin_prog(100, 7, "DIV")),
        ("div_17_5",     bin_prog(17, 5, "DIV")),
        ("div_by_zero",  bin_prog(7, 0, "DIV")),        # -> 0
        ("div_255_1",    bin_prog(255, 1, "DIV")),      # -> 255
        ("div_255_255",  bin_prog(255, 255, "DIV")),    # -> 1
        ("div_0_5",      bin_prog(0, 5, "DIV")),        # -> 0
        ("div_1_2",      bin_prog(1, 2, "DIV")),        # -> 0
        ("mod_100_7",    bin_prog(100, 7, "MOD")),
        ("mod_17_5",     bin_prog(17, 5, "MOD")),
        ("mod_by_zero",  bin_prog(7, 0, "MOD")),        # -> 0
        ("mod_255_16",   bin_prog(255, 16, "MOD")),     # -> 15
        ("mod_7_10",     bin_prog(7, 10, "MOD")),       # -> 7
        # signed (unsigned-folded) edge: (0-1)/1 == 0xFF unsigned at 8-bit.
        ("neg_div_1", [("IMM", 0), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                       ("PSH", 0), ("IMM", 1), ("DIV", 0), ("HALT", 0)]),
        # digit format idiom: 48 + (7 % 10) == 55.
        ("digit_format", [("IMM", 7), ("PSH", 0), ("IMM", 10), ("MOD", 0),
                          ("PSH", 0), ("IMM", 48), ("ADD", 0), ("HALT", 0)]),
        # sequential div->mul->mod chain (depth-1 stack, no deep parking): each op reads
        # the running AX as STACK0. ((100/7=14) *5=70) %13 == 5.
        ("div_mul_mod", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0),   # AX=14
                         ("PSH", 0), ("IMM", 5), ("MUL", 0),                 # AX=70
                         ("PSH", 0), ("IMM", 13), ("MOD", 0), ("HALT", 0)]),  # AX=5
    ]

    print("\n=== (1) 8-bit trace: HF run_program vs lean naive vs lean big-K vs isa.interpret ===",
          flush=True)
    all_ok = True
    for name, prog in battery:
        code = isa.assemble(prog)
        ref = isa.interpret(code, max_steps=200)
        hf = Q.run_program(vm, code, max_steps=200)
        rn = LF.run_program_lean(lean, code, max_steps=200)
        sp = LF.speculative_run_lean(lean, code, block_steps=args.block_k, max_steps=200)
        ok = (hf["ax_trace"] == ref and rn["ax_trace"] == ref and sp.ax_trace == ref)
        all_ok = all_ok and ok
        flag = "OK " if ok else "FAIL"
        print(f"  {flag} {name:14s} final_ax={ref[-1]:4d}  hf={hf['exact']} "
              f"naive={rn['exact']} bigK={sp.exact}", flush=True)
    print(f"[1] {'ALL BYTE-EXACT (8-bit) vs isa.interpret + HF' if all_ok else 'DIVERGENCE'}",
          flush=True)

    # ---- (2) 32-bit result equality (int promotion) ----------------------
    print("\n=== (2) 32-bit result: lean nibble decode(mask=0xFFFFFFFF) vs HF(mask=0xFFFFFFFF) ===",
          flush=True)
    from c4_min.nibble_pure_forward_complete import _decode_reg_from_nibbles
    from c4_min.qwen_full_vm import _snap
    wide = [
        ("mul_0x40_4",  bin_prog(0x40, 4, "MUL"),   256),
        ("mul_ff_ff",   bin_prog(0xFF, 0xFF, "MUL"), 65025),
        ("mul_100_100", bin_prog(100, 100, "MUL"),  10000),
        ("div_255_1",   bin_prog(255, 1, "DIV"),    255),
    ]
    wide_ok = True
    for name, prog, want32 in wide:
        code = isa.assemble(prog)
        hf = Q.run_program(vm, code, max_steps=64, mask=0xFFFFFFFF)
        # lean: decode the ALU op AX at 32 bits from the same nibble path.
        op = code[3].op
        from c4_min.nibble_pure_forward import SP_INIT
        reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
        store_log, cur_pc, got32 = [], 0, None
        for _ in range(64):
            cur_op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
            x, positions = LF._build_stream_and_overlay(lean, code, reg_state, store_log, None)
            with torch.no_grad():
                hidden, _ = lean.forward(x, past=None, q_positions=positions)
            state = hidden[0, -1]
            if cur_op == op:
                got32 = _decode_reg_from_nibbles(state, L, L.AX) & 0xFFFFFFFF
                break
            reg_state = {"PC": _snap(state[L.PC_VAL]),
                         "AX": _snap(state[L.AX_VAL]) & 0xFF,
                         "SP": _snap(state[L.SP_VAL]), "BP": _snap(state[L.BP_VAL]),
                         "STACK0": _snap(state[L.STK_VAL])}
            cur_pc = reg_state["PC"]
        hf32 = hf["ax_trace"][3]
        ok = (got32 == want32 == hf32)
        wide_ok = wide_ok and ok
        print(f"  {'OK ' if ok else 'FAIL'} {name:12s} want32={want32:6d}  lean32={got32}  hf32={hf32}",
              flush=True)
    print(f"[2] {'32-bit int-promotion EXACT (lean==HF==want)' if wide_ok else 'DIVERGENCE'}",
          flush=True)

    # ---- (3) ms/step -----------------------------------------------------
    n = args.spin_steps
    # a DIV-heavy deterministic loop (exercise the recurrent divmod every iter).
    spin = isa.assemble([("IMM", 200), ("PSH", 0), ("IMM", 3), ("DIV", 0),   # 200/3=66
                         ("PSH", 0), ("IMM", 1), ("SUB", 0),                 # 66-1
                         ("BNZ", 1), ("HALT", 0)])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    rn = LF.run_program_lean(lean, spin, max_steps=n)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    naive_ms = (time.perf_counter() - t0) / max(rn["steps"], 1) * 1e3

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    rs = LF.speculative_run_lean(lean, spin, block_steps=args.block_k, max_steps=n)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    spec_wall = time.perf_counter() - t0
    spec_ms = spec_wall / max(rs.steps, 1) * 1e3

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    rh = Q.run_program(vm, spin, max_steps=n)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    hf_ms = (time.perf_counter() - t0) / max(len(rh["ax_trace"]), 1) * 1e3

    print(f"\n=== (3) ms/step on a DIV-heavy deterministic loop ({rn['steps']} steps) ===",
          flush=True)
    print(f"  HF run_program  (per-step)      : {hf_ms:8.3f} ms/step  exact={rh['exact']}", flush=True)
    print(f"  lean naive      (per-step)      : {naive_ms:8.3f} ms/step  exact={rn['exact']}", flush=True)
    print(f"  lean big-K spec (K={args.block_k})          : {spec_ms:8.3f} ms/step  "
          f"({rs.forwards} forwards, {rs.speedup:.1f}x fewer, exact={rs.exact})", flush=True)

    print(f"\n[SUMMARY] 8bit_battery={'PASS' if all_ok else 'FAIL'}  "
          f"32bit={'PASS' if wide_ok else 'FAIL'}", flush=True)
    return 0 if (all_ok and wide_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
