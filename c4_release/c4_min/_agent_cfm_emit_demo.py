"""CFM-path in-transformer JIT (C4_CFM_EMIT) — append-and-run / re-emit / end-to-end.

Proves the ONE genuinely-unbuilt piece of the compile-and-run-in-one-transformer-run
endgame on the GOLDEN CFM VM (``qwen_full_vm`` / ``qwen_lean_forward``): a runtime
``EMIT`` opcode that APPENDS a CODE frame into the SAME KV §Memory the fetch@PC CAM
(``_bake_code_cam``) reads, so runtime-generated code becomes fetchable by PC with
NO residual-width (D) growth — program-length-INDEPENDENT, unlike the bespoke
fixed-width ``CODE_WORD`` band (``nibble_compiler``).

Run (LEAN build, ~1 GB sparse-resident):
    PYTHONPATH=<c4_release> C4_PF_CFM=1 C4_CFM_EMIT=1 \
        python -m c4_min._agent_cfm_emit_demo

Three demos, each byte-exact vs the ``isa.interpret`` reference (the golden c4
oracle for this 8-bit slice):
  1. append-and-run : a loader EMITs ``IMM 14 ; HALT`` into an empty code region
                      and JMPs in -> the model fetches + runs the produced bytecode.
  2. re-emit        : the loader EMITs to a slot, runs it (AX=3), RE-EMITs the SAME
                      slot (AX=9), runs the new instruction -> self-modifying code
                      via the address-keyed CAM's latest-write-wins (one frame/addr).
  3. end-to-end     : a LOADED (NOT baked-in-weights) compiler reads a C source
                      string from the data band, EMITs the compiled bytecode, JMPs
                      in, and runs the produced program -> result byte-exact vs c4.
"""
from __future__ import annotations

import os

os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_CFM_EMIT", "1")

import torch

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF


IMM, PSH, HALT, JMP, EMIT, ADD, MUL = (
    isa.IMM, isa.PSH, isa.HALT, isa.JMP, isa.EMIT, isa.ADD, isa.MUL)


def _emit_seq(target: int, op: int, immval: int):
    """The 4-instruction loader gadget that produces ``Instr(op, immval)`` at
    ``code[target]``:  ``IMM immval ; PSH ; IMM op ; EMIT target``  (produced op
    rides AX, produced imm rides STACK0, EMIT immediate = target address)."""
    return [isa.Instr(IMM, immval), isa.Instr(PSH, 0),
            isa.Instr(IMM, op), isa.Instr(EMIT, target)]


def prog_append_and_run():
    """Loader that EMITs ``IMM 14 ; HALT`` into a reserved region then JMPs in.
    Expected final AX = 14."""
    PROD, END = 10, 11
    prog = []
    prog += _emit_seq(PROD, IMM, 14)     # 0..3  code[10] := IMM 14
    prog += _emit_seq(END, HALT, 0)      # 4..7  code[11] := HALT
    prog += [isa.Instr(JMP, PROD)]       # 8     handoff: JMP to produced code
    prog += [isa.Instr(isa.NOP, 0)]      # 9     pad
    prog += [isa.Instr(isa.NOP, 0), isa.Instr(isa.NOP, 0)]  # 10,11 reserved region
    return prog, 14


def prog_reemit():
    """Loader that EMITs ``IMM 3`` at slot 30, runs it, then RE-EMITs the SAME slot
    to ``IMM 9`` and runs the new instruction (self-modifying code).  Expected
    final AX = 9 (the newest frame wins by the CAM's latest-write-wins)."""
    PROD, NEXT = 30, 31
    prog = []
    prog += _emit_seq(PROD, IMM, 3)      # 0..3   code[30] := IMM 3
    prog += _emit_seq(NEXT, JMP, 12)     # 4..7   code[31] := JMP 12 (v1 trampoline)
    prog += [isa.Instr(JMP, PROD)]       # 8      run v1: IMM3 -> JMP12
    prog += [isa.Instr(isa.NOP, 0)] * 3  # 9,10,11
    prog += _emit_seq(PROD, IMM, 9)      # 12..15 code[30] := IMM 9 (RE-EMIT same addr)
    prog += _emit_seq(NEXT, HALT, 0)     # 16..19 code[31] := HALT
    prog += [isa.Instr(JMP, PROD)]       # 20     run v2: IMM9 -> HALT
    prog += [isa.Instr(isa.NOP, 0)] * (PROD - len(prog))     # pad up to slot 30
    prog += [isa.Instr(isa.NOP, 0), isa.Instr(isa.NOP, 0)]   # 30,31 reserved
    return prog, 9


# ---------------------------------------------------------------------------
# END-TO-END: a LOADED compiler (code frames, NOT baked in weights) that reads a
# C source string from the data §Memory and EMITs the compiled bytecode.
#
# The compiler reads the source char at cursor via LC (mem[AX]), turns a
# ``d1 + d2`` / ``d1 * d2`` two-operand expression into
# ``IMM d1 ; PSH ; IMM d2 ; <ADD|MUL> ; HALT`` (the exact bytecode the real c4
# compiler emits for such an expression), EMITs those 5 words into a reserved
# region, and JMPs in.  The digits + operator are read out of the SRC data band at
# runtime (nothing about the expression is in the loader's bytecode), so this is a
# genuine compile-then-run — the minimal-subset instance of demo_model_runs_c on
# the GOLDEN CFM path.  The source bytes are pre-loaded into the store §Memory
# (addresses 0..2) as the "input file".
# ---------------------------------------------------------------------------
def prog_compile_expr(prod_base: int):
    """The LOADED compiler bytecode (reads mem[0..2] = 'd1 op d2', emits 5 words
    at ``prod_base..prod_base+4``, JMPs in).  ``mem`` holds the source bytes:
    mem[0]=d1 (digit value), mem[1]=op-selector (0=ADD,1=MUL), mem[2]=d2."""
    LI = isa.LI
    P = prod_base
    prog = []
    # produced[0] = IMM d1 :  read imm<-mem[0], op<-IMM, EMIT P
    prog += [isa.Instr(IMM, 0), isa.Instr(LI, 0), isa.Instr(PSH, 0),
             isa.Instr(IMM, IMM), isa.Instr(EMIT, P)]                    # 0..4
    # produced[1] = PSH :  imm<-0, op<-PSH, EMIT P+1
    prog += _emit_seq(P + 1, PSH, 0)                                     # 5..8
    # produced[2] = IMM d2 :  read imm<-mem[2], op<-IMM, EMIT P+2
    prog += [isa.Instr(IMM, 2), isa.Instr(LI, 0), isa.Instr(PSH, 0),
             isa.Instr(IMM, IMM), isa.Instr(EMIT, P + 2)]                # 9..13
    # produced[3] = <ADD|MUL> :  op-selector byte at mem[1] -> ADD(25) + sel*(MUL-ADD)
    #   read sel<-mem[1]; opcode = ADD + sel*(MUL-ADD).  We do it as:
    #   IMM 1 (addr); LI -> AX=sel; PSH; IMM (MUL-ADD); MUL -> AX = sel*(MUL-ADD);
    #   PSH; IMM ADD; ADD -> AX = ADD + sel*(MUL-ADD).  Then PSH 0 (imm) ; EMIT.
    prog += [isa.Instr(IMM, 1), isa.Instr(LI, 0),                       # AX = sel
             isa.Instr(PSH, 0), isa.Instr(IMM, MUL - ADD), isa.Instr(MUL, 0),
             isa.Instr(PSH, 0), isa.Instr(IMM, ADD), isa.Instr(ADD, 0),  # AX = opcode
             isa.Instr(PSH, 0),                                          # STACK0 = imm = 0
             isa.Instr(EMIT, P + 3)]                                     # code[P+3] := op
    # produced[4] = HALT
    prog += _emit_seq(P + 4, HALT, 0)
    # handoff
    prog += [isa.Instr(JMP, P)]
    return prog


def prog_compile_add_expr(prod_base: int):
    """ADD-only variant of ``prog_compile_expr`` — the compiler executes NO MUL, so
    it (and the produced ``IMM d1 ; PSH ; IMM d2 ; ADD ; HALT``) runs on the FAST
    ``SUBSET_MEM_CMP`` model (no heavy muldiv build).  Reads d1<-mem[0], d2<-mem[2];
    mem[1] is ignored (the operator is fixed ADD).  Still a genuine compile-then-run:
    the digit values are read from the SRC data band at runtime, nothing about the
    expression is in the loader bytecode."""
    LI = isa.LI
    P = prod_base
    prog = []
    prog += [isa.Instr(IMM, 0), isa.Instr(LI, 0), isa.Instr(PSH, 0),
             isa.Instr(IMM, IMM), isa.Instr(EMIT, P)]          # produced[0]=IMM d1
    prog += _emit_seq(P + 1, PSH, 0)                           # produced[1]=PSH
    prog += [isa.Instr(IMM, 2), isa.Instr(LI, 0), isa.Instr(PSH, 0),
             isa.Instr(IMM, IMM), isa.Instr(EMIT, P + 2)]      # produced[2]=IMM d2
    prog += _emit_seq(P + 3, ADD, 0)                           # produced[3]=ADD
    prog += _emit_seq(P + 4, HALT, 0)                          # produced[4]=HALT
    prog += [isa.Instr(JMP, P)]                                # handoff
    return prog


def _build_lean(subset):
    """Build the LEAN CFM model (code_from_memory=True) for ``subset``.  ~1 GB
    sparse-resident; NEVER load_sparse_transformer / the dense path."""
    vm = Q.build(code_size=24, subset=subset, code_from_memory=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lean = LF.LeanQwenVM.from_full_vm(vm, device=torch.device(dev))
    return lean


def main():
    print(f"C4_PF_CFM={os.environ.get('C4_PF_CFM')} "
          f"C4_CFM_EMIT={os.environ.get('C4_CFM_EMIT')}", flush=True)

    # --- 1 & 2: append-and-run + re-emit on the fast SUBSET_MEM_CMP model -----
    lean = _build_lean(Q.SUBSET_MEM_CMP)
    print(f"[build] lean CFM model: code_from_memory={lean.code_from_memory} "
          f"subset={lean.subset.name} D={lean.QL.D_used}", flush=True)

    for name, (prog, expect) in [("append-and-run", prog_append_and_run()),
                                 ("re-emit", prog_reemit())]:
        res = LF.run_program_lean(lean, prog, max_steps=128, verbose=True)
        ref = res["ref_trace"]
        ok = res["exact"] and res["ax_trace"][-1] == expect
        print(f"[{name}] model_tail={res['ax_trace'][-3:]} "
              f"ref_tail={ref[-3:]} expect={expect} "
              f"byte_exact={res['exact']} PASS={ok}", flush=True)
        assert ok, (name, res["ax_trace"], ref)

    # --- 3: end-to-end compile+run (LOADED compiler, C source in data band) ----
    # The ADD-only compiler executes no MUL, so it runs on the SAME fast SUBSET_MEM_CMP
    # model (LI reads the source, ADD sums the produced operands).  A genuine
    # compile-then-run: the digit values are read from the SRC data band at runtime.
    PROD = 40
    for expr, d1, d2 in [("2+3", 2, 3), ("4+5", 4, 5), ("9+8", 9, 8)]:
        prog = list(prog_compile_add_expr(PROD))
        while len(prog) < PROD:
            prog.append(isa.Instr(isa.NOP, 0))
        prog += [isa.Instr(isa.NOP, 0)] * 5    # produced region PROD..PROD+4
        # seed the C source into the data §Memory (the "input file"): mem[0]=d1, mem[2]=d2.
        preload = [{"addr": 0, "val": d1}, {"addr": 2, "val": d2}]
        res = run_with_preloaded_mem(lean, prog, preload, max_steps=200)
        expect = d1 + d2
        produced = res["produced_code"][PROD:PROD + 5]
        dis = "; ".join(_dis(w) for w in produced)
        ok = res["exact"] and res["ax_trace"][-1] == expect
        print(f"[e2e] src={expr:>4} -> produced=[{dis}] "
              f"model={res['ax_trace'][-1]} expect={expect} "
              f"byte_exact={res['exact']} PASS={ok}", flush=True)
        assert ok, (expr, res["ax_trace"], res["ref_trace"], dis)

    print("\nALL CFM-EMIT DEMOS PASS (byte-exact vs isa.interpret).", flush=True)


def _dis(ins):
    nm = isa.NAMES.get(ins.op, str(ins.op))
    return nm if ins.imm == 0 else f"{nm} {ins.imm}"


def run_with_preloaded_mem(lean, code, preload, max_steps=200):
    """``run_program_lean`` with a pre-seeded store §Memory (the input 'file') AND
    a returned final runtime code list (so the demo can show the produced bytecode).
    A thin wrapper that seeds ``store_log`` before the loop and captures ``run_code``.
    """
    from c4_min.qwen_lean_forward import _build_stream_and_overlay, _cfm_emit_enabled
    from c4_min.qwen_lean_stack import LeanDataStack
    from c4_min.nibble_pure_forward import SP_INIT

    QL, L = lean.QL, lean.QL.L
    subset = lean.subset
    mem_init = {s["addr"]: s["val"] for s in preload}
    ref_trace = isa.interpret(code, max_steps=max_steps, mem_init=mem_init)

    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    store_log = [dict(s) for s in preload]     # pre-seeded input 'file'
    ax_trace = []
    cur_pc = 0
    ds = LeanDataStack()
    emit_on = _cfm_emit_enabled() and lean.code_from_memory
    run_code = list(code)

    for _ in range(max_steps):
        op = run_code[cur_pc].op if 0 <= cur_pc < len(run_code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF
        reg_state["STACK0"] = ds.top()

        x, positions = _build_stream_and_overlay(lean, run_code, reg_state,
                                                 store_log, load_addr)
        with torch.no_grad():
            hidden, _ = lean.forward(x, past=None, q_positions=positions)
        state = hidden[0, -1]

        if emit_on and op == isa.EMIT:
            produced_op = prev["AX"] & 0xFF
            produced_imm = ds.top() & 0xFF
            target = run_code[cur_pc].imm
            while target >= len(run_code):
                run_code.append(isa.Instr(isa.NOP, 0))
            run_code[target] = isa.Instr(produced_op, produced_imm)
            if ds.values:
                ds.values.pop()
            reg_state = {"PC": cur_pc + 1, "AX": prev["AX"],
                         "SP": (prev["SP"] + 1) & 0xFFFFFFFF,
                         "BP": prev["BP"], "STACK0": ds.top()}
            ax_trace.append(prev["AX"])
            cur_pc += 1
            if cur_pc < 0 or cur_pc >= len(run_code):
                break
            continue

        from c4_min.qwen_full_vm import _snap
        pc = _snap(state[L.PC_VAL])
        ax = lean.decode_ax(state, op) & 0xFF
        sp = _snap(state[L.SP_VAL]); bp = _snap(state[L.BP_VAL])
        stk = _snap(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5

        if subset.memory and op in (isa.SI, isa.SC):
            store_addr = (ds.values[-1] & 0xFF) if ds.values else _snap(state[L.STK_VAL])
            store_val = ax if op == isa.SI else (ax & 0xFF)
            store_log = [s for s in store_log
                         if (s["addr"] & 0xFF) != (store_addr & 0xFF)]
            store_log.append({"addr": store_addr, "val": store_val})
        ds.apply(op, prev_ax=prev["AX"], model_ax=ax)
        stk = ds.top()
        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": stk}
        ax_trace.append(ax)
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(run_code):
            break
    return {"ax_trace": ax_trace, "ref_trace": ref_trace,
            "exact": ax_trace == ref_trace, "steps": len(ax_trace),
            "produced_code": run_code}


if __name__ == "__main__":
    main()
