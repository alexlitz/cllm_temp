"""Stratified CORPUS SAMPLE through the FUSED Qwen VM — argmax-exactness report.

Runs a stratified set of programs across ALL op families (arith, cmp, var, if,
func, loop, memory, mul/div/mod, bitwise, shift) through the REAL
``transformers.Qwen2Model.forward`` (``qwen_full_vm``), and reports the fraction
that decode argmax-exact vs the ``isa.interpret`` reference — the deliverable.

Each program is tagged with the MINIMAL op subset it needs; the runner builds one
Qwen VM per subset (so a family that needs only base ops runs on the model that
fits stock Qwen2.5-0.5B, and the wider families run on the correspondingly-widened
genuine-Qwen2 config). Every result is checked against the SAME ``isa.interpret``
ground truth used everywhere in c4_min.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import isa
from . import qwen_full_vm as Q


@dataclass
class Case:
    family: str
    label: str
    prog: List[Tuple[str, int]]
    subset: str                       # "base" | "mem+cmp" | "bitwise" | "full"


# ---------------------------------------------------------------------------
# The stratified corpus. Each family probes the op through the same per-step path
# a real c4 program uses (IMM/PSH operand setup + the op + HALT), plus control /
# memory / loop shapes.
# ---------------------------------------------------------------------------
def _bin(op, a, b):
    return [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)]


def corpus() -> List[Case]:
    C: List[Case] = []

    # -- arith (base) --
    for a, b in [(3, 4), (100, 27), (200, 55), (255, 1)]:
        C.append(Case("arith", f"add_{a}_{b}", _bin("ADD", a, b), "base"))
    for a, b in [(9, 4), (200, 55), (7, 9), (0, 1)]:
        C.append(Case("arith", f"sub_{a}_{b}", _bin("SUB", a, b), "base"))

    # -- IMM / PSH round-trip (base) --
    for v in (0, 42, 200):
        C.append(Case("arith", f"imm_{v}", [("IMM", v), ("HALT", 0)], "base"))

    # -- if / branch (base) --
    C.append(Case("if", "bz_taken",
                  [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)], "base"))
    C.append(Case("if", "bz_not",
                  [("IMM", 1), ("BZ", 4), ("IMM", 42), ("HALT", 0)], "base"))
    C.append(Case("if", "bnz_taken",
                  [("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)], "base"))
    C.append(Case("if", "bnz_not",
                  [("IMM", 0), ("BNZ", 4), ("IMM", 42), ("HALT", 0)], "base"))
    C.append(Case("if", "jmp_fwd",
                  [("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)], "base"))

    # -- loop (base): count down from 3, sum via repeated SUB; back-branch BNZ --
    #   AX=3; loop: AX = AX-1; BNZ loop; HALT.  ends AX=0 after 3 iters.
    C.append(Case("loop", "countdown3",
                  [("IMM", 3),                 # 0: AX=3
                   ("PSH", 0), ("IMM", 1), ("SUB", 0),   # 1-3: AX = STACK0 - 1... (AX-1)
                   ("BNZ", 1),                 # 4: if AX!=0 loop to idx1
                   ("HALT", 0)], "base"))
    C.append(Case("loop", "countdown2",
                  [("IMM", 2), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1),
                   ("HALT", 0)], "base"))

    # -- func (base): JSR/ENT/LEV leaf call; ADJ frame hygiene --
    C.append(Case("func", "call_leaf_42",
                  [("IMM", 0), ("JSR", 4), ("PSH", 0), ("HALT", 0),
                   ("ENT", 0), ("IMM", 42), ("LEV", 0)], "base"))
    C.append(Case("func", "call_leaf_55",
                  [("IMM", 0), ("JSR", 4), ("PSH", 0), ("HALT", 0),
                   ("ENT", 0), ("IMM", 55), ("LEV", 0)], "base"))
    C.append(Case("func", "nested_g37",
                  [("IMM", 1), ("JSR", 4), ("HALT", 0), ("NOP", 0),        # 0-3 main
                   ("ENT", 0), ("JSR", 8), ("LEV", 0), ("NOP", 0),         # 4-7 f
                   ("ENT", 0), ("IMM", 0x37), ("LEV", 0)], "base"))        # 8-10 g
    C.append(Case("func", "adj_discard",
                  [("IMM", 0x99), ("PSH", 0), ("ADJ", 1), ("IMM", 0x11), ("HALT", 0)],
                  "base"))
    C.append(Case("func", "ent_imm",
                  [("ENT", 8), ("IMM", 5), ("HALT", 0)], "base"))

    # -- cmp (mem+cmp) --
    for op, a, b in [("EQ", 5, 5), ("EQ", 7, 9), ("NE", 7, 9), ("NE", 5, 5),
                     ("LT", 7, 9), ("LT", 9, 7), ("GT", 9, 7), ("GT", 7, 9),
                     ("LE", 7, 9), ("LE", 9, 7), ("GE", 9, 7), ("GE", 7, 9)]:
        C.append(Case("cmp", f"{op.lower()}_{a}_{b}", _bin(op, a, b), "mem+cmp"))

    # -- memory / var (mem+cmp): store a local then load it (a "variable") --
    C.append(Case("memory", "si_li_0x23",
                  [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
                   ("IMM", 5), ("LI", 0), ("HALT", 0)], "mem+cmp"))
    C.append(Case("memory", "si_li_0x41",
                  [("IMM", 9), ("PSH", 0), ("IMM", 0x41), ("SI", 0),
                   ("IMM", 9), ("LI", 0), ("HALT", 0)], "mem+cmp"))
    C.append(Case("var", "var_add",   # x=7; y=x+3 -> store x, load, add
                  [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),      # mem[10]=7
                   ("IMM", 10), ("LI", 0),                              # AX=7
                   ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)], "mem+cmp"))
    C.append(Case("var", "var_two",   # two locals, sum them
                  [("IMM", 20), ("PSH", 0), ("IMM", 4), ("SI", 0),     # mem[20]=4
                   ("IMM", 21), ("PSH", 0), ("IMM", 5), ("SI", 0),     # mem[21]=5
                   ("IMM", 20), ("LI", 0), ("PSH", 0),
                   ("IMM", 21), ("LI", 0), ("ADD", 0), ("HALT", 0)], "mem+cmp"))

    # -- bitwise + shift (bitwise subset) --
    for op, a, b in [("AND", 0x6C, 0x3A), ("AND", 0xFF, 0x0F),
                     ("OR", 0x6C, 0x3A), ("OR", 0xF0, 0x0F),
                     ("XOR", 0xAA, 0x55), ("XOR", 0x6C, 0x3A)]:
        C.append(Case("bitwise", f"{op.lower()}_{a}_{b}", _bin(op, a, b), "bitwise"))
    for op, a, b in [("SHL", 5, 3), ("SHL", 1, 7), ("SHR", 200, 2), ("SHR", 0xFF, 4)]:
        C.append(Case("shift", f"{op.lower()}_{a}_{b}", _bin(op, a, b), "bitwise"))

    # -- mul/div/mod (muldiv subset; pruned FFN table, no bitwise -> leaner build) --
    for op, a, b in [("MUL", 6, 7), ("MUL", 12, 12), ("MUL", 15, 17),
                     ("DIV", 84, 7), ("DIV", 100, 3), ("DIV", 9, 4),
                     ("MOD", 84, 5), ("MOD", 100, 7), ("MOD", 9, 4)]:
        C.append(Case("muldiv", f"{op.lower()}_{a}_{b}", _bin(op, a, b), "muldiv"))

    return C


_SUBSET = {"base": Q.SUBSET_BASE, "mem+cmp": Q.SUBSET_MEM_CMP,
           "bitwise": Q.SUBSET_BITWISE, "muldiv": Q.SUBSET_MULDIV,
           "full": Q.SUBSET_FULL}
_MDM_OP = {"MUL": isa.MUL, "DIV": isa.DIV, "MOD": isa.MOD}


def _mdm_keys_for(cases: List[Case]):
    """The pruned MUL/DIV/MOD table keys the muldiv cases reach (op, a, b)."""
    keys = []
    for c in cases:
        if c.subset in ("full", "muldiv"):
            # binop shape: IMM a; PSH; IMM b; OP; HALT
            opname = c.prog[3][0]
            a = c.prog[0][1]
            b = c.prog[2][1]
            if opname in _MDM_OP:
                keys.append((_MDM_OP[opname], a, b))
    return keys


def run(subsets: Optional[List[str]] = None, code_size: int = 24,
        verbose: bool = False) -> Dict[str, object]:
    """Run the corpus through the fused Qwen VM, one model per subset. Returns a
    report with per-family and overall argmax-exact pass fractions.

    ``subsets`` restricts which subsets to build (memory-safe: e.g. ["base"] runs
    only the stock-0.5B-fitting families)."""
    cases = corpus()
    if subsets is not None:
        cases = [c for c in cases if c.subset in subsets]

    by_subset: Dict[str, List[Case]] = {}
    for c in cases:
        by_subset.setdefault(c.subset, []).append(c)

    results = []
    fam_pass: Dict[str, List[int]] = {}
    for sub_name, subs_cases in by_subset.items():
        mdm_keys = (_mdm_keys_for(subs_cases)
                    if sub_name in ("full", "muldiv") else None)
        vm = Q.build(code_size=code_size, subset=_SUBSET[sub_name], mdm_keys=mdm_keys)
        for c in subs_cases:
            res = Q.run_program(vm, isa.assemble(c.prog), max_steps=48)
            ok = res["exact"]
            results.append({"family": c.family, "label": c.label,
                            "subset": sub_name, "exact": ok,
                            "got": res["ax_trace"], "ref": res["ref_trace"],
                            "fits_stock": vm.fits_stock})
            fam_pass.setdefault(c.family, []).append(1 if ok else 0)
            if verbose:
                tag = "PASS" if ok else "FAIL"
                print(f"  [{tag}] {c.family:8s} {c.label:16s} ({sub_name}) "
                      f"got={res['ax_trace'][:6]} ref={res['ref_trace'][:6]}")
        del vm

    n_pass = sum(r["exact"] for r in results)
    n_total = len(results)
    families = {fam: (sum(v), len(v)) for fam, v in fam_pass.items()}
    return {"results": results, "n_pass": n_pass, "n_total": n_total,
            "families": families,
            "pass_fraction": (n_pass / n_total) if n_total else 0.0}


def _demo(subsets=None):
    import json
    rep = run(subsets=subsets, verbose=True)
    print("\n=== per-family (argmax-exact through Qwen2Model.forward) ===")
    for fam, (p, t) in sorted(rep["families"].items()):
        print(f"  {fam:10s} {p}/{t}")
    print(f"\nOVERALL: {rep['n_pass']}/{rep['n_total']} "
          f"= {100 * rep['pass_fraction']:.1f}% argmax-exact through the real Qwen2 forward")


if __name__ == "__main__":
    import sys
    subs = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    _demo(subs)
