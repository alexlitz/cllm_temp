"""Verify C4_CMP_H4_QVETO (real build path, spec_k=0).

Builds the model with the flag ON (make_cmp_h4_qveto_op fires during the
normal bake pipeline — NO manual weight hack) and traces ALU_LO at block-17
for the block-17-crush programs. Expect id433 (x=35>76) to go clean
(min=-8,argmax@3=-2 -> min~-0.1,argmax@3=+6); x=5 unchanged; id437 unchanged
(separate block-15 crush, out of scope).

Run with C4_CMP_H4_QVETO=1 (ON) and =0 (OFF) to A/B.
"""
import os, sys
os.environ.setdefault("C4_CMP_H4_QVETO", "1")
os.environ.update(dict(
    C4_TEST_SPEC_K="0", C4_SMOKE_SPEC_K="0",
    C4_NO_STACK0_EMIT="1", C4_OPERAND_FROM_MEMSP="1",
    C4_DERIVE_ALU_CLEAR="1", C4_CMP_BYTE0_SE_RECOVER="0",
    C4_SKIP_DIM_INTEGRITY="1", C4_SKIP_GATE_CHECK="1",
))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from src.compiler import compile_c
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe

p = build_groundtruth_probe(); model = p.model
dp = dict(model.dim_positions); dev = p._device
STEP = int(Token.STEP_TOKENS); alo = dp["ALU_LO"]


def alu_at(src, pc, stopblk):
    bc, data = compile_c(src)
    opc_ax, otok = p.runner._oracle_pc_ax_steps(
        bc, data or b"", "", expected_steps=None, with_tokens=True)
    prompt = p.runner._build_element(
        bc, data or b"", [], "", spec_k=1, adaptive_start_k=0,
        expected_steps=None)
    prefix = list(prompt.context); tape = list(prefix)
    for stp in otok:
        tape.extend(stp)
    padded = torch.tensor([tape], dtype=torch.long, device=dev)
    rstep = next(s for s, (pcx, ax) in enumerate(opc_ax) if pcx == pc)
    with torch.no_grad():
        emb = model.embed(padded)[0]
    ss = len(prefix) + rstep * STEP
    axrow = next(ss + off for off in range(STEP)
                 if emb[ss + off, dp["MARK_AX"]].abs().item() > 0.5)
    with torch.no_grad():
        r = model.forward(padded, stop_after_block=stopblk)[0][axrow]
    return [round(float(r[alo + i]), 1) for i in range(16)]


progs = [
    ("int main() { int x; x = 5; if (x > 76) return 1; return 0; }", 106, "x=5"),
    ("int main() { int x; x = 35; if (x > 76) return 1; return 0; }", 106, "id433 x=35"),
    ("int main() { int x; x = 10; if (x > 30) return 1; return 0; }", 106, "id437 x=10"),
]

flag = os.environ.get("C4_CMP_H4_QVETO", "0")
print(f"=== C4_CMP_H4_QVETO={flag} (built via real bake path) ALU_LO@blk17 ===")
for s, pc, l in progs:
    lo = alu_at(s, pc, 17)
    print(f"  {l}: min={min(lo)} argmax@{lo.index(max(lo))}={max(lo)}")
