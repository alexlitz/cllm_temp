import os, sys
for k,v in dict(C4_TEST_SPEC_K="0",C4_SMOKE_SPEC_K="0",C4_NO_STACK0_EMIT="1",C4_OPERAND_FROM_MEMSP="1",C4_SKIP_DIM_INTEGRITY="1",C4_SKIP_GATE_CHECK="1").items():
    os.environ.setdefault(k,v)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from tools.probe_groundtruth import build_groundtruth_probe
p=build_groundtruth_probe(); model=p.model; dp=dict(model.dim_positions)
inv={v:k for k,v in dp.items()}
ffn=model.blocks[15].ffn
def dense(t): return t.to_dense() if t.layout!=torch.strided else t
Wu=dense(ffn.W_up.data); Wd=dense(ffn.W_down.data); Wg=dense(ffn.W_gate.data)
bu=dense(ffn.b_up.data); bg=dense(ffn.b_gate.data)
alo=dp["ALU_LO"]; ahi=dp["ALU_HI"]
# clear units: those writing negative to ALU_LO band cells
cu=sorted(set(torch.nonzero(Wd[alo:alo+16,:]< -0.01)[:,1].tolist()
             + torch.nonzero(Wd[ahi:ahi+16,:]< -0.01)[:,1].tolist()))
print("num clear units (ALU_LO/HI neg-write):", len(cu))
print("clear unit idxs:", cu[:40])
mark_ax=dp["MARK_AX"]
cmp_ops=["OP_EQ","OP_NE","OP_LT","OP_GT","OP_LE","OP_GE"]
opgt=dp["OP_GT"]; opsi=dp["OP_SI"]
for u in cu[:4]:
    row=Wu[u]
    print(f"\nunit {u}: b_up={float(bu[u]):.1f} MARK_AX={float(row[mark_ax]):.1f} "
          f"OP_SI={float(row[opsi]):.1f} OP_GT={float(row[opgt]):.1f}")
    for op in cmp_ops:
        print(f"    W_up[{op}]={float(row[dp[op]]):.3f}", end="")
    print()
    # what down-writes
    dcols=torch.nonzero(dense(ffn.W_down.data)[:,u].abs()>0.01).flatten().tolist()
    print("    down-writes to:", [(inv.get(d,d), round(float(dense(ffn.W_down.data)[d,u]),3)) for d in dcols[:6]])
