import os, sys
HERE="/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/agent-a1b98bc10378dc470"
for _p in (HERE, os.path.join(HERE,"c4_release")):
    if _p not in sys.path: sys.path.insert(0,_p)
import torch
from c4_release.tools.capture_residual_trace import ResidualTracer
t=ResidualTracer(); d=t.dim_positions
def hot(s,thr=0.3): return [(i,round(float(v),3)) for i,v in enumerate(s) if abs(float(v))>thr]
mark_ax=d["MARK_AX"]; mark_se=d.get("MARK_SE_ONLY",d.get("MARK_SE"))
# Put a NOP-ish opcode AFTER the cmp so the cmp step has a downstream SE row.
# Use: cmp then PSH then EXIT so the cmp's SE row exists.
progs={
  "LT 5<7 (->1)": "IMM 7; PSH; IMM 5; LT; PSH; EXIT",
  "GT 5>7 (->0)": "IMM 7; PSH; IMM 5; GT; PSH; EXIT",
  "EQ 5==5(->1)": "IMM 5; PSH; IMM 5; EQ; PSH; EXIT",
  "NE 5!=5(->0)": "IMM 5; PSH; IMM 5; NE; PSH; EXIT",
}
for label,prog in progs.items():
    cap=t.run(prog)
    last=max(cap.after_layer); arr=cap.after_layer[last]
    # find cmp opcode AX row
    opmap={"LT":"OP_LT","GT":"OP_GT","EQ":"OP_EQ","NE":"OP_NE"}
    opdim=d[opmap[label[:2]]]
    ax_op=[r for r in range(arr.shape[1]) if float(arr[0,r,opdim])>1.0 and float(arr[0,r,mark_ax])>0.5]
    if not ax_op: print(label,"no cmp AX row"); continue
    ax_row=ax_op[0]
    se_cand=[s for s in torch.nonzero(arr[0,:,mark_se]>0.5).squeeze(-1).tolist() if 20<=(s-ax_row)<=38]
    if not se_cand: print(label,f"no SE row after cmp AX {ax_row}"); continue
    se_row=se_cand[0]
    b11=cap.after_layer[11]
    secmp=hot(b11[0,se_row,d["SE_CMP"]:d["SE_CMP"]+4])
    seop=hot(b11[0,se_row,opdim.__class__ and d[opmap[label[:2]].replace('OP_','SE_OP_')]:d[opmap[label[:2]].replace('OP_','SE_OP_')]+1])
    cmp_raw=hot(b11[0,se_row,d["CMP"]:d["CMP"]+4])
    out_lo=hot(b11[0,se_row,d["OUTPUT_LO"]:d["OUTPUT_LO"]+4])
    print(f"{label}: result={cap.result!r} ax_row={ax_row} se_row={se_row}")
    print(f"   SE_OP={seop} SE_CMP={secmp} rawCMP(SErow)={cmp_raw} OUTPUT_LO(SErow)={out_lo}")
