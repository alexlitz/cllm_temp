import os, sys
HERE="/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/agent-a1b98bc10378dc470"
for _p in (HERE, os.path.join(HERE,"c4_release")):
    if _p not in sys.path: sys.path.insert(0,_p)
from c4_release.tools.capture_residual_trace import ResidualTracer
t=ResidualTracer()
for li in range(8,15):
    if li>=len(t.model.blocks): break
    a=t.model.blocks[li].attn
    lg=getattr(t.model.blocks[li],"_logical_layer","?")
    HD=a.W_q.shape[0]//a.num_heads
    sl=getattr(a,"alibi_slopes",None)
    slv=None if sl is None else [round(float(x),3) for x in sl.tolist()]
    print(f"blk{li} L{lg} num_heads={a.num_heads} HD={HD} Wq.shape={tuple(a.W_q.shape)} slopes={slv}")
