#!/usr/bin/env python3
"""_agent_analyze_chain.py — trace COO -> Identity -> MatMul chains to see how many
matmul RHS weights are Identity-indirected COO (so sparse-through-Identity would
catch them), and confirm which COO tensors have a genuine non-matmul/non-identity
consumer (needing dense reconstruction)."""
import struct, sys
from collections import defaultdict

binp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse/blockstack.nblbin"
with open(binp, "rb") as f:
    blob = f.read()
pos = [0]
def i32():
    v = struct.unpack_from("<i", blob, pos[0])[0]; pos[0]+=4; return v
magic=i32(); n_tensors=i32(); n_nodes=i32(); in_tid=i32(); out_tid=i32()
t_iscoo=[0]*n_tensors; t_nnz=[0]*n_tensors; t_name=[""]*n_tensors; t_dims=[None]*n_tensors
for i in range(n_tensors):
    nl=i32(); t_name[i]=blob[pos[0]:pos[0]+nl].decode('utf-8','replace'); pos[0]+=nl
    isi=i32(); dt=i32(); rank=i32(); dims=[i32() for _ in range(rank)]; t_dims[i]=dims; ne=i32()
    if isi==1:
        pos[0]+= (4 if dt==0 else 8)*ne
    elif isi==2:
        t_iscoo[i]=1; nnz=i32(); t_nnz[i]=nnz; pos[0]+=8*nnz+4*nnz
OP={0:"MatMul",1:"Add",5:"Gather",7:"Transpose",22:"Identity"}
nodes=[]
for nd in range(n_nodes):
    op=i32(); nin=i32(); ins=[i32() for _ in range(nin)]; nout=i32(); outs=[i32() for _ in range(nout)]
    nattr=i32()
    for _ in range(nattr):
        k=i32(); nv=i32(); pos[0]+=4*nv
    nodes.append((op, ins, outs))

# producer map: tensor -> (op, ins)
produced_by={}
for (op,ins,outs) in nodes:
    for o in outs:
        produced_by[o]=(op,ins)

def is_coo_source(t, depth=0):
    """True if tensor t is COO, or an Identity/Transpose of a COO source."""
    if t<0: return False
    if t_iscoo[t]: return True
    if t in produced_by and depth<8:
        op,ins=produced_by[t]
        if op==22 and len(ins)>=1:           # Identity
            return is_coo_source(ins[0], depth+1)
    return False

matmul_direct_coo=0
matmul_identity_coo=0
matmul_dense=0
mm_coo_nnz=0
for (op,ins,outs) in nodes:
    if op!=0: continue
    rhs=ins[1] if len(ins)>1 else -1
    if rhs>=0 and t_iscoo[rhs]:
        matmul_direct_coo+=1
    elif is_coo_source(rhs):
        matmul_identity_coo+=1
    else:
        matmul_dense+=1

print(f"MatMul nodes: direct-COO-RHS={matmul_direct_coo}  Identity-of-COO-RHS={matmul_identity_coo}  dense-RHS={matmul_dense}")

# which COO tensors are consumed by a genuine non-matmul, non-identity, non-transpose op?
consumers=defaultdict(set)
for (op,ins,outs) in nodes:
    for t in ins:
        if t>=0 and t_iscoo[t]:
            consumers[t].add(op)
genuine=[]
for t in range(n_tensors):
    if not t_iscoo[t]: continue
    ops=consumers[t]
    # allowed sparse-friendly: MatMul(0) as rhs, Identity(22)
    other=ops - {0, 22}
    if other:
        genuine.append((t, t_name[t], t_dims[t], sorted(other)))
print(f"\nCOO tensors with a genuine non-(matmul/identity) consumer: {len(genuine)}")
for g in genuine[:20]:
    print("  ", g)
