#!/usr/bin/env python3
"""_agent_analyze_graph.py — analyze the lowered .nblbin: which tensors are COO,
which ops consume them, and the total dense MAC count broken down by op, to see
where the wall goes and confirm COO tensors are matmul-RHS-only."""
import os, struct, sys
from collections import defaultdict

binp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fullisa_sparse/blockstack.nblbin"

with open(binp, "rb") as f:
    blob = f.read()
pos = [0]
def i32():
    v = struct.unpack_from("<i", blob, pos[0])[0]; pos[0] += 4; return v
def i64():
    v = struct.unpack_from("<q", blob, pos[0])[0]; pos[0] += 8; return v
def f32():
    v = struct.unpack_from("<f", blob, pos[0])[0]; pos[0] += 4; return v

magic = i32(); n_tensors = i32(); n_nodes = i32(); input_tid = i32(); output_tid = i32()
print(f"magic={magic:#x} n_tensors={n_tensors} n_nodes={n_nodes} in={input_tid} out={output_tid}")

t_isinit = [0]*n_tensors
t_dims = [None]*n_tensors
t_nnz = [0]*n_tensors
t_iscoo = [0]*n_tensors
t_name = [""]*n_tensors
for i in range(n_tensors):
    nl = i32(); name = blob[pos[0]:pos[0]+nl].decode('utf-8', 'replace'); pos[0]+=nl
    t_name[i] = name
    isi = i32(); t_isinit[i] = isi
    dt = i32(); rank = i32(); dims = [i32() for _ in range(rank)]; t_dims[i] = dims
    ne = i32()
    if isi == 1:
        if dt == 0:
            pos[0] += 4*ne
        else:
            pos[0] += 8*ne
    elif isi == 2:
        t_iscoo[i] = 1
        nnz = i32(); t_nnz[i] = nnz
        pos[0] += 8*nnz  # int64 idx
        pos[0] += 4*nnz  # f32 vals

OPNAMES = {0:"MatMul",1:"Add",2:"Sub",3:"Mul",4:"Div",5:"Gather",6:"Reshape",7:"Transpose",
           8:"Concat",9:"Unsqueeze",10:"Shape",11:"Cast",12:"Exp",13:"Neg",14:"Abs",
           15:"Range",16:"ReduceMax",17:"ReduceSum",18:"Clip",19:"Trilu",20:"ConstantOfShape",
           21:"Sigmoid",22:"Identity"}

coo_consumed_by = defaultdict(list)   # tid -> [(op, input-slot)]
matmul_rhs_coo = 0
matmul_total = 0
op_counts = defaultdict(int)
for nd in range(n_nodes):
    op = i32(); op_counts[OPNAMES.get(op,op)] += 1
    nin = i32(); ins = [i32() for _ in range(nin)]
    nout = i32(); outs = [i32() for _ in range(nout)]
    nattr = i32()
    for _ in range(nattr):
        key = i32(); nv = i32(); pos[0] += 4*nv
    if op == 0:
        matmul_total += 1
        if nin >= 2 and t_iscoo[ins[1]]:
            matmul_rhs_coo += 1
    for slot, t in enumerate(ins):
        if t >= 0 and t_iscoo[t]:
            coo_consumed_by[t].append((OPNAMES.get(op,op), slot))

n_coo = sum(t_iscoo)
print(f"\nCOO tensors: {n_coo}  total nnz={sum(t_nnz)}")
print(f"MatMul nodes: {matmul_total}  (with COO RHS: {matmul_rhs_coo})")

# check every COO tensor is consumed ONLY as MatMul RHS (slot 1)
bad = []
for t in range(n_tensors):
    if not t_iscoo[t]:
        continue
    uses = coo_consumed_by[t]
    for (opn, slot) in uses:
        if not (opn == "MatMul" and slot == 1):
            bad.append((t, t_name[t], opn, slot))
print(f"\nCOO tensors consumed as NON-(MatMul-RHS): {len(bad)}")
for b in bad[:20]:
    print("  ", b)
if not bad:
    print("  => every COO weight is used ONLY as a MatMul RHS. Dense reconstruction NOT needed.")

print("\nop counts:")
for k, v in sorted(op_counts.items(), key=lambda kv:-kv[1]):
    print(f"  {k:18s} {v}")
