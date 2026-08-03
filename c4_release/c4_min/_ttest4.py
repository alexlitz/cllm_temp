import os; os.environ['OMP_NUM_THREADS']='4'
import torch, torch.nn.functional as F
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.sparse_coo_spmm import CooSpmmFFN, _dense_of
dev = torch.device('cuda:0')
model, L, _ = build_compact_sparse_streaming(code_size=44, compute_mode='sparse_mm')
model.to(dev); model.materialize_dense(dev)
# find distinct block index 239 in the enumeration order used by the check
seen=set(); idx=0; target=None
for bi,b in enumerate(model.blocks):
    if id(b) in seen: continue
    seen.add(id(b))
    if getattr(b,'_routed',False) or getattr(b.ffn,'W_up',None) is None: continue
    if bi==239: target=b
b=target if target is not None else model.blocks[239]
print('block 239 Dff', _dense_of(b.ffn.W_up).shape, 'nnz up', int((_dense_of(b.ffn.W_up)!=0).sum()))
coo = CooSpmmFFN(b.ffn, dev)
torch.manual_seed(0)
K=17
x = torch.randn(1,K,model.dim,device=dev)*0.5
ref = b.ffn.forward(x)
got = coo.forward(x)
d=(ref-got).abs()
print('max abs', d.max().item(), 'argmax', d.argmax().item())
# decompose: check up/gate/down individually
xk = x.reshape(K,model.dim).transpose(0,1).contiguous()
up_ref = F.linear(x, _dense_of(b.ffn.W_up)) + b.ffn.b_up
up_coo = coo.up.forward(xk, coo.b_up).transpose(0,1).reshape(1,K,-1)
print('up L-inf', (up_ref-up_coo).abs().max().item())
gate_ref = F.linear(x, _dense_of(b.ffn.W_gate)) + b.ffn.b_gate
gate_coo = coo.gate.forward(xk, coo.b_gate).transpose(0,1).reshape(1,K,-1)
print('gate L-inf', (gate_ref-gate_coo).abs().max().item())
hid_ref = F.silu(up_ref)*gate_ref
hid_coo = F.silu(up_coo)*gate_coo
print('hidden L-inf', (hid_ref-hid_coo).abs().max().item())
# down: note ref uses hid_ref (from dense), coo uses hid_coo; test down alone on same hid
hk = hid_ref.reshape(K,-1).transpose(0,1).contiguous()
down_ref = F.linear(hid_ref, _dense_of(b.ffn.W_down))
down_coo = coo.down.forward(hk).transpose(0,1).reshape(1,K,-1)
print('down L-inf (same input)', (down_ref-down_coo).abs().max().item())
