"""Find what W_o columns for heads 5/6/7 of block 11.attn contain. Try to id which row Q/K dims fire at REG_AX_mark."""
import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
model = runner.model
model.eval()
attn = model.blocks[11].attn
W_q = attn.W_q.data.to_dense() if attn.W_q.is_sparse else attn.W_q.data
W_k = attn.W_k.data.to_dense() if attn.W_k.is_sparse else attn.W_k.data
W_v = attn.W_v.data.to_dense() if attn.W_v.is_sparse else attn.W_v.data
W_o = attn.W_o.data.to_dense() if attn.W_o.is_sparse else attn.W_o.data
HD = W_o.shape[1] // attn.num_heads

# For each head, find non-zero W_q dims and corresponding W_k dims
for h in range(attn.num_heads):
    base = h * HD
    print(f"\n=== Head {h} (slots {base}:{base+HD}) ===")
    for slot in range(HD):
        wq_row = W_q[base + slot]
        wk_row = W_k[base + slot]
        wv_row = W_v[base + slot]
        if wq_row.abs().sum() > 0 or wk_row.abs().sum() > 0:
            wq_top = wq_row.abs().topk(3)
            wq_dims = [(i.item(), wq_row[i].item()) for i in wq_top.indices if abs(wq_row[i].item()) > 0]
            wk_top = wk_row.abs().topk(3)
            wk_dims = [(i.item(), wk_row[i].item()) for i in wk_top.indices if abs(wk_row[i].item()) > 0]
            wv_top = wv_row.abs().topk(3)
            wv_dims = [(i.item(), wv_row[i].item()) for i in wv_top.indices if abs(wv_row[i].item()) > 0]
            wo_col = W_o[:, base + slot]
            wo_top = wo_col.abs().topk(3)
            wo_dims = [(i.item(), wo_col[i].item()) for i in wo_top.indices if abs(wo_col[i].item()) > 0]
            print(f"  slot {slot}: Q={wq_dims}  K={wk_dims}  V={wv_dims}  O={wo_dims}")
