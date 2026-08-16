"""DIGIT-COMPOSITIONAL operand readout — the fair inline-generalization test.

The 256-way operand classifier CANNOT generalize to unseen operand VALUES (a value
never seen as a label is unreachable). But numbers in text are DIGIT SEQUENCES: a
router that reads the operand DIGIT-BY-DIGIT (per-digit 10-way, then compose base-10)
should generalize to unseen values because the DIGITS recur. This is the honest
inline mechanism and decides whether byte-exact inline routing is achievable
end-to-end, or whether operand reading is a fundamental wall.

We compare, on operand values in a DISJOINT range from training (true value
extrapolation) AND on unseen phrasings:
  (D1) 256-way value classifier from the digit-token hidden states  (baseline, the
       formulation the main experiment used) — expected to NOT generalize.
  (D2) per-DIGIT readout: a learned pointer per digit-position reads the host hidden
       state at each digit token and classifies it 10-way; compose to the value.
       Generalizes iff the host encodes digit identity positionally.

Then we run the FULL inline pipeline with the digit readout feeding the FROZEN
byte-exact ALU and report held-out byte-exact N/N.

Golden 174ece66 not on this path. Lean fp32, polls RSS, aborts > 9 GB. Local only.
"""
from __future__ import annotations
import functools, json, os, random, resource, time
from typing import Dict, List, Tuple
print = functools.partial(print, flush=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from _agent_graft_sgd_vm_rl import (GraftedByteExactALU, construct_byte_exact,
    head_decode, OP_ID as ALU_OP_ID, OPS as ALU_OPS, IN_DIM, N_OPS,
    IN_ONE, IN_A, IN_B, IN_QUOT, IN_CMP_LT, IN_CMP_EQ, IN_CMP_GT, op_semantics)

def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024.0*1024.0)
PEAK=0.0
def poll(lim=9.0):
    global PEAK; r=rss_gb(); PEAK=max(PEAK,r)
    if r>lim: raise MemoryError(f"RSS {r:.2f} GB > {lim} -- ABORT")
    return r
torch.set_default_dtype(torch.float32)

MAXD=3  # up to 3 decimal digits (0..255)
# templates that put A and B at clean, separable positions
TRAIN_T=[" A is {a} and B is {b} , so the sum is",
         " first number {a} second number {b} the result is",
         " compute {a} plus {b} equals",
         " add {a} and {b} to get"]
HELD_T=[" take {a} then add {b} arriving at",
        " with {a} and {b} the total becomes"]

def digits3(x):  # value -> [hundreds,tens,units] as 0..9, left-padded
    s=f"{x:03d}"; return [int(c) for c in s]

def build(n, seed, lo, hi, templates):
    rng=random.Random(seed); rows=[]
    for _ in range(n):
        a=rng.randint(lo,hi); b=rng.randint(0,255-a if a<=255 else 0)
        b=rng.randint(lo,hi); # independent; ADD legal-byte handled by ALU wrap (mod256) but we keep a+b any
        rows.append((rng.choice(templates).format(a=a,b=b), a, b))
    return rows

@torch.no_grad()
def encode_full(host,tok,texts,dev,bs=16):
    hs=[]; ms=[]; maxT=0; buf=[]
    for i in range(0,len(texts),bs):
        enc=tok(texts[i:i+bs],return_tensors="pt",padding=True)
        enc={k:v.to(dev) for k,v in enc.items()}
        out=host.model(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"])
        h=out.last_hidden_state.float().cpu(); m=enc["attention_mask"].cpu()
        buf.append((h,m)); maxT=max(maxT,h.shape[1]); poll()
    for h,m in buf:
        if h.shape[1]<maxT:
            h=torch.cat([h,torch.zeros(h.shape[0],maxT-h.shape[1],h.shape[2])],1)
            m=torch.cat([m,torch.zeros(m.shape[0],maxT-m.shape[1],dtype=m.dtype)],1)
        hs.append(h); ms.append(m)
    return torch.cat(hs,0).to(dev), torch.cat(ms,0).to(dev)

class ValueClassifier(nn.Module):
    """(D1) baseline: attention-pool per operand -> 256-way value classifier."""
    def __init__(self,hid):
        super().__init__()
        self.qa=nn.Parameter(torch.randn(hid)*0.02); self.qb=nn.Parameter(torch.randn(hid)*0.02)
        self.da=nn.Sequential(nn.Linear(hid,512),nn.GELU(),nn.Linear(512,256))
        self.db=nn.Sequential(nn.Linear(hid,512),nn.GELU(),nn.Linear(512,256))
    def _pool(self,H,M,q):
        s=(H*q).sum(-1).masked_fill(~M.bool(),-1e30); w=torch.softmax(s,-1).unsqueeze(-1)
        return (w*H).sum(1)
    def forward(self,H,M):
        return self.da(self._pool(H,M,self.qa)), self.db(self._pool(H,M,self.qb))
    def predict_values(self,H,M):
        la,lb=self.forward(H,M); return la.argmax(-1), lb.argmax(-1)

class DigitReadout(nn.Module):
    """(D2) per-digit: MAXD learned pointers per operand read the host at each digit
    token, each classified 10-way; compose base-10 -> value. Generalizes to unseen
    values iff digits are positionally decodable in the host."""
    def __init__(self,hid):
        super().__init__()
        self.qa=nn.Parameter(torch.randn(MAXD,hid)*0.02)
        self.qb=nn.Parameter(torch.randn(MAXD,hid)*0.02)
        self.da=nn.ModuleList([nn.Sequential(nn.Linear(hid,256),nn.GELU(),nn.Linear(256,10)) for _ in range(MAXD)])
        self.db=nn.ModuleList([nn.Sequential(nn.Linear(hid,256),nn.GELU(),nn.Linear(256,10)) for _ in range(MAXD)])
    def _digit_logits(self,H,M,qset,decs):
        outs=[]
        for d in range(MAXD):
            s=(H*qset[d]).sum(-1).masked_fill(~M.bool(),-1e30)
            w=torch.softmax(s,-1).unsqueeze(-1); pooled=(w*H).sum(1)
            outs.append(decs[d](pooled))            # [B,10]
        return outs   # list of MAXD [B,10]
    def forward(self,H,M):
        return self._digit_logits(H,M,self.qa,self.da), self._digit_logits(H,M,self.qb,self.db)
    def predict_values(self,H,M):
        la,lb=self.forward(H,M)
        va=sum(la[d].argmax(-1)*(10**(MAXD-1-d)) for d in range(MAXD))
        vb=sum(lb[d].argmax(-1)*(10**(MAXD-1-d)) for d in range(MAXD))
        return va, vb

def train_value(m,H,M,rows,dev,steps,lr=1e-3):
    a=torch.tensor([r[1] for r in rows],device=dev); b=torch.tensor([r[2] for r in rows],device=dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        la,lb=m(H,M); loss=F.cross_entropy(la,a)+F.cross_entropy(lb,b)
        loss.backward(); opt.step()
        if s%150==0 or s==steps-1: poll()
    return float(loss.item())

def train_digit(m,H,M,rows,dev,steps,lr=1e-3):
    da=torch.tensor([digits3(r[1]) for r in rows],device=dev)  # [N,MAXD]
    db=torch.tensor([digits3(r[2]) for r in rows],device=dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        la,lb=m(H,M)
        loss=sum(F.cross_entropy(la[d],da[:,d]) for d in range(MAXD)) \
            +sum(F.cross_entropy(lb[d],db[:,d]) for d in range(MAXD))
        loss.backward(); opt.step()
        if s%150==0 or s==steps-1: poll()
    return float(loss.item())

@torch.no_grad()
def eval_read(m,H,M,rows,dev):
    va,vb=m.predict_values(H,M)
    a=torch.tensor([r[1] for r in rows],device=dev); b=torch.tensor([r[2] for r in rows],device=dev)
    a_ok=int((va==a).sum()); b_ok=int((vb==b).sum()); both=int(((va==a)&(vb==b)).sum())
    n=len(rows)
    return dict(a=f"{a_ok}/{n}",b=f"{b_ok}/{n}",both=f"{both}/{n}",both_pct=100.0*both/max(1,n)), va, vb

def main():
    t0=time.time(); quick=os.environ.get("INLINE_QUICK","0")=="1"
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MID="Qwen/Qwen2.5-0.5B-Instruct"
    dev="cuda:0" if torch.cuda.is_available() else "cpu"
    if os.environ.get("INLINE_DEVICE"): dev=os.environ["INLINE_DEVICE"]
    print(f"Host={MID} dev={dev} quick={quick}")
    tok=AutoTokenizer.from_pretrained(MID)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    host=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float32); poll()
    host=host.to(dev).eval()
    for p in host.parameters(): p.requires_grad_(False)
    hid=host.config.hidden_size; print(f"  hidden={hid} RSS={rss_gb():.2f}GB")
    alu=GraftedByteExactALU(temp=30.0).to(dev); construct_byte_exact(alu)
    for p in alu.parameters(): p.requires_grad_(False)

    ntr=500 if not quick else 60; nte=150 if not quick else 30
    steps=1000 if not quick else 150
    results={"host":MID,"dev":dev,"maxd":MAXD}

    # TRAIN operands in 0..149; TEST operands in 150..255 (disjoint value range) +
    # unseen phrasings. True operand-value extrapolation.
    tr=build(ntr,1,0,149,TRAIN_T)
    te=build(nte,9,150,255,HELD_T)
    Htr,Mtr=encode_full(host,tok,[r[0] for r in tr],dev)
    Hte,Mte=encode_full(host,tok,[r[0] for r in te],dev)

    print("\n=== (D1) 256-WAY VALUE classifier (baseline) — disjoint value range ===")
    vc=ValueClassifier(hid).to(dev); train_value(vc,Htr,Mtr,tr,dev,steps)
    e_tr,_,_=eval_read(vc,Htr,Mtr,tr,dev); e_te,_,_=eval_read(vc,Hte,Mte,te,dev)
    print(f"  train both={e_tr['both']} ({e_tr['both_pct']:.1f}%)  "
          f"HELD-OUT both={e_te['both']} ({e_te['both_pct']:.1f}%)  a={e_te['a']} b={e_te['b']}")
    results["value_classifier"]=dict(train=e_tr,heldout=e_te)

    print("\n=== (D2) PER-DIGIT readout (compose base-10) — disjoint value range ===")
    dr=DigitReadout(hid).to(dev); train_digit(dr,Htr,Mtr,tr,dev,steps)
    e_tr2,_,_=eval_read(dr,Htr,Mtr,tr,dev); e_te2,va,vb=eval_read(dr,Hte,Mte,te,dev)
    print(f"  train both={e_tr2['both']} ({e_tr2['both_pct']:.1f}%)  "
          f"HELD-OUT both={e_te2['both']} ({e_te2['both_pct']:.1f}%)  a={e_te2['a']} b={e_te2['b']}")
    results["digit_readout"]=dict(train=e_tr2,heldout=e_te2)

    # FULL inline pipeline on the HELD-OUT set with the DIGIT readout feeding the
    # FROZEN byte-exact ALU: byte-exact result N/N on unseen operand values.
    print("\n=== FULL INLINE PIPELINE (digit-readout -> FROZEN byte-exact ALU), held-out ===")
    feats=torch.zeros(len(te),IN_DIM,device=dev)
    feats[:,IN_ONE]=1.0; feats[:,IN_A]=va.float(); feats[:,IN_B]=vb.float()
    aq=va.clamp(min=1)
    feats[:,IN_QUOT]=torch.where(va>0,(vb//aq).float(),torch.zeros_like(va).float())
    feats[:,IN_CMP_LT]=(va<vb).float(); feats[:,IN_CMP_EQ]=(va==vb).float(); feats[:,IN_CMP_GT]=(va>vb).float()
    oph=F.one_hot(torch.full((len(te),),ALU_OP_ID["ADD"],device=dev),num_classes=N_OPS).float()
    res=alu(feats,oph); emitted=head_decode(res.double())
    tgt=torch.tensor([(r[1]+r[2])&0xFF for r in te],device=dev)
    be=int((emitted==tgt).sum()); n=len(te)
    # byte-exact GIVEN routed operands (isolates ALU exactness from read errors)
    exp_routed=torch.tensor([((int(va[i])+int(vb[i]))&0xFF) for i in range(n)],device=dev)
    be_given=int((emitted==exp_routed).sum())
    print(f"  held-out byte-exact ADD result (true) = {be}/{n} ({100.0*be/n:.1f}%)")
    print(f"  ALU byte-exact GIVEN routed operands  = {be_given}/{n} (isolates ALU: read err aside)")
    results["full_inline_heldout_byteexact"]=f"{be}/{n}"
    results["full_inline_heldout_byteexact_pct"]=100.0*be/n
    results["alu_byteexact_given_routed_heldout"]=f"{be_given}/{n}"

    results["peak_rss_gb"]=PEAK; results["elapsed_s"]=time.time()-t0
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"_agent_inline_digit_readout_results.json")
    with open(out,"w") as f: json.dump(results,f,indent=2,default=lambda o: float(o))
    print(f"\nWrote {out} (elapsed {time.time()-t0:.1f}s PEAK RSS {PEAK:.2f} GB)")

if __name__=="__main__":
    main()
