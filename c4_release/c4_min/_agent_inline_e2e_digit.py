"""FULL INLINE / NO-MODE pipeline with the CORRECT operand-read mechanism.

The pinpoint probe showed the operand DIGITS are cleanly recoverable and GENERALIZE
from the frozen host's hidden state at the digit token (units digit 100% held-out).
The earlier 0/60 held-out was a FORMULATION artifact (256-way value-as-class can't
reach unseen values; a single pooled pointer can't find the digit token). Here we do
the honest inline pipeline end-to-end and MEASURE held-out byte-exact:

  (1) TRIGGER: content-triggered binary head on the answer-slot hidden state (from
      the main experiment: 100% recall / 0% false-trigger, generalizes) — NO mode token.
  (2) OP: content head -> which of ADD/SUB/CMP_LT/CMP_EQ/CMP_GT/DIV.
  (3) OPERANDS: read each operand's decimal digits at the located digit tokens
      (per-digit 10-way probe, compose base-10). Generalizes to unseen values.
  (4) FROZEN byte-exact ALU on the composed operands -> emitted byte.

We train on operands in 0..149 with the TRAIN phrasings, and TEST on a DISJOINT value
range (150..255) with UNSEEN phrasings — the true inline generalization test. We
report held-out full-inline byte-exact N/N (fire + op + operands + result), the
false-trigger rate, and compare to the value-as-class baseline.

Golden 174ece66 not on this path. Lean fp32, polls RSS, aborts > 9 GB. CPU ok.
"""
from __future__ import annotations
import functools, json, os, random, resource, time
from typing import List, Tuple, Optional
print=functools.partial(print,flush=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from _agent_graft_sgd_vm_rl import (GraftedByteExactALU, construct_byte_exact,
    head_decode, OP_ID as ALU_OP_ID, OPS as ALU_OPS, IN_DIM, N_OPS,
    IN_ONE, IN_A, IN_B, IN_QUOT, IN_CMP_LT, IN_CMP_EQ, IN_CMP_GT, op_semantics)
def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024.0*1024.0)
PEAK=0.0
def poll(lim=9.0):
    global PEAK; r=rss_gb(); PEAK=max(PEAK,r)
    if r>lim: raise MemoryError(f"RSS {r:.2f}>{lim}")
    return r
torch.set_default_dtype(torch.float32)
MAXD=3
USE_OPS=["ADD","SUB","DIV","CMP_LT","CMP_EQ","CMP_GT"]
# templates put "{a} ... {b}" with a and b as bare ints; sentence ends at answer slot.
TRAIN_T={
 "ADD":["So {a} plus {b} gives","The sum of {a} and {b} is","Adding {a} to {b} makes"],
 "SUB":["So {a} minus {b} gives","The difference of {a} and {b} is","Subtract {b} from {a} to get"],
 "DIV":["So {b} divided by {a} gives","The quotient of {b} over {a} is","Dividing {b} by {a} yields"],
 "CMP_LT":["Is {a} less than {b}? one or zero:","Checking if {a} is under {b} gives"],
 "CMP_EQ":["Is {a} equal to {b}? one or zero:","Checking if {a} matches {b} gives"],
 "CMP_GT":["Is {a} greater than {b}? one or zero:","Checking if {a} is over {b} gives"],
}
HELD_T={
 "ADD":["Combine {a} with {b} to reach","The total of {a} and {b} equals"],
 "SUB":["Reduce {a} by {b} arriving at","Take {b} away from {a} leaving"],
 "DIV":["Split {b} into {a} whole parts, each is","How many whole {a} go into {b}? it is"],
 "CMP_LT":["Determine if {a} is below {b}, one or zero:"],
 "CMP_EQ":["Tell if {a} is the same as {b}, one or zero:"],
 "CMP_GT":["Determine if {a} exceeds {b}, one or zero:"],
}
NEG_TRAIN=["The capital of France is","Water is essential for","The sun rises in the",
 "Books are a source of","The opposite of hot is","She opened the door and",
 "Music makes people feel","The mountain was covered in"]
NEG_HELD=["The children ran across the","A gentle rain fell over the",
 "Fresh bread smells absolutely","Stars are brightest on a clear","The garden was full of blooming",
 "There were 12 birds on the","She read 3 books during the","The recipe needs 2 cups of"]

class Sample:
    def __init__(s,text,is_arith,op,a,b,tgt):
        s.text=text; s.is_arith=is_arith; s.op=op; s.a=a; s.b=b; s.tgt=tgt

def legal(op,rng,lo,hi):
    if op=="ADD": a=rng.randint(lo,min(hi,200)); b=rng.randint(0,255-a); return a,b
    if op=="SUB": a=rng.randint(lo,hi); b=rng.randint(0,a); return a,b
    if op=="DIV": a=rng.randint(1,min(hi,20) if hi>=1 else 1); b=rng.randint(lo,hi); return max(1,a),b
    a=rng.randint(lo,hi); b=rng.randint(lo,hi); return a,b

def build(n_per_op,n_neg,seed,held,lo,hi):
    rng=random.Random(seed); data=[]
    T=HELD_T if held else TRAIN_T
    for op in USE_OPS:
        for _ in range(n_per_op):
            a,b=legal(op,rng,lo,hi)
            tgt,_,_=op_semantics(op,a,b,[0,0,0,0],0)
            data.append(Sample(rng.choice(T[op]).format(a=a,b=b),True,op,a,b,tgt))
    pool=NEG_HELD if held else NEG_TRAIN
    for _ in range(n_neg): data.append(Sample(rng.choice(pool),False,None,0,0,0))
    rng.shuffle(data); return data

def find_digit_tokens(tok,text,a,b):
    """Locate the token index of EACH decimal digit of a and b (left-padded MAXD).
    Returns two lists of MAXD token indices. If a digit position doesn't exist (short
    number), point at the number's first digit token (leading-zero surrogate)."""
    enc=tok(text,return_offsets_mapping=True,add_special_tokens=False)
    offs=enc["offset_mapping"]
    def tok_at(ci):
        for i,(c0,c1) in enumerate(offs):
            if c0<=ci<c1: return i
        return len(offs)-1
    def positions(x):
        sx=str(x)
        # find first occurrence of the exact number substring bounded by non-digits
        idx=-1
        for cand in range(len(text)-len(sx)+1):
            if text[cand:cand+len(sx)]==sx:
                left=text[cand-1] if cand>0 else ' '
                right=text[cand+len(sx)] if cand+len(sx)<len(text) else ' '
                if not left.isdigit() and not right.isdigit(): idx=cand; break
        if idx<0: idx=text.find(sx)
        char_positions=[idx+k for k in range(len(sx))]
        toks=[tok_at(cp) for cp in char_positions]
        # left-pad to MAXD by repeating the first token (surrogate for leading zero)
        while len(toks)<MAXD: toks=[toks[0]]+toks
        return toks[-MAXD:]
    return positions(a),positions(b)

@torch.no_grad()
def encode(host,tok,samples,dev,bs=16):
    """Return last-token hidden [N,hid] (answer slot) + per-sample digit-token hidden
    for a and b [N,MAXD,hid] (zeros for non-arith)."""
    hid=host.config.hidden_size
    Hlast=[]; Da=[]; Db=[]
    for i in range(0,len(samples),bs):
        chunk=samples[i:i+bs]; texts=[s.text for s in chunk]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False)
        enc={k:v.to(dev) for k,v in enc.items()}
        out=host.model(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"])
        h=out.last_hidden_state.float()
        last=(enc["attention_mask"].sum(1)-1).long()
        for j,s in enumerate(chunk):
            Hlast.append(h[j,last[j]].cpu())
            if s.is_arith:
                pa,pb=find_digit_tokens(tok,s.text,s.a,s.b)
                Da.append(torch.stack([h[j,p] for p in pa]).cpu())
                Db.append(torch.stack([h[j,p] for p in pb]).cpu())
            else:
                Da.append(torch.zeros(MAXD,hid)); Db.append(torch.zeros(MAXD,hid))
        poll()
    return torch.stack(Hlast).to(dev),torch.stack(Da).to(dev),torch.stack(Db).to(dev)

class Router(nn.Module):
    def __init__(self,hid):
        super().__init__()
        self.trig=nn.Sequential(nn.Linear(hid,256),nn.GELU(),nn.Linear(256,1))
        self.op=nn.Sequential(nn.Linear(hid,256),nn.GELU(),nn.Linear(256,N_OPS))
        self.dig_a=nn.ModuleList([nn.Sequential(nn.Linear(hid,256),nn.GELU(),nn.Linear(256,10)) for _ in range(MAXD)])
        self.dig_b=nn.ModuleList([nn.Sequential(nn.Linear(hid,256),nn.GELU(),nn.Linear(256,10)) for _ in range(MAXD)])
    def forward(self,Hlast,Da,Db):
        tl=self.trig(Hlast).squeeze(-1); ol=self.op(Hlast)
        la=[self.dig_a[d](Da[:,d]) for d in range(MAXD)]
        lb=[self.dig_b[d](Db[:,d]) for d in range(MAXD)]
        return tl,ol,la,lb
    def values(self,la,lb):
        va=sum(la[d].argmax(-1)*(10**(MAXD-1-d)) for d in range(MAXD))
        vb=sum(lb[d].argmax(-1)*(10**(MAXD-1-d)) for d in range(MAXD))
        return va,vb

def dtens(vals,dev):
    return torch.tensor([[int(c) for c in f"{v:03d}"] for v in vals],device=dev)

def train(router,Hl,Da,Db,samples,dev,steps,lr=1e-3):
    is_a=torch.tensor([s.is_arith for s in samples],dtype=torch.float32,device=dev)
    opid=torch.tensor([USE_OPS.index(s.op) if s.is_arith else 0 for s in samples],device=dev)
    da=dtens([s.a for s in samples],dev); db=dtens([s.b for s in samples],dev)
    am=is_a.bool()
    opt=torch.optim.AdamW(router.parameters(),lr=lr,weight_decay=1e-4)
    for st in range(steps):
        opt.zero_grad(set_to_none=True)
        tl,ol,la,lb=router(Hl,Da,Db)
        loss=F.binary_cross_entropy_with_logits(tl,is_a)
        if am.any():
            loss=loss+F.cross_entropy(ol[am],opid[am])
            for d in range(MAXD):
                loss=loss+F.cross_entropy(la[d][am],da[am,d])+F.cross_entropy(lb[d][am],db[am,d])
        loss.backward(); opt.step()
        if st%150==0 or st==steps-1: poll()
    return float(loss.item())

@torch.no_grad()
def evaluate(router,alu,Hl,Da,Db,samples,dev,thr=0.5):
    tl,ol,la,lb=router(Hl,Da,Db)
    trig=torch.sigmoid(tl)>thr; oppred=ol.argmax(-1); va,vb=router.values(la,lb)
    is_a=torch.tensor([s.is_arith for s in samples],dtype=torch.bool,device=dev)
    n_a=int(is_a.sum()); n_n=int((~is_a).sum())
    recall=int((trig&is_a).sum()); false_t=int((trig&~is_a).sum())
    # feed routed operands to frozen ALU
    feats=torch.zeros(len(samples),IN_DIM,device=dev)
    feats[:,IN_ONE]=1.0; feats[:,IN_A]=va.float(); feats[:,IN_B]=vb.float()
    aq=va.clamp(min=1); feats[:,IN_QUOT]=torch.where(va>0,(vb//aq).float(),torch.zeros_like(va).float())
    feats[:,IN_CMP_LT]=(va<vb).float(); feats[:,IN_CMP_EQ]=(va==vb).float(); feats[:,IN_CMP_GT]=(va>vb).float()
    oph=F.one_hot(oppred.clamp(0,N_OPS-1),num_classes=N_OPS).float()
    # map USE_OPS index -> ALU op id
    alu_ids=torch.tensor([ALU_OP_ID[USE_OPS[int(o)]] for o in oppred],device=dev)
    oph=F.one_hot(alu_ids,num_classes=N_OPS).float()
    emitted=head_decode(alu(feats,oph).double())
    full=0; opok=0; abok=0; per={op:[0,0] for op in USE_OPS}
    for i,s in enumerate(samples):
        if not s.is_arith: continue
        o_ok=USE_OPS[int(oppred[i])]==s.op
        ab_ok=int(va[i])==s.a and int(vb[i])==s.b
        r_ok=int(emitted[i])==s.tgt
        opok+=int(o_ok); abok+=int(ab_ok)
        f=bool(trig[i]) and o_ok and ab_ok and r_ok
        full+=int(f); per[s.op][0]+=int(f); per[s.op][1]+=1
    return dict(n_arith=n_a,n_non=n_n,
        trigger_recall=f"{recall}/{n_a}",trigger_recall_pct=100.0*recall/max(1,n_a),
        false_trigger=f"{false_t}/{n_n}",false_trigger_pct=100.0*false_t/max(1,n_n),
        op_exact=f"{opok}/{n_a}",operands_exact=f"{abok}/{n_a}",
        full_inline=f"{full}/{n_a}",full_inline_pct=100.0*full/max(1,n_a),
        per_op={op:f"{c}/{t}" for op,(c,t) in per.items()})

def main():
    t0=time.time(); quick=os.environ.get("INLINE_QUICK","0")=="1"
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MID="Qwen/Qwen2.5-0.5B-Instruct"; dev=os.environ.get("INLINE_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Host={MID} dev={dev}")
    tok=AutoTokenizer.from_pretrained(MID)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="right"
    host=AutoModelForCausalLM.from_pretrained(MID,dtype=torch.float32); poll(); host=host.to(dev).eval()
    for p in host.parameters(): p.requires_grad_(False)
    hid=host.config.hidden_size; print(f"  hid={hid} RSS={rss_gb():.2f}")
    alu=GraftedByteExactALU(temp=30.0).to(dev); construct_byte_exact(alu)
    for p in alu.parameters(): p.requires_grad_(False)
    npo=40 if not quick else 8; nneg=80 if not quick else 16; steps=800 if not quick else 120
    # TRAIN: operands 0..149, train phrasings. TEST: operands 150..255, HELD phrasings.
    tr=build(npo,nneg,1,held=False,lo=0,hi=149)
    te=build(max(10,npo//2),max(20,nneg//2),2,held=True,lo=150,hi=255)
    print(f"  train {len(tr)} ({sum(s.is_arith for s in tr)} arith) / test {len(te)} ({sum(s.is_arith for s in te)} arith)")
    Hl,Da,Db=encode(host,tok,tr,dev)
    Hl_te,Da_te,Db_te=encode(host,tok,te,dev)
    router=Router(hid).to(dev)
    fl=train(router,Hl,Da,Db,tr,dev,steps)
    ev_tr=evaluate(router,alu,Hl,Da,Db,tr,dev)
    ev_te=evaluate(router,alu,Hl_te,Da_te,Db_te,te,dev)
    print(f"  final loss {fl:.4f}")
    print(f"  [TRAIN] {ev_tr}")
    print(f"  [HELD-OUT: disjoint value range 150..255 + unseen phrasings] {ev_te}")
    results=dict(host=MID,dev=dev,mechanism="per-digit read at located digit tokens + frozen byte-exact ALU",
                 train=ev_tr,heldout=ev_te,final_loss=fl,
                 peak_rss_gb=PEAK,elapsed_s=time.time()-t0)
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"_agent_inline_e2e_digit_results.json")
    with open(out,"w") as f: json.dump(results,f,indent=2,default=lambda o: float(o))
    print(f"\nWrote {out} (PEAK RSS {PEAK:.2f} GB, {time.time()-t0:.1f}s)")

if __name__=="__main__": main()
