"""PINPOINT probe: is the operand DIGIT recoverable from the host hidden state at the
EXACT digit token, and does it GENERALIZE? Removes the "learned pointer might not find
the digit token" confound: take the host last-layer hidden state at the KNOWN token
offset of each operand's units digit and train a 10-way probe. Test on SAME-range
disjoint pairs AND a disjoint value range (magnitude shift). If even the exact-token
probe fails, operand-read is a REPRESENTATION wall; if it works here but the pooled
readout did not, the wall is ADDRESSING (finding the token) not representation.

Golden 174ece66 not on this path. Lean fp32, polls RSS, aborts > 9 GB. CPU ok.
"""
from __future__ import annotations
import functools, json, os, random, resource, time
print=functools.partial(print,flush=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024.0*1024.0)
PEAK=0.0
def poll(lim=9.0):
    global PEAK; r=rss_gb(); PEAK=max(PEAK,r)
    if r>lim: raise MemoryError(f"RSS {r:.2f}>{lim}")
    return r
torch.set_default_dtype(torch.float32)

def make_row(a,b): return f"The first number is {a} and the second number is {b}."

def digit_token_positions(tok,a,b):
    text=make_row(a,b)
    enc=tok(text,return_offsets_mapping=True,add_special_tokens=False)
    offs=enc["offset_mapping"]
    sa=text.index(f" is {a} ")+4; ea=sa+len(str(a))-1
    sb=text.rindex(f" is {b}.")+4; eb=sb+len(str(b))-1
    def tok_at(ci):
        for i,(c0,c1) in enumerate(offs):
            if c0<=ci<c1: return i
        return len(offs)-1
    return tok_at(ea),tok_at(eb)

@torch.no_grad()
def encode_at(host,tok,rows,dev,bs=16):
    Ha=[]; Hb=[]
    for i in range(0,len(rows),bs):
        chunk=rows[i:i+bs]; texts=[make_row(a,b) for a,b in chunk]
        pos=[digit_token_positions(tok,a,b) for a,b in chunk]
        enc=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False)
        enc={k:v.to(dev) for k,v in enc.items()}
        out=host.model(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"])
        h=out.last_hidden_state.float()
        for j,(ia,ib) in enumerate(pos):
            Ha.append(h[j,ia].cpu()); Hb.append(h[j,ib].cpu())
        poll()
    return torch.stack(Ha).to(dev),torch.stack(Hb).to(dev)

class UnitsProbe(nn.Module):
    def __init__(self,hid):
        super().__init__(); self.net=nn.Sequential(nn.Linear(hid,256),nn.GELU(),nn.Linear(256,10))
    def forward(self,h): return self.net(h)

def units(x): return x%10

def train_eval(host,tok,tr,te,dev,steps,label,results):
    Ha_tr,Hb_tr=encode_at(host,tok,tr,dev); Ha_te,Hb_te=encode_at(host,tok,te,dev)
    ya=torch.tensor([units(a) for a,b in tr],device=dev); yb=torch.tensor([units(b) for a,b in tr],device=dev)
    ya_te=torch.tensor([units(a) for a,b in te],device=dev); yb_te=torch.tensor([units(b) for a,b in te],device=dev)
    hid=Ha_tr.shape[1]; pa=UnitsProbe(hid).to(dev); pb=UnitsProbe(hid).to(dev)
    opt=torch.optim.AdamW(list(pa.parameters())+list(pb.parameters()),lr=1e-3,weight_decay=1e-4)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(pa(Ha_tr),ya)+F.cross_entropy(pb(Hb_tr),yb)
        loss.backward(); opt.step()
        if s%150==0 or s==steps-1: poll()
    with torch.no_grad():
        a_tr=int((pa(Ha_tr).argmax(-1)==ya).sum()); b_tr=int((pb(Hb_tr).argmax(-1)==yb).sum())
        a_te=int((pa(Ha_te).argmax(-1)==ya_te).sum()); b_te=int((pb(Hb_te).argmax(-1)==yb_te).sum())
    ntr=len(tr); nte=len(te)
    print(f"[{label}] UNITS-digit @exact token: train A={a_tr}/{ntr} B={b_tr}/{ntr}  "
          f"TEST A={a_te}/{nte} ({100.0*a_te/nte:.1f}%) B={b_te}/{nte} ({100.0*b_te/nte:.1f}%)")
    results[label]=dict(train_a=f"{a_tr}/{ntr}",train_b=f"{b_tr}/{ntr}",
                        test_a=f"{a_te}/{nte}",test_b=f"{b_te}/{nte}",
                        test_a_pct=100.0*a_te/nte,test_b_pct=100.0*b_te/nte)

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
    print(f"  hid={host.config.hidden_size} RSS={rss_gb():.2f}")
    rng=random.Random(5)
    ntr=500 if not quick else 60; nte=150 if not quick else 30; steps=800 if not quick else 120
    results={"host":MID,"dev":dev}
    tr=[(rng.randint(0,255),rng.randint(0,255)) for _ in range(ntr)]; seen=set(tr); te=[]
    while len(te)<nte:
        p=(rng.randint(0,255),rng.randint(0,255))
        if p not in seen: te.append(p)
    train_eval(host,tok,tr,te,dev,steps,"same-range disjoint pairs",results)
    tr2=[(rng.randint(0,149),rng.randint(0,149)) for _ in range(ntr)]
    te2=[(rng.randint(150,255),rng.randint(150,255)) for _ in range(nte)]
    train_eval(host,tok,tr2,te2,dev,steps,"disjoint value range (mag shift)",results)
    results["peak_rss_gb"]=PEAK; results["elapsed_s"]=time.time()-t0
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"_agent_inline_digit_pinpoint_results.json")
    with open(out,"w") as f: json.dump(results,f,indent=2,default=lambda o: float(o))
    print(f"\nWrote {out} (elapsed {time.time()-t0:.1f}s PEAK RSS {PEAK:.2f} GB)")

if __name__=="__main__": main()
