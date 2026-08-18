"""OPERAND-READ PROBE: is the operand VALUE recoverable from the host, and does the
operand read generalize to UNSEEN operand values?

The inline-routing experiment (_agent_inline_routing_nomode.py) found the trigger
("when to invoke the ALU") generalizes perfectly (100% recall / 0% false-trigger)
but the OPERAND VALUE read (a_head/b_head) does NOT generalize to unseen operands
(held-out 0/N). This probe isolates WHY:

  (P1) final-token-only readout: the router in the main file reads ONLY the last
       hidden state. A digit-value in the sentence may not be linearly present at
       the final token. -> probe.

  (P2) per-DIGIT-TOKEN readout: read the operand at the token(s) that ACTUALLY
       carry its digits (a small "operand pointer" attention over the sentence
       tokens), and decode the value from THOSE hidden states. This is the honest
       inline mechanism (DNC-style learned addressing). Does it generalize to unseen
       operand values?

We measure held-out (UNSEEN operand values, disjoint value ranges) operand-read
accuracy for both. This grounds whether byte-exact inline routing is achievable
end-to-end or whether operand reading is a fundamental limit.

Golden 174ece66 not on this path. Lean fp32, polls RSS, aborts > 9 GB. Local only.
"""
from __future__ import annotations
import functools, json, os, random, resource, time
from typing import Dict, List, Tuple
print = functools.partial(print, flush=True)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024.0*1024.0)
PEAK=0.0
def poll(lim=9.0):
    global PEAK; r=rss_gb(); PEAK=max(PEAK,r)
    if r>lim: raise MemoryError(f"RSS {r:.2f} GB > {lim} -- ABORT")
    return r
torch.set_default_dtype(torch.float32)

# sentences embedding "A op B" where the operand DIGITS live at known token spans.
# We test whether the operand VALUE can be read from the host at (a) the final
# token and (b) the operand's own digit tokens, and whether it GENERALIZES to
# operand values NOT seen at train (disjoint ranges).
TEMPLATES = [
    "So {a} plus {b} gives",
    "The sum of {a} and {b} is",
    "If you add {a} and {b} you get",
    "Adding {a} to {b} makes",
    "Take {a} and {b} together to get",
]
HELD_TEMPLATES = [
    "Combine {a} with {b} to reach",
    "The total of {a} and {b} equals",
]

def build(n, seed, lo, hi, templates):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.randint(lo, hi); b = rng.randint(lo, hi)
        t = rng.choice(templates)
        rows.append((t.format(a=a, b=b), a, b, t))
    return rows

class DigitPointerReadout(nn.Module):
    """Learned attention that POINTS at operand-digit tokens and decodes the operand
    value from the pointed hidden states. One pointer per operand (A,B). This is the
    inline 'read the operand from where it appears in the text' mechanism."""
    def __init__(self, hidden):
        super().__init__()
        self.qa = nn.Parameter(torch.randn(hidden)*0.02)
        self.qb = nn.Parameter(torch.randn(hidden)*0.02)
        self.dec_a = nn.Sequential(nn.Linear(hidden,512), nn.GELU(), nn.Linear(512,256))
        self.dec_b = nn.Sequential(nn.Linear(hidden,512), nn.GELU(), nn.Linear(512,256))
    def _read(self, H, mask, q):
        # H [B,T,hid], mask [B,T]; attention pointer -> pooled hidden [B,hid]
        score = (H*q).sum(-1)                       # [B,T]
        score = score.masked_fill(~mask.bool(), -1e30)
        w = torch.softmax(score, -1).unsqueeze(-1)  # [B,T,1]
        return (w*H).sum(1)                         # [B,hid]
    def forward(self, H, mask):
        pa = self._read(H, mask, self.qa)
        pb = self._read(H, mask, self.qb)
        return self.dec_a(pa), self.dec_b(pb)

class FinalTokenReadout(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dec_a = nn.Sequential(nn.Linear(hidden,512), nn.GELU(), nn.Linear(512,256))
        self.dec_b = nn.Sequential(nn.Linear(hidden,512), nn.GELU(), nn.Linear(512,256))
    def forward(self, hlast):
        return self.dec_a(hlast), self.dec_b(hlast)

@torch.no_grad()
def encode_full(host, tok, texts, device, bs=16):
    """Return [N,T,hid] full hidden states + [N,T] mask + [N] last-idx (right-pad)."""
    all_h, all_m, all_last = [], [], []
    maxT = 0
    enc_list = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        enc = tok(chunk, return_tensors="pt", padding=True)
        enc = {k:v.to(device) for k,v in enc.items()}
        out = host.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        h = out.last_hidden_state.float().cpu()
        m = enc["attention_mask"].cpu()
        enc_list.append((h,m)); maxT=max(maxT,h.shape[1]); poll()
    for h,m in enc_list:
        if h.shape[1]<maxT:
            padh=torch.zeros(h.shape[0],maxT-h.shape[1],h.shape[2])
            padm=torch.zeros(m.shape[0],maxT-m.shape[1],dtype=m.dtype)
            h=torch.cat([h,padh],1); m=torch.cat([m,padm],1)
        all_h.append(h); all_m.append(m)
    H=torch.cat(all_h,0).to(device); M=torch.cat(all_m,0).to(device)
    last=(M.sum(1)-1).long()
    return H, M, last

def train_probe(probe, get_pred, rows, device, steps=600, lr=1e-3):
    a=torch.tensor([r[1] for r in rows],device=device)
    b=torch.tensor([r[2] for r in rows],device=device)
    opt=torch.optim.AdamW(probe.parameters(),lr=lr,weight_decay=1e-4)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        la,lb=get_pred()
        loss=F.cross_entropy(la,a)+F.cross_entropy(lb,b)
        loss.backward(); opt.step()
        if s%100==0 or s==steps-1: poll()
    return float(loss.item())

@torch.no_grad()
def eval_probe(get_pred, rows, device):
    la,lb=get_pred()
    a=torch.tensor([r[1] for r in rows],device=device)
    b=torch.tensor([r[2] for r in rows],device=device)
    a_ok=int((la.argmax(-1)==a).sum()); b_ok=int((lb.argmax(-1)==b).sum())
    both=int(((la.argmax(-1)==a)&(lb.argmax(-1)==b)).sum())
    n=len(rows)
    return dict(a=f"{a_ok}/{n}", b=f"{b_ok}/{n}", both=f"{both}/{n}",
                both_pct=100.0*both/max(1,n))

def main():
    t0=time.time()
    quick=os.environ.get("INLINE_QUICK","0")=="1"
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
    hid=host.config.hidden_size
    print(f"  hidden={hid} RSS={rss_gb():.2f}GB")

    ntr = 400 if not quick else 60
    nte = 120 if not quick else 30
    steps = 800 if not quick else 120
    results={"host":MID,"dev":dev}

    # --- SPLIT 1: SAME operand-range, SAME templates, fresh operand DRAWS (in-dist
    #     generalization: unseen exact (a,b) pairs but from the trained value range)
    tr = build(ntr, 1, 0, 255, TEMPLATES)
    te_indist = build(nte, 7, 0, 255, TEMPLATES)          # unseen pairs, seen range
    # --- SPLIT 2: DISJOINT operand VALUE range (train 0..149, test 150..255) +
    #     unseen templates: the hard operand-value extrapolation
    tr2 = build(ntr, 3, 0, 149, TEMPLATES)
    te_ood = build(nte, 11, 150, 255, HELD_TEMPLATES)     # unseen range + unseen phrasing

    Htr,Mtr,Ltr = encode_full(host,tok,[r[0] for r in tr],dev)
    Hti,Mti,Lti = encode_full(host,tok,[r[0] for r in te_indist],dev)
    Htr2,Mtr2,Ltr2 = encode_full(host,tok,[r[0] for r in tr2],dev)
    Hood,Mood,Lood = encode_full(host,tok,[r[0] for r in te_ood],dev)

    def last_h(H,last): 
        bi=torch.arange(H.shape[0],device=H.device); return H[bi,last]

    print("\n=== (P1) FINAL-TOKEN readout ===")
    for name,(Ht,Mt,Lt,rowst),(He,Me,Le,rowse) in [
        ("in-dist (unseen pairs, seen range+templates)",(Htr,Mtr,Ltr,tr),(Hti,Mti,Lti,te_indist)),
        ("OOD (disjoint value range + unseen templates)",(Htr2,Mtr2,Ltr2,tr2),(Hood,Mood,Lood,te_ood)),
    ]:
        ft=FinalTokenReadout(hid).to(dev)
        gp_tr=lambda: ft(last_h(Ht,Lt))
        gp_te=lambda: ft(last_h(He,Le))
        train_probe(ft,gp_tr,rowst,dev,steps=steps)
        e_tr=eval_probe(gp_tr,rowst,dev); e_te=eval_probe(gp_te,rowse,dev)
        print(f"  [{name}] train both={e_tr['both']} ({e_tr['both_pct']:.1f}%)  "
              f"test both={e_te['both']} ({e_te['both_pct']:.1f}%)  a={e_te['a']} b={e_te['b']}")
        results[f"final_token::{name}"]=dict(train=e_tr,test=e_te)

    print("\n=== (P2) DIGIT-POINTER readout (attends to operand-digit tokens) ===")
    for name,(Ht,Mt,rowst),(He,Me,rowse) in [
        ("in-dist (unseen pairs, seen range+templates)",(Htr,Mtr,tr),(Hti,Mti,te_indist)),
        ("OOD (disjoint value range + unseen templates)",(Htr2,Mtr2,tr2),(Hood,Mood,te_ood)),
    ]:
        dp=DigitPointerReadout(hid).to(dev)
        gp_tr=lambda: dp(Ht,Mt)
        gp_te=lambda: dp(He,Me)
        train_probe(dp,gp_tr,rowst,dev,steps=steps)
        e_tr=eval_probe(gp_tr,rowst,dev); e_te=eval_probe(gp_te,rowse,dev)
        print(f"  [{name}] train both={e_tr['both']} ({e_tr['both_pct']:.1f}%)  "
              f"test both={e_te['both']} ({e_te['both_pct']:.1f}%)  a={e_te['a']} b={e_te['b']}")
        results[f"digit_pointer::{name}"]=dict(train=e_tr,test=e_te)

    results["peak_rss_gb"]=PEAK; results["elapsed_s"]=time.time()-t0
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"_agent_inline_operand_probe_results.json")
    with open(out,"w") as f: json.dump(results,f,indent=2,default=lambda o: float(o))
    print(f"\nWrote {out} (elapsed {time.time()-t0:.1f}s PEAK RSS {PEAK:.2f} GB)")

if __name__=="__main__":
    main()
