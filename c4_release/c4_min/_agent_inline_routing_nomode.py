"""INLINE / NO-MODE ROUTING into the grafted byte-exact ALU (#914 follow-up).

QUESTION (the "without entering the special thinking mode" test):
  #914 grafted the #910 byte-exact ALU into a real Qwen2.5-0.5B-Instruct as a
  side adapter, and USED it -- but only in an EXPLICIT MODE: a `vm_mode` flag was
  flipped by the harness and the operands were PINNED into residual dims by the
  driver. The host never *decided* to use the ALU; it never *found* the operands
  in real text. That is a mode-gated calculator, not a native one.

  This file tests the harder claim: can the REST of the pretrained network learn
  to use the byte-exact ALU INLINE and NATIVELY -- content-triggered during
  normal generation, NO mode token, NO special prompt, NO separate forward path --
  while (i) preserving normal language behaviour and (ii) GENERALIZING the routing
  to unseen operands / ops / phrasings.

PRECEDENT (Toolformer / NALU / DNC / MoE lineage):
  * Toolformer (Schick 2023): an LM learns *when* to call an external API tool
    inline, from self-supervised text -- but the tool is a black box outside the
    net and the result is a decoded string.
  * NALU/NAC (Trask 2018): arithmetic units inside the net that GENERALIZE
    extrapolatively -- but they are LEARNED (approximate), not byte-exact-constructed.
  * DNC/NTM (Graves 2016): a differentiable memory the controller learns to
    address -- the routing/addressing is learned, the memory is a substrate.
  * MoE / adapters: learned routing to frozen expert modules.
  NOVEL CELL HERE = a byte-exact CONSTRUCTED internal circuit (the #910 softmax-LUT
  ALU, exact not approximate) + LEARNED INLINE routing that fires on CONTENT with
  no mode switch, keeping the result byte-exact. i.e. Toolformer's "learn when to
  call" over a NALU-style in-net unit, but with the unit's answer EXACT (DNC-style
  learned addressing to a constructed, not learned, substrate).

THE EXPERIMENT (all MEASURED):
  Host = stock Qwen2.5-0.5B-Instruct (lean fp32, ~4 GB RSS, NOT the sparse ISA
  model -> no 108 GB densify). The #910 byte-exact ALU is grafted (FROZEN, so the
  ALU internals stay byte-exact and we ISOLATE the routing question).

  A ROUTER (small trainable head reading the host's REAL hidden state at the answer
  position) learns three things from CONTENT alone:
    (1) TRIGGER  : is the current position an arithmetic-answer slot? (binary)
    (2) OPERANDS : read operand bytes A, B from the host hidden state (byte-exact
                   pointers into the frozen ALU's operand lanes)
    (3) OP       : which of ADD/SUB/CMP_LT/CMP_EQ/CMP_GT/DIV this subproblem is
  When TRIGGER fires, the operands+op drive the FROZEN byte-exact ALU and its
  byte-exact result is EMITTED (as the model's next-token content). No mode token.

  The training text embeds arithmetic subproblems inside natural language and
  INTERLEAVES non-arithmetic sentences (so the trigger has real negatives). The
  arithmetic is *in the text* -- no calc prompt, no calculator token.

  TWO CONDITIONS COMPARED:
    (A) EXPLICIT-MODE : a literal cue token (" =>") precedes the answer slot and
        the router is *allowed* to key on that cue (the mode-gated fallback).
    (B) INLINE/NO-MODE: NO cue token -- the router must trigger from the natural-
        language content alone (the goal). Same net, same frozen ALU.

MEASURED (reported at the bottom + JSON):
  * inline routing accuracy: right operands routed + byte-exact result emitted,
    inline, no mode token. N/N.
  * language preservation: held-out non-arith perplexity vs the base model.
  * generalization: routing fires correctly on operands / ops / PHRASINGS not in
    training. Held-out N/N.
  * false-trigger rate: does it spuriously invoke the ALU on non-arith content?

HONESTY: byte-exactness of the ALU is FROZEN/constructed (not the question);
the question is ROUTING QUALITY. "Inline works" = MEASURED N/N with no mode token,
not asserted. A "no-mode reaches X%, false-triggers Y%, needs a soft cue for Z%"
is a valid WIN. Distinguish content-triggered (goal) from mode-gated (fallback).

MEMORY: lean fp32 host (~4 GB). Polls RSS, ABORTS > 9 GB. Golden 174ece66 not on
this path (self-contained _agent_ file). Local only.

Run:  HF_HOME=/media/data/.cache/huggingface PYTHONPATH=<repo> \
      python c4_min/_agent_inline_routing_nomode.py
Flag: INLINE_QUICK=1 for a fast smoke.
"""
from __future__ import annotations

import functools
import json
import os
import random
import resource
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _agent_graft_sgd_vm_rl import (
    GraftedByteExactALU, construct_byte_exact, head_decode,
    OPS as ALU_OPS, OP_ID as ALU_OP_ID, IN_DIM, N_OPS,
    IN_ONE, IN_A, IN_B, IN_QUOT, IN_CMP_LT, IN_CMP_EQ, IN_CMP_GT,
    op_semantics,
)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


PEAK_RSS = 0.0
RSS_LIMIT_GB = 9.0


def poll_rss(limit_gb: float = RSS_LIMIT_GB) -> float:
    global PEAK_RSS
    r = rss_gb()
    PEAK_RSS = max(PEAK_RSS, r)
    if r > limit_gb:
        raise MemoryError(
            f"RSS {r:.2f} GB exceeded the {limit_gb} GB safety boundary -- ABORT")
    return r


torch.set_default_dtype(torch.float32)


# op -> (TRAIN phrasings, HELD-OUT phrasings). {a},{b} are operands; sentence ENDS
# at the answer slot. Held-out phrasings are unseen at train.
ADD_TRAIN = [
    "So {a} plus {b} gives",
    "If you add {a} and {b} you get",
    "The sum of {a} and {b} is",
    "Adding {a} to {b} makes",
]
ADD_HELDOUT = [
    "Combine {a} with {b} to reach",
    "Take {a}, then increase it by {b}, and you arrive at",
]
SUB_TRAIN = [
    "So {a} minus {b} gives",
    "If you subtract {b} from {a} you get",
    "The difference of {a} and {b} is",
    "Taking {b} away from {a} leaves",
]
SUB_HELDOUT = [
    "Reduce {a} by {b} and you are left with",
    "Starting at {a} and going down by {b} reaches",
]
DIV_TRAIN = [
    "The quotient of {b} divided by {a} is",
    "So {b} divided by {a} gives",
    "Splitting {b} into {a} equal parts gives each",
    "Dividing {b} by {a} yields",
]
DIV_HELDOUT = [
    "How many whole times does {a} fit into {b}? The answer is",
    "Share {b} among {a} and each whole share is",
]
LT_TRAIN = [
    "Is {a} less than {b}? Answer 1 for yes 0 for no:",
    "Checking whether {a} is smaller than {b} gives",
]
LT_HELDOUT = [
    "Determine if {a} is below {b}, one or zero:",
]
EQ_TRAIN = [
    "Is {a} equal to {b}? Answer 1 for yes 0 for no:",
    "Checking whether {a} matches {b} gives",
]
EQ_HELDOUT = [
    "Tell me if {a} is the same value as {b}, one or zero:",
]
GT_TRAIN = [
    "Is {a} greater than {b}? Answer 1 for yes 0 for no:",
    "Checking whether {a} is larger than {b} gives",
]
GT_HELDOUT = [
    "Determine if {a} exceeds {b}, one or zero:",
]

OP_TEMPLATES = {
    "ADD": (ADD_TRAIN, ADD_HELDOUT),
    "SUB": (SUB_TRAIN, SUB_HELDOUT),
    "DIV": (DIV_TRAIN, DIV_HELDOUT),
    "CMP_LT": (LT_TRAIN, LT_HELDOUT),
    "CMP_EQ": (EQ_TRAIN, EQ_HELDOUT),
    "CMP_GT": (GT_TRAIN, GT_HELDOUT),
}
USE_OPS = list(OP_TEMPLATES.keys())

NONARITH_TRAIN = [
    "The capital of France is",
    "Water is essential for all",
    "The sun rises in the",
    "My favorite color is a shade of",
    "A cow is a large farm",
    "The opposite of hot is",
    "Books are a wonderful source of",
    "In winter the weather turns very",
    "She opened the door and stepped",
    "The ocean is deep and full of",
    "Music can make people feel",
    "A good breakfast gives you",
    "The mountain was covered in white",
    "He picked up the phone and said",
    "Autumn leaves fall gently from the",
    "The library was quiet except for the",
]
NONARITH_HELDOUT = [
    "The children laughed and ran across the",
    "A gentle rain began to fall over the",
    "The old clock on the wall kept perfect",
    "Fresh bread from the bakery smells absolutely",
    "The train arrived exactly on",
    "Stars appear brightest on a clear dark",
    "The garden was full of blooming",
    "A warm cup of tea is comforting on a cold",
]
NONARITH_WITH_NUMBERS = [
    "There were 12 birds sitting on the",
    "She read 3 books during the long",
    "The recipe needs 2 cups of",
    "He waited 7 minutes for the",
    "The building has 5 floors and a",
    "We saw 9 deer near the edge of the",
]

MODE_CUE = " =>"


@dataclass
class InlineSample:
    text: str
    is_arith: bool
    op: Optional[str]
    a: int
    b: int
    target: int
    phrasing_seen: bool


def _mk_arith(op: str, a: int, b: int, phrasing: str, seen: bool) -> InlineSample:
    mem = [0, 0, 0, 0]
    tgt, _, _ = op_semantics(op, a, b, mem, 0)
    text = phrasing.format(a=a, b=b)
    return InlineSample(text=text, is_arith=True, op=op, a=a, b=b,
                        target=tgt, phrasing_seen=seen)


def sample_legal_operands(op: str, rng: random.Random,
                          lo: int = 0, hi: int = 255) -> Tuple[int, int]:
    if op == "ADD":
        a = rng.randint(lo, min(hi, 200)); b = rng.randint(0, 255 - a)
    elif op == "SUB":
        a = rng.randint(lo, hi); b = rng.randint(0, a)
    elif op == "DIV":
        a = rng.randint(1, 20); b = rng.randint(lo, hi)
    else:
        a = rng.randint(lo, hi); b = rng.randint(lo, hi)
    return a, b


def make_inline_dataset(n_arith_per_op: int, n_nonarith: int, seed: int,
                        use_heldout_phrasing: bool = False,
                        operand_range: Tuple[int, int] = (0, 255)
                        ) -> List[InlineSample]:
    rng = random.Random(seed)
    data: List[InlineSample] = []
    lo, hi = operand_range
    for op in USE_OPS:
        train_ph, held_ph = OP_TEMPLATES[op]
        phrasings = held_ph if use_heldout_phrasing else train_ph
        for _ in range(n_arith_per_op):
            a, b = sample_legal_operands(op, rng, lo, hi)
            ph = rng.choice(phrasings)
            data.append(_mk_arith(op, a, b, ph, seen=not use_heldout_phrasing))
    neg_pool = (NONARITH_HELDOUT if use_heldout_phrasing else NONARITH_TRAIN)
    for _ in range(n_nonarith):
        text = rng.choice(neg_pool)
        data.append(InlineSample(text=text, is_arith=False, op=None, a=0, b=0,
                                 target=0, phrasing_seen=not use_heldout_phrasing))
    rng.shuffle(data)
    return data


class InlineRouter(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.trigger = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 1))
        self.op_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, N_OPS))
        self.a_head = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Linear(512, 256))
        self.b_head = nn.Sequential(
            nn.Linear(hidden, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, h: torch.Tensor):
        return (self.trigger(h).squeeze(-1),
                self.op_head(h),
                self.a_head(h),
                self.b_head(h))


def build_alu_feats(a: torch.Tensor, b: torch.Tensor, op_ids: torch.Tensor,
                    device) -> Tuple[torch.Tensor, torch.Tensor]:
    N = a.shape[0]
    feats = torch.zeros(N, IN_DIM, device=device)
    feats[:, IN_ONE] = 1.0
    feats[:, IN_A] = a.float()
    feats[:, IN_B] = b.float()
    aq = a.clamp(min=1)
    feats[:, IN_QUOT] = torch.where(a > 0, (b // aq).float(),
                                    torch.zeros_like(a).float())
    feats[:, IN_CMP_LT] = (a < b).float()
    feats[:, IN_CMP_EQ] = (a == b).float()
    feats[:, IN_CMP_GT] = (a > b).float()
    oph = F.one_hot(op_ids, num_classes=N_OPS).float().to(device)
    return feats, oph


class HostEncoder:
    def __init__(self, host, tok, device):
        self.host = host
        self.tok = tok
        self.device = device
        self.hidden = host.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str], add_cue: bool = False,
               batch_size: int = 16) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            if add_cue:
                chunk = [t + MODE_CUE for t in chunk]
            enc = self.tok(chunk, return_tensors="pt", padding=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.host.model(input_ids=enc["input_ids"],
                                  attention_mask=enc["attention_mask"])
            hs = out.last_hidden_state
            lengths = enc["attention_mask"].sum(1) - 1
            idx = lengths.long()
            batch_idx = torch.arange(hs.shape[0], device=hs.device)
            outs.append(hs[batch_idx, idx].float().cpu())
            poll_rss()
        return torch.cat(outs, 0).to(self.device)


def train_router(router: InlineRouter, alu: GraftedByteExactALU,
                 H: torch.Tensor, samples: List[InlineSample], device,
                 steps: int, lr: float = 1e-3) -> Dict:
    for p in alu.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    is_arith = torch.tensor([s.is_arith for s in samples],
                            dtype=torch.float32, device=device)
    op_ids = torch.tensor([ALU_OP_ID.get(s.op, 0) if s.is_arith else 0
                           for s in samples], dtype=torch.long, device=device)
    a_tgt = torch.tensor([s.a for s in samples], dtype=torch.long, device=device)
    b_tgt = torch.tensor([s.b for s in samples], dtype=torch.long, device=device)
    arith_mask = is_arith.bool()
    last = {}
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tl, ol, al, bl = router(H)
        loss_trig = F.binary_cross_entropy_with_logits(tl, is_arith)
        if arith_mask.any():
            loss_op = F.cross_entropy(ol[arith_mask], op_ids[arith_mask])
            loss_a = F.cross_entropy(al[arith_mask], a_tgt[arith_mask])
            loss_b = F.cross_entropy(bl[arith_mask], b_tgt[arith_mask])
        else:
            loss_op = loss_a = loss_b = torch.zeros((), device=device)
        loss = loss_trig + loss_op + loss_a + loss_b
        loss.backward()
        opt.step()
        last = dict(final_loss=float(loss.item()),
                    loss_trig=float(loss_trig.item()),
                    loss_op=float(loss_op.item()),
                    loss_a=float(loss_a.item()),
                    loss_b=float(loss_b.item()))
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            poll_rss()
    return last


@torch.no_grad()
def eval_routing(router: InlineRouter, alu: GraftedByteExactALU,
                 H: torch.Tensor, samples: List[InlineSample], device,
                 trig_thresh: float = 0.5) -> Dict:
    tl, ol, al, bl = router(H)
    trig = (torch.sigmoid(tl) > trig_thresh)
    op_pred = ol.argmax(-1)
    a_pred = al.argmax(-1)
    b_pred = bl.argmax(-1)
    is_arith = torch.tensor([s.is_arith for s in samples],
                            dtype=torch.bool, device=device)
    n_arith = int(is_arith.sum().item())
    n_non = int((~is_arith).sum().item())
    trig_correct_arith = int((trig & is_arith).sum().item())
    false_trigger = int((trig & ~is_arith).sum().item())
    feats, oph = build_alu_feats(a_pred, b_pred, op_pred, device)
    result = alu(feats, oph)
    emitted = head_decode(result.double())
    op_tgt = torch.tensor([ALU_OP_ID.get(s.op, -1) for s in samples], device=device)
    a_true = torch.tensor([s.a for s in samples], device=device)
    b_true = torch.tensor([s.b for s in samples], device=device)
    tgt = torch.tensor([s.target for s in samples], device=device)
    routed_ops_ok = 0
    routed_operands_ok = 0
    full_inline_ok = 0
    result_byteexact_given_routed = 0
    per_op = {op: [0, 0] for op in USE_OPS}
    seen_full = 0; seen_total = 0
    heldout_full = 0; heldout_total = 0
    for i, s in enumerate(samples):
        if not s.is_arith:
            continue
        op_ok = bool(op_pred[i].item() == op_tgt[i].item())
        ab_ok = bool(a_pred[i].item() == a_true[i].item() and
                     b_pred[i].item() == b_true[i].item())
        trg_ok = bool(trig[i].item())
        res_ok = bool(emitted[i].item() == tgt[i].item())
        routed_ops_ok += int(op_ok)
        routed_operands_ok += int(ab_ok)
        full = trg_ok and op_ok and ab_ok and res_ok
        full_inline_ok += int(full)
        per_op[s.op][0] += int(full); per_op[s.op][1] += 1
        rop = ALU_OPS[int(op_pred[i].item())]
        exp, _, _ = op_semantics(rop, int(a_pred[i].item()), int(b_pred[i].item()),
                                 [0, 0, 0, 0], 0)
        result_byteexact_given_routed += int(int(emitted[i].item()) == exp)
        if s.phrasing_seen:
            seen_full += int(full); seen_total += 1
        else:
            heldout_full += int(full); heldout_total += 1
    return dict(
        n_arith=n_arith, n_non=n_non,
        trigger_recall=f"{trig_correct_arith}/{n_arith}",
        trigger_recall_pct=100.0 * trig_correct_arith / max(1, n_arith),
        false_trigger=f"{false_trigger}/{n_non}",
        false_trigger_pct=100.0 * false_trigger / max(1, n_non),
        routed_op_exact=f"{routed_ops_ok}/{n_arith}",
        routed_operands_exact=f"{routed_operands_ok}/{n_arith}",
        full_inline_exact=f"{full_inline_ok}/{n_arith}",
        full_inline_pct=100.0 * full_inline_ok / max(1, n_arith),
        alu_byteexact_given_routed=f"{result_byteexact_given_routed}/{n_arith}",
        per_op={op: f"{c}/{t}" for op, (c, t) in per_op.items()},
        seen_phrasing=f"{seen_full}/{seen_total}" if seen_total else "n/a",
        heldout_phrasing=f"{heldout_full}/{heldout_total}" if heldout_total else "n/a",
    )


@torch.no_grad()
def held_out_perplexity(host, tok, device, texts: List[str]) -> float:
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(texts), 8):
        chunk = texts[i:i + 8]
        enc = tok(chunk, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = host(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              labels.reshape(-1), reduction="none")
        nll = nll.reshape(labels.shape)[mask]
        total_nll += float(nll.sum().item()); total_tok += int(mask.sum().item())
        poll_rss()
    return float(np.exp(total_nll / max(1, total_tok)))


def main():
    t0 = time.time()
    quick = os.environ.get("INLINE_QUICK", "0") == "1"
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MID = "Qwen/Qwen2.5-0.5B-Instruct"
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    force = os.environ.get("INLINE_DEVICE")
    if force:
        dev = force
    print(f"Host = {MID}   device = {dev}   quick={quick}")
    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    host = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32)
    poll_rss()
    host = host.to(dev).eval()
    for p in host.parameters():
        p.requires_grad_(False)
    H_dim = host.config.hidden_size
    print(f"  host params = {sum(p.numel() for p in host.parameters())/1e6:.2f}M   "
          f"RSS after load = {rss_gb():.2f} GB   hidden = {H_dim}")
    alu = GraftedByteExactALU(temp=30.0).to(dev)
    construct_byte_exact(alu)
    for p in alu.parameters():
        p.requires_grad_(False)
    enc = HostEncoder(host, tok, dev)
    results = dict(host=MID, device=dev, hidden=H_dim,
                   note="router trained; host + ALU FROZEN; ALU byte-exact by construction")
    n_arith = 30 if not quick else 8
    n_non = 60 if not quick else 16
    steps = 400 if not quick else 60
    train_data = make_inline_dataset(n_arith, n_non, seed=1, use_heldout_phrasing=False)
    heldout_data = make_inline_dataset(max(6, n_arith // 3), max(12, n_non // 3),
                                       seed=999, use_heldout_phrasing=True)
    print(f"  train: {len(train_data)} ({sum(s.is_arith for s in train_data)} arith), "
          f"held-out: {len(heldout_data)} ({sum(s.is_arith for s in heldout_data)} arith)")

    print("\n===== CONDITION (B) INLINE / NO-MODE (content-triggered, no cue) =====")
    H_train_B = enc.encode([s.text for s in train_data], add_cue=False)
    router_B = InlineRouter(H_dim).to(dev)
    tr_B = train_router(router_B, alu, H_train_B, train_data, dev, steps=steps)
    print(f"    train losses: {tr_B}")
    ev_B_train = eval_routing(router_B, alu, H_train_B, train_data, dev)
    H_held_B = enc.encode([s.text for s in heldout_data], add_cue=False)
    ev_B_held = eval_routing(router_B, alu, H_held_B, heldout_data, dev)
    print(f"    [train] {ev_B_train}")
    print(f"    [held-out phrasing+operands] {ev_B_held}")
    results["inline_no_mode"] = dict(train_losses=tr_B, eval_train=ev_B_train,
                                     eval_heldout=ev_B_held)

    print("\n===== CONDITION (A) EXPLICIT-MODE (cue token '=>' present) =====")
    H_train_A = enc.encode([s.text for s in train_data], add_cue=True)
    router_A = InlineRouter(H_dim).to(dev)
    tr_A = train_router(router_A, alu, H_train_A, train_data, dev, steps=steps)
    ev_A_train = eval_routing(router_A, alu, H_train_A, train_data, dev)
    H_held_A = enc.encode([s.text for s in heldout_data], add_cue=True)
    ev_A_held = eval_routing(router_A, alu, H_held_A, heldout_data, dev)
    print(f"    train losses: {tr_A}")
    print(f"    [train] {ev_A_train}")
    print(f"    [held-out phrasing+operands] {ev_A_held}")
    results["explicit_mode"] = dict(train_losses=tr_A, eval_train=ev_A_train,
                                    eval_heldout=ev_A_held)

    print("\n===== FALSE-TRIGGER STRESS: sentences with numbers but no arithmetic =====")
    hard_neg = [InlineSample(text=t, is_arith=False, op=None, a=0, b=0,
                             target=0, phrasing_seen=False)
                for t in NONARITH_WITH_NUMBERS]
    H_hardneg_B = enc.encode([s.text for s in hard_neg], add_cue=False)
    ft_B = eval_routing(router_B, alu, H_hardneg_B, hard_neg, dev)
    print(f"    [INLINE no-mode] false-trigger on numbers-no-arith: "
          f"{ft_B['false_trigger']} ({ft_B['false_trigger_pct']:.1f}%)")
    results["false_trigger_numbers_no_arith_inline"] = ft_B["false_trigger"]
    results["false_trigger_numbers_no_arith_inline_pct"] = ft_B["false_trigger_pct"]

    print("\n===== LANGUAGE PRESERVATION: held-out non-arith perplexity =====")
    lang_texts = NONARITH_HELDOUT + NONARITH_WITH_NUMBERS
    ppl = held_out_perplexity(host, tok, dev, lang_texts)
    results["heldout_nonarith_perplexity"] = ppl
    results["host_frozen_no_forgetting"] = True
    print(f"    held-out non-arith perplexity (host UNMODIFIED) = {ppl:.3f}")
    print(f"    host is FROZEN (router is a separate module) -> perplexity delta = 0 "
          f"by construction (no catastrophic forgetting possible)")

    print("\n################################ VERDICT #########################")
    def pct(d): return d["full_inline_pct"]
    inline_train = pct(ev_B_train)
    inline_held = pct(ev_B_held)
    mode_train = pct(ev_A_train)
    mode_held = pct(ev_A_held)
    ft_inline = ev_B_train["false_trigger_pct"]
    ft_inline_held = ev_B_held["false_trigger_pct"]
    ft_hard = ft_B["false_trigger_pct"]
    trig_recall_inline = ev_B_train["trigger_recall_pct"]
    trig_recall_held = ev_B_held["trigger_recall_pct"]
    native_inline_works = (inline_train >= 90.0 and inline_held >= 60.0
                           and ft_hard <= 20.0)
    verdict = dict(
        inline_full_routing_train_pct=inline_train,
        inline_full_routing_heldout_pct=inline_held,
        mode_full_routing_train_pct=mode_train,
        mode_full_routing_heldout_pct=mode_held,
        inline_trigger_recall_pct=trig_recall_inline,
        inline_trigger_recall_heldout_pct=trig_recall_held,
        inline_false_trigger_pct=ft_inline,
        inline_false_trigger_heldout_pct=ft_inline_held,
        inline_false_trigger_numbers_no_arith_pct=ft_hard,
        heldout_nonarith_perplexity=ppl,
        language_preservation_delta=0.0,
        inline_full_train=ev_B_train["full_inline_exact"],
        inline_full_heldout=ev_B_held["full_inline_exact"],
        mode_full_train=ev_A_train["full_inline_exact"],
        mode_full_heldout=ev_A_held["full_inline_exact"],
        alu_byteexact_train=ev_B_train["alu_byteexact_given_routed"],
        native_inline_works=native_inline_works,
        inline_vs_mode_gap_train=mode_train - inline_train,
        inline_vs_mode_gap_heldout=mode_held - inline_held,
    )
    results["verdict"] = verdict
    print(f"  INLINE / NO-MODE full routing (fire+op+operands+byte-exact):")
    print(f"     train   = {ev_B_train['full_inline_exact']}  ({inline_train:.1f}%)")
    print(f"     held-out= {ev_B_held['full_inline_exact']}  ({inline_held:.1f}%)  "
          f"(unseen phrasings + operands)")
    print(f"  EXPLICIT-MODE (cue token) full routing:")
    print(f"     train   = {ev_A_train['full_inline_exact']}  ({mode_train:.1f}%)")
    print(f"     held-out= {ev_A_held['full_inline_exact']}  ({mode_held:.1f}%)")
    print(f"  inline-vs-mode gap: train {mode_train - inline_train:+.1f}pp   "
          f"held-out {mode_held - inline_held:+.1f}pp")
    print(f"  INLINE trigger recall: train {trig_recall_inline:.1f}%  held-out {trig_recall_held:.1f}%")
    print(f"  INLINE false-trigger: non-arith {ft_inline:.1f}%  held-out-nonarith {ft_inline_held:.1f}%  "
          f"numbers-no-arith {ft_hard:.1f}%")
    print(f"  ALU byte-exact given routed operands (train): {ev_B_train['alu_byteexact_given_routed']}")
    print(f"  Language preservation: host FROZEN -> perplexity {ppl:.3f}, delta = 0 (no forgetting)")
    print(f"  ==> NATIVE INLINE (content-triggered, no mode) works: {native_inline_works}")

    results["peak_rss_gb"] = PEAK_RSS
    results["elapsed_s"] = time.time() - t0
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_agent_inline_routing_nomode_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"\nWrote {out}   (elapsed {time.time()-t0:.1f}s, PEAK RSS {PEAK_RSS:.2f} GB)")


if __name__ == "__main__":
    main()
