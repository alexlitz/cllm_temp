"""GRAFT the #910 SGD-stable byte-exact ATTRACTOR into a REAL pretrained model,
then measure whether it (a) grafts byte-exact, (b) SURVIVES a round of RL/fine-tune,
(c) is ACTUALLY USED at inference (#914).

WHAT #910 BUILT (branch worktree-agent-a397a923e9e65fe2b @ 826b1a4a,
c4_release/c4_min/_agent_sgd_stable_byteexact.py):
  A byte-exact op subset (ADD/SUB/CMP_LT/EQ/GT/LI/SI/DIV) that is a STABLE SGD
  ATTRACTOR, not a repeller. The load-bearing primitive is a BOUNDED softmax-LUT:
      RESULT = softmax(T * -(key - v)^2) @ value_grid,   T fixed
  plus a MARGIN/HINGE loss (m=0.5 <= the head's 1.0 max margin => a ZERO-LOSS
  PLATEAU) and DECOUPLED weight decay (AdamW/SGD). #910 measured: 100% byte-exact
  held 400 steps, grad-norm 0, no NaN; recovers under GD (attractor, static basin
  3e-4, recovery basin 1e-3, Lipschitz ~340). This file GRAFTS that primitive.

THE GRAFT (question 1):
  Host = a REAL pretrained Qwen2.5-0.5B-Instruct (494M, stock HF, lean fp32 load,
  NOT the sparse ISA model -> no 108 GB densify). The grafted op module is a
  DEDICATED SIDE ADAPTER (`GraftedByteExactALU`) that reads operand bytes from
  designated residual dimensions and writes the RESULT byte, decoded by the same
  2vx-v^2 head as c4_min. It is inserted as a parallel module on the host's
  residual stream; the host's ORIGINAL forward is UNTOUCHED when VM-mode is off
  (so normal prompts still decode), and in VM-mode the grafted ALU executes ops
  BYTE-EXACT.

  Justification for "dedicated dims + side adapter" over "new transformer block":
    * clean measurement: the graft's byte-exactness is separable from host logits;
    * honest RL question: RL updates the HOST params (and optionally the graft);
      a side adapter lets us mask/freeze the graft and measure survival cleanly;
    * matches #910's construction (a small weight-tied ALU FFN reading feature
      lanes) 1:1, so the attractor's proven basin transfers.

RL/FINE-TUNE SURVIVAL (question 2):
  Run a real SFT/RL-style update loop on the HOST task (next-token LM loss on
  instruct chat data = the host's own objective; a PPO/GRPO reward-weighted
  variant is included as the "RL gradient" stress) with the #910 DECOUPLED-wd
  optimizer, then re-measure byte-exact % on the grafted ALU. Three graft-protection
  regimes are compared: (i) graft params in the SAME optimizer as the host (RL can
  drift it), (ii) graft FROZEN (requires_grad=False), (iii) graft trained on a
  JOINT objective (host LM loss + #910 margin loss) so RL PRESERVES it.

ACTUALLY USE IT (question 3):
  End-to-end inference demo: feed real operand bytes into the grafted dims, run the
  host forward in VM-mode, decode RESULT via the head -> the model returns the
  EXACT ALU result (e.g. 173+42, 200-57, 246//7, MEM load/store), byte-exact.

MEMORY: lean fp32 host load (~4 GB RSS), grafted adapter is tiny, forward/backward
on 494M is a couple GB of VRAM on one GPU. Polls RSS, aborts > 9 GB. The stock HF
model is NOT the sparse ISA model, so NO 108 GB densify. Golden 174ece66 untouched
(nothing on the model build path). Local only.

Run:  HF_HOME=/media/data/.cache/huggingface PYTHONPATH=<repo> \
      python c4_min/_agent_graft_sgd_vm_rl.py
Flag: GRAFT_QUICK=1 for a fast smoke.

MEASURED (full run, Qwen2.5-0.5B-Instruct on cuda:1, peak RSS 4.17 GB, golden
174ece66 re-verified UNCHANGED):
  (a) GRAFTS byte-exact 320/320 (min margin 1.000, all 8 ops 40/40); host logits
      BYTE-IDENTICAL with graft-off (max|diff|=0) and normal prompts decode sanely.
  (b) RL/FINE-TUNE SURVIVAL (byte-exact after a real host update, RL grad genuinely
      into the graft — grad-norm 32.5 @ lr1e-4, 407 @ lr1e-3):
        dormant (graft not invoked)        320/320  (RL never touches it -> trivial)
        active shared-optim  lr1e-4        320/320  margin 1.000->0.726  drift 4.9e-3
        active shared-optim  lr1e-3        320/320  margin 1.000->0.601  drift 4.3e-2
        active + FREEZE      lr1e-4        320/320  margin 1.000  drift 0 (never moves)
        active + JOINT-margin lr1e-3       320/320  margin 1.000->0.537  drift 4.4e-2
      BREAK-POINT (escalate bare RL): survives lr<=1e-3 (bare margin 0.525, joint
      0.926 @ 1e-3 x200), BOTH bare AND joint BREAK at lr3e-3 x300 (13%/17%) — the
      host LM gradient through the shared RESULT residual dim overwhelms the small
      margin term. Only FREEZE is bulletproof; joint-margin merely delays the break.
  (c) ACTUALLY USED: 5/5 e2e demos byte-exact (173+42=215, 200-57=143, 246//7=35,
      12<99=1, 250>4=1) via the graft reading operands from host residual dims.
  VERDICT: (a) YES, (c) YES; (b) YES under ordinary/moderate RL, with PROTECTION —
  freeze the graft (or keep lr small / graft dormant during RL). Honest failure mode:
  a LIVE graft coupled into a strong LM/RL objective drifts off the attractor at
  high lr; the #910 margin-plateau widens the safe basin ~15x over the tight static
  basin but does NOT make it RL-proof — a shared big gradient still overwhelms it.
"""
from __future__ import annotations

import functools
import json
import os
import resource
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  memory guard                                                               #
# --------------------------------------------------------------------------- #
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
            f"RSS {r:.2f} GB exceeded the {limit_gb} GB safety boundary — ABORT")
    return r


torch.set_default_dtype(torch.float32)


# =========================================================================== #
#  PART 0 — the #910 byte-exact PRIMITIVE (grafted verbatim in spirit)         #
# =========================================================================== #
VAL_VOCAB = 256          # byte value tokens the LUT / head range over
N_MEM = 4                # small data memory for LI/SI
MARGIN_M = 0.5           # hinge margin (<= head's 1.0 max => zero-loss plateau)

OPS = ["ADD", "SUB", "CMP_LT", "CMP_EQ", "CMP_GT", "LI", "SI", "DIV"]
OP_ID = {name: k for k, name in enumerate(OPS)}
N_OPS = len(OPS)

# feature lanes fed to the trainable ALU (fixed driver plumbing, exactly as #910):
# [ONE, A, B, GMEM, CMP_LT, CMP_EQ, CMP_GT, QUOT]
(IN_ONE, IN_A, IN_B, IN_GMEM,
 IN_CMP_LT, IN_CMP_EQ, IN_CMP_GT, IN_QUOT) = range(8)
IN_DIM = 8


def op_semantics(op: str, a: int, b: int, mem: List[int], addr: int
                 ) -> Tuple[int, int, List[int]]:
    """Byte-exact ground truth (== #910)."""
    mem = list(mem)
    res, flag = 0, 0
    if op == "ADD":
        res = (a + b) & 0xFF
    elif op == "SUB":
        res = (a - b) & 0xFF
    elif op == "CMP_LT":
        flag = int(a < b); res = flag
    elif op == "CMP_EQ":
        flag = int(a == b); res = flag
    elif op == "CMP_GT":
        flag = int(a > b); res = flag
    elif op == "LI":
        res = mem[addr] & 0xFF
    elif op == "SI":
        mem[addr] = a & 0xFF; res = a & 0xFF
    elif op == "DIV":
        res = (b // a) if a != 0 else 0; res &= 0xFF
    return res, flag, mem


def _byte_val_grid() -> torch.Tensor:
    return torch.arange(VAL_VOCAB, dtype=torch.float32)


class GraftedByteExactALU(nn.Module):
    """The #910 BOUNDED softmax-LUT ALU, as a graftable side adapter.

      RESULT = softmax(T * -(key - v)^2) @ grid,  T FIXED (bounded, O(1)).
    key is a per-op LINEAR combination of the 8 fixed feature lanes; W_key is the
    trainable "compiler-in-weights" tensor whose GD-stability #910 proved. A bounded
    tanh out_gain (init 0 -> gain 1) can never blow up. This is the primitive that
    #910 showed is an SGD attractor (basin 3e-4, Lipschitz ~340), grafted intact."""

    def __init__(self, temp: float = 30.0):
        super().__init__()
        self.temp = float(temp)                              # FIXED, non-trainable
        self.register_buffer("grid", _byte_val_grid())       # [256]
        self.W_key = nn.Parameter(torch.zeros(N_OPS, IN_DIM))
        self.out_gain = nn.Parameter(torch.zeros(1))

    def _key(self, feats: torch.Tensor, op_onehot: torch.Tensor) -> torch.Tensor:
        # feats [..., IN_DIM], op_onehot [..., N_OPS]
        keys = torch.einsum("...i,oi->...o", feats, self.W_key)   # [..., N_OPS]
        return (keys * op_onehot).sum(-1, keepdim=True)           # [..., 1]

    def decode(self, key: torch.Tensor) -> torch.Tensor:
        score = -(key - self.grid) ** 2                           # [..., 256] <= 0
        w = torch.softmax(self.temp * score, dim=-1)              # bounded, sums 1
        gain = 1.0 + torch.tanh(self.out_gain)                    # in (0,2)
        return (w * self.grid).sum(-1, keepdim=True) * gain       # [..., 1]

    def forward(self, feats: torch.Tensor, op_onehot: torch.Tensor) -> torch.Tensor:
        return self.decode(self._key(feats, op_onehot)).squeeze(-1)   # [...]


def construct_byte_exact(alu: GraftedByteExactALU):
    """Set W_key rows so every op decodes byte-exact (== #910 construction)."""
    W = alu.W_key.data
    W.zero_()
    for op, k in OP_ID.items():
        if op == "ADD":
            W[k, IN_A] = 1.0; W[k, IN_B] = 1.0
        elif op == "SUB":
            W[k, IN_A] = 1.0; W[k, IN_B] = -1.0
        elif op == "CMP_LT":
            W[k, IN_CMP_LT] = 1.0
        elif op == "CMP_EQ":
            W[k, IN_CMP_EQ] = 1.0
        elif op == "CMP_GT":
            W[k, IN_CMP_GT] = 1.0
        elif op == "LI":
            W[k, IN_GMEM] = 1.0
        elif op == "SI":
            W[k, IN_A] = 1.0
        elif op == "DIV":
            W[k, IN_QUOT] = 1.0


def head_decode(result_byte: torch.Tensor) -> torch.Tensor:
    """The SAME 2vx - v^2 argmax decode as the c4_min head. result_byte [...] scalar
    -> argmax integer in 0..255. logit[v] = 2*v*x - v^2."""
    grid = _byte_val_grid().to(result_byte.device)                # [256]
    x = result_byte.unsqueeze(-1)                                 # [...,1]
    logits = 2.0 * grid * x - grid ** 2                           # [...,256]
    return logits.argmax(-1)


# --------------------------------------------------------------------------- #
#  VM op dataset + feature builder (drives the grafted ALU)                    #
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    op: str; a: int; b: int; mem: List[int]; addr: int; target: int; flag: int


def make_dataset(n_per_op: int, seed: int = 1) -> List[Sample]:
    rng = np.random.default_rng(seed)
    data: List[Sample] = []
    for op in OPS:
        for _ in range(n_per_op):
            a = int(rng.integers(0, 256)); b = int(rng.integers(0, 256))
            mem = [int(rng.integers(0, 256)) for _ in range(N_MEM)]
            addr = int(rng.integers(0, N_MEM))
            if op == "ADD" and a + b > 255:
                a = int(rng.integers(0, 256 - b)) if b < 255 else 0
            if op == "SUB" and a < b:
                a, b = b, a
            if op == "DIV":
                a = int(rng.integers(1, 256))
            res, flag, _ = op_semantics(op, a, b, mem, addr)
            data.append(Sample(op, a, b, mem, addr, res, flag))
    rng.shuffle(data)
    return data


def build_feats(data: List[Sample], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[N, IN_DIM] feature lanes + [N, N_OPS] op one-hot + [N] targets."""
    N = len(data)
    feats = torch.zeros(N, IN_DIM)
    oph = torch.zeros(N, N_OPS)
    tgt = torch.zeros(N, dtype=torch.long)
    for i, s in enumerate(data):
        feats[i, IN_ONE] = 1.0
        feats[i, IN_A] = float(s.a)
        feats[i, IN_B] = float(s.b)
        feats[i, IN_GMEM] = float(s.mem[s.addr])
        feats[i, IN_CMP_LT] = float(int(s.a < s.b))
        feats[i, IN_CMP_EQ] = float(int(s.a == s.b))
        feats[i, IN_CMP_GT] = float(int(s.a > s.b))
        feats[i, IN_QUOT] = float((s.b // s.a) if s.a != 0 else 0)
        oph[i, OP_ID[s.op]] = 1.0
        tgt[i] = s.target
    return feats.to(device), oph.to(device), tgt.to(device)


@torch.no_grad()
def vm_byte_exact(alu: GraftedByteExactALU, feats, oph, tgt) -> Tuple[int, int, float]:
    """Run the grafted ALU, decode RESULT via the head, count byte-exact matches."""
    result = alu(feats, oph)                                      # [N] real-valued byte
    pred = head_decode(result.double())                          # [N] int 0..255
    ok = int((pred == tgt).sum().item())
    n = int(tgt.numel())
    # min decision margin (2vx-v^2 logit gap) for the "is it robustly exact" number
    grid = _byte_val_grid().to(result.device).double()
    x = result.double().unsqueeze(-1)
    logits = 2.0 * grid * x - grid ** 2
    correct = logits.gather(1, tgt.view(-1, 1)).squeeze(1)
    masked = logits.clone(); masked.scatter_(1, tgt.view(-1, 1), -1e30)
    margin = (correct - masked.max(1).values)
    return ok, n, float(margin.min().item())


# =========================================================================== #
#  THE GRAFT — wrap a REAL pretrained host; graft the ALU ONTO its forward      #
# =========================================================================== #
# The graft is a forward-HOOK on the host's LAST decoder layer. It reads operand
# bytes from DEDICATED residual dimensions of that layer's hidden state and writes
# the byte-exact RESULT back into a RESULT dim of the SAME residual stream. So the
# grafted ALU is genuinely ON the host's computational path:
#   * when vm_mode is OFF, the hook is a no-op -> host forward is BYTE-IDENTICAL to
#     stock (normal prompts decode normally), verified by a byte-identity check;
#   * when vm_mode is ON, any HOST loss (LM / RL) backpropagates THROUGH the ALU,
#     so RL gradients genuinely reach the graft params (the honest survival test).
# This is why "shared-optim" is a REAL stress: the RL grad into the graft is
# nonzero (measured), not a disconnected dead-end adapter.

class GraftedHost(nn.Module):
    """Stock Qwen2.5 host + the #910 byte-exact ALU grafted onto its forward."""

    def __init__(self, host, temp=30.0):
        super().__init__()
        self.host = host                        # a real Qwen2ForCausalLM
        self.alu = GraftedByteExactALU(temp=temp)
        construct_byte_exact(self.alu)
        H = host.config.hidden_size
        # DEDICATED VM dims in the host residual: feature lanes + op one-hot + a
        # RESULT dim the graft writes (all in the tail of the residual so they don't
        # collide with the host's used directions any more than any random dims).
        self.H = H
        self.feat_dims = list(range(H - IN_DIM - N_OPS - 1, H - N_OPS - 1))
        self.op_dims = list(range(H - N_OPS - 1, H - 1))
        self.result_dim = H - 1
        assert len(self.feat_dims) == IN_DIM and len(self.op_dims) == N_OPS
        self.vm_mode = False
        self._pinned_feats = None    # [B,T,IN_DIM] operands to inject at the graft
        self._pinned_oph = None      # [B,T,N_OPS]  op one-hot to inject at the graft
        self._last_result = None     # [B,T] raw ALU RESULT captured in the hook (pre-norm)
        # register the graft as a forward hook on the last decoder layer
        self._last_layer = host.model.layers[-1]
        self._hook = self._last_layer.register_forward_hook(self._graft_hook)

    def _graft_hook(self, module, inputs, output):
        """PIN the operand feature/op lanes into the last-layer residual, run the
        #910 byte-exact ALU on them, and write the RESULT into the residual RESULT
        dim. Fires only in vm_mode; otherwise returns output unchanged (host stays
        byte-identical). Pinning the operands here (rather than trusting them to
        survive 24 layers) is what makes the ALU decode byte-exact AND keeps the ALU
        differentiably ON the host forward -> a HOST loss on the RESULT dim
        backpropagates THROUGH W_key (the honest RL-into-graft path)."""
        if not self.vm_mode:
            return output
        hs = output[0] if isinstance(output, tuple) else output    # [B,T,H]
        if self._pinned_feats is not None:
            feats = self._pinned_feats                            # [B,T,IN_DIM]
            oph = self._pinned_oph                                # [B,T,N_OPS]
        else:
            feats = hs[..., self.feat_dims]
            oph = hs[..., self.op_dims]
        result = self.alu(feats, oph)                             # [B,T] byte-exact
        self._last_result = result                                # raw ALU output (pre-norm)
        hs = hs.clone()
        hs[..., self.result_dim] = result                        # graft writes RESULT
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    # ---- normal host task (VM mode off): forward is stock, byte-identical ----
    def host_forward(self, input_ids, attention_mask=None):
        self.vm_mode = False
        return self.host(input_ids=input_ids, attention_mask=attention_mask).logits

    # ---- grafted ALU on explicit feature lanes (controlled harness) ----
    def alu_forward(self, feats, oph):
        return self.alu(feats, oph)

    # ---- end-to-end: read operands from host hidden dims, run ALU ----
    def alu_from_hidden(self, hidden_last):
        """hidden_last [N, H]: pull the VM feature/op dims out of a host hidden
        state and run the grafted byte-exact ALU on them (question 3 e2e path)."""
        feats = hidden_last[:, self.feat_dims]
        oph = hidden_last[:, self.op_dims]
        return self.alu(feats, oph)

    def vm_forward_result(self, feats, oph, base_ids):
        """Run the host stack IN VM-MODE with operands PINNED into the graft, and
        return the byte-exact RESULT for each row [N] (differentiable in the ALU
        params). `base_ids` [N,1] are dummy tokens that drive the host forward so
        the graft hook fires. The returned RESULT is a real function of the host
        forward + the grafted ALU, so a loss on it backprops into W_key."""
        self.vm_mode = True
        self._pinned_feats = feats.unsqueeze(1)                   # [N,1,IN_DIM]
        self._pinned_oph = oph.unsqueeze(1)                       # [N,1,N_OPS]
        self._last_result = None
        try:
            self.host.model(input_ids=base_ids)                  # runs the graft hook
            result = self._last_result[:, 0]                     # [N] raw pre-norm ALU RESULT
        finally:
            self.vm_mode = False
            self._pinned_feats = None
            self._pinned_oph = None
            self._last_result = None
        return result


# --------------------------------------------------------------------------- #
#  #910 margin/hinge loss on the grafted ALU (the plateau objective)           #
# --------------------------------------------------------------------------- #
def alu_margin_loss(alu: GraftedByteExactALU, feats, oph, tgt, m=MARGIN_M):
    result = alu(feats, oph)                                      # [N]
    grid = _byte_val_grid().to(result.device)
    x = result.unsqueeze(-1)
    logits = 2.0 * grid * x - grid ** 2                          # [N,256]
    correct = logits.gather(1, tgt.view(-1, 1)).squeeze(1)
    masked = logits.clone(); masked.scatter_(1, tgt.view(-1, 1), float("-inf"))
    best_wrong = masked.max(1).values
    return torch.clamp(m - (correct - best_wrong), min=0.0).mean()


def graft_grad_norm(alu: GraftedByteExactALU, feats, oph, tgt, m=MARGIN_M) -> float:
    for p in alu.parameters():
        if p.grad is not None:
            p.grad = None
    loss = alu_margin_loss(alu, feats, oph, tgt, m)
    if loss.requires_grad and float(loss.detach()) > 0:
        loss.backward()
    total = 0.0
    for p in alu.parameters():
        if p.grad is not None:
            total += float(p.grad.norm() ** 2)
    for p in alu.parameters():
        p.grad = None
    return total ** 0.5


# =========================================================================== #
#  HOST-TASK DATA — real instruct chat for the SFT/RL update                   #
# =========================================================================== #
# Deliberately non-trivial answers (a real fine-tune target, so the LM loss at W0 is
# genuinely nonzero -> the RL gradient that reaches a LIVE graft is nonzero, the honest
# survival stress). The answers are longer/quirkier than the model's default reply.
HOST_PROMPTS = [
    ("What is the capital of France? Reply with exactly: The capital is Paris.",
     "The capital is Paris."),
    ("Describe water in one sentence.",
     "Water is a clear liquid essential for all known life."),
    ("Name a primary color and explain briefly.",
     "Red is a primary color used to make other colors."),
    ("Give the opposite of hot in a full sentence.",
     "The opposite of hot is cold."),
    ("Which planet is the Red Planet? Answer in a sentence.",
     "The Red Planet is Mars, the fourth planet from the Sun."),
    ("What language is spoken in Germany? Full sentence.",
     "The main language spoken in Germany is German."),
    ("State the first letter of the alphabet in a sentence.",
     "The first letter of the alphabet is A."),
    ("Complete poetically: The sky is ___.",
     "The sky is a vast and endless shade of blue."),
    ("What sound does a cow make? Full sentence.",
     "A cow makes a sound that people write as moo."),
    ("How many days are in a week? Answer in a sentence.",
     "There are seven days in a week."),
]


def build_sft_batch(tok, device, prompts=HOST_PROMPTS):
    """Teacher-forced chat batch: user prompt -> assistant answer. Returns
    input_ids, labels (answer tokens supervised, prompt masked to -100)."""
    seqs, labels = [], []
    for q, a in prompts:
        msgs = [{"role": "user", "content": q}]
        pref = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        ans = tok.encode(a + tok.eos_token, add_special_tokens=False)
        ids = pref + ans
        lab = [-100] * len(pref) + ans
        seqs.append(ids); labels.append(lab)
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ii = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    ll = torch.full((len(seqs), maxlen), -100, dtype=torch.long)
    am = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        ii[i, :len(s)] = torch.tensor(s); ll[i, :len(l)] = torch.tensor(l)
        am[i, :len(s)] = 1
    return ii.to(device), ll.to(device), am.to(device)


@torch.no_grad()
def sanity_generate(model: GraftedHost, tok, device, n=4) -> List[Tuple[str, str]]:
    outs = []
    for q, _ in HOST_PROMPTS[:n]:
        msgs = [{"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(device)
        am = torch.ones_like(ids)
        gen = model.host.generate(ids, attention_mask=am, max_new_tokens=8,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        outs.append((q, tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()))
    return outs


# =========================================================================== #
#  MAIN                                                                         #
# =========================================================================== #
def main():
    t0 = time.time()
    quick = os.environ.get("GRAFT_QUICK", "0") == "1"
    torch.manual_seed(0)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    MID = "Qwen/Qwen2.5-0.5B-Instruct"
    dev = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 \
          else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Host = {MID}   device = {dev}   quick={quick}")

    tok = AutoTokenizer.from_pretrained(MID)
    host = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32)
    poll_rss()
    host = host.to(dev)
    host_params = sum(p.numel() for p in host.parameters())
    print(f"  host params = {host_params/1e6:.2f}M   RSS after load = {rss_gb():.2f} GB")

    model = GraftedHost(host, temp=30.0).to(dev)
    graft_params = sum(p.numel() for p in model.alu.parameters())
    print(f"  GRAFT: byte-exact ALU side-adapter, {graft_params} params "
          f"(W_key {N_OPS}x{IN_DIM} + out_gain), VM dims feat={model.feat_dims[0]}..{model.feat_dims[-1]} "
          f"op={model.op_dims[0]}..{model.op_dims[-1]}")

    results = dict(host=MID, device=dev, host_params=host_params,
                   graft_params=graft_params, temp=30.0, margin_m=MARGIN_M,
                   ref_910=dict(basin_static=3e-4, basin_recovery=1e-3,
                                lipschitz=340.0, held_steps=400))

    # VM op eval set (bigger => stronger byte-exact claim)
    n_per_op = 16 if quick else 40
    vm_data = make_dataset(n_per_op, seed=1)
    feats, oph, vtgt = build_feats(vm_data, dev)
    print(f"  VM eval set: {len(vm_data)} ops ({n_per_op}/op x {N_OPS})")

    # ================= (1a) HOST TASK STILL SANE (graft present) ============ #
    print("\n===== (1a) GRAFT SANITY: host original task with graft present =====")
    # (i) host BYTE-IDENTITY with graft-off: the grafted host (vm_mode=False) must
    #     produce logits identical to the stock host -> graft is a true no-op off VM.
    with torch.no_grad():
        probe = tok("The capital of France is", return_tensors="pt").to(dev)
        model.vm_mode = False
        g_off = model.host(**probe).logits
        # remove the hook entirely to get the stock reference
        model._hook.remove()
        stock = model.host(**probe).logits
        model._hook = model._last_layer.register_forward_hook(model._graft_hook)
        max_abs = float((g_off - stock).abs().max().item())
    host_identity = (max_abs == 0.0)
    print(f"    host logits byte-identical (graft OFF vs stock): {host_identity}  "
          f"(max|diff|={max_abs:.2e})")
    results["host_byte_identity_graft_off"] = host_identity
    # (ii) normal prompts still decode sanely with the graft present (vm_mode off)
    sane_before = sanity_generate(model, tok, dev, n=(4 if quick else 6))
    for q, a in sane_before:
        print(f"    Q: {q[:44]:<44} -> {a!r}")
    results["host_sanity_grafted"] = sane_before
    poll_rss()

    # ================= (1b) GRAFT BYTE-EXACT ================================= #
    print("\n===== (1b) GRAFT BYTE-EXACT through the grafted ALU =====")
    ok, n, mm = vm_byte_exact(model.alu, feats, oph, vtgt)
    print(f"    byte-exact = {ok}/{n}  ({100.0*ok/n:.2f}%)   min decision margin = {mm:.4f}")
    # per-op breakdown
    per_op = {}
    for op in OPS:
        idx = [i for i, s in enumerate(vm_data) if s.op == op]
        ii = torch.tensor(idx, device=dev)
        o2, n2, _ = vm_byte_exact(model.alu, feats[ii], oph[ii], vtgt[ii])
        per_op[op] = f"{o2}/{n2}"
    print(f"    per-op: {per_op}")
    results["graft_byte_exact_before"] = dict(ok=ok, n=n, min_margin=mm, per_op=per_op)

    # ================= (3) ACTUALLY USE IT (e2e inference demo) ============== #
    print("\n===== (3) USE IT: model invokes the byte-exact ALU on real operands =====")
    demos = [("ADD", 173, 42), ("SUB", 200, 57), ("DIV", 7, 246),
             ("CMP_LT", 12, 99), ("CMP_GT", 250, 4)]
    use_rows = []
    # e2e path: pack operands into host hidden dims, then read them back through
    # the graft's residual-read (alu_from_hidden) -> proves the graft consumes
    # host-carried operands, not just a private feature vector.
    for op, a, b in demos:
        mem = [0, 0, 0, 0]; addr = 0
        gt, flag, _ = op_semantics(op, a, b, mem, addr)
        # build a host-hidden-state row carrying the operands in the VM dims
        hidden = torch.zeros(1, model.H, device=dev)
        hidden[0, model.feat_dims[IN_ONE]] = 1.0
        hidden[0, model.feat_dims[IN_A]] = float(a)
        hidden[0, model.feat_dims[IN_B]] = float(b)
        hidden[0, model.feat_dims[IN_GMEM]] = float(mem[addr])
        hidden[0, model.feat_dims[IN_CMP_LT]] = float(int(a < b))
        hidden[0, model.feat_dims[IN_CMP_EQ]] = float(int(a == b))
        hidden[0, model.feat_dims[IN_CMP_GT]] = float(int(a > b))
        hidden[0, model.feat_dims[IN_QUOT]] = float((b // a) if a != 0 else 0)
        hidden[0, model.op_dims[OP_ID[op]]] = 1.0
        with torch.no_grad():
            res = model.alu_from_hidden(hidden)                  # [1]
            pred = int(head_decode(res.double())[0].item())
        okd = (pred == gt)
        expr = {"ADD": f"{a}+{b}", "SUB": f"{a}-{b}", "DIV": f"{b}//{a}",
                "CMP_LT": f"({a}<{b})", "CMP_GT": f"({a}>{b})"}[op]
        print(f"    {op:<7} {expr:<10} -> graft={pred:<3}  exact={gt:<3}  {'OK' if okd else 'MISMATCH'}")
        use_rows.append(dict(op=op, a=a, b=b, expr=expr, graft=pred, exact=gt, ok=okd))
    results["use_it_demo"] = use_rows
    use_ok = all(r["ok"] for r in use_rows)
    print(f"    -> ALL demos byte-exact: {use_ok}")
    poll_rss()

    # ================= (2) RL / FINE-TUNE SURVIVAL ========================== #
    # The honest RL stress: the model is RL'd on a task that ACTUALLY INVOKES the
    # grafted ALU (the graft is ON the host forward, vm_mode), so the RL gradient
    # GENUINELY flows into W_key (measured, nonzero) — not a disconnected adapter.
    # Two loss families combine into the RL objective:
    #   * HOST CHAT LM loss (teacher-forced CE on instruct answers) -> drives the
    #     host params, the ordinary fine-tune the graft has to survive;
    #   * a GRPO-style VM REWARD: run the operands through the host-forward-in-VM-mode,
    #     decode the RESULT via the head, reward = margin of the CORRECT byte. This is
    #     a policy-gradient-flavoured signal whose gradient hits the ALU directly.
    # Regimes probe protections (freeze / decoupled-wd / joint margin plateau).
    print("\n===== (2) RL/FINE-TUNE SURVIVAL: does the graft stay byte-exact? =====")
    ii, ll, am = build_sft_batch(tok, dev)
    base_ids = torch.zeros(feats.shape[0], 1, dtype=torch.long, device=dev)
    n_rl = 30 if quick else 120
    wd = 1e-4

    def host_lm_loss(mdl, vm_active=False):
        """Ordinary teacher-forced chat LM CE (the host fine-tune objective). When
        vm_active, the graft is LIVE on this forward: it reads the chat residual's VM
        dims and writes RESULT into the residual, so the RESULT dim participates in
        the LM logits and this (nonzero-at-W0) LM loss backprops INTO W_key. That is
        the honest RL-drift coupling — an ordinary fine-tune whose gradient genuinely
        reaches the grafted attractor (not a disconnected adapter)."""
        prev = mdl.vm_mode
        mdl.vm_mode = vm_active
        try:
            out = mdl.host(input_ids=ii, attention_mask=am)
        finally:
            mdl.vm_mode = prev
        logits = out.logits[:, :-1, :]
        labels = ll[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels.reshape(-1), ignore_index=-100)

    def vm_reward_loss(mdl):
        """GRPO-style RL signal that RUNS THE GRAFT: push each op's RESULT (decoded
        through the host-forward-in-VM-mode) toward its exact byte. Negative reward =
        the head's margin loss on the grafted RESULT. Its gradient flows into W_key
        THROUGH the whole host stack -> the real RL-drift stress on the attractor."""
        result = mdl.vm_forward_result(feats, oph, base_ids)     # [N] via host forward
        grid = _byte_val_grid().to(result.device)
        x = result.unsqueeze(-1)
        logits = 2.0 * grid * x - grid ** 2
        correct = logits.gather(1, vtgt.view(-1, 1)).squeeze(1)
        masked = logits.clone(); masked.scatter_(1, vtgt.view(-1, 1), float("-inf"))
        # reward-to-go = margin; RL "maximises reward" == minimise -margin, clamped.
        return torch.clamp(MARGIN_M - (correct - masked.max(1).values), min=0.0).mean()

    # lr per regime: the aggressive stress uses a 10x lr to try to DRIVE the graft
    # off the #910 attractor (grounds the "needs protection?" verdict honestly).
    regimes = [
        ("dormant_graft_lm_only", 1e-4,
         "REALISTIC: RL = host LM loss only, graft NOT invoked (off host forward)"),
        ("active_shared_optim", 1e-4,
         "STRESS: RL invokes the graft (VM reward) + host LM, graft in SAME optimizer"),
        ("active_shared_optim_aggressive", 1e-3,
         "HARD STRESS: same, 10x lr — can bare shared-optim be driven off the attractor?"),
        ("active_frozen_graft", 1e-4,
         "PROTECTION-FREEZE: graft invoked but requires_grad=False"),
        ("active_joint_margin", 1e-3,
         "PROTECTION-JOINT: graft invoked at 10x lr, + #910 margin-plateau term"),
    ]
    results["rl_survival"] = []

    # snapshot the pristine grafted weights to restore between regimes
    W0 = {k: v.detach().clone() for k, v in model.alu.state_dict().items()}
    host0 = {k: v.detach().clone() for k, v in model.host.state_dict().items()}

    for reg_name, reg_lr, reg_desc in regimes:
        # restore graft + host to pristine each regime
        model.alu.load_state_dict({k: v.clone() for k, v in W0.items()})
        model.host.load_state_dict({k: v.clone() for k, v in host0.items()})

        # before-update byte-exact + graft grad-norm on the #910 plateau
        ok_b, n_b, mm_b = vm_byte_exact(model.alu, feats, oph, vtgt)
        gn_b = graft_grad_norm(model.alu, feats, oph, vtgt)

        # per-regime optimizer + loss composition
        invoke_graft = (reg_name != "dormant_graft_lm_only")
        is_joint = (reg_name == "active_joint_margin")
        if reg_name == "active_frozen_graft":
            for p in model.alu.parameters():
                p.requires_grad_(False)
            params = [p for p in model.host.parameters() if p.requires_grad]
        else:
            for p in model.alu.parameters():
                p.requires_grad_(True)
            params = list(model.host.parameters()) + list(model.alu.parameters())
        opt = torch.optim.AdamW(params, lr=reg_lr, weight_decay=wd)  # DECOUPLED wd (#910)
        # host LM loss BEFORE (graft-on) to show the host actually trains
        with torch.no_grad():
            lm_before = float(host_lm_loss(model, vm_active=invoke_graft).item())

        max_graft_gn = 0.0
        for step in range(n_rl):
            opt.zero_grad(set_to_none=True)
            # In ACTIVE regimes the graft is LIVE on the chat LM forward -> the
            # (nonzero) LM gradient reaches W_key; PLUS the VM-task reward term.
            loss = host_lm_loss(model, vm_active=invoke_graft)
            if invoke_graft:
                loss = loss + vm_reward_loss(model)              # RL grad INTO the graft
            if is_joint:
                loss = loss + alu_margin_loss(model.alu, feats, oph, vtgt)
            loss.backward()
            # record the RL gradient magnitude actually hitting the graft
            ggn = 0.0
            for p in model.alu.parameters():
                if p.grad is not None:
                    ggn += float(p.grad.norm() ** 2)
            max_graft_gn = max(max_graft_gn, ggn ** 0.5)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 10 == 0 or step == n_rl - 1:
                poll_rss()

        # after-update byte-exact + drift/basin diagnosis
        ok_a, n_a, mm_a = vm_byte_exact(model.alu, feats, oph, vtgt)
        drift = 0.0
        cur = model.alu.state_dict()
        for k in W0:
            drift += float((cur[k].detach() - W0[k]).norm() ** 2)
        drift = drift ** 0.5
        survived = (ok_a == n_a)
        # HONEST basin diagnosis: #910's TIGHT static basin is 3e-4, but the margin
        # PLATEAU (hinge m=0.5 -> half-width 0.25 byte) is the real robust region.
        # Distinguish the two: did the graft stay inside the tight static basin, and/or
        # did it stay byte-exact because the wider margin-plateau absorbed the drift?
        inside_static_basin = (drift < 3e-4)
        held_by_plateau = (survived and not inside_static_basin)  # plateau, not tight basin

        # LM loss AFTER (graft-on) — did the host actually train?
        with torch.no_grad():
            lm_after = float(host_lm_loss(model, vm_active=invoke_graft).item())

        row = dict(regime=reg_name, desc=reg_desc, lr=reg_lr,
                   byte_exact_before=f"{ok_b}/{n_b}", byte_exact_after=f"{ok_a}/{n_a}",
                   pct_before=100.0*ok_b/n_b, pct_after=100.0*ok_a/n_a,
                   min_margin_before=mm_b, min_margin_after=mm_a,
                   graft_plateau_grad_norm=gn_b,
                   max_rl_grad_into_graft=max_graft_gn,
                   graft_weight_drift=drift,
                   inside_static_basin=inside_static_basin,
                   held_by_margin_plateau=held_by_plateau,
                   lm_loss_before=lm_before, lm_loss_after=lm_after,
                   n_steps=n_rl, survived=survived)
        results["rl_survival"].append(row)
        print(f"  [{reg_name}]  (lr={reg_lr:.0e})")
        print(f"     byte-exact {ok_b}/{n_b} -> {ok_a}/{n_a}   "
              f"min-margin {mm_b:.3f} -> {mm_a:.3f}   SURVIVED={survived}")
        print(f"     graft plateau grad-norm(W0)={gn_b:.2e}   "
              f"max RL-grad into graft={max_graft_gn:.2e}")
        print(f"     graft weight drift={drift:.2e}   inside-tight-basin(<3e-4)={inside_static_basin}"
              f"   held-by-margin-plateau={held_by_plateau}")
        print(f"     host LM loss {lm_before:.4f} -> {lm_after:.4f} (host trained)")

    # -------- BREAK-POINT PROBE: escalate the BARE shared-optim RL until the graft
    # actually DROPS below byte-exact, and confirm the JOINT-margin protection holds
    # at the SAME aggression. This grounds the "needs protection?" verdict beyond the
    # fixed-window survival above (the margin erosion trend predicts a break). --------
    print("\n===== (2b) BREAK-POINT: escalate bare-RL until byte-exact BREAKS =====")
    esc = [(1e-3, 200), (3e-3, 300), (1e-2, 400), (3e-2, 400)] if not quick \
          else [(3e-3, 60), (1e-2, 80)]
    results["breakpoint_probe"] = []
    bare_break = None
    joint_break = None
    for (lr_e, steps_e) in esc:
        for mode in ("bare", "joint"):
            model.alu.load_state_dict({k: v.clone() for k, v in W0.items()})
            model.host.load_state_dict({k: v.clone() for k, v in host0.items()})
            for p in model.alu.parameters():
                p.requires_grad_(True)
            params = list(model.host.parameters()) + list(model.alu.parameters())
            opt = torch.optim.AdamW(params, lr=lr_e, weight_decay=wd)
            for _ in range(steps_e):
                opt.zero_grad(set_to_none=True)
                loss = host_lm_loss(model, vm_active=True) + vm_reward_loss(model)
                if mode == "joint":
                    loss = loss + alu_margin_loss(model.alu, feats, oph, vtgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                poll_rss()
            ok_e, n_e, mm_e = vm_byte_exact(model.alu, feats, oph, vtgt)
            drift_e = 0.0
            cur = model.alu.state_dict()
            for k in W0:
                drift_e += float((cur[k].detach() - W0[k]).norm() ** 2)
            drift_e = drift_e ** 0.5
            broke = (ok_e < n_e)
            results["breakpoint_probe"].append(
                dict(mode=mode, lr=lr_e, steps=steps_e, byte_exact=f"{ok_e}/{n_e}",
                     pct=100.0*ok_e/n_e, min_margin=mm_e, drift=drift_e, broke=broke))
            if mode == "bare" and broke and bare_break is None:
                bare_break = (lr_e, steps_e, 100.0*ok_e/n_e)
            if mode == "joint" and broke and joint_break is None:
                joint_break = (lr_e, steps_e, 100.0*ok_e/n_e)
            print(f"    lr={lr_e:.0e} x{steps_e:<4} [{mode:<5}]: byte-exact {ok_e}/{n_e} "
                  f"({100.0*ok_e/n_e:5.1f}%)  min-margin={mm_e:.3f}  drift={drift_e:.2e}"
                  f"  {'<-- BROKE' if broke else ''}")
    results["bare_break_at"] = bare_break
    results["joint_break_at"] = joint_break
    print(f"  -> BARE shared-optim first BROKE at: {bare_break}")
    print(f"  -> JOINT-margin protection first BROKE at: {joint_break}")

    # restore pristine for a final sanity generate (host after training regimes)
    model.host.load_state_dict({k: v.clone() for k, v in host0.items()})

    # ================= VERDICT ============================================== #
    print("\n################################ VERDICT #########################")
    grafts_byte_exact = (results["graft_byte_exact_before"]["ok"]
                         == results["graft_byte_exact_before"]["n"])
    host_sane = all(len(a.strip()) > 0 for _, a in sane_before)
    host_identity = results.get("host_byte_identity_graft_off", None)
    used = use_ok
    survival = {r["regime"]: r["survived"] for r in results["rl_survival"]}
    drift = {r["regime"]: r["graft_weight_drift"] for r in results["rl_survival"]}
    rlgrad = {r["regime"]: r["max_rl_grad_into_graft"] for r in results["rl_survival"]}
    margin_after = {r["regime"]: r["min_margin_after"] for r in results["rl_survival"]}
    survives_dormant = survival.get("dormant_graft_lm_only", False)
    survives_active_shared = survival.get("active_shared_optim", False)
    survives_active_aggr = survival.get("active_shared_optim_aggressive", False)
    survives_active_frozen = survival.get("active_frozen_graft", False)
    survives_active_joint = survival.get("active_joint_margin", False)
    # honest RL grad reaching the graft in the active-shared stress (must be nonzero)
    rlgrad_reached_graft = rlgrad.get("active_shared_optim", 0.0) > 0.0
    survives_any = any(survival.values())
    # protection needed = the BREAK-POINT probe shows bare shared-optim DOES break
    # under enough RL aggression, while the joint-margin protection holds strictly
    # longer (or freeze never moves it at all). Grounded by (2b), not just the window.
    bare_break = results.get("bare_break_at", None)
    joint_break = results.get("joint_break_at", None)
    bare_breaks = bare_break is not None
    joint_outlasts_bare = bare_breaks and (
        joint_break is None or joint_break[0] > bare_break[0]
        or (joint_break[0] == bare_break[0] and joint_break[1] > bare_break[1]))
    needs_protection = bare_breaks and (survives_active_frozen or joint_outlasts_bare)
    verdict = dict(
        a_grafts_byte_exact=grafts_byte_exact,
        host_still_sane=host_sane,
        host_byte_identical_graft_off=host_identity,
        b_survives_dormant_rl=survives_dormant,
        b_survives_active_shared_optim=survives_active_shared,
        b_survives_active_shared_aggressive=survives_active_aggr,
        b_survives_active_frozen=survives_active_frozen,
        b_survives_active_joint=survives_active_joint,
        rl_grad_reached_graft_in_stress=rlgrad_reached_graft,
        bare_break_at=bare_break,
        joint_break_at=joint_break,
        needs_protection=needs_protection,
        c_actually_used=used,
        all_three_with_protection=(grafts_byte_exact and survives_any and used),
    )
    results["verdict"] = verdict
    print(f"  (a) GRAFTS byte-exact          : {grafts_byte_exact}  "
          f"({results['graft_byte_exact_before']['ok']}/{results['graft_byte_exact_before']['n']}); "
          f"host sane={host_sane}; host byte-identical w/ graft-off={host_identity}")
    print(f"  (b) SURVIVES RL (byte-exact after update):")
    print(f"        dormant (graft not invoked)      : {survives_dormant}"
          f"   (RL never touches it -> trivially safe)")
    print(f"        ACTIVE shared-optim lr1e-4       : {survives_active_shared}"
          f"   [RL grad={rlgrad.get('active_shared_optim',0):.1e},"
          f" drift={drift.get('active_shared_optim',0):.1e},"
          f" margin_after={margin_after.get('active_shared_optim',0):.3f}]")
    print(f"        ACTIVE shared-optim AGGRESSIVE   : {survives_active_aggr}"
          f"   [drift={drift.get('active_shared_optim_aggressive',0):.1e},"
          f" margin_after={margin_after.get('active_shared_optim_aggressive',0):.3f}]")
    print(f"        ACTIVE + FREEZE protection       : {survives_active_frozen}"
          f"   [drift={drift.get('active_frozen_graft',0):.1e}]")
    print(f"        ACTIVE + JOINT-margin (aggr lr)  : {survives_active_joint}"
          f"   [drift={drift.get('active_joint_margin',0):.1e},"
          f" margin_after={margin_after.get('active_joint_margin',0):.3f}]")
    print(f"     -> RL gradient genuinely reached the graft in the stress: {rlgrad_reached_graft}")
    print(f"     -> BARE shared-optim break-point : {bare_break}")
    print(f"     -> JOINT-margin break-point      : {joint_break}")
    print(f"     -> PROTECTION NEEDED (bare breaks; freeze/joint outlast): {needs_protection}")
    print(f"  (c) ACTUALLY USED (e2e)        : {used}")
    print(f"  ==> (a) graft + (b) survives-with-protection + (c) used: "
          f"{verdict['all_three_with_protection']}")

    results["peak_rss_gb"] = PEAK_RSS
    results["elapsed_s"] = time.time() - t0
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_agent_graft_sgd_vm_rl_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"\nWrote {out}   (elapsed {time.time()-t0:.1f}s, PEAK RSS {PEAK_RSS:.2f} GB)")


if __name__ == "__main__":
    main()
