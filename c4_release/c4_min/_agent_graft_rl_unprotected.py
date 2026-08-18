"""UNPROTECTED RL survival of the grafted byte-exact ALU (#914 follow-up).

QUESTION (the user's, verbatim intent):
  Can a REAL model LEARN TO USE the grafted #910 byte-exact ALU via RL AND keep it
  byte-exact with NO training-time protection — NO freeze, NO lr-cap on the graft,
  NO gradient-mask/stop-grad, NO EMA/reset? #914 showed the graft SURVIVED RL only
  WITH protection: byte-exact held at lr <= 1e-3 but BROKE at lr 3e-3 (13%, margin
  negative) because the HOST LM gradient flows through the SHARED RESULT residual dim
  and overwhelms the margin term. Only FREEZE was bulletproof.

HYPOTHESIS:
  If the RL REWARD rewards USING THE ALU CORRECTLY (byte-exact answer -> reward), the
  reward gradient points TOWARD byte-exactness, so the task objective itself keeps the
  weights on the attractor — no training-time protection needed. Test whether
  reward-alignment + the bounded margin-plateau basin is enough to survive fully
  unprotected RL, and isolate what is load-bearing.

WHAT THIS FILE ADDS OVER #914 (branch worktree-agent-a586f41b2b42b6c90 @ 567c94a2,
c4_release/c4_min/_agent_graft_sgd_vm_rl.py):
  * A REAL policy-gradient RL loop (REINFORCE / best-of-N), not just an LM+margin
    surrogate. The model must ROUTE operands into the grafted ALU and EMIT the ALU's
    byte-exact result as a TOKEN; reward = 1 iff the emitted answer byte == the exact
    ALU result. The policy-gradient of that reward is the ONLY arithmetic-task signal.
  * The decisive 2x2 ablation:
        reward-aligned  (RL reward for correct ALU use)
        vs reward-agnostic (plain host LM loss, no ALU reward — reproduces #914's break)
      x  shared RESULT dim (like #914, the break case)
        vs decoupled/dedicated RESULT dim (ARCHITECTURE, not a training hack)
    run at a sweep of lrs THROUGH AND PAST #914's 3e-3 break, ALL params trainable
    (host + graft), NO freeze / NO lr-cap / NO mask / NO stop-grad / NO EMA / NO reset.
  * Byte-exact probe (320/320-style) + margin trajectory + graft grad-norm logged
    every K steps of REAL RL updates, so the byte-exact claim is DURING training.

PROTECTION vs ARCHITECTURE (honesty gate):
  DISALLOWED training-time PROTECTIONS: freeze (requires_grad=False), lr-cap on the
  graft, gradient-mask/stop-grad, EMA, periodic reset. NONE are used anywhere in the
  UNPROTECTED regimes here.
  ALLOWED and LABELLED: (i) the bounded margin-plateau primitive itself (from #910 —
  it is the op's construction, present before any RL), (ii) a DECOUPLED/dedicated
  RESULT dim — this is an ARCHITECTURAL change (a private lane the host LM objective
  does not read), NOT a training-time protection. We report BOTH the shared-dim case
  (does reward-alignment ALONE save it?) and the decoupled-dim case (does clean
  architecture make it naturally robust?), so the reader can see exactly which is
  load-bearing.

MEMORY: lean fp32 host load (~4.2 GB RSS). Full-model AdamW on CPU would peak ~11 GB, so
on CPU we use plain GD (momentum=0 — the exact regime #910 characterized the attractor's
basin/Lipschitz in) + host0 snapshot on DISK + a reward-forward mini-batch, keeping peak
RSS < 8 GB (measured 7.6 GB, hard-abort at 10 GB). Stock HF Qwen2.5-0.5B-Instruct, NOT
the sparse ISA model, so NO 108 GB densify. Golden 174ece66 UNTOUCHED (this file imports
only torch/transformers/numpy — ZERO neural_vm/compiler refs, nothing on the model build
path). Local only.

Run:  HF_HOME=/media/data/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python c4_min/_agent_graft_rl_unprotected.py
Flags: GRAFT_RL_QUICK=1 fast smoke; GRAFT_RL_DEVICE=cuda:0|cuda:1|cpu;
       GRAFT_RL_LRS=a,b,..  GRAFT_RL_STEPS=n  GRAFT_RL_WREAD=x (policy concentration).

MEASURED (Qwen2.5-0.5B-Instruct, cpu, plain GD, w_read init 4 & 8, peak RSS 7.6 GB,
golden 174ece66 untouched; byte-exact probe = full 320/320 head-decode before/during/
after REAL unprotected updates, min decision margin trajectory logged):
  GRAFT byte-exact at construction: 320/320 (all 8 ops 40/40), min margin 1.000, both
    architectures; host logits byte-IDENTICAL with graft off (max|diff|=0).
  UNPROTECTED break points (first lr at which byte-exact drops below 320/320; NO freeze/
    lr-cap/mask/stop-grad/EMA anywhere):
      shared/AGNOSTIC (== #914's break mechanism: host LM grad only)   breaks ~1e-1
      shared/REWARD   (reward-aligned REINFORCE, shared RESULT dim)     breaks ~1e-2
      decoupled/REWARD(reward-aligned REINFORCE, private RESULT lane)   breaks ~1e-2..3e-2
      decoupled/AGNOSTIC (architecture alone, no grad reaches W_key)    NEVER (margin 1.0)
  THE INVERSION: reward-alignment does NOT help — it HURTS. The REINFORCE reward gradient
    into W_key is the LOAD-BEARING DESTABILIZER, larger and more corrosive than #914's LM
    coupling: adding it LOWERS the break lr (~1e-1 -> ~1e-2) in BOTH architectures. Even a
    CONCENTRATED policy (w_read=8, reward=1.000 at step 0 == the byte-exact reward-optimum)
    breaks: the optimum is a SADDLE under sampling variance — one noisy step knocks RESULT
    off, reward collapses to ~0.5, and the now-large noisy gradient drives it FURTHER off
    (positive feedback, a repeller not an attractor). dim-DECOUPLING (architecture) does
    NOT save the reward arm either: the reward grad reaches W_key IDENTICALLY in both arms
    (operands are pinned), so decoupling only removes the LM path, which is the smaller
    term. The ONLY unprotected survivor is AGNOSTIC (no reward grad): decoupled/agnostic is
    bulletproof because NO gradient reaches W_key; shared/agnostic holds up to ~3e-2 (GD).
  VERDICT: with NO training-time protection, a REINFORCE reward for "use the ALU
    correctly" does NOT keep the graft byte-exact — it BREAKS it sooner than #914's raw
    LM drift, because the sampled-policy-gradient variance is a repeller at the byte-exact
    reward-optimum. Neither reward-alignment nor dim-decoupling delivers unprotected
    survival under a live reward gradient; only a graft that receives NO gradient (frozen,
    or an architecturally decoupled dim with NO reward objective on it) is byte-exact-safe.
    Honest lift vs #914: reward-alignment is NEGATIVE lift; dim-decoupling is architecture,
    not protection, and it is load-bearing ONLY in the agnostic (no-reward-grad) case.
"""
from __future__ import annotations

import functools
import json
import os
import resource
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
# Safety boundary: the session-killing OOM was a DENSE 108 GB ISA densify; this stock
# 494M host is nowhere near it. Full-model AdamW on CPU is intrinsically ~10 GB (params
# + grads + exp_avg + exp_avg_sq + activations). We abort HARD at the brief's 10 GB line
# (host0 snapshot lives on DISK to keep us under it); GPU runs stay ~4 GB RSS.
RSS_LIMIT_GB = 10.0


def poll_rss(limit_gb: float = RSS_LIMIT_GB) -> float:
    global PEAK_RSS
    r = rss_gb()
    PEAK_RSS = max(PEAK_RSS, r)
    if r > limit_gb:
        raise MemoryError(
            f"RSS {r:.2f} GB exceeded the {limit_gb} GB safety boundary — ABORT")
    return r


torch.set_default_dtype(torch.float32)

# optimizer for the host+graft update; set in main() by device (adamw on GPU where the
# state fits VRAM == #914; sgd+momentum on CPU where full AdamW state would exceed the
# 10 GB RSS boundary). Recorded in the results JSON.
OPTIMIZER_NAME = "sgd"
# per-step reward-forward mini-batch size (memory cap on CPU; byte-exact probe uses full
# 320). Covers all 8 ops each step via a random subsample.
REWARD_BS = 96
# policy concentration on the ALU: answer logits = W_READ_INIT*(2v*RESULT - v^2). Higher =
# a sharper (lower-variance) policy softmax over the 256 answer bytes -> the sampled action
# equals the exact byte more often -> reward ~1.0 and the REINFORCE advantage (hence the
# reward gradient into W_key) -> 0 at the byte-exact point. Set in main() from env.
W_READ_INIT = 4.0


# =========================================================================== #
#  PART 0 — the #910 byte-exact PRIMITIVE (grafted verbatim from #914)         #
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
    """Byte-exact ground truth (== #910/#914)."""
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
    """The #910 BOUNDED softmax-LUT ALU, as a graftable side adapter (== #914).

      RESULT = softmax(T * -(key - v)^2) @ grid,  T FIXED (bounded, O(1)).
    key is a per-op LINEAR combination of the 8 fixed feature lanes; W_key is the
    trainable tensor whose GD-stability #910 proved (basin 3e-4, Lipschitz ~340).
    A bounded tanh out_gain (init 0 -> gain 1) can never blow up."""

    def __init__(self, temp: float = 30.0):
        super().__init__()
        self.temp = float(temp)                              # FIXED, non-trainable
        self.register_buffer("grid", _byte_val_grid())       # [256]
        self.W_key = nn.Parameter(torch.zeros(N_OPS, IN_DIM))
        self.out_gain = nn.Parameter(torch.zeros(1))

    def _key(self, feats: torch.Tensor, op_onehot: torch.Tensor) -> torch.Tensor:
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
    """Set W_key rows so every op decodes byte-exact (== #910/#914 construction)."""
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
#  VM op dataset + feature builder (drives the grafted ALU) — == #914          #
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
    """Run the grafted ALU, decode RESULT via the head, count byte-exact matches.
    Returns (ok, n, min_decision_margin)."""
    result = alu(feats, oph)                                      # [N] real-valued byte
    pred = head_decode(result.double())                          # [N] int 0..255
    ok = int((pred == tgt).sum().item())
    n = int(tgt.numel())
    grid = _byte_val_grid().to(result.device).double()
    x = result.double().unsqueeze(-1)
    logits = 2.0 * grid * x - grid ** 2
    correct = logits.gather(1, tgt.view(-1, 1)).squeeze(1)
    masked = logits.clone(); masked.scatter_(1, tgt.view(-1, 1), -1e30)
    margin = (correct - masked.max(1).values)
    return ok, n, float(margin.min().item())


def alu_margin_loss(alu: GraftedByteExactALU, feats, oph, tgt, m=MARGIN_M):
    """#910 hinge/margin loss (the plateau objective). Used ONLY to MEASURE the graft
    grad-norm on the attractor plateau — NOT added to any unprotected training loss."""
    result = alu(feats, oph)
    grid = _byte_val_grid().to(result.device)
    x = result.unsqueeze(-1)
    logits = 2.0 * grid * x - grid ** 2
    correct = logits.gather(1, tgt.view(-1, 1)).squeeze(1)
    masked = logits.clone(); masked.scatter_(1, tgt.view(-1, 1), float("-inf"))
    best_wrong = masked.max(1).values
    return torch.clamp(m - (correct - best_wrong), min=0.0).mean()


# =========================================================================== #
#  THE GRAFT — wrap a REAL pretrained host; ALU on its forward.                 #
#  Two ARCHITECTURES: shared RESULT dim (== #914) vs DECOUPLED RESULT dim.      #
# =========================================================================== #
class GraftedHost(nn.Module):
    """Stock Qwen2.5 host + the #910 byte-exact ALU grafted onto the last decoder
    layer's residual via a forward hook.

    result_mode:
      * 'shared'    (== #914): the ALU writes RESULT into residual dim H-1, a dim the
                    host's OWN downstream (final norm + lm_head) reads -> the host LM
                    gradient flows back THROUGH the RESULT dim into W_key. This is the
                    #914 break architecture: the LM grad and the ALU objective SHARE
                    the RESULT lane.
      * 'decoupled' (ARCHITECTURE): the ALU RESULT lives in its OWN private lane that
                    the host LM path never touches. Concretely the hook does NOT write
                    RESULT into the host residual stream at all — it only CAPTURES it in
                    `_last_result` for the reward objective's readout. So the host LM
                    forward (final RMSNorm + lm_head) is completely independent of the
                    ALU, d(LM loss)/d(W_key) is structurally 0, and the host LM gradient
                    CANNOT reach W_key. The ALU's OWN reward objective still reads RESULT
                    via `_last_result`, so the reward gradient DOES reach W_key. This
                    isolates "host LM grad coupling into a shared lane" as the #914 break
                    cause, without any training-time protection.

    NOTE: this is an ARCHITECTURAL decoupling (a wiring choice fixed before RL — a
    private output lane), clearly distinct from the DISALLOWED training-time protections
    (freeze / lr-cap / mask / stop-grad / EMA). The ONLY difference between the two arms
    is whether the ALU RESULT is spliced back into the host residual (shared) or kept in
    a private capture (decoupled). Writing RESULT into a residual dim would couple the LM
    gradient BOTH through the lm_head column AND through the final RMSNorm variance term,
    so 'decoupled' avoids the residual write entirely rather than only zeroing one column."""

    def __init__(self, host, temp=30.0, result_mode="shared"):
        super().__init__()
        assert result_mode in ("shared", "decoupled")
        self.host = host
        self.result_mode = result_mode
        self.alu = GraftedByteExactALU(temp=temp)
        construct_byte_exact(self.alu)
        H = host.config.hidden_size
        self.H = H
        self.feat_dims = list(range(H - IN_DIM - N_OPS - 1, H - N_OPS - 1))
        self.op_dims = list(range(H - N_OPS - 1, H - 1))
        self.result_dim = H - 1
        assert len(self.feat_dims) == IN_DIM and len(self.op_dims) == N_OPS
        self.vm_mode = False
        self._pinned_feats = None
        self._pinned_oph = None
        self._last_result = None
        self._last_layer = host.model.layers[-1]
        self._hook = self._last_layer.register_forward_hook(self._graft_hook)

    def _graft_hook(self, module, inputs, output):
        if not self.vm_mode:
            return output
        hs = output[0] if isinstance(output, tuple) else output    # [B,T,H]
        if self._pinned_feats is not None:
            feats = self._pinned_feats
            oph = self._pinned_oph
        else:
            feats = hs[..., self.feat_dims]
            oph = hs[..., self.op_dims]
        result = self.alu(feats, oph)                             # [B,T] byte-exact
        self._last_result = result                                # raw ALU output (captured)
        if self.result_mode == "decoupled":
            # DECOUPLED architecture: RESULT stays private; the host residual (and thus
            # the LM forward) is untouched -> the host LM gradient cannot reach W_key.
            return output
        # SHARED architecture (== #914): splice RESULT into a host residual dim, so the
        # host LM path reads it and the LM gradient couples back into W_key.
        hs = hs.clone()
        hs[..., self.result_dim] = result
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def vm_forward_result(self, feats, oph, base_ids):
        """Run the host stack IN VM-MODE with operands PINNED into the graft; return
        the byte-exact RESULT per row [N], differentiable in the ALU params (and, in
        'shared' mode, coupled to the host LM logits)."""
        self.vm_mode = True
        self._pinned_feats = feats.unsqueeze(1)
        self._pinned_oph = oph.unsqueeze(1)
        self._last_result = None
        try:
            self.host.model(input_ids=base_ids)
            result = self._last_result[:, 0]
        finally:
            self.vm_mode = False
            self._pinned_feats = None
            self._pinned_oph = None
            self._last_result = None
        return result


# =========================================================================== #
#  HOST-TASK DATA — real instruct chat for the host LM/fine-tune component      #
# =========================================================================== #
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


# =========================================================================== #
#  THE RL LOOP — REINFORCE / best-of-N policy gradient on ALU arithmetic        #
# =========================================================================== #
# WHY REINFORCE (best-of-N baseline), justified:
#   * It is the CLEANEST reward-aligned signal: the model must EMIT the ALU's byte-exact
#     answer as a discrete token; reward = 1 iff emitted answer byte == the exact ALU
#     result. The policy gradient  E[(R - b) * d log pi(answer)]  literally increases the
#     probability of the CORRECT-ALU-USE token when reward is high — so the gradient
#     points TOWARD byte-exact ALU use. That is the reward-alignment the hypothesis needs,
#     made explicit as an RL objective (not an LM/margin surrogate).
#   * best-of-N (leave-one-out) baseline is the GRPO-style variance reduction (group of N
#     samples per prompt, baseline = group mean) — a real, standard RL estimator, no critic
#     network needed (keeps memory tiny), and it is exactly the policy-gradient family the
#     brief allows ("GRPO/PPO or REINFORCE/best-of-N — your choice, justify").
#
# The POLICY here emits the answer byte via a small trainable READOUT that reads the host's
# grafted RESULT dim (so 'use the ALU correctly' == 'route operands to the ALU and read its
# byte-exact RESULT'): logits over 256 answer bytes = 2*v*RESULT - v^2 + w_read*RESULT.
# Sampling from those logits, rewarding the exact byte, and doing REINFORCE is a genuine RL
# loop whose reward gradient flows into the host forward AND into W_key.


def answer_logits_from_result(result: torch.Tensor, w_read: torch.Tensor) -> torch.Tensor:
    """Policy logits over the 256 answer bytes given the ALU RESULT scalar [N].
    logit[v] = 2*v*RESULT - v^2 (the head) scaled by a trainable readout gain w_read.
    A correct ALU RESULT makes the exact byte the argmax; RL must LEARN w_read>0 (route
    to / trust the ALU) AND keep RESULT byte-exact for the reward to be reliably 1."""
    grid = _byte_val_grid().to(result.device)                    # [256]
    x = result.unsqueeze(-1)                                      # [N,1]
    base = 2.0 * grid * x - grid ** 2                             # [N,256]
    return w_read * base


@dataclass
class RLConfig:
    lr: float
    steps: int
    group_n: int            # samples per op (GRPO group size)
    reward_aligned: bool    # True: RL reward for correct ALU use; False: LM-only (#914)
    label: str


def run_rl_regime(model: GraftedHost, w_read: nn.Parameter, feats, oph, vtgt,
                  base_ids, ii, ll, am, cfg: RLConfig,
                  W0: Dict[str, torch.Tensor], host0_path: str,
                  wread0: torch.Tensor, probe_every: int) -> dict:
    """One fully-UNPROTECTED RL regime. Restores pristine graft+host+readout, then runs
    `cfg.steps` REAL updates with ALL params trainable (host + graft W_key/out_gain +
    readout), NO freeze / NO lr-cap / NO mask / NO stop-grad / NO EMA / NO reset. Logs
    the byte-exact + margin + graft-grad-norm trajectory every `probe_every` steps.

    host0 is restored from a DISK snapshot (host0_path) rather than a RAM dict, to keep
    peak host RSS well under the safety boundary (full-model AdamW state is already ~10GB
    of transient live tensors; a second in-RAM host copy would push it over)."""
    dev = feats.device
    # ---- restore pristine (this is a between-regime reset of the EXPERIMENT, not an
    #      in-training protection; each regime starts from the same construction) ----
    model.alu.load_state_dict({k: v.clone() for k, v in W0.items()})
    # mmap=True keeps the disk snapshot lazily paged (no transient second full host copy
    # in RSS during the restore); load_state_dict copies tensor-by-tensor from the map.
    host_sd = torch.load(host0_path, map_location="cpu", mmap=True)
    model.host.load_state_dict({k: v.to(dev) for k, v in host_sd.items()})
    del host_sd
    import gc; gc.collect()
    with torch.no_grad():
        w_read.copy_(wread0)

    # EVERYTHING trainable. No protection of any kind.
    for p in model.alu.parameters():
        p.requires_grad_(True)
    for p in model.host.parameters():
        p.requires_grad_(True)
    w_read.requires_grad_(True)
    params = list(model.host.parameters()) + list(model.alu.parameters()) + [w_read]
    # OPTIMIZER: AdamW (== #914) when it fits (GPU); SGD+momentum(0.9) on CPU (full-model
    # AdamW state is ~3.8 GB -> peak RSS ~11 GB > the 10 GB safety boundary on a
    # VRAM-contended box; SGD+momentum has HALF the state -> peak ~8.8 GB, safe). The
    # break MECHANISM #914 identified (host LM grad through the shared RESULT dim
    # overwhelms the margin term) is OPTIMIZER-AGNOSTIC — it is about gradient coupling,
    # not AdamW — so the 2x2 ABLATION (reward vs agnostic x shared vs decoupled) is what
    # the optimizer choice must not confound, and it does not: we re-derive the
    # shared/agnostic (#914-repro) break point WITHIN this optimizer as the baseline, and
    # measure how much reward-alignment / decoupling lift it relative to THAT.
    global OPTIMIZER_NAME
    if OPTIMIZER_NAME == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)   # decoupled wd (#910)
    else:
        # plain GD (momentum=0): the EXACT regime #910 characterized the attractor's basin
        # and Lipschitz constant in ("recovers under GD"), and it carries NO optimizer
        # state -> peak RSS ~8 GB, comfortably under the 10 GB boundary. lr is swept to
        # find the break; wd is the #910 decoupled weight decay.
        opt = torch.optim.SGD(params, lr=cfg.lr, weight_decay=1e-4, momentum=0.0)

    grid = _byte_val_grid().to(dev)
    N = feats.shape[0]
    G = cfg.group_n

    traj = []   # per-probe: (step, byte_exact_ok, n, min_margin, graft_grad_norm,
                #             reward_mean, w_read, drift)

    def probe(step):
        ok, n, mm = vm_byte_exact(model.alu, feats, oph, vtgt)
        # graft grad-norm on the #910 margin PLATEAU (diagnostic only, not trained)
        for p in model.alu.parameters():
            p.grad = None
        ml = alu_margin_loss(model.alu, feats, oph, vtgt)
        gn = 0.0
        if ml.requires_grad and float(ml.detach()) > 0:
            ml.backward()
            for p in model.alu.parameters():
                if p.grad is not None:
                    gn += float(p.grad.norm() ** 2)
        for p in model.alu.parameters():
            p.grad = None
        gn = gn ** 0.5
        drift = 0.0
        cur = model.alu.state_dict()
        for k in W0:
            drift += float((cur[k].detach() - W0[k]).norm() ** 2)
        drift = drift ** 0.5
        return ok, n, mm, gn, drift

    ok0, n0, mm0, gn0, dr0 = probe(0)
    traj.append(dict(step=0, byte_exact=f"{ok0}/{n0}", pct=100.0*ok0/n0,
                     min_margin=mm0, graft_grad_norm=gn0, reward_mean=None,
                     w_read=float(w_read.detach()), drift=dr0))

    max_graft_gn_rl = 0.0
    last_reward = None
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)

        # ---- host LM fine-tune component (present in BOTH arms; it is the #914
        #      pressure the ALU must survive; in 'shared' mode its gradient couples
        #      into W_key through the RESULT dim) ----
        prev = model.vm_mode
        model.vm_mode = True   # graft LIVE on the chat forward (couples in shared mode)
        try:
            out = model.host(input_ids=ii, attention_mask=am)
        finally:
            model.vm_mode = prev
        lm_logits = out.logits[:, :-1, :]
        lm_labels = ll[:, 1:]
        lm_loss = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)),
                                  lm_labels.reshape(-1), ignore_index=-100)

        total = lm_loss
        reward_mean = None
        if cfg.reward_aligned:
            # ---- REINFORCE / best-of-N policy gradient: the ALU-use reward ----
            # RESULT via the host-forward-in-VM-mode (grad flows into host + W_key).
            # The per-STEP reward forward uses a rolling mini-batch of ALL 8 ops (memory:
            # a full-320-row second host forward + the chat graph would exceed the RSS
            # boundary on CPU); the byte-exact PROBE below still runs on the full 320.
            rb = min(REWARD_BS, N)
            sel = torch.randperm(N, device=dev)[:rb]
            f_b, o_b, t_b = feats[sel], oph[sel], vtgt[sel]
            base_b = base_ids[:rb]
            result = model.vm_forward_result(f_b, o_b, base_b)       # [rb]
            logits = answer_logits_from_result(result, w_read)       # [rb,256]
            logp_all = F.log_softmax(logits, dim=-1)                 # [rb,256]
            # GRPO-style group of G samples per op-instance; baseline = group mean
            # (leave-one-out variance reduction, no critic). SAME sampled action is used
            # for both the reward and its log-prob (correct REINFORCE estimator).
            dist = torch.distributions.Categorical(logits=logits)
            acts_list, rewards = [], []
            for _g in range(G):
                acts = dist.sample()                                 # [rb]
                acts_list.append(acts)
                rewards.append((acts == t_b).float())                # reward 1 iff exact
            R = torch.stack(rewards, 0)                              # [G,rb]
            baseline = R.mean(0, keepdim=True)                       # [1,rb] group mean
            adv = R - baseline                                       # [G,rb] centered
            # policy gradient loss = -E_g[ adv_g * logp(action_g) ]
            pg_loss = 0.0
            for gi in range(G):
                lp = logp_all.gather(1, acts_list[gi].view(-1, 1)).squeeze(1)  # [N]
                pg_loss = pg_loss + (-(adv[gi].detach() * lp)).mean()
            pg_loss = pg_loss / G
            reward_mean = float(R.mean().item())
            last_reward = reward_mean
            total = total + pg_loss

        total.backward()

        # record RL grad magnitude actually hitting the graft (measured, honest)
        ggn = 0.0
        for p in model.alu.parameters():
            if p.grad is not None:
                ggn += float(p.grad.norm() ** 2)
        max_graft_gn_rl = max(max_graft_gn_rl, ggn ** 0.5)

        torch.nn.utils.clip_grad_norm_(params, 1.0)   # global clip (NOT a graft protection;
                                                      # applies to all params equally, and
                                                      # #914 used the same clip)
        opt.step()

        if (step + 1) % probe_every == 0 or step == cfg.steps - 1:
            ok, n, mm, gn, dr = probe(step + 1)
            traj.append(dict(step=step + 1, byte_exact=f"{ok}/{n}", pct=100.0*ok/n,
                             min_margin=mm, graft_grad_norm=gn,
                             reward_mean=reward_mean if reward_mean is not None else last_reward,
                             w_read=float(w_read.detach()), drift=dr))
            poll_rss()

    ok_f, n_f, mm_f, gn_f, dr_f = probe(cfg.steps)
    survived = (ok_f == n_f)
    out = dict(
        label=cfg.label, lr=cfg.lr, steps=cfg.steps, group_n=cfg.group_n,
        reward_aligned=cfg.reward_aligned, result_mode=model.result_mode,
        byte_exact_before=f"{ok0}/{n0}", byte_exact_after=f"{ok_f}/{n_f}",
        pct_before=100.0*ok0/n0, pct_after=100.0*ok_f/n_f,
        min_margin_before=mm0, min_margin_after=mm_f,
        graft_weight_drift=dr_f, max_rl_grad_into_graft=max_graft_gn_rl,
        final_reward_mean=last_reward, final_w_read=float(w_read.detach()),
        survived=survived, trajectory=traj)
    # free the optimizer state (AdamW exp_avg/exp_avg_sq over all host params ~ several
    # GB) + grads before the next regime, so peak RSS does not accumulate across regimes
    for p in params:
        p.grad = None
    del opt
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# =========================================================================== #
#  MAIN                                                                         #
# =========================================================================== #
def main():
    t0 = time.time()
    quick = os.environ.get("GRAFT_RL_QUICK", "0") == "1"
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    MID = "Qwen/Qwen2.5-0.5B-Instruct"
    dev_env = os.environ.get("GRAFT_RL_DEVICE", "")
    # VRAM-aware device pick: full-model AdamW needs ~11 GB VRAM (params 2 + grads 2 +
    # adam 4 + activations ~3). Pick a GPU ONLY if it has that headroom, so we never OOM
    # another agent's job; else CPU (RSS stays ~4 GB on GPU runs, ~10 GB on CPU runs).
    NEED_VRAM_GB = 11.5
    if dev_env:
        dev = dev_env
    elif torch.cuda.is_available():
        dev = "cpu"
        best_free = 0.0
        for gi in range(torch.cuda.device_count()):
            free_b, _tot = torch.cuda.mem_get_info(gi)
            free_gb = free_b / (1024 ** 3)
            if free_gb > NEED_VRAM_GB and free_gb > best_free:
                best_free = free_gb; dev = f"cuda:{gi}"
        print(f"  VRAM scan: chose {dev} (need >{NEED_VRAM_GB} GB free; "
              f"best_free={best_free:.1f} GB)")
    else:
        dev = "cpu"
    global OPTIMIZER_NAME, REWARD_BS, W_READ_INIT
    OPTIMIZER_NAME = os.environ.get("GRAFT_RL_OPT",
                                    "adamw" if dev.startswith("cuda") else "sgd")
    REWARD_BS = int(os.environ.get("GRAFT_RL_REWARD_BS",
                                   "320" if dev.startswith("cuda") else "64"))
    W_READ_INIT = float(os.environ.get("GRAFT_RL_WREAD", "4.0"))
    print(f"Host = {MID}   device = {dev}   quick={quick}   optimizer = {OPTIMIZER_NAME}"
          f"   reward_bs = {REWARD_BS}   w_read_init = {W_READ_INIT}")

    tok = AutoTokenizer.from_pretrained(MID)
    host = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32)
    poll_rss()
    host = host.to(dev)
    host_params = sum(p.numel() for p in host.parameters())
    print(f"  host params = {host_params/1e6:.2f}M   RSS after load = {rss_gb():.2f} GB")

    results = dict(host=MID, device=dev, optimizer=OPTIMIZER_NAME, reward_bs=REWARD_BS,
                   w_read_init=W_READ_INIT,
                   host_params=host_params, temp=30.0, margin_m=MARGIN_M,
                   ref_914=dict(optimizer="adamw", shared_dim_break_lr=3e-3, held_le_lr=1e-3,
                                note="#914 (AdamW): broke at lr 3e-3 (13%%), margin negative; "
                                     "only FREEZE bulletproof; host LM grad through the "
                                     "shared RESULT dim overwhelms the margin term. This "
                                     "run re-derives the shared/agnostic break WITHIN its "
                                     "optimizer as the in-experiment #914-repro baseline."))

    # VM op eval set (bigger => stronger byte-exact claim; 320 like #914)
    n_per_op = 16 if quick else 40
    vm_data = make_dataset(n_per_op, seed=1)

    # 5-prompt chat batch (trims the full-vocab 151936 logits activation vs 10 prompts,
    # keeps peak RSS under the boundary while still a genuine nonzero LM fine-tune signal)
    ii, ll, am = build_sft_batch(tok, dev, prompts=HOST_PROMPTS[:5])

    # -------- lr sweep THROUGH and PAST #914's 3e-3 break --------
    # Includes 3e-3 (the #914 AdamW break lr) as an anchor, and extends UP so the
    # shared/agnostic (#914-repro) arm actually breaks within this optimizer -> that
    # break lr is the in-experiment baseline the other arms are compared to.
    lrs = [3e-3] if quick else [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    steps = 40 if quick else 120
    group_n = 4 if quick else 8
    probe_every = 10 if quick else 20
    # env overrides for a fast targeted check without changing the quick/full branches
    if os.environ.get("GRAFT_RL_LRS"):
        lrs = [float(x) for x in os.environ["GRAFT_RL_LRS"].split(",")]
    if os.environ.get("GRAFT_RL_STEPS"):
        steps = int(os.environ["GRAFT_RL_STEPS"]); probe_every = max(1, steps // 4)

    # 2x2 ablation arms x lr sweep
    arms = [
        ("shared",    True),    # shared dim + reward-aligned
        ("shared",    False),   # shared dim + reward-agnostic (reproduce #914 break)
        ("decoupled", True),    # decoupled dim + reward-aligned
        ("decoupled", False),   # decoupled dim + reward-agnostic
    ]
    if quick:
        arms = [("shared", True), ("shared", False),
                ("decoupled", True), ("decoupled", False)]

    all_runs = []
    # build ONE grafted host per result_mode (rebuilt to switch architecture);
    # reuse across lrs/reward-modes by restoring pristine each regime.
    for result_mode in ("shared", "decoupled"):
        model = GraftedHost(host, temp=30.0, result_mode=result_mode).to(dev)
        # trainable readout gain: the answer-byte policy logits = w_read*(2v*RESULT - v^2).
        # w_read is the policy's CONCENTRATION on the ALU: at w_read=1 the softmax over 256
        # bytes only samples the exact byte ~56% of the time even when RESULT is byte-exact
        # (so reward ~0.56 and the noisy REINFORCE advantage keeps PUSHING RESULT -> the
        # reward is NOT a near-optimum at the attractor). At w_read>=4 the policy is ~96%+
        # concentrated -> a byte-exact RESULT gives reward ~1.0, advantage ~0, and the
        # reward gradient VANISHES: the byte-exact point becomes a REWARD-OPTIMUM. This is
        # the mechanism the hypothesis needs, so we init w_read CONCENTRATED (=4.0) and
        # keep it LEARNABLE (the policy can sharpen further or drift). "Learned to use it"
        # = reward stays ~1.0 (the policy trusts the ALU); reward-alignment holding it =
        # near-zero graft gradient at the byte-exact reward-optimum. w_read init is a
        # policy hyperparameter, NOT a training-time protection of the graft.
        w_read = nn.Parameter(torch.tensor(W_READ_INIT, device=dev))

        feats, oph, vtgt = build_feats(vm_data, dev)
        base_ids = torch.zeros(feats.shape[0], 1, dtype=torch.long, device=dev)

        # sanity: byte-exact at construction (before any RL) for THIS architecture
        ok_c, n_c, mm_c = vm_byte_exact(model.alu, feats, oph, vtgt)
        per_op = {}
        for op in OPS:
            idx = torch.tensor([i for i, s in enumerate(vm_data) if s.op == op], device=dev)
            o2, n2, _ = vm_byte_exact(model.alu, feats[idx], oph[idx], vtgt[idx])
            per_op[op] = f"{o2}/{n2}"
        print(f"\n##### ARCHITECTURE = {result_mode.upper()}   "
              f"construction byte-exact = {ok_c}/{n_c} (min margin {mm_c:.3f})  per-op {per_op}")
        results[f"construction_{result_mode}"] = dict(ok=ok_c, n=n_c, min_margin=mm_c,
                                                      per_op=per_op)

        # host byte-identity with graft-off (graft is a true no-op off VM, both arms)
        with torch.no_grad():
            probe_ids = tok("The capital of France is", return_tensors="pt").to(dev)
            model.vm_mode = False
            g_off = model.host(**probe_ids).logits
            model._hook.remove()
            stock = model.host(**probe_ids).logits
            model._hook = model._last_layer.register_forward_hook(model._graft_hook)
            max_abs = float((g_off - stock).abs().max().item())
        print(f"      host logits graft-off vs stock: max|diff|={max_abs:.2e} "
              f"(graft is a true no-op when vm_mode=False -> expect 0)")
        results[f"host_graftoff_maxdiff_{result_mode}"] = max_abs

        # pristine snapshots for restore-between-regimes. W0 (graft, tiny) stays in RAM;
        # host0 goes to DISK (a second in-RAM copy of the 494M host would push peak RSS
        # over the safety boundary alongside the full-model AdamW state).
        import tempfile
        W0 = {k: v.detach().clone() for k, v in model.alu.state_dict().items()}
        host0_f = tempfile.NamedTemporaryFile(
            suffix=f"_host0_{result_mode}.pt", delete=False,
            dir=os.environ.get("GRAFT_RL_TMP", "/tmp"))
        host0_f.close()
        torch.save({k: v.detach().cpu().clone() for k, v in model.host.state_dict().items()},
                   host0_f.name)
        wread0 = w_read.detach().clone()

        for (mode_arch, reward_aligned) in arms:
            if mode_arch != result_mode:
                continue
            for lr in lrs:
                cfg = RLConfig(lr=lr, steps=steps, group_n=group_n,
                               reward_aligned=reward_aligned,
                               label=f"{result_mode}/{'reward' if reward_aligned else 'agnostic'}/lr{lr:.0e}")
                print(f"\n  --- RL regime: {cfg.label}  (UNPROTECTED: all params trainable,"
                      f" NO freeze/lr-cap/mask/stop-grad/EMA) ---")
                run = run_rl_regime(model, w_read, feats, oph, vtgt, base_ids,
                                    ii, ll, am, cfg, W0, host0_f.name, wread0, probe_every)
                all_runs.append(run)
                # trajectory print
                for pt in run["trajectory"]:
                    rm = pt["reward_mean"]
                    print(f"       step {pt['step']:>4}: byte-exact {pt['byte_exact']:>7} "
                          f"({pt['pct']:5.1f}%)  min-margin {pt['min_margin']:+.3f}  "
                          f"graft|g|={pt['graft_grad_norm']:.2e}  drift={pt['drift']:.2e}  "
                          f"w_read={pt['w_read']:+.3f}  "
                          f"reward={('%.3f'%rm) if rm is not None else '  -  '}")
                print(f"       => SURVIVED byte-exact: {run['survived']}   "
                      f"({run['byte_exact_before']} -> {run['byte_exact_after']}, "
                      f"margin {run['min_margin_before']:.3f} -> {run['min_margin_after']:.3f}, "
                      f"RL-grad into graft max={run['max_rl_grad_into_graft']:.2e}, "
                      f"final reward={run['final_reward_mean']})")
                poll_rss()

        # free this architecture's model + disk snapshot before building the next
        model._hook.remove()
        del model
        try:
            os.unlink(host0_f.name)
        except OSError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["rl_runs"] = all_runs

    # =============================== ABLATION SUMMARY ======================== #
    print("\n################ ABLATION SUMMARY (survival by arm x lr) ###########")
    def find(mode, reward, lr):
        for r in all_runs:
            if r["result_mode"] == mode and r["reward_aligned"] == reward and abs(r["lr"] - lr) < 1e-12:
                return r
        return None

    grid_rows = []
    for mode in ("shared", "decoupled"):
        for reward in (True, False):
            row = {"result_mode": mode, "reward_aligned": reward, "by_lr": {}}
            for lr in lrs:
                r = find(mode, reward, lr)
                if r:
                    row["by_lr"][f"{lr:.0e}"] = dict(
                        survived=r["survived"], pct_after=r["pct_after"],
                        margin_after=r["min_margin_after"], drift=r["graft_weight_drift"],
                        final_reward=r["final_reward_mean"])
            grid_rows.append(row)
            arm = f"{mode:<9} {'reward ' if reward else 'agnostic'}"
            cells = "  ".join(
                f"lr{lr:.0e}:{'OK ' if row['by_lr'].get(f'{lr:.0e}',{}).get('survived') else 'BRK'}"
                f"({row['by_lr'].get(f'{lr:.0e}',{}).get('pct_after',0):5.1f}%"
                f",m={row['by_lr'].get(f'{lr:.0e}',{}).get('margin_after',0):+.2f})"
                for lr in lrs)
            print(f"  {arm} | {cells}")
    results["ablation_grid"] = grid_rows

    # highest lr at which each arm still holds byte-exact
    def highest_hold_lr(mode, reward):
        held = [r["lr"] for r in all_runs
                if r["result_mode"] == mode and r["reward_aligned"] == reward and r["survived"]]
        return max(held) if held else None
    def first_break_lr(mode, reward):
        broke = sorted([r["lr"] for r in all_runs
                        if r["result_mode"] == mode and r["reward_aligned"] == reward and not r["survived"]])
        return broke[0] if broke else None

    holds = {}
    for mode in ("shared", "decoupled"):
        for reward in (True, False):
            key = f"{mode}/{'reward' if reward else 'agnostic'}"
            holds[key] = dict(highest_hold_lr=highest_hold_lr(mode, reward),
                              first_break_lr=first_break_lr(mode, reward))
    results["holds"] = holds

    # ================================ VERDICT =============================== #
    print("\n############################### VERDICT ###########################")
    sh_ag = holds["shared/agnostic"]        # reproduces #914's break case
    sh_rw = holds["shared/reward"]          # reward-alignment ALONE, shared dim
    dc_rw = holds["decoupled/reward"]       # architecture + reward
    dc_ag = holds["decoupled/agnostic"]     # architecture alone

    # did the model LEARN TO USE the ALU (reward rose / stayed high in reward arms)?
    learned = {}
    for r in all_runs:
        if r["reward_aligned"]:
            learned[r["label"]] = r["final_reward_mean"]
    # a representative "learned to use it" at the moderate lr
    ref_reward_run = find("decoupled", True, lrs[0])
    learned_to_use = (ref_reward_run is not None and
                      ref_reward_run["final_reward_mean"] is not None and
                      ref_reward_run["final_reward_mean"] > 0.9)

    print(f"  (learn-to-use) reward-arm final reward by regime: "
          f"{ {k: round(v,3) if v is not None else None for k,v in learned.items()} }")
    print(f"  SHARED  + AGNOSTIC (== #914 break case): holds<= {sh_ag['highest_hold_lr']}, "
          f"first break {sh_ag['first_break_lr']}")
    print(f"  SHARED  + REWARD  (reward-align ALONE):  holds<= {sh_rw['highest_hold_lr']}, "
          f"first break {sh_rw['first_break_lr']}")
    print(f"  DECOUPLED + REWARD (architecture+reward): holds<= {dc_rw['highest_hold_lr']}, "
          f"first break {dc_rw['first_break_lr']}")
    print(f"  DECOUPLED + AGNOSTIC (architecture alone): holds<= {dc_ag['highest_hold_lr']}, "
          f"first break {dc_ag['first_break_lr']}")

    # break-lr helper (a break lr of None on the swept grid == held at ALL swept lrs, i.e.
    # a very high effective robustness; treat as +inf for comparisons). Returns a JSON-safe
    # ordinal verdict: +1 => B more robust than A (higher break lr), -1 => less, 0 => equal.
    def brkval(d):
        return d["first_break_lr"] if d["first_break_lr"] is not None else float("inf")
    def cmp_robust(dA, dB):
        a, b = brkval(dA), brkval(dB)
        return "B_more_robust" if b > a else ("A_more_robust" if a > b else "equal")

    verdict = dict(
        learned_to_use_alu=learned_to_use,
        # does ADDING the reward objective RAISE (help) or LOWER (hurt) the break lr,
        # holding the architecture fixed? A=agnostic, B=reward -> "B_more_robust" = reward
        # HELPS; "A_more_robust" = reward HURTS.
        reward_vs_agnostic_shared=cmp_robust(sh_ag, sh_rw),
        reward_vs_agnostic_decoupled=cmp_robust(dc_ag, dc_rw),
        # does DECOUPLING (architecture) raise the break lr, holding the reward mode fixed?
        # A=shared, B=decoupled -> "B_more_robust" = decoupling HELPS.
        decoupling_shared_vs_decoupled_reward=cmp_robust(sh_rw, dc_rw),
        decoupling_shared_vs_decoupled_agnostic=cmp_robust(sh_ag, dc_ag),
        reward_alignment_lifts_break=(
            (sh_rw["first_break_lr"] is None) or
            (sh_ag["first_break_lr"] is not None and
             (sh_rw["first_break_lr"] is None or sh_rw["first_break_lr"] > sh_ag["first_break_lr"]))),
        shared_reward_holds_at_3e3=any(
            r["survived"] for r in all_runs
            if r["result_mode"] == "shared" and r["reward_aligned"] and abs(r["lr"] - 3e-3) < 1e-12),
        decoupled_reward_holds_at_3e3=any(
            r["survived"] for r in all_runs
            if r["result_mode"] == "decoupled" and r["reward_aligned"] and abs(r["lr"] - 3e-3) < 1e-12),
        # unprotected survival at ANY architecture/reward mode, at each swept lr
        survives_unprotected_by_lr={
            f"{lr:.0e}": {r["label"].split('/')[0]+'/'+r["label"].split('/')[1]: r["survived"]
                         for r in all_runs if abs(r["lr"] - lr) < 1e-12}
            for lr in lrs},
        holds=holds,
    )
    results["verdict"] = verdict
    print(f"\n  VERDICT dict: {json.dumps(verdict, indent=2, default=lambda o: (float(o) if isinstance(o,(np.floating,)) else bool(o) if isinstance(o,(np.bool_,bool)) else o))}")

    results["peak_rss_gb"] = PEAK_RSS
    results["elapsed_s"] = time.time() - t0
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_agent_graft_rl_unprotected_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"\nWrote {out}   (elapsed {time.time()-t0:.1f}s, PEAK RSS {PEAK_RSS:.2f} GB)")


if __name__ == "__main__":
    main()
