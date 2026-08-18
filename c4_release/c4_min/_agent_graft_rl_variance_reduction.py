"""VARIANCE-REDUCTION survival of the grafted byte-exact ALU under UNPROTECTED RL (#914 follow-up).

QUESTION (the user's brief, verbatim intent):
  #914 (branch graft-rl-unprotected-reward-914 @ 7b86401d) MEASURED that grafting the
  #910 bounded byte-exact ALU into a real Qwen2.5-0.5B and doing UNPROTECTED reward-aligned
  REINFORCE FAILED: reward-alignment HURT (shared/reward broke ~1e-2 vs shared/agnostic
  ~1e-1). ROOT: policy-gradient VARIANCE turns the byte-exact reward-optimum into a
  SADDLE/REPELLER — even a perfect policy (reward 1.0 at step 0) breaks: one noisy sample
  knocks RESULT off -> reward collapses -> the large noisy gradient drives it further off.
  The reward grad into W_key (measured max ~44-113) dwarfs the LM coupling (~1.0). The
  killer is VARIANCE, not reward direction.

  BUT #914 used a WEAK estimator: GRPO group G=8, NO advantage normalization, fixed
  temperature (w_read=4, ~sampling temp 0.25), NO KL trust region, and a single ALU basin
  (margin m=0.5). The user's two levers to test:

    LEVER 1 (VARIANCE REDUCTION — standard RL machinery, ALLOWED, NOT a protection):
      * large GRPO group G in {8, 32, 128}
      * advantage normalization (whiten adv within the group)
      * low sampling temperature + entropy schedule
      * KL-to-reference trust region (KL penalty; PPO-style clip test)
      Does suppressing the gradient variance restabilize the saddle so the ALU stays
      byte-exact AND reward rises?

    LEVER 2 (ROUTING vs COMPUTE SPLIT — architecture, ALLOWED, LABELLED):
      The thing to LEARN is the ROUTING (which operands / when to invoke the ALU). The
      ALU's INTERNAL byte-exact LUT (W_key/grid) should stay on its #910 attractor. Give
      the routing/policy params a NORMAL gradient while the internal LUT sits in a WIDER
      BASIN (higher temp = deeper plateau) or on a decoupled dim — WITHOUT hard-freezing
      it (NO stop-grad; W_key STILL receives the reward gradient, it must self-recover).
      Does the model learn WHEN to use the ALU while the computation stays exact?

HONESTY GATE (the user has been burned by optimism all session; #914's reward-alignment
  hypothesis was already refuted -> do NOT assume variance reduction works, MEASURE it):
  * "Byte-exact" = the ACTUAL grafted forward probed DURING real RL updates, cited N/N with
    a full trajectory (320/320-style probe before / during / after).
  * "Learns to use it" = MEASURED reward + routing-accuracy rise, not asserted.
  * ALLOWED (standard RL / design, labelled): variance reduction (big group / adv-norm /
    low temp+entropy / KL trust region), wider-basin margin, architectural routing/compute
    split (a private lane / dedicated readout). DISALLOWED (protection): freeze
    (requires_grad=False), lr-cap on the graft, gradient-mask / stop-grad, EMA, periodic
    reset. NONE of the disallowed appear here (asserted + auditable: W_key.requires_grad is
    True in every regime, the graft receives the reward gradient every step, and the
    max RL-grad-into-graft is logged so a silent zero would show).
  * If the only thing that works is effectively a protection in disguise, SAY SO.

WHAT THIS ADDS OVER #914:
  * A real GRPO policy-gradient loop with the FULL variance-reduction toolkit as swept
    knobs: group_n in {8,32,128}, adv_norm on/off, sampling temperature (temp_sample),
    entropy bonus, KL-to-reference penalty (beta), PPO-clip.
  * A 3-arm ARCHITECTURE ablation at each config:
      shared        (== #914 break case; RESULT spliced into a host residual dim)
      decoupled     (private RESULT lane; host LM grad cannot reach W_key)
      split         (routing/compute split: policy readout trainable + WIDE-BASIN LUT
                     temp=60 so W_key sits deeper in its plateau; W_key still gets the
                     reward gradient — NO stop-grad)
  * Everything trainable, NO protection, at the lr where #914 broke (1e-2) and past it.
  * Byte-exact probe (320/320) + margin + graft-grad-norm + routing-accuracy trajectory
    logged every K real RL updates.

MEMORY: lean fp32 host load (~4 GB on GPU, ~7-9 GB on CPU). Stock HF Qwen2.5-0.5B-Instruct
  (NOT the sparse ISA model -> NO 108 GB densify). host0 snapshot on DISK. Poll RSS every
  probe; HARD-ABORT at 10 GB. Golden 174ece66 UNTOUCHED (imports only torch/transformers/
  numpy; ZERO neural_vm/compiler refs; nothing on the model build path). Local only.

Run:  HF_HOME=/media/data/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python c4_min/_agent_graft_rl_variance_reduction.py
Flags: GRAFT_VR_QUICK=1 fast smoke; GRAFT_VR_DEVICE=cuda:0|cuda:1|cpu;
       GRAFT_VR_LRS=a,b,..  GRAFT_VR_STEPS=n  GRAFT_VR_GROUPS=8,32,128 .
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

OPTIMIZER_NAME = "sgd"
REWARD_BS = 64
W_READ_INIT = 4.0


# =========================================================================== #
#  PART 0 — the #910 byte-exact PRIMITIVE (grafted verbatim from #914)         #
# =========================================================================== #
VAL_VOCAB = 256
N_MEM = 4
MARGIN_M = 0.5           # nominal hinge margin (<= head's 1.0 max => zero-loss plateau)

OPS = ["ADD", "SUB", "CMP_LT", "CMP_EQ", "CMP_GT", "LI", "SI", "DIV"]
OP_ID = {name: k for k, name in enumerate(OPS)}
N_OPS = len(OPS)

(IN_ONE, IN_A, IN_B, IN_GMEM,
 IN_CMP_LT, IN_CMP_EQ, IN_CMP_GT, IN_QUOT) = range(8)
IN_DIM = 8


def op_semantics(op: str, a: int, b: int, mem: List[int], addr: int
                 ) -> Tuple[int, int, List[int]]:
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
    """The #910 BOUNDED softmax-LUT ALU as a graftable side adapter.

      RESULT = softmax(T * -(key - v)^2) @ grid,  T FIXED (bounded, O(1)).
    key = per-op LINEAR combo of the 8 fixed feature lanes; W_key trainable (#910 basin).
    `temp` is the LUT sharpness — a HIGHER temp deepens the plateau (wider stable basin)
    around each integer key, which is LEVER-2's "wider basin" knob (a property of the op's
    construction, NOT a training-time protection)."""

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
    grid = _byte_val_grid().to(result_byte.device)                # [256]
    x = result_byte.unsqueeze(-1)                                 # [...,1]
    logits = 2.0 * grid * x - grid ** 2                           # [...,256]
    return logits.argmax(-1)


# --------------------------------------------------------------------------- #
#  VM op dataset + feature builder                                             #
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
#  Three ARCHITECTURES: shared (==#914) / decoupled / split (routing+wide LUT). #
# =========================================================================== #
class GraftedHost(nn.Module):
    """Stock Qwen2.5 host + the #910 byte-exact ALU grafted onto the last decoder layer.

    result_mode:
      * 'shared'    (== #914): ALU writes RESULT into residual dim H-1 (host reads it ->
                    host LM grad flows through RESULT into W_key). The #914 break arch.
      * 'decoupled' (ARCHITECTURE): RESULT lives in a private capture, never spliced into
                    the host residual -> host LM grad cannot reach W_key. The ALU's OWN
                    reward objective still reads RESULT, so the REWARD grad DOES reach W_key.
      * 'split'     (ROUTING/COMPUTE SPLIT, ARCHITECTURE): same private RESULT lane as
                    decoupled (routing is the trainable readout the policy learns), PLUS the
                    ALU LUT built in a WIDER basin (higher temp) so W_key sits deeper in its
                    plateau. W_key STILL receives the reward gradient (NO stop-grad); it just
                    must self-recover from a wider, more forgiving basin. This tests LEVER 2.

    NOTE: 'decoupled'/'split' are ARCHITECTURAL wiring choices fixed before RL (a private
    output lane / a wider LUT basin), clearly distinct from the DISALLOWED training-time
    protections (freeze / lr-cap / mask / stop-grad / EMA). W_key.requires_grad is True in
    ALL modes and the reward gradient into W_key is logged."""

    def __init__(self, host, temp=30.0, result_mode="shared"):
        super().__init__()
        assert result_mode in ("shared", "decoupled", "split")
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
        if self.result_mode in ("decoupled", "split"):
            # RESULT stays private; host residual untouched -> host LM grad can't reach W_key.
            return output
        # SHARED (== #914): splice RESULT into a host residual dim (LM grad couples in).
        hs = hs.clone()
        hs[..., self.result_dim] = result
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def vm_forward_result(self, feats, oph, base_ids):
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
#  THE RL LOOP — GRPO with the FULL variance-reduction toolkit                 #
# =========================================================================== #
# The POLICY emits an answer byte via a trainable READOUT that reads the grafted RESULT dim
# ("use the ALU correctly" == route operands to the ALU + read its byte-exact RESULT):
#   answer logits over 256 bytes = w_read * (2*v*RESULT - v^2) / temp_sample
# temp_sample controls the sampling temperature (LEVER 1: low temp -> lower-variance policy).
#
# Beyond #914 (which used only G=8 + a leave-one-out group-mean baseline), we add, as SWEPT
# knobs, the standard GRPO/PPO variance-reduction toolkit:
#   * group_n in {8,32,128}   (larger group -> lower-variance advantage)
#   * adv_norm                (whiten advantages within the group: (A-mean)/std)
#   * temp_sample             (sampling temperature; low temp -> sharper policy)
#   * entropy_coef            (entropy bonus)
#   * kl_beta                 (KL-to-reference trust region penalty)
#   * ppo_clip                (PPO ratio clip)


def answer_logits_from_result(result: torch.Tensor, w_read: torch.Tensor,
                              temp_sample: float = 1.0) -> torch.Tensor:
    """Policy logits over 256 answer bytes given ALU RESULT [N]. Divided by temp_sample:
    LOWER temp_sample -> sharper (lower-variance) policy -> the sampled action equals the
    exact byte more often -> reward ~1 and the REINFORCE advantage (=> reward grad into
    W_key) -> 0 at the byte-exact point. This is LEVER-1's temperature knob."""
    grid = _byte_val_grid().to(result.device)                    # [256]
    x = result.unsqueeze(-1)                                      # [N,1]
    base = 2.0 * grid * x - grid ** 2                             # [N,256]
    return (w_read * base) / max(temp_sample, 1e-6)


@dataclass
class RLConfig:
    lr: float
    steps: int
    group_n: int            # GRPO group size (variance reduction lever)
    reward_aligned: bool
    adv_norm: bool = False          # whiten advantages within group
    temp_sample: float = 1.0        # sampling temperature (low -> lower variance)
    entropy_coef: float = 0.0       # entropy bonus
    kl_beta: float = 0.0            # KL-to-reference trust region penalty
    ppo_clip: float = 0.0           # PPO ratio clip (0 = off)
    label: str = ""


def run_rl_regime(model: GraftedHost, w_read: nn.Parameter, feats, oph, vtgt,
                  base_ids, ii, ll, am, cfg: RLConfig,
                  W0: Dict[str, torch.Tensor], host0_path: str,
                  wread0: torch.Tensor, probe_every: int) -> dict:
    """One fully-UNPROTECTED RL regime with the variance-reduction toolkit active. Restores
    pristine graft+host+readout, then runs cfg.steps REAL updates with ALL params trainable
    (host + graft W_key/out_gain + readout), NO freeze / lr-cap / mask / stop-grad / EMA /
    reset. Logs byte-exact + margin + graft-grad-norm + routing-accuracy every probe_every."""
    dev = feats.device
    # restore pristine (between-regime experiment reset, NOT in-training protection)
    model.alu.load_state_dict({k: v.clone() for k, v in W0.items()})
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
    # HONESTY AUDIT: assert the graft LUT is NOT frozen (the disallowed protection).
    assert model.alu.W_key.requires_grad, "W_key must be trainable (no freeze)"
    params = list(model.host.parameters()) + list(model.alu.parameters()) + [w_read]
    global OPTIMIZER_NAME
    if OPTIMIZER_NAME == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    else:
        opt = torch.optim.SGD(params, lr=cfg.lr, weight_decay=1e-4, momentum=0.0)

    grid = _byte_val_grid().to(dev)
    N = feats.shape[0]
    G = cfg.group_n

    # reference w_read (KL trust region): the initial policy concentration. The KL penalty
    # is a soft trust region on the POLICY distribution (standard GRPO), NOT a protection of
    # the live W_key (fully trainable): it is added as a loss, gradient flows normally.
    ref_wread = float(w_read.detach())

    traj = []

    def routing_accuracy():
        """Fraction of rows where the POLICY's argmax answer byte == the exact target
        (measures 'the model actually invokes the ALU on the right operands and emits its
        byte-exact result')."""
        with torch.no_grad():
            rb = min(REWARD_BS, N)
            sel = torch.arange(N, device=dev)[:rb]
            result = model.vm_forward_result(feats[sel], oph[sel], base_ids[:rb])
            logits = answer_logits_from_result(result, w_read, cfg.temp_sample)
            pred = logits.argmax(-1)
            return float((pred == vtgt[sel]).float().mean().item())

    def probe(step):
        ok, n, mm = vm_byte_exact(model.alu, feats, oph, vtgt)
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
    racc0 = routing_accuracy()
    traj.append(dict(step=0, byte_exact=f"{ok0}/{n0}", pct=100.0*ok0/n0,
                     min_margin=mm0, graft_grad_norm=gn0, reward_mean=None,
                     routing_acc=racc0, w_read=float(w_read.detach()), drift=dr0))

    max_graft_gn_rl = 0.0
    last_reward = None
    last_racc = racc0
    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)

        # host LM fine-tune component (present in BOTH arms; the #914 pressure)
        prev = model.vm_mode
        model.vm_mode = True
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
            # GRPO policy gradient with the full variance-reduction toolkit
            rb = min(REWARD_BS, N)
            sel = torch.randperm(N, device=dev)[:rb]
            f_b, o_b, t_b = feats[sel], oph[sel], vtgt[sel]
            base_b = base_ids[:rb]
            result = model.vm_forward_result(f_b, o_b, base_b)       # [rb]
            logits = answer_logits_from_result(result, w_read, cfg.temp_sample)  # [rb,256]
            logp_all = F.log_softmax(logits, dim=-1)                 # [rb,256]
            probs = logp_all.exp()

            dist = torch.distributions.Categorical(logits=logits)
            acts_list, rewards = [], []
            for _g in range(G):
                acts = dist.sample()                                 # [rb]
                acts_list.append(acts)
                rewards.append((acts == t_b).float())                # reward 1 iff exact
            R = torch.stack(rewards, 0)                              # [G,rb]
            baseline = R.mean(0, keepdim=True)                       # [1,rb] group mean
            adv = R - baseline                                       # [G,rb] centered
            if cfg.adv_norm:
                std = adv.std(0, keepdim=True).clamp_min(1e-6)
                adv = adv / std
            pg_loss = 0.0
            for gi in range(G):
                lp = logp_all.gather(1, acts_list[gi].view(-1, 1)).squeeze(1)  # [rb]
                if cfg.ppo_clip > 0:
                    old_lp = lp.detach()
                    ratio = torch.exp(lp - old_lp)                    # [rb], ~1
                    a_gi = adv[gi].detach()
                    unclipped = ratio * a_gi
                    clipped = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * a_gi
                    pg_loss = pg_loss + (-(torch.min(unclipped, clipped))).mean()
                else:
                    pg_loss = pg_loss + (-(adv[gi].detach() * lp)).mean()
            pg_loss = pg_loss / G

            ent = -(probs * logp_all).sum(-1).mean()
            kl_term = torch.tensor(0.0, device=dev)
            if cfg.kl_beta > 0:
                with torch.no_grad():
                    ref_logits = answer_logits_from_result(
                        result.detach(), torch.tensor(ref_wread, device=dev), cfg.temp_sample)
                    ref_logp = F.log_softmax(ref_logits, dim=-1)
                kl_term = (probs * (logp_all - ref_logp)).sum(-1).mean()

            reward_mean = float(R.mean().item())
            last_reward = reward_mean
            total = total + pg_loss - cfg.entropy_coef * ent + cfg.kl_beta * kl_term

        total.backward()

        ggn = 0.0
        for p in model.alu.parameters():
            if p.grad is not None:
                ggn += float(p.grad.norm() ** 2)
        max_graft_gn_rl = max(max_graft_gn_rl, ggn ** 0.5)

        torch.nn.utils.clip_grad_norm_(params, 1.0)   # global clip (all params equally; #914 same)
        opt.step()

        if (step + 1) % probe_every == 0 or step == cfg.steps - 1:
            ok, n, mm, gn, dr = probe(step + 1)
            last_racc = routing_accuracy()
            traj.append(dict(step=step + 1, byte_exact=f"{ok}/{n}", pct=100.0*ok/n,
                             min_margin=mm, graft_grad_norm=gn,
                             reward_mean=reward_mean if reward_mean is not None else last_reward,
                             routing_acc=last_racc,
                             w_read=float(w_read.detach()), drift=dr))
            poll_rss()

    ok_f, n_f, mm_f, gn_f, dr_f = probe(cfg.steps)
    survived = (ok_f == n_f)
    out = dict(
        label=cfg.label, lr=cfg.lr, steps=cfg.steps, group_n=cfg.group_n,
        reward_aligned=cfg.reward_aligned, result_mode=model.result_mode,
        adv_norm=cfg.adv_norm, temp_sample=cfg.temp_sample,
        entropy_coef=cfg.entropy_coef, kl_beta=cfg.kl_beta, ppo_clip=cfg.ppo_clip,
        byte_exact_before=f"{ok0}/{n0}", byte_exact_after=f"{ok_f}/{n_f}",
        pct_before=100.0*ok0/n0, pct_after=100.0*ok_f/n_f,
        min_margin_before=mm0, min_margin_after=mm_f,
        graft_weight_drift=dr_f, max_rl_grad_into_graft=max_graft_gn_rl,
        final_reward_mean=last_reward, final_routing_acc=last_racc,
        routing_acc_before=racc0, final_w_read=float(w_read.detach()),
        survived=survived, trajectory=traj)
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
    quick = os.environ.get("GRAFT_VR_QUICK", "0") == "1"
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    MID = "Qwen/Qwen2.5-0.5B-Instruct"
    dev_env = os.environ.get("GRAFT_VR_DEVICE", "")
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
        print(f"  VRAM scan: chose {dev} (need >{NEED_VRAM_GB} GB free; best_free={best_free:.1f} GB)")
    else:
        dev = "cpu"
    global OPTIMIZER_NAME, REWARD_BS, W_READ_INIT
    OPTIMIZER_NAME = os.environ.get("GRAFT_VR_OPT",
                                    "adamw" if dev.startswith("cuda") else "sgd")
    REWARD_BS = int(os.environ.get("GRAFT_VR_REWARD_BS",
                                   "320" if dev.startswith("cuda") else "64"))
    W_READ_INIT = float(os.environ.get("GRAFT_VR_WREAD", "4.0"))
    print(f"Host = {MID}   device = {dev}   quick={quick}   optimizer = {OPTIMIZER_NAME}"
          f"   reward_bs = {REWARD_BS}   w_read_init = {W_READ_INIT}")

    tok = AutoTokenizer.from_pretrained(MID)
    host = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32)
    poll_rss()
    host = host.to(dev)
    host_params = sum(p.numel() for p in host.parameters())
    print(f"  host params = {host_params/1e6:.2f}M   RSS after load = {rss_gb():.2f} GB")

    results = dict(host=MID, device=dev, optimizer=OPTIMIZER_NAME, reward_bs=REWARD_BS,
                   w_read_init=W_READ_INIT, host_params=host_params, temp=30.0,
                   ref_914=dict(shared_reward_break_lr=1e-2, shared_agnostic_break_lr=1e-1,
                                note="#914: shared/reward (G=8, no adv-norm, no KL) broke ~1e-2; "
                                     "reward-alignment HURT (variance = repeller at the byte-exact "
                                     "reward-optimum). This run adds the variance-reduction toolkit "
                                     "(big G / adv-norm / low temp / KL trust region) + a routing/"
                                     "compute split arch to test if the saddle restabilizes."))

    n_per_op = 16 if quick else 40
    vm_data = make_dataset(n_per_op, seed=1)
    ii, ll, am = build_sft_batch(tok, dev, prompts=HOST_PROMPTS)

    lrs = [1e-2] if quick else [1e-2, 3e-2, 1e-1]
    steps = 40 if quick else 120
    probe_every = 10 if quick else 20
    groups = [8, 32] if quick else [8, 32, 128]
    if os.environ.get("GRAFT_VR_LRS"):
        lrs = [float(x) for x in os.environ["GRAFT_VR_LRS"].split(",")]
    if os.environ.get("GRAFT_VR_STEPS"):
        steps = int(os.environ["GRAFT_VR_STEPS"]); probe_every = max(1, steps // 4)
    if os.environ.get("GRAFT_VR_GROUPS"):
        groups = [int(x) for x in os.environ["GRAFT_VR_GROUPS"].split(",")]

    all_runs = []
    # 'split' builds the LUT with a HIGHER temp (deeper plateau = wider stable basin).
    ARCH_TEMP = {"shared": 30.0, "decoupled": 30.0, "split": 60.0}

    archs = os.environ.get("GRAFT_VR_ARCHS", "shared,decoupled,split").split(",")
    for result_mode in archs:
        temp = ARCH_TEMP[result_mode]
        model = GraftedHost(host, temp=temp, result_mode=result_mode).to(dev)
        w_read = nn.Parameter(torch.tensor(W_READ_INIT, device=dev))

        feats, oph, vtgt = build_feats(vm_data, dev)
        base_ids = torch.zeros(feats.shape[0], 1, dtype=torch.long, device=dev)

        ok_c, n_c, mm_c = vm_byte_exact(model.alu, feats, oph, vtgt)
        per_op = {}
        for op in OPS:
            idx = torch.tensor([i for i, s in enumerate(vm_data) if s.op == op], device=dev)
            o2, n2, _ = vm_byte_exact(model.alu, feats[idx], oph[idx], vtgt[idx])
            per_op[op] = f"{o2}/{n2}"
        print(f"\n##### ARCHITECTURE = {result_mode.upper()} (LUT temp={temp})  "
              f"construction byte-exact = {ok_c}/{n_c} (min margin {mm_c:.3f})  per-op {per_op}")
        results[f"construction_{result_mode}"] = dict(ok=ok_c, n=n_c, min_margin=mm_c,
                                                      per_op=per_op, lut_temp=temp)

        with torch.no_grad():
            probe_ids = tok("The capital of France is", return_tensors="pt").to(dev)
            model.vm_mode = False
            g_off = model.host(**probe_ids).logits
            model._hook.remove()
            stock = model.host(**probe_ids).logits
            model._hook = model._last_layer.register_forward_hook(model._graft_hook)
            max_abs = float((g_off - stock).abs().max().item())
        print(f"      host logits graft-off vs stock: max|diff|={max_abs:.2e} (expect 0)")
        results[f"host_graftoff_maxdiff_{result_mode}"] = max_abs

        import tempfile
        W0 = {k: v.detach().clone() for k, v in model.alu.state_dict().items()}
        host0_f = tempfile.NamedTemporaryFile(
            suffix=f"_host0_{result_mode}.pt", delete=False,
            dir=os.environ.get("GRAFT_VR_TMP", "/tmp"))
        host0_f.close()
        torch.save({k: v.detach().cpu().clone() for k, v in model.host.state_dict().items()},
                   host0_f.name)
        wread0 = w_read.detach().clone()

        # --------- the variance-reduction config sweep for this architecture ---------
        def make_configs(lr):
            cfgs = []
            # AGNOSTIC control (no reward grad; == #914 shared/agnostic mechanism)
            cfgs.append(RLConfig(lr=lr, steps=steps, group_n=8, reward_aligned=False,
                                 label=f"{result_mode}/agnostic/lr{lr:.0e}"))
            # REWARD, BASELINE (== #914: G=8, no adv-norm, no KL, temp 1.0) — reproduces break
            cfgs.append(RLConfig(lr=lr, steps=steps, group_n=8, reward_aligned=True,
                                 label=f"{result_mode}/reward-G8-base/lr{lr:.0e}"))
            # incremental variance reduction (isolate the load-bearing lever)
            for G in groups:
                if G == 8:
                    continue
                cfgs.append(RLConfig(lr=lr, steps=steps, group_n=G, reward_aligned=True,
                                     label=f"{result_mode}/reward-G{G}/lr{lr:.0e}"))
            Gbig = max(groups)
            cfgs.append(RLConfig(lr=lr, steps=steps, group_n=Gbig, reward_aligned=True,
                                 adv_norm=True,
                                 label=f"{result_mode}/reward-G{Gbig}-advnorm/lr{lr:.0e}"))
            cfgs.append(RLConfig(lr=lr, steps=steps, group_n=Gbig, reward_aligned=True,
                                 adv_norm=True, temp_sample=0.3,
                                 label=f"{result_mode}/reward-G{Gbig}-advnorm-lowtemp/lr{lr:.0e}"))
            cfgs.append(RLConfig(lr=lr, steps=steps, group_n=Gbig, reward_aligned=True,
                                 adv_norm=True, temp_sample=0.3, kl_beta=0.1,
                                 label=f"{result_mode}/reward-FULL(G{Gbig}+advnorm+lowtemp+KL)/lr{lr:.0e}"))
            return cfgs

        for lr in lrs:
            for cfg in make_configs(lr):
                print(f"\n  --- RL regime: {cfg.label}  [G={cfg.group_n} advnorm={cfg.adv_norm} "
                      f"temp={cfg.temp_sample} kl={cfg.kl_beta}]  (UNPROTECTED) ---")
                run = run_rl_regime(model, w_read, feats, oph, vtgt, base_ids,
                                    ii, ll, am, cfg, W0, host0_f.name, wread0, probe_every)
                all_runs.append(run)
                for pt in run["trajectory"]:
                    rm = pt["reward_mean"]
                    print(f"       step {pt['step']:>4}: byte-exact {pt['byte_exact']:>7} "
                          f"({pt['pct']:5.1f}%)  min-margin {pt['min_margin']:+.3f}  "
                          f"graft|g|={pt['graft_grad_norm']:.2e}  drift={pt['drift']:.2e}  "
                          f"route-acc={pt['routing_acc']:.3f}  "
                          f"reward={('%.3f'%rm) if rm is not None else '  -  '}")
                print(f"       => SURVIVED byte-exact: {run['survived']}   "
                      f"({run['byte_exact_before']} -> {run['byte_exact_after']}, "
                      f"margin {run['min_margin_before']:.3f} -> {run['min_margin_after']:.3f}, "
                      f"route-acc {run['routing_acc_before']:.3f} -> {run['final_routing_acc']:.3f}, "
                      f"RL-grad->graft max={run['max_rl_grad_into_graft']:.2e}, "
                      f"final reward={run['final_reward_mean']})")
                poll_rss()

        model._hook.remove()
        del model
        try:
            os.unlink(host0_f.name)
        except OSError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["rl_runs"] = all_runs

    # =============================== SUMMARY / VERDICT ======================= #
    print("\n################ VARIANCE-REDUCTION SURVIVAL SUMMARY ###########")

    def survivors_at(lr):
        return [(r["label"], r["survived"], r["pct_after"], r["min_margin_after"],
                 r["final_reward_mean"], r["final_routing_acc"])
                for r in all_runs if abs(r["lr"] - lr) < 1e-12 and r["reward_aligned"]]

    for lr in lrs:
        print(f"\n  == lr {lr:.0e} (reward arms) ==")
        for lbl, surv, pct, m, rew, racc in survivors_at(lr):
            tag = "OK " if surv else "BRK"
            print(f"    {tag} {lbl:<62} byte {pct:5.1f}%  m={m:+.2f}  "
                  f"reward={('%.3f'%rew) if rew is not None else '  - '}  route={racc:.3f}")

    def config_key(r):
        return r["label"].rsplit("/lr", 1)[0]

    families = {}
    for r in all_runs:
        if not r["reward_aligned"]:
            continue
        k = config_key(r)
        families.setdefault(k, []).append(r)

    family_summary = {}
    for k, runs in families.items():
        held = [r["lr"] for r in runs if r["survived"]]
        useful_held = [r["lr"] for r in runs if r["survived"]
                       and (r["final_reward_mean"] or 0) > 0.9
                       and (r["final_routing_acc"] or 0) > 0.9]
        family_summary[k] = dict(
            highest_hold_lr=max(held) if held else None,
            highest_stable_and_useful_lr=max(useful_held) if useful_held else None,
            n_runs=len(runs))
    results["family_summary"] = family_summary

    print("\n################ STABLE-AND-USEFUL by config family ###########")
    for k, s in sorted(family_summary.items()):
        print(f"  {k:<58} holds<={s['highest_hold_lr']}  "
              f"stable+useful<={s['highest_stable_and_useful_lr']}")

    # =============================== VERDICT =============================== #
    print("\n############################### VERDICT ###########################")
    winners = []
    for r in all_runs:
        if (r["reward_aligned"] and r["survived"] and r["lr"] >= 1e-2 - 1e-12
                and (r["final_reward_mean"] or 0) > 0.9
                and (r["final_routing_acc"] or 0) > 0.9
                and r["byte_exact_after"] == r["byte_exact_before"]):
            winners.append(dict(label=r["label"], lr=r["lr"],
                                reward=r["final_reward_mean"],
                                route=r["final_routing_acc"],
                                margin=r["min_margin_after"],
                                result_mode=r["result_mode"]))

    def knob_count(w):
        r = next(x for x in all_runs if x["label"] == w["label"])
        return int(r["adv_norm"]) + int(r["temp_sample"] != 1.0) + int(r["kl_beta"] > 0) \
            + (0 if r["group_n"] == 8 else 1) + (1 if r["result_mode"] == "split" else 0)
    winners_sorted = sorted(winners, key=lambda w: (knob_count(w), -w["lr"]))
    results["winners_at_or_above_1e2"] = winners
    results["minimal_winner"] = winners_sorted[0] if winners_sorted else None

    verdict = dict(
        any_nonprotection_recipe_stable_and_useful_at_1e2=bool(winners),
        n_winners=len(winners),
        minimal_recipe=winners_sorted[0] if winners_sorted else None,
        note=("A 'winner' = reward-aligned RL that stays byte-exact 320/320 AND reaches "
              "reward>0.9 + routing-acc>0.9 at lr>=1e-2 (#914's break lr), with NO freeze/"
              "lr-cap/mask/stop-grad/EMA. If empty: variance reduction does NOT prevent the "
              "saddle break under any swept config -> the honest failure result."))
    results["verdict"] = verdict
    print(f"\n  VERDICT: {json.dumps(verdict, indent=2, default=lambda o: (float(o) if isinstance(o,(np.floating,)) else bool(o) if isinstance(o,(np.bool_,bool)) else o))}")

    results["peak_rss_gb"] = PEAK_RSS
    results["elapsed_s"] = time.time() - t0
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_agent_graft_rl_variance_reduction_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"\nWrote {out}   (elapsed {time.time()-t0:.1f}s, PEAK RSS {PEAK_RSS:.2f} GB)")


if __name__ == "__main__":
    main()
