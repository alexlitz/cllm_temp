"""CROSS-PROGRAM BATCHED perfect-draft speculation.

``pf_speculative.verify_blocks`` batches the STEPS of ONE program into a handful
of block-verify forwards.  This module stacks the block-verify spans of MANY
INDEPENDENT programs into ONE ``[B, W, D]`` batched forward — the 4th lever.

The programs in the corpus are independent, so a block-verify forward that would
run B times (once per program) runs ONCE over a padded batch: the 300-block
forward launch (embedding gather + block stack + per-block KV attention) is
amortised across B programs.  Because each program's block-verify span is causally
self-contained (its query rows attend only to its own frozen prefix, kept in its
own per-block KV cache), the batched forward is byte-identical to running each
program's ``verify_blocks`` separately — only the matmuls are stacked.

Programs are DEPTH-BUCKETED first (by drafted step count) so a batch's members run
a similar number of blocks with no idle slots; a program that HALTs early simply
drops out of the active set.  The perfect draft is free, so the depth is known
before any forward.

Byte-identity: the per-program per-step decode + accept decision is EXACTLY
``verify_blocks``' (same nibble-argmax register decode, same draft comparison,
same commit + eviction).  We assert this in ``run_corpus_stacked --spotcheck-n``
(the batched verdict must match the per-program ``spotcheck_vs_cached``).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .nibble_pure_forward import N_ROLES, _snap_lane
from .nibble_pure_forward_complete import (
    PureForwardCompleteLayout, _decode_reg_from_nibbles,
)
from .pf_speculative import (
    PFDraft, draft_pf_program, build_code_vec, apply_overlay_window_fast,
    verify_blocks, speculative_run,
)
from .nibble_pure_forward_cached import (
    apply_overlay_window, prune_keep_mask_head,
)


# ---------------------------------------------------------------------------
# Depth bucketing (geometric bands) so a batch runs ~its members' depth.
# ---------------------------------------------------------------------------
_EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def _bucket(items: List[dict], batch_cap: int) -> List[List[dict]]:
    buckets: Dict[int, List[dict]] = {}
    for it in items:
        rs = it["draft"].step_count
        key = len(_EDGES)
        for bi, e in enumerate(_EDGES):
            if rs <= e:
                key = bi
                break
        buckets.setdefault(key, []).append(it)
    batches: List[List[dict]] = []
    for key in sorted(buckets):
        band = sorted(buckets[key], key=lambda r: r["draft"].step_count)
        for s in range(0, len(band), batch_cap):
            batches.append(band[s:s + batch_cap])
    return batches


# ---------------------------------------------------------------------------
# Per-program per-block KV cache (batched tensors) — mirrors BlockKVCacheBatched
# but kept per (program, block) so the batch can pad + stack them per forward.
# ---------------------------------------------------------------------------
class _ProgCache:
    __slots__ = ("K", "V", "pos", "total_evicted", "tokens_since_prune")

    def __init__(self, n_blocks: int):
        self.K: List[Optional[torch.Tensor]] = [None] * n_blocks   # [1,H,S,HD]
        self.V: List[Optional[torch.Tensor]] = [None] * n_blocks
        self.pos: List[Optional[torch.Tensor]] = [None] * n_blocks  # [S]
        self.total_evicted = 0
        self.tokens_since_prune = 0

    def cache_len(self, b: int) -> int:
        return 0 if self.pos[b] is None else int(self.pos[b].shape[0])

    def commit(self, b: int, K_new, V_new, pos_new):
        if self.K[b] is None:
            self.K[b], self.V[b], self.pos[b] = K_new, V_new, pos_new
        else:
            self.K[b] = torch.cat([self.K[b], K_new], dim=2)
            self.V[b] = torch.cat([self.V[b], V_new], dim=2)
            self.pos[b] = torch.cat([self.pos[b], pos_new], dim=0)


def _attn_batched(attn, x, past_K, past_V, past_pos, past_valid, q_pos):
    """x [B,W,D]; past_* padded [B,H,Sc,HD]/[B,Sc]; q_pos [B,W].  softmax1+ALiBi.
    Returns (out[B,W,D], (K_all,V_all,pos_all[B,Sc+W])) — mirrors
    nibble_pure_forward_gpu._attn_batched."""
    B, W, D = x.shape
    H, HD = attn.n_heads, attn.head_dim
    Q = attn.W_q.linear(x).view(B, W, H, HD).transpose(1, 2)
    Knew = attn.W_k.linear(x).view(B, W, H, HD).transpose(1, 2)
    Vnew = attn.W_v.linear(x).view(B, W, H, HD).transpose(1, 2)
    new_valid = torch.ones(B, W, dtype=torch.bool, device=x.device)
    if past_K is not None:
        K = torch.cat([past_K, Knew], dim=2)
        Vv = torch.cat([past_V, Vnew], dim=2)
        k_pos = torch.cat([past_pos, q_pos], dim=1)
        valid = torch.cat([past_valid, new_valid], dim=1)
    else:
        K, Vv, k_pos, valid = Knew, Vnew, q_pos, new_valid
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale       # [B,H,W,Sk]
    dist = (q_pos.unsqueeze(2) - k_pos.unsqueeze(1)).abs().float()   # [B,W,Sk]
    scores = scores - attn.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(1)
    fut = (k_pos.unsqueeze(1) > q_pos.unsqueeze(2))                  # [B,W,Sk]
    fut = fut | (~valid).unsqueeze(1)
    scores = scores.masked_fill(fut.unsqueeze(1), float("-inf"))
    m = torch.clamp(scores.max(dim=-1, keepdim=True)[0], min=0.0)
    exp_x = torch.exp(scores - m)
    denom = torch.exp(-m) + exp_x.sum(dim=-1, keepdim=True)
    a = exp_x / denom
    out = torch.matmul(a, Vv).transpose(1, 2).contiguous().view(B, W, D)
    out = x + attn.W_o.linear(out)
    return out, (K, Vv, k_pos, valid)


_FFN_ROW_CHUNK = 2048


def _ffn_chunked(ffn, a):
    B, W, D = a.shape
    rows = B * W
    if rows <= _FFN_ROW_CHUNK:
        return ffn.forward(a)
    flat = a.reshape(rows, 1, D)
    outs = [ffn.forward(flat[s:s + _FFN_ROW_CHUNK])
            for s in range(0, rows, _FFN_ROW_CHUNK)]
    return torch.cat(outs, dim=0).reshape(B, W, D)


def _decode_and_check(state, L, fr, mask):
    """Return (ok, got_dict).  Mirrors verify_blocks' per-row decode + accept."""
    got_pc = _snap_lane(state[L.PC_VAL])
    got_sp = _snap_lane(state[L.SP_VAL])
    got_bp = _snap_lane(state[L.BP_VAL])
    got_ax = _decode_reg_from_nibbles(state, L, L.AX) & mask
    want_ax = fr["ax"] & mask
    if fr.get("is_halt"):
        ok = (got_ax == want_ax)
    else:
        ok = (got_pc == fr["pc"] and got_ax == want_ax
              and got_sp == (fr["sp"] & 0xFFFFFFFF)
              and got_bp == (fr["bp"] & 0xFFFFFFFF))
    return ok, {"pc": got_pc, "ax": got_ax, "sp": got_sp, "bp": got_bp}


def _verify_batch(model, L, batch: List[dict], block_steps: int, device: str,
                  evict: bool, prune_interval: int, mask: int, fast: bool,
                  cos_threshold=0.99, zero_eps=1e-9, recency_eps=1e-6):
    """Block-verify a BATCH of programs together.  Each item dict carries
    ``code``/``draft``; we fill in ``status``/``got``/``forwards``/``accepted``.

    Returns (n_forwards_total_batched, sum_naive_forwards)."""
    n_blocks = len(model.blocks)
    H = model.blocks[0].attn.n_heads
    HD = model.blocks[0].attn.head_dim
    dev = torch.device(device)
    D = model.embed.shape[1]
    # Host mirror of the embedding table (built ONCE) so the per-block window
    # residual (embed gather + structural overlay) is assembled on CPU with cheap
    # scalar writes and copied to device once per forward.
    embed_cpu = (model.embed if model.embed.device.type == "cpu"
                 else model.embed.detach().to("cpu"))

    # per-program state
    for it in batch:
        it["_cache"] = _ProgCache(n_blocks)
        it["_step"] = 0
        it["_accepted"] = 0
        it["_done"] = False
        it["_last_ax"] = None
        it["_mismatch"] = None
        # code_vec built on CPU (the overlay is assembled on the host).
        it["_code_vec"] = (build_code_vec(it["code"], L, D, "cpu",
                                          dtype=embed_cpu.dtype) if fast else None)
    active = list(range(len(batch)))
    n_forwards = 0
    slopes = model.blocks[0].attn.alibi_slopes
    scale = HD ** -0.5

    while active:
        # advance each active program by one BLOCK; build its span + query rows.
        spans = []          # (bi_in_active, span_toks, span_start, q_local, steps)
        Wmax = 0
        for ai in active:
            it = batch[ai]
            draft: PFDraft = it["draft"]
            n_steps = draft.step_count
            step = it["_step"]
            end = min(step + block_steps, n_steps)
            span_start = 0 if step == 0 else draft.win_starts[step]
            span_end = (draft.win_starts[end] + 1) if end < n_steps \
                else len(draft.tokens)
            span_toks = draft.tokens[span_start:span_end]
            q_local = [draft.win_starts[s] - span_start for s in range(step, end)]
            spans.append((ai, span_toks, span_start, q_local, list(range(step, end))))
            Wmax = max(Wmax, len(span_toks))

        Bact = len(spans)
        # pad token windows to Wmax; build overlaid residual on CPU then move once.
        win_toks = torch.zeros(Bact, Wmax, dtype=torch.long)
        q_positions = torch.zeros(Bact, Wmax, dtype=torch.long)
        valid_row = torch.zeros(Bact, Wmax, dtype=torch.bool)
        for i, (ai, span_toks, span_start, q_local, srange) in enumerate(spans):
            S = len(span_toks)
            win_toks[i, :S] = torch.tensor(span_toks, dtype=torch.long)
            q_positions[i, :S] = torch.arange(span_start, span_start + S)
            # pad positions get a sentinel LARGER than any real pos so causal drops.
            q_positions[i, S:] = span_start + S + 10_000_000
            valid_row[i, :S] = True
        with torch.no_grad():
            x_cpu = embed_cpu[win_toks].clone()
            for i, (ai, span_toks, span_start, q_local, srange) in enumerate(spans):
                it = batch[ai]
                S = len(span_toks)
                xi = x_cpu[i:i + 1, :S, :]
                if fast:
                    apply_overlay_window_fast(xi, span_start, L, it["draft"].store_log,
                                              it["_code_vec"], query_rows=q_local)
                else:
                    apply_overlay_window(xi, span_start, it["code"], L,
                                         it["draft"].store_log, is_last_row_query=False)
                    for wi in q_local:
                        for role in range(N_ROLES):
                            xi[0, wi, L.ROLE + role] = 1.0
            x = x_cpu.to(dev)
            q_pos = q_positions.to(dev)
            row_valid = valid_row.to(dev)

            # forward the block stack, batched, with per-program padded KV caches.
            hidden = x
            win_kv = []           # per block: (K_win[B,H,Wmax,HD], V_win, pos_win[B,Wmax])
            sentinel = int(q_pos.max().item()) + 1_000_000
            for b in range(n_blocks):
                lens = [batch[ai]["_cache"].cache_len(b) for (ai, *_r) in spans]
                Sc = max(lens) if lens else 0
                if Sc == 0:
                    pK = pV = pP = pVal = None
                else:
                    pK = torch.zeros(Bact, H, Sc, HD, device=dev)
                    pV = torch.zeros(Bact, H, Sc, HD, device=dev)
                    pP = torch.full((Bact, Sc), sentinel, dtype=torch.long, device=dev)
                    pVal = torch.zeros(Bact, Sc, dtype=torch.bool, device=dev)
                    for i, (ai, *_r) in enumerate(spans):
                        pc = batch[ai]["_cache"]
                        n = lens[i]
                        if n:
                            pK[i, :, :n, :] = pc.K[b]
                            pV[i, :, :n, :] = pc.V[b]
                            pP[i, :n] = pc.pos[b].to(dev)
                            pVal[i, :n] = True
                a, kv = _attn_batched(model.blocks[b].attn, hidden, pK, pV, pP,
                                      pVal, q_pos)
                hidden = _ffn_chunked(model.blocks[b].ffn, a)
                K_all, V_all, pos_all, _v = kv
                win_kv.append((K_all[:, :, -Wmax:, :].contiguous(),
                               V_all[:, :, -Wmax:, :].contiguous(),
                               pos_all[:, -Wmax:].contiguous()))
                del kv, K_all, V_all, pos_all, pK, pV, pP, pVal
            n_forwards += 1

        # -- per-program decode + accept each of its block's query rows -----------
        for i, (ai, span_toks, span_start, q_local, srange) in enumerate(spans):
            it = batch[ai]
            draft = it["draft"]
            n_steps = draft.step_count
            bad = False
            for s, wi in zip(srange, q_local):
                state = hidden[i, wi]
                ok, got = _decode_and_check(state, L, draft.frames[s], mask)
                if not ok:
                    it["_mismatch"] = {"step": s, "got": got,
                                       "want": {k: draft.frames[s][k]
                                                for k in ("pc", "ax", "sp", "bp")}}
                    bad = True
                    break
                it["_accepted"] += 1
                if s == n_steps - 1:
                    it["_last_ax"] = got["ax"]
            if bad:
                it["_done"] = True
                it["_failed"] = True
                continue
            # commit the frozen (non-query) rows of THIS program's span.
            S = len(span_toks)
            q_set = set(q_local)
            keep = [wi for wi in range(S) if wi not in q_set]
            if keep:
                keep_idx = torch.tensor(keep, device=dev, dtype=torch.long)
                pc = it["_cache"]
                for b in range(n_blocks):
                    K_win, V_win, pos_win = win_kv[b]
                    pc.commit(b, K_win[i:i + 1, :, keep_idx, :].clone(),
                              V_win[i:i + 1, :, keep_idx, :].clone(),
                              pos_win[i, keep_idx].clone())
            it["_step"] = srange[-1] + 1
            if it["_step"] >= n_steps:
                it["_done"] = True
            # bounded eviction (per program, same policy as verify_blocks).
            if evict:
                pc = it["_cache"]
                pc.tokens_since_prune += len(srange) * V.FRAME_LEN
                if pc.tokens_since_prune >= prune_interval:
                    pc.tokens_since_prune = 0
                    for b in range(n_blocks):
                        if pc.K[b] is None:
                            continue
                        Kb, Vb, posb = pc.K[b], pc.V[b], pc.pos[b]
                        Sb = posb.shape[0]
                        keep_any = torch.zeros(Sb, dtype=torch.bool)
                        vnorm = Vb[0].norm(dim=-1)
                        hv = (vnorm > zero_eps).any(dim=-1)
                        Kc = Kb[0].cpu(); Vc = Vb[0].cpu(); Pc = posb.cpu()
                        knorm_hs = Kc.norm(dim=-1)
                        mean_key = Kc.mean(dim=1)
                        cm = mean_key.norm(dim=-1) / knorm_hs.mean(dim=-1).clamp(min=1e-30)
                        for h in range(H):
                            if not bool(hv[h]):
                                continue
                            metric = "exact" if float(cm[h]) > 0.9 else "cosine"
                            keep_any |= prune_keep_mask_head(
                                Kc[h], Vc[h], Pc, slope=float(slopes[h]), scale=scale,
                                cos_threshold=cos_threshold, zero_eps=zero_eps,
                                recency_eps=recency_eps, dup_metric=metric)
                        kidx = torch.nonzero(keep_any, as_tuple=False).flatten()
                        if int(kidx.numel()) < Sb:
                            pc.total_evicted += Sb - int(kidx.numel())
                            kdev = kidx.to(Kb.device)
                            pc.K[b] = Kb[:, :, kdev, :]
                            pc.V[b] = Vb[:, :, kdev, :]
                            pc.pos[b] = posb[kdev]
        if device.startswith("cuda"):
            torch.cuda.synchronize(dev)
        active = [ai for ai in active if not batch[ai]["_done"]]

    # -- finalise verdicts --------------------------------------------------------
    sum_naive = 0
    for it in batch:
        draft = it["draft"]
        sum_naive += draft.step_count
        exp = it["expected"] & 0xFFFFFFFF
        if it.get("_failed"):
            mm = it["_mismatch"]
            it["status"] = "FAIL"
            it["got"] = None
            it["detail"] = (f"model argmax != draft at step {mm['step']}: "
                            f"got {mm['got']} want {mm['want']}") if mm else "verify fail"
        else:
            got = (it["_last_ax"] if it["_last_ax"] is not None else 0) & mask
            it["status"] = "PASS" if got == exp else "FAIL"
            it["got"] = got
            it["detail"] = "" if it["status"] == "PASS" else f"exit mismatch: exp {exp} got {got}"
        it["steps"] = draft.step_count
        it["accepted"] = it["_accepted"]
        it["max_cache"] = 0
        # clean transient state
        for k in ("_cache", "_code_vec"):
            it.pop(k, None)
    return n_forwards, sum_naive


def speculative_run_batch(model, L: PureForwardCompleteLayout,
                          items: List[dict], *, block_steps: int = 64,
                          max_steps: int = 300000, device: str = "cpu",
                          evict: bool = True, prune_interval: int = 120,
                          mask: int = 0xFFFFFFFF, fast: bool = True,
                          batch_cap: int = 8,
                          progress: Optional[Callable[[int, int], None]] = None
                          ) -> Tuple[List[dict], int, int]:
    """Cross-program batched speculation over ``items`` (each a dict with
    ``idx``/``cluster``/``description``/``expected``/``code``).  Returns
    ``(result_rows, sum_naive_forwards, sum_spec_forwards)``.

    Each program is drafted (free), depth-bucketed, and its block-verify spans are
    stacked with its bucket-mates into one batched forward per block-step.  A
    program whose draft does not HALT is a TIMEOUT (no forwards)."""
    rows: List[dict] = []
    sum_naive = sum_spec = 0
    # draft everything first (free); split TIMEOUTs (non-halting) out.
    verifiable = []
    for it in items:
        draft = draft_pf_program(it["code"], max_steps=max_steps, mask=mask)
        if not draft.halted:
            rows.append(dict(idx=it["idx"], cluster=it["cluster"], status="TIMEOUT",
                             expected=it["expected"] & 0xFFFFFFFF, got=None,
                             steps=draft.step_count, forwards=0,
                             naive_forwards=draft.step_count, speedup=0.0,
                             detail=f"draft did not HALT within {max_steps} steps",
                             description=it["description"]))
            continue
        it2 = dict(it)
        it2["draft"] = draft
        verifiable.append(it2)

    batches = _bucket(verifiable, batch_cap)
    done = 0
    for batch in batches:
        nf, sn = _verify_batch(model, L, batch, block_steps, device, evict,
                               prune_interval, mask, fast)
        sum_spec += nf
        sum_naive += sn
        for it in batch:
            rows.append(dict(
                idx=it["idx"], cluster=it["cluster"], status=it["status"],
                expected=it["expected"] & 0xFFFFFFFF, got=it["got"],
                steps=it["steps"], forwards=nf, naive_forwards=it["steps"],
                speedup=round(it["steps"] / nf, 2) if nf else 0.0,
                accepted=it["accepted"], max_cache=it.get("max_cache", 0),
                detail=it["detail"], description=it["description"]))
            done += 1
        if progress:
            npass = sum(1 for r in rows if r["status"] == "PASS")
            progress(done, npass)
    return rows, sum_naive, sum_spec
