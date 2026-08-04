"""BYTE-EXACT FLASH ATTENTION (softmax1 + ALiBi) for the c4_min VM — O(S) memory.

The general N^2-memory fix for the c4_min attention.  Both the GLOBAL heads of the
self-emulation / arbitrary-program path (``local_attention.windowed_forward``'s
``_attend_group(..., window=None)``) and the un-windowed ``SparseAttn.forward``
materialise the FULL ``[B, H, Sq, Sk]`` score matrix ``Q@Kᵀ`` and softmax it in
one shot — O(Sq·Sk) VRAM.  For self-emulation the global memory/stack/LEV heads
must reach over the WHOLE growing KV (no direct-CAM draft), so this quadratic
matrix is the wall: at S≈12k the score alone is ~12.9 GiB and OOMs.

THE KEY ENABLER (byte-exact).  ``blogspec_model`` documents that softmax1
(``exp(x)/(1+Σexp(x))``, the spec's only real deviation, §491) is EXACTLY a plain
softmax over ``[BOS-sink, real...]`` with a prepended score-0 / value-0 sink column
whose ``exp(0)=1`` supplies the ``+1`` in the denominator (§40-44).  A plain softmax
is what every tiled / online-softmax FLASH kernel computes — so we can run a
STANDARD flash kernel (tiled, never materialising the score matrix) and recover
softmax1 exactly.  Two mathematically-identical realisations of that sink are used:

  * ``softmax1_ctx = plain_softmax_ctx · sigmoid(LSE)``.  The ``+1`` sink rescales
    the plain-softmax context of every query row by
    ``Σexp(s)/(1+Σexp(s)) = sigmoid(logsumexp(s))``.  The flash / mem-efficient
    kernel already computes the per-row LSE, so we get softmax1 with a single
    ``·sigmoid(LSE)`` — NO physical sink column, NO extra key.  Used by the
    SDPA path.
  * a prepended zero-score / zero-value key column — used inside the custom Triton
    kernel, where the running denominator is simply initialised to ``1`` (the
    sink's ``exp(0)``) instead of ``0``.

ALiBi (``-slope·|q_pos - k_pos|``, §307) is the per-head additive recency bias.  For
the causal region ``q_pos >= k_pos`` so ``|q-k| = q-k``.  The SDPA path folds it into
two extra Q/K dims (``Q@K`` then captures ``-slope·(q-k)`` in the dot product) so no
O(S²) additive mask tensor is ever built; the Triton kernel adds it inside the tile.

Two backends, both byte-exact (fp32; tiled-reduction noise ~1e-6, far below the
nibble-decode margin — tf32 is NOT used, its 10-bit mantissa breaks doom):

  * ``sdpa_flash_softmax1`` — the un-cached FULL case (Sq==Sk, positions 0..S-1,
    top-left causal): PyTorch mem-efficient SDPA + ALiBi-fold + ``·sigmoid(LSE)``.
    O(S) VRAM, uses the vendor flash kernel.
  * ``triton_flash_softmax1`` — the GENERAL cached / windowed case (arbitrary
    ``q_pos`` / ``k_pos`` arrays, bottom-right causal, optional sliding window):
    a custom tiled online-softmax1 Triton kernel.  Handles the KV-cache regime
    (Sq<Sk, q_pos = the newest positions) that SDPA's fixed top-left ``is_causal``
    cannot express.

``flash_softmax1_context`` dispatches to the right backend from the shapes.

Gated by ``C4_FLASH_ATTN`` (checked at the attention call sites); default OFF -> the
masked-full path runs and the golden ``069cc32f`` is unchanged.  No stored weight is
touched -> the fingerprint is byte-identical by construction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:                                   # pragma: no cover
    _HAVE_TRITON = False


# ===========================================================================
# SDPA backend (un-cached full case): ALiBi-fold + mem-efficient flash + LSE.
# ===========================================================================
def _align8(n):
    """Next multiple of 8 >= n.  The mem-efficient SDPA kernel requires the last
    (head) dim of Q/K/V to be 8-byte-stride aligned (a multiple of 8 for fp32);
    an arbitrary VM head_dim (e.g. 90) or the ALiBi-folded HD+4 (94) is not, so we
    zero-pad up to the next multiple of 8.  Zero padding lanes contribute 0 to the
    Q·K dot and 0 to the context, so the result over the real dims is unchanged."""
    return (n + 7) & ~7


def _fold_alibi_qk(Q, K, q_pos, k_pos, slopes, scale):
    """Return ``(Qx, Kx)`` with ALiBi ``-slope·(q_pos-k_pos)`` folded into 2 extra
    head dims, then zero-padded so the last dim is a multiple of 8 (the mem-efficient
    kernel's alignment requirement — an arbitrary VM head_dim like 90 is not aligned).

    ``(Qx·Kx)·scale`` = ``(Q·K)·scale - slope·(q_pos - k_pos)`` on the causal region
    (where ``q_pos >= k_pos`` so ``|q-k| = q-k``).  We need the extra dot to equal
    ``-slope·(q-k)/scale`` so that after SDPA multiplies by ``scale`` it is exactly
    the ALiBi term:  ``extra = (-slope/scale)·q_pos + (slope/scale)·k_pos``.  Split as
    ``Q_extra = [(-c)·q_pos, 1]``, ``K_extra = [1, c·k_pos]`` with ``c = slope/scale``
    per head; the trailing zero pad (to the mult-of-8) contributes 0 to the dot.
    """
    B, H, Sq, HD = Q.shape
    Sk = K.shape[2]
    dev, dt = Q.device, Q.dtype
    HD_A = _align8(HD + 2)                              # aligned padded head dim
    npad = HD_A - HD - 2                                # trailing zero-pad lanes
    c = (slopes / scale).to(dt).view(1, H, 1)                       # [1,H,1]
    ones_q = torch.ones(1, H, Sq, device=dev, dtype=dt)
    ones_k = torch.ones(1, H, Sk, device=dev, dtype=dt)
    qpf = q_pos.view(1, 1, Sq).to(dt)
    kpf = k_pos.view(1, 1, Sk).to(dt)
    qe_cols = [(-c) * qpf, ones_q]                     # [1,H,Sq] each
    ke_cols = [ones_k, c * kpf]
    for _ in range(npad):
        qe_cols.append(torch.zeros(1, H, Sq, device=dev, dtype=dt))
        ke_cols.append(torch.zeros(1, H, Sk, device=dev, dtype=dt))
    qe = torch.stack(qe_cols, dim=-1).expand(B, H, Sq, 2 + npad)    # [B,H,Sq,pad]
    ke = torch.stack(ke_cols, dim=-1).expand(B, H, Sk, 2 + npad)
    Qx = torch.cat([Q, qe], dim=-1).contiguous()
    Kx = torch.cat([K, ke], dim=-1).contiguous()
    return Qx, Kx


def sdpa_flash_softmax1(Q, K, V, q_pos, k_pos, slopes, scale):
    """softmax1 + ALiBi + TOP-LEFT causal via mem-efficient SDPA — O(S) VRAM.

    Byte-exact realisation of the un-cached FULL path (``past_kv is None,
    q_positions is None``): positions are ``0..S-1``, Sq==Sk, and SDPA's
    ``is_causal=True`` (top-left, query row i sees keys 0..i) is exactly the
    reference causal.  ALiBi is folded into the Q/K dims; the softmax1 ``+1`` sink
    is recovered by ``·sigmoid(LSE)`` (LSE returned by the mem-efficient kernel).

    Returns the context ``[B, H, Sq, HD]`` (over the REAL head dims only — the V
    alignment pad is sliced off).
    """
    B, H, Sq, HD = Q.shape
    Qx, Kx = _fold_alibi_qk(Q, K, q_pos, k_pos, slopes, scale)
    # V must also be 8-aligned in the last dim (kernel requirement); zero-pad it and
    # slice the pad off the output.  The pad lanes get context 0 (dropped).
    HDV = V.shape[-1]
    HDV_A = _align8(HDV)
    if HDV_A != HDV:
        Vc = torch.cat(
            [V, torch.zeros(B, H, V.shape[2], HDV_A - HDV,
                            device=V.device, dtype=V.dtype)], dim=-1).contiguous()
    else:
        Vc = V.contiguous()
    # aten mem-efficient attention -> (out, logsumexp, philox_seed, philox_offset).
    out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        Qx, Kx, Vc, None, True, is_causal=True, scale=scale)
    out = out[..., :HDV]                               # drop the V alignment pad
    lse = lse[..., :Sq]                                # kernel pads LSE to mult-of-32
    return out * torch.sigmoid(lse).unsqueeze(-1)


# ===========================================================================
# Triton backend (general cached / windowed case): tiled online-softmax1.
# ===========================================================================
if _HAVE_TRITON:

    @triton.jit
    def _flash_softmax1_kernel(
        Q, K, V, Out,
        QPOS, KPOS, SLOPES,
        scale,
        stride_qh, stride_qs, stride_qd,
        stride_kh, stride_ks, stride_kd,
        stride_vh, stride_vs, stride_vd,
        stride_oh, stride_os, stride_od,
        Sq, Sk, WINDOW,
        HD, HD_POW2: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
        USE_WINDOW: tl.constexpr,
    ):
        """One program = one (head, query-tile).  Online softmax with the softmax1
        ``+1`` sink baked into the running denominator (``l_i`` initialised to 1.0,
        the sink's ``exp(0)``).  ALiBi ``-slope·(qpos-kpos)`` and the position-based
        causal (``kpos <= qpos``) + optional sliding window (``qpos-kpos < WINDOW``)
        are applied per K-tile.  Never materialises the ``[Sq,Sk]`` score matrix.

        HD may be any value (e.g. 80); the head-dim range is padded to the next power
        of two ``HD_POW2`` and the real ``[0,HD)`` lanes are masked (padding lanes load
        0 -> contribute 0 to the dot product and are not stored), so the result over
        the real dims is unchanged.
        """
        pid_q = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        offs_d = tl.arange(0, HD_POW2)
        d_mask = offs_d < HD                                          # real head dims
        q_mask = offs_q < Sq

        q_ptrs = Q + pid_h * stride_qh + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)  # [BQ, HD_POW2]
        qpos = tl.load(QPOS + offs_q, mask=q_mask, other=0).to(tl.float32)  # [BQ]
        slope = tl.load(SLOPES + pid_h).to(tl.float32)

        # online-softmax accumulators (fp32).  The softmax1 ``+1`` sink IS a virtual
        # key of logit 0 and value 0: seed the running max at its logit (m_i=0.0) and
        # the running denominator at its weight (l_i=exp(0-0)=1.0).  Its zero value
        # adds nothing to ``acc``.  Because m_i starts at 0.0 (not -inf) the sink is
        # correctly carried through every online rescale (alpha = exp(m_i-m_new)).
        m_i = tl.zeros([BLOCK_Q], dtype=tl.float32)                 # sink logit 0
        l_i = tl.full([BLOCK_Q], 1.0, dtype=tl.float32)            # sink exp(0-0)=1
        acc = tl.zeros([BLOCK_Q, HD_POW2], dtype=tl.float32)

        for k_start in range(0, Sk, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < Sk
            k_ptrs = K + pid_h * stride_kh + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)  # [BK, HD_POW2]
            kpos = tl.load(KPOS + offs_k, mask=k_mask, other=0).to(tl.float32)  # [BK]

            # scores[bq,bk] = (q . k) * scale - slope*|qpos-kpos|.  input_precision
            # "ieee" forces true fp32 accumulation (NOT tf32's 10-bit mantissa, which
            # would break doom's tight-margin decodes) -> byte-exact.
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale   # [BQ, BK]
            dist = qpos[:, None] - kpos[None, :]                     # signed
            s = s - slope * tl.abs(dist)

            # causal (kpos <= qpos) + valid key + optional window (dist < WINDOW).
            valid = (kpos[None, :] <= qpos[:, None]) & k_mask[None, :]
            if USE_WINDOW:
                valid = valid & (dist < WINDOW)
            s = tl.where(valid, s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)                              # rescale old
            p = tl.exp(s - m_new[:, None])                           # [BQ, BK]
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v_ptrs = V + pid_h * stride_vh + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)  # [BK, HD_POW2]
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v,
                                                input_precision="ieee")
            m_i = m_new

        # softmax1: divide by l_i, which now equals ``exp(0 - m_final) +
        # Σ_j exp(s_j - m_final)`` == the softmax1 denominator ``1 + Σexp`` rescaled
        # by the same ``exp(-m_final)`` as the numerator ``acc``.  The sink's zero
        # value contributed nothing to ``acc`` but its ``exp(0-m)`` is in l_i.
        acc = acc / l_i[:, None]
        o_ptrs = Out + pid_h * stride_oh + offs_q[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty),
                 mask=q_mask[:, None] & d_mask[None, :])


def triton_flash_softmax1(Q, K, V, q_pos, k_pos, slopes, scale, window=None):
    """General tiled online-softmax1 + ALiBi + causal (+ optional window) attention.

    Args (flattened over batch into the head axis by the caller, or B==1):
      Q  [B, H, Sq, HD]   K,V [B, H, Sk, HD]  (fp32)
      q_pos [Sq] long      k_pos [Sk] long     (absolute positions, any ordering with
                                                kpos ascending within the causal set)
      slopes [H] float     scale float          window int|None (drop qpos-kpos>=W)

    Returns context ``[B, H, Sq, HD]``.  Never builds the ``[Sq,Sk]`` matrix -> O(S)
    (each program keeps only its BLOCK_Q×BLOCK_K tile).  Byte-exact to softmax1 +
    ALiBi (fp32 online reduction, ~1e-6).
    """
    assert _HAVE_TRITON, "triton unavailable"
    B, H, Sq, HD = Q.shape
    Sk = K.shape[2]
    assert B == 1, "triton_flash_softmax1 expects B==1 (fold batch into heads)"
    dev = Q.device
    Qf = Q[0].contiguous().float()                      # [H, Sq, HD]
    Kf = K[0].contiguous().float()
    Vf = V[0].contiguous().float()
    out = torch.empty((H, Sq, HD), device=dev, dtype=torch.float32)
    qpos = q_pos.to(dev).to(torch.int32).contiguous()
    kpos = k_pos.to(dev).to(torch.int32).contiguous()
    sl = slopes.to(dev).float().contiguous()
    HD_POW2 = max(16, 1 << (HD - 1).bit_length())      # next power of two >= HD (>=16)
    # fp32 tiles cost 4 bytes/elem; keep BQ·BK·HD_POW2 under the A5000 SMEM limit
    # (~101 KB).  128-wide heads -> 32×32 tiles; 64-wide -> 64×64.
    if HD_POW2 >= 128:
        BLOCK_Q = BLOCK_K = 32
    else:
        BLOCK_Q = BLOCK_K = 64
    W = 0 if window is None else int(window)
    grid = (triton.cdiv(Sq, BLOCK_Q), H)
    _flash_softmax1_kernel[grid](
        Qf, Kf, Vf, out,
        qpos, kpos, sl,
        float(scale),
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Kf.stride(0), Kf.stride(1), Kf.stride(2),
        Vf.stride(0), Vf.stride(1), Vf.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        Sq, Sk, W,
        HD, HD_POW2=HD_POW2, BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        USE_WINDOW=(window is not None),
        num_warps=8,       # fp32 ieee-precision dot has no tensor-core path; 8 warps
                           # per (head,q-tile) hides the latency (~9x vs the default 4)
    )
    return out.unsqueeze(0).to(Q.dtype)


# ===========================================================================
# Dispatcher used by the attention call sites.
# ===========================================================================
def flash_softmax1_context(Q, K, V, q_pos, k_pos, slopes, scale,
                           window=None, uncached_full=False):
    """Byte-exact O(S) flash softmax1 + ALiBi (+ optional window) attention.

    ``uncached_full=True`` (Sq==Sk, positions 0..S-1, no window) picks the SDPA
    mem-efficient + LSE-rescale path (top-left causal == the reference).  Every
    other case (cached KV, bottom-right causal, sliding window) uses the general
    Triton online-softmax1 kernel.

    Returns context ``[B, H, Sq, HD]`` (before ``W_o``).
    """
    if uncached_full and window is None:
        return sdpa_flash_softmax1(Q, K, V, q_pos, k_pos, slopes, scale)
    if _HAVE_TRITON and Q.is_cuda:
        return triton_flash_softmax1(Q, K, V, q_pos, k_pos, slopes, scale,
                                     window=window)
    # CPU / no-triton fallback: chunked reference (still O(S·chunk), byte-exact).
    return _chunked_reference(Q, K, V, q_pos, k_pos, slopes, scale, window)


def _chunked_reference(Q, K, V, q_pos, k_pos, slopes, scale, window):
    """A KV-chunked softmax1 reference for CPU / no-Triton — never materialises the
    full [Sq,Sk] matrix (processes K in chunks with an online softmax1), so it is the
    O(S) fallback and byte-exact."""
    from .blogspec_model import softmax1
    B, H, Sq, HD = Q.shape
    Sk = K.shape[2]
    CH = 1024
    # softmax1 ``+1`` sink = a virtual key of logit 0 / value 0: seed the running max
    # at its logit (0.0, NOT -inf) and the denominator at its weight (exp(0-0)=1.0), so
    # the sink is correctly carried through the online rescale (alpha = exp(m_i-m_new)).
    m_i = torch.zeros((B, H, Sq, 1), device=Q.device, dtype=Q.dtype)   # sink logit 0
    l_i = torch.ones((B, H, Sq, 1), device=Q.device, dtype=Q.dtype)    # sink exp(0-0)=1
    acc = torch.zeros((B, H, Sq, HD), device=Q.device, dtype=Q.dtype)
    for c0 in range(0, Sk, CH):
        c1 = min(c0 + CH, Sk)
        Kc, Vc = K[:, :, c0:c1], V[:, :, c0:c1]
        kp = k_pos[c0:c1]
        s = torch.matmul(Q, Kc.transpose(-2, -1)) * scale
        dist = (q_pos.unsqueeze(1) - kp.unsqueeze(0)).float()
        s = s - slopes.view(1, H, 1, 1) * dist.abs().unsqueeze(0)
        m = (kp.unsqueeze(0) > q_pos.unsqueeze(1))
        if window is not None:
            m = m | (dist >= window)
        s = s.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
        m_new = torch.maximum(m_i, s.max(dim=-1, keepdim=True)[0])
        alpha = torch.exp(m_i - m_new)
        p = torch.exp(s - m_new)
        l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
        acc = acc * alpha + torch.matmul(p, Vc)
        m_i = m_new
    return acc / l_i
