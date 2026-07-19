"""STEP 1 — tensor-sparsity for the c4_min pure-forward VM.

The c4_min pure-forward model is ~99.998 % sparse: the bitwise (non-divmod)
config is 42 blocks / dim 2392 / FFN hidden padded to the global-max 21544, so
7.46 B *dense* params (29.8 GB fp32) of which only ~180 k are nonzero.  Every
``model.forward`` therefore wastes essentially all of its FLOPs multiplying by
zero — and the 30 GB barely fits a 24 GB card.

This module converts the model's weight tensors to sparse (CSR for the 2-D
mat-muls, dense-kept for tiny vectors/biases) and runs the *same* forward with
sparse mat-muls.  At this sparsity the sparse mat-mul does work ∝ nnz (hundreds)
instead of the full ``21544×2392`` dense product, so it wins on BOTH storage and
compute — but the win is measured honestly here (sparse-kernel launch overhead
is real; a matrix that is *empty* or *tiny-dense* is kept dense and logged).

The wrapper is byte-identical (L∞=0) to the dense forward: sparse ``A@x`` is the
same arithmetic as dense ``A@x`` on the nonzeros, and the zero rows/cols
contribute exactly 0 either way.  It is a drop-in for
``run_pure_forward_complete`` (same ``model`` API: ``.embed``, ``.blocks``,
``.dim``, ``.vocab``, block ``.attn`` / ``.ffn`` with the same field names).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .blogspec_model import softmax1


# ---------------------------------------------------------------------------
# A sparse-or-dense 2-D weight: CSR when sparse pays, dense otherwise.
# ---------------------------------------------------------------------------
@dataclass
class SparseWeight:
    """A ``[out, in]`` linear weight stored sparse (CSR) or dense.

    ``linear(x)`` computes ``x @ W.T`` (i.e. ``F.linear(x, W)``) with either a
    sparse CSR mat-mul (work ∝ nnz) or the dense kernel, whichever the builder
    decided pays off for this shape/density.  A CSR that a given backend cannot
    run for a shape falls back to dense automatically and is logged.
    """

    out_dim: int
    in_dim: int
    is_sparse: bool
    dense: Optional[torch.Tensor] = None      # [out, in] when dense
    csr: Optional[torch.Tensor] = None        # [out, in] CSR when sparse
    nnz: int = 0
    fallback_reason: str = ""
    # "sparse_mm"    -> torch.sparse.mm (compute win; fp-accum-order residue only).
    # "dense_kernel" -> materialise the (small) dense weight from CSR and call the
    #                   SAME F.linear as the dense model -> BIT-IDENTICAL (L-inf=0).
    #                   Storage stays sparse; only one weight is dense-transient.
    compute_mode: str = "sparse_mm"
    # RESIDENT dense weight (set by ``materialize_dense``): the CSR densified ONCE
    # and kept on-device so ``linear`` skips the per-call ``csr.to_dense()`` (the
    # driver-bound cost — a 300-block verify re-densifies ~2100 CSR weights EVERY
    # forward).  Using it is BIT-IDENTICAL to ``dense_kernel`` (same F.linear GEMM,
    # same accum order — just the exact same dense tensor cached instead of rebuilt).
    dense_resident: Optional[torch.Tensor] = None

    def to(self, device) -> "SparseWeight":
        if self.dense is not None:
            self.dense = self.dense.to(device)
        if self.csr is not None:
            self.csr = self.csr.to(device)
        if self.dense_resident is not None:
            self.dense_resident = self.dense_resident.to(device)
        return self

    def materialize_dense(self, device=None) -> None:
        """Densify the CSR ONCE into a resident dense weight (on ``device`` if
        given) so ``linear`` reuses it instead of re-running ``csr.to_dense()`` on
        every forward.  Bit-identical to ``dense_kernel`` (same GEMM, same tensor).
        Only meaningful for a sparse weight; a dense-kept weight is already resident.
        """
        if not self.is_sparse:
            if device is not None and self.dense is not None:
                self.dense = self.dense.to(device)
            return
        d = self.csr.to_dense()
        if device is not None:
            d = d.to(device)
        self.dense_resident = d

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ W.T`` for ``x`` of shape ``[..., in_dim]`` -> ``[..., out_dim]``."""
        if not self.is_sparse:
            return F.linear(x, self.dense)
        if self.dense_resident is not None:
            # resident dense (materialize_dense): identical GEMM, no re-densify.
            return F.linear(x, self.dense_resident)
        if self.compute_mode == "dense_kernel":
            # bit-identical to the dense forward: same GEMM, same accum order.
            return F.linear(x, self.csr.to_dense())
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_dim)               # [N, in]
        # CSR @ dense: W[out,in] @ x2d.T[in,N] = [out, N]; transpose -> [N, out].
        try:
            out = torch.sparse.mm(self.csr, x2d.transpose(0, 1))   # [out, N]
        except (RuntimeError, NotImplementedError):
            # some (device, dtype, shape) combos have no CSR kernel -> dense fall.
            dense = self.csr.to_dense()
            self.fallback_reason = "no_csr_kernel"
            return F.linear(x, dense)
        out = out.transpose(0, 1).contiguous()          # [N, out]
        return out.reshape(*orig_shape[:-1], self.out_dim)

    def storage_bytes(self) -> int:
        if not self.is_sparse:
            return self.dense.numel() * self.dense.element_size()
        # CSR: crow_indices (out+1 int64) + col_indices (nnz int64) + values (nnz*4)
        return (self.out_dim + 1) * 8 + self.nnz * 8 + self.nnz * 4

    def dense_equiv_bytes(self) -> int:
        return self.out_dim * self.in_dim * 4


def _make_weight(w: torch.Tensor, density_thresh: float, min_numel: int,
                 log: Dict[str, int], compute_mode: str) -> SparseWeight:
    """Sparsify ``w`` (``[out, in]``) iff it is big enough AND sparse enough."""
    out_dim, in_dim = w.shape
    numel = out_dim * in_dim
    nnz = int((w != 0).sum().item())
    density = nnz / numel if numel else 1.0
    if numel < min_numel:
        log["kept_dense_small"] = log.get("kept_dense_small", 0) + 1
        return SparseWeight(out_dim, in_dim, False, dense=w.contiguous(),
                            nnz=nnz, fallback_reason="small",
                            compute_mode=compute_mode)
    if density >= density_thresh:
        log["kept_dense_dense"] = log.get("kept_dense_dense", 0) + 1
        return SparseWeight(out_dim, in_dim, False, dense=w.contiguous(),
                            nnz=nnz, fallback_reason="dense",
                            compute_mode=compute_mode)
    csr = w.to_sparse_csr()
    log["sparsified"] = log.get("sparsified", 0) + 1
    return SparseWeight(out_dim, in_dim, True, csr=csr, nnz=nnz,
                        compute_mode=compute_mode)


# ---------------------------------------------------------------------------
# Sparse attention / FFN / block mirrors of blogspec_model.
# ---------------------------------------------------------------------------
class SparseAttn:
    """softmax1 + ALiBi attention with sparse Q/K/V/O (mirrors blogspec Attn)."""

    def __init__(self, attn, density_thresh, min_numel, log, compute_mode):
        self.dim = attn.dim
        self.n_heads = attn.n_heads
        self.head_dim = attn.head_dim
        self.scale = attn.scale
        self.max_seq_len = attn.max_seq_len
        self.alibi_slopes = attn.alibi_slopes.clone()
        mk = lambda w: _make_weight(w.detach(), density_thresh, min_numel, log,
                                    compute_mode)
        self.W_q = mk(attn.W_q)
        self.W_k = mk(attn.W_k)
        self.W_v = mk(attn.W_v)
        self.W_o = mk(attn.W_o)

    def to(self, device):
        self.alibi_slopes = self.alibi_slopes.to(device)
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            w.to(device)
        return self

    def materialize_dense(self, device=None):
        for w in (self.W_q, self.W_k, self.W_v, self.W_o):
            w.materialize_dense(device)
        return self

    def forward(self, x: torch.Tensor, past_kv=None, q_positions=None,
                use_cache: bool = False):
        """softmax1 + ALiBi attention — byte-identical to ``blogspec_model.Attn``.

        Mirrors the dense ``Attn.forward`` exactly for BOTH the default un-cached
        path AND the incremental cached / windowed path that
        ``run_pure_forward_cached`` drives.  Q/K/V/O go through the SAME
        ``F.linear`` (in ``dense_kernel`` mode the sparse ``.linear`` materialises
        the weight and calls ``F.linear``, so the GEMM + accum order are identical
        -> L-inf=0), and the ALiBi / causal / softmax1 math is copied verbatim.
        Returns ``out`` (default) or ``(out, (K, V, k_pos))`` when ``use_cache``.
        """
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim
        Q = self.W_q.linear(x).view(B, S, H, HD).transpose(1, 2)
        Knew = self.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)
        Vnew = self.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)

        if q_positions is None:
            q_pos = torch.arange(S, device=x.device)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)

        if past_kv is not None:
            K_cache, V_cache, pos_cache = past_kv
            K = torch.cat([K_cache, Knew], dim=2)
            V = torch.cat([V_cache, Vnew], dim=2)
            k_pos = torch.cat([pos_cache.to(x.device), q_pos], dim=0)
        else:
            K, V, k_pos = Knew, Vnew, q_pos

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if past_kv is None and q_positions is None:
            pos = torch.arange(S, device=x.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
            scores = scores - self.alibi_slopes.view(1, H, 1, 1) * dist
            causal = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal
        else:
            dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs().float()
            scores = scores - self.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(0)
            mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))     # [Sq, Sk]
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = softmax1(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, D)
        out = x + self.W_o.linear(out)
        if use_cache:
            return out, (K, V, k_pos)
        return out


class SparseFFN:
    """SwiGLU FFN with sparse W_up / W_gate / W_down (mirrors blogspec FFN)."""

    def __init__(self, ffn, density_thresh, min_numel, log, compute_mode):
        mk = lambda w: _make_weight(w.detach(), density_thresh, min_numel, log,
                                    compute_mode)
        self.W_up = mk(ffn.W_up)
        self.b_up = ffn.b_up.detach().clone()
        self.W_gate = mk(ffn.W_gate)
        self.b_gate = ffn.b_gate.detach().clone()
        self.W_down = mk(ffn.W_down)
        self.b_down = ffn.b_down.detach().clone()

    def to(self, device):
        self.b_up = self.b_up.to(device)
        self.b_gate = self.b_gate.to(device)
        self.b_down = self.b_down.to(device)
        for w in (self.W_up, self.W_gate, self.W_down):
            w.to(device)
        return self

    def materialize_dense(self, device=None):
        for w in (self.W_up, self.W_gate, self.W_down):
            w.materialize_dense(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.W_up.linear(x) + self.b_up
        gate = self.W_gate.linear(x) + self.b_gate
        hidden = F.silu(up) * gate
        return x + self.W_down.linear(hidden) + self.b_down


class SparseBlock:
    def __init__(self, block, density_thresh, min_numel, log, compute_mode):
        self.attn = SparseAttn(block.attn, density_thresh, min_numel, log,
                               compute_mode)
        self.ffn = SparseFFN(block.ffn, density_thresh, min_numel, log,
                             compute_mode)

    def to(self, device):
        self.attn.to(device); self.ffn.to(device); return self

    def materialize_dense(self, device=None):
        self.attn.materialize_dense(device); self.ffn.materialize_dense(device)
        return self

    def __call__(self, x, past_kv=None, q_positions=None, use_cache: bool = False):
        if use_cache or past_kv is not None or q_positions is not None:
            a, new_kv = self.attn.forward(x, past_kv=past_kv,
                                          q_positions=q_positions, use_cache=True)
            out = self.ffn.forward(a)
            return (out, new_kv) if use_cache else out
        return self.ffn.forward(self.attn.forward(x))


@dataclass
class SparseStats:
    dense_bytes: int
    sparse_bytes: int
    n_sparsified: int
    n_kept_dense_small: int
    n_kept_dense_dense: int
    total_nnz: int

    @property
    def dense_gb(self) -> float:
        return self.dense_bytes / 1e9

    @property
    def sparse_mb(self) -> float:
        return self.sparse_bytes / 1e6


class SparseTransformer:
    """Drop-in sparse mirror of ``blogspec_model.Transformer`` for the driver.

    Exposes ``.embed`` / ``.blocks`` / ``.dim`` / ``.vocab`` and a ``blk(x)``
    call per block, exactly what ``run_pure_forward_complete`` uses.  The forward
    is byte-identical to the dense ``Transformer`` on the default (un-cached)
    path.  The embedding and LM head stay dense (embedding is a gather, not a
    mat-mul; the LM head is only used by the ``.forward`` classifier which the
    driver doesn't call — it reads the residual bands directly).
    """

    def __init__(self, model, density_thresh: float = 0.25,
                 min_numel: int = 4096, compute_mode: str = "sparse_mm"):
        """``compute_mode``: ``"sparse_mm"`` (compute win; fp-accum-order residue
        only, argmax-decode-identical) or ``"dense_kernel"`` (bit-identical
        L-inf=0 to the dense forward; storage still sparse, compute densifies
        one weight transiently)."""
        self.dim = model.dim
        self.vocab = model.vocab
        self.max_seq_len = model.max_seq_len
        self.compute_mode = compute_mode
        self.embed = model.embed.detach().clone()      # gather, kept dense
        self.lm_head = model.lm_head.detach().clone()
        self.lm_bias = model.lm_bias.detach().clone()
        log: Dict[str, int] = {}
        self.blocks = [SparseBlock(b, density_thresh, min_numel, log, compute_mode)
                       for b in model.blocks]
        self._log = log

    def to(self, device):
        self.embed = self.embed.to(device)
        self.lm_head = self.lm_head.to(device)
        self.lm_bias = self.lm_bias.to(device)
        for b in self.blocks:
            b.to(device)
        return self

    def materialize_dense(self, device=None):
        """Densify EVERY block's CSR weights ONCE onto ``device`` (block-by-block
        so peak transient memory is one block's dense weights, not all at once).
        After this, ``linear`` uses the resident dense weight — no per-forward
        ``to_dense``.  Bit-identical to ``dense_kernel`` (same F.linear GEMM).

        VRAM cost is the full dense model (~3.5 GB lean / ~10 GB divmod), so call
        it only when the card can hold model + KV caches + batch activations.
        """
        for b in self.blocks:
            b.materialize_dense(device)
        if device is not None and device != "cpu":
            self.embed = self.embed.to(device)
            self.lm_head = self.lm_head.to(device)
            self.lm_bias = self.lm_bias.to(device)
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed[tokens]
        for blk in self.blocks:
            x = blk(x)
        return F.linear(x, self.lm_head, self.lm_bias)

    def forward_hidden_cached(self, x, past_key_values=None, q_positions=None,
                              use_cache: bool = False):
        """KV-cached block-stack forward (no LM head) — the API
        ``run_pure_forward_cached`` drives.

        Byte-identical to ``blogspec_model.Transformer.forward_hidden_cached``:
        the SPARSE blocks compute the same GEMMs (``dense_kernel`` -> L-inf=0),
        so the incremental K/V cache and the returned hidden are the same the
        dense model would produce.  ``x`` is the NEW query rows' residual
        (``[B, Snew, D]``); ``q_positions`` their absolute positions;
        ``past_key_values`` a per-block ``(K, V, pos)`` list (or ``None``).
        Returns ``(hidden, new_caches)``.
        """
        n = len(self.blocks)
        if past_key_values is None:
            past_key_values = [None] * n
        new_caches = []
        for blk, pkv in zip(self.blocks, past_key_values):
            x, kv = blk(x, past_kv=pkv, q_positions=q_positions, use_cache=True)
            new_caches.append(kv)
        return x, new_caches

    # -- storage accounting -------------------------------------------------
    def stats(self) -> SparseStats:
        dense_bytes = 0
        sparse_bytes = 0
        total_nnz = 0
        # embed + lm head + biases (dense)
        for t in (self.embed, self.lm_head, self.lm_bias):
            dense_bytes += t.numel() * 4
            sparse_bytes += t.numel() * 4
            total_nnz += int((t != 0).sum().item())
        for b in self.blocks:
            for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o,
                      b.ffn.W_up, b.ffn.W_gate, b.ffn.W_down):
                dense_bytes += w.dense_equiv_bytes()
                sparse_bytes += w.storage_bytes()
                total_nnz += w.nnz
            for v in (b.ffn.b_up, b.ffn.b_gate, b.ffn.b_down, b.attn.alibi_slopes):
                dense_bytes += v.numel() * 4
                sparse_bytes += v.numel() * 4
        return SparseStats(
            dense_bytes=dense_bytes, sparse_bytes=sparse_bytes,
            n_sparsified=self._log.get("sparsified", 0),
            n_kept_dense_small=self._log.get("kept_dense_small", 0),
            n_kept_dense_dense=self._log.get("kept_dense_dense", 0),
            total_nnz=total_nnz)
