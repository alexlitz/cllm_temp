#!/usr/bin/env python3
"""Memory-frugal lean extraction for the RECURRENT (divmod-folded) full-ISA model.

``LeanQwenVM.from_full_vm`` iterates ``qmodel.layers`` — which for a recurrent
divmod build is the EXPANDED 102-entry apply sequence (repeated references to 39
distinct modules).  It ``.clone()``s each, materialising 102 distinct GPU tensors
(~31 GB) → OOM on a 24 GB card.

This builder extracts the 39 DISTINCT physical layers ONCE (to the GPU), keeps the
``apply_order`` (102 indices), and drives the lean forward through them — so the
recurrent full-ISA doom model (SUBSET_MULDIV, ~11.85 GB) fits.  The math is identical
to the expanded path (same weights, same apply order), so it stays byte-exact.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from c4_min.qwen_lean_forward import LeanQwenVM, _LeanLayer


def build_lean_recurrent(vm, device: str, dtype: torch.dtype = torch.float32
                         ) -> LeanQwenVM:
    """Extract the DISTINCT physical layers of a (possibly recurrent) ``vm`` to
    ``device`` and return a ``RecurrentLeanQwenVM`` that iterates them via the
    apply order.  Falls back to the normal linear order for a non-recurrent vm."""
    qm = vm.qmodel
    cfg = qm.config
    dev = torch.device(device)
    phys = getattr(qm, "_phys_layers", None)
    if phys is None:
        phys = list(qm.layers)
        apply_order = list(range(len(phys)))
    else:
        # recover the apply order from L._qwen_apply (the indices into phys).
        apply_order = list(getattr(vm.QL.L, "_qwen_apply", range(len(phys))))

    layers: List[_LeanLayer] = []
    with torch.no_grad():
        for layer in phys:                    # 39 DISTINCT modules, extracted ONCE
            sa = layer.self_attn
            mlp = layer.mlp

            def g(lin):
                return lin.weight.detach().to(dev, dtype)

            def b(lin):
                return (lin.bias.detach().to(dev, dtype)
                        if lin.bias is not None else None)

            layers.append(_LeanLayer(
                ln1=layer.input_layernorm.weight.detach().to(dev, dtype),
                ln2=layer.post_attention_layernorm.weight.detach().to(dev, dtype),
                q_w=g(sa.q_proj), k_w=g(sa.k_proj), v_w=g(sa.v_proj), o_w=g(sa.o_proj),
                q_b=b(sa.q_proj), k_b=b(sa.k_proj), v_b=b(sa.v_proj),
                gate_w=g(mlp.gate_proj), up_w=g(mlp.up_proj), down_w=g(mlp.down_proj),
            ))
        final_norm = qm.norm.weight.detach().to(dev, dtype)
    embed = vm.embed.detach().to(dev, dtype)
    head_dim = cfg.head_dim
    half = head_dim // 2
    inv_freq = 1.0 / (cfg.rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev)[:half] / head_dim))

    lean = RecurrentLeanQwenVM(
        layers=layers, final_norm=final_norm, embed=embed,
        hidden_size=cfg.hidden_size, n_layers=len(apply_order),
        n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim, rope_theta=cfg.rope_theta, rms_eps=cfg.rms_norm_eps,
        device=dev, dtype=dtype, QL=vm.QL, subset=vm.subset, inv_freq=inv_freq,
        code_from_memory=vm.code_from_memory,
        efficient_alu=getattr(vm, "efficient_alu", False),
        shift_via_mul=getattr(vm, "shift_via_mul", False))
    lean.apply_order = apply_order
    return lean


class RecurrentLeanQwenVM(LeanQwenVM):
    """LeanQwenVM whose forward iterates ``self.layers`` via ``self.apply_order``
    (the recurrence), so the 39 distinct layers are applied 102x without storing
    102 copies.  Byte-identical to the expanded linear forward."""

    apply_order: List[int] = None

    def forward(self, x: torch.Tensor, past: Optional[List] = None,
                q_positions: Optional[torch.Tensor] = None):
        B, S, H = x.shape
        if q_positions is None:
            q_pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)
            if q_pos.dim() == 1:
                q_pos = q_pos.unsqueeze(0).expand(B, S)
        order = self.apply_order if self.apply_order is not None else range(len(self.layers))
        if past is None:
            past = [None] * len(order)
        new_past: List = []
        h = x
        for step_i, li in enumerate(order):
            layer = self.layers[li]
            xn = self._rmsnorm(h, layer.ln1)
            a, kv = self._attn(layer, xn, past[step_i], q_pos)
            h = h + a
            xn2 = self._rmsnorm(h, layer.ln2)
            mlp = F.linear(F.silu(F.linear(xn2, layer.gate_w)) * F.linear(xn2, layer.up_w),
                           layer.down_w)
            h = h + mlp
            new_past.append(kv)
        h = self._rmsnorm(h, self.final_norm)
        return h, new_past
