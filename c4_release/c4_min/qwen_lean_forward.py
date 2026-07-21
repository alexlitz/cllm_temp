"""LEAN native forward of the COMPACTED (7-14 layer, 6-head) fused C4 VM.

This is the **performance foundation** for the fused Qwen VM
(``qwen_full_vm``).  ``qwen_full_vm.build`` bakes the whole VM step into a genuine
``transformers.Qwen2Model`` (compaction: 5 register CAM heads + 1 memory head, each
register gathered on ONE frame token by a single content-match + recency).  That is
the *correct* reference, but the HF wrapper carries per-forward machinery — a
``Cache`` object, ``position_ids`` / ``cache_position`` plumbing, attention-mask
construction, the ALL_ATTENTION_FUNCTIONS dispatch, ``dynamic_rope_update`` — that
is pure overhead on the deterministic single-token-window VM step.

Here we REUSE ``qwen_full_vm.build``'s bake (the WEIGHTS are byte-identical — we
copy them straight out of the built ``Qwen2Model``) and run them through a
hand-written minimal decoder-only forward: RoPE + RMSNorm + softmax + SwiGLU, GQA
``repeat_kv``, no ``Cache``/dispatch/mask machinery.  The math is IDENTICAL to
``transformers.models.qwen2.modeling_qwen2`` (verified argmax byte-exact vs the HF
model, vs ``isa.interpret``, in ``test_qwen_lean_forward`` and the head-to-head
bench), so the lean forward decodes the SAME bytes — only the *evaluator* is lean.

Entry points (the reusable foundation for the follow-on optimization agents)
============================================================================
  * ``LeanQwenVM.from_full_vm(vm, device)`` — extract the baked weights of a built
    ``qwen_full_vm.QwenFullVM`` into flat tensors on a target device.  ONE bake,
    two evaluators (HF + lean) share the identical weights.
  * ``LeanQwenVM.forward(x, past=None, q_positions=None)`` — the lean block-stack
    forward.  ``past`` is a per-layer ``[(K,V,pos), ...]`` KV cache (or None); with
    a cache ``x`` is only the NEW query rows and ``q_positions`` their absolute
    positions.  Returns ``(hidden [B,S,H], new_past)``.  This is the single hook the
    CUDA-graph / fusion agent wraps.
  * ``run_program_lean(lean, code, ...)`` — the naive one-forward-per-VM-step driver
    (mirrors ``qwen_full_vm.run_program`` exactly; state round-trips through the
    windowed token stream).  Byte-exact vs ``isa.interpret``.
  * ``speculative_run_lean(lean, code, ...)`` — PERFECT-DRAFT speculation on the lean
    forward: the deterministic VM drafts the whole register trace for free, then the
    lean forward VERIFIES K steps per forward against a per-layer KV cache with
    optional bounded eviction.  Returns forwards-saved + the decoded trace.

The follow-on agents (CUDA-graph/fusion, async KV-eviction, iterative muldiv) plug
into ``LeanQwenVM.forward`` + ``speculative_run_lean`` without touching the bake.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .nibble_pure_forward import SP_INIT
from .qwen_full_vm import (
    QwenFullVM, CAM_REGS, _REG_TOKEN, _snap, _address_bits, _signed_imm,
)


# ===========================================================================
# The lean weight bundle — flat tensors extracted from the built Qwen2Model.
# ===========================================================================
@dataclass
class _LeanLayer:
    ln1: torch.Tensor                       # input RMSNorm gamma
    ln2: torch.Tensor                       # post-attention RMSNorm gamma
    q_w: torch.Tensor
    k_w: torch.Tensor
    v_w: torch.Tensor
    o_w: torch.Tensor
    q_b: Optional[torch.Tensor]
    k_b: Optional[torch.Tensor]
    v_b: Optional[torch.Tensor]
    gate_w: torch.Tensor
    up_w: torch.Tensor
    down_w: torch.Tensor


@dataclass
class LeanQwenVM:
    """A LEAN native forward of a baked ``qwen_full_vm.QwenFullVM``.

    Holds the SAME weights as the HF ``Qwen2Model`` (copied out at construction) and
    a hand-written RoPE + RMSNorm + softmax + SwiGLU forward — no ``transformers``
    machinery on the compute path.  Decode is argmax-identical to the HF path.
    """

    layers: List[_LeanLayer]
    final_norm: torch.Tensor
    embed: torch.Tensor                     # [vocab, H] token -> residual
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_eps: float
    device: torch.device
    dtype: torch.dtype
    QL: object                              # the VM layout (shared with qwen_full_vm)
    subset: object
    inv_freq: torch.Tensor = field(default=None)

    # ------------------------------------------------------------------
    @classmethod
    def from_full_vm(cls, vm: QwenFullVM, device: str = "cpu",
                     dtype: torch.dtype = torch.float32) -> "LeanQwenVM":
        """Extract the baked weights of ``vm`` (a built ``QwenFullVM``) into flat
        tensors on ``device``.  The HF ``Qwen2Model`` and this lean bundle then
        share BYTE-IDENTICAL weights; only the forward differs."""
        qm = vm.qmodel
        cfg = qm.config
        dev = torch.device(device)
        layers: List[_LeanLayer] = []
        with torch.no_grad():
            for layer in qm.layers:
                sa = layer.self_attn
                mlp = layer.mlp

                def g(lin):
                    return lin.weight.detach().to(dev, dtype).clone()

                def b(lin):
                    return (lin.bias.detach().to(dev, dtype).clone()
                            if lin.bias is not None else None)

                layers.append(_LeanLayer(
                    ln1=layer.input_layernorm.weight.detach().to(dev, dtype).clone(),
                    ln2=layer.post_attention_layernorm.weight.detach().to(dev, dtype).clone(),
                    q_w=g(sa.q_proj), k_w=g(sa.k_proj), v_w=g(sa.v_proj), o_w=g(sa.o_proj),
                    q_b=b(sa.q_proj), k_b=b(sa.k_proj), v_b=b(sa.v_proj),
                    gate_w=g(mlp.gate_proj), up_w=g(mlp.up_proj), down_w=g(mlp.down_proj),
                ))
            final_norm = qm.norm.weight.detach().to(dev, dtype).clone()
        embed = vm.embed.detach().to(dev, dtype).clone()
        head_dim = cfg.head_dim
        half = head_dim // 2
        # HF Qwen2 default RoPE inv_freq: theta^{-2i/head_dim}, i in 0..head_dim/2.
        inv_freq = 1.0 / (cfg.rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev)[:half] / head_dim))
        return cls(
            layers=layers, final_norm=final_norm, embed=embed,
            hidden_size=cfg.hidden_size, n_layers=cfg.num_hidden_layers,
            n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim, rope_theta=cfg.rope_theta, rms_eps=cfg.rms_norm_eps,
            device=dev, dtype=dtype, QL=vm.QL, subset=vm.subset, inv_freq=inv_freq)

    # ------------------------------------------------------------------
    # LEAN forward — RoPE + RMSNorm + softmax + SwiGLU, no HF machinery.
    # ------------------------------------------------------------------
    def _rmsnorm(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.rms_eps) * gamma

    def _rope_cos_sin(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for absolute ``positions`` ([S] or [B,S]) — the HF Qwen2 layout
        ``emb = cat(freqs, freqs)`` (attention_scaling = 1.0 for default rope).
        Returns cos/sin shaped ``positions.shape + (head_dim,)``."""
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq         # [...,half]
        emb = torch.cat([freqs, freqs], dim=-1)                                   # [...,HD]
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, t: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor) -> torch.Tensor:
        # t: [B, H, S, HD]; cos/sin: [B, S, HD] -> insert head axis at dim 1.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (t * cos) + (self._rotate_half(t) * sin)

    def _attn(self, layer: _LeanLayer, xn: torch.Tensor,
              past: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
              q_pos: torch.Tensor):
        """One attention sub-layer on the NORMED input ``xn`` [B, S, H].

        ``past`` = (K_cache [B,Hkv,Sc,HD], V_cache, pos_cache [B,Sc]) or None.
        ``q_pos`` [B, S] = absolute positions of the query rows.  Returns
        ``(attn_out [B,S,H], (K_all, V_all, pos_all[B,Sk]))``."""
        B, S, H = xn.shape
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        scale = hd ** -0.5
        q = F.linear(xn, layer.q_w, layer.q_b).view(B, S, nh, hd).transpose(1, 2)
        k = F.linear(xn, layer.k_w, layer.k_b).view(B, S, nkv, hd).transpose(1, 2)
        v = F.linear(xn, layer.v_w, layer.v_b).view(B, S, nkv, hd).transpose(1, 2)

        cos_q, sin_q = self._rope_cos_sin(q_pos)                          # [B,S,HD]
        q = self._apply_rope(q, cos_q, sin_q)
        k = self._apply_rope(k, cos_q, sin_q)

        if past is not None and past[0] is not None:
            K_cache, V_cache, pos_cache = past
            K = torch.cat([K_cache, k], dim=2)
            Vv = torch.cat([V_cache, v], dim=2)
            k_pos = torch.cat([pos_cache, q_pos], dim=1)                  # [B,Sk]
        else:
            K, Vv, k_pos = k, v, q_pos

        n_rep = nh // nkv
        if n_rep != 1:
            Sk = K.shape[2]
            Kr = K[:, :, None, :, :].expand(B, nkv, n_rep, Sk, hd).reshape(B, nh, Sk, hd)
            Vr = Vv[:, :, None, :, :].expand(B, nkv, n_rep, Sk, hd).reshape(B, nh, Sk, hd)
        else:
            Kr, Vr = K, Vv

        scores = torch.matmul(q, Kr.transpose(-2, -1)) * scale           # [B,nh,S,Sk]
        # causal mask over ABSOLUTE positions, per batch row (pad rows sit at a far
        # position so they never attend and are never attended to).
        mask = (k_pos.unsqueeze(1) > q_pos.unsqueeze(2))                  # [B,S,Sk]
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
        # concat heads -> [B, S, nh*hd] (o_proj maps nh*hd -> hidden_size).
        out = torch.matmul(attn, Vr).transpose(1, 2).contiguous().view(B, S, nh * hd)
        out = F.linear(out, layer.o_w)
        return out, (K, Vv, k_pos)

    def forward(self, x: torch.Tensor,
                past: Optional[List] = None,
                q_positions: Optional[torch.Tensor] = None):
        """Lean block-stack forward over the embedded/overlaid residual ``x``
        [B, S, H].

        ``past`` = per-layer ``[(K,V,pos), ...]`` KV cache (or None for a fresh
        forward).  With a cache, ``x`` is only the NEW rows and ``q_positions``
        ([S] or [B,S]) their absolute positions; without one the rows sit at
        0..S-1.  Returns ``(hidden [B,S,H], new_past)`` — ``new_past`` the updated
        per-layer cache.  The register/nibble decode reads ``hidden[:, -1]``.
        """
        B, S, H = x.shape
        if q_positions is None:
            q_pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)
            if q_pos.dim() == 1:
                q_pos = q_pos.unsqueeze(0).expand(B, S)
        if past is None:
            past = [None] * self.n_layers
        new_past: List = []
        h = x
        for li, layer in enumerate(self.layers):
            xn = self._rmsnorm(h, layer.ln1)
            a, kv = self._attn(layer, xn, past[li], q_pos)
            h = h + a
            xn2 = self._rmsnorm(h, layer.ln2)
            mlp = F.linear(F.silu(F.linear(xn2, layer.gate_w)) * F.linear(xn2, layer.up_w),
                           layer.down_w)
            h = h + mlp
            new_past.append(kv)
        h = self._rmsnorm(h, self.final_norm)
        return h, new_past


# ===========================================================================
# The naive driver — one VM step = one lean forward (mirrors qwen_full_vm.run_program).
# ===========================================================================
def _build_stream_and_overlay(lean: LeanQwenVM, code: List[isa.Instr],
                              reg_state: dict, store_log: List[dict],
                              load_addr: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the windowed token stream + overlaid residual for ONE lean forward.
    Byte-identical window to ``qwen_full_vm._build_stream_and_overlay`` (BOS sink +
    persistent store frames + latest register frame + STEP_END query row).  Returns
    ``(x [1,S,H], positions [S])``."""
    QL, L = lean.QL, lean.QL.L
    subset = lean.subset
    from .blogspec_memory import ADDR_BITS

    n_store = len(store_log) if subset.memory else 0
    stream: List[int] = [V.BOS]
    stream += [V.MEM] * n_store
    stream += [_REG_TOKEN[r] for r in CAM_REGS]
    stream += [V.STEP_END]
    toks = torch.tensor([stream], device=lean.device)
    x = lean.embed[toks].clone()
    Sn = x.shape[1]

    for i in range(Sn):
        x[0, i, L.ONE] = 1.0
        for k, ins in enumerate(code):
            x[0, i, L.CODE_OP[k]] = float(ins.op)
            x[0, i, L.CODE_IMM[k]] = float(_signed_imm(ins.imm))

    if subset.memory:
        for si, st in enumerate(store_log):
            p = 1 + si
            x[0, p, L.IS_STORE] = 1.0
            for b, bit in enumerate(_address_bits(st["addr"], ADDR_BITS)):
                x[0, p, L.ADDR_BIN + b] = bit
            for j, nv in enumerate(V.nibbles_of_value(st["val"], NIB_PER_REG)):
                x[0, p, L.VAL_NIB + j] = float(nv)

    reg0 = 1 + n_store
    for hh, reg in enumerate(CAM_REGS):
        p = reg0 + hh
        for j, nv in enumerate(V.nibbles_of_value(reg_state[reg], NIB_PER_REG)):
            x[0, p, QL.TOK_NIB + j] = float(nv)
        x[0, p, QL.ROLE + hh] = 1.0
        x[0, p, QL.IS_TOK] = 1.0

    for hh in range(len(CAM_REGS)):
        x[0, -1, QL.ROLE + hh] = 1.0
    if subset.memory and load_addr is not None:
        x[0, -1, L.IS_LOAD] = 1.0
        for b, bit in enumerate(_address_bits(load_addr, ADDR_BITS)):
            x[0, -1, L.QRY_BIN + b] = float(bit)
    positions = torch.arange(Sn, device=lean.device)
    return x, positions


def run_program_lean(lean: LeanQwenVM, code: List[isa.Instr], max_steps: int = 64,
                     verbose: bool = False) -> Dict[str, object]:
    """Execute ``code`` on the LEAN fused VM (one VM step = one lean forward).

    Byte-identical driver to ``qwen_full_vm.run_program`` — the register state
    round-trips through the windowed token stream, the op is computed in the SwiGLU
    MLPs, control-flow inside the forward.  Returns
    ``{"ax_trace","ref_trace","exact","steps"}``."""
    QL, L = lean.QL, lean.QL.L
    subset = lean.subset
    # Match the reference oracle's step budget to the driver's so a loop longer than
    # the default isa.interpret cap (256 steps) is not truncated against a
    # run-to-completion model trace (#691 BUG 2: countdown >= 64 needs > 256 steps).
    ref_trace = isa.interpret(code, max_steps=max_steps)

    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    store_log: List[dict] = []
    ax_trace: List[int] = []
    cur_pc = 0
    call_stack: List[Tuple[Optional[int], Optional[int]]] = []

    for _ in range(max_steps):
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF

        x, positions = _build_stream_and_overlay(lean, code, reg_state, store_log, load_addr)
        with torch.no_grad():
            hidden, _ = lean.forward(x, past=None, q_positions=positions)
        state = hidden[0, -1]

        pc = _snap(state[L.PC_VAL])
        ax = _snap(state[L.AX_VAL]) & 0xFF
        sp = _snap(state[L.SP_VAL])
        bp = _snap(state[L.BP_VAL])
        stk = _snap(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5

        if op == isa.JSR:
            call_stack.append((cur_pc + 1, bp))
        elif op == isa.ENT:
            call_stack.append((None, prev["BP"]))
        elif op == isa.LEV:
            saved_bp = ret_pc = None
            if call_stack:
                _, saved_bp = call_stack.pop()
            if call_stack:
                ret_pc, _ = call_stack.pop()
            if saved_bp is not None:
                bp = saved_bp
            if ret_pc is not None:
                pc = ret_pc
        elif subset.memory and op in (isa.SI, isa.SC):
            store_addr = _snap(state[L.STK_VAL])
            store_val = ax if op == isa.SI else (ax & 0xFF)
            store_log = [s for s in store_log
                         if (s["addr"] & 0xFF) != (store_addr & 0xFF)]
            store_log.append({"addr": store_addr, "val": store_val})

        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": stk}
        ax_trace.append(ax)
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):5s} -> "
                  f"pc={pc} ax={ax} sp={sp} bp={bp} stk={stk} halt={halted}")
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(code):
            break
    return {"ax_trace": ax_trace, "ref_trace": ref_trace,
            "exact": ax_trace == ref_trace, "steps": len(ax_trace)}


# ===========================================================================
# PERFECT-DRAFT SPECULATION on the lean forward.
#
# The fused VM is deterministic, so the driver's exact per-step register + store
# bookkeeping is a FREE perfect draft.  We draft the whole program in Python, then
# the lean forward VERIFIES K steps per forward: one padded window carrying the BOS
# sink + the (compacted) store log + K register frames (one per drafted step) + a
# STEP_END query row per step, against the block stack.  Each step's query row
# attends over its own register frame + the persistent store log, exactly as the
# naive per-step window does, so the decoded state is byte-identical to the naive
# driver — speculation only verifies the model in parallel.
# ===========================================================================
@dataclass
class LeanDraft:
    steps: List[dict]             # per step: reg_state BEFORE, load_addr, store_log snapshot, op
    ref_trace: List[int]
    halted: bool


def draft_program_lean(lean: LeanQwenVM, code: List[isa.Instr],
                       max_steps: int = 4096) -> LeanDraft:
    """Run the driver's EXACT transition in pure Python (zero forwards) and record,
    per step, the (register state, load address, store-log snapshot) the naive
    driver would feed the model — the perfect draft to verify in parallel.

    Mirrors ``run_program_lean``'s control loop but computes the next register state
    with the EXACT model transition (``base_dispatch_rules`` semantics: STACK0 is a
    SINGLE register PSH sets and pop-ops read; SP is -=4 on PSH, +=4 on a pop; AX is
    8-bit-folded) instead of a forward, so the drafted per-step register file — the
    exact input the naive driver feeds the model — is byte-identical.  Returns an
    EMPTY draft (fall back to naive) if the program uses an op outside the
    speculation slice (functions JSR/ENT/LEV, which need the driver's call stack)."""
    subset = lean.subset
    # Match the reference oracle to the draft's step budget so a loop longer than the
    # default isa.interpret cap (256) is not truncated against the full drafted trace
    # (#691 BUG 2). The draft's own loop runs up to max_steps, so the golden must too.
    ref_trace = isa.interpret(code, max_steps=max_steps)

    # register file mirrored EXACTLY as the model computes it (see base_dispatch_rules).
    pc = 0
    ax = 0
    sp = SP_INIT
    bp = SP_INIT
    stk = 0                       # STACK0 register (PSH sets it, a pop reads it)
    mem: Dict[int, int] = {}
    store_log: List[dict] = []
    steps: List[dict] = []
    halted = False
    POP_OPS = (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
               isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
               isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
               isa.SI, isa.SC)

    for _ in range(max_steps):
        if not (0 <= pc < len(code)):
            break
        op = code[pc].op
        imm = code[pc].imm
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = ax & 0xFF
        # snapshot the PRE-step register file (what the naive driver feeds the model).
        steps.append({
            "reg_state": {"PC": pc, "AX": ax & 0xFF, "SP": sp, "BP": bp, "STACK0": stk},
            "store_log": [dict(s) for s in store_log],
            "load_addr": load_addr, "op": op, "pc_at": pc,
        })
        npc = pc + 1
        halted_step = False
        addr = 0
        v = stk & 0xFF            # the value a pop reads (STACK0)
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + imm) & 0xFF
        elif op == isa.PSH:
            stk = ax & 0xFF; sp -= 4
        elif op in POP_OPS:
            sp += 4
            if op == isa.ADD: ax = (v + ax) & 0xFF
            elif op == isa.SUB: ax = (v - ax) & 0xFF
            elif op == isa.MUL: ax = (v * ax) & 0xFF
            elif op == isa.DIV: ax = ((v // ax) if ax else 0) & 0xFF
            elif op == isa.MOD: ax = ((v % ax) if ax else 0) & 0xFF
            elif op == isa.AND: ax = v & ax
            elif op == isa.OR: ax = v | ax
            elif op == isa.XOR: ax = v ^ ax
            elif op == isa.SHL: ax = (v << ax) & 0xFF
            elif op == isa.SHR: ax = (v >> ax) & 0xFF
            elif op == isa.EQ: ax = 1 if v == ax else 0
            elif op == isa.NE: ax = 1 if v != ax else 0
            elif op == isa.LT: ax = 1 if v < ax else 0
            elif op == isa.GT: ax = 1 if v > ax else 0
            elif op == isa.LE: ax = 1 if v <= ax else 0
            elif op == isa.GE: ax = 1 if v >= ax else 0
            elif op in (isa.SI, isa.SC):
                addr = v; mem[addr & 0xFF] = ax & 0xFF
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax & 0xFF, 0) & 0xFF
        elif op == isa.JMP:
            npc = imm
        elif op == isa.BZ:
            npc = imm if (ax & 0xFF) == 0 else npc
        elif op == isa.BNZ:
            npc = imm if (ax & 0xFF) != 0 else npc
        elif op == isa.HALT:
            halted_step = True
        elif op == isa.NOP:
            pass
        else:
            # out-of-slice op (functions, syscalls) -> caller falls back to naive.
            return LeanDraft(steps=[], ref_trace=ref_trace, halted=False)

        if subset.memory and op in (isa.SI, isa.SC):
            store_log = [s for s in store_log
                         if (s["addr"] & 0xFF) != (addr & 0xFF)]
            store_log.append({"addr": addr, "val": ax & 0xFF})
        pc = npc
        if halted_step or pc < 0 or pc >= len(code):
            halted = True
            break
    return LeanDraft(steps=steps, ref_trace=ref_trace, halted=halted)


def _build_spec_batch(lean: LeanQwenVM, code: List[isa.Instr],
                      drafted: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack the per-step windows of ``drafted`` (a slice of draft steps) into ONE
    batched ``[B, S, H]`` residual — B independent single-step windows (each the
    BYTE-IDENTICAL window the naive driver builds for that step: BOS sink + the
    step's compacted store log + its register frame + a STEP_END query row).

    Because each batch row is a SELF-CONTAINED naive window (no cross-step attention;
    padded rows are causally invisible), the decoded state at every row's last
    position is byte-identical to the naive per-step forward — so this verifies B
    VM steps in ONE lean forward with ZERO approximation.  The compacted CAM's
    "exactly one frame per role in the window" invariant is preserved per row.
    Returns ``(x [B, Smax, H], positions [B, Smax])`` (pad rows sit at a far position
    so the causal mask drops them)."""
    QL, L = lean.QL, lean.QL.L
    subset = lean.subset
    from .blogspec_memory import ADDR_BITS

    # per-step token windows (BOS + that step's store log + frame + query).
    windows: List[List[int]] = []
    step_stores: List[List[dict]] = []
    for st in drafted:
        store_rows = st["store_log"] if subset.memory else []
        step_stores.append(store_rows)
        stream = [V.BOS] + [V.MEM] * len(store_rows)
        stream += [_REG_TOKEN[r] for r in CAM_REGS] + [V.STEP_END]
        windows.append(stream)
    B = len(windows)
    Smax = max(len(w) for w in windows)

    win_toks = torch.zeros(B, Smax, dtype=torch.long, device=lean.device)
    positions = torch.zeros(B, Smax, dtype=torch.long, device=lean.device)
    PAD_POS = 10_000_000
    for i, w in enumerate(windows):
        s = len(w)
        win_toks[i, :s] = torch.tensor(w, device=lean.device)
        positions[i, :s] = torch.arange(s, device=lean.device)
        positions[i, s:] = PAD_POS + torch.arange(Smax - s, device=lean.device)
    x = lean.embed[win_toks].clone()

    for i, st in enumerate(drafted):
        w = windows[i]
        s = len(w)
        store_rows = step_stores[i]
        n_store = len(store_rows)
        # program-in-data on the real rows of this window.
        for p in range(s):
            x[i, p, L.ONE] = 1.0
            for k, ins in enumerate(code):
                x[i, p, L.CODE_OP[k]] = float(ins.op)
                x[i, p, L.CODE_IMM[k]] = float(_signed_imm(ins.imm))
        # store frames.
        if subset.memory:
            for sj, srow in enumerate(store_rows):
                p = 1 + sj
                x[i, p, L.IS_STORE] = 1.0
                for b, bit in enumerate(_address_bits(srow["addr"], ADDR_BITS)):
                    x[i, p, L.ADDR_BIN + b] = bit
                for j, nv in enumerate(V.nibbles_of_value(srow["val"], NIB_PER_REG)):
                    x[i, p, L.VAL_NIB + j] = float(nv)
        # the register frame.
        reg0 = 1 + n_store
        reg_state = st["reg_state"]
        for hh, reg in enumerate(CAM_REGS):
            p = reg0 + hh
            for j, nv in enumerate(V.nibbles_of_value(reg_state[reg], NIB_PER_REG)):
                x[i, p, QL.TOK_NIB + j] = float(nv)
            x[i, p, QL.ROLE + hh] = 1.0
            x[i, p, QL.IS_TOK] = 1.0
        # the query row (last real row of this window).
        qrow = s - 1
        for hh in range(len(CAM_REGS)):
            x[i, qrow, QL.ROLE + hh] = 1.0
        if subset.memory and st["load_addr"] is not None:
            x[i, qrow, L.IS_LOAD] = 1.0
            for b, bit in enumerate(_address_bits(st["load_addr"], ADDR_BITS)):
                x[i, qrow, L.QRY_BIN + b] = float(bit)
    return x, positions


@dataclass
class LeanSpecResult:
    status: str                          # PASS | FAIL
    ax_trace: List[int]
    ref_trace: List[int]
    exact: bool
    steps: int
    forwards: int                        # lean forwards run (speculative cost)
    naive_forwards: int                  # == steps
    speedup: float
    accepted: int
    detail: str = ""


def speculative_run_lean(lean: LeanQwenVM, code: List[isa.Instr], *,
                         block_steps: int = 32, max_steps: int = 4096,
                         verbose: bool = False) -> LeanSpecResult:
    """Perfect-draft speculation on the lean forward.

    Drafts the whole program with the deterministic VM (free), then verifies
    ``block_steps`` steps per lean forward.  Each verified step decodes its AX at its
    STEP_END row; the program PASSes iff the decoded AX trace matches
    ``isa.interpret``.  Returns forwards-saved (``naive_forwards / forwards``).
    Falls back to the naive per-step driver for out-of-slice programs (functions)."""
    draft = draft_program_lean(lean, code, max_steps=max_steps)
    ref_trace = draft.ref_trace
    if not draft.steps:
        r = run_program_lean(lean, code, max_steps=max_steps, verbose=verbose)
        n = r["steps"]
        return LeanSpecResult(
            status="PASS" if r["exact"] else "FAIL", ax_trace=r["ax_trace"],
            ref_trace=r["ref_trace"], exact=r["exact"], steps=n, forwards=n,
            naive_forwards=n, speedup=1.0, accepted=n, detail="naive-fallback")

    QL, L = lean.QL, lean.QL.L
    n_steps = len(draft.steps)
    ax_trace: List[int] = []
    forwards = 0
    accepted = 0
    for s0 in range(0, n_steps, block_steps):
        slab = draft.steps[s0:s0 + block_steps]
        # stack the per-step windows into ONE batched forward (each row is the
        # byte-identical naive window for that step; the last real row is its query).
        x, positions = _build_spec_batch(lean, code, slab)
        with torch.no_grad():
            hidden, _ = lean.forward(x, past=None, q_positions=positions)
        forwards += 1
        # each row's query row is the last REAL position (positions < PAD_POS).
        for i, st in enumerate(slab):
            n_store = len(st["store_log"]) if lean.subset.memory else 0
            qrow = (1 + n_store) + len(CAM_REGS)          # BOS + stores + 5 regs + STEP_END
            state = hidden[i, qrow]
            ax = _snap(state[L.AX_VAL]) & 0xFF
            ax_trace.append(ax)
            accepted += 1
    exact = ax_trace == ref_trace
    speedup = (n_steps / forwards) if forwards else 0.0
    return LeanSpecResult(
        status="PASS" if exact else "FAIL", ax_trace=ax_trace, ref_trace=ref_trace,
        exact=exact, steps=n_steps, forwards=forwards, naive_forwards=n_steps,
        speedup=speedup, accepted=accepted,
        detail="" if exact else "spec trace != isa.interpret")
