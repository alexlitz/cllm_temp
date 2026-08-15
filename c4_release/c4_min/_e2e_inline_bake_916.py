"""#916 END-TO-END BAKE + RUN — assemble the WHOLE inline+feed-forward full-ISA
(base + inline fp64 log-sink divmod + all other opcodes) into ONE block-spec model
and RUN real programs through the ACTUAL constructed forward, byte-exact vs
``isa.interpret``.

This closes the #916 GAP the knife-edge audit named: the per-stage ledger proved
each stage byte-exact IN ISOLATION and the base shave end-to-end over 8 branching
programs, but NOBODY had assembled ALL stages (base + inline divmod) into ONE model
and run a program that EXERCISES the inline divmod through the actual layers.

The assembled model is exactly ``qwen_full_vm._block_specs(subset=FULL,
efficient_alu=True, div_logsink=True, recurrent_divmod=False)`` — the UNROLLED
(no subroutine, no reused body) fp64 log-sink divmod + base + mem + cmp + bitwise
folded into ONE block stack.  We build it SPARSE-RESIDENT (never densify -> RSS-safe;
the dense fp64 Qwen2Model at these dims is ~30 GB and would OOM) and run the SAME
byte-identical windowed-stream lean forward the in-tree ``qwen_lean_forward`` uses.

MEMORY SAFETY: watchdog aborts > 4 GB.  We NEVER instantiate the dense Qwen2Model —
we bake each block's FFN into a sparse CSR (gate/up/down) directly, and the ~4
attention layers into small dense q/k/v/o proj tensors.

Golden 174ece66 UNTOUCHED (this module is OFF the build path).

Run:  PYTHONPATH=<c4_release> OMP_NUM_THREADS=2 C4_LOGSINK_DIV=1 \
      python -m c4_min._e2e_inline_bake_916
"""
from __future__ import annotations

import os
import resource
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

MASK32 = 0xFFFFFFFF


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def _rss_watchdog(limit_mb: int = 4000):
    def watch():
        while True:
            if _rss_mb() > limit_mb:
                print(f"RSS ABORT {_rss_mb()} MB > {limit_mb}", flush=True)
                os._exit(3)
            time.sleep(0.25)
    threading.Thread(target=watch, daemon=True).start()


# ---------------------------------------------------------------------------
# A stub HF-attn-like holder: gives the in-tree bake_* functions the .q_proj.weight
# etc. attributes they write into, at the model's (hidden) geometry.  Dense but tiny
# (only the ~4 attention layers get one; hidden ~2600 -> ~47 MB fp64 each, x4).
# ---------------------------------------------------------------------------
class _Lin:
    def __init__(self, out_f: int, in_f: int, dtype, bias: bool = False):
        self.weight = torch.zeros(out_f, in_f, dtype=dtype)
        self.bias = torch.zeros(out_f, dtype=dtype) if bias else None


class _StubAttn:
    def __init__(self, hidden: int, arch, dtype):
        nh, nkv, hd = (arch.num_attention_heads, arch.num_key_value_heads,
                       arch.head_dim)
        self.q_proj = _Lin(nh * hd, hidden, dtype, bias=True)
        self.k_proj = _Lin(nkv * hd, hidden, dtype, bias=True)
        self.v_proj = _Lin(nkv * hd, hidden, dtype, bias=True)
        self.o_proj = _Lin(hidden, nh * hd, dtype, bias=False)


@dataclass
class _SparseLayer:
    ln1: torch.Tensor
    ln2: torch.Tensor
    q_w: Optional[torch.Tensor] = None
    k_w: Optional[torch.Tensor] = None
    v_w: Optional[torch.Tensor] = None
    o_w: Optional[torch.Tensor] = None
    q_b: Optional[torch.Tensor] = None
    k_b: Optional[torch.Tensor] = None
    v_b: Optional[torch.Tensor] = None
    gate_csr: torch.Tensor = None
    up_csr: torch.Tensor = None
    down_csr: torch.Tensor = None
    has_attn: bool = False


def _to_csr(w: torch.Tensor) -> torch.Tensor:
    return w.to_sparse_csr()


def _spmm(csr: torch.Tensor, x2d: torch.Tensor) -> torch.Tensor:
    """csr [out,in] @ x2d [N,in]^T -> [N,out].  Same math as F.linear(x2d, w)."""
    return torch.sparse.mm(csr, x2d.transpose(0, 1)).transpose(0, 1).contiguous()


class SparseLeanVM:
    """A sparse-resident lean forward of the assembled fp64 full-ISA block stack.
    Same RMSNorm+RoPE+softmax+SwiGLU math as ``qwen_lean_forward.LeanQwenVM`` (whose
    forward this mirrors), but MLP weights live sparse (never densified)."""

    def __init__(self, layers, final_norm, embed, hidden, arch, dtype, QL, subset,
                 code_from_memory, efficient_alu, shift_via_mul, rope_theta, rms_eps,
                 div_logsink=False):
        self.layers = layers
        self.final_norm = final_norm
        self.embed = embed
        self.hidden_size = hidden
        self.n_layers = len(layers)
        self.n_heads = arch.num_attention_heads
        self.n_kv_heads = arch.num_key_value_heads
        self.head_dim = arch.head_dim
        self.rope_theta = rope_theta
        self.rms_eps = rms_eps
        self.device = torch.device("cpu")
        self.dtype = dtype
        self.QL = QL
        self.subset = subset
        self.code_from_memory = code_from_memory
        self.efficient_alu = efficient_alu
        self.shift_via_mul = shift_via_mul
        self.div_logsink = div_logsink
        half = self.head_dim // 2
        self.inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32)[:half] / self.head_dim))

    # -- lean math (mirrors qwen_lean_forward.LeanQwenVM) --
    def _rmsnorm(self, x, gamma):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.rms_eps) * gamma

    def _rope_cos_sin(self, positions):
        freqs = positions.to(torch.float32).unsqueeze(-1) * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, t, cos, sin):
        cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
        return (t * cos) + (self._rotate_half(t) * sin)

    def _attn(self, layer, xn, q_pos):
        B, S, H = xn.shape
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim
        scale = hd ** -0.5
        q = F.linear(xn, layer.q_w, layer.q_b).view(B, S, nh, hd).transpose(1, 2)
        k = F.linear(xn, layer.k_w, layer.k_b).view(B, S, nkv, hd).transpose(1, 2)
        v = F.linear(xn, layer.v_w, layer.v_b).view(B, S, nkv, hd).transpose(1, 2)
        cos_q, sin_q = self._rope_cos_sin(q_pos)
        q = self._apply_rope(q, cos_q, sin_q)
        k = self._apply_rope(k, cos_q, sin_q)
        K, Vv, k_pos = k, v, q_pos
        n_rep = nh // nkv
        if n_rep != 1:
            Sk = K.shape[2]
            Kr = K[:, :, None, :, :].expand(B, nkv, n_rep, Sk, hd).reshape(B, nh, Sk, hd)
            Vr = Vv[:, :, None, :, :].expand(B, nkv, n_rep, Sk, hd).reshape(B, nh, Sk, hd)
        else:
            Kr, Vr = K, Vv
        scores = torch.matmul(q, Kr.transpose(-2, -1)) * scale
        mask = (k_pos.unsqueeze(1) > q_pos.unsqueeze(2))
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
        out = torch.matmul(attn, Vr).transpose(1, 2).contiguous().view(B, S, nh * hd)
        out = F.linear(out, layer.o_w)
        return out

    def forward(self, x, q_positions=None):
        B, S, H = x.shape
        if q_positions is None:
            q_pos = torch.arange(S).unsqueeze(0).expand(B, S)
        else:
            q_pos = q_positions.to(dtype=torch.long)
            if q_pos.dim() == 1:
                q_pos = q_pos.unsqueeze(0).expand(B, S)
        h = x
        for layer in self.layers:
            xn = self._rmsnorm(h, layer.ln1)
            if layer.has_attn:
                h = h + self._attn(layer, xn, q_pos)
            xn2 = self._rmsnorm(h, layer.ln2)
            xn2f = xn2.reshape(B * S, H)
            g = _spmm(layer.gate_csr, xn2f)
            u = _spmm(layer.up_csr, xn2f)
            act = F.silu(g) * u
            d = _spmm(layer.down_csr, act).reshape(B, S, H)
            h = h + d
        h = self._rmsnorm(h, self.final_norm)
        return h


# ---------------------------------------------------------------------------
# BUILD the sparse-resident assembled model from qwen_full_vm's block specs.
# ---------------------------------------------------------------------------
def build_sparse(subset_name: str = "full") -> Tuple[SparseLeanVM, dict]:
    os.environ["C4_LOGSINK_DIV"] = "1"
    from . import qwen_full_vm as Q

    subset = {"full": Q.SUBSET_FULL, "muldiv": Q.SUBSET_MULDIV,
              "mem_cmp": Q.SUBSET_MEM_CMP}[subset_name]
    arch = Q.QWEN2_5_ARCH
    K = Q.NORM_K
    code_size = 24
    efficient_alu = True
    recurrent_divmod = False
    code_from_memory = True
    shift_via_mul = True

    QL = Q.QwenFullLayout(code_size, subset, efficient_alu=efficient_alu,
                          recurrent_divmod=recurrent_divmod,
                          code_from_memory=code_from_memory,
                          shift_via_mul=shift_via_mul, div_logsink=True)
    L = QL.L
    block_specs = Q._block_specs(L, code_size, subset, efficient_alu=efficient_alu,
                                 recurrent_divmod=recurrent_divmod,
                                 code_from_memory=code_from_memory,
                                 shift_via_mul=QL.shift_via_mul,
                                 div_logsink=QL.div_logsink)
    block_names = [nm for nm, _ in block_specs]

    dim_needed = QL.D_used + 1
    hidden = arch.hidden_for(dim_needed)
    inter = max(int(s["W_up"].shape[0]) for _, s in block_specs)
    inter = max(inter, arch.num_attention_heads * arch.head_dim, 8)
    n_layers = len(block_specs)
    comp = QL.D_used
    dtype = torch.float64 if QL.div_logsink else torch.float32

    gamma = Q.rmsnorm_identity_gamma(hidden, K).to(dtype)
    embed = Q._build_embedding(L, hidden, comp, K).to(dtype)

    attn_bakers = {0: "register"}
    if code_from_memory:
        attn_bakers[block_names.index("code-cam")] = "code"
    if subset.memory:
        attn_bakers[block_names.index("mem-cam")] = "memory"
    if QL.div_logsink:
        attn_bakers[block_names.index("ls-recip-attn")] = "recip"

    layers: List[_SparseLayer] = []
    for i, (name, spec) in enumerate(block_specs):
        h_units = spec["W_up"].shape[0]
        Dg = spec["W_up"].shape[1]
        bd = spec.get("b_down")
        if bd is not None and float(bd.abs().max()) != 0.0:
            raise AssertionError(f"nonzero b_down in {name}")
        gate_w = torch.zeros(inter, hidden, dtype=dtype)
        up_w = torch.zeros(inter, hidden, dtype=dtype)
        down_w = torch.zeros(hidden, inter, dtype=dtype)
        gate_w[:h_units, :Dg] = spec["W_up"].to(dtype)
        up_w[:h_units, :Dg] = spec["W_gate"].to(dtype)
        down_w[:Dg, :h_units] = spec["W_down"].to(dtype)
        gate_w[:h_units, L.ONE] += spec["b_up"].to(dtype)
        up_w[:h_units, L.ONE] += spec["b_gate"].to(dtype)
        lay = _SparseLayer(ln1=gamma.clone(), ln2=gamma.clone(),
                           gate_csr=_to_csr(gate_w), up_csr=_to_csr(up_w),
                           down_csr=_to_csr(down_w))
        del gate_w, up_w, down_w

        if i in attn_bakers:
            kind = attn_bakers[i]
            stub = _StubAttn(hidden, arch, dtype)
            if kind == "register":
                Q._bake_register_cam(stub, QL, arch, comp, K)
            elif kind == "code":
                Q._bake_code_cam(stub, QL, arch, comp, K)
            elif kind == "memory":
                Q._bake_memory_cam(stub, QL, arch, comp, K, mem_addr_bits=None)
            elif kind == "recip":
                from . import nibble_logsink_blocks as LS
                LS.bake_recip_sink_cam(stub, L, arch, head_idx=1)
            lay.has_attn = True
            lay.q_w = stub.q_proj.weight; lay.k_w = stub.k_proj.weight
            lay.v_w = stub.v_proj.weight; lay.o_w = stub.o_proj.weight
            lay.q_b = stub.q_proj.bias; lay.k_b = stub.k_proj.bias
            lay.v_b = stub.v_proj.bias
        layers.append(lay)

    vm = SparseLeanVM(layers, gamma.clone(), embed, hidden, arch, dtype, QL, subset,
                      code_from_memory, efficient_alu, QL.shift_via_mul,
                      Q.ROPE_THETA, 1e-6, div_logsink=bool(QL.div_logsink))
    info = {"n_layers": n_layers, "hidden": hidden, "intermediate": inter,
            "D_used": QL.D_used, "block_names": block_names,
            "n_divmod_blocks": sum(1 for n in block_names if n.startswith("ls")),
            "dtype": str(dtype), "attn_layers": sorted(attn_bakers.keys()),
            "peak_rss_mb": _rss_mb()}
    return vm, info


# ---------------------------------------------------------------------------
# RUN a program end-to-end through the sparse assembled model, byte-exact vs isa.
# Mirrors qwen_lean_forward.run_program_lean's driver (windowed stream, KV memory,
# call stack) but calls our sparse forward.
# ---------------------------------------------------------------------------
def run_program(vm: SparseLeanVM, code, max_steps: int = 200, mask: int = 0xFF,
                verbose: bool = False) -> dict:
    from . import isa
    from . import qwen_full_vm as Q
    from .nibble_pure_forward_complete import _decode_reg_from_nibbles
    from .qwen_full_vm import SP_INIT, _snap
    QL, L = vm.QL, vm.QL.L
    subset = vm.subset

    ref_trace = isa.interpret(code, max_steps=max_steps)

    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    store_log: List[dict] = []
    ax_trace: List[int] = []
    cur_pc = 0
    call_stack: List[Tuple[Optional[int], Optional[int]]] = []
    stack_kv: List[Tuple[int, int]] = []
    _POP_OPS = {isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD, isa.AND, isa.OR,
                isa.XOR, isa.SHL, isa.SHR, isa.EQ, isa.NE, isa.LT, isa.GT,
                isa.LE, isa.GE}
    run_code = list(code)
    _nib_ax = {isa.MUL, isa.DIV, isa.MOD}
    if vm.shift_via_mul:
        _nib_ax |= {isa.SHL, isa.SHR}

    for _ in range(max_steps):
        op = run_code[cur_pc].op if 0 <= cur_pc < len(run_code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF
        # qwen_full_vm's stream builder seeds the log-sink reciprocal KV rows (which
        # the lean builder does NOT) — mandatory for DIV/MOD through the inline divmod.
        x = Q._build_stream_and_overlay(vm, run_code, reg_state, store_log, load_addr)
        state_h = vm.forward(x, q_positions=None)     # positions default to arange(S)
        state = state_h[0, -1]

        pc = _snap(state[L.PC_VAL])
        if vm.efficient_alu and op in _nib_ax:
            ax = _decode_reg_from_nibbles(state, L, L.AX) & mask
        else:
            ax = _snap(state[L.AX_VAL]) & 0xFF
        sp = _snap(state[L.SP_VAL]); bp = _snap(state[L.BP_VAL])
        halted = float(state[L.HALTED]) > 0.5

        if op == isa.PSH:
            stack_kv.append((sp & 0xFF, prev["AX"] & 0xFF))
        elif op in _POP_OPS and stack_kv:
            stack_kv.pop()

        if op == isa.JSR:
            call_stack.append((cur_pc + 1, prev["BP"]))
        elif op == isa.LEV and call_stack:
            ret_pc, saved_bp = call_stack.pop()
            pc = ret_pc if ret_pc is not None else pc
            bp = saved_bp if saved_bp is not None else bp

        new_stk = stack_kv[-1][1] if stack_kv else 0
        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": new_stk}
        ax_trace.append(ax)
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op)} -> "
                  f"AX={ax} PC={pc} SP={sp} BP={bp}")
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(run_code):
            break

    n = min(len(ax_trace), len(ref_trace))
    exact = ax_trace[:n] == ref_trace[:n] and n > 0
    return {"ax_trace": ax_trace, "ref_trace": ref_trace, "exact": exact,
            "steps": len(ax_trace), "n_cmp": n, "peak_rss_mb": _rss_mb()}
