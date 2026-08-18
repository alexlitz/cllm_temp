"""ASSEMBLED WHOLE-ISA TOKEN PIPELINE (#923, closing #922/#916).

Assembles the base-VM token pipeline (``_token_pipeline_base``) + the divmod token
pipeline (``_token_pipeline_divmod``) into ONE token-pipelined whole-ISA model at a
literal stock-Qwen2.5-0.5B geometry (hidden <= 896, <= 24 layers PER TOKEN) and runs
REAL PROGRAMS byte-exact end-to-end vs ``isa.interpret`` — including programs that USE
full-32-bit DIV/MOD.

Per VM step the forward is TOKEN-PIPELINED:
  * The base VM step (fetch/decode/PC/AX/SP/BP/STACK0/OP_IS/mem-CAM/cmp/bitwise/
    add-sub/dispatch/branch/fold) runs through the whole-ISA sparse forward's baked
    attention CAMs (register/code/mem reconstruction) to reconstruct state, then its
    18 base FFN blocks are SLICED across emission tokens (<= cap FFN layers/token, +1
    gather-attn), carrying transient base scratch across token boundaries via the
    byte-exact softmax1+ALiBi CAM.  Measured per-token base residual: 327 (<=896).
  * A DIV/MOD step's 32-bit-exact quotient/remainder is computed by the #922 divmod
    token pipeline (6 tokens/DIV, per-token d_model 296 <=896) — the radix-16-lean
    single-forward divmod is only ~2^16-exact, so the full-32-bit path is the sliced
    log-sink divmod.  The result nibbles are written into AX.

The assembled per-token geometry is the MAX over base (327 d_model, 512 FFN-hidden,
<=24 layers) and divmod (296 d_model, 2422 FFN-hidden, 24 layers) — a single residual
of <= 896 hidden and <= 24 layers/token.  MEASURED (constructed), not projected.

MEASURED, off-build-path.  Golden 174ece66 UNTOUCHED (NEW file).  RSS-safe (the base
sparse forward + tiny pipelines run in ~1-2 GB); watchdog aborts > 4 GB.

Run:  PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._token_pipeline_whole_isa
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

from . import _e2e_inline_bake_916 as E
from . import _token_pipeline_base as TPB
from . import _token_pipeline_divmod as TPD


# ===========================================================================
# The base whole-ISA sparse forward, but with the FFN blocks split so a VM step
# can run the FFN portion TOKEN-SLICED.  We reuse the byte-exact baked attention
# CAMs (register / code-fetch / mem) from the sparse build; only the FFN datapath
# is re-expressed as the token pipeline.
# ===========================================================================
class WholeISATokenPipeline:
    def __init__(self, cap: int = 23, dtype=torch.float32):
        self.cap = cap
        # Build the whole-ISA sparse forward (radix-16 lean base + attention CAMs).
        # We use its baked attention CAMs + the driver; the FFN datapath is replaced
        # by the compact token pipeline.
        self.vm, self.info = E_build_lean(dtype)
        # Base token pipeline (compact remap of the base FFN blocks).
        base_specs, L, QL = TPB.build_base_blocks()
        self.base_specs = base_specs
        self.L = L
        self.QL = QL
        self.cmap, self.compact_base = TPB.build_compact_base(base_specs, L)
        # Divmod token pipeline (full-32-bit exact).
        self.Ldiv = TPD.TPLayout(896)
        self.div_blocks = TPD.build_parallel_divmod_blocks(self.Ldiv, refine=True)
        # base names in whole-ISA order (for the sparse forward's block loop)
        self.base_names = set(n for n, _ in base_specs)
        # Split the base blocks at the LAST attention block: the reconstruction span
        # (blocks up to + including the last attention CAM) runs the attention on the
        # full sequence; the pure-FFN compute span after it is SLICED across tokens on
        # the compact residual (the scratch-slicing lever).  Map whole-ISA block index
        # -> compact-base spec index for the post-attention FFN span.
        names = self.info["block_names"]
        attn_idx = set(self.info["attn_layers"])
        self.last_attn = max(attn_idx)
        # compact spec index by block name (base order == whole-ISA base order)
        self._compact_by_name = {n: (n, s) for n, s in self.compact_base}
        # the post-attention base FFN block names, in order
        self.post_attn_names = [names[i] for i in range(self.last_attn + 1, len(names))
                                if not TPB._is_divmod_block(names[i])]

    # per-token geometry (CONSTRUCTED)
    def geometry(self):
        base_hidden = max(int(s["W_up"].shape[0]) for _, s in self.compact_base)
        div_hidden = max(int(s["W_up"].shape[0]) for _, s in self.div_blocks)
        return {
            "base_per_token_d_model": self.cmap.D,
            "base_per_token_ffn_hidden": base_hidden,
            "base_blocks": len(self.compact_base),
            "divmod_per_token_d_model": self.Ldiv.D,
            "divmod_per_token_ffn_hidden": div_hidden,
            "whole_per_token_d_model": max(self.cmap.D, self.Ldiv.D),
            "whole_per_token_ffn_hidden": max(base_hidden, div_hidden),
        }


def E_build_lean(dtype):
    """Build the radix-16-lean whole-ISA sparse forward (attention CAMs baked)."""
    from . import _e2e_floor_r16lean_916 as F16
    return F16.build_sparse_r16lean(dtype=dtype)


# ===========================================================================
# The whole-ISA program run.  Mirrors _e2e_inline_bake_916.run_program's driver
# (windowed stream, KV memory, call stack) but:
#   * runs the base FFN step TOKEN-SLICED via the base token pipeline, and
#   * computes DIV/MOD via the divmod token pipeline (full-32-bit exact),
# so the whole ISA runs across the step's emission tokens at <= 896 / <= 24 layers.
# ===========================================================================
def run_program_pipelined(wp: WholeISATokenPipeline, code, max_steps: int = 120,
                          mask: int = 0xFF, verbose: bool = False) -> dict:
    from . import isa
    from . import qwen_full_vm as Q
    from .nibble_pure_forward_complete import _decode_reg_from_nibbles
    from .qwen_full_vm import SP_INIT, _snap
    from . import nibble_muldivmod as NM
    vm = wp.vm
    QL, L = vm.QL, vm.QL.L
    subset = vm.subset
    cmap = wp.cmap
    idx = torch.tensor(cmap.dims)

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

    tok_counts = []          # base tokens/step
    div_tok = None           # tokens/DIV

    for _ in range(max_steps):
        op = run_code[cur_pc].op if 0 <= cur_pc < len(run_code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF

        # ---- attention + base FFN, TOKEN-SLICED ----
        x = Q._build_stream_and_overlay(vm, run_code, reg_state, store_log, load_addr)
        # run attention layers on the full residual (register/code/mem CAM), producing
        # the post-attention residual the FFN blocks consume; then run the base FFN
        # blocks TOKEN-SLICED on the compact residual.  n_base_tok = compute-span tokens
        # (the reconstruction span is 1 attention token) -> base tokens/step = 1 + this.
        state_full, n_base_tok = _forward_base_pipelined(wp, x)
        tok_counts.append(1 + n_base_tok)
        state = state_full  # last-row residual, full layout

        pc = _snap(state[L.PC_VAL])
        sp = _snap(state[L.SP_VAL]); bp = _snap(state[L.BP_VAL])
        halted = float(state[L.HALTED]) > 0.5

        if op in (isa.DIV, isa.MOD):
            # full-32-bit exact via the divmod token pipeline (STACK0=a, AX=b)
            a = prev["STACK0"] & 0xFFFFFFFF
            b = prev["AX"] & 0xFFFFFFFF
            q, r, nt = TPD.run_packed_divmod(a, b, wp.Ldiv, wp.div_blocks, cap=23)
            div_tok = nt
            ax = (q if op == isa.DIV else r) & mask
        elif vm.efficient_alu and op in _nib_ax:
            ax = _decode_reg_from_nibbles(state, L, L.AX) & mask
        else:
            ax = _snap(state[L.AX_VAL]) & 0xFF

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
                  f"AX={ax} PC={pc} SP={sp} BP={bp}  base_tok={n_base_tok}")
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(run_code):
            break

    n = min(len(ax_trace), len(ref_trace))
    exact = ax_trace[:n] == ref_trace[:n] and n > 0
    return {"ax_trace": ax_trace, "ref_trace": ref_trace, "exact": exact,
            "steps": len(ax_trace), "n_cmp": n,
            "base_tokens_per_step": tok_counts, "div_tokens": div_tok,
            "peak_rss_mb": E._rss_mb()}


def _forward_base_pipelined(wp: WholeISATokenPipeline, x):
    """Run ONE VM step's forward with the pure-FFN compute datapath TOKEN-SLICED.

    Two spans:
      * RECONSTRUCTION span (blocks 0..last_attn): the register/code/mem attention
        CAMs run on the FULL sequence to reconstruct PC/AX/SP/BP/STACK0 + fetched
        op/imm/mem onto the query row (byte-exact baked CAMs), interleaved with their
        FFN.  Divmod-bank blocks are skipped (a base op never reads the DIV RES bands).
      * COMPUTE span (post-attention base FFN blocks): NO more attention, so the query
        row is row-independent — we take the query-row residual, project it to the
        COMPACT base residual (<=896), and run the post-attention base FFN blocks
        TOKEN-SLICED via the compact pipeline (scratch carried across token boundaries
        by the byte-exact softmax1+ALiBi CAM), then scatter the result back.

    Returns ``(last_row_residual_full, n_compute_tokens)`` where n_compute_tokens is the
    number of <=cap-layer FFN tokens the compute span used.
    """
    vm = wp.vm
    L = wp.L
    cmap = wp.cmap
    idx = torch.tensor(cmap.dims)
    B, S, H = x.shape
    q_pos = torch.arange(S).unsqueeze(0).expand(B, S)
    h = x
    names = wp.info["block_names"]

    # ---- RECONSTRUCTION span: run blocks 0..last_attn (attention CAMs + FFN) ----
    for li in range(wp.last_attn + 1):
        layer = vm.layers[li]
        name = names[li]
        xn = vm._rmsnorm(h, layer.ln1)
        if layer.has_attn:
            h = h + vm._attn(layer, xn, q_pos)
        if TPB._is_divmod_block(name):
            continue
        xn2 = vm._rmsnorm(h, layer.ln2)
        xn2f = xn2.reshape(B * S, H)
        g = E._spmm(layer.gate_csr, xn2f)
        u = E._spmm(layer.up_csr, xn2f)
        act = F.silu(g) * u
        d = E._spmm(layer.down_csr, act).reshape(B, S, H)
        h = h + d

    # ---- COMPUTE span: post-attention base FFN, TOKEN-SLICED on the compact residual.
    # The query row (index -1) carries the reconstructed state; the remaining base FFN
    # blocks are row-wise, so we compute them on the compact projection of that row.
    qrow = h[0, -1].to(torch.float32)                 # full-layout query-row residual
    xc0 = qrow[idx].clone()                            # -> compact base residual (<=896)
    post_specs = [wp._compact_by_name[n] for n in wp.post_attn_names]
    xc_out, n_tok = TPB.run_base_token_pipeline(xc0, cmap, post_specs, cap=wp.cap)
    # scatter the compact result back into the full-layout query row (only the base
    # footprint dims changed; the rest of the row is untouched register/CAM bands).
    out = qrow.clone()
    out[idx] = xc_out
    # final RMSNorm (identity gamma) matches the sparse forward's tail norm.
    out = vm._rmsnorm(out.unsqueeze(0), vm.final_norm)[0]
    return out.to(h.dtype), n_tok


# ===========================================================================
# FULL-32-BIT DIV/MOD through the divmod token pipeline (the cases the radix-16-lean
# single-forward build DIVERGES on, but the sliced log-sink divmod is byte-exact).
# ===========================================================================
def run_full32_div(wp: WholeISATokenPipeline, op_name, a, b):
    from . import nibble_muldivmod as NM
    q, r, _ = TPD.run_packed_divmod(a & 0xFFFFFFFF, b & 0xFFFFFFFF,
                                    wp.Ldiv, wp.div_blocks, cap=23)
    got = q if op_name == "DIV" else r
    want = NM.div32(a, b) if op_name == "DIV" else NM.mod32(a, b)
    return got & 0xFFFFFFFF, want & 0xFFFFFFFF


def main():
    import random
    from . import isa
    from . import _e2e_battery_916 as Bat
    from . import _e2e_floor_r16lean_916 as F16
    E._rss_watchdog(4000)
    print("=" * 92)
    print("#923 LITERAL-0.5B WHOLE-ISA TOKEN PIPELINE: base pipeline + #922 divmod")
    print("pipeline assembled at hidden <=896 / <=24 layers PER TOKEN, run real")
    print("programs byte-exact end-to-end vs isa.interpret (incl full-32-bit DIV).")
    print("=" * 92)

    wp = WholeISATokenPipeline(cap=23)
    g = wp.geometry()
    print("\nPER-TOKEN GEOMETRY (CONSTRUCTED, not projected):")
    print(f"  base   : d_model {g['base_per_token_d_model']:>4}  FFN-hidden "
          f"{g['base_per_token_ffn_hidden']:>4}  ({g['base_blocks']} FFN blocks, "
          f"reconstruction+compute)")
    print(f"  divmod : d_model {g['divmod_per_token_d_model']:>4}  FFN-hidden "
          f"{g['divmod_per_token_ffn_hidden']:>4}")
    print(f"  WHOLE  : d_model {g['whole_per_token_d_model']:>4} (<=896: "
          f"{g['whole_per_token_d_model'] <= 896})  FFN-hidden "
          f"{g['whole_per_token_ffn_hidden']:>4} (<=4864: "
          f"{g['whole_per_token_ffn_hidden'] <= 4864})")
    print(f"  layers/token: base compute-span <= {wp.cap}+1 attn = <=24;  "
          f"divmod <=24 (knife-edge)")

    prog_ok = prog_tot = 0
    step_ok = step_tot = 0
    dm_ok = dm_tot = 0
    base_tok_max = 0
    for title, progs, is_dm in (
            ("DIVMOD boundary (b=1, b=2^k, b~256, large q, b=0 guard)",
             Bat.divmod_programs(), True),
            ("Multi-op branching / memory / cmp / mul / div-in-loop programs",
             Bat.multiop_programs(), False)):
        print("\n" + "-" * 92)
        print(title)
        print("-" * 92)
        for name, prog in progs.items():
            code = isa.assemble(prog)
            r = run_program_pipelined(wp, code, max_steps=120)
            ok = r["exact"]
            prog_tot += 1
            prog_ok += int(ok)
            step_tot += r["n_cmp"]
            step_ok += (r["n_cmp"] if ok else
                        sum(1 for i in range(r["n_cmp"])
                            if r["ax_trace"][i] == r["ref_trace"][i]))
            if is_dm:
                dm_tot += 1
                dm_ok += int(ok)
            if r["base_tokens_per_step"]:
                base_tok_max = max(base_tok_max, max(r["base_tokens_per_step"]))
            tag = "PASS" if ok else "FAIL"
            extra = "" if ok else f"  ax={r['ax_trace']} ref={r['ref_trace']}"
            print(f"  {name:<22} steps={r['n_cmp']:>3} base_tok/step="
                  f"{r['base_tokens_per_step'][0] if r['base_tokens_per_step'] else '?'} "
                  f"[{tag}]{extra}")

    # FULL-32-BIT DIV/MOD (the sliced log-sink divmod's full-width exactness)
    print("\n" + "-" * 92)
    print("FULL-32-BIT DIV/MOD through the divmod token pipeline (the radix-16-lean")
    print("single-forward build's documented divergences) vs nibble_muldivmod32")
    print("-" * 92)
    f_ok = f_tot = 0
    div_tokens = None
    named = [
        ("DIV", 0xFFFFFFFF, 3, "max/3 (q~1.4e9)"),
        ("MOD", 0xFFFFFFFF, 3, "max %% 3"),
        ("DIV", 0xDEADBEEF, 0x1234, "0xDEADBEEF/0x1234"),
        ("MOD", 0xDEADBEEF, 0x1234, "0xDEADBEEF %% 0x1234"),
        ("DIV", 0xFFFFFFFF, 0xFFFF, "max/65535 (q=65537)"),
        ("DIV", 700003, 7, "700003/7 (q>2^16)"),
        ("DIV", 0xFFFFFFFF, 1, "max/1 (b=1)"),
        ("DIV", 0xFFFFFFFF, 2, "max/2 (b=2^1)"),
        ("DIV", 0xFFFFFFFF, 256, "max/256"),
        ("DIV", 0xCAFEBABE, 0x100, "0xCAFEBABE/256"),
        ("DIV", 0xFFFFFFFF, 0xFFFFFFFF, "max/max (q=1)"),
        ("DIV", 2**31, 3, "2^31/3"),
        ("MOD", 2**31, 7, "2^31 %% 7"),
        ("DIV", 4000000000, 3, "4e9/3"),
        ("DIV", 0xFFFFFFFF, 0x10000, "max/65536 (b=2^16)"),
        ("MOD", 0xFFFFFFFF, 0x10000, "max %% 65536"),
    ]
    for op_name, a, b, desc in named:
        got, want = run_full32_div(wp, op_name, a, b)
        _, _, nt = TPD.run_packed_divmod(a & 0xFFFFFFFF, b & 0xFFFFFFFF, wp.Ldiv,
                                         wp.div_blocks, cap=23)
        div_tokens = nt
        f_tot += 1
        good = got == want
        f_ok += int(good)
        tag = "PASS" if good else "FAIL"
        extra = "" if good else f"  got={got} want={want}"
        print(f"  {desc:<34} [{tag}]{extra}")
    print(f"  named full-32-bit DIV/MOD byte-exact: {f_ok}/{f_tot}  (tokens/DIV={div_tokens})")

    rng = random.Random(20260815)
    r_ok = r_tot = 0
    for _ in range(200):
        a = rng.randint(0, 2**32 - 1)
        b = rng.randint(1, 2**32 - 1)
        dg, dw = run_full32_div(wp, "DIV", a, b)
        mg, mw = run_full32_div(wp, "MOD", a, b)
        r_tot += 2
        r_ok += int(dg == dw) + int(mg == mw)
    print(f"  RANDOM full-32-bit DIV/MOD (200 pairs a,b in [0,2^32)): {r_ok}/{r_tot}")

    print("\n" + "=" * 92)
    print("VERDICT — literal-0.5B whole-ISA token pipeline")
    print("=" * 92)
    print(f"  programs byte-exact (8-bit incl div-in-loop) : {prog_ok}/{prog_tot}")
    print(f"  steps byte-exact                             : {step_ok}/{step_tot}")
    print(f"  8-bit DIVMOD boundary cases                  : {dm_ok}/{dm_tot}")
    print(f"  FULL-32-BIT named DIV/MOD                     : {f_ok}/{f_tot}")
    print(f"  FULL-32-BIT random DIV/MOD                    : {r_ok}/{r_tot}")
    print(f"  per-token d_model (whole)  : {g['whole_per_token_d_model']} (<=896)")
    print(f"  per-token FFN-hidden(whole): {g['whole_per_token_ffn_hidden']} (<=4864)")
    print(f"  base tokens/step (max)     : {base_tok_max}   divmod tokens/DIV: {div_tokens}")
    print(f"  peak RSS                   : {E._rss_mb()} MB")
    allok = (prog_ok == prog_tot and dm_ok == dm_tot and f_ok == f_tot
             and r_ok == r_tot and g['whole_per_token_d_model'] <= 896
             and g['whole_per_token_ffn_hidden'] <= 4864)
    print(f"\n  LITERAL-0.5B WHOLE-ISA BYTE-EXACT END-TO-END: {allok}")
    return allok


if __name__ == "__main__":
    ok = main()
    print(f"\nALL BYTE-EXACT: {ok}")
