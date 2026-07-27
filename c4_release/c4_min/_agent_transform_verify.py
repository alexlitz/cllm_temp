"""EMPIRICAL model_subsets-transform verification harness for the c4_min VM (GPU).

Additive experiment file (golden 069cc32f unchanged: imports only, builds nothing on
the golden path). Builds the runnable standalone-Transformer VM
(``nibble_pure_forward_complete``) ONCE, moves it to a GPU, establishes a byte-exact
BASELINE battery vs the model's own value-faithful reference (``ref_interpret``), then
applies each model_subsets emulation transform to the built weights / forward and
re-runs the battery, reporting byte-exact PASS/FAIL + divergence magnitude.

Reference = this VM family's ``ref_interpret`` (word-addressed SP, LEA=bp+4*imm,
signed cmp) — the value-faithful golden for THIS model (NOT ``isa.interpret``, the
byte-addressed fused-Qwen slice; different memory model).

Transforms (mirror the model_subsets math):
  2. GQA      : average W_k/W_v per head-group      (predict degrade; CAM cap)
  3. MQA      : tie K/V to 1 head                    (predict CAM break)
  4. GeLU     : swap SiLU->Swish_b(b=1.7729) gate   (predict break)
  5. RoPE     : positional="rope" recency           (predict CAM addressing degrade)
  6. BOS-sink : sink="bos_sink"                      (predict byte-exact; +1==exp(0))
(RMSNorm needs a norm-block rebuild + compensator — handled/measured separately.)

Run:  C4_XFORM_DEVICE=cuda:1 python -m c4_release.c4_min._agent_transform_verify
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from . import nibble_pure_forward_complete as N
from . import isa
from . import blogspec_model as BM
from . import blogspec_vocab as V
from .nibble_vm import _snap_lane


BETA = 1.772934   # convert/gelu_to_silu best-fit Swish_beta to GeLU (max-abs floor 0.0139)
DEVICE = os.environ.get("C4_XFORM_DEVICE", "cuda:1")


# ---------------------------------------------------------------------------
# GPU-aware runner: mirrors run_pure_forward_complete's non-file/non-PRTF loop,
# but places the token tensor on the model's device.  The overlay + decode are
# device-agnostic (they read float(state[...]) / write into x in place).
# ---------------------------------------------------------------------------
def run_on_device(model, L, code, max_steps, dev, mask=0xFF):
    init_frame = N._build_frame(0, 0, N.SP_INIT, N.SP_INIT, 0)
    store_log = {}
    stream = [V.BOS] + init_frame
    trace = []
    cur_pc = cur_sp = cur_bp = cur_ax = 0
    cur_sp = cur_bp = N.SP_INIT
    frame_idx = 0
    for _ in range(max_steps):
        overlay = N.make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream], device=dev)
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            for blk in model.blocks:
                x = blk(x)
        state = x[0, -1]
        pc = _snap_lane(state[L.PC_VAL])
        sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL])
        stk = _snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        ax = N._decode_reg_from_nibbles(state, L, L.AX)
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = N._mem_top(store_log, cur_sp); s_val = ax & mask
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = N._build_frame(pc, ax, sp, bp, stk,
                               mem_addr=(s_addr if is_store else 0),
                               mem_val=(s_val if is_store else 0))
        trace.append(ax & mask)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    return trace


# ---------------------------------------------------------------------------
# Battery: small programs covering ALU / mem / branch / cmp / expr.
# div/mod traverse the deep recurrent-divmod span (expensive) so only ONE each.
# ---------------------------------------------------------------------------
def battery():
    A = isa.assemble
    return [
        ("add",      A([('IMM', 5), ('PSH', 0), ('IMM', 3), ('ADD', 0), ('HALT', 0)]), 8),
        ("sub",      A([('IMM', 9), ('PSH', 0), ('IMM', 4), ('SUB', 0), ('HALT', 0)]), 8),
        ("mul",      A([('IMM', 6), ('PSH', 0), ('IMM', 7), ('MUL', 0), ('HALT', 0)]), 8),
        ("div",      A([('IMM', 40), ('PSH', 0), ('IMM', 6), ('DIV', 0), ('HALT', 0)]), 8),
        ("mod",      A([('IMM', 40), ('PSH', 0), ('IMM', 6), ('MOD', 0), ('HALT', 0)]), 8),
        ("expr",     A([('IMM', 5), ('PSH', 0), ('IMM', 3), ('ADD', 0),
                        ('PSH', 0), ('IMM', 2), ('MUL', 0), ('HALT', 0)]), 12),
        ("and",      A([('IMM', 0xF6), ('PSH', 0), ('IMM', 0x0F), ('AND', 0), ('HALT', 0)]), 8),
        ("xor",      A([('IMM', 0xF0), ('PSH', 0), ('IMM', 0x0F), ('XOR', 0), ('HALT', 0)]), 8),
        ("mem_si_li", A([('IMM', 200), ('PSH', 0), ('IMM', 42), ('SI', 0),
                         ('IMM', 200), ('LI', 0), ('HALT', 0)]), 12),
        ("if_lt",    A([('IMM', 7), ('PSH', 0), ('IMM', 9), ('LT', 0), ('BZ', 8),
                        ('IMM', 111), ('HALT', 0), ('IMM', 222), ('HALT', 0)]), 12),
        ("if_lt_f",  A([('IMM', 9), ('PSH', 0), ('IMM', 7), ('LT', 0), ('BZ', 8),
                        ('IMM', 111), ('HALT', 0), ('IMM', 222), ('HALT', 0)]), 12),
        ("eq",       A([('IMM', 5), ('PSH', 0), ('IMM', 5), ('EQ', 0), ('HALT', 0)]), 8),
    ]


def run_battery(model, L, tag, dev, verbose=False):
    n_pass = 0
    fails = []
    total = 0
    for name, code, ms in battery():
        ref = N.ref_interpret(code, max_steps=ms)
        try:
            got = run_on_device(model, L, code, ms, dev)
        except Exception as e:  # pragma: no cover
            fails.append((name, ref, f"EXC:{e}", 256)); total += 1
            if verbose:
                print(f"    {tag:10s} {name:10s} EXC {e}", flush=True)
            continue
        total += 1
        m = max(len(ref), len(got))
        rp = ref + [None] * (m - len(ref)); gp = got + [None] * (m - len(got))
        maxdiv = 0
        for a, b in zip(rp, gp):
            maxdiv = max(maxdiv, 256 if (a is None or b is None) else abs(a - b))
        ok = (ref == got)
        if ok:
            n_pass += 1
        else:
            fails.append((name, ref, got, maxdiv))
        if verbose:
            print(f"    {tag:10s} {name:10s} {'PASS' if ok else 'FAIL':4s} "
                  f"maxdiv={maxdiv:<4d} ref={ref} got={got}", flush=True)
    return n_pass, total, fails


# --- T4: SiLU -> GeLU/Swish_b on the SwiGLU gate --------------------------
def _gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _swish_b(x, b):
    return x * torch.sigmoid(b * x)


def patch_ffn_activation(act_fn):
    orig = BM.FFN.forward

    def fwd(self, x):
        up = F.linear(x, self.W_up) + self.b_up
        gate = F.linear(x, self.W_gate) + self.b_gate
        hidden = act_fn(up) * gate
        return x + F.linear(hidden, self.W_down, self.b_down)

    BM.FFN.forward = fwd
    return lambda: setattr(BM.FFN, "forward", orig)


# --- T2/T3: MHA -> GQA(n_kv) / MQA (group-average W_k/W_v per head) --------
def apply_kv_grouping(model, n_kv):
    saved = []
    H = model.blocks[0].attn.n_heads
    hd = model.blocks[0].attn.head_dim
    assert H % n_kv == 0
    hpg = H // n_kv
    for blk in model.blocks:
        at = blk.attn
        saved.append((at, at.W_k.data.clone(), at.W_v.data.clone()))
        for W in (at.W_k, at.W_v):
            Wv = W.data.view(H, hd, at.dim)
            for g in range(n_kv):
                sl = slice(g * hpg, (g + 1) * hpg)
                Wv[sl] = Wv[sl].mean(dim=0, keepdim=True)

    def restore():
        for at, wk, wv in saved:
            at.W_k.data.copy_(wk); at.W_v.data.copy_(wv)
    return restore


# --- T5/T6: ALiBi<->RoPE and softmax1<->bos_sink (per-Attn config attrs) ---
def set_attn_config(model, positional=None, sink=None):
    saved = []
    for blk in model.blocks:
        at = blk.attn
        saved.append((at, at.positional, at.sink))
        if positional is not None:
            at.positional = positional
        if sink is not None:
            at.sink = sink

    def restore():
        for at, p, s in saved:
            at.positional = p; at.sink = s
    return restore


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print(f"c4_min VM x model_subsets transform verification  (device={DEVICE})")
    print("=" * 78)
    t0 = time.time()
    print("building VM (nibble_pure_forward_complete, recurrent_divmod=True)...", flush=True)
    model, L = N.build_pure_forward_complete_model(code_size=24, recurrent_divmod=True)
    model.eval()
    dev = torch.device(DEVICE)
    model.to(dev)
    print(f"  built+moved in {time.time()-t0:.1f}s | n_blocks={len(model.blocks)} "
          f"dim={model.dim} n_heads={model.blocks[0].attn.n_heads} "
          f"head_dim={model.blocks[0].attn.head_dim} "
          f"norm={model.norm} pos={model.blocks[0].attn.positional} "
          f"sink={model.blocks[0].attn.sink}", flush=True)

    results = {}

    def stage(key, label, verbose=True):
        p, t, f = run_battery(model, L, key, dev, verbose=verbose)
        results[key] = (p, t, f)
        print(f"  => [{label}] {p}/{t} byte-exact ({time.time()-t0:.0f}s)", flush=True)

    print("\n[BASELINE] native VM (silu / alibi / softmax1 / norm-free)", flush=True)
    stage("baseline", "baseline")

    print("\n[T6] softmax1 -> BOS-sink; predict EXACT", flush=True)
    r = set_attn_config(model, sink="bos_sink"); stage("bos_sink", "T6 bos_sink"); r()

    print("\n[T5] ALiBi -> RoPE binary-distance recency; predict DEGRADE", flush=True)
    r = set_attn_config(model, positional="rope"); stage("rope", "T5 rope"); r()

    print(f"\n[T4] SiLU -> Swish_b (GeLU emul, b={BETA}); predict BREAK", flush=True)
    r = patch_ffn_activation(lambda x: _swish_b(x, BETA)); stage("gelu_swish", "T4 gelu_swish"); r()

    print("\n[T4b] SiLU -> pure GeLU; predict BREAK harder", flush=True)
    r = patch_ffn_activation(_gelu); stage("gelu_pure", "T4b gelu_pure"); r()

    H = model.blocks[0].attn.n_heads
    for n_kv in sorted({d for d in (H // 2, 4, 2) if d >= 1 and H % d == 0}, reverse=True):
        print(f"\n[T2] MHA -> GQA n_kv={n_kv}; predict DEGRADE", flush=True)
        r = apply_kv_grouping(model, n_kv); stage(f"gqa{n_kv}", f"T2 gqa{n_kv}"); r()

    print("\n[T3] MHA -> MQA n_kv=1; predict CAM BREAK", flush=True)
    r = apply_kv_grouping(model, 1); stage("mqa", "T3 mqa"); r()

    print("\n" + "=" * 78)
    print("VERIFICATION MATRIX")
    print("=" * 78)

    def maxdiv_of(fails):
        return max([d for (_, _, _, d) in fails if d is not None], default=0)

    order = [
        ("baseline", "native (reference)"),
        ("bos_sink", "exact +1==exp(0)"),
        ("rope", "degrade (recency)"),
        ("gelu_swish", "approx floor .0139"),
        ("gelu_pure", "approx (worse)"),
    ] + [(k, "degrade (retrofit)") for k in results if k.startswith("gqa")] + [
        ("mqa", "break (CAM cap)"),
    ]
    print(f"{'transform':12s} {'ms_predict':20s} {'pass':>7s} {'maxdiv':>7s}  verdict")
    for key, pred in order:
        if key not in results:
            continue
        p, t, f = results[key]
        md = maxdiv_of(f)
        if p == t:
            verd = "byte-exact"
        elif p == 0:
            verd = "BREAKS-all"
        else:
            verd = f"partial({t-p} fail: " + ",".join(n for (n, _, _, _) in f) + ")"
        print(f"{key:12s} {pred:20s} {p:>3d}/{t:<3d} {md:>7d}  {verd}")

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
