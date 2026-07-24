"""Probe: is the matmul step-225 LEA off-by-one (228 -> 229) an fp RESIDUE in the
LEA_Q nibble sum at large context, and does it depend on context length S?

Runs the cached driver up to the failing LEA step, then does ONE full uncached
forward reading the residual AFTER each LEA block (lea-q-reduce / lea-addr-nib) at
the query row to expose LEA_Q, the BP nibble reads, and AXB_LO/HI.

Also runs the SAME LEA (BP=244, IMM=-4) at a SHORT context (freshly, ~few frames)
to compare — isolating whether the drift is context-length-driven.

Run:  OMP_NUM_THREADS=4 python -m c4_min.selfhost._probe_lea_residue --device=cuda:0
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch

from c4_min import isa
from c4_min.selfhost._matmul_src import matmul_c, SCALE


def _compile(A, B):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, data = compile_c(matmul_c(A, B, SCALE))
    return bytecode_to_isa(bc), data


def build(device):
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(
        code_size=256, recurrent_divmod=True, addr32=True)
    if device != "cpu":
        sparse = sparse.to(device)
    return sparse, L


def _block_names(L, n_blocks):
    # L carries _apply_order; block names were assigned in build order. We fetch the
    # 'lea-q-reduce' / 'lea-addr-nib' indices by re-deriving from the layout's stored
    # spec order if available; else scan by probing dims.
    return None


def run_and_capture(model, L, code, device, stop_step, prune_interval=60,
                    hook_step=None):
    """Cached driver clone.  At `hook_step` it wraps each block's forward to record
    the query-row (last) residual per block, exposing LEA internals AS COMPUTED BY
    THE REAL CACHED PATH (blocks run in stored order inside forward_hidden_cached)."""
    from c4_min.nibble_pure_forward_cached import BlockKVCacheBatched, apply_overlay_window
    from c4_min.nibble_pure_forward import _MEM_MARKER_LOCAL, SP_INIT, _snap_lane
    from c4_min.nibble_pure_forward_complete import _build_frame, _decode_reg_from_nibbles, _mem_top
    from c4_min import blogspec_vocab as V
    blocks = model.blocks; n_blocks = len(blocks)
    H = blocks[0].attn.n_heads; HD = blocks[0].attn.head_dim
    caches = [BlockKVCacheBatched(H, HD, blocks[b].attn.alibi_slopes) for b in range(n_blocks)]
    tsp = 0
    store_log = {}
    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    stream = [V.BOS] + init_frame
    cur_pc = 0; cur_sp = cur_bp = SP_INIT; cur_ax = 0; frame_idx = 0
    win_start = 0; win_len = len(stream)
    captured = {"snaps": None}
    for step in range(stop_step + 1):
        win_toks = torch.tensor([stream[win_start:win_start + win_len]], device=device)
        q_positions = torch.arange(win_start, win_start + win_len, device=device)

        snaps = []
        orig_calls = []
        if step == hook_step:
            for b in blocks:
                orig_calls.append(b.__class__.__call__)
            def make_wrap(orig):
                def wrap(self, x, **kw):
                    out = orig(self, x, **kw)
                    y = out[0] if isinstance(out, tuple) else out
                    last = y[0, -1]
                    snaps.append(dict(
                        LEA_Q=float(last[L.LEA_Q]),
                        BP0=float(last[L.BP + 0]), BP1=float(last[L.BP + 1]),
                        IMM0=float(last[L.IMM_NIB + 0]), IMM1=float(last[L.IMM_NIB + 1]),
                        AXB_LO=float(last[L.AXB_LO]), AXB_HI=float(last[L.AXB_HI]),
                        AX0=float(last[L.AX + 0]), AX1=float(last[L.AX + 1]),
                        AXV=float(last[L.AX_VAL]),
                        OP_LEA=float(last[L.OP_IS + isa.LEA]),
                    ))
                    return out
                return wrap
            # monkeypatch the shared Block class once
            BlkCls = blocks[0].__class__
            _orig = BlkCls.__call__
            BlkCls.__call__ = make_wrap(_orig)

        with torch.no_grad():
            x = model.embed[win_toks].clone()
            apply_overlay_window(x, win_start, code, L, store_log, is_last_row_query=True)
            past = [caches[b].as_past_kv() for b in range(n_blocks)]
            hidden, new_kv = model.forward_hidden_cached(
                x, past_key_values=past, q_positions=q_positions, use_cache=True)

        if step == hook_step:
            blocks[0].__class__.__call__ = _orig
            captured["snaps"] = list(snaps)

        state = hidden[0, -1]
        pc = _snap_lane(state[L.PC_VAL]); sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL]); stk = _snap_lane(state[L.STK_VAL])
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        ax = _decode_reg_from_nibbles(state, L, L.AX)
        if step == stop_step:
            return dict(store_log=dict(store_log), cur_pc=cur_pc, cur_bp=cur_bp,
                        cur_sp=cur_sp, op=op, ax=ax, S=len(stream),
                        snaps=captured["snaps"])
        s_addr = s_val = 0; is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & 0xFFFFFFFF
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & 0xFFFFFFFF
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val & 0xFFFFFFFF)
        n_commit = win_len - 1
        if n_commit > 0:
            for b in range(n_blocks):
                K_all, V_all, pos_all = new_kv[b]
                caches[b].commit(K_all[:, :, -win_len:, :][:, :, :n_commit, :],
                                 V_all[:, :, -win_len:, :][:, :, :n_commit, :],
                                 pos_all[-win_len:][:n_commit])
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if pc < 0 or pc >= len(code):
            break
        prev_q = win_start + win_len - 1
        stream += frame; win_start = prev_q; win_len = 1 + V.FRAME_LEN
        tsp += V.FRAME_LEN
        if tsp >= prune_interval:
            for b in range(n_blocks):
                caches[b].evict(0.99, prune_interval, 1e-9, 1e-6)
            tsp = 0
    return None


def dump_snaps(snaps, label):
    print(f"\n==== {label} ====", flush=True)
    if not snaps:
        print("  (no snaps)"); return
    prev_q = 0.0
    for bi, s in enumerate(snaps):
        if abs(s["LEA_Q"]) > 1e-9 or abs(prev_q) > 1e-9 or s["OP_LEA"] > 0.5:
            print(f"  blk{bi:>2}: OP_LEA={s['OP_LEA']:.3f} LEA_Q={s['LEA_Q']:.8f} "
                  f"BP0={s['BP0']:.6f} BP1={s['BP1']:.6f} "
                  f"IMM0={s['IMM0']:.4f} IMM1={s['IMM1']:.4f} "
                  f"AXB_LO={s['AXB_LO']:.6f} AXB_HI={s['AXB_HI']:.6f} "
                  f"AX0={s['AX0']:.6f} AX1={s['AX1']:.6f} AXV={s['AXV']:.4f}", flush=True)
        prev_q = s["LEA_Q"]
    last = snaps[-1]
    dec = int(round(last["AX0"])) + 16 * int(round(last["AX1"]))
    print(f"  FINAL decoded AX byte = {dec}  (AX0={last['AX0']:.4f} AX1={last['AX1']:.4f})",
          flush=True)


def main():
    device = "cpu"
    for a in sys.argv:
        if a.startswith("--device="):
            device = a.split("=", 1)[1]
    A = [[1, 1], [2, 1]]; B = [[1, 2], [3, 1]]
    code, data = _compile(A, B)
    t0 = time.time(); model, L = build(device)
    print(f"build {time.time()-t0:.1f}s device={device}", flush=True)

    # The failing LEA is step 225 (LEA -4 at pc=167, BP=244). Hook the blocks at that
    # step and read the LEA internals from the REAL cached forward.
    cap = run_and_capture(model, L, code, device, stop_step=225, hook_step=225)
    print(f"captured at step 225: op={isa.NAMES.get(cap['op'], cap['op'])} "
          f"pc={cap['cur_pc']} BP={cap['cur_bp']} SP={cap['cur_sp']} S={cap['S']} "
          f"decoded_ax={cap['ax'] & 0xFFFFFFFF}", flush=True)
    dump_snaps(cap["snaps"], "matmul step225 LEA -4 @ BP=244 (deep, S~6781)")

    # Compare: run the SAME LEA -4 at BP=244 at a SHORT context to see if it is
    # context-length driven.  A short program: ENT to set BP=244... simplest is to
    # just re-check an EARLY step in the same run that executes LEA at small S.
    # (already covered by the per-block dump; the short-context comparison is done
    #  in the second probe.)


if __name__ == "__main__":
    main()
