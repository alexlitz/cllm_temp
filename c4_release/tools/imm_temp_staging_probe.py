#!/usr/bin/env python3
"""Wave 1 Cluster B3 probe — TEMP staging from MARK_PC to MARK_AX across L0-L5
for IMM 0xFF.

Per ``docs/IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md``, L5 head 0 (non-first-step
immediate fetch at AX from ``TEMP=PC+1``) is suspected to be landing on the
OPCODE byte at PC+0 for high-bit IMM values in ``[0xE0, 0xFF]``. Q reads
``TEMP+k`` (k=0..15) at the AX-marker row and matches against
``ADDR_KEY+32+k`` on the K side.

Expected staging (for IMM 0xFF at PC=2, IMM byte at code position 3 = 0xFF):
- L3 head 7 stages TEMP at MARK_PC (low nibble of PC byte-1 patterns).
- L4 ``_layer4_pc_plus1_ax_rules`` chain rotates EMBED -> TEMP+0..15 (lo) and
  TEMP+16..31 (hi) at the MARK_AX row, with offset=+1 (so MARK_AX TEMP
  pattern should encode the byte at PC+1 = the IMM byte 0xFF).

For 0xFF the expected TEMP pattern at MARK_AX:
- TEMP+15 high (low nibble F)
- TEMP+(16+15) = TEMP+31 high (hi nibble F)
- All other TEMP+0..14 and TEMP+16..30 ~ low.

Anything else at MARK_AX indicates the L4 PC+1 staging is wrong, OR a
later layer overwrites TEMP before L5 head 0 consumes it.

This probe captures the TEMP+k residual at BOTH the MARK_PC row AND the
MARK_AX row in the AX-marker forward, layer by layer, so the divergence
layer is identifiable.
"""

from __future__ import annotations
import sys
import os
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from neural_vm.embedding import Opcode  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.dim_registry import build_default_registry  # noqa: E402


def make_bytecode(ops):
    bytecode = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    return bytecode


def main():
    print(f"[probe] building runner...", flush=True)
    runner = BatchedPureNeuralRunner()
    model = runner.model
    registry = build_default_registry()

    # Capture TEMP+0..31 plus markers
    dim_names = ["MARK_AX", "MARK_PC", "IS_BYTE", "OP_IMM", "HAS_SE"]
    for k in range(32):
        dim_names.append(f"TEMP+{k}")
    # FETCH_LO/HI are the L5 head 0 OUTPUT band (where it writes the fetched
    # IMM byte). Capture these too to track the downstream consequence.
    for k in range(16):
        dim_names.append(f"FETCH_LO+{k}")
        dim_names.append(f"FETCH_HI+{k}")
    # Also capture ADDR_KEY+32..47 (the K-side high-nibble counters that
    # L5 head 0 matches against) for reference at MARK_PC rows.
    for k in range(16):
        dim_names.append(f"ADDR_KEY+{32 + k}")
    # EMBED_LO/HI carry the raw byte content per-position (L4 PC+1 chain
    # reads EMBED at MARK_AX). Capture for reference.
    for k in range(16):
        dim_names.append(f"EMBED_LO+{k}")
        dim_names.append(f"EMBED_HI+{k}")
    # CLEAN_EMBED is the post-L1/L2 cleaned byte content that L3 head 7 and
    # L4 pc_relay consume. Capture for reference at MARK_AX (post-L4-relay).
    for k in range(16):
        dim_names.append(f"CLEAN_EMBED_LO+{k}")
        dim_names.append(f"CLEAN_EMBED_HI+{k}")

    def dim_pos(name):
        if "+" in name:
            base, offset = name.split("+")
            return registry.slots[base].start + int(offset)
        return registry.slots[name].start

    positions = {n: dim_pos(n) for n in dim_names}

    captured = []

    def make_hook(block_idx, kind):
        # kind: 'pre_attn' (block input) or 'pre_ffn' (post-attn residual)
        def hook(module, args):
            t = args[0].detach().cpu().clone()
            captured.append((block_idx, kind, t))
        return hook

    # Register on every block:
    #   pre_attn = block input residual (Li input = L(i-1) post-FFN output)
    #   pre_ffn  = post-attn residual (Li post-attn, pre-FFN)
    #   post_ffn = post-FFN output (Li output)
    # post_ffn is captured via the next block's pre_attn (handled at lookup
    # time). We additionally register a forward_hook on the final block FFN.
    handles = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attn'):
            handles.append(block.attn.register_forward_pre_hook(make_hook(i, 'pre_attn')))
        if hasattr(block, 'ffn'):
            handles.append(block.ffn.register_forward_pre_hook(make_hook(i, 'pre_ffn')))

    # Add post-FFN capture by hooking the block module itself (full output).
    def make_post_block_hook(block_idx):
        def hook(module, args, output):
            t = output.detach().cpu().clone()
            captured.append((block_idx, 'post_ffn', t))
        return hook
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_post_block_hook(i)))

    # Allow CLI override of which IMM value to probe (default 0xFF).
    imm_val = int(os.environ.get("PROBE_IMM", "255"))
    bytecode = make_bytecode([
        (Opcode.IMM, imm_val), Opcode.PSH,
        (Opcode.IMM, 0xD5), Opcode.XOR,
        Opcode.EXIT,
    ])

    captured.clear()
    print(f"[probe] running IMM 0x{imm_val:02X} / PSH / IMM 0xD5 / XOR / EXIT", flush=True)
    print(f"[probe] expected lo_peak={imm_val & 0xF}, hi_peak={(imm_val >> 4) & 0xF} at MARK_AX", flush=True)
    results = runner.run_batch([bytecode], max_steps=20, bucket_by_predicted_length=False)
    out, exit_code = results[0]
    print(f"[probe] exit_code={hex(exit_code) if isinstance(exit_code, int) else exit_code}", flush=True)
    print(f"[probe] expected 0x2A (= 0xFF XOR 0xD5)", flush=True)
    print(f"[probe] total hook calls: {len(captured)}", flush=True)

    # Group hooks by forward. Each forward emits 2*n_blocks entries (pre_attn
    # + pre_ffn per block). Block 0 pre_attn starts a new forward.
    forward_groups = []
    cur = []
    for entry in captured:
        block_idx, kind, t = entry
        if kind == 'pre_attn' and block_idx == 0 and cur:
            forward_groups.append(cur)
            cur = []
        cur.append(entry)
    if cur:
        forward_groups.append(cur)
    print(f"[probe] forward groups: {len(forward_groups)}", flush=True)

    # Find forwards where MARK_AX exists at the last seq position. The IMM 0xFF
    # AX-emit forward is the first such forward.
    mark_ax_fwds = []
    for fi, fg in enumerate(forward_groups):
        _, _, last_t = fg[-1]
        sl = last_t.shape[1]
        if sl == 0:
            continue
        row = last_t[0, sl-1, :].numpy()
        if float(row[positions["MARK_AX"]]) > 0.5:
            mark_ax_fwds.append(fi)
    print(f"[probe] MARK_AX-at-last-pos forwards: {mark_ax_fwds}", flush=True)

    if not mark_ax_fwds:
        print("[probe] no MARK_AX found -- abort", flush=True)
        return

    # Focus on the IMM 0xFF AX forward (the first MARK_AX forward; per v3 probe
    # this is fwd_idx=6 for this bytecode).
    target_fwd = mark_ax_fwds[0]
    print(f"[probe] target forward: {target_fwd}", flush=True)
    fg = forward_groups[target_fwd]
    _, _, t0 = fg[0]
    seq_len = t0.shape[1]
    print(f"[probe] seq_len={seq_len}", flush=True)

    # Find MARK_PC row and MARK_AX row. We use the input to L0 (block 0 pre_attn)
    # as the row classifier source (markers are baked into embeddings).
    mark_pc_pos = None
    mark_ax_pos = None
    # MARK_AX is at the last position by construction. MARK_PC is the row with
    # MARK_PC>0.5 in the input.
    input_t = t0[0]  # [seq_len, d_model]
    for pos in range(seq_len):
        row = input_t[pos, :].numpy()
        if float(row[positions["MARK_PC"]]) > 0.5 and mark_pc_pos is None:
            mark_pc_pos = pos
        if float(row[positions["MARK_AX"]]) > 0.5 and mark_ax_pos is None:
            mark_ax_pos = pos
    print(f"[probe] mark_pc_pos={mark_pc_pos}, mark_ax_pos={mark_ax_pos}", flush=True)

    if mark_pc_pos is None or mark_ax_pos is None:
        print("[probe] missing marker -- abort", flush=True)
        return

    # Dump position layout: find where the IMM byte (0xFF, nibble pattern lo=15,
    # hi=15) lives in the sequence at L0 input. The L4 PC+1@AX rule reads
    # EMBED at the CODE byte at PC+1; this position needs to be relayed via
    # L4 attention head 0 (V slot from MARK_PC) before TEMP-rotation works.
    print(f"\n[probe] === position layout scan (L0 pre_attn) ===", flush=True)
    for pos in range(seq_len):
        row = t0[0, pos, :].numpy()
        flags = []
        if float(row[positions["MARK_PC"]]) > 0.5:
            flags.append("MARK_PC")
        if float(row[positions["MARK_AX"]]) > 0.5:
            flags.append("MARK_AX")
        if float(row[positions["IS_BYTE"]]) > 0.5:
            flags.append("IS_BYTE")
        # Sample EMBED_LO/HI peaks for byte-content positions
        e_lo = [float(row[positions[f"EMBED_LO+{k}"]]) for k in range(16)]
        e_hi = [float(row[positions[f"EMBED_HI+{k}"]]) for k in range(16)]
        e_lo_peak = max(range(16), key=lambda k: e_lo[k])
        e_hi_peak = max(range(16), key=lambda k: e_hi[k])
        if flags or e_lo[e_lo_peak] > 0.5 or e_hi[e_hi_peak] > 0.5:
            print(f"  pos {pos:3d}: flags={flags} EMBED lo_peak={e_lo_peak}({round(e_lo[e_lo_peak],2)}) hi_peak={e_hi_peak}({round(e_hi[e_hi_peak],2)})", flush=True)

    # Helper: extract nibble pattern (lo+hi 16-slots each) and find the peak.
    def nibble_summary(row, lo_base, hi_base):
        lo = [float(row[positions[f"{lo_base}+{k}"]]) for k in range(16)]
        hi = [float(row[positions[f"{hi_base}+{k}"]]) for k in range(16)]
        lo_nonzero = {k: round(v, 3) for k, v in enumerate(lo) if abs(v) > 1e-2}
        hi_nonzero = {k: round(v, 3) for k, v in enumerate(hi) if abs(v) > 1e-2}
        lo_peak = max(range(16), key=lambda k: lo[k])
        hi_peak = max(range(16), key=lambda k: hi[k])
        return {
            "lo_peak": lo_peak, "lo_peak_val": round(lo[lo_peak], 3),
            "hi_peak": hi_peak, "hi_peak_val": round(hi[hi_peak], 3),
            "lo": lo_nonzero,
            "hi": hi_nonzero,
        }

    # TEMP is special: TEMP+0..15 = lo, TEMP+16..31 = hi
    def temp_summary(row):
        lo = [float(row[positions[f"TEMP+{k}"]]) for k in range(16)]
        hi = [float(row[positions[f"TEMP+{16 + k}"]]) for k in range(16)]
        lo_nonzero = {k: round(v, 3) for k, v in enumerate(lo) if abs(v) > 1e-2}
        hi_nonzero = {k: round(v, 3) for k, v in enumerate(hi) if abs(v) > 1e-2}
        lo_peak = max(range(16), key=lambda k: lo[k])
        hi_peak = max(range(16), key=lambda k: hi[k])
        return {
            "lo_peak": lo_peak, "lo_peak_val": round(lo[lo_peak], 3),
            "hi_peak": hi_peak, "hi_peak_val": round(hi[hi_peak], 3),
            "lo": lo_nonzero,
            "hi": hi_nonzero,
        }

    # Walk L0..L5 (block input + post-attn) and L6 input (== L5 output) at both
    # MARK_PC and MARK_AX rows.
    print(f"\n[probe] === TEMP staging walk L0..L6 ===", flush=True)
    print(f"[probe] IMM byte 0xFF -> expected lo_peak=15 (low nibble F), hi_peak=15 (hi nibble F) at MARK_AX", flush=True)
    print(f"[probe] (At MARK_PC, TEMP holds PC byte0/byte1 pattern from L3 head 7 -- compare to MARK_AX for divergence)", flush=True)

    findings = []
    for layer in range(7):  # 0..6
        for kind in ('pre_attn', 'pre_ffn', 'post_ffn'):
            entry = None
            for b_idx, k, t in fg:
                if b_idx == layer and k == kind:
                    entry = t
                    break
            if entry is None:
                continue
            label = f"L{layer}_{kind}"

            pc_row = entry[0, mark_pc_pos, :].numpy()
            ax_row = entry[0, mark_ax_pos, :].numpy()
            pc_tmp = temp_summary(pc_row)
            ax_tmp = temp_summary(ax_row)
            pc_fetch = nibble_summary(pc_row, "FETCH_LO", "FETCH_HI")
            ax_fetch = nibble_summary(ax_row, "FETCH_LO", "FETCH_HI")
            pc_embed = nibble_summary(pc_row, "EMBED_LO", "EMBED_HI")
            ax_embed = nibble_summary(ax_row, "EMBED_LO", "EMBED_HI")
            pc_cembed = nibble_summary(pc_row, "CLEAN_EMBED_LO", "CLEAN_EMBED_HI")
            ax_cembed = nibble_summary(ax_row, "CLEAN_EMBED_LO", "CLEAN_EMBED_HI")

            print(f"\n  {label}:", flush=True)
            print(f"    MARK_PC  TEMP  lo_peak={pc_tmp['lo_peak']:>2} ({pc_tmp['lo_peak_val']:>7}) "
                  f"hi_peak={pc_tmp['hi_peak']:>2} ({pc_tmp['hi_peak_val']:>7})", flush=True)
            print(f"      TEMP lo nz: {pc_tmp['lo']}", flush=True)
            print(f"      TEMP hi nz: {pc_tmp['hi']}", flush=True)
            print(f"    MARK_PC  FETCH lo_peak={pc_fetch['lo_peak']:>2} ({pc_fetch['lo_peak_val']:>7}) "
                  f"hi_peak={pc_fetch['hi_peak']:>2} ({pc_fetch['hi_peak_val']:>7})", flush=True)
            print(f"      FETCH lo nz: {pc_fetch['lo']}", flush=True)
            print(f"      FETCH hi nz: {pc_fetch['hi']}", flush=True)
            print(f"    MARK_AX  TEMP  lo_peak={ax_tmp['lo_peak']:>2} ({ax_tmp['lo_peak_val']:>7}) "
                  f"hi_peak={ax_tmp['hi_peak']:>2} ({ax_tmp['hi_peak_val']:>7})", flush=True)
            print(f"      TEMP lo nz: {ax_tmp['lo']}", flush=True)
            print(f"      TEMP hi nz: {ax_tmp['hi']}", flush=True)
            print(f"    MARK_AX  FETCH lo_peak={ax_fetch['lo_peak']:>2} ({ax_fetch['lo_peak_val']:>7}) "
                  f"hi_peak={ax_fetch['hi_peak']:>2} ({ax_fetch['hi_peak_val']:>7})", flush=True)
            print(f"      FETCH lo nz: {ax_fetch['lo']}", flush=True)
            print(f"      FETCH hi nz: {ax_fetch['hi']}", flush=True)
            print(f"    MARK_PC  EMBED lo_peak={pc_embed['lo_peak']:>2} ({pc_embed['lo_peak_val']:>7}) "
                  f"hi_peak={pc_embed['hi_peak']:>2} ({pc_embed['hi_peak_val']:>7})", flush=True)
            print(f"      EMBED lo nz: {pc_embed['lo']}", flush=True)
            print(f"      EMBED hi nz: {pc_embed['hi']}", flush=True)
            print(f"    MARK_AX  EMBED lo_peak={ax_embed['lo_peak']:>2} ({ax_embed['lo_peak_val']:>7}) "
                  f"hi_peak={ax_embed['hi_peak']:>2} ({ax_embed['hi_peak_val']:>7})", flush=True)
            print(f"      EMBED lo nz: {ax_embed['lo']}", flush=True)
            print(f"      EMBED hi nz: {ax_embed['hi']}", flush=True)
            print(f"    MARK_PC  CEMBD lo_peak={pc_cembed['lo_peak']:>2} ({pc_cembed['lo_peak_val']:>7}) "
                  f"hi_peak={pc_cembed['hi_peak']:>2} ({pc_cembed['hi_peak_val']:>7})", flush=True)
            print(f"      CEMBD lo nz: {pc_cembed['lo']}", flush=True)
            print(f"      CEMBD hi nz: {pc_cembed['hi']}", flush=True)
            print(f"    MARK_AX  CEMBD lo_peak={ax_cembed['lo_peak']:>2} ({ax_cembed['lo_peak_val']:>7}) "
                  f"hi_peak={ax_cembed['hi_peak']:>2} ({ax_cembed['hi_peak_val']:>7})", flush=True)
            print(f"      CEMBD lo nz: {ax_cembed['lo']}", flush=True)
            print(f"      CEMBD hi nz: {ax_cembed['hi']}", flush=True)

            findings.append({
                "stage": label,
                "mark_pc_temp": pc_tmp,
                "mark_ax_temp": ax_tmp,
                "mark_pc_fetch": pc_fetch,
                "mark_ax_fetch": ax_fetch,
                "mark_pc_embed": pc_embed,
                "mark_ax_embed": ax_embed,
                "mark_pc_cembd": pc_cembed,
                "mark_ax_cembd": ax_cembed,
            })

    # Also dump ADDR_KEY+32..47 at MARK_PC for reference (PC byte K side).
    print(f"\n[probe] === ADDR_KEY+32..47 reference at MARK_PC (L5 pre_attn = K side input) ===", flush=True)
    for b_idx, k, t in fg:
        if b_idx == 5 and k == 'pre_attn':
            pc_row = t[0, mark_pc_pos, :].numpy()
            ax_row = t[0, mark_ax_pos, :].numpy()
            print(f"  MARK_PC ADDR_KEY+32..47: {[round(float(pc_row[positions[f'ADDR_KEY+{32 + i}']]), 2) for i in range(16)]}", flush=True)
            print(f"  MARK_AX ADDR_KEY+32..47: {[round(float(ax_row[positions[f'ADDR_KEY+{32 + i}']]), 2) for i in range(16)]}", flush=True)
            break

    # Save findings to JSON for the doc
    out_path = HERE / "imm_temp_staging_probe_findings.json"
    with open(out_path, "w") as f:
        json.dump({
            "exit_code": int(exit_code) if isinstance(exit_code, int) else None,
            "target_fwd": target_fwd,
            "mark_pc_pos": mark_pc_pos,
            "mark_ax_pos": mark_ax_pos,
            "findings": findings,
        }, f, indent=2)
    print(f"\n[probe] findings written to {out_path}", flush=True)

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
