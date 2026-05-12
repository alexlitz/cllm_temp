"""Per-layer trace harness for the +6 ADD leak.

Strategy:
  - Build the compiled neural VM model (cached) in both `pure_neural=True`
    and handler-mode (`pure_neural=False`) configurations. Both build the
    SAME underlying model (compile_full_vm); the difference is in the
    runner's Python overrides at STEP_END boundaries, not in the residual
    stream during a single forward.
  - Construct a *synthesized* context that contains the bytecode prefix
    PLUS the prior STEP_END sequences for steps 1, 2, 3 (IMM/PSH/IMM) so
    that step 4 (the ADD step) is the one being generated.
  - Identify the AX marker position inside the ADD step's STEP_END output.
    The model emits the STEP_END sequence as 35 tokens; the REG_AX marker
    is at offset 5 of the sequence (PC takes 0..4, AX 5..9).
  - Run the model forward token-by-token until the AX marker token itself
    is the last input. At each block boundary, snapshot the residual
    stream at the AX marker position for dims:
      ALU_LO[0..15], ALU_HI[0..15], AX_CARRY_LO[0..15], AX_CARRY_HI[0..15],
      OUTPUT_LO[0..15], OUTPUT_HI[0..15], OP_ADD.
  - Then continue the forward, and also snapshot when the 1st AX value
    byte is generated. We compare the LO nibble decision and the HI nibble
    decision (which determine the byte the model emits).

This is the harness the prior agent missed building, and which the user
explicitly requested.
"""
from __future__ import annotations

import os
import sys
import torch

# Make c4_release importable when run from repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from c4_release.neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from c4_release.neural_vm.vm_step import Token, _SetDim as BD_DEFAULT  # noqa: E402
from c4_release.neural_vm.embedding import Opcode  # noqa: E402


# Dim positions are resolved at runtime from the compiled model's
# `dim_positions` dict (when available). The set_vm_weights legacy path
# falls back to `_SetDim` constants which differ from the compiler-allocated
# positions — using the wrong positions yields meaningless residual reads.
DIM_RANGE_NAMES = [
    "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
    "OUTPUT_LO", "OUTPUT_HI",
]
DIM_SCALAR_NAMES = ["OP_ADD", "MARK_AX", "OP_PSH", "OP_IMM"]

DIM_RANGES = {}
DIM_SCALARS = {}


def _bind_dims(dim_positions):
    """Populate DIM_RANGES/DIM_SCALARS from the model's actual dim_positions.

    Falls back to _SetDim constants when a name isn't present in the dict.
    """
    DIM_RANGES.clear()
    DIM_SCALARS.clear()

    def D(name):
        if isinstance(dim_positions, dict) and name in dim_positions:
            return dim_positions[name]
        return getattr(BD_DEFAULT, name)

    for name in DIM_RANGE_NAMES:
        DIM_RANGES[name] = (D(name), 16)
    for name in DIM_SCALAR_NAMES:
        DIM_SCALARS[name] = D(name)


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _argmax_of_range(vec, base, n=16):
    sub = vec[base:base + n]
    return int(sub.argmax().item()), float(sub[sub.argmax().item()].item())


def _bake_run_to_get_correct_step_seq(prog):
    """Run handler-mode runner once to obtain the well-formed token stream
    (with PC/AX/SP/BP/STACK0/MEM/STEP_END sections for each step).

    Because handler mode passes test_add_basic (returns 42), the token
    stream up through the ADD step's STEP_END contains the *correct*
    AX = 42 value bytes. We can then use this as a reference for what
    pure_neural mode *should* emit at the ADD step's AX marker.
    """
    runner = AutoregressiveVMRunner(
        pure_neural=False,
        trust_neural_alu=True,
    )
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    bc = _make_bc(prog)
    _, exit_code = runner.run(bc, b"", max_steps=30)
    # Capture last full context. The runner has self.model and the loop
    # context is local to run(); we don't have access to it directly. But
    # we can rerun with hooks to capture it. Simpler: re-do generation via
    # an instrumented call.
    return runner, exit_code


def _capture_handler_mode_context(prog):
    """Run handler-mode and capture the final context (token list)."""
    runner = AutoregressiveVMRunner(
        pure_neural=False,
        trust_neural_alu=True,
    )
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    bc = _make_bc(prog)

    # Monkey-patch run to capture the context. We replicate the body but
    # also save context.
    captured_contexts = []

    real_generate = runner._generate_next_cached

    def _hook(gen_ctx):
        captured_contexts.append(list(gen_ctx))
        return real_generate(gen_ctx)

    runner._generate_next_cached = _hook
    out, exit_code = runner.run(bc, b"", max_steps=30)
    return runner, captured_contexts, exit_code


def _find_add_step_ax_marker_idx(bytecode, context):
    """Locate the token index of REG_AX inside the STEP_END sequence
    emitted *after* the ADD instruction dispatched.

    Strategy: walk the context, find STEP_END tokens (262). The N-th
    STEP_END corresponds to step N. The ADD step is the 4th instruction
    in `IMM 10 PSH IMM 32 ADD EXIT` (indices: 0=IMM, 1=PSH, 2=IMM,
    3=ADD, 4=EXIT). So we want the 4th STEP_END.

    The STEP_END sequence is laid out (35 tokens):
        REG_PC v0 v1 v2 v3   (5 tokens, positions 0..4)
        REG_AX v0 v1 v2 v3   (5 tokens, positions 5..9)
        REG_SP v0 v1 v2 v3   (5 tokens, positions 10..14)
        REG_BP v0 v1 v2 v3   (5 tokens, positions 15..19)
        STACK0 v0 v1 v2 v3   (5 tokens, positions 20..24)
        MEM addr0..3 v0..v3  (9 tokens, positions 25..33)
        STEP_END             (1 token, position 34)

    The REG_AX marker token is REG_AX (=258).
    Find the 4th REG_AX in the context (one per step). It's the one
    where the value bytes that follow encode 42.
    """
    ax_positions = []
    for i, t in enumerate(context):
        if t == Token.REG_AX:
            ax_positions.append(i)
    return ax_positions


def _decode_register_value(context, marker_idx):
    """Read 4 value bytes following marker_idx in context, little-endian."""
    val = 0
    for j in range(4):
        if marker_idx + 1 + j < len(context):
            val |= (context[marker_idx + 1 + j] & 0xFF) << (j * 8)
    return val


def _build_handler_context_up_to_add_ax(prog):
    """Run handler mode, capture context just before the model needs to emit
    the AX value bytes of the ADD step (i.e., the REG_AX marker for the ADD
    step is the *last* token in the partial context).

    Returns (context_truncated, full_context, ax_marker_idx_in_context).
    """
    runner, contexts, exit_code = _capture_handler_mode_context(prog)
    full = contexts[-1] + [
        # Append the very last emitted token to fully recover
    ] if contexts else []
    # Take the longest context captured (last call to _generate_next_cached).
    # We want the context just before the ADD step's AX byte 0 is emitted.
    # Find AX markers in the longest available context.
    longest = max(contexts, key=len) if contexts else []
    # Locate steps by PC markers (REG_PC + 4 bytes encodes pc). A clean step
    # has PC = 2 + step_idx*8. Step 4 (ADD-completed) has PC = 2 + 4*8 = 34.
    pc_positions = [i for i, t in enumerate(longest) if t == Token.REG_PC]
    print(f"  All REG_PC positions: {pc_positions}")
    pc_to_idx = {}
    for i, pc_idx in enumerate(pc_positions):
        pc_val = _decode_register_value(longest, pc_idx)
        print(f"    step{i+1} PC marker @ idx {pc_idx} → PC={pc_val}")
        pc_to_idx[pc_val] = pc_idx

    ax_positions = _find_add_step_ax_marker_idx(None, longest)
    print(f"  All REG_AX positions: {ax_positions}")
    for i, ax_idx in enumerate(ax_positions):
        val = _decode_register_value(longest, ax_idx)
        print(f"    step{i+1} AX marker @ idx {ax_idx} → value={val}")
    # ADD is the 4th instruction. PSH is no-op for AX. So:
    #   step1 (IMM 10): AX=10
    #   step2 (PSH):    AX=10
    #   step3 (IMM 32): AX=32
    #   step4 (ADD):    AX=42 (expected)
    #   step5 (EXIT):   AX unchanged from step4
    # Pick the 4th REG_AX marker (index 3).
    if len(ax_positions) < 4:
        print(f"WARNING: Only found {len(ax_positions)} REG_AX markers "
              f"in context (need >= 4)")
        return longest, longest, None
    # Find the AX marker that comes AFTER PC=34 (ADD step). The step format
    # is PC, AX, SP, BP, STK0, MEM, SE; AX immediately follows PC's 4 value
    # bytes.
    target_pc = 34
    add_ax_idx = None
    if target_pc in pc_to_idx:
        pc_idx = pc_to_idx[target_pc]
        # AX marker should be at pc_idx + 5 (PC marker + 4 value bytes + AX marker)
        cand = pc_idx + 5
        if cand < len(longest) and longest[cand] == Token.REG_AX:
            add_ax_idx = cand
    if add_ax_idx is None:
        # Fallback: take the 4th REG_AX
        add_ax_idx = ax_positions[3]
    add_ax_value = _decode_register_value(longest, add_ax_idx)
    print(f"  Handler-mode ADD step REG_AX at idx {add_ax_idx}: value={add_ax_value} "
          f"(expect 42 = 10+32)")
    if add_ax_value != 42:
        print(f"  WARNING: handler-mode ADD AX = {add_ax_value}, not 42!")
    # We want the truncated context where AX marker is the LAST token.
    # That way the model is about to emit byte 0 of AX = 42.
    truncated = longest[:add_ax_idx + 1]
    print(f"  Truncated context length: {len(truncated)} (full: {len(longest)})")
    return truncated, longest, add_ax_idx


def _hook_and_forward(model, token_ids, target_pos):
    """Run a forward pass, snapshotting the residual stream at `target_pos`
    after each block. Returns a list of {block_idx, key: stats} dicts.
    """
    snaps = []

    def make_hook(block_idx):
        def _hook(module, inputs, output):
            # output is the residual after this block. Capture target_pos.
            try:
                if isinstance(output, tuple):
                    out_tensor = output[0]
                else:
                    out_tensor = output
                vec = out_tensor[0, target_pos].detach().clone().cpu()
                snaps.append((block_idx, vec))
            except Exception as e:
                snaps.append((block_idx, f"hook error: {e}"))
        return _hook

    handles = []
    for i, block in enumerate(model.blocks):
        h = block.register_forward_hook(make_hook(i))
        handles.append(h)

    # Also snapshot post-embedding.
    embed_snap = []

    def embed_hook(module, inputs, output):
        try:
            vec = output[0, target_pos].detach().clone().cpu()
            embed_snap.append(vec)
        except Exception as e:
            embed_snap.append(f"hook error: {e}")

    h_embed = model.embed.register_forward_hook(embed_hook)
    handles.append(h_embed)

    try:
        with torch.no_grad():
            _ = model.forward(token_ids)
    finally:
        for h in handles:
            h.remove()

    return embed_snap, snaps


def _summarize_residual(vec):
    """Extract our dims of interest from a residual snapshot."""
    if isinstance(vec, str):
        return vec
    out = {}
    for name, scalar_dim in DIM_SCALARS.items():
        out[name] = float(vec[scalar_dim].item())
    for name, (base, n) in DIM_RANGES.items():
        sub = vec[base:base + n]
        argmax = int(sub.argmax().item())
        max_v = float(sub[argmax].item())
        sum_v = float(sub.sum().item())
        # Top-3 nibbles by absolute value
        absvals = sub.abs()
        top3_idx = absvals.topk(min(3, n)).indices.tolist()
        top3 = [(int(i), float(sub[int(i)].item())) for i in top3_idx]
        out[name] = {"argmax": argmax, "max": max_v, "sum": sum_v, "top3": top3}
    return out


def _print_summary(label, embed_snap, snaps):
    print(f"\n=== {label} ===")
    if embed_snap:
        s = _summarize_residual(embed_snap[0])
        print(f"  post-embed:")
        _print_residual(s)
    for block_idx, vec in snaps:
        s = _summarize_residual(vec)
        print(f"  after L{block_idx}:")
        _print_residual(s)


def _print_residual(s):
    if isinstance(s, str):
        print(f"    {s}")
        return
    # Print scalars first
    scalars = []
    for name in DIM_SCALARS:
        v = s[name]
        if abs(v) > 0.01:
            scalars.append(f"{name}={v:.2f}")
    if scalars:
        print(f"    scalars: " + ", ".join(scalars))
    # Then nibble groups - only show if there's a clear winner or notable mass
    for name in DIM_RANGES:
        d = s[name]
        if d["max"] > 0.01 or abs(d["sum"]) > 0.01:
            top3 = d.get("top3", [])
            top3_str = ", ".join([f"[{i}]={v:.2f}" for i, v in top3])
            print(f"    {name}: argmax={d['argmax']:2d} max={d['max']:6.2f} "
                  f"sum={d['sum']:7.2f}  top3={top3_str}")


def _diff_summary(s_pn, s_hm):
    """Compare two residual summaries. Return list of (key, pn, hm)."""
    if isinstance(s_pn, str) or isinstance(s_hm, str):
        return []
    diffs = []
    for name in DIM_SCALARS:
        if abs(s_pn[name] - s_hm[name]) > 0.1:
            diffs.append((name, s_pn[name], s_hm[name]))
    for name in DIM_RANGES:
        a = s_pn[name]
        b = s_hm[name]
        if a["argmax"] != b["argmax"] or abs(a["max"] - b["max"]) > 0.5:
            diffs.append((f"{name}.argmax", a["argmax"], b["argmax"]))
            diffs.append((f"{name}.max", a["max"], b["max"]))
    return diffs


def _print_token_stream(label, tokens):
    print(f"\n  --- {label} ({len(tokens)} tokens) ---")
    TOKEN_NAMES = {
        256: "SEP", 257: "PC", 258: "AX", 259: "SP", 260: "BP",
        261: "MEM", 262: "SE", 263: "HALT", 264: "CS", 265: "CE",
        266: "DS", 267: "DE", 268: "STK0", 269: "UIS", 270: "UIE",
        271: "TCL", 272: "THS", 273: "THE", 274: "IOEB", 275: "IOET",
    }
    line = []
    for i, t in enumerate(tokens):
        if t in TOKEN_NAMES:
            line.append(f"[{i}:{TOKEN_NAMES[t]}]")
        else:
            line.append(f"{t}")
        if len(line) >= 15:
            print("    " + " ".join(line))
            line = []
    if line:
        print("    " + " ".join(line))


def main():
    prog = [
        (Opcode.IMM, 10),
        Opcode.PSH,
        (Opcode.IMM, 32),
        Opcode.ADD,
        Opcode.EXIT,
    ]

    print(">>> Building handler-mode runner (reference) and capturing contexts...")
    truncated_ctx, full_ctx, add_ax_idx = _build_handler_context_up_to_add_ax(prog)
    if add_ax_idx is None:
        print("FATAL: could not find ADD-step AX marker in handler-mode context")
        return

    _print_token_stream("handler-mode full context", full_ctx)

    # Now build a fresh pure_neural runner. It re-uses the cached model via
    # AutoregressiveVMRunner._MODEL_CACHE; we just want access to model.
    print("\n>>> Building pure_neural runner...")
    pn_runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
    )
    pn_runner._func_call_handlers = {}
    pn_runner._syscall_handlers = {}
    pn_model = pn_runner.model

    # Build a handler-mode runner that shares the same compile (cached).
    hm_runner = AutoregressiveVMRunner(
        pure_neural=False,
        trust_neural_alu=True,
    )
    hm_model = hm_runner.model

    print(f"\n  Same model object? {pn_model is hm_model}")
    print(f"  pn n_blocks: {len(pn_model.blocks)}, hm n_blocks: {len(hm_model.blocks)}")

    # Dump dim_positions if available, and rebind DIM_RANGES/SCALARS.
    dp = getattr(pn_model, 'dim_positions', None)
    if isinstance(dp, dict):
        print("\n  dim_positions for key dims:")
        for name in ("ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                      "OUTPUT_LO", "OUTPUT_HI", "AX_FULL_LO", "AX_FULL_HI",
                      "EMBED_LO", "EMBED_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
                      "OP_ADD", "OP_PSH", "OP_IMM", "MARK_AX", "MARK_STACK0"):
            if name in dp:
                print(f"    {name} = {dp[name]}")
    _bind_dims(dp)
    print(f"\n  Rebound dims: ALU_LO={DIM_RANGES['ALU_LO'][0]}, "
          f"OUTPUT_LO={DIM_RANGES['OUTPUT_LO'][0]}, "
          f"OP_ADD={DIM_SCALARS['OP_ADD']}")

    print("\n>>> Block types:")
    for i, block in enumerate(pn_model.blocks):
        ffn_type = type(block.ffn).__name__
        attn_type = type(block.attn).__name__
        # check for sparse W
        ffn_W_up = getattr(block.ffn, 'W_up', None)
        is_alu = 'AddSub' in ffn_type or 'ALU' in ffn_type or '_AddSub' in ffn_type
        print(f"    L{i}: attn={attn_type:25s} ffn={ffn_type:40s}"
              f"{' [ALU]' if is_alu else ''}")

    # Now forward both models on the truncated context (where AX marker
    # is the LAST token, so the next token to emit is AX byte 0 of the ADD result).
    device = next(pn_model.parameters()).device
    token_ids = torch.tensor([truncated_ctx], dtype=torch.long, device=device)
    target_pos = len(truncated_ctx) - 1  # AX marker position

    print(f"\n  target_pos (AX marker) = {target_pos}")
    print(f"  context[{target_pos}] = {truncated_ctx[target_pos]} "
          f"(REG_AX = {Token.REG_AX})")

    print("\n>>> Forward pure_neural mode with hooks...")
    pn_embed, pn_snaps = _hook_and_forward(pn_model, token_ids, target_pos)

    print(">>> Forward handler mode with hooks...")
    hm_embed, hm_snaps = _hook_and_forward(hm_model, token_ids, target_pos)

    # If they're the same model instance, the per-block residuals are
    # IDENTICAL. The difference must be at the RUNTIME layer (Python
    # overrides). Let's verify this hypothesis quickly.
    same_model = pn_model is hm_model
    if same_model:
        print("\n  NOTE: handler and pure_neural runners share the same model "
              "object. Per-block residuals will be IDENTICAL — leak must be "
              "in token emission (argmax of head logits) or in autoregressive "
              "feedback (prior STEP_END tokens differ between modes).")

    # Print pn summary in full
    _print_summary("pure_neural (truncated handler-context)", pn_embed, pn_snaps)

    # Now check: what token does the model want to emit?
    print("\n>>> Logit decode at AX marker position...")
    with torch.no_grad():
        logits = pn_model.forward(token_ids)
    top_token = int(logits[0, target_pos].argmax(-1).item())
    print(f"  Argmax token at AX marker: {top_token} (expect byte 42 = 0x2A)")
    top5 = logits[0, target_pos].topk(5)
    for i, (v, t) in enumerate(zip(top5.values, top5.indices)):
        print(f"    rank {i}: token {int(t.item()):3d} logit={float(v.item()):7.2f}")

    # Predict byte 0 of AX value. If it's 42 then handler-mode emits the
    # right byte. If it's 48 then we've reproduced the +6 leak directly
    # from the truncated handler context.
    if top_token == 42:
        print("\n  *** Argmax = 42 — the model emits the correct byte when "
              "fed the *handler-mode* context. The +6 leak must therefore "
              "live in pure_neural's *own* upstream tokens (IMM/PSH/IMM "
              "STEP_END emissions diverge), not in the ALU itself!")
    else:
        print(f"\n  *** Argmax = {top_token} — the model emits {top_token} "
              f"instead of 42 even on the well-formed handler context. The "
              f"+6 leak lives in the ALU FFN chain that runs on the AX "
              f"marker position.")

    # ---- Phase 2: also run pure_neural's actual autoregressive output and
    # capture the context where it diverges ----
    print("\n>>> Now running pure_neural runner to capture *its* token stream...")

    pn_runner._memory = {}
    pn_runner._mem_history = {}
    pn_runner._mem_access_order = []
    captured_pn = []
    real_pn = pn_runner._generate_next_cached

    def _pn_hook(gen_ctx):
        captured_pn.append(list(gen_ctx))
        return real_pn(gen_ctx)

    pn_runner._generate_next_cached = _pn_hook
    bc = _make_bc(prog)
    _, pn_exit = pn_runner.run(bc, b"", max_steps=30)
    pn_runner._generate_next_cached = real_pn
    print(f"  pure_neural exit code: {pn_exit}")

    if captured_pn:
        pn_longest = max(captured_pn, key=len)
        pn_ax_positions = _find_add_step_ax_marker_idx(None, pn_longest)
        print(f"  pure_neural REG_AX count: {len(pn_ax_positions)}")
        if len(pn_ax_positions) >= 4:
            pn_add_ax_idx = pn_ax_positions[3]
            pn_add_ax_value = _decode_register_value(pn_longest, pn_add_ax_idx)
            print(f"  pure_neural ADD-step AX value: {pn_add_ax_value} "
                  f"(handler-mode reference: 42)")

            # Now: compare pure_neural's truncated context (just before AX byte 0
            # of ADD step) with handler-mode's truncated context. Are the
            # prior STEP_END sequences identical?
            pn_trunc = pn_longest[:pn_add_ax_idx + 1]
            hm_trunc = truncated_ctx
            mismatches = []
            for i in range(min(len(pn_trunc), len(hm_trunc))):
                if pn_trunc[i] != hm_trunc[i]:
                    mismatches.append((i, pn_trunc[i], hm_trunc[i]))
            print(f"  Context mismatches (pn vs hm): {len(mismatches)}")
            for (i, p, h) in mismatches[:20]:
                print(f"    pos {i}: pn={p} hm={h}")
            if not mismatches and pn_add_ax_value != 42:
                print(f"\n  *** Contexts IDENTICAL but pure_neural gets "
                      f"{pn_add_ax_value} while handler gets 42!")
                print(f"  *** This means the AX byte 0 ARGMAX itself differs "
                      f"between the two modes' forward passes — but they share "
                      f"the same model... probably KV cache state differs.")


if __name__ == "__main__":
    main()
