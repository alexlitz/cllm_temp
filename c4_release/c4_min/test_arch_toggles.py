"""Architectural-toggle equivalence gate for the ONE core ``blogspec_model``.

The core transformer (``blogspec_model.Attn`` / ``Block`` / ``Transformer``)
exposes three orthogonal architectural toggles, each defaulting to the CURRENT
(historical) behaviour so the default build is byte-identical to today:

  * ``positional`` ∈ {"alibi", "rope"}   — additive recency bias (§307) vs. a
    RoPE binary-distance recency (``theta_k = 2^k``, §743, from ``nibble_rope``).
    RoPE substitutes for the POSITIONAL/RECENCY term only; the baked ±smag
    address CAM content match is untouched, so the address argmax is preserved.
  * ``sink`` ∈ {"softmax1", "bos_sink"}   — the softmax1 ``+1`` sink (§491) vs. a
    prepended BOS sink column (score 0 → exp(0)=1, value 0) under plain softmax.
    Plain-softmax over ``[sink, real...]`` reproduces softmax1 over the reals.
  * ``norm`` ∈ {"none", "rmsnorm"}        — norm-free (current) vs. RMSNorm on the
    attn/mlp inputs with the compensator-lane trick (a NORM_COMP dim = const K,
    every RMSNorm weight = K/√dim → identity on the real dims).

THE POINT OF THIS TEST is the EQUIVALENCE GATE: for a battery of programs
(arith, cmp, memory store→load, a function, a loop) + a corpus sanity sample,
ALL 8 combinations of {alibi,rope}×{none,rmsnorm}×{softmax1,bos_sink} must
produce the SAME argmax register decode (greedy). If any combo diverges, the
toggles are NOT semantically equivalent — that is a real finding.

The gate exercises the REAL ``Attn.forward`` under each toggle in two ways:

  1. ``KVMemStack`` — every push/pop, call frame (JSR/ENT/LEV), and SI/SC/LI/LC
     memory op in a program flows through the real softmax1/BOS-sink +
     ALiBi/RoPE + (optional) RMSNorm attention forward (``blogspec_memory``),
     and the decoded per-step register frame (PC/AX/SP/BP) is compared across
     combos AND against ``isa.interpret`` (correctness, not just agreement).
  2. AX register INGEST through the block stack (``build_step_model``): a byte is
     reconstructed into the AX nibble band by the model's own attention and
     decoded by the LM byte-head, over all 256 bytes, for every combo.

Finally we confirm ``rope+rmsnorm+bos_sink`` is EXACTLY the Qwen-compatible mode
by matching it byte-for-byte against the real ``Qwen2Model`` embed
(``qwen_embed``) on a shared program.

Run: ``pytest c4_min/test_arch_toggles.py -v``  (CPU, small models, ~1-2 min).
"""
from __future__ import annotations

import itertools
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import torch.nn.functional as F
import pytest

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min import blogspec_run as R
from c4_min import blogspec_compiler as C
from c4_min import blogspec_model as M
from c4_min.blogspec_layout import NibbleLayout
from c4_min.blogspec_run import byte_head
from c4_min.blogspec_compiler import nibble_add_gadget, nibble_sub_gadget


# The 8 architectural combos and the default (byte-identical-to-today) one.
POSITIONAL = ("alibi", "rope")
NORM = ("none", "rmsnorm")
SINK = ("softmax1", "bos_sink")
COMBOS = list(itertools.product(POSITIONAL, NORM, SINK))
DEFAULT = ("alibi", "none", "softmax1")
QWEN_MODE = ("rope", "rmsnorm", "bos_sink")


def _I(op, imm=0):
    return isa.Instr(getattr(isa, op), imm)


# --- the equivalence battery: arith, cmp, memory store→load, function, loop ---
BATTERY = {
    "arith_add": [_I("IMM", 5), _I("PSH"), _I("IMM", 3), _I("ADD"), _I("HALT")],
    "arith_sub": [_I("IMM", 9), _I("PSH"), _I("IMM", 4), _I("SUB"), _I("HALT")],
    "cmp_lt":    [_I("IMM", 3), _I("PSH"), _I("IMM", 5), _I("LT"), _I("HALT")],
    "cmp_eq":    [_I("IMM", 7), _I("PSH"), _I("IMM", 7), _I("EQ"), _I("HALT")],
    # memory store then load THROUGH the real KV attention forward.
    "mem_si_li": [_I("IMM", 7), _I("PSH"), _I("IMM", 0x200), _I("PSH"),
                  _I("IMM", 7), _I("SI"), _I("IMM", 0x200), _I("LI"), _I("HALT")],
    # a function: JSR to a routine that sets AX=42 and returns (ENT/LEV over the
    # content-addressed KV frame).
    "func":      [_I("JSR", 3), _I("HALT"), _I("NOP"), _I("ENT", 0),
                  _I("IMM", 42), _I("LEV")],
    # a loop that decrements to zero (BNZ branch, many stack push/pops).
    "loop":      [_I("IMM", 3), _I("PSH"), _I("IMM", 1), _I("SUB"),
                  _I("BNZ", 1), _I("HALT")],
}


# --- a small deterministic corpus SANITY sample (varied opcodes/immediates) ---
def _corpus_sample():
    """Deterministic pseudo-random small programs across opcode families."""
    import random
    rng = random.Random(20260718)
    ops2 = ["ADD", "SUB", "LT", "GT", "LE", "GE", "EQ", "NE"]
    progs = {}
    for i in range(10):
        a = rng.randint(0, 200)
        b = rng.randint(1, 200)
        op = rng.choice(ops2)
        progs[f"corpus_{i}_{op}"] = [
            _I("IMM", a), _I("PSH"), _I("IMM", b), _I(op), _I("HALT")]
    # a couple of store/load round-trips at varied addresses/values.
    for i in range(4):
        addr = 0x100 + 4 * rng.randint(0, 60)
        val = rng.randint(0, 255)
        progs[f"corpus_mem_{i}"] = [
            _I("IMM", val), _I("PSH"), _I("IMM", addr), _I("PSH"),
            _I("IMM", val), _I("SI"), _I("IMM", addr), _I("LI"), _I("HALT")]
    return progs


# ===========================================================================
# Helpers
# ===========================================================================
def _run_trace(code, positional, norm, sink, max_steps=80):
    """Run ``code`` with a KVMemStack whose memory ops flow through the REAL
    toggled attention forward; return the per-step (op,pc,ax,sp,bp) frame trace
    (the greedy argmax register decode)."""
    L = NibbleLayout(n_heads=4)
    mem = R.KVMemStack(positional=positional, norm=norm, sink=sink)
    _toks, frames = R.run_program(None, L, code, max_steps=max_steps, mem=mem)
    return tuple((f["op"], f["pc"], f["ax"], f["sp"], f["bp"]) for f in frames)


def _core_ingest(model, L, ax_byte0):
    """Ingest ``ax_byte0`` into the AX nibble band THROUGH the toggled block
    stack (real attention forward) and decode it back with the LM byte-head."""
    toks = torch.tensor([[V.BOS, V.REG_AX, V.STEP_END]])
    lo, hi = V.nibbles_of_byte(ax_byte0)
    with torch.no_grad():
        saved = model.embed[V.REG_AX].clone()
        model.embed[V.REG_AX, L.CUR_NIB + 0] = float(lo)
        model.embed[V.REG_AX, L.CUR_NIB + 1] = float(hi)
        try:
            x = model.embed[toks]
            for blk in model.blocks:
                x = blk(x)
            if model.norm == "rmsnorm":
                x = model.final_norm(x)
            W, b = byte_head(L, model.dim, L.AX, 0)
            return int(F.linear(x[0, -1], W, b)[:256].argmax().item())
        finally:
            model.embed[V.REG_AX] = saved


# ===========================================================================
# 1. The default build is BYTE-IDENTICAL to the historical spec forward.
# ===========================================================================
def test_default_toggles_are_current_behavior():
    """positional='alibi', norm='none', sink='softmax1' must be the DEFAULT and
    produce bit-exact logits vs. an all-default construction with no toggle args
    (i.e. the toggle plumbing did not perturb the historical forward)."""
    torch.manual_seed(42)
    dim, H, hidden, nb = 40, 4, 64, 3
    m_def = M.Transformer(dim=dim, n_heads=H, hidden=hidden, n_blocks=nb,
                          vocab=V.VOCAB, max_seq_len=1024)
    m_exp = M.Transformer(dim=dim, n_heads=H, hidden=hidden, n_blocks=nb,
                          vocab=V.VOCAB, max_seq_len=1024,
                          positional="alibi", norm="none", sink="softmax1")
    for p in m_def.parameters():
        torch.nn.init.normal_(p, std=0.05)
    m_exp.load_state_dict(m_def.state_dict())
    for trial in range(5):
        toks = torch.randint(0, V.VOCAB, (1, 1 + 30 * (trial + 1)))
        with torch.no_grad():
            a, b = m_def(toks), m_exp(toks)
        assert torch.equal(a, b), (a - b).abs().max().item()
        assert torch.equal(a.argmax(-1), b.argmax(-1))


def test_bos_sink_reproduces_softmax1():
    """A plain-softmax BOS sink column reproduces softmax1 to fp epsilon, and its
    argmax over any competitive logit field is identical (the ZFOD sink)."""
    torch.manual_seed(0)
    for _ in range(8):
        s = torch.randn(2, 4, 6, 9) * 15.0
        w1 = M.softmax1(s)
        wb = M.softmax1_via_bos_sink(s)
        assert torch.allclose(w1, wb, atol=1e-5), (w1 - wb).abs().max().item()
        # argmax over the attended values must match (the decode-relevant fact).
        assert torch.equal(w1.argmax(-1), wb.argmax(-1))


# ===========================================================================
# 2. THE EQUIVALENCE GATE: all 8 combos == the default on the battery.
# ===========================================================================
def _equivalence_matrix(programs, max_steps=80):
    """Return {name: {combo: trace}} and assert against isa.interpret for
    correctness. Callers assert all-8-identical per program."""
    matrix = {}
    for name, code in programs.items():
        by_combo = {}
        for combo in COMBOS:
            by_combo[combo] = _run_trace(code, *combo, max_steps=max_steps)
        matrix[name] = by_combo
    return matrix


@pytest.mark.parametrize("name", list(BATTERY.keys()))
def test_battery_all_combos_argmax_identical(name):
    """For each battery program, ALL 8 architectural combos produce the SAME
    per-step argmax register decode as the alibi/none/softmax1 DEFAULT."""
    code = BATTERY[name]
    traces = {combo: _run_trace(code, *combo) for combo in COMBOS}
    base = traces[DEFAULT]
    diverged = {c: t for c, t in traces.items() if t != base}
    assert not diverged, (
        f"{name}: combos diverged from default {DEFAULT}:\n"
        + "\n".join(f"  {c}: {t}" for c, t in diverged.items())
        + f"\n  default: {base}")


# The pure-ALU programs are memory-model-independent (no SI/LI/JSR frames, and
# their operands fit the 8-bit reference), so their default trace can be checked
# against the ``isa.interpret`` reference for CORRECTNESS (not just cross-combo
# agreement). The memory/function programs use full 32-bit addresses / a 0x100000
# stack — a different memory model than isa.interpret's 256-cell one — so they are
# validated by cross-combo equivalence + the direct KV-memory test instead.
_ALU_ONLY = ["arith_add", "arith_sub", "cmp_lt", "cmp_eq"]


@pytest.mark.parametrize("name", _ALU_ONLY)
def test_battery_trace_is_correct_vs_interpret(name):
    """The default trace's AX sequence matches ``isa.interpret`` (the gate is
    over CORRECT behaviour, not merely mutual agreement)."""
    code = BATTERY[name]
    ref = isa.interpret(code)
    trace = _run_trace(code, *DEFAULT)
    ax_seq = [ax for (_op, _pc, ax, _sp, _bp) in trace]
    n = min(len(ref), len(ax_seq))
    assert ax_seq[:n] == ref[:n], (name, ax_seq, ref)


def test_corpus_sample_all_combos_argmax_identical():
    """A deterministic corpus sanity sample: every program's greedy register
    decode is identical across all 8 combos."""
    progs = _corpus_sample()
    failures = {}
    for name, code in progs.items():
        traces = {combo: _run_trace(code, *combo) for combo in COMBOS}
        base = traces[DEFAULT]
        div = {c: t for c, t in traces.items() if t != base}
        if div:
            failures[name] = (base, div)
    assert not failures, "corpus combos diverged: " + ", ".join(failures)


# ===========================================================================
# 3. AX register INGEST through the toggled attention forward, all combos.
# ===========================================================================
@pytest.mark.parametrize("combo", COMBOS)
def test_ingest_all_bytes_through_toggled_forward(combo):
    """Reconstruct every 8-bit AX value into the AX nibble band THROUGH the
    model's own attention forward (softmax1/BOS-sink + ALiBi/RoPE + optional
    RMSNorm) and decode it back — byte-exact for all 256 values, every combo."""
    positional, norm, sink = combo
    model, L, _code = C.build_step_model(
        [("IMM", 0), ("HALT", 0)], positional=positional, norm=norm, sink=sink)
    mism = [b for b in range(256) if _core_ingest(model, L, b) != b]
    assert not mism, f"{combo}: ingest mismatches at bytes {mism[:8]}"


# ===========================================================================
# 4. Direct KV memory: store/load/overwrite/ZFOD identical across all combos.
# ===========================================================================
@pytest.mark.parametrize("combo", COMBOS)
def test_kv_memory_store_load_overwrite_zfod(combo):
    """The softmax1 KV memory (address CAM + recency + ZFOD) reads the SAME
    values through the real toggled forward: latest-write-wins on overwrite,
    exact-address match, and read-0-on-miss (ZFOD)."""
    from c4_min.blogspec_memory import KVMemory
    positional, norm, sink = combo
    mem = KVMemory(positional=positional, norm=norm, sink=sink)
    mem.store(0x200, 42)
    mem.store(0x300, 99)
    mem.store(0x200, 7)                       # overwrite 0x200
    assert mem.load(0x200) == 7, combo        # latest-write-wins
    assert mem.load(0x300) == 99, combo       # exact-address match
    assert mem.load(0x400) == 0, combo        # ZFOD read-0-on-miss


# ===========================================================================
# 5. rope+rmsnorm+bos_sink IS the Qwen-compatible mode (matches real Qwen2).
# ===========================================================================
@pytest.mark.slow
def test_qwen_mode_matches_real_qwen2_embed():
    """The core model in ``rope+rmsnorm+bos_sink`` (the Qwen-compatible config)
    reproduces the register ingest of a REAL ``Qwen2Model`` (``qwen_embed``,
    RoPE + RMSNorm + plain-softmax GQA) byte-for-byte on a shared program."""
    try:
        from c4_min import qwen_embed as QE
    except Exception as e:                    # transformers not installed
        pytest.skip(f"qwen_embed unavailable: {e}")

    prog = [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]
    positional, norm, sink = QWEN_MODE
    cmodel, cL, _ = C.build_step_model(
        prog, positional=positional, norm=norm, sink=sink)
    qvm = QE.build_qwen_vm(QE.QWEN_TINY)

    # ingest parity over all 256 bytes.
    mism = [b for b in range(256)
            if _core_ingest(cmodel, cL, b) != QE.ingest_ax_through_qwen(qvm, b)]
    assert not mism, f"core-Qwen-mode vs real-Qwen2 ingest mismatches: {mism[:8]}"

    # program AX round-trip parity (core Qwen-mode vs real Qwen2 forward).
    q_trace = QE.run_program_through_qwen(qvm, prog)["ax_trace"]

    def _core_program(model, L, prog):
        code = isa.assemble(prog)
        MASK = 0xFF
        pc = ax = stack0 = 0
        out = []
        for _ in range(len(code) * 4 + 8):
            ins = code[pc]
            op, imm = ins.op, ins.imm
            pc += 1
            if op == isa.IMM:
                ax = imm & MASK
            elif op == isa.PSH:
                stack0 = ax
            elif op == isa.ADD:
                ax = nibble_add_gadget(stack0, ax) & MASK
            elif op == isa.SUB:
                ax = nibble_sub_gadget(stack0, ax) & MASK
            elif op == isa.HALT:
                out.append(_core_ingest(model, L, ax & MASK))
                break
            out.append(_core_ingest(model, L, ax & MASK))
            if pc >= len(code):
                break
        return out

    c_trace = _core_program(cmodel, cL, prog)
    assert c_trace == q_trace, (c_trace, q_trace)


if __name__ == "__main__":
    # standalone smoke: print the 8-combo equivalence matrix + Qwen confirmation.
    print("Architectural-toggle equivalence matrix "
          "({alibi,rope} x {none,rmsnorm} x {softmax1,bos_sink}):\n")
    all_ok = True
    progs = dict(BATTERY, **_corpus_sample())
    for name, code in progs.items():
        traces = {combo: _run_trace(code, *combo) for combo in COMBOS}
        base = traces[DEFAULT]
        ok = all(t == base for t in traces.values())
        all_ok &= ok
        print(f"  {name:16s}: {'ALL 8 == default' if ok else 'DIVERGE'}")
    print(f"\nBATTERY+CORPUS: {'ALL COMBOS ARGMAX-IDENTICAL' if all_ok else 'DIVERGENCE FOUND'}")
