"""ELIZA, running through a GENUINE HuggingFace ``transformers.Qwen2Model``.

This is the interactive deliverable: a user types a chat message, ELIZA reads it,
pattern-matches it, and replies — and every VM step of that pattern-match is ONE
real ``transformers.models.qwen2.Qwen2Model.forward`` (RoPE + RMSNorm + plain-
softmax GQA + SwiGLU). The transformer computes ALL the compute + control (the
per-byte compares, the branches, the loads through the §Memory CAM); the only
Python on the path is the standard autoregressive argmax-emit + the §Tool-Use I/O
boundary (READ pulls the user line off stdin, PRTF formats the reply) — exactly the
one op class the blogspec does NOT compute neurally.

It FUSES two pieces:
  * ``chat_eliza`` — the ELIZA chat program (``build_chat_min``: read a line,
    prefix-match each keyword, PRTF the reply) + the plain-python reference
    (``run_eliza_reference``) that is the BYTE-EXACT oracle.
  * ``qwen_full_vm`` — the FULL C4 VM fused into a real ``Qwen2Model.forward``
    (``build(subset=SUBSET_MEM_CMP)`` gives a genuine ``Qwen2Model(cfg)`` whose
    weights ARE the VM; ONE ``Qwen2Model.forward`` = ONE VM step).

Honest framing of "is this really Qwen2?"
------------------------------------------
The ARCHITECTURE is a real, unmodified ``transformers.Qwen2Model`` — constructed
via ``Qwen2Model(cfg)`` with a genuine ``Qwen2Config`` (head_dim=64, rope_theta=1e6,
GQA 14/2, SwiGLU MLP, RMSNorm). It is NOT ``from_pretrained``: the WEIGHTS are the
C4 VM's (baked by ``qwen_full_vm.build``), not the pretrained language-model weights.
So the claim is precise: **the model that runs ELIZA is a genuine Qwen2Model whose
weights compute the VM**. ``test_compute_is_in_the_qwen_forward`` (in
``test_qwen_full_vm.py``) proves the arithmetic is Qwen's own SwiGLU, not a Python
gadget (zeroing the Qwen MLPs annihilates the result).

The D-budget
------------
ELIZA needs string ops (LC reads the buffer, EQ compares, SI/LI + the §Memory CAM,
BZ/JMP branches) but NOT mul/div/mod — so it AVOIDS the 45 GB byte-table wall. The
minimal subset it needs is ``SUBSET_MEM_CMP`` (memory + cmp): a genuine Qwen2 config
with ``hidden_size=1152, intermediate_size=896, 10 layers``. That is +256 hidden
over stock Qwen2.5-0.5B (896) — i.e. a real Qwen2 that is slightly WIDER than the
stock 0.5B, still built via ``Qwen2Model(cfg)`` (same head_dim/rope/GQA/SwiGLU/
RMSNorm). See ``qwen_full_vm.fit_report()``: the wall only appears at +muldiv
(intermediate 160465 ≈ 45 GB), which ELIZA does not cross.

Which ELIZA
-----------
The transformer runs ONE ``Qwen2Model.forward`` over the WHOLE growing window PER VM
step, so a turn's wall-time grows with (steps × window). The full substring-scan
ELIZA (``chat_eliza.build_eliza``) is ~140-1200 steps/turn — too slow to run whole
through the forward. So we run ``build_chat_min`` — the SAME read/compute/write loop
with a CHEAP prefix match (compare the message's leading bytes to each keyword, no
O(len) scan): ~20-60 steps/turn, tractable end-to-end through the real forward. The
match is genuine (the model LOADs the user's bytes through the §Memory CAM and
branches on them); only the match SHAPE (prefix vs substring) is cheaper. Both the
Qwen path and the plain-python reference run the IDENTICAL bytecode → byte-exact.

Usage
-----
    OMP_NUM_THREADS=4 python -m c4_min.run_eliza_qwen_hf          # interactive
    OMP_NUM_THREADS=4 python -m c4_min.run_eliza_qwen_hf --demo   # scripted demo

Type a message and press enter; ELIZA replies. Keyword rules (prefix-matched, the
keyword must be at the START of your line): bye / hello / sad / happy / mother /
dream / yes / no. ``bye`` ends the session. Anything else → the fallback.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import isa
from . import blogspec_vocab as V
from . import nibble_filesys as FS
from . import qwen_full_vm as Q
from .blogspec_layout import NIB_PER_REG
from . import chat_eliza as E


# ===========================================================================
# Seed the ELIZA data segment (keyword + response string literals) into the
# §Memory KV store-log so an LC(addr) reads it back through the Qwen memory CAM.
#
# The fused Qwen VM's memory CAM content-addresses the STORE FRAMES (the KV write
# log), not a separate read-only data segment. So the ELIZA string table (which the
# c4 loader would place in a data segment) is laid into the store-log as ONE MEM
# frame per byte — address = the byte's data-segment address, value = the byte. A
# later LC(addr) then attends to exactly that frame and loads the byte, byte-exact
# to the reference (whose ``mem`` dict holds the same data segment).
# ===========================================================================
def _seed_data_seg(data_seg: Dict[int, int]) -> List[dict]:
    """One persistent MEM store frame per data-segment byte (addr -> byte)."""
    return [{"addr": a & 0xFF, "val": b & 0xFF} for a, b in sorted(data_seg.items())]


# ===========================================================================
# THE I/O-ENABLED FORWARD DRIVER — one VM step = one Qwen2Model.forward, PLUS the
# §Tool-Use I/O boundary (READ / PRTF) the compute-only ``qwen_full_vm.run_program``
# does not service.
#
# This mirrors ``qwen_full_vm.run_program`` (register CAM reads state; the SwiGLU
# MLPs compute the op; control-flow all inside the forward) but adds:
#   * a pre-seeded store-log carrying the ELIZA data segment (string literals),
#   * READ  — the driver pulls the user line off ``fio.stdin`` and lays each byte
#             into the store-log as its OWN MEM frame (so LC reads it back), AX=n,
#   * PRTF  — the driver reads the response C-string out of the memory view
#             (store-log + data seg), formats it, and appends to ``fio.stdout``.
# The file ops are the ONE class the transformer does NOT compute (§Tool Use Mode):
# for those rows the driver overrides the registers (PC += 1; AX := runner result),
# exactly as the pure-forward complete driver does.
# ===========================================================================
def run_eliza_turn_qwen(vm: Q.QwenFullVM, code: List[isa.Instr],
                        data_seg: Dict[int, int], fio: FS.FileOpState,
                        max_steps: int = 4000,
                        verbose: bool = False) -> Tuple[str, int]:
    """Run ONE ELIZA turn through the genuine ``Qwen2Model.forward``.

    ``fio`` carries the user message on its input-KV (READ fd 0) and collects the
    PRTF reply on ``fio.runner.stdout``. Returns ``(reply_text, n_steps)``.
    """
    QL, L = vm.QL, vm.QL.L

    reg_state = {"PC": 0, "AX": 0, "SP": E.SP_INIT_REF, "BP": E.SP_INIT_REF,
                 "STACK0": 0}
    # persistent memory KV log, pre-seeded with the ELIZA string table.
    store_log: List[dict] = _seed_data_seg(data_seg)
    # a Python-side view of memory for the §Tool-Use I/O marshalling ONLY (READ's
    # destination buffer, PRTF's format string). The transformer computes every
    # non-I/O op; this view exists purely to service the tool boundary, matching
    # ``run_eliza_reference``'s ``mem`` dict. Reads resolve store-log then data seg.
    def _mem_byte(addr: int) -> int:
        addr &= 0xFF
        best = None
        for i, st in enumerate(store_log):
            if (st["addr"] & 0xFF) == addr:
                best = st           # latest write wins (later frame overrides)
        if best is not None:
            return best["val"] & 0xFF
        return data_seg.get(addr, 0) & 0xFF

    def _read_cstring(addr: int, limit: int = 256) -> str:
        out = []
        for i in range(limit):
            b = _mem_byte(addr + i)
            if b == 0:
                break
            out.append(b)
        return bytes(out).decode("latin-1")

    cur_pc = 0
    for step in range(max_steps):
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        imm = code[cur_pc].imm if 0 <= cur_pc < len(code) else 0
        prev = dict(reg_state)

        # ---- §Tool-Use I/O boundary: READ / PRTF (not computed neurally) --------
        if op in FS.FILE_OPCODES:
            new_pc = cur_pc + 1
            if op == isa.READ:
                # READ(fd=0, buf=BUF, n=READ_N): the c4 marshalling pushed fd then
                # buf then set n=AX. The driver reads the line off the input-KV and
                # lays each byte into the store-log as its own MEM frame so a later
                # LC(BUF+k) reads it back through the Qwen memory CAM.
                n = prev["AX"] & 0xFF
                buf = E.BUF
                fd = FS.STDIN_FD
                call = FS.ToolCall(fio._new_id(), FS.TOOL_TYPE[op],
                                   {"fd": fd, "buf": buf, "n": n})
                resp = fio.runner.handle(call)
                data = resp.data or b""
                for i, byte in enumerate(data):
                    a = (buf + i) & 0xFF                # each READ byte as a MEM
                    store_log = [s for s in store_log   # frame (latest-write-wins)
                                 if (s["addr"] & 0xFF) != a]
                    store_log.append({"addr": a, "val": byte & 0xFF})
                fio.calls.append(call)
                new_ax = len(data) & 0xFF
            else:  # PRTF(fmt_ptr) -> n_written
                fmt_ptr = prev["STACK0"] & 0xFF        # the pushed format pointer
                fmt = _read_cstring(fmt_ptr)
                call = FS.ToolCall(fio._new_id(), FS.TOOL_TYPE[op],
                                   {"fmt": fmt, "fmt_ptr": fmt_ptr, "args": [],
                                    "strings": {}})
                resp = fio.runner.handle(call)
                fio.calls.append(call)
                new_ax = resp.result & 0xFF
            reg_state = {"PC": new_pc, "AX": new_ax, "SP": prev["SP"],
                         "BP": prev["BP"], "STACK0": prev["STACK0"]}
            if verbose:
                print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):5s} (I/O) "
                      f"-> pc={new_pc} ax={new_ax}")
            cur_pc = new_pc
            if cur_pc < 0 or cur_pc >= len(code):
                break
            continue

        # ---- COMPUTE / CONTROL: ONE genuine Qwen2Model.forward ------------------
        load_addr = None
        if op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF
        x = Q._build_stream_and_overlay(vm, code, reg_state, store_log, load_addr)
        state = Q._forward(vm, x)

        pc = Q._snap(state[L.PC_VAL])
        ax = Q._snap(state[L.AX_VAL]) & 0xFF
        sp = Q._snap(state[L.SP_VAL])
        bp = Q._snap(state[L.BP_VAL])
        stk = Q._snap(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5

        if op in (isa.SI, isa.SC):
            store_addr = Q._snap(state[L.STK_VAL]) & 0xFF
            # §Memory latest-write-wins as KV-log compaction (see qwen_full_vm.
            # run_program): a re-store supersedes the prior write, so each address
            # appears at most once — a clean one-frame content-address for the CAM.
            store_log = [s for s in store_log
                         if (s["addr"] & 0xFF) != store_addr]
            store_log.append({"addr": store_addr, "val": ax & 0xFF})

        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": stk}
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):5s} -> "
                  f"pc={pc} ax={ax} sp={sp} bp={bp} stk={stk} halt={halted}")
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(code):
            break

    return bytes(fio.runner.stdout).decode("latin-1"), step + 1


# ===========================================================================
# The interactive chat driver.
# ===========================================================================
@dataclass
class QwenElizaChat:
    """A built, ready-to-chat ELIZA on a genuine ``Qwen2Model``.

    ``vm`` is the constructed ``Qwen2Model(cfg)`` (weights = the VM); ``eliza`` the
    compiled bytecode + data segment. ``reply(msg)`` runs ONE turn through the real
    ``Qwen2Model.forward`` and (by default) cross-checks it byte-exact against the
    plain-python reference on the same bytecode."""
    vm: Q.QwenFullVM
    eliza: E.Eliza
    max_steps: int = 4000

    def reply(self, message: str, check_reference: bool = True,
              verbose: bool = False) -> Tuple[str, int, Optional[bool]]:
        """Run ONE turn. Returns ``(reply, n_steps, byte_exact_vs_reference)``.

        The message enters via the input-KV (READ fd 0); the reply comes out via
        PRTF. ``byte_exact_vs_reference`` is ``None`` if ``check_reference`` is off.
        """
        fio = E._fresh_fio(message)
        reply, steps = run_eliza_turn_qwen(
            self.vm, self.eliza.code, self.eliza.data_seg, fio,
            max_steps=self.max_steps, verbose=verbose)
        ok: Optional[bool] = None
        if check_reference:
            ref = E.chat_turn_ref(self.eliza, message)
            ok = (ref == reply)
        return reply, steps, ok


# A COMPACT default keyword set for the interactive session. Fewer, short rules =>
# a smaller code + data segment => a smaller per-forward token window => a snappy
# ~5-10 s/turn on CPU. The FULL 8-rule ELIZA (``--full``) is the classic table but
# ~30 s/turn (a ~240-token window). Both run byte-exact through the real forward;
# this is purely a wall-time / conversational-breadth trade (the per-forward window
# grows with the code + data-segment size, and the whole window is re-forwarded
# every VM step).
FAST_RULES: List[Tuple[str, str]] = [
    ("bye",   "BYE. TAKE CARE.\n"),
    ("hi",    "HELLO. HOW ARE YOU?\n"),
    ("sad",   "WHY ARE YOU SAD?\n"),
    ("happy", "WHAT MAKES YOU HAPPY?\n"),
    ("yes",   "YOU SEEM SURE.\n"),
    ("no",    "WHY NOT?\n"),
]


def build_qwen_eliza(eliza: Optional[E.Eliza] = None,
                     full: bool = False,
                     verbose: bool = True) -> QwenElizaChat:
    """Construct the genuine ``Qwen2Model`` + compile the ELIZA bytecode.

    Uses ``SUBSET_MEM_CMP`` — the minimal op subset ELIZA needs (memory + cmp; no
    mul/div/mod, so no 45 GB byte-table wall). The result is a real
    ``Qwen2Model(cfg)`` (hidden 1152-1536, intermediate 896, 10 layers) whose
    weights compute the VM. ``full`` selects the classic 8-rule ELIZA (wider window,
    ~30 s/turn) vs the compact default (~5-10 s/turn); both are byte-exact.
    """
    if eliza is None:
        eliza = E.build_chat_min() if full else E.build_chat_min(rules=FAST_RULES)
    code_size = len(eliza.code) + 2
    if verbose:
        print(f"[build] compiling a genuine Qwen2Model (subset=mem+cmp, "
              f"code_size={code_size}) ...", flush=True)
    t0 = time.time()
    vm = Q.build(code_size=code_size, subset=Q.SUBSET_MEM_CMP)
    if verbose:
        cls = type(vm.qmodel).__module__ + "." + type(vm.qmodel).__name__
        print(f"[build] built {cls}: hidden_size={vm.hidden_size} "
              f"intermediate_size={vm.intermediate_size} n_layers={vm.n_layers} "
              f"query_heads={vm.arch.num_attention_heads} "
              f"head_dim={vm.arch.head_dim} rope_theta={int(Q.ROPE_THETA)} "
              f"(constructed via Qwen2Model(cfg), weights = the C4 VM) "
              f"in {time.time()-t0:.1f}s", flush=True)
        print(f"[build] fits stock Qwen2.5-0.5B budget: {vm.fits_stock} "
              f"(mem+cmp is +256 hidden over stock 896; the mul/div/mod byte-table "
              f"wall at intermediate 160465 ≈ 45 GB is NOT crossed)", flush=True)
    return QwenElizaChat(vm=vm, eliza=eliza)


def _greeting(chat: QwenElizaChat) -> str:
    kws = " / ".join(kw.decode("latin-1") for kw, _ in chat.eliza.rules)
    return ("ELIZA (running through a genuine transformers.Qwen2Model.forward).\n"
            f"Talk to me. Keywords (must START your line, prefix match): {kws}.\n"
            "Anything else -> a generic reply. Type 'bye' to end.\n")


def interactive(chat: Optional[QwenElizaChat] = None,
                full: bool = False, check_reference: bool = True) -> None:
    """Read a line of user input, run ELIZA through the Qwen forward, print the
    reply, and loop. Reads stdin so ``python -m c4_min.run_eliza_qwen_hf`` is a
    real interactive terminal chat."""
    chat = chat or build_qwen_eliza(full=full)
    print("\n" + _greeting(chat))
    while True:
        try:
            sys.stdout.write("you> ")
            sys.stdout.flush()
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "":                       # EOF (Ctrl-D)
            break
        msg = line.rstrip("\n")
        if not msg:
            continue
        t0 = time.time()
        reply, steps, ok = chat.reply(msg, check_reference=check_reference)
        dt = time.time() - t0
        sys.stdout.write("eliza> " + reply)
        if not reply.endswith("\n"):
            sys.stdout.write("\n")
        tag = "" if ok is None else (" [byte-exact vs reference: "
                                     f"{'YES' if ok else 'NO'}]")
        sys.stdout.write(f"        ({steps} Qwen2Model.forward steps, "
                         f"{dt:.1f}s{tag})\n\n")
        sys.stdout.flush()
        # stop after the 'bye' rule fires (its reply is the terse BYE line).
        if reply.startswith("BYE"):
            break
    print("[session ended]")


def demo(messages: Optional[List[str]] = None,
         full: bool = False, check_reference: bool = True) -> None:
    """A scripted multi-turn exchange through the genuine ``Qwen2Model.forward``,
    each turn byte-exact-checked against the plain-python reference."""
    if messages is None:
        messages = (["hello there", "sad today", "mother knows", "yes indeed",
                     "flibberty jibbet", "bye now"] if full else
                    ["hi there", "sad today", "happy now", "yes indeed",
                     "flibberty jibbet", "bye now"])
    chat = build_qwen_eliza(full=full)
    print("\n=== ELIZA through a genuine transformers.Qwen2Model.forward ===\n")
    total_steps = 0
    total_t = 0.0
    all_ok = True
    for msg in messages:
        t0 = time.time()
        reply, steps, ok = chat.reply(msg, check_reference=check_reference)
        dt = time.time() - t0
        total_steps += steps
        total_t += dt
        all_ok = all_ok and (ok is not False)
        print(f"you>   {msg}")
        print(f"eliza> {reply}", end="" if reply.endswith("\n") else "\n")
        tag = "" if ok is None else f" byte-exact-vs-reference={'YES' if ok else 'NO'}"
        print(f"       [{steps} Qwen2Model.forward steps, {dt:.1f}s{tag}]\n")
    print(f"=== {len(messages)} turns, {total_steps} total Qwen2Model.forward "
          f"steps, {total_t:.1f}s wall "
          f"({total_t/max(1,total_steps):.2f}s/step); "
          f"all byte-exact vs reference: {'YES' if all_ok else 'NO'} ===")


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    full = "--full" in argv
    check = "--no-check" not in argv
    if "--demo" in argv:
        demo(full=full, check_reference=check)
    else:
        interactive(full=full, check_reference=check)


if __name__ == "__main__":
    main()
