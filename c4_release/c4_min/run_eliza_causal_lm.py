"""ELIZA driven by the GENUINE HuggingFace generation stack — ``model.generate`` +
``TextIteratorStreamer`` + chat template + ``reasoning_content`` / ``<think>``.

This is the interactive deliverable of ``vm_causal_lm``: the fused Qwen C4 VM is a
real ``C4VMForCausalLM`` (``PreTrainedModel`` + ``GenerationMixin``), so an ELIZA
turn is driven by the NATIVE HF stack, not a bespoke Python driver:

  * ``model.generate(do_sample=False, ...)`` autoregressively RUNS the VM (each
    frame token is greedy argmax over the ~267 frame-token vocab);
  * a ``ToolCallStoppingCriteria`` halts ``generate()`` at a §Tool-Use I/O op
    (READ / PRTF); the AGENTIC loop (``run_agentic_generate``) services it (READ
    pulls the user line into the memory KV log; PRTF captures the reply byte(s))
    and RESUMES ``generate()`` — the ONE op class the blogspec does not compute
    neurally;
  * a ``C4VMTokenizer`` (chat template + reasoning parser) exposes the VM's
    register frames as ``<think>…</think>`` in the raw stream AND as
    ``reasoning_content`` (structured), and the PRTF bytes as the assistant OUTPUT;
  * ``TextIteratorStreamer`` streams the frame tokens live.

The reply is byte-exact vs the plain-python ELIZA reference (``chat_eliza``).
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

from . import isa
from . import blogspec_vocab as V
from . import qwen_full_vm as Q
from . import nibble_filesys as FS
from . import chat_eliza as E
from . import run_eliza_qwen_hf as HF
from .vm_causal_lm import (
    C4VMForCausalLM, C4Program, run_agentic_generate, build_c4_causal_lm,
)
from .vm_tokenizer import C4VMTokenizer


# ===========================================================================
# The §Tool-Use I/O service for the agentic loop.  Mirrors run_eliza_qwen_hf's
# I/O boundary exactly (READ lays the input line into the memory KV log so a
# later LC reads it through the Qwen memory CAM; PRTF reads the response C-string
# out of the memory view and captures the output byte(s)).
# ===========================================================================
def make_eliza_tool_service(fio: FS.FileOpState, data_seg: Dict[int, int]):
    """Return a ``tool_service(op, reg, program)`` closure for ``run_agentic_generate``.

    It services one READ / PRTF against ``fio`` and ``data_seg``, returning
    ``(next_reg, store_additions, visible_bytes)``.
    """
    def _mem_byte(store_log: List[dict], addr: int) -> int:
        addr &= 0xFF
        best = None
        for st in store_log:
            if (st["addr"] & 0xFF) == addr:
                best = st                       # latest write wins
        if best is not None:
            return best["val"] & 0xFF
        return data_seg.get(addr, 0) & 0xFF

    def _read_cstring(store_log: List[dict], addr: int, limit: int = 256) -> str:
        out = []
        for i in range(limit):
            b = _mem_byte(store_log, addr + i)
            if b == 0:
                break
            out.append(b)
        return bytes(out).decode("latin-1")

    def tool_service(op: int, reg: dict, program: C4Program):
        store_log = program.store_log
        new_pc = None
        # the file op executes at reg["PC"]; find its PC to advance past it.
        # (reg is the state ENTERING the op; PC advances by 1 as the ISA specifies.)
        cur_pc = reg["PC"]
        store_add: List[dict] = []
        visible: List[int] = []
        if op == isa.READ:
            n = reg["AX"] & 0xFF
            buf = E.BUF
            call = FS.ToolCall(fio._new_id(), FS.TOOL_TYPE[op],
                               {"fd": FS.STDIN_FD, "buf": buf, "n": n})
            resp = fio.runner.handle(call)
            data = resp.data or b""
            for i, byte in enumerate(data):
                store_add.append({"addr": (buf + i) & 0xFF, "val": byte & 0xFF})
            fio.calls.append(call)
            new_ax = len(data) & 0xFF
        else:  # PRTF
            fmt_ptr = reg["STACK0"] & 0xFF
            fmt = _read_cstring(store_log, fmt_ptr)
            call = FS.ToolCall(fio._new_id(), FS.TOOL_TYPE[op],
                               {"fmt": fmt, "fmt_ptr": fmt_ptr, "args": [],
                                "strings": {}})
            before = len(fio.runner.stdout)
            resp = fio.runner.handle(call)
            fio.calls.append(call)
            visible = list(fio.runner.stdout[before:])   # the PRTF OUTPUT bytes
            new_ax = resp.result & 0xFF
        next_reg = {"PC": cur_pc + 1, "AX": new_ax, "SP": reg["SP"],
                    "BP": reg["BP"], "STACK0": reg["STACK0"]}
        return next_reg, store_add, bytes(visible)

    return tool_service


# ===========================================================================
# One ELIZA turn through the NATIVE HF stack.
# ===========================================================================
def eliza_turn_causal_lm(eliza: E.Eliza, message: str, verbose: bool = False,
                         ) -> Tuple[str, str, dict]:
    """Run ONE ELIZA turn for ``message`` via ``model.generate`` + the agentic loop.

    Returns ``(reply_text, thinking_text, meta)`` where ``reply`` is the OUTSIDE-
    think OUTPUT (byte-exact vs the reference), ``thinking`` is the register-frame
    reasoning, and ``meta`` carries the raw tokens + tool count.
    """
    code_size = len(eliza.code) + 2
    model = build_c4_causal_lm(eliza.code, subset=Q.SUBSET_MEM_CMP,
                               store_log=HF._seed_data_seg(eliza.data_seg),
                               code_size=code_size, max_steps=4000)
    fio = E._fresh_fio(message)
    service = make_eliza_tool_service(fio, dict(eliza.data_seg))
    result = run_agentic_generate(model, service, max_new_tokens=8192)
    reply = result["visible"].decode("latin-1")
    tok = C4VMTokenizer()
    thinking = tok.reasoning_content(result["tokens"])
    meta = {"tokens": result["tokens"], "n_tool": result["n_tool"],
            "n_visible": len(result["visible"])}
    return reply, thinking, meta


def chat_turn_openai(eliza: E.Eliza, message: str) -> Dict[str, str]:
    """One ELIZA turn as an OpenAI / R1-style assistant message:
        {"role": "assistant", "reasoning_content": <register-frame reasoning>,
         "content": <visible reply>}
    — the exact shape a reasoning client (``choice.message.reasoning_content``)
    consumes.  Uses the chat template for the prompt framing and the native
    ``model.generate`` + agentic loop for the run."""
    tok = C4VMTokenizer()
    # the system/user prompt via the chat template (enable_thinking convention).
    _prompt = tok.apply_chat_template(
        [{"role": "system", "content": "You are ELIZA."},
         {"role": "user", "content": message}],
        tokenize=False, enable_thinking=True)
    reply, thinking, meta = eliza_turn_causal_lm(eliza, message)
    return {"role": "assistant", "reasoning_content": thinking, "content": reply,
            "_prompt": _prompt}


def stream_chat_turn(eliza: E.Eliza, message: str, out=sys.stdout) -> Dict[str, str]:
    """Stream ONE ELIZA turn LIVE via a genuine ``transformers.TextIteratorStreamer``
    over ``model.generate`` (the whole agentic turn runs in a worker thread; the
    streamer yields the ``<think>`` reasoning + visible reply as they are emitted).
    Returns the OpenAI/R1-style assistant message.
    """
    import threading
    from transformers import TextIteratorStreamer

    code_size = len(eliza.code) + 2
    model = build_c4_causal_lm(eliza.code, subset=Q.SUBSET_MEM_CMP,
                               store_log=HF._seed_data_seg(eliza.data_seg),
                               code_size=code_size, max_steps=4000)
    fio = E._fresh_fio(message)
    service = make_eliza_tool_service(fio, dict(eliza.data_seg))
    tok = C4VMTokenizer()

    out.write("[chat template + enable_thinking]\n")
    out.write(tok.apply_chat_template(
        [{"role": "system", "content": "You are ELIZA."},
         {"role": "user", "content": message}], tokenize=False, enable_thinking=True))
    out.write("\n[live stream via transformers.TextIteratorStreamer]\n")
    out.flush()

    # A genuine TextIteratorStreamer issues an end-of-stream after each generate()
    # call, so it cannot span the agentic loop's PAUSED segments (each tool boundary
    # ends a segment).  We therefore attach a FRESH streamer to each segment and
    # consume it live — the whole turn streams, segment by segment, through the real
    # transformers streamer.  Each segment's generate() runs in a worker thread while
    # the main thread drains its streamer queue.
    import torch
    from transformers import StoppingCriteriaList
    from .vm_causal_lm import (HaltStoppingCriteria, ToolCallStoppingCriteria,
                               _step_frame_tokens)

    stop = StoppingCriteriaList([HaltStoppingCriteria(), ToolCallStoppingCriteria(model)])
    ids: List[int] = [V.BOS]
    visible = bytearray()
    while True:
        model.pending_tool = None
        streamer = TextIteratorStreamer(tok, skip_prompt=True)
        box: Dict[str, object] = {}

        def _seg(ids_now):
            box["out"] = model.generate(
                input_ids=torch.tensor([ids_now]), do_sample=False,
                max_new_tokens=8192, streamer=streamer,
                stopping_criteria=stop, pad_token_id=V.HALT)

        th = threading.Thread(target=_seg, args=(list(ids),))
        th.start()
        for piece in streamer:
            out.write(piece); out.flush()
        th.join()
        ids = box["out"][0].tolist()
        if model.pending_tool is None:
            break
        tool = model.pending_tool
        while ids and ids[-1] == V.HALT:
            ids.pop()
        next_reg, store_add, vis = service(tool["op"], tool["reg"], model.program)
        for st in store_add:
            a = st["addr"] & 0xFF
            model.program.store_log = [s for s in model.program.store_log
                                       if (s["addr"] & 0xFF) != a]
            model.program.store_log.append({"addr": a, "val": st["val"] & 0xFF})
        for byte in vis:
            ids += [V.THINK_END, byte & 0xFF, V.THINK_START]
            visible.append(byte & 0xFF)
            out.write(chr(byte)); out.flush()   # visible OUTPUT byte, streamed live
        ids += _step_frame_tokens(next_reg["PC"], next_reg["AX"], next_reg["SP"],
                                  next_reg["BP"], 0, next_reg["STACK0"])
        model._frame_memo.clear()
        model.pending_tool = None
    out.write("\n")
    return {"role": "assistant",
            "reasoning_content": tok.reasoning_content(ids),
            "content": bytes(visible).decode("latin-1")}


def demo(messages: List[str] = None, full: bool = False) -> None:
    """A scripted ELIZA exchange through the native HF ``generate()`` + agentic loop,
    each reply byte-exact-checked against the plain-python reference."""
    eliza = E.build_chat_min() if full else E.build_chat_min(rules=HF.FAST_RULES)
    if messages is None:
        messages = ["hi there", "sad today", "happy now", "bye now"]
    print("\n=== ELIZA through the NATIVE HF stack "
          "(model.generate + agentic tool-I/O loop) ===\n")
    all_ok = True
    for msg in messages:
        t0 = time.time()
        reply, thinking, meta = eliza_turn_causal_lm(eliza, msg)
        dt = time.time() - t0
        ref = E.chat_turn_ref(eliza, msg)
        ok = (ref == reply)
        all_ok = all_ok and ok
        print(f"you>   {msg}")
        print(f"eliza> {reply}", end="" if reply.endswith("\n") else "\n")
        print(f"       [{meta['n_tool']} tool-I/O services, {dt:.1f}s, "
              f"byte-exact-vs-reference={'YES' if ok else 'NO'}]\n")
    print(f"=== all byte-exact vs reference: {'YES' if all_ok else 'NO'} ===")


def main(argv: List[str] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    demo(full="--full" in argv)


if __name__ == "__main__":
    main()
