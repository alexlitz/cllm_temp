"""A tokenizer + reasoning parser + chat template for the C4 VM frame stream.

The fused Qwen C4 VM's ``C4VMForCausalLM`` generates over a ~267-token FRAME vocab
(``blogspec_vocab``): byte tokens 0..255 + a handful of structural markers
(REG_PC / REG_AX / REG_SP / REG_BP / MEM / STEP_END / BOS / HALT / THINK_START /
THINK_END).  ``C4VMTokenizer`` gives that stream the HF text surfaces the reasoning
stack expects:

  * ``decode(ids)`` -> raw text with the register frames wrapped in
    ``<think>…</think>`` and the PRTF OUTPUT bytes as the visible assistant text —
    exactly what a DeepSeek-R1 / Qwen-thinking reasoning parser consumes.
  * ``reasoning_content(ids)`` / ``content(ids)`` -> the STRUCTURED split (the
    ``reasoning_content`` vs ``content`` fields an OpenAI/R1 client reads).
  * ``apply_chat_template(messages, enable_thinking=...)`` -> the system/user/
    assistant prompt convention (with the ``enable_thinking`` flag), returning the
    BOS seed the VM decodes from.

The frame stream opens inside a think block (``THINK_START`` right after BOS); a
PRTF step emits ``THINK_END, <byte>, THINK_START`` so the byte is visible OUTPUT;
everything else (the register frames) is hidden reasoning.  This IS the think-tag
protocol the blogspec uses to separate internal VM thinking from user-facing stdout
(§Printing and Reading Input).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from . import blogspec_vocab as V


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _frame_to_text(frame: List[int]) -> str:
    """Render one 30-token register frame as a compact reasoning line."""
    dec = V.parse_step_frame(frame)
    return (f"[step] PC={dec['pc']} AX={dec['ax']} SP={dec['sp']} "
            f"BP={dec['bp']} MEM[{dec['mem_addr']}]={dec['mem_val']}")


class C4VMTokenizer:
    """Text / reasoning-content surface over the C4 VM frame-token stream."""

    def __init__(self):
        self.vocab_size = V.VOCAB
        self.bos_token_id = V.BOS
        self.eos_token_id = V.HALT
        self.pad_token_id = V.HALT
        # the reasoning-parser tags standard clients look for.
        self.think_start = THINK_OPEN
        self.think_end = THINK_CLOSE

    # -- structured split: reasoning_content vs content ----------------------
    def split(self, ids: List[int]) -> Dict[str, object]:
        """Split a generated frame stream into structured fields:
            {"reasoning_frames": [line, ...],   # the register-frame reasoning
             "reasoning_content": str,          # joined reasoning (R1/OpenAI field)
             "content": str}                    # the visible assistant OUTPUT (PRTF)
        A PRTF byte is OUTPUT (outside think); a register frame is reasoning.
        """
        frames: List[str] = []
        content = bytearray()
        inside_think = True                     # the stream opens inside THINK
        i = 0
        N = len(ids)
        while i < N:
            t = ids[i]
            if t == V.THINK_START:
                inside_think = True; i += 1; continue
            if t == V.THINK_END:
                inside_think = False; i += 1; continue
            if (t == V.REG_PC and i + V.FRAME_LEN <= N
                    and ids[i + V.FRAME_LEN - 1] == V.STEP_END):
                frames.append(_frame_to_text(ids[i:i + V.FRAME_LEN]))
                i += V.FRAME_LEN
                continue
            if not inside_think and 0 <= t <= 255:
                content.append(t)               # a visible PRTF OUTPUT byte
            i += 1
        return {"reasoning_frames": frames,
                "reasoning_content": "\n".join(frames),
                "content": content.decode("latin-1")}

    def reasoning_content(self, ids: List[int]) -> str:
        """The ``reasoning_content`` field (the hidden register-frame reasoning)."""
        return self.split(ids)["reasoning_content"]

    def content(self, ids: List[int]) -> str:
        """The visible assistant ``content`` (the PRTF OUTPUT bytes)."""
        return self.split(ids)["content"]

    # -- raw decode: <think>reasoning</think> + visible content --------------
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Render the stream as raw text a reasoning parser consumes: the register
        frames inside ``<think>…</think>``, the PRTF bytes as the visible output.

        With ``skip_special_tokens=True`` the ``<think>`` block is dropped entirely
        (the clean assistant message), matching a client that hides reasoning."""
        parts = self.split(ids)
        if skip_special_tokens:
            return parts["content"]
        think = parts["reasoning_content"]
        body = parts["content"]
        out = ""
        if think:
            out += f"{THINK_OPEN}\n{think}\n{THINK_CLOSE}\n"
        out += body
        return out

    # -- chat template -------------------------------------------------------
    def apply_chat_template(self, messages: List[Dict[str, str]],
                            tokenize: bool = True,
                            add_generation_prompt: bool = True,
                            enable_thinking: bool = True):
        """The system/user/assistant chat-template convention.  Returns the BOS seed
        the VM decodes from (``tokenize=True`` -> the ``[[BOS]]`` input id tensor;
        ``tokenize=False`` -> the rendered prompt text).

        The VM's "prompt" is the seeded program + the user line the READ pulls in;
        the chat template records the system/user framing and the ``enable_thinking``
        convention (whether the assistant surfaces its reasoning).  ``generate()``
        then runs from BOS.  This mirrors the Qwen ``enable_thinking`` template: with
        ``enable_thinking`` the assistant opens a ``<think>`` block; without it the
        reasoning is suppressed and only OUTPUT is surfaced."""
        rendered_lines = []
        for m in messages:
            rendered_lines.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            head = "<|im_start|>assistant\n"
            head += THINK_OPEN if enable_thinking else ""
            rendered_lines.append(head)
        text = "\n".join(rendered_lines)
        if not tokenize:
            return text
        import torch
        # the VM decodes its frame stream from BOS; the chat framing lives in the
        # template text (the user line enters via the READ tool-I/O boundary).
        return torch.tensor([[V.BOS]])
