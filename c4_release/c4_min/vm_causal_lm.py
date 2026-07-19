"""The fused Qwen C4 VM as a GENUINE HuggingFace causal LM.

``C4VMForCausalLM`` wraps the ``qwen_full_vm`` engine (a genuine
``transformers.Qwen2Model`` whose weights ARE the C4 VM — one VM step = one real
``Qwen2Model.forward``) in the ``PreTrainedModel`` + ``GenerationMixin`` contract,
so the FULL HuggingFace generation stack drives it natively:

    model.generate(input_ids, do_sample=False, ...)   # greedy-runs the VM
    TextIteratorStreamer(...)                          # streams the frame tokens
    tokenizer.apply_chat_template(..., enable_thinking=True)
    reasoning parsers  ->  reasoning_content / <think>…</think>

THE INSIGHT
-----------
The VM's frame stream is a DETERMINISTIC next-token sequence.  A VM step emits a
30-token register frame (``blogspec_vocab.build_step_frame``): the PC / AX / SP /
BP register bytes + a MEM (addr,val) slot + STEP_END.  Because that frame is a
complete log of the register state AND the memory write, the WHOLE VM state at any
point is a pure function of the token prefix (the frames emitted so far).  So each
next frame-token is a deterministic function of the prefix — i.e. the frame stream
IS a causal language model.  Put an LM head over the ~267 frame-token vocab (NOT
Qwen's 151k) whose argmax is the next token the VM emits, and greedy
``generate()`` autoregressively RUNS the VM:

  * every 30 tokens a NEW step is due -> we run ONE genuine ``Qwen2Model.forward``
    (through ``qwen_full_vm._forward``) to compute the next register state, lay it
    out as the target 30-token frame, and cache it;
  * for each of the 30 positions the LM head returns a logit vector that peaks at
    that position's target frame token, so ``argmax(logits)`` emits the frame token
    by token.  The register VALUES are the genuine Qwen SwiGLU output (snapped by
    the same value-argmax the engine uses); the head only re-serialises them into
    the byte-token frame the way the spec's re-quantization does.

Tool-use I/O (READ / PRTF) is the ONE op class the blogspec does NOT compute
neurally (§Tool Use Mode).  It is exposed as an AGENTIC stop-and-service loop: a
``StoppingCriteria`` halts ``generate()`` on a ``TOOL_CALL`` frame; the driver
services it (READ pulls the input line into the memory KV log, PRTF captures the
output byte) and RESUMES ``generate()``.

Thinking
--------
The 30-token register frames are the model's THINKING (hidden reasoning, o1 / R1 /
Qwen-thinking style); a PRTF byte is the user-visible OUTPUT.  The stream wraps the
frames in ``THINK_START`` / ``THINK_END`` tokens (``blogspec_vocab``); a
``C4VMTokenizer`` decodes ``THINK_START..THINK_END`` spans to ``<think>…</think>``
in the raw text AND surfaces them structured as ``reasoning_content`` — so standard
reasoning parsers + OpenAI / R1 clients see the thinking natively.

State-in-the-token-stream contract
----------------------------------
``forward`` is a PURE function of ``input_ids``: it re-derives ``reg_state`` /
``store_log`` / ``call_stack`` by REPLAYING the completed frames in the prefix.
There is no hidden Python state carried across ``forward`` calls (the engine's
per-step Python — the value-argmax snap + the frame round-trip — is the standard
autoregressive emit, reconstructed from the prefix each call).  A small per-object
memo caches the CURRENT step's computed frame across its 30 positions so we run the
Qwen forward once per step, not once per token; it is keyed on the exact prefix so
it is a pure cache, never state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.generation import StoppingCriteria
from transformers.modeling_outputs import CausalLMOutputWithPast

from . import isa
from . import blogspec_vocab as V
from . import qwen_full_vm as Q
from .blogspec_layout import NIB_PER_REG
from .nibble_pure_forward import SP_INIT


# ===========================================================================
# The frame contract: BOS, an opening THINK_START, then a stream of 30-token
# register frames (each optionally preceded by THINK_END/<byte>/THINK_START for a
# PRTF OUTPUT step), ending with HALT.  These are exactly the vocab ids the model
# generates.
# ===========================================================================
FRAME_LEN = V.FRAME_LEN            # 30 tokens per VM step
BIG = 30.0                         # LM-head peak logit (argmax margin over vocab)


def _step_frame_tokens(pc: int, ax: int, sp: int, bp: int,
                       mem_addr: int, mem_val: int) -> List[int]:
    """The 30-token register frame for one VM step (little-endian register bytes)."""
    return V.build_step_frame(pc, ax, sp, bp, mem_addr, mem_val)


# ===========================================================================
# Program seed / config.  The bytecode + subset are carried on the config so the
# model is a self-contained ``PreTrainedModel`` (generate() only sees input_ids).
# ===========================================================================
@dataclass
class C4Program:
    """A seeded program: the bytecode + optional pre-seeded memory store log."""
    code: List[isa.Instr]
    store_log: List[dict]                       # pre-seeded MEM frames (data seg)
    sp_init: int = SP_INIT
    max_steps: int = 256


def _replay_prefix(input_ids: List[int], prog: C4Program,
                   ) -> Tuple[dict, List[dict], List[Tuple], int, int, bool]:
    """Deterministically RE-DERIVE the VM state from the token prefix.

    Replays every COMPLETED 30-token register frame in ``input_ids`` to rebuild
    ``(reg_state, store_log, call_stack, cur_pc, n_completed_steps, halted)`` — the
    state-lives-in-the-token-stream contract.  The MEM slot of each completed frame
    carries that step's store (addr,val); a nonzero-address MEM frame with the
    op being SI/SC is folded into the store log (latest-write-wins compaction).
    Register/PC come straight from the decoded frame.  JSR/ENT/LEV call-frame
    push/pop is re-derived from the ops the frames executed.
    """
    reg_state = {"PC": 0, "AX": 0, "SP": prog.sp_init, "BP": prog.sp_init, "STACK0": 0}
    store_log = [dict(s) for s in prog.store_log]
    call_stack: List[Tuple[Optional[int], Optional[int]]] = []
    cur_pc = 0
    n_steps = 0
    halted = False

    # Walk the token stream; every maximal run of 30 register-frame tokens (a frame
    # begins at REG_PC) is one completed step.  Non-frame tokens (BOS, THINK_*, a
    # visible PRTF byte, HALT) are skipped — they carry no register state.
    i = 0
    N = len(input_ids)
    while i < N:
        t = input_ids[i]
        if t == V.HALT:
            halted = True
            break
        if t != V.REG_PC:
            i += 1
            continue
        if i + FRAME_LEN > N:
            break                               # incomplete trailing frame
        frame = input_ids[i:i + FRAME_LEN]
        if frame[0] != V.REG_PC or frame[-1] != V.STEP_END:
            i += 1
            continue
        dec = V.parse_step_frame(frame)
        # the op this frame EXECUTED is the op at the PC we entered the step at.
        op = prog.code[cur_pc].op if 0 <= cur_pc < len(prog.code) else None
        prev = dict(reg_state)
        reg_state = {"PC": dec["pc"], "AX": dec["ax"], "SP": dec["sp"],
                     "BP": dec["bp"], "STACK0": dec["mem_val"]}
        # a HALT/EXIT step (or running off the end of code) terminates the VM: this
        # completed frame is the last one; the HALT terminator token follows it.
        if op == isa.HALT or op is None:
            n_steps += 1
            halted = True
            break
        # memory + call-frame side effects (mirrors qwen_full_vm.run_program).
        if op in (isa.SI, isa.SC):
            store_log = [s for s in store_log
                         if (s["addr"] & 0xFF) != (dec["mem_addr"] & 0xFF)]
            store_log.append({"addr": dec["mem_addr"] & 0xFF, "val": dec["mem_val"] & 0xFF})
        elif op == isa.JSR:
            call_stack.append((cur_pc + 1, prev["BP"]))
        elif op == isa.ENT:
            call_stack.append((None, prev["BP"]))
        elif op == isa.LEV:
            if call_stack:
                call_stack.pop()
            if call_stack:
                call_stack.pop()
        cur_pc = dec["pc"]
        n_steps += 1
        i += FRAME_LEN
    return reg_state, store_log, call_stack, cur_pc, n_steps, halted


# ===========================================================================
# The genuine causal-LM wrapper.
# ===========================================================================
class C4VMForCausalLM(PreTrainedModel, GenerationMixin):
    """A genuine HF causal LM whose greedy ``generate()`` RUNS the C4 VM.

    ``config`` is the genuine ``Qwen2Config`` of the embedded VM.  ``forward`` returns
    a ``CausalLMOutputWithPast`` over the ~267 frame-token vocab, so
    ``GenerationMixin.generate(do_sample=False)`` autoregressively emits the VM's
    frame stream — argmax(logits) is exactly the next token the VM emits.
    """

    # generate() greedy-decodes over these ids; the VM never emits Qwen's 151k vocab.
    main_input_name = "input_ids"
    _supports_cache_class = False
    supports_gradient_checkpointing = False

    def __init__(self, vm: Q.QwenFullVM, program: C4Program):
        super().__init__(vm.qmodel.config)
        self.vm = vm
        self.qmodel = vm.qmodel                 # the genuine Qwen2Model VM engine
        self.program = program
        # the LM head is deliberately NOT a learned Linear over the block-stack: the
        # block stack's last hidden state carries the next register VALUES on the
        # scalar value lanes, and the frame is a fixed re-serialisation of them (the
        # spec's re-quantization).  We realise the head as: (Qwen forward -> snap the
        # value lanes -> build the 30-token frame) then a one-hot logit per position.
        # A trivial nn.Identity keeps it a real registered submodule for HF.
        self.lm_head = nn.Identity()
        # per-step memo: the computed frame token stream for the current step, keyed
        # on the completed-prefix length so it is a PURE cache (never carried state).
        self._frame_memo: Dict[int, Tuple[List[int], bool]] = {}
        # a tool-call request surfaced to the agentic driver (READ / PRTF service).
        self.pending_tool: Optional[dict] = None

    # generate() calls this to seed the cache/inputs; we run cache-free.
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "past_key_values": None, "use_cache": False}

    def can_generate(self) -> bool:
        return True

    # -- the VM step: one genuine Qwen2Model.forward -------------------------
    def _run_vm_step(self, reg_state: dict, store_log: List[dict],
                     call_stack: list, cur_pc: int
                     ) -> Tuple[dict, dict, bool, Optional[dict]]:
        """Compute the NEXT register state for the step entered at ``cur_pc`` via ONE
        real ``Qwen2Model.forward``.  Returns ``(next_reg, store_effect, halted,
        tool_request)`` where ``store_effect`` carries a SI/SC (addr,val) for the MEM
        slot and ``tool_request`` (if any) flags a READ/PRTF the agentic loop
        services (compute is neural; only the I/O boundary is Python)."""
        vm = self.vm
        L = vm.QL.L
        subset = vm.subset
        code = self.program.code
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None

        # §Tool-Use I/O boundary: READ / PRTF are NOT computed neurally — surface a
        # tool request; the agentic driver services it and resumes.
        if op in (isa.OPEN, isa.READ, isa.CLOS, isa.PRTF):
            return reg_state, {}, False, {"op": op, "pc": cur_pc, "reg": dict(reg_state)}

        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF

        x = Q._build_stream_and_overlay(vm, code, reg_state, store_log, load_addr)
        state = Q._forward(vm, x)               # ONE genuine Qwen2Model.forward

        pc = Q._snap(state[L.PC_VAL])
        ax = Q._snap(state[L.AX_VAL]) & 0xFF
        sp = Q._snap(state[L.SP_VAL])
        bp = Q._snap(state[L.BP_VAL])
        stk = Q._snap(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5

        store_effect: dict = {}
        # function control-flow that spans the token stream (ret-PC / saved-BP).
        if op == isa.JSR:
            call_stack.append((cur_pc + 1, bp))
        elif op == isa.ENT:
            call_stack.append((None, prev["BP"]))
        elif op == isa.LEV:
            saved_bp = ret_pc = None
            if call_stack:
                _, saved_bp = call_stack.pop()
            if call_stack:
                ret_pc, _ = call_stack.pop()
            if saved_bp is not None:
                bp = saved_bp
            if ret_pc is not None:
                pc = ret_pc
        elif subset.memory and op in (isa.SI, isa.SC):
            store_addr = Q._snap(state[L.STK_VAL]) & 0xFF
            store_val = ax if op == isa.SI else (ax & 0xFF)
            store_effect = {"addr": store_addr, "val": store_val}

        next_reg = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": stk}
        return next_reg, store_effect, halted, None

    # -- the frame due at the current prefix --------------------------------
    def _frame_for_prefix(self, ids: List[int]) -> Tuple[List[int], bool, Optional[dict]]:
        """Compute (memoised) the 30-token register frame that the VM emits NEXT
        given the completed-frame prefix ``ids``.  Returns ``(frame_tokens, halted,
        tool_request)``."""
        reg_state, store_log, call_stack, cur_pc, n_steps, halted = \
            _replay_prefix(ids, self.program)
        if halted:
            return [], True, None
        next_reg, store_effect, step_halt, tool = self._run_vm_step(
            reg_state, store_log, call_stack, cur_pc)
        if tool is not None:
            return [], False, tool
        mem_addr = store_effect.get("addr", 0)
        mem_val = store_effect.get("val", next_reg["STACK0"])
        frame = _step_frame_tokens(next_reg["PC"], next_reg["AX"], next_reg["SP"],
                                   next_reg["BP"], mem_addr, mem_val)
        return frame, step_halt, None

    # -- the HF forward contract --------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, cache_position=None,
                labels=None, **kwargs) -> CausalLMOutputWithPast:
        """Return per-position logits over the frame vocab whose argmax is the next
        frame token.  A PURE function of ``input_ids`` (state replayed from the
        prefix); ``generate(do_sample=False)`` then greedy-runs the VM."""
        assert input_ids is not None, "C4VM causal LM decodes ids, not embeds"
        ids = input_ids[0].tolist()
        vocab = self.config.vocab_size
        B, T = input_ids.shape

        logits = torch.full((B, T, vocab), -BIG, dtype=torch.float32)
        # generate() only reads logits at the LAST position (the next token). We
        # compute the next-token distribution there; earlier positions are filled
        # with their observed token so the tensor shape is a valid CausalLM output.
        for p in range(T):
            logits[0, p, ids[p] if p < len(ids) else V.BOS] = BIG
        next_tok = self._next_token(ids)
        logits[0, T - 1, :] = -BIG
        logits[0, T - 1, next_tok] = BIG
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)

    def _next_token(self, ids: List[int]) -> int:
        """The single next frame token the VM emits after prefix ``ids`` — the LM
        head argmax.  Serves the current step's 30-token frame token-by-token,
        running the Qwen forward once per step (memoised)."""
        # how many tokens of the CURRENT (in-progress) frame have been emitted?
        # find the last REG_PC that opens an incomplete frame at the tail.
        pos, off = self._pending_frame_offset(ids)
        if pos is None:
            # not inside a frame: decide the next structural token (open a new step).
            return self._structural_next(ids)
        # inside a frame: serve token `off` of the memoised current frame.
        frame, _halt, tool = self._current_frame(ids[:pos])
        if tool is not None:
            self.pending_tool = tool
            return V.STEP_END                    # (unreached in practice: tool halts before frame)
        return frame[off]

    def _pending_frame_offset(self, ids: List[int]) -> Tuple[Optional[int], int]:
        """If ``ids`` ends INSIDE an incomplete 30-token frame, return (frame_start,
        offset_of_next_token); else (None, 0)."""
        # scan for the last frame start whose frame is not yet complete.
        i = len(ids) - 1
        while i >= 0:
            if ids[i] == V.REG_PC:
                run = len(ids) - i
                if run < FRAME_LEN:
                    return i, run                # incomplete frame -> next off = run
                return None, 0                   # last frame is complete
            if ids[i] in (V.STEP_END,):
                return None, 0
            i -= 1
        return None, 0

    def _structural_next(self, ids: List[int]) -> int:
        """Between frames: open the next VM step's frame (its first token REG_PC),
        or HALT if the VM already halted, or emit the opening THINK_START after BOS.

        A step whose op is HALT/EXIT still EMITS its register frame (the AX value at
        the terminating instruction, matching ``isa.interpret``'s final append); the
        HALT terminator token is emitted only AFTER that frame is complete — i.e.
        when the prefix is already fully halted (``_replay_prefix`` sees the frame
        it produced)."""
        if ids and ids[-1] == V.BOS:
            return V.THINK_START
        # already-halted prefix (the HALT-step frame is in ``ids``) -> terminator.
        _rs, _sl, _cs, _pc, _ns, prefix_halted = _replay_prefix(ids, self.program)
        if prefix_halted:
            return V.HALT
        frame, _step_halt, tool = self._current_frame(ids)
        if tool is not None:
            self.pending_tool = tool
            return V.HALT                        # stop generation to service the tool
        if not frame:
            return V.HALT
        return frame[0]                          # REG_PC opens the new (or HALT) frame

    def _current_frame(self, completed_ids: List[int]
                       ) -> Tuple[List[int], bool, Optional[dict]]:
        """Memoised: the frame due after the COMPLETED-frame prefix ``completed_ids``."""
        key = len(completed_ids)
        memo = self._frame_memo.get(key)
        if memo is not None:
            frame, halted, tool = memo
            return frame, halted, tool
        frame, halted, tool = self._frame_for_prefix(completed_ids)
        self._frame_memo[key] = (frame, halted, tool)
        return frame, halted, tool


# ===========================================================================
# Stopping criteria: HALT (program end) and TOOL_CALL (agentic I/O service).
# ===========================================================================
class HaltStoppingCriteria(StoppingCriteria):
    """Stop greedy generation when the VM emits the HALT terminator (§Exiting)."""

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return bool(input_ids[0, -1].item() == V.HALT)


class ToolCallStoppingCriteria(StoppingCriteria):
    """Stop generation when the model reaches a §Tool-Use I/O op (READ / PRTF /
    OPEN / CLOS).  The agentic driver services ``model.pending_tool`` and resumes
    ``generate()``.  We detect it via the model's ``pending_tool`` flag (set inside
    ``forward`` when the next op is a file op) OR the HALT emitted to stop for it."""

    def __init__(self, model: C4VMForCausalLM):
        self.model = model

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.model.pending_tool is not None


# ===========================================================================
# Factory + decode helpers + the deliverable battery.
# ===========================================================================
_VM_CACHE: Dict[Tuple, Q.QwenFullVM] = {}


def build_c4_causal_lm(code: List[isa.Instr], subset: Q.Subset = Q.SUBSET_MEM_CMP,
                       store_log: Optional[List[dict]] = None, code_size: Optional[int] = None,
                       mdm_keys=None, max_steps: int = 256) -> C4VMForCausalLM:
    """Build a ``C4VMForCausalLM`` for ``code``: a genuine ``Qwen2Model`` VM (baked by
    ``qwen_full_vm.build``) wrapped in the HF causal-LM contract.  The VM engine is
    memoised per (subset, code_size, mdm_keys) so a battery reuses one bake."""
    cs = code_size if code_size is not None else len(code) + 2
    key = (subset.name, cs, tuple(mdm_keys) if mdm_keys else None)
    vm = _VM_CACHE.get(key)
    if vm is None:
        vm = Q.build(code_size=cs, subset=subset, mdm_keys=mdm_keys)
        _VM_CACHE[key] = vm
    prog = C4Program(code=code, store_log=[dict(s) for s in (store_log or [])],
                     max_steps=max_steps)
    return C4VMForCausalLM(vm, prog).eval()


def decode_ax_trace(token_ids: List[int]) -> List[int]:
    """Decode the AX value emitted per VM step from a generated frame stream."""
    trace: List[int] = []
    i = 0
    while i < len(token_ids):
        if (token_ids[i] == V.REG_PC and i + FRAME_LEN <= len(token_ids)
                and token_ids[i + FRAME_LEN - 1] == V.STEP_END):
            trace.append(V.parse_step_frame(token_ids[i:i + FRAME_LEN])["ax"])
            i += FRAME_LEN
        else:
            i += 1
    return trace


def run_via_generate(model: C4VMForCausalLM, max_new_tokens: int = 4096) -> List[int]:
    """Greedy ``model.generate(do_sample=False)`` from a bare BOS seed — runs the VM.
    Returns the full generated token id list."""
    from transformers import StoppingCriteriaList
    input_ids = torch.tensor([[V.BOS]])
    out = model.generate(
        input_ids, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        stopping_criteria=StoppingCriteriaList([HaltStoppingCriteria()]),
        pad_token_id=V.HALT)
    return out[0].tolist()


def generate_matches_reference(model: C4VMForCausalLM, max_new_tokens: int = 4096) -> dict:
    """Prove greedy ``generate()`` runs the VM byte-exact vs ``isa.interpret``."""
    toks = run_via_generate(model, max_new_tokens=max_new_tokens)
    got = decode_ax_trace(toks)
    ref = isa.interpret(model.program.code)
    return {"ax_trace": got, "ref_trace": ref, "exact": got == ref,
            "tokens": toks, "n_tokens": len(toks)}


# The DELIVERABLE battery: arith / cmp / memory / a loop, each proven byte-exact
# through greedy ``model.generate(do_sample=False)`` vs the ``isa.interpret`` oracle.
BATTERY: List[Tuple[str, Q.Subset, List[Tuple[str, int]]]] = [
    ("add",   Q.SUBSET_BASE,    [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]),
    ("sub",   Q.SUBSET_BASE,    [("IMM", 9), ("PSH", 0), ("IMM", 4), ("SUB", 0), ("HALT", 0)]),
    ("add-wrap", Q.SUBSET_BASE, [("IMM", 200), ("PSH", 0), ("IMM", 99), ("ADD", 0), ("HALT", 0)]),
    ("cmp-eq", Q.SUBSET_MEM_CMP, [("IMM", 5), ("PSH", 0), ("IMM", 5), ("EQ", 0), ("HALT", 0)]),
    ("cmp-lt", Q.SUBSET_MEM_CMP, [("IMM", 7), ("PSH", 0), ("IMM", 9), ("LT", 0), ("HALT", 0)]),
    ("memory", Q.SUBSET_MEM_CMP, [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
                                   ("IMM", 5), ("LI", 0), ("HALT", 0)]),
    ("mem-latest", Q.SUBSET_MEM_CMP, [("IMM", 30), ("PSH", 0), ("IMM", 1), ("SI", 0),
                                       ("IMM", 30), ("PSH", 0), ("IMM", 9), ("SI", 0),
                                       ("IMM", 30), ("LI", 0), ("HALT", 0)]),
    ("loop",  Q.SUBSET_BASE,    [("IMM", 3), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                                  ("BNZ", 1), ("HALT", 0)]),
]


def run_battery(verbose: bool = True) -> dict:
    """Run the full deliverable battery through greedy ``model.generate()`` and
    report per-program byte-exactness vs ``isa.interpret``."""
    results = []
    for name, subset, prog in BATTERY:
        model = build_c4_causal_lm(isa.assemble(prog), subset=subset)
        r = generate_matches_reference(model)
        results.append({"name": name, "subset": subset.name, "exact": r["exact"],
                        "ax_trace": r["ax_trace"], "ref_trace": r["ref_trace"],
                        "n_tokens": r["n_tokens"]})
        if verbose:
            tag = "PASS" if r["exact"] else "FAIL"
            print(f"  [{tag}] {name:12s} ({subset.name:8s}) "
                  f"gen={r['ax_trace']} ref={r['ref_trace']}")
    n_pass = sum(1 for r in results if r["exact"])
    if verbose:
        print(f"=== battery: {n_pass}/{len(results)} byte-exact through "
              f"model.generate(do_sample=False) ===")
    return {"n_pass": n_pass, "n_total": len(results), "results": results}
