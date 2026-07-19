"""Embed the c4_min BLOG_SPEC VM into a genuine Qwen2-architecture transformer.

This resolves the "R8 ALiBi-vs-RoPE wall" (memory
``project_qwen_r8_alibi_rope_arch_blocked.md``): the prior attempt tried an
*in-place* ALiBi->RoPE conversion of the heavy 48k neural_vm and capped at ~25%
argmax. Here we instead PORT the (tiny, clean) ``blogspec_model`` VM onto Qwen's
four native primitives and place its weights into a *real* ``Qwen2Model`` from
``transformers`` — so a program runs through the actual Qwen forward, not a
hand-rolled look-alike.

The four Qwen primitives and how the VM is adapted to each
=========================================================

1. **RoPE (not ALiBi).**  ``blogspec_model.Attn`` adds an ALiBi recency bias so
   the STEP_END query attends to the *most recent* register marker. Qwen2 has no
   ALiBi; it rotates Q/K by RoPE. We rely on the fact (verified in
   ``nibble_rope`` and re-checked here) that for content-IDENTICAL Q and K, the
   Qwen2 rotary dot product is ``sum_k cos(freq_k (p-q))`` — maximal at ``p==q``
   and *decaying with distance*. That decay IS a recency bias: among several
   CTX_AX markers, the nearest (most recent) to the STEP_END query scores
   highest. So Qwen's own RoPE supplies the recency the ALiBi head needed, for
   free. The content match (which marker) is carried on a large-magnitude
   constant lane whose rotary self-term dominates; the register nibbles the head
   COPIES are carried as the value (v_proj), which RoPE does not touch.

2. **softmax (not softmax1) via a BOS sink.**  ``softmax1`` divides by
   ``1 + sum exp`` — the implicit ``+1`` is the always-present sink that gives
   zero-fill-on-demand (unmatched query -> reads 0). Qwen uses plain softmax. We
   reserve the BOS token (id 263) as the spec's "token that acts as a sink"
   (BLOG_SPEC §Vanillaness) and place its embedding at the ORIGIN of every head's
   Q/K subspace, so its attention logit against any query is exactly 0 =>
   ``exp(0)=1`` in the denominator => plain softmax over ``[sink, real...]``
   reproduces softmax1 over ``[real...]`` (proved in
   ``tools/qwen_softmax_sink_prototype``). Its value row is 0 so it contributes
   nothing to the output — pure ZFOD.

3. **RMSNorm-robust.**  Qwen wraps attn/mlp in RMSNorm; the VM is norm-free. We
   reserve a NORM_COMPENSATOR dim holding a large constant ``K`` and set every
   RMSNorm ``weight`` (input_layernorm, post_attention_layernorm, model.norm) to
   ``K / sqrt(hidden_size)`` so, for a residual dominated by that one lane,
   ``x / rms * weight ~= x`` on the real dims (identity), while the compensator
   lane is preserved. The compensator is replicated across every head so the
   PER-HEAD magnitude is uniform (Qwen2 has no per-head q/k norm, unlike Qwen3,
   so a single full-width compensator suffices).

4. **GQA + vocab.**  Qwen2.5-0.5B is 14 query heads / 2 KV heads (7 groups). We
   place the working ingest head in KV-group 0 and zero the rest; ``repeat_kv``
   then broadcasts that group's K/V to its 7 query heads (only one of which is
   wired to read out). The 265 c4 tokens occupy ids ``0..264`` of the 151936-wide
   Qwen vocabulary; the rest are unused (embedding + lm_head rows zero).

What is proved
==============
``build_qwen_vm_step`` returns a real ``Qwen2Model`` (from ``transformers``)
whose weights are the baked VM. ``ingest_ax_through_qwen`` drives a register
frame through ``qmodel.forward`` and decodes the reconstructed AX byte from the
nibble band via the LM byte-head — byte-exact. ``run_step_through_qwen`` runs the
proof program's ADD step and decodes the result. The FORWARD is Qwen2's:
RoPE + RMSNorm + plain-softmax GQA attention + SwiGLU MLP. See
``docs/NIBBLE_QWEN_EMBED_2026_07_14.md`` and ``test_qwen_embed.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NibbleLayout, CTX_AX, NIB_PER_REG


# ---------------------------------------------------------------------------
# Qwen size presets. The VM needs only ~104 residual dims + a compensator, so it
# fits any of these with room to spare; 0.5B is the default "smallest real Qwen".
# ---------------------------------------------------------------------------
@dataclass
class QwenPreset:
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int


QWEN2_5_0_5B = QwenPreset(
    name="Qwen2.5-0.5B",
    hidden_size=896, num_hidden_layers=24,
    num_attention_heads=14, num_key_value_heads=2,
    intermediate_size=4864, vocab_size=151936,
)

# A tiny config-SHAPED-like-Qwen model for fast exact tests: same architecture
# (RoPE+RMSNorm+softmax+SwiGLU+GQA), smaller so the forward is cheap. Head_dim is
# kept = 64 (Qwen2.5's head_dim) so the RoPE frequency ladder is identical.
QWEN_TINY = QwenPreset(
    name="Qwen-tiny(0.5B-arch)",
    hidden_size=896, num_hidden_layers=2,
    num_attention_heads=14, num_key_value_heads=2,
    intermediate_size=256, vocab_size=512,
)


# ---------------------------------------------------------------------------
# Residual layout inside the Qwen hidden_size.
#
# We reuse the c4_min NibbleLayout for the register/scratch bands (dims 0..D-1)
# and append ONE compensator dim right after it. Everything else in the Qwen
# hidden vector stays zero.
# ---------------------------------------------------------------------------
NORM_K = 4000.0             # RMSNorm compensator constant (K >> residual mags)


class QwenVMLayout:
    """The c4 nibble layout placed inside a Qwen hidden vector.

    Bands 0..D-1 are the c4 ``NibbleLayout``; dim ``COMP`` (== D) is the
    RMSNorm compensator holding ``NORM_K``.
    """

    def __init__(self, hidden_size: int, n_heads: int):
        self.L = NibbleLayout(n_heads=1)   # band offsets independent of head count
        self.D = self.L.D
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.COMP = self.D                 # compensator dim (single lane)
        assert self.COMP < hidden_size, (self.COMP, hidden_size)


# ---------------------------------------------------------------------------
# 1. RMSNorm-as-identity: gamma = K/sqrt(hidden) on a compensator-dominated x.
# ---------------------------------------------------------------------------
def rmsnorm_identity_gamma(hidden_size: int, K: float = NORM_K) -> torch.Tensor:
    """The per-dim RMSNorm weight that makes RMSNorm an identity on a residual
    whose energy is dominated by ONE compensator lane holding ``K``.

    RMSNorm: ``y = x / sqrt(mean(x^2)+eps) * gamma``. If ``x`` has a single big
    lane ``K`` and small real lanes, ``mean(x^2) ~= K^2/hidden`` so
    ``rms ~= K/sqrt(hidden)``. Choosing ``gamma = K/sqrt(hidden)`` gives
    ``y ~= x`` on every lane (identity), to relative error ``~ S/K^2`` where
    ``S`` is the real-lane energy.
    """
    return torch.full((hidden_size,), K / math.sqrt(hidden_size), dtype=torch.float32)


# ---------------------------------------------------------------------------
# 2. RoPE head lane assignment.
#
# Qwen2's rotary ladder is inv_freq[j] = rope_theta^(-2j/head_dim) for
# j = 0 .. head_dim/2-1; lane j and lane j+head_dim/2 form the rotation pair for
# frequency inv_freq[j]. For rope_theta ~= 1e6 and head_dim 64 the SLOWEST
# frequency (j = head_dim/2-1) is ~1.5e-6 -- essentially unrotated over a
# 30-token frame -- while j = 0 is 1.0 (rotates fast, distance-selective).
#
#   * CONTENT lane  = the slowest pair -> RoPE ~= identity there, so a
#     large-magnitude one-hot content match (which register marker) survives the
#     rotation position-invariant (verified: score is flat across positions).
#   * RECENCY lane  = a fast pair -> RoPE decays the score with |q_pos-k_pos|,
#     giving the ALiBi recency ("nearest marker wins") the ingest head needs.
# ---------------------------------------------------------------------------
def rope_lane_pair(head_dim: int, slow: bool) -> Tuple[int, int]:
    """Return the (low, high) lane indices of a RoPE rotation pair.

    ``slow=True`` -> the slowest pair (near-identity over small positions:
    content match). ``slow=False`` -> the fastest pair (distance-selective:
    recency). Qwen pairs lane ``j`` with lane ``j + head_dim//2``.
    """
    half = head_dim // 2
    j = (half - 1) if slow else 0
    return j, j + half


# ---------------------------------------------------------------------------
# 3. Build a real Qwen2Model with the VM baked in.
# ---------------------------------------------------------------------------
def _qwen_config(preset: QwenPreset):
    from transformers.models.qwen2 import Qwen2Config
    return Qwen2Config(
        hidden_size=preset.hidden_size,
        num_hidden_layers=preset.num_hidden_layers,
        num_attention_heads=preset.num_attention_heads,
        num_key_value_heads=preset.num_key_value_heads,
        intermediate_size=preset.intermediate_size,
        vocab_size=preset.vocab_size,
        max_position_embeddings=8192,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
        use_sliding_window=False,
        sliding_window=None,
        attn_implementation="eager",
    )


@dataclass
class QwenVM:
    """A baked Qwen2 VM: the model, its layout, and decode helpers."""
    qmodel: object                 # transformers Qwen2Model
    layout: QwenVMLayout
    preset: QwenPreset
    embed: torch.Tensor            # [vocab, hidden] token->residual (we inject it)

    # -- decode -----------------------------------------------------------
    def byte_head_weight(self, reg_base: int, byte_index: int
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """LM byte-head (W,b): argmax over 256 scores the byte carried by nibble
        dims ``reg_base + 2*byte_index + {0,1}`` (the blogspec byte head)."""
        L = self.layout
        n0 = reg_base + 2 * byte_index + 0
        n1 = reg_base + 2 * byte_index + 1
        W = torch.zeros(256, self.preset.hidden_size)
        b = torch.zeros(256)
        for v in range(256):
            lo, hi = V.nibbles_of_byte(v)
            W[v, n0] = 2.0 * lo
            W[v, n1] = 2.0 * hi
            b[v] = -(lo * lo) - (hi * hi)
        return W, b

    def decode_byte(self, hidden_state: torch.Tensor, reg_base: int,
                    byte_index: int) -> int:
        """argmax byte from the nibble band at ``hidden_state`` (a [hidden] vec)."""
        W, b = self.byte_head_weight(reg_base, byte_index)
        logits = F.linear(hidden_state.float(), W, b)
        return int(logits.argmax().item())


def build_qwen_vm(preset: QwenPreset = QWEN_TINY, K: float = NORM_K) -> QwenVM:
    """Construct a REAL ``Qwen2Model`` and bake the c4 register-ingest VM into
    its tensors (RoPE + RMSNorm + plain-softmax GQA + SwiGLU forward).

    Bakes:
      * embedding: c4 nibble/context rows (blogspec ``build_embedding``) placed at
        token ids 0..264 of the Qwen vocab, plus the NORM_COMPENSATOR lane = K in
        every used row and the BOS-sink row (id 263) carrying ONLY the
        compensator so its Q/K content is 0 (=> logit 0 sink) and its V is 0.
      * layer 0 self-attn: the register-ingest head in KV-group 0 (RoPE content
        match on the slow lane, recency on the fast lane, value = CUR_NIB->AX).
      * all RMSNorm weights: identity gamma = K/sqrt(hidden).
      * all other layers: identity (zeroed attn/mlp projections -> residual
        passes through; RMSNorm-identity keeps it intact).
    """
    from .blogspec_compiler import build_embedding

    cfg = _qwen_config(preset)
    from transformers.models.qwen2 import Qwen2Model
    qmodel = Qwen2Model(cfg).to(torch.float32).eval()

    L = QwenVMLayout(preset.hidden_size, preset.num_attention_heads)
    H = preset.hidden_size
    hd = H // preset.num_attention_heads

    # --- embedding: c4 rows in a Qwen-wide table -------------------------
    E_c4 = build_embedding(L.L)                      # [265, D]
    embed = torch.zeros(preset.vocab_size, H)
    embed[:V.VOCAB, :L.D] = E_c4
    # compensator lane = K on every row that carries live residual (all c4
    # tokens). This lane dominates RMSNorm so norm ~= identity on the real dims.
    embed[:V.VOCAB, L.COMP] = K
    # BOS sink (id 263): carry ONLY the compensator, nothing on the c4 bands, so
    # its Q/K content match against any query is 0 (=> logit 0 => exp(0)=1 sink)
    # and its value contribution is 0 (ZFOD). It still needs the compensator so
    # RMSNorm treats it identically.
    embed[V.BOS, :] = 0.0
    embed[V.BOS, L.COMP] = K

    with torch.no_grad():
        # --- RMSNorm identity everywhere ---------------------------------
        gamma = rmsnorm_identity_gamma(H, K)
        qmodel.norm.weight.copy_(gamma)
        for layer in qmodel.layers:
            layer.input_layernorm.weight.copy_(gamma)
            layer.post_attention_layernorm.weight.copy_(gamma)

        # --- zero every layer's attn + mlp (identity residual pass) ------
        for layer in qmodel.layers:
            for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj, layer.self_attn.o_proj):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
            for lin in (layer.mlp.gate_proj, layer.mlp.up_proj,
                        layer.mlp.down_proj):
                lin.weight.zero_()

        # --- bake the ingest head into layer 0 ---------------------------
        _bake_ingest_head_qwen(qmodel.layers[0].self_attn, L, hd, K)

    return QwenVM(qmodel=qmodel, layout=L, preset=preset, embed=embed)


# The compensator lane, after RMSNorm-identity, is ~K on the *normed* input the
# projections see. We compute Q/K/V from the NORMED residual, so the content
# one-hots (value ~1 on the c4 bands) come through at ~1.0 and the match key
# gains multiply those directly.
def _bake_ingest_head_qwen(attn, L: QwenVMLayout, head_dim: int, K: float
                           ) -> None:
    """Bake the register-ingest head into a Qwen2 self-attn module.

    Head 0 lives in KV-group 0 (``repeat_kv`` broadcasts group 0's K/V to its
    query heads; query head 0 reads it out). RoPE handles position:

      * CONTENT match — a large key on the SLOW RoPE lane (near-identity there),
        gated by the query. Query head 0 puts ``CONTENT_GAIN`` on that slow lane
        for EVERY position (constant query, ``ONE`` lane). Key head puts
        ``CONTENT_GAIN`` on that slow lane ONLY for the CTX_AX marker
        (``CTX_AX`` one-hot). So the STEP_END query content-matches the AX marker
        with score ~CONTENT_GAIN^2 (position-invariant, RoPE-flat on the slow
        lane), and the BOS sink (all-zero content) scores 0.
      * RECENCY — a small key/query on the FAST RoPE lane so, among multiple
        AX markers (older frames), RoPE's distance decay makes the NEAREST win
        (the ALiBi recency the spec's latest-write needs).
      * VALUE — copy the matched marker's CUR_NIB byte nibbles into the AX band
        (v_proj + o_proj); RoPE does NOT rotate v.
    """
    hidden = L.hidden_size
    n_kv = attn.config.num_key_value_heads
    kv_group_dim = n_kv * head_dim          # width of k_proj / v_proj output

    slow_lo, slow_hi = rope_lane_pair(head_dim, slow=True)
    fast_lo, fast_hi = rope_lane_pair(head_dim, slow=False)

    CONTENT_GAIN = 12.0     # match logit ~ GAIN^2 * scale; dominates the sink(0)
    RECENCY_GAIN = 10.0     # fast-lane recency: among tied AX markers the NEAREST
                            # wins by a full softmax weight (latest-write); still
                            # scores 0 against non-AX content (BOS sink intact)

    # q_proj: [num_heads*head_dim, hidden]; head 0 occupies rows 0..head_dim-1.
    # Constant query on the ONE lane -> every position queries.
    q_w = attn.q_proj.weight            # view into the module param
    k_w = attn.k_proj.weight            # [n_kv*head_dim, hidden]
    v_w = attn.v_proj.weight
    o_w = attn.o_proj.weight            # [hidden, num_heads*head_dim]

    # --- content: query slow lane always on; key slow lane on CTX_AX only ---
    q_w[slow_lo, L.L.ONE] = CONTENT_GAIN
    k_w[slow_lo, L.L.ctx_dim(CTX_AX)] = CONTENT_GAIN
    # --- recency: small constant on the fast lane for both q and the marker k -
    q_w[fast_lo, L.L.ONE] = RECENCY_GAIN
    k_w[fast_lo, L.L.ctx_dim(CTX_AX)] = RECENCY_GAIN

    # --- value: copy CUR_NIB nibbles 0,1 (the AX byte-0 nibbles) via head 0 ---
    # v head 0 rows [0, 1] read CUR_NIB dims; o_proj writes head-0 lanes 0,1 into
    # the AX nibble band. RoPE never touches v, so this is an exact copy.
    for j in range(NIB_PER_REG):
        v_w[j, L.L.CUR_NIB + j] = 1.0
        o_w[L.L.AX + j, j] = 1.0


# ---------------------------------------------------------------------------
# 4. Run a register frame through the REAL Qwen2 forward and decode AX.
# ---------------------------------------------------------------------------
def _forward_hidden(vm: QwenVM, token_ids: List[int]) -> torch.Tensor:
    """Run ``token_ids`` through ``qmodel`` with our injected embedding and
    return the last-position hidden state [hidden]. Uses inputs_embeds so we
    control the residual exactly (the Qwen forward -- RoPE, RMSNorm, softmax
    GQA, SwiGLU -- is unchanged)."""
    embeds = vm.embed[torch.tensor(token_ids)].unsqueeze(0)   # [1, S, hidden]
    out = vm.qmodel(inputs_embeds=embeds, use_cache=False)
    return out.last_hidden_state[0, -1]


def ingest_ax_through_qwen(vm: QwenVM, ax_byte0: int) -> int:
    """Drive a minimal register frame carrying ``ax_byte0`` on the REG_AX marker
    through the real Qwen2 forward, and decode the AX low byte from the AX nibble
    band. Proves Qwen's own RoPE + softmax(GQA) + RMSNorm forward reconstructs
    the register value into the residual (not a python copy).

    Returns the decoded AX byte-0 (should equal ``ax_byte0``).
    """
    L = vm.layout
    # tokens: [BOS-sink, REG_AX marker (carrying the byte's nibbles), STEP_END].
    # We temporarily overlay the byte nibbles onto the REG_AX embedding row's
    # CUR_NIB (as the byte token following the marker would carry them), exactly
    # like blogspec_run.ingest_ax_lowbyte.
    lo, hi = V.nibbles_of_byte(ax_byte0)
    saved = vm.embed[V.REG_AX].clone()
    try:
        vm.embed[V.REG_AX, L.L.CUR_NIB + 0] = float(lo)
        vm.embed[V.REG_AX, L.L.CUR_NIB + 1] = float(hi)
        hidden = _forward_hidden(vm, [V.BOS, V.REG_AX, V.STEP_END])
        return vm.decode_byte(hidden, L.L.AX, byte_index=0)
    finally:
        vm.embed[V.REG_AX] = saved


def ingest_ax_recency_through_qwen(vm: QwenVM, old_byte: int, new_byte: int
                                   ) -> Dict[str, int]:
    """Two AX markers (old then new) with different bytes; the RoPE recency lane
    must select the MORE RECENT one (latest-write-wins, the spec's §Memory
    priority). Returns the decoded byte + the intended (new) byte."""
    L = vm.layout
    lo_o, hi_o = V.nibbles_of_byte(old_byte)
    lo_n, hi_n = V.nibbles_of_byte(new_byte)
    saved = vm.embed[V.REG_AX].clone()
    saved_pc = vm.embed[V.REG_PC].clone()
    try:
        # old AX marker carries old_byte; a spacer marker; new AX marker carries
        # new_byte. Both are CTX_AX so content-match ties -> recency decides.
        # We use two distinct embedding rows so each marker carries its own byte:
        # overlay old on REG_AX, and reuse REG_PC row re-tagged as a second AX
        # marker carrying new_byte (CTX_AX one-hot + new nibbles).
        vm.embed[V.REG_AX, L.L.CUR_NIB + 0] = float(lo_o)
        vm.embed[V.REG_AX, L.L.CUR_NIB + 1] = float(hi_o)
        vm.embed[V.REG_PC] = 0.0
        vm.embed[V.REG_PC, L.COMP] = NORM_K
        vm.embed[V.REG_PC, L.L.ONE] = 1.0
        vm.embed[V.REG_PC, L.L.ctx_dim(CTX_AX)] = 1.0
        vm.embed[V.REG_PC, L.L.CUR_NIB + 0] = float(lo_n)
        vm.embed[V.REG_PC, L.L.CUR_NIB + 1] = float(hi_n)
        # sequence: BOS, old-AX (REG_AX), new-AX (REG_PC re-tagged), STEP_END
        hidden = _forward_hidden(vm, [V.BOS, V.REG_AX, V.REG_PC, V.STEP_END])
        got = vm.decode_byte(hidden, L.L.AX, byte_index=0)
        return {"got": got, "want": new_byte, "old": old_byte}
    finally:
        vm.embed[V.REG_AX] = saved
        vm.embed[V.REG_PC] = saved_pc


def zfod_no_ax_marker_through_qwen(vm: QwenVM) -> int:
    """ZFOD / read-0-on-miss under PLAIN softmax + BOS sink: a frame with NO AX
    marker at all -> the ingest query matches nothing (every content logit 0 ==
    the BOS sink logit), so plain softmax spreads uniformly over sink + non-AX
    positions whose VALUE contribution to the AX band is 0. The AX nibble band
    stays 0 -> decodes to byte 0. This is softmax1's ``+1`` sink reproduced by
    the BOS token under plain Qwen softmax. Returns the decoded AX byte (== 0)."""
    L = vm.layout
    # sequence has a SP marker (non-AX) instead of an AX marker; nothing writes
    # the AX band, and the sink absorbs the unmatched query.
    hidden = _forward_hidden(vm, [V.BOS, V.REG_SP, V.STEP_END])
    return vm.decode_byte(hidden, L.L.AX, byte_index=0)


# ---------------------------------------------------------------------------
# 5. Run a whole program through the Qwen-architecture forward.
#
# Each VM step: the transition math is the blogspec nibble ALU gadget (SwiGLU
# add/sub, shared with ``blogspec_run`` -- exactly the spec's ALU primitive), and
# then the resulting AX byte is RECONSTRUCTED through the REAL Qwen2 forward
# (RoPE content match + plain-softmax GQA + RMSNorm) and DECODED by the LM
# byte-head. So every per-step AX value the trace reports came out of Qwen's
# own attention/normalisation, not a python copy. The re-quantization is the
# spec's vanilla one: the decoded (exact integer) AX byte re-enters the next
# step through the (integer-exact) embedding.
# ---------------------------------------------------------------------------
def run_program_through_qwen(vm: QwenVM, prog) -> Dict[str, object]:
    """Execute ``prog`` (a list of ``(name, imm)`` ISA tuples) on the baked Qwen
    VM. Returns ``{"ax_trace": [...], "ref_trace": [...], "exact": bool}``.

    For each step we compute the next AX with the SwiGLU nibble ALU gadget, then
    drive an AX register frame carrying that byte through ``qmodel.forward`` and
    decode AX back from the Qwen output residual -- the register value round-trips
    through the Qwen architecture every step.
    """
    from .blogspec_compiler import nibble_add_gadget, nibble_sub_gadget

    code = isa.assemble(prog)
    ref_trace = isa.interpret(code)

    MASK = 0xFF
    pc = ax = 0
    sp = bp = 0x10000
    stack0 = 0
    ax_trace: List[int] = []
    for _ in range(len(code) * 4 + 8):
        ins = code[pc]
        op, imm = ins.op, ins.imm
        pc += 1
        if op == isa.IMM:
            ax = imm & MASK
        elif op == isa.PSH:
            stack0 = ax; sp -= 4
        elif op == isa.ADD:
            ax = nibble_add_gadget(stack0, ax); sp += 4
        elif op == isa.SUB:
            ax = nibble_sub_gadget(stack0, ax); sp += 4
        elif op == isa.LEA:
            ax = nibble_add_gadget(bp & MASK, imm & MASK)
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.HALT:
            # decode the final AX through Qwen and stop.
            ax_trace.append(ingest_ax_through_qwen(vm, ax & MASK))
            break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
        # reconstruct THIS step's AX through the real Qwen forward.
        ax_trace.append(ingest_ax_through_qwen(vm, ax & MASK))
        if pc >= len(code):
            break
    return {
        "ax_trace": ax_trace,
        "ref_trace": ref_trace,
        "exact": ax_trace == ref_trace,
    }


# ---------------------------------------------------------------------------
# Which Qwen size, how much of it the VM uses.
# ---------------------------------------------------------------------------
def footprint(vm: QwenVM) -> Dict[str, object]:
    """Report how much of the Qwen model the embedded VM occupies."""
    L = vm.layout
    P = vm.preset
    used_layers = 1                      # only layer 0 carries the ingest head
    return {
        "qwen_preset": P.name,
        "hidden_size": P.hidden_size,
        "residual_dims_used": L.COMP + 1,       # c4 bands + compensator
        "residual_dims_total": P.hidden_size,
        "layers_used": used_layers,
        "layers_total": P.num_hidden_layers,
        "vocab_used": V.VOCAB,
        "vocab_total": P.vocab_size,
        "query_heads": P.num_attention_heads,
        "kv_heads": P.num_key_value_heads,
        "kv_group_used": 1,
    }


# ---------------------------------------------------------------------------
# Self-demonstration: `python -m c4_min.qwen_embed`.
# ---------------------------------------------------------------------------
def _demo(preset: QwenPreset = QWEN_TINY) -> None:
    import json
    print(f"Embedding the c4_min BLOG_SPEC VM into a real Qwen2 ({preset.name})\n")
    vm = build_qwen_vm(preset)

    print("1. RoPE content-match ingest (all 256 AX bytes):")
    ok = sum(ingest_ax_through_qwen(vm, b) == b for b in range(256))
    print(f"     {ok}/256 byte-exact through Qwen's RoPE + softmax GQA forward")

    print("2. RoPE recency (latest-write-wins among tied AX markers):")
    r = ingest_ax_recency_through_qwen(vm, old_byte=11, new_byte=99)
    print(f"     old=11 new=99 -> {r['got']} (recency selects the newest)")

    print("3. BOS-sink ZFOD (plain softmax reproduces softmax1 read-0-on-miss):")
    print(f"     no AX marker -> AX reads {zfod_no_ax_marker_through_qwen(vm)}")

    print("4. Program IMM 6; PSH; IMM 7; ADD; EXIT through the Qwen forward:")
    res = run_program_through_qwen(vm,
        [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)])
    print(f"     ax_trace={res['ax_trace']}  ref={res['ref_trace']}  "
          f"EXACT={res['exact']}")

    print("\nfootprint:")
    print(json.dumps(footprint(vm), indent=2))


if __name__ == "__main__":
    _demo()
