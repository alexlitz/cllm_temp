"""Split ALUAddSub into 5 stage modules sharing intermediate GE-format state.

Phase 0 (2026-05-10): the monolithic `ALUAddSub` runtime wrapper is being
flattened into 5 sequential FFN stages so the compiler can manage them
individually. Each stage is a standalone `nn.Module` installable as
`block.ffn`.

Math is identical to `PureNeuralALU(operations='add_sub').forward`. The
forward is split into 5 chunks; intermediate GE-format state ([B, seq, 8, 160])
is passed via a shared `_AddSubGEState` container that all 5 stages reference.

Stage layout:
  0. _AddSubBDToGE     -- BD→GE projection (BDToGEConverter equivalent)
  1. _AddSubStage1     -- AddRawAndGenFFN + SubRawAndGenFFN, opcode-merged
  2. _AddSubStage2     -- AddCarryLookaheadFFN + SubBorrowLookaheadFFN, merged
  3. _AddSubStage3     -- AddFinalResultFFN + SubFinalResultFFN, merged
  4. _AddSubGEToBD     -- GE→BD writeback (GEToBDConverter equivalent)

Stages 1-3 each merge ADD and SUB pipelines: both run in parallel on a
clone of the GE state, results combined via opcode mask. This mirrors the
original `ALUAddSub.forward` logic exactly.

Output is byte-identical to the original wrapper since:
  - The same BDToGEConverter / GEToBDConverter weights are used
  - The same build_add_layers / build_sub_layers stages are used
  - The same opcode-mask merging is done at the same point (between stage 3
    and the GE→BD writeback)
"""

import torch
import torch.nn as nn

from .alu.chunk_config import NIBBLE
from .alu.ops.add import build_add_layers
from .alu.ops.sub import build_sub_layers
from .alu.ops.common import GenericE
from .efficient_alu_neural import BDToGEConverter, GEToBDConverter


class _AddSubGEState(nn.Module):
    """Shared mutable state container for the 5 add/sub stages.

    Holds the GE-format buffer between stages. Single instance is shared by
    all 5 stages (they each store a reference). State is per-forward and
    cleared on each call to stage 0.
    """

    def __init__(self):
        super().__init__()
        # Last-forward state (set by stages, read by next stage).
        # `_x_ge` is the cumulative GE state. `_x_ge_add` and `_x_ge_sub`
        # are parallel ADD/SUB pipelines (set by stage 1, mutated by 2-3,
        # merged into `_x_ge` at the end of stage 3).
        self._x_ge = None
        self._x_ge_add = None
        self._x_ge_sub = None
        self._x_ge_out = None
        self._opcode_mask = None
        self._original_shape = None  # (B, seq_len)


class _AddSubBDToGE(nn.Module):
    """Stage 0: BD → GE format conversion.

    Calls the BDToGEConverter (Python forward, mathematically a linear
    projection with opcode normalization). Stores the GE-format result in
    the shared state container.

    Output: returns x_bd unchanged (residual identity). Side effect:
    populates `state._x_ge`, `state._x_ge_add`, `state._x_ge_sub`.
    """

    def __init__(self, BD, state: _AddSubGEState):
        super().__init__()
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.state = state
        self.bd_to_ge = BDToGEConverter(BD, self.ge)

    def forward(self, x_bd):
        B, seq_len, _ = x_bd.shape
        x_ge = self.bd_to_ge(x_bd)  # [B, seq_len, 8, 160]

        # Flatten for per-position processing
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)

        # Stash for downstream stages
        self.state._x_ge = x_ge_flat.clone()
        self.state._x_ge_add = x_ge_flat.clone()
        self.state._x_ge_sub = x_ge_flat.clone()
        self.state._x_ge_out = x_ge_flat.clone()
        self.state._original_shape = (B, seq_len)
        self.state._x_bd_input = x_bd  # needed by GE→BD writeback

        return x_bd  # residual identity


class _AddSubStage1(nn.Module):
    """Stage 1: AddRawAndGenFFN + SubRawAndGenFFN.

    Runs add_layers[0] and sub_layers[0] on parallel ADD and SUB GE
    buffers from the shared state.
    """

    def __init__(self, state: _AddSubGEState):
        super().__init__()
        self.state = state
        self.add_layer = build_add_layers(NIBBLE, opcode=25)[0]
        self.sub_layer = build_sub_layers(NIBBLE, opcode=26)[0]

    def forward(self, x_bd):
        self.state._x_ge_add = self.add_layer(self.state._x_ge_add)
        self.state._x_ge_sub = self.sub_layer(self.state._x_ge_sub)
        return x_bd  # residual identity


class _AddSubStage2(nn.Module):
    """Stage 2: AddCarryLookaheadFFN + SubBorrowLookaheadFFN."""

    def __init__(self, state: _AddSubGEState):
        super().__init__()
        self.state = state
        self.add_layer = build_add_layers(NIBBLE, opcode=25)[1]
        self.sub_layer = build_sub_layers(NIBBLE, opcode=26)[1]

    def forward(self, x_bd):
        self.state._x_ge_add = self.add_layer(self.state._x_ge_add)
        self.state._x_ge_sub = self.sub_layer(self.state._x_ge_sub)
        return x_bd


class _AddSubStage3(nn.Module):
    """Stage 3: AddFinalResultFFN + SubFinalResultFFN, plus opcode merge."""

    def __init__(self, BD, state: _AddSubGEState):
        super().__init__()
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.state = state
        self.add_layer = build_add_layers(NIBBLE, opcode=25)[2]
        self.sub_layer = build_sub_layers(NIBBLE, opcode=26)[2]

    def forward(self, x_bd):
        x_add = self.add_layer(self.state._x_ge_add)
        x_sub = self.sub_layer(self.state._x_ge_sub)

        ge = self.ge

        # Opcode masks (matches PureNeuralALU.forward `add_sub` branch)
        x_ge_flat = self.state._x_ge
        op_add = (x_ge_flat[:, 0, ge.OP_START + 25] > 0.1).float()
        op_sub = (x_ge_flat[:, 0, ge.OP_START + 26] > 0.1).float()
        op_total = op_add + op_sub

        x_ge_out = self.state._x_ge_out
        x_ge_out[:, :, ge.RESULT] = (
            x_add[:, :, ge.RESULT] * op_add[:, None]
            + x_sub[:, :, ge.RESULT] * op_sub[:, None]
        )

        self.state._x_ge_out = x_ge_out
        self.state._opcode_mask = op_total
        return x_bd


class _AddSubGEToBD(nn.Module):
    """Stage 4: GE → BD writeback.

    Reads RESULT from the merged GE state, applies opcode mask + AX-marker
    gating, and writes OUTPUT_LO/HI + CARRY into x_bd. Returns the updated
    x_bd as the FFN's output (matches the original ALUAddSub.forward
    return signature).
    """

    def __init__(self, BD, S, state: _AddSubGEState):
        super().__init__()
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.state = state
        self.S = S
        self.ge_to_bd = GEToBDConverter(BD, self.ge, S)

    def forward(self, x_bd):
        BD = self.BD
        B, seq_len = self.state._original_shape

        x_ge_out = self.state._x_ge_out.view(B, seq_len, 8, self.ge.DIM)
        opcode_mask = self.state._opcode_mask.view(B, seq_len)

        # Gate on MARK_AX (matches PureNeuralALU.forward)
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        # The transitional L8 block still contains legacy lookup ADD/SUB
        # units before this efficient post-op. Let this stage be authoritative
        # for active ADD/SUB markers by clearing the legacy marker result and
        # intra-byte carry lanes before GE->BD writes the neural result.
        active = opcode_mask[:, :, None]
        x_bd_clean = x_bd.clone()
        x_bd_clean[:, :, BD.OUTPUT_LO:BD.OUTPUT_LO + 16] *= (1.0 - active)
        x_bd_clean[:, :, BD.OUTPUT_HI:BD.OUTPUT_HI + 16] *= (1.0 - active)
        x_bd_clean[:, :, BD.CARRY:BD.CARRY + 3] *= (1.0 - active)

        x_bd_out = self.ge_to_bd(x_ge_out, x_bd_clean, opcode_mask=opcode_mask)
        return x_bd_out

    # Stub methods for compatibility with vm_step.py model utilities.
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


def make_addsub_stage_modules(BD, S):
    """Construct the 5-stage pipeline + shared state.

    Returns a tuple of (state, stage0, stage1, stage2, stage3, stage4).
    Each stage references the shared state container.
    """
    state = _AddSubGEState()
    stage0 = _AddSubBDToGE(BD, state)
    stage1 = _AddSubStage1(state)
    stage2 = _AddSubStage2(state)
    stage3 = _AddSubStage3(BD, state)
    stage4 = _AddSubGEToBD(BD, S, state)
    return state, stage0, stage1, stage2, stage3, stage4


class AddSub5StageBlock(nn.Module):
    """Composite wrapper holding all 5 add/sub stages.

    Used as a transitional container by `set_vm_weights` so the wrapper
    can be installed in one place. `_expand_wrapper_blocks` splits this
    into 5 successive single-stage blocks at runtime, eliminating the
    composite wrapper from the final model topology.

    Mathematically equivalent to ALUAddSub.forward (same computation,
    just split across 5 stage modules sharing state).

    Vanilla refactor (2026-05-10): forward is now an ``nn.Sequential``
    composition over the 5 stages. Each stage's signature is
    ``x_bd -> x_bd`` (residual-identity for stages 0-3; stage 4 returns
    the BD with OUTPUT/CARRY written), so the BD residual is the only
    tensor threaded by Sequential. The shared ``_AddSubGEState`` lives
    as a side channel: each stage holds a reference to the same state
    object, so Sequential does not need to forward it through the call
    chain. This keeps the standard ``nn.Sequential`` semantics while
    preserving the cross-stage GE workspace.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        state, s0, s1, s2, s3, s4 = make_addsub_stage_modules(BD, S)
        self.state = state
        # Vanilla nn.Sequential pipeline over the 5 stages. The shared
        # `_AddSubGEState` reference each stage carries internally is the
        # side channel — Sequential only forwards the BD tensor.
        self.pipeline = nn.Sequential(s0, s1, s2, s3, s4)

    @property
    def stages(self):
        """Back-compat: expose the 5 sub-stages as an indexable sequence.

        Older code paths (and the original ModuleList-based implementation)
        walked `block.ffn.stages[i]`. The canonical container is now the
        `nn.Sequential` pipeline; expose it under the historical name to
        avoid duplicate parameter registration.
        """
        return self.pipeline

    def forward(self, x):
        return self.pipeline(x)

    # Compatibility stubs (mirror PureNeuralALU)
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class DeclarativeAddSubBlock(nn.Module):
    """Declarative byte-0 ADD/SUB composite — two rule-derived FFN passes.

    The migration replacement for ``AddSub5StageBlock`` (2026-06-14). Holds
    two ``PureFFN`` modules, both lowered purely from
    ``wide_alu_dsl.wide_add_rules`` / ``wide_sub_rules`` (NO imperative
    BDToGEConverter / GE add-sub layers / GEToBDConverter forward compute):

      * ``lo_ffn`` (pass 1): byte-0 LOW nibble add/sub lookup. Reads operand A
        from ALU_LO, operand B from AX_CARRY_LO; writes the lo result to
        OUTPUT_LO and the inter-nibble carry/borrow to CARRY+0.
      * ``hi_ffn`` (pass 2): byte-0 HIGH nibble lookup. Reads ALU_HI /
        AX_CARRY_HI + the CARRY+0 carry-in written by pass 1; writes the hi
        result to OUTPUT_HI and the byte-level overflow/borrow to CARRY+1
        (ADD) / CARRY+2 (SUB) for the downstream L10 CarryPropagationPostOp.

    A single FFN forward cannot self-cascade the inter-nibble carry (``W_up``
    only reads the input residual), so the two passes run SEQUENTIALLY inside
    this one block's forward — exactly the way the imperative
    ``AddSub5StageBlock`` ran its 5-stage ``nn.Sequential`` inside one block.
    Keeping it ONE block preserves the model's physical block count (so
    absolute-position-dependent ops like lea are unaffected).

    The OUTPUT result is written at a DOMINANT amplitude so the correct cell
    out-votes the downstream block-N L9 ALU_LO->OUTPUT_LO leak (the folded-in
    +5 fix). Both FFNs are residual-add ``PureFFN``s, so the composite forward
    is just ``hi_ffn(lo_ffn(x))``.
    """

    def __init__(self, lo_ffn: nn.Module, hi_ffn: nn.Module,
                 cleanup_ffn: nn.Module = None):
        super().__init__()
        # Optional operand-cleanup pre-pass: subtracts the constant
        # operand-gather artifacts (index-0 ~5.4 etc.) so the lo/hi lookups
        # read clean one-hots on the model's DIRTY MARK_AX operand bands
        # (the same div-style cleanup pattern the bitwise wrap uses).
        self.cleanup_ffn = cleanup_ffn
        self.lo_ffn = lo_ffn
        self.hi_ffn = hi_ffn
        # Mark so the L8 wrap's idempotent guard recognises an existing attach.
        self._is_addsub_decl_wrap = True

    def forward(self, x):
        # (Optional) operand cleanup -> pass 1 (lo nibble + CARRY+0) ->
        # pass 2 (hi nibble reading CARRY+0 + CARRY+1/2). Each PureFFN adds
        # its writes to the residual.
        if self.cleanup_ffn is not None:
            x = self.cleanup_ffn(x)
        return self.hi_ffn(self.lo_ffn(x))

    # Compatibility stubs (mirror PureNeuralALU / AddSub5StageBlock).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass
