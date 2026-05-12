"""Split EfficientDivMod_Neural (ALUDivMod) into compiler-installed sub-stages.

Phase 0 (2026-05-10): the monolithic ``ALUDivMod`` runtime wrapper (alias
``EfficientDivMod_Neural``) is the LAST surviving ALU wrapper. This module
provides ``FlattenedDivMod`` — a composite ``nn.Module`` whose internal
sub-stages are installed by 3 phase-ordered compiler ops (one per stage)
plus an ``install`` op that swaps the composite into ``model.blocks[10].
post_ops[-1]``.

Stages:
  0. ``bd_to_ge``          — ``BDToGEConverter`` (BD one-hot → GE scalar). Phase=10.0.
  1. ``div_layers`` /
     ``mod_layers``        — long-division pipeline:
                                 ClearDivSlots → LongDivision → EmitDivResult.
                                 Built via ``build_div_layers`` /
                                 ``build_mod_layers`` (which now dispatch to
                                 ``divmod_longdiv.build_*_layers_longdiv``
                                 since commit ``dd71fe7`` replaced fp64 MAGIC
                                 with nibble-level long division). Phase=10.1.
  2. ``ge_to_bd``           — ``GEToBDConverter`` (GE result → BD one-hot
                                 OUTPUT_LO/HI), opcode-mask + AX-marker
                                 gated. Phase=10.2.

The install op (``make_l10_alu_divmod_install_op``, kind="block",
layer_idx=10, phase=10.8) runs AFTER ``make_l10_post_op_attach_op``
(phase=10.7) has populated ``model.blocks[10].post_ops``, and replaces
the final post_op with the fully-constructed composite.

Forward semantics ("vanilla" 2026-05-10):
  ``FlattenedDivMod.forward`` is now an ``nn.Sequential`` chain over 4
  stage modules sharing per-forward state:

    Stage 0 — ``_DivModBDToGEStage``:    BD→GE projection; stashes the
                                          flattened GE buffer into a
                                          ``_DivModPipelineState``.
    Stage 1 — ``_DivModDivPipelineStage``: clones GE state, runs the 3
                                          DIV sub-FFNs (ClearDivSlots,
                                          LongDivisionModule, EmitDiv).
    Stage 2 — ``_DivModModPipelineStage``: clones GE state, runs the 3
                                          MOD sub-FFNs (analogous, emits
                                          remainder).
    Stage 3 — ``_DivModGEToBDStage``:    opcode-merges DIV/MOD RESULT,
                                          AX-marker gates, GE→BD writeback.

  Identity-passthrough (returns ``x_bd`` unchanged) is used between
  stages 0/1/2; stage 3 returns the final BD tensor. State flows via a
  shared ``_DivModPipelineState`` container, mirroring the pattern used
  by ``ALUShiftComposite`` and ``AddSub5StageBlock``.

  The internal long-division iterations (8 outer × 3 sub-FFN-equivalent
  ops) remain wrapped inside ``LongDivisionModule`` rather than being
  promoted to top-level transformer blocks: 24+ separate blocks would
  bloat the residual stream and the compiler dependency graph without
  any computational benefit since the iterations are tightly coupled.

Forward output is byte-identical to the previous
``EfficientDivMod_Neural`` / ``ALUDivMod.forward`` /
``PureNeuralALU(operations='div_mod').forward`` — same ops, same tensor
shapes, same intermediate threshold (0.1) for opcode normalization,
same MARK_AX-only OUTPUT gating.
"""

import torch
import torch.nn as nn

from .alu.chunk_config import NIBBLE
from .alu.ops.common import GenericE
from .alu.ops.div import build_div_layers
from .alu.ops.mod import build_mod_layers
from .efficient_alu_neural import BDToGEConverter, GEToBDConverter


class _DivModPipelineState:
    """Per-composite scratch space for the 4-stage DIV/MOD pipeline.

    Holds intermediate tensors so each ``nn.Sequential`` stage can run as
    a standalone module. Mirrors ``ShiftPipelineState`` and the
    ``_AddSubGEState`` containers used by other flattened ALU composites.

    Lifecycle: stage 0 populates everything; stages 1/2 read ``x_ge_flat``
    and write ``x_div`` / ``x_mod``; stage 3 consumes all of the above and
    returns the final BD tensor.
    """

    def __init__(self):
        self.x_bd_in = None       # [B, seq, d_model] original input
        self.x_ge = None          # [B, seq, 8, 160]
        self.x_ge_flat = None     # [B*seq, 8, 160]
        self.x_div = None         # [B*seq, 8, 160]  — DIV pipeline result
        self.x_mod = None         # [B*seq, 8, 160]  — MOD pipeline result
        self.original_shape = None  # (B, seq_len)


class _DivModBDToGEStage(nn.Module):
    """Stage 0: BD → GenericE format conversion.

    Wraps ``BDToGEConverter`` (mathematically a linear projection +
    opcode normalization). Side effect: populates the shared state
    container with the GE buffer that downstream stages will read.

    Returns ``x_bd`` unchanged (residual identity through the rest of
    the chain).
    """

    def __init__(self, S, BD, state: _DivModPipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        self.bd_to_ge = BDToGEConverter(BD, self.ge)

    def forward(self, x_bd):
        x_ge = self.bd_to_ge(x_bd)  # [B, seq, 8, 160]
        B, seq_len, _, _ = x_ge.shape
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)
        self.state.x_bd_in = x_bd
        self.state.x_ge = x_ge
        self.state.x_ge_flat = x_ge_flat
        self.state.original_shape = (B, seq_len)
        return x_bd


class _DivModDivPipelineStage(nn.Module):
    """Stage 1: DIV long-division pipeline.

    Clones the GE state and runs the 3-layer DIV pipeline
    (ClearDivSlotsFFN → LongDivisionModule → EmitDivResultModule for
    opcode 31). The ``LongDivisionModule`` itself does 8 outer × 3
    sub-FFN-equivalent ops internally — those iterations are kept
    inside the module since they are tightly coupled and cannot be
    expressed as independent transformer blocks without 24+ residual
    crossings.

    Returns ``x_bd`` unchanged.
    """

    def __init__(self, S, BD, state: _DivModPipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        self.div_layers = nn.ModuleList(build_div_layers(NIBBLE, opcode=31))

    def forward(self, x_bd):
        x_div = self.state.x_ge_flat.clone()
        for layer in self.div_layers:
            x_div = layer(x_div)
        self.state.x_div = x_div
        return x_bd


class _DivModModPipelineStage(nn.Module):
    """Stage 2: MOD long-division pipeline.

    Clones the GE state and runs the 3-layer MOD pipeline
    (ClearDivSlotsFFN → LongDivisionModule → EmitDivResultModule with
    ``emit_remainder=True`` for opcode 32). Same internal structure as
    DIV stage, distinct opcode and final-emit slot.

    Returns ``x_bd`` unchanged.
    """

    def __init__(self, S, BD, state: _DivModPipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        self.mod_layers = nn.ModuleList(build_mod_layers(NIBBLE, opcode=32))

    def forward(self, x_bd):
        x_mod = self.state.x_ge_flat.clone()
        for layer in self.mod_layers:
            x_mod = layer(x_mod)
        self.state.x_mod = x_mod
        return x_bd


class _DivModGEToBDStage(nn.Module):
    """Stage 3: opcode-merge DIV/MOD RESULT, AX-marker gate, GE→BD writeback.

    Reads ``state.x_ge_flat`` (original GE state for opcode flags),
    ``state.x_div`` and ``state.x_mod`` (from stages 1/2). Threshold-
    normalizes opcode flags (>0.1 to match
    ``PureNeuralALU.forward``'s div_mod branch), opcode-merges
    RESULT, restricts to MARK_AX positions, then runs
    ``GEToBDConverter`` to write OUTPUT_LO/HI back into BD format.

    Returns the updated BD tensor — this is the only stage that returns
    a value other than the unchanged input.
    """

    def __init__(self, S, BD, state: _DivModPipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        self.ge_to_bd = GEToBDConverter(BD, self.ge, S)

    def forward(self, x_bd):
        BD = self.BD
        ge = self.ge

        x_ge_flat = self.state.x_ge_flat
        x_div = self.state.x_div
        x_mod = self.state.x_mod
        B, seq_len = self.state.original_shape

        x_ge_out = x_ge_flat.clone()

        # Normalize opcode values to 0/1 (BDToGEConverter writes opcode flags
        # at variable magnitudes; threshold at 0.1 to match
        # PureNeuralALU.forward's div_mod branch).
        op_div = (x_ge_flat[:, 0, ge.OP_START + 31] > 0.1).float()
        op_mod = (x_ge_flat[:, 0, ge.OP_START + 32] > 0.1).float()
        op_total = op_div + op_mod

        # Opcode-merge RESULT (matches PureNeuralALU div_mod branch exactly).
        x_ge_out[:, :, ge.RESULT] = (
            x_div[:, :, ge.RESULT] * op_div[:, None] +
            x_mod[:, :, ge.RESULT] * op_mod[:, None]
        )

        # Reshape back.
        x_ge_out = x_ge_out.view(B, seq_len, 8, ge.DIM)
        opcode_mask = op_total.view(B, seq_len)

        # Only write OUTPUT at AX marker positions (MARK_AX > 0.5). Without
        # this, the ALU writes result "0" at byte positions where operands
        # are zero, corrupting the passthrough from L10 head 1.
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        x_bd_out = self.ge_to_bd(x_ge_out, x_bd, opcode_mask=opcode_mask)
        return x_bd_out


class FlattenedDivMod(nn.Module):
    """Flattened (compiler-baked) DIV/MOD ALU — vanilla nn.Sequential form.

    Byte-identical to ``ALUDivMod.forward`` (=
    ``PureNeuralALU(operations='div_mod').forward``) but exposes the
    BD↔GE converters and the 3-layer long-division pipeline (per opcode)
    as 4 individually-installable stage submodules so the unified
    compiler can bake each stage as a discrete ``Operation``.

    Sub-stages, installed by 3 + 1 compiler ops:

      - ``bd_to_ge``       ``BDToGEConverter``                 — phase=10.0
      - ``div_layers`` +
        ``mod_layers``     ``build_div_layers(NIBBLE, 31)`` /
                            ``build_mod_layers(NIBBLE, 32)``    — phase=10.1
      - ``ge_to_bd``       ``GEToBDConverter``                 — phase=10.2

    Plus a kind="block" install op at phase=10.8 (after
    ``make_l10_post_op_attach_op`` at 10.7) that swaps the composite into
    ``model.blocks[10].post_ops``. Forward is byte-identical to the
    previous ``EfficientDivMod_Neural`` instance.

    Operates on BD residual stream (same shape contract as the previous
    runtime wrapper) since this is installed as a ``post_op`` (consumed
    by ``TransformerBlock.forward`` as ``x = post_op(x)`` after the FFN).

    Vanilla (2026-05-10): forward is realized as an ``nn.Sequential``
    chain over 4 stage modules (``_DivModBDToGEStage``,
    ``_DivModDivPipelineStage``, ``_DivModModPipelineStage``,
    ``_DivModGEToBDStage``) sharing a ``_DivModPipelineState`` container.

    Why ``nn.Sequential`` (Option B) rather than residual-stream split
    (Option A): the long-division iterations (8 outer × 3 sub-FFN-
    equivalent ops) are tightly coupled and require a 9-nibble
    accumulator that lives only in the GE workspace. Promoting them to
    24+ separate transformer blocks would (a) require routing the
    accumulator through the 512-dim BD residual stream every step, and
    (b) bloat the compiler dependency graph with no benefit. Stage
    granularity stops at the 4 outer phases (BDToGE / DIV pipeline /
    MOD pipeline / merge+GEToBD); the long-division module remains a
    single ``nn.Module`` internally.

    Backward-compatibility: ``self.bd_to_ge``, ``self.div_layers``,
    ``self.mod_layers``, ``self.ge_to_bd`` remain accessible as
    properties (delegating to the corresponding stage modules) so
    callers that introspect the composite — e.g. compiler ops asserting
    install state — still work.
    """

    # Stage names (in pipeline order). Used as keys into ``self.stages``
    # (an ``nn.ModuleDict``) and as the attribute order of the
    # ``nn.Sequential`` rebuild.
    _STAGE_NAMES = ('bdtoge', 'div', 'mod', 'getobd')

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)

        # Shared state container (per-forward scratch).
        self._state = _DivModPipelineState()

        # Single source of truth for the stage modules. Sole owner of the
        # parameters — avoids double-registration when ``self.pipeline``
        # (the ``nn.Sequential``) is also assembled below: ``Sequential``
        # references ``self.stages``' members by identity, but
        # ``Sequential`` itself is held under a non-tracked attribute
        # (``__dict__``) so PyTorch's ``__setattr__`` doesn't add it as a
        # second child module.
        self.stages = nn.ModuleDict()

        # nn.Sequential view (assembled once all 4 stages are installed).
        # Stored via __dict__ to bypass nn.Module's auto-registration —
        # the underlying stage modules are already owned by ``stages``.
        self.__dict__['pipeline'] = None

    # --- Compatibility properties -------------------------------------
    # Pre-vanilla, callers accessed ``self.bd_to_ge`` / ``self.div_layers``
    # / ``self.mod_layers`` / ``self.ge_to_bd`` directly. The vanilla
    # form keeps those readable by delegating to the stage modules.

    @property
    def bd_to_ge(self):
        if 'bdtoge' not in self.stages:
            return None
        return self.stages['bdtoge'].bd_to_ge

    @property
    def ge_to_bd(self):
        if 'getobd' not in self.stages:
            return None
        return self.stages['getobd'].ge_to_bd

    @property
    def div_layers(self):
        if 'div' not in self.stages:
            return nn.ModuleList()
        return self.stages['div'].div_layers

    @property
    def mod_layers(self):
        if 'mod' not in self.stages:
            return nn.ModuleList()
        return self.stages['mod'].mod_layers

    # --- Per-stage installers (called by compiler ops) -----------------

    def _maybe_build_sequential(self):
        """Assemble the ``nn.Sequential`` view once all 4 stages are installed.

        Idempotent: callable repeatedly. Only sets ``self.pipeline``
        when every stage slot is populated; partial bakes leave it
        None and forward() will raise with a missing-stage list.

        The ``Sequential`` is stored in ``self.__dict__`` (not as an
        ``nn.Module`` attribute) to avoid double-registering the stage
        modules — they are already owned by ``self.stages``.
        """
        if all(name in self.stages for name in self._STAGE_NAMES):
            self.__dict__['pipeline'] = nn.Sequential(
                *(self.stages[name] for name in self._STAGE_NAMES)
            )
        else:
            self.__dict__['pipeline'] = None

    def install_bdtoge(self):
        """phase=10.0: install BD → GE converter stage.

        Equivalent to constructing ``self.bd_to_ge`` inside
        ``PureNeuralALU.__init__(operations='div_mod')``. In the vanilla
        form this builds ``_DivModBDToGEStage`` and (re)assembles the
        ``nn.Sequential`` view if all 4 stages are present.
        """
        self.stages['bdtoge'] = _DivModBDToGEStage(self.S, self.BD, self._state)
        self._maybe_build_sequential()

    def install_longdiv(self):
        """phase=10.1: install both DIV and MOD long-division pipelines.

        Builds ``_DivModDivPipelineStage`` (opcode 31) and
        ``_DivModModPipelineStage`` (opcode 32). Each stage owns its
        own ``ModuleList`` so re-installing one doesn't disturb the
        other.

        Each stage's ``forward`` runs the 3-layer pipeline
        (ClearDivSlots → LongDivisionModule → EmitDivResult) on a clone
        of the shared GE buffer.
        """
        # Re-install resets, so a re-bake doesn't accumulate state.
        self.stages['div'] = _DivModDivPipelineStage(self.S, self.BD, self._state)
        self.stages['mod'] = _DivModModPipelineStage(self.S, self.BD, self._state)
        self._maybe_build_sequential()

    def install_getobd(self):
        """phase=10.2: install GE → BD converter stage.

        Equivalent to constructing ``self.ge_to_bd`` inside
        ``PureNeuralALU.__init__(operations='div_mod')``. In the vanilla
        form this builds ``_DivModGEToBDStage`` and (re)assembles the
        ``nn.Sequential`` view if all 4 stages are present.
        """
        self.stages['getobd'] = _DivModGEToBDStage(self.S, self.BD, self._state)
        self._maybe_build_sequential()

    # --- Forward (byte-identical to PureNeuralALU div_mod branch) -----

    def forward(self, x_bd):
        pipeline = self.__dict__.get('pipeline')
        if pipeline is None:
            missing = [name for name in self._STAGE_NAMES if name not in self.stages]
            # Translate internal stage names back to the historical
            # field names used by the previous installer-pattern
            # FlattenedDivMod for parity with diagnostic messages.
            translation = {
                'bdtoge': 'bd_to_ge',
                'div': 'div_layers',
                'mod': 'mod_layers',
                'getobd': 'ge_to_bd',
            }
            missing_legacy = [translation[m] for m in missing]
            raise RuntimeError(
                f"FlattenedDivMod: missing stages {missing_legacy}. "
                "All 3 stage compiler ops (phase 10.0/10.1/10.2) plus "
                "the install op (phase 10.8) must run before forward()."
            )

        # ---- Opcode-gated early-out (perf optimization, 2026-05-11) ----
        # The full long-division pipeline (~370 ms/call dominated by an
        # 8-outer × 15-inner Python loop inside LongDivisionModule) runs
        # unconditionally and is masked to zero at the GE→BD stage when
        # neither OP_DIV nor OP_MOD is active. For the vast majority of
        # tests, neither opcode is active, so this work is pure waste.
        #
        # Stage 3 (`_DivModGEToBDStage`) ultimately gates the writeback on
        # `(OP_DIV>0.1 OR OP_MOD>0.1) AND MARK_AX>0.5` per sequence
        # position. If neither flag exceeds 0.1 anywhere in the batch,
        # the pipeline contributes nothing and we can return x_bd
        # unchanged. The `.item()` forces one CPU/GPU sync (~few µs) — a
        # rounding error compared to the ~370 ms loop it elides.
        #
        # Correctness: this is a strict subset of the mask already
        # applied at stage 3 (we only skip when the mask would zero the
        # contribution everywhere). Numerical output is identical when
        # DIV/MOD is active.
        #
        # ONNX export & torch.compile: the `.item()` calls would force a
        # CPU/GPU sync and graph break. Under tracing or compilation we
        # always run the full pipeline; the stage-3 mask zeroes the
        # writeback correctly when DIV/MOD is inactive, so output stays
        # byte-identical. Per docs/ONNX_EXPORT_STATUS_2026_05_11.md
        # blocker 1.
        if torch.onnx.is_in_onnx_export() or torch.compiler.is_compiling():
            return pipeline(x_bd)

        BD = self.BD
        op_div_max = x_bd[..., BD.OP_DIV].max()
        op_mod_max = x_bd[..., BD.OP_MOD].max()
        if float(op_div_max.item()) < 0.1 and float(op_mod_max.item()) < 0.1:
            return x_bd

        return pipeline(x_bd)

    # Stub methods for compatibility with vm_step.py (mirror PureNeuralALU).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass
