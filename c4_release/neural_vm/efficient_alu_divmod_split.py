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

Forward semantics are byte-identical to ``ALUDivMod.forward`` (=
``PureNeuralALU(operations='div_mod').forward``):
  * BD → GE projection
  * Two parallel pipelines (DIV opcode 31, MOD opcode 32) on cloned GE state
  * Opcode-mask merge of RESULT (op_div + op_mod, normalized at threshold 0.1)
  * AX-marker gate restricts OUTPUT writes
  * GE → BD writeback via ``GEToBDConverter`` (sets OUTPUT_LO/HI, no
    CARRY for DIV/MOD since they don't trigger ADD-carry / SUB-borrow).
"""

import torch
import torch.nn as nn

from .alu.chunk_config import NIBBLE
from .alu.ops.common import GenericE
from .alu.ops.div import build_div_layers
from .alu.ops.mod import build_mod_layers
from .efficient_alu_neural import BDToGEConverter, GEToBDConverter


class FlattenedDivMod(nn.Module):
    """Flattened (compiler-baked) DIV/MOD ALU.

    Byte-identical to ``ALUDivMod.forward`` (=
    ``PureNeuralALU(operations='div_mod').forward``) but exposes the
    BD↔GE converters and the 3-layer long-division pipeline (per opcode)
    as individually-installable submodules so the unified compiler can
    bake each stage as a discrete ``Operation``.

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
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)

        # Sub-stages filled in by the 3 stage installer ops. Each
        # ``install_*`` mutates one slot. Forward raises if any is
        # missing so a partial bake fails loudly.
        self.bd_to_ge = None        # phase=10.0 — BDToGEConverter
        self.div_layers = nn.ModuleList()  # phase=10.1 — long-division DIV pipeline
        self.mod_layers = nn.ModuleList()  # phase=10.1 — long-division MOD pipeline
        self.ge_to_bd = None        # phase=10.2 — GEToBDConverter

    # --- Per-stage installers (called by compiler ops) -----------------

    def install_bdtoge(self):
        """phase=10.0: install BD → GE converter.

        Equivalent to constructing ``self.bd_to_ge`` inside
        ``PureNeuralALU.__init__(operations='div_mod')``.
        """
        self.bd_to_ge = BDToGEConverter(self.BD, self.ge)

    def install_longdiv(self):
        """phase=10.1: install the 3-layer long-division pipeline.

        Builds DIV and MOD pipelines via ``build_div_layers`` /
        ``build_mod_layers`` which (since commit ``dd71fe7``) dispatch
        to the nibble-level long-division pipeline:
          ClearDivSlotsFFN → LongDivisionModule → EmitDivResultModule.

        Both pipelines share the GE buffer at runtime (each is run on
        a clone of the post-BD-to-GE state), then opcode-merged in the
        forward.
        """
        # Re-install resets, so a re-bake doesn't accumulate state.
        self.div_layers = nn.ModuleList(build_div_layers(NIBBLE, opcode=31))
        self.mod_layers = nn.ModuleList(build_mod_layers(NIBBLE, opcode=32))

    def install_getobd(self):
        """phase=10.2: install GE → BD converter.

        Equivalent to constructing ``self.ge_to_bd`` inside
        ``PureNeuralALU.__init__(operations='div_mod')``.
        """
        self.ge_to_bd = GEToBDConverter(self.BD, self.ge, self.S)

    # --- Forward (byte-identical to PureNeuralALU div_mod branch) -----

    def forward(self, x_bd):
        if (self.bd_to_ge is None or self.ge_to_bd is None or
                len(self.div_layers) == 0 or len(self.mod_layers) == 0):
            missing = []
            if self.bd_to_ge is None:
                missing.append('bd_to_ge')
            if len(self.div_layers) == 0:
                missing.append('div_layers')
            if len(self.mod_layers) == 0:
                missing.append('mod_layers')
            if self.ge_to_bd is None:
                missing.append('ge_to_bd')
            raise RuntimeError(
                f"FlattenedDivMod: missing stages {missing}. "
                "All 3 stage compiler ops (phase 10.0/10.1/10.2) plus "
                "the install op (phase 10.8) must run before forward()."
            )

        B, seq_len, _ = x_bd.shape
        BD = self.BD

        # Convert BD → GE format.
        x_ge = self.bd_to_ge(x_bd)  # [B, seq_len, 8, 160]

        # Flatten for efficient layer processing.
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)  # [B*seq_len, 8, 160]

        x_ge_out = x_ge_flat.clone()

        # Run DIV pipeline on a clone.
        x_div = x_ge_flat.clone()
        for layer in self.div_layers:
            x_div = layer(x_div)

        # Run MOD pipeline on a clone.
        x_mod = x_ge_flat.clone()
        for layer in self.mod_layers:
            x_mod = layer(x_mod)

        # Normalize opcode values to 0/1 (BDToGEConverter writes opcode flags
        # at variable magnitudes; threshold at 0.1 to match
        # PureNeuralALU.forward's div_mod branch).
        op_div = (x_ge_flat[:, 0, self.ge.OP_START + 31] > 0.1).float()
        op_mod = (x_ge_flat[:, 0, self.ge.OP_START + 32] > 0.1).float()

        op_total = op_div + op_mod

        # Opcode-merge RESULT (matches PureNeuralALU div_mod branch exactly).
        x_ge_out[:, :, self.ge.RESULT] = (
            x_div[:, :, self.ge.RESULT] * op_div[:, None] +
            x_mod[:, :, self.ge.RESULT] * op_mod[:, None]
        )

        opcode_mask_flat = op_total

        # Reshape back.
        x_ge_out = x_ge_out.view(B, seq_len, 8, self.ge.DIM)

        opcode_mask = opcode_mask_flat.view(B, seq_len)

        # Only write OUTPUT at AX marker positions (MARK_AX > 0.5). Without
        # this, the ALU writes result "0" at byte positions where operands
        # are zero, corrupting the passthrough from L10 head 1.
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        x_bd_out = self.ge_to_bd(x_ge_out, x_bd, opcode_mask=opcode_mask)

        return x_bd_out

    # Stub methods for compatibility with vm_step.py (mirror PureNeuralALU).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass
