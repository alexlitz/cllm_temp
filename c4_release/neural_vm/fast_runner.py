"""
Fast Speculative Runner for Neural VM

Uses DraftVM for fast execution and transformer for validation.
Expected speedup: 10-35x depending on hit rate.
"""

import torch
from typing import List, Tuple, Optional
from .vm_step import AutoregressiveVM, Token, set_vm_weights
from .embedding import Opcode
from .speculative import DraftVM


class SpeculativeRunner:
    """
    Fast speculative VM runner.

    Strategy:
    1. DraftVM executes instructions (fast Python)
    2. DraftVM produces 35 draft tokens per step
    3. Transformer validates all 35 tokens in ONE forward pass
    4. If validation passes, continue; otherwise fall back

    Expected speedup: 10-35x (35 tokens validated per forward pass)
    """

    def __init__(
        self,
        d_model=512,
        n_layers=16,
        n_heads=8,
        ffn_hidden=4096,
        max_seq_len=4096,
        validate_every=1,  # Validate every N steps
    ):
        self.validate_every = validate_every

        # Create transformer model
        self.model = AutoregressiveVM(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_hidden=ffn_hidden,
            max_seq_len=max_seq_len,
        )
        self.model.eval()
        set_vm_weights(self.model)
        self.model.compact(block_size=32)
        self.model.compact_moe()

        self.draft_vm = None
        self.context = []
        self.validations = 0
        self.mismatches = 0
        self.total_steps = 0

    def run(
        self,
        bytecode: List[int],
        data: Optional[bytes] = None,
        argv: Optional[List[str]] = None,
        max_steps: int = 10000,
    ) -> Tuple[str, int]:
        """
        Run bytecode with speculative execution.

        Returns:
            (output_string, exit_code)
        """
        self.validations = 0
        self.mismatches = 0
        self.total_steps = 0

        # Initialize DraftVM
        self.draft_vm = DraftVM(bytecode)

        # Build initial context (code + data prefix)
        self.context = self._build_context(bytecode, data or b'', argv or [])

        output = []
        step = 0

        while step < max_steps:
            # 1. DraftVM executes one instruction
            if not self.draft_vm.step():
                break  # Halted

            # 2. DraftVM produces 35 draft tokens
            draft_tokens = self.draft_vm.draft_tokens()

            # 3. Validate with transformer (periodically)
            should_validate = (step % self.validate_every == 0)
            accepted = 35  # Assume accepted if not validating

            if should_validate:
                accepted = self._validate_step(draft_tokens)
                self.validations += 1
                if accepted < 35:
                    self.mismatches += 1

            # 4. Add tokens to context
            if accepted == 35:
                self.context.extend(draft_tokens)
            else:
                # Validation failed - fall back to transformer for this step
                self._fall_back_transformer(draft_tokens[:accepted])

            step += 1
            self.total_steps = step

            # Check for halt
            if draft_tokens[-1] == Token.HALT:
                break

        return "".join(output), self.draft_vm.ax

    def _validate_step(self, draft_tokens: List[int]) -> int:
        """
        Validate draft tokens against transformer.

        Returns:
            Number of accepted tokens (0-35)
        """
        # Use the model's verification method
        accepted = self.model.verify_speculative_step(self.context, draft_tokens)

        # Add accepted tokens to context
        if accepted > 0:
            self.context.extend(draft_tokens[:accepted])

        return accepted

    def _fall_back_transformer(self, partial_tokens: List[int]):
        """Fall back to pure transformer generation."""
        # Generate remaining tokens one by one
        remaining = 35 - len(partial_tokens)
        for _ in range(remaining):
            token = self.model.generate_next(self.context)
            self.context.append(token)
            if token == Token.STEP_END:
                break

    def _build_context(
        self,
        bytecode: List[int],
        data: bytes,
        argv: List[str],
    ) -> List[int]:
        """Build initial context with bytecode and data."""
        context = []

        # Code section
        context.append(Token.CODE_START)
        for instr in bytecode:
            # Encode instruction as bytes (little-endian)
            for i in range(8):
                context.append((instr >> (i * 8)) & 0xFF)
        context.append(Token.CODE_END)

        # Data section
        context.append(Token.DATA_START)
        context.extend(list(data))
        context.append(Token.DATA_END)

        return context

    def get_stats(self) -> dict:
        """Get execution statistics."""
        match_rate = 1.0 - (self.mismatches / max(1, self.validations))
        return {
            "total_steps": self.total_steps,
            "validations": self.validations,
            "mismatches": self.mismatches,
            "match_rate": match_rate,
        }


def create_fast_runner(**kwargs) -> SpeculativeRunner:
    """Create a fast speculative runner."""
    return SpeculativeRunner(**kwargs)
