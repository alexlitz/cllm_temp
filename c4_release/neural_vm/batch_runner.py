"""
Batched Speculative Runner for Neural VM

Processes multiple programs in parallel using batch dimension.

Key optimizations:
1. Batch dimension: Run N programs simultaneously
2. Speculative execution: DraftVM + transformer validation
3. Batch validation: Verify N programs in one forward pass

Expected speedup: N * 10-35x (where N is batch size)
"""

import torch
from typing import List, Tuple, Optional
from .vm_step import AutoregressiveVM, Token, set_vm_weights
from .embedding import Opcode
from .speculative import DraftVM


class BatchedSpeculativeRunner:
    """
    Batched speculative VM runner.

    Runs multiple programs in parallel using the batch dimension.
    Each program has its own DraftVM for fast execution.

    Example:
        runner = BatchedSpeculativeRunner(batch_size=4)
        # Run 4 programs at once
        bytecodes = [[...], [...], [...], [...]]
        results = runner.run_batch(bytecodes)
    """

    def __init__(
        self,
        batch_size: int = 4,
        d_model=512,
        n_layers=16,
        n_heads=8,
        ffn_hidden=4096,
        max_seq_len=4096,
        validate_every=1,
    ):
        self.batch_size = batch_size
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

        # Per-program state
        self.draft_vms = []
        self.contexts = []
        self.validations = 0
        self.mismatches = 0
        self.total_steps = 0

    def run_batch(
        self,
        bytecodes: List[List[int]],
        data_list: Optional[List[bytes]] = None,
        argv_list: Optional[List[List[str]]] = None,
        max_steps: int = 10000,
    ) -> List[Tuple[str, int]]:
        """
        Run multiple bytecodes in parallel.

        Args:
            bytecodes: List of bytecode lists
            data_list: Optional list of data sections
            argv_list: Optional list of argv lists
            max_steps: Maximum steps per program

        Returns:
            List of (output_string, exit_code) tuples
        """
        if len(bytecodes) > self.batch_size:
            raise ValueError(f"Too many programs: {len(bytecodes)} > batch_size {self.batch_size}")

        # Initialize per-program state
        self.draft_vms = [DraftVM(bc) for bc in bytecodes]
        self.contexts = [
            self._build_context(bc, (data_list or [b''])[i], (argv_list or [[]])[i])
            for i, bc in enumerate(bytecodes)
        ]

        results = []
        active = [True] * len(bytecodes)

        step = 0
        while step < max_steps and any(active):
            # 1. Execute active DraftVMs
            draft_tokens_batch = []
            for i, vm in enumerate(self.draft_vms):
                if active[i]:
                    if vm.step():
                        draft_tokens_batch.append(vm.draft_tokens())
                    else:
                        active[i] = False
                        draft_tokens_batch.append([Token.HALT] * 35)
                else:
                    draft_tokens_batch.append([Token.HALT] * 35)

            # 2. Batch validate with transformer
            if step % self.validate_every == 0:
                accepted_batch = self._validate_batch(draft_tokens_batch)
                self.validations += 1
                for i, accepted in enumerate(accepted_batch):
                    if accepted < 35:
                        self.mismatches += 1
            else:
                accepted_batch = [35] * len(bytecodes)

            # 3. Update contexts
            for i, (draft, accepted) in enumerate(zip(draft_tokens_batch, accepted_batch)):
                if accepted == 35:
                    self.contexts[i].extend(draft)
                elif active[i]:
                    # Partial acceptance - add what was accepted
                    self.contexts[i].extend(draft[:accepted])

            step += 1
            self.total_steps = step

        # Collect results
        return [("", vm.ax) for vm in self.draft_vms]

    def _validate_batch(self, draft_tokens_batch: List[List[int]]) -> List[int]:
        """
        Validate draft tokens for all programs in ONE forward pass.

        Args:
            draft_tokens_batch: List of draft token lists (one per program)

        Returns:
            List of accepted token counts (one per program)
        """
        # Build batch input for transformer
        # Contexts with draft tokens appended
        contexts_with_draft = []
        draft_lens = []
        for i, (ctx, draft) in enumerate(zip(self.contexts, draft_tokens_batch)):
            contexts_with_draft.append(ctx + draft)
            draft_lens.append(len(draft))

        # Use the model's batch verification method
        accepted_batch = self.model.verify_speculative_batch(contexts_with_draft, draft_lens)

        # Update contexts with accepted tokens
        for i, (ctx, draft, accepted) in enumerate(zip(self.contexts, draft_tokens_batch, accepted_batch)):
            self.contexts[i] = ctx + draft[:accepted]

        return accepted_batch

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
        match_rate = 1.0 - (self.mismatches / max(1, self.validations * self.batch_size))
        return {
            "total_steps": self.total_steps,
            "validations": self.validations,
            "mismatches": self.mismatches,
            "batch_size": self.batch_size,
            "match_rate": match_rate,
        }


def create_batched_runner(batch_size: int = 4, **kwargs) -> BatchedSpeculativeRunner:
    """Create a batched speculative runner."""
    return BatchedSpeculativeRunner(batch_size=batch_size, **kwargs)
