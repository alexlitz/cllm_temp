"""
Semantic Metadata Extraction for KV Cache Eviction.

Extracts semantic information from VM execution context to enable
smart eviction policies based on VM semantics.
"""

import torch
from typing import List, Dict
from .embedding import E


def extract_semantic_metadata(context_tokens: List[int]) -> List[Dict]:
    """
    Extract semantic metadata from VM execution context tokens.

    For each token position, determines:
    - Type: memory_write, memory_read, io, register, other
    - Address: For memory operations
    - Register: For register operations
    - Value: For zero-write detection

    Args:
        context_tokens: List of token IDs from VM execution

    Returns:
        List of metadata dicts, one per token:
        {
            'type': str,  # 'memory_write', 'memory_read', 'io', 'register', 'other'
            'address': int (optional),
            'register': str (optional),
            'value': int (optional),
        }
    """
    # For now, return simple metadata
    # This is a placeholder - full implementation would decode the embedding
    # to extract actual semantic information

    # Each VM step generates 35 tokens in our current setup
    # We group tokens and analyze semantics

    metadata = []

    for i, token in enumerate(context_tokens):
        meta = {'type': 'other'}

        # Simple heuristic: Analyze token position in sequence
        # Real implementation would decode embedding values

        # Position in current VM step (35 tokens per step)
        step_pos = i % 35

        # Heuristic classification based on position
        # (This is simplified - real version would decode embedding)

        if step_pos >= 0 and step_pos < 8:
            # Arithmetic unit tokens
            meta['type'] = 'other'
        elif step_pos >= 8 and step_pos < 16:
            # Memory/register tokens
            meta['type'] = 'register'
            meta['register'] = f'pos_{step_pos}'
        elif step_pos >= 16 and step_pos < 24:
            # Could be memory operations
            meta['type'] = 'memory_read'
        elif step_pos >= 24 and step_pos < 32:
            # Could be I/O
            meta['type'] = 'io'

        metadata.append(meta)

    return metadata


def extract_semantic_metadata_from_embedding(
    embeddings: torch.Tensor,
    batch_idx: int = 0
) -> List[Dict]:
    """
    Extract semantic metadata from embedding tensor.

    Args:
        embeddings: Embedding tensor [batch, seq_len, d_model]
        batch_idx: Which batch element to analyze

    Returns:
        List of metadata dicts, one per sequence position
    """
    seq_len = embeddings.shape[1]
    metadata = []

    for pos in range(seq_len):
        # Get embedding at this position
        emb = embeddings[batch_idx, pos, :]  # [d_model]

        meta = {'type': 'other'}

        # Check for memory write
        if emb.shape[0] > E.MEM_WRITE:
            mem_write = emb[E.MEM_WRITE].item()
            mem_read = emb[E.MEM_READ].item()

            if mem_write > 0.5:
                meta['type'] = 'memory_write'

                # Extract memory address (8 nibbles)
                if emb.shape[0] >= E.MEM_ADDR_BASE + 8:
                    addr_nibbles = []
                    for i in range(8):
                        nibble_val = emb[E.MEM_ADDR_BASE + i].item()
                        # Convert from normalized value to nibble (0-15)
                        nibble = int(round(nibble_val * 15 / E.SCALE))
                        addr_nibbles.append(nibble)

                    # Combine nibbles into address (little-endian)
                    address = 0
                    for i, nib in enumerate(addr_nibbles):
                        address |= (nib & 0xF) << (i * 4)
                    meta['address'] = address

                # Extract memory data to detect zero writes
                if emb.shape[0] >= E.MEM_DATA_BASE + 8:
                    data_nibbles = []
                    for i in range(8):
                        nibble_val = emb[E.MEM_DATA_BASE + i].item()
                        nibble = int(round(nibble_val * 15 / E.SCALE))
                        data_nibbles.append(nibble)

                    # Combine nibbles into value
                    value = 0
                    for i, nib in enumerate(data_nibbles):
                        value |= (nib & 0xF) << (i * 4)
                    meta['value'] = value

            elif mem_read > 0.5:
                meta['type'] = 'memory_read'

                # Extract address for reads too
                if emb.shape[0] >= E.MEM_ADDR_BASE + 8:
                    addr_nibbles = []
                    for i in range(8):
                        nibble_val = emb[E.MEM_ADDR_BASE + i].item()
                        nibble = int(round(nibble_val * 15 / E.SCALE))
                        addr_nibbles.append(nibble)

                    address = 0
                    for i, nib in enumerate(addr_nibbles):
                        address |= (nib & 0xF) << (i * 4)
                    meta['address'] = address

        # Check for I/O operations
        if emb.shape[0] > E.IO_OUTPUT_READY:
            io_output = emb[E.IO_OUTPUT_READY].item()
            io_need_input = emb[E.IO_NEED_INPUT].item()

            if io_output > 0.5 or io_need_input > 0.5:
                meta['type'] = 'io'

        # Check for AX register updates (common register)
        if emb.shape[0] >= E.AX_BASE + 8:
            # Simple heuristic: If AX values are non-zero, consider it an update
            ax_sum = sum(abs(emb[E.AX_BASE + i].item()) for i in range(8))
            if ax_sum > 0.1:
                meta['type'] = 'register'
                meta['register'] = 'AX'

                # Extract AX value to detect zero writes
                ax_nibbles = []
                for i in range(8):
                    nibble_val = emb[E.AX_BASE + i].item()
                    nibble = int(round(nibble_val * 15 / E.SCALE))
                    ax_nibbles.append(nibble)

                value = 0
                for i, nib in enumerate(ax_nibbles):
                    value |= (nib & 0xF) << (i * 4)
                meta['value'] = value

        metadata.append(meta)

    return metadata


def should_cache_token(metadata: Dict) -> bool:
    """
    Determine if a token should be cached based on metadata.

    Returns False for:
    - Zero writes (memory or register writes with value=0)
    - Certain I/O operations that are ephemeral

    Args:
        metadata: Token metadata dict

    Returns:
        True if token should be cached, False to skip
    """
    # Skip zero writes (they represent freed memory)
    if metadata.get('value') == 0:
        if metadata.get('type') in ['memory_write', 'register']:
            return False

    # Always cache reads and other operations
    return True
