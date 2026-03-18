"""
Base neural layers for Neural VM V7.

PureFFN: SwiGLU FFN with fixed forward, subclasses bake weights
PureAttention: Attention with fixed forward, subclasses bake weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps

from .embedding import E


def sparse_linear(x, weight_sparse, bias=None):
    """F.linear drop-in for sparse COO weight matrices.

    Computes x @ weight.T + bias, where weight is sparse [out_D, D].
    Handles 2D [N, D] and 3D [B, S, D] inputs.
    """
    if x.dim() == 3:
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
    else:
        x_flat = x
    # W_sparse: [out_D, D], x_flat.T: [D, N] -> result: [out_D, N] -> [N, out_D]
    out = torch.sparse.mm(weight_sparse, x_flat.t()).t()
    if bias is not None:
        out = out + bias
    if x.dim() == 3:
        out = out.reshape(B, S, -1)
    return out


def bake_weights(method):
    """
    Decorator for _bake_weights methods.

    Wraps the method in torch.no_grad() context for efficient weight initialization.
    Also provides S (scale factor) as a convenience.

    Usage:
        @bake_weights
        def _bake_weights(self):
            S = E.SCALE
            self.W_up[0, E.NIB_A] = S
            ...
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with torch.no_grad():
            return method(self, *args, **kwargs)
    return wrapper


class PureFFN(nn.Module):
    """
    Pure SwiGLU FFN with FINAL forward pass.

    Uses nn.Linear internally for standard PyTorch patterns while
    providing direct weight access for baking via W_up, W_gate, W_down properties.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Use nn.Linear for cleaner code
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=True)

        # Additional bias parameters (not used in forward, but used for weight baking)
        self.b_up = nn.Parameter(torch.zeros(hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))

        # Zero initialize
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

        self._bake_weights()

    # Direct weight access for baking (forwards to nn.Linear weights)
    def __getattr__(self, name):
        if name in ('W_up', 'W_gate', 'W_down', 'b_up', 'b_gate', 'b_down'):
            if name == 'W_up':
                return self.up.weight.data  # Return .data to allow indexed assignment
            elif name == 'b_up':
                # Return the actual b_up parameter
                return object.__getattribute__(self, 'b_up').data
            elif name == 'W_gate':
                return self.gate.weight.data
            elif name == 'b_gate':
                # Return the actual b_gate parameter
                return object.__getattribute__(self, 'b_gate').data
            elif name == 'W_down':
                return self.down.weight.data
            elif name == 'b_down':
                return self.down.bias.data
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        # Check for weight/bias assignments (after __init__ has set up the modules)
        if name in ('W_up', 'W_gate', 'W_down', 'b_up', 'b_gate', 'b_down'):
            # Use object.__getattribute__ to avoid recursion
            if not hasattr(self, '_modules'):
                # Still in __init__, use normal setattr
                object.__setattr__(self, name, value)
                return

            # Check if this is an initial assignment (like in __init__)
            # For b_up, b_gate, b_down, they might not exist yet
            try:
                obj_attr = object.__getattribute__(self, name)
                has_attr = True
            except AttributeError:
                has_attr = False

            if not has_attr:
                # Initial assignment, use normal setattr
                object.__setattr__(self, name, value)
                return

            # Reassignment to existing parameter
            if name == 'W_up':
                self.up.weight.data = value
            elif name == 'b_up':
                object.__getattribute__(self, 'b_up').data = value
            elif name == 'W_gate':
                self.gate.weight.data = value
            elif name == 'b_gate':
                object.__getattribute__(self, 'b_gate').data = value
            elif name == 'W_down':
                self.down.weight.data = value
            elif name == 'b_down':
                self.down.bias.data = value
        else:
            super().__setattr__(name, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FINAL - Standard SwiGLU FFN. DO NOT OVERRIDE."""
        up = self.up(x)
        gate = self.gate(x)
        hidden = F.silu(up) * gate
        return x + self.down(hidden)

    def sparsify(self):
        """Convert weight matrices to COO sparse format."""
        self.up.weight = nn.Parameter(self.up.weight.data.to_sparse_coo().coalesce())
        self.gate.weight = nn.Parameter(self.gate.weight.data.to_sparse_coo().coalesce())
        self.down.weight = nn.Parameter(self.down.weight.data.to_sparse_coo().coalesce())

    def compact(self, block_size=1):
        """Prune to dense sub-matrices of only active hidden units.

        Identifies hidden units where up, gate, or down have non-zero
        weights, keeps only those rows/columns, and stores dense sub-matrices.

        Args:
            block_size: Align active units to blocks of this size.
        """
        # Get dense weights
        W_up = self.up.weight.data.to_dense() if self.up.weight.is_sparse else self.up.weight.data
        W_gate = self.gate.weight.data.to_dense() if self.gate.weight.is_sparse else self.gate.weight.data
        W_down = self.down.weight.data.to_dense() if self.down.weight.is_sparse else self.down.weight.data
        H = W_up.shape[0]

        # Find active hidden units
        active = (
            (W_up.abs().sum(dim=1) > 0)
            | (W_gate.abs().sum(dim=1) > 0)
            | (self.b_up.data.abs() > 0)
            | (self.b_gate.data.abs() > 0)
            | (W_down.abs().sum(dim=0) > 0)
        )

        if block_size > 1:
            active_blocks = set()
            for idx in active.nonzero(as_tuple=True)[0].tolist():
                active_blocks.add(idx // block_size)
            indices = []
            for blk in sorted(active_blocks):
                start = blk * block_size
                indices.extend(range(start, min(start + block_size, H)))
            active_idx = torch.tensor(indices, dtype=torch.long) if indices else torch.tensor([0], dtype=torch.long)
        else:
            active_idx = active.nonzero(as_tuple=True)[0]
            if len(active_idx) == 0:
                active_idx = torch.tensor([0], dtype=torch.long)

        n = len(active_idx)
        self._compact_size = n
        self.hidden_dim = n

        # Replace with compacted Linear layers
        self.up = nn.Linear(self.dim, n, bias=False)
        self.gate = nn.Linear(self.dim, n, bias=False)
        self.down = nn.Linear(n, self.dim, bias=True)

        # Copy weights
        self.up.weight = nn.Parameter(W_up[active_idx].contiguous())
        self.gate.weight = nn.Parameter(W_gate[active_idx].contiguous())
        self.down.weight = nn.Parameter(W_down[:, active_idx].contiguous())
        nn.init.zeros_(self.down.bias)

        # Copy biases
        self.b_up = nn.Parameter(self.b_up.data[active_idx].contiguous())
        self.b_gate = nn.Parameter(self.b_gate.data[active_idx].contiguous())

    def compact_moe(self, opcode_range=None, relay_map=None):
        """Partition compact FFN into MoE experts by opcode.

        After compact(), further partitions hidden units into:
        - shared: always computed (no opcode dependency)
        - per-opcode experts: only computed when that opcode is active

        At runtime, call set_active_opcode(dim) before forward().

        Args:
            opcode_range: range of dims that are opcode one-hot flags.
            relay_map: dict mapping additional up weight dims to opcode dims.
        """
        if opcode_range is None:
            opcode_range = range(262, 294)
        if relay_map is None:
            relay_map = {}

        W_up = self.up.weight.data
        W_gate = self.gate.weight.data
        b_up = self.b_up.data
        b_gate = self.b_gate.data
        H = W_up.shape[0]

        # For each unit, find opcode dependencies
        opcode_to_units = {}
        shared_indices = []

        for i in range(H):
            opcodes = set()
            # Check W_up for positive opcode weights
            for d in opcode_range:
                if d < W_up.shape[1] and W_up[i, d].item() > 0.5:
                    opcodes.add(d)
            # Check W_gate for significant opcode weights
            for d in opcode_range:
                if d < W_gate.shape[1] and abs(W_gate[i, d].item()) > 0.5:
                    opcodes.add(d)
            # Check relay dims
            for relay_dim, target in relay_map.items():
                if relay_dim < W_up.shape[1] and W_up[i, relay_dim].item() > 0.5:
                    if isinstance(target, (list, tuple, set)):
                        opcodes.update(target)
                    else:
                        opcodes.add(target)

            if opcodes:
                for d in opcodes:
                    opcode_to_units.setdefault(d, []).append(i)
            else:
                shared_indices.append(i)

        if not opcode_to_units:
            self._moe_shared = None
            self._moe_experts = None
            self._active_opcode = None
            return

        dim = W_up.shape[1]

        # Store shared sub-matrices
        if shared_indices:
            idx = torch.tensor(shared_indices, dtype=torch.long)
            self._moe_shared = {
                'W_up': W_up[idx].contiguous(),
                'b_up': b_up[idx].contiguous(),
                'W_gate': W_gate[idx].contiguous(),
                'b_gate': b_gate[idx].contiguous(),
                'W_down': self.down.weight.data[:, idx].contiguous(),
            }
        else:
            self._moe_shared = None

        # Store per-opcode expert sub-matrices
        self._moe_experts = {}
        for d, units in opcode_to_units.items():
            idx = torch.tensor(sorted(set(units)), dtype=torch.long)
            self._moe_experts[d] = {
                'W_up': W_up[idx].contiguous(),
                'b_up': b_up[idx].contiguous(),
                'W_gate': W_gate[idx].contiguous(),
                'b_gate': b_gate[idx].contiguous(),
                'W_down': self.down.weight.data[:, idx].contiguous(),
            }

        self._active_opcode = None

        # Pre-concatenate shared + expert matrices for each opcode
        self._moe_combined = {}
        shared_mats = self._moe_shared
        for d, expert in self._moe_experts.items():
            if shared_mats is not None:
                self._moe_combined[d] = {
                    'W_up': torch.cat([shared_mats['W_up'], expert['W_up']], dim=0),
                    'b_up': torch.cat([shared_mats['b_up'], expert['b_up']], dim=0),
                    'W_gate': torch.cat([shared_mats['W_gate'], expert['W_gate']], dim=0),
                    'b_gate': torch.cat([shared_mats['b_gate'], expert['b_gate']], dim=0),
                    'W_down': torch.cat([shared_mats['W_down'], expert['W_down']], dim=1),
                }
            else:
                self._moe_combined[d] = expert

        if shared_mats is not None:
            self._moe_combined['_shared'] = shared_mats

        # Full matrices fallback
        self._moe_combined['_full'] = {
            'W_up': W_up,
            'b_up': b_up,
            'W_gate': W_gate,
            'b_gate': b_gate,
            'W_down': self.down.weight.data,
        }

        # Empty fallback
        self._moe_combined['_empty'] = {
            'W_up': torch.zeros(0, dim),
            'b_up': torch.zeros(0),
            'W_gate': torch.zeros(0, dim),
            'b_gate': torch.zeros(0),
            'W_down': torch.zeros(dim, 0),
        }

    def _activate_moe(self, opcode_dim):
        """Swap weight tensors to the active opcode's sub-matrices.

        Called by AutoregressiveVM.set_active_opcode().
        """
        if self._moe_combined is None:
            return

        if opcode_dim is None:
            c = self._moe_combined['_full']
        else:
            c = self._moe_combined.get(opcode_dim)
            if c is None:
                c = self._moe_combined.get('_shared')
            if c is None:
                c = self._moe_combined['_empty']

        # Swap in the expert weights
        n = c['W_up'].shape[0]
        self.up = nn.Linear(self.dim, n, bias=False)
        self.gate = nn.Linear(self.dim, n, bias=False)
        self.down = nn.Linear(n, self.dim, bias=True)

        self.up.weight = nn.Parameter(c['W_up'].contiguous())
        self.gate.weight = nn.Parameter(c['W_gate'].contiguous())
        self.down.weight = nn.Parameter(c['W_down'].contiguous())
        nn.init.zeros_(self.down.bias)

        # Swap biases
        self.b_up = nn.Parameter(c['b_up'].contiguous())
        self.b_gate = nn.Parameter(c['b_gate'].contiguous())

    def _bake_weights(self):
        """Override to bake operation-specific weights."""
        pass


class PureAttention(nn.Module):
    """
    Pure Attention with FINAL forward pass.

    Uses nn.Linear internally for standard PyTorch patterns while
    providing direct weight access for baking via W_q, W_k, W_v, W_o properties.

    Subclasses ONLY override _bake_weights().
    Used for carry propagation between nibble positions.
    """

    def __init__(self, dim: int, num_heads: int = 1, causal: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        # Use nn.Linear for cleaner code
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Zero initialize
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.k_proj.weight)
        nn.init.zeros_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

        # Default mask is zeros (no masking)
        self.register_buffer('mask', torch.zeros(E.NUM_POSITIONS, E.NUM_POSITIONS))

        self._bake_weights()

    # Direct weight access for baking
    def __getattr__(self, name):
        if name in ('W_q', 'W_k', 'W_v', 'W_o'):
            if name == 'W_q':
                return self.q_proj.weight.data
            elif name == 'W_k':
                return self.k_proj.weight.data
            elif name == 'W_v':
                return self.v_proj.weight.data
            elif name == 'W_o':
                return self.out_proj.weight.data
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        # Redirect weight assignments to nn.Linear weights
        if name == 'W_q':
            self.q_proj.weight.data = value
        elif name == 'W_k':
            self.k_proj.weight.data = value
        elif name == 'W_v':
            self.v_proj.weight.data = value
        elif name == 'W_o':
            self.out_proj.weight.data = value
        else:
            super().__setattr__(name, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FINAL - Standard attention. DO NOT OVERRIDE."""
        B, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        Q = self.q_proj(x).view(B, S, H, HD).transpose(1, 2)
        K = self.k_proj(x).view(B, S, H, HD).transpose(1, 2)
        V = self.v_proj(x).view(B, S, H, HD).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Use mask only if S <= NUM_POSITIONS, otherwise no masking
        if S <= self.mask.shape[0]:
            scores = scores + self.mask[:S, :S]

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return x + self.out_proj(out)

    def _bake_weights(self):
        """Override to bake attention weights."""
        pass


class FlattenedPureFFN(nn.Module):
    """
    Wrapper that flattens input, applies a PureFFN, and unflattens output.

    For cross-position operations (reading from one position, writing to another).
    Uses composition - the inner ffn is a standard PureFFN with no forward override.

    Subclasses override _bake_weights() to set weights via self.ffn.W_up, etc.
    Use _flat_idx(pos, slot) to compute indices into the flattened dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dim = E.DIM
        self.flat_dim = E.NUM_POSITIONS * E.DIM
        self.hidden_dim = hidden_dim
        # Use standard PureFFN on flattened dimension
        self.ffn = PureFFN(dim=self.flat_dim, hidden_dim=hidden_dim)
        # Call subclass bake_weights after ffn is created
        self._bake_weights()

    def _flat_idx(self, pos: int, slot: int) -> int:
        """Get flattened index for position and slot."""
        return pos * self.dim + slot

    # Convenience properties to access inner FFN weights
    @property
    def W_up(self):
        return self.ffn.W_up

    @property
    def b_up(self):
        return self.ffn.b_up

    @property
    def W_gate(self):
        return self.ffn.W_gate

    @property
    def b_gate(self):
        return self.ffn.b_gate

    @property
    def W_down(self):
        return self.ffn.W_down

    @property
    def b_down(self):
        return self.ffn.b_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten -> PureFFN -> Unflatten. FINAL - DO NOT OVERRIDE."""
        B, N, D = x.shape
        x_flat = x.reshape(B, 1, N * D)  # [B, 1, flat_dim] - single "position"
        y_flat = self.ffn(x_flat)
        return y_flat.reshape(B, N, D)

    def _bake_weights(self):
        """Override to bake operation-specific weights."""
        pass
