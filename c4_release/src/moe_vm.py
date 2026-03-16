"""
MoE Neural VM - Mixture of Experts for Efficient Soft Execution

Instead of computing ALL operations and blending (wasteful),
use the opcode as a router to select which "expert" (operation) to run.

This gives us:
- Soft/differentiable dispatch (router is a softmax)
- Efficient compute (only top-k experts activated)
- Trainable routing (can learn better dispatch)

Architecture:
    opcode -> Router (softmax) -> top-k experts -> weighted sum

Each "expert" is a specific operation (ADD, MUL, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


class Expert(nn.Module):
    """Base class for operation experts."""

    def forward(self, state: 'VMState') -> 'VMState':
        raise NotImplementedError


class VMState:
    """Mutable VM state passed between experts."""

    def __init__(self):
        self.ax = None      # Accumulator [4, 256]
        self.sp = None      # Stack pointer [4, 256]
        self.bp = None      # Base pointer [4, 256]
        self.pc = None      # Program counter [4, 256]
        self.imm = None     # Current immediate [4, 256]
        self.stack_top = None  # Value at top of stack [4, 256]
        self.halted = torch.tensor([0.0])

        # Memory (for experts that need it)
        self.memory_keys = []
        self.memory_values = []

    def clone(self) -> 'VMState':
        """Create a copy of state."""
        new = VMState()
        new.ax = self.ax.clone() if self.ax is not None else None
        new.sp = self.sp.clone() if self.sp is not None else None
        new.bp = self.bp.clone() if self.bp is not None else None
        new.pc = self.pc.clone() if self.pc is not None else None
        new.imm = self.imm.clone() if self.imm is not None else None
        new.stack_top = self.stack_top.clone() if self.stack_top is not None else None
        new.halted = self.halted.clone()
        new.memory_keys = self.memory_keys  # Shared reference OK
        new.memory_values = self.memory_values
        return new


class MoEALU(nn.Module):
    """ALU operations for MoE experts."""

    def __init__(self):
        super().__init__()
        self._build_tables()

    def _build_tables(self):
        """Build nibble operation tables."""
        # Add table
        add_table = torch.zeros(16, 16, 2, 16)
        carry_table = torch.zeros(16, 16, 2, 2)
        for a in range(16):
            for b in range(16):
                for c in range(2):
                    total = a + b + c
                    add_table[a, b, c, total % 16] = 1.0
                    carry_table[a, b, c, total // 16] = 1.0
        self.register_buffer('add_table', add_table)
        self.register_buffer('carry_table', carry_table)

        # Bitwise tables
        for op_name, op_fn in [('and', lambda a, b: a & b),
                               ('or', lambda a, b: a | b),
                               ('xor', lambda a, b: a ^ b)]:
            table = torch.zeros(16, 16, 16)
            for a in range(16):
                for b in range(16):
                    table[a, b, op_fn(a, b)] = 1.0
            self.register_buffer(f'{op_name}_table', table)

        # Byte <-> Nibble conversion
        b2n = torch.zeros(256, 32)
        n2b = torch.zeros(32, 256)
        for b in range(256):
            h, l = b >> 4, b & 0xF
            b2n[b, h] = 1.0
            b2n[b, 16 + l] = 1.0
            n2b[h, b] = 1.0
            n2b[16 + l, b] = 1.0
        self.register_buffer('b2n', b2n)
        self.register_buffer('n2b', n2b)

    def encode(self, val: int) -> torch.Tensor:
        result = torch.zeros(4, 256)
        for i in range(4):
            result[i, (val >> (i * 8)) & 0xFF] = 1.0
        return result

    def decode(self, enc: torch.Tensor) -> int:
        result = 0
        for i in range(4):
            result |= int(torch.argmax(enc[i]).item()) << (i * 8)
        return result

    def _to_nibbles(self, byte_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = F.linear(byte_enc, self.b2n.T)
        return F.softmax(combined[:16] * 100, dim=-1), F.softmax(combined[16:] * 100, dim=-1)

    def _from_nibbles(self, h: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h, l])
        return F.softmax(F.linear(combined, self.n2b.T) * 100, dim=-1)

    def _nibble_add(self, a, b, cin):
        weights = torch.einsum('i,j,k->ijk', a, b, cin)
        sum_r = torch.einsum('ijk,ijkl->l', weights, self.add_table)
        carry_r = torch.einsum('ijk,ijkl->l', weights, self.carry_table)
        return F.softmax(sum_r * 100, dim=-1), F.softmax(carry_r * 100, dim=-1)

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(4, 256)
        carry = torch.tensor([1.0, 0.0])
        for i in range(4):
            ah, al = self._to_nibbles(a[i])
            bh, bl = self._to_nibbles(b[i])
            sl, carry = self._nibble_add(al, bl, carry)
            sh, carry = self._nibble_add(ah, bh, carry)
            result[i] = self._from_nibbles(sh, sl)
        return result

    def subtract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        b_neg = self._negate(b)
        return self.add(a, b_neg)

    def _negate(self, x: torch.Tensor) -> torch.Tensor:
        flipped = torch.zeros(4, 256)
        ff = torch.zeros(16)
        ff[15] = 1.0
        for i in range(4):
            xh, xl = self._to_nibbles(x[i])
            nh = self._nibble_bitwise(xh, ff, self.xor_table)
            nl = self._nibble_bitwise(xl, ff, self.xor_table)
            flipped[i] = self._from_nibbles(nh, nl)
        return self.add(flipped, self.encode(1))

    def _nibble_bitwise(self, a, b, table):
        weights = torch.einsum('i,j->ij', a, b)
        return F.softmax(torch.einsum('ij,ijk->k', weights, table) * 100, dim=-1)

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        av, bv = float(self.decode(a)), float(self.decode(b))
        at, bt = torch.tensor(av), torch.tensor(bv)
        result = F.silu(at) * bt + F.silu(-at) * (-bt)
        return self.encode(int(round(result.item())) & 0xFFFFFFFF)

    def divide(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        av, bv = self.decode(a), self.decode(b)
        if bv == 0:
            return self.encode(0)
        return self.encode((av // bv) & 0xFFFFFFFF)

    def bitwise(self, a: torch.Tensor, b: torch.Tensor, op: str) -> torch.Tensor:
        table = getattr(self, f'{op}_table')
        result = torch.zeros(4, 256)
        for i in range(4):
            ah, al = self._to_nibbles(a[i])
            bh, bl = self._to_nibbles(b[i])
            result[i] = self._from_nibbles(
                self._nibble_bitwise(ah, bh, table),
                self._nibble_bitwise(al, bl, table)
            )
        return result

    def is_zero(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([1.0 if self.decode(x) == 0 else 0.0])

    def compare(self, a, b):
        diff = self.decode(self.subtract(a, b))
        is_zero = 1.0 if diff == 0 else 0.0
        is_neg = 1.0 if diff >= 0x80000000 else 0.0
        return torch.tensor([is_neg]), torch.tensor([is_zero]), torch.tensor([1.0 - is_zero - is_neg])

    def blend(self, a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return (1 - alpha) * a + alpha * b


# ============================================================================
# EXPERTS - Each is a specific operation
# ============================================================================

class ImmExpert(nn.Module):
    """IMM: ax = imm"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.ax = state.imm.clone()
        return new


class LeaExpert(nn.Module):
    """LEA: ax = bp + imm"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.ax = alu.add(state.bp, state.imm)
        return new


class AddExpert(nn.Module):
    """ADD: ax = pop() + ax"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.ax = alu.add(state.stack_top, state.ax)
        new.sp = alu.add(state.sp, alu.encode(8))
        return new


class SubExpert(nn.Module):
    """SUB: ax = pop() - ax"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.ax = alu.subtract(state.stack_top, state.ax)
        new.sp = alu.add(state.sp, alu.encode(8))
        return new


class MulExpert(nn.Module):
    """MUL: ax = pop() * ax"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.ax = alu.multiply(state.stack_top, state.ax)
        new.sp = alu.add(state.sp, alu.encode(8))
        return new


class DivExpert(nn.Module):
    """DIV: ax = pop() / ax"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.ax = alu.divide(state.stack_top, state.ax)
        new.sp = alu.add(state.sp, alu.encode(8))
        return new


class ModExpert(nn.Module):
    """MOD: ax = pop() % ax"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        a, b = state.stack_top, state.ax
        quot = alu.divide(a, b)
        prod = alu.multiply(quot, b)
        new.ax = alu.subtract(a, prod)
        new.sp = alu.add(state.sp, alu.encode(8))
        return new


class CompareExpert(nn.Module):
    """Base for comparison experts."""
    def __init__(self, cmp_type: str):
        super().__init__()
        self.cmp_type = cmp_type

    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        lt, eq, gt = alu.compare(state.stack_top, state.ax)
        zero, one = alu.encode(0), alu.encode(1)

        if self.cmp_type == 'eq':
            new.ax = alu.blend(zero, one, eq)
        elif self.cmp_type == 'ne':
            new.ax = alu.blend(one, zero, eq)
        elif self.cmp_type == 'lt':
            new.ax = alu.blend(zero, one, lt)
        elif self.cmp_type == 'gt':
            new.ax = alu.blend(zero, one, gt)
        elif self.cmp_type == 'le':
            le = lt + eq - lt * eq
            new.ax = alu.blend(zero, one, le)
        elif self.cmp_type == 'ge':
            ge = gt + eq - gt * eq
            new.ax = alu.blend(zero, one, ge)

        new.sp = alu.add(state.sp, alu.encode(8))
        return new


class BitwiseExpert(nn.Module):
    """Base for bitwise experts."""
    def __init__(self, op: str):
        super().__init__()
        self.op = op

    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.ax = alu.bitwise(state.stack_top, state.ax, self.op)
        new.sp = alu.add(state.sp, alu.encode(8))
        return new


class JmpExpert(nn.Module):
    """JMP: pc = imm"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.pc = state.imm.clone()
        return new


class BzExpert(nn.Module):
    """BZ: if ax == 0, pc = imm"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        is_zero = alu.is_zero(state.ax)
        next_pc = alu.add(state.pc, alu.encode(8))
        new.pc = alu.blend(next_pc, state.imm, is_zero)
        return new


class BnzExpert(nn.Module):
    """BNZ: if ax != 0, pc = imm"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        is_zero = alu.is_zero(state.ax)
        next_pc = alu.add(state.pc, alu.encode(8))
        new.pc = alu.blend(state.imm, next_pc, is_zero)
        return new


class PshExpert(nn.Module):
    """PSH: push ax"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.sp = alu.subtract(state.sp, alu.encode(8))
        # Memory write handled by VM
        return new


class NoopExpert(nn.Module):
    """NOOP: do nothing (for unhandled ops)"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        return state.clone()


class ExitExpert(nn.Module):
    """EXIT: halt"""
    def forward(self, state: VMState, alu: MoEALU) -> VMState:
        new = state.clone()
        new.halted = torch.tensor([1.0])
        return new


# ============================================================================
# MoE ROUTER
# ============================================================================

class MoERouter(nn.Module):
    """
    Routes opcodes to experts using soft attention.

    Unlike dense MoE where you learn the router, here we use
    fixed routing based on opcode identity (one-hot match).
    For trainable routing, you'd learn the routing weights.
    """

    def __init__(self, num_ops: int = 64, top_k: int = 1):
        super().__init__()
        self.num_ops = num_ops
        self.top_k = top_k

        # For each opcode, which expert to use
        # This is a fixed mapping, but could be learned
        self.register_buffer('routing_table', torch.eye(num_ops))

    def forward(self, op_enc: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Route opcode to experts.

        Returns:
            weights: Soft weights for each expert
            indices: Top-k expert indices to activate
        """
        # Compute routing scores (just the opcode encoding)
        scores = op_enc

        # Get top-k
        if self.top_k < self.num_ops:
            topk_vals, topk_idx = torch.topk(scores, self.top_k)
            # Renormalize top-k weights
            weights = F.softmax(topk_vals * 100, dim=0)
            indices = topk_idx.tolist()
        else:
            weights = F.softmax(scores * 100, dim=0)
            indices = list(range(self.num_ops))

        return weights, indices


# ============================================================================
# MoE VM
# ============================================================================

class MoEVM(nn.Module):
    """
    Mixture-of-Experts Neural VM.

    Each operation is an "expert". The opcode routes to the relevant
    expert(s) with soft weights. Only top-k experts are computed.

    With top_k=1: Only one expert runs (efficient)
    With top_k=3: Top 3 experts run and blend (more robust)
    """

    # Opcode constants
    OP_LEA, OP_IMM, OP_JMP, OP_JSR = 0, 1, 2, 3
    OP_BZ, OP_BNZ, OP_ENT, OP_ADJ = 4, 5, 6, 7
    OP_LEV, OP_LI, OP_LC, OP_SI = 8, 9, 10, 11
    OP_SC, OP_PSH = 12, 13
    OP_OR, OP_XOR, OP_AND = 14, 15, 16
    OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE = 17, 18, 19, 20, 21, 22
    OP_SHL, OP_SHR, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD = 23, 24, 25, 26, 27, 28, 29
    OP_EXIT = 38
    NUM_OPS = 64

    def __init__(self, top_k: int = 1):
        super().__init__()
        self.top_k = top_k
        self.alu = MoEALU()
        self.router = MoERouter(self.NUM_OPS, top_k)

        # Create experts
        self.experts = nn.ModuleDict({
            str(self.OP_IMM): ImmExpert(),
            str(self.OP_LEA): LeaExpert(),
            str(self.OP_ADD): AddExpert(),
            str(self.OP_SUB): SubExpert(),
            str(self.OP_MUL): MulExpert(),
            str(self.OP_DIV): DivExpert(),
            str(self.OP_MOD): ModExpert(),
            str(self.OP_AND): BitwiseExpert('and'),
            str(self.OP_OR): BitwiseExpert('or'),
            str(self.OP_XOR): BitwiseExpert('xor'),
            str(self.OP_EQ): CompareExpert('eq'),
            str(self.OP_NE): CompareExpert('ne'),
            str(self.OP_LT): CompareExpert('lt'),
            str(self.OP_GT): CompareExpert('gt'),
            str(self.OP_LE): CompareExpert('le'),
            str(self.OP_GE): CompareExpert('ge'),
            str(self.OP_JMP): JmpExpert(),
            str(self.OP_BZ): BzExpert(),
            str(self.OP_BNZ): BnzExpert(),
            str(self.OP_PSH): PshExpert(),
            str(self.OP_EXIT): ExitExpert(),
        })
        self.noop = NoopExpert()

        self.reset()

    def reset(self):
        """Reset VM state."""
        self.state = VMState()
        self.state.ax = self.alu.encode(0)
        self.state.sp = self.alu.encode(0x10000)
        self.state.bp = self.alu.encode(0x10000)
        self.state.pc = self.alu.encode(0)
        self.state.halted = torch.tensor([0.0])
        self.code_tokens = []
        self.eight = self.alu.encode(8)

    def load_bytecode(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load bytecode."""
        self.code_tokens = []
        for i, instr in enumerate(bytecode):
            addr = i * 8
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code_tokens.append((
                self.alu.encode(addr),
                self._encode_op(op),
                self.alu.encode(imm & 0xFFFFFFFF)
            ))

        if data:
            base = 0x10000
            for i, b in enumerate(data):
                self.state.memory_keys.append(self.alu.encode(base + i))
                self.state.memory_values.append(self.alu.encode(b))

    def _encode_op(self, op: int) -> torch.Tensor:
        result = torch.zeros(self.NUM_OPS)
        if 0 <= op < self.NUM_OPS:
            result[op] = 1.0
        return result

    def _fetch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch instruction via attention."""
        if not self.code_tokens:
            return torch.zeros(self.NUM_OPS), self.alu.encode(0)

        pc_flat = self.state.pc.flatten()
        scores = []
        for addr_enc, _, _ in self.code_tokens:
            match = (addr_enc.flatten() * pc_flat).sum()
            scores.append(match)

        scores = torch.stack(scores) * 100
        weights = F.softmax(scores, dim=0)

        op_result = torch.zeros(self.NUM_OPS)
        imm_result = torch.zeros(4, 256)
        for i, (_, op_enc, imm_enc) in enumerate(self.code_tokens):
            op_result = op_result + weights[i] * op_enc
            imm_result = imm_result + weights[i] * imm_enc

        return op_result, imm_result

    def _mem_load(self, addr: torch.Tensor) -> torch.Tensor:
        """Load from memory."""
        if not self.state.memory_keys:
            return self.alu.encode(0)

        addr_flat = addr.flatten()
        scores = []
        for key in self.state.memory_keys:
            match = (key.flatten() * addr_flat).sum()
            scores.append(match)

        scores = torch.stack(scores) * 100
        positions = torch.arange(len(scores), dtype=torch.float32)
        scores = scores + positions * 0.01
        weights = F.softmax(scores, dim=0)

        result = torch.zeros(4, 256)
        for i, val in enumerate(self.state.memory_values):
            result = result + weights[i] * val
        return result

    def _mem_store(self, addr: torch.Tensor, val: torch.Tensor):
        """Store to memory."""
        self.state.memory_keys.append(addr.clone())
        self.state.memory_values.append(val.clone())

    def step(self) -> torch.Tensor:
        """Execute one step using MoE dispatch."""
        # Fetch
        op_enc, imm_enc = self._fetch()

        # Prepare state
        self.state.imm = imm_enc
        self.state.stack_top = self._mem_load(self.state.sp)

        # Route to experts
        weights, indices = self.router(op_enc)

        # Run selected experts
        expert_states = []
        for idx in indices:
            key = str(idx)
            if key in self.experts:
                expert = self.experts[key]
            else:
                expert = self.noop
            expert_state = expert(self.state, self.alu)
            expert_states.append(expert_state)

        # Blend expert outputs
        if len(expert_states) == 1:
            new_state = expert_states[0]
        else:
            # Weighted blend of states
            new_state = VMState()
            new_state.ax = torch.zeros(4, 256)
            new_state.sp = torch.zeros(4, 256)
            new_state.bp = torch.zeros(4, 256)
            new_state.pc = torch.zeros(4, 256)
            new_state.halted = torch.tensor([0.0])

            for i, es in enumerate(expert_states):
                w = weights[i]
                new_state.ax = new_state.ax + w * es.ax
                new_state.sp = new_state.sp + w * es.sp
                new_state.bp = new_state.bp + w * es.bp
                new_state.pc = new_state.pc + w * es.pc
                new_state.halted = new_state.halted + w * es.halted

        # Handle memory writes (PSH, SI, ENT, JSR)
        psh_weight = op_enc[self.OP_PSH]
        if psh_weight > 0.5:
            new_sp = self.alu.subtract(self.state.sp, self.eight)
            self._mem_store(new_sp, self.state.ax)

        # Handle special ops that weren't covered by experts
        # ENT, ADJ, LEV, JSR, LI, SI need special handling
        ent_weight = op_enc[self.OP_ENT]
        if ent_weight > 0.5:
            # Push BP, set BP=SP, SP -= imm
            old_sp = self.state.sp
            self._mem_store(self.alu.subtract(old_sp, self.eight), self.state.bp)
            new_state.bp = self.alu.subtract(old_sp, self.eight)
            new_state.sp = self.alu.subtract(new_state.bp, imm_enc)

        adj_weight = op_enc[self.OP_ADJ]
        if adj_weight > 0.5:
            new_state.sp = self.alu.add(self.state.sp, imm_enc)

        lev_weight = op_enc[self.OP_LEV]
        if lev_weight > 0.5:
            new_state.sp = self.state.bp
            new_state.bp = self._mem_load(self.state.bp)
            bp_plus_8 = self.alu.add(self.state.bp, self.eight)
            new_state.pc = self._mem_load(bp_plus_8)
            new_state.sp = self.alu.add(bp_plus_8, self.eight)

        jsr_weight = op_enc[self.OP_JSR]
        if jsr_weight > 0.5:
            next_pc = self.alu.add(self.state.pc, self.eight)
            new_sp = self.alu.subtract(self.state.sp, self.eight)
            self._mem_store(new_sp, next_pc)
            new_state.sp = new_sp
            new_state.pc = imm_enc

        li_weight = op_enc[self.OP_LI]
        if li_weight > 0.5:
            new_state.ax = self._mem_load(self.state.ax)

        si_weight = op_enc[self.OP_SI]
        if si_weight > 0.5:
            addr = self._mem_load(self.state.sp)
            self._mem_store(addr, self.state.ax)
            new_state.sp = self.alu.add(self.state.sp, self.eight)

        # Advance PC (unless jump/branch handled it)
        jump_ops = op_enc[self.OP_JMP] + op_enc[self.OP_JSR] + op_enc[self.OP_BZ] + op_enc[self.OP_BNZ] + op_enc[self.OP_LEV]
        if jump_ops < 0.5:
            new_state.pc = self.alu.add(self.state.pc, self.eight)

        # Update state
        new_state.memory_keys = self.state.memory_keys
        new_state.memory_values = self.state.memory_values
        self.state = new_state

        return self.state.halted

    def run(self, max_steps: int = 100000) -> int:
        """Run until halted."""
        for _ in range(max_steps):
            halted = self.step()
            if halted > 0.5:
                break
        return self.alu.decode(self.state.ax)


def compile_and_run_moe(source: str, top_k: int = 1, max_steps: int = 100000) -> int:
    """Compile and run on MoE VM."""
    from .compiler import compile_c
    bytecode, data = compile_c(source)
    vm = MoEVM(top_k=top_k)
    vm.load_bytecode(bytecode, data)
    return vm.run(max_steps)


__all__ = [
    'MoEVM',
    'MoEALU',
    'MoERouter',
    'VMState',
    'compile_and_run_moe',
]
