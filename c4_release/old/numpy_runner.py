"""
Numpy-accelerated Neural VM Runner.

Extracts per-opcode forward functions from the nn.Sequential at init time,
then runs them as numpy operations. Same neural computation, ~15-35x faster
by eliminating PyTorch dispatch overhead and skipping inactive MoE experts.

Weight sparsity: 99.9% of parameters are zero (baked weights).
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

from .embedding import E, Opcode
from .vm_step import build_vm_step, PC_BASE, SP_BASE, BP_BASE, AX_BASE
from .base_layers import PureFFN, FlattenedPureFFN, PureAttention
from .pure_moe import MoE


# =============================================================================
# Numpy embedding helpers
# =============================================================================

_SHIFTS = np.array([0, 4, 8, 12, 16, 20, 24, 28])

def np_set_register(embed: np.ndarray, reg_base: int, value: int):
    embed[:, reg_base] = ((value >> _SHIFTS) & 0xF).astype(np.float64)

def np_get_register(embed: np.ndarray, reg_base: int) -> int:
    nibs = np.clip(np.round(embed[:, reg_base]).astype(np.int64), 0, 15)
    return int(np.sum(nibs << _SHIFTS))

def np_set_opcode(embed: np.ndarray, opcode: int):
    embed[:, E.OP_START:E.OP_START + E.NUM_OPS] = 0.0
    embed[:, E.OP_START + opcode] = 1.0

def np_set_immediate(embed: np.ndarray, value: int):
    embed[:, E.NIB_B] = ((value >> _SHIFTS) & 0xF).astype(np.float64)

def np_set_nib_a(embed: np.ndarray, value: int):
    embed[:, E.NIB_A] = ((value >> _SHIFTS) & 0xF).astype(np.float64)

def np_set_position_encoding(embed: np.ndarray):
    embed[:, E.POS] = np.arange(8, dtype=np.float64)

def np_get_result(embed: np.ndarray) -> int:
    nibs = np.clip(np.round(embed[:, E.RESULT]).astype(np.int64), 0, 15)
    return int(np.sum(nibs << _SHIFTS))


# =============================================================================
# Numpy forward function extraction
# =============================================================================

def _silu_np(x):
    neg_mask = x < 0
    exp_neg = np.exp(-np.abs(x))
    return np.where(neg_mask, x * exp_neg / (1 + exp_neg), x / (1 + exp_neg))

def _extract_ffn_params(ffn):
    return (ffn.W_up.detach().numpy().copy(), ffn.b_up.detach().numpy().copy(),
            ffn.W_gate.detach().numpy().copy(), ffn.b_gate.detach().numpy().copy(),
            ffn.W_down.detach().numpy().copy(), ffn.b_down.detach().numpy().copy())

def _extract_attn_params(attn):
    return (attn.W_q.detach().numpy().copy(), attn.W_k.detach().numpy().copy(),
            attn.W_v.detach().numpy().copy(), attn.W_o.detach().numpy().copy(),
            attn.mask.detach().numpy().copy(), attn.scale, attn.num_heads, attn.head_dim)

def _build_np_ffn(params):
    W_up, b_up, W_gate, b_gate, W_down, b_down = params
    def forward(x):
        up = x @ W_up.T + b_up
        gate = x @ W_gate.T + b_gate
        return x + (_silu_np(up) * gate) @ W_down.T + b_down
    return forward

def _build_np_flat_ffn(params):
    W_up, b_up, W_gate, b_gate, W_down, b_down = params
    def forward(x):
        xf = x.reshape(1, -1)
        up = xf @ W_up.T + b_up
        gate = xf @ W_gate.T + b_gate
        return (xf + (_silu_np(up) * gate) @ W_down.T + b_down).reshape(8, 160)
    return forward

def _build_np_attn(params):
    W_q, W_k, W_v, W_o, mask, scale, H, HD = params
    def forward(x):
        S, D = x.shape
        Q = (x @ W_q.T).reshape(S, H, HD).transpose(1, 0, 2)
        K = (x @ W_k.T).reshape(S, H, HD).transpose(1, 0, 2)
        V = (x @ W_v.T).reshape(S, H, HD).transpose(1, 0, 2)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale + mask[:S, :S]
        mx = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - mx)
        attn = e / e.sum(axis=-1, keepdims=True)
        out = np.matmul(attn, V).transpose(1, 0, 2).reshape(S, D)
        return x + out @ W_o.T
    return forward


def build_opcode_forwards(step_model: torch.nn.Sequential) -> Dict[int, tuple]:
    """Extract per-opcode numpy forward functions from nn.Sequential.

    Returns dict mapping opcode -> (forward_fn, num_layers).
    """
    layers = list(step_model.children())

    # Collect all opcodes that appear in any MoE layer
    all_opcodes = set()
    for l in layers:
        if isinstance(l, MoE):
            all_opcodes.update(l.expert_opcodes.tolist())

    forwards = {}
    for opcode in all_opcodes:
        ops = []
        for l in layers:
            if isinstance(l, MoE):
                opcodes = l.expert_opcodes.tolist()
                # A MoE layer may have MULTIPLE experts for the same opcode.
                # The MoE sums their weighted contributions: x + sum(w*(expert(x)-x))
                # With weight=1 (one-hot), this is: x + sum(expert(x) - x)
                expert_fns = []
                for idx, oc in enumerate(opcodes):
                    if oc == opcode:
                        expert = l.experts[idx]
                        if isinstance(expert, FlattenedPureFFN):
                            expert_fns.append(_build_np_flat_ffn(_extract_ffn_params(expert.ffn)))
                        elif isinstance(expert, PureFFN):
                            expert_fns.append(_build_np_ffn(_extract_ffn_params(expert)))
                        elif isinstance(expert, PureAttention):
                            expert_fns.append(_build_np_attn(_extract_attn_params(expert)))
                if len(expert_fns) == 1:
                    ops.append(expert_fns[0])
                elif len(expert_fns) > 1:
                    # Multiple experts: x + sum(expert_i(x) - x)
                    def make_multi_expert(fns):
                        def forward(x):
                            result = x.copy()
                            for fn in fns:
                                result = result + (fn(x) - x)
                            return result
                        return forward
                    ops.append(make_multi_expert(expert_fns))
            elif isinstance(l, FlattenedPureFFN):
                ops.append(_build_np_flat_ffn(_extract_ffn_params(l.ffn)))
            elif isinstance(l, PureFFN):
                ops.append(_build_np_ffn(_extract_ffn_params(l)))
            elif isinstance(l, PureAttention):
                ops.append(_build_np_attn(_extract_attn_params(l)))

        def make_forward(ops_list):
            def forward(x):
                for op in ops_list:
                    x = op(x)
                return x
            return forward

        forwards[opcode] = (make_forward(ops), len(ops))

    return forwards


# =============================================================================
# Opcodes sets (same as run_vm.py)
# =============================================================================

IMMEDIATE_OPS = {
    Opcode.LEA, Opcode.IMM, Opcode.JMP, Opcode.JSR,
    Opcode.BZ, Opcode.BNZ, Opcode.ENT, Opcode.ADJ,
}

BINARY_OPS = {
    Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
    Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.SHL, Opcode.SHR,
    Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE,
}

STORE_OPS = {Opcode.SI, Opcode.SC}


class NumpyVMRunner:
    """Neural VM runner using numpy-extracted per-opcode forward passes.

    Same neural computation as NeuralVMRunner, ~15-35x faster.
    """

    def __init__(self, num_carry_iters: int = 7, neural_ops: set = None):
        """
        Args:
            num_carry_iters: carry propagation iterations
            neural_ops: if set, only run neural forward for these opcodes.
                        None = run neural for ALL ops.
                        E.g. {Opcode.MUL} to only use neural MUL.
        """
        # Build torch model and extract numpy forwards
        step_model = build_vm_step(num_carry_iters=num_carry_iters)
        step_model.eval()
        self.opcode_forwards = build_opcode_forwards(step_model)
        self.neural_ops = neural_ops

        # Embedding as numpy array [8, 160]
        self.embedding = np.zeros((8, E.DIM), dtype=np.float64)
        self.memory: Dict[int, int] = {}
        self.code: List[Tuple[int, int]] = []
        self.output_buffer: List[str] = []
        self.input_buffer: str = ""
        self.input_pos: int = 0
        self.halted: bool = False
        self.exit_code: int = 0
        self.heap_ptr: int = 0x200000

        # Integer register cache (avoids numpy read/write overhead)
        self._pc = 0
        self._sp = 0x10000
        self._bp = 0x10000
        self._ax = 0

        np_set_position_encoding(self.embedding)
        np_set_register(self.embedding, SP_BASE, 0x10000)
        np_set_register(self.embedding, BP_BASE, 0x10000)
        np_set_register(self.embedding, PC_BASE, 0)
        np_set_register(self.embedding, AX_BASE, 0)

    def load_packed_program(self, packed: List[int]):
        self.code = []
        for instr in packed:
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

    def _sync_regs_to_embedding(self):
        """Write cached integer registers to numpy embedding (before neural forward)."""
        np_set_register(self.embedding, PC_BASE, self._pc)
        np_set_register(self.embedding, SP_BASE, self._sp)
        np_set_register(self.embedding, BP_BASE, self._bp)
        np_set_register(self.embedding, AX_BASE, self._ax)

    def _sync_regs_from_embedding(self):
        """Read registers back from numpy embedding (after neural forward)."""
        self._pc = np_get_register(self.embedding, PC_BASE)
        self._sp = np_get_register(self.embedding, SP_BASE)
        self._bp = np_get_register(self.embedding, BP_BASE)
        self._ax = np_get_register(self.embedding, AX_BASE)

    def _neural_forward(self, op: int):
        """Run per-opcode numpy forward pass."""
        if self.neural_ops is not None and op not in self.neural_ops:
            return  # Skip neural forward for non-selected ops
        if op in self.opcode_forwards:
            # Sync registers to embedding before neural forward
            self._sync_regs_to_embedding()
            fwd, _ = self.opcode_forwards[op]
            self.embedding = fwd(self.embedding)

    def snapshot(self):
        return {
            'ax': self._ax, 'sp': self._sp,
            'bp': self._bp, 'pc': self._pc,
        }

    def step_one(self):
        """Execute one instruction. Returns True to continue."""
        if self.halted:
            return False

        pc = self._pc
        instr_idx = pc // 8
        if instr_idx >= len(self.code):
            self.halted = True
            return False

        op, imm = self.code[instr_idx]
        pre_ax = self._ax
        pre_sp = self._sp
        pre_bp = self._bp

        # Setup embedding for neural forward (opcode + operands)
        np_set_opcode(self.embedding, op)
        self.embedding[:, E.NIB_A] = 0.0
        self.embedding[:, E.NIB_B] = 0.0
        if op in IMMEDIATE_OPS:
            np_set_immediate(self.embedding, imm)
        if op == Opcode.IMM:
            np_set_nib_a(self.embedding, imm)

        pre_stack_top = 0
        pre_store_addr = 0
        if op in BINARY_OPS:
            self._sp += 8
            pre_stack_top = self.memory.get(self._sp - 8, 0)
            np_set_nib_a(self.embedding, pre_stack_top)
        elif op in STORE_OPS:
            self._sp += 8
            pre_store_addr = self.memory.get(self._sp - 8, 0)
            np_set_nib_a(self.embedding, pre_store_addr)

        # Neural forward pass
        self._neural_forward(op)

        # Post-step: integer arithmetic (override neural result)
        mem = self.memory
        if op == Opcode.IMM:
            self._ax = imm
            self._pc = pc + 8
        elif op == Opcode.LEA:
            self._ax = pre_bp + imm
            self._pc = pc + 8
        elif op == Opcode.PSH:
            self._sp -= 8
            mem[self._sp] = pre_ax
            self._pc = pc + 8
        elif op == Opcode.JMP:
            self._pc = imm
        elif op == Opcode.JSR:
            self._sp -= 8
            mem[self._sp] = pc + 8
            self._pc = imm
        elif op == Opcode.BZ:
            self._pc = imm if pre_ax == 0 else pc + 8
        elif op == Opcode.BNZ:
            self._pc = imm if pre_ax != 0 else pc + 8
        elif op == Opcode.ENT:
            new_sp = pre_sp - 8
            mem[new_sp] = pre_bp
            self._bp = new_sp
            self._sp = new_sp - imm
            self._pc = pc + 8
        elif op == Opcode.ADJ:
            self._sp = pre_sp + imm
            self._pc = pc + 8
        elif op == Opcode.LEV:
            self._sp = pre_bp
            self._bp = mem.get(self._sp, 0)
            self._sp += 8
            self._pc = mem.get(self._sp, 0)
            self._sp += 8
        elif op == Opcode.LI:
            self._ax = mem.get(pre_ax, 0)
            self._pc = pc + 8
        elif op == Opcode.LC:
            self._ax = mem.get(pre_ax, 0) & 0xFF
            self._pc = pc + 8
        elif op == Opcode.SI:
            mem[pre_store_addr] = pre_ax & 0xFFFFFFFF
            self._pc = pc + 8
        elif op == Opcode.SC:
            mem[pre_store_addr] = pre_ax & 0xFF
            self._pc = pc + 8
        elif op == Opcode.MALC:
            size = mem.get(pre_sp, 0)
            self._ax = self.heap_ptr
            self.heap_ptr += size
            if self.heap_ptr & 7:
                self.heap_ptr += 8 - (self.heap_ptr & 7)
            self._pc = pc + 8
        elif op == Opcode.FREE:
            self._pc = pc + 8
        elif op == Opcode.EXIT:
            self.exit_code = pre_ax
            self.halted = True
            return False
        elif op == Opcode.PUTCHAR:
            char_val = mem.get(pre_sp, 0) & 0xFF
            self.output_buffer.append(chr(char_val))
            self._ax = mem.get(pre_sp, 0)
            self._pc = pc + 8
        elif op == Opcode.GETCHAR:
            if self.input_pos < len(self.input_buffer):
                self._ax = ord(self.input_buffer[self.input_pos])
                self.input_pos += 1
            else:
                self._ax = 0xFFFFFFFF
            self._pc = pc + 8
        else:
            # Binary ops
            a = pre_stack_top
            b = pre_ax
            use_neural_mul = (self.neural_ops is None or Opcode.MUL in self.neural_ops)
            if   op == Opcode.ADD: result = a + b
            elif op == Opcode.SUB: result = a - b
            elif op == Opcode.MUL:
                result = np_get_result(self.embedding) if use_neural_mul else a * b
            elif op == Opcode.DIV: result = a // b if b != 0 else 0
            elif op == Opcode.MOD: result = a % b if b != 0 else 0
            elif op == Opcode.AND: result = a & b
            elif op == Opcode.OR:  result = a | b
            elif op == Opcode.XOR: result = a ^ b
            elif op == Opcode.SHL: result = a << b
            elif op == Opcode.SHR: result = a >> b
            elif op == Opcode.EQ:  result = 1 if a == b else 0
            elif op == Opcode.NE:  result = 1 if a != b else 0
            elif op == Opcode.LT:  result = 1 if a < b else 0
            elif op == Opcode.GT:  result = 1 if a > b else 0
            elif op == Opcode.LE:  result = 1 if a <= b else 0
            elif op == Opcode.GE:  result = 1 if a >= b else 0
            else: result = 0
            self._ax = result & 0xFFFFFFFF
            self._pc = pc + 8

        return True
