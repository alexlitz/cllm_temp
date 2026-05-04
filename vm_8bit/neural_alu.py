"""
8-bit Neural ALU - forward is a pure sequence of SwiGLU FFN layers.

Two nibble positions (8 bits = 2 x 4-bit nibbles). ADD and SUB use RESULT
(slot 5) with proper opcode gating. MUL uses RESULT_MUL (slot 10) to avoid
corrupting RESULT; a final merge layer selects based on opcode.

Bitwise ops (AND/OR/XOR) use 2-layer pipelines per op: extract bits then
combine. DIV/MOD/SHL/SHR use custom tensor-op modules.

forward(x): for layer in self.layers: x = layer(x); return x. No branching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


S = 100.0
BASE = 16
N = 2


class E8:
    NIB_A = 0
    NIB_B = 1
    RAW_SUM = 2
    CARRY_IN = 3
    CARRY_OUT = 4
    RESULT = 5
    TEMP = 6
    OP_ADD = 7
    OP_SUB = 8
    OP_MUL = 9
    RESULT_MUL = 10
    BIT_A0 = 11; BIT_A1 = 12; BIT_A2 = 13; BIT_A3 = 14
    BIT_B0 = 15; BIT_B1 = 16; BIT_B2 = 17; BIT_B3 = 18
    OP_AND = 19; OP_OR = 20; OP_XOR = 21
    OP_DIV = 22; OP_MOD = 23
    OP_SHL = 24; OP_SHR = 25
    DIM = 26
    NUM_POSITIONS = 2


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.W_up = nn.Parameter(torch.zeros(hidden_dim, dim))
        self.b_up = nn.Parameter(torch.zeros(hidden_dim))
        self.W_gate = nn.Parameter(torch.zeros(hidden_dim, dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.W_down = nn.Parameter(torch.zeros(dim, hidden_dim))
        self.b_down = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        up = F.linear(x, self.W_up, self.b_up)
        gate = F.linear(x, self.W_gate, self.b_gate)
        return x + F.linear(F.silu(up) * gate, self.W_down, self.b_down)


class FlatFFN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dim = E8.DIM
        self.ffn = SwiGLUFFN(N * E8.DIM, hidden_dim)

    def _fi(self, pos, slot):
        return pos * self.dim + slot

    def forward(self, x):
        B, NP, D = x.shape
        y = self.ffn(x.reshape(B, 1, NP * D))
        return y.reshape(B, NP, D)


def _clr(ffn, h, op_slot, slot):
    ffn.W_up.data[h, op_slot] = S
    ffn.W_gate.data[h, slot] = -1.0
    ffn.W_down.data[slot, h] = 1.0 / S
    ffn.W_up.data[h + 1, op_slot] = -S
    ffn.W_gate.data[h + 1, slot] = 1.0
    ffn.W_down.data[slot, h + 1] = 1.0 / S


# -- ADD pipeline (3 layers, opcode-gated, writes RESULT) -----------------


class AddRawGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = SwiGLUFFN(E8.DIM, 6)
        E, op = E8, E8.OP_ADD
        with torch.no_grad():
            W, b, G, D = self.ffn.W_up, self.ffn.b_up, self.ffn.W_gate, self.ffn.W_down
            W[0, E.NIB_A] = S; W[0, E.NIB_B] = S; G[0, op] = S
            D[E.RAW_SUM, 0] = 1.0 / (S * S)
            W[1, E.NIB_A] = -S; W[1, E.NIB_B] = -S; G[1, op] = -S
            D[E.RAW_SUM, 1] = 1.0 / (S * S)
            W[2, E.NIB_A] = S; W[2, E.NIB_B] = S; b[2] = -S * 15
            G[2, op] = 1.0; D[E.CARRY_OUT, 2] = 1.0 / S
            W[3, E.NIB_A] = S; W[3, E.NIB_B] = S; b[3] = -S * 16
            G[3, op] = 1.0; D[E.CARRY_OUT, 3] = -1.0 / S
            W[4, E.NIB_A] = S; W[4, E.NIB_B] = S; b[4] = -S * 14
            G[4, op] = 1.0; D[E.TEMP, 4] = 1.0 / S
            W[5, E.NIB_A] = S; W[5, E.NIB_B] = S; b[5] = -S * 15
            G[5, op] = 1.0; D[E.TEMP, 5] = -1.0 / S
            D[E.TEMP, 2] = -1.0 / S; D[E.TEMP, 3] = 1.0 / S

    def forward(self, x):
        return self.ffn(x)


class AddCarry(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = FlatFFN(9)
        fi = self.flat._fi; E, op = E8, E8.OP_ADD
        with torch.no_grad():
            W, b = self.flat.ffn.W_up, self.flat.ffn.b_up
            G, D = self.flat.ffn.W_gate, self.flat.ffn.W_down
            h = 0
            W[h, fi(0, E.CARRY_OUT)] = S; b[h] = -S * 0.5
            G[h, fi(0, op)] = 1.0; D[fi(1, E.CARRY_IN), h] = 2.0 / S; h += 1
            for p in range(N):
                _clr(self.flat.ffn, h, fi(p, op), fi(p, E.CARRY_OUT)); h += 2
                _clr(self.flat.ffn, h, fi(p, op), fi(p, E.TEMP)); h += 2

    def forward(self, x):
        return self.flat(x)


class AddFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = SwiGLUFFN(E8.DIM, 10)
        E, op = E8, E8.OP_ADD
        with torch.no_grad():
            W, b, G, D = self.ffn.W_up, self.ffn.b_up, self.ffn.W_gate, self.ffn.W_down
            W[0, op] = S; G[0, E.RAW_SUM] = 1.0; D[E.RESULT, 0] = 1.0 / S
            W[1, op] = -S; G[1, E.RAW_SUM] = -1.0; D[E.RESULT, 1] = 1.0 / S
            W[2, op] = S; G[2, E.CARRY_IN] = 1.0; D[E.RESULT, 2] = 1.0 / S
            W[3, op] = -S; G[3, E.CARRY_IN] = -1.0; D[E.RESULT, 3] = 1.0 / S
            W[4, E.RAW_SUM] = S; W[4, E.CARRY_IN] = S; b[4] = -S * 15
            G[4, op] = 1.0; D[E.RESULT, 4] = -16.0 / S
            W[5, E.RAW_SUM] = S; W[5, E.CARRY_IN] = S; b[5] = -S * 16
            G[5, op] = 1.0; D[E.RESULT, 5] = 16.0 / S
            _clr(self.ffn, 6, op, E.RAW_SUM)
            _clr(self.ffn, 8, op, E.CARRY_IN)

    def forward(self, x):
        return self.ffn(x)


# -- SUB pipeline (3 layers, opcode-gated, writes RESULT) -----------------


class SubRawBorrow(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = SwiGLUFFN(E8.DIM, 7)
        E, op = E8, E8.OP_SUB
        with torch.no_grad():
            W, b, G, D = self.ffn.W_up, self.ffn.b_up, self.ffn.W_gate, self.ffn.W_down
            W[0, op] = S; G[0, E.NIB_A] = 1.0; G[0, E.NIB_B] = -1.0
            D[E.RAW_SUM, 0] = 1.0 / S
            W[1, op] = -S; G[1, E.NIB_A] = -1.0; G[1, E.NIB_B] = 1.0
            D[E.RAW_SUM, 1] = 1.0 / S
            W[2, E.NIB_B] = S; W[2, E.NIB_A] = -S; b[2] = 0
            G[2, op] = 1.0; D[E.CARRY_OUT, 2] = 1.0 / S
            W[3, E.NIB_B] = S; W[3, E.NIB_A] = -S; b[3] = -S * 1
            G[3, op] = 1.0; D[E.CARRY_OUT, 3] = -1.0 / S
            W[4, E.NIB_A] = S; W[4, E.NIB_B] = -S; b[4] = S * 1
            G[4, op] = 1.0; D[E.TEMP, 4] = 1.0 / S
            W[5, E.NIB_A] = S; W[5, E.NIB_B] = -S; b[5] = 0
            G[5, op] = 1.0; D[E.TEMP, 5] = -2.0 / S
            W[6, E.NIB_A] = S; W[6, E.NIB_B] = -S; b[6] = -S * 1
            G[6, op] = 1.0; D[E.TEMP, 6] = 1.0 / S

    def forward(self, x):
        return self.ffn(x)


class SubBorrow(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = FlatFFN(9)
        fi = self.flat._fi; E, op = E8, E8.OP_SUB
        with torch.no_grad():
            W, b = self.flat.ffn.W_up, self.flat.ffn.b_up
            G, D = self.flat.ffn.W_gate, self.flat.ffn.W_down
            h = 0
            W[h, fi(0, E.CARRY_OUT)] = S; b[h] = -S * 0.5
            G[h, fi(0, op)] = 1.0; D[fi(1, E.CARRY_IN), h] = 2.0 / S; h += 1
            for p in range(N):
                _clr(self.flat.ffn, h, fi(p, op), fi(p, E.CARRY_OUT)); h += 2
                _clr(self.flat.ffn, h, fi(p, op), fi(p, E.TEMP)); h += 2

    def forward(self, x):
        return self.flat(x)


class SubFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = SwiGLUFFN(E8.DIM, 10)
        E, op = E8, E8.OP_SUB
        with torch.no_grad():
            W, b, G, D = self.ffn.W_up, self.ffn.b_up, self.ffn.W_gate, self.ffn.W_down
            W[0, op] = S; G[0, E.RAW_SUM] = 1.0; D[E.RESULT, 0] = 1.0 / S
            W[1, op] = -S; G[1, E.RAW_SUM] = -1.0; D[E.RESULT, 1] = 1.0 / S
            W[2, op] = S; G[2, E.CARRY_IN] = -1.0; D[E.RESULT, 2] = 1.0 / S
            W[3, op] = -S; G[3, E.CARRY_IN] = 1.0; D[E.RESULT, 3] = 1.0 / S
            W[4, E.CARRY_IN] = S; W[4, E.RAW_SUM] = -S; b[4] = 0
            G[4, op] = 1.0; D[E.RESULT, 4] = 16.0 / S
            W[5, E.CARRY_IN] = S; W[5, E.RAW_SUM] = -S; b[5] = -S * 1
            G[5, op] = 1.0; D[E.RESULT, 5] = -16.0 / S
            _clr(self.ffn, 6, op, E.RAW_SUM)
            _clr(self.ffn, 8, op, E.CARRY_IN)

    def forward(self, x):
        return self.ffn(x)


# -- MUL pipeline (writes RESULT_MUL, NOT RESULT) -------------------------


class MulSchoolbook(nn.Module):
    def __init__(self):
        super().__init__()
        E, op = E8, E8.OP_MUL
        RM = E.RESULT_MUL
        hidden_dim = 3 * 2 + N * 2
        self.flat = FlatFFN(hidden_dim)
        fi = self.flat._fi
        with torch.no_grad():
            h = 0
            for p in range(N):
                _clr(self.flat.ffn, h, fi(p, op), fi(p, RM)); h += 2
            for i, j, k in [(0, 0, 0), (0, 1, 1), (1, 0, 1)]:
                self.flat.ffn.W_up.data[h, fi(i, E.NIB_A)] = S
                self.flat.ffn.W_gate.data[h, fi(j, E.NIB_B)] = 1.0
                self.flat.ffn.W_down.data[fi(k, RM), h] = 1.0 / S; h += 1
                self.flat.ffn.W_up.data[h, fi(i, E.NIB_A)] = -S
                self.flat.ffn.W_gate.data[h, fi(j, E.NIB_B)] = -1.0
                self.flat.ffn.W_down.data[fi(k, RM), h] = 1.0 / S; h += 1

    def forward(self, x):
        return self.flat(x)


class MulCarryPass(nn.Module):
    def __init__(self, max_carry, pass_idx):
        super().__init__()
        E, op = E8, E8.OP_MUL
        RM = E.RESULT_MUL
        add_units = 2 if pass_idx > 0 else 0
        clr_units = 2 * N if pass_idx > 0 else 0
        hidden_dim = add_units + clr_units + 2 * max_carry * N
        self.flat = FlatFFN(hidden_dim)
        fi = self.flat._fi
        with torch.no_grad():
            W, b = self.flat.ffn.W_up, self.flat.ffn.b_up
            G, D = self.flat.ffn.W_gate, self.flat.ffn.W_down
            og = fi(0, op); h = 0
            if pass_idx > 0:
                cf = fi(0, E.CARRY_OUT); ri = fi(1, RM)
                W[h, cf] = S; G[h, og] = 1.0; D[ri, h] = 1.0 / S; h += 1
                W[h, cf] = -S; G[h, og] = -1.0; D[ri, h] = 1.0 / S; h += 1
                for p in range(N):
                    _clr(self.flat.ffn, h, fi(p, op), fi(p, E.CARRY_OUT)); h += 2
            for p in range(N):
                ri, ci = fi(p, RM), fi(p, E.CARRY_OUT)
                for k in range(1, max_carry + 1):
                    thr = k * BASE
                    W[h, ri] = S; b[h] = -S * (thr - 1)
                    G[h, og] = 1.0; D[ri, h] = -float(BASE) / S; D[ci, h] = 1.0 / S; h += 1
                    W[h, ri] = S; b[h] = -S * float(thr)
                    G[h, og] = 1.0; D[ri, h] = float(BASE) / S; D[ci, h] = -1.0 / S; h += 1

    def forward(self, x):
        return self.flat(x)


class MulGenProp(nn.Module):
    def __init__(self):
        super().__init__()
        E, op = E8, E8.OP_MUL; RM = E.RESULT_MUL
        hidden_dim = 2 + 6 * (N - 1) + 2 * N
        self.flat = FlatFFN(hidden_dim)
        fi = self.flat._fi
        with torch.no_grad():
            W, b = self.flat.ffn.W_up, self.flat.ffn.b_up
            G, D = self.flat.ffn.W_gate, self.flat.ffn.W_down
            og = fi(0, op); h = 0
            for p in range(N):
                ri = fi(p, RM)
                if p > 0:
                    cf = fi(p - 1, E.CARRY_OUT)
                    W[h, cf] = S; G[h, og] = 1.0; D[ri, h] = 1.0 / S; h += 1
                    W[h, cf] = -S; G[h, og] = -1.0; D[ri, h] = 1.0 / S; h += 1
                    W[h, ri] = S; W[h, cf] = S; b[h] = -S * 15
                    G[h, og] = 1.0; D[ri, h] = -16.0 / S
                    D[fi(p, E.CARRY_OUT), h] = 1.0 / S; h += 1
                    W[h, ri] = S; W[h, cf] = S; b[h] = -S * 16
                    G[h, og] = 1.0; D[ri, h] = 16.0 / S
                    D[fi(p, E.CARRY_OUT), h] = -1.0 / S; h += 1
                    W[h, ri] = S; W[h, cf] = S; b[h] = -S * 14
                    G[h, og] = 1.0; D[fi(p, E.TEMP), h] = 1.0 / S; h += 1
                    W[h, ri] = S; W[h, cf] = S; b[h] = -S * 15
                    G[h, og] = 1.0; D[fi(p, E.TEMP), h] = -1.0 / S; h += 1
                else:
                    W[h, ri] = S; b[h] = -S * 14; G[h, og] = 1.0
                    D[fi(p, E.TEMP), h] = 1.0 / S; h += 1
                    W[h, ri] = S; b[h] = -S * 15; G[h, og] = 1.0
                    D[fi(p, E.TEMP), h] = -1.0 / S; h += 1
            for p in range(N):
                _clr(self.flat.ffn, h, fi(p, op), fi(p, E.CARRY_OUT)); h += 2

    def forward(self, x):
        return self.flat(x)


class MulLookahead(nn.Module):
    def __init__(self):
        super().__init__()
        E, op = E8, E8.OP_MUL
        hidden_dim = N * (N - 1) // 2 + 4 * N
        self.flat = FlatFFN(hidden_dim)
        fi = self.flat._fi
        with torch.no_grad():
            W, b = self.flat.ffn.W_up, self.flat.ffn.b_up
            G, D = self.flat.ffn.W_gate, self.flat.ffn.W_down
            og = fi(0, op); h = 0
            for i in range(1, N):
                W[h, fi(i - 1, E.CARRY_OUT)] = S; b[h] = -S * 0.5
                G[h, og] = 1.0; D[fi(i, E.CARRY_IN), h] = 2.0 / S; h += 1
            for p in range(N):
                _clr(self.flat.ffn, h, fi(p, op), fi(p, E.CARRY_OUT)); h += 2
                _clr(self.flat.ffn, h, fi(p, op), fi(p, E.TEMP)); h += 2

    def forward(self, x):
        return self.flat(x)


class MulCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        E, op = E8, E8.OP_MUL; RM = E.RESULT_MUL
        hidden_dim = 4 * (N - 1) + 2 * N
        self.flat = FlatFFN(hidden_dim)
        fi = self.flat._fi
        with torch.no_grad():
            W, b = self.flat.ffn.W_up, self.flat.ffn.b_up
            G, D = self.flat.ffn.W_gate, self.flat.ffn.W_down
            og = fi(0, op); h = 0
            for p in range(N):
                ri, ci = fi(p, RM), fi(p, E.CARRY_IN)
                if p > 0:
                    W[h, ci] = S; G[h, og] = 1.0; D[ri, h] = 1.0 / S; h += 1
                    W[h, ci] = -S; G[h, og] = -1.0; D[ri, h] = 1.0 / S; h += 1
                    W[h, ri] = S; W[h, ci] = S; b[h] = -S * 15
                    G[h, og] = 1.0; D[ri, h] = -16.0 / S; h += 1
                    W[h, ri] = S; W[h, ci] = S; b[h] = -S * 16
                    G[h, og] = 1.0; D[ri, h] = 16.0 / S; h += 1
            for p in range(N):
                _clr(self.flat.ffn, h, fi(p, op), fi(p, E.CARRY_IN)); h += 2

    def forward(self, x):
        return self.flat(x)


# -- Merge: RESULT_MUL -> RESULT (opcode-gated) --------------------------


class MergeResult(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = SwiGLUFFN(E8.DIM, 2)
        E = E8
        with torch.no_grad():
            W, G, D = self.ffn.W_up, self.ffn.W_gate, self.ffn.W_down
            W[0, E.RESULT_MUL] = S; G[0, E.OP_MUL] = 1.0
            D[E.RESULT, 0] = 1.0 / S
            W[1, E.RESULT_MUL] = -S; G[1, E.OP_MUL] = -1.0
            D[E.RESULT, 1] = 1.0 / S

    def forward(self, x):
        return self.ffn(x)


# -- Bitwise AND/OR/XOR pipelines ----------------------------------------


class BitExtract(nn.Module):
    def __init__(self, opcode):
        super().__init__()
        k = 4
        total_pairs = sum((1 << (k - j)) - 1 for j in range(k))
        hidden_dim = total_pairs * 4
        self.ffn = SwiGLUFFN(E8.DIM, hidden_dim)
        with torch.no_grad():
            W, b, G, D = self.ffn.W_up, self.ffn.b_up, self.ffn.W_gate, self.ffn.W_down
            h = 0
            for source, bit_base in [(E8.NIB_A, E8.BIT_A0), (E8.NIB_B, E8.BIT_B0)]:
                for j in range(k):
                    M = (1 << (k - j)) - 1
                    step_size = 1 << j
                    for m in range(1, M + 1):
                        threshold = m * step_size
                        sign = 1.0 if m % 2 == 1 else -1.0
                        W[h, source] = S
                        b[h] = -S * (threshold - 1.0)
                        G[h, opcode] = 1.0
                        D[bit_base + j, h] = sign / S
                        h += 1
                        W[h, source] = S
                        b[h] = -S * float(threshold)
                        G[h, opcode] = 1.0
                        D[bit_base + j, h] = -sign / S
                        h += 1

    def forward(self, x):
        return self.ffn(x)


class BitAndCombine(nn.Module):
    def __init__(self):
        super().__init__()
        k = 4
        hidden_dim = 2 * k + 4 * k
        self.ffn = SwiGLUFFN(E8.DIM, hidden_dim)
        with torch.no_grad():
            W, G, D = self.ffn.W_up, self.ffn.W_gate, self.ffn.W_down
            h = 0
            for j in range(k):
                weight = float(1 << j)
                a_s = E8.BIT_A0 + j
                b_s = E8.BIT_B0 + j
                W[h, a_s] = S; G[h, b_s] = 1.0
                D[E8.RESULT, h] = weight / S; h += 1
                W[h, a_s] = -S; G[h, b_s] = -1.0
                D[E8.RESULT, h] = weight / S; h += 1
            for j in range(k):
                _clr(self.ffn, h, E8.OP_AND, E8.BIT_A0 + j); h += 2
            for j in range(k):
                _clr(self.ffn, h, E8.OP_AND, E8.BIT_B0 + j); h += 2

    def forward(self, x):
        return self.ffn(x)


class BitOrCombine(nn.Module):
    def __init__(self):
        super().__init__()
        k = 4
        hidden_dim = 4 * k + 4 * k
        self.ffn = SwiGLUFFN(E8.DIM, hidden_dim)
        with torch.no_grad():
            W, G, D = self.ffn.W_up, self.ffn.W_gate, self.ffn.W_down
            h = 0
            for j in range(k):
                weight = float(1 << j)
                a_s = E8.BIT_A0 + j
                b_s = E8.BIT_B0 + j
                W[h, E8.OP_OR] = S; G[h, a_s] = 1.0
                D[E8.RESULT, h] = weight / S; h += 1
                W[h, E8.OP_OR] = S; G[h, b_s] = 1.0
                D[E8.RESULT, h] = weight / S; h += 1
                W[h, a_s] = S; G[h, b_s] = 1.0
                D[E8.RESULT, h] = -weight / S; h += 1
                W[h, a_s] = -S; G[h, b_s] = -1.0
                D[E8.RESULT, h] = -weight / S; h += 1
            for j in range(k):
                _clr(self.ffn, h, E8.OP_OR, E8.BIT_A0 + j); h += 2
            for j in range(k):
                _clr(self.ffn, h, E8.OP_OR, E8.BIT_B0 + j); h += 2

    def forward(self, x):
        return self.ffn(x)


class BitXorCombine(nn.Module):
    def __init__(self):
        super().__init__()
        k = 4
        hidden_dim = 4 * k + 4 * k
        self.ffn = SwiGLUFFN(E8.DIM, hidden_dim)
        with torch.no_grad():
            W, G, D = self.ffn.W_up, self.ffn.W_gate, self.ffn.W_down
            h = 0
            for j in range(k):
                weight = float(1 << j)
                a_s = E8.BIT_A0 + j
                b_s = E8.BIT_B0 + j
                W[h, E8.OP_XOR] = S; G[h, a_s] = 1.0
                D[E8.RESULT, h] = weight / S; h += 1
                W[h, E8.OP_XOR] = S; G[h, b_s] = 1.0
                D[E8.RESULT, h] = weight / S; h += 1
                W[h, a_s] = S; G[h, b_s] = 1.0
                D[E8.RESULT, h] = -2.0 * weight / S; h += 1
                W[h, a_s] = -S; G[h, b_s] = -1.0
                D[E8.RESULT, h] = -2.0 * weight / S; h += 1
            for j in range(k):
                _clr(self.ffn, h, E8.OP_XOR, E8.BIT_A0 + j); h += 2
            for j in range(k):
                _clr(self.ffn, h, E8.OP_XOR, E8.BIT_B0 + j); h += 2

    def forward(self, x):
        return self.ffn(x)


# -- DIV module (tensor ops, fp64 softmax1 reciprocal) --------------------


class DivCompute(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('base_powers', torch.tensor([1.0, 16.0], dtype=torch.float64))

    def forward(self, x):
        B, NP, D = x.shape
        opcode = x[:, 0, E8.OP_DIV]

        nib_a = x[:, :2, E8.NIB_A].double()
        nib_b = x[:, :2, E8.NIB_B].double()
        dividend = (nib_a * self.base_powers).sum(dim=-1)
        divisor = (nib_b * self.base_powers).sum(dim=-1)

        SS = 60.0
        val = (nib_b * self.base_powers).clamp(min=0.5)
        active = nib_b > 0.5
        scores = torch.where(active, torch.log(val), torch.full_like(val, -SS))
        mx = scores.max(dim=-1, keepdim=True).values
        ex = torch.exp(scores - mx)
        reciprocal = torch.exp(-mx.squeeze(-1)) / ex.sum(dim=-1).clamp(min=1e-30)

        q_float = dividend * reciprocal
        is_nonzero = divisor > 0.5
        q_float = torch.where(is_nonzero, q_float, torch.zeros_like(q_float))

        MAGIC = 12582912.0
        q32 = q_float.float()
        floor_all = ((q32 - 0.5 + 0.001 + MAGIC) - MAGIC)
        floor_hi = ((q32 / 16.0 - 0.5 + 0.001 + MAGIC) - MAGIC)
        chunk_lo = (floor_all - 16.0 * floor_hi).clamp(0, 15)

        delta = torch.zeros_like(x)
        delta[:, 0, E8.RESULT] = (opcode * chunk_lo.float())
        delta[:, 1, E8.RESULT] = (opcode * floor_hi.float())
        return x + delta


# -- MOD module (tensor ops, DIV + correction) ---------------------------


class ModCompute(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('base_powers', torch.tensor([1.0, 16.0], dtype=torch.float64))

    def forward(self, x):
        B, NP, D = x.shape
        opcode = x[:, 0, E8.OP_MOD]

        nib_a = x[:, :2, E8.NIB_A].double()
        nib_b = x[:, :2, E8.NIB_B].double()
        dividend = (nib_a * self.base_powers).sum(dim=-1)
        divisor = (nib_b * self.base_powers).sum(dim=-1)

        SS = 60.0
        val = (nib_b * self.base_powers).clamp(min=0.5)
        active = nib_b > 0.5
        scores = torch.where(active, torch.log(val), torch.full_like(val, -SS))
        mx = scores.max(dim=-1, keepdim=True).values
        ex = torch.exp(scores - mx)
        reciprocal = torch.exp(-mx.squeeze(-1)) / ex.sum(dim=-1).clamp(min=1e-30)

        q_float = (dividend - 1.0) * reciprocal
        is_nonzero = divisor > 0.5
        q_float = torch.where(is_nonzero, q_float, torch.zeros_like(q_float))

        MAGIC = 12582912.0
        q32 = q_float.float()
        quotient = ((q32 - 0.5 + 0.001 + MAGIC) - MAGIC).clamp(min=0)

        remainder = dividend - quotient.double() * divisor
        remainder = torch.where(is_nonzero, remainder, torch.zeros_like(dividend))
        needs_fix = (remainder >= divisor).double()
        remainder = remainder - needs_fix * divisor
        rem_int = torch.round(remainder).long().clamp(0, 255)

        r_lo = (rem_int % 16).float()
        r_hi = (rem_int // 16).clamp(max=15).float()

        delta = torch.zeros_like(x)
        delta[:, 0, E8.RESULT] = opcode * r_lo
        delta[:, 1, E8.RESULT] = opcode * r_hi
        return x + delta


# -- SHL/SHR modules (tensor ops, integer shift) -------------------------


class ShlCompute(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('base_powers', torch.tensor([1.0, 16.0], dtype=torch.float64))

    def forward(self, x):
        B, NP, D = x.shape
        opcode = x[:, 0, E8.OP_SHL]

        nib_a = x[:, :2, E8.NIB_A].double()
        nib_b = x[:, :2, E8.NIB_B].double()
        a_val = (nib_a * self.base_powers).sum(dim=-1).round().long().clamp(0, 255)
        n_val = (nib_b * self.base_powers).sum(dim=-1).round().long().clamp(0, 255)

        n_clamped = n_val.clamp(0, 7)
        valid = (n_val < 8)
        shift_factor = torch.pow(2, n_clamped.long())
        result = torch.where(valid, (a_val * shift_factor) & 255, torch.zeros_like(a_val))

        r_lo = (result % 16).float()
        r_hi = (result // 16).clamp(max=15).float()

        delta = torch.zeros_like(x)
        delta[:, 0, E8.RESULT] = opcode * r_lo
        delta[:, 1, E8.RESULT] = opcode * r_hi
        return x + delta


class ShrCompute(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('base_powers', torch.tensor([1.0, 16.0], dtype=torch.float64))

    def forward(self, x):
        B, NP, D = x.shape
        opcode = x[:, 0, E8.OP_SHR]

        nib_a = x[:, :2, E8.NIB_A].double()
        nib_b = x[:, :2, E8.NIB_B].double()
        a_val = (nib_a * self.base_powers).sum(dim=-1).round().long().clamp(0, 255)
        n_val = (nib_b * self.base_powers).sum(dim=-1).round().long().clamp(0, 255)

        n_clamped = n_val.clamp(0, 7)
        valid = (n_val < 8)
        shift_factor = torch.pow(2, n_clamped.long())
        result = torch.where(valid, (a_val / shift_factor).long(), torch.zeros_like(a_val))

        r_lo = (result % 16).float()
        r_hi = (result // 16).clamp(max=15).float()

        delta = torch.zeros_like(x)
        delta[:, 0, E8.RESULT] = opcode * r_lo
        delta[:, 1, E8.RESULT] = opcode * r_hi
        return x + delta


# -- Helpers --------------------------------------------------------------


def _mul_carry_passes():
    max_val = N * 15 * 15
    passes = []
    while True:
        mc = max_val // BASE
        if mc == 0:
            break
        passes.append(mc)
        if mc <= 1:
            break
        max_val = (BASE - 1) + mc
    return passes


# -- NeuralALU: forward is purely sequential ------------------------------


class NeuralALU(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            AddRawGen(), AddCarry(), AddFinal(),
            SubRawBorrow(), SubBorrow(), SubFinal(),
            MulSchoolbook(),
        ]
        for idx, mc in enumerate(_mul_carry_passes()):
            layers.append(MulCarryPass(mc, pass_idx=idx))
        layers += [MulGenProp(), MulLookahead(), MulCorrect()]
        layers.append(MergeResult())
        layers += [
            BitExtract(E8.OP_AND), BitAndCombine(),
            BitExtract(E8.OP_OR), BitOrCombine(),
            BitExtract(E8.OP_XOR), BitXorCombine(),
        ]
        layers += [DivCompute(), ModCompute(), ShlCompute(), ShrCompute()]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
