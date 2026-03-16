#!/usr/bin/env python3
"""
Comprehensive test suite for V5 Pure Generative VM.
Tests all operations with edge cases and boundary conditions.
"""
import sys
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')

import torch
from pure_gen_vm_v5 import (
    EmbedDimsV5, EfficientAddSubLayer, EfficientMulProductsLayer,
    EfficientComparisonLayer, EfficientDivLayer, BranchZeroLayer,
    EfficientBitwiseLayer, EfficientShiftLayer, ValueToOneHotV5,
    ControlFlowLayer, StackFrameLayer, Opcode
)

E = EmbedDimsV5
DIM = E.DIM

def create_input(a, b, opcode):
    """Create value-encoded input tensor."""
    x = torch.zeros(1, 1, DIM)
    for nib in range(8):
        x[0, 0, E.OP_A_VAL_START + nib] = float((a >> (nib * 4)) & 0xF)
        x[0, 0, E.OP_B_VAL_START + nib] = float((b >> (nib * 4)) & 0xF)
    x[0, 0, E.OPCODE_START + opcode] = 1.0
    return x

def read_result(out):
    """Read value-encoded result as integer."""
    result = 0
    for nib in range(8):
        val = out[0, 0, E.RESULT_VAL_START + nib].item()
        result |= (int(round(val)) & 0xF) << (nib * 4)
    return result

def run_tests(name, layer, tests, result_reader=None):
    """Run a batch of tests and report results."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print('='*60)

    passed = 0
    failed = 0
    failures = []

    for test in tests:
        if len(test) == 4:
            a, b, opcode, expected = test
            op_name = ""
        else:
            a, b, opcode, expected, op_name = test

        x = create_input(a, b, opcode)
        with torch.no_grad():
            out = layer(x)

        if result_reader:
            result = result_reader(out)
        else:
            result = read_result(out)

        if result == expected:
            passed += 1
            print(f"  PASS: {op_name} {hex(a)} op {hex(b)} = {hex(expected)}")
        else:
            failed += 1
            failures.append((op_name, a, b, expected, result))
            print(f"  FAIL: {op_name} {hex(a)} op {hex(b)} = {hex(expected)}, got {hex(result)}")

    print(f"\n{name}: {passed}/{passed+failed} passed")
    return passed, failed, failures

# =============================================================================
# ADD TESTS
# =============================================================================
add_layer = EfficientAddSubLayer(DIM)

add_tests = [
    # Basic single nibble
    (0, 0, Opcode.ADD, 0, "0+0"),
    (0, 1, Opcode.ADD, 1, "0+1"),
    (1, 0, Opcode.ADD, 1, "1+0"),
    (5, 3, Opcode.ADD, 8, "5+3"),
    (7, 8, Opcode.ADD, 15, "7+8"),
    (15, 0, Opcode.ADD, 15, "15+0"),
    (0, 15, Opcode.ADD, 15, "0+15"),

    # Single nibble with carry
    (9, 9, Opcode.ADD, 0x12, "9+9 carry"),
    (15, 1, Opcode.ADD, 0x10, "15+1 carry"),
    (15, 15, Opcode.ADD, 0x1E, "15+15 carry"),
    (8, 8, Opcode.ADD, 0x10, "8+8 carry"),

    # Multi-nibble
    (0x12, 0x34, Opcode.ADD, 0x46, "0x12+0x34"),
    (0xFF, 0x01, Opcode.ADD, 0x100, "0xFF+0x01 carry"),
    (0xFF, 0xFF, Opcode.ADD, 0x1FE, "0xFF+0xFF"),
    (0x1234, 0x5678, Opcode.ADD, 0x68AC, "0x1234+0x5678"),
    (0xFFFF, 0x0001, Opcode.ADD, 0x10000, "0xFFFF+0x0001"),

    # 32-bit
    (0x12345678, 0x11111111, Opcode.ADD, 0x23456789, "32-bit add"),
    (0xFFFFFFFF, 0x00000001, Opcode.ADD, 0x00000000, "32-bit overflow"),
    (0x80000000, 0x80000000, Opcode.ADD, 0x00000000, "max+max overflow"),

    # Carry propagation chain
    (0x0FFFFFFF, 0x00000001, Opcode.ADD, 0x10000000, "long carry chain"),
    (0x00FF00FF, 0x00010001, Opcode.ADD, 0x01000100, "multi carry"),
]

# =============================================================================
# SUB TESTS
# =============================================================================
sub_tests = [
    # Basic single nibble (no borrow)
    (0, 0, Opcode.SUB, 0, "0-0"),
    (5, 3, Opcode.SUB, 2, "5-3"),
    (15, 0, Opcode.SUB, 15, "15-0"),
    (15, 15, Opcode.SUB, 0, "15-15"),
    (10, 3, Opcode.SUB, 7, "10-3"),
    (8, 8, Opcode.SUB, 0, "8-8"),

    # Single nibble with borrow
    (0, 1, Opcode.SUB, 0xFFFFFFFF, "0-1 borrow"),
    (5, 10, Opcode.SUB, 0xFFFFFFFB, "5-10 borrow"),
    (0, 15, Opcode.SUB, 0xFFFFFFF1, "0-15 borrow"),

    # Multi-nibble with borrow
    (0x100, 0x01, Opcode.SUB, 0xFF, "0x100-0x01"),
    (0x1000, 0x0001, Opcode.SUB, 0x0FFF, "0x1000-0x0001"),
    (0x5678, 0x1234, Opcode.SUB, 0x4444, "0x5678-0x1234"),

    # 32-bit
    (0x12345678, 0x11111111, Opcode.SUB, 0x01234567, "32-bit sub"),
    (0x00000000, 0x00000001, Opcode.SUB, 0xFFFFFFFF, "0-1 32bit"),
    (0x10000000, 0x00000001, Opcode.SUB, 0x0FFFFFFF, "long borrow"),

    # Borrow chain
    (0x10000000, 0x00000001, Opcode.SUB, 0x0FFFFFFF, "borrow chain"),
    (0x01000100, 0x00010001, Opcode.SUB, 0x00FF00FF, "multi borrow"),
]

# =============================================================================
# MUL TESTS
# =============================================================================
mul_layer = EfficientMulProductsLayer(DIM)

def read_mul_result(out):
    """Read single product from MUL output."""
    return int(round(out[0, 0, E.MUL_PROD_START].item()))

mul_tests = [
    # Basic
    (0, 0, Opcode.MUL, 0, "0*0"),
    (0, 1, Opcode.MUL, 0, "0*1"),
    (1, 0, Opcode.MUL, 0, "1*0"),
    (1, 1, Opcode.MUL, 1, "1*1"),
    (2, 3, Opcode.MUL, 6, "2*3"),
    (5, 3, Opcode.MUL, 15, "5*3"),
    (7, 6, Opcode.MUL, 42, "7*6"),

    # Boundary
    (15, 1, Opcode.MUL, 15, "15*1"),
    (1, 15, Opcode.MUL, 15, "1*15"),
    (15, 15, Opcode.MUL, 225, "15*15"),
    (8, 8, Opcode.MUL, 64, "8*8"),
    (10, 10, Opcode.MUL, 100, "10*10"),

    # All nibble values
    (2, 2, Opcode.MUL, 4, "2*2"),
    (3, 3, Opcode.MUL, 9, "3*3"),
    (4, 4, Opcode.MUL, 16, "4*4"),
    (5, 5, Opcode.MUL, 25, "5*5"),
    (6, 6, Opcode.MUL, 36, "6*6"),
    (7, 7, Opcode.MUL, 49, "7*7"),
    (9, 9, Opcode.MUL, 81, "9*9"),
    (11, 11, Opcode.MUL, 121, "11*11"),
    (12, 12, Opcode.MUL, 144, "12*12"),
    (13, 13, Opcode.MUL, 169, "13*13"),
    (14, 14, Opcode.MUL, 196, "14*14"),
]

# =============================================================================
# DIV TESTS
# =============================================================================
div_layer = EfficientDivLayer(DIM)

div_tests = [
    # Basic
    (0, 1, Opcode.DIV, 0, "0/1"),
    (1, 1, Opcode.DIV, 1, "1/1"),
    (5, 1, Opcode.DIV, 5, "5/1"),
    (10, 2, Opcode.DIV, 5, "10/2"),
    (15, 3, Opcode.DIV, 5, "15/3"),
    (100, 10, Opcode.DIV, 10, "100/10"),

    # Truncation
    (7, 2, Opcode.DIV, 3, "7/2 trunc"),
    (10, 3, Opcode.DIV, 3, "10/3 trunc"),
    (100, 7, Opcode.DIV, 14, "100/7"),
    (255, 3, Opcode.DIV, 85, "255/3"),

    # Large values
    (1000, 10, Opcode.DIV, 100, "1000/10"),
    (0x1000, 0x10, Opcode.DIV, 0x100, "0x1000/0x10"),
    (0xFFFF, 0xFF, Opcode.DIV, 0x101, "0xFFFF/0xFF"),

    # Edge cases
    (15, 15, Opcode.DIV, 1, "15/15"),
    (1, 15, Opcode.DIV, 0, "1/15 trunc to 0"),
    (14, 15, Opcode.DIV, 0, "14/15 trunc to 0"),
]

# =============================================================================
# MOD TESTS
# =============================================================================
mod_tests = [
    # Basic
    (0, 1, Opcode.MOD, 0, "0%1"),
    (1, 1, Opcode.MOD, 0, "1%1"),
    (5, 3, Opcode.MOD, 2, "5%3"),
    (10, 3, Opcode.MOD, 1, "10%3"),
    (100, 7, Opcode.MOD, 2, "100%7"),

    # Zero remainder
    (15, 3, Opcode.MOD, 0, "15%3"),
    (15, 5, Opcode.MOD, 0, "15%5"),
    (255, 3, Opcode.MOD, 0, "255%3"),
    (256, 16, Opcode.MOD, 0, "256%16"),

    # All remainders
    (17, 5, Opcode.MOD, 2, "17%5"),
    (18, 5, Opcode.MOD, 3, "18%5"),
    (19, 5, Opcode.MOD, 4, "19%5"),
    (20, 5, Opcode.MOD, 0, "20%5"),
    (21, 5, Opcode.MOD, 1, "21%5"),

    # Large values
    (1000, 13, Opcode.MOD, 12, "1000%13"),
    (0xFFFF, 0x100, Opcode.MOD, 0xFF, "0xFFFF%0x100"),
]

# =============================================================================
# COMPARISON TESTS
# =============================================================================
cmp_layer = EfficientComparisonLayer(DIM)

def read_cmp_eq(out):
    return 1 if out[0, 0, E.CMP_EQ].item() > 0.5 else 0

def read_cmp_ne(out):
    return 1 if out[0, 0, E.CMP_NE].item() > 0.5 else 0

def read_cmp_lt(out):
    return 1 if out[0, 0, E.CMP_LT].item() > 0.5 else 0

def read_cmp_gt(out):
    return 1 if out[0, 0, E.CMP_GT].item() > 0.5 else 0

def read_cmp_le(out):
    return 1 if out[0, 0, E.CMP_LE].item() > 0.5 else 0

def read_cmp_ge(out):
    return 1 if out[0, 0, E.CMP_GE].item() > 0.5 else 0

# EQ tests
eq_tests = [
    (0, 0, Opcode.EQ, 1, "0==0"),
    (1, 1, Opcode.EQ, 1, "1==1"),
    (15, 15, Opcode.EQ, 1, "15==15"),
    (5, 3, Opcode.EQ, 0, "5==3"),
    (0, 1, Opcode.EQ, 0, "0==1"),
    (0x1234, 0x1234, Opcode.EQ, 1, "0x1234==0x1234"),
    (0x1234, 0x1235, Opcode.EQ, 0, "0x1234==0x1235"),
    (0xFFFFFFFF, 0xFFFFFFFF, Opcode.EQ, 1, "max==max"),
    (0xFFFFFFFF, 0xFFFFFFFE, Opcode.EQ, 0, "max==max-1"),
    (0x12345678, 0x12345678, Opcode.EQ, 1, "32bit eq"),
    (0x12345678, 0x12345679, Opcode.EQ, 0, "32bit ne"),
]

# NE tests
ne_tests = [
    (0, 0, Opcode.NE, 0, "0!=0"),
    (5, 3, Opcode.NE, 1, "5!=3"),
    (0x1234, 0x1234, Opcode.NE, 0, "0x1234!=0x1234"),
    (0x1234, 0x1235, Opcode.NE, 1, "0x1234!=0x1235"),
]

# LT tests
lt_tests = [
    (0, 0, Opcode.LT, 0, "0<0"),
    (0, 1, Opcode.LT, 1, "0<1"),
    (1, 0, Opcode.LT, 0, "1<0"),
    (3, 5, Opcode.LT, 1, "3<5"),
    (5, 3, Opcode.LT, 0, "5<3"),
    (5, 5, Opcode.LT, 0, "5<5"),
    (0x1234, 0x1235, Opcode.LT, 1, "0x1234<0x1235"),
    (0x1235, 0x1234, Opcode.LT, 0, "0x1235<0x1234"),
    (0x1234, 0x2234, Opcode.LT, 1, "0x1234<0x2234 MSB"),
    (0x00FF, 0x0100, Opcode.LT, 1, "0x00FF<0x0100"),
]

# GT tests
gt_tests = [
    (0, 0, Opcode.GT, 0, "0>0"),
    (1, 0, Opcode.GT, 1, "1>0"),
    (0, 1, Opcode.GT, 0, "0>1"),
    (5, 3, Opcode.GT, 1, "5>3"),
    (3, 5, Opcode.GT, 0, "3>5"),
    (5, 5, Opcode.GT, 0, "5>5"),
    (0x1235, 0x1234, Opcode.GT, 1, "0x1235>0x1234"),
    (0x1234, 0x1235, Opcode.GT, 0, "0x1234>0x1235"),
    (0x2234, 0x1234, Opcode.GT, 1, "0x2234>0x1234 MSB"),
]

# LE tests
le_tests = [
    (0, 0, Opcode.LE, 1, "0<=0"),
    (0, 1, Opcode.LE, 1, "0<=1"),
    (1, 0, Opcode.LE, 0, "1<=0"),
    (5, 5, Opcode.LE, 1, "5<=5"),
    (3, 5, Opcode.LE, 1, "3<=5"),
    (5, 3, Opcode.LE, 0, "5<=3"),
]

# GE tests
ge_tests = [
    (0, 0, Opcode.GE, 1, "0>=0"),
    (1, 0, Opcode.GE, 1, "1>=0"),
    (0, 1, Opcode.GE, 0, "0>=1"),
    (5, 5, Opcode.GE, 1, "5>=5"),
    (5, 3, Opcode.GE, 1, "5>=3"),
    (3, 5, Opcode.GE, 0, "3>=5"),
]

# =============================================================================
# BITWISE TESTS
# =============================================================================
bitwise_layer = EfficientBitwiseLayer(DIM)

# AND tests
and_tests = [
    (0x00, 0x00, Opcode.AND, 0x00, "0&0"),
    (0xFF, 0x00, Opcode.AND, 0x00, "FF&0"),
    (0x00, 0xFF, Opcode.AND, 0x00, "0&FF"),
    (0xFF, 0xFF, Opcode.AND, 0xFF, "FF&FF"),
    (0xAA, 0x55, Opcode.AND, 0x00, "AA&55"),
    (0xFF, 0x55, Opcode.AND, 0x55, "FF&55"),
    (0x0F, 0xF0, Opcode.AND, 0x00, "0F&F0"),
    (0x12345678, 0xF0F0F0F0, Opcode.AND, 0x10305070, "mask"),
    (0xFFFFFFFF, 0x12345678, Opcode.AND, 0x12345678, "identity"),
    (0x00000000, 0xFFFFFFFF, Opcode.AND, 0x00000000, "zero"),
]

# OR tests
or_tests = [
    (0x00, 0x00, Opcode.OR, 0x00, "0|0"),
    (0xFF, 0x00, Opcode.OR, 0xFF, "FF|0"),
    (0x00, 0xFF, Opcode.OR, 0xFF, "0|FF"),
    (0xFF, 0xFF, Opcode.OR, 0xFF, "FF|FF"),
    (0xAA, 0x55, Opcode.OR, 0xFF, "AA|55"),
    (0x0F, 0xF0, Opcode.OR, 0xFF, "0F|F0"),
    (0x12340000, 0x00005678, Opcode.OR, 0x12345678, "combine"),
    (0x00000000, 0xFFFFFFFF, Opcode.OR, 0xFFFFFFFF, "all ones"),
]

# XOR tests
xor_tests = [
    (0x00, 0x00, Opcode.XOR, 0x00, "0^0"),
    (0xFF, 0x00, Opcode.XOR, 0xFF, "FF^0"),
    (0x00, 0xFF, Opcode.XOR, 0xFF, "0^FF"),
    (0xFF, 0xFF, Opcode.XOR, 0x00, "FF^FF"),
    (0xAA, 0x55, Opcode.XOR, 0xFF, "AA^55"),
    (0xFF, 0xAA, Opcode.XOR, 0x55, "FF^AA"),
    (0x12345678, 0x12345678, Opcode.XOR, 0x00000000, "self=0"),
    (0xAAAAAAAA, 0x55555555, Opcode.XOR, 0xFFFFFFFF, "alt"),
]

# =============================================================================
# SHIFT TESTS
# =============================================================================
shift_layer = EfficientShiftLayer(DIM)

# SHL tests
shl_tests = [
    (0x01, 0, Opcode.SHL, 0x01, "1<<0"),
    (0x01, 1, Opcode.SHL, 0x02, "1<<1"),
    (0x01, 4, Opcode.SHL, 0x10, "1<<4"),
    (0x01, 8, Opcode.SHL, 0x100, "1<<8"),
    (0x01, 16, Opcode.SHL, 0x10000, "1<<16"),
    (0x0F, 4, Opcode.SHL, 0xF0, "F<<4"),
    (0xFF, 8, Opcode.SHL, 0xFF00, "FF<<8"),
    (0x12, 8, Opcode.SHL, 0x1200, "12<<8"),
    (0x01, 31, Opcode.SHL, 0x80000000, "1<<31"),
    (0xFFFFFFFF, 1, Opcode.SHL, 0xFFFFFFFE, "max<<1"),
]

# SHR tests
shr_tests = [
    (0x01, 0, Opcode.SHR, 0x01, "1>>0"),
    (0x02, 1, Opcode.SHR, 0x01, "2>>1"),
    (0x10, 4, Opcode.SHR, 0x01, "10>>4"),
    (0x100, 8, Opcode.SHR, 0x01, "100>>8"),
    (0xF0, 4, Opcode.SHR, 0x0F, "F0>>4"),
    (0xFF00, 8, Opcode.SHR, 0xFF, "FF00>>8"),
    (0xFF, 1, Opcode.SHR, 0x7F, "FF>>1"),
    (0xFFFF, 8, Opcode.SHR, 0xFF, "FFFF>>8"),
    (0x80000000, 31, Opcode.SHR, 0x01, "80000000>>31"),
    (0xFFFFFFFF, 1, Opcode.SHR, 0x7FFFFFFF, "max>>1"),
]

# =============================================================================
# BZ/BNZ TESTS
# =============================================================================
bz_layer = BranchZeroLayer(DIM)

def read_is_zero(out):
    return 1 if out[0, 0, E.CMP_EQ].item() > 0.5 else 0

bz_tests = [
    (0, "zero"),
    (1, "one"),
    (0xF, "nibble max"),
    (0x10, "second nibble"),
    (0x100, "third nibble"),
    (0x1000, "fourth nibble"),
    (0x10000, "fifth nibble"),
    (0x100000, "sixth nibble"),
    (0x1000000, "seventh nibble"),
    (0x10000000, "eighth nibble"),
    (0xFFFFFFFF, "all ones"),
]

# =============================================================================
# CONTROL FLOW TESTS (JMP, JSR, BZ, BNZ)
# =============================================================================
ctrl_layer = ControlFlowLayer(DIM)

def create_ctrl_input(pc, ax, imm, opcode):
    """Create input for control flow testing."""
    x = torch.zeros(1, 1, DIM)
    # Set PC
    for nib in range(8):
        x[0, 0, E.PC_VAL_START + nib] = float((pc >> (nib * 4)) & 0xF)
    # Set AX (accumulator)
    for nib in range(8):
        x[0, 0, E.AX_VAL_START + nib] = float((ax >> (nib * 4)) & 0xF)
    # Set immediate
    for nib in range(8):
        x[0, 0, E.IMM_VAL_START + nib] = float((imm >> (nib * 4)) & 0xF)
    # Set opcode
    x[0, 0, E.OPCODE_START + opcode] = 1.0
    return x

def read_pc(out):
    """Read PC value from output."""
    result = 0
    for nib in range(8):
        val = out[0, 0, E.PC_VAL_START + nib].item()
        result |= (int(round(val)) & 0xF) << (nib * 4)
    return result

def read_branch_taken(out):
    """Read branch taken flag."""
    return out[0, 0, E.BRANCH_TAKEN].item() > 0.5

# JMP tests: PC = immediate
jmp_tests = [
    (0x100, 0, 0x200, Opcode.JMP, 0x200, True, "JMP to 0x200"),
    (0x0, 0, 0x1000, Opcode.JMP, 0x1000, True, "JMP to 0x1000"),
    (0xFFFF, 0, 0x0, Opcode.JMP, 0x0, True, "JMP to 0x0"),
]

# BZ tests: PC = immediate if AX == 0, else PC + 8
bz_tests_ctrl = [
    (0x100, 0, 0x200, Opcode.BZ, 0x200, True, "BZ taken (AX=0)"),
    (0x100, 1, 0x200, Opcode.BZ, 0x108, False, "BZ not taken (AX=1)"),
    (0x100, 0xFF, 0x200, Opcode.BZ, 0x108, False, "BZ not taken (AX=FF)"),
    (0x100, 0x12345678, 0x200, Opcode.BZ, 0x108, False, "BZ not taken (AX large)"),
]

# BNZ tests: PC = immediate if AX != 0, else PC + 8
bnz_tests = [
    (0x100, 0, 0x200, Opcode.BNZ, 0x108, False, "BNZ not taken (AX=0)"),
    (0x100, 1, 0x200, Opcode.BNZ, 0x200, True, "BNZ taken (AX=1)"),
    (0x100, 0xFF, 0x200, Opcode.BNZ, 0x200, True, "BNZ taken (AX=FF)"),
]

# =============================================================================
# STACK FRAME TESTS (ENT, ADJ, LEV)
# =============================================================================
stack_layer = StackFrameLayer(DIM)

def create_stack_input(sp, bp, imm, opcode):
    """Create input for stack frame testing."""
    x = torch.zeros(1, 1, DIM)
    # Set SP
    for nib in range(8):
        x[0, 0, E.SP_VAL_START + nib] = float((sp >> (nib * 4)) & 0xF)
    # Set BP
    for nib in range(8):
        x[0, 0, E.BP_VAL_START + nib] = float((bp >> (nib * 4)) & 0xF)
    # Set immediate
    for nib in range(8):
        x[0, 0, E.IMM_VAL_START + nib] = float((imm >> (nib * 4)) & 0xF)
    # Set opcode
    x[0, 0, E.OPCODE_START + opcode] = 1.0
    return x

def read_sp(out):
    """Read SP value from output."""
    result = 0
    for nib in range(8):
        val = out[0, 0, E.SP_VAL_START + nib].item()
        result |= (int(round(val)) & 0xF) << (nib * 4)
    return result

def read_bp(out):
    """Read BP value from output."""
    result = 0
    for nib in range(8):
        val = out[0, 0, E.BP_VAL_START + nib].item()
        result |= (int(round(val)) & 0xF) << (nib * 4)
    return result

# ADJ tests: SP = SP + imm
adj_tests = [
    (0x1000, 0, 0x10, Opcode.ADJ, 0x1010, "ADJ +16"),
    (0x1000, 0, 0x100, Opcode.ADJ, 0x1100, "ADJ +256"),
    (0x1000, 0, 0x8, Opcode.ADJ, 0x1008, "ADJ +8"),
    (0x0FF0, 0, 0x20, Opcode.ADJ, 0x1010, "ADJ with carry"),
]

# ENT tests: BP = SP, SP = SP - imm
ent_tests = [
    (0x1000, 0x500, 0x10, Opcode.ENT, 0x0FF0, 0x1000, "ENT -16"),
    (0x1000, 0x500, 0x100, Opcode.ENT, 0x0F00, 0x1000, "ENT -256"),
    (0x2000, 0x100, 0x8, Opcode.ENT, 0x1FF8, 0x2000, "ENT -8"),
]

# LEV tests: SP = BP
lev_tests = [
    (0x1000, 0x2000, 0, Opcode.LEV, 0x2000, "LEV restore SP"),
    (0x500, 0x1000, 0, Opcode.LEV, 0x1000, "LEV restore SP 2"),
]


# =============================================================================
# RUN ALL TESTS
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("V5 COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    all_failures = []

    # ADD
    p, f, failures = run_tests("ADD", add_layer, add_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("ADD", *fail) for fail in failures])

    # SUB
    p, f, failures = run_tests("SUB", add_layer, sub_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("SUB", *fail) for fail in failures])

    # MUL
    p, f, failures = run_tests("MUL (single nibble products)", mul_layer, mul_tests, read_mul_result)
    total_passed += p
    total_failed += f
    all_failures.extend([("MUL", *fail) for fail in failures])

    # DIV
    p, f, failures = run_tests("DIV", div_layer, div_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("DIV", *fail) for fail in failures])

    # MOD
    p, f, failures = run_tests("MOD", div_layer, mod_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("MOD", *fail) for fail in failures])

    # EQ
    p, f, failures = run_tests("EQ", cmp_layer, eq_tests, read_cmp_eq)
    total_passed += p
    total_failed += f
    all_failures.extend([("EQ", *fail) for fail in failures])

    # NE
    p, f, failures = run_tests("NE", cmp_layer, ne_tests, read_cmp_ne)
    total_passed += p
    total_failed += f
    all_failures.extend([("NE", *fail) for fail in failures])

    # LT
    p, f, failures = run_tests("LT", cmp_layer, lt_tests, read_cmp_lt)
    total_passed += p
    total_failed += f
    all_failures.extend([("LT", *fail) for fail in failures])

    # GT
    p, f, failures = run_tests("GT", cmp_layer, gt_tests, read_cmp_gt)
    total_passed += p
    total_failed += f
    all_failures.extend([("GT", *fail) for fail in failures])

    # LE
    p, f, failures = run_tests("LE", cmp_layer, le_tests, read_cmp_le)
    total_passed += p
    total_failed += f
    all_failures.extend([("LE", *fail) for fail in failures])

    # GE
    p, f, failures = run_tests("GE", cmp_layer, ge_tests, read_cmp_ge)
    total_passed += p
    total_failed += f
    all_failures.extend([("GE", *fail) for fail in failures])

    # AND
    p, f, failures = run_tests("AND", bitwise_layer, and_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("AND", *fail) for fail in failures])

    # OR
    p, f, failures = run_tests("OR", bitwise_layer, or_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("OR", *fail) for fail in failures])

    # XOR
    p, f, failures = run_tests("XOR", bitwise_layer, xor_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("XOR", *fail) for fail in failures])

    # SHL
    p, f, failures = run_tests("SHL", shift_layer, shl_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("SHL", *fail) for fail in failures])

    # SHR
    p, f, failures = run_tests("SHR", shift_layer, shr_tests)
    total_passed += p
    total_failed += f
    all_failures.extend([("SHR", *fail) for fail in failures])

    # BZ (zero detection)
    print(f"\n{'='*60}")
    print("Testing BZ Layer (Zero Detection)")
    print('='*60)
    bz_passed = 0
    bz_failed = 0
    for val, name in bz_tests:
        x = torch.zeros(1, 1, DIM)
        for nib in range(8):
            x[0, 0, E.RESULT_VAL_START + nib] = float((val >> (nib * 4)) & 0xF)
        with torch.no_grad():
            out = bz_layer(x)
        is_zero = read_is_zero(out)
        expected = 1 if val == 0 else 0
        if is_zero == expected:
            bz_passed += 1
            print(f"  PASS: {name}: {hex(val)} is_zero={is_zero}")
        else:
            bz_failed += 1
            all_failures.append(("BZ", name, val, expected, is_zero))
            print(f"  FAIL: {name}: {hex(val)} expected is_zero={expected}, got {is_zero}")
    print(f"\nBZ Layer: {bz_passed}/{bz_passed+bz_failed} passed")
    total_passed += bz_passed
    total_failed += bz_failed

    # JMP
    print(f"\n{'='*60}")
    print("Testing JMP (Unconditional Jump)")
    print('='*60)
    jmp_passed = 0
    jmp_failed = 0
    for pc, ax, imm, opcode, expected_pc, expected_taken, name in jmp_tests:
        x = create_ctrl_input(pc, ax, imm, opcode)
        with torch.no_grad():
            out = ctrl_layer(x)
        result_pc = read_pc(out)
        taken = read_branch_taken(out)
        if result_pc == expected_pc and taken == expected_taken:
            jmp_passed += 1
            print(f"  PASS: {name}: PC={hex(result_pc)}, taken={taken}")
        else:
            jmp_failed += 1
            all_failures.append(("JMP", name, expected_pc, result_pc))
            print(f"  FAIL: {name}: expected PC={hex(expected_pc)}, got {hex(result_pc)}")
    print(f"\nJMP: {jmp_passed}/{jmp_passed+jmp_failed} passed")
    total_passed += jmp_passed
    total_failed += jmp_failed

    # BZ (control flow)
    print(f"\n{'='*60}")
    print("Testing BZ (Branch if Zero - Control Flow)")
    print('='*60)
    bz_ctrl_passed = 0
    bz_ctrl_failed = 0
    for pc, ax, imm, opcode, expected_pc, expected_taken, name in bz_tests_ctrl:
        x = create_ctrl_input(pc, ax, imm, opcode)
        with torch.no_grad():
            out = ctrl_layer(x)
        result_pc = read_pc(out)
        taken = read_branch_taken(out)
        if result_pc == expected_pc:
            bz_ctrl_passed += 1
            print(f"  PASS: {name}: PC={hex(result_pc)}, taken={taken}")
        else:
            bz_ctrl_failed += 1
            all_failures.append(("BZ_CTRL", name, expected_pc, result_pc))
            print(f"  FAIL: {name}: expected PC={hex(expected_pc)}, got {hex(result_pc)}")
    print(f"\nBZ Control: {bz_ctrl_passed}/{bz_ctrl_passed+bz_ctrl_failed} passed")
    total_passed += bz_ctrl_passed
    total_failed += bz_ctrl_failed

    # BNZ
    print(f"\n{'='*60}")
    print("Testing BNZ (Branch if Not Zero)")
    print('='*60)
    bnz_passed = 0
    bnz_failed = 0
    for pc, ax, imm, opcode, expected_pc, expected_taken, name in bnz_tests:
        x = create_ctrl_input(pc, ax, imm, opcode)
        with torch.no_grad():
            out = ctrl_layer(x)
        result_pc = read_pc(out)
        taken = read_branch_taken(out)
        if result_pc == expected_pc:
            bnz_passed += 1
            print(f"  PASS: {name}: PC={hex(result_pc)}, taken={taken}")
        else:
            bnz_failed += 1
            all_failures.append(("BNZ", name, expected_pc, result_pc))
            print(f"  FAIL: {name}: expected PC={hex(expected_pc)}, got {hex(result_pc)}")
    print(f"\nBNZ: {bnz_passed}/{bnz_passed+bnz_failed} passed")
    total_passed += bnz_passed
    total_failed += bnz_failed

    # ADJ
    print(f"\n{'='*60}")
    print("Testing ADJ (Adjust Stack Pointer)")
    print('='*60)
    adj_passed = 0
    adj_failed = 0
    for sp, bp, imm, opcode, expected_sp, name in adj_tests:
        x = create_stack_input(sp, bp, imm, opcode)
        with torch.no_grad():
            out = stack_layer(x)
        result_sp = read_sp(out)
        if result_sp == expected_sp:
            adj_passed += 1
            print(f"  PASS: {name}: SP={hex(result_sp)}")
        else:
            adj_failed += 1
            all_failures.append(("ADJ", name, expected_sp, result_sp))
            print(f"  FAIL: {name}: expected SP={hex(expected_sp)}, got {hex(result_sp)}")
    print(f"\nADJ: {adj_passed}/{adj_passed+adj_failed} passed")
    total_passed += adj_passed
    total_failed += adj_failed

    # ENT
    print(f"\n{'='*60}")
    print("Testing ENT (Enter Function)")
    print('='*60)
    ent_passed = 0
    ent_failed = 0
    for sp, bp, imm, opcode, expected_sp, expected_bp, name in ent_tests:
        x = create_stack_input(sp, bp, imm, opcode)
        with torch.no_grad():
            out = stack_layer(x)
        result_sp = read_sp(out)
        result_bp = read_bp(out)
        if result_sp == expected_sp and result_bp == expected_bp:
            ent_passed += 1
            print(f"  PASS: {name}: SP={hex(result_sp)}, BP={hex(result_bp)}")
        else:
            ent_failed += 1
            all_failures.append(("ENT", name, (expected_sp, expected_bp), (result_sp, result_bp)))
            print(f"  FAIL: {name}: expected SP={hex(expected_sp)}, BP={hex(expected_bp)}")
            print(f"         got SP={hex(result_sp)}, BP={hex(result_bp)}")
    print(f"\nENT: {ent_passed}/{ent_passed+ent_failed} passed")
    total_passed += ent_passed
    total_failed += ent_failed

    # LEV
    print(f"\n{'='*60}")
    print("Testing LEV (Leave Function)")
    print('='*60)
    lev_passed = 0
    lev_failed = 0
    for sp, bp, imm, opcode, expected_sp, name in lev_tests:
        x = create_stack_input(sp, bp, imm, opcode)
        with torch.no_grad():
            out = stack_layer(x)
        result_sp = read_sp(out)
        if result_sp == expected_sp:
            lev_passed += 1
            print(f"  PASS: {name}: SP={hex(result_sp)}")
        else:
            lev_failed += 1
            all_failures.append(("LEV", name, expected_sp, result_sp))
            print(f"  FAIL: {name}: expected SP={hex(expected_sp)}, got {hex(result_sp)}")
    print(f"\nLEV: {lev_passed}/{lev_passed+lev_failed} passed")
    total_passed += lev_passed
    total_failed += lev_failed

    # =============================================================================
    # SUMMARY
    # =============================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total: {total_passed}/{total_passed + total_failed} tests passed")
    print(f"Pass rate: {100*total_passed/(total_passed+total_failed):.1f}%")

    if all_failures:
        print(f"\nFailed tests ({len(all_failures)}):")
        for fail in all_failures:
            print(f"  {fail}")
    else:
        print("\nALL TESTS PASSED!")

    # Weight count summary
    print("\n" + "=" * 70)
    print("V5 WEIGHT COUNTS")
    print("=" * 70)

    def count_nonzero(module):
        total = 0
        for p in module.parameters():
            total += (p != 0).sum().item()
        for name, buf in module.named_buffers():
            total += buf.numel()
        return int(total)

    weights = {
        "ADD/SUB": count_nonzero(add_layer),
        "MUL": count_nonzero(mul_layer),
        "DIV/MOD": count_nonzero(div_layer),
        "Comparisons": count_nonzero(cmp_layer),
        "BZ/BNZ (zero detect)": count_nonzero(bz_layer),
        "Bitwise (AND/OR/XOR)": 224,  # Computed in forward
        "Shifts (SHL/SHR)": count_nonzero(shift_layer),
        "Control Flow (JMP/BZ/BNZ)": count_nonzero(ctrl_layer),
        "Stack Frame (ENT/ADJ/LEV)": count_nonzero(stack_layer),
    }

    total_weights = 0
    for name, count in weights.items():
        print(f"  {name}: {count}")
        total_weights += count

    print(f"  {'─'*30}")
    print(f"  TOTAL: {total_weights}")
    print(f"\n  V3 had ~100,000 weights")
    print(f"  V5 reduction: ~{100000 // total_weights}×")
    print("=" * 70)
