"""Tests for autoregressive unified memory execution.

Verifies that the neural embedding can infer executable regions from
the token stream without external runner hints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neural_vm.neural_embedding import NeuralVMEmbedding
from neural_vm.vm_step import Token, _SetDim


def test_code_region_executable():
    """MEM sections in code region (< 0x10000) should be marked executable."""
    torch.manual_seed(42)  # Deterministic for testing
    emb = NeuralVMEmbedding(272, 512)
    assert emb._autoregressive_exec, "Autoregressive mode should be enabled by default"

    # MEM at address 0x100 (in code region)
    tokens = torch.tensor([[
        Token.MEM,
        0x00, 0x01, 0x00, 0x00,  # addr = 0x100
        0x25, 0x00, 0x00, 0x00   # val = ADD opcode
    ]])

    x = emb(tokens)
    mem_exec = x[0, 0, _SetDim.MEM_EXEC].item()
    # Injection SETS to 1.0
    assert abs(mem_exec - 1.0) < 0.01, f"MEM_EXEC should be 1.0 for code region, got {mem_exec}"

    # Check ADDR_KEY injection on value bytes - specific bits should be 1.0
    # addr 0x100 = 256: lo=0, hi=0, top=1
    assert abs(x[0, 5, _SetDim.ADDR_KEY + 0].item() - 1.0) < 0.01, "ADDR_KEY lo=0 not set"
    assert abs(x[0, 5, _SetDim.ADDR_KEY + 16 + 0].item() - 1.0) < 0.01, "ADDR_KEY hi=0 not set"
    assert abs(x[0, 5, _SetDim.ADDR_KEY + 32 + 1].item() - 1.0) < 0.01, "ADDR_KEY top=1 not set"

    print("PASS: test_code_region_executable")


def test_data_region_not_executable():
    """MEM sections in data region (>= 0x10000) should NOT be marked executable."""
    torch.manual_seed(42)
    emb = NeuralVMEmbedding(272, 512)

    # MEM at address 0x10000 (data region)
    tokens_data = torch.tensor([[
        Token.MEM,
        0x00, 0x00, 0x01, 0x00,  # addr = 0x10000 (data region)
        0x42, 0x00, 0x00, 0x00
    ]])

    # MEM at address 0x100 (code region) for comparison
    tokens_code = torch.tensor([[
        Token.MEM,
        0x00, 0x01, 0x00, 0x00,  # addr = 0x100 (code region)
        0x42, 0x00, 0x00, 0x00
    ]])

    x_data = emb(tokens_data)
    x_code = emb(tokens_code)

    mem_exec_data = x_data[0, 0, _SetDim.MEM_EXEC].item()
    mem_exec_code = x_code[0, 0, _SetDim.MEM_EXEC].item()

    # Code region has MEM_EXEC SET to 1.0, data region has random init
    # Code region should be exactly 1.0
    assert abs(mem_exec_code - 1.0) < 0.01, f"Code region MEM_EXEC should be 1.0, got {mem_exec_code}"
    # Data region should NOT be 1.0 (random init, unlikely to be exactly 1.0)
    assert abs(mem_exec_data - 1.0) > 0.1, f"Data region MEM_EXEC should not be 1.0, got {mem_exec_data}"

    print("PASS: test_data_region_not_executable")


def test_jump_target_detection():
    """PC jumps should be detected and target addresses marked executable."""
    emb = NeuralVMEmbedding(272, 512)

    # Simulate two steps: PC goes from 2 to 0x20000 (non-sequential = jump)
    tokens = torch.tensor([[
        Token.REG_PC, 0x02, 0x00, 0x00, 0x00,  # PC = 2
        Token.STEP_END,
        Token.REG_PC, 0x00, 0x00, 0x02, 0x00,  # PC = 0x20000 (jump!)
        Token.STEP_END,
    ]])

    targets = emb._extract_jump_targets_autoregressive(tokens)
    assert 0x20000 in targets, f"0x20000 should be detected as jump target, got {targets}"

    print("PASS: test_jump_target_detection")


def test_jump_makes_data_executable():
    """Jump to data region should mark MEM at that address as executable."""
    torch.manual_seed(42)
    emb = NeuralVMEmbedding(272, 512)

    # PC jumps to 0x20000, and there's a MEM section at 0x20000
    tokens = torch.tensor([[
        Token.REG_PC, 0x02, 0x00, 0x00, 0x00,
        Token.STEP_END,
        Token.REG_PC, 0x00, 0x00, 0x02, 0x00,  # Jump to 0x20000
        Token.MEM, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00,  # MEM at 0x20000
        Token.STEP_END,
    ]])

    x = emb(tokens)
    # MEM marker is at position 11 (after REG_PC[5] + STEP_END + REG_PC[5])
    mem_exec = x[0, 11, _SetDim.MEM_EXEC].item()
    # Should be SET to exactly 1.0 due to jump target detection
    assert abs(mem_exec - 1.0) < 0.01, f"MEM_EXEC should be 1.0 for jump target, got {mem_exec}"

    print("PASS: test_jump_makes_data_executable")


def test_autoregressive_mode_toggle():
    """Autoregressive mode can be toggled off for external hints."""
    torch.manual_seed(42)
    emb = NeuralVMEmbedding(272, 512)

    # Disable autoregressive mode
    emb.set_autoregressive_exec(False)
    assert not emb._autoregressive_exec

    # MEM in code region should NOT be auto-marked anymore (random init only)
    tokens = torch.tensor([[
        Token.MEM, 0x00, 0x01, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00
    ]])
    x_before = emb(tokens)
    mem_exec_before = x_before[0, 0, _SetDim.MEM_EXEC].item()
    # Should NOT be exactly 1.0 (random init)
    assert abs(mem_exec_before - 1.0) > 0.1, \
        f"Without hints, MEM_EXEC should not be 1.0, got {mem_exec_before}"

    # Now use external hint
    emb.add_exec_addr(0x100)
    x_after = emb(tokens)
    mem_exec_after = x_after[0, 0, _SetDim.MEM_EXEC].item()

    # Should be SET to exactly 1.0
    assert abs(mem_exec_after - 1.0) < 0.01, \
        f"With external hint, MEM_EXEC should be 1.0, got {mem_exec_after}"

    print("PASS: test_autoregressive_mode_toggle")


def test_multiple_jumps():
    """Multiple jump targets should all be detected."""
    emb = NeuralVMEmbedding(272, 512)

    # PC sequence: 2 -> 100 -> 200 (two jumps)
    tokens = torch.tensor([[
        Token.REG_PC, 0x02, 0x00, 0x00, 0x00,  # PC = 2
        Token.STEP_END,
        Token.REG_PC, 0x64, 0x00, 0x00, 0x00,  # PC = 100 (jump)
        Token.STEP_END,
        Token.REG_PC, 0xC8, 0x00, 0x00, 0x00,  # PC = 200 (jump)
        Token.STEP_END,
    ]])

    targets = emb._extract_jump_targets_autoregressive(tokens)
    assert 100 in targets, f"100 should be detected as jump target, got {targets}"
    assert 200 in targets, f"200 should be detected as jump target, got {targets}"

    print("PASS: test_multiple_jumps")


def test_sequential_not_jump():
    """Sequential PC increments should NOT be detected as jumps."""
    emb = NeuralVMEmbedding(272, 512)

    # PC sequence: 2 -> 10 -> 18 (sequential, INSTR_WIDTH=8)
    tokens = torch.tensor([[
        Token.REG_PC, 0x02, 0x00, 0x00, 0x00,  # PC = 2
        Token.STEP_END,
        Token.REG_PC, 0x0A, 0x00, 0x00, 0x00,  # PC = 10 (sequential)
        Token.STEP_END,
        Token.REG_PC, 0x12, 0x00, 0x00, 0x00,  # PC = 18 (sequential)
        Token.STEP_END,
    ]])

    targets = emb._extract_jump_targets_autoregressive(tokens)
    assert len(targets) == 0, f"Sequential PCs should not create jump targets, got {targets}"

    print("PASS: test_sequential_not_jump")


def test_no_external_hints_needed():
    """Verify the system works without any add_exec_addr() calls."""
    torch.manual_seed(42)
    emb = NeuralVMEmbedding(272, 512)

    # Never call add_exec_addr() - autoregressive should handle it
    assert len(emb._exec_addrs) == 0, "Should start with no external exec addrs"

    # MEM in code region should still be executable
    tokens = torch.tensor([[
        Token.MEM, 0x00, 0x01, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00
    ]])

    x = emb(tokens)
    mem_exec = x[0, 0, _SetDim.MEM_EXEC].item()
    # Should be SET to 1.0 via autoregressive detection
    assert abs(mem_exec - 1.0) < 0.01, f"Should work without external hints, got {mem_exec}"

    print("PASS: test_no_external_hints_needed")


if __name__ == "__main__":
    test_code_region_executable()
    test_data_region_not_executable()
    test_jump_target_detection()
    test_jump_makes_data_executable()
    test_autoregressive_mode_toggle()
    test_multiple_jumps()
    test_sequential_not_jump()
    test_no_external_hints_needed()
    print("\nAll autoregressive exec tests passed!")
