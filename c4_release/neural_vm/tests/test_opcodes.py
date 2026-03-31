"""Comprehensive tests for the autoregressive neural VM (3000+ tests).

Tests all 256 byte values for IMM, step structure, register propagation,
marker ordering, override semantics, context encoding, and architecture.

All 256 IMM programs are run once and cached; test methods check cached results.
"""

import unittest
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode


def run_program(model, bytecode, max_steps=5):
    """Run a program and return (exit_code, generated_tokens, step_list).

    Each step is a list of tokens: [REG_PC, b0..b3, REG_AX, b0..b3, ..., SE/HALT]
    """
    from neural_vm.run_vm import AutoregressiveVMRunner
    runner = AutoregressiveVMRunner()
    runner.model = model
    context = runner._build_context(bytecode, b'', [])

    generated = []
    for _ in range(max_steps * Token.STEP_TOKENS + 10):
        tok = model.generate_next(context)
        context.append(tok)
        generated.append(tok)
        if tok == Token.HALT:
            break

    # Parse steps
    steps = []
    step_tokens = []
    for tok in generated:
        step_tokens.append(tok)
        if tok in (Token.STEP_END, Token.HALT):
            steps.append(step_tokens)
            step_tokens = []

    # Extract exit code
    exit_code = 0
    for i in range(len(context) - 1, -1, -1):
        if context[i] == Token.REG_AX and i + 4 < len(context):
            exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
            break

    return exit_code, generated, steps


def run_programs_batch(model, bytecodes_list, max_steps=5, batch_size=256):
    """Run multiple programs in parallel via batched forward passes.

    All bytecodes must have the same length (same number of instructions).
    Returns: list of (exit_code, generated_tokens, steps) tuples.
    """
    from neural_vm.run_vm import AutoregressiveVMRunner

    all_results = [None] * len(bytecodes_list)

    for chunk_start in range(0, len(bytecodes_list), batch_size):
        chunk = bytecodes_list[chunk_start:chunk_start + batch_size]

        # Build contexts (all same length since same bytecode length)
        runner = AutoregressiveVMRunner()
        runner.model = model
        contexts = [runner._build_context(bc, b'', []) for bc in chunk]

        generated = [[] for _ in chunk]
        halted = [False] * len(chunk)

        total_tokens = max_steps * Token.STEP_TOKENS + 10
        for _ in range(total_tokens):
            if all(halted):
                break

            next_tokens = model.generate_next_batch(contexts)

            for i in range(len(chunk)):
                if not halted[i]:
                    contexts[i].append(next_tokens[i])
                    generated[i].append(next_tokens[i])
                    if next_tokens[i] == Token.HALT:
                        halted[i] = True
                else:
                    # Halted programs: append pad token to keep lengths aligned
                    contexts[i].append(0)

        # Parse results
        for i in range(len(chunk)):
            steps = []
            step_tokens = []
            for tok in generated[i]:
                step_tokens.append(tok)
                if tok in (Token.STEP_END, Token.HALT):
                    steps.append(step_tokens)
                    step_tokens = []

            exit_code = 0
            for j in range(len(contexts[i]) - 1, -1, -1):
                if contexts[i][j] == Token.REG_AX and j + 4 < len(contexts[i]):
                    exit_code = sum(contexts[i][j + 1 + k] << (k * 8) for k in range(4))
                    break

            all_results[chunk_start + i] = (exit_code, generated[i], steps)

    return all_results


def extract_register(step, marker_token):
    """Extract 32-bit register value from a step's token list."""
    for i, tok in enumerate(step):
        if tok == marker_token and i + 4 < len(step):
            return sum(step[i + 1 + j] << (j * 8) for j in range(4))
    return None


def extract_register_bytes(step, marker_token):
    """Extract raw 4 bytes from a register section."""
    for i, tok in enumerate(step):
        if tok == marker_token and i + 4 < len(step):
            return [step[i + 1 + j] for j in range(4)]
    return None


# =========================================================================
# Shared fixtures - model and program result caches
# =========================================================================

_shared_model = None

def _get_model():
    global _shared_model
    if _shared_model is None:
        import os
        cache_path = os.path.join(os.path.dirname(__file__), '.compact_moe_model.pt')
        if os.path.exists(cache_path):
            _shared_model = AutoregressiveVM.load_compact(cache_path)
        else:
            _shared_model = AutoregressiveVM()
            set_vm_weights(_shared_model)
            _shared_model.compact(block_size=32)
            _shared_model.compact_moe()
            _shared_model.save_compact(cache_path)
    return _shared_model


_imm_cache = None
_override_cache = None
_triple_cache = None

_OVERRIDE_VALUES = [0, 1, 15, 16, 17, 31, 32, 64, 100, 127, 128, 191, 200, 240, 254, 255]
_TRIPLE_VALUES = [0, 42, 128, 255]


def _get_imm_cache():
    """Cache results for IMM v; EXIT for all v in 0..255."""
    global _imm_cache
    if _imm_cache is None:
        model = _get_model()
        bytecodes = [[Opcode.IMM | (v << 8), Opcode.EXIT] for v in range(256)]
        results = run_programs_batch(model, bytecodes)
        _imm_cache = {v: results[v] for v in range(256)}
    return _imm_cache


def _get_override_cache():
    """Cache results for IMM a; IMM b; EXIT for 16x16 pairs."""
    global _override_cache
    if _override_cache is None:
        model = _get_model()
        pairs = [(a, b) for a in _OVERRIDE_VALUES for b in _OVERRIDE_VALUES]
        bytecodes = [[Opcode.IMM | (a << 8), Opcode.IMM | (b << 8), Opcode.EXIT]
                     for a, b in pairs]
        results = run_programs_batch(model, bytecodes)
        _override_cache = {pairs[i]: results[i] for i in range(len(pairs))}
    return _override_cache


def _get_triple_cache():
    """Cache results for IMM a; IMM b; IMM c; EXIT for 4x4x4 triples."""
    global _triple_cache
    if _triple_cache is None:
        model = _get_model()
        triples = [(a, b, c) for a in _TRIPLE_VALUES
                   for b in _TRIPLE_VALUES for c in _TRIPLE_VALUES]
        bytecodes = [[Opcode.IMM | (a << 8), Opcode.IMM | (b << 8),
                      Opcode.IMM | (c << 8), Opcode.EXIT]
                     for a, b, c in triples]
        results = run_programs_batch(model, bytecodes, max_steps=6)
        _triple_cache = {triples[i]: results[i] for i in range(len(triples))}
    return _triple_cache


# =========================================================================
# 1. All 256 IMM exit codes (256 tests)
# =========================================================================

class TestIMMExitCodes(unittest.TestCase):
    """IMM v; EXIT -> exit_code=v for all 256 byte values."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_exit_code_test(val):
    def test(self):
        ec, _, _ = self.cache[val]
        self.assertEqual(ec, val)
    test.__doc__ = f"IMM {val}; EXIT -> exit_code={val}"
    return test

for _v in range(256):
    setattr(TestIMMExitCodes, f'test_imm_{_v:03d}_exit', _make_exit_code_test(_v))


# =========================================================================
# 2. All 256 IMM AX values in step 0 (256 tests)
# =========================================================================

class TestIMMAXValues(unittest.TestCase):
    """AX register holds immediate value after IMM for all 256 values."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_ax_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        self.assertGreater(len(steps), 0)
        ax = extract_register(steps[0], Token.REG_AX)
        self.assertEqual(ax, val)
    test.__doc__ = f"IMM {val}: AX={val} in step 0"
    return test

for _v in range(256):
    setattr(TestIMMAXValues, f'test_ax_{_v:03d}', _make_ax_test(_v))


# =========================================================================
# 3. All 256 IMM AX preserved across EXIT (256 tests)
# =========================================================================

class TestIMMAXPreserved(unittest.TestCase):
    """AX preserved from IMM step to EXIT step for all 256 values."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_ax_preserved_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        self.assertGreaterEqual(len(steps), 2)
        ax0 = extract_register(steps[0], Token.REG_AX)
        ax1 = extract_register(steps[1], Token.REG_AX)
        self.assertEqual(ax0, ax1)
    test.__doc__ = f"IMM {val}: AX preserved across EXIT"
    return test

for _v in range(256):
    setattr(TestIMMAXPreserved, f'test_ax_preserved_{_v:03d}', _make_ax_preserved_test(_v))


# =========================================================================
# 4. All 256 step token counts (256 tests)
# =========================================================================

class TestIMMStepTokens(unittest.TestCase):
    """Each step has exactly 35 tokens for all 256 IMM values."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_step_tokens_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        self.assertGreater(len(steps), 0, "No steps generated")
        for i, step in enumerate(steps):
            self.assertEqual(len(step), Token.STEP_TOKENS,
                             f"Step {i}: {len(step)} tokens, expected {Token.STEP_TOKENS}")
    test.__doc__ = f"IMM {val}: all steps have 35 tokens"
    return test

for _v in range(256):
    setattr(TestIMMStepTokens, f'test_step_tokens_{_v:03d}', _make_step_tokens_test(_v))


# =========================================================================
# 5. All 256 HALT emitted (256 tests)
# =========================================================================

class TestIMMHaltEmitted(unittest.TestCase):
    """Last generated token is HALT for all 256 IMM values."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_halt_test(val):
    def test(self):
        _, gen, _ = self.cache[val]
        self.assertEqual(gen[-1], Token.HALT)
    test.__doc__ = f"IMM {val}: HALT at end"
    return test

for _v in range(256):
    setattr(TestIMMHaltEmitted, f'test_halt_{_v:03d}', _make_halt_test(_v))


# =========================================================================
# 6. All 256 marker order (256 tests)
# =========================================================================

class TestIMMMarkerOrder(unittest.TestCase):
    """Markers in correct order for all 256 IMM values."""
    EXPECTED = [Token.REG_PC, Token.REG_AX, Token.REG_SP,
                Token.REG_BP, Token.STACK0, Token.MEM]

    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_marker_order_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        for step_idx, step in enumerate(steps):
            markers = [t for t in step if t > 255 and t not in (Token.STEP_END, Token.HALT)]
            self.assertEqual(markers, self.EXPECTED,
                             f"Step {step_idx}: wrong markers {markers}")
    test.__doc__ = f"IMM {val}: correct marker order"
    return test

for _v in range(256):
    setattr(TestIMMMarkerOrder, f'test_marker_order_{_v:03d}', _make_marker_order_test(_v))


# =========================================================================
# 7. All 256 byte range (256 tests)
# =========================================================================

class TestIMMByteRange(unittest.TestCase):
    """All non-marker tokens are byte values for all 256 IMM values."""
    MARKERS = {Token.REG_PC, Token.REG_AX, Token.REG_SP, Token.REG_BP,
               Token.STACK0, Token.MEM, Token.STEP_END, Token.HALT}

    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_byte_range_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        for step_idx, step in enumerate(steps):
            for i, tok in enumerate(step):
                if tok not in self.MARKERS:
                    self.assertTrue(0 <= tok <= 255,
                                    f"Step {step_idx}[{i}]: token {tok} not a byte")
    test.__doc__ = f"IMM {val}: all data tokens in byte range"
    return test

for _v in range(256):
    setattr(TestIMMByteRange, f'test_byte_range_{_v:03d}', _make_byte_range_test(_v))


# =========================================================================
# 8. All 256 STEP_END + HALT positions (256 tests)
# =========================================================================

class TestStepEndHaltPositions(unittest.TestCase):
    """Intermediate steps end with STEP_END, final step ends with HALT."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_se_halt_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        self.assertGreaterEqual(len(steps), 2)
        self.assertEqual(steps[0][-1], Token.STEP_END)
        self.assertEqual(steps[-1][-1], Token.HALT)
    test.__doc__ = f"IMM {val}: STEP_END then HALT"
    return test

for _v in range(256):
    setattr(TestStepEndHaltPositions, f'test_se_halt_{_v:03d}', _make_se_halt_test(_v))


# =========================================================================
# 9. All 256 marker spacing (256 tests)
# =========================================================================

class TestMarkerSpacing(unittest.TestCase):
    """Verify marker positions: PC@0, AX@5, SP@10, BP@15, STACK0@20, MEM@25."""
    EXPECTED_POS = {'PC': 0, 'AX': 5, 'SP': 10, 'BP': 15, 'STACK0': 20, 'MEM': 25}

    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_spacing_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        marker_map = {
            Token.REG_PC: 'PC', Token.REG_AX: 'AX', Token.REG_SP: 'SP',
            Token.REG_BP: 'BP', Token.STACK0: 'STACK0', Token.MEM: 'MEM',
        }
        for step_idx, step in enumerate(steps):
            positions = {}
            for i, tok in enumerate(step):
                if tok in marker_map:
                    positions[marker_map[tok]] = i
            for name, expected_pos in self.EXPECTED_POS.items():
                self.assertEqual(positions.get(name), expected_pos,
                                 f"Step {step_idx}: {name} at {positions.get(name)}, expected {expected_pos}")
    test.__doc__ = f"IMM {val}: correct marker spacing"
    return test

for _v in range(256):
    setattr(TestMarkerSpacing, f'test_spacing_{_v:03d}', _make_spacing_test(_v))


# =========================================================================
# 10. All 256 AX byte decomposition (256 tests)
# =========================================================================

class TestAXByteDecomposition(unittest.TestCase):
    """Verify AX register bytes encode value correctly (little-endian)."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_ax_bytes_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        bytes_ = extract_register_bytes(steps[0], Token.REG_AX)
        self.assertIsNotNone(bytes_)
        self.assertEqual(bytes_[0], val, f"AX byte 0 should be {val}")
        self.assertEqual(bytes_[1], 0, "AX byte 1 should be 0")
        self.assertEqual(bytes_[2], 0, "AX byte 2 should be 0")
        self.assertEqual(bytes_[3], 0, "AX byte 3 should be 0")
    test.__doc__ = f"IMM {val}: AX bytes = [{val}, 0, 0, 0]"
    return test

for _v in range(256):
    setattr(TestAXByteDecomposition, f'test_ax_bytes_{_v:03d}', _make_ax_bytes_test(_v))


# =========================================================================
# 11. All 256 context encodings (256 tests)
# =========================================================================

class TestContextEncoding(unittest.TestCase):
    """Verify _build_context produces correct token sequences."""
    @classmethod
    def setUpClass(cls):
        from neural_vm.run_vm import AutoregressiveVMRunner
        cls.runner = AutoregressiveVMRunner()

def _make_context_test(val):
    def test(self):
        ctx = self.runner._build_context([Opcode.IMM | (val << 8), Opcode.EXIT], b'', [])
        self.assertEqual(ctx[0], Token.CODE_START)
        self.assertEqual(ctx[1], Opcode.IMM)
        self.assertEqual(ctx[2], val)       # imm byte 0
        self.assertEqual(ctx[3], 0)         # imm byte 1
        self.assertEqual(ctx[4], 0)         # imm byte 2
        self.assertEqual(ctx[5], 0)         # imm byte 3
        self.assertEqual(ctx[6], Opcode.EXIT)
        self.assertEqual(ctx[-1], Token.DATA_END)
    test.__doc__ = f"Context encoding for IMM {val}; EXIT"
    return test

for _v in range(256):
    setattr(TestContextEncoding, f'test_ctx_{_v:03d}', _make_context_test(_v))


# =========================================================================
# 12. Override pairs: exit codes (16x16 = 256 tests)
# =========================================================================

class TestIMMOverridePairs(unittest.TestCase):
    """IMM a; IMM b; EXIT -> exit_code=b for 16x16 pairs."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_override_cache()

def _make_override_test(a, b):
    def test(self):
        ec, _, _ = self.cache[(a, b)]
        self.assertEqual(ec, b, f"IMM {a}; IMM {b}; EXIT: expected {b}, got {ec}")
    test.__doc__ = f"IMM {a}; IMM {b}; EXIT -> {b}"
    return test

for _a in _OVERRIDE_VALUES:
    for _b in _OVERRIDE_VALUES:
        setattr(TestIMMOverridePairs, f'test_override_{_a:03d}_{_b:03d}',
                _make_override_test(_a, _b))


# =========================================================================
# 13. Override pairs: AX values per step (16x16 = 256 tests)
# =========================================================================

class TestOverrideAXValues(unittest.TestCase):
    """AX should hold correct value after each IMM in override sequence."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_override_cache()

def _make_override_ax_test(a, b):
    def test(self):
        _, _, steps = self.cache[(a, b)]
        ax0 = extract_register(steps[0], Token.REG_AX)
        self.assertEqual(ax0, a, f"Step 0: AX={ax0}, expected {a}")
        ax1 = extract_register(steps[1], Token.REG_AX)
        self.assertEqual(ax1, b, f"Step 1: AX={ax1}, expected {b}")
    test.__doc__ = f"Override AX: IMM {a} -> {a}, then IMM {b} -> {b}"
    return test

for _a in _OVERRIDE_VALUES:
    for _b in _OVERRIDE_VALUES:
        setattr(TestOverrideAXValues, f'test_override_ax_{_a:03d}_{_b:03d}',
                _make_override_ax_test(_a, _b))


# =========================================================================
# 14. Triple override: exit codes (4x4x4 = 64 tests)
# =========================================================================

class TestIMMTripleOverride(unittest.TestCase):
    """IMM a; IMM b; IMM c; EXIT -> exit_code=c for 4x4x4 triples."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_triple_cache()

def _make_triple_test(a, b, c):
    def test(self):
        ec, _, _ = self.cache[(a, b, c)]
        self.assertEqual(ec, c, f"IMM {a}; IMM {b}; IMM {c}; EXIT: expected {c}, got {ec}")
    test.__doc__ = f"IMM {a}; IMM {b}; IMM {c}; EXIT -> {c}"
    return test

for _a in _TRIPLE_VALUES:
    for _b in _TRIPLE_VALUES:
        for _c in _TRIPLE_VALUES:
            setattr(TestIMMTripleOverride, f'test_triple_{_a:03d}_{_b:03d}_{_c:03d}',
                    _make_triple_test(_a, _b, _c))


# =========================================================================
# 15. Override step structure (16x16 = 256 tests)
# =========================================================================

class TestOverrideStepStructure(unittest.TestCase):
    """Step structure correct for all override pairs."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_override_cache()

def _make_override_struct_test(a, b):
    def test(self):
        _, _, steps = self.cache[(a, b)]
        self.assertEqual(len(steps), 3, f"Expected 3 steps, got {len(steps)}")
        for i, step in enumerate(steps):
            self.assertEqual(len(step), Token.STEP_TOKENS,
                             f"Step {i}: {len(step)} tokens")
        self.assertEqual(steps[0][-1], Token.STEP_END)
        self.assertEqual(steps[1][-1], Token.STEP_END)
        self.assertEqual(steps[2][-1], Token.HALT)
    test.__doc__ = f"Override ({a},{b}): 3 steps, correct terminators"
    return test

for _a in _OVERRIDE_VALUES:
    for _b in _OVERRIDE_VALUES:
        setattr(TestOverrideStepStructure, f'test_ovr_struct_{_a:03d}_{_b:03d}',
                _make_override_struct_test(_a, _b))


# =========================================================================
# 16. PC progression for all IMM values (256 tests)
# =========================================================================

class TestPCAllValues(unittest.TestCase):
    """PC should be 0 in step 0, 5 in step 1, for all IMM values."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_pc_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        pc0 = extract_register(steps[0], Token.REG_PC)
        self.assertEqual(pc0, 0, f"Step 0 PC={pc0}, expected 0")
        pc1 = extract_register(steps[1], Token.REG_PC)
        self.assertEqual(pc1, 5, f"Step 1 PC={pc1}, expected 5")
    test.__doc__ = f"IMM {val}: PC=0 then PC=5"
    return test

for _v in range(256):
    setattr(TestPCAllValues, f'test_pc_{_v:03d}', _make_pc_test(_v))


# =========================================================================
# 17. SP default zero for all values (256 tests)
# =========================================================================

class TestSPDefault(unittest.TestCase):
    """SP should be 0 in all steps for IMM+EXIT programs."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_sp_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        for i, step in enumerate(steps):
            sp = extract_register(step, Token.REG_SP)
            self.assertEqual(sp, 0, f"Step {i}: SP={sp}, expected 0")
    test.__doc__ = f"IMM {val}: SP=0 in all steps"
    return test

for _v in range(256):
    setattr(TestSPDefault, f'test_sp_{_v:03d}', _make_sp_test(_v))


# =========================================================================
# 18. BP default zero for all values (256 tests)
# =========================================================================

class TestBPDefault(unittest.TestCase):
    """BP should be 0 in all steps for IMM+EXIT programs."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_bp_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        for i, step in enumerate(steps):
            bp = extract_register(step, Token.REG_BP)
            self.assertEqual(bp, 0, f"Step {i}: BP={bp}, expected 0")
    test.__doc__ = f"IMM {val}: BP=0 in all steps"
    return test

for _v in range(256):
    setattr(TestBPDefault, f'test_bp_{_v:03d}', _make_bp_test(_v))


# =========================================================================
# 19. Two steps generated for all values (256 tests)
# =========================================================================

class TestTwoSteps(unittest.TestCase):
    """IMM+EXIT should produce exactly 2 steps for all values."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

def _make_two_steps_test(val):
    def test(self):
        _, _, steps = self.cache[val]
        self.assertEqual(len(steps), 2)
    test.__doc__ = f"IMM {val}: exactly 2 steps"
    return test

for _v in range(256):
    setattr(TestTwoSteps, f'test_two_steps_{_v:03d}', _make_two_steps_test(_v))


# =========================================================================
# 20. Runner end-to-end for all values (256 tests)
# =========================================================================

class TestRunnerE2E(unittest.TestCase):
    """Test AutoregressiveVMRunner end-to-end for all 256 values."""
    @classmethod
    def setUpClass(cls):
        cls.model = _get_model()
        from neural_vm.run_vm import AutoregressiveVMRunner
        cls.runner_cls = AutoregressiveVMRunner

    def _run_e2e(self, bytecode, max_steps=5):
        runner = self.runner_cls()
        runner.model = self.model
        return runner.run(bytecode, max_steps=max_steps)

def _make_runner_test(val):
    def test(self):
        _, ec = self._run_e2e([Opcode.IMM | (val << 8), Opcode.EXIT])
        self.assertEqual(ec, val)
    test.__doc__ = f"Runner: IMM {val}; EXIT -> exit_code={val}"
    return test

for _v in range(256):
    setattr(TestRunnerE2E, f'test_runner_imm_{_v:03d}', _make_runner_test(_v))


# =========================================================================
# 21. Override PC progression (16x16 = 256 tests)
# =========================================================================

class TestOverridePCProgression(unittest.TestCase):
    """PC should be 2, 7, 12 across 3-instruction programs."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_override_cache()

def _make_override_pc_test(a, b):
    def test(self):
        _, _, steps = self.cache[(a, b)]
        self.assertEqual(extract_register(steps[0], Token.REG_PC), 2)
        self.assertEqual(extract_register(steps[1], Token.REG_PC), 7)
        self.assertEqual(extract_register(steps[2], Token.REG_PC), 12)
    test.__doc__ = f"Override ({a},{b}): PC=2,7,12"
    return test

for _a in _OVERRIDE_VALUES:
    for _b in _OVERRIDE_VALUES:
        setattr(TestOverridePCProgression, f'test_ovr_pc_{_a:03d}_{_b:03d}',
                _make_override_pc_test(_a, _b))


# =========================================================================
# Model architecture tests
# =========================================================================

class TestModelArchitecture(unittest.TestCase):
    """Test model configuration and basic properties."""
    @classmethod
    def setUpClass(cls):
        cls.model = _get_model()

    def test_d_model(self):
        self.assertEqual(self.model.d_model, 512)

    def test_n_layers(self):
        self.assertEqual(len(self.model.blocks), 16)

    def test_vocab_size(self):
        self.assertEqual(self.model.vocab_size, Token.VOCAB_SIZE)

    def test_vocab_size_is_274(self):
        """Vocab size is 274: 256 bytes + 18 special tokens (includes THINKING_START/END)."""
        self.assertEqual(Token.VOCAB_SIZE, 274)

    def test_step_tokens_is_35(self):
        self.assertEqual(Token.STEP_TOKENS, 35)

    def test_marker_token_values(self):
        self.assertEqual(Token.REG_PC, 257)
        self.assertEqual(Token.REG_AX, 258)
        self.assertEqual(Token.REG_SP, 259)
        self.assertEqual(Token.REG_BP, 260)
        self.assertEqual(Token.MEM, 261)
        self.assertEqual(Token.STEP_END, 262)
        self.assertEqual(Token.HALT, 263)
        self.assertEqual(Token.CODE_START, 264)
        self.assertEqual(Token.CODE_END, 265)
        self.assertEqual(Token.DATA_START, 266)
        self.assertEqual(Token.DATA_END, 267)
        self.assertEqual(Token.STACK0, 268)

    def test_forward_pass_shape(self):
        ctx = [Token.CODE_START, Opcode.IMM, 42, 0, 0, 0,
               Opcode.EXIT, 0, 0, 0, 0,
               Token.CODE_END, Token.DATA_START, Token.DATA_END]
        token_ids = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(token_ids)
        self.assertEqual(logits.shape, (1, len(ctx), Token.VOCAB_SIZE))

    def test_forward_pass_deterministic(self):
        ctx = [Token.CODE_START, Opcode.IMM, 42, 0, 0, 0,
               Opcode.EXIT, 0, 0, 0, 0,
               Token.CODE_END, Token.DATA_START, Token.DATA_END]
        token_ids = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            logits1 = self.model(token_ids).clone()
            logits2 = self.model(token_ids).clone()
        self.assertTrue(torch.equal(logits1, logits2))

    def test_generate_next_returns_int(self):
        ctx = [Token.CODE_START, Opcode.IMM, 42, 0, 0, 0,
               Opcode.EXIT, 0, 0, 0, 0,
               Token.CODE_END, Token.DATA_START, Token.DATA_END]
        tok = self.model.generate_next(ctx)
        self.assertIsInstance(tok, int)
        self.assertTrue(0 <= tok < Token.VOCAB_SIZE)

    def test_generate_next_deterministic(self):
        ctx = [Token.CODE_START, Opcode.IMM, 42, 0, 0, 0,
               Opcode.EXIT, 0, 0, 0, 0,
               Token.CODE_END, Token.DATA_START, Token.DATA_END]
        tok1 = self.model.generate_next(ctx)
        tok2 = self.model.generate_next(ctx)
        self.assertEqual(tok1, tok2)

    def test_first_token_is_reg_pc(self):
        ctx = [Token.CODE_START, Opcode.IMM, 42, 0, 0, 0,
               Opcode.EXIT, 0, 0, 0, 0,
               Token.CODE_END, Token.DATA_START, Token.DATA_END]
        tok = self.model.generate_next(ctx)
        self.assertEqual(tok, Token.REG_PC)


# =========================================================================
# Context building edge cases
# =========================================================================

class TestContextEdgeCases(unittest.TestCase):
    """Test context building with various data and argv combinations."""
    @classmethod
    def setUpClass(cls):
        from neural_vm.run_vm import AutoregressiveVMRunner
        cls.runner = AutoregressiveVMRunner()

    def _ctx(self, bytecode, data=b'', argv=None):
        return self.runner._build_context(bytecode, data, argv or [])

    def test_empty_data(self):
        ctx = self._ctx([Opcode.IMM | (0 << 8), Opcode.EXIT])
        ds = ctx.index(Token.DATA_START)
        de = ctx.index(Token.DATA_END)
        self.assertEqual(de, ds + 1)

    def test_data_bytes(self):
        ctx = self._ctx([Opcode.IMM | (0 << 8), Opcode.EXIT], data=b'\x48\x65\x6c')
        ds = ctx.index(Token.DATA_START)
        de = ctx.index(Token.DATA_END)
        self.assertEqual(ctx[ds+1:de], [0x48, 0x65, 0x6c])

    def test_argv_encoding(self):
        ctx = self._ctx([Opcode.IMM | (0 << 8), Opcode.EXIT], argv=['hi'])
        de = ctx.index(Token.DATA_END)
        self.assertEqual(ctx[de+1], ord('h'))
        self.assertEqual(ctx[de+2], ord('i'))
        self.assertEqual(ctx[de+3], 0)

    def test_multiple_argv(self):
        ctx = self._ctx([Opcode.IMM | (0 << 8), Opcode.EXIT], argv=['a', 'b'])
        de = ctx.index(Token.DATA_END)
        self.assertEqual(ctx[de+1], ord('a'))
        self.assertEqual(ctx[de+2], 0)
        self.assertEqual(ctx[de+3], ord('b'))
        self.assertEqual(ctx[de+4], 0)

    def test_multi_instruction_encoding(self):
        ctx = self._ctx([
            Opcode.IMM | (10 << 8),
            Opcode.IMM | (20 << 8),
            Opcode.EXIT,
        ])
        self.assertEqual(ctx[0], Token.CODE_START)
        self.assertEqual(ctx[1], Opcode.IMM)
        self.assertEqual(ctx[2], 10)
        self.assertEqual(ctx[6], Opcode.IMM)
        self.assertEqual(ctx[7], 20)
        self.assertEqual(ctx[11], Opcode.EXIT)

    def test_large_immediate_encoding(self):
        imm = 0x12345678
        ctx = self._ctx([Opcode.IMM | (imm << 8), Opcode.EXIT])
        self.assertEqual(ctx[1], Opcode.IMM)
        self.assertEqual(ctx[2], 0x78)  # little-endian byte 0
        self.assertEqual(ctx[3], 0x56)
        self.assertEqual(ctx[4], 0x34)
        self.assertEqual(ctx[5], 0x12)

    def test_code_end_present(self):
        ctx = self._ctx([Opcode.EXIT])
        self.assertIn(Token.CODE_END, ctx)

    def test_data_start_after_code_end(self):
        ctx = self._ctx([Opcode.EXIT])
        ce = ctx.index(Token.CODE_END)
        ds = ctx.index(Token.DATA_START)
        self.assertEqual(ds, ce + 1)

    def test_context_starts_with_code_start(self):
        ctx = self._ctx([Opcode.EXIT])
        self.assertEqual(ctx[0], Token.CODE_START)

    def test_context_ends_with_data_end(self):
        ctx = self._ctx([Opcode.EXIT])
        self.assertEqual(ctx[-1], Token.DATA_END)


# =========================================================================
# _SetDim allocation tests
# =========================================================================

class TestSetDimAllocations(unittest.TestCase):
    """Verify _SetDim constants are self-consistent."""

    def test_no_overlap_mark_flags(self):
        mark_dims = [_SetDim.MARK_PC, _SetDim.MARK_AX, _SetDim.MARK_SP,
                     _SetDim.MARK_BP, _SetDim.MARK_MEM, _SetDim.MARK_SE,
                     _SetDim.MARK_STACK0, _SetDim.MARK_CS, _SetDim.MARK_SE_ONLY]
        self.assertEqual(len(mark_dims), len(set(mark_dims)))

    def test_no_overlap_next_flags(self):
        next_dims = [_SetDim.NEXT_PC, _SetDim.NEXT_AX, _SetDim.NEXT_SP,
                     _SetDim.NEXT_BP, _SetDim.NEXT_STACK0, _SetDim.NEXT_MEM,
                     _SetDim.NEXT_SE, _SetDim.NEXT_HALT]
        self.assertEqual(len(next_dims), len(set(next_dims)))

    def test_embed_output_no_overlap(self):
        embed_lo = set(range(_SetDim.EMBED_LO, _SetDim.EMBED_LO + 16))
        embed_hi = set(range(_SetDim.EMBED_HI, _SetDim.EMBED_HI + 16))
        out_lo = set(range(_SetDim.OUTPUT_LO, _SetDim.OUTPUT_LO + 16))
        out_hi = set(range(_SetDim.OUTPUT_HI, _SetDim.OUTPUT_HI + 16))
        all_ranges = [embed_lo, embed_hi, out_lo, out_hi]
        for i in range(len(all_ranges)):
            for j in range(i+1, len(all_ranges)):
                self.assertEqual(len(all_ranges[i] & all_ranges[j]), 0)

    def test_all_dims_within_d_model(self):
        for name in dir(_SetDim):
            if name.startswith('_'):
                continue
            val = getattr(_SetDim, name)
            if isinstance(val, int):
                self.assertLess(val, 512, f"{name}={val} >= 512")

    def test_head_dims_spaced_by_7(self):
        heads = [_SetDim.H0, _SetDim.H1, _SetDim.H2, _SetDim.H3,
                 _SetDim.H4, _SetDim.H5, _SetDim.H6, _SetDim.H7]
        for i in range(len(heads) - 1):
            self.assertEqual(heads[i+1] - heads[i], 7,
                             f"H{i}->H{i+1}: gap={heads[i+1]-heads[i]}")

    def test_byte_index_dims_contiguous(self):
        bi = [_SetDim.BYTE_INDEX_0, _SetDim.BYTE_INDEX_1,
              _SetDim.BYTE_INDEX_2, _SetDim.BYTE_INDEX_3]
        for i in range(len(bi) - 1):
            self.assertEqual(bi[i+1] - bi[i], 1)


# =========================================================================
# Token constants tests
# =========================================================================

class TestTokenConstants(unittest.TestCase):
    """Verify Token constant values and relationships."""

    def test_byte_range_no_conflict(self):
        specials = [Token.SEP, Token.REG_PC, Token.REG_AX, Token.REG_SP,
                    Token.REG_BP, Token.MEM, Token.STEP_END, Token.HALT,
                    Token.CODE_START, Token.CODE_END, Token.DATA_START,
                    Token.DATA_END, Token.STACK0]
        for s in specials:
            self.assertGreater(s, 255)

    def test_special_tokens_unique(self):
        specials = [Token.SEP, Token.REG_PC, Token.REG_AX, Token.REG_SP,
                    Token.REG_BP, Token.MEM, Token.STEP_END, Token.HALT,
                    Token.CODE_START, Token.CODE_END, Token.DATA_START,
                    Token.DATA_END, Token.STACK0]
        self.assertEqual(len(specials), len(set(specials)))

    def test_vocab_covers_all(self):
        max_special = max(Token.SEP, Token.REG_PC, Token.REG_AX, Token.REG_SP,
                          Token.REG_BP, Token.MEM, Token.STEP_END, Token.HALT,
                          Token.CODE_START, Token.CODE_END, Token.DATA_START,
                          Token.DATA_END, Token.STACK0,
                          Token.USER_INPUT_START, Token.USER_INPUT_END,
                          Token.TOOL_CALL)
        self.assertEqual(Token.VOCAB_SIZE, max_special + 1)

    def test_step_tokens_formula(self):
        self.assertEqual(5 + 5 + 5 + 5 + 5 + 9 + 1, Token.STEP_TOKENS)


# =========================================================================
# 22. Multi-instruction programs (HALT bug regression)
# =========================================================================

_multi_cache = None
_MULTI_PROGRAMS = None

def _get_multi_programs():
    global _MULTI_PROGRAMS
    if _MULTI_PROGRAMS is None:
        _MULTI_PROGRAMS = []
        # 2-instruction: IMM v; EXIT
        for v in [0, 1, 42, 127, 128, 255]:
            _MULTI_PROGRAMS.append((
                f"IMM {v}; EXIT",
                [Opcode.IMM | (v << 8), Opcode.EXIT],
                v, 2,
            ))
        # 3-instruction: IMM a; IMM b; EXIT
        for a, b in [(5, 3), (0, 255), (128, 42), (10, 20)]:
            _MULTI_PROGRAMS.append((
                f"IMM {a}; IMM {b}; EXIT",
                [Opcode.IMM | (a << 8), Opcode.IMM | (b << 8), Opcode.EXIT],
                b, 3,
            ))
        # 4-instruction: IMM a; IMM b; IMM c; EXIT
        for a, b, c in [(10, 20, 30), (1, 2, 3), (255, 128, 42)]:
            _MULTI_PROGRAMS.append((
                f"IMM {a}; IMM {b}; IMM {c}; EXIT",
                [Opcode.IMM | (a << 8), Opcode.IMM | (b << 8),
                 Opcode.IMM | (c << 8), Opcode.EXIT],
                c, 4,
            ))
    return _MULTI_PROGRAMS

def _get_multi_cache():
    global _multi_cache
    if _multi_cache is None:
        model = _get_model()
        _multi_cache = {}
        # Group by bytecode length for batching
        from collections import defaultdict
        groups = defaultdict(list)
        for name, bytecode, expected_ec, expected_steps in _get_multi_programs():
            groups[len(bytecode)].append((name, bytecode))
        for bc_len, items in groups.items():
            keys = [name for name, bc in items]
            bytecodes = [bc for name, bc in items]
            results = run_programs_batch(model, bytecodes, max_steps=8)
            for i, k in enumerate(keys):
                _multi_cache[k] = results[i]
    return _multi_cache


class TestMultiInstructionExitCode(unittest.TestCase):
    """Multi-instruction programs produce correct exit codes (HALT bug regression)."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_multi_cache()
        cls.programs = _get_multi_programs()

def _make_multi_ec_test(name, expected_ec):
    def test(self):
        ec, _, _ = self.cache[name]
        self.assertEqual(ec, expected_ec, f"{name}: expected exit_code={expected_ec}, got {ec}")
    test.__doc__ = f"{name} -> exit_code={expected_ec}"
    return test

for _name, _bc, _ec, _steps in _get_multi_programs():
    safe_name = _name.replace(';', '_').replace(' ', '').replace(',', '_')
    setattr(TestMultiInstructionExitCode, f'test_ec_{safe_name}',
            _make_multi_ec_test(_name, _ec))


class TestMultiInstructionStepCount(unittest.TestCase):
    """Multi-instruction programs produce correct number of steps."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_multi_cache()
        cls.programs = _get_multi_programs()

def _make_multi_steps_test(name, expected_steps):
    def test(self):
        _, _, steps = self.cache[name]
        self.assertEqual(len(steps), expected_steps,
                         f"{name}: expected {expected_steps} steps, got {len(steps)}")
    test.__doc__ = f"{name} -> {expected_steps} steps"
    return test

for _name, _bc, _ec, _steps in _get_multi_programs():
    safe_name = _name.replace(';', '_').replace(' ', '').replace(',', '_')
    setattr(TestMultiInstructionStepCount, f'test_steps_{safe_name}',
            _make_multi_steps_test(_name, _steps))


class TestMultiInstructionHaltPosition(unittest.TestCase):
    """HALT only at last step, STEP_END at intermediate steps."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_multi_cache()
        cls.programs = _get_multi_programs()

def _make_multi_halt_test(name, expected_steps):
    def test(self):
        _, _, steps = self.cache[name]
        self.assertEqual(len(steps), expected_steps, f"Wrong step count for {name}")
        for i in range(expected_steps - 1):
            self.assertEqual(steps[i][-1], Token.STEP_END,
                             f"{name} step {i}: expected STEP_END, got {steps[i][-1]}")
        self.assertEqual(steps[-1][-1], Token.HALT,
                         f"{name} last step: expected HALT, got {steps[-1][-1]}")
    test.__doc__ = f"{name}: SE then HALT"
    return test

for _name, _bc, _ec, _steps in _get_multi_programs():
    safe_name = _name.replace(';', '_').replace(' ', '').replace(',', '_')
    setattr(TestMultiInstructionHaltPosition, f'test_halt_{safe_name}',
            _make_multi_halt_test(_name, _steps))


# =========================================================================
# 23. NOP tests
# =========================================================================

_nop_cache = None

def _get_nop_cache():
    global _nop_cache
    if _nop_cache is None:
        model = _get_model()
        # All 3-instruction programs → same context length
        keys = []
        bytecodes = []
        for v in [0, 1, 42, 99, 128, 255]:
            keys.append(('nop_imm', v))
            bytecodes.append([Opcode.NOP, Opcode.IMM | (v << 8), Opcode.EXIT])
        for v in [0, 42, 255]:
            keys.append(('imm_nop', v))
            bytecodes.append([Opcode.IMM | (v << 8), Opcode.NOP, Opcode.EXIT])
        results = run_programs_batch(model, bytecodes)
        _nop_cache = {keys[i]: results[i] for i in range(len(keys))}
    return _nop_cache


class TestNOPExitCodes(unittest.TestCase):
    """NOP doesn't affect exit code."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_nop_cache()

    def test_nop_imm_42_exit(self):
        ec, _, _ = self.cache[('nop_imm', 42)]
        self.assertEqual(ec, 42)

    def test_nop_imm_0_exit(self):
        ec, _, _ = self.cache[('nop_imm', 0)]
        self.assertEqual(ec, 0)

    def test_nop_imm_255_exit(self):
        ec, _, _ = self.cache[('nop_imm', 255)]
        self.assertEqual(ec, 255)

    def test_nop_imm_99_exit(self):
        ec, _, _ = self.cache[('nop_imm', 99)]
        self.assertEqual(ec, 99)

    def test_nop_imm_128_exit(self):
        ec, _, _ = self.cache[('nop_imm', 128)]
        self.assertEqual(ec, 128)

    def test_nop_imm_1_exit(self):
        ec, _, _ = self.cache[('nop_imm', 1)]
        self.assertEqual(ec, 1)

    def test_imm_nop_exit_42(self):
        """IMM 42; NOP; EXIT -> exit_code=42 (NOP preserves AX)."""
        ec, _, _ = self.cache[('imm_nop', 42)]
        self.assertEqual(ec, 42)

    def test_imm_nop_exit_0(self):
        ec, _, _ = self.cache[('imm_nop', 0)]
        self.assertEqual(ec, 0)

    def test_imm_nop_exit_255(self):
        ec, _, _ = self.cache[('imm_nop', 255)]
        self.assertEqual(ec, 255)


class TestNOPStepStructure(unittest.TestCase):
    """NOP programs have correct step count and structure."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_nop_cache()

    def test_nop_imm_exit_3_steps(self):
        _, _, steps = self.cache[('nop_imm', 42)]
        self.assertEqual(len(steps), 3)

    def test_imm_nop_exit_3_steps(self):
        _, _, steps = self.cache[('imm_nop', 42)]
        self.assertEqual(len(steps), 3)

    def test_nop_imm_exit_step_ends(self):
        _, _, steps = self.cache[('nop_imm', 42)]
        self.assertEqual(steps[0][-1], Token.STEP_END)
        self.assertEqual(steps[1][-1], Token.STEP_END)
        self.assertEqual(steps[2][-1], Token.HALT)


# =========================================================================
# 24. JMP tests
# =========================================================================

_jmp_cache = None

def _get_jmp_cache():
    global _jmp_cache
    if _jmp_cache is None:
        model = _get_model()
        _jmp_cache = {}
        # JMP 12; EXIT; IMM 99; EXIT -> skip EXIT at PC=7, land at IMM 99 (PC=12)
        _jmp_cache['skip_exit'] = run_program(
            model,
            [Opcode.JMP | (12 << 8), Opcode.EXIT, Opcode.IMM | (99 << 8), Opcode.EXIT],
            max_steps=8,
        )
        # JMP 7; IMM 42; EXIT -> jump to IMM 42 (no skip, target = next instr)
        _jmp_cache['jmp_next'] = run_program(
            model,
            [Opcode.JMP | (7 << 8), Opcode.IMM | (42 << 8), Opcode.EXIT],
            max_steps=8,
        )
    return _jmp_cache


class TestJMPExitCodes(unittest.TestCase):
    """JMP correctly redirects PC."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_jmp_cache()

    def test_jmp_skip_exit(self):
        """JMP 12; EXIT; IMM 99; EXIT -> exit_code=99."""
        ec, _, _ = self.cache['skip_exit']
        self.assertEqual(ec, 99)

    def test_jmp_next(self):
        """JMP 7; IMM 42; EXIT -> exit_code=42 (JMP to next instruction)."""
        ec, _, _ = self.cache['jmp_next']
        self.assertEqual(ec, 42)


class TestJMPStepStructure(unittest.TestCase):
    """JMP programs have correct step structure."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_jmp_cache()

    def test_jmp_skip_exit_steps(self):
        """JMP 12; EXIT; IMM 99; EXIT -> 3 steps (JMP, IMM, EXIT)."""
        _, _, steps = self.cache['skip_exit']
        # Step 0: JMP (PC=2), Step 1: IMM 99 (PC=12), Step 2: EXIT (PC=17)
        self.assertEqual(len(steps), 3)

    def test_jmp_skip_exit_halt(self):
        _, _, steps = self.cache['skip_exit']
        self.assertEqual(steps[0][-1], Token.STEP_END)
        self.assertEqual(steps[1][-1], Token.STEP_END)
        self.assertEqual(steps[2][-1], Token.HALT)

    def test_jmp_skip_exit_pc(self):
        """After JMP 12, PC should be 12 in step 1."""
        _, _, steps = self.cache['skip_exit']
        pc1 = extract_register(steps[1], Token.REG_PC)
        self.assertEqual(pc1, 12)

    def test_jmp_next_steps(self):
        """JMP 7; IMM 42; EXIT -> 3 steps."""
        _, _, steps = self.cache['jmp_next']
        self.assertEqual(len(steps), 3)


# =========================================================================
# 25. PC carry tests
# =========================================================================

class TestPCCarry(unittest.TestCase):
    """PC increment correctly handles nibble carry (e.g., 12+5=17)."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_triple_cache()
        cls.override_cache = _get_override_cache()

    def test_pc_12_to_17(self):
        """IMM; IMM; IMM; EXIT -> PC goes 2, 7, 12, 17."""
        _, _, steps = self.cache[(0, 42, 128)]
        self.assertEqual(len(steps), 4)
        self.assertEqual(extract_register(steps[0], Token.REG_PC), 2)
        self.assertEqual(extract_register(steps[1], Token.REG_PC), 7)
        self.assertEqual(extract_register(steps[2], Token.REG_PC), 12)
        self.assertEqual(extract_register(steps[3], Token.REG_PC), 17)

    def test_pc_7_to_12(self):
        """IMM; IMM; EXIT -> PC=2, 7, 12."""
        _, _, steps = self.override_cache[(42, 99)]
        self.assertEqual(extract_register(steps[0], Token.REG_PC), 2)
        self.assertEqual(extract_register(steps[1], Token.REG_PC), 7)
        self.assertEqual(extract_register(steps[2], Token.REG_PC), 12)


# =========================================================================
# 26. PSH tests (push AX to stack)
# =========================================================================

_psh_cache = None

def _get_psh_cache():
    global _psh_cache
    if _psh_cache is None:
        model = _get_model()
        _psh_cache = {}
        # Group 1: 4-instruction programs (IMM a; PSH; IMM b; EXIT)
        keys4 = []
        bytecodes4 = []
        for a, b in [(5, 3), (0, 255), (42, 42), (128, 1)]:
            keys4.append(('psh', a, b))
            bytecodes4.append([Opcode.IMM | (a << 8), Opcode.PSH,
                               Opcode.IMM | (b << 8), Opcode.EXIT])
        results4 = run_programs_batch(model, bytecodes4, max_steps=8)
        for i, k in enumerate(keys4):
            _psh_cache[k] = results4[i]
        # Group 2: 3-instruction programs (IMM a; PSH; EXIT)
        keys3 = []
        bytecodes3 = []
        for a in [0, 5, 42, 128, 255]:
            keys3.append(('psh_exit', a))
            bytecodes3.append([Opcode.IMM | (a << 8), Opcode.PSH, Opcode.EXIT])
        results3 = run_programs_batch(model, bytecodes3, max_steps=8)
        for i, k in enumerate(keys3):
            _psh_cache[k] = results3[i]
    return _psh_cache


class TestPSH(unittest.TestCase):
    """PSH pushes AX to stack, decrements SP by 8."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_psh_cache()

    def test_psh_preserves_ax(self):
        """IMM 5; PSH; IMM 3; EXIT -> exit=3 (AX overwritten by second IMM)."""
        ec, _, _ = self.cache[('psh', 5, 3)]
        self.assertEqual(ec, 3)

    def test_psh_exit_preserves_ax(self):
        """IMM 42; PSH; EXIT -> exit=42 (PSH doesn't change AX)."""
        ec, _, _ = self.cache[('psh_exit', 42)]
        self.assertEqual(ec, 42)

    def test_psh_sp_decrements(self):
        """After PSH, SP byte 0 should decrease by 8."""
        _, _, steps = self.cache[('psh_exit', 42)]
        sp0 = extract_register(steps[0], Token.REG_SP)  # IMM step
        sp1 = extract_register(steps[1], Token.REG_SP)  # PSH step
        # Model produces byte 0 correctly; runner handles multi-byte correction
        self.assertEqual(sp1 & 0xFF, (sp0 - 8) & 0xFF)

    def test_psh_stack0_set(self):
        """After PSH, STACK0 should contain the pushed AX value."""
        _, _, steps = self.cache[('psh_exit', 42)]
        # PSH step should set STACK0 = AX = 42
        st0 = extract_register(steps[1], Token.STACK0)
        self.assertEqual(st0, 42)

    def test_psh_stack0_zero(self):
        """PSH with AX=0 sets STACK0=0."""
        _, _, steps = self.cache[('psh_exit', 0)]
        st0 = extract_register(steps[1], Token.STACK0)
        self.assertEqual(st0, 0)

    def test_psh_stack0_255(self):
        """PSH with AX=255 sets STACK0=255."""
        _, _, steps = self.cache[('psh_exit', 255)]
        st0 = extract_register(steps[1], Token.STACK0)
        self.assertEqual(st0, 255)


# =========================================================================
# 27. Binary op tests (ADD, SUB, comparisons, bitwise, MUL)
# =========================================================================

_binop_cache = None

_BINOP_TESTS = [
    # ADD
    (5, 3, Opcode.ADD, 'add'),
    (0, 0, Opcode.ADD, 'add'),
    (200, 100, Opcode.ADD, 'add'),
    (255, 1, Opcode.ADD, 'add'),
    (128, 128, Opcode.ADD, 'add'),
    (15, 1, Opcode.ADD, 'add'),   # nibble boundary
    # SUB
    (10, 3, Opcode.SUB, 'sub'),
    (0, 1, Opcode.SUB, 'sub'),
    (100, 100, Opcode.SUB, 'sub'),
    (255, 255, Opcode.SUB, 'sub'),
    (16, 1, Opcode.SUB, 'sub'),   # nibble boundary
    # EQ
    (5, 5, Opcode.EQ, 'eq'),
    (5, 3, Opcode.EQ, 'eq'),
    (0, 0, Opcode.EQ, 'eq'),
    (255, 255, Opcode.EQ, 'eq'),
    # NE
    (5, 3, Opcode.NE, 'ne'),
    (5, 5, Opcode.NE, 'ne'),
    # LT
    (3, 5, Opcode.LT, 'lt'),
    (5, 3, Opcode.LT, 'lt'),
    (5, 5, Opcode.LT, 'lt'),
    (0, 255, Opcode.LT, 'lt'),
    # GT
    (5, 3, Opcode.GT, 'gt'),
    (3, 5, Opcode.GT, 'gt'),
    (5, 5, Opcode.GT, 'gt'),
    (255, 0, Opcode.GT, 'gt'),
    # LE
    (3, 5, Opcode.LE, 'le'),
    (5, 5, Opcode.LE, 'le'),
    (5, 3, Opcode.LE, 'le'),
    # GE
    (5, 3, Opcode.GE, 'ge'),
    (5, 5, Opcode.GE, 'ge'),
    (3, 5, Opcode.GE, 'ge'),
    # OR
    (0xF0, 0x0F, Opcode.OR, 'or'),
    (0xAA, 0x55, Opcode.OR, 'or'),
    (0, 0, Opcode.OR, 'or'),
    # XOR
    (0xF0, 0x0F, Opcode.XOR, 'xor'),
    (0xFF, 0xFF, Opcode.XOR, 'xor'),
    (0, 0, Opcode.XOR, 'xor'),
    # AND
    (0xF0, 0x0F, Opcode.AND, 'and'),
    (0xFF, 0x0F, Opcode.AND, 'and'),
    (0, 0xFF, Opcode.AND, 'and'),
    # MUL
    (3, 4, Opcode.MUL, 'mul'),
    (0, 255, Opcode.MUL, 'mul'),
    (15, 17, Opcode.MUL, 'mul'),
    (16, 16, Opcode.MUL, 'mul'),
    # DIV
    (10, 3, Opcode.DIV, 'div'),
    (100, 10, Opcode.DIV, 'div'),
    (255, 1, Opcode.DIV, 'div'),
    (7, 7, Opcode.DIV, 'div'),
    (0, 5, Opcode.DIV, 'div'),
    (3, 5, Opcode.DIV, 'div'),
    (200, 50, Opcode.DIV, 'div'),
    (255, 16, Opcode.DIV, 'div'),
    # MOD
    (10, 3, Opcode.MOD, 'mod'),
    (100, 10, Opcode.MOD, 'mod'),
    (255, 1, Opcode.MOD, 'mod'),
    (7, 7, Opcode.MOD, 'mod'),
    (0, 5, Opcode.MOD, 'mod'),
    (3, 5, Opcode.MOD, 'mod'),
    (200, 50, Opcode.MOD, 'mod'),
    (255, 16, Opcode.MOD, 'mod'),
]

def _get_binop_cache():
    """Cache: IMM a; PSH; IMM b; OP; EXIT -> exit_code = a OP b."""
    global _binop_cache
    if _binop_cache is None:
        model = _get_model()
        keys = [(name, a, b) for a, b, op, name in _BINOP_TESTS]
        bytecodes = [
            [Opcode.IMM | (a << 8), Opcode.PSH, Opcode.IMM | (b << 8), op, Opcode.EXIT]
            for a, b, op, name in _BINOP_TESTS
        ]
        results = run_programs_batch(model, bytecodes, max_steps=10)
        _binop_cache = {keys[i]: results[i] for i in range(len(keys))}
    return _binop_cache


def _expected_binop(name, a, b):
    """Compute expected result for a binary op."""
    ops = {
        'add': lambda a, b: (a + b) & 0xFF,
        'sub': lambda a, b: (a - b) & 0xFF,
        'eq': lambda a, b: 1 if a == b else 0,
        'ne': lambda a, b: 1 if a != b else 0,
        'lt': lambda a, b: 1 if a < b else 0,
        'gt': lambda a, b: 1 if a > b else 0,
        'le': lambda a, b: 1 if a <= b else 0,
        'ge': lambda a, b: 1 if a >= b else 0,
        'or': lambda a, b: a | b,
        'xor': lambda a, b: a ^ b,
        'and': lambda a, b: a & b,
        'mul': lambda a, b: (a * b) & 0xFF,
        'div': lambda a, b: (a // b) if b != 0 else 0,
        'mod': lambda a, b: (a % b) if b != 0 else 0,
    }
    return ops[name](a, b)


class TestBinaryOps(unittest.TestCase):
    """IMM a; PSH; IMM b; OP; EXIT -> exit_code = a OP b."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_binop_cache()


def _make_binop_test(name, a, b):
    def test(self):
        ec, _, steps = self.cache[(name, a, b)]
        expected = _expected_binop(name, a, b)
        self.assertEqual(ec, expected,
                         f"{a} {name} {b}: expected {expected}, got {ec}")
    test.__doc__ = f"{a} {name} {b} = {_expected_binop(name, a, b)}"
    return test

# Generate test methods for all binary op cases
for _key in [
    ('add', 5, 3), ('add', 0, 0), ('add', 200, 100), ('add', 255, 1),
    ('add', 128, 128), ('add', 15, 1),
    ('sub', 10, 3), ('sub', 0, 1), ('sub', 100, 100), ('sub', 255, 255),
    ('sub', 16, 1),
    ('eq', 5, 5), ('eq', 5, 3), ('eq', 0, 0), ('eq', 255, 255),
    ('ne', 5, 3), ('ne', 5, 5),
    ('lt', 3, 5), ('lt', 5, 3), ('lt', 5, 5), ('lt', 0, 255),
    ('gt', 5, 3), ('gt', 3, 5), ('gt', 5, 5), ('gt', 255, 0),
    ('le', 3, 5), ('le', 5, 5), ('le', 5, 3),
    ('ge', 5, 3), ('ge', 5, 5), ('ge', 3, 5),
    ('or', 0xF0, 0x0F), ('or', 0xAA, 0x55), ('or', 0, 0),
    ('xor', 0xF0, 0x0F), ('xor', 0xFF, 0xFF), ('xor', 0, 0),
    ('and', 0xF0, 0x0F), ('and', 0xFF, 0x0F), ('and', 0, 0xFF),
    ('mul', 3, 4), ('mul', 0, 255), ('mul', 15, 17), ('mul', 16, 16),
    ('div', 10, 3), ('div', 100, 10), ('div', 255, 1), ('div', 7, 7),
    ('div', 0, 5), ('div', 3, 5), ('div', 200, 50), ('div', 255, 16),
    ('mod', 10, 3), ('mod', 100, 10), ('mod', 255, 1), ('mod', 7, 7),
    ('mod', 0, 5), ('mod', 3, 5), ('mod', 200, 50), ('mod', 255, 16),
]:
    _name, _a, _b = _key
    setattr(TestBinaryOps, f'test_{_name}_{_a}_{_b}',
            _make_binop_test(_name, _a, _b))


# =========================================================================
# 27b. DIV/MOD edge cases (division by zero)
# =========================================================================

_divmod_zero_cache = None

def _get_divmod_zero_cache():
    """Cache DIV/MOD by zero cases via batch (model-only)."""
    global _divmod_zero_cache
    if _divmod_zero_cache is None:
        model = _get_model()
        cases = [
            (42, 0, Opcode.DIV, 'div'),
            (0, 0, Opcode.DIV, 'div'),
            (255, 0, Opcode.DIV, 'div'),
            (42, 0, Opcode.MOD, 'mod'),
            (0, 0, Opcode.MOD, 'mod'),
            (255, 0, Opcode.MOD, 'mod'),
        ]
        bytecodes = [
            [Opcode.IMM | (a << 8), Opcode.PSH, Opcode.IMM | (b << 8), op, Opcode.EXIT]
            for a, b, op, name in cases
        ]
        results = run_programs_batch(model, bytecodes, max_steps=10)
        _divmod_zero_cache = {(name, a, b): results[i] for i, (a, b, op, name) in enumerate(cases)}
    return _divmod_zero_cache


class TestDivModEdgeCases(unittest.TestCase):
    """DIV/MOD by zero -> result 0."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_divmod_zero_cache()

    def test_div_42_0(self):
        """42 div 0 = 0"""
        ec, _, _ = self.cache[('div', 42, 0)]
        self.assertEqual(ec, 0)

    def test_div_0_0(self):
        """0 div 0 = 0"""
        ec, _, _ = self.cache[('div', 0, 0)]
        self.assertEqual(ec, 0)

    def test_div_255_0(self):
        """255 div 0 = 0"""
        ec, _, _ = self.cache[('div', 255, 0)]
        self.assertEqual(ec, 0)

    def test_mod_42_0(self):
        """42 mod 0 = 0"""
        ec, _, _ = self.cache[('mod', 42, 0)]
        self.assertEqual(ec, 0)

    def test_mod_0_0(self):
        """0 mod 0 = 0"""
        ec, _, _ = self.cache[('mod', 0, 0)]
        self.assertEqual(ec, 0)

    def test_mod_255_0(self):
        """255 mod 0 = 0"""
        ec, _, _ = self.cache[('mod', 255, 0)]
        self.assertEqual(ec, 0)


# =========================================================================
# 28. Binary op SP increment (pop semantics)
# =========================================================================

class TestBinaryOpSP(unittest.TestCase):
    """Binary ops pop from stack: SP increases by 8."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_binop_cache()

    def test_add_sp_increment(self):
        """After ADD, SP byte 0 should be 8 more (mod 256) than after PSH."""
        _, _, steps = self.cache[('add', 5, 3)]
        # Steps: IMM(0), PSH(1), IMM(2), ADD(3), EXIT(4)
        sp_psh = extract_register(steps[1], Token.REG_SP)  # after PSH
        sp_add = extract_register(steps[3], Token.REG_SP)  # after ADD
        # SP arithmetic is byte 0 only (no cross-byte carry)
        self.assertEqual(sp_add & 0xFF, (sp_psh + 8) & 0xFF)


# =========================================================================
# 29. BZ/BNZ tests (conditional branch)
# =========================================================================

_bz_cache = None

def _get_bz_cache():
    global _bz_cache
    if _bz_cache is None:
        model = _get_model()
        # All 4 programs have 6 instructions → same context length → one batch
        keys = ['bz_taken', 'bz_not_taken', 'bnz_taken', 'bnz_not_taken']
        bytecodes = [
            # BZ taken: IMM 0; BZ 20; IMM 1; EXIT; IMM 2; EXIT
            # Instruction 4 (IMM 2) is at address 4*5=20
            [Opcode.IMM | (0 << 8), Opcode.BZ | (20 << 8),
             Opcode.IMM | (1 << 8), Opcode.EXIT,
             Opcode.IMM | (2 << 8), Opcode.EXIT],
            # BZ not taken: IMM 1; BZ 20; IMM 42; EXIT; IMM 99; EXIT
            [Opcode.IMM | (1 << 8), Opcode.BZ | (20 << 8),
             Opcode.IMM | (42 << 8), Opcode.EXIT,
             Opcode.IMM | (99 << 8), Opcode.EXIT],
            # BNZ taken: IMM 1; BNZ 20; IMM 1; EXIT; IMM 2; EXIT
            [Opcode.IMM | (1 << 8), Opcode.BNZ | (20 << 8),
             Opcode.IMM | (1 << 8), Opcode.EXIT,
             Opcode.IMM | (2 << 8), Opcode.EXIT],
            # BNZ not taken: IMM 0; BNZ 20; IMM 42; EXIT; IMM 99; EXIT
            [Opcode.IMM | (0 << 8), Opcode.BNZ | (20 << 8),
             Opcode.IMM | (42 << 8), Opcode.EXIT,
             Opcode.IMM | (99 << 8), Opcode.EXIT],
        ]
        results = run_programs_batch(model, bytecodes, max_steps=10)
        _bz_cache = {keys[i]: results[i] for i in range(len(keys))}
    return _bz_cache


class TestBZBNZ(unittest.TestCase):
    """BZ/BNZ conditional branch tests."""
    @classmethod
    def setUpClass(cls):
        cls.cache = _get_bz_cache()

    def test_bz_taken(self):
        """IMM 0; BZ target; ... -> branches to target, exit=2."""
        ec, _, _ = self.cache['bz_taken']
        self.assertEqual(ec, 2)

    def test_bz_not_taken(self):
        """IMM 1; BZ target; IMM 42; EXIT -> falls through, exit=42."""
        ec, _, _ = self.cache['bz_not_taken']
        self.assertEqual(ec, 42)

    def test_bnz_taken(self):
        """IMM 1; BNZ target; ... -> branches to target, exit=2."""
        ec, _, _ = self.cache['bnz_taken']
        self.assertEqual(ec, 2)

    def test_bnz_not_taken(self):
        """IMM 0; BNZ target; IMM 42; EXIT -> falls through, exit=42."""
        ec, _, _ = self.cache['bnz_not_taken']
        self.assertEqual(ec, 42)

    def test_bz_preserves_ax(self):
        """BZ preserves AX when not taken."""
        _, _, steps = self.cache['bz_not_taken']
        # After BZ (step 1), AX should still be 1
        ax = extract_register(steps[1], Token.REG_AX)
        self.assertEqual(ax, 1)

    def test_bnz_preserves_ax(self):
        """BNZ preserves AX when not taken."""
        _, _, steps = self.cache['bnz_not_taken']
        ax = extract_register(steps[1], Token.REG_AX)
        self.assertEqual(ax, 0)


# =========================================================================
# 30. Function call opcodes (LEA, JSR, ENT, LEV)
# =========================================================================

_func_call_cache = None

def _get_func_call_cache():
    """Cache function-call test results using the runner pipeline."""
    global _func_call_cache
    if _func_call_cache is None:
        from neural_vm.run_vm import AutoregressiveVMRunner
        model = _get_model()
        _func_call_cache = {}

        def _run(name, bytecode, max_steps=50):
            runner = AutoregressiveVMRunner()
            runner.model = model
            output_str, exit_code = runner.run(bytecode, max_steps=max_steps)
            _func_call_cache[name] = exit_code

        # LEA tests (initial BP=0)
        _run('lea_42', [Opcode.LEA | (42 << 8), Opcode.EXIT])
        _run('lea_0', [Opcode.LEA | (0 << 8), Opcode.EXIT])
        _run('lea_100', [Opcode.LEA | (100 << 8), Opcode.EXIT])
        _run('lea_7', [Opcode.LEA | (7 << 8), Opcode.EXIT])

        # JSR + ENT + LEV round trip: JSR func; EXIT; func: ENT 0; IMM 42; LEV
        # Target = instruction 2, immediate = 5*(2-1) = 5
        _run('jsr_lev_simple', [
            Opcode.JSR | (5 << 8),       # 0: JSR func (target=inst 2)
            Opcode.EXIT,                  # 1: EXIT
            Opcode.ENT | (0 << 8),        # 2: func: ENT 0
            Opcode.IMM | (42 << 8),       # 3: IMM 42
            Opcode.LEV,                   # 4: LEV
        ])

        # JSR preserves AX: IMM 99; JSR func; EXIT; func: ENT 0; LEV
        # Target = instruction 3, immediate = 5*(3-1) = 10
        _run('jsr_preserves_ax', [
            Opcode.IMM | (99 << 8),       # 0: IMM 99
            Opcode.JSR | (10 << 8),       # 1: JSR func (target=inst 3)
            Opcode.EXIT,                  # 2: EXIT
            Opcode.ENT | (0 << 8),        # 3: func: ENT 0
            Opcode.LEV,                   # 4: LEV
        ])

        # ENT with locals: JSR func; EXIT; func: ENT 16; IMM 77; LEV
        # Target = instruction 2, immediate = 5*(2-1) = 5
        _run('ent_with_locals', [
            Opcode.JSR | (5 << 8),        # 0: JSR func (target=inst 2)
            Opcode.EXIT,                  # 1: EXIT
            Opcode.ENT | (16 << 8),       # 2: func: ENT 16
            Opcode.IMM | (77 << 8),       # 3: IMM 77
            Opcode.LEV,                   # 4: LEV
        ])

        # Nested calls: JSR f; EXIT; f: ENT 0; IMM 7; JSR g; LEV; g: ENT 0; LEV
        # Target f = instruction 2, immediate = 5*(2-1) = 5
        # Target g = instruction 6, immediate = 5*(6-1) = 25
        _run('nested_calls', [
            Opcode.JSR | (5 << 8),        # 0: JSR f (target=inst 2)
            Opcode.EXIT,                  # 1: EXIT
            Opcode.ENT | (0 << 8),        # 2: f: ENT 0
            Opcode.IMM | (7 << 8),        # 3: IMM 7
            Opcode.JSR | (25 << 8),       # 4: JSR g (target=inst 6)
            Opcode.LEV,                   # 5: LEV (f returns)
            Opcode.ENT | (0 << 8),        # 6: g: ENT 0
            Opcode.LEV,                   # 7: LEV (g returns)
        ])

        # Function that modifies AX: JSR func; EXIT; func: ENT 0; IMM 200; LEV
        _run('func_returns_200', [
            Opcode.JSR | (5 << 8),        # 0: JSR func (target=inst 2)
            Opcode.EXIT,                  # 1: EXIT
            Opcode.ENT | (0 << 8),        # 2: func: ENT 0
            Opcode.IMM | (200 << 8),      # 3: IMM 200
            Opcode.LEV,                   # 4: LEV
        ])

        # LEA inside function: JSR func; EXIT; func: ENT 0; LEA 8; LEV
        _run('lea_in_func', [
            Opcode.JSR | (5 << 8),        # 0: JSR func (target=inst 2)
            Opcode.EXIT,                  # 1: EXIT
            Opcode.ENT | (0 << 8),        # 2: func: ENT 0
            Opcode.LEA | (8 << 8),        # 3: LEA 8
            Opcode.LEV,                   # 4: LEV
        ])

    return _func_call_cache


class TestFunctionCallOpcodes(unittest.TestCase):
    """Tests for function-call opcodes: LEA, JSR, ENT, LEV."""

    @classmethod
    def setUpClass(cls):
        cls.cache = _get_func_call_cache()

    # --- LEA tests ---

    def test_lea_42(self):
        """LEA 42; EXIT -> AX = BP + 42 = 0 + 42 = 42."""
        self.assertEqual(self.cache['lea_42'], 42)

    def test_lea_0(self):
        """LEA 0; EXIT -> AX = 0 + 0 = 0."""
        self.assertEqual(self.cache['lea_0'], 0)

    def test_lea_100(self):
        """LEA 100; EXIT -> AX = 0 + 100 = 100."""
        self.assertEqual(self.cache['lea_100'], 100)

    def test_lea_7(self):
        """LEA 7; EXIT -> AX = 0 + 7 = 7."""
        self.assertEqual(self.cache['lea_7'], 7)

    # --- JSR + ENT + LEV tests ---

    def test_jsr_lev_simple(self):
        """JSR func; EXIT; func: ENT 0; IMM 42; LEV -> exit=42."""
        self.assertEqual(self.cache['jsr_lev_simple'], 42)

    def test_jsr_preserves_ax(self):
        """IMM 99; JSR func; EXIT; func: ENT 0; LEV -> exit=99."""
        self.assertEqual(self.cache['jsr_preserves_ax'], 99)

    def test_ent_with_locals(self):
        """JSR func; EXIT; func: ENT 16; IMM 77; LEV -> exit=77."""
        self.assertEqual(self.cache['ent_with_locals'], 77)

    def test_nested_calls(self):
        """JSR f; EXIT; f: ENT 0; IMM 7; JSR g; LEV; g: ENT 0; LEV -> exit=7."""
        self.assertEqual(self.cache['nested_calls'], 7)

    def test_func_returns_200(self):
        """JSR func; EXIT; func: ENT 0; IMM 200; LEV -> exit=200."""
        self.assertEqual(self.cache['func_returns_200'], 200)

    # --- LEA inside function ---

    def test_lea_in_func(self):
        """JSR func; EXIT; func: ENT 0; LEA 8; LEV -> AX = BP + 8."""
        # BP is set by ENT to (old_sp - 8). The exact value depends on
        # initial SP (0) minus JSR's SP decrement (-8) minus ENT push (-8).
        # BP = (0 - 8 - 8) & 0xFFFFFFFF = 0xFFFFFFF0.
        # LEA 8: AX = 0xFFFFFFF0 + 8 = 0xFFFFFFF8.
        self.assertEqual(self.cache['lea_in_func'], 0xFFFFFFF8)


# =========================================================================
# Memory operation tests (PSH/LI/SI/SC/LC, MEM section, STACK0, ZFOD)
# =========================================================================


class TestMemoryOps(unittest.TestCase):
    """Test autoregressive memory operations.

    Tests verify that the model generates correct MEM section bytes,
    STACK0 values, and AX values for load/store operations.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _get_model()
        from neural_vm.run_vm import AutoregressiveVMRunner
        cls.runner_cls = AutoregressiveVMRunner

    def _run(self, bytecode, max_steps=10, data=b""):
        runner = self.runner_cls()
        runner.model = self.model
        return runner.run(bytecode, data=data, max_steps=max_steps)

    def _run_steps(self, bytecode, max_steps=10):
        """Run and return (exit_code, steps)."""
        exit_code, generated, steps = run_program(self.model, bytecode, max_steps)
        return exit_code, steps

    def test_psh_generates_mem_section(self):
        """PSH should generate non-zero MEM addr/val bytes."""
        # IMM 42; PSH; IMM 0; EXIT
        bytecode = [
            Opcode.IMM | (42 << 8),
            Opcode.PSH,
            Opcode.IMM | (0 << 8),
            Opcode.EXIT,
        ]
        _, _, steps = run_program(self.model, bytecode, max_steps=6)
        # Step 1 is PSH — check MEM section exists with addr/val
        if len(steps) > 1:
            psh_step = steps[1]
            mem_idx = None
            for i, tok in enumerate(psh_step):
                if tok == Token.MEM:
                    mem_idx = i
                    break
            self.assertIsNotNone(mem_idx, "MEM marker not found in PSH step")

    def test_imm_exit_still_works(self):
        """Regression: IMM v; EXIT should still produce correct exit code."""
        for v in [0, 1, 42, 127, 128, 255]:
            _, ec = self._run([Opcode.IMM | (v << 8), Opcode.EXIT])
            self.assertEqual(ec, v, f"IMM {v}; EXIT failed")

    def test_stack0_identity(self):
        """STACK0 at step 0 should be 0 (initial *SP when SP=0)."""
        bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
        _, _, steps = run_program(self.model, bytecode, max_steps=3)
        if len(steps) > 0:
            stack0 = extract_register(steps[0], Token.STACK0)
            # Initial STACK0 is *SP where SP=0, memory uninitialized → ZFOD → 0
            self.assertEqual(stack0, 0, "Initial STACK0 should be 0 (ZFOD)")

    # --- MEM section helpers ---

    @staticmethod
    def _extract_mem(step):
        """Extract (addr, val) from MEM section of a step's token list."""
        for i, tok in enumerate(step):
            if tok == Token.MEM and i + 8 < len(step):
                addr = sum(step[i + 1 + j] << (j * 8) for j in range(4))
                val = sum(step[i + 5 + j] << (j * 8) for j in range(4))
                return addr, val
        return None, None

    # --- PSH MEM section tests ---

    def test_psh_mem_val_is_ax(self):
        """PSH: MEM val bytes should equal AX value being pushed."""
        bytecode = [Opcode.IMM | (42 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.EXIT]
        _, _, steps = run_program(self.model, bytecode, max_steps=6)
        self.assertGreater(len(steps), 1, "Need at least 2 steps")
        _, val = self._extract_mem(steps[1])
        self.assertEqual(val, 42, "PSH MEM val should be AX=42")

    def test_psh_mem_addr_is_sp(self):
        """PSH: MEM addr should be new SP (old SP - 8)."""
        bytecode = [Opcode.IMM | (42 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.EXIT]
        _, _, steps = run_program(self.model, bytecode, max_steps=6)
        self.assertGreater(len(steps), 1)
        addr, _ = self._extract_mem(steps[1])
        sp = extract_register(steps[1], Token.REG_SP)
        # SP should be 0-8 = 0xFFFFFFF8. MEM addr should equal SP.
        self.assertIsNotNone(addr)
        self.assertEqual(addr, sp, f"PSH MEM addr={addr:#x} should equal SP={sp:#x}")

    # --- PSH + LI semantics ---

    def test_psh_li_zfod(self):
        """IMM 42; PSH; LI -> AX should be 0 (LI loads from memory[AX]=memory[42], never written)."""
        bytecode = [
            Opcode.IMM | (42 << 8),
            Opcode.PSH,
            Opcode.LI,
            Opcode.EXIT,
        ]
        _, ec = self._run(bytecode)
        self.assertEqual(ec, 0, "PSH then LI from AX (not SP) should ZFOD")

    def test_si_li_round_trip_values(self):
        """SI+LI round-trip for various values including lo-nibble-0."""
        addr = 0x100
        for v in [0, 1, 16, 42, 128, 255]:
            bytecode = [
                Opcode.IMM | (addr << 8),   # AX = addr
                Opcode.PSH,                  # push addr (STACK0 = addr)
                Opcode.IMM | (v << 8),       # AX = v
                Opcode.SI,                   # memory[addr] = v
                Opcode.IMM | (addr << 8),    # AX = addr
                Opcode.LI,                   # AX = memory[addr] = v
                Opcode.EXIT,
            ]
            _, ec = self._run(bytecode, max_steps=10)
            self.assertEqual(ec, v, f"SI+LI round-trip failed for v={v}")

    # --- SI + LI ---

    def test_si_li_round_trip(self):
        """SI stores AX at *STACK0, then LI loads from the same address."""
        # Push address, then store value, then load it back
        # IMM addr; PSH; IMM val; SI; IMM addr; LI; EXIT
        addr_val = 0x100
        store_val = 77
        bytecode = [
            Opcode.IMM | (addr_val << 8),  # AX = addr
            Opcode.PSH,                     # push addr (STACK0 = addr)
            Opcode.IMM | (store_val << 8),  # AX = val
            Opcode.SI,                      # memory[STACK0] = AX
            Opcode.IMM | (addr_val << 8),   # AX = addr again
            Opcode.LI,                      # AX = memory[AX]
            Opcode.EXIT,
        ]
        _, ec = self._run(bytecode, max_steps=10)
        self.assertEqual(ec, store_val, f"SI+LI: expected {store_val}, got {ec}")

    # --- SC + LC ---

    def test_sc_lc_round_trip(self):
        """SC stores byte, LC loads byte."""
        addr_val = 0x200
        store_val = 0xAB
        bytecode = [
            Opcode.IMM | (addr_val << 8),
            Opcode.PSH,
            Opcode.IMM | (store_val << 8),
            Opcode.SC,                      # memory[STACK0] = AX & 0xFF
            Opcode.IMM | (addr_val << 8),
            Opcode.LC,                      # AX = memory[AX] & 0xFF
            Opcode.EXIT,
        ]
        _, ec = self._run(bytecode, max_steps=10)
        self.assertEqual(ec, store_val & 0xFF, f"SC+LC: expected {store_val & 0xFF}, got {ec}")

    # --- ZFOD (read uninitialized → 0) ---

    def test_zfod_li(self):
        """LI from uninitialized address should return 0 (ZFOD)."""
        bytecode = [
            Opcode.IMM | (0x1000 << 8),  # AX = address never written
            Opcode.LI,
            Opcode.EXIT,
        ]
        _, ec = self._run(bytecode, max_steps=5)
        self.assertEqual(ec, 0, "ZFOD: LI from uninitialized addr should be 0")

    def test_zfod_lc(self):
        """LC from uninitialized address should return 0 (ZFOD)."""
        bytecode = [
            Opcode.IMM | (0x2000 << 8),
            Opcode.LC,
            Opcode.EXIT,
        ]
        _, ec = self._run(bytecode, max_steps=5)
        self.assertEqual(ec, 0, "ZFOD: LC from uninitialized addr should be 0")

    # --- IMM all lo-nibble-0 values (regression) ---

    def test_imm_lo_nibble_zero(self):
        """Regression: all 0xN0 values should work correctly."""
        for hi in range(16):
            v = hi << 4
            _, ec = self._run([Opcode.IMM | (v << 8), Opcode.EXIT])
            self.assertEqual(ec, v, f"IMM 0x{v:02x} failed")

    # --- Multiple writes to same address (latest wins) ---

    def test_multiple_writes_latest_wins(self):
        """Two SI to same address, then LI should read latest write."""
        addr = 0x100
        bytecode = [
            Opcode.IMM | (addr << 8),  # AX = addr
            Opcode.PSH,                # STACK0 = addr
            Opcode.IMM | (10 << 8),    # AX = 10
            Opcode.SI,                 # memory[addr] = 10
            Opcode.IMM | (20 << 8),    # AX = 20
            Opcode.SI,                 # memory[addr] = 20 (overwrite)
            Opcode.IMM | (addr << 8),  # AX = addr
            Opcode.LI,                 # AX = memory[addr] = 20 (latest)
            Opcode.EXIT,
        ]
        _, ec = self._run(bytecode, max_steps=12)
        self.assertEqual(ec, 20, "Latest write should win")


# =========================================================================
# Speculative decoding tests
# =========================================================================

from neural_vm.speculative import DraftVM


def run_programs_batch_speculative(model, bytecodes_list, max_steps=5, batch_size=256):
    """Run multiple programs using speculative decoding (1 fwd pass per step).

    Uses DraftVM to predict all 35 tokens per step, then verifies against the
    transformer in a single forward pass. Falls back to token-by-token generation
    on mismatch.

    Returns: list of (exit_code, generated_tokens, steps, accepted_counts) tuples.
        accepted_counts: list of int per step (35 = full match).
    """
    from neural_vm.run_vm import AutoregressiveVMRunner

    all_results = [None] * len(bytecodes_list)

    for chunk_start in range(0, len(bytecodes_list), batch_size):
        chunk = bytecodes_list[chunk_start:chunk_start + batch_size]

        runner = AutoregressiveVMRunner()
        runner.model = model
        contexts = [runner._build_context(bc, b'', []) for bc in chunk]

        # Initialize draft VMs
        drafts = [DraftVM(bc) for bc in chunk]
        generated = [[] for _ in chunk]
        halted = [False] * len(chunk)
        accepted_counts = [[] for _ in chunk]

        for step_num in range(max_steps):
            if all(halted):
                break

            # Step 1: execute draft VMs
            for i in range(len(chunk)):
                if not halted[i]:
                    drafts[i].step()

            # Step 2: build draft tokens
            draft_tokens_list = []
            for i in range(len(chunk)):
                if not halted[i]:
                    draft_tokens_list.append(drafts[i].draft_tokens())
                else:
                    draft_tokens_list.append([0] * Token.STEP_TOKENS)

            # Step 3: append draft tokens to contexts for verification
            contexts_with_draft = []
            for i in range(len(chunk)):
                contexts_with_draft.append(contexts[i] + draft_tokens_list[i])

            # Pad to same length
            max_len = max(len(c) for c in contexts_with_draft)
            for i in range(len(contexts_with_draft)):
                while len(contexts_with_draft[i]) < max_len:
                    contexts_with_draft[i].insert(0, 0)  # left-pad

            # Step 4: verify in one batched forward pass
            draft_lens = [Token.STEP_TOKENS] * len(chunk)
            accepted = model.verify_speculative_batch(contexts_with_draft, draft_lens)

            # Step 5: process results
            for i in range(len(chunk)):
                if halted[i]:
                    continue

                if accepted[i] == Token.STEP_TOKENS:
                    # Full match: accept all draft tokens
                    contexts[i].extend(draft_tokens_list[i])
                    generated[i].extend(draft_tokens_list[i])
                    accepted_counts[i].append(accepted[i])
                    if draft_tokens_list[i][-1] == Token.HALT:
                        halted[i] = True
                else:
                    # Mismatch: fall back to token-by-token for this step
                    accepted_counts[i].append(accepted[i])
                    # Remove draft tokens from context, generate one at a time
                    for _ in range(Token.STEP_TOKENS):
                        tok = model.generate_next(contexts[i])
                        contexts[i].append(tok)
                        generated[i].append(tok)
                        if tok in (Token.STEP_END, Token.HALT):
                            if tok == Token.HALT:
                                halted[i] = True
                            break

        # Parse results (same as run_programs_batch)
        for i in range(len(chunk)):
            steps = []
            step_tokens = []
            for tok in generated[i]:
                step_tokens.append(tok)
                if tok in (Token.STEP_END, Token.HALT):
                    steps.append(step_tokens)
                    step_tokens = []

            exit_code = 0
            for j in range(len(contexts[i]) - 1, -1, -1):
                if contexts[i][j] == Token.REG_AX and j + 4 < len(contexts[i]):
                    exit_code = sum(contexts[i][j + 1 + k] << (k * 8) for k in range(4))
                    break

            all_results[chunk_start + i] = (exit_code, generated[i], steps, accepted_counts[i])

    return all_results


_spec_imm_cache = None

def _get_spec_imm_cache():
    """Cache speculative results for IMM v; EXIT for all v in 0..255."""
    global _spec_imm_cache
    if _spec_imm_cache is None:
        model = _get_model()
        bytecodes = [[Opcode.IMM | (v << 8), Opcode.EXIT] for v in range(256)]
        results = run_programs_batch_speculative(model, bytecodes)
        _spec_imm_cache = {v: results[v] for v in range(256)}
    return _spec_imm_cache


class TestSpeculativeDecoding(unittest.TestCase):
    """Verify speculative decoding produces identical results to standard decoding."""

    @classmethod
    def setUpClass(cls):
        cls.spec_cache = _get_spec_imm_cache()
        cls.orig_cache = _get_imm_cache()

    def test_imm_exit_codes_match(self):
        """All 256 IMM speculative exit codes match original."""
        for v in range(256):
            spec_ec = self.spec_cache[v][0]
            orig_ec = self.orig_cache[v][0]
            self.assertEqual(spec_ec, orig_ec, f"IMM {v}: spec={spec_ec}, orig={orig_ec}")

    def test_imm_tokens_match(self):
        """All 256 IMM speculative tokens match original token-by-token."""
        for v in range(256):
            spec_gen = self.spec_cache[v][1]
            orig_gen = self.orig_cache[v][1]
            self.assertEqual(spec_gen, orig_gen,
                             f"IMM {v}: speculative tokens differ from original")

    def test_acceptance_rate_100(self):
        """All draft tokens should be accepted for correct transformer weights."""
        for v in range(256):
            counts = self.spec_cache[v][3]
            for step_idx, count in enumerate(counts):
                self.assertEqual(count, Token.STEP_TOKENS,
                                 f"IMM {v} step {step_idx}: only {count}/35 accepted")

    def test_override_speculative(self):
        """Override pairs give correct results via speculative decoding."""
        model = _get_model()
        pairs = [(a, b) for a in [0, 42, 128, 255] for b in [0, 42, 128, 255]]
        bytecodes = [[Opcode.IMM | (a << 8), Opcode.IMM | (b << 8), Opcode.EXIT]
                     for a, b in pairs]
        spec_results = run_programs_batch_speculative(model, bytecodes)
        for i, (a, b) in enumerate(pairs):
            self.assertEqual(spec_results[i][0], b,
                             f"IMM {a}; IMM {b}; EXIT: expected {b}, got {spec_results[i][0]}")
            for step_idx, count in enumerate(spec_results[i][3]):
                self.assertEqual(count, Token.STEP_TOKENS,
                                 f"Override ({a},{b}) step {step_idx}: {count}/35 accepted")

    def test_psh_speculative(self):
        """PSH programs give correct results via speculative decoding."""
        model = _get_model()
        bytecodes = [
            [Opcode.IMM | (42 << 8), Opcode.PSH, Opcode.EXIT],
            [Opcode.IMM | (0 << 8), Opcode.PSH, Opcode.EXIT],
            [Opcode.IMM | (255 << 8), Opcode.PSH, Opcode.EXIT],
        ]
        spec_results = run_programs_batch_speculative(model, bytecodes, max_steps=6)
        for i, val in enumerate([42, 0, 255]):
            self.assertEqual(spec_results[i][0], val,
                             f"IMM {val}; PSH; EXIT: expected {val}, got {spec_results[i][0]}")

    def test_jmp_speculative(self):
        """JMP programs give correct results via speculative decoding."""
        model = _get_model()
        # JMP past IMM 1 to IMM 2: IMM 99; JMP 12; IMM 1; EXIT; (never reached)
        # But we need EXIT reachable — actually JMP target = instr_idx * 5 + 2
        # JMP to instruction 2 (idx=2): target = 2*5+2 = 12
        # Program: IMM 99; JMP 12; IMM 1; EXIT
        # idx 0: IMM 99 (pc after = 7)
        # idx 1: JMP 12 (pc jumps to 12, idx=2)
        # idx 2: IMM 1 -- skipped by JMP? No, JMP 12 goes TO idx 2.
        # Let's do: JMP forward over an IMM
        # idx 0: IMM 99; idx 1: JMP 17; idx 2: IMM 1; idx 3: EXIT
        # JMP 17 -> idx=(17-2)/5=3 -> EXIT. AX stays 99.
        bytecodes = [
            [Opcode.IMM | (99 << 8), Opcode.JMP | (17 << 8),
             Opcode.IMM | (1 << 8), Opcode.EXIT],
        ]
        spec_results = run_programs_batch_speculative(model, bytecodes, max_steps=6)
        self.assertEqual(spec_results[0][0], 99)

    def test_binop_speculative(self):
        """Binary op programs via speculative decoding."""
        model = _get_model()
        # IMM a; PSH; IMM b; ADD; EXIT
        cases = [(5, 3, Opcode.ADD, 8), (10, 3, Opcode.SUB, 7),
                 (5, 5, Opcode.EQ, 1), (5, 3, Opcode.EQ, 0)]
        bytecodes = [
            [Opcode.IMM | (a << 8), Opcode.PSH, Opcode.IMM | (b << 8), op, Opcode.EXIT]
            for a, b, op, _ in cases
        ]
        spec_results = run_programs_batch_speculative(model, bytecodes, max_steps=10)
        for i, (a, b, op, expected) in enumerate(cases):
            self.assertEqual(spec_results[i][0], expected,
                             f"{a} op {b}: expected {expected}, got {spec_results[i][0]}")


if __name__ == "__main__":
    unittest.main()
