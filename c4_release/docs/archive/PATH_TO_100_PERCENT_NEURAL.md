# Path to 100% Neural Execution

**Date**: 2026-04-09
**Current Status**: ~95% Neural (ENT 80%, LEV 10%)
**Goal**: 100% Pure Autoregressive VM with Zero Handlers

---

## Executive Summary

The C4 Transformer VM is currently ~95% neural for VM operations. This document outlines the path to achieve 100% pure autoregressive execution by:

1. Verifying and cleaning up existing neural implementations
2. Completing ENT neural (remove 13-line handler)
3. Completing LEV neural (remove 8-line handler)

**Estimated Total Effort**: 5-8 hours (testing + ENT) to 20-30 hours (testing + ENT + LEV)

---

## Current Handler Status

### Function Call Handlers (run_vm.py lines 220-243)

| Operation | Status | Neural % | Handler Lines | Notes |
|-----------|--------|----------|---------------|-------|
| IMM | ✅ Removed | 100% | 0 | L6 FFN relay |
| LEA | ✅ Removed | 100% | 0 | L7/L8/L9 ALU |
| JSR | ✅ Removed | 100% | 0 | L6 PC + STACK0 + L14 MEM |
| PSH | ✅ Removed | 100% | 0 | L6 FFN SP -= 8 |
| ENT | ⏳ **Minimized** | **80%** | **13** | Only SP -= (8+imm) remains |
| LEV | ⏳ **Retained** | **10%** | **8** | Full handler for 3 mem reads |
| ADD/SUB/MUL/DIV/MOD | ✅ Removed | 100% | 0 | L8-L10 ALU |
| OR/XOR/AND | ✅ Removed | 100% | 0 | L8-L10 ALU |
| SHL/SHR | ✅ Removed | 100% | 0 | L8-L10 ALU |

**Total Function Handlers**: 2 operations, 21 lines (down from 200+ originally)

### Syscall Handlers (run_vm.py lines 165-182)

| Operation | Status | Implementation | Action Required |
|-----------|--------|----------------|-----------------|
| ADJ | ⚠️ **Handler Active** | Neural (L7/L8/L9) | **Verify + Remove Handler** |
| MALC | ✅ C4 Stdlib | Bytecode subroutine | Keep for stdlib |
| FREE | ✅ C4 Stdlib | Bytecode subroutine | Keep for stdlib |
| MSET | ✅ C4 Stdlib | Bytecode subroutine | Keep for stdlib |
| MCMP | ✅ C4 Stdlib | Bytecode subroutine | Keep for stdlib |
| PUTCHAR | ✅ External Tool | I/O boundary | Keep (external) |
| GETCHAR | ✅ External Tool | I/O boundary | Keep (external) |
| OPEN | ✅ External Tool | I/O boundary | Keep (external) |
| READ | ✅ External Tool | I/O boundary | Keep (external) |
| CLOS | ✅ External Tool | I/O boundary | Keep (external) |
| PRTF | ✅ External Tool | I/O boundary | Keep (external) |

**Key Issue**: ADJ has neural implementation (vm_step.py L7/L8/L9) but handler is still active and not blocked in pure mode.

### SP Correction Handlers (run_vm.py lines 390-418)

| Correction | Status | Neural Implementation | Notes |
|------------|--------|----------------------|-------|
| Binary pop SP += 8 | ✅ Removed | vm_step.py:6015-6066 | 15+ opcodes (ADD, SUB, etc.) |
| PSH SP -= 8 | ✅ Removed | vm_step.py:3615-3641 | Multi-byte borrow |
| ADJ SP multi-byte | ✅ Removed | vm_step.py L7/L8/L9/L6 | Reuses LEA pattern |

**All SP corrections removed** - commented out in code.

---

## What "100% Neural" Means

**VM Operations to Make Neural:**
- ✅ Register arithmetic (PC, SP, BP, AX, STACK0) - DONE
- ✅ Instruction fetch and decode - DONE
- ✅ Memory reads (LI, LC via L15 softmax1) - DONE
- ✅ Memory writes (SI, SC, PSH via L14 MEM tokens) - DONE
- ✅ ALU operations (ADD, SUB, MUL, DIV, MOD, bitwise, shifts) - DONE
- ✅ Stack operations (PSH, binary pops) - DONE
- ⏳ Function calls (JSR ✅, ENT 80%, LEV 10%)
- ⚠️ Stack pointer adjustments (ADJ - neural but handler active)

**NOT "Handlers" to Remove:**
- ✅ External I/O boundaries (PUTCHAR, GETCHAR, OPEN, READ, CLOS, PRTF)
- ✅ C4 stdlib functions (MALC, FREE, MSET, MCMP as bytecode subroutines)

These are **intentional boundaries** where the VM interfaces with the outside world or user-provided C4 code. They are not VM semantics that need to be neural.

---

## Phase 1: Verify and Clean Existing Neural Implementations

**Goal**: Ensure all claimed neural implementations actually work without handlers.

**Estimated Time**: 2-3 hours

### 1.1 Verify ADJ Neural Implementation

**Current Issue**:
- Neural implementation exists: L7 head 1, L8 FFN, L9 FFN (vm_step.py:4420-4690)
- Handler still active: `_syscall_adj` (run_vm.py:1033-1049)
- Handler NOT blocked in pure_attention_memory mode (removed from _RUNNER_VM_MEMORY_OPS line 55)

**Test Plan**:
```python
# Test 1: Simple local variable allocation
int main() {
    int a, b, c;  # ADJ -12 (allocate 3 ints)
    a = 1;
    b = 2;
    c = 3;
    return a + b + c;  # Should return 6
}

# Test 2: Positive adjustment (deallocate)
int main() {
    int x;        # ADJ -4
    x = 42;
    # ADJ +4 (implicit at end)
    return x;
}

# Test 3: Large adjustment
int arr[100];     # ADJ -400
int main() {
    arr[0] = 7;
    arr[99] = 13;
    return arr[0] + arr[99];  # Should return 20
}

# Test 4: Sign extension (negative immediate)
# Test with ADJ -0x800000 (min 24-bit signed) → should become -8388608
```

**Action Steps**:
1. Create test file `tests/test_adj_neural.py` with above tests
2. Run tests with `pure_attention_memory=False` (handler active) - baseline
3. Run tests with `pure_attention_memory=True` and ADJ added to _RUNNER_VM_MEMORY_OPS (handler blocked)
4. If tests pass: Remove `_syscall_adj` handler entirely from `_syscall_handlers` (line 173)
5. Remove ADJ from `_syscall_handlers` dict

**Success Criteria**: All tests pass with handler blocked/removed.

---

### 1.2 Verify Binary Pop SP Corrections

**Current Status**:
- Neural implementation: vm_step.py:6015-6066
- Handler removed: lines 392-398 commented out

**Test Plan**:
```python
# Test 1: Simple binary operation
int main() {
    int x = 5;
    int y = 3;
    return x + y;  # ADD pops one operand, SP should += 8
}

# Test 2: Multi-byte SP carry
# Arrange SP near byte boundary (e.g., 0x10000 - 16)
# Perform operations that pop stack
# Verify SP crosses byte boundary correctly
```

**Action Steps**:
1. Run existing test suite with `pure_attention_memory=True`
2. Specifically test programs with heavy stack usage
3. Confirm no SP-related failures

**Success Criteria**: All 1096 tests pass without SP corrections.

---

### 1.3 Verify PSH SP -= 8

**Current Status**:
- Neural implementation: vm_step.py:3615-3641
- Handler removed: lines 400-406 commented out

**Test Plan**:
```python
# Test 1: Simple push
int main() {
    int x = 42;
    return x;  # PSH literal, SP -= 8
}

# Test 2: Multiple pushes
int main() {
    int a = 1;
    int b = 2;
    int c = 3;
    return a + b + c;  # Multiple PSH operations
}

# Test 3: Multi-byte borrow
# Arrange SP at boundary (e.g., 0x10000)
# PSH should borrow correctly across bytes
```

**Action Steps**:
1. Run existing test suite (PSH is heavily used)
2. Verify no PSH-related failures

**Success Criteria**: All tests pass without PSH corrections.

---

## Phase 2: Complete ENT Neural Implementation

**Goal**: Eliminate ENT handler entirely (currently 13 lines, 80% neural).

**Estimated Time**: 3-5 hours

### 2.1 Current ENT Status

**Neural Components** (80%):
- ✅ STACK0 = old_BP: L5 head 5 + L6 FFN units 978-1009
- ✅ BP = old_SP - 8: L5 head 6 + L6 FFN units 1010-1041
- ✅ MEM token (old_BP): L14 STACK0 source
- ✅ AX passthrough: L6 FFN units 1042-1073

**Handler Component** (20%):
- ❌ SP -= (8 + imm): Python handler override (run_vm.py:1504-1538 lines 1527-1538)

### 2.2 ENT Neural Implementation Plan

**Approach**: Reuse ADJ/LEA pattern for `SP -= (8 + imm)` arithmetic.

**Architecture**:
```
L7 Head 2: ENT-specific SP gather
  - Fires when OP_ENT active
  - Gathers SP OUTPUT → ALU at AX marker
  - Similar to ADJ (L7 head 1) but separate head

L8 FFN: ENT lo nibble subtraction
  - 256 units: enumerate all (sp_lo[0] - (8 + imm_lo[0])) combinations
  - Borrow detection: 120 units for pairs where sp_byte < (8 + imm_byte)
  - Gate on OP_ENT

L9 FFN: ENT hi nibbles with borrow propagation
  - 512 units: 256 × 2 (with/without borrow from lower byte)
  - Propagate borrow across all 4 bytes
  - Gate on OP_ENT

L6 FFN: ENT SP writeback
  - 32 units: carry NEXT_SP → OUTPUT at SP marker
  - Gate on OP_ENT
```

**Total Units**: ~1500 FFN units (similar to ADJ pattern)

**Dimensions Needed**:
- No new dimensions (reuse ALU, FETCH, CARRY, NEXT_SP)

### 2.3 Implementation Steps

1. **Extend L7 Attention** (vm_step.py ~line 4422)
   ```python
   # Head 2: ENT — SP OUTPUT → ALU at AX marker
   base = 2 * HD
   attn.W_q[base, BD.OP_ENT] = L  # fires when ENT active
   attn.W_k[base, BD.MARK_SP] = L  # attends to SP
   # V: copy OUTPUT_LO/HI (SP's byte-0 output from L6)
   for k in range(16):
       attn.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
       attn.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
   # O: route to ALU (SP value for subtraction)
   for k in range(16):
       attn.W_o[BD.ALU_LO + k, base + 1 + k] = 1.0
       attn.W_o[BD.ALU_HI + k, base + 17 + k] = 1.0
   ```

2. **Add L8 FFN ENT Lo Nibble** (vm_step.py ~line 4695)
   ```python
   # === ENT: lo nibble (256 units) ===
   # ENT computes: SP = SP - (8 + signed_immediate)
   # First compute: new_lo = sp_lo - ((8 + imm) & 0xF)
   for a in range(16):  # sp_lo
       for b in range(16):  # (8 + imm_lo) & 0xF
           result = (a - b) % 16  # Modulo handles negative
           ffn.W_up[unit, BD.MARK_AX] = S
           ffn.W_up[unit, BD.ALU_LO + a] = S      # sp_lo from L7
           ffn.W_up[unit, BD.FETCH_LO + b] = S    # (8+imm) lo nibble
           ffn.b_up[unit] = -S * 15.5
           ffn.W_gate[unit, BD.OP_ENT] = 1.0
           ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
           unit += 1

   # === ENT borrow detection (120 units) ===
   for a in range(16):
       for b in range(16):
           if a < b:  # Need borrow
               ffn.W_up[unit, BD.MARK_AX] = S
               ffn.W_up[unit, BD.ALU_LO + a] = S
               ffn.W_up[unit, BD.FETCH_LO + b] = S
               ffn.b_up[unit] = -S * 15.5
               ffn.W_gate[unit, BD.OP_ENT] = 1.0
               ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)  # Signal borrow
               unit += 1
   ```

3. **Add L9 FFN ENT Hi Nibbles** (vm_step.py ~line 4850)
   ```python
   # === ENT: hi nibbles with borrow (512 units) ===
   for byte_pos in range(1, 4):  # Bytes 1-3
       for a in range(16):  # sp_hi nibble
           for borrow in [0, 1]:  # Borrow from lower byte
               b = ((8 >> (byte_pos * 4)) & 0xF)  # 8's contribution at this byte (usually 0)
               result = (a - b - borrow) % 16
               needs_borrow = (a < b + borrow)

               ffn.W_up[unit, BD.MARK_AX] = S
               ffn.W_up[unit, BD.ALU_HI + (byte_pos-1)*16 + a] = S
               ffn.W_up[unit, BD.CARRY + (byte_pos-1)] = S * borrow  # Conditional
               ffn.b_up[unit] = -S * (10.5 + borrow * 5)  # Threshold
               ffn.W_gate[unit, BD.OP_ENT] = 1.0
               ffn.W_down[BD.OUTPUT_HI + (byte_pos-1)*16 + result, unit] = 2.0 / S
               if needs_borrow:
                   ffn.W_down[BD.CARRY + byte_pos, unit] = 2.0 / (S * 5.0)
               unit += 1
   ```

4. **Test ENT Neural**
   ```python
   # Test 1: Simple function with locals
   int helper(int x) {
       int local = x * 2;  # ENT allocates space
       return local;
   }
   int main() {
       return helper(21);  # Should return 42
   }

   # Test 2: Multiple locals
   int func() {
       int a, b, c;  # ENT -12
       a = 1;
       b = 2;
       c = 3;
       return a + b + c;
   }

   # Test 3: Nested calls
   int inner(int x) { int tmp; tmp = x + 1; return tmp; }
   int outer(int x) { int tmp; tmp = inner(x); return tmp; }
   int main() { return outer(5); }  # Should return 6
   ```

5. **Remove ENT Handler**
   - Delete `_handler_ent` function (run_vm.py:1504-1538)
   - Remove `Opcode.ENT: self._handler_ent` from `_func_call_handlers` (line 226)

**Success Criteria**: All ENT tests pass with handler removed. ENT is 100% neural.

---

## Phase 3: Complete LEV Neural Implementation

**Goal**: Eliminate LEV handler entirely (currently 8 lines, 10% neural).

**Estimated Time**: 14-22 hours

### 3.1 Current LEV Status

**Neural Components** (10%):
- ✅ AX passthrough: L6 FFN units (preserve return value)

**Handler Components** (90%):
- ❌ saved_bp = mem[BP]: Memory read
- ❌ return_addr = mem[BP+8]: Memory read
- ❌ stack0_val = mem[new_sp]: Memory read (depends on SP calculation)
- ❌ SP = BP + 16: Arithmetic
- ❌ BP = saved_bp: Register assignment
- ❌ PC = return_addr: Register assignment
- ❌ STACK0 = stack0_val: Register assignment

### 3.2 LEV Neural Implementation Plan

**Challenge**: LEV requires **3 memory reads** in one step, but L15 currently supports only **1 memory read per step** (4 heads for 4 bytes of one address).

**Solution**: Extend L15 to support **parallel memory reads** via 8 heads.

**Architecture**:
```
L15 Extension: Parallel Memory Lookups
  - Heads 0-3: Load saved_bp from memory[BP] (4 bytes)
  - Heads 4-7: Load return_addr from memory[BP+8] (4 bytes)
  - Both lookups happen in same transformer step
  - Results stored in TEMP dims for routing

L15 Address Encoding:
  - Add BP marker support (like existing SP/STACK0 support)
  - Q: Fire when OP_LEV active at BP marker
  - K: Match memory addresses based on BP value
  - V: Return 4 bytes of data

State Passing via TEMP:
  - TEMP[0-31]: saved_bp (32 nibbles for 4 bytes × 2 nibbles/byte)
  - TEMP[32-63]: return_addr
  - Available: TEMP region is [480-511] = 32 dims

L16 (New Layer): Register Routing
  - FFN units to route TEMP → OUTPUT at multiple markers:
    - SP marker: BP + 16 (simple addition)
    - BP marker: TEMP[0-31] (saved_bp)
    - PC marker: TEMP[32-63] (return_addr)
  - ~400 FFN units

L17 or Extended L15: STACK0 Lookup
  - Third memory read: memory[new_sp]
  - Depends on SP = BP + 16 (computed in L16)
  - Options:
    A. Add L17 layer for third lookup (increases depth)
    B. Multi-step LEV (break into 2 VM steps)
    C. 12-head L15 (heads 8-11 for stack0_val)
```

**Recommended Approach**: **Option C - 12-head L15**
- Heads 0-3: saved_bp from memory[BP]
- Heads 4-7: return_addr from memory[BP+8]
- Heads 8-11: stack0_val from memory[BP+16] (new_sp)
- All in one step, no new layers, maintains instruction semantics

**Total Units**: ~600-800 FFN units
- L15 extension: 8 heads × 24 dims = 192 units (parallel lookups)
- L16 routing: ~400 units (TEMP → OUTPUT at 4 markers)

**Dimensions Needed**:
- TEMP[0-31]: saved_bp (use existing TEMP region [480-511])
- TEMP[32-63]: return_addr (use existing TEMP region)
- Mark BP support in L15 (reuse existing marker infrastructure)

### 3.3 Implementation Steps

1. **Extend L15 for BP Memory Lookups** (vm_step.py ~line 5705)
   ```python
   # Add BP marker activation (similar to STACK0)
   # Head 0-3: Load saved_bp from memory[BP]
   for h in range(4):  # 4 bytes
       base = h * HD
       # Q: Fire when OP_LEV active at BP marker
       attn.W_q[base, BD.OP_LEV] = L
       attn.W_q[base, BD.MARK_BP] = L
       # K: Match memory addresses = BP value
       # (Use same address encoding as existing LI/LC)
       attn.W_k[base, BD.H1 + BP_I] = L  # BP byte 0
       attn.W_k[base, BD.H2 + BP_I] = L  # BP byte 1
       # ... similar for bytes 2-3
       # V: Copy memory bytes (from MEM sections in context)
       # O: Route to TEMP[0-31] for saved_bp

   # Heads 4-7: Load return_addr from memory[BP+8]
   for h in range(4, 8):
       base = h * HD
       # Q: Fire when OP_LEV active at BP marker
       attn.W_q[base, BD.OP_LEV] = L
       attn.W_q[base, BD.MARK_BP] = L
       # K: Match memory addresses = BP + 8
       # (Add offset logic similar to existing byte position offsets)
       # O: Route to TEMP[32-63] for return_addr

   # Heads 8-11: Load stack0_val from memory[BP+16]
   for h in range(8, 12):
       base = h * HD
       # Q: Fire when OP_LEV active at BP marker
       attn.W_q[base, BD.OP_LEV] = L
       attn.W_q[base, BD.MARK_BP] = L
       # K: Match memory addresses = BP + 16
       # O: Route to TEMP[64-95] for stack0_val (or directly to AX_CARRY)
   ```

2. **Add L16 Layer for Register Routing** (new layer after L15)
   ```python
   def _set_layer16_lev_routing(self, attn, ffn):
       """L16: Route LEV memory reads to output registers."""
       unit = 0
       S = self.swiglu_scale

       # Route saved_bp (TEMP[0-31]) to BP marker OUTPUT
       for k in range(32):  # 32 nibbles (4 bytes × 2 nibbles/byte × 4 split)
           ffn.W_up[unit, BD.OP_LEV] = S
           ffn.W_up[unit, BD.MARK_BP] = S
           ffn.b_up[unit] = -S * 1.5
           ffn.W_gate[unit, BD.TEMP + k] = 1.0  # Read from TEMP
           ffn.W_down[BD.OUTPUT_LO + (k % 16), unit] = 2.0 / S  # Write to OUTPUT
           # ... similar for HI nibbles
           unit += 1

       # Route return_addr (TEMP[32-63]) to PC marker OUTPUT
       # ... similar pattern

       # Compute SP = BP + 16 and route to SP marker OUTPUT
       # Reuse ADD pattern from existing ALU
       # ... arithmetic units
   ```

3. **Test LEV Neural**
   ```python
   # Test 1: Simple return
   int helper() { return 42; }
   int main() { return helper(); }

   # Test 2: Nested calls
   int c() { return 42; }
   int b() { return c(); }
   int a() { return b(); }
   int main() { return a(); }  # Should return 42

   # Test 3: With locals
   int func(int x) {
       int local = x * 2;
       return local;
   }
   int main() { return func(21); }  # Should return 42

   # Test 4: Recursive
   int fib(int n) {
       if (n <= 1) return n;
       return fib(n-1) + fib(n-2);
   }
   int main() { return fib(7); }  # Should return 13
   ```

4. **Remove LEV Handler**
   - Delete `_handler_lev` function (run_vm.py:1540-1560)
   - Remove `Opcode.LEV: self._handler_lev` from `_func_call_handlers` (line 227)

**Success Criteria**: All LEV tests pass with handler removed. LEV is 100% neural.

**Complexity Note**: This is the most complex phase due to architectural changes (extending L15 heads from 4 to 12, adding L16 layer). Requires careful testing of attention mechanics and state passing.

---

## Phase 4: Final Cleanup and Verification

**Goal**: Ensure 100% neural execution with zero handlers.

**Estimated Time**: 2-3 hours

### 4.1 Remove Residual Handler Infrastructure

1. **Remove ADJ Syscall Handler**
   - Delete `_syscall_adj` function (run_vm.py:1033-1049)
   - Remove `Opcode.ADJ: self._syscall_adj` from `_syscall_handlers` (line 173)

2. **Clean Up Handler Dispatch**
   - Verify `_func_call_handlers` is empty (lines 220-243)
   - Verify `_syscall_handlers` only has external tools (PUTCHAR, etc.) and C4 stdlib (MALC, etc.)

3. **Remove Tracking State**
   - Consider removing `_last_sp`, `_last_bp`, `_last_ax` if no longer needed
   - These were used by handlers for multi-byte correctness

### 4.2 Full Test Suite Validation

1. **Run Full 1096 Test Suite**
   ```bash
   # Test in pure mode
   python tests/test_suite_1000.py --pure-mode
   # Or via pytest
   pytest tests/test_vm.py -v --pure-mode
   ```

2. **Run Recursive Programs**
   ```bash
   # Fibonacci (tests JSR + ENT + LEV heavily)
   python tests/test_recursive.py
   ```

3. **Run Complex C4 Programs**
   ```bash
   # Sudoku solver, Mandelbrot, etc.
   python tests/test_full_programs.py
   ```

### 4.3 Purity Validation

Enable and verify purity report:
```python
runner = AutoregressiveVMRunner(pure_attention_memory=True)
result = runner.run(bytecode, data)
report = runner.get_pure_attention_report()

# Verify:
# - Zero blocked_vm_memory_ops
# - Zero function handler overrides
# - Only external_tool_ops (PUTCHAR, etc.)
print(report)
```

**Expected Output**:
```
{
    'blocked_vm_memory_ops': {},  # Empty - no VM ops blocked
    'external_tool_ops': {
        'PUTCHAR': 42,  # Only I/O boundaries
        'GETCHAR': 10,
        # ... other tool ops
    }
}
```

### 4.4 Performance Benchmarking

Measure performance impact of pure neural execution:

```python
import time

# Hybrid mode (with handlers)
runner_hybrid = AutoregressiveVMRunner(pure_attention_memory=False)
start = time.time()
for i in range(1000):
    runner_hybrid.run(bytecode, data)
hybrid_time = time.time() - start

# Pure mode (100% neural)
runner_pure = AutoregressiveVMRunner(pure_attention_memory=True)
start = time.time()
for i in range(1000):
    runner_pure.run(bytecode, data)
pure_time = time.time() - start

print(f"Hybrid: {hybrid_time:.2f}s ({1000/hybrid_time:.0f} runs/sec)")
print(f"Pure: {pure_time:.2f}s ({1000/pure_time:.0f} runs/sec)")
print(f"Slowdown: {pure_time/hybrid_time:.1%}")
```

**Target**: Pure mode within 15% of hybrid mode performance (speculative execution compensates for extra FFN units).

---

## Implementation Priority & Sequencing

### Recommended Sequence

**Week 1 (5-8 hours): Core Verification + ENT**
1. ✅ Phase 1.1: Verify ADJ neural (2h)
2. ✅ Phase 1.2-1.3: Verify binary pop + PSH (1h)
3. ✅ Phase 2: Complete ENT neural (3-5h)
4. 🎯 **Milestone**: 97% neural (only LEV handler remains)

**Week 2-3 (14-22 hours): LEV (Optional)**
5. ⏳ Phase 3: Complete LEV neural (14-22h)
6. 🎯 **Milestone**: 100% neural - no VM operation handlers

**Week 3-4 (2-3 hours): Finalization**
7. ✅ Phase 4: Cleanup + validation (2-3h)
8. 🎯 **Final Milestone**: 100% pure autoregressive VM

### Minimal Path (Just ENT)

If LEV is deferred:
1. Verify ADJ (2h)
2. Complete ENT (3-5h)
3. Cleanup (1h)
4. **Total**: 6-8 hours to reach 97% neural

**Justification for Deferring LEV**:
- LEV executes rarely (only at function returns)
- ENT executes at every function entry (more common)
- LEV requires significant architectural changes (L15 extension, new layer)
- 97% neural is a huge achievement, 100% is perfectionism

---

## Success Criteria

### Phase 1 Success
- ✅ All 1096 tests pass with ADJ handler removed
- ✅ Binary pop and PSH tests pass
- ✅ ADJ removed from `_syscall_handlers`

### Phase 2 Success (ENT)
- ✅ ENT tests pass (simple functions, nested calls, locals)
- ✅ ENT handler removed from `_func_call_handlers`
- ✅ Recursive programs work (fibonacci, factorial)
- 🎯 **97% neural execution achieved**

### Phase 3 Success (LEV) - Optional
- ✅ LEV tests pass (returns, nested returns, recursive)
- ✅ LEV handler removed from `_func_call_handlers`
- ✅ L15 extended to 12 heads
- ✅ L16 layer added for register routing
- 🎯 **100% neural execution achieved**

### Phase 4 Success (Final)
- ✅ Zero VM operation handlers in `_syscall_handlers` (only I/O tools remain)
- ✅ Zero function handlers in `_func_call_handlers` (empty dict)
- ✅ All 1096 tests pass in pure mode
- ✅ Purity report shows zero blocked ops
- ✅ Performance within 15% of hybrid mode
- 🎯 **100% pure autoregressive VM - Mission Complete**

---

## Files to Modify

### neural_vm/vm_step.py
- L7 attention: Add ENT head 2 (~line 4440)
- L8 FFN: Add ENT lo nibble + borrow (~line 4695)
- L9 FFN: Add ENT hi nibbles (~line 4850)
- L15 attention: Extend to 12 heads for LEV (~line 5705)
- L16: Add new layer for LEV register routing (~line 5900)

### neural_vm/run_vm.py
- Remove `_syscall_adj` handler (lines 1033-1049)
- Remove `Opcode.ADJ` from `_syscall_handlers` (line 173)
- Remove `_handler_ent` (lines 1504-1538)
- Remove `Opcode.ENT` from `_func_call_handlers` (line 226)
- Remove `_handler_lev` (lines 1540-1560) - Phase 3 only
- Remove `Opcode.LEV` from `_func_call_handlers` (line 227) - Phase 3 only

### tests/ (New test files)
- `tests/test_adj_neural.py` - ADJ verification tests
- `tests/test_ent_neural.py` - ENT neural tests (may already exist as `test_jsr_neural.py` covers similar)
- `tests/test_lev_neural.py` - LEV neural tests
- `tests/test_pure_mode.py` - Comprehensive pure mode validation

### Documentation
- Update `HANDLER_REMOVAL_FINAL.md` with final status
- Create `100_PERCENT_NEURAL.md` celebration document
- Update `README.md` with pure mode instructions

---

## Risk Mitigation

### Risk 1: Neural Implementation Bugs
**Probability**: Medium
**Impact**: High (tests fail)
**Mitigation**:
- Incremental testing (verify each phase before proceeding)
- Keep handlers as fallback during development
- Extensive test coverage before removal

### Risk 2: Performance Regression
**Probability**: Low-Medium
**Impact**: Medium (slower execution)
**Mitigation**:
- Extra FFN units increase forward pass time (~10-15%)
- Speculative execution (DraftVM) compensates
- Compact MoE reduces overhead (opcode-specific experts)
- Target: <15% slowdown acceptable

### Risk 3: L15 Extension Complexity
**Probability**: High (LEV phase)
**Impact**: High (architectural changes)
**Mitigation**:
- Defer LEV to Phase 3 (optional)
- Prototype L15 extension separately
- Extensive testing before integrating

### Risk 4: Dimension Exhaustion
**Probability**: Low
**Impact**: High
**Mitigation**:
- TEMP region [480-511] has 32 dims available
- Only need ~32 dims for LEV (saved_bp + return_addr)
- ENT uses existing dims (no new allocation)

---

## Open Questions

1. **ADJ Handler**: Should we block it immediately or wait for Phase 1 testing?
   - **Recommendation**: Test first (Phase 1.1), then remove.

2. **ENT vs LEV Priority**: Should we do ENT first or both together?
   - **Recommendation**: ENT first (simpler, higher impact), LEV optional.

3. **L15 12-Head Extension**: Is 12 heads acceptable or should we use multi-step LEV?
   - **Recommendation**: 12 heads (maintains instruction semantics, single step).

4. **Performance Target**: How much slowdown is acceptable for pure mode?
   - **Recommendation**: <15% slowdown (speculative execution compensates).

5. **Testing Strategy**: Run full 1096 suite after each phase or at end?
   - **Recommendation**: After each phase (catch regressions early).

---

## Next Immediate Actions

**If Starting Today** (in priority order):

1. ✅ **Create `tests/test_adj_neural.py`** - 30 min
2. ✅ **Run ADJ tests** - 30 min
3. ✅ **Verify ADJ works neurally** - 1 hour
4. ✅ **Remove ADJ handler** - 15 min
5. ✅ **Run full test suite** - 30 min
6. 🎯 **Checkpoint**: ADJ verified and removed (~3 hours total)

**Then Proceed to ENT**:
7. ✅ **Implement ENT L7/L8/L9 extensions** - 2-3 hours
8. ✅ **Test ENT neural** - 1 hour
9. ✅ **Remove ENT handler** - 15 min
10. 🎯 **Checkpoint**: 97% neural achieved (~6-8 hours total)

**Optional LEV**:
11. ⏳ **Design L15 12-head extension** - 3-4 hours
12. ⏳ **Implement L16 routing layer** - 4-6 hours
13. ⏳ **Test LEV neural** - 4-6 hours
14. ⏳ **Remove LEV handler** - 15 min
15. 🎯 **Final Goal**: 100% neural (~20-30 hours total)

---

## Conclusion

The path to 100% neural execution is clear:

**Short-term (Week 1)**: Verify ADJ + Complete ENT → **97% neural** (6-8 hours)
**Long-term (Week 2-3)**: Complete LEV → **100% neural** (20-30 hours total)

**Current Status**: 95% neural (JSR ✅, ENT 80%, LEV 10%)
**After Phase 1-2**: 97% neural (JSR ✅, ENT ✅, LEV ⏳)
**After Phase 3**: 100% neural (JSR ✅, ENT ✅, LEV ✅)

The C4 Transformer VM will be the **first fully autoregressive virtual machine** with zero special handling for VM operations or memory, executing entirely through learned transformer weights.

**Status**: Plan Complete - Ready for Implementation

---

**Date**: 2026-04-09
**Author**: Claude Sonnet 4.5
**Estimated Total Effort**: 6-8 hours (97% neural) or 20-30 hours (100% neural)
**Recommendation**: Start with Phase 1-2 (ADJ + ENT), defer LEV based on cost/benefit

🎯 **Next Action**: Create `tests/test_adj_neural.py` and begin Phase 1.1
