# Neural Output Generation Architecture

**Goal**: Implement fully neural printf output generation in transformer weights

---

## Token Flow

### Normal Execution
```
REG_PC(5) + REG_AX(5) + REG_SP(5) + REG_BP(5) + STACK0(5) + MEM(9) + STEP_END(1)
```

### PRTF Execution (Conversational I/O Mode)
```
Step N: REG_PC(5) + ... + STEP_END(1)          [normal VM step]
Step N+1: THINKING_END(1)                       [exit thinking, PRTF detected]
Step N+2: byte_0                                [first output byte]
Step N+3: byte_1                                [second output byte]
...
Step N+k: byte_n                                [last output byte]
Step N+k+1: THINKING_START(1)                   [re-enter thinking]
Step N+k+2: REG_PC(5) + ... + STEP_END(1)      [next VM step, PC advanced]
```

---

## State Tracking via Lookback

The model determines its mode by looking back at recent tokens:

### Mode Detection (L2 Attention)
- **Normal Mode**: Default state, generate 35-token VM step
- **Output Mode Start**: Last token = THINKING_END → enter output mode
- **Output Mode Active**: Last token = byte (0-255) AND in output mode → continue
- **Output Mode End**: Format string complete → emit THINKING_START
- **Resume Normal**: Last token = THINKING_START → resume 35-token format

### Implementation (L2 Attention Head)
```python
# Head 0: Detect last token type
# Q: CONST (always query)
# K: Token embeddings (THINKING_END, THINKING_START, bytes)
# V: Copy flags (IS_THINKING_END, IS_THINKING_START, IS_BYTE)
# O: Write to state dimensions

# Lookback distance = 1, so attend to t-1 from position t
# Use causal mask + ALiBi to prefer most recent token
```

---

## Phase 1: Literal Strings (No Format Specifiers)

### Scope
Handle only `printf("literal text\n")` - no %d, %x, %c, %s

### Required Components

#### 1. Format String Pointer Extraction (L7 Attention)
**Goal**: Read format string pointer from stack at SP+0

```python
# When in output mode start (just after THINKING_END):
# Q: IO_IN_OUTPUT_MODE_START (active at current position)
# K: MARK_SP (attend to SP marker from previous step's context)
# V: Copy STACK0 bytes (4 bytes = 32-bit pointer)
# O: Write to FORMAT_PTR_LO, FORMAT_PTR_HI

# This extracts the format string pointer that PRTF pushed to stack
```

#### 2. Format String Position Counter (L8 FFN)
**Goal**: Track current position in format string (0, 1, 2, ...)

```python
# Initialize: IO_FORMAT_POS = 0 when entering output mode
# Increment: IO_FORMAT_POS += 1 after each byte emission
# Uses: nibble arithmetic (same pattern as PC increment)
```

#### 3. Format String Fetch (L9 Attention)
**Goal**: Read byte at current position from format string

```python
# Q: FORMAT_PTR + FORMAT_POS (compute address to fetch)
# K: ADDR_KEY (memory key matching)
# V: Byte value at that address
# O: Write to OUTPUT_BYTE (the byte to emit)

# Uses same memory attention as L5 code fetch
```

#### 4. Null Terminator Detection (L10 FFN)
**Goal**: Detect when format string is complete (byte = 0)

```python
# Detect: OUTPUT_BYTE == 0 (all nibbles zero)
# Set: IO_OUTPUT_COMPLETE flag
# Clear: IO_IN_OUTPUT_MODE flag
# Set: NEXT_THINKING_START (emit THINKING_START next)
```

#### 5. Output Byte Emission (L15 Output Head)
**Goal**: Route OUTPUT_BYTE to token generation

```python
# When IO_IN_OUTPUT_MODE AND NOT IO_OUTPUT_COMPLETE:
#   Output = OUTPUT_BYTE (emit byte from format string)
# When IO_OUTPUT_COMPLETE:
#   Output = THINKING_START (exit output mode)
# When last token was THINKING_START:
#   Output = REG_PC (resume normal 35-token format)
```

---

## Layer Allocation

| Layer | Component | Purpose |
|-------|-----------|---------|
| L2 | Lookback detection | Detect last token type (THINKING_END/START/byte) |
| L3 | State initialization | Set IO_IN_OUTPUT_MODE when entering |
| L7 | Format pointer extraction | Read format string pointer from STACK0 |
| L8 | Position counter | Track IO_FORMAT_POS, increment each step |
| L9 | Format string fetch | Attention to read byte at FORMAT_PTR + FORMAT_POS |
| L10 | Terminator detection | Detect byte == 0, set IO_OUTPUT_COMPLETE |
| L13 | Mode routing | Control flow between output/normal modes |
| L15 | Output emission | Emit OUTPUT_BYTE or THINKING_START |

---

## Dimension Allocations

Already allocated:
- `IO_IS_PRTF = 460` - PRTF opcode detected
- `IO_STATE = 462` - State machine flag
- `IO_FORMAT_POS = 464` - Position in format string

Need to add:
- `IO_IN_OUTPUT_MODE = 465` - Flag: currently emitting output
- `IO_OUTPUT_COMPLETE = 466` - Flag: format string done
- `FORMAT_PTR_LO = 467` - Format string pointer (lo 16 bits)
- `FORMAT_PTR_HI = 483` - Format string pointer (hi 16 bits)
- `OUTPUT_BYTE_LO = 499` - Output byte (lo nibble)
- `OUTPUT_BYTE_HI = 500` - Output byte (hi nibble)
- `LAST_WAS_THINKING_END = 501` - Lookback: prev token was THINKING_END
- `LAST_WAS_THINKING_START = 502` - Lookback: prev token was THINKING_START
- `LAST_WAS_BYTE = 503` - Lookback: prev token was byte (0-255)

---

## Implementation Plan

### Step 1: Lookback Detection (L2 Attention)
- Add head to detect previous token type
- Write to LAST_WAS_* dimensions

### Step 2: State Initialization (L3 FFN)
- When LAST_WAS_THINKING_END: set IO_IN_OUTPUT_MODE = 1

### Step 3: Format Pointer Extraction (L7 Attention)
- When IO_IN_OUTPUT_MODE just activated: read STACK0 → FORMAT_PTR

### Step 4: Position Counter (L8 FFN)
- Initialize FORMAT_POS = 0 when entering mode
- Increment FORMAT_POS after each byte emission

### Step 5: Format String Fetch (L9 Attention)
- Compute address = FORMAT_PTR + FORMAT_POS
- Attend to memory with this address
- Write byte to OUTPUT_BYTE

### Step 6: Terminator Detection (L10 FFN)
- Detect OUTPUT_BYTE == 0
- Set IO_OUTPUT_COMPLETE, clear IO_IN_OUTPUT_MODE

### Step 7: Output Routing (L15 Output Head)
- Route OUTPUT_BYTE when in output mode
- Route THINKING_START when output complete
- Resume normal routing after THINKING_START

### Step 8: Testing
- Test with simple printf("Hi\n")
- Test with printf("Hello, World!\n")
- Verify null terminator detection
- Verify THINKING_START emission
- Verify resume to normal mode

---

## Estimated Effort

| Task | Hours |
|------|-------|
| Lookback detection | 2-3 |
| State initialization | 1-2 |
| Format pointer extraction | 2-3 |
| Position counter | 2-3 |
| Format string fetch | 3-4 |
| Terminator detection | 2-3 |
| Output routing | 2-3 |
| Testing & debugging | 4-6 |
| **Phase 1 Total** | **18-27 hours** |

---

## Phase 2: Format Specifiers (Future)

After Phase 1 works, add:
- Format specifier detection (%d, %x, %c)
- Argument extraction from stack
- Integer-to-string conversion
- Format string state machine (literal vs specifier mode)

Estimated additional: 12-18 hours

---

## Next Action

Implement Step 1: Lookback detection in L2 attention
