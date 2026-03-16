# Tool-Use I/O Protocol

The C4 ONNX runtime supports a tool-use I/O protocol that allows programs to communicate with external handlers (LLMs, MCP servers, Python scripts, etc.) instead of using direct I/O.

## Quick Start

```bash
# Normal mode (default) - direct I/O
./onnx_runner_c4 6 '*' 7
# Output: 42

# Tool-use mode - generates tool calls
./onnx_runner_c4 -t 6 '*' 7
# Output: TOOL_CALL:result:1:{"value":42,"expr":"6 * 7"}

# Or use the [TOOLUSE] token (for LLM integration)
./onnx_runner_c4 '[TOOLUSE]' 6 '*' 7
```

## Enabling Tool-Use Mode

Tool-use mode is **disabled by default**. Enable it with:

| Method | Example |
|--------|---------|
| `-t` flag | `./onnx_runner_c4 -t 6 '*' 7` |
| `[TOOLUSE]` token | `./onnx_runner_c4 '[TOOLUSE]' 6 '*' 7` |

The `[TOOLUSE]` token can appear anywhere in the arguments and will be stripped before processing.

## Protocol Format

### Tool Calls (Program → Handler)

```
TOOL_CALL:<type>:<id>:{<params_json>}
```

| Field | Description |
|-------|-------------|
| `type` | Operation type (see below) |
| `id` | Unique call ID (incrementing integer) |
| `params_json` | JSON object with operation parameters |

### Tool Responses (Handler → Program)

```
TOOL_RESPONSE:<id>:<result>
```

| Field | Description |
|-------|-------------|
| `id` | Must match the call ID |
| `result` | Integer result value |

## Tool Call Types

### Output Operations

| Type | Params | Description |
|------|--------|-------------|
| `putchar` | `{"char": <ascii>}` | Output single character |
| `print` | `{"text": "<string>"}` | Output string (escaped) |
| `result` | `{"value": <int>, "expr": "<str>"}` | Computation result |

### Input Operations

| Type | Params | Response |
|------|--------|----------|
| `getchar` | `{}` | ASCII code or -1 for EOF |

### Control Operations

| Type | Params | Description |
|------|--------|-------------|
| `exit` | `{"code": <int>}` | Program exit |

## Example Session

**Program output:**
```
TOOL_CALL:print:1:{"text":"Enter a number: "}
TOOL_CALL:getchar:2:{}
```

**Handler response:**
```
TOOL_RESPONSE:2:53
```
(53 = ASCII '5')

**Program output:**
```
TOOL_CALL:result:3:{"value":25,"expr":"5 * 5"}
TOOL_CALL:exit:4:{"code":0}
```

## Python Handler

Use `tooluse_c_handler.py` to run programs with tool-use:

```bash
python3 tooluse_c_handler.py ./onnx_runner_c4 -t 6 '*' 7
```

Or programmatically:

```python
from tooluse_c_handler import CToolUseHandler

handler = CToolUseHandler(
    output_callback=lambda s: print(s, end=''),
    input_callback=input
)
handler.run_program('./onnx_runner_c4', ['-t', '6', '*', '7'])
```

## VM-Based Tool Use (Python)

For running C programs compiled to bytecode:

```python
from tooluse_io import ToolUseVM, ToolUseIOHandler
from src.compiler import compile_c

# Compile C source
bytecode, data = compile_c(source_code)

# Create handler
handler = ToolUseIOHandler(
    output_callback=lambda s: print(s, end=''),
    input_callback=lambda: input() + '\n'
)

# Run
vm = ToolUseVM(handler)
vm.load(bytecode, data)
vm.run()
```

## LLM Integration

The `[TOOLUSE]` token is designed for LLM integration. An LLM can include this token when it wants the program to generate structured tool calls instead of direct output:

**Without token (normal):**
```
User: Calculate 6 * 7
LLM: ./onnx_runner_c4 6 '*' 7
Output: 42
```

**With token (tool-use):**
```
User: Calculate 6 * 7 and return structured result
LLM: ./onnx_runner_c4 [TOOLUSE] 6 '*' 7
Output: TOOL_CALL:result:1:{"value":42,"expr":"6 * 7"}
```

The LLM can then parse the JSON and use the result in its response.

## Files

| File | Description |
|------|-------------|
| `onnx_runner_c4.c` | Main ONNX runtime with tool-use support |
| `tooluse_io.py` | Python VM with tool-use I/O |
| `tooluse_c_handler.py` | Python handler for C programs |
| `run_eliza_tooluse.py` | Eliza demo with tool-use |
