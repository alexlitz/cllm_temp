# How to add a tool handler

A short, opinionated recipe for extending the tool-use I/O subsystem with a
new tool call. The extensible handler API lets you add tools (file I/O,
RNG, timers, MCP shims, ...) **without touching `ToolUseVM` core code**:
register a callback, dispatch from a new `ToolCallType`, done.

For protocol-level background (the C-runtime `TOOL_CALL:<type>:<id>:{...}`
wire format, the `[TOOLUSE]` token, LLM integration) see
[`TOOLUSE.md`](TOOLUSE.md). This guide is the contributor-facing recipe.

## The four pieces

| Piece                | Lives in                                | Role                                                   |
|----------------------|-----------------------------------------|--------------------------------------------------------|
| `ToolCallType`       | `tools/tooluse_io.py`                   | Enum tag identifying the operation                     |
| `ToolCall`           | `tools/tooluse_io.py`, `neural_vm/run_vm.py` | Request: `(call_type, call_id, params)`           |
| `ToolResponse`       | `tools/tooluse_io.py`, `neural_vm/run_vm.py` | Reply: `(call_id, success, result, error?)`       |
| Handler callback     | `ToolUseIOHandler` (or your own class)  | `def _handle_X(self, call: ToolCall) -> ToolResponse`  |

The VM (`ToolUseVM` in `tools/tooluse_io.py`) emits a `ToolCall` when an
I/O opcode fires, pauses execution, and resumes after the handler returns
a `ToolResponse`. Whether the handler is `ToolUseIOHandler` (in-process
Python), the C runtime's wire protocol, or a custom MCP shim is up to you
— `ToolUseVM.__init__` takes any object that exposes
`handle(call: ToolCall) -> ToolResponse`.

## 1. Add a new `ToolCallType` enum value

Append to `ToolCallType` in `tools/tooluse_io.py`. Pick a stable string
value — it appears in `to_dict()` / `to_json()` and on the wire when
talking to LLMs or external processes.

```python
class ToolCallType(Enum):
    # ... existing entries ...
    RNG = "rng"  # deterministic / mock RNG
```

Keep the value snake_case to match `file_open`, `user_input`, etc.

## 2. Implement a handler callback

Two paths, pick one:

**Path A — extend `ToolUseIOHandler`.** Add a `_handle_rng(self, call)`
method and register it in the `handlers` dict of `handle()`:

```python
def _handle_rng(self, call: ToolCall) -> ToolResponse:
    n = call.params.get("n", 1)
    seed = call.params.get("seed", 0)
    # deterministic mock: returns seed, seed+1, seed+2, ...
    values = [(seed + i) & 0xFFFFFFFF for i in range(n)]
    return ToolResponse(call.call_id, True, result=values)
```

Path A is the right choice when the new tool is a first-class part of
the standard handler (file I/O, stdout, malloc, ...).

**Path B — register a callback without subclassing.** Construct any
object with a `handle(call) -> ToolResponse` method and pass it to
`ToolUseVM`. This is the extensible escape hatch: a new tool can be
added in user code with **zero edits to `ToolUseIOHandler`**.

```python
class RNGHandler:
    """Standalone handler that dispatches RNG and delegates the rest."""

    def __init__(self, inner: ToolUseIOHandler, seed: int = 0):
        self.inner = inner
        self.seed = seed
        self.counter = 0

    def handle(self, call: ToolCall) -> ToolResponse:
        if call.call_type == ToolCallType.RNG:
            n = call.params.get("n", 1)
            values = [(self.seed + self.counter + i) & 0xFFFFFFFF
                      for i in range(n)]
            self.counter += n
            return ToolResponse(call.call_id, True, result=values)
        # delegate everything else
        return self.inner.handle(call)
```

Path B is preferred when:

- The new tool is a one-off (test fixture, demo, mock).
- The new tool needs state that doesn't belong on `ToolUseIOHandler`
  (e.g., an open MCP socket).
- You want byte-identical core behavior for all *other* tool types.

## 3. Register with `ToolUseVM(tool_handler=handler)`

`ToolUseVM.__init__` takes the handler as the only required arg:

```python
inner = ToolUseIOHandler(output_callback=lambda s: print(s, end=""))
handler = RNGHandler(inner, seed=42)

vm = ToolUseVM(io_handler=handler)
vm.load(bytecode, data)
vm.run()
```

The neural runner (`AutoregressiveVMRunner.run` in
`neural_vm/run_vm.py`) uses the same protocol but takes a **callable**
instead of an object:

```python
runner.run(
    bytecode, data=data,
    tool_handler=lambda call: handler.handle(call),
)
```

Both call paths use the same `ToolCall` / `ToolResponse` dataclasses
(they are defined in both `tools/tooluse_io.py` and
`neural_vm/run_vm.py` for import-decoupling; the shapes match by
contract).

## 4. End-to-end example: a mock deterministic RNG tool

Wire a brand-new `RNG` tool through the system without modifying the
core VM. The full source is in
[`tests/test_tool_use_io.py::test_extensible_handler_demo`](../tests/test_tool_use_io.py).

```python
from tools.tooluse_io import (
    ToolCall, ToolCallType, ToolResponse,
    ToolUseIOHandler, ToolUseVM,
)

# Step 1: add RNG to ToolCallType (one line in tooluse_io.py)
# (in the test we monkey-patch the enum via a string fallback;
#  in real code you commit the enum entry.)

# Step 2: build a wrapper handler that owns RNG and delegates the rest
class RNGHandler:
    def __init__(self, inner, seed=0):
        self.inner = inner
        self.seed = seed
        self.counter = 0
        self.call_history = []

    def handle(self, call):
        self.call_history.append(call)
        if call.call_type == ToolCallType.RNG:
            n = call.params.get("n", 1)
            values = [(self.seed + self.counter + i) & 0xFFFFFFFF
                      for i in range(n)]
            self.counter += n
            return ToolResponse(call.call_id, True, result=values)
        return self.inner.handle(call)

# Step 3: wire it up
inner = ToolUseIOHandler()
handler = RNGHandler(inner, seed=100)
vm = ToolUseVM(io_handler=handler)

# Step 4: synthesize a tool call directly (the VM emits these from I/O
# opcodes; here we drive the protocol by hand to demonstrate it works
# end-to-end without touching ToolUseVM):
call = ToolCall(ToolCallType.RNG, call_id=1, params={"n": 3, "seed": 0})
resp = handler.handle(call)
assert resp.success
assert resp.result == [100, 101, 102]
```

Determinism matters: a mock RNG that returns `seed, seed+1, ...` makes
golden-file comparison trivial. Real RNGs belong behind a seed knob too
— the C4 corpus is regression-tested by byte identity (see
[`HOW_TO_ADD_A_CORRECTIVE_OP.md`](HOW_TO_ADD_A_CORRECTIVE_OP.md) §6 for
the byte-identity gate philosophy).

## 5. (Optional) emit the new tool from an opcode

If your tool should fire from a bytecode-level opcode (not just be
invoked programmatically by the host), wire the opcode → `ToolCall`
emission in `ToolUseVM.step()` (`tools/tooluse_io.py`). The pattern is:

```python
elif op == ExtendedOpcode.RNG:
    n = self.memory.get(self.sp, 0); self.sp += 8
    self.pending_call = ToolCall(
        ToolCallType.RNG,
        self._next_call_id(),
        {"n": n, "seed": 0},
    )
    return self.pending_call
```

and the matching branch in `provide_response()` to write the result
back to `self.ax` (or to memory via the params).

If your tool is host-driven (e.g., the runner injects an RNG draw
between steps), you can skip this step entirely — the handler API does
not require an opcode.

## 6. Anti-patterns

- **Do not edit the `handlers` dict in `ToolUseIOHandler.handle()`
  from a subclass** without overriding `handle()` itself — the dict is
  rebuilt every call. Either override `handle()`, or use Path B.
- **Do not raise from the handler.** Return
  `ToolResponse(call_id, success=False, error=str(e))` instead — the
  VM checks `response.success` and writes -1 / 0xFFFFFFFF to `ax`.
- **Do not mutate `call.params` in place.** Treat the call as
  immutable; copy if you need to transform.
- **Do not skip `call_id`.** The neural runner matches responses to
  calls by id; mismatched ids raise `ValueError` in
  `ToolUseVM.provide_response()`.

## 7. References

- [`tools/tooluse_io.py`](../tools/tooluse_io.py) — `ToolUseVM`,
  `ToolUseIOHandler`, `ToolCallType`, `ToolCall`, `ToolResponse`,
  `FileHandle`.
- [`neural_vm/run_vm.py`](../neural_vm/run_vm.py) — neural-VM-side
  `ToolCall` / `ToolResponse` and `_TOOL_CALL_OPS`; syscall handlers
  `_syscall_open` / `_syscall_read` / `_syscall_clos` / `_syscall_prtf`
  show the in-runner dispatch pattern.
- [`tests/test_tool_use_io.py`](../tests/test_tool_use_io.py) — protocol
  tests, including `test_extensible_handler_demo` which proves a new
  tool can be added without touching core VM code.
- [`TOOLUSE.md`](TOOLUSE.md) — protocol-level reference (wire format,
  `[TOOLUSE]` token, C-runtime integration).
