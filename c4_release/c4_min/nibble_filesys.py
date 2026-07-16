"""File operations for the nibble VM — OPEN / READ / CLOS via the TOOL_CALL
token protocol + a stub runner (BLOG_SPEC §File Operations / §Tool Use Mode).

Why a tool call
===============
Everything else in this VM is computed *neurally* — the ALU ops, the calling
convention, even ``malloc``/``memset`` are just bytecode that runs through the
transformer's FFN/attention (§Tool Use Mode: "runtime library functions ... are
not tool calls"). File operations are the exception. Per BLOG_SPEC §693-695:

    "File operations pretty intrinsically require a tool call ... in addition to
     implementing these instructions via tool-calling I implemented support for
     stdin reading via reading user messages"

and §849-851 (§Tool Use Mode):

    "In tool calling mode, I/O opcodes (PRTF, OPEN, READ, CLOS) emit a TOOL_CALL
     token at the end of the VM step, which the external runner intercepts to
     perform the actual I/O — reading files, formatting output, etc. — before
     resuming execution."

So an OPEN/READ/CLOS opcode does NOT compute a value inside the transformer.
Instead, at the end of its VM step it emits a **TOOL_CALL token** carrying the
syscall name + its arguments (marshalled from the register file / memory). An
external **runner** intercepts that token, performs the real I/O against a
filesystem (here a stubbed in-memory one), and feeds the integer result **back
into AX** (a TOOL_RESPONSE); for READ it also writes the read bytes into VM
memory. Execution then resumes at the next step.

The runner protocol (the token stream contract)
================================================
This is the interface an external runner (an LLM harness, an MCP server, a
python script) must implement. It is the same wire format as the existing C4
runtime (``docs/TOOLUSE.md``):

    program -> runner :   TOOL_CALL:<type>:<id>:{<params_json>}
    runner  -> program:   TOOL_RESPONSE:<id>:<result>

``type`` is ``open`` / ``read`` / ``close``; ``id`` is a monotonically
incrementing call id; ``params_json`` is a JSON object whose fields marshal the
opcode's operands (see ``ToolCall.params`` below). ``result`` is the single
integer the opcode's AX receives (the fd, the byte count, or 0). READ also
carries the read bytes so the runner can lay them into the destination buffer;
those go in the ``TOOL_RESPONSE`` payload as a ``bytes`` field and are written
to VM memory by ``FileRunner.apply``.

The c4 syscall semantics we honour (matching the C4 / POSIX subset):

    OPEN(name_ptr, flags) -> fd            AX = open(path) ; -1 on failure
    READ(fd, buf_ptr, n)  -> n_read        reads up to n bytes into *buf ; AX=n_read
    CLOS(fd)              -> 0              AX = 0 (close always succeeds here)

Argument marshalling (the c4 calling convention)
------------------------------------------------
c4 pushes syscall arguments left-to-right, so at the syscall the stack (top ->
down) holds the *last* argument first. The full-VM lowers e.g. ``open(name,O)``
to ``PSH name ; PSH O ; OPEN`` and reads the two args back off the stack. In
this minimal, single-AX slice we keep it explicit and unambiguous: the opcode's
operands are taken from **AX** (the last-pushed / primary arg) and the immediate
+ the stack, exactly as ``blogspec_run._apply_op`` threads them (see
``marshal_*`` below). The important invariant the proof rests on is that the
runner is a pure function of ``(syscall, args, filesystem)`` and its result is
the ONLY thing that re-enters the VM — the transformer never "sees" the file.

Neural READ pathway (§File Operations / §Printing and Reading Input)
--------------------------------------------------------------------
The blog also gives READ a *native* pathway: like ``getchar``/stdin, READ can be
satisfied from the **input KV** — bytes injected into the token stream between
``USER_INPUT_START``/``END`` markers and read out by attention, with no runner
intervention (§849-851: "AX = read(fd,buf,n) via input KV"). We model that here
with ``InputKVStream`` (fd 0 == stdin), so a READ on the stdin fd pulls its
bytes from the injected input buffer rather than the file runner — the same
result byte-for-byte, but on the 100%-native path. Regular file fds go through
the tool-call runner. This is exactly the blog's two-mode story: file reads via
tool call, stdin reads via input KV.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import isa


# ===========================================================================
# The wire protocol: TOOL_CALL / TOOL_RESPONSE (docs/TOOLUSE.md format).
# ===========================================================================
FILE_OPCODES = frozenset({isa.OPEN, isa.READ, isa.CLOS, isa.PRTF})

# opcode -> tool-call ``type`` string (the syscall name on the wire).
TOOL_TYPE = {isa.OPEN: "open", isa.READ: "read", isa.CLOS: "close",
             isa.PRTF: "printf"}

STDIN_FD = 0   # the neural-read (input-KV) fd — READ(0,...) reads stdin.
STDOUT_FD = 1  # PRTF writes formatted output to stdout (captured by the runner).


@dataclass
class ToolCall:
    """One TOOL_CALL emitted at a VM step end. Wire form:

        TOOL_CALL:<type>:<id>:{<params_json>}
    """
    id: int
    type: str                       # "open" | "read" | "close"
    params: Dict[str, object]       # the marshalled syscall args (JSON-able)

    def to_token(self) -> str:
        """The exact TOOL_CALL token string (the emitted token payload)."""
        return f"TOOL_CALL:{self.type}:{self.id}:{json.dumps(self.params, separators=(',', ':'))}"

    @staticmethod
    def parse(token: str) -> "ToolCall":
        """Inverse of ``to_token`` — a runner uses this to read the emitted call."""
        assert token.startswith("TOOL_CALL:"), token
        _, typ, cid, params = token.split(":", 3)
        return ToolCall(id=int(cid), type=typ, params=json.loads(params))


@dataclass
class ToolResponse:
    """The runner's reply that re-enters the VM. Wire form:

        TOOL_RESPONSE:<id>:<result>

    ``result`` is the single integer written into AX (fd / n_read / 0). ``data``
    carries READ's bytes (base is transport-only; ``FileRunner.apply`` lays them
    into VM memory). The AX write is the only VM-visible effect of ``result``.
    """
    id: int
    result: int
    data: bytes = b""

    def to_token(self) -> str:
        # bytes are hex-encoded so the response stays a plain ascii token.
        payload = str(self.result)
        if self.data:
            payload += ":" + self.data.hex()
        return f"TOOL_RESPONSE:{self.id}:{payload}"

    @staticmethod
    def parse(token: str) -> "ToolResponse":
        assert token.startswith("TOOL_RESPONSE:"), token
        _, cid, rest = token.split(":", 2)
        parts = rest.split(":", 1)
        result = int(parts[0])
        data = bytes.fromhex(parts[1]) if len(parts) > 1 and parts[1] else b""
        return ToolResponse(id=int(cid), result=result, data=data)


# ===========================================================================
# Argument marshalling — opcode operands -> the tool-call params JSON.
#
# In the minimal single-AX slice the operands are threaded exactly as the rest
# of ``_apply_op`` threads them: AX is the primary/last-pushed argument, the
# stack (``pop``) supplies earlier args, the immediate supplies static ones.
#
#   OPEN : name_ptr = pop()  (the C string address), flags = imm
#   READ : n = AX (count),   buf_ptr = pop(),   fd = pop()
#   CLOS : fd = AX
#
# The runner never needs the raw pointers except to write READ's bytes back, so
# ``buf_ptr`` is carried in params for that single purpose. Filenames are read
# out of VM memory (the null-terminated C string at ``name_ptr``) by
# ``marshal_open`` so the runner receives a concrete ``path`` string.
# ===========================================================================
def read_cstring(mem, addr: int, limit: int = 4096) -> str:
    """Read a NUL-terminated C string out of byte-addressed VM memory."""
    out = bytearray()
    for i in range(limit):
        b = mem.load_int(addr + i, 1) & 0xFF
        if b == 0:
            break
        out.append(b)
    return out.decode("latin-1")


def _format_printf(fmt: str, args: List[int], strings: Dict[str, str]) -> str:
    """The c4 printf subset: %d %c %s %x %% + literal bytes (BLOG_SPEC / C4).

    ``args`` are the integer varargs (in call order); a ``%s`` consumes the arg as
    a data-segment pointer whose C-string was pre-resolved into ``strings`` (keyed
    by the pointer value as a string) by the marshaller — the runner never touches
    VM memory. Returns the exact formatted byte string PRTF appends to stdout."""
    out = []
    ai = 0
    i = 0
    n = len(fmt)
    while i < n:
        c = fmt[i]
        if c != "%":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= n:
            out.append("%")
            break
        spec = fmt[i]
        i += 1
        if spec == "%":
            out.append("%")
        elif spec == "d":
            out.append(str(_sign32(args[ai]))); ai += 1
        elif spec == "u":
            out.append(str(args[ai] & 0xFFFFFFFF)); ai += 1
        elif spec == "x":
            out.append(format(args[ai] & 0xFFFFFFFF, "x")); ai += 1
        elif spec == "c":
            out.append(chr(args[ai] & 0xFF)); ai += 1
        elif spec == "s":
            out.append(strings.get(str(args[ai]), "")); ai += 1
        else:                                    # unknown -> emit verbatim (%<c>)
            out.append("%" + spec)
    return "".join(out)


def _sign32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


# ===========================================================================
# The filesystem stub + the runner that performs the I/O.
# ===========================================================================
class StubFilesystem:
    """An in-memory filesystem: ``name -> bytes``. Stands in for the real disk
    the external runner would touch — the proof only needs the runner to be a
    pure function of (syscall, args, files)."""

    def __init__(self, files: Optional[Dict[str, bytes]] = None):
        self.files: Dict[str, bytes] = dict(files or {})

    def exists(self, name: str) -> bool:
        return name in self.files

    def contents(self, name: str) -> bytes:
        return self.files[name]


class InputKVStream:
    """The neural-read (input-KV) byte buffer — the bytes injected between
    ``USER_INPUT_START``/``END`` that stdin reads attend to (§Printing and
    Reading Input). A READ on ``STDIN_FD`` is satisfied from here, with no file
    runner and no real disk — the 100%-native pathway."""

    def __init__(self, data: bytes = b""):
        self.data = bytes(data)
        self.pos = 0

    def read(self, n: int) -> bytes:
        chunk = self.data[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk


@dataclass
class _OpenFile:
    name: str
    data: bytes
    pos: int = 0


class FileRunner:
    """The external runner: intercepts TOOL_CALL tokens, performs the syscall
    against the stub filesystem (or the input-KV stream for stdin), and produces
    the TOOL_RESPONSE that re-enters the VM.

    This is the reference implementation of the runner protocol documented at the
    top of this module. A real runner (LLM harness / MCP server) implements the
    same ``handle(ToolCall) -> ToolResponse`` contract against a real disk.
    """

    def __init__(self, fs: Optional[StubFilesystem] = None,
                 stdin: Optional[InputKVStream] = None):
        self.fs = fs or StubFilesystem()
        self.stdin = stdin or InputKVStream()
        self._fds: Dict[int, _OpenFile] = {}
        self._next_fd = 3            # 0/1/2 reserved (stdin/stdout/stderr)
        self.log: List[Tuple[str, str]] = []   # (call_token, response_token)
        self.stdout = bytearray()   # PRTF appends its formatted output here.

    # -- the protocol handler ------------------------------------------------
    def handle(self, call: ToolCall) -> ToolResponse:
        """Perform ``call`` and return the response (result + optional bytes).

        A pure function of (syscall, args, filesystem/stdin state). The ONLY
        thing that re-enters the VM is ``ToolResponse.result`` (into AX) and,
        for READ, ``ToolResponse.data`` (into VM memory by ``apply``)."""
        if call.type == "open":
            return self._open(call)
        if call.type == "read":
            return self._read(call)
        if call.type == "close":
            return self._close(call)
        if call.type == "printf":
            return self._printf(call)
        raise NotImplementedError(f"runner: unknown tool call {call.type!r}")

    def _open(self, call: ToolCall) -> ToolResponse:
        path = call.params["path"]
        if not self.fs.exists(path):
            return ToolResponse(call.id, result=-1)          # open failure -> -1
        fd = self._next_fd
        self._next_fd += 1
        self._fds[fd] = _OpenFile(name=path, data=self.fs.contents(path))
        return ToolResponse(call.id, result=fd)

    def _read(self, call: ToolCall) -> ToolResponse:
        fd = call.params["fd"]
        n = call.params["n"]
        if fd == STDIN_FD:                                   # neural-read: input KV
            chunk = self.stdin.read(n)
        else:
            f = self._fds.get(fd)
            if f is None:
                return ToolResponse(call.id, result=-1)      # bad fd -> -1
            chunk = f.data[f.pos:f.pos + n]
            f.pos += len(chunk)
        return ToolResponse(call.id, result=len(chunk), data=chunk)

    def _close(self, call: ToolCall) -> ToolResponse:
        fd = call.params["fd"]
        self._fds.pop(fd, None)
        return ToolResponse(call.id, result=0)               # close -> 0

    def _printf(self, call: ToolCall) -> ToolResponse:
        """PRTF(fmt_ptr, args...) -> n_written.  The runner formats the C string
        at ``fmt`` with the integer ``args`` (the c4 printf subset: %d / %c / %s /
        %x / %% + literal bytes) and appends the bytes to ``stdout``.  AX receives
        the number of bytes written — the same result the C runtime returns.  The
        formatted text is the ONLY external effect; nothing re-enters VM memory."""
        text = _format_printf(call.params["fmt"], call.params.get("args", []),
                              call.params.get("strings", {}))
        out = text.encode("latin-1")
        self.stdout.extend(out)
        return ToolResponse(call.id, result=len(out))

    # -- applying a response back into VM state ------------------------------
    def apply(self, call: ToolCall, resp: ToolResponse, mem) -> int:
        """Lay ``resp`` back into the VM: READ writes its ``data`` bytes into the
        destination buffer (``buf_ptr`` from the call params); every op returns
        the integer that the caller writes into AX. This is the boundary re-entry
        (§Tool Use Mode: "feeds the result back into AX; execution resumes")."""
        if call.type == "read" and resp.data:
            buf = call.params["buf"]
            for i, byte in enumerate(resp.data):
                mem.store_int(buf + i, byte, 1)              # byte -> VM memory
        self.log.append((call.to_token(), resp.to_token()))
        return resp.result


# ===========================================================================
# The dispatch entry point — called from ``blogspec_run._apply_op``.
#
# A file opcode's whole effect is: marshal the args -> emit the TOOL_CALL ->
# runner performs it -> write the result into AX (+ READ's bytes into mem). This
# is the ONE place the VM crosses the boundary; the transformer computes nothing.
# ===========================================================================
@dataclass
class FileOpState:
    """Per-run file-op context threaded through ``_apply_op`` (like ``mem``).

    ``runner`` performs the I/O; ``calls`` accumulates the emitted TOOL_CALLs (the
    proof reads this to confirm the exact token stream)."""
    runner: FileRunner = field(default_factory=FileRunner)
    calls: List[ToolCall] = field(default_factory=list)
    _next_id: int = 1

    def _new_id(self) -> int:
        cid = self._next_id
        self._next_id += 1
        return cid


def dispatch_file_op(op: int, ax: int, imm: int, pop, mem,
                     fio: FileOpState) -> int:
    """Execute a file opcode via the TOOL_CALL protocol; return the new AX.

    ``pop`` is ``_apply_op``'s stack pop (advances SP); ``mem`` the byte-addressed
    VM memory; ``fio`` the run's file-op context. Emits the TOOL_CALL, has the
    runner perform the syscall, applies the response (READ bytes -> mem), and
    returns the integer AX receives.
    """
    if op == isa.OPEN:
        # OPEN(name_ptr=pop, flags=imm) -> fd.  Read the C string filename out of
        # VM memory so the runner gets a concrete path.
        name_ptr = pop()
        path = read_cstring(mem, name_ptr)
        call = ToolCall(fio._new_id(), TOOL_TYPE[op],
                        {"path": path, "name_ptr": name_ptr, "flags": imm})
    elif op == isa.READ:
        # READ(fd=pop, buf_ptr=pop, n=AX) -> n_read.  fd==0 -> input-KV (stdin).
        n = ax
        buf_ptr = pop()
        fd = pop()
        call = ToolCall(fio._new_id(), TOOL_TYPE[op],
                        {"fd": fd, "buf": buf_ptr, "n": n})
    elif op == isa.CLOS:
        # CLOS(fd=AX) -> 0.
        call = ToolCall(fio._new_id(), TOOL_TYPE[op], {"fd": ax})
    elif op == isa.PRTF:
        # PRTF(fmt_ptr=pop, args...=stack) -> n_written.  The c4 lowering pushes
        # the format pointer FIRST then each arg, so at the syscall the stack is
        # (top->down) last-arg ... first-arg, fmt_ptr.  In this single-AX slice we
        # keep it explicit: fmt = pop(), varargs are threaded via ``fio.args`` set
        # by the caller (the driver marshals them from the KV store log).
        fmt_ptr = pop()
        fmt = read_cstring(mem, fmt_ptr)
        args = list(getattr(fio, "pending_args", []) or [])
        strings = {str(a): read_cstring(mem, a) for a in args}
        call = ToolCall(fio._new_id(), TOOL_TYPE[op],
                        {"fmt": fmt, "fmt_ptr": fmt_ptr, "args": args,
                         "strings": strings})
    else:
        raise NotImplementedError(f"{isa.NAMES.get(op, op)} is not a file opcode")

    fio.calls.append(call)                    # record the emitted TOOL_CALL token
    resp = fio.runner.handle(call)            # runner performs the real I/O
    return fio.runner.apply(call, resp, mem)  # response re-enters -> new AX


# ===========================================================================
# PURE-FORWARD DRIVER integration.
#
# The pure-forward VM (``nibble_pure_forward_cached.run_pure_forward_cached``)
# does NOT keep a python ``mem`` object: its whole memory lives in the token
# stream as a KV **store log** (``store_log[frame_idx] = (addr, val)``) that the
# transformer attends to.  A load (LI/LC/pop) is the model attending to the store
# token whose binary address matches the query — latest-write-wins.
#
# A FILE op is the exception the blog carves out (§Tool Use Mode): it is NOT
# computed neurally.  So for the pure-forward driver the file op's whole effect
# is realised by the driver, exactly as on the hybrid VM, but backed by a
# ``store_log`` view instead of a ``DictMemStack``:
#
#   * ``StoreLogMem`` gives ``read_cstring`` / the runner the SAME
#     ``load_int``/``store_int`` interface, reading the seeded data segment (the
#     string literals the c4 loader places) + the KV store log, and RECORDING
#     every write (READ's bytes) as ``(addr, byte)`` byte-stores.
#   * ``dispatch_file_op_driver`` marshals the args off the KV stack (the same
#     PSH store log the model reads), runs the op via ``dispatch_file_op``, and
#     returns ``(new_ax, new_sp, byte_stores)``.  The driver then (a) overrides
#     the model's AX with ``new_ax``, (b) sets SP to ``new_sp`` (the popped
#     args), and (c) appends each READ byte-store as its OWN emitted frame so the
#     model can read those bytes back with LC — the bytes literally re-enter the
#     token stream as §Memory KV entries, byte-for-byte the file contents.
# ===========================================================================
class StoreLogMem:
    """A ``load_int``/``store_int`` view over the pure-forward driver's KV state.

    Reads resolve against (in priority order) the pending writes of THIS file op,
    the run's ``store_log`` (byte/word stores, latest-write-wins), and a seeded
    ``data_seg`` (the read-only data segment: string literals).  Writes are
    recorded per-byte into ``pending`` (the driver lays them into the store log as
    fresh KV entries so subsequent LC loads attend to them)."""

    def __init__(self, store_log: Dict[int, Tuple[int, int]],
                 data_seg: Optional[Dict[int, int]] = None):
        self.store_log = store_log
        self.data_seg = data_seg or {}
        self.pending: List[Tuple[int, int]] = []   # (addr, byte) writes to inject
        self._byte_cache: Dict[int, int] = {}      # addr -> latest byte written

    def _byte_at(self, addr: int) -> int:
        if addr in self._byte_cache:
            return self._byte_cache[addr] & 0xFF
        # scan the store log for the latest write covering this byte address.
        best_fi = None
        for fi, (a, v) in self.store_log.items():
            # each store writes a 4-byte word at ``a`` (or a single byte for SC /
            # a byte-store).  A byte at ``addr`` is covered if a <= addr < a+4.
            if a <= addr < a + 4 and (best_fi is None or fi > best_fi):
                best_fi = fi
        if best_fi is not None:
            a, v = self.store_log[best_fi]
            return (v >> (8 * (addr - a))) & 0xFF
        return self.data_seg.get(addr, 0) & 0xFF

    def load_int(self, addr: int, width: int = 4) -> int:
        val = 0
        for i in range(width):
            val |= self._byte_at(addr + i) << (8 * i)
        return val

    def store_int(self, addr: int, val: int, width: int = 4) -> None:
        for i in range(width):
            b = (val >> (8 * i)) & 0xFF
            self._byte_cache[addr + i] = b
            self.pending.append((addr + i, b))


def dispatch_file_op_driver(op: int, ax: int, imm: int, cur_sp: int,
                            store_log: Dict[int, Tuple[int, int]],
                            fio: FileOpState,
                            data_seg: Optional[Dict[int, int]] = None,
                            slot: int = 8) -> Tuple[int, int, List[Tuple[int, int]]]:
    """Execute a file opcode on the PURE-FORWARD driver's KV state.

    ``cur_sp`` is the pre-op stack pointer; ``store_log`` the KV store log the
    model attends to; ``data_seg`` the seeded read-only data segment (string
    literals).  Returns ``(new_ax, new_sp, byte_stores)``:

      new_ax      : the integer the runner returns (fd / n_read / 0 / n_written).
      new_sp      : SP after popping the op's stack args (the c4 ``ADJ`` is folded
                    in here: the popped arg slots are reclaimed).
      byte_stores : for READ, the ``(addr, byte)`` list to inject into the store
                    log as fresh KV frames (so LC reads the file bytes back).

    The marshalling matches ``dispatch_file_op`` (the same single-AX slice proven
    on the hybrid VM): a ``pop`` reads MEM[SP] off the KV log and advances SP by
    one slot.
    """
    mem = StoreLogMem(store_log, data_seg)
    sp = [cur_sp]

    def pop() -> int:
        v = mem.load_int(sp[0], 4)
        sp[0] += slot
        return v

    new_ax = dispatch_file_op(op, ax, imm, pop, mem, fio)
    return new_ax, sp[0], list(mem.pending)


__all__ = [
    "FILE_OPCODES", "TOOL_TYPE", "STDIN_FD", "STDOUT_FD",
    "ToolCall", "ToolResponse",
    "StubFilesystem", "InputKVStream", "FileRunner", "FileOpState",
    "read_cstring", "dispatch_file_op",
    "StoreLogMem", "dispatch_file_op_driver",
]
