# BLOG_SPEC revision — the honest accounting of "native neural I/O"

Date: 2026-07-20 · Branch: `neural-io-position-wt` (off `unify-full-vm-no-lean-splits @ 49907f07`)

Companion to `BLOG_SPEC_REVISIONS.md` item 15 ("Native conversational I/O vs.
tool-call I/O"). This entry corrects and qualifies the **"100% native"** framing
around I/O (`docs/BLOG_SPEC.md` §695) with what the implementation actually does,
after wiring the position-signature substrate into the stdin read path.

## The claim (BLOG_SPEC §695, §704-717)

> "File operations pretty intrinsically require a tool call, however I wanted to
> make sure basic IO was possible with a **100% native** version of the
> transformer so ... I implemented support for **stdin reading** via reading user
> messages and **stdout writing** via system messages outside the thinking tag."

and the mechanism (§710): multiple attention heads share a fixed BOS key with
distinct ALiBi slopes; for a token at distance `d`, head `k` contributes
`exp(-m_k·d)`; the tuple `(exp(-m_1·d), …, exp(-m_K·d))` uniquely identifies
position `d`; "to retrieve the byte at position N, we construct a query that
matches the exponential signature of distance N."

## What was actually implemented before this change (the audit finding)

The **stdin read** was a Python buffer slice, NOT attention:

```python
class InputKVStream:            # nibble_filesys.py
    def read(self, n):
        chunk = self.data[self.pos:self.pos + n]   # a Python list slice
        self.pos += len(chunk)
        return chunk
```

No position-signature heads were baked, no `exp(-m_k·d)` tuple was constructed,
and no `model.forward` ran. So the "100% native" wording was **not honored** for
the stdin path as shipped — the byte located "by position N" was located by a
Python index. (The `nibble_io_position.py` substrate proving the mechanism
existed on the `nibble-io-position` branch but was **not wired** into the read
path — it was tested in isolation only.)

## What is genuinely neural NOW (this change)

`InputKVStream.read` (default `neural=True`) retrieves each returned byte through
the **real** `blogspec_model.Attn` softmax1 + ALiBi forward via
`nibble_io_position.IOPositionBuffer.read_run_neural`:

- the injected input bytes are laid out as a token stream — a BOS/marker token
  then one value token per byte (§706);
- every head shares the BOS key and carries a distinct ALiBi slope (§704), so the
  per-head `exp(-m_k·d)` decay IS the position signature (§710);
- the byte at buffer offset `o` sits at ALiBi distance `o+1` from the marker; the
  query at that distance wins under softmax1 and the value lane copies the byte
  out — i.e. the byte is **located and copied by attention**, not indexed in
  Python;
- out-of-range reads return 0 by softmax1 ZFOD (§491), giving correct short-read
  semantics.

Verified byte-for-byte identical to the reference slice and end-to-end through
the TOOL_CALL runner into VM memory (`test_neural_stdin_read.py`,
`test_nibble_io_position.py`; `Attn.forward` invocation confirmed by a spy).
The retrieval head is a tiny standalone `Attn` (dim = 257·n_heads), never the
dense full VM.

The stdin path is now the honest realization of §704-712: `getchar` / stdin /
argv reads (all of which are position-addressed buffer fetches, §851) go through
the position-signature attention.

## What is still Python (and why that is honest, not a violation)

- **PRTF / stdout formatting** and **file OPEN/READ/CLOS** go through the
  `FileRunner` — a Python tool-call handler. This is **correct by the spec's own
  framing**: §693-695 states file operations "pretty intrinsically require a tool
  call", and §1 explicitly considers tool calling "not part of the LLM itself."
  These are the tool-call I/O mode, not the native mode; they were never claimed
  to be neural.
- **stdout position tracking / PUTCHAR** (§717): the *output* buffer is addressed
  by the same position signature in reverse. The read-side substrate
  (`read_run_neural`) demonstrates the mechanism; a symmetric neural PUTCHAR
  emit-side was not built here (the output bytes are still assembled by the
  runner's `stdout` bytearray). This remains Python-driver.

## Recommended spec wording (qualify, don't overclaim)

State the two I/O modes precisely (extends `BLOG_SPEC_REVISIONS.md` item 15):

- **Native mode — stdin / getchar / argv READ**: genuinely neural. Bytes are
  located by the multi-slope BOS ALiBi position signature and retrieved through
  the vanilla attention forward (§704-712). This is the part that earns "native."
- **Tool-call mode — PRTF, OPEN/READ/CLOS (files)**: an external Python runner,
  by design (§693-695). NOT neural, and the spec should not imply it is.
- **stdout PUTCHAR emit-side**: the position-signature addressing is symmetric
  and demonstrated for reads; the emit-side is currently still runner-assembled.

Drop the unqualified "100% native neural I/O" phrasing. The accurate statement is
"**stdin/getchar/argv reads are 100% native** (position-signature attention);
file ops and printf formatting are tool calls, by design." That is what the code
now does.

## Addendum (2026-07-22) — raster demos should emit a real image by default

Recommendation for the blog's raster demos (mandelbrot, and any pixel output):
**make actual-image output the default**, not ASCII art. The program writes a
PPM header plus raw RGB bytes through `PRTF`, so the output is a real file you
pipe straight to a viewer:

    model_run mandelbrot > mandelbrot.ppm    # P6: "P6\n<w> <h>\n255\n" then w*h*3 raw RGB bytes

Why this is the better default:

- it exercises the **same** byte-output channel (`PRTF`, the tool-call stdout
  mode described above) — nothing new is required, the demo just chooses an
  image payload instead of characters;
- correctness is both **visually obvious** and **byte-checkable** against a CPU
  escape-time reference (the model reproduces the reference byte stream exactly);
- it makes the "the weights ARE the program, and the program's output is a file
  you can open" point far more forcefully than ASCII art.

Reference artifact: `c4_min/mandelbrot.ppm` / `mandelbrot.png` (320×240, P6,
230,415 bytes, `max_iter=120`) — the exact bytes a C4 `mandelbrot.c` emits via
`printf`/`PRTF`. Keep ASCII art only as a fallback for terminals with no image
viewer.
