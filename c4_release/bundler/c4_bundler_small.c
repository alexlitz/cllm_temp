/*
 * c4_bundler_small.c — the C4-C version of the CHK-4 small-model bundler.
 *
 * BLOG_SPEC §"Bundling Programs" (line 910): "a bundler which takes the
 * [runtime], the [model] weights and the target c program, compiles them into a
 * single binary."  This is that bundler, written in the C4 subset of C — the
 * SAME dialect + validation gate (gcc `-fsyntax-only -w`) as the sibling
 * `bundler/c4_bundler.c` / `neural_c4_bundler.c` (open/read/close/printf/
 * putchar/malloc, `while` loops, pointer arithmetic, global fixed arrays; no
 * structs, no `for`).  It is compiled with gcc as a stand-alone tool; VERIFIED
 * to produce a bundle byte-for-byte identical to the Python bundler.
 *
 * It produces the SAME .c4bundle container as c4_min/bundle_small.py — a flat
 * binary:  MAGIC(8) VERSION(u32) header_json_len(u32) header_json
 *          [runtime][weights][bytecode].
 *
 * Division of labour (the torch-free split):
 *   - the WEIGHT serialisation (torch tensors -> sparse COO) needs torch, so it
 *     stays in Python: `python -m c4_min.bundle_small prepare --source ... \
 *         --out-dir DIR` writes the four raw parts (header.bin / runtime.bin /
 *     weights.bin / bytecode.bin) + a manifest.txt.  That step is the COMPILE of
 *     the target C program's bytecode (§912) + the model export.
 *   - the CONTAINER FUSION (concatenate the parts into the single self-contained
 *     file) is pure byte I/O — THIS program does it, in C4-C, byte-for-byte
 *     identical to what bundle_small.assemble_bundle would have written.
 *
 * So the sequence
 *     python -m c4_min.bundle_small prepare --source S --out-dir D ...
 *     ./c4_bundler_small D/manifest.txt > out.c4bundle    (this program)
 * yields the same bytes as
 *     python -m c4_min.bundle_small assemble --source S --out out.c4bundle ...
 *
 * Usage:
 *   gcc -w -static -o c4_bundler_small bundler/c4_bundler_small.c
 *   ./c4_bundler_small DIR/manifest.txt > out.c4bundle
 *
 * The manifest lists, one per line, "<filename> <length>" in FUSE ORDER
 * (header.bin, runtime.bin, weights.bin, bytecode.bin).  The bundler reads each
 * named file (resolved relative to the manifest's directory) and streams its
 * bytes straight to stdout in order.  No parsing of the header is needed: the
 * prepare step already finalised the offsets in header.bin.
 *
 * SCOPE NOTE: the minimal teaching-compiler `src/compiler.py` compiles a
 * NARROWER C4 (no global `[N]` arrays, syscalls are pre-bound so protos are a
 * redeclare error) — the sibling `bundler/*.c` files do not compile under it
 * either; gcc is the shared C4-bundler toolchain here.
 */

/* C4-compatible syscall / stdlib declarations. */
int open(char *path, int mode);
int read(int fd, char *buf, int n);
int close(int fd);
int printf(char *fmt, ...);
int putchar(int c);
int exit(int code);
char *malloc(int size);

/* Globals (C4 has no locals-of-arrays beyond fixed size; keep buffers global). */
char *manifest_path;
char dir_prefix[1024];      /* directory of the manifest, incl. trailing '/'   */
int  dir_len;
char line_name[1024];       /* one part filename from the manifest             */
char full_path[2048];       /* dir_prefix + line_name                          */
char iobuf[65536];          /* streaming copy buffer                           */

/* Copy the directory portion of manifest_path into dir_prefix (with a trailing
 * slash).  If the path has no '/', dir_prefix is empty (cwd-relative parts). */
int compute_dir_prefix() {
    int i;
    int last_slash;
    i = 0;
    last_slash = -1;
    while (manifest_path[i] != 0) {
        if (manifest_path[i] == '/') {
            last_slash = i;
        }
        i = i + 1;
    }
    dir_len = 0;
    if (last_slash >= 0) {
        i = 0;
        while (i <= last_slash) {
            dir_prefix[i] = manifest_path[i];
            i = i + 1;
        }
        dir_len = last_slash + 1;
    }
    dir_prefix[dir_len] = 0;
    return 0;
}

/* Build full_path = dir_prefix + line_name (NUL-terminated). */
int build_full_path() {
    int i;
    int j;
    i = 0;
    while (i < dir_len) {
        full_path[i] = dir_prefix[i];
        i = i + 1;
    }
    j = 0;
    while (line_name[j] != 0) {
        full_path[i] = line_name[j];
        i = i + 1;
        j = j + 1;
    }
    full_path[i] = 0;
    return 0;
}

/* Stream one named part file to stdout, byte-for-byte.  Returns bytes emitted,
 * or -1 on open failure. */
int emit_part() {
    int fd;
    int n;
    int i;
    int total;

    build_full_path();
    fd = open(full_path, 0);
    if (fd < 0) {
        return -1;
    }
    total = 0;
    n = read(fd, iobuf, 65536);
    while (n > 0) {
        i = 0;
        while (i < n) {
            /* putchar writes the raw byte (0..255) to stdout unmodified. */
            putchar(iobuf[i] & 255);
            i = i + 1;
        }
        total = total + n;
        n = read(fd, iobuf, 65536);
    }
    close(fd);
    return total;
}

/* Read the whole manifest into iobuf (manifests are tiny). Returns length. */
int read_manifest(char *buf, int cap) {
    int fd;
    int n;
    int total;
    fd = open(manifest_path, 0);
    if (fd < 0) {
        return -1;
    }
    total = 0;
    n = read(fd, buf + total, cap - total);
    while (n > 0) {
        total = total + n;
        n = read(fd, buf + total, cap - total);
    }
    close(fd);
    buf[total] = 0;
    return total;
}

int main(int argc, char **argv) {
    char *man;
    int mlen;
    int p;
    int k;
    int emitted;
    int grand_total;
    char c;

    if (argc < 2) {
        /* Usage goes to stderr conceptually; C4 only has stdout, so keep it out
         * of the way — a real invocation always redirects stdout to the bundle,
         * so we print nothing on the happy path. */
        printf("usage: c4_bundler_small MANIFEST > out.c4bundle\n");
        return 1;
    }
    manifest_path = argv[1];
    compute_dir_prefix();

    man = malloc(65536);
    mlen = read_manifest(man, 65535);
    if (mlen < 0) {
        return 1;
    }

    /* Walk the manifest line by line.  Each line: "<name> <length>".  We only
     * need the NAME (the byte count is advisory / for verification); stream that
     * file's bytes to stdout.  Parts appear in FUSE ORDER, so a straight pass
     * concatenates them correctly. */
    grand_total = 0;
    p = 0;
    while (p < mlen) {
        /* skip leading spaces / blank lines */
        while (p < mlen) {
            c = man[p];
            if (c == ' ') { p = p + 1; }
            else { if (c == 10) { p = p + 1; } else { if (c == 13) { p = p + 1; } else { p = p; break; } } }
        }
        if (p >= mlen) { break; }
        /* read the filename token (up to space) */
        k = 0;
        while (p < mlen) {
            c = man[p];
            if (c == ' ') { break; }
            if (c == 10) { break; }
            if (c == 13) { break; }
            line_name[k] = c;
            k = k + 1;
            p = p + 1;
        }
        line_name[k] = 0;
        /* skip the rest of the line (the length field) */
        while (p < mlen) {
            c = man[p];
            p = p + 1;
            if (c == 10) { break; }
        }
        if (k > 0) {
            emitted = emit_part();
            if (emitted < 0) {
                /* On a missing part we abort; the partial stdout is discarded by
                 * the caller (it checks the exit code before using the file). */
                return 2;
            }
            grand_total = grand_total + emitted;
        }
    }

    /* Success: stdout now holds the fused .c4bundle.  We deliberately print
     * nothing else so the stream is byte-clean. */
    return 0;
}
