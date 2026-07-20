/*
 * onnx_kernel_c4subset.c  —  the SMALLEST genuine slice of the fixed-point ONNX
 * runtime, rewritten in the *actual* c4 subset (Swierczek's c4 grammar) so it
 * compiles under c4vm.  This is the Rel-1 tractability probe: the full
 * onnx_runtime_nibble_fixedpoint.c does NOT compile under c4 (2D/3D global
 * arrays, macro/expression array bounds, `long`, varargs printf, fscanf) — see
 * docs/SELFHOST_3LAYER_FEASIBILITY.md.  This isolates the load-bearing inner
 * kernel of the runtime (a fixed-point MatMul: the dominant cost of every
 * transformer forward) using ONLY constructs c4 accepts:
 *
 *   - only `int` (no `long`, no `float`); 16.16-style fixed-point in a 32-bit int
 *   - arrays via malloc'd pointers (c4 has NO `int A[n]` array-decl syntax) —
 *     EXACTLY how the runtime allocs every tv[] tensor
 *   - no varargs printf beyond c4's printf, no fscanf/fopen
 *   - while-loops only
 *
 * It computes C[M,N] = A[M,K] @ B[K,N] in fixed-point, matching the runtime's
 * matmul() inner loop, then returns a checksum.  Under c4vm this runs the SAME
 * per-MAC instruction sequence the full runtime does, so its measured bytecode
 * size + per-call VM step count is the honest unit cost the full runtime pays,
 * multiplied by the ~500k MACs a single tiny-model forward needs.
 */

int SCALE_BITS;

int fpmul(int a, int b) {
  int p;
  p = a * b;
  return p >> 12;   /* 12-bit fractional scale, round-toward-zero */
}

int matmul(int *A, int *B, int *C, int M, int K, int N) {
  int p; int q; int r; int acc;
  p = 0;
  while (p < M) {
    q = 0;
    while (q < N) {
      acc = 0;
      r = 0;
      while (r < K) {
        acc = acc + fpmul(A[p * K + r], B[r * N + q]);
        r = r + 1;
      }
      C[p * N + q] = acc;
      q = q + 1;
    }
    p = p + 1;
  }
  return 0;
}

int main() {
  int *A; int *B; int *C;
  int cs;
  SCALE_BITS = 12;
  A = malloc(4 * 4);   /* 4 ints */
  B = malloc(4 * 4);
  C = malloc(4 * 4);
  /* A = [[1.0,2.0],[3.0,0.5]] scaled by 4096 ; B = [[0.5,1.0],[2.0,1.0]] */
  A[0] = 4096; A[1] = 8192; A[2] = 12288; A[3] = 2048;
  B[0] = 2048; B[1] = 4096; B[2] = 8192; B[3] = 4096;
  matmul(A, B, C, 2, 2, 2);
  cs = C[0] + C[1] + C[2] + C[3];
  return cs;
}
