int FRACUNIT;
int iabs(int a) { if (a < 0) return 0 - a; return a; }
int FixedMul(int a, int b) { return a * b / FRACUNIT; }
int FixedDiv(int a, int b) {
  int q;
  if (iabs(a) < iabs(b)) { q = a * FRACUNIT / b; return q; }
  return a / b * FRACUNIT;
}
int clamp(int x, int lo, int hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}
int main() {
  int m;
  int d;
  int c;
  FRACUNIT = 16;
  m = FixedMul(48, 32);
  d = FixedDiv(m, 3);
  c = clamp(d, 10, 40);
  return c;
}
