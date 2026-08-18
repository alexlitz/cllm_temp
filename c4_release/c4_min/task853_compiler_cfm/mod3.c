int base;
int addup(int a, int b) { return a + b + base; }
int scale(int a, int b) { int t; t = a * b; return t - base; }
int pick(int a, int b) { if (a > b) return a; return b; }
int main() {
  int x;
  int y;
  base = 7;
  x = addup(10, 20);
  y = scale(x, 2);
  return pick(y, 30);
}
