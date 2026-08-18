int scale;
int addmul(int a, int b) { return a * b + scale; }
int diff(int a, int b) { int t; t = a - b; return t * t; }
int main() { int r; r = addmul(3, 4); return diff(r, 5); }
