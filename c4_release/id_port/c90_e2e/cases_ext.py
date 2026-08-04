"""Extended C90 conformance case set (task #839 — broaden the c90_e2e battery).

Each case is a small, self-contained C program in the c4 language SUBSET the repo
compiler (``src.compiler.compile_c``) accepts, with a DETERMINISTIC integer return
value.  The three-stage battery (``run_battery_ext.py``) checks, per case:

  1. gcc reference   : ``gcc-15 -std=c90`` compiled + run, exit code == return & 0xFF
  2. native ./c4     : ``ref_interpret`` (byte-exact golden == the native c4 VM)
  3. transformer     : ``run_pure_forward_complete`` (lean streaming CFM build)

The return value is compared byte-exact across all three.  We use RETURN VALUES
(not printf) as the observable because the transformer decodes AX cleanly and the
gcc exit-code channel is unambiguous; every ``EXPECT`` is the process exit status
gcc would report, i.e. ``return_value & 0xFF``.

IMPORTANT — compiler subset (empirically verified against src/compiler.py):
  SUPPORTED : int, char, enum, pointers (* &), if/else, while, return, sizeof,
              functions (recursion + mutual via forward decl), globals, locals,
              + - * / % & | ^ << >> == != < > <= >= && || ?: = , unary -/*/&,
              array indexing p[k] on a pointer, ++/-- (pre+post), string
              literals, printf (Sys), malloc/free/memset (stdlib .c4).
  NOT       : for, do-while, switch, goto, break, continue, struct, union,
              bitfields, static, extern, array declarations `T a[N]`, multi-dim
              arrays, compound assignment (+=), function pointers, varargs.

So the "for / do / switch / goto / break / continue / struct / union / bitfield"
C90 features are EXERCISED where the subset allows an equivalent (for->while,
switch->if/else chain, arrays via malloc, struct via parallel arrays), and are
FLAGGED as out-of-subset otherwise.  Every case here is inside the subset.

The EXPECT value is the ground truth we independently recompute with gcc at battery
time; the literal here is a human cross-check.
"""

# Each entry: (name, category, source, expect_mod256)
CASES = [

    # ---- integer arithmetic + operator coverage --------------------------------
    ("op_add",            "arith",   "int main(){ return 2+3; }", 5),
    ("op_sub",            "arith",   "int main(){ return 40-7; }", 33),
    ("op_mul",            "arith",   "int main(){ return 6*7; }", 42),
    ("op_div",            "arith",   "int main(){ return 84/2; }", 42),
    ("op_mod",            "arith",   "int main(){ return 17%5; }", 2),
    ("op_div_exact",      "arith",   "int main(){ return 100/4; }", 25),
    ("op_mod_zero_rem",   "arith",   "int main(){ return 20%4; }", 0),
    ("op_neg",            "arith",   "int main(){ int a; a=5; return 0-a; }", 251),  # -5 & 0xFF
    ("op_unary_minus",    "arith",   "int main(){ return -12; }", 244),              # -12 & 0xFF
    ("op_precedence",     "arith",   "int main(){ return 2+3*4-1; }", 13),
    ("op_paren",          "arith",   "int main(){ return (2+3)*4; }", 20),
    ("op_nested_arith",   "arith",   "int main(){ return ((10-2)*3+4)/2; }", 14),

    # ---- bitwise + shift -------------------------------------------------------
    ("bit_and",           "bitwise", "int main(){ return 12 & 10; }", 8),
    ("bit_or",            "bitwise", "int main(){ return 12 | 3; }", 15),
    ("bit_xor",           "bitwise", "int main(){ return 12 ^ 10; }", 6),
    ("bit_and_or",        "bitwise", "int main(){ return (12 & 10) | 1; }", 9),
    ("shift_left",        "bitwise", "int main(){ return 1 << 5; }", 32),
    ("shift_right",       "bitwise", "int main(){ return 200 >> 2; }", 50),
    ("shift_combo",       "bitwise", "int main(){ return (1<<4) | (1<<1); }", 18),
    ("bit_mask_low",      "bitwise", "int main(){ return 0xAB & 0x0F; }", 11),
    ("bit_set_clear",     "bitwise", "int main(){ int x; x=0xF0; x=x|0x0C; return x&0xFC; }", 252),

    # ---- comparison / relational / logical -------------------------------------
    ("cmp_lt_true",       "cmp",     "int main(){ return 3<5; }", 1),
    ("cmp_lt_false",      "cmp",     "int main(){ return 5<3; }", 0),
    ("cmp_gt",            "cmp",     "int main(){ return 9>2; }", 1),
    ("cmp_le_eq",         "cmp",     "int main(){ return 5<=5; }", 1),
    ("cmp_ge",            "cmp",     "int main(){ return 7>=8; }", 0),
    ("cmp_eq",            "cmp",     "int main(){ return 4==4; }", 1),
    ("cmp_ne",            "cmp",     "int main(){ return 4!=5; }", 1),
    ("cmp_chain",         "cmp",     "int main(){ return (3<5)+(2>1)+(4==4); }", 3),
    ("logic_and",         "cmp",     "int main(){ return (1 && 1) + (1 && 0); }", 1),
    ("logic_or",          "cmp",     "int main(){ return (0 || 1) + (0 || 0); }", 1),
    ("logic_short_and",   "cmp",     "int main(){ int a; a=0; if(a && (10/a)) return 9; return 7; }", 7),
    ("logic_short_or",    "cmp",     "int main(){ int a; a=1; if(a || (10/0)) return 3; return 9; }", 3),

    # ---- integer promotion / conversion / overflow wraparound ------------------
    ("wrap_add_256",      "overflow","int main(){ int a; a=200; a=a+100; return a; }", 300 % 256),  # 44
    ("wrap_mul",          "overflow","int main(){ int a; a=50; a=a*10; return a; }", 500 % 256),    # 244
    ("char_trunc",        "overflow","int main(){ char c; c=300; return c; }", 300 % 256),          # 44
    ("char_signext",      "overflow","int main(){ char c; int i; c=200; i=c; return i&0xFF; }", 200),
    ("large_const",       "overflow","int main(){ return 1000000 & 0xFF; }", 1000000 & 0xFF),        # 64
    ("mul_big",           "overflow","int main(){ return (1000*1000) & 0xFF; }", (1000000) & 0xFF),  # 64

    # ---- control flow: if / else / nested --------------------------------------
    ("if_true",           "control", "int main(){ if(1) return 5; return 0; }", 5),
    ("if_else",           "control", "int main(){ int a; a=9; if(a>5) return 1; else return 0; }", 1),
    ("if_elif_chain",     "control", "int main(){ int a; a=2; if(a==1) return 10; else if(a==2) return 20; else return 30; }", 20),
    ("if_nested",         "control", "int main(){ int a; int b; a=3; b=4; if(a>0){ if(b>0) return 7; } return 0; }", 7),
    ("switch_as_ifelse",  "control", "int main(){ int x; x=3; if(x==1) return 11; if(x==2) return 22; if(x==3) return 33; return 0; }", 33),
    ("ternary_basic",     "control", "int main(){ int a; a=5; return a>3?10:20; }", 10),
    ("ternary_nested",    "control", "int main(){ int a; a=2; return a==1?1:(a==2?2:3); }", 2),

    # ---- control flow: while (loops; for/do emulated by while) ------------------
    ("while_sum",         "loops",   "int main(){ int i; int s; i=0; s=0; while(i<5){ s=s+i; i=i+1;} return s; }", 10),
    ("while_countdown",   "loops",   "int main(){ int i; i=10; while(i>0){ i=i-1; } return i; }", 0),
    ("for_emulated",      "loops",   "int main(){ int i; int s; i=1; s=0; while(i<=4){ s=s+i; i=i+1; } return s; }", 10),
    ("while_break_emul",  "loops",   "int main(){ int i; int r; i=0; r=0; while(i<100){ if(i==5){ r=i; i=100; } else i=i+1; } return r; }", 5),
    ("while_continue_emul","loops",  "int main(){ int i; int s; i=0; s=0; while(i<6){ i=i+1; if(i==3) s=s; else s=s+i; } return s; }", 18),
    ("nested_while",      "loops",   "int main(){ int i; int j; int c; i=0; c=0; while(i<3){ j=0; while(j<3){ c=c+1; j=j+1; } i=i+1; } return c; }", 9),
    ("while_factorial",   "loops",   "int main(){ int n; int f; n=5; f=1; while(n>1){ f=f*n; n=n-1; } return f&0xFF; }", 120),
    ("do_while_emul",     "loops",   "int main(){ int i; i=0; i=i+1; while(i<3){ i=i+1; } return i; }", 3),

    # ---- functions: call / args / recursion / mutual recursion -----------------
    ("fn_call",           "functions","int add(int a,int b){ return a+b; } int main(){ return add(3,4); }", 7),
    ("fn_three_args",     "functions","int f(int a,int b,int c){ return a*100+b*10+c; } int main(){ return f(1,2,3)&0xFF; }", 123 & 0xFF),
    ("fn_five_args",      "functions","int g(int a,int b,int c,int d,int e){ return a+b+c+d+e; } int main(){ return g(1,2,3,4,5); }", 15),
    ("fn_recursion_fac",  "functions","int fac(int n){ if(n<2) return 1; return n*fac(n-1); } int main(){ return fac(5)&0xFF; }", 120),
    ("fn_recursion_fib",  "functions","int fib(int n){ if(n<2) return n; return fib(n-1)+fib(n-2); } int main(){ return fib(10); }", 55),
    ("fn_recursion_sum",  "functions","int s(int n){ if(n==0) return 0; return n+s(n-1); } int main(){ return s(10); }", 55),
    ("fn_mutual_even",    "functions","int isodd(int n); int iseven(int n){ if(n==0) return 1; return isodd(n-1); } int isodd(int n){ if(n==0) return 0; return iseven(n-1); } int main(){ return iseven(8); }", 1),
    ("fn_mutual_odd",     "functions","int isodd(int n); int iseven(int n){ if(n==0) return 1; return isodd(n-1); } int isodd(int n){ if(n==0) return 0; return iseven(n-1); } int main(){ return isodd(7); }", 1),
    ("fn_ackermann_sm",   "functions","int ack(int m,int n){ if(m==0) return n+1; if(n==0) return ack(m-1,1); return ack(m-1,ack(m,n-1)); } int main(){ return ack(2,2); }", 7),
    ("fn_gcd",            "functions","int gcd(int a,int b){ if(b==0) return a; return gcd(b,a%b); } int main(){ return gcd(48,36); }", 12),
    ("fn_power",          "functions","int p(int b,int e){ if(e==0) return 1; return b*p(b,e-1); } int main(){ return p(2,6); }", 64),

    # ---- globals / scope / storage ---------------------------------------------
    ("global_rw",         "storage", "int g; int main(){ g=42; return g; }", 42),
    ("global_two",        "storage", "int x; int y; int main(){ x=10; y=20; return x+y; }", 30),
    ("global_fn_shared",  "storage", "int counter; int bump(){ counter=counter+1; return counter; } int main(){ counter=0; bump(); bump(); return bump(); }", 3),
    ("local_shadows_glob","storage", "int v; int main(){ int v; v=7; return v; }", 7),
    ("global_accumulate", "storage", "int acc; int add(int n){ acc=acc+n; return acc; } int main(){ acc=0; add(3); add(4); return add(5); }", 12),

    # ---- pointers: basic / arith / char-vs-int stride (the #822 class) ---------
    ("ptr_deref",         "pointers","int main(){ int a; int *p; a=7; p=&a; return *p; }", 7),
    ("ptr_write",         "pointers","int main(){ int a; int *p; p=&a; *p=99; return a; }", 99),
    ("ptr_to_ptr",        "pointers","int main(){ int a; int *p; int **pp; a=5; p=&a; pp=&p; return **pp; }", 5),
    ("ptr_int_stride",    "pointers","int main(){ int *p; p=malloc(24); *p=10; *(p+1)=20; *(p+2)=30; return *(p+1); }", 20),
    ("ptr_char_stride",   "pointers","int main(){ char *s; s=malloc(4); *s=65; *(s+1)=66; *(s+2)=67; return *(s+1); }", 66),
    ("ptr_index_int",     "pointers","int main(){ int *p; p=malloc(24); p[0]=1; p[1]=2; p[2]=3; return p[2]; }", 3),
    ("ptr_index_char",    "pointers","int main(){ char *s; s=malloc(4); s[0]=88; s[1]=89; return s[1]; }", 89),
    ("ptr_diff_int",      "pointers","int main(){ int *p; int *q; p=malloc(40); q=p+3; return q-p; }", 3),
    ("ptr_walk_sum",      "pointers","int main(){ int *p; int i; int s; p=malloc(40); i=0; while(i<5){ p[i]=i+1; i=i+1; } i=0; s=0; while(i<5){ s=s+p[i]; i=i+1; } return s; }", 15),
    ("str_literal_index", "pointers","int main(){ char *s; s=\"ABCD\"; return s[2]; }", 67),
    ("str_first_char",    "pointers","int main(){ char *s; s=\"hi\"; return *s; }", 104),

    # ---- arrays-via-malloc: sort / hash / manipulation -------------------------
    ("arr_reverse_sum",   "arrays",  "int main(){ int *a; int i; int s; a=malloc(32); i=0; while(i<4){ a[i]=(i+1)*10; i=i+1; } s=a[0]+a[3]; return s; }", 50),
    ("arr_max",           "arrays",  "int main(){ int *a; int i; int m; a=malloc(40); a[0]=3; a[1]=9; a[2]=2; a[3]=7; a[4]=5; m=a[0]; i=1; while(i<5){ if(a[i]>m) m=a[i]; i=i+1; } return m; }", 9),
    ("arr_bubble_sort",   "arrays",  "int main(){ int *a; int i; int j; int t; a=malloc(40); a[0]=5; a[1]=2; a[2]=8; a[3]=1; a[4]=3; i=0; while(i<5){ j=0; while(j<4){ if(a[j]>a[j+1]){ t=a[j]; a[j]=a[j+1]; a[j+1]=t; } j=j+1; } i=i+1; } return a[0]*10+a[4]; }", 18),
    ("arr_linear_search", "arrays",  "int main(){ int *a; int i; a=malloc(40); a[0]=11; a[1]=22; a[2]=33; a[3]=44; a[4]=55; i=0; while(i<5){ if(a[i]==33) return i; i=i+1; } return 99; }", 2),
    ("arr_dotproduct",    "arrays",  "int main(){ int *a; int *b; int i; int s; a=malloc(24); b=malloc(24); i=0; while(i<3){ a[i]=i+1; b[i]=i+2; i=i+1; } s=0; i=0; while(i<3){ s=s+a[i]*b[i]; i=i+1; } return s; }", 20),
    ("hash_simple",       "arrays",  "int main(){ int h; int i; char *s; s=\"abc\"; h=0; i=0; while(s[i]!=0){ h=(h*31+s[i])&0xFF; i=i+1; } return h; }", (((((0*31+97)&0xFF)*31+98)&0xFF)*31+99)&0xFF),

    # ---- enums -----------------------------------------------------------------
    ("enum_basic",        "enum",    "enum { RED, GREEN, BLUE }; int main(){ return GREEN; }", 1),
    ("enum_explicit",     "enum",    "enum { A=10, B=20, C=30 }; int main(){ return B; }", 20),
    ("enum_mixed",        "enum",    "enum { X=5, Y, Z }; int main(){ return Y+Z; }", 13),
    ("enum_in_expr",      "enum",    "enum { ONE=1, TWO=2, FOUR=4 }; int main(){ return ONE|TWO|FOUR; }", 7),

    # ---- expressions: sizeof / precedence / ternary / assignment ---------------
    ("sizeof_int",        "expr",    "int main(){ return sizeof(int); }", 8),
    ("sizeof_char",       "expr",    "int main(){ return sizeof(char); }", 1),
    ("sizeof_ptr",        "expr",    "int main(){ return sizeof(int*); }", 8),
    ("assign_chain_val",  "expr",    "int main(){ int a; int b; a=5; b=a; b=b+1; return a+b; }", 11),
    ("assign_in_cond",    "expr",    "int main(){ int a; a=3; if((a=a+1)>3) return a; return 0; }", 4),
    ("complex_expr",      "expr",    "int main(){ int a; int b; a=6; b=4; return (a+b)*(a-b)/(a%b); }", (10*2)//2),
    ("cast_int",          "expr",    "int main(){ char c; c=65; return (int)c; }", 65),

    # ---- small real programs: fixed-point / bit tricks / string ----------------
    ("bit_count",         "programs","int main(){ int n; int c; n=0xB7; c=0; while(n>0){ c=c+(n&1); n=n>>1; } return c; }", bin(0xB7).count("1")),
    ("bit_reverse4",      "programs","int main(){ int n; int r; int i; n=0xD; r=0; i=0; while(i<4){ r=(r<<1)|(n&1); n=n>>1; i=i+1; } return r; }", int('{:04b}'.format(0xD)[::-1],2)),
    ("is_power_of_two",   "programs","int main(){ int n; n=64; if(n>0 && (n&(n-1))==0) return 1; return 0; }", 1),
    ("fixed_point_mul",   "programs","int main(){ int a; int b; a=3<<4; b=2<<4; return ((a*b)>>4)&0xFF; }", (((3<<4)*(2<<4))>>4)&0xFF),  # 6<<4=96
    ("collatz_steps",     "programs","int main(){ int n; int s; n=6; s=0; while(n>1){ if(n%2==0) n=n/2; else n=3*n+1; s=s+1; } return s; }", 8),
    ("digit_sum",         "programs","int main(){ int n; int s; n=12345; s=0; while(n>0){ s=s+n%10; n=n/10; } return s; }", 15),
    ("reverse_number",    "programs","int main(){ int n; int r; n=123; r=0; while(n>0){ r=r*10+n%10; n=n/10; } return r&0xFF; }", 321 & 0xFF),
    ("string_length",     "programs","int slen(char *s){ int n; n=0; while(s[n]!=0){ n=n+1; } return n; } int main(){ return slen(\"hello\"); }", 5),
    ("string_upper_first","programs","int main(){ char *s; s=malloc(4); s[0]=97; if(s[0]>=97 && s[0]<=122) s[0]=s[0]-32; return s[0]; }", 65),
    ("count_vowels",      "programs","int main(){ char *s; int i; int c; s=\"education\"; i=0; c=0; while(s[i]!=0){ if(s[i]==97||s[i]==101||s[i]==105||s[i]==111||s[i]==117) c=c+1; i=i+1; } return c; }", sum(1 for ch in "education" if ch in "aeiou")),
    ("min_of_three",      "programs","int min3(int a,int b,int c){ int m; m=a; if(b<m) m=b; if(c<m) m=c; return m; } int main(){ return min3(7,3,5); }", 3),
    ("clamp",             "programs","int clamp(int x,int lo,int hi){ if(x<lo) return lo; if(x>hi) return hi; return x; } int main(){ return clamp(15,0,10); }", 10),
    ("triangle_number",   "programs","int tri(int n){ if(n==0) return 0; return n+tri(n-1); } int main(){ return tri(9); }", 45),
    ("count_bits_set_16", "programs","int main(){ int n; int c; n=0xFF; c=0; while(n>0){ c=c+(n&1); n=n>>1; } return c; }", 8),
]
