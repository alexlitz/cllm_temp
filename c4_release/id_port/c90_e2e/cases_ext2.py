"""Extended C90 conformance case set — WAVE 2 (task #839 follow-on, +52 cases).

Grows the ``id_port/c90_e2e`` battery breadth on the CPU path (gcc + the faithful
full-word ``native_c4`` VM) beyond the 112-case ``cases_ext.CASES``.  These cases
deliberately push the UNDER-TESTED corners of the c4 C-subset:

  integer promotions/conversions, bitfield-equivalent pack/unpack (masks+shifts),
  enum/typedef edges, pointer arithmetic + multidim-via-malloc + pointer-to-pointer,
  struct/union-via-reinterpret, string/escape handling, ternary/sequence points,
  switch-as-if/else + default + fallthrough, static/extern linkage (globals),
  recursion (deep + mutual-3), signed negative div/mod, const-expr / sizeof.

Same three-stage contract as ``cases_ext``:  every case is a small self-contained C
program with a DETERMINISTIC integer return value; the observable is the process exit
status (``return_value & 0xFF``) checked BYTE-EXACT across gcc, the native c4 VM, and
(in a future GPU pass) the transformer.

Each entry: ``(name, category, source, expect_mod256, tf_note)`` where ``tf_note`` is
the STATICALLY-PREDICTED transformer divergence class (this module does NOT run the
transformer — it tags likely-divergent cases from the compiled c4 bytecode using the
SAME heuristic as ``run_battery_flash.classify`` so they feed the existing fix
backlog).  ``tf_note == "ok"`` means the bytecode has no data-segment loads, no
arg-passing calls, no wide-ALU/cmp/shift risk → expected byte-exact on the transformer.

c4-vs-x86 gaps (``sizeof(int)==8`` vs 4) are tagged in ``C4_X86_GAP`` and are EXPECTED,
not failures.  Out-of-subset C90 features (function pointers, varargs) live in
``SUBSET_LIMIT_CASES`` — they are NOT in the green battery; they DOCUMENT the c4-subset
boundary (gcc compiles them, the c4 compiler rejects them by design).
"""

# ---------------------------------------------------------------------------
# In-subset cases: byte-exact native_c4 == gcc (verified CPU-only at authoring).
# Each: (name, category, source, expect_mod256, tf_note)
# ---------------------------------------------------------------------------
CASES2 = [

    # ---- integer promotions / conversions / truncation -----------------------
    ("prom_char_arith",   "promote",  "int main(){ char a; char b; a=100; b=50; return (a+b)&0xFF; }", 150, "ok"),
    ("prom_char_mul",     "promote",  "int main(){ char c; c=20; return (c*c)&0xFF; }", 144, "width/ALU-32bit"),
    ("conv_int_to_char",  "promote",  "int main(){ int i; char c; i=0x1FF; c=i; return c&0xFF; }", 255, "ok"),
    ("conv_neg_char",     "promote",  "int main(){ char c; c=0-1; return c&0xFF; }", 255, "ok"),
    ("char_wrap_127",     "promote",  "int main(){ char c; c=127; c=c+1; return c&0xFF; }", 128, "ok"),
    ("char_wrap_neg",     "promote",  "int main(){ char c; c=0-128; c=c-1; return c&0xFF; }", 127, "ok"),
    ("int_to_char_hi",    "promote",  "int main(){ int i; char c; i=0x1234; c=i; return c&0xFF; }", 0x34, "ok"),
    ("mixed_char_int",    "promote",  "int main(){ char c; int i; c=10; i=1000; return (i+c)&0xFF; }", 1010 & 0xFF, "ok"),
    ("promote_cmp",       "promote",  "int main(){ char c; c=200; if(c<0) return 1; return 0; }", 1, "cmp/EQ-decode"),
    ("uchar_via_mask",    "promote",  "int main(){ int x; x=250; return (x&0xFF); }", 250, "ok"),
    ("bool_from_cmp",     "promote",  "int main(){ int a; a=5; return (a>3)+(a<10)+(a==5); }", 3, "cmp/EQ-decode"),

    # ---- signed negative division / modulo (C-truncating, sign of dividend) ---
    ("div_trunc_neg",     "signed",   "int main(){ int a; a=0-7; return (a/2)&0xFF; }", (-3) & 0xFF, "width/ALU-32bit"),
    ("mod_neg",           "signed",   "int main(){ int a; a=0-7; return (a%3)&0xFF; }", (-1) & 0xFF, "width/ALU-32bit"),
    ("shr_arith_neg",     "signed",   "int main(){ int a; a=0-16; return (a>>2)&0xFF; }", (-4) & 0xFF, "shift-width"),
    ("div_trunc_neg2",    "signed",   "int main(){ int a; a=0-9; return (a/4)&0xFF; }", (-2) & 0xFF, "width/ALU-32bit"),
    ("mod_neg2",          "signed",   "int main(){ int a; a=0-10; return (a%4)&0xFF; }", (-2) & 0xFF, "width/ALU-32bit"),
    ("neg_mul",           "signed",   "int main(){ int a; a=0-3; return (a*4)&0xFF; }", (-12) & 0xFF, "width/ALU-32bit"),
    ("abs_via_if",        "signed",   "int myabs(int x){ if(x<0) return 0-x; return x; } int main(){ return myabs(0-17); }", 17, "call-frame/arg-passing"),
    ("sign_function",     "signed",   "int sgn(int x){ if(x>0) return 1; if(x<0) return 0-1; return 0; } int main(){ return sgn(0-9)&0xFF; }", (-1) & 0xFF, "call-frame/arg-passing"),

    # ---- bitfield-equivalent pack / unpack (shift + mask) --------------------
    ("bitfield_pack",     "bitfield", "int main(){ int f; int r; int g; int b; r=5; g=6; b=7; f=(r<<6)|(g<<3)|b; return ((f>>3)&7); }", 6, "shift-width"),
    ("bitfield_extract",  "bitfield", "int main(){ int w; w=0xABCD; return (w>>8)&0xFF; }", 0xAB, "shift-width"),
    ("bitfield_set",      "bitfield", "int main(){ int flags; flags=0; flags=flags|(1<<2); flags=flags|(1<<5); return flags; }", (1 << 2) | (1 << 5), "shift-width"),
    ("bitfield_clear",    "bitfield", "int main(){ int f; f=0xFF; f=f&(0-1-(1<<3)); return f&0xFF; }", 0xFF & ~(1 << 3), "shift-width"),
    ("bitfield_toggle",   "bitfield", "int main(){ int f; f=0x0F; f=f^0x05; return f; }", 0x0F ^ 0x05, "ok"),

    # ---- enum / typedef edges -----------------------------------------------
    ("enum_as_size",      "enum",     "enum { N=4 }; int main(){ int *p; p=malloc(N*8); p[N-1]=42; return p[3]; }", 42, "data-segment-recall (global/string literal load)"),
    ("enum_gap",          "enum",     "enum { A=1, B=10, C, D }; int main(){ return C+D; }", 11 + 12, "ok"),
    ("enum_in_index",     "enum",     "enum { FIRST, SECOND, THIRD }; int main(){ int *p; p=malloc(24); p[FIRST]=1; p[SECOND]=2; p[THIRD]=3; return p[THIRD]; }", 3, "data-segment-recall (global/string literal load)"),

    # ---- pointer arithmetic / multidim-via-malloc / pointer-to-pointer -------
    ("ptr_2d_index",      "pointers", "int main(){ int *m; m=malloc(72); m[0*3+0]=1; m[1*3+2]=7; return m[1*3+2]; }", 7, "data-segment-recall (global/string literal load)"),
    ("ptr_row_stride",    "pointers", "int main(){ int *m; int r; int c; m=malloc(48); r=1; c=1; m[r*2+c]=9; return m[r*2+c]; }", 9, "data-segment-recall (global/string literal load)"),
    ("ptr_sub_scale",     "pointers", "int main(){ int *p; int *q; p=malloc(80); q=p+7; return q-p; }", 7, "data-segment-recall (global/string literal load)"),
    ("ptr_char_diff",     "pointers", "int main(){ char *p; char *q; p=malloc(16); q=p+5; return q-p; }", 5, "data-segment-recall (global/string literal load)"),
    ("ptr_cast_charint",  "pointers", "int main(){ int *p; char *c; p=malloc(8); *p=0; c=(char*)p; c[0]=65; return *p; }", 65, "data-segment-recall (global/string literal load)"),
    ("union_byte_view",   "pointers", "int main(){ int *p; char *c; p=malloc(8); *p=0x41424344; c=(char*)p; return c[0]; }", 0x44, "data-segment-recall (global/string literal load)"),
    ("post_dec_ptr",      "pointers", "int main(){ int *p; int *q; p=malloc(40); q=p+3; q--; return q-p; }", 2, "data-segment-recall (global/string literal load)"),
    ("matrix_diag_sum",   "pointers", "int main(){ int *m; int i; int s; m=malloc(72); i=0; while(i<9){ m[i]=i+1; i=i+1; } s=m[0]+m[4]+m[8]; return s; }", 15, "data-segment-recall (global/string literal load)"),
    ("matrix_transpose",  "pointers", "int main(){ int *a; int *b; int i; int j; a=malloc(32); b=malloc(32); i=0; while(i<2){ j=0; while(j<2){ a[i*2+j]=i*2+j; j=j+1; } i=i+1; } i=0; while(i<2){ j=0; while(j<2){ b[j*2+i]=a[i*2+j]; j=j+1; } i=i+1; } return b[1*2+0]; }", 1, "data-segment-recall (global/string literal load)"),
    ("ptr_array_of_ptr",  "pointers", "int main(){ int **pp; int *a; int *b; int x; int y; x=11; y=22; a=&x; b=&y; pp=malloc(16); pp[0]=a; pp[1]=b; return *pp[1]; }", 22, "data-segment-recall (global/string literal load)"),
    ("global_ptr_buf",    "pointers", "int *buf; int main(){ buf=malloc(24); buf[0]=7; buf[1]=8; buf[2]=9; return buf[0]+buf[1]+buf[2]; }", 24, "data-segment-recall (global/string literal load)"),

    # ---- string literals / escapes ------------------------------------------
    ("str_escape_nl",     "strings",  "int main(){ char *s; s=\"a\\nb\"; return s[1]; }", 10, "data-segment-recall (global/string literal load)"),
    ("str_escape_tab",    "strings",  "int main(){ char *s; s=\"x\\ty\"; return s[1]; }", 9, "data-segment-recall (global/string literal load)"),
    ("str_len_escape",    "strings",  "int slen(char*s){int n;n=0;while(s[n]!=0)n=n+1;return n;} int main(){ return slen(\"a\\nb\\tc\"); }", 5, "call-frame/arg-passing"),
    ("str_reverse_char",  "strings",  "int main(){ char *s; int n; s=\"abcd\"; n=0; while(s[n]!=0) n=n+1; return s[n-1]; }", ord('d'), "data-segment-recall (global/string literal load)"),
    ("str_compare",       "strings",  "int seq(char*a,char*b){ int i; i=0; while(a[i]!=0){ if(a[i]!=b[i]) return 0; i=i+1; } return b[i]==0; } int main(){ return seq(\"abc\",\"abc\"); }", 1, "call-frame/arg-passing"),
    ("str_count_char",    "strings",  "int main(){ char *s; int i; int c; s=\"mississippi\"; i=0; c=0; while(s[i]!=0){ if(s[i]==115) c=c+1; i=i+1; } return c; }", "mississippi".count("s"), "data-segment-recall (global/string literal load)"),
    ("atoi_simple",       "strings",  "int main(){ char *s; int i; int v; s=\"12345\"; i=0; v=0; while(s[i]!=0){ v=v*10+(s[i]-48); i=i+1; } return v&0xFF; }", 12345 & 0xFF, "data-segment-recall (global/string literal load)"),
    ("global_str_ptr",    "strings",  "char *msg; int main(){ msg=\"XYZ\"; return msg[1]; }", ord('Y'), "data-segment-recall (global/string literal load)"),

    # ---- ternary / sequence points / assignment order -----------------------
    ("ternary_assign",    "seqpoint", "int main(){ int a; int b; a=5; b=(a>3)?100:200; return b&0xFF; }", 100, "cmp/EQ-decode"),
    ("ternary_side",      "seqpoint", "int main(){ int a; int b; a=1; b=0; a>0?(b=7):(b=8); return b; }", 7, "cmp/EQ-decode"),
    ("nested_ternary3",   "seqpoint", "int main(){ int x; x=3; return x==1?10:x==2?20:x==3?30:40; }", 30, "cmp/EQ-decode"),
    ("post_inc_val",      "seqpoint", "int main(){ int i; int j; i=5; j=i++; return j*10+i; }", 5 * 10 + 6, "width/ALU-32bit"),
    ("pre_inc_val",       "seqpoint", "int main(){ int i; int j; i=5; j=++i; return j*10+i; }", 6 * 10 + 6, "width/ALU-32bit"),
    ("chain_assign3",     "seqpoint", "int main(){ int a; int b; int c; a=b=c=5; return a+b+c; }", 15, "ok"),
    ("side_effect_order", "seqpoint", "int main(){ int i; int *a; a=malloc(24); i=0; a[i]=i; i=i+1; a[i]=i; i=i+1; a[i]=i; return a[0]+a[1]+a[2]; }", 3, "data-segment-recall (global/string literal load)"),

    # ---- switch-as-if/else: default + fallthrough ---------------------------
    ("switch_default",    "control",  "int main(){ int x; x=9; if(x==1)return 1; if(x==2)return 2; return 99; }", 99, "cmp/EQ-decode"),
    ("switch_fallthru",   "control",  "int main(){ int x; int r; x=2; r=0; if(x<=3)r=r+1; if(x<=2)r=r+1; if(x<=1)r=r+1; return r; }", 2, "cmp/EQ-decode"),
    ("switch_5way",       "control",  "int classify(int x){ if(x==0)return 100; if(x==1)return 101; if(x==2)return 102; if(x==3)return 103; return 199; } int main(){ return classify(2); }", 102, "call-frame/arg-passing"),
    ("triple_nested_if",  "control",  "int main(){ int a; int b; int c; a=1; b=2; c=3; if(a==1){ if(b==2){ if(c==3) return 42; } } return 0; }", 42, "cmp/EQ-decode"),

    # ---- linkage: extern-equivalent (shared global) / static-local via global -
    ("static_counter",    "linkage",  "int seq; int next(){ seq=seq+1; return seq; } int main(){ seq=0; next(); next(); return next(); }", 3, "data-segment-recall (global/string literal load)"),
    ("linkage_extern",    "linkage",  "int shared; int producer(){ shared=77; return 0; } int consumer(){ return shared; } int main(){ producer(); return consumer(); }", 77, "data-segment-recall (global/string literal load)"),

    # ---- recursion: deep + mutual-3 -----------------------------------------
    ("rec_deep_sum",      "recursion","int s(int n){ if(n==0) return 0; return 1+s(n-1); } int main(){ return s(30); }", 30, "call-frame/arg-passing"),
    ("rec_mutual3",       "recursion","int a(int n); int b(int n); int c(int n); int a(int n){if(n==0)return 0;return 1+b(n-1);} int b(int n){if(n==0)return 0;return 1+c(n-1);} int c(int n){if(n==0)return 0;return 1+a(n-1);} int main(){return a(9);}", 9, "call-frame/arg-passing"),
    ("hanoi_moves",       "recursion","int h(int n){ if(n==0) return 0; return 2*h(n-1)+1; } int main(){ return h(6)&0xFF; }", 63, "call-frame/arg-passing"),
    ("binary_digits",     "recursion","int nd(int n){ if(n==0) return 0; return 1+nd(n/2); } int main(){ return nd(100); }", 7, "call-frame/arg-passing"),
    ("sum_of_squares",    "recursion","int sq(int n){ if(n==0) return 0; return n*n+sq(n-1); } int main(){ return sq(5); }", 55, "call-frame/arg-passing"),
    ("ackermann_2_3",     "recursion","int ack(int m,int n){ if(m==0)return n+1; if(n==0)return ack(m-1,1); return ack(m-1,ack(m,n-1)); } int main(){ return ack(2,3); }", 9, "call-frame/arg-passing"),

    # ---- const-expr / sizeof -------------------------------------------------
    ("sizeof_charptr",    "expr",     "int main(){ return sizeof(char*); }", 8, "ok"),
    ("sizeof_intptrptr",  "expr",     "int main(){ return sizeof(int**); }", 8, "ok"),
    ("sizeof_in_malloc",  "expr",     "int main(){ int *p; p=malloc(3*sizeof(int)); p[2]=99; return p[2]; }", 99, "data-segment-recall (global/string literal load)"),
    ("constexpr_arith",   "expr",     "int main(){ return 3*4+5*6-2; }", 40, "width/ALU-32bit"),
    ("const_shift_expr",  "expr",     "int main(){ return (1<<3)+(1<<2)+(1<<1)+(1<<0); }", 15, "shift-width"),
    ("const_fold_big",    "expr",     "int main(){ return (255*255)&0xFF; }", (255 * 255) & 0xFF, "width/ALU-32bit"),

    # ---- small real programs -------------------------------------------------
    ("fib_iter",          "programs", "int main(){ int a; int b; int t; int i; a=0; b=1; i=0; while(i<10){ t=a+b; a=b; b=t; i=i+1; } return a; }", 55, "ok"),
    ("gcd_iter",          "programs", "int main(){ int a; int b; int t; a=48; b=36; while(b!=0){ t=a%b; a=b; b=t; } return a; }", 12, "width/ALU-32bit"),
    ("prime_count",       "programs", "int isp(int n){ int d; if(n<2)return 0; d=2; while(d*d<=n){ if(n%d==0)return 0; d=d+1; } return 1; } int main(){ int c; int i; c=0; i=2; while(i<20){ c=c+isp(i); i=i+1; } return c; }", 8, "call-frame/arg-passing"),
    ("sum_digits_hex",    "programs", "int main(){ int n; int s; n=0xABC; s=0; while(n>0){ s=s+(n&0xF); n=n>>4; } return s; }", 0xA + 0xB + 0xC, "shift-width"),
    ("ptr_strcpy",        "programs", "int main(){ char *s; char *d; int i; s=\"hey\"; d=malloc(8); i=0; while(s[i]!=0){ d[i]=s[i]; i=i+1; } d[i]=0; return d[1]; }", ord('e'), "data-segment-recall (global/string literal load)"),
    ("nested_loop_mul",   "programs", "int main(){ int i; int j; int p; p=0; i=1; while(i<=3){ j=1; while(j<=3){ p=p+i*j; j=j+1; } i=i+1; } return p; }", 36, "width/ALU-32bit"),
    ("euclid_lcm",        "programs", "int gcd(int a,int b){ if(b==0)return a; return gcd(b,a%b); } int main(){ int a; int b; a=4; b=6; return a*b/gcd(a,b); }", 12, "call-frame/arg-passing"),
    ("power_mod",         "programs", "int main(){ int b; int e; int r; b=3; e=4; r=1; while(e>0){ r=(r*b)&0xFF; e=e-1; } return r; }", (3 ** 4) & 0xFF, "width/ALU-32bit"),
    ("loop_break_flag",   "programs", "int main(){ int i; int found; found=0; i=0; while(i<100 && found==0){ if(i*i>50){ found=i; } i=i+1; } return found; }", 8, "cmp/EQ-decode"),
]

# c4-vs-x86 EXPECTED gaps (c4 sizeof(int)==8 vs x86 4). Tagged, not failures.
C4_X86_GAP2 = {"sizeof_arith"}

CASES2.append(
    ("sizeof_arith", "expr", "int main(){ return sizeof(int)+sizeof(char)*2; }", 6, "ok")
)

# ---------------------------------------------------------------------------
# Out-of-subset C90 features: gcc compiles them, the c4 compiler REJECTS them by
# design.  NOT in the green battery — these DOCUMENT the c4-subset boundary so a
# reader can see what a genuine subset limit (vs a c4 bug) looks like.
# Each: (name, feature, source, gcc_expect, subset_reason)
# ---------------------------------------------------------------------------
SUBSET_LIMIT_CASES = [
    ("fnptr_call", "function-pointers",
     "int apply(int f(int),int x){ return f(x); } int dbl(int x){ return x*2; } int main(){ return apply(dbl,5); }",
     10, "c4 parse_parameter_list has no function-type parameter grammar (SyntaxError)"),
    ("varargs", "varargs",
     "int sum(int n,...){ return n; } int main(){ return sum(3,1,2,3); }",
     3, "c4 has no '...' ellipsis token / va_list (SyntaxError)"),
    ("char_lit_escape", "char-literal-escape",
     "int main(){ return '\\n'; }",
     10, "c4 Lexer consumes the char after the opening quote BEFORE testing for a "
     "backslash, so \\n / \\t / \\0 char literals mis-lex (string-literal escapes DO work)"),
]
