#!/usr/bin/env python3
"""Broad C90 corpus for the SOURCE-path transpiler conformance harness.

Every case is a self-contained C90 program with a deterministic observable
(exit code mod 256, and/or stdout).  Unlike ``cases_ext*`` (which write in the
c4 SUBSET directly to test the compiler/native-VM/transformer), THESE cases are
written in *vanilla C90* -- using for/switch/continue/break/goto/struct/union/
compound-assignment/function-pointers/multidim-arrays -- exactly the constructs
the TRANSPILER must lower.  They exercise the transpiler's lowering, not the c4
subset.

Each entry: ``(name, category, source, expect_mod256, sizeof_gap)`` where
``sizeof_gap=True`` marks a case whose result legitimately differs between
gcc -m32 (sizeof(int)==4) and the c4 word VM (sizeof(int)==8).

Categories map to the transpiler bug CLASSES under test:
  multidim   -- multi-dimensional arrays  a[i][j] (Class-1 compound-index stride)
  ptrarith   -- pointer arithmetic, pointer-to-pointer
  structarr  -- struct / array member strides (Class-3 &arr[i]->arr[i*N], Class-7 field width)
  compound   -- compound assignment += -= |= &= etc (Class-2 dropped-to-=)
  loops      -- for / while / do-while + continue / break (Class-4 for-continue increment)
  goto       -- goto / labels
  switch     -- switch/case/default incl fallthrough
  bitwise    -- bitwise + signed/unsigned shifts (Class-6 arithmetic vs logical >>)
  fnptr      -- function pointers (Class-5 stubbing risk)
  union      -- unions
  enum       -- enums
  varargs    -- varargs
  recursion  -- recursion
  strings    -- string ops
  storage    -- static/global init
"""

CORPUS = [

    # =====================================================================
    # multi-dimensional arrays  (Class-1 compound-index stride)
    # =====================================================================
    ("md_2d_local", "multidim",
     "int main(){ int a[3][4]; int i; int j; for(i=0;i<3;i++) for(j=0;j<4;j++) a[i][j]=i*4+j; return a[2][3]; }",
     11, False),
    ("md_2d_sum", "multidim",
     "int main(){ int a[2][2]; int s; a[0][0]=1; a[0][1]=2; a[1][0]=3; a[1][1]=4; s=a[0][0]+a[0][1]+a[1][0]+a[1][1]; return s; }",
     10, False),
    ("md_3d", "multidim",
     "int main(){ int a[2][2][2]; a[1][1][1]=42; a[0][0][0]=1; return a[1][1][1]+a[0][0][0]; }",
     43, False),
    ("md_row_major", "multidim",
     "int main(){ int m[3][3]; int i; int j; for(i=0;i<3;i++) for(j=0;j<3;j++) m[i][j]=(i==j)?1:0; return m[0][0]+m[1][1]+m[2][2]; }",
     3, False),
    ("md_global_2d", "multidim",
     "int g[4][4]; int main(){ int i; int j; for(i=0;i<4;i++) for(j=0;j<4;j++) g[i][j]=i+j; return g[3][3]; }",
     6, False),
    ("md_flat_index", "multidim",
     "int main(){ int a[12]; int i; int k; k=3; for(i=0;i<12;i++) a[i]=i; return a[k*2+1]; }",
     7, False),
    ("md_2d_transpose", "multidim",
     "int main(){ int a[2][3]; int b[3][2]; int i; int j; for(i=0;i<2;i++) for(j=0;j<3;j++) a[i][j]=i*3+j; for(i=0;i<2;i++) for(j=0;j<3;j++) b[j][i]=a[i][j]; return b[2][1]; }",
     5, False),

    # =====================================================================
    # pointer arithmetic / pointer-to-pointer  (Class-1/3 stride)
    # =====================================================================
    ("pa_array_ptr", "ptrarith",
     "int main(){ int a[5]; int *p; int i; for(i=0;i<5;i++) a[i]=i*i; p=a+2; return *p; }",
     4, False),
    ("pa_ptr_walk", "ptrarith",
     "int main(){ int a[4]; int *p; int s; a[0]=1;a[1]=2;a[2]=3;a[3]=4; s=0; for(p=a;p<a+4;p++) s+=*p; return s; }",
     10, False),
    ("pa_ptr_diff", "ptrarith",
     "int main(){ int a[10]; int *p; int *q; p=a+2; q=a+9; return q-p; }",
     7, False),
    ("pa_ptr_to_ptr", "ptrarith",
     "int main(){ int x; int *p; int **pp; x=17; p=&x; pp=&p; **pp=99; return x; }",
     99, False),
    ("pa_char_ptr_idx", "ptrarith",
     "int main(){ char b[8]; char *p; int i; for(i=0;i<8;i++) b[i]=65+i; p=b; return p[3]; }",
     68, False),
    ("pa_arr_of_ptr", "ptrarith",
     "int main(){ int x; int y; int z; int *a[3]; x=1;y=2;z=3; a[0]=&x;a[1]=&y;a[2]=&z; return *a[0]+*a[1]+*a[2]; }",
     6, False),
    ("pa_ptr_index_write", "ptrarith",
     "int main(){ int a[6]; int *p; int i; p=a; for(i=0;i<6;i++) p[i]=i+10; return p[5]; }",
     15, False),
    ("pa_double_deref", "ptrarith",
     "int main(){ int arr[3]; int *p; int **pp; arr[1]=55; p=arr; pp=&p; return (*pp)[1]; }",
     55, False),

    # =====================================================================
    # struct / array member strides  (Class-3 &arr[i], Class-7 field width)
    #
    # NOTE: the transpiler's struct support is DOOM-idiom-scoped -- it lowers
    # ``T_t arr[N]`` / ``T_t *p`` GLOBAL struct arrays/pointers (the linuxdoom
    # style: `_t`-suffixed typedef alias + global storage), which exercises the
    # Class-1/3/7 stride+field-width lowering.  The COMMON-C90 forms (tagged
    # ``struct X`` non-typedef, LOCAL struct-value variables, non-`_t` aliases)
    # are ALSO tested here to characterise the transpiler's real surface -- they
    # reveal the "struct support is contract/naming-convention-bound" limitation.
    # =====================================================================
    # --- DOOM-idiom (`_t` typedef + global): SHOULD lower correctly ---
    ("st_t_global_arr", "structarr",
     "typedef struct {int x; int y;} p_t; p_t arr[4]; int main(){ int i; for(i=0;i<4;i++){ arr[i].x=i; arr[i].y=i*10; } return arr[3].x+arr[3].y; }",
     33, False),
    ("st_t_ptr", "structarr",
     "typedef struct {int x; int y;} p_t; p_t arr[4]; int main(){ p_t *p; arr[2].x=5; arr[2].y=6; p=&arr[2]; return p->x+p->y; }",
     11, False),
    ("st_t_ptr_walk", "structarr",
     "typedef struct {int id; int qty;} it_t; it_t items[3]; int main(){ it_t *it; int t; items[0].qty=2; items[1].qty=3; items[2].qty=5; t=0; for(it=items; it<items+3; it++) t+=it->qty; return t; }",
     10, False),
    ("st_t_addr_of_elem", "structarr",
     "typedef struct {int x; int y;} p_t; p_t a[4]; int main(){ p_t *p; int i; for(i=0;i<4;i++) a[i].x=i*i; p=&a[3]; return p->x; }",
     9, False),
    ("st_t_char_field", "structarr",
     "typedef struct {char a; char b; int c;} s_t; s_t g[1]; int main(){ g[0].a=1; g[0].b=2; g[0].c=100; return g[0].a+g[0].b+g[0].c; }",
     103, False),
    ("st_t_short_field", "structarr",
     "typedef struct {short a; short b;} s_t; s_t g[1]; int main(){ g[0].a=300; g[0].b=100; return (g[0].a+g[0].b)&0xFF; }",
     144, False),
    ("st_t_node_list", "structarr",
     "typedef struct {int v; int next;} node_t; node_t pool[5]; int main(){ int i; for(i=0;i<5;i++){ pool[i].v=i+1; pool[i].next=i+2; } return pool[4].v+pool[0].next; }",
     7, False),
    # --- common-C90 forms (tagged struct / local struct-value): limitation probes ---
    ("st_tag_local", "structarr",
     "struct P{int x; int y;}; int main(){ struct P p; p.x=3; p.y=4; return p.x+p.y; }",
     7, False),
    ("st_tag_ptr", "structarr",
     "struct P{int x; int y;}; int main(){ struct P p; struct P *q; p.x=10; q=&p; q->y=20; return p.x+p.y; }",
     30, False),
    ("st_typedef_local", "structarr",
     "typedef struct {int x; int y;} P; int main(){ P p; p.x=3; p.y=4; return p.x+p.y; }",
     7, False),
    ("st_sizeof", "structarr",
     "struct P{int x; int y;}; int main(){ return sizeof(struct P); }",
     8, True),

    # =====================================================================
    # compound assignment  (Class-2 dropped-to-=)
    # =====================================================================
    ("ca_plus", "compound",
     "int main(){ int a; a=10; a+=5; return a; }", 15, False),
    ("ca_minus", "compound",
     "int main(){ int a; a=10; a-=3; return a; }", 7, False),
    ("ca_mul", "compound",
     "int main(){ int a; a=6; a*=7; return a; }", 42, False),
    ("ca_div", "compound",
     "int main(){ int a; a=42; a/=6; return a; }", 7, False),
    ("ca_mod", "compound",
     "int main(){ int a; a=17; a%=5; return a; }", 2, False),
    ("ca_or", "compound",
     "int main(){ int a; a=0x10; a|=0x05; return a; }", 0x15, False),
    ("ca_and", "compound",
     "int main(){ int a; a=0xFF; a&=0x0F; return a; }", 0x0F, False),
    ("ca_xor", "compound",
     "int main(){ int a; a=0xFF; a^=0x0F; return a; }", 0xF0, False),
    ("ca_shl", "compound",
     "int main(){ int a; a=1; a<<=4; return a; }", 16, False),
    ("ca_shr", "compound",
     "int main(){ int a; a=256; a>>=3; return a; }", 32, False),
    ("ca_on_index", "compound",
     "int main(){ int a[4]; int i; for(i=0;i<4;i++) a[i]=i; a[2]+=100; return a[2]; }", 102, False),
    ("ca_on_field", "compound",
     "struct S{int x;}; int main(){ struct S s; s.x=5; s.x+=10; s.x*=2; return s.x; }", 30, False),
    ("ca_on_deref", "compound",
     "int main(){ int x; int *p; x=7; p=&x; *p+=3; *p*=2; return x; }", 20, False),
    ("ca_accumulate", "compound",
     "int main(){ int s; int i; s=0; for(i=1;i<=10;i++) s+=i; return s; }", 55, False),

    # =====================================================================
    # loops: for / while / do-while + continue / break  (Class-4)
    # =====================================================================
    ("lp_for_sum", "loops",
     "int main(){ int i; int s; s=0; for(i=0;i<10;i++) s=s+i; return s; }", 45, False),
    ("lp_for_continue", "loops",
     "int main(){ int i; int s; s=0; for(i=0;i<10;i++){ if(i%2==0) continue; s+=i; } return s; }", 25, False),
    ("lp_for_break", "loops",
     "int main(){ int i; int s; s=0; for(i=0;i<100;i++){ if(i>=5) break; s+=i; } return s; }", 10, False),
    ("lp_while_continue", "loops",
     "int main(){ int i; int s; i=0; s=0; while(i<10){ i++; if(i%3==0) continue; s+=i; } return s; }", 37, False),
    ("lp_do_while", "loops",
     "int main(){ int i; int s; i=0; s=0; do{ s+=i; i++; }while(i<5); return s; }", 10, False),
    ("lp_do_while_break", "loops",
     "int main(){ int i; int s; i=0; s=0; do{ if(i==3) break; s+=i; i++; }while(i<100); return s; }", 3, False),
    ("lp_nested_break", "loops",
     "int main(){ int i; int j; int c; c=0; for(i=0;i<5;i++){ for(j=0;j<5;j++){ if(j==2) break; c++; } } return c; }", 10, False),
    ("lp_nested_continue", "loops",
     "int main(){ int i; int j; int c; c=0; for(i=0;i<3;i++){ for(j=0;j<3;j++){ if(i==j) continue; c++; } } return c; }", 6, False),
    ("lp_for_multi_update", "loops",
     "int main(){ int i; int j; int c; c=0; for(i=0,j=10; i<j; i++,j--) c++; return c; }", 5, False),
    ("lp_for_continue_update", "loops",
     "int main(){ int i; int c; c=0; for(i=0;i<10;i++){ if(i<5) continue; c++; } return c; }", 5, False),
    ("lp_for_empty_body", "loops",
     "int main(){ int i; for(i=0;i<1000;i++); return i&0xFF; }", 1000 & 0xFF, False),
    ("lp_while_nested_cont", "loops",
     "int main(){ int i; int j; int s; i=0; s=0; while(i<4){ j=0; while(j<4){ j++; if(j==2) continue; s++; } i++; } return s; }", 12, False),

    # =====================================================================
    # goto / labels
    # =====================================================================
    ("gt_forward", "goto",
     "int main(){ int x; x=5; goto skip; x=99; skip: return x; }", 5, False),
    ("gt_loop", "goto",
     "int main(){ int i; int s; i=0; s=0; loop: if(i>=10) goto done; s+=i; i++; goto loop; done: return s; }", 45, False),
    ("gt_break_out", "goto",
     "int main(){ int i; int j; for(i=0;i<10;i++){ for(j=0;j<10;j++){ if(i*j>6) goto found; } } found: return i*10+j; }",
     None, False),  # expect filled below via gcc oracle only (deterministic though)
    ("gt_error_cleanup", "goto",
     "int main(){ int r; r=0; if(r==0) goto err; return 1; err: return 42; }", 42, False),

    # =====================================================================
    # switch / case / default incl fallthrough
    # =====================================================================
    ("sw_basic", "switch",
     "int main(){ int x; x=2; switch(x){ case 1: return 10; case 2: return 20; case 3: return 30; } return 0; }", 20, False),
    ("sw_default", "switch",
     "int main(){ int x; x=9; switch(x){ case 1: return 1; case 2: return 2; default: return 99; } return 0; }", 99, False),
    ("sw_fallthrough", "switch",
     "int main(){ int x; int r; x=1; r=0; switch(x){ case 1: r+=1; case 2: r+=2; case 3: r+=3; break; case 4: r+=100; } return r; }", 6, False),
    ("sw_fallthrough_mid", "switch",
     "int main(){ int x; int r; x=2; r=0; switch(x){ case 1: r+=1; case 2: r+=2; case 3: r+=4; break; default: r=200; } return r; }", 6, False),
    ("sw_no_break_default", "switch",
     "int main(){ int x; int r; x=5; r=0; switch(x){ case 1: r=1; break; default: r+=10; case 6: r+=20; } return r; }", 30, False),
    ("sw_in_loop", "switch",
     "int main(){ int i; int s; s=0; for(i=0;i<5;i++){ switch(i){ case 0: s+=1; break; case 2: s+=10; break; case 4: s+=100; break; } } return s; }", 111, False),
    ("sw_char", "switch",
     "int classify(char c){ switch(c){ case 'a': return 1; case 'b': return 2; default: return 0; } } int main(){ return classify('b'); }", 2, False),

    # =====================================================================
    # bitwise + signed / unsigned shifts  (Class-6)
    # =====================================================================
    ("bw_and_or_xor", "bitwise",
     "int main(){ int a; int b; a=0xF0; b=0x3C; return (a&b)+(a|b)+(a^b) & 0xFFFF; }", (0xF0&0x3C)+(0xF0|0x3C)+(0xF0^0x3C), False),
    ("bw_not", "bitwise",
     "int main(){ int a; a=0; return (~a)&0xFF; }", 0xFF, False),
    ("bw_shl", "bitwise",
     "int main(){ int a; a=3; return a<<5; }", 96, False),
    ("bw_shr_pos", "bitwise",
     "int main(){ int a; a=1000; return a>>3; }", 125, False),
    ("bw_shr_signed", "bitwise",
     "int main(){ int a; a=-256; return (a>>4)&0xFF; }", (-16)&0xFF, False),
    ("bw_shr_unsigned", "bitwise",
     "int main(){ unsigned int a; a=0xFFFFFFF0u; return (int)(a>>4)&0xFF; }", 0xFF, True),
    ("bw_shr_uchar", "bitwise",
     "int main(){ unsigned char a; a=0xF0; return a>>4; }", 0x0F, False),
    ("bw_mask_shift", "bitwise",
     "int main(){ int w; w=0x12345678; return (w>>8)&0xFF; }", 0x56, False),
    ("bw_pack", "bitwise",
     "int main(){ int r; int g; int b; r=5; g=6; b=7; return (r<<6)|(g<<3)|b; }", (5<<6)|(6<<3)|7, False),
    ("bw_angle_shift", "bitwise",
     "int main(){ unsigned int a; a=0x80000000u; return (int)((a>>19)&0xFF); }", ((0x80000000>>19)&0xFF), True),
    ("bw_signed_div_neg", "bitwise",
     "int main(){ int a; a=-7; return (a/2)&0xFF; }", (-3)&0xFF, False),
    ("bw_signed_mod_neg", "bitwise",
     "int main(){ int a; a=-7; return (a%3)&0xFF; }", (-1)&0xFF, False),

    # =====================================================================
    # function pointers  (Class-5 stubbing risk)
    # =====================================================================
    ("fp_call", "fnptr",
     "int dbl(int x){ return x*2; } int main(){ int (*f)(int); f=dbl; return f(21); }", 42, False),
    ("fp_apply", "fnptr",
     "int add(int a,int b){ return a+b; } int apply(int (*f)(int,int),int a,int b){ return f(a,b); } int main(){ return apply(add,3,4); }", 7, False),
    ("fp_table", "fnptr",
     "int f0(){ return 10; } int f1(){ return 20; } int f2(){ return 30; } int main(){ int (*t[3])(); int i; int s; t[0]=f0; t[1]=f1; t[2]=f2; s=0; for(i=0;i<3;i++) s+=t[i](); return s; }", 60, False),
    ("fp_dispatch", "fnptr",
     "int inc(int x){return x+1;} int dec(int x){return x-1;} int main(){ int (*op)(int); int r; op=inc; r=op(5); op=dec; r=op(r); return r; }", 5, False),

    # =====================================================================
    # unions
    # =====================================================================
    ("un_int_char", "union",
     "union U{int i; char c[4];}; int main(){ union U u; u.i=0x41424344; return u.c[0]&0xFF; }", 0x44, False),
    ("un_reuse", "union",
     "union U{int a; int b;}; int main(){ union U u; u.a=100; return u.b; }", 100, False),
    ("un_in_struct", "union",
     "struct S{ int tag; union { int i; char c; } val; }; int main(){ struct S s; s.tag=1; s.val.i=77; return s.val.i; }", 77, False),
    ("un_two_view", "union",
     "union U{ int words[2]; char bytes[8]; }; int main(){ union U u; u.words[0]=0x01020304; return u.bytes[0]&0xFF; }", 0x04, False),

    # =====================================================================
    # enums
    # =====================================================================
    ("en_basic", "enum",
     "enum Color{RED,GREEN,BLUE}; int main(){ return RED+GREEN+BLUE; }", 0+1+2, False),
    ("en_explicit", "enum",
     "enum E{A=1,B=10,C,D=100}; int main(){ return A+B+C+D; }", 1+10+11+100, False),
    ("en_as_index", "enum",
     "enum{X,Y,Z}; int main(){ int a[3]; a[X]=1; a[Y]=2; a[Z]=3; return a[Z]; }", 3, False),
    ("en_in_switch", "enum",
     "enum State{IDLE,RUN,STOP}; int main(){ enum State s; s=RUN; switch(s){ case IDLE: return 0; case RUN: return 42; case STOP: return 99; } return -1; }", 42, False),

    # =====================================================================
    # varargs
    # =====================================================================
    ("va_sum", "varargs",
     "#include <stdarg.h>\nint vsum(int n,...){ va_list ap; int s; int i; va_start(ap,n); s=0; for(i=0;i<n;i++) s+=va_arg(ap,int); va_end(ap); return s; } int main(){ return vsum(4,10,20,30,40); }", 100, False),
    ("va_printf", "varargs",
     "#include <stdio.h>\nint main(){ printf(\"%d %d\\n\", 12, 34); return 0; }", 0, False),

    # =====================================================================
    # recursion
    # =====================================================================
    ("rc_fact", "recursion",
     "int fact(int n){ if(n<=1) return 1; return n*fact(n-1); } int main(){ return fact(5)&0xFF; }", 120, False),
    ("rc_fib", "recursion",
     "int fib(int n){ if(n<2) return n; return fib(n-1)+fib(n-2); } int main(){ return fib(10); }", 55, False),
    ("rc_ackermann", "recursion",
     "int ack(int m,int n){ if(m==0) return n+1; if(n==0) return ack(m-1,1); return ack(m-1,ack(m,n-1)); } int main(){ return ack(2,3); }", 9, False),
    ("rc_mutual", "recursion",
     "int is_even(int n); int is_odd(int n){ if(n==0) return 0; return is_even(n-1); } int is_even(int n){ if(n==0) return 1; return is_odd(n-1); } int main(){ return is_even(10); }", 1, False),

    # =====================================================================
    # string ops
    # =====================================================================
    ("str_len", "strings",
     "int slen(char *s){ int n; n=0; while(s[n]) n++; return n; } int main(){ return slen(\"hello world\"); }", 11, False),
    ("str_cpy", "strings",
     "int main(){ char dst[16]; char *src; int i; src=\"abcd\"; i=0; while(src[i]){ dst[i]=src[i]; i++; } dst[i]=0; return dst[3]; }", ord('d'), False),
    ("str_cmp", "strings",
     "int scmp(char *a,char *b){ while(*a && *a==*b){ a++; b++; } return *a - *b; } int main(){ return (scmp(\"abc\",\"abc\")==0)?1:0; }", 1, False),
    ("str_upper", "strings",
     "int main(){ char *s; int i; int c; s=\"aBcD\"; c=0; for(i=0;s[i];i++){ if(s[i]>='a'&&s[i]<='z') c++; } return c; }", 2, False),
    ("str_printf_s", "strings",
     "#include <stdio.h>\nint main(){ printf(\"[%s]\\n\", \"hi\"); return 0; }", 0, False),
    ("str_atoi", "strings",
     "int myatoi(char *s){ int v; v=0; while(*s>='0'&&*s<='9'){ v=v*10+(*s-'0'); s++; } return v; } int main(){ return myatoi(\"123\")&0xFF; }", 123, False),

    # =====================================================================
    # static / global init
    # =====================================================================
    ("sg_global_int", "storage",
     "int g; int main(){ g=42; return g; }", 42, False),
    ("sg_global_init", "storage",
     "int g = 77; int main(){ return g; }", 77, False),
    ("sg_global_arr_init", "storage",
     "int t[5] = {10,20,30,40,50}; int main(){ return t[2]; }", 30, False),
    ("sg_static_local", "storage",
     "int counter(){ static int n = 0; n++; return n; } int main(){ counter(); counter(); return counter(); }", 3, False),
    ("sg_static_arr", "storage",
     "int main(){ static int a[3] = {5,6,7}; return a[0]+a[1]+a[2]; }", 18, False),
    ("sg_global_string", "storage",
     "char *msg = \"XYZ\"; int main(){ return msg[1]; }", ord('Y'), False),
    ("sg_multi_global", "storage",
     "int a=1; int b=2; int c=3; int main(){ return a*100+b*10+c; }", 123, False),
    ("sg_char_arr_init", "storage",
     "char buf[4] = {'a','b','c',0}; int main(){ return buf[2]; }", ord('c'), False),
    ("sg_2d_global_init", "storage",
     "int m[2][2] = {{1,2},{3,4}}; int main(){ return m[1][0]+m[0][1]; }", 5, False),

    # =====================================================================
    # small real programs (integration)
    # =====================================================================
    ("pg_bubble_sort", "programs",
     "int main(){ int a[5]; int i; int j; int t; a[0]=5;a[1]=2;a[2]=8;a[3]=1;a[4]=9; for(i=0;i<5;i++) for(j=0;j<4;j++) if(a[j]>a[j+1]){ t=a[j]; a[j]=a[j+1]; a[j+1]=t; } return a[0]*10+a[4]; }", 19, False),
    ("pg_matrix_mul", "programs",
     "int main(){ int a[2][2]; int b[2][2]; int c[2][2]; int i; int j; int k; a[0][0]=1;a[0][1]=2;a[1][0]=3;a[1][1]=4; b[0][0]=5;b[0][1]=6;b[1][0]=7;b[1][1]=8; for(i=0;i<2;i++) for(j=0;j<2;j++){ c[i][j]=0; for(k=0;k<2;k++) c[i][j]+=a[i][k]*b[k][j]; } return c[1][1]&0xFF; }", (3*6+4*8)&0xFF, False),
    ("pg_linked_list", "programs",
     "struct Node{ int val; int next; }; struct Node pool[10]; int main(){ int head; int i; int sum; for(i=0;i<5;i++){ pool[i].val=(i+1)*2; pool[i].next=i+1; } pool[4].next=-1; head=0; sum=0; while(head!=-1){ sum+=pool[head].val; head=pool[head].next; } return sum; }", 30, False),
    ("pg_stack_machine", "programs",
     "int main(){ int stack[16]; int sp; int r; sp=0; stack[sp++]=3; stack[sp++]=4; r=stack[--sp]; r=r+stack[--sp]; stack[sp++]=r; return stack[0]; }", 7, False),
    ("pg_state_switch", "programs",
     "int main(){ int state; int steps; int out; state=0; steps=0; out=0; while(steps<10){ switch(state){ case 0: out+=1; state=1; break; case 1: out+=2; state=2; break; case 2: out+=4; state=0; break; } steps++; } return out&0xFF; }", None, False),
    ("pg_hist_count", "programs",
     "int main(){ char *s; int hist[26]; int i; s=\"mississippi\"; for(i=0;i<26;i++) hist[i]=0; for(i=0;s[i];i++) hist[s[i]-'a']++; return hist['s'-'a']; }", 4, False),
    ("pg_fixed_mul", "programs",
     "int FixedMul(int a,int b){ return (int)(((long long)a*(long long)b)>>16); } int main(){ return FixedMul(3<<16, 4<<16)>>16; }", 12, False),
]
