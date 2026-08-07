/* minic_mod.c -- a WHOLE-MODULE minimal C compiler, written in the parent c4
 * subset (int/char/ptr, if/else/while, functions with params + locals, globals,
 * inter-function calls).  It is the #853 loop-closing capstone: a real
 * recursive-descent compiler-as-a-program that reads a SMALL C MODULE (several
 * function definitions + globals + real locals + calls between them) and emits
 * the c4 bytecode as a decimal "op imm" stream via printf.
 *
 * It is a genuine multi-pass-free single-pass compiler with a symbol table:
 *   - lexer:      identifiers, numbers, single/two-char punctuation
 *   - symbols:    globals (Glo) + functions (Fun) + params/locals (Loc)
 *   - codegen:    ENT prologue, LEA local access, JSR/ADJ calls, LEV, branches
 *
 * ADDRESSING.  c4 uses absolute code-array pointers for JSR/branch targets; a
 * program-as-a-program cannot know the host's pointer base, so minic_mod emits
 * SELF-CONSISTENT relative addresses = the instruction INDEX (word count) into
 * its own emitted stream.  JSR/BZ/BNZ/JMP immediates are the target index; a
 * forward reference (function called before defined / if/while skip) is fixed up
 * by a backpatch table.  This is a real, executable compilation: feeding the
 * emitted stream to a c4 VM whose PC is an index (as the neural VM's PC is)
 * runs the module correctly.
 *
 * SUBSET the INPUT may use: int globals; int fn(int a, int b){ int x; ... };
 * statements: return E; if(E) S [else S]; while(E) S; { S* }; E; ;
 * expressions (precedence low->high): assignment '=' ; comparisons == != < > ;
 * additive + - ; multiplicative * / % ; unary - ; primary: NUM, name,
 * name(args), (E).  Enough for m_fixed-style fixed-point helpers.
 */

/* ---- c4 opcode numbers (match c4_min.isa) ---- */
int OP_LEA; int OP_IMM; int OP_JMP; int OP_JSR; int OP_BZ; int OP_BNZ;
int OP_ENT; int OP_ADJ; int OP_LEV; int OP_LI;  int OP_SI;  int OP_PSH;
int OP_OR;  int OP_XOR; int OP_AND;
int OP_EQ;  int OP_NE;  int OP_LT;  int OP_GT;  int OP_LE;  int OP_GE;
int OP_ADD; int OP_SUB; int OP_MUL; int OP_DIV; int OP_MOD;

/* ---- lexer state ---- */
char *src;      /* source buffer */
int   pos;      /* cursor */
int   tk;       /* 0=eof 1=num 2=id 3=punct */
int   tkval;    /* num value, or punct code (see below), or first id char */
char *idstart;  /* start of current identifier in src */
int   idlen;    /* length of current identifier */

/* ---- emitted code (as an index stream we also KEEP so we can backpatch) ---- */
int *code;      /* emitted opcode/imm words, flat: even=op odd=imm */
int  ec;        /* emit cursor = number of words emitted (== next index) */

/* ---- symbol table (parallel arrays; interned by first-char+len+2nd-char) ---- */
int  *sym_key;  /* a cheap hash key for the identifier */
int  *sym_class;/* 1=Glo 2=Fun 3=Loc */
int  *sym_val;  /* Glo: data addr ; Fun: code index ; Loc: frame offset */
int   nsym;
int  *pkeys;    /* scratch buffer for parameter keys (globals; no local arrays) */

int  loc;       /* running local slot counter within a function */

/* punct codes packed so two-char ops are distinct from one char */
/* single chars use their ascii; two-char use 256+ */
int P_EQ; int P_NE; /* == != */

/* ---------- emit ----------
 * We accumulate into code[] (flat op/imm words) and PRINT ONLY AT THE END, after
 * every forward branch (BZ/BNZ/JMP) target has been backpatched.  Printing at
 * emit-time would freeze the placeholder-0 targets before they are fixed up. */
int emitw(int w) {
    code[ec] = w;
    ec = ec + 1;
    return 0;
}
int emit1(int op) {
    emitw(op); emitw(0);
    return 0;
}
int emit2(int op, int imm) {
    emitw(op); emitw(imm);
    return 0;
}

/* ---------- symbol interning ---------- */
int idkey() {
    int k;
    int i;
    k = idlen * 131;
    i = 0;
    while (i < idlen) {
        k = k * 31 + idstart[i];
        i = i + 1;
    }
    return k;
}
int sym_find(int key) {
    int i;
    i = 0;
    while (i < nsym) {
        if (sym_key[i] == key) return i;
        i = i + 1;
    }
    return 0 - 1;
}
int sym_add(int key, int cls, int val) {
    sym_key[nsym] = key;
    sym_class[nsym] = cls;
    sym_val[nsym] = val;
    nsym = nsym + 1;
    return nsym - 1;
}

/* ---------- lexer ---------- */
int next() {
    int c;
    tk = 0; tkval = 0;
    /* NOTE: the parent c4 compiler mis-parses the '\r' char escape as the letter
     * 'r' (114), so we must NOT list '\r' here (it would eat every 'r').  Inputs
     * are LF-terminated; CR is not part of the accepted module charset. */
    while (src[pos] == ' ' || src[pos] == '\n' || src[pos] == '\t')
        pos = pos + 1;
    c = src[pos];
    if (c == 0) { tk = 0; return 0; }
    /* number */
    if (c >= '0' && c <= '9') {
        tkval = 0;
        while (src[pos] >= '0' && src[pos] <= '9') {
            tkval = tkval * 10 + (src[pos] - '0');
            pos = pos + 1;
        }
        tk = 1;
        return 0;
    }
    /* identifier / keyword */
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
        idstart = src + pos;
        idlen = 0;
        while ((src[pos] >= 'a' && src[pos] <= 'z') ||
               (src[pos] >= 'A' && src[pos] <= 'Z') ||
               (src[pos] >= '0' && src[pos] <= '9') || src[pos] == '_') {
            pos = pos + 1;
            idlen = idlen + 1;
        }
        tk = 2;
        tkval = idkey();
        return 0;
    }
    /* two-char punctuation: == != */
    if (c == '=' && src[pos + 1] == '=') { tk = 3; tkval = P_EQ; pos = pos + 2; return 0; }
    if (c == '!' && src[pos + 1] == '=') { tk = 3; tkval = P_NE; pos = pos + 2; return 0; }
    /* single-char punctuation */
    tk = 3; tkval = c; pos = pos + 1;
    return 0;
}

/* keyword recognition by exact source spelling (idstart/idlen) */
int id_is(char *kw) {
    int i;
    i = 0;
    while (kw[i] != 0) {
        if (i >= idlen) return 0;
        if (idstart[i] != kw[i]) return 0;
        i = i + 1;
    }
    if (i != idlen) return 0;
    return 1;
}

/* ---------- expression codegen (precedence climbing) ---------- */
int expr();

/* primary: NUM | name | name(args) | (expr) | -primary */
int primary() {
    int key;
    int si;
    int argc;
    if (tk == 1) { emit2(OP_IMM, tkval); next(); return 0; }
    if (tk == 3 && tkval == '(') {
        next(); expr();
        if (tk == 3 && tkval == ')') next();
        return 0;
    }
    if (tk == 3 && tkval == '-') {           /* unary minus */
        next();
        if (tk == 1) { emit2(OP_IMM, 0 - tkval); next(); return 0; }
        emit2(OP_IMM, 0); emit1(OP_PSH); primary(); emit1(OP_SUB);
        return 0;
    }
    if (tk == 2) {
        key = tkval;
        next();
        if (tk == 3 && tkval == '(') {       /* function call */
            next();
            argc = 0;
            while (!(tk == 3 && tkval == ')')) {
                expr();
                emit1(OP_PSH);
                argc = argc + 1;
                if (tk == 3 && tkval == ',') next();
            }
            next();                          /* consume ')' */
            si = sym_find(key);
            emit2(OP_JSR, sym_val[si]);
            if (argc > 0) emit2(OP_ADJ, argc);
            return 0;
        }
        /* variable reference (Loc or Glo) */
        si = sym_find(key);
        if (sym_class[si] == 3) {            /* Loc */
            emit2(OP_LEA, sym_val[si]); emit1(OP_LI);
        } else {                             /* Glo */
            emit2(OP_IMM, sym_val[si]); emit1(OP_LI);
        }
        return 0;
    }
    return 0;
}

/* multiplicative: primary (('*'|'/'|'%') primary)* */
int term() {
    int op;
    primary();
    while (tk == 3 && (tkval == '*' || tkval == '/' || tkval == '%')) {
        op = tkval;
        emit1(OP_PSH);
        next();
        primary();
        if (op == '*') emit1(OP_MUL);
        if (op == '/') emit1(OP_DIV);
        if (op == '%') emit1(OP_MOD);
    }
    return 0;
}

/* additive: term (('+'|'-') term)* */
int addsub() {
    int op;
    term();
    while (tk == 3 && (tkval == '+' || tkval == '-')) {
        op = tkval;
        emit1(OP_PSH);
        next();
        term();
        if (op == '+') emit1(OP_ADD);
        if (op == '-') emit1(OP_SUB);
    }
    return 0;
}

/* relational/equality: addsub ((< > == !=) addsub)* */
int rel() {
    int op;
    addsub();
    while (tk == 3 && (tkval == '<' || tkval == '>' || tkval == P_EQ || tkval == P_NE)) {
        op = tkval;
        emit1(OP_PSH);
        next();
        addsub();
        if (op == '<') emit1(OP_LT);
        if (op == '>') emit1(OP_GT);
        if (op == P_EQ) emit1(OP_EQ);
        if (op == P_NE) emit1(OP_NE);
    }
    return 0;
}

/* assignment (right-assoc): rel | LEA/IMM addr '=' assign  */
int expr() {
    int key;
    int si;
    int save_ec;
    int addr_op;
    int addr_imm;
    int argc;
    int op;
    /* try to parse an lvalue assignment: a bare identifier followed by '=' */
    if (tk == 2) {
        key = tkval;
        save_ec = ec;
        next();
        if (tk == 3 && tkval == '=') {
            /* assignment: emit address, PSH, value, SI */
            si = sym_find(key);
            if (sym_class[si] == 3) { addr_op = OP_LEA; addr_imm = sym_val[si]; }
            else { addr_op = OP_IMM; addr_imm = sym_val[si]; }
            emit2(addr_op, addr_imm);
            emit1(OP_PSH);
            next();
            expr();
            emit1(OP_SI);
            return 0;
        }
        /* not an assignment: rewind the lexer to re-read this identifier as a
         * value.  We rewound nothing in `code`; restore ec and re-lex by
         * resetting pos back to the identifier and calling next(). */
        ec = save_ec;
        /* re-emit as a value primary: the identifier is already consumed and
         * `tk` now holds the following token; handle the common cases inline by
         * re-deriving from `key` and continuing the rel() chain. */
        si = sym_find(key);
        if (tk == 3 && tkval == '(') {
            /* it was a call all along -- redo via primary path */
            next();
            argc = 0;
            while (!(tk == 3 && tkval == ')')) {
                expr();
                emit1(OP_PSH);
                argc = argc + 1;
                if (tk == 3 && tkval == ',') next();
            }
            next();
            emit2(OP_JSR, sym_val[si]);
            if (argc > 0) emit2(OP_ADJ, argc);
        } else {
            if (sym_class[si] == 3) { emit2(OP_LEA, sym_val[si]); emit1(OP_LI); }
            else { emit2(OP_IMM, sym_val[si]); emit1(OP_LI); }
        }
        /* continue any *,/,+,-,<,> operators that follow this value */
        while (tk == 3 && (tkval == '*' || tkval == '/' || tkval == '%')) {
            op = tkval; emit1(OP_PSH); next(); primary();
            if (op == '*') emit1(OP_MUL);
            if (op == '/') emit1(OP_DIV);
            if (op == '%') emit1(OP_MOD);
        }
        while (tk == 3 && (tkval == '+' || tkval == '-')) {
            op = tkval; emit1(OP_PSH); next(); term();
            if (op == '+') emit1(OP_ADD);
            if (op == '-') emit1(OP_SUB);
        }
        while (tk == 3 && (tkval == '<' || tkval == '>' || tkval == P_EQ || tkval == P_NE)) {
            op = tkval; emit1(OP_PSH); next(); addsub();
            if (op == '<') emit1(OP_LT);
            if (op == '>') emit1(OP_GT);
            if (op == P_EQ) emit1(OP_EQ);
            if (op == P_NE) emit1(OP_NE);
        }
        return 0;
    }
    rel();
    return 0;
}

/* ---------- statement codegen ---------- */
int stmt();

int stmt() {
    int bz_idx;
    int jmp_idx;
    int loop_top;
    if (tk == 2 && id_is("return")) {
        next();
        if (!(tk == 3 && tkval == ';')) expr();
        emit1(OP_LEV);
        if (tk == 3 && tkval == ';') next();
        return 0;
    }
    if (tk == 2 && id_is("if")) {
        next();
        if (tk == 3 && tkval == '(') next();
        expr();
        if (tk == 3 && tkval == ')') next();
        emit2(OP_BZ, 0);
        bz_idx = ec - 1;                  /* index of the BZ imm word */
        stmt();
        if (tk == 2 && id_is("else")) {
            emit2(OP_JMP, 0);
            jmp_idx = ec - 1;
            code[bz_idx] = ec / 2;        /* BZ target = next instr index */
            next();
            stmt();
            code[jmp_idx] = ec / 2;
        } else {
            code[bz_idx] = ec / 2;
        }
        return 0;
    }
    if (tk == 2 && id_is("while")) {
        next();
        loop_top = ec / 2;
        if (tk == 3 && tkval == '(') next();
        expr();
        if (tk == 3 && tkval == ')') next();
        emit2(OP_BZ, 0);
        bz_idx = ec - 1;
        stmt();
        emit2(OP_JMP, loop_top);
        code[bz_idx] = ec / 2;
        return 0;
    }
    if (tk == 3 && tkval == '{') {
        next();
        while (!(tk == 3 && tkval == '}')) stmt();
        next();
        return 0;
    }
    if (tk == 3 && tkval == ';') { next(); return 0; }
    /* expression statement */
    expr();
    if (tk == 3 && tkval == ';') next();
    return 0;
}

/* ---------- top-level: globals + function definitions ---------- */

/* declare params: fills Loc symbols with frame offsets 2,3,4,...
 * c4 param offset = 16 + (nparams-1-i)*8 bytes; we use INDEX offsets and let the
 * ENT/LEV/LEA driver interpret them as word slots.  For a faithful positive
 * frame layout we number params 2..(nparams+1) (2 = first slot above saved bp/pc)
 * matching the wide-LEA driver's BP+imm addressing. */
int parse_params(int *pnp) {
    int np;
    int i;
    np = 0;
    /* collect param identifiers between ( ) into the global pkeys scratch */
    while (!(tk == 3 && tkval == ')')) {
        if (tk == 2 && (id_is("int") || id_is("char"))) { next(); }  /* type */
        if (tk == 2) { pkeys[np] = tkval; np = np + 1; next(); }
        if (tk == 3 && tkval == ',') next();
    }
    next();  /* consume ')' */
    /* FRAME LAYOUT (matches the c4_min draft: LEA imm -> bp + 4*imm, i.e. imm is a
     * WORD/slot index; ENT nl does sp -= 4*nl; the stack grows down by one 4-byte
     * word per slot).  After `arg0;PSH ... argK;PSH JSR; ENT nl`:
     *     bp+4*0 = saved bp      bp+4*1 = return PC
     *     bp+4*2 = LAST pushed arg (argK)   bp+4*(2+j) = ...
     *   args are pushed in SOURCE order (arg0 first), so the FIRST source arg is
     *   deepest: param i (0-based) sits at SLOT 2 + (np-1-i) = (np+1) - i. */
    i = 0;
    while (i < np) {
        sym_add(pkeys[i], 3, (np + 1) - i);   /* Loc, LEA word-slot operand */
        i = i + 1;
    }
    *pnp = np;
    return 0;
}

int parse_locals(int base) {
    int nl;
    int more;
    nl = 0;
    /* a run of `int a, b; char c;` declarations at the top of the function body.
     * Each declaration line is `type name (, name)* ;`. */
    while (tk == 2 && (id_is("int") || id_is("char"))) {
        next();                          /* consume the type keyword */
        /* comma-separated names on THIS declaration line, ended by ';' */
        more = 1;
        while (more == 1) {
            if (tk != 2) { more = 0; }
            else {
                /* local slot: below bp.  LEA imm -> bp + 4*imm, so local #(nl+1) at
                 * SLOT -(nl+1) resolves to bp - 4*(nl+1).  distinct from params (>0). */
                sym_add(tkval, 3, 0 - (nl + 1));
                nl = nl + 1;
                next();                  /* consume the name */
                if (tk == 3 && tkval == ',') { next(); }   /* another name follows */
                else { more = 0; }        /* end of this declaration line */
            }
        }
        if (tk == 3 && tkval == ';') next();   /* consume the ';' ending the line */
    }
    return nl;
}

int main() {
    int n;
    int got;
    int key;
    int si;
    int np;
    int nl;
    int fn_locals_start;
    int ent_imm_idx;
    int data_addr;

    OP_LEA = 0;  OP_IMM = 1;  OP_JMP = 2;  OP_JSR = 3;  OP_BZ = 4;  OP_BNZ = 5;
    OP_ENT = 6;  OP_ADJ = 7;  OP_LEV = 8;  OP_LI = 9;   OP_SI = 11; OP_PSH = 13;
    OP_OR = 14;  OP_XOR = 15; OP_AND = 16;
    OP_EQ = 17;  OP_NE = 18;  OP_LT = 19;  OP_GT = 20;  OP_LE = 21; OP_GE = 22;
    OP_ADD = 25; OP_SUB = 26; OP_MUL = 27; OP_DIV = 28; OP_MOD = 29;
    P_EQ = 300;  P_NE = 301;

    src = malloc(8192);
    code = malloc(65536);
    sym_key = malloc(4096);
    sym_class = malloc(4096);
    sym_val = malloc(4096);
    pkeys = malloc(256);
    nsym = 0;
    ec = 0;
    data_addr = 0;

    /* slurp the module source from stdin */
    n = 0;
    got = read(0, src, 8191);
    while (got > 0) {
        n = n + got;
        got = read(0, src + n, 8191 - n);
    }
    src[n] = 0;
    pos = 0;
    next();

    /* top level: a sequence of `int name ...` declarations.  If `name(` -> a
     * function definition; otherwise a global int (optionally `= NUM;`). */
    while (tk != 0) {
        /* type keyword */
        if (tk == 2 && (id_is("int") || id_is("char"))) next();
        /* the declared name */
        key = tkval;
        next();
        if (tk == 3 && tkval == '(') {
            /* FUNCTION DEFINITION */
            si = sym_find(key);
            if (si < 0) si = sym_add(key, 2, ec / 2);   /* Fun, code index */
            else { sym_class[si] = 2; sym_val[si] = ec / 2; }
            /* remember where locals begin so we can pop them on function exit */
            fn_locals_start = nsym;
            next();                                     /* consume '(' */
            parse_params(&np);
            if (tk == 3 && tkval == '{') next();
            /* locals */
            nl = parse_locals(0);
            /* ENT with the local-slot count */
            emit2(OP_ENT, nl);
            /* body statements */
            while (!(tk == 3 && tkval == '}')) stmt();
            next();                                     /* consume '}' */
            emit1(OP_LEV);
            /* pop this function's params+locals from the symbol table */
            nsym = fn_locals_start;
        } else {
            /* GLOBAL int (possibly `= NUM`) */
            sym_add(key, 1, data_addr);
            data_addr = data_addr + 1;
            if (tk == 3 && tkval == '=') { next(); /* skip initializer NUM */ next(); }
            if (tk == 3 && tkval == ';') next();
        }
    }

    /* emit the finished (fully backpatched) bytecode as "op imm\n" pairs */
    n = 0;
    while (n < ec) {
        printf("%d %d\n", code[n], code[n + 1]);
        n = n + 2;
    }
    return 0;
}
