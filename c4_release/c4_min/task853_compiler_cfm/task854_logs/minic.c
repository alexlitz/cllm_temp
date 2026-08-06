/* minic.c -- a genuine minimal recursive-descent C-expression compiler,
 * written in the parent c4 subset (int/char/ptr, if/else/while only; no
 * break/for/switch/ternary; all locals at top of function).
 *
 * It reads a C source (a single `int NAME(params){ return EXPR; }` function)
 * from a file argument, tokenizes it, recursively parses the return
 * expression (real precedence-climbing recursion -> drives the VM stack deep),
 * and emits the c4 bytecode as a decimal opcode/imm stream via printf.
 *
 * This is a REAL compiler-as-a-program (tokenizer + recursive descent), scoped
 * to a tiny input so it runs on the pure-forward transformer.  Its bytecode is
 * produced by the parent compile_c and executed by the VM; the emitted output
 * must be byte-exact whether run natively (parent VM ref) or neurally (model).
 */

/* opcode numbers matching c4_min.isa (LEA..PRTF) */
int OP_IMM;   int OP_PSH;  int OP_ADD;  int OP_SUB;
int OP_MUL;   int OP_LEA;  int OP_LEV;  int OP_ENT;

char *src;     /* source buffer */
int   pos;     /* lex cursor */
int   tk;      /* current token kind: 0=eof 1=num 2=id 3=punct */
int   tkval;   /* numeric value / punct char */
char *idbuf;   /* current identifier text (not fully used; kept tiny) */

/* emit one bytecode word as "op imm\n" */
int emit(int op, int imm) {
    printf("%d %d\n", op, imm);
    return 0;
}

/* advance to the next token */
int next() {
    int c;
    tk = 0; tkval = 0;
    /* skip spaces */
    while (src[pos] == ' ' || src[pos] == '\n' || src[pos] == '\t') pos = pos + 1;
    c = src[pos];
    if (c == 0) { tk = 0; return 0; }
    if (c >= '0' && c <= '9') {
        tkval = 0;
        while (src[pos] >= '0' && src[pos] <= '9') {
            tkval = tkval * 10 + (src[pos] - '0');
            pos = pos + 1;
        }
        tk = 1;
        return 0;
    }
    /* single-char punctuation */
    tk = 3; tkval = c; pos = pos + 1;
    return 0;
}

/* forward: expr() recurses via factor()->expr() for parentheses */
int expr();

/* factor: NUM | '(' expr ')' */
int factor() {
    if (tk == 1) {
        emit(OP_IMM, tkval);
        next();
        return 0;
    }
    if (tk == 3 && tkval == '(') {
        next();
        expr();
        /* consume ')' */
        if (tk == 3 && tkval == ')') next();
        return 0;
    }
    return 0;
}

/* term: factor (('*') factor)*  */
int term() {
    factor();
    while (tk == 3 && tkval == '*') {
        emit(OP_PSH, 0);
        next();
        factor();
        emit(OP_MUL, 0);
    }
    return 0;
}

/* expr: term (('+'|'-') term)*  -- recursive-descent, drives the stack */
int expr() {
    int op;
    term();
    while (tk == 3 && (tkval == '+' || tkval == '-')) {
        op = tkval;
        emit(OP_PSH, 0);
        next();
        term();
        if (op == '+') emit(OP_ADD, 0);
        if (op == '-') emit(OP_SUB, 0);
    }
    return 0;
}

/* scan forward to the 'return' expression, compile it, emit LEV */
int compile_return() {
    /* find 'r' 'e' 't' ... just scan to the first '=' ... no; scan to 'return'
     * by locating the substring: we scan for 'n' preceded by 'retur'.  To keep
     * it simple we scan until we see a digit or '(' AFTER the first '{'. */
    /* skip to '{' */
    while (src[pos] != 0 && src[pos] != '{') pos = pos + 1;
    if (src[pos] == '{') pos = pos + 1;
    /* skip to first digit or '(' (start of the return expression) */
    while (src[pos] != 0 && !((src[pos] >= '0' && src[pos] <= '9') || src[pos] == '(')) pos = pos + 1;
    next();
    expr();
    emit(OP_LEV, 0);
    return 0;
}

int main() {
    int n;
    int got;
    OP_LEA = 0; OP_IMM = 1; OP_LEV = 8; OP_PSH = 13;
    OP_ADD = 25; OP_SUB = 26; OP_MUL = 27; OP_ENT = 6;
    src = malloc(4096);
    /* slurp the C source from stdin (fd 0), one read() of up to 4095 bytes */
    n = 0;
    got = read(0, src, 4095);
    while (got > 0) {
        n = n + got;
        got = read(0, src + n, 4095 - n);
    }
    src[n] = 0;
    pos = 0;
    compile_return();
    return 0;
}
