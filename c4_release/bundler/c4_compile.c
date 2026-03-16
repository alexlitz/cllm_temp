/*
 * C4 Bytecode Compiler - Compiles C to bytecode and dumps it
 *
 * Usage:
 *   ./c4_compile program.c > bytecode.h
 *
 * Output: C header with bytecode array and data segment
 */

/* C4-compatible declarations */
int open(char *path, int mode);
int read(int fd, char *buf, int n);
int close(int fd);
char *malloc(int size);
int printf(char *fmt, ...);
int strlen(char *s);
int exit(int code);

/* ============ Constants ============ */

int POOL_SIZE;
int DATA_BASE;

/* Token types */
int TK_NUM;   int TK_ID;    int TK_STR;
int TK_INT;   int TK_CHAR;  int TK_ENUM;
int TK_IF;    int TK_ELSE;  int TK_WHILE;
int TK_RET;   int TK_SIZEOF;
int TK_ASSIGN; int TK_OR;   int TK_AND;
int TK_EQ;    int TK_NE;    int TK_LT;
int TK_GT;    int TK_LE;    int TK_GE;
int TK_SHL;   int TK_SHR;   int TK_ADD;
int TK_SUB;   int TK_MUL;   int TK_DIV;
int TK_MOD;   int TK_INC;   int TK_DEC;
int TK_BRAK;  int TK_EOF;

/* Opcodes */
int OP_LEA;  int OP_IMM;  int OP_JMP;  int OP_JSR;
int OP_BZ;   int OP_BNZ;  int OP_ENT;  int OP_ADJ;
int OP_LEV;  int OP_LI;   int OP_LC;   int OP_SI;
int OP_SC;   int OP_PSH;
int OP_OR;   int OP_XOR;  int OP_AND;
int OP_EQ;   int OP_NE;   int OP_LT;   int OP_GT;
int OP_LE;   int OP_GE;   int OP_SHL;  int OP_SHR;
int OP_ADD;  int OP_SUB;  int OP_MUL;  int OP_DIV;
int OP_MOD;  int OP_EXIT;
int OP_OPEN; int OP_READ; int OP_CLOS; int OP_PRTF;
int OP_MALC; int OP_MSET; int OP_MCMP;
int OP_GETCHAR; int OP_PUTCHAR;

/* Type constants */
int TY_CHAR; int TY_INT; int TY_PTR;

/* ============ Global State ============ */

/* Lexer */
char *src;
int src_pos;
int line;
int tk;
int tk_val;
char *tk_str;
int tk_len;

/* Parser */
int *code;
int code_pos;
char *data;
int data_pos;
int expr_type;

/* Symbol table */
char **sym_names;
int *sym_class;
int *sym_type;
int *sym_val;
int *sym_hclass;
int *sym_htype;
int *sym_hval;
int sym_count;
int MAX_SYMS;

int local_offset;

int init_constants() {
    POOL_SIZE = 65536;
    DATA_BASE = 0x10000;

    TK_NUM = 1;   TK_ID = 2;    TK_STR = 3;
    TK_INT = 10;  TK_CHAR = 11; TK_ENUM = 12;
    TK_IF = 13;   TK_ELSE = 14; TK_WHILE = 15;
    TK_RET = 16;  TK_SIZEOF = 17;
    TK_ASSIGN = 20;
    TK_OR = 22;   TK_AND = 23;
    TK_EQ = 27;   TK_NE = 28;
    TK_LT = 29;   TK_GT = 30;   TK_LE = 31;   TK_GE = 32;
    TK_SHL = 33;  TK_SHR = 34;
    TK_ADD = 35;  TK_SUB = 36;
    TK_MUL = 37;  TK_DIV = 38;  TK_MOD = 39;
    TK_INC = 40;  TK_DEC = 41;
    TK_BRAK = 42;
    TK_EOF = 99;

    OP_LEA = 0;  OP_IMM = 1;  OP_JMP = 2;  OP_JSR = 3;
    OP_BZ = 4;   OP_BNZ = 5;  OP_ENT = 6;  OP_ADJ = 7;
    OP_LEV = 8;  OP_LI = 9;   OP_LC = 10;  OP_SI = 11;
    OP_SC = 12;  OP_PSH = 13;
    OP_OR = 14;  OP_XOR = 15; OP_AND = 16;
    OP_EQ = 17;  OP_NE = 18;  OP_LT = 19;  OP_GT = 20;
    OP_LE = 21;  OP_GE = 22;  OP_SHL = 23; OP_SHR = 24;
    OP_ADD = 25; OP_SUB = 26; OP_MUL = 27; OP_DIV = 28;
    OP_MOD = 29;
    OP_OPEN = 30; OP_READ = 31; OP_CLOS = 32; OP_PRTF = 33;
    OP_MALC = 34; OP_MSET = 36; OP_MCMP = 37;
    OP_EXIT = 38;
    OP_GETCHAR = 64;
    OP_PUTCHAR = 65;

    TY_CHAR = 0; TY_INT = 1; TY_PTR = 2;
    MAX_SYMS = 256;

    return 0;
}

/* ============ String Utilities ============ */

int str_eq(char *a, int alen, char *b, int blen) {
    int i;
    if (alen != blen) return 0;
    i = 0;
    while (i < alen) {
        if (a[i] != b[i]) return 0;
        i = i + 1;
    }
    return 1;
}

/* ============ Symbol Table ============ */

int find_sym(char *name, int len) {
    int i;
    i = 0;
    while (i < sym_count) {
        if (str_eq(sym_names[i], strlen(sym_names[i]), name, len)) return i;
        i = i + 1;
    }
    return -1;
}

int add_sym(char *name, int len, int cls, int typ, int val) {
    int idx;
    char *n;
    int i;

    idx = sym_count;
    sym_count = sym_count + 1;

    n = malloc(len + 1);
    i = 0;
    while (i < len) { n[i] = name[i]; i = i + 1; }
    n[len] = 0;

    sym_names[idx] = n;
    sym_class[idx] = cls;
    sym_type[idx] = typ;
    sym_val[idx] = val;
    sym_hclass[idx] = 0;
    sym_htype[idx] = 0;
    sym_hval[idx] = 0;

    return idx;
}

/* ============ Lexer ============ */

int next() {
    char c;
    int start;

    while (1) {
        c = src[src_pos];
        if (c == 0) { tk = TK_EOF; return 0; }

        if (c == '\n') { line = line + 1; src_pos = src_pos + 1; }
        else if (c == ' ' || c == '\t' || c == '\r') { src_pos = src_pos + 1; }
        else if (c == '/') {
            if (src[src_pos + 1] == '/') {
                src_pos = src_pos + 2;
                while (src[src_pos] != 0 && src[src_pos] != '\n') src_pos = src_pos + 1;
            }
            else if (src[src_pos + 1] == '*') {
                src_pos = src_pos + 2;
                while (src[src_pos] != 0) {
                    if (src[src_pos] == '*' && src[src_pos + 1] == '/') {
                        src_pos = src_pos + 2;
                        break;
                    }
                    if (src[src_pos] == '\n') line = line + 1;
                    src_pos = src_pos + 1;
                }
            }
            else {
                tk = TK_DIV; src_pos = src_pos + 1; return 0;
            }
        }
        else if (c >= '0' && c <= '9') {
            tk_val = 0;
            if (c == '0' && (src[src_pos + 1] == 'x' || src[src_pos + 1] == 'X')) {
                src_pos = src_pos + 2;
                while (1) {
                    c = src[src_pos];
                    if (c >= '0' && c <= '9') tk_val = tk_val * 16 + (c - '0');
                    else if (c >= 'a' && c <= 'f') tk_val = tk_val * 16 + (c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') tk_val = tk_val * 16 + (c - 'A' + 10);
                    else break;
                    src_pos = src_pos + 1;
                }
            } else {
                while (src[src_pos] >= '0' && src[src_pos] <= '9') {
                    tk_val = tk_val * 10 + (src[src_pos] - '0');
                    src_pos = src_pos + 1;
                }
            }
            tk = TK_NUM;
            return 0;
        }
        else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
            start = src_pos;
            while ((src[src_pos] >= 'a' && src[src_pos] <= 'z') ||
                   (src[src_pos] >= 'A' && src[src_pos] <= 'Z') ||
                   (src[src_pos] >= '0' && src[src_pos] <= '9') ||
                   src[src_pos] == '_') {
                src_pos = src_pos + 1;
            }
            tk_str = src + start;
            tk_len = src_pos - start;

            if (str_eq(tk_str, tk_len, "int", 3)) tk = TK_INT;
            else if (str_eq(tk_str, tk_len, "char", 4)) tk = TK_CHAR;
            else if (str_eq(tk_str, tk_len, "enum", 4)) tk = TK_ENUM;
            else if (str_eq(tk_str, tk_len, "if", 2)) tk = TK_IF;
            else if (str_eq(tk_str, tk_len, "else", 4)) tk = TK_ELSE;
            else if (str_eq(tk_str, tk_len, "while", 5)) tk = TK_WHILE;
            else if (str_eq(tk_str, tk_len, "return", 6)) tk = TK_RET;
            else if (str_eq(tk_str, tk_len, "sizeof", 6)) tk = TK_SIZEOF;
            else tk = TK_ID;
            return 0;
        }
        else if (c == '"') {
            src_pos = src_pos + 1;
            tk_val = data_pos + DATA_BASE;
            while (src[src_pos] != '"' && src[src_pos] != 0) {
                c = src[src_pos];
                src_pos = src_pos + 1;
                if (c == '\\') {
                    c = src[src_pos];
                    src_pos = src_pos + 1;
                    if (c == 'n') c = '\n';
                    else if (c == 't') c = '\t';
                    else if (c == '0') c = 0;
                    else if (c == '\\') c = '\\';
                }
                data[data_pos] = c;
                data_pos = data_pos + 1;
            }
            data[data_pos] = 0;
            data_pos = data_pos + 1;
            src_pos = src_pos + 1;
            tk = TK_STR;
            return 0;
        }
        else if (c == '\'') {
            src_pos = src_pos + 1;
            tk_val = src[src_pos];
            if (tk_val == '\\') {
                src_pos = src_pos + 1;
                c = src[src_pos];
                if (c == 'n') tk_val = '\n';
                else if (c == 't') tk_val = '\t';
                else if (c == '0') tk_val = 0;
                else tk_val = c;
            }
            src_pos = src_pos + 1;
            if (src[src_pos] == '\'') src_pos = src_pos + 1;
            tk = TK_NUM;
            return 0;
        }
        else if (c == '=') {
            src_pos = src_pos + 1;
            if (src[src_pos] == '=') { src_pos = src_pos + 1; tk = TK_EQ; }
            else tk = TK_ASSIGN;
            return 0;
        }
        else if (c == '!') {
            src_pos = src_pos + 1;
            if (src[src_pos] == '=') { src_pos = src_pos + 1; tk = TK_NE; }
            else { tk = '!'; }
            return 0;
        }
        else if (c == '<') {
            src_pos = src_pos + 1;
            if (src[src_pos] == '=') { src_pos = src_pos + 1; tk = TK_LE; }
            else if (src[src_pos] == '<') { src_pos = src_pos + 1; tk = TK_SHL; }
            else tk = TK_LT;
            return 0;
        }
        else if (c == '>') {
            src_pos = src_pos + 1;
            if (src[src_pos] == '=') { src_pos = src_pos + 1; tk = TK_GE; }
            else if (src[src_pos] == '>') { src_pos = src_pos + 1; tk = TK_SHR; }
            else tk = TK_GT;
            return 0;
        }
        else if (c == '|') {
            src_pos = src_pos + 1;
            if (src[src_pos] == '|') { src_pos = src_pos + 1; tk = TK_OR; }
            else { tk = '|'; }
            return 0;
        }
        else if (c == '&') {
            src_pos = src_pos + 1;
            if (src[src_pos] == '&') { src_pos = src_pos + 1; tk = TK_AND; }
            else { tk = '&'; }
            return 0;
        }
        else if (c == '+') {
            src_pos = src_pos + 1;
            if (src[src_pos] == '+') { src_pos = src_pos + 1; tk = TK_INC; }
            else tk = TK_ADD;
            return 0;
        }
        else if (c == '-') {
            src_pos = src_pos + 1;
            if (src[src_pos] == '-') { src_pos = src_pos + 1; tk = TK_DEC; }
            else tk = TK_SUB;
            return 0;
        }
        else if (c == '*') { src_pos = src_pos + 1; tk = TK_MUL; return 0; }
        else if (c == '%') { src_pos = src_pos + 1; tk = TK_MOD; return 0; }
        else if (c == '[') { src_pos = src_pos + 1; tk = TK_BRAK; return 0; }
        else {
            tk = c;
            src_pos = src_pos + 1;
            return 0;
        }
    }
}

/* ============ Code Generation ============ */

int emit(int op, int imm) {
    code[code_pos] = op | (imm << 8);
    code_pos = code_pos + 1;
    return code_pos - 1;
}

int patch_jmp(int idx, int target) {
    int op;
    op = code[idx] & 0xFF;
    code[idx] = op | (target << 8);
    return 0;
}

/* ============ Expression Parser ============ */

int expr(int level);

int primary() {
    int sym_idx;
    int t;
    int argc;
    int i;
    int sz;

    if (tk == TK_NUM) {
        emit(OP_IMM, tk_val);
        next();
        expr_type = TY_INT;
    }
    else if (tk == TK_STR) {
        emit(OP_IMM, tk_val);
        next();
        expr_type = TY_PTR;
    }
    else if (tk == TK_SIZEOF) {
        next();
        if (tk == '(') next();
        t = TY_INT;
        if (tk == TK_INT) { next(); t = TY_INT; }
        else if (tk == TK_CHAR) { next(); t = TY_CHAR; }
        while (tk == TK_MUL) { next(); t = TY_PTR; }
        if (tk == ')') next();
        emit(OP_IMM, (t >= TY_PTR || t == TY_INT) ? 8 : 1);
        expr_type = TY_INT;
    }
    else if (tk == TK_ID) {
        sym_idx = find_sym(tk_str, tk_len);
        if (sym_idx < 0) {
            sym_idx = add_sym(tk_str, tk_len, 0, TY_INT, 0);
        }
        next();

        if (tk == '(') {
            next();
            argc = 0;
            while (tk != ')' && tk != TK_EOF) {
                expr(TK_ASSIGN);
                emit(OP_PSH, 0);
                argc = argc + 1;
                if (tk == ',') next();
            }
            if (tk == ')') next();

            if (sym_class[sym_idx] == 3) {
                emit(sym_val[sym_idx], 0);
            } else {
                emit(OP_JSR, sym_val[sym_idx]);
            }
            if (argc) emit(OP_ADJ, argc * 8);
            expr_type = sym_type[sym_idx];
        }
        else if (sym_class[sym_idx] == 1) {
            emit(OP_IMM, sym_val[sym_idx]);
            expr_type = sym_type[sym_idx];
        }
        else if (sym_class[sym_idx] == 5) {
            emit(OP_LEA, sym_val[sym_idx]);
            expr_type = sym_type[sym_idx];
            if (expr_type == TY_CHAR) emit(OP_LC, 0);
            else emit(OP_LI, 0);
        }
        else if (sym_class[sym_idx] == 4) {
            emit(OP_IMM, sym_val[sym_idx]);
            expr_type = sym_type[sym_idx];
            if (expr_type == TY_CHAR) emit(OP_LC, 0);
            else emit(OP_LI, 0);
        }
        else {
            expr_type = sym_type[sym_idx];
        }
    }
    else if (tk == '(') {
        next();
        if (tk == TK_INT || tk == TK_CHAR) {
            t = TY_INT;
            if (tk == TK_CHAR) t = TY_CHAR;
            next();
            while (tk == TK_MUL) { next(); t = TY_PTR; }
            if (tk == ')') next();
            primary();
            expr_type = t;
        } else {
            expr(TK_ASSIGN);
            if (tk == ')') next();
        }
    }
    else if (tk == TK_MUL) {
        next();
        primary();
        if (expr_type == TY_CHAR) emit(OP_LC, 0);
        else emit(OP_LI, 0);
        if (expr_type > TY_INT) expr_type = expr_type - 1;
    }
    else if (tk == '&') {
        next();
        primary();
        if ((code[code_pos - 1] & 0xFF) == OP_LI || (code[code_pos - 1] & 0xFF) == OP_LC) {
            code_pos = code_pos - 1;
        }
        expr_type = TY_PTR;
    }
    else if (tk == '!') {
        next();
        primary();
        emit(OP_IMM, 0);
        emit(OP_EQ, 0);
        expr_type = TY_INT;
    }
    else if (tk == TK_SUB) {
        next();
        emit(OP_IMM, -1);
        emit(OP_PSH, 0);
        primary();
        emit(OP_MUL, 0);
        expr_type = TY_INT;
    }
    else if (tk == TK_INC || tk == TK_DEC) {
        t = tk;
        next();
        primary();
        if ((code[code_pos - 1] & 0xFF) == OP_LI || (code[code_pos - 1] & 0xFF) == OP_LC) {
            code[code_pos - 1] = OP_PSH;
            if (expr_type == TY_CHAR) emit(OP_LC, 0);
            else emit(OP_LI, 0);
        }
        emit(OP_PSH, 0);
        emit(OP_IMM, (expr_type >= TY_PTR) ? 8 : 1);
        if (t == TK_INC) emit(OP_ADD, 0);
        else emit(OP_SUB, 0);
        if (expr_type == TY_CHAR) emit(OP_SC, 0);
        else emit(OP_SI, 0);
    }
    return 0;
}

int expr(int level) {
    int t;
    int addr;
    int sz;

    primary();

    while (tk >= level) {
        t = expr_type;

        if (tk == TK_ASSIGN) {
            next();
            if ((code[code_pos - 1] & 0xFF) == OP_LI || (code[code_pos - 1] & 0xFF) == OP_LC) {
                code[code_pos - 1] = OP_PSH;
            }
            expr(TK_ASSIGN);
            if (t == TY_CHAR) emit(OP_SC, 0);
            else emit(OP_SI, 0);
            expr_type = t;
        }
        else if (tk == TK_OR) {
            next();
            addr = emit(OP_BNZ, 0);
            expr(TK_AND);
            patch_jmp(addr, code_pos * 8);
            expr_type = TY_INT;
        }
        else if (tk == TK_AND) {
            next();
            addr = emit(OP_BZ, 0);
            expr(TK_EQ);
            patch_jmp(addr, code_pos * 8);
            expr_type = TY_INT;
        }
        else if (tk == TK_EQ) {
            next(); emit(OP_PSH, 0); expr(TK_LT); emit(OP_EQ, 0); expr_type = TY_INT;
        }
        else if (tk == TK_NE) {
            next(); emit(OP_PSH, 0); expr(TK_LT); emit(OP_NE, 0); expr_type = TY_INT;
        }
        else if (tk == TK_LT) {
            next(); emit(OP_PSH, 0); expr(TK_SHL); emit(OP_LT, 0); expr_type = TY_INT;
        }
        else if (tk == TK_GT) {
            next(); emit(OP_PSH, 0); expr(TK_SHL); emit(OP_GT, 0); expr_type = TY_INT;
        }
        else if (tk == TK_LE) {
            next(); emit(OP_PSH, 0); expr(TK_SHL); emit(OP_LE, 0); expr_type = TY_INT;
        }
        else if (tk == TK_GE) {
            next(); emit(OP_PSH, 0); expr(TK_SHL); emit(OP_GE, 0); expr_type = TY_INT;
        }
        else if (tk == TK_SHL) {
            next(); emit(OP_PSH, 0); expr(TK_ADD); emit(OP_SHL, 0); expr_type = TY_INT;
        }
        else if (tk == TK_SHR) {
            next(); emit(OP_PSH, 0); expr(TK_ADD); emit(OP_SHR, 0); expr_type = TY_INT;
        }
        else if (tk == TK_ADD) {
            next(); emit(OP_PSH, 0); expr(TK_MUL);
            sz = (t >= TY_PTR) ? 8 : 1;
            if (sz > 1) {
                emit(OP_PSH, 0);
                emit(OP_IMM, sz);
                emit(OP_MUL, 0);
            }
            emit(OP_ADD, 0);
            expr_type = t;
        }
        else if (tk == TK_SUB) {
            next(); emit(OP_PSH, 0); expr(TK_MUL);
            sz = (t >= TY_PTR) ? 8 : 1;
            if (sz > 1) {
                emit(OP_PSH, 0);
                emit(OP_IMM, sz);
                emit(OP_MUL, 0);
            }
            emit(OP_SUB, 0);
            expr_type = t;
        }
        else if (tk == TK_MUL) {
            next(); emit(OP_PSH, 0); expr(TK_INC); emit(OP_MUL, 0); expr_type = TY_INT;
        }
        else if (tk == TK_DIV) {
            next(); emit(OP_PSH, 0); expr(TK_INC); emit(OP_DIV, 0); expr_type = TY_INT;
        }
        else if (tk == TK_MOD) {
            next(); emit(OP_PSH, 0); expr(TK_INC); emit(OP_MOD, 0); expr_type = TY_INT;
        }
        else if (tk == TK_INC || tk == TK_DEC) {
            if ((code[code_pos - 1] & 0xFF) == OP_LI || (code[code_pos - 1] & 0xFF) == OP_LC) {
                code[code_pos - 1] = OP_PSH;
                if (t == TY_CHAR) emit(OP_LC, 0);
                else emit(OP_LI, 0);
            }
            emit(OP_PSH, 0);
            emit(OP_IMM, (t >= TY_PTR) ? 8 : 1);
            if (tk == TK_INC) emit(OP_ADD, 0);
            else emit(OP_SUB, 0);
            if (t == TY_CHAR) emit(OP_SC, 0);
            else emit(OP_SI, 0);
            emit(OP_PSH, 0);
            emit(OP_IMM, (t >= TY_PTR) ? 8 : 1);
            if (tk == TK_INC) emit(OP_SUB, 0);
            else emit(OP_ADD, 0);
            next();
            expr_type = t;
        }
        else if (tk == TK_BRAK) {
            next(); emit(OP_PSH, 0); expr(TK_ASSIGN);
            if (t >= TY_PTR) {
                emit(OP_PSH, 0);
                emit(OP_IMM, 8);
                emit(OP_MUL, 0);
            }
            emit(OP_ADD, 0);
            if (tk == ']') next();
            if (t == TY_PTR + TY_CHAR) emit(OP_LC, 0);
            else emit(OP_LI, 0);
            expr_type = (t >= TY_PTR) ? t - 1 : TY_INT;
        }
        else break;
    }
    return 0;
}

/* ============ Statement Parser ============ */

int stmt() {
    int addr1;
    int addr2;

    if (tk == TK_IF) {
        next();
        if (tk == '(') next();
        expr(TK_ASSIGN);
        if (tk == ')') next();
        addr1 = emit(OP_BZ, 0);
        stmt();
        if (tk == TK_ELSE) {
            next();
            addr2 = emit(OP_JMP, 0);
            patch_jmp(addr1, code_pos * 8);
            stmt();
            patch_jmp(addr2, code_pos * 8);
        } else {
            patch_jmp(addr1, code_pos * 8);
        }
    }
    else if (tk == TK_WHILE) {
        next();
        addr1 = code_pos * 8;
        if (tk == '(') next();
        expr(TK_ASSIGN);
        if (tk == ')') next();
        addr2 = emit(OP_BZ, 0);
        stmt();
        emit(OP_JMP, addr1);
        patch_jmp(addr2, code_pos * 8);
    }
    else if (tk == TK_RET) {
        next();
        if (tk != ';') expr(TK_ASSIGN);
        emit(OP_LEV, 0);
        if (tk == ';') next();
    }
    else if (tk == '{') {
        next();
        while (tk != '}' && tk != TK_EOF) stmt();
        if (tk == '}') next();
    }
    else if (tk == ';') {
        next();
    }
    else {
        expr(TK_ASSIGN);
        if (tk == ';') next();
    }
    return 0;
}

/* ============ Declaration Parser ============ */

int decl() {
    int base_type;
    int t;
    int sym_idx;
    int params;
    int func_addr;
    int enum_val;
    int i;

    if (tk == TK_ENUM) {
        next();
        if (tk == TK_ID) next();
        if (tk == '{') {
            next();
            enum_val = 0;
            while (tk != '}' && tk != TK_EOF) {
                if (tk == TK_ID) {
                    add_sym(tk_str, tk_len, 1, TY_INT, enum_val);
                    next();
                }
                if (tk == TK_ASSIGN) {
                    next();
                    if (tk == TK_NUM) {
                        enum_val = tk_val;
                        next();
                    }
                }
                enum_val = enum_val + 1;
                if (tk == ',') next();
            }
            if (tk == '}') next();
        }
        if (tk == ';') next();
        return 0;
    }

    base_type = TY_INT;
    if (tk == TK_INT) { next(); base_type = TY_INT; }
    else if (tk == TK_CHAR) { next(); base_type = TY_CHAR; }

    while (tk != ';' && tk != TK_EOF) {
        t = base_type;
        while (tk == TK_MUL) { next(); t = TY_PTR; }

        if (tk != TK_ID) break;

        sym_idx = find_sym(tk_str, tk_len);
        if (sym_idx < 0) {
            sym_idx = add_sym(tk_str, tk_len, 0, t, 0);
        }
        next();

        if (tk == '(') {
            sym_class[sym_idx] = 2;
            sym_val[sym_idx] = code_pos * 8;
            next();

            local_offset = 16;
            params = 0;
            while (tk != ')' && tk != TK_EOF) {
                t = TY_INT;
                if (tk == TK_INT) { next(); t = TY_INT; }
                else if (tk == TK_CHAR) { next(); t = TY_CHAR; }
                while (tk == TK_MUL) { next(); t = TY_PTR; }

                if (tk == TK_ID) {
                    i = find_sym(tk_str, tk_len);
                    if (i >= 0) {
                        sym_hclass[i] = sym_class[i];
                        sym_htype[i] = sym_type[i];
                        sym_hval[i] = sym_val[i];
                    } else {
                        i = add_sym(tk_str, tk_len, 0, 0, 0);
                    }
                    sym_class[i] = 5;
                    sym_type[i] = t;
                    sym_val[i] = local_offset;
                    local_offset = local_offset + 8;
                    params = params + 1;
                    next();
                }
                if (tk == ',') next();
            }
            if (tk == ')') next();

            i = 0;
            while (i < sym_count) {
                if (sym_class[i] == 5 && sym_val[i] >= 16) {
                    sym_val[i] = (params - 1 - (sym_val[i] - 16) / 8) * 8 + 16;
                }
                i = i + 1;
            }

            local_offset = 0;

            if (tk == '{') {
                next();
                while (tk == TK_INT || tk == TK_CHAR) {
                    t = TY_INT;
                    if (tk == TK_CHAR) t = TY_CHAR;
                    next();

                    while (tk != ';' && tk != TK_EOF) {
                        while (tk == TK_MUL) { next(); t = TY_PTR; }

                        if (tk == TK_ID) {
                            i = find_sym(tk_str, tk_len);
                            if (i >= 0) {
                                sym_hclass[i] = sym_class[i];
                                sym_htype[i] = sym_type[i];
                                sym_hval[i] = sym_val[i];
                            } else {
                                i = add_sym(tk_str, tk_len, 0, 0, 0);
                            }
                            local_offset = local_offset - 8;
                            sym_class[i] = 5;
                            sym_type[i] = t;
                            sym_val[i] = local_offset;
                            next();
                        }
                        if (tk == ',') { next(); t = base_type; }
                        else break;
                    }
                    if (tk == ';') next();
                }

                func_addr = code_pos;
                emit(OP_ENT, 0 - local_offset);

                while (tk != '}' && tk != TK_EOF) stmt();

                emit(OP_LEV, 0);
                if (tk == '}') next();

                i = 0;
                while (i < sym_count) {
                    if (sym_hclass[i]) {
                        sym_class[i] = sym_hclass[i];
                        sym_type[i] = sym_htype[i];
                        sym_val[i] = sym_hval[i];
                        sym_hclass[i] = 0;
                    }
                    i = i + 1;
                }
            }
            return 0;
        }
        else {
            sym_class[sym_idx] = 4;
            sym_val[sym_idx] = data_pos + DATA_BASE;
            data_pos = data_pos + 8;
        }

        if (tk == ',') next();
    }
    if (tk == ';') next();
    return 0;
}

/* ============ Compile ============ */

int compile(char *source) {
    int main_idx;
    int i;

    src = source;
    src_pos = 0;
    line = 1;

    code = malloc(POOL_SIZE * 4);
    data = malloc(POOL_SIZE);
    code_pos = 1;
    data_pos = 0;

    sym_names = malloc(MAX_SYMS * 8);
    sym_class = malloc(MAX_SYMS * 4);
    sym_type = malloc(MAX_SYMS * 4);
    sym_val = malloc(MAX_SYMS * 4);
    sym_hclass = malloc(MAX_SYMS * 4);
    sym_htype = malloc(MAX_SYMS * 4);
    sym_hval = malloc(MAX_SYMS * 4);
    sym_count = 0;

    i = 0;
    while (i < MAX_SYMS) { sym_hclass[i] = 0; i = i + 1; }

    /* Built-in functions */
    add_sym("open", 4, 3, TY_INT, OP_OPEN);
    add_sym("read", 4, 3, TY_INT, OP_READ);
    add_sym("close", 5, 3, TY_INT, OP_CLOS);
    add_sym("printf", 6, 3, TY_INT, OP_PRTF);
    add_sym("malloc", 6, 3, TY_PTR, OP_MALC);
    add_sym("memset", 6, 3, TY_PTR, OP_MSET);
    add_sym("memcmp", 6, 3, TY_INT, OP_MCMP);
    add_sym("getchar", 7, 3, TY_INT, OP_GETCHAR);
    add_sym("putchar", 7, 3, TY_INT, OP_PUTCHAR);
    add_sym("exit", 4, 3, TY_INT, OP_EXIT);

    next();
    while (tk != TK_EOF) {
        decl();
    }

    main_idx = find_sym("main", 4);
    if (main_idx >= 0) {
        code[0] = OP_JSR | (sym_val[main_idx] << 8);
    }

    return 0;
}

/* ============ Dump Bytecode ============ */

int dump_bytecode() {
    int i;

    printf("/* Auto-generated bytecode */\n\n");

    printf("int program_code[] = {\n");
    i = 0;
    while (i < code_pos) {
        if ((i & 7) == 0) printf("    ");
        printf("0x%x", code[i]);
        if (i + 1 < code_pos) printf(",");
        if ((i & 7) == 7 || i + 1 == code_pos) printf("\n");
        i = i + 1;
    }
    printf("};\n");
    printf("int program_code_len = %d;\n\n", code_pos);

    printf("char program_data[] = {\n");
    i = 0;
    while (i < data_pos) {
        if ((i & 15) == 0) printf("    ");
        printf("0x%x", data[i] & 255);
        if (i + 1 < data_pos) printf(",");
        if ((i & 15) == 15 || i + 1 == data_pos) printf("\n");
        i = i + 1;
    }
    if (data_pos == 0) printf("    0\n");
    printf("};\n");
    printf("int program_data_len = %d;\n", data_pos);

    return 0;
}

/* ============ Main ============ */

int main(int argc, char **argv) {
    int fd;
    int src_len;
    char *source;

    init_constants();

    if (argc < 2) {
        printf("Usage: %s program.c\n", argv[0]);
        return 1;
    }

    fd = open(argv[1], 0);
    if (fd < 0) {
        printf("Error: cannot open %s\n", argv[1]);
        return 1;
    }

    source = malloc(POOL_SIZE);
    src_len = read(fd, source, POOL_SIZE - 1);
    source[src_len] = 0;
    close(fd);

    compile(source);
    dump_bytecode();

    return 0;
}
