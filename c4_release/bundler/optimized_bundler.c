/*
 * Optimized Neural Bundler — Standard C
 *
 * Bundles bytecode + model + VM runtime into a single C file.
 *
 * Usage:
 *   ./optimized_bundler model.c4onnx program.c optimized_runtime.c > bundled.c
 *   gcc -O2 -o program bundled.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============ Constants ============ */

#define POOL_SIZE 65536
#define DATA_BASE 0x10000
#define MAX_SYMS  256

/* Token types — must be > 125 (highest ASCII char '}'), ascending precedence */
typedef enum {
    TK_NUM = 128, TK_ID = 129, TK_STR = 130,
    TK_INT = 131, TK_CHAR = 132, TK_ENUM = 133,
    TK_IF = 134, TK_ELSE = 135, TK_WHILE = 136,
    TK_RET = 137, TK_SIZEOF = 138,
    TK_ASSIGN = 140, TK_OR = 142, TK_AND = 143,
    TK_EQ = 147, TK_NE = 148, TK_LT = 149, TK_GT = 150,
    TK_LE = 151, TK_GE = 152, TK_SHL = 153, TK_SHR = 154,
    TK_ADD = 155, TK_SUB = 156, TK_MUL = 157,
    TK_DIV = 158, TK_MOD = 159, TK_INC = 160,
    TK_DEC = 161, TK_BRAK = 162, TK_EOF = 199
} TokenType;

/* Opcodes */
typedef enum {
    OP_LEA = 0, OP_IMM = 1, OP_JMP = 2, OP_JSR = 3,
    OP_BZ = 4, OP_BNZ = 5, OP_ENT = 6, OP_ADJ = 7,
    OP_LEV = 8, OP_LI = 9, OP_LC = 10, OP_SI = 11,
    OP_SC = 12, OP_PSH = 13,
    OP_OR = 14, OP_XOR = 15, OP_AND = 16,
    OP_EQ = 17, OP_NE = 18, OP_LT = 19, OP_GT = 20,
    OP_LE = 21, OP_GE = 22, OP_SHL = 23, OP_SHR = 24,
    OP_ADD = 25, OP_SUB = 26, OP_MUL = 27, OP_DIV = 28,
    OP_MOD = 29,
    OP_OPEN = 30, OP_READ = 31, OP_CLOS = 32, OP_PRTF = 33,
    OP_MALC = 34, OP_MSET = 36, OP_MCMP = 37,
    OP_EXIT = 38,
    OP_GETCHAR = 64, OP_PUTCHAR = 65
} Opcode;

/* Type system */
typedef enum { TY_CHAR = 0, TY_INT = 1, TY_PTR = 2 } Type;

/* Symbol classes: 1=Num(enum const), 2=Fun, 3=Syscall, 4=Glo, 5=Loc */

/* ============ Symbol Table ============ */

typedef struct {
    char *name;
    int cls, type, val;
    int h_cls, h_type, h_val;  /* saved during local scope */
} Symbol;

static Symbol syms[MAX_SYMS];
static int sym_count;

/* ============ Compiler State ============ */

static char *src;
static int src_pos, line;
static int tk, tk_val;
static char *tk_str;
static int tk_len;

static int *code;
static int code_pos;
static char *data;
static int data_pos;
static int expr_type;
static int local_offset;

/* Model and runtime data */
static unsigned char *model_data;
static int model_len;
static char *runtime_data;
static int runtime_len;

/* ============ String/Symbol Utilities ============ */

static int str_eq(const char *a, int alen, const char *b, int blen) {
    int i;
    if (alen != blen) return 0;
    for (i = 0; i < alen; i++) { if (a[i] != b[i]) return 0; }
    return 1;
}

static int find_sym(const char *name, int len) {
    int i;
    for (i = 0; i < sym_count; i++) {
        if (str_eq(syms[i].name, (int)strlen(syms[i].name), name, len)) return i;
    }
    return -1;
}

static int add_sym(const char *name, int len, int cls, int typ, int val) {
    int idx = sym_count++;
    char *n = (char *)malloc(len + 1);
    memcpy(n, name, len);
    n[len] = 0;
    syms[idx].name = n;
    syms[idx].cls = cls;
    syms[idx].type = typ;
    syms[idx].val = val;
    syms[idx].h_cls = 0;
    syms[idx].h_type = 0;
    syms[idx].h_val = 0;
    return idx;
}

/* ============ Lexer ============ */

static void next(void) {
    char c;
    int start;
    while (1) {
        c = src[src_pos];
        if (c == 0) { tk = TK_EOF; return; }
        if (c == '\n') { line++; src_pos++; continue; }
        if (c == ' ' || c == '\t' || c == '\r') { src_pos++; continue; }

        if (c == '/') {
            if (src[src_pos + 1] == '/') {
                src_pos += 2;
                while (src[src_pos] != 0 && src[src_pos] != '\n') src_pos++;
                continue;
            } else if (src[src_pos + 1] == '*') {
                src_pos += 2;
                while (src[src_pos] != 0) {
                    if (src[src_pos] == '*' && src[src_pos + 1] == '/') { src_pos += 2; break; }
                    if (src[src_pos] == '\n') line++;
                    src_pos++;
                }
                continue;
            } else { tk = TK_DIV; src_pos++; return; }
        }

        if (c >= '0' && c <= '9') {
            tk_val = 0;
            if (c == '0' && (src[src_pos + 1] == 'x' || src[src_pos + 1] == 'X')) {
                src_pos += 2;
                while (1) {
                    c = src[src_pos];
                    if (c >= '0' && c <= '9') tk_val = tk_val * 16 + (c - '0');
                    else if (c >= 'a' && c <= 'f') tk_val = tk_val * 16 + (c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') tk_val = tk_val * 16 + (c - 'A' + 10);
                    else break;
                    src_pos++;
                }
            } else {
                while (src[src_pos] >= '0' && src[src_pos] <= '9') {
                    tk_val = tk_val * 10 + (src[src_pos] - '0');
                    src_pos++;
                }
            }
            tk = TK_NUM; return;
        }

        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
            start = src_pos;
            while ((src[src_pos] >= 'a' && src[src_pos] <= 'z') ||
                   (src[src_pos] >= 'A' && src[src_pos] <= 'Z') ||
                   (src[src_pos] >= '0' && src[src_pos] <= '9') ||
                   src[src_pos] == '_') { src_pos++; }
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
            return;
        }

        if (c == '"') {
            src_pos++;
            tk_val = data_pos + DATA_BASE;
            while (src[src_pos] != '"' && src[src_pos] != 0) {
                c = src[src_pos]; src_pos++;
                if (c == '\\') {
                    c = src[src_pos]; src_pos++;
                    if (c == 'n') c = '\n';
                    else if (c == 't') c = '\t';
                    else if (c == '0') c = 0;
                }
                data[data_pos++] = c;
            }
            data[data_pos++] = 0;
            src_pos++;
            tk = TK_STR; return;
        }

        if (c == '\'') {
            src_pos++; tk_val = src[src_pos];
            if (tk_val == '\\') {
                src_pos++; c = src[src_pos];
                if (c == 'n') tk_val = '\n';
                else if (c == 't') tk_val = '\t';
                else if (c == '0') tk_val = 0;
                else tk_val = c;
            }
            src_pos++;
            if (src[src_pos] == '\'') src_pos++;
            tk = TK_NUM; return;
        }

        if (c == '=') { src_pos++; if (src[src_pos] == '=') { src_pos++; tk = TK_EQ; } else tk = TK_ASSIGN; return; }
        if (c == '!') { src_pos++; if (src[src_pos] == '=') { src_pos++; tk = TK_NE; } else tk = '!'; return; }
        if (c == '<') { src_pos++; if (src[src_pos] == '=') { src_pos++; tk = TK_LE; } else if (src[src_pos] == '<') { src_pos++; tk = TK_SHL; } else tk = TK_LT; return; }
        if (c == '>') { src_pos++; if (src[src_pos] == '=') { src_pos++; tk = TK_GE; } else if (src[src_pos] == '>') { src_pos++; tk = TK_SHR; } else tk = TK_GT; return; }
        if (c == '|') { src_pos++; if (src[src_pos] == '|') { src_pos++; tk = TK_OR; } else tk = '|'; return; }
        if (c == '&') { src_pos++; if (src[src_pos] == '&') { src_pos++; tk = TK_AND; } else tk = '&'; return; }
        if (c == '+') { src_pos++; if (src[src_pos] == '+') { src_pos++; tk = TK_INC; } else tk = TK_ADD; return; }
        if (c == '-') { src_pos++; if (src[src_pos] == '-') { src_pos++; tk = TK_DEC; } else tk = TK_SUB; return; }
        if (c == '*') { src_pos++; tk = TK_MUL; return; }
        if (c == '%') { src_pos++; tk = TK_MOD; return; }
        if (c == '[') { src_pos++; tk = TK_BRAK; return; }

        tk = c; src_pos++; return;
    }
}

/* ============ Code Generation ============ */

static int emit(int op, int imm) {
    code[code_pos] = op | (imm << 8);
    return code_pos++;
}

static void patch_jmp(int idx, int target) {
    int op = code[idx] & 0xFF;
    code[idx] = op | (target << 8);
}

/* ============ Expression Parser ============ */

static void expr(int level);

static void primary(void) {
    int sym_idx, t, argc;
    if (tk == TK_NUM) { emit(OP_IMM, tk_val); next(); expr_type = TY_INT; }
    else if (tk == TK_STR) { emit(OP_IMM, tk_val); next(); expr_type = TY_PTR; }
    else if (tk == TK_SIZEOF) {
        next(); if (tk == '(') next();
        t = TY_INT;
        if (tk == TK_INT) { next(); t = TY_INT; }
        else if (tk == TK_CHAR) { next(); t = TY_CHAR; }
        while (tk == TK_MUL) { next(); t += TY_PTR; }
        if (tk == ')') next();
        emit(OP_IMM, (t >= TY_PTR || t == TY_INT) ? 8 : 1);
        expr_type = TY_INT;
    }
    else if (tk == TK_ID) {
        sym_idx = find_sym(tk_str, tk_len);
        if (sym_idx < 0) sym_idx = add_sym(tk_str, tk_len, 0, TY_INT, 0);
        next();
        if (tk == '(') {
            next(); argc = 0;
            while (tk != ')' && tk != TK_EOF) {
                expr(TK_ASSIGN); emit(OP_PSH, 0); argc++;
                if (tk == ',') next();
            }
            if (tk == ')') next();
            if (syms[sym_idx].cls == 3) emit(syms[sym_idx].val, 0);
            else emit(OP_JSR, syms[sym_idx].val);
            if (argc) emit(OP_ADJ, argc * 8);
            expr_type = syms[sym_idx].type;
        }
        else if (syms[sym_idx].cls == 1) { emit(OP_IMM, syms[sym_idx].val); expr_type = syms[sym_idx].type; }
        else if (syms[sym_idx].cls == 5) {
            emit(OP_LEA, syms[sym_idx].val); expr_type = syms[sym_idx].type;
            if (expr_type == TY_CHAR) emit(OP_LC, 0); else emit(OP_LI, 0);
        }
        else if (syms[sym_idx].cls == 4) {
            emit(OP_IMM, syms[sym_idx].val); expr_type = syms[sym_idx].type;
            if (expr_type == TY_CHAR) emit(OP_LC, 0); else emit(OP_LI, 0);
        }
        else expr_type = syms[sym_idx].type;
    }
    else if (tk == '(') {
        next();
        if (tk == TK_INT || tk == TK_CHAR) {
            t = TY_INT; if (tk == TK_CHAR) t = TY_CHAR; next();
            while (tk == TK_MUL) { next(); t += TY_PTR; }
            if (tk == ')') next();
            primary(); expr_type = t;
        } else { expr(TK_ASSIGN); if (tk == ')') next(); }
    }
    else if (tk == TK_MUL) {
        next(); primary();
        if (expr_type == TY_PTR) emit(OP_LC, 0); else emit(OP_LI, 0);
        if (expr_type >= TY_PTR) expr_type -= TY_PTR;
    }
    else if (tk == '&') {
        next(); primary();
        if ((code[code_pos - 1] & 0xFF) == OP_LI || (code[code_pos - 1] & 0xFF) == OP_LC) code_pos--;
        expr_type += TY_PTR;
    }
    else if (tk == '!') { next(); primary(); emit(OP_IMM, 0); emit(OP_EQ, 0); expr_type = TY_INT; }
    else if (tk == TK_SUB) { next(); emit(OP_IMM, -1); emit(OP_PSH, 0); primary(); emit(OP_MUL, 0); expr_type = TY_INT; }
    else if (tk == TK_INC || tk == TK_DEC) {
        t = tk; next(); primary();
        if ((code[code_pos - 1] & 0xFF) == OP_LI || (code[code_pos - 1] & 0xFF) == OP_LC) {
            code[code_pos - 1] = OP_PSH;
            if (expr_type == TY_CHAR) emit(OP_LC, 0); else emit(OP_LI, 0);
        }
        emit(OP_PSH, 0); emit(OP_IMM, (expr_type > TY_PTR) ? 8 : 1);
        if (t == TK_INC) emit(OP_ADD, 0); else emit(OP_SUB, 0);
        if (expr_type == TY_CHAR) emit(OP_SC, 0); else emit(OP_SI, 0);
    }
}

static void expr(int level) {
    int t, addr, sz;
    primary();
    while (tk >= level) {
        t = expr_type;
        if (tk == TK_ASSIGN) {
            next();
            if ((code[code_pos - 1] & 0xFF) == OP_LI || (code[code_pos - 1] & 0xFF) == OP_LC) code[code_pos - 1] = OP_PSH;
            expr(TK_ASSIGN);
            if (t == TY_CHAR) emit(OP_SC, 0); else emit(OP_SI, 0);
            expr_type = t;
        }
        else if (tk == TK_OR) { next(); addr = emit(OP_BNZ, 0); expr(TK_AND); patch_jmp(addr, code_pos * 8); expr_type = TY_INT; }
        else if (tk == TK_AND) { next(); addr = emit(OP_BZ, 0); expr(TK_EQ); patch_jmp(addr, code_pos * 8); expr_type = TY_INT; }
        else if (tk == TK_EQ) { next(); emit(OP_PSH, 0); expr(TK_LT); emit(OP_EQ, 0); expr_type = TY_INT; }
        else if (tk == TK_NE) { next(); emit(OP_PSH, 0); expr(TK_LT); emit(OP_NE, 0); expr_type = TY_INT; }
        else if (tk == TK_LT) { next(); emit(OP_PSH, 0); expr(TK_SHL); emit(OP_LT, 0); expr_type = TY_INT; }
        else if (tk == TK_GT) { next(); emit(OP_PSH, 0); expr(TK_SHL); emit(OP_GT, 0); expr_type = TY_INT; }
        else if (tk == TK_LE) { next(); emit(OP_PSH, 0); expr(TK_SHL); emit(OP_LE, 0); expr_type = TY_INT; }
        else if (tk == TK_GE) { next(); emit(OP_PSH, 0); expr(TK_SHL); emit(OP_GE, 0); expr_type = TY_INT; }
        else if (tk == TK_SHL) { next(); emit(OP_PSH, 0); expr(TK_ADD); emit(OP_SHL, 0); expr_type = TY_INT; }
        else if (tk == TK_SHR) { next(); emit(OP_PSH, 0); expr(TK_ADD); emit(OP_SHR, 0); expr_type = TY_INT; }
        else if (tk == TK_ADD) {
            next(); emit(OP_PSH, 0); expr(TK_MUL);
            sz = (t > TY_PTR) ? 8 : 1;
            if (sz > 1) { emit(OP_PSH, 0); emit(OP_IMM, sz); emit(OP_MUL, 0); }
            emit(OP_ADD, 0); expr_type = t;
        }
        else if (tk == TK_SUB) {
            next(); emit(OP_PSH, 0); expr(TK_MUL);
            sz = (t > TY_PTR) ? 8 : 1;
            if (sz > 1) { emit(OP_PSH, 0); emit(OP_IMM, sz); emit(OP_MUL, 0); }
            emit(OP_SUB, 0); expr_type = t;
        }
        else if (tk == TK_MUL) { next(); emit(OP_PSH, 0); expr(TK_INC); emit(OP_MUL, 0); expr_type = TY_INT; }
        else if (tk == TK_DIV) { next(); emit(OP_PSH, 0); expr(TK_INC); emit(OP_DIV, 0); expr_type = TY_INT; }
        else if (tk == TK_MOD) { next(); emit(OP_PSH, 0); expr(TK_INC); emit(OP_MOD, 0); expr_type = TY_INT; }
        else if (tk == TK_INC || tk == TK_DEC) {
            if ((code[code_pos - 1] & 0xFF) == OP_LI || (code[code_pos - 1] & 0xFF) == OP_LC) {
                code[code_pos - 1] = OP_PSH;
                if (t == TY_CHAR) emit(OP_LC, 0); else emit(OP_LI, 0);
            }
            emit(OP_PSH, 0); emit(OP_IMM, (t > TY_PTR) ? 8 : 1);
            if (tk == TK_INC) emit(OP_ADD, 0); else emit(OP_SUB, 0);
            if (t == TY_CHAR) emit(OP_SC, 0); else emit(OP_SI, 0);
            emit(OP_PSH, 0); emit(OP_IMM, (t > TY_PTR) ? 8 : 1);
            if (tk == TK_INC) emit(OP_SUB, 0); else emit(OP_ADD, 0);
            next(); expr_type = t;
        }
        else if (tk == TK_BRAK) {
            next(); emit(OP_PSH, 0); expr(TK_ASSIGN);
            if (t > TY_PTR) { emit(OP_PSH, 0); emit(OP_IMM, 8); emit(OP_MUL, 0); }
            emit(OP_ADD, 0); if (tk == ']') next();
            if (t == TY_PTR) emit(OP_LC, 0); else emit(OP_LI, 0);
            expr_type = (t >= TY_PTR) ? t - TY_PTR : TY_INT;
        }
        else break;
    }
}

/* ============ Statement Parser ============ */

static void stmt(void) {
    int addr1, addr2;
    if (tk == TK_IF) {
        next(); if (tk == '(') next(); expr(TK_ASSIGN); if (tk == ')') next();
        addr1 = emit(OP_BZ, 0); stmt();
        if (tk == TK_ELSE) { next(); addr2 = emit(OP_JMP, 0); patch_jmp(addr1, code_pos * 8); stmt(); patch_jmp(addr2, code_pos * 8); }
        else patch_jmp(addr1, code_pos * 8);
    }
    else if (tk == TK_WHILE) {
        next(); addr1 = code_pos * 8;
        if (tk == '(') next(); expr(TK_ASSIGN); if (tk == ')') next();
        addr2 = emit(OP_BZ, 0); stmt(); emit(OP_JMP, addr1); patch_jmp(addr2, code_pos * 8);
    }
    else if (tk == TK_RET) { next(); if (tk != ';') expr(TK_ASSIGN); emit(OP_LEV, 0); if (tk == ';') next(); }
    else if (tk == '{') { next(); while (tk != '}' && tk != TK_EOF) stmt(); if (tk == '}') next(); }
    else if (tk == ';') { next(); }
    else { expr(TK_ASSIGN); if (tk == ';') next(); }
}

/* ============ Declaration Parser ============ */

static void decl(void) {
    int base_type, t, sym_idx, params, enum_val, i;
    if (tk == TK_ENUM) {
        next(); if (tk == TK_ID) next();
        if (tk == '{') {
            next(); enum_val = 0;
            while (tk != '}' && tk != TK_EOF) {
                if (tk == TK_ID) { add_sym(tk_str, tk_len, 1, TY_INT, enum_val); next(); }
                if (tk == TK_ASSIGN) { next(); if (tk == TK_NUM) { enum_val = tk_val; next(); } }
                enum_val++; if (tk == ',') next();
            }
            if (tk == '}') next();
        }
        if (tk == ';') next(); return;
    }
    base_type = TY_INT;
    if (tk == TK_INT) { next(); base_type = TY_INT; }
    else if (tk == TK_CHAR) { next(); base_type = TY_CHAR; }
    while (tk != ';' && tk != TK_EOF) {
        t = base_type; while (tk == TK_MUL) { next(); t += TY_PTR; }
        if (tk != TK_ID) break;
        sym_idx = find_sym(tk_str, tk_len);
        if (sym_idx < 0) sym_idx = add_sym(tk_str, tk_len, 0, t, 0);
        next();
        if (tk == '(') {
            syms[sym_idx].cls = 2; syms[sym_idx].val = code_pos * 8;
            next(); local_offset = 16; params = 0;
            while (tk != ')' && tk != TK_EOF) {
                t = TY_INT;
                if (tk == TK_INT) { next(); t = TY_INT; }
                else if (tk == TK_CHAR) { next(); t = TY_CHAR; }
                while (tk == TK_MUL) { next(); t += TY_PTR; }
                if (tk == TK_ID) {
                    i = find_sym(tk_str, tk_len);
                    if (i >= 0) { syms[i].h_cls = syms[i].cls; syms[i].h_type = syms[i].type; syms[i].h_val = syms[i].val; }
                    else i = add_sym(tk_str, tk_len, 0, 0, 0);
                    syms[i].cls = 5; syms[i].type = t; syms[i].val = local_offset;
                    local_offset += 8; params++; next();
                }
                if (tk == ',') next();
            }
            if (tk == ')') next();
            for (i = 0; i < sym_count; i++) {
                if (syms[i].cls == 5 && syms[i].val >= 16)
                    syms[i].val = (params - 1 - (syms[i].val - 16) / 8) * 8 + 16;
            }
            local_offset = 0;
            if (tk == '{') {
                next();
                while (tk == TK_INT || tk == TK_CHAR) {
                    t = TY_INT; if (tk == TK_CHAR) t = TY_CHAR; next();
                    while (tk != ';' && tk != TK_EOF) {
                        while (tk == TK_MUL) { next(); t += TY_PTR; }
                        if (tk == TK_ID) {
                            i = find_sym(tk_str, tk_len);
                            if (i >= 0) { syms[i].h_cls = syms[i].cls; syms[i].h_type = syms[i].type; syms[i].h_val = syms[i].val; }
                            else i = add_sym(tk_str, tk_len, 0, 0, 0);
                            local_offset -= 8;
                            syms[i].cls = 5; syms[i].type = t; syms[i].val = local_offset; next();
                        }
                        if (tk == ',') { next(); t = base_type; } else break;
                    }
                    if (tk == ';') next();
                }
                emit(OP_ENT, -local_offset);
                while (tk != '}' && tk != TK_EOF) stmt();
                emit(OP_LEV, 0); if (tk == '}') next();
                for (i = 0; i < sym_count; i++) {
                    if (syms[i].h_cls) {
                        syms[i].cls = syms[i].h_cls; syms[i].type = syms[i].h_type;
                        syms[i].val = syms[i].h_val; syms[i].h_cls = 0;
                    }
                }
            }
            return;
        } else {
            syms[sym_idx].cls = 4; syms[sym_idx].val = data_pos + DATA_BASE;
            data_pos += 8;
        }
        if (tk == ',') next();
    }
    if (tk == ';') next();
}

/* ============ Compile ============ */

static void compile(char *source) {
    int main_idx;
    src = source; src_pos = 0; line = 1;
    code = (int *)malloc(POOL_SIZE * sizeof(int));
    data = (char *)malloc(POOL_SIZE);
    code_pos = 2; data_pos = 0;

    sym_count = 0;
    memset(syms, 0, sizeof(syms));

    /* Built-in syscalls */
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
    while (tk != TK_EOF) decl();

    main_idx = find_sym("main", 4);
    if (main_idx >= 0) {
        code[0] = OP_JSR | (syms[main_idx].val << 8);
        code[1] = OP_EXIT;
    }
}

/* ============ Read File ============ */

static char *read_file(const char *path, int *out_len) {
    FILE *f;
    char *buf;
    long len;

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);

    buf = (char *)malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = 0;
    fclose(f);

    *out_len = (int)len;
    return buf;
}

/* ============ Output Bundle ============ */

static void output_bundle(void) {
    int i, op, imm;

    printf("/*\n * Neural VM Bundle - Auto-generated\n *\n");
    printf(" * This file contains:\n");
    printf(" *   - Embedded model weights (%d bytes)\n", model_len);
    printf(" *   - Compiled bytecode (%d instructions)\n", code_pos);
    printf(" *   - Program data (%d bytes)\n", data_pos);
    printf(" *   - Full Neural VM runtime (ALL computation through neural weights)\n");
    printf(" *\n * Every arithmetic operation flows through the neural model's\n");
    printf(" * weight matrices via matmul + softmax + activation.\n */\n\n");

    /* Embedded model */
    printf("static unsigned char embedded_model[] = {\n");
    for (i = 0; i < model_len; i++) {
        if ((i & 15) == 0) printf("    ");
        printf("0x%02x", model_data[i]);
        if (i + 1 < model_len) printf(",");
        if ((i & 15) == 15 || i + 1 == model_len) printf("\n");
    }
    printf("};\nstatic int embedded_model_len = %d;\n\n", model_len);

    /* Bytecode */
    printf("static int program_code[][2] = {\n");
    for (i = 0; i < code_pos; i++) {
        op = code[i] & 0xFF;
        imm = code[i] >> 8;
        printf("    {%d, %d},\n", op, imm);
    }
    printf("};\nstatic int program_code_len = %d;\n\n", code_pos);

    /* Data segment */
    printf("static unsigned char program_data[] = {");
    if (data_pos == 0) {
        printf("0");
    } else {
        for (i = 0; i < data_pos; i++) {
            if ((i & 15) == 0) printf("\n    ");
            printf("0x%02x", (unsigned char)data[i]);
            if (i + 1 < data_pos) printf(",");
        }
        printf("\n");
    }
    printf("};\nstatic int program_data_len = %d;\n\n", data_pos);

    /* Emit runtime verbatim */
    fwrite(runtime_data, 1, runtime_len, stdout);
    printf("\n");
}

/* ============ Main ============ */

int main(int argc, char **argv) {
    char *source;
    int src_len;

    if (argc < 4) {
        fprintf(stderr, "Usage: %s model.c4onnx program.c optimized_runtime.c > bundled.c\n", argv[0]);
        return 1;
    }

    /* Read model */
    model_data = (unsigned char *)read_file(argv[1], &model_len);
    if (!model_data) return 1;

    /* Read and compile source */
    source = read_file(argv[2], &src_len);
    if (!source) return 1;

    /* Read runtime template */
    runtime_data = read_file(argv[3], &runtime_len);
    if (!runtime_data) return 1;

    compile(source);
    output_bundle();

    return 0;
}
