// c4.c preprocessed - removed #define int long long, simplified for our compiler

char *p, *lp, *data;
int *e, *le, *id, *sym;
int tk, ival, ty, loc, line, src, debug;

// Token enum values
enum {
  Num, Fun, Sys, Glo, Loc, Id,
  Char, Else, Enum, If, Int, Return, Sizeof, While,
  Assign, Cond, Lor, Lan, Or, Xor, And, Eq, Ne, Lt, Gt, Le, Ge, Shl, Shr, Add, Sub, Mul, Div, Mod, Inc, Dec, Brak
};

// Opcodes
enum { LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV, LI, LC, SI, SC, PSH,
       OR_OP, XOR_OP, AND_OP, EQ_OP, NE_OP, LT_OP, GT_OP, LE_OP, GE_OP,
       SHL_OP, SHR_OP, ADD_OP, SUB_OP, MUL_OP, DIV_OP, MOD_OP,
       OPEN, READ, CLOS, PRTF, MALC, FREE, MSET, MCMP, EXIT_OP };

// Types  
enum { CHAR_T, INT_T, PTR_T };

// Symbol table offsets
enum { Tk_off, Hash, Name, Class, Type, Val, HClass, HType, HVal, Idsz };

// Minimal test - just the VM execution loop
int main() {
    int *pc;
    int *sp;
    int *bp;
    int ax;
    int cycle;
    int i;
    int t;
    
    // Simple program: compute 6 * 7
    // IMM 6, PSH, IMM 7, MUL, EXIT
    int code[10];
    code[0] = IMM;
    code[1] = 6;
    code[2] = PSH;
    code[3] = 0;
    code[4] = IMM;
    code[5] = 7;
    code[6] = MUL_OP;
    code[7] = 0;
    code[8] = EXIT_OP;
    code[9] = 0;
    
    pc = code;
    ax = 0;
    cycle = 0;
    
    while (1) {
        i = *pc;
        pc = pc + 1;
        
        if (i == IMM) { ax = *pc; pc = pc + 1; }
        else if (i == PSH) { sp = sp - 1; *sp = ax; }
        else if (i == MUL_OP) { ax = *sp * ax; sp = sp + 1; }
        else if (i == EXIT_OP) { return ax; }
        
        cycle = cycle + 1;
        if (cycle > 100) { return 0; }
    }
    return ax;
}
