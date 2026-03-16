// Minimal C4 VM loop

int code[20];
int stack[100];
int ax;
int pc_idx;
int sp_idx;
int op;
int cycle;

int main() {
    // Program: 6 * 7 = 42
    code[0] = 1;   // IMM
    code[1] = 6;
    code[2] = 13;  // PSH
    code[3] = 0;
    code[4] = 1;   // IMM
    code[5] = 7;
    code[6] = 27;  // MUL
    code[7] = 0;
    code[8] = 38;  // EXIT
    
    pc_idx = 0;
    sp_idx = 99;
    ax = 0;
    cycle = 0;
    
    while (cycle < 50) {
        op = code[pc_idx];
        pc_idx = pc_idx + 1;
        
        if (op == 1) {
            ax = code[pc_idx];
            pc_idx = pc_idx + 1;
        }
        if (op == 13) {
            stack[sp_idx] = ax;
            sp_idx = sp_idx - 1;
        }
        if (op == 27) {
            sp_idx = sp_idx + 1;
            ax = stack[sp_idx] * ax;
        }
        if (op == 38) {
            return ax;
        }
        cycle = cycle + 1;
    }
    return ax;
}
