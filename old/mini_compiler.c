// Mini expression compiler that runs on the transformer
// Compiles simple expressions like "6*7" to bytecode

int input[100];   // Input expression as ASCII codes
int output[100];  // Output bytecode
int ip;           // Input pointer
int op;           // Output pointer
int ch;           // Current char
int num;          // Current number

int peek() {
    return input[ip];
}

int advance() {
    ch = input[ip];
    ip = ip + 1;
    return ch;
}

int emit(int opcode) {
    output[op] = opcode;
    op = op + 1;
    return 0;
}

int emit2(int opcode, int imm) {
    output[op] = opcode + imm * 256;
    op = op + 1;
    return 0;
}

int parse_num() {
    num = 0;
    while (ch >= 48) {
        if (ch > 57) { return num; }
        num = num * 10 + ch - 48;
        advance();
    }
    return num;
}

int parse_expr();

int parse_atom() {
    if (ch >= 48) {
        if (ch <= 57) {
            parse_num();
            emit2(1, num);
            return 0;
        }
    }
    return 0;
}

int parse_term() {
    int oper;
    parse_atom();
    while (1) {
        if (ch == 42) {
            emit(13);
            oper = ch;
            advance();
            parse_atom();
            emit(27);
        }
        if (ch == 47) {
            emit(13);
            oper = ch;
            advance();
            parse_atom();
            emit(28);
        }
        if (ch != 42) {
            if (ch != 47) {
                return 0;
            }
        }
    }
    return 0;
}

int parse_expr() {
    int oper;
    parse_term();
    while (1) {
        if (ch == 43) {
            emit(13);
            oper = ch;
            advance();
            parse_term();
            emit(25);
        }
        if (ch == 45) {
            emit(13);
            oper = ch;
            advance();
            parse_term();
            emit(26);
        }
        if (ch != 43) {
            if (ch != 45) {
                return 0;
            }
        }
    }
    return 0;
}

int main() {
    // Input: "6*7" = [54, 42, 55, 0]
    input[0] = 54;
    input[1] = 42;
    input[2] = 55;
    input[3] = 0;
    
    ip = 0;
    op = 0;
    
    advance();
    parse_expr();
    emit(38);
    
    // Return first bytecode instruction for testing
    return output[0];
}
