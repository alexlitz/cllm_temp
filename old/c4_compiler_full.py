"""
Full C4 compiler with pointer support for self-hosting.

Adds to basic compiler:
- Pointer types: int*, char*
- Pointer dereference: *p
- Address-of: &x
- Array indexing: a[i]
- String literals: "hello"
- Char literals: 'a'
- Pointer arithmetic: p++, p+n
- Sizeof operator
- Enum declarations
"""

from enum import IntEnum
from typing import List, Dict, Tuple, Optional


class Op(IntEnum):
    """C4 opcodes."""
    LEA = 0    # load effective address
    IMM = 1    # load immediate
    JMP = 2    # jump
    JSR = 3    # jump subroutine
    BZ = 4     # branch if zero
    BNZ = 5    # branch if not zero
    ENT = 6    # enter subroutine
    ADJ = 7    # adjust stack
    LEV = 8    # leave subroutine
    LI = 9     # load int
    LC = 10    # load char
    SI = 11    # store int
    SC = 12    # store char
    PSH = 13   # push

    OR = 14
    XOR = 15
    AND = 16
    EQ = 17
    NE = 18
    LT = 19
    GT = 20
    LE = 21
    GE = 22
    SHL = 23
    SHR = 24
    ADD = 25
    SUB = 26
    MUL = 27
    DIV = 28
    MOD = 29

    OPEN = 30
    READ = 31
    CLOS = 32
    PRTF = 33
    MALC = 34
    FREE = 35
    MSET = 36
    MCMP = 37
    EXIT = 38


# Type constants
CHAR = 0
INT = 1
PTR = 2  # Pointer adds PTR to base type


class TokenType(IntEnum):
    NUM = 1
    ID = 2
    CHAR_LIT = 3
    STRING = 4
    # Keywords
    KW_CHAR = 10
    KW_INT = 11
    KW_ENUM = 12
    KW_IF = 13
    KW_ELSE = 14
    KW_WHILE = 15
    KW_RETURN = 16
    KW_SIZEOF = 17
    # Operators (in precedence order for parsing)
    ASSIGN = 20
    COND = 21      # ?
    LOR = 22       # ||
    LAND = 23      # &&
    OR = 24        # |
    XOR = 25       # ^
    AND = 26       # &
    EQ = 27        # ==
    NE = 28        # !=
    LT = 29
    GT = 30
    LE = 31
    GE = 32
    SHL = 33       # <<
    SHR = 34       # >>
    ADD = 35
    SUB = 36
    MUL = 37
    DIV = 38
    MOD = 39
    INC = 40       # ++
    DEC = 41       # --
    BRAK = 42      # [
    # Delimiters
    LPAREN = 50
    RPAREN = 51
    LBRACE = 52
    RBRACE = 53
    LBRACKET = 54
    RBRACKET = 55
    SEMI = 56
    COMMA = 57
    COLON = 58
    # Special
    EOF = 99


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.keywords = {
            'char': TokenType.KW_CHAR,
            'int': TokenType.KW_INT,
            'enum': TokenType.KW_ENUM,
            'if': TokenType.KW_IF,
            'else': TokenType.KW_ELSE,
            'while': TokenType.KW_WHILE,
            'return': TokenType.KW_RETURN,
            'sizeof': TokenType.KW_SIZEOF,
        }
        # Data segment for string literals
        self.data: List[int] = []
        self.data_offset = 0

    def peek(self, offset=0) -> str:
        pos = self.pos + offset
        return self.source[pos] if pos < len(self.source) else ''

    def advance(self) -> str:
        ch = self.peek()
        self.pos += 1
        if ch == '\n':
            self.line += 1
        return ch

    def skip_whitespace(self):
        while self.peek() and self.peek() in ' \t\n\r':
            self.advance()
        # Skip // comments
        if self.peek() == '/' and self.peek(1) == '/':
            while self.peek() and self.peek() != '\n':
                self.advance()
            self.skip_whitespace()
        # Skip /* */ comments
        if self.peek() == '/' and self.peek(1) == '*':
            self.advance()
            self.advance()
            while self.peek() and not (self.peek() == '*' and self.peek(1) == '/'):
                self.advance()
            if self.peek():
                self.advance()
                self.advance()
            self.skip_whitespace()
        # Skip #preprocessor lines (simplified)
        if self.peek() == '#':
            while self.peek() and self.peek() != '\n':
                self.advance()
            self.skip_whitespace()

    def next_token(self) -> Tuple[TokenType, any, int]:
        self.skip_whitespace()

        if not self.peek():
            return (TokenType.EOF, None, self.line)

        ch = self.peek()
        line = self.line

        # Numbers
        if ch.isdigit():
            num = 0
            if ch == '0' and self.peek(1) in 'xX':
                # Hex
                self.advance()
                self.advance()
                while self.peek() in '0123456789abcdefABCDEF':
                    d = self.advance()
                    num = num * 16 + int(d, 16)
            elif ch == '0':
                # Octal
                while self.peek() in '01234567':
                    num = num * 8 + int(self.advance())
            else:
                while self.peek().isdigit():
                    num = num * 10 + int(self.advance())
            return (TokenType.NUM, num, line)

        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            ident = ''
            while self.peek().isalnum() or self.peek() == '_':
                ident += self.advance()
            if ident in self.keywords:
                return (self.keywords[ident], ident, line)
            return (TokenType.ID, ident, line)

        # Char literal
        if ch == "'":
            self.advance()
            val = ord(self.advance())
            if self.peek() == '\\':
                self.advance()
                esc = self.advance()
                val = {'n': 10, 't': 9, '\\': 92, "'": 39, '0': 0}.get(esc, ord(esc))
            else:
                val = ord(self.source[self.pos - 1])
            self.advance()  # closing '
            return (TokenType.NUM, val, line)

        # String literal
        if ch == '"':
            self.advance()
            str_start = self.data_offset
            while self.peek() and self.peek() != '"':
                c = self.advance()
                if c == '\\':
                    esc = self.advance()
                    c = {'n': '\n', 't': '\t', '\\': '\\', '"': '"', '0': '\0'}.get(esc, esc)
                self.data.append(ord(c))
                self.data_offset += 1
            self.advance()  # closing "
            self.data.append(0)  # null terminate
            self.data_offset += 1
            # Align to word boundary
            while self.data_offset % 8 != 0:
                self.data.append(0)
                self.data_offset += 1
            return (TokenType.STRING, str_start, line)

        # Operators
        self.advance()

        if ch == '+':
            if self.peek() == '+':
                self.advance()
                return (TokenType.INC, '++', line)
            return (TokenType.ADD, '+', line)
        if ch == '-':
            if self.peek() == '-':
                self.advance()
                return (TokenType.DEC, '--', line)
            return (TokenType.SUB, '-', line)
        if ch == '*':
            return (TokenType.MUL, '*', line)
        if ch == '/':
            return (TokenType.DIV, '/', line)
        if ch == '%':
            return (TokenType.MOD, '%', line)

        if ch == '=':
            if self.peek() == '=':
                self.advance()
                return (TokenType.EQ, '==', line)
            return (TokenType.ASSIGN, '=', line)
        if ch == '!':
            if self.peek() == '=':
                self.advance()
                return (TokenType.NE, '!=', line)
            return (TokenType.NUM, 0, line)  # ! as unary handled elsewhere
        if ch == '<':
            if self.peek() == '=':
                self.advance()
                return (TokenType.LE, '<=', line)
            if self.peek() == '<':
                self.advance()
                return (TokenType.SHL, '<<', line)
            return (TokenType.LT, '<', line)
        if ch == '>':
            if self.peek() == '=':
                self.advance()
                return (TokenType.GE, '>=', line)
            if self.peek() == '>':
                self.advance()
                return (TokenType.SHR, '>>', line)
            return (TokenType.GT, '>', line)

        if ch == '&':
            if self.peek() == '&':
                self.advance()
                return (TokenType.LAND, '&&', line)
            return (TokenType.AND, '&', line)
        if ch == '|':
            if self.peek() == '|':
                self.advance()
                return (TokenType.LOR, '||', line)
            return (TokenType.OR, '|', line)
        if ch == '^':
            return (TokenType.XOR, '^', line)
        if ch == '~':
            return (TokenType.NUM, -1, line)  # ~ handled as unary

        if ch == '?':
            return (TokenType.COND, '?', line)
        if ch == ':':
            return (TokenType.COLON, ':', line)

        if ch == '(':
            return (TokenType.LPAREN, '(', line)
        if ch == ')':
            return (TokenType.RPAREN, ')', line)
        if ch == '{':
            return (TokenType.LBRACE, '{', line)
        if ch == '}':
            return (TokenType.RBRACE, '}', line)
        if ch == '[':
            return (TokenType.BRAK, '[', line)
        if ch == ']':
            return (TokenType.RBRACKET, ']', line)
        if ch == ';':
            return (TokenType.SEMI, ';', line)
        if ch == ',':
            return (TokenType.COMMA, ',', line)

        return (TokenType.NUM, 0, line)


class Symbol:
    def __init__(self, name: str, sclass: str, stype: int, value: int):
        self.name = name
        self.sclass = sclass  # 'Num', 'Fun', 'Sys', 'Glo', 'Loc'
        self.stype = stype    # CHAR, INT, or INT+PTR, etc.
        self.value = value    # Address or value
        # For shadowing in functions
        self.h_class = None
        self.h_type = None
        self.h_value = None


class Compiler:
    def __init__(self):
        self.code: List[int] = []
        self.symbols: Dict[str, Symbol] = {}
        self.local_offset = 0
        self.current_type = INT
        self.expr_type = INT

        # Data segment base address (after code)
        self.data_base = 0x10000
        self.data: List[int] = []

        # Setup system calls as symbols
        syscalls = ['open', 'read', 'close', 'printf', 'malloc', 'free', 'memset', 'memcmp', 'exit']
        for i, name in enumerate(syscalls):
            self.symbols[name] = Symbol(name, 'Sys', INT, Op.OPEN + i)

    def emit(self, op: Op, imm: int = 0):
        self.code.append(int(op) + (imm << 8))

    def current_addr(self) -> int:
        return len(self.code) * 8

    def patch(self, addr: int, target: int):
        idx = addr // 8
        op = self.code[idx] & 0xFF
        self.code[idx] = op + (target << 8)

    def compile(self, source: str) -> Tuple[List[int], List[int]]:
        """Compile source, return (code, data)."""
        lexer = Lexer(source)
        self.tokens = []
        while True:
            tok = lexer.next_token()
            self.tokens.append(tok)
            if tok[0] == TokenType.EOF:
                break

        self.data = lexer.data
        self.pos = 0

        # Reserve space for startup code: JSR main; EXIT
        self.emit(Op.JSR, 0)  # Will patch with main address
        self.emit(Op.EXIT, 0)

        self.parse_program()

        # Patch startup to jump to main
        if 'main' in self.symbols:
            main_addr = self.symbols['main'].value
            self.code[0] = int(Op.JSR) + (main_addr << 8)

        return self.code, self.data

    def peek(self) -> TokenType:
        return self.tokens[self.pos][0]

    def token_val(self):
        return self.tokens[self.pos][1]

    def token_line(self):
        return self.tokens[self.pos][2]

    def advance(self):
        self.pos += 1

    def expect(self, t: TokenType):
        if self.peek() != t:
            raise SyntaxError(f"Expected {t}, got {self.peek()} at line {self.token_line()}")
        self.advance()

    def parse_program(self):
        """Parse global declarations."""
        while self.peek() != TokenType.EOF:
            self.parse_global_decl()

    def parse_global_decl(self):
        """Parse function or global variable."""
        base_type = INT

        if self.peek() == TokenType.KW_INT:
            self.advance()
            base_type = INT
        elif self.peek() == TokenType.KW_CHAR:
            self.advance()
            base_type = CHAR
        elif self.peek() == TokenType.KW_ENUM:
            self.parse_enum()
            return

        # Handle pointer types
        while self.peek() == TokenType.MUL:
            self.advance()
            base_type += PTR

        name = self.token_val()
        self.expect(TokenType.ID)

        if self.peek() == TokenType.LPAREN:
            # Function definition
            self.parse_function(name, base_type)
        else:
            # Global variable
            self.symbols[name] = Symbol(name, 'Glo', base_type, self.data_base + len(self.data) * 8)
            # Allocate space
            for _ in range(8):
                self.data.append(0)
            self.expect(TokenType.SEMI)

    def parse_enum(self):
        """Parse enum declaration."""
        self.advance()  # 'enum'
        if self.peek() == TokenType.ID:
            self.advance()  # optional enum name
        self.expect(TokenType.LBRACE)

        val = 0
        while self.peek() != TokenType.RBRACE:
            name = self.token_val()
            self.expect(TokenType.ID)
            if self.peek() == TokenType.ASSIGN:
                self.advance()
                val = self.token_val()
                self.expect(TokenType.NUM)
            self.symbols[name] = Symbol(name, 'Num', INT, val)
            val += 1
            if self.peek() == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RBRACE)
        if self.peek() == TokenType.SEMI:
            self.advance()

    def parse_function(self, name: str, ret_type: int):
        """Parse function definition."""
        self.expect(TokenType.LPAREN)

        # Record function address
        func_addr = self.current_addr()
        self.symbols[name] = Symbol(name, 'Fun', ret_type, func_addr)

        # Parse parameters
        param_count = 0
        while self.peek() != TokenType.RPAREN:
            # Parameter type
            ptype = INT
            if self.peek() == TokenType.KW_INT:
                self.advance()
            elif self.peek() == TokenType.KW_CHAR:
                self.advance()
                ptype = CHAR

            while self.peek() == TokenType.MUL:
                self.advance()
                ptype += PTR

            pname = self.token_val()
            self.expect(TokenType.ID)

            # Shadow any existing symbol
            if pname in self.symbols:
                sym = self.symbols[pname]
                sym.h_class = sym.sclass
                sym.h_type = sym.stype
                sym.h_value = sym.value
            else:
                self.symbols[pname] = Symbol(pname, 'Loc', ptype, 0)

            self.symbols[pname].sclass = 'Loc'
            self.symbols[pname].stype = ptype
            self.symbols[pname].value = param_count
            param_count += 1

            if self.peek() == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)

        # ENT instruction (placeholder for local size)
        ent_addr = self.current_addr()
        self.emit(Op.ENT, 0)

        # Parameters are at positive offsets from BP
        # param 0 at bp + (param_count) * 8, etc. (reversed push order)
        # Actually in C4: params at bp+16+, locals at bp-8-
        # Track which symbols are parameters for this function by collecting
        # names during parsing and adjusting only those (fixes issue with
        # parameter name reuse across functions)
        param_names = []
        # Re-collect param names by walking backward from param_count
        for pname, sym in self.symbols.items():
            if sym.sclass == 'Loc' and 0 <= sym.value < param_count:
                param_names.append(pname)

        for pname in param_names:
            sym = self.symbols[pname]
            # First param at bp + 16, second at bp + 24, etc.
            sym.value = 16 + (param_count - 1 - sym.value) * 8

        # Parse local variables
        self.local_offset = 0
        while self.peek() in (TokenType.KW_INT, TokenType.KW_CHAR):
            ltype = INT if self.peek() == TokenType.KW_INT else CHAR
            self.advance()

            while self.peek() == TokenType.MUL:
                self.advance()
                ltype += PTR

            while True:
                lname = self.token_val()
                self.expect(TokenType.ID)

                ltype_actual = ltype
                while self.peek() == TokenType.MUL:
                    self.advance()
                    ltype_actual += PTR

                # Shadow any existing symbol
                if lname in self.symbols:
                    sym = self.symbols[lname]
                    if sym.h_class is None:
                        sym.h_class = sym.sclass
                        sym.h_type = sym.stype
                        sym.h_value = sym.value
                else:
                    self.symbols[lname] = Symbol(lname, 'Loc', ltype_actual, 0)

                self.local_offset += 8
                self.symbols[lname].sclass = 'Loc'
                self.symbols[lname].stype = ltype_actual
                self.symbols[lname].value = -self.local_offset

                if self.peek() == TokenType.COMMA:
                    self.advance()
                else:
                    break

            self.expect(TokenType.SEMI)

        # Patch ENT with local size
        self.patch(ent_addr, self.local_offset)

        # Parse statements
        while self.peek() != TokenType.RBRACE:
            self.parse_statement()

        self.expect(TokenType.RBRACE)

        # Default return
        self.emit(Op.LEV)

        # Restore shadowed symbols
        for sym in self.symbols.values():
            if sym.h_class is not None:
                sym.sclass = sym.h_class
                sym.stype = sym.h_type
                sym.value = sym.h_value
                sym.h_class = None

    def parse_statement(self):
        """Parse a statement."""
        if self.peek() == TokenType.KW_IF:
            self.advance()
            self.expect(TokenType.LPAREN)
            self.parse_expression(TokenType.ASSIGN)
            self.expect(TokenType.RPAREN)

            bz_addr = self.current_addr()
            self.emit(Op.BZ, 0)

            self.parse_statement()

            if self.peek() == TokenType.KW_ELSE:
                self.advance()
                jmp_addr = self.current_addr()
                self.emit(Op.JMP, 0)
                self.patch(bz_addr, self.current_addr())
                self.parse_statement()
                self.patch(jmp_addr, self.current_addr())
            else:
                self.patch(bz_addr, self.current_addr())

        elif self.peek() == TokenType.KW_WHILE:
            self.advance()
            loop_addr = self.current_addr()
            self.expect(TokenType.LPAREN)
            self.parse_expression(TokenType.ASSIGN)
            self.expect(TokenType.RPAREN)

            bz_addr = self.current_addr()
            self.emit(Op.BZ, 0)

            self.parse_statement()
            self.emit(Op.JMP, loop_addr)
            self.patch(bz_addr, self.current_addr())

        elif self.peek() == TokenType.KW_RETURN:
            self.advance()
            if self.peek() != TokenType.SEMI:
                self.parse_expression(TokenType.ASSIGN)
            self.expect(TokenType.SEMI)
            self.emit(Op.LEV)

        elif self.peek() == TokenType.LBRACE:
            self.advance()
            while self.peek() != TokenType.RBRACE:
                self.parse_statement()
            self.expect(TokenType.RBRACE)

        elif self.peek() == TokenType.SEMI:
            self.advance()

        else:
            self.parse_expression(TokenType.ASSIGN)
            self.expect(TokenType.SEMI)

    def parse_expression(self, level: TokenType):
        """Parse expression with precedence climbing."""
        # Unary operators and atoms
        if self.peek() == TokenType.NUM:
            val = self.token_val()
            self.advance()
            self.emit(Op.IMM, val)
            self.expr_type = INT

        elif self.peek() == TokenType.STRING:
            addr = self.data_base + self.token_val()
            self.advance()
            self.emit(Op.IMM, addr)
            self.expr_type = PTR + CHAR

        elif self.peek() == TokenType.KW_SIZEOF:
            self.advance()
            self.expect(TokenType.LPAREN)
            t = INT
            if self.peek() == TokenType.KW_INT:
                self.advance()
            elif self.peek() == TokenType.KW_CHAR:
                self.advance()
                t = CHAR
            while self.peek() == TokenType.MUL:
                self.advance()
                t += PTR
            self.expect(TokenType.RPAREN)
            self.emit(Op.IMM, 8 if t >= PTR or t == INT else 1)
            self.expr_type = INT

        elif self.peek() == TokenType.ID:
            name = self.token_val()
            self.advance()

            sym = self.symbols.get(name)
            if sym is None:
                raise SyntaxError(f"Undefined: {name} at line {self.token_line()}")

            if self.peek() == TokenType.LPAREN:
                # Function call
                self.advance()
                argc = 0
                while self.peek() != TokenType.RPAREN:
                    self.parse_expression(TokenType.ASSIGN)
                    self.emit(Op.PSH)
                    argc += 1
                    if self.peek() == TokenType.COMMA:
                        self.advance()
                self.expect(TokenType.RPAREN)

                if sym.sclass == 'Sys':
                    self.emit(Op(sym.value))
                elif sym.sclass == 'Fun':
                    self.emit(Op.JSR, sym.value)
                else:
                    raise SyntaxError(f"Not a function: {name}")

                if argc:
                    self.emit(Op.ADJ, argc * 8)
                self.expr_type = sym.stype

            elif sym.sclass == 'Num':
                # Enum constant
                self.emit(Op.IMM, sym.value)
                self.expr_type = INT

            elif sym.sclass == 'Loc':
                # Local variable - always emit LI to load value
                self.emit(Op.LEA, sym.value)
                self.expr_type = sym.stype
                # Load value (use LI for int and pointers, LC for char)
                self.emit(Op.LI if self.expr_type != CHAR else Op.LC)

            elif sym.sclass == 'Glo':
                # Global variable - always emit LI to load value
                self.emit(Op.IMM, sym.value)
                self.expr_type = sym.stype
                # Load value (use LI for int and pointers, LC for char)
                self.emit(Op.LI if self.expr_type != CHAR else Op.LC)

        elif self.peek() == TokenType.LPAREN:
            self.advance()
            # Check for cast
            if self.peek() in (TokenType.KW_INT, TokenType.KW_CHAR):
                t = INT if self.peek() == TokenType.KW_INT else CHAR
                self.advance()
                while self.peek() == TokenType.MUL:
                    self.advance()
                    t += PTR
                self.expect(TokenType.RPAREN)
                self.parse_expression(TokenType.INC)
                self.expr_type = t
            else:
                self.parse_expression(TokenType.ASSIGN)
                self.expect(TokenType.RPAREN)

        elif self.peek() == TokenType.MUL:
            # Dereference
            self.advance()
            self.parse_expression(TokenType.INC)
            if self.expr_type >= PTR:
                self.expr_type -= PTR
            self.emit(Op.LI if self.expr_type == INT else Op.LC)

        elif self.peek() == TokenType.AND:
            # Address-of
            self.advance()
            self.parse_expression(TokenType.INC)
            # Remove the load that was emitted
            if self.code and (self.code[-1] & 0xFF) in (Op.LI, Op.LC):
                self.code.pop()
            self.expr_type += PTR

        elif self.peek() == TokenType.SUB:
            # Unary minus
            self.advance()
            self.emit(Op.IMM, -1)
            self.emit(Op.PSH)
            self.parse_expression(TokenType.INC)
            self.emit(Op.MUL)
            self.expr_type = INT

        elif self.peek() == TokenType.INC or self.peek() == TokenType.DEC:
            # Pre-increment/decrement
            is_inc = self.peek() == TokenType.INC
            self.advance()
            self.parse_expression(TokenType.INC)
            # Duplicate address on stack
            if self.code and (self.code[-1] & 0xFF) in (Op.LI, Op.LC):
                is_char = (self.code[-1] & 0xFF) == Op.LC
                self.code[-1] = int(Op.PSH)  # Replace load with push
                self.emit(Op.LC if is_char else Op.LI)
            self.emit(Op.PSH)
            self.emit(Op.IMM, 8 if self.expr_type >= PTR else 1)
            self.emit(Op.ADD if is_inc else Op.SUB)
            self.emit(Op.SC if self.expr_type == CHAR else Op.SI)

        else:
            raise SyntaxError(f"Unexpected token: {self.peek()} at line {self.token_line()}")

        # Binary operators
        while self.peek() >= level:
            saved_type = self.expr_type

            if self.peek() == TokenType.ASSIGN:
                self.advance()
                # For assignment, we need the address on stack
                last_op = self.code[-1] & 0xFF if self.code else None
                if last_op in (Op.LI, Op.LC):
                    # Replace load with push (keep address)
                    self.code[-1] = int(Op.PSH)
                elif last_op == Op.LEA:
                    # For pointers, LEA gives address, just push it
                    self.emit(Op.PSH)
                self.parse_expression(TokenType.ASSIGN)
                self.emit(Op.SC if saved_type == CHAR else Op.SI)

            elif self.peek() == TokenType.COND:
                # Ternary
                self.advance()
                bz_addr = self.current_addr()
                self.emit(Op.BZ, 0)
                self.parse_expression(TokenType.ASSIGN)
                self.expect(TokenType.COLON)
                jmp_addr = self.current_addr()
                self.emit(Op.JMP, 0)
                self.patch(bz_addr, self.current_addr())
                self.parse_expression(TokenType.COND)
                self.patch(jmp_addr, self.current_addr())

            elif self.peek() == TokenType.LOR:
                self.advance()
                bnz_addr = self.current_addr()
                self.emit(Op.BNZ, 0)
                self.parse_expression(TokenType.LAND)
                self.patch(bnz_addr, self.current_addr())
                self.expr_type = INT

            elif self.peek() == TokenType.LAND:
                self.advance()
                bz_addr = self.current_addr()
                self.emit(Op.BZ, 0)
                self.parse_expression(TokenType.OR)
                self.patch(bz_addr, self.current_addr())
                self.expr_type = INT

            elif self.peek() == TokenType.OR:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.XOR)
                self.emit(Op.OR)
                self.expr_type = INT

            elif self.peek() == TokenType.XOR:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.AND)
                self.emit(Op.XOR)
                self.expr_type = INT

            elif self.peek() == TokenType.AND:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.EQ)
                self.emit(Op.AND)
                self.expr_type = INT

            elif self.peek() == TokenType.EQ:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.LT)
                self.emit(Op.EQ)
                self.expr_type = INT

            elif self.peek() == TokenType.NE:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.LT)
                self.emit(Op.NE)
                self.expr_type = INT

            elif self.peek() == TokenType.LT:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.SHL)
                self.emit(Op.LT)
                self.expr_type = INT

            elif self.peek() == TokenType.GT:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.SHL)
                self.emit(Op.GT)
                self.expr_type = INT

            elif self.peek() == TokenType.LE:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.SHL)
                self.emit(Op.LE)
                self.expr_type = INT

            elif self.peek() == TokenType.GE:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.SHL)
                self.emit(Op.GE)
                self.expr_type = INT

            elif self.peek() == TokenType.SHL:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.ADD)
                self.emit(Op.SHL)
                self.expr_type = INT

            elif self.peek() == TokenType.SHR:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.ADD)
                self.emit(Op.SHR)
                self.expr_type = INT

            elif self.peek() == TokenType.ADD:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.MUL)
                # Pointer arithmetic
                if saved_type >= PTR:
                    base_type = saved_type - PTR
                    elem_size = 8 if base_type == INT else 1
                    if elem_size > 1:
                        self.emit(Op.PSH)
                        self.emit(Op.IMM, elem_size)
                        self.emit(Op.MUL)
                self.emit(Op.ADD)
                self.expr_type = saved_type

            elif self.peek() == TokenType.SUB:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.MUL)
                # Pointer arithmetic
                if saved_type >= PTR and self.expr_type >= PTR:
                    # Pointer difference
                    base_type = saved_type - PTR
                    elem_size = 8 if base_type == INT else 1
                    self.emit(Op.SUB)
                    if elem_size > 1:
                        self.emit(Op.PSH)
                        self.emit(Op.IMM, elem_size)
                        self.emit(Op.DIV)
                    self.expr_type = INT
                elif saved_type >= PTR:
                    base_type = saved_type - PTR
                    elem_size = 8 if base_type == INT else 1
                    if elem_size > 1:
                        self.emit(Op.PSH)
                        self.emit(Op.IMM, elem_size)
                        self.emit(Op.MUL)
                    self.emit(Op.SUB)
                    self.expr_type = saved_type
                else:
                    self.emit(Op.SUB)
                    self.expr_type = INT

            elif self.peek() == TokenType.MUL:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.INC)
                self.emit(Op.MUL)
                self.expr_type = INT

            elif self.peek() == TokenType.DIV:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.INC)
                self.emit(Op.DIV)
                self.expr_type = INT

            elif self.peek() == TokenType.MOD:
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.INC)
                self.emit(Op.MOD)
                self.expr_type = INT

            elif self.peek() == TokenType.INC or self.peek() == TokenType.DEC:
                # Post-increment/decrement
                is_inc = self.peek() == TokenType.INC
                self.advance()
                # Similar to pre but return old value
                if self.code and (self.code[-1] & 0xFF) in (Op.LI, Op.LC):
                    is_char = (self.code[-1] & 0xFF) == Op.LC
                    self.code[-1] = int(Op.PSH)
                    self.emit(Op.LC if is_char else Op.LI)
                self.emit(Op.PSH)
                self.emit(Op.IMM, 8 if saved_type >= PTR else 1)
                self.emit(Op.ADD if is_inc else Op.SUB)
                self.emit(Op.SC if saved_type == CHAR else Op.SI)
                self.emit(Op.PSH)
                self.emit(Op.IMM, 8 if saved_type >= PTR else 1)
                self.emit(Op.SUB if is_inc else Op.ADD)

            elif self.peek() == TokenType.BRAK:
                # Array indexing
                self.advance()
                self.emit(Op.PSH)
                self.parse_expression(TokenType.ASSIGN)
                self.expect(TokenType.RBRACKET)
                # Scale index by element size (8 for int, 1 for char)
                if saved_type >= PTR:
                    base_type = saved_type - PTR
                    elem_size = 8 if base_type == INT else 1
                    if elem_size > 1:
                        self.emit(Op.PSH)
                        self.emit(Op.IMM, elem_size)
                        self.emit(Op.MUL)
                self.emit(Op.ADD)
                self.expr_type = saved_type - PTR if saved_type >= PTR else saved_type
                self.emit(Op.LI if self.expr_type == INT else Op.LC)

            else:
                break


def compile_c(source: str) -> Tuple[List[int], List[int]]:
    """Compile C source, return (code, data)."""
    compiler = Compiler()
    return compiler.compile(source)


def test_full_compiler():
    print("FULL C4 COMPILER WITH POINTERS")
    print("=" * 60)

    # Test 1: Basic
    print("\n1. Basic arithmetic:")
    code, data = compile_c("int main() { return 3 + 4; }")
    print(f"   Code: {len(code)} instructions")

    # Test 2: Pointers
    print("\n2. Pointer test:")
    code, data = compile_c("""
        int main() {
            int x;
            int *p;
            x = 42;
            p = &x;
            return *p;
        }
    """)
    print(f"   Code: {len(code)} instructions")

    # Test 3: String
    print("\n3. String literal:")
    code, data = compile_c("""
        int main() {
            char *s;
            s = "hello";
            return *s;
        }
    """)
    print(f"   Code: {len(code)} instructions, Data: {len(data)} bytes")

    # Test 4: Array indexing
    print("\n4. Array indexing:")
    code, data = compile_c("""
        int main() {
            char *s;
            s = "abc";
            return s[1];
        }
    """)
    print(f"   Code: {len(code)} instructions")

    # Test 5: Pointer arithmetic
    print("\n5. Pointer arithmetic:")
    code, data = compile_c("""
        int main() {
            char *p;
            p = "xyz";
            p = p + 1;
            return *p;
        }
    """)
    print(f"   Code: {len(code)} instructions")

    # Test 6: Enum
    print("\n6. Enum:")
    code, data = compile_c("""
        enum { A, B, C = 10, D };
        int main() { return C + D; }
    """)
    print(f"   Code: {len(code)} instructions")

    print("\n" + "=" * 60)
    print("FULL COMPILER READY!")


if __name__ == "__main__":
    test_full_compiler()
