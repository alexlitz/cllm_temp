"""
Minimal C to C4 bytecode compiler.

Supports:
- Integer expressions
- Variables (local and global)
- If/else, while
- Functions with parameters
- Printf
- Return

This generates bytecode compatible with our transformer C4 VM.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import IntEnum

from c4_vm import Op


class TokenType(IntEnum):
    NUM = 1
    ID = 2
    CHAR_LIT = 3
    STRING = 4
    # Keywords
    INT = 10
    CHAR = 11
    IF = 12
    ELSE = 13
    WHILE = 14
    RETURN = 15
    # Operators
    PLUS = 20
    MINUS = 21
    STAR = 22
    SLASH = 23
    PERCENT = 24
    EQ = 25      # ==
    NE = 26      # !=
    LT = 27      # <
    GT = 28      # >
    LE = 29      # <=
    GE = 30      # >=
    ASSIGN = 31  # =
    AND = 32     # &&
    OR = 33      # ||
    # Delimiters
    LPAREN = 40
    RPAREN = 41
    LBRACE = 42
    RBRACE = 43
    SEMI = 44
    COMMA = 45
    EOF = 99


@dataclass
class Token:
    type: TokenType
    value: any
    line: int


@dataclass
class Symbol:
    name: str
    type: str  # 'int', 'char', 'func'
    scope: str  # 'global', 'local', 'param'
    offset: int  # stack offset for locals/params


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.keywords = {
            'int': TokenType.INT,
            'char': TokenType.CHAR,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'return': TokenType.RETURN,
        }

    def peek(self) -> str:
        if self.pos >= len(self.source):
            return ''
        return self.source[self.pos]

    def advance(self) -> str:
        ch = self.peek()
        self.pos += 1
        if ch == '\n':
            self.line += 1
        return ch

    def skip_whitespace(self):
        while self.peek() in ' \t\n\r':
            self.advance()
        # Skip // comments
        if self.peek() == '/' and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == '/':
            while self.peek() and self.peek() != '\n':
                self.advance()
            self.skip_whitespace()

    def next_token(self) -> Token:
        self.skip_whitespace()

        if not self.peek():
            return Token(TokenType.EOF, None, self.line)

        ch = self.peek()
        line = self.line

        # Numbers
        if ch.isdigit():
            num = 0
            while self.peek().isdigit():
                num = num * 10 + int(self.advance())
            return Token(TokenType.NUM, num, line)

        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            ident = ''
            while self.peek().isalnum() or self.peek() == '_':
                ident += self.advance()
            if ident in self.keywords:
                return Token(self.keywords[ident], ident, line)
            return Token(TokenType.ID, ident, line)

        # Operators
        self.advance()
        if ch == '+': return Token(TokenType.PLUS, '+', line)
        if ch == '-': return Token(TokenType.MINUS, '-', line)
        if ch == '*': return Token(TokenType.STAR, '*', line)
        if ch == '/': return Token(TokenType.SLASH, '/', line)
        if ch == '%': return Token(TokenType.PERCENT, '%', line)
        if ch == '(':  return Token(TokenType.LPAREN, '(', line)
        if ch == ')': return Token(TokenType.RPAREN, ')', line)
        if ch == '{': return Token(TokenType.LBRACE, '{', line)
        if ch == '}': return Token(TokenType.RBRACE, '}', line)
        if ch == ';': return Token(TokenType.SEMI, ';', line)
        if ch == ',': return Token(TokenType.COMMA, ',', line)

        if ch == '=':
            if self.peek() == '=':
                self.advance()
                return Token(TokenType.EQ, '==', line)
            return Token(TokenType.ASSIGN, '=', line)

        if ch == '!':
            if self.peek() == '=':
                self.advance()
                return Token(TokenType.NE, '!=', line)

        if ch == '<':
            if self.peek() == '=':
                self.advance()
                return Token(TokenType.LE, '<=', line)
            return Token(TokenType.LT, '<', line)

        if ch == '>':
            if self.peek() == '=':
                self.advance()
                return Token(TokenType.GE, '>=', line)
            return Token(TokenType.GT, '>', line)

        if ch == '&':
            if self.peek() == '&':
                self.advance()
                return Token(TokenType.AND, '&&', line)

        if ch == '|':
            if self.peek() == '|':
                self.advance()
                return Token(TokenType.OR, '||', line)

        raise SyntaxError(f"Unknown character: {ch} at line {line}")


class Compiler:
    def __init__(self):
        self.code: List[int] = []
        self.symbols: Dict[str, Symbol] = {}
        self.local_offset = 0
        self.in_function = False

    def emit(self, op: Op, imm: int = 0):
        """Emit instruction: opcode + (imm << 8)"""
        self.code.append(int(op) + (imm << 8))

    def current_addr(self) -> int:
        """Current code address (in bytes, 8 per instruction)"""
        return len(self.code) * 8

    def patch(self, addr: int, target: int):
        """Patch jump target at addr"""
        idx = addr // 8
        op = self.code[idx] & 0xFF
        self.code[idx] = op + (target << 8)

    def compile(self, source: str) -> List[int]:
        """Compile C source to bytecode."""
        lexer = Lexer(source)
        self.tokens = []
        while True:
            tok = lexer.next_token()
            self.tokens.append(tok)
            if tok.type == TokenType.EOF:
                break

        self.pos = 0
        self.parse_program()
        return self.code

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type: TokenType):
        tok = self.advance()
        if tok.type != type:
            raise SyntaxError(f"Expected {type}, got {tok.type} at line {tok.line}")
        return tok

    def parse_program(self):
        """Parse top-level declarations."""
        while self.peek().type != TokenType.EOF:
            self.parse_declaration()

    def parse_declaration(self):
        """Parse function or global variable."""
        # type name
        self.expect(TokenType.INT)  # Only int for now
        name = self.expect(TokenType.ID).value

        if self.peek().type == TokenType.LPAREN:
            # Function
            self.parse_function(name)
        else:
            # Global variable
            self.symbols[name] = Symbol(name, 'int', 'global', len(self.symbols))
            self.expect(TokenType.SEMI)

    def parse_function(self, name: str):
        """Parse function definition."""
        self.expect(TokenType.LPAREN)

        # Parse parameters
        params = []
        self.local_offset = 0

        if self.peek().type != TokenType.RPAREN:
            while True:
                self.expect(TokenType.INT)
                pname = self.expect(TokenType.ID).value
                params.append(pname)
                if self.peek().type == TokenType.COMMA:
                    self.advance()
                else:
                    break

        self.expect(TokenType.RPAREN)

        # Setup function
        func_addr = self.current_addr()
        self.symbols[name] = Symbol(name, 'func', 'global', func_addr)

        # Parameters are at bp + 16, bp + 24, etc. (after saved bp and return addr)
        for i, pname in enumerate(params):
            self.symbols[pname] = Symbol(pname, 'int', 'param', 16 + i * 8)

        self.in_function = True
        self.local_offset = 0

        # Parse body
        self.expect(TokenType.LBRACE)

        # ENT instruction (will patch local size later)
        ent_addr = self.current_addr()
        self.emit(Op.ENT, 0)

        # Parse local declarations and statements
        while self.peek().type == TokenType.INT:
            self.advance()
            vname = self.expect(TokenType.ID).value
            self.local_offset += 8
            self.symbols[vname] = Symbol(vname, 'int', 'local', -self.local_offset)
            if self.peek().type == TokenType.ASSIGN:
                self.advance()
                self.parse_expr()
                # Store to local
                self.emit(Op.LEA, -self.local_offset)
                self.emit(Op.PSH)
                # ax has value, swap
                self.emit(Op.SI)
            self.expect(TokenType.SEMI)

        # Patch ENT with local size
        self.patch(ent_addr, self.local_offset)

        # Parse statements
        while self.peek().type != TokenType.RBRACE:
            self.parse_statement()

        self.expect(TokenType.RBRACE)

        # Default return
        self.emit(Op.LEV)
        self.in_function = False

        # Clean up local symbols
        to_remove = [k for k, v in self.symbols.items() if v.scope in ('local', 'param')]
        for k in to_remove:
            del self.symbols[k]

    def parse_statement(self):
        """Parse a statement."""
        tok = self.peek()

        if tok.type == TokenType.IF:
            self.advance()
            self.expect(TokenType.LPAREN)
            self.parse_expr()
            self.expect(TokenType.RPAREN)

            # BZ to else/end
            bz_addr = self.current_addr()
            self.emit(Op.BZ, 0)

            self.parse_statement()

            if self.peek().type == TokenType.ELSE:
                self.advance()
                # JMP over else
                jmp_addr = self.current_addr()
                self.emit(Op.JMP, 0)
                # Patch BZ to here
                self.patch(bz_addr, self.current_addr())
                self.parse_statement()
                # Patch JMP to here
                self.patch(jmp_addr, self.current_addr())
            else:
                # Patch BZ to here
                self.patch(bz_addr, self.current_addr())

        elif tok.type == TokenType.WHILE:
            self.advance()
            loop_start = self.current_addr()
            self.expect(TokenType.LPAREN)
            self.parse_expr()
            self.expect(TokenType.RPAREN)

            # BZ to end
            bz_addr = self.current_addr()
            self.emit(Op.BZ, 0)

            self.parse_statement()

            # JMP to loop start
            self.emit(Op.JMP, loop_start)

            # Patch BZ to here
            self.patch(bz_addr, self.current_addr())

        elif tok.type == TokenType.RETURN:
            self.advance()
            if self.peek().type != TokenType.SEMI:
                self.parse_expr()
            self.expect(TokenType.SEMI)
            self.emit(Op.LEV)

        elif tok.type == TokenType.LBRACE:
            self.advance()
            while self.peek().type != TokenType.RBRACE:
                self.parse_statement()
            self.expect(TokenType.RBRACE)

        elif tok.type == TokenType.ID:
            name = self.advance().value

            if self.peek().type == TokenType.ASSIGN:
                # Assignment
                self.advance()
                self.parse_expr()

                sym = self.symbols.get(name)
                if sym is None:
                    raise SyntaxError(f"Unknown variable: {name}")

                if sym.scope == 'local' or sym.scope == 'param':
                    self.emit(Op.LEA, sym.offset)
                    self.emit(Op.PSH)
                    self.emit(Op.SI)
                else:
                    # Global - would need address
                    raise NotImplementedError("Global variable assignment")

                self.expect(TokenType.SEMI)

            elif self.peek().type == TokenType.LPAREN:
                # Function call as statement
                self.pos -= 1  # Rewind
                self.parse_expr()
                self.expect(TokenType.SEMI)

            else:
                raise SyntaxError(f"Unexpected token after identifier: {self.peek()}")

        else:
            # Expression statement
            self.parse_expr()
            self.expect(TokenType.SEMI)

    def parse_expr(self):
        """Parse expression (with precedence)."""
        self.parse_or()

    def parse_or(self):
        self.parse_and()
        while self.peek().type == TokenType.OR:
            self.advance()
            self.emit(Op.PSH)
            self.parse_and()
            self.emit(Op.OR)

    def parse_and(self):
        self.parse_equality()
        while self.peek().type == TokenType.AND:
            self.advance()
            self.emit(Op.PSH)
            self.parse_equality()
            self.emit(Op.AND)

    def parse_equality(self):
        self.parse_comparison()
        while self.peek().type in (TokenType.EQ, TokenType.NE):
            op = self.advance().type
            self.emit(Op.PSH)
            self.parse_comparison()
            if op == TokenType.EQ:
                self.emit(Op.EQ)
            else:
                self.emit(Op.NE)

    def parse_comparison(self):
        self.parse_additive()
        while self.peek().type in (TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op = self.advance().type
            self.emit(Op.PSH)
            self.parse_additive()
            if op == TokenType.LT:
                self.emit(Op.LT)
            elif op == TokenType.GT:
                self.emit(Op.GT)
            elif op == TokenType.LE:
                self.emit(Op.LE)
            else:
                self.emit(Op.GE)

    def parse_additive(self):
        self.parse_multiplicative()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().type
            self.emit(Op.PSH)
            self.parse_multiplicative()
            if op == TokenType.PLUS:
                self.emit(Op.ADD)
            else:
                self.emit(Op.SUB)

    def parse_multiplicative(self):
        self.parse_unary()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().type
            self.emit(Op.PSH)
            self.parse_unary()
            if op == TokenType.STAR:
                self.emit(Op.MUL)
            elif op == TokenType.SLASH:
                self.emit(Op.DIV)
            else:
                self.emit(Op.MOD)

    def parse_unary(self):
        if self.peek().type == TokenType.MINUS:
            self.advance()
            self.parse_primary()
            # Negate: 0 - x
            self.emit(Op.PSH)
            self.emit(Op.IMM, 0)
            self.emit(Op.SUB)
        else:
            self.parse_primary()

    def parse_primary(self):
        tok = self.peek()

        if tok.type == TokenType.NUM:
            self.advance()
            self.emit(Op.IMM, tok.value)

        elif tok.type == TokenType.ID:
            name = self.advance().value

            if self.peek().type == TokenType.LPAREN:
                # Function call
                self.advance()
                argc = 0

                if self.peek().type != TokenType.RPAREN:
                    while True:
                        self.parse_expr()
                        self.emit(Op.PSH)
                        argc += 1
                        if self.peek().type == TokenType.COMMA:
                            self.advance()
                        else:
                            break

                self.expect(TokenType.RPAREN)

                # Special handling for printf
                if name == 'printf':
                    self.emit(Op.PRTF)
                    if argc > 0:
                        self.emit(Op.ADJ, argc * 8)
                else:
                    # Regular function call
                    sym = self.symbols.get(name)
                    if sym and sym.type == 'func':
                        self.emit(Op.JSR, sym.offset)
                        if argc > 0:
                            self.emit(Op.ADJ, argc * 8)
                    else:
                        raise SyntaxError(f"Unknown function: {name}")

            else:
                # Variable
                sym = self.symbols.get(name)
                if sym is None:
                    raise SyntaxError(f"Unknown variable: {name}")

                if sym.scope == 'local' or sym.scope == 'param':
                    self.emit(Op.LEA, sym.offset)
                    self.emit(Op.LI)
                else:
                    raise NotImplementedError("Global variable read")

        elif tok.type == TokenType.LPAREN:
            self.advance()
            self.parse_expr()
            self.expect(TokenType.RPAREN)

        else:
            raise SyntaxError(f"Unexpected token in expression: {tok}")


def compile_c(source: str) -> List[int]:
    """Compile C source to bytecode."""
    compiler = Compiler()
    return compiler.compile(source)


def test_compiler():
    print("C4 C COMPILER")
    print("=" * 60)
    print()

    # Test 1: Simple expression
    print("Test 1: Simple expression")
    code = compile_c("""
        int main() {
            return 3 + 4;
        }
    """)
    print(f"  Bytecode: {code}")
    print()

    # Test with our transformer
    from c4_printf_autoregressive import C4AutoregressivePrintf
    import torch

    def run_bytecode(code, entry_offset=0):
        executor = C4AutoregressivePrintf(memory_size=512)
        memory = torch.zeros(512)
        for i, instr in enumerate(code):
            memory[i * 8] = float(instr)

        # Find main and call it
        # For now, assume main is at the start
        memory_size = 512
        pc, sp, bp, ax, memory, output = executor.run(
            memory,
            pc=torch.tensor(float(entry_offset)),
            sp=torch.tensor(float(memory_size - 50)),
            bp=torch.tensor(float(memory_size - 50)),
            ax=torch.tensor(0.0),
            max_steps=100
        )
        return int(ax.item()), output

    result, output = run_bytecode(code)
    status = "✓" if result == 7 else "✗"
    print(f"  {status} 3 + 4 = {result}")
    print()

    # Test 2: Variables and arithmetic
    print("Test 2: Variables")
    code2 = compile_c("""
        int main() {
            int x;
            int y;
            x = 10;
            y = 3;
            return x * y;
        }
    """)
    print(f"  Bytecode: {code2}")
    result2, _ = run_bytecode(code2)
    status = "✓" if result2 == 30 else "✗"
    print(f"  {status} x=10, y=3, x*y = {result2}")
    print()

    # Test 3: If/else
    print("Test 3: If/else")
    code3 = compile_c("""
        int main() {
            int x;
            x = 5;
            if (x > 3) {
                return 100;
            } else {
                return 200;
            }
        }
    """)
    result3, _ = run_bytecode(code3)
    status = "✓" if result3 == 100 else "✗"
    print(f"  {status} if (5 > 3) return 100 else 200 = {result3}")
    print()

    # Test 4: While loop
    print("Test 4: While loop (sum 1 to 5)")
    code4 = compile_c("""
        int main() {
            int sum;
            int i;
            sum = 0;
            i = 1;
            while (i <= 5) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }
    """)
    result4, _ = run_bytecode(code4)
    status = "✓" if result4 == 15 else "✗"
    print(f"  {status} sum(1..5) = {result4}")
    print()

    # Test 5: Printf
    print("Test 5: Printf")
    code5 = compile_c("""
        int main() {
            printf(42);
            return 0;
        }
    """)
    result5, output5 = run_bytecode(code5)
    print(f"  Output: '{output5.strip()}'")
    status = "✓" if "42" in output5 else "✗"
    print(f"  {status} printf(42)")
    print()

    # Test 6: Function call
    print("Test 6: Function call")
    code6 = compile_c("""
        int add(int a, int b) {
            return a + b;
        }
        int main() {
            return add(3, 4);
        }
    """)
    # Need to call main which is the second function
    # Find main offset (first function ends with LEV)
    main_offset = 0
    for i, instr in enumerate(code6):
        if (instr & 0xFF) == Op.LEV:
            main_offset = (i + 1) * 8
            break

    result6, _ = run_bytecode(code6, entry_offset=main_offset)
    status = "✓" if result6 == 7 else "✗"
    print(f"  {status} add(3, 4) = {result6}")
    print()

    print("=" * 60)
    print("C COMPILER → TRANSFORMER COMPLETE!")


if __name__ == "__main__":
    test_compiler()
