#!/usr/bin/env python3
"""
Neural Quine Generator

Creates a C program that:
1. Embeds neural model weights
2. Prints its own exact source code when run

Usage:
    python gen_quine.py model.c4onnx > quine.c
    gcc -o quine quine.c
    ./quine > quine2.c
    diff quine.c quine2.c  # Should be empty
"""

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_quine.py model.c4onnx > quine.c", file=sys.stderr)
        sys.exit(1)

    # Read model bytes
    with open(sys.argv[1], 'rb') as f:
        model = list(f.read())

    model_str = ','.join(str(b) for b in model)
    model_len = len(model)

    # The quine uses a simple encoding:
    # @ = newline, $ = quote, # = backslash, ^ = percent

    # Line 1: comment
    L1 = "/* Neural Quine with embedded model */"
    # Line 2: declarations
    L2 = "int printf(char*f,...);int putchar(int c);"
    # Line 3: model array (will be generated)
    L3_pre = "char M[]={"
    L3_post = "};"
    # Line 4: model length
    L4 = f"int L={model_len};"
    # Line 5: quine string (will contain encoded program)
    # Line 6: main function

    # The main function needs to print:
    # - L1, L2 verbatim
    # - L3 with model bytes
    # - L4 verbatim
    # - L5 (the Q string itself, escaped)
    # - L6 (itself)

    # Build the main function code
    main_code = '''int main(){int i;char*p;'''
    main_code += '''printf("/* Neural Quine with embedded model */\\n");'''
    main_code += '''printf("int printf(char*f,...);int putchar(int c);\\n");'''
    main_code += '''printf("char M[]={");'''
    main_code += '''for(i=0;i<L;i++){if(i)putchar(',');printf("%d",M[i]&255);}'''
    main_code += '''printf("};\\n");'''
    main_code += f'''printf("int L=%d;\\n",L);'''
    main_code += '''printf("char*Q=\\"");'''
    main_code += '''p=Q;while(*p){if(*p=='@')printf("\\\\n");else if(*p=='$')printf("\\\\\\"");else if(*p=='#')printf("\\\\\\\\");else if(*p=='^')printf("\\\\%%");else if(*p==10)printf("@");else if(*p=='"')printf("$");else if(*p=='\\\\')printf("#");else if(*p=='%')printf("^");else putchar(*p);p++;}'''
    main_code += '''printf("\\";\\n");'''
    main_code += '''p=Q;while(*p){if(*p=='@')putchar(10);else if(*p=='$')putchar('"');else if(*p=='#')putchar('\\\\');else if(*p=='^')putchar('%');else putchar(*p);p++;}'''
    main_code += '''putchar(10);return 0;}'''

    # Encode main_code for Q string (replace special chars)
    Q_content = main_code.replace('\\', '#').replace('"', '$').replace('\n', '@').replace('%', '^')

    # Output the quine
    print(L1)
    print(L2)
    print(f"char M[]={{{model_str}}};")
    print(f"int L={model_len};")
    print(f'char*Q="{Q_content}";')
    print(main_code)

if __name__ == '__main__':
    main()
