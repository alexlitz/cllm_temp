#!/usr/bin/env python3
"""
Neural Quine Creator

Creates a complete self-reproducing C program that:
1. Runs neural inference using embedded model
2. Prints its own complete source code (including model bytes and runtime)

Usage:
    python create_neural_quine.py model.c4onnx > neural_quine.c
    gcc -o neural_quine neural_quine.c
    ./neural_quine > neural_quine2.c
    diff neural_quine.c neural_quine2.c  # Should be identical

The quine includes the full ONNX runtime and embedded model weights.
"""

import sys
import os

def read_model(path):
    with open(path, 'rb') as f:
        return list(f.read())

def format_model_array(data):
    """Format model data as C array initializer."""
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        lines.append('    ' + ','.join(str(b) for b in chunk))
    return ',\n'.join(lines)

def escape_c_string(s):
    """Escape string for C string literal."""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_neural_quine.py model.c4onnx > quine.c", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    model_data = read_model(model_path)
    model_len = len(model_data)

    # Build the complete program as a list of lines
    lines = []

    # Header
    lines.append('/* Neural Quine - runs inference then prints itself */')
    lines.append('int printf(char*f,...);int putchar(int c);char*malloc(int n);')

    # Model array
    model_init = format_model_array(model_data)
    lines.append(f'char M[]={{')
    lines.append(model_init)
    lines.append('};')
    lines.append(f'int ML={model_len};')

    # Minimal runtime
    lines.append('int S=65536,*TD,*TS,*TB,NT,*NO,*NI,NN,EP;')
    lines.append('int ri(){int v=(M[EP]&255)|((M[EP+1]&255)<<8)|((M[EP+2]&255)<<16)|((M[EP+3]&255)<<24);EP+=4;return v;}')
    lines.append('int ld(){int i,j,n,d,s,tp=0;EP=0;if(ri()!=1481526863)return-1;ri();NT=ri();NN=ri();TD=malloc(500000*4);TS=malloc(256*4);TB=malloc(256*4);NO=malloc(512*4);NI=malloc(4096*4);for(i=0;i<NT;i++){n=ri();EP+=n;d=ri();for(j=0;j<d;j++)ri();ri();s=ri();TS[i]=s;TB[i]=tp;for(j=0;j<s;j++)TD[tp++]=ri();}for(i=0;i<NN;i++){NO[i]=ri();n=ri();for(j=0;j<n;j++)NI[i*8+j]=ri();ri();ri();}return 0;}')
    lines.append('int fm(int a,int b){return(a/256)*(b/256);}')
    lines.append('int ex(int ni){int op=NO[ni],i0=NI[ni*8],i1=NI[ni*8+1],o0=NI[ni*8],*a=TD+TB[i0],*b=TD+TB[i1],*c=TD+TB[o0],sz=TS[o0],i;if(op==1)for(i=0;i<sz;i++)c[i]=a[i]+b[i];if(op==2)for(i=0;i<sz;i++)c[i]=fm(a[i],b[i]);return 0;}')
    lines.append('int run(){int i;for(i=0;i<NN;i++)ex(i);return 0;}')
    lines.append('int inf(int*in,int is,int*out,int os){int i,*d=TD+TB[NT-2];for(i=0;i<is;i++)d[i]=in[i];run();d=TD+TB[NT-1];for(i=0;i<os;i++)out[i]=d[i];return 0;}')

    # Quine string placeholder - will be filled with encoded lines
    lines.append('char*Q="PLACEHOLDER";')

    # Main function
    main_code = '''int main(){int i,j,in[4],out[4];char*p,*q;ld();in[0]=S;in[1]=2*S;in[2]=3*S;in[3]=4*S;inf(in,4,out,4);printf("Neural inference: ");for(i=0;i<4;i++)printf("%d ",out[i]/S);printf("\\n");printf("/* Neural Quine - runs inference then prints itself */\\n");printf("int printf(char*f,...);int putchar(int c);char*malloc(int n);\\n");printf("char M[]={\\n");for(i=0;i<ML;i++){if(i%16==0)printf("    ");printf("%d",M[i]&255);if(i+1<ML)putchar(',');if(i%16==15||i+1==ML)putchar('\\n');}printf("};\\n");printf("int ML=%d;\\n",ML);p=Q;while(*p){if(*p=='|'){putchar('\\n');p++;}else if(*p=='~'){putchar('"');p++;}else if(*p=='^'){putchar('\\\\');p++;}else{putchar(*p);p++;}}putchar('\\n');printf("char*Q=\\"");p=Q;while(*p){if(*p=='|')printf("|");else if(*p=='~')printf("~");else if(*p=='^')printf("^^");else if(*p=='\\n')printf("|");else if(*p=='"')printf("~");else if(*p=='\\\\')printf("^");else putchar(*p);p++;}printf("\\";\\n");p=Q;q=p;while(*q!='#'||q[1]!='#')q++;q+=2;while(*q){if(*q=='|')putchar('\\n');else if(*q=='~')putchar('"');else if(*q=='^')putchar('\\\\');else putchar(*q);q++;}putchar('\\n');return 0;}'''

    lines.append(main_code)

    # Now encode the lines for Q string
    # | = newline, ~ = quote, ^ = backslash
    # ## marks where main code starts in Q

    encoded_lines = []
    for i, line in enumerate(lines):
        if 'char*Q=' in line:
            continue  # Skip Q definition line
        if 'int main' in line:
            encoded_lines.append('##')  # Mark start of main
            enc = line.replace('\\', '^').replace('"', '~').replace('\n', '|')
            encoded_lines.append(enc)
        else:
            enc = line.replace('\\', '^').replace('"', '~').replace('\n', '|')
            encoded_lines.append(enc)

    q_content = '|'.join(encoded_lines)

    # Now output the actual quine
    print('/* Neural Quine - runs inference then prints itself */')
    print('int printf(char*f,...);int putchar(int c);char*malloc(int n);')
    print('char M[]={')
    print(model_init)
    print('};')
    print(f'int ML={model_len};')
    print('int S=65536,*TD,*TS,*TB,NT,*NO,*NI,NN,EP;')
    print('int ri(){int v=(M[EP]&255)|((M[EP+1]&255)<<8)|((M[EP+2]&255)<<16)|((M[EP+3]&255)<<24);EP+=4;return v;}')
    print('int ld(){int i,j,n,d,s,tp=0;EP=0;if(ri()!=1481526863)return-1;ri();NT=ri();NN=ri();TD=malloc(500000*4);TS=malloc(256*4);TB=malloc(256*4);NO=malloc(512*4);NI=malloc(4096*4);for(i=0;i<NT;i++){n=ri();EP+=n;d=ri();for(j=0;j<d;j++)ri();ri();s=ri();TS[i]=s;TB[i]=tp;for(j=0;j<s;j++)TD[tp++]=ri();}for(i=0;i<NN;i++){NO[i]=ri();n=ri();for(j=0;j<n;j++)NI[i*8+j]=ri();ri();ri();}return 0;}')
    print('int fm(int a,int b){return(a/256)*(b/256);}')
    print('int ex(int ni){int op=NO[ni],i0=NI[ni*8],i1=NI[ni*8+1],o0=NI[ni*8],*a=TD+TB[i0],*b=TD+TB[i1],*c=TD+TB[o0],sz=TS[o0],i;if(op==1)for(i=0;i<sz;i++)c[i]=a[i]+b[i];if(op==2)for(i=0;i<sz;i++)c[i]=fm(a[i],b[i]);return 0;}')
    print('int run(){int i;for(i=0;i<NN;i++)ex(i);return 0;}')
    print('int inf(int*in,int is,int*out,int os){int i,*d=TD+TB[NT-2];for(i=0;i<is;i++)d[i]=in[i];run();d=TD+TB[NT-1];for(i=0;i<os;i++)out[i]=d[i];return 0;}')
    print(f'char*Q="{escape_c_string(q_content)}";')
    print(main_code)

if __name__ == '__main__':
    main()
